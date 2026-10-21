# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audits of what a run did, from its own records: the epochs its seals read, and whether its checks were
deterministic.

* :func:`audit_epochs` checks invariants I3 and I4 over a seal journal against the publication log: every seal's batch
  epoch is the one in force when it read the pointer (I4: an epoch published before the seal is reflected in it, and
  none newer is), and epochs never decrease in commit order (I3). A pinned seal (``SealDecision.target_pin``) relaxes
  I4's adoption on purpose: it is reported as pinned, and checked against its pin and against what its pointer read
  named, instead.
* :func:`verdict_conflicts` finds identical executions that got different verdicts. The contract presumes that a check
  gives one verdict per execution; a test whose outcome follows an unpinned seed (a string-hash seed that orders a
  set, for instance) breaks that, and the seal does not check it. Every raw output the records hold (each record's
  own runs, and every re-execution its scores made) is graded again under the checks of the given versions; outputs of
  the same request (the same artifact as applied, the same execution identity) whose verdicts under the same grading
  identity differ are reported.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .canonical import FrozenDict, freeze, thaw
from .records import EpochRef, ExecutionRecord, Outcome, ScoreRecord, SealDecision, TestRun
from .scoring import Adapter
from .store import ContentStore
from .versions import GraderVersion, Publication, iter_checks

__all__ = [
    "EpochAudit",
    "VerdictConflicts",
    "audit_epochs",
    "verdict_conflicts",
]


# ---------------------------------------------------------------------------------------------------------- epochs


@dataclass(frozen=True)
class EpochAudit:
    """The epochs of a journal's seals, checked against the publication log.

    ``seals`` has one entry per decision, in commit order: its seal and batch, the batch epoch's ``seq``, what the
    pointer named at the read (``pointer``; the batch epoch unless pinned), the epoch in force at the read time by the
    log (``in_force``) and a ``status``: ``ok``; ``race`` (the pointer missed only publications made within
    ``tolerance_s`` of the read: a publication and a read that overlapped, not a violation); ``unexpected`` (I4
    fails: the pointer named another epoch than the one in force, ``problem`` says how); ``pinned`` (the batch
    epoch is the pinned one, and the pointer read is checked as above). ``unexpected`` lists the seals that fail I4,
    ``decreasing`` the seals whose epoch is older than an earlier seal's (I3), ``pinned`` the pinned seals."""

    seals: tuple[Mapping[str, Any], ...]
    unexpected: tuple[str, ...]
    decreasing: tuple[str, ...]
    pinned: tuple[str, ...]
    races: tuple[str, ...]
    ok: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "seals", tuple(freeze(s) for s in self.seals))

    def to_dict(self) -> dict:
        return {"seals": [thaw(s) for s in self.seals], "unexpected": list(self.unexpected),
                "decreasing": list(self.decreasing), "pinned": list(self.pinned), "races": list(self.races),
                "ok": self.ok}


def _publications(epochs: Any) -> list[Publication]:
    if hasattr(epochs, "publications"):
        out = list(epochs.publications("grader"))
    else:
        out = [p for p in epochs if p.kind == "grader"]
    return sorted(out, key=lambda p: p.seq)


def _in_force(graders: Sequence[Publication], t: float | None) -> Publication | None:
    if t is None:
        return None
    current = None
    for p in graders:
        if p.published_at <= t:
            current = p
    return current


def _check_read(ref: EpochRef | None, graders: Sequence[Publication], read_at: float | None,
                tolerance_s: float) -> tuple[str, str | None, Publication | None]:
    """(status, problem, in force) of one pointer read that returned ``ref`` at ``read_at``."""
    in_force = _in_force(graders, read_at)
    by_seq = {p.seq: p for p in graders}
    if ref is not None:
        known = by_seq.get(ref.seq)
        if known is None or known.version_id != ref.version_id:
            return "unexpected", "not_a_grader_publication", in_force
    if read_at is None:
        return ("ok", None, in_force) if ref is not None else ("ok", None, None)
    if in_force is None:
        if ref is None:
            return "ok", None, None
        return "unexpected", "newer_than_in_force", None  # read an epoch published after the read: clocks disagree
    if ref is None:
        late = in_force.published_at > read_at - tolerance_s and all(
            p.published_at > read_at - tolerance_s for p in graders if p.seq <= in_force.seq)
        return ("race", "no_epoch", in_force) if late else ("unexpected", "no_epoch", in_force)
    if ref.seq == in_force.seq:
        return "ok", None, in_force
    if ref.seq > in_force.seq:
        return "unexpected", "newer_than_in_force", in_force
    missed = [p for p in graders if ref.seq < p.seq <= in_force.seq]
    if all(p.published_at > read_at - tolerance_s for p in missed):
        return "race", "older_than_in_force", in_force
    return "unexpected", "older_than_in_force", in_force


def audit_epochs(decisions: Iterable[SealDecision], epochs: Any, *, tolerance_s: float = 1.0) -> EpochAudit:
    """Check I4 and I3 over ``decisions`` (a seal journal, in commit order) against the publication log of
    ``epochs`` (a registry, or its publications). Publication times and the seals' read times must come from
    comparable clocks. A failed seal (no decision made) is checked like any other: its read still counts.

    A pinned seal's batch epoch must be its pin's publication (``"e0"``: the first grader publication), resolved
    when the log held it at the read; its pointer read (``target_pin.published``) is checked as an unpinned seal's.
    The read time is the batch epoch's ``read_at`` (for a seal without an epoch, its ``opened_at``)."""
    graders = _publications(epochs)
    seals: list[dict] = []
    unexpected, decreasing, pinned, races = [], [], [], []
    last_seq: int | None = None
    for d in decisions:
        pin = d.target_pin
        pointer = pin.published if pin is not None else d.epoch
        read_at = (d.epoch.read_at if d.epoch is not None else None)
        if read_at is None and pointer is not None:
            read_at = pointer.read_at
        if read_at is None:
            read_at = d.opened_at
        status, problem, in_force = _check_read(pointer, graders, read_at, tolerance_s)
        entry: dict[str, Any] = {"seal_id": d.seal_id, "batch_id": d.batch_id,
                                 "epoch": d.epoch.seq if d.epoch is not None else None,
                                 "pointer": pointer.seq if pointer is not None else None,
                                 "in_force": in_force.seq if in_force is not None else None, "read_at": read_at,
                                 "status": status, "problem": problem}
        if pin is not None:
            entry["pin"] = pin.pin
            if pin.pin == "e0":
                want = graders[0] if graders else None
            else:
                want = next((p for p in graders if p.seq == pin.pin), None)
            if pointer is None or (want is not None and want.seq > pointer.seq):
                want = None  # nothing read, or published after the read: the pin could not resolve
            got = (d.epoch.seq, d.epoch.version_id) if d.epoch is not None else None
            if got != ((want.seq, want.version_id) if want is not None else None) or \
                    bool(pin.resolved) != (got is not None):
                entry["status"], entry["problem"] = "unexpected", "not_the_pinned_epoch"
            elif status == "ok":
                entry["status"] = "pinned"  # the read was right, and the batch epoch is the pin's
            pinned.append(d.seal_id)
        if entry["status"] == "unexpected":
            unexpected.append(d.seal_id)
        elif entry["status"] == "race":
            races.append(d.seal_id)
        if d.epoch is not None:
            if last_seq is not None and d.epoch.seq < last_seq:
                decreasing.append(d.seal_id)
                entry["decreasing"] = True
            last_seq = d.epoch.seq if last_seq is None else max(last_seq, d.epoch.seq)
        seals.append(entry)
    return EpochAudit(seals=tuple(seals), unexpected=tuple(unexpected), decreasing=tuple(decreasing),
                      pinned=tuple(pinned), races=tuple(races), ok=not unexpected and not decreasing)


# ---------------------------------------------------------------------------------------------------------- verdicts


@dataclass(frozen=True)
class VerdictConflicts:
    """Identical executions with different verdicts.

    An *observation* is one raw output (a run) of a request (what runs: the artifact as applied and the check's
    execution identity, :attr:`agent_ecc.recompute.ExecRequest.request_id`), graded under one grading identity. A
    *group* is every observation of one (request, grading identity); it is *repeated* when it holds more than one
    distinct run, and *conflicting* when distinct runs got a pass and a fail. ``conflicts`` lists the conflicting
    groups (task, check, the runs by outcome, the records they belong to); ``by_task`` counts per task."""

    groups: int
    repeated: int
    conflicts: tuple[Mapping[str, Any], ...]
    by_task: Mapping[str, Any] = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "conflicts", tuple(freeze(c) for c in self.conflicts))
        object.__setattr__(self, "by_task", freeze(self.by_task))

    @property
    def deterministic(self) -> bool:
        return not self.conflicts

    def to_dict(self) -> dict:
        return {"groups": self.groups, "repeated": self.repeated, "conflicting": len(self.conflicts),
                "conflicts": [thaw(c) for c in self.conflicts], "by_task": thaw(self.by_task)}


def verdict_conflicts(records: Iterable[ExecutionRecord], scores: Iterable[ScoreRecord],
                      versions: Mapping[str, GraderVersion] | Iterable[GraderVersion],
                      adapters: Mapping[str, Adapter] | Sequence[Adapter], store: ContentStore) -> VerdictConflicts:
    """Grade every raw output the records hold under every check of ``versions`` that has its execution identity,
    and report the requests whose outputs got both verdicts under one grading identity (see
    :class:`VerdictConflicts`). The outputs are each record's own runs and the runs its scores' re-executions made
    (:meth:`agent_ecc.records.ScoreRecord.new_runs`). A record whose artifact gives no request (nothing to execute,
    such as an empty patch) has nothing to compare. Grading uses the adapters' ``parse``, a pure function of the
    check, the run and the objects it reads, so a conflict is between outputs, never between readings."""
    from .scoring import PartVerdict

    if not isinstance(adapters, Mapping):
        adapters = {a.family: a for a in adapters}
    if not isinstance(versions, Mapping):
        versions = {v.version_id: v for v in versions}
    records = list(records)
    runs_of: dict[str, list[TestRun]] = defaultdict(list)
    for r in records:
        runs_of[r.record_id].extend(r.runs)
    for s in scores:
        runs_of[s.execution_id].extend(s.new_runs())
    observations: dict[tuple, dict[str, Any]] = {}
    for record in records:
        adapter = adapters.get(record.family)
        if adapter is None:
            continue
        seen_checks: set[tuple[str, str]] = set()
        for version in versions.values():
            lookup = version.lookup(record.task_id, record.world_version)
            if lookup.spec is None:
                continue
            for check in iter_checks(lookup.spec):
                ids = (check.execution_digest, check.grading_digest)
                if ids in seen_checks:
                    continue
                seen_checks.add(ids)
                try:
                    request = adapter.reexec_request(record, check, store)
                except Exception:  # noqa: BLE001 - an artifact the adapter cannot read
                    request = None
                if request is None:
                    continue
                for run in runs_of.get(record.record_id, ()):
                    if run.execution_digest != check.execution_digest:
                        continue
                    try:
                        verdict = adapter.parse(check, run, store)
                    except Exception:  # noqa: BLE001 - a parse failure is not a verdict
                        verdict = PartVerdict.undeterminable("parse_error")
                    if not verdict.outcome.logical:
                        continue
                    key = (request.request_id, check.grading_digest)
                    group = observations.setdefault(key, {"task_id": record.task_id, "check": check.name,
                                                          "request_id": request.request_id,
                                                          "grading_digest": check.grading_digest, "runs": {},
                                                          "records": set()})
                    group["runs"][run.run_digest] = verdict.outcome
                    group["records"].add(record.record_id)
    conflicts, by_task = [], defaultdict(lambda: {"groups": 0, "repeated": 0, "conflicting": 0})
    repeated = 0
    for key in sorted(observations):
        group = observations[key]
        outcomes = group["runs"]
        task = by_task[group["task_id"]]
        task["groups"] += 1
        if len(outcomes) > 1:
            repeated += 1
            task["repeated"] += 1
        passes = sorted(d for d, o in outcomes.items() if o is Outcome.PASS)
        fails = sorted(d for d, o in outcomes.items() if o is Outcome.FAIL)
        if passes and fails:
            task["conflicting"] += 1
            conflicts.append({"task_id": group["task_id"], "check": group["check"], "request_id": group["request_id"],
                              "grading_digest": group["grading_digest"], "pass_runs": passes, "fail_runs": fails,
                              "records": sorted(group["records"])})
    return VerdictConflicts(groups=len(observations), repeated=repeated, conflicts=tuple(conflicts),
                            by_task=dict(sorted(by_task.items())))
