# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifests: what an evaluation measured, and what training actually consumed.

**Evaluation manifests** (:func:`evaluation_manifest`) state a result under exactly one grader version: the record
set (by digest), the version, one score per record, and the undeterminable records counted separately (never as
fails). Scores under another version are refused, so a manifest cannot mix graders. Because the record set and the
version are both content-addressed, a manifest can be reproduced later from the same records
(:func:`reproduce_evaluation`), and two manifests over the same records under two versions say exactly which verdicts
the change flipped (:func:`compare_evaluations`).

**Training consumption** (:func:`consumption_records`, :func:`training_manifest`) is established from the learner's
side: its per-record log (one line per consumed sample, written where the loss consumes it) and its step completion
events, joined with the seal decision that handed the batch over. A step counts as consumed only if it completed and
its records were logged; a seal decision alone is not consumption. The join checks, per step, that every sealed
record was consumed and nothing else was, once, in its sealed group, with its sealed reward, on verified tokens, with
an advantage that is its group's sealed rewards normalized, on the tokens it was handed over with, and on at least one
token; and that the step consumed nothing another seal handed over under another epoch (a batch mixing epochs).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from .canonical import FrozenDict, digest_json, freeze, thaw
from .records import (
    ConsumedRecord,
    ConsumptionRecord,
    ExecutionRecord,
    Method,
    Outcome,
    ScoreRecord,
    SealDecision,
)
from .scoring import ScoreInput, Scorer
from .versions import Epoch, GraderVersion

__all__ = [
    "EvaluationManifest",
    "EvaluationRow",
    "LearnerLine",
    "MixedVersionError",
    "StepCompletion",
    "TrainingManifest",
    "TrainingReport",
    "compare_evaluations",
    "consumption_records",
    "evaluate",
    "evaluation_manifest",
    "group_normalized",
    "reproduce_evaluation",
    "training_manifest",
]

SCHEMA_EVALUATION = "agent-ecc/evaluation-manifest/1"
SCHEMA_TRAINING = "agent-ecc/training-manifest/1"


class MixedVersionError(ValueError):
    """Scores under different grader versions were given to one evaluation."""


# ---------------------------------------------------------------------------------------------------------- evaluation


@dataclass(frozen=True)
class EvaluationRow:
    execution_id: str
    task_id: str
    trajectory_id: str
    outcome: Outcome
    method: Method
    score_id: str
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", Outcome(self.outcome))
        object.__setattr__(self, "method", Method(self.method))

    def to_dict(self) -> dict:
        return {"execution_id": self.execution_id, "task_id": self.task_id, "trajectory_id": self.trajectory_id,
                "outcome": self.outcome.value, "method": self.method.value, "score_id": self.score_id,
                "reason": self.reason}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EvaluationRow:
        return cls(execution_id=d["execution_id"], task_id=d["task_id"], trajectory_id=d["trajectory_id"],
                   outcome=d["outcome"], method=d["method"], score_id=d["score_id"], reason=d.get("reason"))


@dataclass(frozen=True)
class EvaluationManifest:
    """A result under one grader version over one record set."""

    grader_version: str
    version_name: str | None
    record_set: str
    rows: tuple[EvaluationRow, ...]
    counts: Mapping[str, Any]
    by_task: Mapping[str, Any]
    meta: Mapping[str, Any] = field(default_factory=FrozenDict)
    schema: str = SCHEMA_EVALUATION

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(r if isinstance(r, EvaluationRow) else EvaluationRow.from_dict(r)
                                               for r in self.rows))
        object.__setattr__(self, "counts", freeze(self.counts))
        object.__setattr__(self, "by_task", freeze(self.by_task))
        object.__setattr__(self, "meta", freeze(self.meta))

    def body(self) -> dict:
        return {"schema": self.schema, "grader_version": self.grader_version, "version_name": self.version_name,
                "record_set": self.record_set, "rows": [r.to_dict() for r in self.rows],
                "counts": thaw(self.counts), "by_task": thaw(self.by_task), "meta": thaw(self.meta)}

    @cached_property
    def manifest_id(self) -> str:
        return digest_json(self.body())

    def to_dict(self) -> dict:
        return {**self.body(), "manifest_id": self.manifest_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EvaluationManifest:
        manifest = cls(grader_version=d["grader_version"], version_name=d.get("version_name"),
                       record_set=d["record_set"], rows=tuple(d.get("rows") or ()), counts=d.get("counts") or {},
                       by_task=d.get("by_task") or {}, meta=d.get("meta") or {},
                       schema=d.get("schema", SCHEMA_EVALUATION))
        if d.get("manifest_id") is not None and d["manifest_id"] != manifest.manifest_id:
            raise ValueError("evaluation manifest id does not match its content")
        return manifest

    @property
    def undeterminable(self) -> int:
        return int(self.counts.get("undeterminable", 0))

    def outcomes(self) -> dict[str, Outcome]:
        return {r.execution_id: r.outcome for r in self.rows}


def record_set_digest(execution_ids: Iterable[str]) -> str:
    return digest_json(sorted(set(execution_ids)))


def evaluation_manifest(records: Sequence[ExecutionRecord], scores: Sequence[ScoreRecord],
                        version: GraderVersion | Epoch | str, *, version_name: str | None = None,
                        meta: Mapping[str, Any] | None = None) -> EvaluationManifest:
    """The manifest of ``records`` under ``version``: exactly one score per record, every score under the version."""
    if isinstance(version, Epoch):
        version = version.version
    version_id = version.version_id if isinstance(version, GraderVersion) else str(version)
    if isinstance(version, GraderVersion) and version_name is None:
        version_name = version.name
    by_record: dict[str, ScoreRecord] = {}
    for score in scores:
        if score.grader_version != version_id:
            raise MixedVersionError(f"score {score.score_id} is under {score.grader_version}, not {version_id}")
        known = by_record.get(score.execution_id)
        if known is not None and not known.same_verdict(score):
            raise ValueError(f"record {score.execution_id} has two different verdicts under {version_id}")
        by_record.setdefault(score.execution_id, score)
    ids = [r.record_id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("the record set holds a record twice")
    missing = [i for i in ids if i not in by_record]
    if missing:
        raise ValueError(f"{len(missing)} records have no score under {version_id} (first: {missing[0]})")
    extra = set(by_record) - set(ids)
    if extra:
        raise ValueError(f"{len(extra)} scores are for records outside the record set")
    rows = []
    counts: dict[str, Any] = {"records": len(records), "pass": 0, "fail": 0, "undeterminable": 0, "by_method": {}}
    tasks: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0, "undeterminable": 0})
    for record in sorted(records, key=lambda r: r.record_id):
        score = by_record[record.record_id]
        rows.append(EvaluationRow(execution_id=record.record_id, task_id=record.task_id,
                                  trajectory_id=record.trajectory_id, outcome=score.outcome, method=score.method,
                                  score_id=score.score_id, reason=score.reason))
        counts[score.outcome.value] += 1
        counts["by_method"][score.method.value] = counts["by_method"].get(score.method.value, 0) + 1
        tasks[record.task_id][score.outcome.value] += 1
    decided = counts["pass"] + counts["fail"]
    counts["pass_rate"] = (counts["pass"] / decided) if decided else None
    by_task = {}
    for task, c in sorted(tasks.items()):
        n = c["pass"] + c["fail"]
        by_task[task] = {**c, "pass_rate": (c["pass"] / n) if n else None}
    return EvaluationManifest(grader_version=version_id, version_name=version_name, record_set=record_set_digest(ids),
                              rows=tuple(rows), counts=counts, by_task=by_task, meta=meta or {})


def evaluate(records: Sequence[ExecutionRecord], version: GraderVersion | Epoch, scorer: Scorer, *,
             prior: Mapping[str, Sequence[ScoreRecord]] | None = None,
             meta: Mapping[str, Any] | None = None) -> tuple[EvaluationManifest, list[ScoreRecord]]:
    """Score ``records`` under ``version`` (each with its prior scores, by record id) and return the manifest and the
    new scores."""
    prior = prior or {}
    scored = scorer.score([ScoreInput(r, tuple(prior.get(r.record_id, ()))) for r in records], version)
    pending = [s for s in scored if s.score is None]
    if pending:
        raise ValueError(f"{len(pending)} records need an action the scorer does not allow ({pending[0].required})")
    scores = [s.score for s in scored]  # type: ignore[misc]
    return evaluation_manifest(records, scores, version, meta=meta), scores  # type: ignore[arg-type]


def compare_evaluations(a: EvaluationManifest, b: EvaluationManifest) -> dict:
    """What changed between two manifests over the same record set: per-record outcome changes and per-task rates."""
    if a.record_set != b.record_set:
        raise ValueError("the manifests cover different record sets")
    oa, ob = a.outcomes(), b.outcomes()
    rows_b = {r.execution_id: r for r in b.rows}
    changes = []
    for row in a.rows:
        if oa[row.execution_id] is not ob[row.execution_id]:
            changes.append({"execution_id": row.execution_id, "task_id": row.task_id,
                            "from": oa[row.execution_id].value, "to": ob[row.execution_id].value,
                            "method": rows_b[row.execution_id].method.value})
    flips = [c for c in changes if c["from"] != "undeterminable" and c["to"] != "undeterminable"]
    return {"from_version": a.grader_version, "to_version": b.grader_version, "records": len(a.rows),
            "changed": len(changes), "flips": len(flips),
            "pass_to_fail": sum(1 for c in flips if c["from"] == "pass"),
            "fail_to_pass": sum(1 for c in flips if c["from"] == "fail"),
            "became_undeterminable": sum(1 for c in changes if c["to"] == "undeterminable"),
            "changes": changes,
            "pass_rate": {"from": a.counts.get("pass_rate"), "to": b.counts.get("pass_rate")}}


def reproduce_evaluation(manifest: EvaluationManifest, records: Sequence[ExecutionRecord],
                         version: GraderVersion | Epoch, scorer: Scorer, *,
                         prior: Mapping[str, Sequence[ScoreRecord]] | None = None) -> dict:
    """Score the records again under the manifest's version and compare outcome by outcome."""
    if isinstance(version, Epoch):
        version = version.version
    if version.version_id != manifest.grader_version:
        raise MixedVersionError("the version given is not the manifest's")
    if record_set_digest(r.record_id for r in records) != manifest.record_set:
        raise ValueError("the records given are not the manifest's record set")
    again, _ = evaluate(records, version, scorer, prior=prior)
    before, after = manifest.outcomes(), again.outcomes()
    differ = sorted(k for k in before if before[k] is not after[k])
    return {"reproduced": not differ, "records": len(before), "differ": differ,
            "manifest_id": manifest.manifest_id, "again_id": again.manifest_id}


# ---------------------------------------------------------------------------------------------------------- training


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


@dataclass(frozen=True)
class LearnerLine:
    """One line of a learner's per-record log: one sample consumed by one learner part.

    Every field is read from the sample's own row where the loss consumes it: ``reward`` is the reward the sample was
    handed over with, ``normalized`` the learner's group-normalized reward, ``advantage`` (and its min and max over the
    sample's loss tokens) the advantage the loss used, ``loss_tokens`` the tokens it trained on in this part,
    ``handoff_loss_tokens`` the count stamped at hand-off, ``identity`` whether the trained tokens are the stamped ones
    (verified, mismatch, unstamped). ``reporter`` names the data-parallel shard, optionally ``<shard>:cp:<part>`` for a
    context-parallel part; ``step_part`` and ``microbatch`` locate the consumption inside the step."""

    batch_id: str
    member_id: str
    group_id: str | None = None
    step: int | str | None = None
    reward: float | None = None
    normalized: float | None = None
    advantage: float | None = None
    advantage_min: float | None = None
    advantage_max: float | None = None
    loss_tokens: int | None = None
    handoff_loss_tokens: int | None = None
    identity: str | None = None
    reporter: str | None = None
    microbatch: Any = None
    step_part: Any = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LearnerLine:
        def opt_int(v):
            return None if v is None else int(v)

        return cls(batch_id=str(d["batch_id"]), member_id=str(d["member_id"]),
                   group_id=None if d.get("group_id") is None else str(d["group_id"]), step=d.get("step"),
                   reward=_num(d.get("reward")), normalized=_num(d.get("normalized")),
                   advantage=_num(d.get("advantage")), advantage_min=_num(d.get("advantage_min")),
                   advantage_max=_num(d.get("advantage_max")), loss_tokens=opt_int(d.get("loss_tokens")),
                   handoff_loss_tokens=opt_int(d.get("handoff_loss_tokens")), identity=d.get("identity"),
                   reporter=None if d.get("reporter") is None else str(d["reporter"]),
                   microbatch=d.get("microbatch"), step_part=d.get("step_part"))

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @property
    def shard(self) -> str:
        return str(self.reporter).split(":cp:")[0]


@dataclass(frozen=True)
class StepCompletion:
    """A learner step finished: the batch it consumed, its step number, when, and whether it succeeded."""

    batch_id: str
    step: int | str | None = None
    completed_at: float | None = None
    ok: bool = True

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> StepCompletion:
        return cls(batch_id=str(d["batch_id"]), step=d.get("step"), completed_at=_num(d.get("completed_at")),
                   ok=bool(d.get("ok", True)))

    def to_dict(self) -> dict:
        return {"batch_id": self.batch_id, "step": self.step, "completed_at": self.completed_at, "ok": self.ok}


def group_normalized(rewards: Sequence[float], *, std: bool, eps: float = 1e-6) -> list[float]:
    """GRPO normalization of one group's rewards: mean-centred, optionally divided by (sample std + eps)."""
    mean = sum(rewards) / len(rewards)
    out = [r - mean for r in rewards]
    if std and len(rewards) > 1:
        sd = math.sqrt(sum(x * x for x in out) / (len(rewards) - 1))
        if sd > 0:
            out = [x / (sd + eps) for x in out]
    return out


#: the join's checks, each a list of issues per consumed step:
#:   records_unsealed        a record the step's seal did not hand over (or the step has no seal decision); each entry
#:                           names the other seals that handed it over (``sealed_in``: a record delivered again)
#:   records_unconsumed      a sealed record no learner line consumed
#:   group_mismatch          the record's learner group is not its sealed group
#:   reward_mismatch         the reward the record was handed over with is not the seal's
#:   identity_unverified     the tokens trained are not the ones handed over (or carry no stamp)
#:   duplicate_consumption   one record consumed more than once in the step (two shards, micro-batches or step parts)
#:   loss_tokens_mismatch    a consumption's loss tokens differ from the count stamped at hand-off
#:   advantage_inconsistent  the normalized reward or the advantage is not the sealed group's rewards normalized
#:   zero_loss_tokens        a sealed record consumed on no loss token at all (removed from the loss, before or after
#:                           the hand-off stamp: its stamp and its learner lines then agree on 0, which no other check
#:                           compares with the seal)
#:   epoch_mixed             the step consumed a record another seal handed over under another epoch than its own
ISSUE_KEYS = ("records_unsealed", "records_unconsumed", "group_mismatch", "reward_mismatch", "identity_unverified",
              "duplicate_consumption", "loss_tokens_mismatch", "advantage_inconsistent", "zero_loss_tokens",
              "epoch_mixed")


@dataclass(frozen=True)
class TrainingManifest:
    """What training consumed, step by step, and whether every consumption matched its seal decision."""

    steps: tuple[Mapping[str, Any], ...]
    consumption_ids: tuple[str, ...]
    records_consumed: int
    labels: Mapping[str, Any]
    by_grader_version: Mapping[str, Any]
    issues: Mapping[str, Any]
    incomplete_batches: tuple[str, ...]
    unconfirmed_batches: tuple[str, ...]
    unconsumed_seals: tuple[str, ...]
    resealed_batches: tuple[str, ...]
    normalization: Mapping[str, Any]
    ok: bool
    meta: Mapping[str, Any] = field(default_factory=FrozenDict)
    schema: str = SCHEMA_TRAINING

    def __post_init__(self) -> None:
        for name in ("steps",):
            object.__setattr__(self, name, tuple(freeze(s) for s in getattr(self, name)))
        for name in ("consumption_ids", "incomplete_batches", "unconfirmed_batches", "unconsumed_seals",
                     "resealed_batches"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("labels", "by_grader_version", "issues", "normalization", "meta"):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    def body(self) -> dict:
        return {"schema": self.schema, "steps": [thaw(s) for s in self.steps],
                "consumption_ids": list(self.consumption_ids), "records_consumed": self.records_consumed,
                "labels": thaw(self.labels), "by_grader_version": thaw(self.by_grader_version),
                "issues": thaw(self.issues), "incomplete_batches": list(self.incomplete_batches),
                "unconfirmed_batches": list(self.unconfirmed_batches),
                "unconsumed_seals": list(self.unconsumed_seals), "resealed_batches": list(self.resealed_batches),
                "normalization": thaw(self.normalization), "ok": self.ok, "meta": thaw(self.meta)}

    @cached_property
    def manifest_id(self) -> str:
        return digest_json(self.body())

    def to_dict(self) -> dict:
        return {**self.body(), "manifest_id": self.manifest_id}


@dataclass(frozen=True)
class TrainingReport:
    records: tuple[ConsumptionRecord, ...]
    manifest: TrainingManifest


def _dedup_lines(lines: Iterable[LearnerLine]) -> list[LearnerLine]:
    """A micro-batch whose loss ran twice is logged twice: keep the last line per (batch, step part, reporter,
    micro-batch, member)."""
    last: dict = {}
    for line in lines:
        last[(line.batch_id, str(line.step_part), line.reporter, str(line.microbatch), line.member_id)] = line
    return list(last.values())


def consumption_records(decisions: Sequence[SealDecision], lines: Iterable[LearnerLine | Mapping[str, Any]],
                        completions: Iterable[StepCompletion | Mapping[str, Any]], *,
                        normalization: str = "auto", tolerance: float = 1e-4, eps: float = 1e-6,
                        meta: Mapping[str, Any] | None = None) -> TrainingReport:
    """Join seal decisions with the learner's per-record log and step completions.

    ``normalization`` is the learner's group normalization: ``mean`` (mean-centred), ``mean_std`` (also divided by the
    sample standard deviation + ``eps``) or ``auto`` (whichever explains the log better). Returns one
    :class:`ConsumptionRecord` per completed step whose records were logged, and the :class:`TrainingManifest`."""
    if normalization not in ("auto", "mean", "mean_std"):
        raise ValueError("normalization must be auto, mean or mean_std")
    lines = _dedup_lines(x if isinstance(x, LearnerLine) else LearnerLine.from_dict(x) for x in lines)
    completions = [c if isinstance(c, StepCompletion) else StepCompletion.from_dict(c) for c in completions]
    sealed: dict[str, SealDecision] = {}
    resealed: set[str] = set()
    for d in decisions:
        if d.batch_id is None:
            continue
        if d.batch_id in sealed:
            resealed.add(d.batch_id)
        sealed[d.batch_id] = d  # a retried step reseals: the last decision is the one handed over
    completed = {}
    for c in completions:
        if c.ok:
            completed[c.batch_id] = c
    # every seal that handed each record over (the last decision per batch), to name where a stray record came from
    handed: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for batch_id, d in sealed.items():
        for _, m in d.members():
            handed[m.member_id].append((batch_id, d.epoch))

    def elsewhere(mid: str, batch: str) -> list[dict]:
        return [{"batch_id": b, "epoch_seq": e.seq if e is not None else None,
                 "version_id": e.version_id if e is not None else None}
                for b, e in sorted(handed.get(mid, ()), key=lambda x: x[0]) if b != batch]

    by_batch: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in lines:
        agg = by_batch[line.batch_id].setdefault(line.member_id, {
            "lines": [], "shards": set(), "parts": defaultdict(set), "uses": {}})
        agg["lines"].append(line)
        agg["shards"].add(line.shard)
        agg["parts"][(str(line.step_part), str(line.reporter))].add(str(line.microbatch))
        use = agg["uses"].setdefault((str(line.step_part), line.shard, str(line.microbatch)),
                                     {"cp_parts": 0, "loss_tokens": 0, "handed": None})
        use["cp_parts"] += 1
        use["loss_tokens"] = (None if line.loss_tokens is None or use["loss_tokens"] is None
                              else use["loss_tokens"] + int(line.loss_tokens))
        if line.handoff_loss_tokens is not None:
            use["handed"] = int(line.handoff_loss_tokens)

    errors = {"mean": 0.0, "mean_std": 0.0}
    staged = []  # (batch, issues, consumed rows, advantage checks)
    for batch in sorted(set(completed) & set(by_batch)):
        learned = by_batch[batch]
        issues: dict[str, list] = {k: [] for k in ISSUE_KEYS}
        decision = sealed.get(batch)
        if decision is None:
            issues["records_unsealed"].extend({"member_id": m, "reason": "no_seal_decision",
                                               "sealed_in": elsewhere(m, batch)} for m in sorted(learned))
            staged.append((batch, None, issues, [], []))
            continue
        members = {m.member_id: (g, m) for g, m in decision.members()}
        groups = {g.group_id: {m.member_id for m in g.members} for g in decision.groups}
        issues["records_unconsumed"].extend({"member_id": m} for m in sorted(members) if m not in learned)
        learner_groups: dict = defaultdict(set)
        for mid, agg in learned.items():
            learner_groups[agg["lines"][-1].group_id].add(mid)
        expected = {}
        for gid, mids in groups.items():
            ordered = sorted(mids)
            rewards = [members[m][1].trained_reward for m in ordered]
            if any(r is None for r in rewards):
                continue
            on = group_normalized([float(r) for r in rewards], std=True, eps=eps)  # type: ignore[arg-type]
            off = group_normalized([float(r) for r in rewards], std=False, eps=eps)  # type: ignore[arg-type]
            for m, a, b in zip(ordered, on, off):
                expected[m] = (b, a)  # (mean, mean_std)
        rows, checks = [], []
        for mid in sorted(learned):
            agg = learned[mid]
            line = agg["lines"][-1]
            twice = {f"{p}:{r}": sorted(mbs) for (p, r), mbs in agg["parts"].items() if len(mbs) > 1}
            if len(agg["uses"]) > 1:
                # consumed more than once in one step: by two shards, in two micro-batches, or in two step parts
                # (a context-parallel record's parts are one consumption)
                issues["duplicate_consumption"].append({
                    "member_id": mid, "shards": sorted(agg["shards"]), "microbatches": twice,
                    "consumptions": sorted([list(k) for k in agg["uses"]])})
            if mid not in members:
                others = elsewhere(mid, batch)
                issues["records_unsealed"].append({"member_id": mid, "reason": "not_in_sealed_batch",
                                                   "sealed_in": others})
                own = decision.epoch
                mixed = [o for o in others if own is None or o["version_id"] != own.version_id]
                if mixed:
                    # handed over by a seal under another epoch: this step's batch mixes epochs
                    issues["epoch_mixed"].append({"member_id": mid,
                                                  "batch_epoch": {"seq": own.seq, "version_id": own.version_id}
                                                  if own is not None else None, "sealed_in": mixed})
                rows.append(ConsumedRecord(member_id=mid, group_id=line.group_id, execution_id=None, score_id=None,
                                           reward=line.reward, sealed_reward=None, advantage=line.advantage,
                                           loss_tokens=line.loss_tokens, identity=line.identity,
                                           shards=tuple(sorted(agg["shards"]))))
                continue
            group, member = members[mid]
            for (part, shard, mb), use in agg["uses"].items():
                if use["handed"] is not None and use["loss_tokens"] != use["handed"]:
                    issues["loss_tokens_mismatch"].append({"member_id": mid, "step_part": part, "shard": shard,
                                                           "microbatch": mb, "handed_over": use["handed"],
                                                           "learner": use["loss_tokens"]})
            counted = [u["loss_tokens"] for u in agg["uses"].values()]
            if counted and all(c is not None for c in counted) and sum(counted) == 0:
                # removed from the loss: a stamp of 0 and a learner count of 0 agree, but the sealed record trained
                # on nothing
                issues["zero_loss_tokens"].append({"member_id": mid, "loss_tokens": 0, "handed_over": sorted(
                    {u["handed"] for u in agg["uses"].values()}, key=lambda v: (v is None, v))})
            if learner_groups.get(line.group_id, set()) != groups[group.group_id] or line.group_id != group.group_id:
                issues["group_mismatch"].append({"member_id": mid, "sealed_group": group.group_id,
                                                 "learner_group": line.group_id,
                                                 "sealed_members": sorted(groups[group.group_id]),
                                                 "learner_members": sorted(learner_groups.get(line.group_id, ()))})
            sealed_reward = member.trained_reward
            if line.reward is None or sealed_reward is None or abs(line.reward - sealed_reward) > tolerance:
                issues["reward_mismatch"].append({"member_id": mid, "sealed": sealed_reward,
                                                  "learner": line.reward})
            if line.identity != "verified":
                issues["identity_unverified"].append({"member_id": mid, "identity": line.identity})
            if mid in expected and line.normalized is not None:
                mean, mean_std = expected[mid]
                errors["mean"] += (line.normalized - mean) ** 2
                errors["mean_std"] += (line.normalized - mean_std) ** 2
                advantages = [x.advantage for x in agg["lines"] if x.advantage is not None]
                lows = [x.advantage_min for x in agg["lines"] if x.advantage_min is not None]
                highs = [x.advantage_max for x in agg["lines"] if x.advantage_max is not None]
                checks.append((mid, line.normalized, advantages, min(lows, default=None), max(highs, default=None),
                               mean, mean_std))
            rows.append(ConsumedRecord(member_id=mid, group_id=line.group_id, execution_id=member.execution_id,
                                       score_id=member.score_id, reward=line.reward, sealed_reward=sealed_reward,
                                       advantage=line.advantage, loss_tokens=sum(
                                           u["loss_tokens"] or 0 for u in agg["uses"].values()),
                                       identity=line.identity, shards=tuple(sorted(agg["shards"]))))
        staged.append((batch, decision, issues, rows, checks))

    hypothesis = normalization if normalization != "auto" else min(errors, key=lambda k: (errors[k], k != "mean_std"))
    worst = 0.0
    records = []
    steps = []
    labels: dict[str, int] = defaultdict(int)
    by_version: dict[str, int] = defaultdict(int)
    for batch, decision, issues, rows, checks in staged:
        for mid, reward, advantages, low, high, mean, mean_std in checks:
            want = mean_std if hypothesis == "mean_std" else mean
            problems = []
            if abs(reward - want) > tolerance:
                problems.append("normalized_reward_not_group_normalized")
            if any(abs(a - reward) > tolerance for a in advantages):
                problems.append("advantage_not_normalized_reward")
            if low is not None and high is not None and high - low > tolerance:
                problems.append("advantage_not_constant")
            worst = max(worst, abs(reward - want), *(abs(a - reward) for a in advantages))
            if problems:
                issues["advantage_inconsistent"].append({"member_id": mid, "normalized": reward, "expected": want,
                                                         "advantages": advantages, "range": [low, high],
                                                         "problems": problems})
        completion = completed[batch]
        record = ConsumptionRecord(batch_id=batch, step=completion.step,
                                   seal_id=decision.seal_id if decision else "",
                                   decision_id=decision.decision_id if decision else "",
                                   epoch=decision.epoch if decision else None, completed_at=completion.completed_at,
                                   records=tuple(rows), issues={k: v for k, v in issues.items()})
        records.append(record)
        if decision is not None:
            members = {m.member_id: (g, m) for g, m in decision.members()}
            for row in rows:
                if row.member_id in members:
                    g, m = members[row.member_id]
                    labels["neutralized" if g.neutralized else m.action] += 1
                    by_version[decision.epoch.version_id if decision.epoch else "none"] += 1
        steps.append({"batch_id": batch, "step": completion.step, "records": len(rows), "ok": record.ok,
                      "consumption_id": record.consumption_id,
                      "epoch_seq": decision.epoch.seq if decision and decision.epoch else None,
                      "issues": {k: len(v) for k, v in issues.items() if v}})
    totals = {k: sum(len(r.issues.get(k, ())) for r in records) for k in ISSUE_KEYS}
    incomplete = tuple(sorted(set(by_batch) - set(completed)))
    unconfirmed = tuple(sorted(set(completed) - set(by_batch)))
    unconsumed = tuple(sorted(set(sealed) - set(completed) - set(by_batch)))
    manifest = TrainingManifest(
        steps=tuple(steps), consumption_ids=tuple(r.consumption_id for r in records),
        records_consumed=sum(len(r.records) for r in records), labels=dict(labels),
        by_grader_version=dict(by_version), issues=totals, incomplete_batches=incomplete,
        unconfirmed_batches=unconfirmed, unconsumed_seals=unconsumed, resealed_batches=tuple(sorted(resealed)),
        normalization={"hypothesis": hypothesis, "squared_error": errors, "max_abs_error": worst},
        # a completed step whose consumption no log confirms is a gap, not a pass
        ok=all(r.ok for r in records) and not unconfirmed, meta=meta or {})
    return TrainingReport(records=tuple(records), manifest=manifest)


def training_manifest(decisions: Sequence[SealDecision], lines: Iterable[LearnerLine | Mapping[str, Any]],
                      completions: Iterable[StepCompletion | Mapping[str, Any]], **kwargs: Any) -> TrainingManifest:
    """The training consumption manifest alone (see :func:`consumption_records`)."""
    return consumption_records(decisions, lines, completions, **kwargs).manifest
