# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The four record types of agent-ecc, and the values they are built from.

* :class:`ExecutionRecord`   what happened: one trajectory's evidence (by digest) and the test runs made when it was
                              graded. Immutable and content-addressed; never revised.
* :class:`ScoreRecord`       a verdict: one execution under one grader version, with per-test outcomes, the method
                              that produced it (recorded / reparse / reexec / undeterminable) and its cost.
* :class:`SealDecision`      what one seal intends to train: the epoch it read once, a decision for every member,
                              neutralized groups, groups filtered out after the decision, backfill and fillers.
* :class:`ConsumptionRecord` what one learner step actually consumed, confirmed by the learner's per-record log and the
                              step's completion. A SealDecision alone is not consumption.

Every record type is a frozen dataclass. Its identity (``record_id``, ``score_id``, ``decision_id``,
``consumption_id``) is the digest of its canonical JSON body, so equal content means equal identity and a record that
is changed is a different record. ``to_dict()`` adds the identity; ``from_dict()`` verifies it.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, TypeVar

from .canonical import FrozenDict, digest_json, freeze, is_digest, thaw

__all__ = [
    "ConsumedRecord",
    "ConsumptionRecord",
    "Cost",
    "EpochRef",
    "EvidenceRef",
    "Exclusion",
    "ExecutionRecord",
    "GroupDecision",
    "IdentityMismatch",
    "MemberDecision",
    "Method",
    "Outcome",
    "PartScore",
    "RUN_ERROR",
    "RUN_OK",
    "ScoreRecord",
    "SealDecision",
    "TargetPin",
    "TestOutcome",
    "TestRun",
    "dump_jsonl",
    "load_jsonl",
    "record_from_dict",
]

SCHEMA_EXECUTION = "agent-ecc/execution/1"
SCHEMA_SCORE = "agent-ecc/score/1"
SCHEMA_SEAL = "agent-ecc/seal/1"
SCHEMA_CONSUMPTION = "agent-ecc/consumption/1"

RUN_OK = "ok"  # the run produced its outputs; parsing decides whether they hold a logical verdict
RUN_ERROR = "error"  # the run did not produce usable outputs (infrastructure); never a verdict


class IdentityMismatch(ValueError):
    """A serialized record's identity does not match its content."""


class Outcome(str, enum.Enum):
    """A verdict. Only PASS and FAIL are logical verdicts; UNDETERMINABLE is the absence of one."""

    PASS = "pass"
    FAIL = "fail"
    UNDETERMINABLE = "undeterminable"

    @property
    def logical(self) -> bool:
        return self is not Outcome.UNDETERMINABLE


class Method(str, enum.Enum):
    """How a verdict was obtained under a grader version."""

    RECORDED = "recorded"  # taken from a verdict recorded earlier, whose identities equal the target's
    REPARSE = "reparse"  # recomputed in-process from recorded outputs (no execution)
    REEXEC = "reexec"  # recomputed by executing the target's tests on the recorded artifact
    UNDETERMINABLE = "undeterminable"  # no verdict could be obtained

    @property
    def rank(self) -> int:
        return _METHOD_RANK[self]


_METHOD_RANK = {Method.RECORDED: 0, Method.REPARSE: 1, Method.REEXEC: 2, Method.UNDETERMINABLE: 3}

T = TypeVar("T")


def _set(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


def _tuple_of(cls: type[T], items: Iterable[Any] | None) -> tuple[T, ...]:
    out = []
    for item in items or ():
        out.append(item if isinstance(item, cls) else cls.from_dict(item))  # type: ignore[attr-defined]
    return tuple(out)


def _opt(cls: type[T], value: Any) -> T | None:
    if value is None or isinstance(value, cls):
        return value
    return cls.from_dict(value)  # type: ignore[attr-defined]


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def _verify(cls_name: str, body: Mapping[str, Any], given: str | None, computed: str) -> None:
    if given is not None and given != computed:
        raise IdentityMismatch(f"{cls_name} identity {given} does not match its content ({computed})")


# ---------------------------------------------------------------------------------------------------------- evidence


@dataclass(frozen=True)
class EvidenceRef:
    """A piece of evidence by content address. ``role`` names what it is (patch, test_log, answer, end_state, ...)."""

    role: str
    digest: str
    size: int | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        if not is_digest(self.digest):
            raise ValueError(f"evidence {self.role!r}: not a sha256 digest: {self.digest!r}")

    def to_dict(self) -> dict:
        return {"role": self.role, "digest": self.digest, "size": self.size, "media_type": self.media_type}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EvidenceRef:
        return cls(role=d["role"], digest=d["digest"], size=d.get("size"), media_type=d.get("media_type"))


@dataclass(frozen=True)
class TestRun:
    """One execution of a test suite against an artifact, and its raw outputs.

    ``execution`` is the execution identity of what ran (for SWE-bench: the image, the eval script and the
    patch-application commands, by digest). Outputs are raw (a test log, per-test outcomes); a run never carries a
    verdict: verdicts are derived from outputs under a grading identity, in a :class:`ScoreRecord`.
    """

    __test__ = False  # not a pytest test class

    check: str
    execution: Mapping[str, str]
    status: str = RUN_OK
    outputs: tuple[EvidenceRef, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if self.status not in (RUN_OK, RUN_ERROR):
            raise ValueError(f"unknown run status {self.status!r}")
        _set(self, "execution", freeze(self.execution))
        _set(self, "outputs", _tuple_of(EvidenceRef, self.outputs))
        _set(self, "facts", freeze(self.facts))

    @cached_property
    def execution_digest(self) -> str:
        return digest_json(self.execution)

    @cached_property
    def run_digest(self) -> str:
        return digest_json(self.to_dict())

    def output(self, role: str) -> EvidenceRef | None:
        return next((ref for ref in self.outputs if ref.role == role), None)

    def to_dict(self) -> dict:
        return {"check": self.check, "execution": thaw(self.execution), "status": self.status,
                "outputs": [o.to_dict() for o in self.outputs], "facts": thaw(self.facts)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TestRun:
        return cls(check=d["check"], execution=d.get("execution") or {}, status=d.get("status", RUN_OK),
                   outputs=d.get("outputs") or (), facts=d.get("facts") or {})


@dataclass(frozen=True)
class ExecutionRecord:
    """What one trajectory did, immutable and content-addressed.

    ``world_version`` names the one world version the trajectory resolved when it started and observed throughout
    (invariant I1). ``world_observations`` lists versions seen at later serving events, if the producer records them;
    a record that observed another version than the one it names is *torn* and no grader verdict is defined for it.
    ``artifacts`` is what the agent produced (a patch, a submitted answer, an end state), ``runs`` the test executions
    made when it was graded. ``facts`` holds small producer facts with meaning to an adapter (for example that the
    agent's patch was empty); ``meta`` is free-form and has no semantics.
    """

    family: str
    task_id: str
    trajectory_id: str
    world_version: str | None = None
    world_observations: tuple[str, ...] = ()
    artifacts: tuple[EvidenceRef, ...] = ()
    runs: tuple[TestRun, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=FrozenDict)
    meta: Mapping[str, Any] = field(default_factory=FrozenDict)
    schema: str = SCHEMA_EXECUTION

    def __post_init__(self) -> None:
        if self.schema != SCHEMA_EXECUTION:
            raise ValueError(f"unsupported execution record schema {self.schema!r}")
        _set(self, "world_observations", tuple(str(w) for w in self.world_observations))
        _set(self, "artifacts", _tuple_of(EvidenceRef, self.artifacts))
        _set(self, "runs", _tuple_of(TestRun, self.runs))
        _set(self, "facts", freeze(self.facts))
        _set(self, "meta", freeze(self.meta))

    def body(self) -> dict:
        return {"schema": self.schema, "family": self.family, "task_id": self.task_id,
                "trajectory_id": self.trajectory_id, "world_version": self.world_version,
                "world_observations": list(self.world_observations),
                "artifacts": [a.to_dict() for a in self.artifacts], "runs": [r.to_dict() for r in self.runs],
                "facts": thaw(self.facts), "meta": thaw(self.meta)}

    @cached_property
    def record_id(self) -> str:
        return digest_json(self.body())

    def to_dict(self) -> dict:
        return {**self.body(), "record_id": self.record_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, verify: bool = True) -> ExecutionRecord:
        record = cls(family=d["family"], task_id=d["task_id"], trajectory_id=d["trajectory_id"],
                     world_version=d.get("world_version"), world_observations=tuple(d.get("world_observations") or ()),
                     artifacts=d.get("artifacts") or (), runs=d.get("runs") or (), facts=d.get("facts") or {},
                     meta=d.get("meta") or {}, schema=d.get("schema", SCHEMA_EXECUTION))
        if verify:
            _verify("ExecutionRecord", d, d.get("record_id"), record.record_id)
        return record

    def artifact(self, role: str) -> EvidenceRef | None:
        return next((a for a in self.artifacts if a.role == role), None)

    def evidence(self) -> tuple[EvidenceRef, ...]:
        """Every evidence reference the record holds: its artifacts and its runs' outputs."""
        return self.artifacts + tuple(o for r in self.runs for o in r.outputs)

    @property
    def torn(self) -> bool:
        """True when the record observed a world version other than the one it names (invariant I1 violated)."""
        return any(w != self.world_version for w in self.world_observations)


# ---------------------------------------------------------------------------------------------------------- scores


@dataclass(frozen=True)
class TestOutcome:
    """One test's status in a run, as the grader read it (passed, failed, error, skipped, xfail, missing)."""

    __test__ = False

    name: str
    status: str
    group: str | None = None

    @property
    def passed(self) -> bool:
        return self.status in ("passed", "xfail")

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "group": self.group}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TestOutcome:
        return cls(name=d["name"], status=d["status"], group=d.get("group"))


@dataclass(frozen=True)
class PartScore:
    """The verdict of one check (one atomic test spec) of a grader version's spec.

    ``execution_digest`` and ``grading_digest`` are the identities of the check that produced it: a later grader whose
    check has the same two identities has the same verdict, which is what lets a verdict be reused.
    """

    check: str
    execution_digest: str
    grading_digest: str
    outcome: Outcome
    method: Method
    reason: str | None = None
    tests: tuple[TestOutcome, ...] = ()
    run: TestRun | None = None
    attempts: int = 0
    seconds: float = 0.0

    def __post_init__(self) -> None:
        _set(self, "outcome", Outcome(self.outcome))
        _set(self, "method", Method(self.method))
        _set(self, "tests", _tuple_of(TestOutcome, self.tests))
        _set(self, "run", _opt(TestRun, self.run))
        _set(self, "seconds", float(self.seconds))

    @property
    def logical(self) -> bool:
        return self.outcome.logical

    def to_dict(self) -> dict:
        return {"check": self.check, "execution_digest": self.execution_digest,
                "grading_digest": self.grading_digest, "outcome": self.outcome.value, "method": self.method.value,
                "reason": self.reason, "tests": [t.to_dict() for t in self.tests],
                "run": self.run.to_dict() if self.run is not None else None, "attempts": self.attempts,
                "seconds": self.seconds}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PartScore:
        return cls(check=d["check"], execution_digest=d["execution_digest"], grading_digest=d["grading_digest"],
                   outcome=d["outcome"], method=d["method"], reason=d.get("reason"), tests=d.get("tests") or (),
                   run=d.get("run"), attempts=int(d.get("attempts") or 0), seconds=float(d.get("seconds") or 0.0))


@dataclass(frozen=True)
class Cost:
    """What obtaining a verdict cost: wall seconds, test executions, sandbox seconds, and executions served by sharing
    the result of an identical request."""

    seconds: float = 0.0
    executions: int = 0
    container_seconds: float = 0.0
    shared: int = 0

    def __post_init__(self) -> None:
        _set(self, "seconds", float(self.seconds))
        _set(self, "container_seconds", float(self.container_seconds))

    def __add__(self, other: Cost) -> Cost:
        return Cost(self.seconds + other.seconds, self.executions + other.executions,
                    self.container_seconds + other.container_seconds, self.shared + other.shared)

    def to_dict(self) -> dict:
        return {"seconds": self.seconds, "executions": self.executions,
                "container_seconds": self.container_seconds, "shared": self.shared}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Cost:
        return cls(seconds=d.get("seconds") or 0.0, executions=int(d.get("executions") or 0),
                   container_seconds=d.get("container_seconds") or 0.0, shared=int(d.get("shared") or 0))


@dataclass(frozen=True)
class ScoreRecord:
    """One execution's verdict under one grader version.

    ``grader_version`` is the version id the verdict is for (None for a verdict recorded under a grader that was never
    published as a version; its parts still carry the identities that graded it). ``parts`` are the evaluated checks in
    order; ``skipped`` names checks a conjunction never evaluated because an earlier check failed logically.
    ``based_on`` is the score a reused verdict was taken from.
    """

    execution_id: str
    grader_version: str | None
    outcome: Outcome
    method: Method
    spec_digest: str | None = None
    reason: str | None = None
    parts: tuple[PartScore, ...] = ()
    skipped: tuple[str, ...] = ()
    cost: Cost = field(default_factory=Cost)
    based_on: str | None = None
    meta: Mapping[str, Any] = field(default_factory=FrozenDict)
    schema: str = SCHEMA_SCORE

    def __post_init__(self) -> None:
        if self.schema != SCHEMA_SCORE:
            raise ValueError(f"unsupported score record schema {self.schema!r}")
        _set(self, "outcome", Outcome(self.outcome))
        _set(self, "method", Method(self.method))
        _set(self, "parts", _tuple_of(PartScore, self.parts))
        _set(self, "skipped", tuple(str(s) for s in self.skipped))
        _set(self, "cost", self.cost if isinstance(self.cost, Cost) else Cost.from_dict(self.cost))
        _set(self, "meta", freeze(self.meta))
        if self.outcome is Outcome.UNDETERMINABLE and self.method is not Method.UNDETERMINABLE:
            raise ValueError("an undeterminable outcome has method undeterminable")
        if self.outcome.logical and self.method is Method.UNDETERMINABLE:
            raise ValueError("a logical verdict needs a method other than undeterminable")

    def body(self) -> dict:
        return {"schema": self.schema, "execution_id": self.execution_id, "grader_version": self.grader_version,
                "spec_digest": self.spec_digest, "outcome": self.outcome.value, "method": self.method.value,
                "reason": self.reason, "parts": [p.to_dict() for p in self.parts], "skipped": list(self.skipped),
                "cost": self.cost.to_dict(), "based_on": self.based_on, "meta": thaw(self.meta)}

    @cached_property
    def score_id(self) -> str:
        return digest_json(self.body())

    def to_dict(self) -> dict:
        return {**self.body(), "score_id": self.score_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, verify: bool = True) -> ScoreRecord:
        record = cls(execution_id=d["execution_id"], grader_version=d.get("grader_version"), outcome=d["outcome"],
                     method=d["method"], spec_digest=d.get("spec_digest"), reason=d.get("reason"),
                     parts=d.get("parts") or (), skipped=tuple(d.get("skipped") or ()),
                     cost=Cost.from_dict(d.get("cost") or {}), based_on=d.get("based_on"), meta=d.get("meta") or {},
                     schema=d.get("schema", SCHEMA_SCORE))
        if verify:
            _verify("ScoreRecord", d, d.get("score_id"), record.score_id)
        return record

    @property
    def passed(self) -> bool | None:
        """True / False for a logical verdict, None when undeterminable."""
        return None if self.outcome is Outcome.UNDETERMINABLE else self.outcome is Outcome.PASS

    @property
    def logical(self) -> bool:
        return self.outcome.logical

    def tests(self) -> tuple[TestOutcome, ...]:
        return tuple(t for p in self.parts for t in p.tests)

    def new_runs(self) -> tuple[TestRun, ...]:
        """Runs this score created by re-executing tests: new evidence a later version may re-parse."""
        return tuple(p.run for p in self.parts if p.method is Method.REEXEC and p.run is not None)

    def same_verdict(self, other: ScoreRecord) -> bool:
        """Equal outcome and equal per-check outcomes (cost, timing and provenance aside)."""
        mine = [(p.execution_digest, p.grading_digest, p.outcome) for p in self.parts]
        theirs = [(p.execution_digest, p.grading_digest, p.outcome) for p in other.parts]
        return self.outcome is other.outcome and mine == theirs and self.skipped == other.skipped


# ---------------------------------------------------------------------------------------------------------- seals


@dataclass(frozen=True)
class EpochRef:
    """A published grader version as one seal read it: publication sequence, version id, name, publication time and
    the time the pointer was read (the batch's linearization point)."""

    seq: int
    version_id: str
    name: str | None = None
    published_at: float | None = None
    read_at: float | None = None

    def to_dict(self) -> dict:
        return {"seq": self.seq, "version_id": self.version_id, "name": self.name,
                "published_at": self.published_at, "read_at": self.read_at}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EpochRef:
        return cls(seq=int(d["seq"]), version_id=d["version_id"], name=d.get("name"),
                   published_at=_float(d.get("published_at")), read_at=_float(d.get("read_at")))


MEMBER_ACTIONS = ("reuse", "reparse", "reexec", "undeterminable", "ungoverned", "pending", "off", "failed")
_UNDETERMINABLE_STATUS = {"unknown_world": "unknown_world", "torn_observation": "torn", "no_epoch": "no_epoch",
                          "no_record": "no_record", "no_adapter": "no_adapter"}


@dataclass(frozen=True)
class MemberDecision:
    """A seal's decision for one member of a group.

    ``action`` is how its label under the batch epoch was decided: reuse (verdicts recorded earlier with equal
    identities), reparse, reexec, undeterminable, ungoverned (the epoch makes no claim about the task: the recorded
    label stands), pending (audit mode: the label would need an execution the mode does not run), off, or failed (the
    seal failed). ``outcome`` is the target verdict, ``target_reward`` the reward it implies, ``trained_reward`` what
    the seal hands to the learner after neutralization (and, in audit or off mode, the recorded reward).
    ``consistent`` is true when the member was graded under exactly the batch epoch's spec: its label is the recorded
    verdict itself, reused as a whole, so its recorded reward is the target reward. ``short_circuit`` is true when a
    logical fail decided the member's conjunction while some of its checks were never evaluated.
    """

    member_id: str
    execution_id: str | None
    action: str
    outcome: Outcome | None = None
    reason: str | None = None
    score_id: str | None = None
    recorded_reward: float | None = None
    target_reward: float | None = None
    trained_reward: float | None = None
    flipped: bool | None = None
    short_circuit: bool = False
    consistent: bool = False

    def __post_init__(self) -> None:
        if self.action not in MEMBER_ACTIONS:
            raise ValueError(f"unknown member action {self.action!r}")
        _set(self, "outcome", None if self.outcome is None else Outcome(self.outcome))
        for name in ("recorded_reward", "target_reward", "trained_reward"):
            _set(self, name, _float(getattr(self, name)))
        if self.consistent and self.action != "reuse":
            raise ValueError("only a reused verdict can be consistent")

    @property
    def status(self) -> str:
        """A compact status name. ``same`` / ``flip`` compare the target verdict with the recorded label:

        * consistent: graded under exactly the batch epoch's spec (the recorded verdict, reused as a whole);
        * repaired_reuse_same / _flip: decided from earlier verdicts alone, but not the recorded verdict of this very
          spec (a verdict from a later score, parts of a larger spec, a rollback);
        * repaired_short_same / _flip: a logical fail decided it while other checks were never evaluated, and nothing
          was executed;
        * repaired_same / _flip: a re-parse took part (no execution); repaired_exec_same / _flip: an execution did;
        * ungoverned, unknown_world, torn, no_epoch, no_record, unrepairable (no label), needs_reexec / needs_reparse
          (pending), off, failed."""
        a = self.action
        change = "flip" if self.flipped else "same"
        if a == "reuse":
            if self.consistent:
                return "consistent"
            return f"repaired_short_{change}" if self.short_circuit else f"repaired_reuse_{change}"
        if a == "reparse":
            return f"repaired_short_{change}" if self.short_circuit else f"repaired_{change}"
        if a == "reexec":
            return f"repaired_exec_{change}"
        if a == "undeterminable":
            return _UNDETERMINABLE_STATUS.get(self.reason or "", "unrepairable")
        if a == "pending":
            return "needs_reparse" if self.reason == "needs_reparse" else "needs_reexec"
        return a

    @property
    def decided(self) -> bool:
        """True when the member's label under the batch epoch is defined (a logical target verdict)."""
        return self.outcome is not None and self.outcome.logical

    def to_dict(self) -> dict:
        return {"member_id": self.member_id, "execution_id": self.execution_id, "action": self.action,
                "status": self.status, "outcome": None if self.outcome is None else self.outcome.value,
                "reason": self.reason, "score_id": self.score_id, "recorded_reward": self.recorded_reward,
                "target_reward": self.target_reward, "trained_reward": self.trained_reward, "flipped": self.flipped,
                "short_circuit": self.short_circuit, "consistent": self.consistent}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> MemberDecision:
        return cls(member_id=d["member_id"], execution_id=d.get("execution_id"), action=d["action"],
                   outcome=d.get("outcome"), reason=d.get("reason"), score_id=d.get("score_id"),
                   recorded_reward=d.get("recorded_reward"), target_reward=d.get("target_reward"),
                   trained_reward=d.get("trained_reward"), flipped=d.get("flipped"),
                   short_circuit=bool(d.get("short_circuit")), consistent=bool(d.get("consistent")))


GROUP_ROLES = ("selected", "backfill", "filler")


@dataclass(frozen=True)
class GroupDecision:
    """A decided group (one GRPO unit): its members, the decision round (1 for the initial selection, later rounds for
    backfill), its role in the batch, and whether its rewards were neutralized (all equal, zero advantage)."""

    group_id: str
    members: tuple[MemberDecision, ...]
    round: int = 1
    role: str = "selected"
    source: str | None = None
    neutralized: bool = False
    neutralize_reason: str | None = None

    def __post_init__(self) -> None:
        if self.role not in GROUP_ROLES:
            raise ValueError(f"unknown group role {self.role!r}")
        _set(self, "members", _tuple_of(MemberDecision, self.members))

    def rewards(self) -> list[float | None]:
        """The rewards this group trains with, member by member."""
        return [m.trained_reward for m in self.members]

    def replace(self, **changes: Any) -> GroupDecision:
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return {"group_id": self.group_id, "members": [m.to_dict() for m in self.members], "round": self.round,
                "role": self.role, "source": self.source, "neutralized": self.neutralized,
                "neutralize_reason": self.neutralize_reason}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GroupDecision:
        return cls(group_id=d["group_id"], members=d.get("members") or (), round=int(d.get("round") or 1),
                   role=d.get("role", "selected"), source=d.get("source"), neutralized=bool(d.get("neutralized")),
                   neutralize_reason=d.get("neutralize_reason"))


@dataclass(frozen=True)
class Exclusion:
    """A group the seal decided and then left out of the batch (e.g. no reward variance under the batch epoch)."""

    group: GroupDecision
    reason: str

    def to_dict(self) -> dict:
        return {"group": self.group.to_dict(), "reason": self.reason}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Exclusion:
        return cls(group=GroupDecision.from_dict(d["group"]), reason=d["reason"])


@dataclass(frozen=True)
class TargetPin:
    """A seal whose batch epoch was pinned instead of read from the pointer (the ``never`` arm of a grader switch).

    ``pin`` names the pinned epoch: ``"e0"`` (the first grader publication, the startup epoch) or a grader
    publication's ``seq``. ``resolved`` is false when the publication log held no such epoch at the seal's read (the
    batch then has no epoch, and repair neutralizes it). ``published`` is what the pointer named at the same read: the
    epoch an unpinned seal would have used. A pinned seal keeps I1 to I3; it relaxes I4's adoption on purpose, and
    this record is what lets an audit tell the two apart (:func:`agent_ecc.audit.audit_epochs`)."""

    pin: str | int
    resolved: bool
    published: EpochRef | None = None

    def __post_init__(self) -> None:
        _set(self, "published", _opt(EpochRef, self.published))

    def to_dict(self) -> dict:
        return {"pin": self.pin, "resolved": self.resolved,
                "published": self.published.to_dict() if self.published else None}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TargetPin:
        return cls(pin=d["pin"], resolved=bool(d["resolved"]), published=d.get("published"))


@dataclass(frozen=True)
class SealDecision:
    """What one seal intends to train. ``groups`` is the batch in hand-off order, every member decided under
    ``epoch``, the one epoch the seal read. ``excluded`` holds decided groups left out of the batch, ``target_groups``
    the batch size the caller asked for. ``failed`` is set when the seal could not finish; the batch is then
    neutralized (strict) or trains recorded labels (non-strict). ``target_pin`` is set only for a seal whose batch
    epoch was pinned (:class:`TargetPin`); ``epoch`` is then the pinned epoch and ``target_pin.published`` what the
    pointer named. A decision without a pin has no ``target_pin`` key in its body, so its identity is the one it had
    before pins existed."""

    seal_id: str
    epoch: EpochRef | None
    mode: str
    policy: str
    strict: bool
    groups: tuple[GroupDecision, ...]
    batch_id: str | None = None
    excluded: tuple[Exclusion, ...] = ()
    target_groups: int | None = None
    rounds: int = 1
    failed: str | None = None
    opened_at: float | None = None
    closed_at: float | None = None
    stats: Mapping[str, Any] = field(default_factory=FrozenDict)
    target_pin: TargetPin | None = None
    schema: str = SCHEMA_SEAL

    def __post_init__(self) -> None:
        if self.schema != SCHEMA_SEAL:
            raise ValueError(f"unsupported seal decision schema {self.schema!r}")
        _set(self, "epoch", _opt(EpochRef, self.epoch))
        _set(self, "groups", _tuple_of(GroupDecision, self.groups))
        _set(self, "excluded", _tuple_of(Exclusion, self.excluded))
        _set(self, "stats", freeze(self.stats))
        _set(self, "target_pin", _opt(TargetPin, self.target_pin))

    def body(self) -> dict:
        body = {"schema": self.schema, "seal_id": self.seal_id, "batch_id": self.batch_id,
                "epoch": self.epoch.to_dict() if self.epoch else None, "mode": self.mode, "policy": self.policy,
                "strict": self.strict, "groups": [g.to_dict() for g in self.groups],
                "excluded": [e.to_dict() for e in self.excluded], "target_groups": self.target_groups,
                "rounds": self.rounds, "failed": self.failed, "opened_at": self.opened_at,
                "closed_at": self.closed_at, "stats": thaw(self.stats)}
        if self.target_pin is not None:
            body["target_pin"] = self.target_pin.to_dict()
        return body

    @cached_property
    def decision_id(self) -> str:
        return digest_json(self.body())

    def to_dict(self) -> dict:
        return {**self.body(), "decision_id": self.decision_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, verify: bool = True) -> SealDecision:
        record = cls(seal_id=d["seal_id"], epoch=d.get("epoch"), mode=d["mode"], policy=d["policy"],
                     strict=bool(d["strict"]), groups=d.get("groups") or (), batch_id=d.get("batch_id"),
                     excluded=d.get("excluded") or (), target_groups=d.get("target_groups"),
                     rounds=int(d.get("rounds") or 1), failed=d.get("failed"), opened_at=_float(d.get("opened_at")),
                     closed_at=_float(d.get("closed_at")), stats=d.get("stats") or {},
                     target_pin=d.get("target_pin"), schema=d.get("schema", SCHEMA_SEAL))
        if verify:
            _verify("SealDecision", d, d.get("decision_id"), record.decision_id)
        return record

    @property
    def shortfall(self) -> int:
        """Groups missing from the batch the caller asked for (fillers off and backfill exhausted)."""
        if self.target_groups is None:
            return 0
        return max(0, int(self.target_groups) - len(self.groups))

    @property
    def neutralized_groups(self) -> int:
        return sum(1 for g in self.groups if g.neutralized)

    def members(self) -> Iterator[tuple[GroupDecision, MemberDecision]]:
        for group in self.groups:
            for member in group.members:
                yield group, member

    def rewards(self) -> dict[str, float | None]:
        """member id -> the reward the learner should train it with."""
        return {m.member_id: m.trained_reward for _, m in self.members()}


# ---------------------------------------------------------------------------------------------------------- consumption


@dataclass(frozen=True)
class ConsumedRecord:
    """One record a learner step consumed: the reward it trained with as the learner logged it, the label the seal
    decided for it, its advantage and loss tokens, and whether its identity was verified (the tokens trained are the
    tokens handed over)."""

    member_id: str
    group_id: str | None
    execution_id: str | None
    score_id: str | None
    reward: float | None
    sealed_reward: float | None
    advantage: float | None = None
    loss_tokens: int | None = None
    identity: str | None = None
    shards: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _set(self, "reward", _float(self.reward))
        _set(self, "sealed_reward", _float(self.sealed_reward))
        _set(self, "advantage", _float(self.advantage))
        _set(self, "shards", tuple(str(s) for s in self.shards))

    def to_dict(self) -> dict:
        return {"member_id": self.member_id, "group_id": self.group_id, "execution_id": self.execution_id,
                "score_id": self.score_id, "reward": self.reward, "sealed_reward": self.sealed_reward,
                "advantage": self.advantage, "loss_tokens": self.loss_tokens, "identity": self.identity,
                "shards": list(self.shards)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ConsumedRecord:
        return cls(member_id=d["member_id"], group_id=d.get("group_id"), execution_id=d.get("execution_id"),
                   score_id=d.get("score_id"), reward=d.get("reward"), sealed_reward=d.get("sealed_reward"),
                   advantage=d.get("advantage"), loss_tokens=d.get("loss_tokens"), identity=d.get("identity"),
                   shards=tuple(d.get("shards") or ()))


@dataclass(frozen=True)
class ConsumptionRecord:
    """What one completed learner step consumed, joined with the seal decision that handed it over.

    ``issues`` maps a check name (records_unsealed, records_unconsumed, group_mismatch, reward_mismatch,
    identity_unverified, duplicate_consumption, loss_tokens_mismatch, advantage_inconsistent, zero_loss_tokens,
    epoch_mixed; :data:`agent_ecc.export.ISSUE_KEYS`) to the issues found; the record is ``ok`` when every list is
    empty."""

    batch_id: str
    step: int | str | None
    seal_id: str
    decision_id: str
    epoch: EpochRef | None
    completed_at: float | None
    records: tuple[ConsumedRecord, ...]
    issues: Mapping[str, Any] = field(default_factory=FrozenDict)
    schema: str = SCHEMA_CONSUMPTION

    def __post_init__(self) -> None:
        if self.schema != SCHEMA_CONSUMPTION:
            raise ValueError(f"unsupported consumption record schema {self.schema!r}")
        _set(self, "epoch", _opt(EpochRef, self.epoch))
        _set(self, "records", _tuple_of(ConsumedRecord, self.records))
        _set(self, "issues", freeze(self.issues))
        _set(self, "completed_at", _float(self.completed_at))

    def body(self) -> dict:
        return {"schema": self.schema, "batch_id": self.batch_id, "step": self.step, "seal_id": self.seal_id,
                "decision_id": self.decision_id, "epoch": self.epoch.to_dict() if self.epoch else None,
                "completed_at": self.completed_at, "records": [r.to_dict() for r in self.records],
                "issues": thaw(self.issues)}

    @cached_property
    def consumption_id(self) -> str:
        return digest_json(self.body())

    def to_dict(self) -> dict:
        return {**self.body(), "consumption_id": self.consumption_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, verify: bool = True) -> ConsumptionRecord:
        record = cls(batch_id=d["batch_id"], step=d.get("step"), seal_id=d["seal_id"], decision_id=d["decision_id"],
                     epoch=d.get("epoch"), completed_at=d.get("completed_at"), records=d.get("records") or (),
                     issues=d.get("issues") or {}, schema=d.get("schema", SCHEMA_CONSUMPTION))
        if verify:
            _verify("ConsumptionRecord", d, d.get("consumption_id"), record.consumption_id)
        return record

    @property
    def ok(self) -> bool:
        return not any(len(v) for v in self.issues.values())


# ---------------------------------------------------------------------------------------------------------- JSONL

_BY_SCHEMA: dict[str, Any] = {
    SCHEMA_EXECUTION: ExecutionRecord,
    SCHEMA_SCORE: ScoreRecord,
    SCHEMA_SEAL: SealDecision,
    SCHEMA_CONSUMPTION: ConsumptionRecord,
}


def record_from_dict(d: Mapping[str, Any], *, verify: bool = True) -> Any:
    """Load any of the four record types by its ``schema`` field."""
    cls = _BY_SCHEMA.get(d.get("schema"))  # type: ignore[arg-type]
    if cls is None:
        raise ValueError(f"unknown record schema {d.get('schema')!r}")
    return cls.from_dict(d, verify=verify)


def dump_jsonl(path: str | os.PathLike, records: Iterable[Any]) -> int:
    """Append records (anything with ``to_dict``) to a JSON-lines file in one write. Returns lines written."""
    data = "".join(json.dumps(r.to_dict(), sort_keys=True) + "\n" for r in records)
    if not data:
        return 0
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        view = memoryview(data.encode("utf-8"))
        while view:
            view = view[os.write(fd, view):]
    finally:
        os.close(fd)
    return data.count("\n")


def load_jsonl(path: str | os.PathLike, cls: Any = None, *, verify: bool = True) -> list:
    """Read records from a JSON-lines file. A torn last line (a live writer's partial append) is ignored; a malformed
    line elsewhere is an error. ``cls`` defaults to dispatch by schema."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.split("\n")
    complete, tail = lines[:-1], lines[-1]
    out = []
    for i, line in enumerate(complete):
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(cls.from_dict(d, verify=verify) if cls is not None else record_from_dict(d, verify=verify))
    if tail.strip():
        try:
            d = json.loads(tail)
        except ValueError:
            return out  # a partial append in progress
        out.append(cls.from_dict(d, verify=verify) if cls is not None else record_from_dict(d, verify=verify))
    return out
