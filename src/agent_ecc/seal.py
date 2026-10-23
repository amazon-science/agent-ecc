# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The seal: deciding, under one grader epoch, every label a training batch will consume.

A :class:`SealSession` is one seal of one batch:

1. It reads the epoch pointer **once**, when it opens. That read is the batch's linearization point: every member of
   every group the session decides, however many rounds it takes, is decided under that one epoch ``g_b``.
2. ``decide()`` scores each member under ``g_b`` (reuse, reparse or re-execute, see :mod:`agent_ecc.compare`). In
   repair mode a group with a member whose label cannot be decided under ``g_b`` is **neutralized**: every reward in
   it is set to the same value, so it contributes no policy gradient.
3. ``assemble()`` applies a reward-dependent filter (by default: the group's decided rewards must have variance)
   **after** the decision, replaces filtered groups through a caller callback (**backfill**), deciding the new groups
   in the same session and so under the same ``g_b``, and optionally fills remaining slots with neutralized
   **fillers** (off by default: the batch is then delivered short, and the shortfall is recorded).
4. ``finish()`` emits the :class:`agent_ecc.records.SealDecision`: what the batch intends to train. It is not
   consumption; what a learner step actually consumed is established from the learner's own log
   (:mod:`agent_ecc.export`).

Modes: ``repair`` enforces the contract; ``audit`` decides what repair would (re-parsing, never executing) but trains
the recorded labels; ``discard`` recomputes nothing and neutralizes every group holding a member that was not graded
under exactly ``g_b``'s spec (validate-and-abort); ``off`` decides nothing.

A member is **consistent** when it was graded under exactly the batch epoch's spec: its recorded verdict is reused as
a whole (every evaluated check from that one score, with the same spec and outcome). A consistent member trains its
recorded reward (which may be shaped); every other decided member trains the reward of its target verdict.

Strict mode (default): a seal that cannot finish (its decision raises, or its log cannot be written) neutralizes the
whole batch (:meth:`SealSession.fail`); with no epoch published, ``discard`` neutralizes. Non-strict mode falls back to
the recorded labels in those cases.

A **target pin** (``pin="e0"``, or a grader publication's ``seq``) makes the batch epoch the pinned epoch, read from the
publication log, whatever the pointer names; the pointer is still read once, and the decision records both
(:class:`agent_ecc.records.TargetPin`). A pinned seal keeps I1 to I3 and relaxes I4's adoption on purpose: it is the
control of an experiment that asks what adopting a new grader version changes.

A **preparer** (``prepared=``, :class:`agent_ecc.prepare.Preparer`) starts the re-executions a repair seal will need
when a group is admitted. The seal still reads the pointer once and decides every member itself; it takes a prepared
result only for the request it would execute, under its own epoch, and only when the result gives the check a logical
verdict, so preparation changes when executions run, never a label.

The comparison arms of a grader-switch experiment are seal configurations (:class:`Arm`, :func:`arm_session`):
``keep`` (audit), ``discard``, ``full_compatible`` (repair; once the batch epoch is not the startup epoch, every member
is re-verified under every target spec, sharing executions through the scorer's cache), ``ecc`` (repair) and ``never``
(repair with the target pinned to the startup epoch).
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .canonical import FrozenDict, freeze
from .compare import Action, Policy
from .records import (
    Exclusion,
    ExecutionRecord,
    GroupDecision,
    MemberDecision,
    Method,
    Outcome,
    ScoreRecord,
    SealDecision,
    TargetPin,
    load_jsonl,
)
from .recompute.base import _merge_transport
from .scoring import ScoreInput, Scorer, _recorded_score
from .versions import Epoch, iter_checks

__all__ = [
    "Arm",
    "Assembly",
    "Candidate",
    "EXCLUSION_ZERO_VARIANCE",
    "EpochSource",
    "Member",
    "Mode",
    "MonotonicityError",
    "PIN_STARTUP",
    "SealFailed",
    "SealJournal",
    "SealSession",
    "apply_rewards",
    "arm_session",
    "binary_reward",
    "has_variance",
    "pinned_epoch",
    "seal",
]

EXCLUSION_ZERO_VARIANCE = "zero_variance_after_seal"
EXCLUSION_FILTERED = "filtered_after_seal"
VARIANCE_EPSILON = 1e-6


class Mode(str, enum.Enum):
    REPAIR = "repair"
    AUDIT = "audit"
    DISCARD = "discard"
    OFF = "off"


class SealFailed(RuntimeError):
    """The seal could not decide the batch's initial selection; the caller delivers :meth:`SealSession.fail`."""


class MonotonicityError(RuntimeError):
    """A seal decision would be committed after one that read a later epoch (invariant I3)."""


class EpochSource(Protocol):
    def read_epoch(self) -> Epoch | None: ...


@dataclass(frozen=True)
class Member:
    """One member of a group offered to a seal: the caller's id for it, its execution record, the scores it already
    has (its recorded verdict, and any later scores: evaluations, earlier seals), and the reward it was recorded with.
    The recorded verdict is the first prior score that is recorded and not derived from another."""

    member_id: str
    execution: ExecutionRecord | None = None
    prior: tuple[ScoreRecord, ...] = ()
    recorded_reward: float | None = None

    def __post_init__(self) -> None:
        prior = self.prior
        if isinstance(prior, ScoreRecord):
            prior = (prior,)
        object.__setattr__(self, "prior", tuple(p for p in prior if p is not None))
        if self.recorded_reward is not None:
            object.__setattr__(self, "recorded_reward", float(self.recorded_reward))


@dataclass(frozen=True)
class Candidate:
    """A group (one GRPO unit) offered to a seal. ``handle`` is the caller's own object for it; it is not recorded."""

    group_id: str
    members: tuple[Member, ...]
    source: str = "selected"
    handle: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(self.members))


@dataclass(frozen=True)
class Assembly:
    """A batch assembled by :meth:`SealSession.assemble`: the groups to hand over in order (selected, backfilled,
    fillers), the groups excluded after the decision and not used as fillers (the caller retires them), candidates
    offered but never decided, the batch size asked for, and a summary."""

    batch: tuple[tuple[Candidate, GroupDecision], ...]
    excluded: tuple[tuple[Candidate, Exclusion], ...]
    unused: tuple[Candidate, ...]
    target_groups: int
    info: Mapping[str, Any]

    @property
    def groups(self) -> list[GroupDecision]:
        return [g for _, g in self.batch]

    @property
    def shortfall(self) -> int:
        return max(0, self.target_groups - len(self.batch))


def binary_reward(score: ScoreRecord) -> float:
    """1.0 for a pass, 0.0 for a fail."""
    return 1.0 if score.outcome is Outcome.PASS else 0.0


def has_variance(rewards: Sequence[float | None]) -> bool:
    """The GRPO group rule: at least two rewards, none missing, sample standard deviation above 1e-6."""
    if len(rewards) < 2 or any(r is None for r in rewards):
        return False
    values = [float(r) for r in rewards]  # type: ignore[arg-type]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5 > VARIANCE_EPSILON


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:500]


_MODE_ALLOW = {Mode.REPAIR: Action.REEXEC, Mode.AUDIT: Action.REPARSE, Mode.DISCARD: Action.REUSE}

#: the pin that names the startup epoch: the first grader publication of the log
PIN_STARTUP = "e0"


def _validated_pin(pin: str | int | None) -> str | int | None:
    """``"e0"``, a grader publication's ``seq`` (an int, or its decimal digits), or None (no pin)."""
    if pin is None:
        return None
    if isinstance(pin, bool):
        raise ValueError(f"unknown target pin {pin!r}: {PIN_STARTUP!r} (the startup epoch) or a publication seq")
    if isinstance(pin, int):
        if pin < 0:
            raise ValueError(f"a target pin's publication seq is >= 0, got {pin}")
        return pin
    value = str(pin).strip()
    if value == PIN_STARTUP:
        return PIN_STARTUP
    if value.isdigit() and value.isascii():
        return int(value)
    raise ValueError(f"unknown target pin {pin!r}: {PIN_STARTUP!r} (the startup epoch) or a publication seq")


def pinned_epoch(epochs: Any, pin: str | int, *, pointer: Epoch | None) -> Epoch | None:
    """The epoch a target pin names, read from the publication log of ``epochs`` (never from the pointer): ``"e0"``
    is the first grader publication, an int the grader publication with that ``seq``. ``pointer`` is the epoch the
    seal's one read of the pointer returned; the pinned epoch must have been published by then (its ``seq`` at most
    the pointer's), and it carries that read's time. None when no such publication exists at the read (the batch has
    no epoch: repair neutralizes it)."""
    pin = _validated_pin(pin)
    if pointer is None:
        return None
    graders = list(epochs.publications("grader"))
    if pin == PIN_STARTUP:
        publication = graders[0] if graders else None
    else:
        publication = next((p for p in graders if p.seq == pin), None)
    if publication is None or publication.seq > pointer.ref.seq:
        return None
    return Epoch(ref=publication.ref(read_at=pointer.ref.read_at), version=epochs.grader(publication.version_id))


class SealSession:
    """One seal of one batch; see the module documentation.

    ``epochs`` is anything with ``read_epoch()`` (a :class:`agent_ecc.versions.Registry`), read once now, or an
    :class:`agent_ecc.versions.Epoch` to seal under (replays), or None (nothing published). ``reward`` maps a decided
    score to a training reward; a consistent member (graded under exactly the batch epoch's spec) keeps its recorded
    reward. ``neutral_reward`` is the value every reward of a neutralized group is set to. ``batch_id`` is the
    caller's key for the learner step that will consume the batch (the join key with the learner's log).

    ``finish`` accepts only groups this session decided, as it handed them out (by ``decide`` or ``assemble``), whose
    members' scores are the session's own under its epoch.

    ``pin`` pins the batch epoch (``"e0"``: the startup epoch, the first grader publication; or a grader publication's
    ``seq``): the session still reads the pointer once, then takes the pinned epoch from the publication log
    (:func:`pinned_epoch`), and its decision records both (``SealDecision.target_pin``). A pin needs a registry; a pin
    the log does not hold at the read leaves the batch without an epoch.

    ``prepared`` is a :class:`agent_ecc.prepare.Preparer`: in repair mode the session's scorer takes, for a request it
    would execute, a result the preparer made for exactly that request under exactly this batch's epoch, when the
    result gives the check a logical verdict (otherwise it executes the request itself); the session tells the
    preparer which members it decides and which it sealed.
    """

    def __init__(
        self,
        epochs: EpochSource | Epoch | None,
        scorer: Scorer | None,
        *,
        mode: Mode | str = Mode.REPAIR,
        strict: bool = True,
        reward: Callable[[ScoreRecord], float] = binary_reward,
        neutral_reward: float = 0.0,
        batch_id: str | None = None,
        seal_id: str | None = None,
        clock: Callable[[], float] = time.time,
        pin: str | int | None = None,
        prepared: Any = None,
    ) -> None:
        self.mode = Mode(mode)
        self.strict = bool(strict)
        self.reward = reward
        self.neutral_reward = float(neutral_reward)
        self.batch_id = None if batch_id is None else str(batch_id)
        self.seal_id = seal_id or uuid.uuid4().hex
        self.clock = clock
        self.pin = _validated_pin(pin)
        self.pointer: Epoch | None = None  # what the pointer named at the read (a pinned seal only)
        self.opened_at = clock()
        if epochs is None or isinstance(epochs, Epoch):
            if self.pin is not None:
                raise ValueError("a target pin is resolved in the publication log: give the registry, not an epoch")
            self.epoch: Epoch | None = epochs
        else:
            self.epoch = epochs.read_epoch()  # the one read of the pointer for this batch
            if self.pin is not None:
                # the batch epoch is the pinned one, from the publication log; the pointer's is recorded beside it
                self.pointer = self.epoch
                self.epoch = pinned_epoch(epochs, self.pin, pointer=self.pointer)
        if self.mode is not Mode.OFF and scorer is None:
            raise ValueError(f"mode {self.mode.value!r} needs a scorer")
        self.scorer = scorer.with_options(allow=_MODE_ALLOW[self.mode]) if scorer is not None and \
            self.mode in _MODE_ALLOW else scorer
        self.prepared = prepared
        self._prepared_view = None
        if prepared is not None and self.mode is Mode.REPAIR and self.epoch is not None and self.scorer is not None:
            # results prepared for exactly this batch epoch (its publication and content) and nothing else
            self._prepared_view = prepared.view(self.epoch, seal_id=self.seal_id, batch_id=self.batch_id)
            self.scorer = self.scorer.with_options(prepared=self._prepared_view)
        self.policy = self.scorer.policy.value if self.scorer is not None else "none"
        self.rounds = 0
        self.scores: list[ScoreRecord] = []
        self._decided: dict[str, tuple[str, ...]] = {}
        # every GroupDecision this session issued, by group id: decide()'s, and assemble()'s role and filler versions
        self._issued: dict[str, list[GroupDecision]] = {}
        self._lock = threading.Lock()
        self._decide_s = 0.0
        self._transport: dict = {}  # how this seal's own executions went over a transport (Scored.transport)
        self._finished = False

    @property
    def epoch_ref(self):
        return self.epoch.ref if self.epoch is not None else None

    @property
    def target_pin(self) -> TargetPin | None:
        """The pin record of a pinned seal (None otherwise)."""
        if self.pin is None:
            return None
        return TargetPin(pin=self.pin, resolved=self.epoch is not None,
                         published=self.pointer.ref if self.pointer is not None else None)

    def _note_prepared(self, kind: str, member_ids: Sequence[str]) -> None:
        """Tell the preparer which members this seal decides or sealed. Never fails the seal."""
        if self.prepared is None:
            return
        try:
            if kind == "deciding":
                self.prepared.note_deciding(member_ids, self.epoch, seal_id=self.seal_id)
            else:
                self.prepared.note_sealed(member_ids, seal_id=self.seal_id)
        except Exception:  # noqa: BLE001 - preparation never fails a seal
            pass

    # ------------------------------------------------------------------------------------------------ decide

    def decide(self, candidates: Sequence[Candidate]) -> list[GroupDecision]:
        """Decide every member of ``candidates`` under the session's epoch; one GroupDecision per candidate."""
        with self._lock:
            if self._finished:
                raise RuntimeError("the seal is finished")
            for c in candidates:
                if c.group_id in self._decided:
                    raise ValueError(f"group {c.group_id!r} is decided twice in one seal")
            self.rounds += 1
            round_ = self.rounds
        self._note_prepared("deciding", [m.member_id for c in candidates for m in c.members])
        t0 = time.perf_counter()
        try:
            decisions = self._decide_members(candidates)
        finally:
            self._decide_s += time.perf_counter() - t0
        out = []
        for c, members in zip(candidates, decisions):
            out.append(self._group(c, members, round_))
        with self._lock:
            for c, g in zip(candidates, out):
                self._decided[c.group_id] = tuple(m.member_id for m in c.members)
                self._issued.setdefault(c.group_id, []).append(g)
        return out

    def _issue(self, group: GroupDecision) -> GroupDecision:
        """Register a version of a decided group this session hands out (a role or a filler change)."""
        with self._lock:
            self._issued.setdefault(group.group_id, []).append(group)
        return group

    def _decide_members(self, candidates: Sequence[Candidate]) -> list[list[MemberDecision]]:
        flat = [(ci, m) for ci, c in enumerate(candidates) for m in c.members]
        decisions: list[MemberDecision | None] = [None] * len(flat)
        if self.mode is Mode.OFF:
            decisions = [self._recorded(m, "off", None) for _, m in flat]
        elif self.epoch is None:
            for i, (_, m) in enumerate(flat):
                if self.mode is Mode.REPAIR:
                    decisions[i] = self._undecided(m, "undeterminable", "no_epoch")
                elif self.mode is Mode.DISCARD:
                    decisions[i] = self._undecided(m, "undeterminable", "no_epoch") if self.strict else \
                        self._recorded(m, "off", "no_epoch")
                else:
                    decisions[i] = self._recorded(m, "undeterminable", "no_epoch")
        else:
            assert self.scorer is not None
            to_score = [(i, m) for i, (_, m) in enumerate(flat) if m.execution is not None]
            inputs = [ScoreInput(m.execution, m.prior) for _, m in to_score]  # type: ignore[arg-type]
            scored = self.scorer.score(inputs, self.epoch)  # the epoch (publication and body): its identity binds
            with self._lock:
                for s in scored:
                    _merge_transport(self._transport, getattr(s, "transport", None))
            for (i, m), s in zip(to_score, scored):
                if s.score is not None:
                    self.scores.append(s.score)
                decisions[i] = self._member(m, s)
            for i, (_, m) in enumerate(flat):
                if decisions[i] is None:
                    decisions[i] = self._undecided(m, "undeterminable", "no_record")
        grouped: list[list[MemberDecision]] = [[] for _ in candidates]
        for (ci, _), d in zip(flat, decisions):
            grouped[ci].append(d)  # type: ignore[arg-type]
        return grouped

    def _recorded_pass(self, m: Member) -> bool | None:
        if m.recorded_reward is not None:
            return m.recorded_reward >= 0.5
        for score in m.prior:
            if score.method is Method.RECORDED and score.based_on is None and score.logical:
                return score.passed
        return None

    def _member(self, m: Member, s) -> MemberDecision:
        score: ScoreRecord | None = s.score
        if s.action == "pending":
            return MemberDecision(member_id=m.member_id, execution_id=s.execution_id, action="pending",
                                  reason=f"needs_{s.required}", recorded_reward=m.recorded_reward,
                                  trained_reward=m.recorded_reward)
        assert score is not None
        outcome = score.outcome
        if not outcome.logical:
            return MemberDecision(member_id=m.member_id, execution_id=s.execution_id, action="undeterminable",
                                  outcome=outcome, reason=score.reason, score_id=score.score_id,
                                  recorded_reward=m.recorded_reward,
                                  trained_reward=None if self.mode is Mode.REPAIR else m.recorded_reward)
        consistent = s.action == "reuse" and self._consistent(m, score)
        if (consistent or s.action == "ungoverned") and m.recorded_reward is not None:
            target = m.recorded_reward  # the recorded verdict itself: its reward stands (it may be shaped)
        else:
            target = float(self.reward(score))
        recorded_pass = self._recorded_pass(m)
        flipped = None if recorded_pass is None else (outcome is Outcome.PASS) != recorded_pass
        trained = target if self.mode is Mode.REPAIR else m.recorded_reward
        return MemberDecision(member_id=m.member_id, execution_id=s.execution_id, action=s.action, outcome=outcome,
                              reason=score.reason, score_id=score.score_id, recorded_reward=m.recorded_reward,
                              target_reward=target, trained_reward=trained, flipped=flipped,
                              short_circuit=s.short_circuit, consistent=consistent)

    def _consistent(self, m: Member, score: ScoreRecord) -> bool:
        """Whether a reused verdict is the member's recorded verdict of exactly the target spec: every evaluated check
        was taken from the recorded score, which was graded under the same spec (by its spec digest; for a recorded
        score without one, by its checks' identities, in order) and has the same outcome."""
        recorded = _recorded_score(m.prior)
        if recorded is None or not recorded.logical or score.based_on != recorded.score_id:
            return False
        if score.outcome is not recorded.outcome:
            return False
        if recorded.spec_digest is not None:
            return recorded.spec_digest == score.spec_digest
        assert self.epoch is not None and m.execution is not None
        spec = self.epoch.version.lookup(m.execution.task_id, m.execution.world_version).spec
        if spec is None or recorded.skipped:
            return False
        want = [(c.execution_digest, c.grading_digest) for c in iter_checks(spec)]
        return [(p.execution_digest, p.grading_digest) for p in recorded.parts] == want

    def _recorded(self, m: Member, action: str, reason: str | None) -> MemberDecision:
        return MemberDecision(member_id=m.member_id, execution_id=m.execution.record_id if m.execution else None,
                              action=action, reason=reason, recorded_reward=m.recorded_reward,
                              trained_reward=m.recorded_reward)

    def _undecided(self, m: Member, action: str, reason: str) -> MemberDecision:
        """A member with no label under the batch epoch (no record, no epoch). Repair trains nothing for it (its group
        is neutralized); the other modes train its recorded reward, as they do for a member whose score is
        undeterminable (audit trains the recorded labels; discard neutralizes the group anyway)."""
        return MemberDecision(member_id=m.member_id, execution_id=m.execution.record_id if m.execution else None,
                              action=action, outcome=Outcome.UNDETERMINABLE, reason=reason,
                              recorded_reward=m.recorded_reward,
                              trained_reward=None if self.mode is Mode.REPAIR else m.recorded_reward)

    def _group(self, c: Candidate, members: list[MemberDecision], round_: int) -> GroupDecision:
        reason = None
        if self.mode is Mode.REPAIR and any(not d.decided for d in members):
            reason = "undeterminable_member"
        elif self.mode is Mode.DISCARD:
            if self.epoch is None:
                reason = "no_epoch" if self.strict else None
            elif any(not (d.consistent or d.action == "ungoverned") for d in members):
                reason = "not_consistent"
        if reason is not None:
            members = [dataclasses.replace(d, trained_reward=self.neutral_reward) for d in members]
        return GroupDecision(group_id=c.group_id, members=tuple(members), round=round_, role="selected",
                             source=c.source, neutralized=reason is not None, neutralize_reason=reason)

    # ------------------------------------------------------------------------------------------------ assemble

    def assemble(
        self,
        initial: Sequence[Candidate],
        *,
        keep: Callable[[Sequence[float | None]], bool] | None = has_variance,
        backfill: Callable[[int, frozenset, float | None], Iterable[Candidate]] | None = None,
        fillers: str = "none",
        deadline_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> Assembly:
        """Decide ``initial``, filter the decided groups with ``keep``, and refill filtered slots.

        ``backfill(need, considered, remaining_s)`` returns up to ``need`` further candidates not in ``considered``
        (group ids already offered), waiting at most ``remaining_s`` seconds if it waits at all (None: no deadline);
        an empty answer ends backfill. Every candidate it returns is decided by this session, under the session's
        epoch. A new round starts only while ``deadline_s`` has not elapsed since backfill began; a round that has
        started is not interrupted. ``fillers="neutral"`` fills slots still missing with the oldest excluded groups,
        neutralized; ``"none"`` (default) delivers the batch short.

        A failure deciding ``initial`` raises :class:`SealFailed`; a failure deciding a backfill round stops
        backfilling (its candidates are returned unused, never trained). Exceptions raised by ``backfill`` propagate.
        """
        if fillers not in ("none", "neutral"):
            raise ValueError("fillers must be 'none' or 'neutral'")
        initial = list(initial)
        target = len(initial)
        try:
            first = self.decide(initial)
        except Exception as error:  # noqa: BLE001 - the caller delivers fail()
            raise SealFailed(_describe(error)) from error
        if keep is None:
            return Assembly(batch=tuple(zip(initial, first)), excluded=(), unused=(), target_groups=target,
                            info=freeze({"target_groups": target, "rounds": self.rounds, "filter": None}))
        sizes = {len(c.members) for c in initial}
        if len(sizes) > 1:
            return Assembly(batch=tuple(zip(initial, first)), excluded=(), unused=(), target_groups=target,
                            info=freeze({"target_groups": target, "rounds": self.rounds,
                                         "skipped": "unequal_group_sizes"}))
        size = sizes.pop() if sizes else 0
        reason = EXCLUSION_ZERO_VARIANCE if keep is has_variance else EXCLUSION_FILTERED
        kept: list[tuple[Candidate, GroupDecision]] = []
        excluded: list[tuple[Candidate, GroupDecision]] = []

        def classify(pairs, role):
            for cand, group in pairs:
                if keep(group.rewards()):
                    kept.append((cand, self._issue(group.replace(role=role))))
                else:
                    excluded.append((cand, group))

        classify(zip(initial, first), "selected")
        considered = {c.group_id for c in initial}
        unused: list[Candidate] = []
        error = None
        decided_n = len(initial)
        t_begin = clock()
        while backfill is not None and len(kept) < target:
            elapsed = clock() - t_begin
            if deadline_s is not None and elapsed >= deadline_s:
                break
            need = target - len(kept)
            remaining = None if deadline_s is None else max(0.0, deadline_s - elapsed)
            offered = list(backfill(need, frozenset(considered), remaining))
            fresh, seen = [], set()
            for c in offered:
                if c.group_id in considered or c.group_id in seen:
                    continue
                seen.add(c.group_id)
                fresh.append(c)
            cands, beyond = fresh[:need], fresh[need:]
            unused.extend(beyond)
            if not cands:
                break
            considered.update(c.group_id for c in cands)
            usable = [c for c in cands if len(c.members) == size]
            unused.extend(c for c in cands if len(c.members) != size)
            if not usable:
                continue
            try:
                groups = self.decide(usable)
            except Exception as err:  # noqa: BLE001 - stop backfilling; these candidates were never decided
                error = _describe(err)
                unused.extend(usable)
                break
            decided_n += len(usable)
            classify(zip(usable, groups), "backfill")
        backfill_s = clock() - t_begin
        missing = target - len(kept)
        filler_pairs: list[tuple[Candidate, GroupDecision]] = []
        if fillers == "neutral" and missing > 0:
            for cand, group in excluded[:missing]:
                members = tuple(dataclasses.replace(m, trained_reward=self.neutral_reward) for m in group.members)
                filler_pairs.append((cand, self._issue(group.replace(
                    members=members, role="filler", neutralized=True,
                    neutralize_reason=group.neutralize_reason or "filler"))))
        rest = excluded[len(filler_pairs):]
        info = {
            "target_groups": target, "rounds": self.rounds, "decided_groups": decided_n,
            "excluded_n": len(excluded), "excluded_neutralized_n": sum(1 for _, g in excluded if g.neutralized),
            "backfilled_n": sum(1 for _, g in kept if g.role == "backfill"), "fillers_n": len(filler_pairs),
            "fillers": fillers, "shortfall": max(0, target - len(kept) - len(filler_pairs)),
            "backfill_s": round(backfill_s, 6), "deadline_s": deadline_s, "error": error,
            "unused_n": len(unused),
        }
        return Assembly(batch=tuple(kept + filler_pairs),
                        excluded=tuple((c, Exclusion(group=g, reason=reason)) for c, g in rest),
                        unused=tuple(unused), target_groups=target, info=freeze(info))

    # ------------------------------------------------------------------------------------------------ finish

    def finish(self, batch: Assembly | Sequence[GroupDecision], *, excluded: Sequence[Exclusion] = (),
               journal: SealJournal | None = None) -> SealDecision:
        """The batch's SealDecision. ``batch`` is an :class:`Assembly` or the decided groups in hand-off order.

        With a ``journal``, the decision is appended to it; if that fails, a strict seal returns :meth:`fail`'s
        neutralized decision instead (an unlogged seal cannot be audited)."""
        info: Mapping[str, Any] = {}
        if isinstance(batch, Assembly):
            groups = [g for _, g in batch.batch]
            excluded = [e for _, e in batch.excluded]
            target = batch.target_groups
            info = batch.info
        else:
            groups = list(batch)
            target = None
        seen: set[str] = set()
        scores = {s.score_id: s for s in self.scores}
        version_id = self.epoch.ref.version_id if self.epoch is not None else None
        for g in list(groups) + [e.group for e in excluded]:
            # only a group this session decided, as it handed it out: never one decided by another session (under
            # another epoch) or edited afterwards
            with self._lock:
                issued = list(self._issued.get(g.group_id, ()))
            if not any(g == mine for mine in issued):
                raise ValueError(f"group {g.group_id!r} was not decided by this seal")
            for m in g.members:
                if m.score_id is None:
                    continue
                score = scores.get(m.score_id)
                if score is None or score.grader_version != version_id:
                    raise ValueError(f"member {m.member_id!r} of group {g.group_id!r} has a score this seal did "
                                     f"not produce under its epoch")
            if g.group_id in seen:
                raise ValueError(f"group {g.group_id!r} appears twice")
            seen.add(g.group_id)
        with self._lock:
            self._finished = True
        decision = SealDecision(seal_id=self.seal_id, epoch=self.epoch_ref, mode=self.mode.value,
                                policy=self.policy, strict=self.strict, groups=tuple(groups),
                                batch_id=self.batch_id, excluded=tuple(excluded), target_groups=target,
                                rounds=self.rounds, failed=None, opened_at=self.opened_at, closed_at=self.clock(),
                                stats=self._stats(groups, excluded, info), target_pin=self.target_pin)
        # the members this seal decided (handed over or excluded) are never prepared again
        self._note_prepared("sealed", [m.member_id for g in list(groups) + [e.group for e in excluded]
                                       for m in g.members])
        if journal is not None:
            try:
                journal.append(decision)
            except Exception as error:  # noqa: BLE001
                if self.strict:
                    return self.fail(f"journal: {_describe(error)}", groups)
                decision = dataclasses.replace(decision, stats=freeze({**decision.stats,
                                                                      "log_error": _describe(error)}))
        return decision

    def fail(self, error: BaseException | str, groups: Sequence[Candidate | GroupDecision]) -> SealDecision:
        """The decision for a seal that could not finish. Strict: every group neutralized (no member trains a label
        not decided under the epoch). Non-strict: the recorded rewards train."""
        with self._lock:
            self._finished = True
        reason = error if isinstance(error, str) else _describe(error)
        out = []
        for g in groups:
            if isinstance(g, Candidate):
                pairs = [(m.member_id, m.execution.record_id if m.execution else None, m.recorded_reward)
                         for m in g.members]
                gid, source, round_ = g.group_id, g.source, 0
            else:
                pairs = [(m.member_id, m.execution_id, m.recorded_reward) for m in g.members]
                gid, source, round_ = g.group_id, g.source, g.round
            members = tuple(MemberDecision(member_id=mid, execution_id=eid, action="failed", reason="seal_failed",
                                           recorded_reward=rec,
                                           trained_reward=self.neutral_reward if self.strict else rec)
                            for mid, eid, rec in pairs)
            out.append(GroupDecision(group_id=gid, members=members, round=max(round_, 0) or 1, role="selected",
                                     source=source, neutralized=self.strict,
                                     neutralize_reason="seal_failed" if self.strict else None))
        self._note_prepared("sealed", [m.member_id for g in out for m in g.members])
        return SealDecision(seal_id=self.seal_id, epoch=self.epoch_ref, mode=self.mode.value, policy=self.policy,
                            strict=self.strict, groups=tuple(out), batch_id=self.batch_id, excluded=(),
                            target_groups=len(out), rounds=self.rounds, failed=reason, opened_at=self.opened_at,
                            closed_at=self.clock(), stats=freeze({"groups": len(out), **self._transport_stats()}),
                            target_pin=self.target_pin)

    def _stats(self, groups: Sequence[GroupDecision], excluded: Sequence[Exclusion],
               info: Mapping[str, Any]) -> FrozenDict:
        statuses: dict[str, int] = {}
        for g in groups:
            for m in g.members:
                statuses[m.status] = statuses.get(m.status, 0) + 1
        by_id = {s.score_id: s for s in self.scores}
        used = [by_id[m.score_id] for g in groups for m in g.members if m.score_id in by_id]
        executions = sum(s.cost.executions for s in self.scores)
        stats = {
            "groups": len(groups),
            "members": sum(len(g.members) for g in groups),
            "neutralized_groups": sum(1 for g in groups if g.neutralized),
            "filler_groups": sum(1 for g in groups if g.role == "filler"),
            "backfill_groups": sum(1 for g in groups if g.role == "backfill"),
            "excluded_groups": len(excluded),
            "statuses": statuses,
            "reparsed": sum(1 for s in used if s.method is Method.REPARSE),
            "reexecuted": sum(1 for s in used if s.method is Method.REEXEC),
            "executions": executions,
            "container_seconds": round(sum(s.cost.container_seconds for s in self.scores), 6),
            "decide_seconds": round(self._decide_s, 6),
            "assembly": dict(info),
        }
        if self._prepared_view is not None:
            # how every request this seal would have executed was answered: by a prepared result, or at the seal
            stats["prepared"] = self._prepared_view.summary()
        stats.update(self._transport_stats())
        return freeze(stats)

    def _transport_stats(self) -> dict:
        """``{"transport": tally}`` when this seal's executions went over a transport (attempts, retries, by reason)."""
        with self._lock:
            return {"transport": copy.deepcopy(self._transport)} if self._transport else {}


class Arm(str, enum.Enum):
    """The comparison arms of a grader-switch experiment, as seal configurations (:func:`arm_session`).

    * ``keep``: audit mode. Decides what repair would (re-parsing, never executing) and trains the recorded labels.
    * ``discard``: validate and discard. Recomputes nothing; a group with a member not graded under exactly the batch
      epoch's spec (and not ungoverned) is neutralized.
    * ``full_compatible``: repair. Once the batch epoch is not the startup epoch (after the switch), every member is
      re-verified under every target spec, its recorded verdicts not trusted (policy ``full``; the short circuit
      stays), with executions shared within a batch and across seals through the scorer's cache; before the switch it
      is the ``ecc`` arm.
    * ``ecc``: repair. Every label is the batch epoch's verdict, reused, re-parsed or re-executed as the identities
      require; an undecidable member neutralizes its group.
    * ``never`` (also ``pinned``): repair under the ``ecc`` policy with the batch epoch pinned to the startup epoch for
      the whole run (``SealSession(pin="e0")``). Later publications still happen and producers still grade under
      them; only the seal ignores them, so a member graded under a later epoch trains the startup spec's verdict (its
      recorded parts of that spec reused). The control that prices adopting the new version: I1 to I3 hold, I4's
      adoption is relaxed on purpose and recorded (``SealDecision.target_pin``).
    """

    KEEP = "keep"
    DISCARD = "discard"
    FULL_COMPATIBLE = "full_compatible"
    ECC = "ecc"
    NEVER = "never"

    @classmethod
    def _missing_(cls, value):  # noqa: ANN001, ANN206
        if value == "pinned":  # the main text's name for the never arm
            return cls.NEVER
        return None


_ARM_MODE = {Arm.KEEP: Mode.AUDIT, Arm.DISCARD: Mode.DISCARD, Arm.FULL_COMPATIBLE: Mode.REPAIR, Arm.ECC: Mode.REPAIR,
             Arm.NEVER: Mode.REPAIR}


def arm_session(arm: Arm | str, epochs: EpochSource | Epoch | None, scorer: Scorer, *,
                startup: int | None = None, cache: dict | None = None, **kwargs: Any) -> SealSession:
    """A :class:`SealSession` that seals one batch as ``arm`` does (see :class:`Arm`). The epoch pointer is read once,
    here. ``cache`` (a dict shared by every seal of the run) is given to the scorer; ``full_compatible`` needs one,
    on the scorer or here, so that its re-verifications share executions across seals. The cache is keyed by the
    execution request (:attr:`agent_ecc.recompute.ExecRequest.request_id`: the task, the check's execution identity,
    the patch as it is applied after the check's patch filter, the eval script and the request's parameters), so every
    request a seal executed to a logical verdict answers the identical request later. ``startup`` is the publication
    ``seq`` of the startup
    epoch; with a registry it defaults to the first grader publication. The remaining keyword arguments go to the
    session.

    ``never`` pins the batch epoch to the startup epoch (``startup``'s seq when given, else the first grader
    publication) and so needs a registry, whose pointer the session still reads once; given an :class:`Epoch` (a
    replay that already knows the pinned epoch), it seals under that epoch as ``ecc`` does, without a pin record."""
    arm = Arm(arm)
    if arm is Arm.NEVER:
        if cache is not None:
            scorer = scorer.with_options(cache=cache)
        scorer = scorer.with_options(policy=Policy.ECC)
        if epochs is None or isinstance(epochs, Epoch):
            return SealSession(epochs, scorer, mode=Mode.REPAIR, **kwargs)
        pin = PIN_STARTUP if startup is None else int(startup)
        return SealSession(epochs, scorer, mode=Mode.REPAIR, pin=pin, **kwargs)  # one read of the pointer, there
    if epochs is None or isinstance(epochs, Epoch):
        epoch = epochs
    else:
        epoch = epochs.read_epoch()  # the one read of the pointer for this batch
    if arm in (Arm.KEEP, Arm.DISCARD):
        # they plan as ecc does (reuse what the identities allow); their modes bound what may be recomputed
        return SealSession(epoch, scorer.with_options(policy=Policy.ECC), mode=_ARM_MODE[arm], **kwargs)
    if cache is not None:
        scorer = scorer.with_options(cache=cache)
    if arm is Arm.ECC:
        return SealSession(epoch, scorer.with_options(policy=Policy.ECC), mode=Mode.REPAIR, **kwargs)
    if scorer.cache is None:
        raise ValueError("the full_compatible arm shares its re-verifications through the scorer's cache: pass "
                         "cache={} (one dict for every seal of the run)")
    if startup is None:
        publications = getattr(epochs, "publications", None)
        if publications is None:
            raise ValueError("pass startup= (the startup epoch's publication seq) when sealing under a given epoch")
        first = next(iter(publications("grader")), None)
        startup = first.seq if first is not None else None
    switched = epoch is not None and startup is not None and epoch.ref.seq != int(startup)
    return SealSession(epoch, scorer.with_options(policy=Policy.FULL if switched else Policy.ECC), mode=Mode.REPAIR,
                       **kwargs)


def seal(epochs: EpochSource | Epoch | None, scorer: Scorer | None, candidates: Sequence[Candidate],
         **kwargs: Any) -> tuple[SealDecision, list[ScoreRecord]]:
    """Seal a batch in one call: one decision round, no filtering. Returns the decision and the scores it produced.
    A failure while deciding returns :meth:`SealSession.fail`'s decision."""
    journal = kwargs.pop("journal", None)
    session = SealSession(epochs, scorer, **kwargs)
    try:
        groups = session.decide(candidates)
    except Exception as error:  # noqa: BLE001
        return session.fail(error, candidates), list(session.scores)
    return session.finish(groups, journal=journal), list(session.scores)


class SealJournal:
    """An append-only JSON-lines log of seal decisions, in commit order.

    ``require_monotonic`` (default) refuses a decision whose epoch was published before the epoch of the last decision
    appended: batches must commit in non-decreasing epoch order (invariant I3), which holds when seals are serialized
    (one sealed, uncommitted batch at a time)."""

    def __init__(self, path: str | os.PathLike, *, require_monotonic: bool = True) -> None:
        self.path = os.fspath(path)
        self.require_monotonic = require_monotonic
        self._lock = threading.Lock()
        self._last_seq: int | None = None
        if os.path.exists(self.path):
            for decision in self.read():
                if decision.epoch is not None:
                    self._last_seq = decision.epoch.seq

    def append(self, decision: SealDecision) -> None:
        with self._lock:
            seq = decision.epoch.seq if decision.epoch is not None else None
            if self.require_monotonic and seq is not None and self._last_seq is not None and seq < self._last_seq:
                raise MonotonicityError(f"seal {decision.seal_id} read epoch seq {seq} after a seal that read "
                                        f"{self._last_seq}")
            line = (json.dumps(decision.to_dict(), sort_keys=True) + "\n").encode("utf-8")
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                view = memoryview(line)
                while view:
                    view = view[os.write(fd, view):]
            finally:
                os.close(fd)
            if seq is not None:
                self._last_seq = seq

    def read(self) -> list[SealDecision]:
        if not os.path.exists(self.path):
            return []
        return load_jsonl(self.path, SealDecision)


def apply_rewards(decision: SealDecision, samples: Iterable[Any], *, member_id: Callable[[Any], str],
                  set_reward: Callable[[Any, float], None]) -> int:
    """Give delivered samples the rewards a seal decided, matched by member id. All or nothing: every delivered sample
    is matched exactly once and every decided member is delivered, checked before any reward moves. Returns samples
    set."""
    decided = decision.rewards()
    moves, seen = [], set()
    for sample in samples:
        mid = member_id(sample)
        if mid not in decided:
            raise ValueError(f"delivered member {mid!r} has no reward decided by seal {decision.seal_id}")
        if mid in seen:
            raise ValueError(f"member {mid!r} is delivered twice")
        if decided[mid] is None:
            raise ValueError(f"member {mid!r} has no trainable reward")
        seen.add(mid)
        moves.append((sample, decided[mid]))
    if len(moves) != len(decided):
        raise ValueError(f"seal {decision.seal_id} decided {len(decided)} members but {len(moves)} are delivered")
    for sample, reward in moves:
        set_reward(sample, float(reward))
    return len(moves)
