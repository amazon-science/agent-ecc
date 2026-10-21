# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deciding, per member and per check, how its verdict under a target grader version is obtained.

For one check of the target spec and one execution record, :func:`plan_check` chooses one action from the record's
execution identity, its recorded runs and its recorded verdicts:

========================  =================================================================================
reuse                     a recorded verdict whose execution and grading identities both equal the check's
reparse                   a recorded run whose execution identity equals the check's, with its outputs available:
                          judge those outputs again under the check's grading identity (no execution)
reexec                    execute the check's tests on the recorded artifact (the identities differ, or no usable
                          run exists)
undeterminable            none of the above is possible with the evidence held
========================  =================================================================================

Policies say how much is taken on trust:

* ``ecc`` (default): reuse when nothing the verdict depends on changed; reparse when only grading inputs changed;
  re-execute when execution inputs changed.
* ``reparse_all``: never take a recorded verdict on trust, but re-execute only when the execution identity differs;
  otherwise re-parse the recorded outputs.
* ``full``: re-execute every check (the full re-grade baseline). A re-verification arm that re-executes every member
  under every target spec after a switch is ``full`` from the switch on, usually with the scorer's cache of logical
  results; one that also evaluates every check of a conjunction after a logical fail turns the short circuit off
  (``Scorer(short_circuit=False)``).

Infrastructure errors are never verdicts: a run that did not produce usable outputs, or outputs that do not parse to
a logical verdict, never decide a check. They are re-executed when the artifact allows it, and otherwise leave the
check undeterminable.

Conjunctions (:class:`agent_ecc.versions.AllOf`) are three-valued ANDs, which are monotone (:func:`combine`,
:class:`Conjunction`): a logical fail of any part fails the whole whatever the other parts say, undeterminable ones
included; the whole passes only when every part passes; otherwise it is undeterminable. So a logical fail decides the
conjunction as soon as it is known, and the parts not yet evaluated are skipped (the short circuit); an undeterminable
part decides nothing, and the parts after it are still evaluated (one of them may fail).
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .records import RUN_ERROR, RUN_OK, Outcome, PartScore, TestRun
from .versions import Check, TestSpec, iter_checks

__all__ = [
    "Action",
    "CheckPlan",
    "Conjunction",
    "Policy",
    "combine",
    "plan_check",
]


class Policy(str, enum.Enum):
    ECC = "ecc"
    REPARSE_ALL = "reparse_all"
    FULL = "full"

    @classmethod
    def _missing_(cls, value):  # noqa: ANN001, ANN206
        if value == "full_compatible":
            raise ValueError(
                "policy 'full_compatible' is now 'reparse_all' (re-parse every recorded output whose execution "
                "identity is unchanged, re-execute the rest). A re-verification that re-executes every member under "
                "every target spec is 'full', with the scorer's cache of logical results; the full_compatible "
                "comparison arm is seal.arm_session('full_compatible', ...) (seal.Arm.FULL_COMPATIBLE): repair under "
                "policy 'full' once the batch epoch is not the startup epoch")
        return None


class Action(str, enum.Enum):
    REUSE = "reuse"
    REPARSE = "reparse"
    REEXEC = "reexec"
    UNDETERMINABLE = "undeterminable"

    @property
    def rank(self) -> int:
        return {Action.REUSE: 0, Action.REPARSE: 1, Action.REEXEC: 2, Action.UNDETERMINABLE: 3}[self]


@dataclass(frozen=True)
class CheckPlan:
    """How one check's verdict will be obtained, and why."""

    check: Check
    action: Action
    reason: str
    run: TestRun | None = None  # the recorded run to re-parse
    prior: PartScore | None = None  # the recorded verdict to reuse


def plan_check(
    check: Check,
    *,
    runs: Sequence[TestRun] = (),
    parts: Sequence[PartScore] = (),
    output_available: Callable[[TestRun], bool] = lambda run: True,
    can_reexec: bool = False,
    policy: Policy | str = Policy.ECC,
    recorded_infra: str = "reexec",
) -> CheckPlan:
    """Choose the action for one check. Pure: depends only on its arguments.

    ``runs`` are the record's runs (and runs created by earlier re-executions), ``parts`` the check-level verdicts of
    its earlier scores, ``output_available`` tells whether a run's outputs can be read, ``can_reexec`` whether the
    artifact needed to execute the check is held. ``recorded_infra`` says what a recorded run that hit an
    infrastructure error leads to: ``"reexec"`` (default) or ``"undeterminable"``.
    """
    policy = Policy(policy)
    exe, grd = check.execution_digest, check.grading_digest
    if policy is Policy.ECC:
        for part in reversed(list(parts)):
            if part.execution_digest == exe and part.grading_digest == grd and part.logical:
                return CheckPlan(check, Action.REUSE, "identity_equal", prior=part)
    reason = "full"
    if policy is not Policy.FULL:
        matching = [r for r in runs if r.execution_digest == exe]
        usable = [r for r in matching if r.status == RUN_OK and output_available(r)]
        if usable:
            if policy is Policy.REPARSE_ALL:
                why = "reparse_all"
            elif any(p.execution_digest == exe for p in parts):
                why = "grading_changed"
            else:
                why = "no_recorded_verdict"
            return CheckPlan(check, Action.REPARSE, why, run=usable[-1])
        same_execution = [p for p in parts if p.execution_digest == exe]
        if any(r.status == RUN_ERROR for r in matching) or (
                not matching and same_execution and not any(p.logical for p in same_execution)):
            reason = "recorded_infra_error"
            if recorded_infra == "undeterminable":
                return CheckPlan(check, Action.UNDETERMINABLE, reason)
        elif matching or same_execution:
            reason = "output_unavailable"  # the same tests ran, but their outputs are not held
        elif runs or parts:
            reason = "execution_changed"
        else:
            reason = "no_run"
    if can_reexec:
        return CheckPlan(check, Action.REEXEC, reason)
    return CheckPlan(check, Action.UNDETERMINABLE, f"missing_artifact:{reason}")


def combine(outcomes: Sequence[Outcome]) -> tuple[Outcome, int]:
    """The three-valued AND of a conjunction whose parts, in order, have ``outcomes``, and how many parts decided it.

    Monotone: a logical FAIL of any part decides FAIL (n is the index of the first fail + 1), whatever the other
    parts are, undeterminable ones included; PASS needs every given part to pass (complete only when all parts are
    given); otherwise the outcome is UNDETERMINABLE and n counts every given part (a later part could still fail)."""
    for i, outcome in enumerate(outcomes):
        if outcome is Outcome.FAIL:
            return Outcome.FAIL, i + 1
    if all(outcome is Outcome.PASS for outcome in outcomes):
        return Outcome.PASS, len(outcomes)
    return Outcome.UNDETERMINABLE, len(outcomes)


class Conjunction:
    """A spec's checks resolved in any order, combined by the monotone three-valued AND.

    ``resolve(part, index)`` records the verdict of the check at ``index`` (without an index: the first check not yet
    resolved). With ``short_circuit`` (default) the conjunction is decided FAIL as soon as one part fails logically,
    and the checks still unresolved when evaluation stops are skipped; otherwise every check is resolved before it is
    decided. An
    undeterminable part never decides: the conjunction is undeterminable only once every check is resolved and none
    failed (an infrastructure error never implies a fail, nor hides a later one)."""

    def __init__(self, spec: TestSpec, *, short_circuit: bool = True) -> None:
        self.spec = spec
        self.checks: tuple[Check, ...] = tuple(iter_checks(spec))
        self.stop_on_fail = bool(short_circuit)
        self._parts: dict[int, PartScore] = {}

    @property
    def parts(self) -> list[PartScore]:
        """The resolved parts, in check order."""
        return [self._parts[i] for i in sorted(self._parts)]

    @property
    def parts_by_index(self) -> dict[int, PartScore]:
        """The resolved parts by their check's position."""
        return dict(self._parts)

    def resolved(self, index: int) -> bool:
        return index in self._parts

    def unresolved(self) -> list[int]:
        return [i for i in range(len(self.checks)) if i not in self._parts]

    @property
    def outcome(self) -> Outcome | None:
        """The decided outcome, or None while it is not decided."""
        failed = any(p.outcome is Outcome.FAIL for p in self._parts.values())
        complete = len(self._parts) == len(self.checks)
        if failed and (self.stop_on_fail or complete):
            return Outcome.FAIL
        if not complete:
            return None
        if all(p.outcome is Outcome.PASS for p in self._parts.values()):
            return Outcome.PASS
        return Outcome.UNDETERMINABLE

    @property
    def done(self) -> bool:
        return self.outcome is not None

    def next_check(self) -> Check | None:
        """The first check not yet resolved, in order (None once decided)."""
        if self.done:
            return None
        pending = self.unresolved()
        return self.checks[pending[0]] if pending else None

    def resolve(self, part: PartScore, index: int | None = None) -> None:
        """Record the verdict of the check at ``index`` (default: the first unresolved one). Checks may still be
        resolved after a logical fail decided the conjunction (information that cost nothing); the outcome stays a
        fail, since the conjunction is monotone."""
        if index is None:
            pending = self.unresolved()
            if not pending:
                raise RuntimeError("every check of the conjunction is already resolved")
            index = pending[0]
        if index in self._parts:
            raise ValueError(f"check {self.checks[index].name!r} is already resolved")
        expected = self.checks[index]
        if part.execution_digest != expected.execution_digest or part.grading_digest != expected.grading_digest:
            raise ValueError(f"part {part.check!r} does not resolve check {expected.name!r}")
        self._parts[index] = part

    @property
    def skipped(self) -> tuple[str, ...]:
        """Names of the checks never evaluated (only meaningful once done: a logical fail decided first)."""
        return tuple(self.checks[i].name for i in self.unresolved()) if self.done else ()

    @property
    def short_circuit(self) -> bool:
        return self.done and self.outcome is Outcome.FAIL and len(self._parts) < len(self.checks)
