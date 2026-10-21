# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence levels, and what each level lets a grader change re-grade.

What a record keeps decides what a later grader version can do with it. The levels are cumulative:

=========  ==========================================================================================================
L0         the recorded reward only
L0+        plus identity stamps: the world version observed and the execution and grading identities that decided
           the recorded verdict
L1         plus model I/O and tool calls with their outputs, and the submitted answer
L2         plus the environment's state as the evaluator saw it: an end-state diff (database change logs), or the
           agent's patch and the raw test log
L2'        plus a full environment snapshot or image reference, for re-execution
=========  ==========================================================================================================

For each type of grader change, :func:`disposition` gives what a record at a level allows:

* ``keep``        reuse the recorded verdict;
* ``reparse``     re-grade in-process from recorded outputs (milliseconds);
* ``reexec``      re-execute the tests (not the episode) on the recorded artifact;
* ``replay``      re-run the recorded tool calls against the new tools and keep the records whose outputs reproduce;
* ``rerun``       only a new episode gives a label: in training, the record's group is neutralized;
* ``neutralize``  no label is defined for the record under the new version, at any cost.

:func:`mask` removes the evidence above a level from a record, so an evidence ablation can re-decide recorded
members as if less had been kept.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .records import ExecutionRecord, ScoreRecord
from .versions import Check

__all__ = [
    "ChangeType",
    "Level",
    "ROLE_LEVELS",
    "Sufficiency",
    "classify_change",
    "disposition",
    "level_of",
    "mask",
    "mask_score",
    "minimum_level",
    "sufficient",
    "table",
]


class Level(str, enum.Enum):
    SCORE = "L0"
    STAMPS = "L0+"
    IO = "L1"
    STATE = "L2"
    SNAPSHOT = "L2'"

    @property
    def rank(self) -> int:
        """Position in the order L0 < L0+ < L1 < L2 < L2' (compare levels by rank, not by their string values)."""
        return _LEVEL_ORDER.index(self)


_LEVEL_ORDER = [Level.SCORE, Level.STAMPS, Level.IO, Level.STATE, Level.SNAPSHOT]

#: The level each evidence role belongs to. Adapters use these role names; others can be added by passing a mapping.
ROLE_LEVELS: Mapping[str, Level] = {
    "transcript": Level.IO,
    "tool_calls": Level.IO,
    "commands": Level.IO,
    "answer": Level.IO,
    "patch": Level.STATE,
    "test_log": Level.STATE,
    "end_state": Level.STATE,
    "tests": Level.STATE,
    "snapshot": Level.SNAPSHOT,
    "image": Level.SNAPSHOT,
}


class ChangeType(str, enum.Enum):
    UNTOUCHED = "untouched"  # the change does not touch the record's task or tests
    ANSWER_TRUTH = "answer_truth"  # an expected answer (label-only truth) changed
    WORLD_DATA = "world_data"  # world data changed, the episode pinned to (or loaded at the start of) one version
    STATE_TEST = "state_test"  # a state-dependent test, or the evaluator code, changed
    LABELS = "labels"  # test labels, the log parser or the grading rules changed
    EXECUTION = "execution"  # the eval script, the image or the patch-application commands changed
    TOOL_BEHAVIOR = "tool_behavior"  # tool implementations or backends return different outputs
    INTERFACE = "interface"  # tool interfaces, documentation or prompts changed: the policy saw another task
    NEW_WORLD_ONLY = "new_world_only"  # truth is defined only for a newer world than the record observed


# change -> {level: disposition}; a level not listed inherits the disposition of the closest lower level listed
_TABLE: dict[ChangeType, dict[Level, str]] = {
    ChangeType.UNTOUCHED: {Level.SCORE: "rerun", Level.STAMPS: "keep"},
    ChangeType.ANSWER_TRUTH: {Level.SCORE: "rerun", Level.IO: "reparse"},
    ChangeType.WORLD_DATA: {Level.SCORE: "rerun", Level.IO: "reparse"},
    ChangeType.STATE_TEST: {Level.SCORE: "rerun", Level.IO: "replay", Level.STATE: "reexec"},
    ChangeType.LABELS: {Level.SCORE: "rerun", Level.STATE: "reparse"},
    ChangeType.EXECUTION: {Level.SCORE: "rerun", Level.STATE: "reexec"},
    ChangeType.TOOL_BEHAVIOR: {Level.SCORE: "rerun", Level.IO: "replay"},
    ChangeType.INTERFACE: {Level.SCORE: "neutralize"},
    ChangeType.NEW_WORLD_ONLY: {Level.SCORE: "neutralize"},
}

# what each (change, disposition) needs the record to hold, by role (any one of the alternatives)
_NEEDS: dict[tuple[ChangeType, str], tuple[str, ...]] = {
    (ChangeType.ANSWER_TRUTH, "reparse"): ("answer",),
    (ChangeType.WORLD_DATA, "reparse"): ("answer",),
    (ChangeType.STATE_TEST, "replay"): ("commands", "tool_calls"),
    (ChangeType.STATE_TEST, "reexec"): ("end_state",),
    (ChangeType.LABELS, "reparse"): ("test_log",),
    (ChangeType.EXECUTION, "reexec"): ("patch", "end_state", "snapshot"),
    (ChangeType.TOOL_BEHAVIOR, "replay"): ("commands", "tool_calls"),
}


def disposition(change: ChangeType | str, level: Level | str) -> str:
    """What a record at ``level`` allows after a grader change of type ``change``."""
    change, level = ChangeType(change), Level(level)
    row = _TABLE[change]
    best = row[Level.SCORE]
    for lv in _LEVEL_ORDER:
        if lv in row and lv.rank <= level.rank:
            best = row[lv]
    return best


def minimum_level(change: ChangeType | str) -> Level | None:
    """The lowest level at which a change is handled without a new episode (None when no level suffices)."""
    change = ChangeType(change)
    for lv in _LEVEL_ORDER:
        if disposition(change, lv) not in ("rerun", "neutralize"):
            return lv
    return None


def table() -> list[dict]:
    """The disposition table, one row per change type (for documentation and reports)."""
    rows = []
    for change in ChangeType:
        rows.append({"change": change.value, **{lv.value: disposition(change, lv) for lv in _LEVEL_ORDER},
                     "minimum_level": (minimum_level(change).value if minimum_level(change) else None)})
    return rows


def _roles(record: ExecutionRecord) -> set[str]:
    roles = {a.role for a in record.artifacts}
    roles.update(o.role for r in record.runs for o in r.outputs)
    return roles


def level_of(record: ExecutionRecord, role_levels: Mapping[str, Level] = ROLE_LEVELS) -> Level:
    """The highest level of evidence the record holds."""
    level = Level.SCORE
    if record.world_version is not None or record.runs:
        level = Level.STAMPS
    for role in _roles(record):
        lv = role_levels.get(role)
        if lv is not None and level.rank < lv.rank:
            level = lv
    return level


@dataclass(frozen=True)
class Sufficiency:
    """Whether a record's evidence supports the disposition its level promises for a change."""

    change: ChangeType
    level: Level
    disposition: str
    ok: bool
    missing: tuple[str, ...] = ()


def sufficient(record: ExecutionRecord, change: ChangeType | str,
               role_levels: Mapping[str, Level] = ROLE_LEVELS) -> Sufficiency:
    """The disposition a change gets for this record, checked against the roles it actually holds: a record at L2 that
    lacks the one role the change needs falls back to what the next lower level allows."""
    change = ChangeType(change)
    held = _roles(record)
    level = level_of(record, role_levels)
    missing: tuple[str, ...] = ()
    for lv in reversed(_LEVEL_ORDER):
        if lv.rank > level.rank:
            continue
        disp = disposition(change, lv)
        needs = _NEEDS.get((change, disp))
        if needs is not None and not held.intersection(needs):
            if not missing:
                missing = tuple(sorted(needs))  # what the record's own level would have needed
            continue
        return Sufficiency(change, lv, disp, ok=disp not in ("rerun", "neutralize"), missing=missing)
    raise AssertionError("the L0 disposition needs no evidence")  # pragma: no cover


def classify_change(old: Check, new: Check) -> str:
    """``none``, ``grading`` or ``execution``: which identity of a check changed (execution wins)."""
    if old.execution_digest != new.execution_digest:
        return "execution"
    if old.grading_digest != new.grading_digest:
        return "grading"
    return "none"


def mask(record: ExecutionRecord, level: Level | str,
         role_levels: Mapping[str, Level] = ROLE_LEVELS) -> ExecutionRecord:
    """A copy of ``record`` holding only the evidence at or below ``level`` (a different record, with its own id).

    L0 keeps no stamps: no world version and no runs. L0+ keeps the runs' execution identities but none of their
    outputs, and no artifacts. Higher levels keep the artifacts and outputs whose role is at or below the level; a role
    with no level in ``role_levels`` is kept from L1 up."""
    level = Level(level)
    if level is Level.SCORE:
        return dataclasses.replace(record, world_version=None, world_observations=(), artifacts=(), runs=())

    def keep(role: str) -> bool:
        return level is not Level.STAMPS and role_levels.get(role, Level.IO).rank <= level.rank

    runs = tuple(dataclasses.replace(r, outputs=tuple(o for o in r.outputs if keep(o.role))) for r in record.runs)
    artifacts = tuple(a for a in record.artifacts if keep(a.role))
    return dataclasses.replace(record, artifacts=artifacts, runs=runs)


def mask_score(score: ScoreRecord, level: Level | str, execution_id: str | None = None) -> ScoreRecord:
    """A recorded score as it would be kept at ``level``: at L0 only the outcome (no identities, so it cannot be
    reused under a changed grader). ``execution_id`` re-targets the score to a masked record's id."""
    level = Level(level)
    changes: dict = {}
    if execution_id is not None:
        changes["execution_id"] = execution_id
    if level is Level.SCORE:
        changes.update(parts=(), skipped=(), spec_digest=None)
    return dataclasses.replace(score, **changes) if changes else score


def roles_needed(changes: Iterable[ChangeType | str]) -> set[str]:
    """Every role some change type would need for its best disposition."""
    out: set[str] = set()
    for change in changes:
        change = ChangeType(change)
        for (c, _), roles in _NEEDS.items():
            if c is change:
                out.update(roles)
    return out
