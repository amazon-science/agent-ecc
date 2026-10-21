# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent ECC: version-consistent experience consumption for agent training.

Validate recorded experience against a chosen verifier version, re-evaluate it when needed, and track what each
training step actually consumes.

The four record types live in :mod:`agent_ecc.records`, grader and world versions in :mod:`agent_ecc.versions`,
per-member decisions in :mod:`agent_ecc.compare`, scoring in :mod:`agent_ecc.scoring`, the seal in
:mod:`agent_ecc.seal`, prepared re-execution in :mod:`agent_ecc.prepare`, manifests in :mod:`agent_ecc.export`,
audits of a run's epochs and of its checks' determinism in :mod:`agent_ecc.audit`, evidence levels in
:mod:`agent_ecc.evidence`, re-execution backends in :mod:`agent_ecc.recompute`, and task-family adapters in
:mod:`agent_ecc.adapters`.
"""

from .audit import EpochAudit, VerdictConflicts, audit_epochs, verdict_conflicts
from .compare import Action, Policy, plan_check
from .export import (
    EvaluationManifest,
    LearnerLine,
    StepCompletion,
    TrainingManifest,
    compare_evaluations,
    consumption_records,
    evaluate,
    evaluation_manifest,
    reproduce_evaluation,
    training_manifest,
)
from .records import (
    ConsumedRecord,
    ConsumptionRecord,
    Cost,
    EpochRef,
    EvidenceRef,
    Exclusion,
    ExecutionRecord,
    GroupDecision,
    MemberDecision,
    Method,
    Outcome,
    PartScore,
    ScoreRecord,
    SealDecision,
    TargetPin,
    TestOutcome,
    TestRun,
    dump_jsonl,
    load_jsonl,
)
from .prepare import PreparedView, Preparer
from .scoring import Adapter, PartVerdict, ScoreInput, Scored, Scorer, grade_recorded
from .seal import (
    Arm,
    Assembly,
    Candidate,
    Member,
    Mode,
    SealFailed,
    SealJournal,
    SealSession,
    apply_rewards,
    arm_session,
    binary_reward,
    has_variance,
    seal,
)
from .store import ChainStore, ContentStore, DirectoryStore, LocatorStore, MemoryStore
from .versions import (
    ANY_WORLD,
    AllOf,
    Check,
    DirectoryRegistry,
    Epoch,
    EpochFirstError,
    GraderVersion,
    MemoryRegistry,
    Publication,
    Registry,
    WorldVersion,
)

__version__ = "0.2.1"

__all__ = [
    "ANY_WORLD",
    "Action",
    "Adapter",
    "AllOf",
    "Arm",
    "Assembly",
    "Candidate",
    "ChainStore",
    "Check",
    "ConsumedRecord",
    "ConsumptionRecord",
    "ContentStore",
    "Cost",
    "DirectoryRegistry",
    "DirectoryStore",
    "Epoch",
    "EpochAudit",
    "EpochFirstError",
    "EpochRef",
    "EvaluationManifest",
    "EvidenceRef",
    "Exclusion",
    "ExecutionRecord",
    "GraderVersion",
    "GroupDecision",
    "LearnerLine",
    "LocatorStore",
    "Member",
    "MemberDecision",
    "MemoryRegistry",
    "MemoryStore",
    "Method",
    "Mode",
    "Outcome",
    "PartScore",
    "PartVerdict",
    "Policy",
    "PreparedView",
    "Preparer",
    "Publication",
    "Registry",
    "ScoreInput",
    "ScoreRecord",
    "Scored",
    "Scorer",
    "SealDecision",
    "SealFailed",
    "SealJournal",
    "SealSession",
    "StepCompletion",
    "TargetPin",
    "TestOutcome",
    "TestRun",
    "TrainingManifest",
    "VerdictConflicts",
    "WorldVersion",
    "__version__",
    "apply_rewards",
    "arm_session",
    "audit_epochs",
    "binary_reward",
    "compare_evaluations",
    "consumption_records",
    "dump_jsonl",
    "evaluate",
    "evaluation_manifest",
    "grade_recorded",
    "has_variance",
    "load_jsonl",
    "plan_check",
    "reproduce_evaluation",
    "seal",
    "training_manifest",
    "verdict_conflicts",
]
