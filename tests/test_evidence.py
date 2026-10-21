# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from agent_ecc.evidence import (
    ChangeType,
    Level,
    classify_change,
    disposition,
    level_of,
    mask,
    mask_score,
    minimum_level,
    sufficient,
    table,
)
from agent_ecc.canonical import digest_bytes
from agent_ecc.records import EvidenceRef, ExecutionRecord, TestRun
from agent_ecc.scoring import ScoreInput, Scorer, grade_recorded
from agent_ecc.store import MemoryStore
from agent_ecc.versions import Check, GraderVersion

from toy import ToyAdapter, ToyBackend, check


def ref(role):
    return EvidenceRef(role=role, digest=digest_bytes(role.encode()))


def record(*roles, runs_outputs=(), world="w0", runs=True):
    run = TestRun(check="tests", execution={"image": "i"}, outputs=tuple(ref(r) for r in runs_outputs))
    return ExecutionRecord(family="x", task_id="t", trajectory_id="tr", world_version=world,
                           artifacts=tuple(ref(r) for r in roles), runs=(run,) if runs else ())


def test_disposition_table():
    assert disposition("labels", "L2") == "reparse" and disposition("labels", "L1") == "rerun"
    assert disposition(ChangeType.EXECUTION, Level.STATE) == "reexec"
    assert disposition("execution", "L2'") == "reexec"  # inherited from L2
    assert disposition("state_test", "L1") == "replay" and disposition("state_test", "L2") == "reexec"
    assert disposition("answer_truth", "L1") == "reparse" and disposition("answer_truth", "L0+") == "rerun"
    assert disposition("untouched", "L0") == "rerun" and disposition("untouched", "L0+") == "keep"
    assert disposition("interface", "L2'") == "neutralize"
    assert minimum_level("labels") is Level.STATE and minimum_level("untouched") is Level.STAMPS
    assert minimum_level("new_world_only") is None
    rows = {r["change"]: r for r in table()}
    assert rows["tool_behavior"]["L1"] == "replay" and rows["world_data"]["minimum_level"] == "L1"
    assert Level.SCORE.rank < Level.STAMPS.rank < Level.IO.rank < Level.STATE.rank < Level.SNAPSHOT.rank


def test_level_of_records():
    assert level_of(record(runs=False, world=None)) is Level.SCORE
    assert level_of(record()) is Level.STAMPS
    assert level_of(record("answer")) is Level.IO
    assert level_of(record("answer", runs_outputs=["test_log"])) is Level.STATE
    assert level_of(record("patch", "snapshot")) is Level.SNAPSHOT


def test_sufficiency_checks_the_roles_actually_held():
    s = sufficient(record("patch", runs_outputs=["test_log"]), "labels")
    assert s.ok and s.disposition == "reparse" and s.level is Level.STATE
    s = sufficient(record("patch"), "labels")  # L2 by its patch, but no log to re-parse
    assert not s.ok and s.disposition == "rerun" and s.missing == ("test_log",)
    assert sufficient(record("patch"), "execution").disposition == "reexec"
    assert sufficient(record("answer"), "answer_truth").ok
    assert sufficient(record(runs=False, world=None), "untouched").disposition == "rerun"
    assert sufficient(record(), "untouched").ok
    assert not sufficient(record("patch", "answer"), "interface").ok


def test_classify_change():
    a = Check(name="t", execution={"x": "1"}, grading={"y": "1"})
    assert classify_change(a, a) == "none"
    assert classify_change(a, Check(name="t", execution={"x": "1"}, grading={"y": "2"})) == "grading"
    assert classify_change(a, Check(name="t", execution={"x": "2"}, grading={"y": "2"})) == "execution"


def test_masking_removes_evidence_above_a_level():
    full = record("answer", "patch", "snapshot", runs_outputs=["test_log"])
    assert level_of(mask(full, "L0")) is Level.SCORE and mask(full, "L0").runs == ()
    stamps = mask(full, Level.STAMPS)
    assert stamps.artifacts == () and stamps.runs[0].outputs == ()
    assert stamps.runs[0].execution == full.runs[0].execution
    assert {a.role for a in mask(full, "L1").artifacts} == {"answer"} and mask(full, "L1").runs[0].outputs == ()
    l2 = mask(full, "L2")
    assert {a.role for a in l2.artifacts} == {"answer", "patch"} and l2.runs[0].outputs[0].role == "test_log"
    assert mask(full, "L2'") == full
    assert mask(full, "L1").record_id != full.record_id


def test_an_ablation_re_decides_masked_records():
    store, adapter, backend = MemoryStore(), ToyAdapter(), ToyBackend()
    scorer = Scorer([adapter], store, backend)
    c1 = check("t", "plain", "ab", store)
    rec = adapter.record(store, task="t", trajectory="x", answer="ab", runs=[adapter.run(store, c1, "ab")])
    recorded = grade_recorded(adapter, rec, c1, store)
    relabel = GraderVersion(name="g2", entries={"t": check("t", "plain", "zz", store)})
    roles = {"answer": Level.STATE, "output": Level.STATE}  # the toy family's roles
    outcomes = {}
    for level in Level:
        masked = mask(rec, level, roles)
        prior = mask_score(recorded, level, execution_id=masked.record_id)
        outcomes[level.value] = scorer.score([ScoreInput(masked, prior)], relabel)[0].action
    assert outcomes == {"L0": "undeterminable", "L0+": "undeterminable", "L1": "undeterminable", "L2": "reparse",
                        "L2'": "reparse"}
