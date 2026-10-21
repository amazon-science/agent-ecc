# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import pytest

from agent_ecc.records import (
    ConsumedRecord,
    ConsumptionRecord,
    Cost,
    EpochRef,
    EvidenceRef,
    Exclusion,
    ExecutionRecord,
    GroupDecision,
    IdentityMismatch,
    MemberDecision,
    PartScore,
    ScoreRecord,
    SealDecision,
    TestOutcome,
    TestRun,
    dump_jsonl,
    load_jsonl,
    record_from_dict,
)
from agent_ecc.canonical import digest_bytes

D1 = digest_bytes(b"patch")
D2 = digest_bytes(b"log")


def execution(**changes):
    base = dict(family="toy", task_id="t1", trajectory_id="traj-1", world_version="w0",
                artifacts=[{"role": "patch", "digest": D1, "size": 5}],
                runs=[{"check": "tests", "execution": {"image": "img", "script": D1},
                       "outputs": [{"role": "test_log", "digest": D2}], "facts": {"applied": True}}],
                facts={"note": [1, 2]})
    base.update(changes)
    return ExecutionRecord(**base)


def score(record, **changes):
    base = dict(execution_id=record.record_id, grader_version="sha256:" + "1" * 64, outcome="fail",
                method="reparse", spec_digest="sha256:" + "2" * 64, reason="tests_failed",
                parts=[PartScore(check="tests", execution_digest="sha256:" + "3" * 64,
                                 grading_digest="sha256:" + "4" * 64, outcome="fail", method="reparse",
                                 tests=[TestOutcome("a", "failed", "FAIL_TO_PASS")])],
                cost=Cost(seconds=0.5))
    base.update(changes)
    return ScoreRecord(**base)


def test_execution_record_is_content_addressed_and_immutable():
    a, b = execution(), execution()
    assert a.record_id == b.record_id and a == b and hash(a) == hash(b)
    assert execution(trajectory_id="traj-2").record_id != a.record_id
    assert isinstance(a.runs[0], TestRun) and isinstance(a.artifacts[0], EvidenceRef)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.task_id = "x"  # type: ignore[misc]
    with pytest.raises(TypeError):
        a.facts["note"] = 3  # type: ignore[index]
    assert a.artifact("patch").digest == D1 and a.artifact("missing") is None
    assert {r.role for r in a.evidence()} == {"patch", "test_log"}


def test_execution_record_round_trip_verifies_identity():
    a = execution()
    d = json.loads(json.dumps(a.to_dict()))
    assert ExecutionRecord.from_dict(d) == a
    d["task_id"] = "tampered"
    with pytest.raises(IdentityMismatch):
        ExecutionRecord.from_dict(d)
    assert ExecutionRecord.from_dict(d, verify=False).task_id == "tampered"


def test_torn_record_observed_another_world():
    assert not execution().torn
    assert not execution(world_observations=("w0", "w0")).torn
    assert execution(world_observations=("w0", "w1")).torn


def test_evidence_ref_requires_a_digest():
    with pytest.raises(ValueError):
        EvidenceRef(role="x", digest="md5:abc")


def test_score_record_round_trip_and_consistency_rules():
    rec = execution()
    s = score(rec)
    d = json.loads(json.dumps(s.to_dict()))
    assert ScoreRecord.from_dict(d) == s and s.passed is False and s.logical
    assert s.tests()[0].name == "a"
    with pytest.raises(ValueError):
        score(rec, outcome="undeterminable", method="reparse")
    with pytest.raises(ValueError):
        score(rec, outcome="pass", method="undeterminable")
    und = score(rec, outcome="undeterminable", method="undeterminable", parts=())
    assert und.passed is None and not und.logical
    d["outcome"] = "pass"
    with pytest.raises(IdentityMismatch):
        ScoreRecord.from_dict(d)


def test_same_verdict_ignores_cost_and_provenance():
    rec = execution()
    a, b = score(rec), score(rec, cost=Cost(seconds=9.0), based_on="sha256:" + "5" * 64)
    assert a.score_id != b.score_id and a.same_verdict(b)
    assert not a.same_verdict(score(rec, outcome="pass", method="reparse", parts=()))


def test_new_runs_are_reexec_parts_only():
    rec = execution()
    run = TestRun(check="tests", execution={"image": "img2"})
    p = PartScore(check="tests", execution_digest="sha256:" + "6" * 64, grading_digest="sha256:" + "7" * 64,
                  outcome="pass", method="reexec", run=run)
    s = score(rec, outcome="pass", method="reexec", parts=[p])
    assert s.new_runs() == (run,)


def _decision(**changes):
    members = (MemberDecision(member_id="m0", execution_id="sha256:" + "8" * 64, action="reparse", outcome="pass",
                              recorded_reward=0.0, target_reward=1.0, trained_reward=1.0, flipped=True),
               MemberDecision(member_id="m1", execution_id=None, action="undeterminable", outcome="undeterminable",
                              reason="torn_observation", recorded_reward=1.0, trained_reward=0.0))
    base = dict(seal_id="s1", epoch=EpochRef(seq=3, version_id="sha256:" + "9" * 64, name="g", published_at=1.0,
                                             read_at=2.0),
                mode="repair", policy="ecc", strict=True,
                groups=(GroupDecision(group_id="u0", members=members, neutralized=True,
                                      neutralize_reason="undeterminable_member"),),
                batch_id="7", target_groups=2,
                excluded=(Exclusion(group=GroupDecision(group_id="u1", members=()), reason="zero_variance"),))
    base.update(changes)
    return SealDecision(**base)


def test_member_status_names():
    d = _decision()
    statuses = [m.status for _, m in d.members()]
    assert statuses == ["repaired_flip", "torn"]
    assert MemberDecision(member_id="x", execution_id=None, action="reuse", consistent=True).status == "consistent"
    assert MemberDecision(member_id="x", execution_id=None, action="reuse", short_circuit=True).status == \
        "repaired_short_same"
    assert MemberDecision(member_id="x", execution_id=None, action="reuse", flipped=True).status == \
        "repaired_reuse_flip"
    assert MemberDecision(member_id="x", execution_id=None, action="reparse", short_circuit=True,
                          flipped=True).status == "repaired_short_flip"
    assert MemberDecision(member_id="x", execution_id=None, action="pending", reason="needs_reparse").status == \
        "needs_reparse"
    with pytest.raises(ValueError):
        MemberDecision(member_id="x", execution_id=None, action="reparse", consistent=True)
    assert MemberDecision(member_id="x", execution_id=None, action="reexec", flipped=False).status == \
        "repaired_exec_same"
    assert MemberDecision(member_id="x", execution_id=None, action="undeterminable",
                          reason="missing_artifact:no_run").status == "unrepairable"
    with pytest.raises(ValueError):
        MemberDecision(member_id="x", execution_id=None, action="guess")


def test_seal_decision_round_trip_shortfall_and_rewards():
    d = _decision()
    again = SealDecision.from_dict(json.loads(json.dumps(d.to_dict())))
    assert again == d and again.decision_id == d.decision_id
    assert d.shortfall == 1 and d.neutralized_groups == 1
    assert d.rewards() == {"m0": 1.0, "m1": 0.0}
    with pytest.raises(ValueError):
        GroupDecision(group_id="g", members=(), role="extra")


def test_consumption_record_ok_and_round_trip():
    c = ConsumptionRecord(batch_id="7", step=7, seal_id="s1", decision_id="sha256:" + "a" * 64,
                          epoch=EpochRef(seq=1, version_id="sha256:" + "b" * 64), completed_at=3.0,
                          records=(ConsumedRecord(member_id="m0", group_id="u0", execution_id=None, score_id=None,
                                                  reward=1.0, sealed_reward=1.0),),
                          issues={"reward_mismatch": []})
    assert c.ok
    again = ConsumptionRecord.from_dict(json.loads(json.dumps(c.to_dict())))
    assert again == c
    bad = dataclasses.replace(c, issues={"reward_mismatch": [{"member_id": "m0"}]})
    assert not bad.ok and bad.consumption_id != c.consumption_id


def test_jsonl_dispatch_and_torn_tail(tmp_path):
    path = tmp_path / "records.jsonl"
    rec = execution()
    s = score(rec)
    d = _decision()
    assert dump_jsonl(path, [rec, s, d]) == 3
    with open(path, "a") as f:
        f.write('{"schema": "agent-ecc/score/1", "execution_')  # a writer's partial append
    loaded = load_jsonl(path)
    assert loaded == [rec, s, d]
    assert record_from_dict(rec.to_dict()) == rec
    with pytest.raises(ValueError):
        record_from_dict({"schema": "nope"})
