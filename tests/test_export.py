# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from agent_ecc.export import (
    LearnerLine,
    MixedVersionError,
    StepCompletion,
    compare_evaluations,
    consumption_records,
    evaluate,
    evaluation_manifest,
    group_normalized,
    reproduce_evaluation,
    training_manifest,
)
from agent_ecc.scoring import Scorer, grade_recorded
from agent_ecc.seal import Candidate, Member, SealSession
from agent_ecc.store import MemoryStore
from agent_ecc.versions import GraderVersion, MemoryRegistry

from toy import ToyAdapter, ToyBackend, check


@pytest.fixture
def world():
    store = MemoryStore()
    adapter = ToyAdapter()
    backend = ToyBackend()
    scorer = Scorer([adapter], store, backend)
    c1 = check("t", "plain", "ab", store)
    g1 = GraderVersion(name="g1", entries={"t": c1})
    records, recorded = [], {}
    for i, answer in enumerate(["ab", "ba", "AB", "ab"]):
        rec = adapter.record(store, task="t", trajectory=f"traj-{i}", answer=answer,
                             runs=[adapter.run(store, c1, answer)])
        records.append(rec)
        recorded[rec.record_id] = [grade_recorded(adapter, rec, c1, store, grader_version=g1.version_id)]
    return store, adapter, backend, scorer, g1, records, recorded


def test_evaluation_manifests_under_two_versions(world):
    store, adapter, backend, scorer, g1, records, recorded = world
    m1, s1 = evaluate(records, g1, scorer, prior=recorded)
    assert m1.counts["pass"] == 2 and m1.counts["fail"] == 2 and m1.undeterminable == 0
    assert m1.counts["by_method"] == {"recorded": 4} and m1.by_task["t"]["pass_rate"] == 0.5
    g2 = GraderVersion(name="g2", entries={"t": check("t", "up", "AB", store)})
    m2, s2 = evaluate(records, g2, scorer, prior=recorded)
    # two records hold the same answer: one execution serves both
    assert m2.counts["pass"] == 3 and m2.counts["by_method"] == {"reexec": 4} and len(backend.executed) == 3
    assert m1.record_set == m2.record_set and m1.manifest_id != m2.manifest_id
    diff = compare_evaluations(m1, m2)
    assert diff["flips"] == 1 and diff["fail_to_pass"] == 1 and diff["pass_to_fail"] == 0
    assert diff["pass_rate"] == {"from": 0.5, "to": 0.75}
    # old results stay reproducible: scoring again under g1 gives the same outcomes, with nothing executed
    before = len(backend.executed)
    assert reproduce_evaluation(m1, records, g1, scorer, prior=recorded)["reproduced"]
    assert len(backend.executed) == before
    again = json.loads(json.dumps(m1.to_dict()))
    assert again["manifest_id"] == m1.manifest_id


def test_evaluation_manifest_refuses_mixed_or_missing_scores(world):
    store, adapter, backend, scorer, g1, records, recorded = world
    _, s1 = evaluate(records, g1, scorer, prior=recorded)
    g2 = GraderVersion(name="g2", entries={"t": check("t", "plain", "ba", store)})
    _, s2 = evaluate(records, g2, scorer, prior=recorded)
    with pytest.raises(MixedVersionError):
        evaluation_manifest(records, s1[:2] + s2[2:], g1)
    with pytest.raises(ValueError):
        evaluation_manifest(records, s1[:3], g1)
    with pytest.raises(ValueError):
        evaluation_manifest(records[:3], s1, g1)
    with pytest.raises(ValueError):
        evaluation_manifest(records + records[:1], s1, g1)


def test_undeterminable_records_are_counted_separately(world):
    store, adapter, backend, scorer, g1, records, recorded = world
    bare = adapter.record(store, task="t", trajectory="bare", answer=None)
    g2 = GraderVersion(name="g2", entries={"t": check("t", "rev", "ba", store)})
    m, _ = evaluate(records + [bare], g2, scorer, prior=recorded)
    assert m.undeterminable == 1 and m.counts["pass"] + m.counts["fail"] == 4
    assert m.counts["pass_rate"] == 2 / 4


# ---------------------------------------------------------------------------------------------- consumption


def sealed_batch(world, batch_id="b1"):
    store, adapter, backend, scorer, g1, records, recorded = world
    registry = MemoryRegistry()
    registry.publish_grader(g1)
    members = [Member(member_id=f"m{i}", execution=r, prior=recorded[r.record_id],
                      recorded_reward=1.0 if recorded[r.record_id][0].passed else 0.0)
               for i, r in enumerate(records)]
    session = SealSession(registry, scorer, batch_id=batch_id)
    groups = session.decide([Candidate("u0", tuple(members[:2])), Candidate("u1", tuple(members[2:]))])
    return session.finish(groups)


def learner_lines(decision, *, std=True, **overrides):
    out = []
    for group in decision.groups:
        rewards = [m.trained_reward for m in group.members]
        norm = group_normalized(rewards, std=std)
        for m, n in zip(group.members, norm):
            line = dict(batch_id=decision.batch_id, member_id=m.member_id, group_id=group.group_id, step=1,
                        reward=m.trained_reward, normalized=n, advantage=n, advantage_min=n, advantage_max=n,
                        loss_tokens=10, handoff_loss_tokens=10, identity="verified", reporter="dp0",
                        microbatch=0)
            line.update(overrides.get(m.member_id, {}))
            out.append(LearnerLine(**line))
    return out


def test_consumption_join_ok(world):
    decision = sealed_batch(world)
    report = consumption_records([decision], learner_lines(decision), [StepCompletion("b1", step=1, completed_at=5)])
    (record,) = report.records
    assert record.ok and len(record.records) == 4 and record.decision_id == decision.decision_id
    assert all(r.execution_id and r.score_id for r in record.records)
    m = report.manifest
    assert m.ok and m.records_consumed == 4 and m.labels == {"reuse": 4}
    assert m.by_grader_version == {decision.epoch.version_id: 4}
    assert m.normalization["hypothesis"] == "mean_std" and m.consumption_ids == (record.consumption_id,)


def test_mean_only_normalization_is_detected(world):
    decision = sealed_batch(world)
    m = training_manifest([decision], learner_lines(decision, std=False), [StepCompletion("b1", step=1)])
    assert m.ok and m.normalization["hypothesis"] == "mean"
    forced = training_manifest([decision], learner_lines(decision, std=False), [StepCompletion("b1", step=1)],
                               normalization="mean_std")
    assert not forced.ok and forced.issues["advantage_inconsistent"] == 4  # both groups have variance


def test_each_mismatch_is_reported(world):
    decision = sealed_batch(world)
    lines = learner_lines(decision, m0={"reward": 0.0}, m1={"identity": "mismatch"}, m2={"group_id": "u0"},
                          m3={"loss_tokens": 7})
    lines.append(LearnerLine(batch_id="b1", member_id="stranger", group_id="u9", reward=1.0, identity="verified",
                             reporter="dp0", microbatch=0))
    report = consumption_records([decision], lines, [StepCompletion("b1", step=1)])
    issues = report.manifest.issues
    assert issues["reward_mismatch"] == 1 and issues["identity_unverified"] == 1
    assert issues["records_unsealed"] == 1 and issues["loss_tokens_mismatch"] == 1
    assert issues["group_mismatch"] >= 2  # m2 moved to u0: both groups' memberships are now wrong
    assert not report.manifest.ok


def test_unconsumed_and_duplicate_consumption(world):
    decision = sealed_batch(world)
    lines = [x for x in learner_lines(decision) if x.member_id != "m3"]
    dup = learner_lines(decision)[0]
    lines.append(LearnerLine(**{**dup.to_dict(), "reporter": "dp1"}))  # consumed by a second shard
    lines.append(LearnerLine(**{**dup.to_dict(), "microbatch": 5}))  # and in a second micro-batch
    report = consumption_records([decision], lines, [StepCompletion("b1", step=1)])
    assert report.manifest.issues["records_unconsumed"] == 1
    assert report.manifest.issues["duplicate_consumption"] == 1


def test_a_record_consumed_in_two_step_parts_is_a_duplicate_consumption(world):
    """One record trained in two parts of one step (two optimizer steps over one batch) by the same shard."""
    decision = sealed_batch(world)
    lines = learner_lines(decision)
    again = next(x for x in lines if x.member_id == "m0")
    lines.append(LearnerLine(**{**again.to_dict(), "step_part": 1}))
    report = consumption_records([decision], lines, [StepCompletion("b1", step=1)])
    (issue,) = report.records[0].issues["duplicate_consumption"]
    assert issue["member_id"] == "m0" and len(issue["consumptions"]) == 2 and not report.manifest.ok


def test_a_repeated_microbatch_and_context_parallel_parts_count_once(world):
    decision = sealed_batch(world)
    lines = learner_lines(decision)
    lines += learner_lines(decision)  # the loss ran twice: identical lines
    halves = []
    for line in learner_lines(decision):
        if line.member_id == "m0":
            halves += [LearnerLine(**{**line.to_dict(), "reporter": "dp0:cp:0", "loss_tokens": 6}),
                       LearnerLine(**{**line.to_dict(), "reporter": "dp0:cp:1", "loss_tokens": 4})]
    lines = [x for x in lines if x.member_id != "m0"] + halves
    report = consumption_records([decision], lines, [StepCompletion("b1", step=1)])
    assert report.manifest.ok, report.manifest.issues


def test_only_completed_logged_steps_are_consumption(world):
    d1 = sealed_batch(world, "b1")
    d2 = sealed_batch(world, "b2")
    d3 = sealed_batch(world, "b3")
    lines = learner_lines(d1) + learner_lines(d2)
    completions = [StepCompletion("b1", step=1), StepCompletion("b3", step=3), StepCompletion("b2", ok=False)]
    report = consumption_records([d1, d2, d3], lines, completions)
    m = report.manifest
    assert [r.batch_id for r in report.records] == ["b1"]
    assert m.incomplete_batches == ("b2",)  # logged, but the step never completed
    assert m.unconfirmed_batches == ("b3",)  # completed, but no log confirms what it consumed
    assert not m.ok
    report = consumption_records([d1, d2, d3], learner_lines(d1), [StepCompletion("b1", step=1)])
    assert report.manifest.unconsumed_seals == ("b2", "b3") and report.manifest.ok


def test_a_reseal_replaces_the_earlier_decision(world):
    first = sealed_batch(world, "b1")
    second = sealed_batch(world, "b1")
    report = consumption_records([first, second], learner_lines(second), [StepCompletion("b1", step=1)])
    assert report.manifest.resealed_batches == ("b1",)
    assert report.records[0].decision_id == second.decision_id and report.manifest.ok


def test_learner_lines_without_a_seal(world):
    decision = sealed_batch(world)
    report = consumption_records([], learner_lines(decision), [StepCompletion("b1", step=1)])
    assert report.manifest.issues["records_unsealed"] == 4 and not report.manifest.ok


def test_learner_line_from_dict_is_tolerant():
    line = LearnerLine.from_dict({"batch_id": 7, "member_id": "m", "reward": "1.0", "loss_tokens": "12",
                                  "advantage": None, "extra": "ignored"})
    assert line.batch_id == "7" and line.reward == 1.0 and line.loss_tokens == 12 and line.shard == "None"
    assert group_normalized([1.0, 0.0], std=False) == [0.5, -0.5]
    assert group_normalized([1.0, 1.0], std=True) == [0.0, 0.0]
