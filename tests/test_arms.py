# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The four arms of a grader switch (keep, discard, full-compatible, ECC) on a SWE-ABS conjunction epoch.

e0 grades each task by its original suite; e1 by the original suite AND a hardened suite graded against a reference set
G, on the patch without its diffs of the hardened test files. Members are graded under the epoch they pinned at start;
the seal decides them under the batch epoch: an original-suite fail decides an e1 fail without executing anything (the
short circuit), an original-suite pass needs the hardened suite executed, a member graded under e1 is consistent, and
a member whose label cannot be decided neutralizes its group (ECC and full-compatible)."""

import dataclasses
import json

import pytest

from abs_world import World, statuses, trained

from agent_ecc.records import Outcome
from agent_ecc.recompute.base import RecordedBackend, RecordingBackend
from agent_ecc.scoring import Scorer
from agent_ecc.seal import Arm, Candidate, Member, arm_session
from agent_ecc.versions import Epoch


def test_an_original_suite_fail_decides_an_e1_fail_without_execution():
    w = World()
    w.switch()
    _, d = w.seal("ecc", [[w.member(w.e0, [False]), w.member(w.e0, [False])]])
    assert statuses(d) == ["repaired_short_same", "repaired_short_same"] and w.executions == []
    assert trained(d) == [0.0, 0.0] and d.neutralized_groups == 0
    assert all(m.short_circuit for _, m in d.members())


def test_an_original_suite_pass_executes_only_the_hardened_suite():
    w = World()
    w.switch()
    _, d = w.seal("ecc", [[w.member(w.e0, [True], patch="diff fix"),
                           w.member(w.e0, [True], patch="diff fix fragile")]])
    assert statuses(d) == ["repaired_exec_same", "repaired_exec_flip"] and trained(d) == [1.0, 0.0]
    assert sorted(w.executions) == [("diff fix", "hardened"), ("diff fix fragile", "hardened")]


def test_members_graded_under_the_batch_epoch_are_consistent():
    w = World()
    w.switch()
    group = [w.member(w.e1, [True, True]), w.member(w.e1, [True, False]), w.member(w.e1, [False, None])]
    _, d = w.seal("ecc", [group])
    assert statuses(d) == ["consistent"] * 3 and trained(d) == [1.0, 0.0, 0.0] and w.executions == []


def test_an_infrastructure_error_of_the_original_suite_never_short_circuits():
    w = World()
    w.switch()
    _, d = w.seal("ecc", [[w.member(w.e1, [None, False], patch="diff fix fragile"), w.member(w.e0, [True])]])
    # a recorded hardened fail decides e1 whatever the errored original suite says; nothing is executed for it
    assert statuses(d)[0] == "consistent" and d.groups[0].members[0].outcome is Outcome.FAIL
    assert ("diff fix fragile", "tests") not in w.executions


def test_an_unrepairable_member_neutralizes_its_group():
    w = World(error_on={"hardened"})
    w.switch()
    _, d = w.seal("ecc", [[w.member(w.e0, [True]), w.member(w.e0, [False])]])
    assert statuses(d) == ["unrepairable", "repaired_short_same"]
    assert d.neutralized_groups == 1 and trained(d) == [0.0, 0.0]
    assert len(w.executions) == 3  # the hardened suite, retried twice


def test_an_unrepairable_member_neutralizes_its_group_under_full_compatible():
    w = World(error_on={"hardened"})
    w.switch()
    _, d = w.seal("full_compatible", [[w.member(w.e1, [True, True], patch="diff fix"),
                                       w.member(w.e0, [False], patch="diff nothing")]])
    # the re-verification's hardened suite never gives a verdict: the member is unrepairable, its group neutralized
    assert d.policy == "full" and statuses(d) == ["unrepairable", "repaired_exec_same"]
    assert d.neutralized_groups == 1 and d.groups[0].neutralize_reason == "undeterminable_member"
    assert trained(d) == [0.0, 0.0]
    assert w.executions.count(("diff fix", "hardened")) == 3  # executed, then retried twice


def test_a_member_recorded_undeterminable_is_unrepairable_under_every_arm_in_the_trainer_compatible_setting():
    """A member whose recorded verdict is undeterminable (its original suite's run errored when it was graded under
    e0). By default ecc and full_compatible execute it again and keep and discard leave it pending. With
    Scorer(recorded_undeterminable="final"), the setting that reproduces a trainer which calls such a member
    unrepairable, every arm calls it unrepairable without executing its patch: ecc, full_compatible and discard
    neutralize its group, keep trains the recorded rewards."""
    by_default = {"keep": "needs_reexec", "discard": "needs_reexec", "ecc": "repaired_exec_flip",
                  "full_compatible": "repaired_exec_flip"}
    for arm in ("keep", "discard", "ecc", "full_compatible"):
        for setting in ("resolve", "final"):
            w = World()
            w.switch()
            scorer = Scorer([w.adapter], w.store, w.backend, recorded_undeterminable=setting)
            session = arm_session(arm, w.registry, scorer, cache={} if arm in ("ecc", "full_compatible") else None)
            errored, failed = w.member(w.e0, [None], patch="diff fix"), w.member(w.e0, [False], patch="diff nothing")
            assert errored.prior[0].outcome is Outcome.UNDETERMINABLE and errored.recorded_reward == 0.0
            d = session.finish(session.decide([Candidate(group_id="g", members=(errored, failed))]))
            first = d.groups[0].members[0]
            executed_errored = any(patch == "diff fix" for patch, _ in w.executions)
            if setting == "resolve":
                assert first.status == by_default[arm], arm
                assert executed_errored == (arm in ("ecc", "full_compatible")), arm
                continue
            assert first.status == "unrepairable" and first.reason == "recorded_undeterminable", arm
            assert not executed_errored, arm
            assert d.groups[0].neutralized == (arm != "keep"), arm
            assert trained(d) == [0.0, 0.0], arm


def test_a_member_without_a_record_trains_its_recorded_reward_only_under_keep():
    """A member whose record the seal cannot find (its producer's record was stored under another id) has no label
    under the batch epoch. keep trains the recorded labels, its own included, so the post-seal filter keeps a group
    whose recorded rewards vary; the other arms neutralize the group."""
    for arm in ("keep", "discard", "ecc", "full_compatible"):
        w = World()
        w.switch()
        scorer = Scorer([w.adapter], w.store, w.backend)
        session = arm_session(arm, w.registry, scorer, cache={} if arm in ("ecc", "full_compatible") else None)
        lost = Member(member_id="lost", recorded_reward=1.0)
        assembly = session.assemble([Candidate(group_id="g", members=(lost, w.member(w.e0, [False])))],
                                    fillers="neutral")
        ((_, group),) = assembly.batch
        assert group.members[0].status == "no_record", arm
        if arm == "keep":
            assert group.role == "selected" and not group.neutralized and group.rewards() == [1.0, 0.0]
        else:
            assert group.role == "filler" and group.neutralized and group.rewards() == [0.0, 0.0], arm


def test_arm_seals_sharing_a_cache_replay_from_their_recordings(tmp_path):
    """Two full_compatible seals of one run share executions through the run's cache. Recorded live
    (RecordingBackend) and replayed from the recordings alone (RecordedBackend, a fresh cache for the run, each seal
    under the epoch in force when it read the pointer), they decide the same, and the replay asks for exactly the
    executions the live seals made, each once."""
    w = World()
    w.switch()
    results = tmp_path / "results.jsonl"
    recording = RecordingBackend(w.backend, results)
    plans = [[[w.member(w.e1, [True, True], patch="diff fix"), w.member(w.e0, [True], patch="diff fix fragile")]],
             [[w.member(w.e0, [True], patch="diff fix"), w.member(w.e0, [False], patch="diff nothing")]]]

    def run(backend, cache, epochs, startup=None):
        out = []
        for i, groups in enumerate(plans):
            scorer = Scorer([w.adapter], w.store, backend, clock=lambda: 0.0)
            source = epochs(i) if callable(epochs) else epochs
            session = arm_session("full_compatible", source, scorer, cache=cache, startup=startup, seal_id=f"s{i}",
                                  batch_id=f"step-{i}", clock=lambda: 5.0)
            candidates = [Candidate(group_id=f"s{i}g{j}", members=tuple(g)) for j, g in enumerate(groups)]
            out.append((session.epoch.ref.read_at, session.finish(session.decide(candidates))))
        return out

    live = run(recording, {}, w.registry)
    # the second seal's first member was answered by the cache: both its suites ran in the first seal
    assert w.executions.count(("diff fix", "tests")) == 1 and w.executions.count(("diff fix", "hardened")) == 1
    assert len(w.executions) == 5 and [d.policy for _, d in live] == ["full", "full"]

    def epoch_at(i):
        read_at = live[i][0]
        ref = w.registry.epoch_at(read_at)
        return Epoch(ref=dataclasses.replace(ref, read_at=read_at), version=w.registry.grader(ref.version_id))

    recorded = RecordedBackend.from_jsonl(results)
    startup = w.registry.publications("grader")[0].seq
    again = run(recorded, {}, epoch_at, startup=startup)
    def essence(decision):  # everything a seal decided, its timing aside
        d = decision.to_dict()
        for key in ("opened_at", "closed_at", "decision_id"):
            d.pop(key)
        d["stats"].pop("decide_seconds")
        return d

    assert [essence(d) for _, d in again] == [essence(d) for _, d in live]
    executed = [json.loads(line)["request_id"] for line in open(results, encoding="utf-8")]
    assert recorded.missing == [] and sorted(recorded.served) == sorted(executed) and len(set(executed)) == 5


def test_the_ecc_arm_shares_executions_across_seals():
    w = World()
    w.switch()
    w.seal("ecc", [[w.member(w.e0, [True], patch="diff fix")]])
    _, d = w.seal("ecc", [[w.member(w.e0, [True], patch="diff fix")]])
    assert w.executions == [("diff fix", "hardened")] and statuses(d) == ["repaired_exec_same"]


def test_keep_decides_what_repair_would_and_trains_the_recorded_labels():
    w = World()
    w.switch()
    session, d = w.seal("keep", [[w.member(w.e0, [True]), w.member(w.e0, [False])]])
    assert session.mode.value == "audit" and d.mode == "audit"
    assert statuses(d) == ["needs_reexec", "repaired_short_same"] and trained(d) == [1.0, 0.0]
    assert w.executions == [] and d.neutralized_groups == 0


def test_discard_neutralizes_every_group_with_a_member_graded_under_another_epoch():
    w = World()
    w.switch()
    old = [w.member(w.e0, [False]), w.member(w.e0, [True])]
    new = [w.member(w.e1, [True, True]), w.member(w.e1, [True, False])]
    _, d = w.seal("discard", [old, new])
    assert [g.neutralized for g in d.groups] == [True, False] and w.executions == []
    assert trained(d) == [0.0, 0.0, 1.0, 0.0]
    assert statuses(d)[:2] == ["repaired_short_same", "needs_reexec"]


def test_full_compatible_re_verifies_every_member_after_the_switch():
    w = World()
    w.switch()
    session, d = w.seal("full_compatible", [[w.member(w.e1, [True, True], patch="diff fix fragile"),
                                             w.member(w.e0, [True], patch="diff nothing")]])
    assert session.policy == "full" and d.policy == "full"
    # the e1-graded member's recorded pass is not trusted; the short circuit stays (the original suite fails 'nothing')
    assert statuses(d) == ["repaired_exec_flip", "repaired_exec_flip"] and trained(d) == [0.0, 0.0]
    assert sorted(w.executions) == [("diff fix fragile", "hardened"), ("diff fix fragile", "tests"),
                                    ("diff nothing", "tests")]


def test_full_compatible_deduplicates_within_and_caches_across_seals():
    w = World()
    w.switch()
    _, d = w.seal("full_compatible", [[w.member(w.e1, [True, True], patch="diff fix"),
                                       w.member(w.e0, [True], patch="diff fix")]])
    assert sorted(w.executions) == [("diff fix", "hardened"), ("diff fix", "tests")]
    assert statuses(d) == ["repaired_exec_same", "repaired_exec_same"]
    _, d = w.seal("full_compatible", [[w.member(w.e1, [True, True], patch="diff fix")]])
    assert len(w.executions) == 2 and statuses(d) == ["repaired_exec_same"]  # both suites answered by the cache


def test_full_compatible_trusts_the_stamps_before_the_switch():
    w = World()
    session, d = w.seal("full_compatible", [[w.member(w.e0, [True]), w.member(w.e0, [False])]])
    assert session.policy == "ecc" and statuses(d) == ["consistent", "consistent"] and w.executions == []
    w.switch()
    session, d = w.seal("full_compatible", [[w.member(w.e0, [False])]])
    # after the switch the recorded fail is not trusted: both suites run, in spec order
    assert session.policy == "full" and w.executions == [("diff fix", "tests"), ("diff fix", "hardened")]
    assert statuses(d) == ["repaired_exec_flip"]


def test_full_compatible_needs_a_cache_and_a_startup_epoch():
    w = World()
    scorer = Scorer([w.adapter], w.store, w.backend)
    with pytest.raises(ValueError, match="cache"):
        arm_session("full_compatible", w.registry, scorer)
    epoch = w.registry.read_epoch()
    with pytest.raises(ValueError, match="startup"):
        arm_session(Arm.FULL_COMPATIBLE, epoch, scorer, cache={})
    assert arm_session(Arm.FULL_COMPATIBLE, epoch, scorer, cache={}, startup=epoch.ref.seq).policy == "ecc"
    with pytest.raises(ValueError):
        Arm("fullcompat")


def test_every_arm_plans_as_ecc_does_whatever_the_scorer_policy():
    """keep and discard reuse what the identities allow even when the scorer given was built for a re-verification."""
    for arm in ("keep", "discard"):
        w = World()
        w.switch()
        scorer = Scorer([w.adapter], w.store, w.backend, policy="full")
        session = arm_session(arm, w.registry, scorer)
        group = [w.member(w.e1, [True, True]), w.member(w.e1, [True, False])]
        d = session.finish(session.decide([Candidate(group_id="g", members=tuple(group))]))
        assert session.policy == "ecc" and statuses(d) == ["consistent", "consistent"], arm
        assert d.neutralized_groups == 0 and w.executions == []


def test_a_patchless_member_is_decided_by_its_evidence_under_every_arm():
    for arm in ("keep", "discard", "full_compatible", "ecc"):
        w = World()
        w.switch()
        _, d = w.seal(arm, [[w.member(w.e0, [False], patch="")]])
        (member,) = [m for _, m in d.members()]
        assert member.outcome is Outcome.FAIL and member.reason == "no_agent_patch", arm
        assert w.executions == [], arm


def test_an_arm_session_reads_the_epoch_pointer_once():
    w = World()
    scorer = Scorer([w.adapter], w.store, w.backend)
    session = arm_session("ecc", w.registry, scorer)
    w.switch()  # published after the read: this batch stays under e0
    assert session.epoch.ref.name == "e0"
    _, d = w.seal("ecc", [[w.member(w.e0, [True])]])
    assert d.epoch.name == "e1"
