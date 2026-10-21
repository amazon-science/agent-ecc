# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A target pin and the ``never`` arm: e1 is published and producers grade under it, but every seal's batch epoch
stays the startup e0. A member graded under e1 is decided from its recorded e0 suite (a reuse, nothing executed); a
lagging member graded under e0 is consistent; the decision records the pin, the batch epoch and what the pointer
named; a pin the log does not hold leaves the batch without an epoch."""

import dataclasses

import pytest
from abs_world import World, statuses, trained

from agent_ecc.records import SealDecision
from agent_ecc.seal import Arm, Candidate, SealJournal, SealSession, arm_session, pinned_epoch


def _essence(decision: SealDecision) -> dict:
    d = decision.to_dict()
    for key in ("opened_at", "closed_at", "decision_id", "seal_id", "target_pin"):
        d.pop(key, None)
    d["epoch"].pop("read_at")
    d["stats"].pop("decide_seconds")
    return d


@pytest.mark.parametrize("value, want", [("e0", "e0"), (" e0 ", "e0"), (3, 3), ("3", 3), (0, 0)])
def test_the_pin_is_e0_or_a_publication_seq(value, want):
    w = World()
    assert SealSession(w.registry, w.scorer(), pin=value).pin == want


@pytest.mark.parametrize("value", ["e1", "E0", "-1", "0x1", "e0,e1", -1, True, "٣"])
def test_an_unknown_pin_is_refused(value):
    w = World()
    with pytest.raises(ValueError, match="pin"):
        SealSession(w.registry, w.scorer(), pin=value)


def test_a_pin_is_resolved_in_the_publication_log_not_given_an_epoch():
    w = World()
    with pytest.raises(ValueError, match="registry"):
        SealSession(w.registry.read_epoch(), w.scorer(), pin="e0")


def test_after_the_switch_every_label_is_e0s_from_the_recorded_e0_suite():
    w = World()
    w.switch()
    group = [w.member(w.e1, [True, False]),   # retracted by e1: trained as e0's pass
             w.member(w.e1, [True, True]),
             w.member(w.e1, [False]),         # e0 failed at grading; the hardened suite never ran
             w.member(w.e0, [True]),          # lagging (a staged release): graded under e0
             w.member(w.e0, [False])]
    session, d = w.seal("never", [group])
    assert d.epoch.name == "e0" and d.mode == "repair" and d.policy == "ecc"
    assert d.target_pin.pin == "e0" and d.target_pin.resolved and d.target_pin.published.name == "e1"
    assert d.target_pin.published.seq == w.registry.publications("grader")[1].seq
    assert statuses(d) == ["repaired_reuse_flip", "repaired_reuse_same", "repaired_reuse_same", "consistent",
                           "consistent"]
    assert trained(d) == [1.0, 1.0, 0.0, 1.0, 0.0]
    assert w.executions == [] and d.neutralized_groups == 0  # reuses: nothing executed


def test_before_the_switch_the_pin_changes_nothing():
    decisions = {}
    for arm in ("ecc", "never"):
        w = World()
        _, decisions[arm] = w.seal(arm, [[w.member(w.e0, [True]), w.member(w.e0, [False])]])
    assert decisions["never"].target_pin.published.name == "e0" and decisions["ecc"].target_pin is None
    assert _essence(decisions["never"]) == _essence(decisions["ecc"])


def test_without_the_pin_the_seal_adopts_e1():
    w = World()
    w.switch()
    _, d = w.seal("ecc", [[w.member(w.e1, [True, False]), w.member(w.e1, [True, True])]])
    assert d.target_pin is None and d.epoch.name == "e1" and statuses(d) == ["consistent", "consistent"]
    assert "target_pin" not in d.body()  # an unpinned decision's identity is the one it had before pins


def test_a_pin_the_log_does_not_hold_neutralizes_the_batch():
    w = World()
    w.switch()
    session = SealSession(w.registry, w.scorer(), pin=7)
    d = session.finish(session.decide([Candidate(group_id="g", members=(w.member(w.e1, [True, True]),
                                                                        w.member(w.e1, [False])))]))
    assert d.epoch is None and d.target_pin.resolved is False and d.target_pin.published.name == "e1"
    assert statuses(d) == ["no_epoch", "no_epoch"] and d.neutralized_groups == 1 and trained(d) == [0.0, 0.0]


def test_a_pin_to_a_publication_seq_seals_under_that_epoch():
    w = World()
    w.switch()
    seq = w.registry.publications("grader")[1].seq
    session = SealSession(w.registry, w.scorer(), pin=seq)
    assert session.epoch.ref.name == "e1" and session.target_pin.pin == seq
    w.publish_again(w.e0)  # a rollback published after the read changes nothing for this batch
    assert session.epoch.ref.name == "e1"


def test_the_pinned_epoch_is_read_from_the_log_never_newer_than_the_read():
    w = World()
    pointer = w.registry.read_epoch()
    w.switch()  # e1 published after the read
    late = w.registry.publications("grader")[1].seq
    assert pinned_epoch(w.registry, late, pointer=pointer) is None
    assert pinned_epoch(w.registry, "e0", pointer=None) is None
    pinned = pinned_epoch(w.registry, "e0", pointer=pointer)
    assert pinned.ref.name == "e0" and pinned.ref.read_at == pointer.ref.read_at
    assert pinned.version.version_id == w.e0.version_id


def test_every_decision_round_of_a_pinned_session_is_pinned():
    """The post-seal filter's backfill rounds decide under the session's epoch: the pinned one, in every round."""
    w = World()
    w.switch()
    session = arm_session("never", w.registry, w.scorer(), cache={})
    first = Candidate(group_id="a", members=(w.member(w.e1, [True, False]), w.member(w.e1, [True, True])))
    later = [Candidate(group_id="b", members=(w.member(w.e1, [True, True]), w.member(w.e0, [False])))]
    assembly = session.assemble([first], backfill=lambda need, considered, remaining: later[:need] if later else [],
                                clock=lambda: 0.0)
    d = session.finish(assembly)
    # under e0, group a trains [1, 1]: no variance, excluded; group b is backfilled under e0 too
    assert [g.group_id for g in d.groups] == ["b"] and [e.group.group_id for e in d.excluded] == ["a"]
    assert d.epoch.name == "e0" and d.rounds == 2 and statuses(d) == ["repaired_reuse_same", "consistent"]
    assert [g.round for g in d.groups] == [2] and w.executions == []


def test_a_member_without_a_logical_recorded_verdict_is_unrepairable_in_the_trainer_compatible_setting():
    """A member graded under e1 whose recorded e0 suite errored (and so its conjunction decided nothing). The trainer
    calls it unrepairable without executing (Scorer(recorded_undeterminable="final")): the pinned seal neutralizes its
    group as ECC does. agent-ecc's default re-executes its e0 suite instead."""
    for setting in ("final", "resolve"):
        w = World()
        w.switch()
        session = arm_session("never", w.registry, w.scorer(recorded_undeterminable=setting), cache={})
        errored, other = w.member(w.e1, [None], patch="diff fix"), w.member(w.e1, [True, True])
        d = session.finish(session.decide([Candidate(group_id="g", members=(errored, other))]))
        if setting == "final":
            assert statuses(d) == ["unrepairable", "repaired_reuse_same"] and w.executions == []
            assert d.neutralized_groups == 1 and trained(d) == [0.0, 0.0]
        else:
            assert statuses(d) == ["repaired_exec_flip", "repaired_reuse_same"]
            assert w.executions == [("diff fix", "tests")] and trained(d) == [1.0, 1.0]


def test_never_is_also_called_pinned_and_seals_under_a_given_epoch_without_a_pin():
    assert Arm("pinned") is Arm.NEVER and Arm("never") is Arm.NEVER
    w = World()
    w.switch()
    e0 = w.registry.publications("grader")[0]
    from agent_ecc.versions import Epoch

    given = Epoch(ref=e0.ref(read_at=w.now), version=w.e0)  # a replay that already knows the pinned epoch
    session = arm_session("never", given, w.scorer(), cache={})
    assert session.epoch.ref.name == "e0" and session.target_pin is None and session.pin is None


def test_a_pinned_decision_round_trips_and_its_journal_stays_monotonic(tmp_path):
    w = World()
    w.switch()
    journal = SealJournal(tmp_path / "seals.jsonl")
    for i in range(2):
        session = arm_session("never", w.registry, w.scorer(), cache={}, batch_id=f"step-{i}")
        session.finish(session.decide([Candidate(group_id=f"g{i}", members=(w.member(w.e1, [True, True]),))]),
                       journal=journal)
    decisions = journal.read()
    assert [d.epoch.name for d in decisions] == ["e0", "e0"]
    assert all(d.target_pin.published.name == "e1" for d in decisions)
    again = SealDecision.from_dict(decisions[0].to_dict())
    assert again == decisions[0] and again.decision_id == decisions[0].decision_id
    # an unpinned seal after the pinned ones reads e1: a later epoch, accepted
    _, d = w.seal("ecc", [[w.member(w.e1, [True, True])]])
    journal.append(d)
    assert [x.epoch.name for x in journal.read()] == ["e0", "e0", "e1"]


def test_a_failed_pinned_seal_records_its_pin():
    w = World()
    w.switch()
    session = arm_session("never", w.registry, w.scorer(), cache={})
    group = Candidate(group_id="g", members=(w.member(w.e1, [True, True]),))
    d = session.fail(RuntimeError("boom"), [group])
    assert d.failed and d.target_pin.pin == "e0" and d.epoch.name == "e0" and d.neutralized_groups == 1
    assert dataclasses.replace(d, target_pin=None).decision_id != d.decision_id
