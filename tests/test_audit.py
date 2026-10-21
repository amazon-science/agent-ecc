# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audits from a run's own records: every seal read the epoch in force (I4) and epochs never decreased in commit
order (I3), pinned seals told apart; and identical executions that got different verdicts (a check whose outcome
follows an unpinned seed)."""

import dataclasses

from abs_world import World

from agent_ecc.audit import audit_epochs, verdict_conflicts
from agent_ecc.records import EpochRef
from agent_ecc.seal import Candidate, SealJournal, arm_session


def _seal(w, arm, *, batch_id, member=None):
    session = arm_session(arm, w.registry, w.scorer(), cache={}, batch_id=batch_id,
                          clock=lambda: w.now)
    member = member or w.member(w.e1, [True, True])
    return session.finish(session.decide([Candidate(group_id=batch_id, members=(member,))]))


def test_a_run_whose_seals_read_the_epoch_in_force_passes(tmp_path):
    w = World()
    journal = SealJournal(tmp_path / "seals.jsonl")
    journal.append(_seal(w, "ecc", batch_id="step-1", member=w.member(w.e0, [True])))
    w.switch()
    journal.append(_seal(w, "ecc", batch_id="step-2"))
    journal.append(_seal(w, "ecc", batch_id="step-3"))
    audit = audit_epochs(journal.read(), w.registry)
    assert audit.ok and audit.unexpected == () and audit.decreasing == () and audit.pinned == ()
    assert [s["status"] for s in audit.seals] == ["ok", "ok", "ok"]
    assert [s["epoch"] for s in audit.seals] == [s["in_force"] for s in audit.seals]


def test_pinned_seals_are_reported_as_pinned_not_as_violations():
    w = World()
    w.switch()
    decisions = [_seal(w, "never", batch_id=f"step-{i}") for i in range(2)]
    audit = audit_epochs(decisions, w.registry)
    assert audit.ok and audit.pinned == tuple(d.seal_id for d in decisions)
    assert [s["status"] for s in audit.seals] == ["pinned", "pinned"]
    assert all(s["epoch"] < s["in_force"] == s["pointer"] for s in audit.seals)


def test_a_seal_on_an_older_epoch_than_in_force_fails_i4():
    w = World()
    w.switch()
    good = _seal(w, "ecc", batch_id="step-1")
    e0 = w.registry.publications("grader")[0]
    stale = dataclasses.replace(good, seal_id="stale", epoch=e0.ref(read_at=good.epoch.read_at))
    audit = audit_epochs([stale], w.registry)
    assert not audit.ok and audit.unexpected == ("stale",)
    assert audit.seals[0]["problem"] == "older_than_in_force"


def test_a_read_that_overlapped_a_publication_is_a_race_not_a_violation():
    w = World()
    first = _seal(w, "ecc", batch_id="step-1", member=w.member(w.e0, [True]))
    w.now += 0.5
    w.registry.publish_grader(w.e1, source="switch")  # published 0.5 s before the read time below
    overlapping = dataclasses.replace(first, seal_id="overlap",
                                      epoch=dataclasses.replace(first.epoch, read_at=w.now + 0.25))
    audit = audit_epochs([overlapping], w.registry, tolerance_s=1.0)
    assert audit.ok and audit.races == ("overlap",) and audit.seals[0]["status"] == "race"
    assert not audit_epochs([overlapping], w.registry, tolerance_s=0.1).ok


def test_a_pin_that_names_another_epoch_than_the_batchs_fails():
    w = World()
    w.switch()
    pinned = _seal(w, "never", batch_id="step-1")
    e1 = w.registry.publications("grader")[1]
    forged = dataclasses.replace(pinned, seal_id="forged", epoch=e1.ref(read_at=pinned.epoch.read_at))
    audit = audit_epochs([forged], w.registry)
    assert audit.unexpected == ("forged",) and audit.seals[0]["problem"] == "not_the_pinned_epoch"


def test_an_epoch_that_decreases_in_commit_order_fails_i3():
    w = World()
    w.switch()
    later = _seal(w, "ecc", batch_id="step-2")
    w.now += 50
    w.publish_again(w.e0)  # a rollback: a newer publication of the old content
    rollback = _seal(w, "ecc", batch_id="step-3", member=w.member(w.e0, [True]))
    assert rollback.epoch.seq > later.epoch.seq
    audit = audit_epochs([rollback, later], w.registry)  # committed out of order
    assert not audit.ok and audit.decreasing == (later.seal_id,)


def test_an_epoch_the_log_never_published_and_a_missing_epoch_fail():
    w = World()
    w.switch()
    d = _seal(w, "ecc", batch_id="step-1")
    unknown = dataclasses.replace(d, seal_id="unknown", epoch=EpochRef(seq=d.epoch.seq, version_id=w.e0.version_id,
                                                                       read_at=d.epoch.read_at))
    missing = dataclasses.replace(d, seal_id="missing", epoch=None, opened_at=d.epoch.read_at)
    audit = audit_epochs([unknown, missing], w.registry.publications())
    assert audit.unexpected == ("unknown", "missing")
    assert [s["problem"] for s in audit.seals] == ["not_a_grader_publication", "no_epoch"]


# ---------------------------------------------------------------------------------------------- verdicts


def test_identical_executions_with_both_verdicts_are_reported():
    """Two trajectories submitted byte-identical patches; their recorded runs of the same suite disagree (a test that
    follows an unpinned hash seed). A third, graded under e1, recorded the hardened suite failing where a seal's
    re-execution of the same request passed it."""
    w = World()
    a = w.member(w.e0, [True], patch="diff fix")
    b = w.member(w.e0, [False], patch="diff fix")
    c = w.member(w.e1, [True, False], patch="diff fix")
    clean = w.member(w.e0, [True], patch="diff fix other")
    empty = w.member(w.e0, [False], patch="")
    w.switch()
    session = arm_session("ecc", w.registry, w.scorer(), cache={})
    session.finish(session.decide([Candidate(group_id="g", members=(a,))]))  # executes a's hardened suite: pass
    records = [m.execution for m in (a, b, c, clean, empty)]
    report = verdict_conflicts(records, session.scores, [w.e0, w.e1], [w.adapter], w.store)
    assert not report.deterministic and len(report.conflicts) == 2
    by_check = {x["check"]: x for x in report.conflicts}
    assert set(by_check) == {"tests", "hardened"}
    assert set(by_check["tests"]["records"]) == {a.execution.record_id, b.execution.record_id,
                                                 c.execution.record_id}
    assert set(by_check["hardened"]["records"]) == {a.execution.record_id, c.execution.record_id}
    assert all(x["pass_runs"] and x["fail_runs"] for x in report.conflicts)
    assert report.by_task["pkg__pkg-1"]["conflicting"] == 2 and report.repeated == 2
    assert report.to_dict()["conflicting"] == 2


def test_a_deterministic_run_has_no_conflicts():
    w = World()
    members = [w.member(w.e0, [True], patch="diff fix"), w.member(w.e0, [True], patch="diff fix"),
               w.member(w.e0, [False], patch="diff nothing")]
    report = verdict_conflicts([m.execution for m in members], [], {w.e0.version_id: w.e0}, {"swebench": w.adapter},
                               w.store)
    # byte-identical runs of one request are one observation, not a repetition
    assert report.deterministic and report.groups == 2 and report.repeated == 0
