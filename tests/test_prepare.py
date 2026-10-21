# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared re-execution: the seal's re-executions start when a group is admitted; the seal only verifies.

The invariants under test:

* a prepared result answers a seal's request only if its epoch identity (publication and content) is the batch
  epoch's, its request (the content address of what runs, the patch among it) is the one the seal built from the
  member's own record, and it gives the check a logical verdict;
* otherwise the request is executed at the seal, as a synchronous seal executes it (a publication between preparation
  and the seal, another patch, a failed or undeterminable job, a job not yet started);
* a job still running is waited for and never executed twice; requests to execute at the seal start at once;
* the labels, statuses, trained rewards and neutralizations are those of the synchronous seal;
* preparation never writes the run's cache and never decides a label;
* nothing is prepared under the startup epoch, or for a member a seal decided.
"""

import random
import threading
import time

import pytest
from abs_world import World, statuses, trained

from agent_ecc.prepare import Preparer
from agent_ecc.recompute.base import ExecResult
from agent_ecc.scoring import Scorer
from agent_ecc.seal import Candidate, SealSession, arm_session


def _group(w, name="u1", suffix=""):
    """A group graded under e0 during a staged release: two e0 passes (one the hardened suite rejects), one e0 fail.
    Groups with the same ``suffix`` hold byte-identical patches (identical requests)."""
    return Candidate(group_id=name, members=(w.member(w.e0, [True], patch="diff fix" + suffix),
                                             w.member(w.e0, [True], patch="diff fix fragile" + suffix),
                                             w.member(w.e0, [False], patch="diff nothing" + suffix)))


def _seal(w, candidates, *, prepared=None, scorer=None, **kwargs):
    scorer = scorer or w.scorer()
    session = arm_session("ecc", w.registry, scorer, cache=w.cache, prepared=prepared, **kwargs)
    if isinstance(candidates, Candidate):
        candidates = [candidates]
    return session, session.finish(session.decide(list(candidates)))


def _uses(prep):
    return [(e["how"], e["why"]) for e in prep.events if e["kind"] == "use"]


def _plans(prep):
    return [e["decision"] for e in prep.events if e["kind"] == "plan"]


def _essence(decision):
    """What a seal decided, its timing, identity and execution accounting aside."""
    d = decision.to_dict()
    for key in ("opened_at", "closed_at", "decision_id", "seal_id", "stats"):
        d.pop(key)
    d["epoch"].pop("read_at")
    for g in d["groups"] + [e["group"] for e in d["excluded"]]:
        for m in g["members"]:
            m.pop("score_id")  # a score's cost differs (shared vs executed); its verdict does not
    return d


class RaisingBackend:
    """A backend whose execute raises (or blocks first), for jobs that fail."""

    def __init__(self, block=None):
        self.block, self.calls = block, 0

    def execute(self, requests, store):
        self.calls += 1
        if self.block is not None:
            self.block.wait(10)
        raise RuntimeError("the recompute service is unreachable")


# ---------------------------------------------------------------------------------------------- the seal uses them


def test_a_prepared_result_answers_the_seal_and_the_labels_are_synchronous_eccs():
    ref = World()
    ref.switch()
    _, sync = _seal(ref, _group(ref))
    assert statuses(sync) == ["repaired_exec_same", "repaired_exec_flip", "repaired_short_same"]
    assert len(ref.executions) == 2 and sync.stats["executions"] == 2

    w = World()
    prep = Preparer(w.registry, w.scorer(cache=w.cache))
    group = _group(w)
    prep.admit(group)
    assert prep.plan() == 1 and _plans(prep) == [] and not prep.queue  # nothing under the startup epoch
    w.switch()
    assert prep.poll()  # e1 published: the admitted group is swept
    prep.plan()
    assert sorted(_plans(prep)) == ["job", "job", "nothing"]
    assert prep.run() == 2 and len(w.executions) == 2
    assert w.cache == {}  # preparation never writes the run's cache

    _, d = _seal(w, group, prepared=prep)
    assert len(w.executions) == 2  # nothing executed at the seal
    assert _essence(d) == _essence(sync)
    assert statuses(d) == statuses(sync) and trained(d) == trained(sync) == [1.0, 0.0, 0.0]
    assert d.stats["executions"] == 0 and d.stats["prepared"]["prepared"] == 2 and d.stats["prepared"]["sync"] == 0
    assert _uses(prep) == [("prepared", "ready"), ("prepared", "ready")]
    assert len(w.cache) == 2  # the seal recorded the results it used, as it records its own executions


def test_a_session_opened_before_a_publication_still_uses_its_epochs_results():
    w = World()
    w.switch()
    prep = Preparer(w.registry, w.scorer())
    group = _group(w)
    prep.admit(group)
    prep.plan()
    prep.run()
    session = arm_session("ecc", w.registry, w.scorer(), prepared=prep)  # g_b: e1, read now
    w.publish_again(w.e1)
    assert prep.poll()  # the target is now the republication
    d = session.finish(session.decide([group]))
    assert d.epoch.name == "e1" and len(w.executions) == 2
    assert statuses(d) == ["repaired_exec_same", "repaired_exec_flip", "repaired_short_same"]


def test_a_publication_between_preparation_and_the_seal_voids_the_prepared_results():
    w = World()
    w.switch()
    prep = Preparer(w.registry, w.scorer())
    group = _group(w)
    prep.admit(group)
    prep.plan()
    assert prep.run() == 2
    again = w.publish_again(w.e1)  # the same content, published again: another epoch identity
    prep.poll()
    _, d = _seal(w, group, prepared=prep)
    assert d.epoch.seq == again.seq and len(w.executions) == 4  # both requests executed again, at the seal
    assert set(_uses(prep)) == {("sync", "other_identity")}
    assert statuses(d) == ["repaired_exec_same", "repaired_exec_flip", "repaired_short_same"]
    assert trained(d) == [1.0, 0.0, 0.0] and d.stats["prepared"]["prepared"] == 0


def test_queued_jobs_for_the_old_epoch_are_superseded_and_never_run():
    w = World()
    w.switch()
    prep = Preparer(w.registry, w.scorer())
    group = _group(w)
    prep.admit(group)
    prep.plan()
    old = [j.id for j in prep.queue]
    assert len(old) == 2
    w.publish_again(w.e1)
    prep.poll()
    assert {prep.by_id[j].state for j in old} == {"superseded"}
    prep.plan()  # the sweep plans the group again under the new target: fresh jobs
    assert prep.run() == 2 and len(w.executions) == 2
    _, d = _seal(w, group, prepared=prep)
    assert len(w.executions) == 2 and d.stats["prepared"]["prepared"] == 2


def test_another_patch_in_the_record_is_executed_at_the_seal():
    """The member the seal is given holds another record (a later attempt replaced the patch): the prepared result is
    for the evidence the preparer was given, not the record the seal reads."""
    w = World()
    w.switch()
    prep = Preparer(w.registry, w.scorer())
    prep.admit(Candidate(group_id="u", members=(w.member(w.e0, [True], patch="diff fix", member_id="m"),)))
    prep.plan()
    prep.run()
    replaced = Candidate(group_id="u", members=(w.member(w.e0, [True], patch="diff fix fragile", member_id="m"),))
    _, d = _seal(w, replaced, prepared=prep)
    assert _uses(prep) == [("sync", "no_job")] and w.executions[-1] == ("diff fix fragile", "hardened")
    assert statuses(d) == ["repaired_exec_flip"] and trained(d) == [0.0]


@pytest.mark.parametrize("how, why", [("raises", "failed"), ("error", "error"), ("truncated", "undeterminable")])
def test_a_failed_or_unusable_job_is_executed_at_the_seal(how, why):
    w = World()
    w.switch()
    if how == "raises":
        scorer = Scorer([w.adapter], w.store, RaisingBackend())
    else:
        scorer = w.scorer()
        setattr(w, "error_on" if how == "error" else "truncate_on", {"hardened"})
    prep = Preparer(w.registry, scorer)
    group = _group(w)
    prep.admit(group)
    prep.plan()
    prep.run()
    w.error_on, w.truncate_on = set(), set()
    _, d = _seal(w, group, prepared=prep)
    assert set(_uses(prep)) == {("sync", why)}
    assert statuses(d) == ["repaired_exec_same", "repaired_exec_flip", "repaired_short_same"]
    assert d.neutralized_groups == 0 and d.stats["executions"] == 2


def test_a_queued_job_is_taken_over_by_the_seal_and_never_run_after():
    w = World()
    w.switch()
    prep = Preparer(w.registry, w.scorer())
    group = _group(w)
    prep.admit(group)
    prep.plan()
    jobs = list(prep.queue)
    _, d = _seal(w, group, prepared=prep)
    assert set(_uses(prep)) == {("sync", "taken_over")} and len(w.executions) == 2
    assert {j.state for j in jobs} == {"taken"} and prep.take_job() is None and prep.run() == 0
    assert {j.state for j in prep.by_id.values() if j.seal} == {"seal_done"}
    assert trained(d) == [1.0, 0.0, 0.0] and d.neutralized_groups == 0


# ---------------------------------------------------------------------------------------------- waiting


def test_a_running_job_is_waited_for_and_not_executed_twice():
    gate = threading.Event()
    w = World(block=gate)
    w.switch()
    prep = Preparer(w.registry, w.scorer())
    member = w.member(w.e0, [True], patch="diff fix fragile")
    prep.admit(Candidate(group_id="u", members=(member,)))
    prep.plan()
    job = prep.take_job()
    worker = threading.Thread(target=prep.run_job, args=(job,))
    worker.start()
    threading.Timer(0.3, gate.set).start()
    t0 = time.monotonic()
    _, d = _seal(w, Candidate(group_id="u", members=(member,)), prepared=prep)
    worker.join(5)
    assert time.monotonic() - t0 >= 0.2 and len(w.executions) == 1
    ((how, why),) = _uses(prep)
    assert (how, why) == ("prepared", "waited") and d.stats["prepared"]["waited_s"] > 0.1
    assert statuses(d) == ["repaired_exec_flip"] and trained(d) == [0.0]


def test_a_waited_for_job_that_fails_is_executed_when_it_ends():
    gate = threading.Event()
    w = World()
    w.switch()
    prep = Preparer(w.registry, Scorer([w.adapter], w.store, RaisingBackend(block=gate)))
    group = Candidate(group_id="u", members=(w.member(w.e0, [True], patch="diff fix"),
                                             w.member(w.e0, [True], patch="diff fix fragile")))
    prep.admit(group)
    prep.plan()
    jobs = [prep.take_job(), prep.take_job()]
    workers = [threading.Thread(target=prep.run_job, args=(j,)) for j in jobs]
    for t in workers:
        t.start()
    threading.Timer(0.3, gate.set).start()
    _, d = _seal(w, group, prepared=prep)
    for t in workers:
        t.join(5)
    assert set(_uses(prep)) == {("sync", "failed")} and len(w.executions) == 2
    assert statuses(d) == ["repaired_exec_same", "repaired_exec_flip"] and d.neutralized_groups == 0


def test_requests_to_execute_start_while_the_seal_waits():
    gate = threading.Event()
    w = World()
    w.switch()
    started = []
    inner = w.backend

    class Timed:
        def execute(self, requests, store):
            started.append(time.monotonic())
            return inner.execute(requests, store)

    prep = Preparer(w.registry, Scorer([w.adapter], w.store, RaisingBackend(block=gate)))
    group = Candidate(group_id="u", members=(w.member(w.e0, [True], patch="diff fix"),
                                             w.member(w.e0, [True], patch="diff fix fragile")))
    prep.admit(group)
    prep.plan()
    running = prep.take_job()  # the first runs (and will fail); the second stays queued and is taken over
    worker = threading.Thread(target=prep.run_job, args=(running,))
    worker.start()
    threading.Timer(0.6, gate.set).start()
    t0 = time.monotonic()
    _, d = _seal(w, group, prepared=prep, scorer=Scorer([w.adapter], w.store, Timed()))
    worker.join(5)
    assert started and started[0] - t0 < 0.3  # the taken-over request was sent at once, not after the wait
    assert sorted(_uses(prep)) == [("sync", "failed"), ("sync", "taken_over")]
    assert statuses(d) == ["repaired_exec_same", "repaired_exec_flip"]


# ---------------------------------------------------------------------------------------------- what is planned


def test_identical_requests_are_executed_once():
    w = World()
    w.switch()
    prep = Preparer(w.registry, w.scorer())
    group = Candidate(group_id="u", members=(w.member(w.e0, [True], patch="diff fix fragile"),
                                             w.member(w.e0, [True], patch="diff fix fragile"),
                                             w.member(w.e0, [True], patch="diff fix")))
    prep.admit(group)
    prep.plan()
    assert sorted(_plans(prep)) == ["attached", "job", "job"]
    assert prep.run() == 2
    _, d = _seal(w, group, prepared=prep)
    assert len(w.executions) == 2 and trained(d) == [0.0, 0.0, 1.0]


def test_nothing_is_prepared_under_the_startup_epoch_or_for_decided_members():
    w = World()
    prep = Preparer(w.registry, w.scorer())
    early, late = _group(w, "early"), _group(w, "late")
    prep.admit([early, late])
    prep.plan()
    assert _plans(prep) == [] and not prep.queue
    _, d = _seal(w, early, prepared=prep)  # under e0: consistent
    assert set(statuses(d)) == {"consistent"}
    w.switch()
    prep.poll()
    prep.plan()
    planned = {e["group"] for e in prep.events if e["kind"] == "plan"}
    assert planned == {"late"}  # the decided group is not even swept
    target = [e for e in prep.events if e["kind"] == "target"][-1]
    assert target["active"] is True and target["swept_groups"] == 1


def test_members_graded_under_the_target_need_nothing():
    w = World()
    w.switch()
    prep = Preparer(w.registry, w.scorer())
    prep.admit(Candidate(group_id="u", members=(w.member(w.e1, [True, True]), w.member(w.e1, [True, False]),
                                                w.member(w.e1, [False]))))
    prep.plan()
    assert set(_plans(prep)) == {"nothing"} and not prep.queue


def test_planning_reads_the_runs_cache_and_never_writes_it():
    w = World()
    w.switch()
    _seal(w, Candidate(group_id="first", members=(w.member(w.e0, [True], patch="diff fix"),)))
    assert len(w.cache) == 1 and len(w.executions) == 1
    before = dict(w.cache)
    prep = Preparer(w.registry, w.scorer(cache=w.cache))
    prep.admit(Candidate(group_id="u", members=(w.member(w.e0, [True], patch="diff fix"),
                                                w.member(w.e0, [True], patch="diff fix fragile"))))
    prep.plan()
    assert sorted(_plans(prep)) == ["job", "nothing"]  # the cached request is answered, the other prepared
    prep.run()
    assert w.cache == before


def test_a_conjunctions_next_suite_is_prepared_once_the_first_passes():
    """A member graded under neither suite's identity (another eval script for the original suite): the original suite
    is prepared first, and the hardened suite only once its result passes; nothing after a logical fail."""
    w = World()
    w.switch()
    # members whose recorded runs are not e1's original suite: their records hold no run of it
    fixed = w.member(w.e1, [], patch="diff fix")
    broken = w.member(w.e1, [], patch="diff nothing")
    prep = Preparer(w.registry, w.scorer())
    prep.admit(Candidate(group_id="u", members=(fixed, broken)))
    prep.plan()
    assert sorted(_plans(prep)) == ["job", "job"]
    prep.run()  # runs both original suites, then plans and runs the fixed member's hardened suite
    assert sorted(w.executions) == [("diff fix", "hardened"), ("diff fix", "tests"), ("diff nothing", "tests")]
    _, d = _seal(w, Candidate(group_id="u", members=(fixed, broken)), prepared=prep)
    assert len(w.executions) == 3 and trained(d) == [1.0, 0.0]
    assert d.stats["prepared"]["prepared"] == 3


def test_queued_jobs_of_decided_members_are_dropped():
    w = World()
    w.switch()
    prep = Preparer(w.registry, w.scorer())
    group = _group(w)
    prep.admit(group)
    prep.plan()
    prep.note_sealed([m.member_id for m in group.members], seal_id="elsewhere")
    assert prep.run() == 0 and w.executions == []
    assert sorted(e["state"] for e in prep.events if e["kind"] == "job") == ["dropped", "dropped"]


def test_members_a_seal_decides_under_another_epoch_are_not_prepared():
    w = World()
    w.switch()
    prep = Preparer(w.registry, w.scorer())
    group = _group(w)
    e0 = w.registry.publications("grader")[0]
    from agent_ecc.versions import Epoch

    prep.note_deciding([m.member_id for m in group.members], Epoch(ref=e0.ref(), version=w.e0), seal_id="s0")
    prep.admit(group)
    prep.plan()
    assert set(_plans(prep)) == {"skip_other_epoch_session"} and not prep.queue
    # a later seal's marks end the earlier seal's: the members can be prepared again
    prep.note_deciding([], None, seal_id="s1")
    prep.poll()
    prep.planned.clear()
    prep.admit(group)
    prep.plan()
    assert sorted(_plans(prep)[3:]) == ["job", "job", "nothing"]


def test_a_member_planned_while_the_seal_executes_attaches_to_that_execution():
    gate = threading.Event()
    w = World(block=gate)
    w.switch()
    prep = Preparer(w.registry, w.scorer(cache=w.cache))
    first = w.member(w.e0, [True], patch="diff fix fragile")
    out = {}

    def seal():
        out["d"] = _seal(w, Candidate(group_id="a", members=(first,)), prepared=prep)[1]

    t = threading.Thread(target=seal)
    t.start()
    deadline = time.monotonic() + 5
    while not any(j.seal and j.state == "seal" for j in prep.by_id.values()) and time.monotonic() < deadline:
        time.sleep(0.01)
    prep.admit(Candidate(group_id="b", members=(w.member(w.e0, [True], patch="diff fix fragile"),)))
    prep.plan()
    assert _plans(prep) == ["attached_seal"]
    gate.set()
    t.join(5)
    assert len(w.executions) == 1 and statuses(out["d"]) == ["repaired_exec_flip"]
    # when the seal's execution ended, the attached member was planned again: the run's cache answers it
    assert _plans(prep) == ["attached_seal", "nothing"] and not prep.queue


def test_a_member_whose_planning_raises_is_skipped_and_preparation_goes_on():
    w = World()
    w.switch()
    prep = Preparer(w.registry, w.scorer())
    bad = w.member(w.e0, [True])
    other = w.member(w.e0, [True], patch="diff fix fragile")
    wrong = bad.prior[0].__class__.from_dict({**bad.prior[0].to_dict(), "execution_id": other.execution.record_id,
                                              "score_id": None})
    broken = type(bad)(member_id="broken", execution=bad.execution, prior=(wrong,), recorded_reward=1.0)
    prep.admit(Candidate(group_id="u", members=(broken, other)))
    prep.plan()
    assert _plans(prep) == ["plan_error", "job"]


def test_a_backend_that_answers_wrongly_fails_the_job():
    w = World()
    w.switch()

    class Wrong:
        def execute(self, requests, store):
            return [ExecResult(request_id="sha256:" + "0" * 64, status="ok")]

    prep = Preparer(w.registry, Scorer([w.adapter], w.store, Wrong()))
    group = Candidate(group_id="u", members=(w.member(w.e0, [True]),))
    prep.admit(group)
    prep.plan()
    prep.run()
    assert [e["state"] for e in prep.events if e["kind"] == "job"] == ["failed"]
    _, d = _seal(w, group, prepared=prep)
    assert _uses(prep) == [("sync", "failed")] and statuses(d) == ["repaired_exec_same"]


def test_a_disabled_preparer_leaves_every_execution_to_the_seal():
    w = World()
    w.switch()
    prep = Preparer(w.registry, w.scorer())
    group = _group(w)
    prep.admit(group)
    prep.plan()
    prep.disable("operator")
    assert [e["state"] for e in prep.events if e["kind"] == "job"] == ["disabled", "disabled"]
    _, d = _seal(w, group, prepared=prep)
    assert set(_uses(prep)) == {("sync", "disabled")} and len(w.executions) == 2
    assert statuses(d) == ["repaired_exec_same", "repaired_exec_flip", "repaired_short_same"]


def test_audit_and_discard_seals_never_consult_preparation():
    for arm, want in (("keep", ["needs_reexec", "needs_reexec", "repaired_short_same"]),
                      ("discard", ["needs_reexec", "needs_reexec", "repaired_short_same"])):
        w = World()
        w.switch()
        prep = Preparer(w.registry, w.scorer())
        group = _group(w)
        prep.admit(group)
        prep.plan()
        prep.run()
        session = arm_session(arm, w.registry, w.scorer(), prepared=prep)
        d = session.finish(session.decide([group]))
        assert statuses(d) == want and _uses(prep) == [] and "prepared" not in d.stats, arm


def test_backfill_rounds_use_prepared_results_under_the_same_epoch():
    w = World()
    w.switch()
    prep = Preparer(w.registry, w.scorer())
    flat = Candidate(group_id="flat", members=(w.member(w.e0, [True], patch="diff fix fragile"),
                                               w.member(w.e0, [False], patch="diff nothing"),
                                               w.member(w.e0, [True], patch="diff fix fragile v2")))
    good = _group(w, "good")  # the same size: a backfill candidate
    prep.admit([flat, good])
    prep.plan()
    prep.run()
    executed = len(w.executions)
    session = SealSession(w.registry, w.scorer(), batch_id="step-3", prepared=prep)
    pending = [good]
    assembly = session.assemble([flat], backfill=lambda need, considered, remaining: pending[:need],
                                clock=lambda: 0.0)
    d = session.finish(assembly)
    assert [g.group_id for g in d.groups] == ["good"] and d.rounds == 2
    assert len(w.executions) == executed and d.stats["prepared"]["sync"] == 0


def test_the_threaded_preparer_prepares_a_publication_backlog_and_new_admissions():
    w = World()
    with Preparer(w.registry, w.scorer(), poll_s=0.01, threads=True) as prep:
        backlog = _group(w, "backlog")
        prep.admit(backlog)
        w.switch()
        fresh = _group(w, "fresh", suffix=" v2")
        prep.admit(fresh)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            stats = prep.stats()
            if stats["jobs"].get("done", 0) >= 4:
                break
            time.sleep(0.01)
        assert prep.stats()["jobs"] == {"done": 4}
        _, d = _seal(w, [backlog, fresh], prepared=prep)
    assert len(w.executions) == 4 and d.stats["prepared"] == {**d.stats["prepared"], "prepared": 4, "sync": 0}
    assert trained(d) == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_the_backends_capacity_is_the_seals_first():
    w = World()
    w.switch()
    prep = Preparer(w.registry, w.scorer(), capacity=2, slots=2)
    prep.admit(_group(w))
    prep.plan()
    with prep.lock:
        prep.seal_inflight = 2  # the seal holds every sandbox
    assert prep.take_job() is None
    with prep.lock:
        prep.seal_inflight = 1
    assert prep.take_job() is not None and prep.take_job() is None  # one left: running + the seal's = capacity


@pytest.mark.parametrize("seed", range(30))
def test_preparation_never_changes_a_label(seed):
    """Random groups, random preparation (members admitted or not, jobs run, left queued, or failing), random
    publications in between: the prepared seal decides exactly as a synchronous seal of the same world."""
    rng = random.Random(seed)
    patches = ["diff fix", "diff fix fragile", "diff nothing", "", "diff fix fragile v2", "diff fix v3"]
    plan = []
    for gi in range(rng.randint(1, 4)):
        members = []
        for _ in range(rng.randint(2, 4)):
            under = rng.choice(["e0", "e0", "e1"])
            if under == "e0":
                passes = [rng.random() < 0.6]
            else:
                passes = rng.choice([[True, True], [True, False], [False], [None, False]])
            members.append((under, passes, rng.choice(patches)))
        plan.append((f"g{gi}", members, rng.random() < 0.7))
    republish = rng.random() < 0.3
    run_jobs = rng.choice(["all", "some", "none"])
    error_prep = rng.random() < 0.2

    def build(w):
        out = []
        for gid, members, _ in plan:
            out.append(Candidate(group_id=gid, members=tuple(
                w.member(w.e0 if under == "e0" else w.e1, list(passes), patch=patch, member_id=f"{gid}m{i}")
                for i, (under, passes, patch) in enumerate(members))))
        return out

    ref = World()
    ref.switch()
    if republish:
        ref.publish_again(ref.e1)
    _, sync = _seal(ref, build(ref))

    w = World()
    w.switch()
    prep = Preparer(w.registry, w.scorer(cache=w.cache))
    candidates = build(w)
    prep.admit([c for c, (_, _, admitted) in zip(candidates, plan) if admitted])
    prep.plan()
    if error_prep:
        w.error_on = {"hardened"}
    if run_jobs == "all":
        prep.run()
    elif run_jobs == "some":
        prep.run(limit=1)
    w.error_on = set()
    if republish:
        w.publish_again(w.e1)
        prep.poll()
        prep.plan()
        if run_jobs == "all":
            prep.run()
    _, d = _seal(w, candidates, prepared=prep)
    assert _essence(d) == _essence(sync)
    assert w.cache.keys() == ref.cache.keys()  # the seal recorded what it used, as the synchronous seal did


def test_a_prepared_seal_replays_from_recordings_without_a_preparer(tmp_path):
    """The seal's decisions depend only on the raw results of the requests it asked for: with the preparer's and the
    seal's executions recorded, the seal replays from the recordings alone, under the epoch it read, to the same
    decisions."""
    import dataclasses

    from agent_ecc.recompute.base import RecordedBackend, RecordingBackend
    from agent_ecc.versions import Epoch

    w = World()
    w.switch()
    results = tmp_path / "results.jsonl"
    recording = RecordingBackend(w.backend, results)
    prep = Preparer(w.registry, Scorer([w.adapter], w.store, recording, cache=w.cache))
    group, other = _group(w), _group(w, "other", suffix=" v2")
    prep.admit(group)  # prepared; `other` is not: the seal executes it itself
    prep.plan()
    prep.run()
    session = arm_session("ecc", w.registry, Scorer([w.adapter], w.store, recording), cache=w.cache, prepared=prep,
                          seal_id="s", batch_id="step-1", clock=lambda: 5.0)
    live = session.finish(session.decide([group, other]))
    assert live.stats["prepared"]["prepared"] == 2 and live.stats["prepared"]["sync"] == 2

    read_at = live.epoch.read_at
    ref = w.registry.epoch_at(read_at)
    epoch = Epoch(ref=dataclasses.replace(ref, read_at=read_at), version=w.registry.grader(ref.version_id))
    recorded = RecordedBackend.from_jsonl(results)
    again_session = arm_session("ecc", epoch, Scorer([w.adapter], w.store, recorded), cache={}, seal_id="s",
                                batch_id="step-1", clock=lambda: 5.0)
    again = again_session.finish(again_session.decide([group, other]))
    assert _essence(again) == _essence(live) and recorded.missing == []


def test_an_error_in_the_preparers_view_never_fails_a_seal():
    """A view that cannot classify the seal's requests disables preparation, and the seal executes what it needs."""
    w = World()
    w.switch()
    prep = Preparer(w.registry, w.scorer())
    group = _group(w)
    prep.admit(group)
    prep.plan()
    prep.run()
    prep.jobs = None  # a broken preparer: every lookup raises
    _, d = _seal(w, group, prepared=prep)
    assert prep.disabled and prep.disabled.startswith("view.claim")
    assert statuses(d) == ["repaired_exec_same", "repaired_exec_flip", "repaired_short_same"]
    assert d.neutralized_groups == 0 and d.stats["executions"] == 2
