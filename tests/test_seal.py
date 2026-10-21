# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import pytest

from agent_ecc.records import Outcome, SealDecision
from agent_ecc.scoring import ScoreInput, Scorer, grade_recorded
from agent_ecc.seal import (
    Candidate,
    Member,
    Mode,
    MonotonicityError,
    SealFailed,
    SealJournal,
    SealSession,
    apply_rewards,
    has_variance,
    seal,
)
from agent_ecc.store import MemoryStore
from agent_ecc.versions import GraderVersion, MemoryRegistry

from toy import ToyAdapter, ToyBackend, check, conj


class Env:
    def __init__(self):
        self.store = MemoryStore()
        self.adapter = ToyAdapter()
        self.backend = ToyBackend()
        self.scorer = Scorer([self.adapter], self.store, self.backend)
        self.registry = MemoryRegistry()
        self.c1 = check("t", "plain", "ab", self.store)  # the grader at production time
        self.g1 = GraderVersion(name="g1", entries={"t": self.c1})
        self.registry.publish_grader(self.g1, source="startup")
        self.n = 0

    def publish(self, name, spec):
        version = GraderVersion(name=name, entries={"t": spec})
        self.registry.publish_grader(version, source="switch", require_coverage=False)
        return version

    def member(self, answer, *, keep_answer=True, error=False):
        self.n += 1
        run = self.adapter.run(self.store, self.c1, answer, error=error)
        record = self.adapter.record(self.store, task="t", trajectory=f"traj-{self.n}",
                                     answer=answer if keep_answer else None, runs=[run])
        recorded = grade_recorded(self.adapter, record, self.c1, self.store, grader_version=self.g1.version_id)
        reward = None if recorded.passed is None else (1.0 if recorded.passed else 0.0)
        return Member(member_id=f"m{self.n}", execution=record, prior=recorded, recorded_reward=reward)

    def group(self, gid, answers, **kw):
        return Candidate(group_id=gid, members=tuple(self.member(a, **kw) for a in answers))


@pytest.fixture
def env():
    return Env()


def statuses(group):
    return [m.status for m in group.members]


def test_repair_decides_every_member_under_the_one_epoch(env):
    env.publish("g2", check("t", "plain", "ba", env.store))  # a label fix
    session = SealSession(env.registry, env.scorer, batch_id="step-3")
    (group,) = session.decide([env.group("u0", ["ab", "ba", "ba", "zz"])])
    assert statuses(group) == ["repaired_flip", "repaired_flip", "repaired_flip", "repaired_same"]
    assert group.rewards() == [0.0, 1.0, 1.0, 0.0] and not group.neutralized
    decision = session.finish([group])
    assert decision.epoch.name == "g2" and decision.batch_id == "step-3" and decision.failed is None
    assert decision.stats["statuses"] == {"repaired_flip": 3, "repaired_same": 1}
    assert {s.grader_version for s in session.scores} == {decision.epoch.version_id}
    assert SealDecision.from_dict(json.loads(json.dumps(decision.to_dict()))) == decision


def test_an_undecidable_member_neutralizes_its_group(env):
    env.publish("g2", check("t", "rev", "ba", env.store))  # needs re-execution
    session = SealSession(env.registry, env.scorer)
    good = env.group("u0", ["ab", "ba"])
    bad = Candidate(group_id="u1", members=(env.member("ab"), env.member("ba", keep_answer=False)))
    g_good, g_bad = session.decide([good, bad])
    assert not g_good.neutralized and g_good.rewards() == [1.0, 0.0]
    assert g_bad.neutralized and g_bad.neutralize_reason == "undeterminable_member"
    assert g_bad.rewards() == [0.0, 0.0] and statuses(g_bad) == ["repaired_exec_same", "unrepairable"]
    assert g_bad.members[0].target_reward == 1.0  # its own label was decidable; the group trains no gradient


def test_the_epoch_is_read_once_even_while_backfilling(env):
    env.publish("g2", check("t", "plain", "ba", env.store))
    session = SealSession(env.registry, env.scorer)
    late = env.group("u-late", ["ba", "ab"])

    def backfill(need, considered, remaining):
        env.publish("g3", check("t", "plain", "zz", env.store))  # published while the seal is open
        return [late]

    assembly = session.assemble([env.group("u0", ["zz", "zz"])], backfill=backfill)
    assert env.registry.read_epoch().version.name == "g3"
    (pair,) = assembly.batch
    assert pair[0].group_id == "u-late" and pair[1].role == "backfill" and pair[1].round == 2
    assert pair[1].rewards() == [1.0, 0.0]  # decided under g2, not g3 (under g3 both would be 0)
    decision = session.finish(assembly)
    assert decision.epoch.name == "g2" and decision.rounds == 2
    assert [e.group.group_id for e in decision.excluded] == ["u0"]


def test_filtering_happens_after_the_decision(env):
    env.publish("g2", check("t", "plain", "ba", env.store))
    hidden = env.group("u-hidden", ["ba", "zz", "ba", "zz"])  # all 0 recorded, mixed under g2
    assert [m.recorded_reward for m in hidden.members] == [0.0, 0.0, 0.0, 0.0]
    assert not has_variance([m.recorded_reward for m in hidden.members])
    session = SealSession(env.registry, env.scorer)
    assembly = session.assemble([hidden])
    assert [c.group_id for c, _ in assembly.batch] == ["u-hidden"] and assembly.excluded == ()
    assert assembly.batch[0][1].rewards() == [1.0, 0.0, 1.0, 0.0]


def test_fillers_are_off_by_default_and_neutral_when_asked(env):
    env.publish("g2", check("t", "plain", "ba", env.store))
    session = SealSession(env.registry, env.scorer)
    assembly = session.assemble([env.group("u0", ["ba", "zz"]), env.group("dead", ["zz", "zz"])])
    assert [c.group_id for c, _ in assembly.batch] == ["u0"] and assembly.shortfall == 1
    decision = session.finish(assembly)
    assert decision.shortfall == 1 and decision.target_groups == 2 and len(decision.excluded) == 1
    assert decision.excluded[0].reason == "zero_variance_after_seal"

    session = SealSession(env.registry, env.scorer)
    assembly = session.assemble([env.group("u1", ["ba", "zz"]), env.group("dead2", ["zz", "zz"])],
                                fillers="neutral")
    roles = [(c.group_id, g.role, g.neutralized) for c, g in assembly.batch]
    assert roles == [("u1", "selected", False), ("dead2", "filler", True)] and assembly.shortfall == 0
    assert session.finish(assembly).stats["filler_groups"] == 1


def test_an_ecc_neutralized_group_is_excluded_and_backfilled(env):
    env.publish("g2", check("t", "rev", "ba", env.store))
    session = SealSession(env.registry, env.scorer)
    broken = Candidate(group_id="broken", members=(env.member("ab"), env.member("ab", keep_answer=False)))
    spare = env.group("spare", ["ab", "ba"])
    assembly = session.assemble([broken], backfill=lambda need, considered, remaining: [spare])
    assert [c.group_id for c, _ in assembly.batch] == ["spare"]
    (_, excluded), = assembly.excluded
    assert excluded.group.neutralized and assembly.info["excluded_neutralized_n"] == 1


def test_backfill_bookkeeping(env):
    session = SealSession(env.registry, env.scorer)
    first = env.group("u0", ["zz", "zz"])
    offers = [[first, env.group("dup", ["ab", "zz"]), env.group("dup", ["ab", "zz"]), env.group("extra", ["ab", "zz"])],
              [env.group("wrong-size", ["ab", "zz", "ab"])], []]
    asked = []

    def backfill(need, considered, remaining):
        asked.append((need, set(considered), remaining))
        return offers[len(asked) - 1]

    assembly = session.assemble([first], backfill=backfill, deadline_s=30.0)
    assert [c.group_id for c, _ in assembly.batch] == ["dup"]
    assert sorted(c.group_id for c in assembly.unused) == ["extra"]
    assert asked[0][0] == 1 and asked[0][1] == {"u0"} and asked[0][2] is not None
    assert len(asked) == 1  # the batch was full after the first round


def test_backfill_deadline_and_wrong_sizes(env):
    clock = iter([0.0, 0.0, 10.0, 25.0, 25.0])  # backfill begins; round 1; round 2; the deadline has passed
    session = SealSession(env.registry, env.scorer)
    calls = []

    def backfill(need, considered, remaining):
        calls.append(remaining)
        return [env.group(f"odd{len(calls)}", ["ab", "zz", "ab"])]

    assembly = session.assemble([env.group("u0", ["zz", "zz"])], backfill=backfill, deadline_s=20.0,
                                clock=lambda: next(clock))
    assert len(calls) == 2 and calls[0] == 20.0 and calls[1] == 10.0
    assert len(assembly.unused) == 2 and assembly.batch == () and assembly.shortfall == 1


def test_unequal_initial_sizes_skip_filtering(env):
    session = SealSession(env.registry, env.scorer)
    assembly = session.assemble([env.group("a", ["zz", "zz"]), env.group("b", ["ab", "zz", "zz"])])
    assert len(assembly.batch) == 2 and assembly.info["skipped"] == "unequal_group_sizes"


def test_failures_of_the_first_decision_and_of_a_backfill_round(env):
    session = SealSession(env.registry, env.scorer)
    broken_scorer = Scorer([env.adapter], env.store, env.backend)
    broken_scorer.score = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("scorer down"))  # type: ignore
    failing = SealSession(env.registry, broken_scorer)
    with pytest.raises(SealFailed):
        failing.assemble([env.group("u0", ["ab", "zz"])])

    class Flaky(Scorer):
        calls = 0

        def score(self, items, version):
            Flaky.calls += 1
            if Flaky.calls > 1:
                raise RuntimeError("backfill round failed")
            return super().score(items, version)

    flaky = Flaky([env.adapter], env.store, env.backend)
    session = SealSession(env.registry, flaky)
    spare = env.group("spare", ["ab", "zz"])
    assembly = session.assemble([env.group("u0", ["zz", "zz"])], backfill=lambda n, c, r: [spare])
    assert assembly.batch == () and assembly.unused == (spare,) and "backfill round failed" in assembly.info["error"]


def test_audit_trains_recorded_labels_and_never_executes(env):
    env.publish("g2", conj("all", check("t", "plain", "ba", env.store), check("t", "rev", "ab", env.store)))
    session = SealSession(env.registry, env.scorer, mode="audit")
    (group,) = session.decide([env.group("u0", ["ab", "ba", "zz"])])
    # "ab" and "zz" fail the relabelled base suite (re-parsed; no execution: a short circuit), "ba" passes it and
    # would need the second suite executed, which audit does not do
    assert statuses(group) == ["repaired_short_flip", "needs_reexec", "repaired_short_same"]
    assert [m.short_circuit for m in group.members] == [True, False, True]
    assert group.rewards() == [1.0, 0.0, 0.0]  # recorded
    assert env.backend.executed == [] and not group.neutralized


def test_a_member_without_a_record_trains_its_recorded_reward_except_in_repair(env):
    """A member the seal has no execution record for (its producer's record was never found) has no label under the
    epoch. Audit trains the recorded labels, its own included, so the post-seal filter keeps a group whose recorded
    rewards vary; repair and discard neutralize the group."""
    env.publish("g2", check("t", "plain", "ba", env.store))

    def group(gid):
        lost = Member(member_id=f"{gid}-lost", recorded_reward=1.0)  # no record: none was found for it
        return Candidate(group_id=gid, members=(lost, env.member("zz")))

    audit = SealSession(env.registry, env.scorer, mode="audit")
    assembly = audit.assemble([group("u0")], fillers="neutral")
    ((cand, kept),) = assembly.batch
    assert kept.role == "selected" and not kept.neutralized and kept.rewards() == [1.0, 0.0]
    assert statuses(kept) == ["no_record", "repaired_same"] and kept.members[0].outcome is Outcome.UNDETERMINABLE
    decision = audit.finish(assembly)
    assert decision.rewards() == {"u0-lost": 1.0, f"m{env.n}": 0.0} and decision.excluded == ()
    for mode, reason in (("repair", "undeterminable_member"), ("discard", "not_consistent")):
        session = SealSession(env.registry, env.scorer, mode=mode)
        assembly = session.assemble([group(f"u-{mode}")], fillers="neutral")
        ((_, filler),) = assembly.batch
        assert filler.role == "filler" and filler.neutralized and filler.neutralize_reason == reason, mode
        assert filler.rewards() == [0.0, 0.0], mode


def test_discard_neutralizes_groups_not_graded_under_the_epoch(env):
    session = SealSession(env.registry, env.scorer, mode=Mode.DISCARD)
    (same,) = session.decide([env.group("u0", ["ab", "zz"])])
    assert not same.neutralized and same.rewards() == [1.0, 0.0]
    env.publish("g2", check("t", "plain", "ba", env.store))
    session = SealSession(env.registry, env.scorer, mode="discard")
    (stale,) = session.decide([env.group("u1", ["ab", "zz"])])
    assert stale.neutralized and stale.neutralize_reason == "not_consistent" and stale.rewards() == [0.0, 0.0]
    assert env.adapter.parsed == 4  # only the producer's own grading parsed: discard recomputes nothing


def test_off_mode_and_no_epoch(env):
    session = SealSession(env.registry, None, mode="off")
    (group,) = session.decide([env.group("u0", ["ab", "zz"])])
    assert statuses(group) == ["off", "off"] and group.rewards() == [1.0, 0.0]
    nothing = MemoryRegistry()
    repair = SealSession(nothing, env.scorer)
    (g,) = repair.decide([env.group("u1", ["ab", "zz"])])
    assert g.neutralized and statuses(g) == ["no_epoch", "no_epoch"] and repair.finish([g]).epoch is None
    (d,) = SealSession(nothing, env.scorer, mode="discard").decide([env.group("u2", ["ab", "zz"])])
    assert d.neutralized and d.neutralize_reason == "no_epoch"
    (lax,) = SealSession(nothing, env.scorer, mode="discard", strict=False).decide([env.group("u3", ["ab", "zz"])])
    assert not lax.neutralized and lax.rewards() == [1.0, 0.0]
    with pytest.raises(ValueError):
        SealSession(env.registry, None, mode="repair")


def test_fail_neutralizes_in_strict_mode_only(env):
    candidates = [env.group("u0", ["ab", "zz"])]
    strict = SealSession(env.registry, env.scorer).fail(RuntimeError("log unavailable"), candidates)
    assert strict.failed.startswith("RuntimeError") and strict.groups[0].neutralized
    assert list(strict.rewards().values()) == [0.0, 0.0]
    lax = SealSession(env.registry, env.scorer, strict=False).fail("boom", candidates)
    assert not lax.groups[0].neutralized and list(lax.rewards().values()) == [1.0, 0.0]


def test_finish_accepts_only_this_seals_groups(env):
    s1, s2 = SealSession(env.registry, env.scorer), SealSession(env.registry, env.scorer)
    (g1,) = s1.decide([env.group("u0", ["ab", "zz"])])
    (g2,) = s2.decide([env.group("u1", ["ab", "zz"])])
    with pytest.raises(ValueError):
        s1.finish([g1, g2])
    with pytest.raises(ValueError):
        s1.finish([g1, g1])
    with pytest.raises(ValueError):
        s1.decide([env.group("u0", ["ab", "zz"])])  # decided twice in one seal
    s1.finish([g1])
    with pytest.raises(RuntimeError):
        s1.decide([env.group("u9", ["ab"])])


def test_journal_enforces_commit_order(env, tmp_path):
    journal = SealJournal(tmp_path / "seals.jsonl")
    early = SealSession(env.registry, env.scorer)  # reads g1
    env.publish("g2", check("t", "plain", "ba", env.store))
    late = SealSession(env.registry, env.scorer)  # reads g2
    (gl,) = late.decide([env.group("u1", ["ba", "zz"])])
    late.finish([gl], journal=journal)
    (ge,) = early.decide([env.group("u0", ["ab", "zz"])])
    with pytest.raises(MonotonicityError):
        journal.append(early.finish([ge]))
    assert [d.epoch.name for d in journal.read()] == ["g2"]
    reopened = SealJournal(tmp_path / "seals.jsonl")
    assert reopened._last_seq == 1


def test_an_unloggable_seal_is_neutralized_when_strict(env, tmp_path):
    journal = SealJournal(tmp_path / "logs" / "seals.jsonl")
    (tmp_path / "logs").write_text("a file where the log directory should be")  # every append now fails
    session = SealSession(env.registry, env.scorer)
    (g,) = session.decide([env.group("u0", ["ab", "zz"])])
    decision = session.finish([g], journal=journal)
    assert decision.failed.startswith("journal") and decision.groups[0].neutralized
    lax = SealSession(env.registry, env.scorer, strict=False)
    (g,) = lax.decide([env.group("u1", ["ab", "zz"])])
    decision = lax.finish([g], journal=journal)
    assert decision.failed is None and "log_error" in decision.stats and decision.rewards() == {"m3": 1.0, "m4": 0.0}


def test_apply_rewards_is_all_or_nothing(env):
    env.publish("g2", check("t", "plain", "ba", env.store))
    decision, scores = seal(env.registry, env.scorer, [env.group("u0", ["ab", "ba"])])
    assert decision.rewards() == {"m1": 0.0, "m2": 1.0} and len(scores) == 2
    samples = [{"id": "m1", "reward": 1.0}, {"id": "m2", "reward": 0.0}]

    def set_reward(s, r):
        s["reward"] = r

    assert apply_rewards(decision, samples, member_id=lambda s: s["id"], set_reward=set_reward) == 2
    assert [s["reward"] for s in samples] == [0.0, 1.0]
    partial = [{"id": "m1", "reward": 5.0}]
    with pytest.raises(ValueError):
        apply_rewards(decision, partial, member_id=lambda s: s["id"], set_reward=set_reward)
    assert partial[0]["reward"] == 5.0
    with pytest.raises(ValueError):
        apply_rewards(decision, samples + [{"id": "m1"}], member_id=lambda s: s["id"], set_reward=set_reward)
    with pytest.raises(ValueError):
        apply_rewards(decision, [{"id": "zz"}], member_id=lambda s: s["id"], set_reward=set_reward)


def test_seal_helper_returns_a_failed_decision_on_error(env):
    broken = Scorer([env.adapter], env.store, env.backend)
    broken.score = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))  # type: ignore
    decision, scores = seal(env.registry, broken, [env.group("u0", ["ab", "zz"])])
    assert decision.failed and decision.groups[0].neutralized and scores == []


def test_reused_members_keep_their_recorded_reward(env):
    session = SealSession(env.registry, env.scorer, reward=lambda s: 0.25 if s.passed else -1.0)
    member = env.member("ab")
    shaped = Member(member_id=member.member_id, execution=member.execution, prior=member.prior, recorded_reward=0.7)
    (g,) = session.decide([Candidate(group_id="u", members=(shaped, env.member("zz")))])
    assert g.rewards() == [0.7, 0.0] and [m.outcome for m in g.members] == [Outcome.PASS, Outcome.FAIL]


# ---------------------------------------------------------------------------------------- consistency and provenance


def test_a_reused_verdict_from_a_later_score_trains_its_own_reward(env):
    """Recorded FAIL under g1; an evaluation under g2 re-executed it to PASS. A seal under g2 given both scores reuses
    the g2 verdict: it is not the recorded verdict, so the member trains the reward of its target verdict."""
    member = env.member("ba")
    assert member.recorded_reward == 0.0
    g2 = env.publish("g2", check("t", "rev", "ab", env.store))  # a runner change: "ba" reversed passes
    (evaluated,) = env.scorer.score([ScoreInput(member.execution, member.prior)], g2)
    assert evaluated.action == "reexec" and evaluated.score.passed
    later = Member(member_id=member.member_id, execution=member.execution, prior=member.prior + (evaluated.score,),
                   recorded_reward=member.recorded_reward)
    (group,) = SealSession(env.registry, env.scorer).decide([Candidate(group_id="u0", members=(later,))])
    (m,) = group.members
    assert m.action == "reuse" and m.outcome is Outcome.PASS and not m.consistent
    assert m.trained_reward == 1.0 and m.status == "repaired_reuse_flip"


def test_a_conjunction_reused_from_two_scores_trains_the_conjunction_label(env):
    """e0 recorded PASS; an evaluation under e1 = e0 AND hardened executed the hardened suite: FAIL. Sealed under e1
    with both scores, every part is reused, but the label is the evaluation's, not the recorded one."""
    member = env.member("ab")
    e1 = env.publish("e1", conj("all", env.c1, check("t", "rev", "ab", env.store)))
    (evaluated,) = env.scorer.score([ScoreInput(member.execution, member.prior)], e1)
    assert evaluated.score.outcome is Outcome.FAIL
    later = Member(member_id=member.member_id, execution=member.execution, prior=member.prior + (evaluated.score,),
                   recorded_reward=member.recorded_reward)
    (group,) = SealSession(env.registry, env.scorer).decide([Candidate(group_id="u0", members=(later,))])
    (m,) = group.members
    assert m.action == "reuse" and m.outcome is Outcome.FAIL and m.trained_reward == 0.0
    assert m.status == "repaired_reuse_flip"


def test_a_rollback_to_a_sub_spec_trains_the_reused_parts_label(env):
    """Graded under e1 = A AND B (A passes, B fails: recorded 0). A rollback publishes A alone: A's recorded part is
    reused, the member passes, and it trains 1.0 (the recorded conjunction reward does not stand)."""
    a, b = env.c1, check("t", "rev", "ab", env.store)
    e1 = GraderVersion(name="e1", entries={"t": conj("all", a, b)})
    record = env.adapter.record(env.store, task="t", trajectory="rb", answer="ab",
                                runs=[env.adapter.run(env.store, a, "ab"), env.adapter.run(env.store, b, "ab")])
    recorded = grade_recorded(env.adapter, record, e1.entries["t"]["*"], env.store, grader_version=e1.version_id)
    assert recorded.outcome is Outcome.FAIL and len(recorded.parts) == 2  # every suite that ran is graded
    env.registry.publish_grader(e1)
    env.publish("e0-again", a)
    member = Member(member_id="m-rb", execution=record, prior=recorded, recorded_reward=0.0)
    (group,) = SealSession(env.registry, env.scorer).decide([Candidate(group_id="u0", members=(member,))])
    (m,) = group.members
    assert m.outcome is Outcome.PASS and m.trained_reward == 1.0 and m.status == "repaired_reuse_flip"


def test_a_consistent_member_keeps_its_recorded_reward_and_others_do_not(env):
    shaped = env.member("ab")
    shaped = Member(member_id=shaped.member_id, execution=shaped.execution, prior=shaped.prior, recorded_reward=0.7)
    (g,) = SealSession(env.registry, env.scorer).decide([Candidate(group_id="u", members=(shaped,))])
    assert g.members[0].consistent and g.members[0].status == "consistent" and g.rewards() == [0.7]
    env.publish("g2", check("t", "dup", "abab", env.store))  # a runner change: "ab" re-executed, still passing
    (g,) = SealSession(env.registry, env.scorer).decide([Candidate(group_id="v", members=(shaped,))])
    assert not g.members[0].consistent and g.rewards() == [1.0]  # re-executed: the target verdict's reward


def test_discard_keeps_members_graded_under_exactly_the_batch_conjunction(env):
    """Validate-and-discard keeps a group whose members were graded under the batch epoch's spec, whatever their
    verdicts, a conjunction that failed at its first suite included."""
    a, b = env.c1, check("t", "rev", "ba", env.store)
    e1 = env.publish("e1", conj("all", a, b))
    members = []
    for i, answer in enumerate(["ab", "zz"]):  # "ab" passes both suites; "zz" fails the first
        runs = [env.adapter.run(env.store, a, answer), env.adapter.run(env.store, b, answer)]
        record = env.adapter.record(env.store, task="t", trajectory=f"d{i}", answer=answer, runs=runs)
        recorded = grade_recorded(env.adapter, record, e1.entries["t"]["*"], env.store, grader_version=e1.version_id)
        members.append(Member(member_id=f"d{i}", execution=record, prior=recorded,
                              recorded_reward=1.0 if recorded.passed else 0.0))
    (group,) = SealSession(env.registry, env.scorer, mode="discard").decide([Candidate("u0", tuple(members))])
    assert not group.neutralized and group.rewards() == [1.0, 0.0]
    assert [m.status for m in group.members] == ["consistent", "consistent"]
    # graded under e0 only: decided by a reused fail (a short circuit), but not graded under the batch spec
    stale = Candidate("u1", (env.member("zz"), env.member("ab")))
    (group,) = SealSession(env.registry, env.scorer, mode="discard").decide([stale])
    assert group.neutralized and group.neutralize_reason == "not_consistent"
    assert group.members[0].status == "repaired_short_same" and group.members[1].status == "needs_reexec"


def test_finish_refuses_a_group_decided_under_another_epoch(env):
    candidate = env.group("u0", ["ab", "ba"])
    old = SealSession(env.registry, env.scorer)
    (old_group,) = old.decide([candidate])
    env.publish("g2", check("t", "plain", "ba", env.store))
    new = SealSession(env.registry, env.scorer)
    (new_group,) = new.decide([candidate])
    assert old_group.rewards() != new_group.rewards()
    with pytest.raises(ValueError):
        new.finish([old_group])  # the same group id and members, decided under g1
    with pytest.raises(ValueError):
        new.finish([new_group.replace(role="backfill")])  # edited after the decision
    edited = new_group.replace(members=tuple(dataclasses.replace(m, trained_reward=0.5) for m in new_group.members))
    with pytest.raises(ValueError):
        new.finish([edited])
    assert new.finish([new_group]).epoch.name == "g2"


def test_finish_accepts_the_versions_assemble_hands_out(env):
    env.publish("g2", check("t", "plain", "ba", env.store))
    session = SealSession(env.registry, env.scorer)
    spare = env.group("spare", ["ba", "zz"])
    assembly = session.assemble([env.group("dead", ["zz", "zz"]), env.group("u0", ["ba", "zz"])],
                                backfill=lambda need, considered, remaining: [spare], fillers="neutral")
    assert [(g.group_id, g.role) for g in assembly.groups] == [("u0", "selected"), ("spare", "backfill")]
    decision = session.finish(assembly)
    assert decision.excluded[0].group.group_id == "dead"


def test_regrade_without_the_short_circuit_evaluates_every_suite(env):
    """A full re-grade that also evaluates the checks after a logical fail: Policy.FULL with short_circuit off."""
    e1 = env.publish("e1", conj("all", check("t", "plain", "zz", env.store), check("t", "rev", "ba", env.store)))
    member = env.member("ab")
    lazy = env.scorer.with_options(policy="full")
    (group,) = SealSession(env.registry, lazy).decide([Candidate(group_id="u0", members=(member,))])
    assert group.members[0].short_circuit and len(env.backend.executed) == 1
    full = env.scorer.with_options(policy="full", short_circuit=False)
    session = SealSession(env.registry, full)
    (group,) = session.decide([Candidate(group_id="u1", members=(member,))])
    assert not group.members[0].short_circuit and group.members[0].outcome is Outcome.FAIL
    (score,) = session.scores
    assert [p.outcome for p in score.parts] == [Outcome.FAIL, Outcome.PASS] and score.skipped == ()
    assert e1.name == "e1"


def test_a_recorded_fail_beside_a_suite_without_a_verdict_is_consistent(env):
    """Graded under e1 = A AND B: A's run errored and the record holds no artifact to run it again, B failed. The
    recorded verdict (a fail: the conjunction is monotone) is the label under e1, so the member is consistent and a
    discard-mode group keeps it."""
    a, b = env.c1, check("t", "dup", "abab", env.store)
    e1 = env.publish("e1", conj("all", a, b))
    runs = [env.adapter.run(env.store, a, "zz", error=True), env.adapter.run(env.store, b, "zz")]
    record = env.adapter.record(env.store, task="t", trajectory="u-f", answer=None, runs=runs)
    recorded = grade_recorded(env.adapter, record, e1.entries["t"]["*"], env.store, grader_version=e1.version_id)
    assert recorded.outcome is Outcome.FAIL
    passing = env.adapter.record(env.store, task="t", trajectory="p-p", answer="ab",
                                 runs=[env.adapter.run(env.store, a, "ab"), env.adapter.run(env.store, b, "ab")])
    passing_recorded = grade_recorded(env.adapter, passing, e1.entries["t"]["*"], env.store,
                                      grader_version=e1.version_id)
    members = (Member(member_id="uf", execution=record, prior=recorded, recorded_reward=0.0),
               Member(member_id="pp", execution=passing, prior=passing_recorded,
                      recorded_reward=1.0 if passing_recorded.passed else 0.0))
    (group,) = SealSession(env.registry, env.scorer, mode="discard").decide([Candidate("u0", members)])
    assert group.members[0].consistent and group.members[0].outcome is Outcome.FAIL
    assert group.members[1].status == "consistent" and group.rewards() == [0.0, 1.0] and not group.neutralized
