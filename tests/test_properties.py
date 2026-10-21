# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Property tests for the training-commit contract (docs/contract.md), over random histories.

Each history interleaves grader publications (changing runners, expected values, conjunctions, per-world truth and
the governed set), world publications (only when the current grader covers the world: epoch first), trajectories
that pin a world and an epoch when they start and are graded under that epoch, random evidence loss and torn
observations, flaky re-execution, and seals that assemble a batch with post-decision filtering and backfill.
After every seal it checks, against an oracle that computes verdicts from the answers alone:

* I1  no trained (non-neutralized) member is torn;
* I2  every trained label is the batch epoch's verdict on the record (for an ungoverned task: its recorded verdict);
* I3  the batch has one epoch; epochs never decrease in commit order;
* I4  the epoch is the latest one published when the seal read the pointer, and no label comes from another version;
* I5  a group is neutralized only if a member is undecidable, and a member is undecidable only for a reason the
      evidence explains (torn, uncovered world, missing evidence, or exhausted retries);

plus: recomputation only where something changed, and the conjunction short circuit only on logical fails.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pytest

from agent_ecc.compare import Policy
from agent_ecc.records import Method, Outcome
from agent_ecc.recompute.base import infra_result
from agent_ecc.scoring import Scorer, grade_recorded
from agent_ecc.seal import Candidate, Member, SealJournal, SealSession
from agent_ecc.store import MemoryStore
from agent_ecc.versions import (
    ANY_WORLD,
    EpochFirstError,
    GraderVersion,
    MemoryRegistry,
    iter_checks,
    spec_digest,
)

from toy import ToyAdapter, ToyBackend, check, conj, oracle_verdict

TASKS = ("t0", "t1", "t2")
WORLDS = ("w0", "w1")
RUNNERS = ("plain", "rev", "up", "dup")
ANSWERS = ("ab", "ba", "AB", "abab", "BA")
EXPECTED = ("ab", "ba", "AB", "abab", "BA", "abab")


class FlakyBackend(ToyBackend):
    def __init__(self, rng: random.Random, p: float):
        super().__init__()
        self.rng, self.p = rng, p
        self.failures = 0

    def _run(self, request, store):
        if self.rng.random() < self.p:
            self.failures += 1
            self.executed.append(request.request_id)
            return infra_result(request, "flaky sandbox")
        return super()._run(request, store)


@dataclass
class Trajectory:
    member: Member
    answer: str
    task: str
    world: str
    torn: bool
    pinned: GraderVersion
    lost_answer: bool
    used: bool = False


@dataclass
class History:
    rng: random.Random
    store: MemoryStore = field(default_factory=MemoryStore)
    adapter: ToyAdapter = field(default_factory=ToyAdapter)
    registry: MemoryRegistry = field(default_factory=MemoryRegistry)
    trajectories: list = field(default_factory=list)
    n: int = 0

    def random_check(self, name):
        return check(name, self.rng.choice(RUNNERS), self.rng.choice(EXPECTED), self.store)

    def random_spec(self, task):
        if self.rng.random() < 0.3:
            return conj("all", self.random_check(f"{task}-base"), self.random_check(f"{task}-hard"))
        return self.random_check(task)

    def random_version(self, k):
        entries = {}
        for task in TASKS:
            roll = self.rng.random()
            if roll < 0.1:
                entries[task] = {self.rng.choice(WORLDS): self.random_spec(task)}  # truth for one world only
            elif roll < 0.35:
                entries[task] = {w: self.random_spec(task) for w in WORLDS}
            else:
                entries[task] = {ANY_WORLD: self.random_spec(task)}
        governed = None
        if self.rng.random() < 0.25:
            governed = frozenset(t for t in TASKS if self.rng.random() < 0.7)
        return GraderVersion(name=f"g{k}", entries=entries, governed=governed)

    def start_and_grade(self):
        pin = self.registry.resolve_start()
        if pin.epoch is None or pin.world is None:
            return
        version = self.registry.grader(pin.epoch.version_id)
        task, answer = self.rng.choice(TASKS), self.rng.choice(ANSWERS)
        self.n += 1
        lookup = version.lookup(task, pin.world)
        runs = []
        if lookup.spec is not None:
            for c in iter_checks(lookup.spec):
                roll = self.rng.random()
                runs.append(self.adapter.run(self.store, c, answer, keep_output=roll > 0.15, error=roll < 0.05))
        lost = self.rng.random() < 0.1
        torn = self.rng.random() < 0.05
        other = [w for w in WORLDS if w != pin.world][0]
        record = self.adapter.record(self.store, task=task, trajectory=f"traj-{self.n}",
                                     answer=None if lost else answer, world=pin.world, runs=runs,
                                     observed=(pin.world, other) if torn else ())
        prior = ()
        if lookup.spec is not None:
            prior = (grade_recorded(self.adapter, record, lookup.spec, self.store,
                                    grader_version=version.version_id),)
        reward = None
        if prior and prior[0].logical:
            reward = 1.0 if prior[0].passed else 0.0
        member = Member(member_id=f"m{self.n}", execution=record, prior=prior, recorded_reward=reward)
        self.trajectories.append(Trajectory(member, answer, task, pin.world, torn, version, lost))

    def candidates(self, k, size):
        """Up to ``k`` groups of ``size`` unused trajectories of one task each, oldest first."""
        out = []
        for task in TASKS:
            pool = [t for t in self.trajectories if not t.used and t.task == task]
            while len(pool) >= size and len(out) < k:
                group, pool = pool[:size], pool[size:]
                for t in group:
                    t.used = True
                out.append(Candidate(group_id=f"u{group[0].member.member_id}",
                                     members=tuple(t.member for t in group), handle=group))
        return out


def check_seal(h: History, decision, session, read_seq, scorer_retries, policy):
    by_member = {t.member.member_id: t for t in h.trajectories}
    scores = {s.score_id: s for s in session.scores}
    epoch = decision.epoch
    version = h.registry.grader(epoch.version_id)
    assert epoch.seq == read_seq  # I4: the latest publication when the pointer was read
    for group in decision.groups:
        undecided = [m for m in group.members if not m.decided]
        assert group.neutralized == bool(undecided) or group.role == "filler"  # I5 (first half)
        for m in group.members:
            t = by_member[m.member_id]
            if m.score_id is not None:
                assert scores[m.score_id].grader_version == epoch.version_id  # I3 / I4: one version per batch
            if not m.decided:
                # I5 (second half): undecidable only for a reason the evidence explains
                lookup = version.lookup(t.task, t.world)
                explained = (t.torn or lookup.status == "uncovered" or t.lost_answer
                             or m.reason.startswith(("infra_error", "missing_artifact", "reparse_undeterminable"))
                             or m.reason in ("no_recorded_verdict",))
                assert explained, (m, t)
                continue
            assert not t.torn  # I1
            lookup = version.lookup(t.task, t.world)
            if lookup.status == "ungoverned":
                expected = t.member.prior[0].passed  # the recorded label stands
            else:
                expected = oracle_verdict(lookup.spec, t.answer, h.store)
            assert (m.outcome is Outcome.PASS) == expected, (m, t.answer, lookup)  # I2
            if not group.neutralized:
                assert m.trained_reward == m.target_reward
                if lookup.status != "ungoverned":
                    # I2 for the reward itself: the trained reward is the batch epoch's verdict's (recorded rewards
                    # here are the recorded verdicts' binary rewards, so a consistent member's recorded reward is too)
                    assert m.trained_reward == (1.0 if m.outcome is Outcome.PASS else 0.0), (m, t.answer)
            if m.consistent:
                # consistent: graded under exactly the batch epoch's spec, its recorded label kept
                pinned = t.pinned.lookup(t.task, t.world)
                assert pinned.spec is not None and spec_digest(pinned.spec) == spec_digest(lookup.spec)
                assert m.action == "reuse" and m.target_reward == m.recorded_reward
            score = scores.get(m.score_id)
            if score is not None and lookup.status != "ungoverned":
                # infrastructure errors never become verdicts: a pass needs every check to pass, a fail a check that
                # failed logically (an undeterminable check beside it decides nothing)
                if score.outcome is Outcome.PASS:
                    assert all(p.outcome is Outcome.PASS for p in score.parts) and not score.skipped
                else:
                    assert any(p.outcome is Outcome.FAIL for p in score.parts)
                if score.skipped:
                    assert score.outcome is Outcome.FAIL  # the short circuit only on a logical fail
                if m.action == "reuse" and policy is Policy.ECC:
                    prior = {(p.execution_digest, p.grading_digest) for p in t.member.prior[0].parts}
                    reused = {(p.execution_digest, p.grading_digest) for p in score.parts
                              if p.method is Method.RECORDED}
                    assert reused <= prior


def run_history(seed, *, policy=Policy.ECC, steps=60, flaky=0.1, retries=2, journal=None, on_seal=None):
    rng = random.Random(seed)
    h = History(rng)
    backend = FlakyBackend(rng, flaky)
    scorer = Scorer([h.adapter], h.store, backend, policy=policy, retries=retries)
    startup = h.random_version(0)
    while not startup.covers_world("w0"):  # epoch first: the startup world needs truth before it is resolvable
        startup = h.random_version(0)
    h.registry.publish_grader(startup, source="startup")
    h.registry.publish_world("w0")
    k = 1
    seals = []
    for step in range(steps):
        roll = rng.random()
        if roll < 0.12:
            try:
                h.registry.publish_grader(h.random_version(k), source="switch")
            except EpochFirstError:
                h.registry.publish_grader(h.random_version(k), source="switch", require_coverage=False)
            k += 1
        elif roll < 0.17:
            try:
                h.registry.publish_world(rng.choice(WORLDS))
            except EpochFirstError:
                pass  # the epoch-first rule held the world back
        elif roll < 0.75:
            h.start_and_grade()
        else:
            initial = h.candidates(2, 2)
            if not initial:
                continue
            read_seq = h.registry.publications("grader")[-1].seq
            session = SealSession(h.registry, scorer, batch_id=f"step-{step}")
            if on_seal is not None:
                on_seal(h, session.epoch, initial)

            def backfill(need, considered, remaining):
                if rng.random() < 0.3:  # a publication while the seal is open must not reach this batch
                    try:
                        h.registry.publish_grader(h.random_version(100 + step), source="race",
                                                  require_coverage=False)
                    except EpochFirstError:
                        pass
                return h.candidates(need, 2)

            assembly = session.assemble(initial, backfill=backfill, fillers=rng.choice(["none", "neutral"]))
            decision = session.finish(assembly, journal=journal)
            assert decision.failed is None
            check_seal(h, decision, session, read_seq, retries, policy)
            seals.append(decision)
    return h, seals, backend


@pytest.mark.parametrize("seed", range(40))
def test_invariants_hold_over_random_histories(seed, tmp_path):
    journal = SealJournal(tmp_path / "seals.jsonl")
    _, seals, _ = run_history(seed, journal=journal)
    committed = journal.read()
    assert [d.decision_id for d in committed] == [d.decision_id for d in seals]
    seqs = [d.epoch.seq for d in committed]
    assert seqs == sorted(seqs)  # I3: non-decreasing in commit order


@pytest.mark.parametrize("seed", range(12))
def test_conservative_policies_agree_with_ecc_wherever_both_decide(seed):
    """At every seal of a history, decide the same groups under the same epoch with each policy (a replay: the seal
    is given the epoch it read). Wherever two policies both decide a member, they give it the same label; the more
    conservative ones never decide a member ECC cannot, and execute at least as much."""
    compared = {"members": 0}

    def on_seal(h, epoch, candidates):
        results = {}
        for policy in Policy:
            backend = ToyBackend()
            scorer = Scorer([h.adapter], h.store, backend, policy=policy)
            groups = SealSession(epoch, scorer).decide(candidates)
            results[policy] = ({m.member_id: m for g in groups for m in g.members}, len(backend.executed))
        ecc, ecc_exec = results[Policy.ECC]
        for policy in (Policy.REPARSE_ALL, Policy.FULL):
            other, other_exec = results[policy]
            assert other_exec >= ecc_exec
            for mid, m in other.items():
                if m.decided:
                    assert ecc[mid].decided and ecc[mid].outcome is m.outcome, (policy, m, ecc[mid])
                    compared["members"] += 1

    run_history(seed, flaky=0.0, on_seal=on_seal)
    assert compared["members"] > 0


@pytest.mark.parametrize("seed", range(12))
def test_recomputation_happens_only_where_something_changed(seed):
    """Without publications after startup, every seal reuses recorded verdicts: nothing is parsed again and nothing
    is executed, except where the recorded evidence held no verdict (an infrastructure error, a missing output)."""
    rng = random.Random(seed)
    h = History(rng)
    backend = ToyBackend()
    scorer = Scorer([h.adapter], h.store, backend)
    startup = h.random_version(0)
    while not startup.covers_world("w0"):
        startup = h.random_version(0)
    h.registry.publish_grader(startup)
    h.registry.publish_world("w0")
    for _ in range(30):
        h.start_and_grade()
    session = SealSession(h.registry, scorer)
    groups = session.decide(h.candidates(20, 2))
    executed_for = set()
    for g in groups:
        for m in g.members:
            t = next(x for x in h.trajectories if x.member.member_id == m.member_id)
            recorded = t.member.prior[0] if t.member.prior else None
            if recorded is not None and recorded.logical and not t.torn:
                assert m.action in ("reuse", "ungoverned"), (m, recorded)
            elif m.action == "reexec":
                executed_for.add(m.member_id)
    assert len(backend.executed) <= len(executed_for) * 2 + 1  # at most one run per check of a 2-check spec


def test_a_batch_never_mixes_epochs_under_racing_publications():
    """Publish a new version before every backfill round and check that each backfilled group was decided under
    the epoch the seal read, not the newer ones."""
    for seed in range(20):
        rng = random.Random(1000 + seed)
        h = History(rng)
        scorer = Scorer([h.adapter], h.store, ToyBackend())
        startup = h.random_version(0)
        while not startup.covers_world("w0"):
            startup = h.random_version(0)
        h.registry.publish_grader(startup)
        h.registry.publish_world("w0")
        for _ in range(40):
            h.start_and_grade()
        session = SealSession(h.registry, scorer)
        read = session.epoch.ref

        def backfill(need, considered, remaining, _k=[0]):
            _k[0] += 1
            h.registry.publish_grader(h.random_version(500 + _k[0]), require_coverage=False)
            return h.candidates(need, 2)

        decision = session.finish(session.assemble(h.candidates(3, 2), backfill=backfill))
        assert decision.epoch == read
        assert {s.grader_version for s in session.scores} <= {read.version_id}
