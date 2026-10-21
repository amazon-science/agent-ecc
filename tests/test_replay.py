# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replaying seals from their inputs: a seal's decision is a function of what it was given.

An integration that logs what each seal was given (the candidates offered in each round, with their records and
prior scores; the time the seal read the epoch pointer) and records its re-executions' raw outputs
(:class:`agent_ecc.recompute.RecordingBackend`) can re-decide every seal later and get the same decision, member by
member. The logged decisions are never inputs: scrambling them changes nothing. This is also how an integration is
checked against a built-in implementation it replaces: a reader turns that implementation's logs into these inputs,
and the replayed decisions are compared with the ones it logged (docs/integrations/trainer.md, section 7).
"""

from __future__ import annotations

import dataclasses
import json

from agent_ecc.recompute.base import RecordedBackend, RecordingBackend
from agent_ecc.records import SealDecision
from agent_ecc.scoring import Scorer, grade_recorded
from agent_ecc.seal import Candidate, Member, SealJournal, SealSession
from agent_ecc.store import MemoryStore
from agent_ecc.versions import Epoch, GraderVersion, MemoryRegistry

from toy import ToyAdapter, ToyBackend, check, conj


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


def _world():
    store, adapter = MemoryStore(), ToyAdapter()
    clock = Clock()
    registry = MemoryRegistry(clock=clock)
    c1 = check("t", "plain", "ab", store)
    g1 = GraderVersion(name="g1", entries={"t": c1})
    registry.publish_grader(g1)
    members = {}
    for i, answer in enumerate(["ab", "ba", "zz", "abab", "ab", "ba", "BA", "zz"]):
        record = adapter.record(store, task="t", trajectory=f"a{i}", answer=answer,
                                runs=[adapter.run(store, c1, answer)])
        recorded = grade_recorded(adapter, record, c1, store, grader_version=g1.version_id)
        members[f"m{i}"] = Member(member_id=f"m{i}", execution=record, prior=recorded,
                                  recorded_reward=1.0 if recorded.passed else 0.0)
    # a hardened suite, then a label fix of the first suite and another hardened suite, published while training runs
    registry.publish_grader(GraderVersion(name="g2", entries={"t": conj("all", c1, check("t", "rev", "ba", store))}))
    registry.publish_grader(GraderVersion(name="g3", entries={"t": conj("all", check("t", "plain", "ba", store),
                                                                       check("t", "up", "BA", store))}))
    return store, adapter, registry, members


def _groups(members, plan):
    return [Candidate(group_id=gid, members=tuple(members[m] for m in mids)) for gid, mids in plan]


ROUNDS = [[("u0", ["m0", "m1"]), ("u1", ["m2", "m3"])], [("u2", ["m4", "m5"])], [("u3", ["m6", "m7"])]]


def _seal(registry, scorer, members, *, epoch=None, journal=None, seal_id="s0", clock=None):
    rounds = [_groups(members, plan) for plan in ROUNDS]
    session = SealSession(epoch if epoch is not None else registry, scorer, batch_id="step-7", seal_id=seal_id,
                          clock=clock or (lambda: 5.0))
    later = rounds[1:]
    assembly = session.assemble(rounds[0], backfill=lambda need, considered, remaining: later.pop(0) if later else [],
                                clock=lambda: 0.0)
    return session, session.finish(assembly, journal=journal)


def _essence(decision: SealDecision) -> dict:
    """Everything a seal decided (its timing aside)."""
    d = decision.to_dict()
    for key in ("opened_at", "closed_at", "decision_id"):
        d.pop(key)
    d["epoch"].pop("read_at")
    d["stats"].pop("decide_seconds")
    return d


def test_a_seal_replays_from_its_inputs_alone(tmp_path):
    store, adapter, registry, members = _world()
    results = tmp_path / "results.jsonl"
    live = Scorer([adapter], store, RecordingBackend(ToyBackend(), results), clock=lambda: 0.0)
    journal = SealJournal(tmp_path / "seals.jsonl")
    session, decision = _seal(registry, live, members, journal=journal)
    assert decision.epoch.name == "g3" and decision.rounds == 2 and decision.stats["executions"] > 0
    assert [g.group_id for g in decision.groups] == ["u0", "u2"] and decision.rewards() == {
        "m0": 0.0, "m1": 1.0, "m4": 0.0, "m5": 1.0}  # u1 has no variance under g3: excluded, then backfilled
    read_at = session.epoch.ref.read_at

    # the replay's inputs: the publication log, the read time, the offered rounds and the recorded raw outputs
    ref = registry.epoch_at(read_at)
    epoch = Epoch(ref=dataclasses.replace(ref, read_at=read_at), version=registry.grader(ref.version_id))
    recorded = RecordedBackend.from_jsonl(results)
    replay = Scorer([adapter], store, recorded, clock=lambda: 0.0)
    _, again = _seal(registry, replay, members, epoch=epoch)
    (logged,) = journal.read()
    assert _essence(again) == _essence(logged)
    assert recorded.missing == []  # every execution the replay asked for was one the seal made


def test_logged_decisions_are_not_inputs(tmp_path):
    store, adapter, registry, members = _world()
    results = tmp_path / "results.jsonl"
    live = Scorer([adapter], store, RecordingBackend(ToyBackend(), results), clock=lambda: 0.0)
    session, decision = _seal(registry, live, members)
    read_at = session.epoch.ref.read_at
    scrambled = json.loads(json.dumps(decision.to_dict()))
    for group in scrambled["groups"]:
        for m in group["members"]:
            m.update(trained_reward=0.5, action="failed", consistent=False)
    # the replay never reads a logged decision: re-deciding gives the original one, not the scrambled one
    ref = registry.epoch_at(read_at)
    epoch = Epoch(ref=dataclasses.replace(ref, read_at=read_at), version=registry.grader(ref.version_id))
    _, again = _seal(registry, Scorer([adapter], store, RecordedBackend.from_jsonl(results), clock=lambda: 0.0),
                     members, epoch=epoch)
    assert _essence(again) == _essence(decision) and again.rewards() != {
        m["member_id"]: m["trained_reward"] for g in scrambled["groups"] for m in g["members"]}


def test_a_replay_under_another_epoch_is_detected(tmp_path):
    store, adapter, registry, members = _world()
    results = tmp_path / "results.jsonl"
    live = Scorer([adapter], store, RecordingBackend(ToyBackend(), results), clock=lambda: 0.0)
    _, decision = _seal(registry, live, members)
    g2 = registry.publications("grader")[1]
    wrong = Epoch(ref=g2.ref(read_at=0.0), version=registry.grader(g2.version_id))
    _, again = _seal(registry, Scorer([adapter], store, RecordedBackend.from_jsonl(results), clock=lambda: 0.0),
                     members, epoch=wrong)
    assert again.epoch.name == "g2" and _essence(again)["groups"] != _essence(decision)["groups"]
