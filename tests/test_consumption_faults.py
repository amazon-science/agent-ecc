# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The consumption join against seeded faults.

Each fault is a bug in the hand-off between the seal and the learner (*pre-stamp*: what was delivered differs from
what was sealed, and the hand-off stamp records what was delivered) or in the learner (*post-stamp*: the learner
trains something other than what was stamped). They are the ten kinds of a fault-injection test of a trainer's
consumption check: a stale label delivered (F1a) or trained (F1b); a sealed record removed from the loss before (F2a)
or after (F2b) the stamp; a record dropped from the delivered list for another (F2-list); a record consumed twice in
one step (F3a) or again in a later step (F3b); a group sealed under one epoch delivered into a batch sealed under
another (F4); advantages from stale group statistics (F5); positional attribution of one record's tokens to another
(F6); a record normalized in another group (F7).

For each kind the join must report the primary finding on the primary records, nothing on the records the fault did
not touch, and nothing at all on the clean run. F2a was the one kind a trainer's join missed (its stamp and its
learner lines agree on 0 tokens): ``zero_loss_tokens`` reports it."""

import math

import pytest

from agent_ecc.canonical import thaw
from agent_ecc.export import ISSUE_KEYS, consumption_records, group_normalized
from agent_ecc.records import EpochRef, GroupDecision, MemberDecision, SealDecision

E0 = EpochRef(seq=0, version_id="sha256:" + "0" * 64, name="e0", published_at=1.0, read_at=2.0)
E1 = EpochRef(seq=1, version_id="sha256:" + "1" * 64, name="e1", published_at=50.0, read_at=51.0)
STEPS, GROUPS, SIZE, SWITCH = 6, 4, 4, 3


def _rewards(step, g):
    """Sealed (trained) and recorded rewards of a group: after the switch the seal changed some labels."""
    trained = [1.0 if (step + g + k) % 3 == 0 else 0.0 for k in range(SIZE)]
    trained[0], trained[1] = 1.0, 0.0  # every group is PG-active
    recorded = list(trained)
    if step >= SWITCH and g % 2 == 0:
        recorded[2] = 1.0 - trained[2]  # a label the seal changed under the new epoch
    return trained, recorded


def _decision(step):
    groups = []
    for g in range(GROUPS):
        trained, recorded = _rewards(step, g)
        members = tuple(MemberDecision(member_id=f"s{step}g{g}m{k}", execution_id=f"x{step}{g}{k}", action="reuse",
                                       outcome="pass" if t else "fail", score_id=f"sc{step}{g}{k}",
                                       recorded_reward=r, target_reward=t, trained_reward=t,
                                       flipped=t != r)
                        for k, (t, r) in enumerate(zip(trained, recorded)))
        groups.append(GroupDecision(group_id=f"s{step}g{g}", members=members))
    return SealDecision(seal_id=f"seal-{step}", epoch=E0 if step < SWITCH else E1, mode="repair", policy="ecc",
                        strict=True, groups=tuple(groups), batch_id=f"step-{step}", target_groups=GROUPS)


def _tokens(mid):
    return 100 + (sum(map(ord, mid)) % 50)


class Run:
    """A clean run: per step, the seal's decision and the learner lines that consume it exactly. Faults edit a step's
    delivered samples (pre-stamp) or the learner's view of them (post-stamp); ``lines()`` recomputes each line from
    the samples as the learner would (group-normalized rewards over the delivered groups)."""

    def __init__(self):
        self.decisions = [_decision(s) for s in range(STEPS)]
        self.delivered = {}  # step -> list of samples (dicts), in hand-off order
        for d in self.decisions:
            step = int(d.batch_id.split("-")[1])
            samples = []
            for g in d.groups:
                for m in g.members:
                    samples.append({"member_id": m.member_id, "group_id": g.group_id, "reward": m.trained_reward,
                                    "stamp_reward": m.trained_reward, "tokens": _tokens(m.member_id),
                                    "stamp_tokens": _tokens(m.member_id), "identity": "verified",
                                    "train_reward": m.trained_reward, "norm_group": g.group_id, "removed": False,
                                    "recorded": m.recorded_reward})
            self.delivered[step] = samples
        self.touched: dict[int, set] = {s: set() for s in range(STEPS)}

    def member(self, step, g, k):
        return self.delivered[step][g * SIZE + k]

    def lines(self):
        out = []
        for step, samples in self.delivered.items():
            groups: dict = {}
            for s in samples:
                if s.get("stale_stats") is None:
                    groups.setdefault(s["norm_group"], []).append(s)
            normalized = {}
            for key, members in groups.items():
                values = group_normalized([m["train_reward"] for m in members], std=True)
                for m, v in zip(members, values):
                    normalized[id(m)] = v
            for s in samples:
                if s.get("stale_stats") is not None:
                    mean, sd = s["stale_stats"]
                    normalized[id(s)] = (s["train_reward"] - mean) / (sd + 1e-6)
            for i, s in enumerate(samples):
                adv = None if s["removed"] else normalized[id(s)]
                out.append({"batch_id": f"step-{step}", "member_id": s["member_id"], "group_id": s["group_id"],
                            "step": step, "reward": s["stamp_reward"], "normalized": normalized[id(s)],
                            "advantage": adv, "advantage_min": adv, "advantage_max": adv,
                            "loss_tokens": 0 if s["removed"] else s["tokens"],
                            "handoff_loss_tokens": s["stamp_tokens"], "identity": s["identity"],
                            "reporter": f"dp{i % 2}", "microbatch": i // 2, "step_part": 0})
        return out

    def completions(self):
        return [{"batch_id": f"step-{s}", "step": s, "completed_at": float(s), "ok": True} for s in range(STEPS)]

    def report(self):
        return consumption_records(self.decisions, self.lines(), self.completions(), normalization="mean_std")

    def touch(self, step, *mids):
        self.touched[step].update(mids)

    def touch_group(self, step, gid):
        self.touched[step].update(s["member_id"] for s in self.delivered[step] if s["group_id"] == gid)


def findings(report):
    """(step, category, member) for every finding."""
    out = set()
    for r in report.records:
        step = int(r.batch_id.split("-")[1])
        for category, items in r.issues.items():
            for item in items:
                out.add((step, category, item["member_id"]))
    return out


def _sealed_member(run, step, g, k):
    return run.decisions[step].groups[g].members[k]


# ---------------------------------------------------------------------------------------------- the faults


def f1a(run, step=4, g=0, k=2):
    """Pre-stamp: the hand-off delivers the record's recorded label instead of the sealed one."""
    s = run.member(step, g, k)
    assert s["recorded"] != s["reward"]
    s["reward"] = s["stamp_reward"] = s["train_reward"] = s["recorded"]
    run.touch_group(step, s["group_id"])
    return "reward_mismatch", [s["member_id"]]


def f1b(run, step=4, g=2, k=2):
    """Post-stamp: the learner trains the record's pre-seal reward (it reads a stale field); the stamp is right."""
    s = run.member(step, g, k)
    assert s["recorded"] != s["reward"]
    s["train_reward"] = s["recorded"]
    run.touch_group(step, s["group_id"])
    return "advantage_inconsistent", [s["member_id"]]


def f2a(run, step=1, g=1, k=1):
    """Pre-stamp: a sealed record is removed from the loss at the hand-off: its stamp says 0 tokens, it trains 0."""
    s = run.member(step, g, k)
    s["removed"], s["stamp_tokens"] = True, 0
    run.touch(step, s["member_id"])
    return "zero_loss_tokens", [s["member_id"]]


def f2b(run, step=2, g=0, k=2):
    """Post-stamp: the learner removes a sealed record from its loss."""
    s = run.member(step, g, k)
    s["removed"] = True
    run.touch(step, s["member_id"])
    return "loss_tokens_mismatch", [s["member_id"]]


def f3a(run, step=1, g=2, k=1):
    """Pre-stamp: the next member's slot gets a copy of the source: consumed twice; the victim is not consumed."""
    samples = run.delivered[step]
    i = g * SIZE + k
    source, victim = samples[i], samples[i + 1]
    run.touch_group(step, source["group_id"])  # the victim among them
    samples[i + 1] = dict(source)
    return "duplicate_consumption", [source["member_id"]], [("records_unconsumed", victim["member_id"])]


def f3b(run, step=5, g=1, k=3, source_step=4):
    """Pre-stamp: a record consumed at the previous step is delivered again, in one slot of this step."""
    samples = run.delivered[step]
    again = dict(run.member(source_step, g, 0))
    victim = samples[g * SIZE + k]
    run.touch_group(step, victim["group_id"])  # the victim among them
    samples[g * SIZE + k] = {**again, "group_id": victim["group_id"], "norm_group": victim["group_id"]}
    run.touch(step, again["member_id"])
    return "records_unsealed", [again["member_id"]], [("records_unconsumed", victim["member_id"])]


def f4(run, step=4, g=3, source_step=2, source_g=3):
    """Pre-stamp: a whole group sealed under e0 at an earlier step replaces one group of an e1 batch."""
    samples = run.delivered[step]
    old = [dict(s) for s in run.delivered[source_step][source_g * SIZE:(source_g + 1) * SIZE]]
    victims = [s["member_id"] for s in samples[g * SIZE:(g + 1) * SIZE]]
    samples[g * SIZE:(g + 1) * SIZE] = old
    run.touch(step, *victims, *(s["member_id"] for s in old))
    return "epoch_mixed", [s["member_id"] for s in old], [("records_unconsumed", v) for v in victims] + [
        ("records_unsealed", s["member_id"]) for s in old]


def f5(run, step=5, g=0):
    """Post-stamp: advantages from stale group statistics (the recorded labels' mean and standard deviation)."""
    members = run.delivered[step][g * SIZE:(g + 1) * SIZE]
    recorded = [s["recorded"] for s in members]
    mean = sum(recorded) / len(recorded)
    sd = math.sqrt(sum((r - mean) ** 2 for r in recorded) / (len(recorded) - 1))
    for s in members:
        s["stale_stats"] = (mean, sd)
    run.touch_group(step, members[0]["group_id"])
    return "advantage_inconsistent", [s["member_id"] for s in members]


def f6(run, step=1, g=3, ks=(0, 1)):
    """Post-stamp: two members' tokens are trained under each other's identity (positional attribution)."""
    a, b = run.member(step, g, ks[0]), run.member(step, g, ks[1])
    for key in ("member_id", "stamp_reward", "stamp_tokens"):
        a[key], b[key] = b[key], a[key]
    a["identity"] = b["identity"] = "mismatch"
    run.touch_group(step, a["group_id"])
    return "identity_unverified", [a["member_id"], b["member_id"]]


def f7(run, step=2, g=1, k=0, dest=2):
    """Post-stamp: a record is normalized in another delivered group."""
    s = run.member(step, g, k)
    s["norm_group"] = f"s{step}g{dest}"
    run.touch_group(step, s["group_id"])
    run.touch_group(step, f"s{step}g{dest}")
    return "advantage_inconsistent", [s["member_id"]]


FAULTS = {"F1a": (f1a, 4), "F1b": (f1b, 4), "F2a": (f2a, 1), "F2b": (f2b, 2), "F3a": (f3a, 1), "F3b": (f3b, 5),
          "F4": (f4, 4), "F5": (f5, 5), "F6": (f6, 1), "F7": (f7, 2)}


def _apply(run, kind):
    fn, step = FAULTS[kind]
    out = fn(run)
    category, primary = out[0], out[1]
    secondary = out[2] if len(out) > 2 else []
    return step, category, primary, secondary


# ---------------------------------------------------------------------------------------------- tests


def test_the_clean_run_has_no_finding():
    report = Run().report()
    assert report.manifest.ok and findings(report) == set()
    assert report.manifest.records_consumed == STEPS * GROUPS * SIZE
    assert set(report.records[0].issues) == set(ISSUE_KEYS)


@pytest.mark.parametrize("kind", sorted(FAULTS))
def test_every_seeded_fault_is_reported_in_its_category_with_no_false_positive(kind):
    run = Run()
    step, category, primary, secondary = _apply(run, kind)
    report = run.report()
    found = findings(report)
    assert not report.manifest.ok
    for mid in primary:
        assert (step, category, mid) in found, (kind, mid, sorted(f for f in found if f[2] == mid))
    for cat, mid in secondary:  # the victims of a dropped slot; the re-delivered records
        assert (step, cat, mid) in found, (kind, cat, mid)
    stray = {f for f in found if f[0] != step or f[2] not in run.touched[step]}
    assert stray == set(), (kind, stray)


def test_every_kind_at_once_is_reported_and_nothing_else():
    """The registered schedule's shape: several kinds per step, every step with a clean group left."""
    run = Run()
    expected = [_apply(run, kind) for kind in ("F2a", "F3a", "F6", "F2b", "F7", "F3b", "F1a", "F4", "F1b", "F5")]
    found = findings(run.report())
    for step, category, primary, secondary in expected:
        for mid in primary:
            assert (step, category, mid) in found, (category, mid)
        for cat, mid in secondary:
            assert (step, cat, mid) in found, (cat, mid)
    stray = {f for f in found if f[2] not in run.touched[f[0]]}
    assert stray == set()


def test_a_record_removed_before_the_stamp_is_found_only_by_its_zero_tokens():
    """The trainer's join missed this kind: the stamp and the learner agree on 0 tokens, the record has a learner
    line, its normalized reward is unchanged and it has no advantage to check."""
    run = Run()
    step, _, (mid,), _ = _apply(run, "F2a")
    found = {f for f in findings(run.report()) if f[2] == mid}
    assert found == {(step, "zero_loss_tokens", mid)}


def test_a_record_delivered_again_names_the_seal_that_handed_it_over():
    run = Run()
    step, _, (again,), _ = _apply(run, "F3b")
    report = run.report()
    (entry,) = [i for r in report.records for i in r.issues["records_unsealed"] if i["member_id"] == again]
    assert thaw(entry["sealed_in"]) == [{"batch_id": "step-4", "epoch_seq": 1, "version_id": E1.version_id}]
    # the same epoch: delivered again, but no epoch mixing
    assert not any(i["member_id"] == again for r in report.records for i in r.issues["epoch_mixed"])


def test_a_batch_mixing_epochs_is_reported_as_such():
    run = Run()
    step, _, old, _ = _apply(run, "F4")
    report = run.report()
    (record,) = [r for r in report.records if r.batch_id == f"step-{step}"]
    mixed = record.issues["epoch_mixed"]
    assert sorted(i["member_id"] for i in mixed) == sorted(old)
    assert all(i["batch_epoch"]["seq"] == 1 and i["sealed_in"][0]["epoch_seq"] == 0 for i in mixed)
    assert report.manifest.issues["epoch_mixed"] == len(old)
