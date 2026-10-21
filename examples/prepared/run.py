# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A grader switch sealed three ways, on synthetic data: synchronous ECC, ECC with prepared re-execution, and the
never arm (the batch epoch pinned to the startup epoch).

Twelve SWE-bench-style trajectories (three tasks, one group of four each) are graded under e0 and admitted to the
buffer. Then e1 is published: the same tasks, with a hardened suite added to two of them (e1 = original suite AND
hardened suite). Each arm seals the same groups on its own copy of the world:

* ``sync``      ECC: the seal re-executes, at the seal, the hardened suite of every e0 pass of a hardened task;
* ``prepared``  ECC with a :class:`agent_ecc.prepare.Preparer`: the admitted groups are planned under e1 as soon as it
                is published and their re-executions run in the background; the seal takes each prepared result only
                for the request it would execute, under its own epoch, and executes nothing itself;
* ``never``     the batch epoch stays e0: the e0 passes that e1 retracts still train as passes; the decision records
                the pin and the epoch the pointer named.

The example checks that ``prepared`` decides exactly what ``sync`` decides, audits every arm's seal log against the
publication log (``ecc`` and ``prepared`` read the epoch in force; ``never`` is reported as pinned), and writes under
--out each arm's seal log, the preparer's event log and a summary.

Run:  python examples/prepared/run.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import agent_ecc  # noqa: F401
except ImportError:  # running from a source checkout without installing
    sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "lifecycle"))

from agent_ecc import (  # noqa: E402
    Candidate,
    MemoryRegistry,
    MemoryStore,
    Member,
    Preparer,
    Scorer,
    SealJournal,
    arm_session,
    audit_epochs,
)
from agent_ecc.adapters.swebench import SimpleHarness, SWEBenchAdapter  # noqa: E402
from agent_ecc.scoring import grade_recorded  # noqa: E402

import simworld  # noqa: E402


def t(module: str, name: str) -> str:
    return f"tests/test_{module}.py::test_{name}"


TASKS = {
    "demo__calc-1": dict(f2p=[t("calc", "round")], p2p=[t("calc", "add")], passing={t("calc", "add")},
                         hardened=[t("calc", "round_half")]),
    "demo__text-2": dict(f2p=[t("text", "upper")], p2p=[t("text", "lower")], passing={t("text", "lower")},
                         hardened=[t("text", "upper_unicode")]),
    "demo__text-3": dict(f2p=[t("text", "strip")], p2p=[t("text", "join")], passing={t("text", "join")},
                         hardened=None),
}
PATCHES = {
    "demo__calc-1": [f"fix:{t('calc', 'round')}", f"fix:{t('calc', 'round')} fix:{t('calc', 'round_half')}",
                     "an unrelated edit", f"fix:{t('calc', 'round')} break:{t('calc', 'add')}"],
    "demo__text-2": [f"fix:{t('text', 'upper')}", f"fix:{t('text', 'upper')}",
                     f"fix:{t('text', 'upper')} fix:{t('text', 'upper_unicode')}", ""],
    "demo__text-3": [f"fix:{t('text', 'strip')}", "", "another unrelated edit", f"fix:{t('text', 'strip')}"],
}


def row(task: str) -> dict:
    spec = TASKS[task]
    return {"instance_id": task, "repo": "demo/" + task.split("__")[1].split("-")[0], "version": "1.0",
            "image": simworld.IMAGE, "eval_script": simworld.eval_script(sorted(spec["f2p"] + spec["p2p"])),
            "log_parser": "parse_log_pytest", "eval_type": "pass_and_fail", "FAIL_TO_PASS": spec["f2p"],
            "PASS_TO_PASS": spec["p2p"]}


def hardened_row(task: str) -> dict:
    spec = TASKS[task]
    tests = sorted(spec["f2p"] + spec["p2p"] + spec["hardened"])
    return {"instance_id": task, "FAIL_TO_PASS": spec["f2p"] + spec["hardened"], "PASS_TO_PASS": spec["p2p"],
            "eval_script": simworld.eval_script(tests)}


class World:
    """One copy of the run: a registry, a store, the simulated evaluator, e0 published, the groups graded under it."""

    def __init__(self) -> None:
        self.store = MemoryStore()
        self.adapter = SWEBenchAdapter(harness=SimpleHarness())
        self.backend = simworld.SimulatedEvaluator({task: spec["passing"] for task, spec in TASKS.items()})
        self.registry = MemoryRegistry()
        self.e0 = self.adapter.version_from_rows([row(task) for task in TASKS], self.store, name="e0")
        self.e1 = self.adapter.conjunction_version([row(task) for task in TASKS],
                                                   [hardened_row(task) for task, s in TASKS.items() if s["hardened"]],
                                                   self.store, name="e1")
        self.registry.publish_grader(self.e0, source="startup")
        self.registry.publish_world(simworld.IMAGE, source="startup")
        self.cache: dict = {}  # the run's cache: one dict for every seal
        self.groups = [self._group(task) for task in TASKS]

    def scorer(self) -> Scorer:
        return Scorer([self.adapter], self.store, self.backend, cache=self.cache)

    def _group(self, task: str) -> Candidate:
        pin = self.registry.resolve_start()  # every trajectory pins the world and the grader epoch at its start
        check = self.registry.grader(pin.epoch.version_id).entries[task]["*"]
        script = self.store.get(str(check.execution["eval_script"])).decode()
        members = []
        for i, patch in enumerate(PATCHES[task]):
            log, applied = simworld.run(patch, script, passing_by_default=TASKS[task]["passing"])
            runs = [self.adapter.run(check, store=self.store, log=log, applied=applied)] if patch.strip() else []
            record = self.adapter.record(store=self.store, task_id=task, trajectory_id=f"{task}/a{i}",
                                         image=pin.world, patch=patch, runs=runs)
            recorded = grade_recorded(self.adapter, record, check, self.store, grader_version=pin.epoch.version_id)
            members.append(Member(member_id=record.trajectory_id, execution=record, prior=(recorded,),
                                  recorded_reward=1.0 if recorded.passed else 0.0))
        return Candidate(group_id=task, members=tuple(members))

    def switch(self) -> None:
        self.registry.publish_grader(self.e1, source="switch")


def seal(world: World, arm: str, out: str, **kwargs) -> dict:
    """Seal the run's three groups as ``arm`` does; returns what the seal decided and what it executed."""
    journal = SealJournal(os.path.join(out, f"seals-{arm}.jsonl"))
    before = world.backend.calls
    session = arm_session("never" if arm == "never" else "ecc", world.registry, world.scorer(), cache=world.cache,
                          batch_id="step-1", **kwargs)
    decision = session.finish(session.decide(list(world.groups)), journal=journal)
    audit = audit_epochs(journal.read(), world.registry)
    return {"decision": decision, "epoch": decision.epoch.name, "executed_at_seal": world.backend.calls - before,
            "pinned_to": decision.target_pin.pin if decision.target_pin else None,
            "pointer": decision.target_pin.published.name if decision.target_pin else decision.epoch.name,
            "audit": [s["status"] for s in audit.seals], "audit_ok": audit.ok}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=None, help="output directory (default: a new temporary directory)")
    args = ap.parse_args(argv)
    out = args.out or tempfile.mkdtemp(prefix="agent-ecc-prepared-")
    os.makedirs(out, exist_ok=True)

    # synchronous ECC
    sync_world = World()
    sync_world.switch()
    sync = seal(sync_world, "sync", out)

    # ECC with prepared re-execution: the groups are admitted under e0, prepared once e1 is published
    world = World()
    preparer = Preparer(world.registry, world.scorer(), log=os.path.join(out, "prepared.jsonl"))
    preparer.admit(world.groups)
    preparer.plan()  # under the startup epoch: nothing to prepare
    nothing_under_e0 = not preparer.queue
    world.switch()
    preparer.poll()  # the publication: every admitted group no seal decided is planned under e1
    preparer.plan()
    background = preparer.run()  # the background jobs (a deployment runs them in threads: Preparer(threads=True))
    background_executions = world.backend.calls
    prepared = seal(world, "prepared", out, prepared=preparer)

    # the never arm: the batch epoch pinned to e0
    never_world = World()
    never_world.switch()
    never = seal(never_world, "never", out)

    def essence(decision):
        return [(m.member_id, m.status, m.trained_reward) for _, m in decision.members()]

    summary = {
        "out": out,
        "members": sum(len(g.members) for g in world.groups),
        "nothing_prepared_under_e0": nothing_under_e0,
        "sync": {"epoch": sync["epoch"], "executed_at_seal": sync["executed_at_seal"],
                 "rewards": sync["decision"].rewards(), "audit": sync["audit"]},
        "prepared": {"epoch": prepared["epoch"], "background_jobs": background,
                     "background_executions": background_executions,
                     "executed_at_seal": prepared["executed_at_seal"],
                     "answers": dict(prepared["decision"].stats["prepared"]["by_reason"]),
                     "same_decisions_as_sync": essence(prepared["decision"]) == essence(sync["decision"]),
                     "audit": prepared["audit"]},
        "never": {"epoch": never["epoch"], "pinned_to": never["pinned_to"], "pointer": never["pointer"],
                  "executed_at_seal": never["executed_at_seal"], "rewards": never["decision"].rewards(),
                  "audit": never["audit"], "audit_ok": never["audit_ok"]},
        "retracted_by_e1": sorted(m for m, r in sync["decision"].rewards().items()
                                  if r == 0.0 and never["decision"].rewards()[m] == 1.0),
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
