# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The record lifecycle, end to end, on synthetic data.

One set of SWE-bench-style trajectories is produced and graded under grader version g1. Then g2 is published:

* demo__calc-1  a label fix (a test that fails in every environment is removed from PASS_TO_PASS): re-parse;
* demo__calc-2  a new eval script that runs one more test, and the labels that list it: re-execute (a patch that
                did not apply fails again without running: the eval script does not change whether it applies);
* demo__text-3  unchanged: reuse;
* demo__text-4  SWE-ABS style hardening: original suite AND a hardened suite (short circuit on original fails);
* demo__text-5  unchanged; every attempt fails.

The example writes, under --out:

* records.jsonl, scores.jsonl          the execution records and every score they received
* eval_g1.json, eval_g2.json           two evaluation manifests over the same record set
* eval_diff.json                       what g2 changed, verdict by verdict
* registry/                            the publication log and version bodies
* seals.jsonl                          the seal decisions (what each step intended to train)
* learner.jsonl, completions.jsonl     a simulated learner's per-record log and step completions
* consumption.jsonl, training.json     consumption records and the training consumption manifest

Run:  python examples/lifecycle/run.py [--out DIR]
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
sys.path.insert(0, HERE)

from agent_ecc import (  # noqa: E402
    Candidate,
    DirectoryRegistry,
    DirectoryStore,
    LearnerLine,
    Member,
    ScoreInput,
    Scorer,
    SealJournal,
    SealSession,
    StepCompletion,
    compare_evaluations,
    consumption_records,
    dump_jsonl,
    evaluation_manifest,
    reproduce_evaluation,
)
from agent_ecc.adapters.swebench import SimpleHarness, SWEBenchAdapter  # noqa: E402
from agent_ecc.export import group_normalized  # noqa: E402
from agent_ecc.scoring import grade_recorded  # noqa: E402

import simworld  # noqa: E402

T = "tests/test_{}.py::test_{}"


def t(module: str, name: str) -> str:
    return T.format(module, name)


TASKS = {
    "demo__calc-1": dict(f2p=[t("calc", "div_zero")], p2p=[t("calc", "add"), t("calc", "mul")],
                         passing={t("calc", "add")}),  # test_mul fails in every environment
    "demo__calc-2": dict(f2p=[t("calc", "round")], p2p=[t("calc", "sub")], passing={t("calc", "sub")}),
    "demo__text-3": dict(f2p=[t("text", "strip")], p2p=[t("text", "join")], passing={t("text", "join")}),
    "demo__text-4": dict(f2p=[t("text", "upper")], p2p=[t("text", "lower")], passing={t("text", "lower")}),
    "demo__text-5": dict(f2p=[t("text", "split")], p2p=[t("text", "join")], passing={t("text", "join")}),
}

PATCHES = {
    "demo__calc-1": ["fix:" + t("calc", "div_zero"), "", "a patch that changes nothing",
                     f"fix:{t('calc', 'div_zero')} break:{t('calc', 'add')}"],
    "demo__calc-2": ["fix:" + t("calc", "round"), f"fix:{t('calc', 'round')} fix:{t('calc', 'round_half')}",
                     "corrupt patch", ""],
    "demo__text-3": ["fix:" + t("text", "strip"), "", "fix:" + t("text", "strip"), "unrelated edit",
                     "fix:" + t("text", "strip"), "unrelated edit", "", "fix:" + t("text", "strip")],
    "demo__text-4": ["fix:" + t("text", "upper"), f"fix:{t('text', 'upper')} fix:{t('text', 'upper_unicode')}", "",
                     f"break:{t('text', 'lower')} fix:{t('text', 'upper')}"],
    "demo__text-5": ["an attempt", "another attempt", "", "a third attempt"],
}


def row(task: str, *, f2p=None, p2p=None, extra_tests=()) -> dict:
    spec = TASKS[task]
    f2p = list(f2p if f2p is not None else spec["f2p"])
    p2p = list(p2p if p2p is not None else spec["p2p"])
    return {"instance_id": task, "repo": "demo/" + task.split("__")[1].split("-")[0], "version": "1.0",
            "image": simworld.IMAGE, "eval_script": simworld.eval_script(sorted(set(f2p + p2p + list(extra_tests)))),
            "log_parser": "parse_log_pytest", "eval_type": "pass_and_fail", "FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=None, help="output directory (default: a new temporary directory)")
    args = ap.parse_args(argv)
    out = args.out or tempfile.mkdtemp(prefix="agent-ecc-lifecycle-")
    os.makedirs(out, exist_ok=True)
    store = DirectoryStore(os.path.join(out, "objects"))
    adapter = SWEBenchAdapter(harness=SimpleHarness())
    backend = simworld.SimulatedEvaluator({task: spec["passing"] for task, spec in TASKS.items()})
    # results are cached by request id: an execution made for evaluation is not repeated for training
    scorer = Scorer([adapter], store, backend, cache={})
    registry = DirectoryRegistry(os.path.join(out, "registry"))

    # ---------------------------------------------------------------- grader versions
    g1 = adapter.version_from_rows([row(task) for task in TASKS], store, name="g1")
    # the broken test is removed from the labels; the eval script still runs it
    base_rows = [row("demo__calc-1", p2p=[t("calc", "add")], extra_tests=[t("calc", "mul")]),
                 row("demo__calc-2", f2p=[t("calc", "round"), t("calc", "round_half")]),
                 row("demo__text-3"), row("demo__text-4"), row("demo__text-5")]
    hardened = [{"instance_id": "demo__text-4", "FAIL_TO_PASS": [t("text", "upper"), t("text", "upper_unicode")],
                 "PASS_TO_PASS": [t("text", "lower")],
                 "eval_script": simworld.eval_script([t("text", "lower"), t("text", "upper"),
                                                      t("text", "upper_unicode")])}]
    g2 = adapter.conjunction_version(base_rows, hardened, store, name="g2")
    registry.publish_grader(g1, source="startup")
    registry.publish_world(simworld.IMAGE)

    # ---------------------------------------------------------------- collect: trajectories graded under g1
    records, recorded = [], {}
    pin = registry.resolve_start()  # every trajectory pins the world and the grader epoch when it starts
    pinned = registry.grader(pin.epoch.version_id)
    for task, patches in PATCHES.items():
        check = pinned.entries[task]["*"]
        for i, patch in enumerate(patches):
            log, applied = simworld.run(patch, check_script(store, check), passing_by_default=TASKS[task]["passing"])
            run = adapter.run(check, store=store, log=log, applied=applied)
            record = adapter.record(store=store, task_id=task, trajectory_id=f"{task}/a{i}", image=pin.world,
                                    patch=patch, runs=[run] if patch.strip() else [])
            records.append(record)
            recorded[record.record_id] = [grade_recorded(adapter, record, check, store,
                                                         grader_version=pinned.version_id)]
    dump_jsonl(os.path.join(out, "records.jsonl"), records)
    all_scores = [s for ss in recorded.values() for s in ss]

    # ---------------------------------------------------------------- evaluate under g1 and g2
    def evaluate(version):
        scored = scorer.score([ScoreInput(r, tuple(recorded[r.record_id])) for r in records], version)
        scores = [s.score for s in scored]
        return evaluation_manifest(records, scores, version), scores

    m1, s1 = evaluate(g1)
    m2, s2 = evaluate(g2)
    eval_executions = backend.calls
    all_scores += s1 + s2
    diff = compare_evaluations(m1, m2)
    reproduced = reproduce_evaluation(m1, records, g1, scorer, prior=recorded)
    write_json(os.path.join(out, "eval_g1.json"), m1.to_dict())
    write_json(os.path.join(out, "eval_g2.json"), m2.to_dict())
    write_json(os.path.join(out, "eval_diff.json"), diff)

    # ---------------------------------------------------------------- train under g2
    registry.publish_grader(g2, source="switch")
    journal = SealJournal(os.path.join(out, "seals.jsonl"))
    by_task: dict[str, list] = {}
    for record in records:
        by_task.setdefault(record.task_id, []).append(record)

    def candidate(task: str, start: int, size: int = 4) -> Candidate:
        members = []
        for record in by_task[task][start:start + size]:
            prior = recorded[record.record_id]
            members.append(Member(member_id=record.trajectory_id, execution=record, prior=tuple(prior),
                                  recorded_reward=1.0 if prior[0].passed else 0.0))
        return Candidate(group_id=f"{task}#{start // size}", members=tuple(members))

    spare = [candidate("demo__text-3", 4)]
    plan = [("step-1", [candidate("demo__calc-1", 0), candidate("demo__calc-2", 0)]),
            ("step-2", [candidate("demo__text-4", 0), candidate("demo__text-5", 0)]),
            ("step-3", [candidate("demo__text-3", 0)])]
    decisions, lines, completions = [], [], []
    for step, (batch_id, initial) in enumerate(plan, start=1):
        session = SealSession(registry, scorer, batch_id=batch_id)  # reads the epoch pointer once

        def backfill(need, considered, remaining):
            offered = [c for c in spare if c.group_id not in considered][:need]
            return offered

        assembly = session.assemble(initial, backfill=backfill)
        decision = session.finish(assembly, journal=journal)
        all_scores += session.scores
        decisions.append(decision)
        # the learner consumes the batch: one line per record, where the loss reads it
        for group in decision.groups:
            rewards = [m.trained_reward for m in group.members]
            for m, adv in zip(group.members, group_normalized(rewards, std=True)):
                lines.append(LearnerLine(batch_id=batch_id, member_id=m.member_id, group_id=group.group_id,
                                         step=step, reward=m.trained_reward, normalized=adv, advantage=adv,
                                         advantage_min=adv, advantage_max=adv, loss_tokens=128,
                                         handoff_loss_tokens=128, identity="verified", reporter="dp0",
                                         microbatch=0))
        if batch_id != "step-3":  # the learner stops before step 3 completes
            completions.append(StepCompletion(batch_id=batch_id, step=step, completed_at=float(step)))
    report = consumption_records(decisions, lines, completions)
    dump_jsonl(os.path.join(out, "scores.jsonl"), all_scores)
    with open(os.path.join(out, "learner.jsonl"), "w") as f:
        f.writelines(json.dumps(line.to_dict()) + "\n" for line in lines)
    with open(os.path.join(out, "completions.jsonl"), "w") as f:
        f.writelines(json.dumps(c.to_dict()) + "\n" for c in completions)
    dump_jsonl(os.path.join(out, "consumption.jsonl"), report.records)
    write_json(os.path.join(out, "training.json"), report.manifest.to_dict())

    summary = {
        "out": out,
        "records": len(records),
        "g1": {"pass": m1.counts["pass"], "fail": m1.counts["fail"], "undeterminable": m1.undeterminable,
               "by_method": dict(m1.counts["by_method"])},
        "g2": {"pass": m2.counts["pass"], "fail": m2.counts["fail"], "undeterminable": m2.undeterminable,
               "by_method": dict(m2.counts["by_method"])},
        "flips": {"fail_to_pass": diff["fail_to_pass"], "pass_to_fail": diff["pass_to_fail"]},
        "g1_reproduced": reproduced["reproduced"],
        "executions": {"evaluation": eval_executions, "training": backend.calls - eval_executions},
        "seals": [{"batch": d.batch_id, "epoch": d.epoch.name, "groups": [g.group_id for g in d.groups],
                   "excluded": [e.group.group_id for e in d.excluded], "statuses": dict(d.stats["statuses"])}
                  for d in decisions],
        "training": {"ok": report.manifest.ok, "steps": [s["batch_id"] for s in report.manifest.steps],
                     "records_consumed": report.manifest.records_consumed,
                     "labels": dict(report.manifest.labels),
                     "incomplete": list(report.manifest.incomplete_batches)},
    }
    print(json.dumps(summary, indent=2))
    return summary


def check_script(store, check) -> str:
    from agent_ecc.store import get_text

    return get_text(store, str(check.execution["eval_script"]))


def write_json(path: str, value) -> None:
    with open(path, "w") as f:
        json.dump(value, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
