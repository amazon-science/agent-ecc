# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The examples run end to end and tell the stories their docstrings tell."""

import importlib.util
import json
import os
import sys

from agent_ecc.export import EvaluationManifest
from agent_ecc.records import ConsumptionRecord, ExecutionRecord, ScoreRecord, SealDecision, load_jsonl
from agent_ecc.versions import DirectoryRegistry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_example(name="lifecycle"):
    path = os.path.join(ROOT, "examples", name, "run.py")
    sys.path.insert(0, os.path.dirname(path))
    try:
        spec = importlib.util.spec_from_file_location(f"{name}_run", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(os.path.dirname(path))


def test_lifecycle_example(tmp_path, capsys):
    out = str(tmp_path / "out")
    summary = load_example().main(["--out", out])
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(summary))
    assert summary["g1"]["pass"] == 8 and summary["g2"]["pass"] == 7
    assert summary["g1"]["by_method"] == {"recorded": 24}
    # under g2: 14 verdicts reused (one a patch that did not apply, carried over to the new eval script without
    # running it), 4 suites re-executed, 4 label fixes re-parsed and 2 empty patches decided by their evidence alone
    # under suites they were never graded under (one of them the hardened suite)
    assert summary["g2"]["by_method"] == {"recorded": 14, "reexec": 4, "reparse": 6}
    assert summary["flips"] == {"fail_to_pass": 1, "pass_to_fail": 2} and summary["g1_reproduced"]
    assert summary["executions"] == {"evaluation": 4, "training": 0}
    assert [s["excluded"] for s in summary["seals"]] == [[], ["demo__text-5#0"], []]
    assert summary["training"]["ok"] and summary["training"]["steps"] == ["step-1", "step-2"]
    assert summary["training"]["incomplete"] == ["step-3"] and summary["training"]["records_consumed"] == 16

    # every artifact the example wrote loads back and verifies its identity
    records = load_jsonl(os.path.join(out, "records.jsonl"), ExecutionRecord)
    scores = load_jsonl(os.path.join(out, "scores.jsonl"), ScoreRecord)
    seals = load_jsonl(os.path.join(out, "seals.jsonl"), SealDecision)
    consumed = load_jsonl(os.path.join(out, "consumption.jsonl"), ConsumptionRecord)
    assert len(records) == 24 and len(seals) == 3 and len(consumed) == 2
    assert {s.execution_id for s in scores} == {r.record_id for r in records}
    m1 = EvaluationManifest.from_dict(json.load(open(os.path.join(out, "eval_g1.json"))))
    m2 = EvaluationManifest.from_dict(json.load(open(os.path.join(out, "eval_g2.json"))))
    assert m1.record_set == m2.record_set and m1.grader_version != m2.grader_version
    registry = DirectoryRegistry(os.path.join(out, "registry"))
    assert [p.name for p in registry.publications("grader")] == ["g1", "g2"]
    assert {d.epoch.version_id for d in seals} == {m2.grader_version}
    training = json.load(open(os.path.join(out, "training.json")))
    assert training["ok"] and training["consumption_ids"] == [c.consumption_id for c in consumed]


def test_prepared_example(tmp_path, capsys):
    out = str(tmp_path / "out")
    summary = load_example("prepared").main(["--out", out])
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(summary))
    assert summary["members"] == 12 and summary["nothing_prepared_under_e0"]
    # synchronous ECC executes the hardened suites at the seal; with preparation they ran before it, once each, and
    # the seal decided exactly the same
    assert summary["sync"]["executed_at_seal"] == 4 and summary["prepared"]["executed_at_seal"] == 0
    assert summary["prepared"]["background_executions"] == 4 and summary["prepared"]["answers"] == {
        "prepared:ready": 4}
    assert summary["prepared"]["same_decisions_as_sync"]
    # the never arm trains e0's verdicts, the three passes e1 retracts included, and records the pin
    assert summary["never"]["epoch"] == "e0" and summary["never"]["pointer"] == "e1"
    assert summary["never"]["pinned_to"] == "e0" and summary["never"]["executed_at_seal"] == 0
    assert summary["retracted_by_e1"] == ["demo__calc-1/a0", "demo__text-2/a0", "demo__text-2/a1"]
    assert summary["sync"]["audit"] == summary["prepared"]["audit"] == ["ok"]
    assert summary["never"]["audit"] == ["pinned"] and summary["never"]["audit_ok"]
    seals = load_jsonl(os.path.join(out, "seals-never.jsonl"), SealDecision)
    assert seals[0].target_pin.published.name == "e1"
    events = [json.loads(line) for line in open(os.path.join(out, "prepared.jsonl"), encoding="utf-8")]
    assert {e["kind"] for e in events} >= {"start", "target", "plan", "job", "use", "deciding", "sealed"}
