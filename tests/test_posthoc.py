# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The post hoc example's outcome-digest check: a sidecar whose digest is missing or does not bind its trajectory to
the patch its evidence names is unrepairable, and its evidence is never read."""

import hashlib
import json
import os

import pytest
from test_example import load_example
from test_swebench import F2P, IMAGE, P2P, eval_script, log

from agent_ecc.adapters import swebench
from agent_ecc.canonical import digest_bytes
from agent_ecc.store import DirectoryStore

TASK = "pkg__pkg-1"
posthoc = load_example("posthoc")


def _capture_digest(sc: dict) -> str:
    """The outcome digest as the trainer's capture computes it (written independently of the example's)."""
    patch = sc["evidence"]["agent_patch"]["digest"]
    value = {"task_id": sc["task_id"], "trajectory_id": sc["trajectory_id"], "attempt_id": sc["trajectory_id"],
             "status": sc["execution_status"], "artifacts": {"agent_patch": patch}}
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _put(root, data: bytes) -> dict:
    digest = digest_bytes(data)
    path = posthoc.artifact_path(root, digest)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return {"digest": digest, "size": len(data)}


def _sidecar(root, i, patch, statuses, spec):
    text = log(statuses)
    resolved = all(s == "PASSED" for s in statuses.values())
    sc = {"schema": posthoc.CONJUNCTION_SCHEMA, "task_id": TASK, "trajectory_id": f"request:{i}:attempt:0",
          "execution_status": "completed", "task": {"repo": "pkg/pkg", "version": "1.0"},
          "stamps": {"image": IMAGE, "specs": [spec]},
          "evidence": {"agent_patch": _put(root, patch.encode()),
                       "suites": [{"name": "s0", "class": "pass" if resolved else "fail",
                                   "test_log": _put(root, text.encode()), "has_output_markers": True}]},
          "verdict": {"resolved": resolved, "valid": True, "reason": None if resolved else "tests_failed"},
          "graded_epoch": {"epoch": 0, "name": "e0", "covered": True}}
    sc["outcome_digest"] = _capture_digest(sc)
    return sc


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A run directory with one epoch and two graded attempts: 0 resolved, 1 failed."""
    adapter = swebench.SWEBenchAdapter  # the official harness is not needed: the simple one parses these logs
    monkeypatch.setattr(swebench, "SWEBenchAdapter", lambda **kw: adapter(harness=swebench.SimpleHarness(), **kw))
    root = str(tmp_path / "run")
    store = DirectoryStore(os.path.join(root, "ecc", "store"))
    labels = store.put(json.dumps({"FAIL_TO_PASS": F2P, "PASS_TO_PASS": P2P}).encode())
    spec = {"name": "s0", "labels": labels, "eval_script": store.put(eval_script().encode()),
            "log_parser": "parse_log_pytest", "eval_type": "pass_and_fail", "image": IMAGE}
    check = swebench.SWEBenchAdapter().check(store=store, instance_id=TASK, repo="pkg/pkg", version="1.0", image=IMAGE,
                                             eval_script=eval_script(), log_parser="parse_log_pytest",
                                             eval_type="pass_and_fail", fail_to_pass=F2P, pass_to_pass=P2P, name="s0",
                                             rule=swebench.verdict_rule("logical"))
    spec.update(apply_cmds=check.execution["apply_cmds"], parser=check.grading["parser"],
                grading_code=check.grading["grading_code"])  # the identities the trainer stamps
    os.makedirs(os.path.join(root, "reval-meta"))
    with open(os.path.join(root, "ecc", "mode"), "w") as f:
        f.write("repair\n")
    with open(os.path.join(root, "ecc", "publish.log"), "w") as f:
        f.write(json.dumps({"epoch": 0, "t": 1.0, "source": "startup"}) + "\n")
    os.makedirs(os.path.join(root, "ecc", "epochs"))
    with open(os.path.join(root, "ecc", "epochs", "0.json"), "w") as f:
        json.dump({"kind": "swe_bench", "epoch": 0, "name": "e0", "store": os.path.join(root, "ecc", "store"),
                   "swe": {TASK: {"specs": [spec]}}}, f)
    for i, (patch, f2p) in enumerate([("diff a", "PASSED"), ("diff b", "FAILED")]):
        _write(root, i, _sidecar(root, i, patch, {F2P[0]: f2p, P2P[0]: "PASSED"}, spec))
    return root


def _path(root, i):
    return os.path.join(root, "reval-meta", f"request_{i}_attempt_0.json")


def _write(root, i, sc):
    with open(_path(root, i), "w") as f:
        json.dump(sc, f)


def _read(root, i):
    with open(_path(root, i)) as f:
        return json.load(f)


def _main(root, tmp_path, *extra):
    summary = posthoc.main(["--run", root, "--out", str(tmp_path / "out"), "--threads", "2", *extra])
    with open(tmp_path / "out" / "eval_e0.json") as f:
        outcomes = {r["trajectory_id"]: r["outcome"] for r in json.load(f)["rows"]}
    with open(tmp_path / "out" / "records.jsonl") as f:
        records = {r["trajectory_id"]: r for r in map(json.loads, f)}
    return summary, outcomes, records


@pytest.fixture(autouse=True)
def _quiet(capsys):
    yield
    capsys.readouterr()


def test_records_with_their_own_evidence_are_scored(run, tmp_path):
    summary, outcomes, _ = _main(run, tmp_path)
    assert outcomes == {"0/a0": "pass", "1/a0": "fail"}
    assert summary["records"]["outcome_digest"] == {"checked": True, "unrepairable": 0}


def test_a_patch_and_log_swapped_with_another_record_fail_closed(run, tmp_path):
    mine, theirs = _read(run, 0), _read(run, 1)
    mine["evidence"] = theirs["evidence"]  # seeded bug: attempt 0's sidecar names attempt 1's evidence
    _write(run, 0, mine)

    summary, outcomes, records = _main(run, tmp_path)

    assert outcomes == {"0/a0": "undeterminable", "1/a0": "fail"}  # never attempt 1's fail
    assert records["0/a0"]["artifacts"] == [] and records["0/a0"]["runs"] == []
    assert summary["records"]["outcome_digest"] == {"checked": True, "unrepairable": 1, "mismatch": 1}
    assert summary["reproduction"]["first_version"]["against_run_log"] == {"unrepairable": 1, "equal": 1}


def test_a_missing_outcome_digest_fails_closed_and_the_check_can_be_turned_off(run, tmp_path):
    sc = _read(run, 1)
    sc.pop("outcome_digest")
    _write(run, 1, sc)

    summary, outcomes, _ = _main(run, tmp_path)
    assert outcomes == {"0/a0": "pass", "1/a0": "undeterminable"}
    assert summary["records"]["outcome_digest"] == {"checked": True, "unrepairable": 1, "missing": 1}

    summary, outcomes, _ = _main(run, tmp_path / "off", "--no-outcome-digest")
    assert outcomes == {"0/a0": "pass", "1/a0": "fail"}
    assert summary["records"]["outcome_digest"] == {"checked": False, "unrepairable": 0}


def test_the_outcome_digest_check_reads_the_sidecar_only():
    sc = {"task_id": TASK, "trajectory_id": "request:3:attempt:0", "execution_status": "completed",
          "evidence": {"agent_patch": {"uri": "/nowhere", "digest": "sha256:aa"}}}
    sc["outcome_digest"] = _capture_digest(sc)
    assert posthoc.outcome_digest_check(sc) == "ok"
    for change in ({"trajectory_id": "request:4:attempt:0"}, {"execution_status": "truncated"},
                   {"evidence": {"agent_patch": {"digest": "sha256:bb"}}}, {"evidence": {}}):
        assert posthoc.outcome_digest_check({**sc, **change}) == "mismatch"
    assert posthoc.outcome_digest_check({**sc, "outcome_digest": ""}) == "missing"
    assert posthoc.outcome_digest_check(None) == "missing"
