# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The post hoc example's integrity checks on the trainer's sidecars. A sidecar whose outcome digest is missing or does
not bind its trajectory to the patch its evidence names is unrepairable, and so is one whose evidence binding (the
digest of the whole sidecar) is missing or no longer matches: its evidence is never read. Both checks cover every
sidecar."""

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


def _binding_digest(sc: dict) -> str:
    """The trainer's evidence binding, written independently of the example's: the sidecar but its volatile fields,
    as canonical JSON of {"v": 1, "sidecar": ...}."""
    body = {k: v for k, v in sc.items() if k not in ("events", "timing", "evidence_binding")}
    payload = json.dumps({"v": 1, "sidecar": json.loads(json.dumps(body, sort_keys=True, default=str))},
                         sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _rebind(sc: dict) -> dict:
    sc["evidence_binding"] = {"version": 1, "digest": _binding_digest(sc)}
    return sc


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
    sc["task"]["base_commit"] = "0123abcd"
    sc["events"], sc["timing"] = {"persist_t": 1.0}, {"grade_s": 2.0}  # written after the verdict: not bound
    return _rebind(sc)


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
    assert summary["records"]["evidence_binding"] == {"checked": True, "unrepairable": 0}


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

    _rebind(sc)  # a run whose sidecars record a binding but no outcome digest
    _write(run, 1, sc)
    summary, outcomes, _ = _main(run, tmp_path / "off", "--no-outcome-digest")
    assert outcomes == {"0/a0": "pass", "1/a0": "fail"}
    assert summary["records"]["outcome_digest"] == {"checked": False, "unrepairable": 0}
    assert summary["records"]["evidence_binding"] == {"checked": True, "unrepairable": 0}


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


# ---------------------------------------------------------------------------------------------- the evidence binding


def _mutate(root, i, change):
    sc = _read(root, i)
    change(sc)
    _write(root, i, sc)  # the stored binding is left as it was: a sidecar edited on disk after it was written


def _flip_suite_class(sc):
    suite = sc["evidence"]["suites"][0]
    suite["class"] = "fail" if suite["class"] == "pass" else "pass"


CORRUPTIONS = {
    "suite class flipped": _flip_suite_class,
    "execution configuration changed": lambda sc: sc["stamps"]["specs"][0].update(
        {"apply_cmds": "sha256:" + "ab" * 32}),
    "base commit marker": lambda sc: sc["task"].update({"base_commit": sc["task"]["base_commit"] + "~CORRUPT"}),
    "verdict changed": lambda sc: sc["verdict"].update({"resolved": not sc["verdict"]["resolved"]}),
    "log reference swapped": lambda sc: sc["evidence"]["suites"][0].update(
        {"test_log": {"digest": "sha256:" + "cd" * 32, "size": 1}}),
    "binding removed": lambda sc: sc.pop("evidence_binding"),
    "binding of another version": lambda sc: sc["evidence_binding"].update({"version": 2}),
}


@pytest.mark.parametrize("attempt", [0, 1])  # a resolved attempt and a failed one: every sidecar is checked
@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_a_sidecar_edited_after_it_was_written_fails_closed(run, tmp_path, name, attempt):
    """None of these touches what the outcome digest binds (the trajectory, the status and the patch's digest)."""
    _mutate(run, attempt, CORRUPTIONS[name])

    summary, outcomes, records = _main(run, tmp_path)

    kept = f"{1 - attempt}/a0"
    assert outcomes[f"{attempt}/a0"] == "undeterminable" and outcomes[kept] == ("pass", "fail")[1 - attempt]
    assert records[f"{attempt}/a0"]["artifacts"] == [] and records[f"{attempt}/a0"]["runs"] == []  # never read
    assert records[kept]["artifacts"] and records[kept]["runs"]
    assert summary["records"]["outcome_digest"] == {"checked": True, "unrepairable": 0}  # the digest never saw it
    want = {"missing": 1} if name.startswith("binding") else {"mismatch": 1}
    assert summary["records"]["evidence_binding"] == {"checked": True, "unrepairable": 1, **want}


def test_a_sidecar_that_fails_both_checks_is_rejected_for_its_outcome_digest(run, tmp_path):
    def swap_evidence(sc):
        sc["evidence"] = _read(run, 1)["evidence"]  # another record's patch and log, binding untouched

    _mutate(run, 0, swap_evidence)
    summary, outcomes, _ = _main(run, tmp_path)
    assert outcomes == {"0/a0": "undeterminable", "1/a0": "fail"}
    assert summary["records"]["outcome_digest"] == {"checked": True, "unrepairable": 1, "mismatch": 1}
    assert summary["records"]["evidence_binding"] == {"checked": True, "unrepairable": 1, "mismatch": 1}


def test_volatile_fields_do_not_move_the_binding_and_old_runs_can_skip_it(run, tmp_path):
    _mutate(run, 0, lambda sc: sc.update({"events": {"persist_t": 99.0}, "timing": {"grade_s": 9.0}}))
    summary, outcomes, _ = _main(run, tmp_path)
    assert outcomes == {"0/a0": "pass", "1/a0": "fail"}
    assert summary["records"]["evidence_binding"] == {"checked": True, "unrepairable": 0}

    _mutate(run, 0, lambda sc: sc.pop("evidence_binding"))  # a run from before the trainer recorded bindings
    summary, outcomes, _ = _main(run, tmp_path / "off", "--no-evidence-binding")
    assert outcomes == {"0/a0": "pass", "1/a0": "fail"}
    assert summary["records"]["evidence_binding"] == {"checked": False, "unrepairable": 0}


def test_the_binding_digest_equals_the_trainers():
    """A fixed sidecar and the digest the trainer's own function gives it (a float, non-ASCII text, volatile fields)."""
    sc = {"schema": "swe-bench-sidecar/2", "task_id": TASK, "trajectory_id": "request:7:attempt:0",
          "stamps": {"image": "registry/img@sha256:ab", "specs": [{"name": "s0", "labels": "sha256:cd"}]},
          "task": {"repo": "pkg/pkg", "version": "1.0", "base_commit": "0123abcd"},
          "evidence": {"agent_patch": {"digest": "sha256:ee", "size": 12}, "suites": [{"name": "s0", "class": "pass"}]},
          "verdict": {"resolved": True, "valid": True, "reason": None, "ratio": 0.5, "note": "caf\u00e9"},
          "events": {"persist_t": 1.0}, "timing": {"grade_s": 2.0}}
    golden = "sha256:510785f3768ca3a6a827d5667edf573750ec9cf2852482e6cf4d43a950055570"
    assert posthoc.evidence_binding_digest(sc) == golden == _binding_digest(sc)
    assert posthoc.evidence_binding_digest({**sc, "events": {"persist_t": 99.0}}) == golden
    sc["evidence_binding"] = {"version": 1, "digest": golden}
    assert posthoc.evidence_binding_check(sc) == "ok"
    assert posthoc.evidence_binding_check({**sc, "task": {**sc["task"], "base_commit": "0123abce"}}) == "mismatch"
    assert posthoc.evidence_binding_check({**sc, "evidence_binding": {"version": 2, "digest": golden}}) == "missing"
    assert posthoc.evidence_binding_check({k: v for k, v in sc.items() if k != "evidence_binding"}) == "missing"
    assert posthoc.evidence_binding_check(None) == "missing"
