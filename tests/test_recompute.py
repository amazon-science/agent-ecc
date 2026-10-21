# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backends: replay, recording, the HTTP reference service, and local Docker (with a stand-in harness)."""

import dataclasses
import json
import os
import threading
import types

import pytest

from agent_ecc.adapters.swebench import SimpleHarness, SWEBenchAdapter
from agent_ecc.recompute.base import (
    CallableBackend,
    ExecRequest,
    ExecResult,
    RecordedBackend,
    RecordingBackend,
    infra_result,
)
from agent_ecc.recompute.docker import LocalDockerBackend
from agent_ecc.recompute.service import HTTPBackend, ReferenceService, make_server
from agent_ecc.scoring import ScoreInput, Scorer, grade_recorded
from agent_ecc.store import MemoryStore, get_text
from agent_ecc.versions import GraderVersion

from toy import ToyAdapter, ToyBackend, check


def toy_world():
    store, adapter = MemoryStore(), ToyAdapter()
    c1 = check("t", "plain", "ab", store)
    items = []
    for i, answer in enumerate(["ab", "ba", "AB"]):
        rec = adapter.record(store, task="t", trajectory=f"x{i}", answer=answer,
                             runs=[adapter.run(store, c1, answer)])
        items.append(ScoreInput(rec, grade_recorded(adapter, rec, c1, store)))
    target = GraderVersion(name="g2", entries={"t": check("t", "up", "AB", store)})
    return store, adapter, items, target


def test_exec_request_identity():
    a = ExecRequest(family="toy", kind="toy/run", task_id="t", execution={"runner": "up"},
                    inputs={"answer": "sha256:" + "1" * 64}, params={"n": 1})
    b = ExecRequest.from_dict(json.loads(json.dumps(a.to_dict())))
    assert a.request_id == b.request_id
    d = a.to_dict()
    d["params"] = {"n": 2}
    with pytest.raises(ValueError):
        ExecRequest.from_dict(d)
    r = infra_result(a, "x" * 1000)
    assert r.status == "error" and len(r.error) == 500 and ExecResult.from_dict(r.to_dict()) == r
    with pytest.raises(ValueError):
        ExecResult(request_id=a.request_id, status="maybe")


def test_recording_then_replay_reproduces_every_verdict(tmp_path):
    store, adapter, items, target = toy_world()
    live = ToyBackend()
    recorder = RecordingBackend(live, tmp_path / "results.jsonl", tmp_path / "requests.jsonl")
    first = Scorer([adapter], store, recorder).score(items, target)
    replay = RecordedBackend.from_jsonl(tmp_path / "results.jsonl")
    again = Scorer([adapter], store, replay).score(items, target)
    assert [a.score.outcome for a in first] == [b.score.outcome for b in again]
    assert all(a.score.same_verdict(b.score) for a, b in zip(first, again))
    assert len(replay.served) == 3 and replay.missing == []
    assert len(open(tmp_path / "requests.jsonl").read().splitlines()) == 3


def test_a_replay_never_invents_an_execution():
    store, adapter, items, target = toy_world()
    replay = RecordedBackend()
    out = Scorer([adapter], store, replay, retries=0).score(items, target)
    assert all(o.score.reason == "infra_error:not_recorded" for o in out) and len(replay.missing) == 3
    live = ToyBackend()
    results = live.execute([adapter.reexec_request(items[0].execution, target.entries["t"]["*"], store)], store)
    empty_store_replay = RecordedBackend(results)
    (r,) = empty_store_replay.execute([adapter.reexec_request(items[0].execution, target.entries["t"]["*"], store)],
                                      MemoryStore())
    assert r.status == "error" and "not in store" in r.error


def test_callable_backend_turns_exceptions_into_infrastructure_errors():
    store, adapter, items, target = toy_world()
    request = adapter.reexec_request(items[0].execution, target.entries["t"]["*"], store)

    def boom(req, st):
        raise OSError("no sandbox")

    (r,) = CallableBackend(boom).execute([request], store)
    assert r.status == "error" and "no sandbox" in r.error
    (r,) = CallableBackend(lambda req, st: infra_result(
        ExecRequest(family="x", kind="k", task_id="t", execution={}, inputs={}), "x")).execute([request], store)
    assert "another request" in r.error


@pytest.fixture
def service():
    server_store = MemoryStore()
    backend = ToyBackend()
    svc = ReferenceService(backend, server_store, workers=2, name="toy")
    server = make_server(svc, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield svc, backend, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_http_service_round_trip(service):
    svc, remote_backend, url = service
    store, adapter, items, target = toy_world()
    client = HTTPBackend(url, timeout_s=30, batch_size=2)
    assert client.health()["backend"] == "toy"
    remote = Scorer([adapter], store, client).score(items, target)
    local = Scorer([adapter], store, ToyBackend()).score(items, target)
    assert all(a.score.same_verdict(b.score) for a, b in zip(remote, local))
    assert len(remote_backend.executed) == 3 and svc.served == 3
    for o in remote:  # every returned output object was verified and stored locally
        for run in o.score.new_runs():
            assert all(store.contains(ref.digest) for ref in run.outputs)


def test_http_service_failures_are_infrastructure_errors(service):
    _, _, url = service
    store, adapter, items, target = toy_world()
    request = adapter.reexec_request(items[0].execution, target.entries["t"]["*"], store)
    (r,) = HTTPBackend("http://127.0.0.1:9", timeout_s=2).execute([request], store)
    assert r.status == "error" and r.error.startswith("service:")
    # the service cannot read inputs that were not uploaded: its backend reports an infrastructure error
    (r,) = HTTPBackend(url, send_objects=False).execute([request], store)
    assert r.status == "error"


# ---------------------------------------------------------------------------------------------- local Docker


class FakeHarness:
    """Stands in for the official run_instance: writes the logs it would write, without containers."""

    def __init__(self, root, behavior):
        self.root, self.behavior, self.calls = root, behavior, []

    def make_test_spec(self, row):
        return types.SimpleNamespace(instance_id=row["instance_id"], eval_script=row["eval_script"], row=row)

    def run_instance(self, spec, pred, client, run_id, timeout=None):
        self.calls.append((spec, pred, run_id, timeout))
        d = os.path.join(self.root, run_id, pred["model_name_or_path"], spec.instance_id)
        os.makedirs(d, exist_ok=True)
        if self.behavior == "ok":
            with open(os.path.join(d, "test_output.txt"), "w") as f:
                f.write(f"ran {pred['model_patch']} with {spec.eval_script}")
        elif self.behavior == "no_apply":
            with open(os.path.join(d, "run_instance.log"), "w") as f:
                f.write(">>>>> Patch Apply Failed:\nerror")
        elif self.behavior == "crash":
            raise RuntimeError("docker daemon unreachable")


def swe_request(store):
    adapter = SWEBenchAdapter(harness=SimpleHarness())
    c = adapter.check(store=store, instance_id="pkg__pkg-1", repo="pkg/pkg", version="1.0",
                      image="example.org/img@sha256:" + "0" * 64, eval_script="#!/bin/bash\necho run\n",
                      log_parser="parse_log_pytest", eval_type="pass_and_fail", fail_to_pass=["a"], pass_to_pass=[])
    rec = adapter.record(store=store, task_id="pkg__pkg-1", trajectory_id="t", image="img", patch="diff --git a b")
    return adapter.reexec_request(rec, c, store)


@pytest.mark.parametrize("behavior", ["ok", "no_apply", "silent", "crash"])
def test_local_docker_backend_maps_harness_outcomes(tmp_path, behavior):
    store = MemoryStore()
    fake = FakeHarness(str(tmp_path), behavior)
    backend = LocalDockerBackend(workers=2, timeout_s=60, log_root=str(tmp_path), client=object(),
                                 run_instance=fake.run_instance, make_test_spec=fake.make_test_spec)
    request = swe_request(store)
    (result,) = backend.execute([request], store)
    if behavior == "ok":
        assert result.status == "ok" and result.facts["applied"] is True
        text = get_text(store, result.outputs[0].digest)
        assert "diff --git a b" in text and "echo run" in text and result.cost.executions == 1
        spec, pred, run_id, timeout = fake.calls[0]
        assert spec.row["log_parser"] == "parse_log_pytest" and timeout == 60
        assert not os.path.exists(os.path.join(str(tmp_path), run_id))  # per-request logs removed
    elif behavior == "no_apply":
        assert result.status == "ok" and result.outputs == () and result.facts["applied"] is False
    else:
        assert result.status == "error"
    assert backend.execute([], store) == []


def _timeout_world(tmp_path, log):
    """A record graded under e0 and a version e2 whose new eval script re-executes it through LocalDockerBackend, whose
    stand-in run_instance writes ``log`` as the test output."""

    def run_instance(spec, pred, client, run_id, timeout=None, skip_patch=False):
        d = os.path.join(str(tmp_path), run_id, pred["model_name_or_path"], spec.instance_id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "test_output.txt"), "w") as f:
            f.write(log)

    store = MemoryStore()
    adapter = SWEBenchAdapter(harness=SimpleHarness())
    row = {"instance_id": "pkg__pkg-1", "repo": "pkg/pkg", "version": "1.0", "image": "img",
           "eval_script": "#!/bin/bash\n", "log_parser": "parse_log_pytest", "eval_type": "pass_and_fail",
           "FAIL_TO_PASS": ["t::fix"], "PASS_TO_PASS": ["t::keep"]}
    e0 = adapter.version_from_rows([row], store, name="e0")
    e2 = adapter.version_from_rows([{**row, "eval_script": "#!/bin/bash\n# v2\n"}], store, name="e2")
    check0 = e0.entries["pkg__pkg-1"]["*"]
    record = adapter.record(store=store, task_id="pkg__pkg-1", trajectory_id="t", image="img", patch="diff fix",
                            runs=[adapter.run(check0, store=store, log=None, error="recorded run lost")])
    backend = LocalDockerBackend(log_root=str(tmp_path), client=object(), run_instance=run_instance,
                                 make_test_spec=FakeHarness(str(tmp_path), "ok").make_test_spec)
    (result,) = backend.execute([adapter.reexec_request(record, e2.entries["pkg__pkg-1"]["*"], store)], store)
    scored = Scorer([adapter], store, backend, retries=0).score([ScoreInput(record)], e2)[0]
    return result, scored


def test_only_the_harness_timeout_marker_at_the_end_of_a_log_is_a_timeout(tmp_path):
    """The words of the harness's timeout message inside a complete log (a test printed them) are no timeout: the log
    is read, and it holds a logical verdict. The official runner's own marker at the end of the log is a timeout: no
    verdict under the logical rule."""
    from agent_ecc.adapters.swebench import END_TEST_OUTPUT, START_TEST_OUTPUT, TEST_EXIT_CODE

    complete = (f"{START_TEST_OUTPUT}\nPASSED t::fix\nPASSED t::keep\n"
                "captured stdout: Timeout error: 30 seconds exceeded, retrying the request\n"
                f"===== 2 passed =====\n{END_TEST_OUTPUT}\n{TEST_EXIT_CODE}: 0\n")
    result, scored = _timeout_world(tmp_path / "mid", complete)
    assert result.status == "ok" and result.facts["timed_out"] is False
    assert scored.action == "reexec" and scored.score.outcome.value == "pass"
    killed = f"{START_TEST_OUTPUT}\nPASSED t::fix\n\n\nTimeout error: 1800 seconds exceeded."
    result, scored = _timeout_world(tmp_path / "end", killed)
    assert result.status == "ok" and result.facts["timed_out"] is True
    assert scored.score.outcome.value == "undeterminable" and scored.score.reason == "timeout"


def test_an_empty_patch_runs_the_tests_on_the_unmodified_tree(tmp_path):
    """A hardened suite whose patch filter removed every diff: the eval script runs without a patch."""
    store = MemoryStore()
    calls = []

    def run_instance(spec, pred, client, run_id, timeout=None, skip_patch=False):
        calls.append(skip_patch)
        d = os.path.join(str(tmp_path), run_id, pred["model_name_or_path"], spec.instance_id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "test_output.txt"), "w") as f:
            f.write("ran on the base tree")

    fake = FakeHarness(str(tmp_path), "ok")
    backend = LocalDockerBackend(log_root=str(tmp_path), client=object(), run_instance=run_instance,
                                 make_test_spec=fake.make_test_spec)
    request = swe_request(store)
    empty = dataclasses.replace(request, inputs={**request.inputs, "patch": store.put(b"")})
    (result,) = backend.execute([empty], store)
    assert result.status == "ok" and result.facts["applied"] is True and calls == [True]
    (result,) = backend.execute([request], store)
    assert result.status == "ok" and calls == [True, False]
    # a runner that cannot skip the patch cannot run the suite: an infrastructure error, never a verdict
    old = LocalDockerBackend(log_root=str(tmp_path), client=object(), run_instance=fake.run_instance,
                             make_test_spec=fake.make_test_spec)
    (result,) = old.execute([empty], store)
    assert result.status == "error" and "empty patch" in result.error


def test_local_docker_backend_without_a_harness(monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def no_swebench(name, *args, **kwargs):
        if name.startswith("swebench"):
            raise ImportError("no swebench here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_swebench)
    store = MemoryStore()
    request = swe_request(store)
    (result,) = LocalDockerBackend(log_root=str(tmp_path)).execute([request], store)
    assert result.status == "error" and "harness unavailable" in result.error
