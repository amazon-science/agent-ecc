# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-execute SWE-bench requests in local Docker containers with the official harness (extra ``agent-ecc[swebench]``).

Each request becomes one call of the official ``run_instance``: a fresh container from the request's image, the
agent's patch applied with the official fallback commands, the eval script run, the raw test log captured. The
backend returns the raw log (never a verdict); the SWE-bench adapter grades it exactly as it grades a recorded log.
The result's ``timed_out`` fact is set only by the harness's timeout marker at the end of the log, as the adapter's
own log reading has it.
A request whose patch is empty (a hardened suite's patch filter removed every diff) runs the eval script on the
unmodified tree (``run_instance(..., skip_patch=True)``).
"""

from __future__ import annotations

import inspect
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..adapters.swebench import _TRAILING_TIMEOUT_RE
from ..records import RUN_OK, Cost, EvidenceRef
from ..store import ContentStore, get_text
from .base import ExecRequest, ExecResult, infra_result

__all__ = ["LocalDockerBackend"]

REQUEST_KIND = "swebench/eval"
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"


class LocalDockerBackend:
    """Runs re-execution requests with the official ``swebench.harness.run_evaluation.run_instance``.

    ``workers`` bounds concurrent containers; ``timeout_s`` is the eval-script timeout; ``log_root`` is where the
    harness writes its per-instance logs (a temporary directory by default; per-request directories are removed after
    their log is stored unless ``keep_logs``). ``run_instance``, ``make_test_spec`` and ``client`` can be injected
    (tests, or a harness installed elsewhere)."""

    def __init__(self, *, workers: int = 4, timeout_s: float = 1800.0, log_root: str | None = None,
                 keep_logs: bool = False, client: Any = None, run_instance: Callable | None = None,
                 make_test_spec: Callable | None = None, run_id_prefix: str = "agent-ecc") -> None:
        self.workers = max(1, int(workers))
        self.timeout_s = float(timeout_s)
        self.keep_logs = keep_logs
        self.run_id_prefix = run_id_prefix
        self._client = client
        self._run_instance = run_instance
        self._make_test_spec = make_test_spec
        self._log_root = Path(log_root) if log_root else None
        self._lock = threading.Lock()

    def _harness(self):
        with self._lock:
            if self._run_instance is None or self._make_test_spec is None:
                from swebench.harness import run_evaluation  # the extra agent-ecc[swebench]
                from swebench.harness.utils import make_test_spec

                if self._log_root is None:
                    import tempfile

                    self._log_root = Path(tempfile.mkdtemp(prefix="agent-ecc-swebench-"))
                run_evaluation.RUN_EVALUATION_LOG_DIR = self._log_root  # where run_instance writes its logs
                self._run_instance = self._run_instance or run_evaluation.run_instance
                self._make_test_spec = self._make_test_spec or make_test_spec
                if self._client is None:
                    make_client = getattr(run_evaluation, "_docker_client", None)
                    if make_client is None:
                        import docker

                        make_client = docker.from_env
                    self._client = make_client()
            if self._log_root is None:
                import tempfile

                self._log_root = Path(tempfile.mkdtemp(prefix="agent-ecc-swebench-"))
            return self._run_instance, self._make_test_spec, self._client

    def execute(self, requests: Sequence[ExecRequest], store: ContentStore) -> list[ExecResult]:
        if not requests:
            return []
        try:
            harness = self._harness()
        except Exception as error:  # noqa: BLE001 - no harness: every request is an infrastructure error
            return [infra_result(r, f"harness unavailable: {type(error).__name__}: {error}") for r in requests]
        with ThreadPoolExecutor(max_workers=min(self.workers, len(requests))) as pool:
            return list(pool.map(lambda r: self._one(r, store, harness), requests))

    def _one(self, request: ExecRequest, store: ContentStore, harness) -> ExecResult:
        run_instance, make_test_spec, client = harness
        if request.kind != REQUEST_KIND:
            return infra_result(request, f"unsupported request kind {request.kind!r}")
        t0 = time.monotonic()
        try:
            patch = get_text(store, str(request.inputs["patch"]))
            skip_patch = not patch.strip()  # a patch filter left nothing to apply: the tests run on the base tree
            if skip_patch and "skip_patch" not in inspect.signature(run_instance).parameters:
                return infra_result(request, "empty patch: this run_instance cannot run the tests without a patch")
            script = get_text(store, str(request.inputs["eval_script"]))
            p = request.params
            instance_id = str(p.get("instance_id") or request.task_id)
            spec = make_test_spec({"instance_id": instance_id, "image": str(p.get("image") or
                                                                          request.execution.get("image")),
                                   "repo": str(p.get("repo")), "version": str(p.get("version")),
                                   "log_parser": str(p.get("log_parser")), "eval_type": str(p.get("eval_type")),
                                   "eval_script": script, "FAIL_TO_PASS": [], "PASS_TO_PASS": []})
            run_id = f"{self.run_id_prefix}-{request.request_id[7:19]}-{uuid.uuid4().hex[:8]}"
            pred = {"instance_id": instance_id, "model_name_or_path": "agent-ecc", "model_patch": patch}
            run_instance(spec, pred, client, run_id, timeout=int(self.timeout_s),
                         **({"skip_patch": True} if skip_patch else {}))
        except Exception as error:  # noqa: BLE001
            return infra_result(request, f"{type(error).__name__}: {error}", seconds=time.monotonic() - t0)
        seconds = time.monotonic() - t0
        assert self._log_root is not None
        log_dir = self._log_root / run_id / "agent-ecc" / instance_id
        try:
            test_output = log_dir / "test_output.txt"
            instance_log = log_dir / "run_instance.log"
            inst = instance_log.read_text(errors="replace") if instance_log.exists() else ""
            cost = Cost(seconds=seconds, executions=1, container_seconds=seconds)
            if test_output.exists():
                log = test_output.read_text(errors="replace")
                data = log.encode("utf-8")
                ref = EvidenceRef(role="test_log", digest=store.put(data), size=len(data), media_type="text/plain")
                # a timeout is the harness's own marker at the end of the log (the official runner appends it after a
                # blank line), never the words anywhere in it: a test may print them
                timed_out = _TRAILING_TIMEOUT_RE.search(log) is not None
                return ExecResult(request_id=request.request_id, status=RUN_OK, outputs=(ref,),
                                  facts={"applied": True, "timed_out": timed_out}, cost=cost)
            if APPLY_PATCH_FAIL in inst:
                # the agent's patch does not apply to this image: a logical fail, with no test log
                return ExecResult(request_id=request.request_id, status=RUN_OK, outputs=(),
                                  facts={"applied": False, "timed_out": False}, cost=cost)
            return infra_result(request, f"no test output: {inst[-400:]}", seconds=seconds)
        finally:
            if not self.keep_logs:
                shutil.rmtree(self._log_root / run_id, ignore_errors=True)
