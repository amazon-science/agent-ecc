# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A synthetic SWE-bench-like world for the lifecycle example (no containers, no real repositories).

Each task has a few tests. An agent's patch is plain text; a test passes when the patch contains the token the test
looks for (``fix:<test>``), and a passing-to-passing test fails when the patch contains ``break:<test>``. A patch
containing ``corrupt`` does not apply. An eval script is a shell-like text whose ``pytest -rA ...`` line names the
tests it runs; the simulated runner executes exactly those tests and prints a pytest-style log between the SWE-bench
output markers, followed by the test exit code. Everything here is deterministic.
"""

from __future__ import annotations

import re

from agent_ecc.adapters.swebench import END_TEST_OUTPUT, START_TEST_OUTPUT, TEST_EXIT_CODE
from agent_ecc.records import Cost, EvidenceRef
from agent_ecc.recompute.base import CallableBackend, ExecResult
from agent_ecc.store import get_text

IMAGE = "example.org/swe-eval/demo@sha256:" + "0" * 64


def eval_script(tests: list[str], *, flags: str = "") -> str:
    return ("#!/bin/bash\nset -uxo pipefail\ncd /testbed\n"
            f": '{START_TEST_OUTPUT}'\npytest -rA {flags}{' '.join(tests)}\n: '{END_TEST_OUTPUT}'\n")


def tests_in(script: str) -> list[str]:
    match = re.search(r"^pytest -rA (?:-\S+ )*(.+)$", script, re.M)
    return match.group(1).split() if match else []


def run(patch: str, script: str, *, passing_by_default: set[str]) -> tuple[str | None, bool]:
    """(log, applied) of running ``script`` with ``patch`` applied."""
    if "corrupt" in patch:
        return None, False
    statuses = {}
    for test in tests_in(script):
        if f"break:{test}" in patch:
            statuses[test] = "FAILED"
        elif f"fix:{test}" in patch or test in passing_by_default:
            statuses[test] = "PASSED"
        else:
            statuses[test] = "FAILED"
    failed = sum(1 for s in statuses.values() if s == "FAILED")
    lines = [f"{status} {test}" for test, status in statuses.items()]
    summary = f"===== {failed} failed, {len(statuses) - failed} passed in 0.42s ====="
    log = "\n".join(["+ pytest -rA", START_TEST_OUTPUT, *lines, summary, END_TEST_OUTPUT,
                     f"{TEST_EXIT_CODE}: {1 if failed else 0}", ""])
    return log, True


class SimulatedEvaluator(CallableBackend):
    """A re-execution backend that runs requests in the simulated world (one 'container' per request)."""

    def __init__(self, passing_by_default: dict[str, set[str]]):
        self.passing_by_default = passing_by_default
        super().__init__(self._execute)

    def _execute(self, request, store):
        patch = get_text(store, str(request.inputs["patch"]))
        script = get_text(store, str(request.inputs["eval_script"]))
        log, applied = run(patch, script, passing_by_default=self.passing_by_default.get(request.task_id, set()))
        outputs = ()
        if log is not None:
            data = log.encode()
            outputs = (EvidenceRef(role="test_log", digest=store.put(data), size=len(data), media_type="text/plain"),)
        return ExecResult(request_id=request.request_id, status="ok", outputs=outputs,
                          facts={"applied": applied, "timed_out": False},
                          cost=Cost(seconds=0.0, executions=1, container_seconds=60.0))
