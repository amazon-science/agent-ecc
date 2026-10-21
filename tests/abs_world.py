# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A SWE-ABS grader switch for tests: e0 grades a task by its original suite; e1 by the original suite AND a hardened
suite graded against a reference set G, on the patch without its diffs of the hardened test files. Re-executions run
in a simulated backend that grades patches by their text: the original suite passes a patch containing 'fix'; the
hardened suite also needs it not to be 'fragile'."""

from __future__ import annotations

import threading

from agent_ecc.adapters.swebench import (
    END_TEST_OUTPUT,
    START_TEST_OUTPUT,
    TEST_EXIT_CODE,
    SimpleHarness,
    SWEBenchAdapter,
)
from agent_ecc.records import Cost, EvidenceRef, Outcome
from agent_ecc.recompute.base import CallableBackend, ExecResult
from agent_ecc.scoring import Scorer, grade_recorded
from agent_ecc.seal import Candidate, Member, arm_session
from agent_ecc.store import MemoryStore, get_text
from agent_ecc.versions import MemoryRegistry

IMAGE = "example.org/swe/eval@sha256:" + "0" * 64
TASK = "pkg__pkg-1"
F2P, P2P = ["tests/test_mod.py::test_fix"], ["tests/test_mod.py::test_keep"]
G = ["tests/test_h.py::test_h1", "tests/test_h.py::test_h2"]
HARDENED_TEST_PATCH = ("diff --git a/tests/test_h.py b/tests/test_h.py\nnew file mode 100644\n--- /dev/null\n"
                       "+++ b/tests/test_h.py\n@@ -0,0 +1,2 @@\n+def test_h1(): pass\n+def test_h2(): pass\n")


def script(tests):
    return (f"#!/bin/bash\ncd /testbed\n: '{START_TEST_OUTPUT}'\npytest -rA {' '.join(tests)}\n"
            f": '{END_TEST_OUTPUT}'\n")


def log(statuses):
    failed = sum(1 for s in statuses.values() if s == "FAILED")
    body = "\n".join(f"{status} {name}" for name, status in statuses.items())
    return (f"{START_TEST_OUTPUT}\n{body}\n===== {failed} failed, {len(statuses) - failed} passed =====\n"
            f"{END_TEST_OUTPUT}\n{TEST_EXIT_CODE}: {1 if failed else 0}\n")


def outcome_log(suite, passed):
    tests = F2P + P2P if suite == "tests" else G
    return log({t: "PASSED" if passed or t in P2P else "FAILED" for t in tests})


class World:
    """Two epochs, a registry whose clock the test moves, and a re-execution backend.

    ``error_on`` names suites whose executions fail in the sandbox (infrastructure errors); ``truncate_on`` suites
    whose executions return a log cut before its end marker (outputs without a logical verdict); ``block`` makes every
    execution wait for the event first (a job that is still running)."""

    def __init__(self, *, error_on=(), truncate_on=(), block: threading.Event | None = None):
        self.store = MemoryStore()
        self.adapter = SWEBenchAdapter(harness=SimpleHarness())
        row = {"instance_id": TASK, "repo": "pkg/pkg", "version": "1.0", "image": IMAGE,
               "eval_script": script(F2P + P2P), "log_parser": "parse_log_pytest", "eval_type": "pass_and_fail",
               "FAIL_TO_PASS": F2P, "PASS_TO_PASS": P2P}
        self.e0 = self.adapter.version_from_rows([row], self.store, name="e0")
        self.e1 = self.adapter.conjunction_version(
            [row], [{"instance_id": TASK, "eval_script": script(G), "test_patch": HARDENED_TEST_PATCH,
                     "reference_set": G}], self.store, name="e1")
        self.base, self.hardened = self.e1.entries[TASK]["*"].parts
        self.now = 100.0
        self.registry = MemoryRegistry(clock=lambda: self.now)
        self.registry.publish_grader(self.e0, source="startup")
        self.executions = []
        self.error_on = set(error_on)
        self.truncate_on = set(truncate_on)
        self.block = block
        self._lock = threading.Lock()
        self.backend = CallableBackend(self._execute)
        self.cache: dict = {}
        self.n = 0

    def switch(self):
        self.now += 10
        self.registry.publish_grader(self.e1, source="switch")
        self.now += 10

    def publish_again(self, version):
        """The same content published again: a new publication (another epoch identity) of an existing version."""
        self.now += 10
        publication = self.registry.publish_grader(version, source="again")
        self.now += 10
        return publication

    def _execute(self, request, store):
        if self.block is not None:
            self.block.wait(10)
        patch, eval_script = get_text(store, request.inputs["patch"]), get_text(store, request.inputs["eval_script"])
        suite = "hardened" if "test_h.py" in eval_script else "tests"
        with self._lock:
            self.executions.append((patch, suite))
        if suite in self.error_on:
            raise RuntimeError("sandbox creation failed")
        passed = "fix" in patch and (suite == "tests" or "fragile" not in patch)
        text = outcome_log(suite, passed)
        if suite in self.truncate_on:
            text = text.split(END_TEST_OUTPUT)[0]  # cut before the end marker: no logical verdict
        data = text.encode()
        ref = EvidenceRef(role="test_log", digest=store.put(data), size=len(data), media_type="text/plain")
        return ExecResult(request_id=request.request_id, status="ok", outputs=(ref,), facts={"applied": True},
                          cost=Cost(seconds=0.01, executions=1, container_seconds=60.0))

    def member(self, graded_under, passes, *, patch="diff fix", member_id=None):
        """A member graded under ``graded_under`` (e0 or e1); ``passes`` lists each suite's recorded outcome
        (True, False, or None for an infrastructure error)."""
        self.n += 1
        spec = graded_under.entries[TASK]["*"]
        checks = [spec] if graded_under is self.e0 else [self.base, self.hardened]
        runs = []
        for check, passed in zip(checks, passes):
            suite = "hardened" if check is self.hardened else "tests"
            if passed is None:
                runs.append(self.adapter.run(check, store=self.store, log=None, error="pod evicted"))
            else:
                runs.append(self.adapter.run(check, store=self.store, log=outcome_log(suite, passed)))
        mid = member_id or f"m{self.n}"
        record = self.adapter.record(store=self.store, task_id=TASK, trajectory_id=mid, image=IMAGE, patch=patch,
                                     runs=runs)
        recorded = grade_recorded(self.adapter, record, spec, self.store, grader_version=graded_under.version_id)
        reward = 1.0 if recorded.outcome is Outcome.PASS else 0.0
        return Member(member_id=mid, execution=record, prior=(recorded,), recorded_reward=reward)

    def scorer(self, **kwargs):
        return Scorer([self.adapter], self.store, self.backend, **kwargs)

    def seal(self, arm, groups, **kwargs):
        session = arm_session(arm, self.registry, self.scorer(),
                              cache=self.cache if arm in ("ecc", "full_compatible", "never") else None, **kwargs)
        candidates = [Candidate(group_id=f"g{i}", members=tuple(g)) for i, g in enumerate(groups)]
        return session, session.finish(session.decide(candidates))


def statuses(decision):
    return [m.status for _, m in decision.members()]


def trained(decision):
    return [m.trained_reward for _, m in decision.members()]
