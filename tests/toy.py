# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A toy task family for tests: every decision path, no third-party code.

The agent's artifact is an answer string (role ``answer``). A check runs a *runner* on the answer, producing an
output string (role ``output``); the check passes when the output equals its expected value. The runner is the
check's execution identity, the expected value its grading identity:

* changing the expected value can be decided by re-parsing a recorded output;
* changing the runner needs the answer to be run again (re-execution).

``oracle_verdict`` computes the true verdict of any spec directly from the answer, independently of recorded
evidence, which is what the property tests compare trained labels against.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from agent_ecc.records import RUN_ERROR, RUN_OK, EvidenceRef, ExecutionRecord, Outcome, TestRun
from agent_ecc.recompute.base import CallableBackend, ExecRequest, ExecResult, infra_result
from agent_ecc.records import Cost
from agent_ecc.scoring import PartVerdict
from agent_ecc.store import ContentStore, get_json, get_text, put_json
from agent_ecc.versions import AllOf, Check, TestSpec, iter_checks

FAMILY = "toy"


def run_answer(runner: str, answer: str) -> str:
    """What a runner outputs for an answer."""
    if runner == "rev":
        return answer[::-1]
    if runner == "up":
        return answer.upper()
    if runner == "dup":
        return answer + answer
    return answer


def check(name: str, runner: str, expected: str, store: ContentStore) -> Check:
    return Check(name=name, execution={"runner": runner}, grading={"expected": put_json(store, {"value": expected})},
                 params={"kind": "toy"})


def expected_of(c: Check, store: ContentStore) -> str:
    return get_json(store, str(c.grading["expected"]))["value"]


def oracle_check(c: Check, answer: str, store: ContentStore) -> bool:
    return run_answer(str(c.execution["runner"]), answer) == expected_of(c, store)


def oracle_verdict(spec: TestSpec, answer: str, store: ContentStore) -> bool:
    """The true verdict of ``spec`` on ``answer``: every check passes."""
    return all(oracle_check(c, answer, store) for c in iter_checks(spec))


class ToyAdapter:
    family = FAMILY

    def __init__(self, *, fail_parse_for: Iterable[str] = ()) -> None:
        self.parsed = 0
        self.fail_parse_for = set(fail_parse_for)  # outputs whose parse reports an infrastructure error

    def record(self, store: ContentStore, *, task: str, trajectory: str, answer: str | None, world: str = "w0",
               runs: Iterable[TestRun] = (), observed: Iterable[str] = (), empty: bool = False) -> ExecutionRecord:
        artifacts = []
        if answer is not None:
            data = answer.encode()
            artifacts.append(EvidenceRef(role="answer", digest=store.put(data), size=len(data)))
        return ExecutionRecord(family=FAMILY, task_id=task, trajectory_id=trajectory, world_version=world,
                               world_observations=tuple(observed), artifacts=tuple(artifacts), runs=tuple(runs),
                               facts={"empty": True} if empty else {})

    def run(self, store: ContentStore, c: Check, answer: str, *, keep_output: bool = True,
            error: bool = False) -> TestRun:
        if error:
            return TestRun(check=c.name, execution=c.execution, status=RUN_ERROR, facts={"error": "boom"})
        outputs = ()
        if keep_output:
            data = run_answer(str(c.execution["runner"]), answer).encode()
            outputs = (EvidenceRef(role="output", digest=store.put(data), size=len(data)),)
        return TestRun(check=c.name, execution=c.execution, status=RUN_OK, outputs=outputs)

    # scorer protocol
    def parse(self, c: Check, run: TestRun, store: ContentStore) -> PartVerdict:
        self.parsed += 1
        if run.status == RUN_ERROR:
            return PartVerdict.undeterminable("infra_error")
        ref = run.output("output")
        if ref is None:
            return PartVerdict.undeterminable("missing_output")
        output = get_text(store, ref.digest)
        if output in self.fail_parse_for:
            return PartVerdict.undeterminable("garbled_output")
        if output == expected_of(c, store):
            return PartVerdict.passed(reason="match")
        return PartVerdict.failed("mismatch")

    def reexec_request(self, record: ExecutionRecord, c: Check, store: ContentStore) -> ExecRequest | None:
        answer = record.artifact("answer")
        if answer is None or not store.contains(answer.digest):
            return None
        return ExecRequest(family=FAMILY, kind="toy/run", task_id=record.task_id, execution=c.execution,
                           inputs={"answer": answer.digest})

    def decide_without_execution(self, record: ExecutionRecord, c: Check) -> PartVerdict | None:
        if record.facts.get("empty"):
            return PartVerdict.failed("empty_answer")
        return None


class ToyBackend(CallableBackend):
    """Runs toy requests. ``infra`` maps a request id to the number of times it fails before it succeeds;
    ``always_fail`` makes every execution an infrastructure error."""

    def __init__(self, *, infra: Mapping[str, int] | None = None, always_fail: bool = False,
                 garble: bool = False) -> None:
        self.infra = dict(infra or {})
        self.always_fail = always_fail
        self.garble = garble
        self.executed: list[str] = []
        self.batches: list[int] = []
        super().__init__(self._run)

    def execute(self, requests, store):
        self.batches.append(len(requests))
        return super().execute(requests, store)

    def _run(self, request: ExecRequest, store: ContentStore) -> ExecResult:
        self.executed.append(request.request_id)
        if self.always_fail:
            return infra_result(request, "sandbox unavailable")
        left = self.infra.get(request.request_id, 0)
        if left > 0:
            self.infra[request.request_id] = left - 1
            return infra_result(request, "transient")
        answer = get_text(store, str(request.inputs["answer"]))
        output = run_answer(str(request.execution["runner"]), answer)
        if self.garble:
            output = "\x00garbled"
        data = output.encode()
        ref = EvidenceRef(role="output", digest=store.put(data), size=len(data))
        return ExecResult(request_id=request.request_id, status=RUN_OK, outputs=(ref,),
                          cost=Cost(seconds=0.01, executions=1, container_seconds=1.0))


def conj(name: str, *parts: TestSpec) -> AllOf:
    return AllOf(name=name, parts=tuple(parts))


def outcome(passed: bool) -> Outcome:
    return Outcome.PASS if passed else Outcome.FAIL
