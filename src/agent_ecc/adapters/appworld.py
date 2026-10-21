# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AppWorld: evidence format, answer re-grading, and state re-grading with the AppWorld evaluator.

An AppWorld task is graded by the world's evaluator on the episode's end state: one test checks the submitted answer
(for question tasks), the others check the database state. The adapter splits a task's spec into two checks:

* ``answer``: execution identity ``{evaluator, comparator}``, grading identity ``{answer}`` (the expected answer).
  An expected-answer change is **re-parsed** in-process from the recorded submitted answer, with AppWorld's answer
  normalization (:func:`answer_to_text`), which is exact while the evaluator is unchanged. An evaluator change is
  re-executed: the target evaluator's own answer test decides.
* ``state``: execution identity ``{evaluator, state_truth}`` (the evaluator code and the state-dependent truth). A
  change of either is **re-executed**: the target version's evaluator runs on the recorded end state (the task's
  database change logs), with no replay of the episode (:class:`AppWorldEvaluatorBackend`, extra
  ``agent-ecc[appworld]``). The state check passes when every non-answer test passes and the evaluator's own verdict
  (``success``: every test of the task passed, none missing) does not say otherwise.

The spec is ``AllOf(answer, state)``: a wrong answer fails the task without re-running the evaluator. A task without
an answer test has the state check only.

One evaluator run decides both checks, so both ask the same request, which names the whole truth the evaluator runs
with (evaluator, data version, state truth, expected answer). An evaluator run compares the submitted answer with
the expected answer of the data it ran with, so its answer test is that answer's only: the run is stamped with the
truth it ran with (:meth:`AppWorldAdapter.reexec_run`) and carries the submitted answer, and a later version that
changes only the expected answer re-parses the submitted answer instead of reading the old answer test.

Evidence roles: ``answer`` (``{"found": bool, "value": ...}``), ``end_state`` (``{"files": {path: text},
"truncated": bool}``), ``tests`` (per-test outcomes ``[{"requirement", "passed"}]``), ``commands`` (the code the agent
executed; level L1).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

from ..canonical import thaw
from ..records import (
    RUN_ERROR,
    RUN_OK,
    Cost,
    EvidenceRef,
    ExecutionRecord,
    Method,
    Outcome,
    PartScore,
    ScoreRecord,
    TestOutcome,
    TestRun,
)
from ..recompute.base import ExecRequest, ExecResult, infra_result
from ..scoring import PartVerdict
from ..store import ContentStore, get_json, put_json
from ..versions import AllOf, Check, GraderVersion, TestSpec, iter_checks, spec_digest

__all__ = [
    "AppWorldAdapter",
    "AppWorldEvaluatorBackend",
    "CHECK_ANSWER",
    "CHECK_STATE",
    "FAMILY",
    "TaskTruth",
    "answer_to_text",
    "is_answer_test",
    "task_truth_from_dir",
]

FAMILY = "appworld"
CHECK_ANSWER = "answer"
CHECK_STATE = "state"
ROLE_ANSWER = "answer"
ROLE_END_STATE = "end_state"
ROLE_TESTS = "tests"
ROLE_COMMANDS = "commands"
REQUEST_KIND = "appworld/evaluate"
ANSWER_COMPARATOR = "appworld/answer-to-text/1"


def answer_to_text(answer: Any) -> str:
    """AppWorld's answer normalization, reproduced: None is "null", lists are sorted and joined with " | ", quoted
    strings are unquoted, numbers are rounded to two decimals (integers stay integers), text is lowercased."""
    if answer is None:
        return "null"
    if isinstance(answer, (list, tuple)):
        return " | ".join(sorted(answer_to_text(v) for v in answer))
    if isinstance(answer, str):
        answer = answer.strip()
        if (answer.startswith('"') and answer.endswith('"')) or (answer.startswith("'") and answer.endswith("'")):
            answer = answer[1:-1]
    try:
        number = float(answer)
        answer = int(number) if number.is_integer() else round(number, 2)
    except (TypeError, ValueError):
        pass
    return str(answer).strip().lower()


def is_answer_test(requirement: str) -> bool:
    """Whether an evaluator test is the answer test (AppWorld names it "assert answers match")."""
    return "answers match" in str(requirement).lower()


@dataclass(frozen=True)
class TaskTruth:
    """What one grader version says about one task in one world.

    ``answer`` is the expected answer (``has_answer=False`` for tasks without an answer test); ``state_digest`` is the
    digest of the state-dependent truth (the task's private and public data and test code); ``evaluator`` the digest
    of the evaluator code; ``data_version`` names the data bundle a re-grading backend loads this truth from."""

    answer: Any = None
    has_answer: bool = True
    state_digest: str = ""
    evaluator: str = ""
    data_version: str | None = None


def task_truth_from_dir(task_dir: str | os.PathLike, *, evaluator: str, data_version: str | None = None,
                        has_answer: bool | None = None) -> TaskTruth:
    """A task's truth from an AppWorld task directory (``ground_truth/answer.json`` and the other ground-truth
    files). ``evaluator`` is the evaluator-code digest (for example of the installed package version and the task's
    evaluation code)."""
    gt = os.path.join(os.fspath(task_dir), "ground_truth")
    answer_path = os.path.join(gt, "answer.json")
    answer = None
    if os.path.exists(answer_path):
        with open(answer_path, encoding="utf-8") as f:
            answer = json.load(f)
    h = hashlib.sha256()
    for name in sorted(os.listdir(gt)):
        path = os.path.join(gt, name)
        if name == "answer.json" or not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            h.update(name.encode() + b"\0" + f.read() + b"\0")
    return TaskTruth(answer=answer, has_answer=(answer is not None) if has_answer is None else has_answer,
                     state_digest=f"sha256:{h.hexdigest()}", evaluator=evaluator, data_version=data_version)


class AppWorldAdapter:
    """The AppWorld family: specs from task truth, records from graded reports, and the scorer protocol."""

    family = FAMILY

    # ------------------------------------------------------------------------------------------- versions

    def spec(self, task_id: str, truth: TaskTruth, store: ContentStore) -> TestSpec:
        """The task's spec. Each check's ``params`` also name the other check's truth, so that both checks ask the
        one evaluator run of the whole truth (:meth:`reexec_request`)."""
        answer_digest = put_json(store, {"answer": truth.answer}) if truth.has_answer else None
        state = Check(name=CHECK_STATE, execution={"evaluator": truth.evaluator, "state_truth": truth.state_digest},
                      grading={}, params={"kind": CHECK_STATE, "task_id": task_id, "data_version": truth.data_version,
                                          "answer": answer_digest})
        if not truth.has_answer:
            return state
        answer = Check(name=CHECK_ANSWER, execution={"evaluator": truth.evaluator, "comparator": ANSWER_COMPARATOR},
                       grading={"answer": answer_digest},
                       params={"kind": CHECK_ANSWER, "task_id": task_id, "data_version": truth.data_version,
                               "state_truth": truth.state_digest})
        return AllOf(name="all_of", parts=(answer, state))

    def version(self, truths: Mapping[str, Mapping[str, TaskTruth]], store: ContentStore, *, name: str,
                governed: Iterable[str] | None = None, meta: Mapping[str, Any] | None = None) -> GraderVersion:
        """A grader version from ``task -> world version -> truth`` (truth is per world: a world change can change
        it)."""
        entries = {task: {world: self.spec(task, truth, store) for world, truth in by_world.items()}
                   for task, by_world in truths.items()}
        return GraderVersion(name=name, entries=entries,
                             governed=frozenset(governed) if governed is not None else None, family=FAMILY,
                             meta=meta or {})

    # ------------------------------------------------------------------------------------------- records

    def record_from_report(self, report: Mapping[str, Any], *, store: ContentStore, task_id: str,
                           trajectory_id: str, world_version: str, spec: TestSpec,
                           grader_version: str | None = None, observed_worlds: Sequence[str] = (),
                           meta: Mapping[str, Any] | None = None) -> tuple[ExecutionRecord, ScoreRecord | None]:
        """An execution record and its recorded score from a graded report, as an in-sandbox grading endpoint
        returns it: ``success``, ``tests`` (``requirement``, ``passed``), ``submitted_answer`` (``found``, ``value``),
        ``end_state`` (``files``, ``truncated``), ``executed_commands``. ``spec`` is the spec the trajectory was graded
        under (from the grader version it pinned when it started). Returns no score when the report holds no
        verdict (an evaluation error, or no ``success``); its partial tests are then recorded as a failed state run,
        never as outputs a later version could read a verdict from. The submitted answer is the agent's own artifact
        and is kept either way."""
        artifacts, facts = [], {}
        submitted = report.get("submitted_answer")
        answer_ref = None
        if isinstance(submitted, Mapping) and "found" in submitted:
            answer_obj = {"found": bool(submitted["found"]), "value": submitted.get("value")}
            answer_ref = _ref(store, ROLE_ANSWER, answer_obj)
            artifacts.append(answer_ref)
        end_state = report.get("end_state")
        if isinstance(end_state, Mapping) and end_state.get("files") is not None:
            artifacts.append(_ref(store, ROLE_END_STATE, {"files": dict(end_state["files"]),
                                                          "truncated": bool(end_state.get("truncated"))}))
            facts["end_state_truncated"] = bool(end_state.get("truncated"))
        if report.get("executed_commands") is not None:
            artifacts.append(_ref(store, ROLE_COMMANDS, list(report["executed_commands"])))
        tests = report.get("tests")
        state_tests = None
        if isinstance(tests, list):
            state_tests = [{"requirement": str(t.get("requirement", "")), "passed": bool(t.get("passed"))}
                           for t in tests if not is_answer_test(t.get("requirement", ""))]
        error = report.get("error")
        graded = error is None and "success" in report
        runs = []
        for check in iter_checks(spec):
            kind = check.params.get("kind")
            if kind == CHECK_ANSWER and answer_ref is not None:
                runs.append(TestRun(check=check.name, execution=check.execution, outputs=(answer_ref,)))
            elif kind == CHECK_STATE and not graded:
                # the evaluation did not finish: whatever tests it reported are not a verdict
                why = str(error)[:300] if error is not None else "the report holds no verdict"
                runs.append(TestRun(check=check.name, execution=check.execution, status=RUN_ERROR,
                                    facts={"error": f"evaluation_error: {why}"}))
            elif kind == CHECK_STATE and state_tests is not None:
                run_facts = {}
                if report.get("other_tests_passed") is not None:
                    run_facts["other_tests_passed"] = bool(report["other_tests_passed"])  # the evaluator's own
                runs.append(TestRun(check=check.name, execution=check.execution,
                                    outputs=(_ref(store, ROLE_TESTS, state_tests),), facts=run_facts))
        if error is not None:
            facts["grading_error"] = str(error)[:300]
        record = ExecutionRecord(family=FAMILY, task_id=str(task_id), trajectory_id=str(trajectory_id),
                                 world_version=str(world_version), world_observations=tuple(observed_worlds),
                                 artifacts=tuple(artifacts), runs=tuple(runs), facts=facts, meta=meta or {})
        if not graded:
            return record, None
        parts = []
        for check in iter_checks(spec):
            kind = check.params.get("kind")
            if kind == CHECK_ANSWER:
                passed = report.get("answer_test_passed")
                if passed is None and isinstance(tests, list):
                    passed = next((bool(t.get("passed")) for t in tests if is_answer_test(t.get("requirement", ""))),
                                  None)
            else:
                passed = report.get("other_tests_passed")
                if passed is None and state_tests is not None:
                    passed = all(t["passed"] for t in state_tests)
            if passed is None:
                parts = []
                break
            parts.append(PartScore(check=check.name, execution_digest=check.execution_digest,
                                   grading_digest=check.grading_digest,
                                   outcome=Outcome.PASS if passed else Outcome.FAIL, method=Method.RECORDED,
                                   reason=None if passed else f"{kind}_failed"))
        outcome = Outcome.PASS if report.get("success") else Outcome.FAIL
        if parts and (all(p.outcome is Outcome.PASS for p in parts)) != (outcome is Outcome.PASS):
            raise ValueError("the report's success does not equal its tests' outcomes")
        score = ScoreRecord(execution_id=record.record_id, grader_version=grader_version, outcome=outcome,
                            method=Method.RECORDED, spec_digest=spec_digest(spec), parts=tuple(parts))
        return record, score

    # ------------------------------------------------------------------------------------------- scorer protocol

    def parse(self, check: Check, run: TestRun, store: ContentStore) -> PartVerdict:
        if run.status == RUN_ERROR:
            return PartVerdict.undeterminable(f"infra_error:{run.facts.get('error') or 'unknown'}")
        kind = check.params.get("kind")
        tests_ref = run.output(ROLE_TESTS)
        if kind == CHECK_ANSWER:
            expected_digest = str(check.grading["answer"])
            truth = run.facts.get("truth")
            if tests_ref is not None and isinstance(truth, Mapping) and truth.get("answer") == expected_digest:
                # an evaluator run made with this very expected answer decided the answer test
                tests = get_json(store, tests_ref.digest)
                answer_tests = [t for t in tests if is_answer_test(t.get("requirement", ""))]
                if not answer_tests:
                    return PartVerdict.undeterminable("answer_test_missing")
                passed = all(bool(t.get("passed")) for t in answer_tests)
                return PartVerdict.passed(reason="answer_matches") if passed else \
                    PartVerdict.failed("answer_mismatch")
            ref = run.output(ROLE_ANSWER)
            if ref is None:
                # an evaluator's answer test compared with another expected answer than this check's says nothing
                return PartVerdict.undeterminable("answer_test_for_another_truth" if tests_ref is not None
                                                  else "missing_answer")
            submitted = get_json(store, ref.digest)
            expected = get_json(store, expected_digest)["answer"]
            if not submitted.get("found"):
                return PartVerdict.failed("no_answer_submitted")
            if answer_to_text(submitted.get("value")) == answer_to_text(expected):
                return PartVerdict.passed(reason="answer_matches")
            return PartVerdict.failed("answer_mismatch")
        if kind == CHECK_STATE:
            if tests_ref is None:
                return PartVerdict.undeterminable("missing_tests")
            return _state_verdict(get_json(store, tests_ref.digest), run.facts)
        return PartVerdict.undeterminable(f"unknown_check_kind:{kind}", final=True)

    def reexec_request(self, record: ExecutionRecord, check: Check, store: ContentStore) -> ExecRequest | None:
        end_state = record.artifact(ROLE_END_STATE)
        data_version = check.params.get("data_version")
        if end_state is None or record.facts.get("end_state_truncated") or not data_version:
            return None
        if not store.contains(end_state.digest):
            return None
        # one evaluator run decides every test, so both checks of a task ask the same request (executed once); it
        # names the whole truth the evaluator runs with, so another truth is another request
        truth = {"evaluator": check.execution.get("evaluator"), "data_version": data_version,
                 "state_truth": check.execution.get("state_truth", check.params.get("state_truth")),
                 "answer": check.grading.get("answer", check.params.get("answer"))}
        inputs = {"end_state": end_state.digest}
        answer = record.artifact(ROLE_ANSWER)
        if answer is not None and store.contains(answer.digest):
            inputs["answer"] = answer.digest  # carried into the run: a later expected answer re-parses it
        return ExecRequest(family=FAMILY, kind=REQUEST_KIND, task_id=record.task_id, execution=truth, inputs=inputs,
                           params={"task_id": record.task_id})

    def reexec_run(self, check: Check, request: ExecRequest, result: ExecResult) -> TestRun:
        """The run an evaluator re-run becomes: its outputs and the submitted answer it was given, stamped with the
        truth it ran with (``facts["truth"]``), since its answer test holds for that expected answer only."""
        outputs = list(result.outputs)
        answer = request.inputs.get("answer")
        if answer and not any(o.role == ROLE_ANSWER for o in outputs):
            outputs.append(EvidenceRef(role=ROLE_ANSWER, digest=str(answer), media_type="application/json"))
        return TestRun(check=check.name, execution=check.execution, status=RUN_OK, outputs=tuple(outputs),
                       facts={**thaw(result.facts), "truth": thaw(request.execution)})

    def decide_without_execution(self, record: ExecutionRecord, check: Check) -> PartVerdict | None:
        return None


def _state_verdict(tests: Sequence[Mapping[str, Any]], facts: Mapping[str, Any]) -> PartVerdict:
    """The state check's verdict from an evaluation's tests and what the evaluator itself reported.

    AppWorld's own verdict is ``success``: every test of the task passed, none missing. A graded report's
    ``other_tests_passed`` is the evaluator's own state verdict. The state passes when every reported non-answer test
    passed and neither says otherwise; a failure that only the answer test explains leaves the state passing (the
    answer check fails the task); a verdict that contradicts the tests is not a verdict."""
    state = [t for t in tests if not is_answer_test(t.get("requirement", ""))]
    outcomes = tuple(TestOutcome(name=str(t.get("requirement", ""))[:300],
                                 status="passed" if t.get("passed") else "failed") for t in state)
    state_failed = any(not t.get("passed") for t in state)
    own = facts.get("other_tests_passed")
    if own is not None:
        if bool(own) and state_failed:
            return PartVerdict.undeterminable("state_verdict_contradicts_tests", outcomes)
        return PartVerdict.passed(outcomes, "state_tests_passed") if own else \
            PartVerdict.failed("state_tests_failed", outcomes)
    success = facts.get("success")
    if success is True:
        if state_failed or any(not t.get("passed") for t in tests):
            return PartVerdict.undeterminable("success_contradicts_tests", outcomes)
        return PartVerdict.passed(outcomes, "state_tests_passed")
    if state_failed:
        return PartVerdict.failed("state_tests_failed", outcomes)
    if success is False:
        num_tests = facts.get("num_tests")
        missing = isinstance(num_tests, int) and len(tests) < num_tests
        answer_failed = any(not t.get("passed") for t in tests if is_answer_test(t.get("requirement", "")))
        if missing or not answer_failed:
            # every reported state test passed, yet the evaluator says the task is not solved: tests are missing
            return PartVerdict.failed("evaluator_not_success", outcomes)
    return PartVerdict.passed(outcomes, "state_tests_passed")


def _ref(store: ContentStore, role: str, value: Any) -> EvidenceRef:
    digest = put_json(store, value)
    return EvidenceRef(role=role, digest=digest, size=len(store.get(digest)), media_type="application/json")


# ---------------------------------------------------------------------------------------------------------- backend

_ROOT: str | None = None


def _init_worker(data_dir: str) -> None:
    """Each worker process serves one data version: AppWorld reads its root once per process."""
    global _ROOT
    _ROOT = tempfile.mkdtemp(prefix="agent-ecc-appworld-")
    os.symlink(os.path.realpath(data_dir), os.path.join(_ROOT, "data"))
    os.environ["APPWORLD_ROOT"] = _ROOT


def _evaluate_end_state(task_id: str, files: Mapping[str, str]) -> dict:
    from appworld.apps.model_lib import CachedDBHandler  # the extra agent-ecc[appworld]
    from appworld.evaluator import evaluate_task

    assert _ROOT is not None
    experiment = "agent_ecc_" + uuid.uuid4().hex[:12]
    out_dir = os.path.join(_ROOT, "experiments", "outputs", experiment, "tasks", task_id)
    t0 = time.time()
    try:
        for rel, text in files.items():
            if os.path.isabs(rel) or ".." in rel.split("/"):
                raise ValueError(f"unsafe end-state path {rel!r}")
            path = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        os.makedirs(os.path.join(out_dir, "dbs"), exist_ok=True)
        tracker = evaluate_task(task_id=task_id, experiment_name=experiment, suppress_errors=True, save_report=False)
        tests = ([{"requirement": str(p.get("requirement", ""))[:300], "passed": True} for p in tracker.passes]
                 + [{"requirement": str(f.get("requirement", ""))[:300], "passed": False} for f in tracker.failures])
        return {"tests": tests, "success": bool(tracker.success), "num_tests": int(tracker.num_tests),
                "seconds": time.time() - t0}
    except Exception as error:  # noqa: BLE001 - reported as an infrastructure error
        return {"error": f"{type(error).__name__}: {error}"[:300], "seconds": time.time() - t0}
    finally:
        CachedDBHandler.reset()
        shutil.rmtree(os.path.join(_ROOT, "experiments", "outputs", experiment), ignore_errors=True)


class AppWorldEvaluatorBackend:
    """Re-runs a data version's AppWorld evaluator on recorded end states (extra ``agent-ecc[appworld]``).

    ``data_dirs`` maps a ``data_version`` (as a version's checks name it) to an AppWorld data directory. Each data
    version gets its own pool of ``workers`` processes, started on first use and kept warm."""

    def __init__(self, data_dirs: Mapping[str, str | os.PathLike], *, workers: int = 4) -> None:
        self.data_dirs = {k: os.fspath(v) for k, v in data_dirs.items()}
        self.workers = max(1, int(workers))
        self._pools: dict[str, ProcessPoolExecutor] = {}

    def _pool(self, data_version: str) -> ProcessPoolExecutor:
        if data_version not in self._pools:
            self._pools[data_version] = ProcessPoolExecutor(max_workers=self.workers, initializer=_init_worker,
                                                            initargs=(self.data_dirs[data_version],))
        return self._pools[data_version]

    def execute(self, requests: Sequence[ExecRequest], store: ContentStore) -> list[ExecResult]:
        futures = []
        for request in requests:
            data_version = str(request.execution.get("data_version"))
            if request.kind != REQUEST_KIND or data_version not in self.data_dirs:
                futures.append((request, None, f"no data directory for data version {data_version!r}"))
                continue
            try:
                end_state = get_json(store, str(request.inputs["end_state"]))
            except Exception as error:  # noqa: BLE001
                futures.append((request, None, f"end state unreadable: {error}"))
                continue
            if end_state.get("truncated") or not end_state.get("files"):
                futures.append((request, None, "end state truncated or empty"))
                continue
            futures.append((request, self._pool(data_version).submit(
                _evaluate_end_state, str(request.params.get("task_id") or request.task_id), end_state["files"]),
                None))
        out = []
        for request, future, error in futures:
            if future is None:
                out.append(infra_result(request, error or "not executed"))
                continue
            try:
                res = future.result()
            except Exception as err:  # noqa: BLE001
                out.append(infra_result(request, f"{type(err).__name__}: {err}"))
                continue
            if "error" in res:
                out.append(infra_result(request, res["error"], seconds=float(res.get("seconds") or 0.0)))
                continue
            ref = _ref(store, ROLE_TESTS, res["tests"])
            out.append(ExecResult(request_id=request.request_id, status=RUN_OK, outputs=(ref,),
                                  facts={"success": res["success"], "num_tests": res.get("num_tests")},
                                  cost=Cost(seconds=float(res.get("seconds") or 0.0), executions=1,
                                            container_seconds=float(res.get("seconds") or 0.0))))
        return out

    def close(self) -> None:
        for pool in self._pools.values():
            pool.shutdown(wait=False, cancel_futures=True)
        self._pools.clear()


def evaluator_digest(package_version: str, evaluation_code: bytes | str = b"") -> str:
    """An evaluator identity from the evaluator package version and a task's evaluation code."""
    if isinstance(evaluation_code, str):
        evaluation_code = evaluation_code.encode("utf-8")
    h = hashlib.sha256()
    h.update(f"appworld-evaluator/{package_version}".encode() + b"\0")
    h.update(evaluation_code)
    return f"sha256:{h.hexdigest()}"
