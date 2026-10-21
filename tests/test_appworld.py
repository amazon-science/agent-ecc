# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AppWorld adapter: answer normalization, records from graded reports, answer re-parse, state re-execution."""

import json

import pytest

from agent_ecc.adapters.appworld import (
    AppWorldAdapter,
    AppWorldEvaluatorBackend,
    TaskTruth,
    answer_to_text,
    evaluator_digest,
    is_answer_test,
    task_truth_from_dir,
)
from agent_ecc.records import EvidenceRef, Method, Outcome
from agent_ecc.recompute.base import CallableBackend, ExecResult
from agent_ecc.scoring import ScoreInput, Scorer
from agent_ecc.seal import Candidate, Member, SealSession
from agent_ecc.store import MemoryStore, get_json, put_json
from agent_ecc.versions import MemoryRegistry

TASK = "task_a_1"
W0, W1 = "world-0", "world-1"
EVAL_V1, EVAL_V2 = evaluator_digest("0.1.3"), evaluator_digest("0.1.3", "patched evaluation code")
STATE_A, STATE_B = "sha256:" + "a" * 64, "sha256:" + "b" * 64


def report(answer, *, success, others=True, found=True, end_state=True, truncated=False):
    """A graded report as an in-sandbox grading endpoint returns it; ``success`` is the recorded verdict, ``others``
    whether the non-answer tests passed (so the answer test passed iff the verdict is a pass or only others failed)."""
    answer_ok = success or not others
    tests = [{"requirement": "assert answers match", "passed": answer_ok},
             {"requirement": "assert no extra changes", "passed": others}]
    return {"success": success, "num_tests": 2, "num_tests_passed": sum(t["passed"] for t in tests), "tests": tests,
            "answer_test_passed": answer_ok, "other_tests_passed": others,
            "submitted_answer": {"found": found, "value": answer},
            "end_state": {"truncated": truncated, "files": {"dbs/music.jsonl": f"[\"UPDATE\", [\"{answer}\"]]"}}
            if end_state else None,
            "executed_commands": [f"apis.supervisor.complete_task(answer={answer!r})"]}


@pytest.fixture
def aw():
    store = MemoryStore()
    adapter = AppWorldAdapter()
    bug = TaskTruth(answer="Silver Lining", state_digest=STATE_A, evaluator=EVAL_V1, data_version="data-1")
    fix = TaskTruth(answer="A Love That Never Was", state_digest=STATE_A, evaluator=EVAL_V1, data_version="data-1")
    v_bug = adapter.version({TASK: {W0: bug}}, store, name="bug")
    v_fix = adapter.version({TASK: {W0: fix}}, store, name="fix")
    return store, adapter, bug, fix, v_bug, v_fix


def produce(store, adapter, version, answer, *, success, others=True, trajectory=None, **kw):
    spec = version.entries[TASK][W0]
    return adapter.record_from_report(report(answer, success=success, others=others, **kw), store=store,
                                      task_id=TASK, trajectory_id=trajectory or f"traj-{answer}", world_version=W0,
                                      spec=spec, grader_version=version.version_id)


class FakeEvaluator(CallableBackend):
    """Re-runs 'the evaluator' on an end state: the answer test passes iff the end state names ``correct``."""

    def __init__(self, correct, others=True):
        self.correct, self.others, self.requests = correct, others, []
        super().__init__(self._run)

    def _run(self, request, store):
        self.requests.append(request)
        state = get_json(store, request.inputs["end_state"])
        answer_ok = self.correct in json.dumps(state)
        tests = [{"requirement": "assert answers match", "passed": answer_ok},
                 {"requirement": "assert no extra changes", "passed": self.others}]
        ref = EvidenceRef(role="tests", digest=put_json(store, tests))
        return ExecResult(request_id=request.request_id, status="ok", outputs=(ref,),
                          facts={"success": answer_ok and self.others})


def test_answer_to_text():
    assert answer_to_text(None) == "null"
    assert answer_to_text(" 'Silver Lining' ") == "silver lining"
    assert answer_to_text(["b", "A"]) == "a | b"
    assert answer_to_text("3.0") == "3" and answer_to_text(2.456) == "2.46" and answer_to_text(7) == "7"
    assert answer_to_text('"x"') == "x"
    assert is_answer_test("Assert Answers Match") and not is_answer_test("assert no changes")


def test_spec_shapes(aw):
    store, adapter, bug, *_ = aw
    spec = adapter.spec(TASK, bug, store)
    assert [c.name for c in spec.parts] == ["answer", "state"]
    assert spec.parts[0].execution["evaluator"] == EVAL_V1 and "answer" in spec.parts[0].grading
    no_answer = adapter.spec(TASK, TaskTruth(has_answer=False, state_digest=STATE_A, evaluator=EVAL_V1), store)
    assert no_answer.name == "state"


def test_records_from_graded_reports(aw):
    store, adapter, bug, fix, v_bug, v_fix = aw
    rec, recorded = produce(store, adapter, v_bug, "Silver Lining", success=True)
    assert {a.role for a in rec.artifacts} == {"answer", "end_state", "commands"}
    assert [r.check for r in rec.runs] == ["answer", "state"]
    assert recorded.passed and [p.outcome for p in recorded.parts] == [Outcome.PASS, Outcome.PASS]
    assert recorded.grader_version == v_bug.version_id and recorded.method is Method.RECORDED
    failed = adapter.record_from_report({"error": "world crashed"}, store=store, task_id=TASK, trajectory_id="x",
                                        world_version=W0, spec=v_bug.entries[TASK][W0])
    assert failed[1] is None and failed[0].facts["grading_error"] == "world crashed"
    with pytest.raises(ValueError):
        adapter.record_from_report({**report("x", success=True), "other_tests_passed": False}, store=store,
                                   task_id=TASK, trajectory_id="y", world_version=W0, spec=v_bug.entries[TASK][W0])


def test_an_answer_fix_is_reparsed_from_the_submitted_answer(aw):
    store, adapter, bug, fix, v_bug, v_fix = aw
    shortcut = ScoreInput(*produce(store, adapter, v_bug, "Silver Lining", success=True))
    correct = ScoreInput(*produce(store, adapter, v_bug, "A Love That Never Was", success=False))
    broke_state = ScoreInput(*produce(store, adapter, v_bug, "A Love That Never Was", success=False, others=False,
                                      trajectory="broke"))
    unanswered = ScoreInput(*produce(store, adapter, v_bug, None, success=False, found=False, trajectory="none"))
    backend = FakeEvaluator(correct="A Love That Never Was")
    out = Scorer([adapter], store, backend).score([shortcut, correct, broke_state, unanswered], v_fix)
    assert [o.score.passed for o in out] == [False, True, False, False]
    # the recorded state verdicts are reused (their identities did not change); the answers are re-parsed, except
    # where the reused state fail already decides the task
    assert [o.action for o in out] == ["reparse", "reparse", "reuse", "reparse"]
    assert [p.method for p in out[0].score.parts] == [Method.REPARSE, Method.RECORDED] and not out[0].short_circuit
    assert out[2].score.reason == "state_failed" and out[2].short_circuit and out[2].score.skipped == ("answer",)
    assert out[3].score.reason == "no_answer_submitted"
    assert backend.requests == []  # nothing re-run


def test_a_seal_repairs_the_answer_fix(aw):
    store, adapter, bug, fix, v_bug, v_fix = aw
    members = []
    for i, (answer, ok) in enumerate((("Silver Lining", True), ("A Love That Never Was", False))):
        rec, recorded = produce(store, adapter, v_bug, answer, success=ok)
        members.append(Member(member_id=f"{i}/a0", execution=rec, prior=recorded, recorded_reward=1.0 if ok else 0.0))
    registry = MemoryRegistry()
    registry.publish_grader(v_bug)
    registry.publish_grader(v_fix)  # the fix is published before the seal
    session = SealSession(registry, Scorer([adapter], store))
    (group,) = session.decide([Candidate("u0", tuple(members))])
    assert group.rewards() == [0.0, 1.0] and [m.status for m in group.members] == ["repaired_flip", "repaired_flip"]


def test_an_evaluator_change_re_runs_the_evaluator_once_per_record(aw):
    store, adapter, bug, fix, v_bug, v_fix = aw
    patched = TaskTruth(answer="Silver Lining", state_digest=STATE_A, evaluator=EVAL_V2, data_version="data-2")
    v_eval = adapter.version({TASK: {W0: patched}}, store, name="evaluator-fix")
    item = ScoreInput(*produce(store, adapter, v_bug, "Silver Lining", success=True))
    backend = FakeEvaluator(correct="nothing matches")  # the fixed evaluator rejects the old shortcut
    out = Scorer([adapter], store, backend).score([item], v_eval)[0]
    assert out.action == "reexec" and out.score.passed is False and len(backend.requests) == 1
    accepting = FakeEvaluator(correct="Silver Lining")
    out = Scorer([adapter], store, accepting).score([item], v_eval)[0]
    assert out.score.passed and [p.method for p in out.score.parts] == [Method.REEXEC, Method.REEXEC]
    assert len(accepting.requests) == 1  # both checks asked the same request: one evaluator run
    assert out.score.cost.shared == 1


def test_a_state_truth_change_re_runs_only_the_state_tests(aw):
    store, adapter, bug, fix, v_bug, v_fix = aw
    new_state = TaskTruth(answer="Silver Lining", state_digest=STATE_B, evaluator=EVAL_V1, data_version="data-2")
    v_state = adapter.version({TASK: {W0: new_state}}, store, name="state-fix")
    item = ScoreInput(*produce(store, adapter, v_bug, "Silver Lining", success=True))
    backend = FakeEvaluator(correct="Silver Lining", others=False)
    out = Scorer([adapter], store, backend).score([item], v_state)[0]
    assert [p.method for p in out.score.parts] == [Method.RECORDED, Method.REEXEC]
    assert out.score.passed is False and out.score.reason == "state_tests_failed"


def test_missing_state_or_world_truth_is_undeterminable(aw):
    store, adapter, bug, fix, v_bug, v_fix = aw
    new_state = TaskTruth(answer="Silver Lining", state_digest=STATE_B, evaluator=EVAL_V1, data_version="data-2")
    v_state = adapter.version({TASK: {W0: new_state}}, store, name="state-fix")
    truncated = ScoreInput(*produce(store, adapter, v_bug, "Silver Lining", success=True, truncated=True))
    no_state = ScoreInput(*produce(store, adapter, v_bug, "Silver Lining", success=True, end_state=False,
                                   trajectory="no-state"))
    scorer = Scorer([adapter], store, FakeEvaluator(correct="Silver Lining"))
    for o in scorer.score([truncated, no_state], v_state):
        assert o.score.outcome is Outcome.UNDETERMINABLE and o.score.reason.startswith("missing_artifact")
    new_world_only = adapter.version({TASK: {W1: fix}}, store, name="new-world")
    o = scorer.score([truncated], new_world_only)[0]
    assert o.score.reason == "unknown_world"
    torn, recorded = adapter.record_from_report(report("Silver Lining", success=True), store=store, task_id=TASK,
                                                trajectory_id="torn", world_version=W0, spec=v_bug.entries[TASK][W0],
                                                observed_worlds=(W0, W1))
    assert scorer.score([ScoreInput(torn, recorded)], v_fix)[0].score.reason == "torn_observation"


def test_ungoverned_tasks_keep_their_recorded_verdict(aw):
    store, adapter, bug, fix, v_bug, v_fix = aw
    item = ScoreInput(*produce(store, adapter, v_bug, "Silver Lining", success=True))
    other = adapter.version({TASK: {W0: fix}}, store, name="governs-nothing", governed=["another_task"])
    out = Scorer([adapter], store).score([item], other)[0]
    assert out.action == "ungoverned" and out.score.passed


def test_evaluator_backend_request_checks(aw, tmp_path):
    store, adapter, bug, fix, v_bug, v_fix = aw
    backend = AppWorldEvaluatorBackend({"data-1": str(tmp_path)})
    rec, _ = produce(store, adapter, v_bug, "Silver Lining", success=True)
    check = v_fix.entries[TASK][W0].parts[1]
    request = adapter.reexec_request(rec, check, store)
    # the request names the whole truth the evaluator runs with, and carries the submitted answer
    assert request.execution == {"evaluator": EVAL_V1, "data_version": "data-1", "state_truth": STATE_A,
                                 "answer": v_fix.entries[TASK][W0].parts[0].grading["answer"]}
    assert request.inputs["answer"] == rec.artifact("answer").digest
    other = adapter.reexec_request(rec, v_fix.entries[TASK][W0].parts[0], store)
    assert other.request_id == request.request_id  # both checks of the task ask the one evaluator run
    alien = request.__class__(family="appworld", kind="appworld/evaluate", task_id=TASK,
                              execution={"evaluator": EVAL_V1, "data_version": "data-9"}, inputs=request.inputs)
    (result,) = backend.execute([alien], store)
    assert result.status == "error" and "data-9" in result.error
    truncated, _ = produce(store, adapter, v_bug, "x", success=False, truncated=True, trajectory="trunc")
    assert adapter.reexec_request(truncated, check, store) is None


def test_task_truth_from_dir(tmp_path):
    gt = tmp_path / "task" / "ground_truth"
    gt.mkdir(parents=True)
    (gt / "answer.json").write_text(json.dumps("42"))
    (gt / "private_data.json").write_text("{}")
    truth = task_truth_from_dir(tmp_path / "task", evaluator=EVAL_V1, data_version="d")
    assert truth.answer == "42" and truth.has_answer and truth.state_digest.startswith("sha256:")
    (gt / "private_data.json").write_text('{"changed": 1}')
    assert task_truth_from_dir(tmp_path / "task", evaluator=EVAL_V1).state_digest != truth.state_digest
    (gt / "answer.json").write_text("null")
    assert not task_truth_from_dir(tmp_path / "task", evaluator=EVAL_V1).has_answer


class EverythingPasses(CallableBackend):
    """An evaluator re-run that reports every test passed: what the evaluator of a data version whose expected answer
    is the submitted one reports."""

    def __init__(self, *, success=True, tests=None):
        self.success = success
        self.tests = tests if tests is not None else [{"requirement": "assert answers match", "passed": True},
                                                      {"requirement": "assert no extra changes", "passed": True}]
        self.requests = []
        super().__init__(self._run)

    def _run(self, request, store):
        self.requests.append(request)
        ref = EvidenceRef(role="tests", digest=put_json(store, self.tests))
        return ExecResult(request_id=request.request_id, status="ok", outputs=(ref,), facts={"success": self.success})


def test_an_answer_change_after_an_evaluator_rerun_compares_with_the_new_answer():
    """v2 patches the evaluator: both checks are re-executed, the evaluator's tests computed with v2's expected answer.
    v3 changes only the expected answer: the answer check re-parses the submitted answer against v3's, never the
    evaluator's old answer test."""
    store, adapter = MemoryStore(), AppWorldAdapter()
    truths = [TaskTruth(answer="x", state_digest=STATE_A, evaluator=EVAL_V1, data_version="d1"),
              TaskTruth(answer="x", state_digest=STATE_A, evaluator=EVAL_V2, data_version="d2"),
              TaskTruth(answer="y", state_digest=STATE_A, evaluator=EVAL_V2, data_version="d3")]
    v1, v2, v3 = (adapter.version({TASK: {W0: t}}, store, name=f"v{i + 1}") for i, t in enumerate(truths))
    rec, recorded = adapter.record_from_report(report("x", success=True), store=store, task_id=TASK,
                                               trajectory_id="t0", world_version=W0, spec=v1.entries[TASK][W0],
                                               grader_version=v1.version_id)
    backend = EverythingPasses()
    s2 = Scorer([adapter], store, backend).score([ScoreInput(rec, (recorded,))], v2)[0]
    assert s2.action == "reexec" and s2.score.passed and len(backend.requests) == 1
    run = s2.score.parts[0].run
    assert run.facts["truth"]["answer"] == v2.entries[TASK][W0].parts[0].grading["answer"]
    assert run.output("answer").digest == rec.artifact("answer").digest  # the submitted answer travels with it
    s3 = Scorer([adapter], store, backend).score([ScoreInput(rec, (recorded, s2.score))], v3)[0]
    assert s3.score.outcome is Outcome.FAIL and s3.score.reason == "answer_mismatch"
    assert [p.method for p in s3.score.parts] == [Method.REPARSE, Method.RECORDED] and len(backend.requests) == 1


def test_the_partial_tests_of_an_errored_evaluation_are_not_a_verdict():
    store, adapter = MemoryStore(), AppWorldAdapter()
    v1 = adapter.version({TASK: {W0: TaskTruth(has_answer=False, state_digest=STATE_A, evaluator=EVAL_V1,
                                              data_version="d1")}}, store, name="v1")
    errored = {"error": "RuntimeError: evaluation crashed after the first test",
               "tests": [{"requirement": "assert song liked", "passed": True}]}
    for rep in (errored, {"tests": [{"requirement": "assert song liked", "passed": True}]}):  # no success either
        rec, recorded = adapter.record_from_report(rep, store=store, task_id=TASK, trajectory_id="t0",
                                                   world_version=W0, spec=v1.entries[TASK][W0])
        assert recorded is None and rec.runs[0].status == "error"
        out = Scorer([adapter], store).score([rec], v1)[0]
        assert out.score.outcome is Outcome.UNDETERMINABLE, out.score.reason


def test_a_reexecuted_state_verdict_agrees_with_the_evaluators_own_success():
    """AppWorld's verdict is success (every test passed, none missing): an evaluator run that reports fewer tests than
    the task has is not a pass, and a failure only the answer test explains leaves the state passing."""
    store, adapter = MemoryStore(), AppWorldAdapter()
    no_answer = {TASK: {W0: TaskTruth(has_answer=False, state_digest=STATE_B, evaluator=EVAL_V1, data_version="d1")}}
    v1 = adapter.version(no_answer, store, name="v1")
    v2 = adapter.version({TASK: {W0: TaskTruth(has_answer=False, state_digest=STATE_B, evaluator=EVAL_V2,
                                              data_version="d2")}}, store, name="v2")
    failed = {"success": False, "other_tests_passed": False,
              "tests": [{"requirement": "assert song liked", "passed": False}],
              "end_state": {"truncated": False, "files": {"dbs/music.jsonl": "[]"}}}
    rec, recorded = adapter.record_from_report(failed, store=store, task_id=TASK, trajectory_id="t0",
                                               world_version=W0, spec=v1.entries[TASK][W0],
                                               grader_version=v1.version_id)
    missing = EverythingPasses(success=False, tests=[{"requirement": "assert song liked", "passed": True}])
    out = Scorer([adapter], store, missing).score([ScoreInput(rec, (recorded,))], v2)[0]
    assert out.score.outcome is Outcome.FAIL and out.score.reason == "evaluator_not_success"
    contradicted = EverythingPasses(success=True, tests=[{"requirement": "assert song liked", "passed": False}])
    out = Scorer([adapter], store, contradicted, retries=0).score([ScoreInput(rec, (recorded,))], v2)[0]
    assert out.score.outcome is Outcome.UNDETERMINABLE
    # with an answer test: the answer's failure explains success False, the state passes, the answer check fails
    v_eval = adapter.version({TASK: {W0: TaskTruth(answer="Silver Lining", state_digest=STATE_A, evaluator=EVAL_V2,
                                                  data_version="d2")}}, store, name="evaluator-fix")
    v_bug = adapter.version({TASK: {W0: TaskTruth(answer="Silver Lining", state_digest=STATE_A, evaluator=EVAL_V1,
                                                 data_version="d1")}}, store, name="bug")
    rec, recorded = produce(store, adapter, v_bug, "Silver Lining", success=True)
    wrong = EverythingPasses(success=False, tests=[{"requirement": "assert answers match", "passed": False},
                                                   {"requirement": "assert no extra changes", "passed": True}])
    out = Scorer([adapter], store, wrong).score([ScoreInput(rec, (recorded,))], v_eval)[0]
    assert out.score.outcome is Outcome.FAIL and out.score.reason == "answer_mismatch"


def test_requests_name_the_whole_truth():
    """Two truths under one data version label are two requests: a shared cache never answers one with the other."""
    store, adapter = MemoryStore(), AppWorldAdapter()
    old = adapter.spec(TASK, TaskTruth(has_answer=False, state_digest=STATE_A, evaluator=EVAL_V1,
                                       data_version="d1"), store)
    new = adapter.spec(TASK, TaskTruth(has_answer=False, state_digest=STATE_B, evaluator=EVAL_V1,
                                       data_version="d1"), store)
    rec, _ = adapter.record_from_report({"success": True, "other_tests_passed": True, "tests": [],
                                         "end_state": {"truncated": False, "files": {"dbs/x.jsonl": "[]"}}},
                                        store=store, task_id=TASK, trajectory_id="t0", world_version=W0, spec=old)
    assert adapter.reexec_request(rec, old, store).request_id != adapter.reexec_request(rec, new, store).request_id
    a = adapter.spec(TASK, TaskTruth(answer="x", state_digest=STATE_A, evaluator=EVAL_V1, data_version="d1"), store)
    b = adapter.spec(TASK, TaskTruth(answer="y", state_digest=STATE_A, evaluator=EVAL_V1, data_version="d1"), store)
    assert adapter.reexec_request(rec, a.parts[1], store).request_id != \
        adapter.reexec_request(rec, b.parts[1], store).request_id


def test_discard_keeps_a_wrong_answer_graded_under_the_batch_epoch(aw):
    store, adapter, bug, fix, v_bug, v_fix = aw
    members = []
    for i, (answer, ok) in enumerate((("Silver Lining", True), ("wrong", False))):
        rec, recorded = produce(store, adapter, v_bug, answer, success=ok, others=True, trajectory=f"t{i}")
        members.append(Member(member_id=f"t{i}", execution=rec, prior=recorded, recorded_reward=1.0 if ok else 0.0))
    registry = MemoryRegistry()
    registry.publish_grader(v_bug)
    (group,) = SealSession(registry, Scorer([adapter], store), mode="discard").decide([Candidate("u0", tuple(members))])
    assert not group.neutralized and group.rewards() == [1.0, 0.0]
    assert [m.status for m in group.members] == ["consistent", "consistent"]
