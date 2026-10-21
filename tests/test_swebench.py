# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SWE-bench adapter: logical verdicts, the reference-set rule, infrastructure errors, the patch filter, records,
versions, SWE-ABS conjunctions, and the official harness."""

import dataclasses
import importlib.util

import pytest

from agent_ecc.adapters.swebench import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TEST_EXIT_CODE,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    OfficialHarness,
    SimpleHarness,
    SWEBenchAdapter,
    apply_infrastructure_error,
    canonical_labels,
    classify_log,
    exclude_paths,
    filter_patch,
    official_verdict,
    parse_log,
    reference_set,
    substitute_test_patch,
    test_patch_paths,
    touched_files,
    verdict_rule,
)
from agent_ecc.records import Method, Outcome
from agent_ecc.recompute.base import CallableBackend, ExecResult
from agent_ecc.records import EvidenceRef
from agent_ecc.scoring import ScoreInput, Scorer, grade_recorded
from agent_ecc.store import MemoryStore, get_json, get_text
from agent_ecc.versions import AllOf

HAS_SWEBENCH = importlib.util.find_spec("swebench") is not None
IMAGE = "example.org/swe/eval@sha256:" + "0" * 64
F2P, P2P = ["tests/test_mod.py::test_fix"], ["tests/test_mod.py::test_keep"]
TEST_PATCH = """diff --git a/tests/test_mod.py b/tests/test_mod.py
--- a/tests/test_mod.py
+++ b/tests/test_mod.py
@@ -1,2 +1,4 @@
 def test_keep():
     assert True
+def test_fix():
+    assert fixed()
"""
HARDENED_PATCH = TEST_PATCH.replace("assert fixed()", "assert fixed() and fixed(edge=True)")


def eval_script(test_patch=TEST_PATCH, extra=""):
    return (f"#!/bin/bash\nset -uxo pipefail\ncd /testbed\ngit checkout abc123 tests/test_mod.py\n"
            f"git apply -v - <<'EOF_9f86d081884c'\n{test_patch}EOF_9f86d081884c\n"
            f": '{START_TEST_OUTPUT}'\npytest -rA tests/test_mod.py{extra}\n: '{END_TEST_OUTPUT}'\n"
            f"git checkout abc123 tests/test_mod.py\n")


def log(statuses, *, markers=True, exit_code="auto", extra=""):
    """A pytest-style log as the official eval script writes it: the results between the output markers, then the test
    command's exit code (``auto``: 1 when a test failed or errored, else 0; None: no exit-code record)."""
    body = "\n".join(f"{status} {name}" for name, status in statuses.items())
    passed = sum(1 for s in statuses.values() if s == "PASSED")
    failed = sum(1 for s in statuses.values() if s == "FAILED")
    summary = f"===== {failed} failed, {passed} passed in 0.12s =====" if statuses else ""
    text = "+ pytest -rA tests/test_mod.py\n"
    if markers:
        text += f"{START_TEST_OUTPUT}\n{body}\n{summary}\n{END_TEST_OUTPUT}\n"
    else:
        text += f"{body}\n{summary}\n"
    if exit_code == "auto":
        exit_code = 1 if any(s in ("FAILED", "ERROR") for s in statuses.values()) else 0
    if exit_code is not None:
        text += f"{TEST_EXIT_CODE}: {exit_code}\n"
    return text + extra


def row(**changes):
    base = {"instance_id": "pkg__pkg-1", "repo": "pkg/pkg", "version": "1.0", "image": IMAGE,
            "eval_script": eval_script(), "log_parser": "parse_log_pytest", "eval_type": "pass_and_fail",
            "FAIL_TO_PASS": F2P, "PASS_TO_PASS": P2P, "test_patch": TEST_PATCH}
    base.update(changes)
    return base


LABELS = canonical_labels(F2P, P2P)
SIMPLE = SimpleHarness()


def classify(text, **kw):
    kw.setdefault("harness", SIMPLE)
    return classify_log(text, labels=kw.pop("labels", LABELS), log_parser="parse_log_pytest",
                        eval_type=kw.pop("eval_type", "pass_and_fail"), instance={"instance_id": "pkg__pkg-1"}, **kw)


# ---------------------------------------------------------------------------------------------- logical verdicts


def test_resolved_and_failed_logs():
    v = classify(log({F2P[0]: "PASSED", P2P[0]: "PASSED"}, exit_code=0))
    assert v.outcome is Outcome.PASS and {t.status for t in v.tests} == {"passed"}
    v = classify(log({F2P[0]: "FAILED", P2P[0]: "PASSED"}, exit_code=1))
    assert v.outcome is Outcome.FAIL and v.reason == "tests_failed"
    assert [t.group for t in v.tests] == ["FAIL_TO_PASS", "PASS_TO_PASS"]
    assert classify(log({F2P[0]: "PASSED", P2P[0]: "SKIPPED"})).outcome is Outcome.PASS  # a skipped P2P is kept


def test_infrastructure_errors_are_not_verdicts():
    """In this order: no log; the harness's timeout marker ending the log; no start marker, or no end marker after it;
    the end marker without the exit-code record after it; a parser that raises."""
    good = {F2P[0]: "PASSED", P2P[0]: "PASSED"}
    assert classify(None).reason == "missing_log" and classify("").reason == "missing_log"
    assert classify(log(good, extra="\n\nTimeout error: 1800 seconds exceeded.")).reason == "timeout"
    assert classify(log(good, extra="\n\nTimeout error: 1800.0 seconds exceeded.\n")).reason == "timeout"
    assert classify(log(good, markers=False)).reason == "missing_output_markers"
    assert classify(f"{START_TEST_OUTPUT}\nPASSED {F2P[0]}\n").reason == "missing_output_markers"  # truncated
    assert classify(f"{END_TEST_OUTPUT}\n{START_TEST_OUTPUT}\nPASSED {F2P[0]}\n").reason == "missing_output_markers"
    assert classify(log(good, exit_code=None)).reason == "no_exit_code_record"
    record_first = f"{TEST_EXIT_CODE}: 0\n" + log(good, exit_code=None)  # the record must follow the end marker
    assert classify(record_first).reason == "no_exit_code_record"
    for text in (None, log(good, markers=False), log(good, exit_code=None)):
        assert classify(text).outcome is Outcome.UNDETERMINABLE


def test_only_the_timeout_marker_at_the_end_of_the_log_is_a_timeout():
    good = {F2P[0]: "PASSED", P2P[0]: "PASSED"}
    # the same words inside the log (a test printing them, say) are not the harness's marker
    assert classify(log(good, extra="note: Timeout error: 5 seconds exceeded. (from a test)\n")).outcome is Outcome.PASS
    assert classify("partial output\n\nTimeout error: 1800.0 seconds exceeded.").reason == "timeout"


def test_a_complete_log_the_official_reading_rejects_is_a_logical_fail():
    """A harness failure marker, no status and no sign the suite ran, or a non-zero exit code with no failing test:
    the official reading rejects the log, the status map is empty, and the run fails."""
    good = {F2P[0]: "PASSED", P2P[0]: "PASSED"}
    for text in (log(good, extra=f"\n{TESTS_TIMEOUT}\n"), log(good, extra=f"\n{TESTS_ERROR}\n"),
                 log(good, extra=f"\n{RESET_FAILED}\n"), log(good, extra=f"\n{APPLY_PATCH_FAIL}\n"),
                 log({}), log(good, exit_code=2)):
        v = classify(text)
        assert v.outcome is Outcome.FAIL and v.reason == "log_rejected", (text, v)
        assert [t.status for t in v.tests] == ["missing", "missing"]
    parsed = parse_log(log(good, extra=f"\n{TESTS_ERROR}\n"), log_parser="parse_log_pytest", instance={},
                       harness=SIMPLE)
    assert parsed.rejected and parsed.infra is None and parsed.markers == (TESTS_ERROR,) and not parsed.status_map


def test_patch_apply_failure_is_a_logical_fail():
    store = MemoryStore()
    adapter = SWEBenchAdapter(harness=SIMPLE)
    check = adapter.check_from_row(row(), store)
    stuck = adapter.run(check, store=store, log=None, applied=False)
    v = adapter.parse(check, stuck, store)
    assert v.outcome is Outcome.FAIL and v.reason == "patch_did_not_apply"


def test_an_application_stopped_by_the_executor_is_no_verdict():
    """An attempt that errored in the sandbox executor or exited with code 124, the restore between two attempts or
    the already-applied check doing so, or a strict application that stopped at it: infrastructure, never a fail."""
    genuine = {"applied": False, "attempts": [{"command": "git apply --verbose", "exit_code": 1, "error": None},
                                              {"command": "patch --batch --fuzz=5 -p1 -i", "exit_code": 1}]}
    assert apply_infrastructure_error(genuine) is None and apply_infrastructure_error(None) is None
    assert "exit 124" in apply_infrastructure_error(
        {"attempts": [{"command": "git apply --verbose", "exit_code": 124}]})
    assert "executor error" in apply_infrastructure_error(
        {"attempts": [{"command": "git apply --verbose", "exit_code": 1, "error": "broken pipe"}]})
    assert apply_infrastructure_error({"infra_error": "restore: exit_code=124"}) == "restore: exit_code=124"
    assert "executor error" in apply_infrastructure_error({"executor_error": "executor unreachable"})
    assert "exit 124" in apply_infrastructure_error({"steps": [{"command": "reset", "exit_code": 124}]})
    assert "--reverse" in apply_infrastructure_error({"attempts": [], "reverse_check": {"exit_code": 124}})
    store = MemoryStore()
    adapter = SWEBenchAdapter(harness=SIMPLE)
    logical = adapter.check_from_row(row(), store)
    official = SWEBenchAdapter(harness=SIMPLE, verdict="official").check_from_row(row(), store)
    stopped = adapter.run(logical, store=store, log=None, applied=False,
                          application={"applied": False, "infra_error": "git apply --verbose: exit_code=124"})
    v = adapter.parse(logical, stopped, store)
    assert v.outcome is Outcome.UNDETERMINABLE and v.reason.startswith("apply_infra_error")
    # the official rule reproduces a report that counts every application failure as a fail
    assert adapter.parse(official, stopped, store).outcome is Outcome.FAIL
    genuine_run = adapter.run(logical, store=store, log=None, applied=False, application=genuine)
    assert adapter.parse(logical, genuine_run, store).reason == "patch_did_not_apply"


def test_a_listed_test_without_a_status_fails():
    text = log({F2P[0]: "PASSED"})
    v = classify(text)
    assert v.outcome is Outcome.FAIL and v.reason == "tests_failed"
    assert [t.status for t in v.tests] == ["passed", "missing"]
    assert classify(log({"other": "FAILED"}), eval_type="fail_only").outcome is Outcome.PASS  # the official fail_only


def test_skips_follow_the_official_report():
    assert classify(log({F2P[0]: "PASSED", P2P[0]: "SKIPPED"})).outcome is Outcome.PASS  # a skipped P2P is maintained
    assert classify(log({F2P[0]: "SKIPPED", P2P[0]: "PASSED"})).outcome is Outcome.FAIL  # a skipped F2P fails
    assert classify(log({F2P[0]: "XFAIL", P2P[0]: "PASSED"})).outcome is Outcome.PASS


def test_the_superseded_missing_status_options_are_refused():
    for option in ({"missing_status": "undeterminable"}, {"missing_status": "fail"}, {"cross_check": False}):
        with pytest.raises(ValueError, match="no longer options"):
            SWEBenchAdapter(harness=SIMPLE, **option)
    with pytest.raises(TypeError):
        SWEBenchAdapter(harness=SIMPLE, lenient=True)
    with pytest.raises(TypeError):
        verdict_rule("logical", missing_status="undeterminable")


def test_a_parser_exception_is_not_a_verdict():
    class Broken(SimpleHarness):
        def status_map(self, *a, **k):
            raise RuntimeError("bad log")

    v = classify(log({F2P[0]: "PASSED"}), harness=Broken())
    assert v.outcome is Outcome.UNDETERMINABLE and v.reason == "parser_error:RuntimeError"

    class BrokenReport(SimpleHarness):
        def resolved(self, *a, **k):
            raise KeyError("FAIL_TO_PASS")

    v = classify(log({F2P[0]: "PASSED", P2P[0]: "PASSED"}), harness=BrokenReport())
    assert v.reason == "grading_error:KeyError"


# ---------------------------------------------------------------------------------------------- the reference set


G = ["tests/test_h.py::test_h1", "tests/test_h.py::test_h2", "tests/test_h.py::test_h3"]
G_LABELS = canonical_labels([], G)


def hardened(statuses, **kw):
    return classify(log(statuses, **kw), labels=G_LABELS, rule="reference_set")


def test_the_hardened_suite_passes_iff_every_test_of_g_passes_or_is_skipped():
    assert hardened({G[0]: "PASSED", G[1]: "PASSED", G[2]: "SKIPPED"}).outcome is Outcome.PASS
    assert hardened({G[0]: "PASSED", G[1]: "PASSED", G[2]: "SKIPPED", "other": "FAILED"}).outcome is Outcome.PASS
    v = hardened({G[0]: "PASSED", G[1]: "FAILED", G[2]: "PASSED"})
    assert v.outcome is Outcome.FAIL and v.reason == "tests_failed"
    assert [(t.name, t.status, t.group) for t in v.tests][1] == (G[1], "failed", "REFERENCE_SET")
    assert hardened({G[0]: "PASSED", G[1]: "XFAIL", G[2]: "PASSED"}).outcome is Outcome.FAIL  # XFAIL is a failure
    assert hardened({G[0]: "PASSED", G[1]: "ERROR", G[2]: "PASSED"}).outcome is Outcome.FAIL


def test_a_test_of_g_without_a_status_fails_and_ids_are_exact():
    v = hardened({G[0]: "PASSED", G[1]: "PASSED"})
    assert v.outcome is Outcome.FAIL and [t.status for t in v.tests] == ["passed", "passed", "missing"]
    truncated = canonical_labels([], ["tests/test_h.py::test_p[a"])  # no prefix matching under the reference set
    v = classify(log({"tests/test_h.py::test_p[a-b]": "PASSED"}), labels=truncated, rule="reference_set")
    assert v.outcome is Outcome.FAIL
    # a log the official reading rejects (here: a module that fails to import and runs nothing) fails
    v = classify(log({}), labels=G_LABELS, rule="reference_set")
    assert v.outcome is Outcome.FAIL and v.reason == "log_rejected"
    assert classify(log({}, exit_code=None), labels=G_LABELS, rule="reference_set").outcome is Outcome.UNDETERMINABLE


def test_ok_is_passed_and_statuses_are_case_insensitive():
    class OkHarness(SimpleHarness):
        def status_map(self, section, log, log_parser, instance):
            return {G[0]: "ok", G[1]: "OK", G[2]: "skipped"} if "PASSED" in section else {}

    v = classify(log({G[0]: "PASSED"}), labels=G_LABELS, rule="reference_set", harness=OkHarness())
    assert v.outcome is Outcome.PASS and [t.status for t in v.tests] == ["passed", "passed", "skipped"]


def test_a_reference_set_check_holds_g_as_pass_to_pass_alone():
    for labels, problem in ((canonical_labels(F2P, G), "fail_to_pass_not_empty"),
                            (canonical_labels([], []), "empty_reference_set")):
        v = classify(log({G[0]: "PASSED"}), labels=labels, rule="reference_set")
        assert v.outcome is Outcome.UNDETERMINABLE and v.final and v.reason.endswith(problem)
    store = MemoryStore()
    adapter = SWEBenchAdapter(harness=SIMPLE)
    with pytest.raises(ValueError, match="reference_set"):
        adapter.check(store=store, instance_id="pkg__pkg-1", repo="pkg/pkg", version="1.0", image=IMAGE,
                      eval_script=eval_script(), log_parser="parse_log_pytest", eval_type="pass_and_fail",
                      fail_to_pass=F2P, pass_to_pass=G, rule=verdict_rule("reference_set"))
    with pytest.raises(ValueError):
        adapter.reference_check(store=store, instance_id="pkg__pkg-1", repo="pkg/pkg", version="1.0", image=IMAGE,
                                eval_script=eval_script(), log_parser="parse_log_pytest", eval_type="pass_and_fail",
                                reference=[], patch_exclude=())
    with pytest.raises(ValueError, match="reference_check"):
        SWEBenchAdapter(harness=SIMPLE, verdict="reference_set")
    with pytest.raises(ValueError):
        classify(log({}), rule="official")  # the official rule has its own function


def gold(statuses, **kw):
    return parse_log(log(statuses, **kw), log_parser="parse_log_pytest", instance={}, harness=SIMPLE)


def test_g_is_the_tests_passing_in_both_gold_runs():
    a = gold({G[0]: "PASSED", G[1]: "PASSED", G[2]: "SKIPPED"})
    b = gold({G[0]: "PASSED", G[1]: "PASSED", "tests/test_h.py::test_only_b": "PASSED"})
    out = reference_set([a, b])
    assert out["valid"] and out["reference"] == [G[0], G[1]] and out["disagreements"] == []


def test_an_invalid_or_flaky_gold_has_no_g():
    passing = gold({G[0]: "PASSED"})
    cases = (([passing], "expected 2 gold runs"),
             ([passing, gold({G[0]: "PASSED"}, exit_code=None)], "invalid_run:infrastructure:no_exit_code_record"),
             ([passing, gold({G[0]: "PASSED"}, extra=f"\n{TESTS_ERROR}\n")], "invalid_run:log_rejected"),
             ([passing, gold({G[0]: "PASSED", G[1]: "FAILED"})], "invalid_run:gold_tests_not_passing:1"),
             ([gold({G[0]: "PASSED", G[1]: "SKIPPED"}), gold({G[0]: "PASSED", G[2]: "PASSED"})], None))
    for runs, reason in cases:
        out = reference_set(runs)
        if reason is None:
            assert out["valid"] and out["reference"] == [G[0]]
        else:
            assert not out["valid"] and out["reference"] is None and out["reason"].startswith(reason), out
    no_overlap = reference_set([gold({G[0]: "PASSED"}), gold({G[1]: "PASSED"})])
    assert not no_overlap["valid"] and no_overlap["reason"] == "empty_reference_set"
    # runs that disagree on a test they share: one run's map passes it, the other's does not (an XFAIL here)
    disagree = reference_set([gold({G[0]: "PASSED", G[1]: "PASSED"}),
                              parse_log(log({G[0]: "PASSED", G[1]: "XFAIL"}), log_parser="parse_log_pytest",
                                        instance={}, harness=SIMPLE)])
    assert not disagree["valid"] and disagree["disagreements"] == [G[1]]


def official(text, **kw):
    kw.setdefault("harness", SIMPLE)
    return official_verdict(text, labels=kw.pop("labels", LABELS), log_parser="parse_log_pytest",
                            eval_type=kw.pop("eval_type", "pass_and_fail"), instance={"instance_id": "pkg__pkg-1"},
                            **kw)


def test_the_official_rule_counts_unusable_logs_as_fails():
    good = {F2P[0]: "PASSED", P2P[0]: "PASSED"}
    assert official(log(good, exit_code=0)).outcome is Outcome.PASS
    assert official(log(good, exit_code=None)).outcome is Outcome.PASS  # the official report needs no exit-code record
    for text, reason in ((log(good, markers=False), "no_output_markers"),
                         (log(good, exit_code=2), "exit_code_contradicts_log"),
                         (log({}), "suite_did_not_run"),
                         (log(good, extra=f"\n{TESTS_ERROR}\n"), "tests_errored"),
                         (log({F2P[0]: "PASSED"}), "tests_failed"),
                         (log({F2P[0]: "FAILED", P2P[0]: "PASSED"}, exit_code=1), "tests_failed")):
        v = official(text)
        assert v.outcome is Outcome.FAIL and v.reason == reason, (reason, v)
    assert official(None).outcome is Outcome.UNDETERMINABLE and official("").reason == "missing_log"


def test_the_adapter_verdict_rule():
    store = MemoryStore()
    logical, rule = SWEBenchAdapter(harness=SIMPLE), SWEBenchAdapter(harness=SIMPLE, verdict="official")
    check = logical.version_from_rows([row()], store, name="e0").entries["pkg__pkg-1"]["*"]
    official_check = rule.version_from_rows([row()], store, name="e0").entries["pkg__pkg-1"]["*"]
    assert check.grading["verdict_rule"] == {"rule": "logical"} == verdict_rule()
    assert official_check.grading["verdict_rule"] == {"rule": "official"}
    assert check.execution_digest == official_check.execution_digest  # the same tests run
    assert check.grading_digest != official_check.grading_digest  # but their logs are judged otherwise
    good = {F2P[0]: "PASSED", P2P[0]: "PASSED"}
    truncated = logical.run(check, store=store, log=log(good, exit_code=None))
    timed_out = logical.run(check, store=store, log=log(good, exit_code=0), timed_out=True)
    contradicted = logical.run(check, store=store, log=log(good, exit_code=2))
    # a run is graded by the rule of the check it is graded under, whichever adapter instance parses it
    for adapter in (logical, rule):
        assert adapter.parse(check, truncated, store).reason == "no_exit_code_record"
        assert adapter.parse(official_check, truncated, store).outcome is Outcome.PASS
        assert adapter.parse(check, timed_out, store).reason == "timeout"
        v = adapter.parse(official_check, timed_out, store)
        assert v.outcome is Outcome.FAIL and v.reason == "timeout"
        assert adapter.parse(check, contradicted, store).reason == "log_rejected"
        assert adapter.parse(official_check, contradicted, store).outcome is Outcome.FAIL
    errored = logical.run(check, store=store, log=None, error="container did not start")
    assert rule.parse(official_check, errored, store).outcome is Outcome.UNDETERMINABLE  # no output: no verdict
    with pytest.raises(ValueError):
        SWEBenchAdapter(verdict="lenient")


def test_the_verdict_rule_is_part_of_the_grading_identity():
    store = MemoryStore()
    digests = {SWEBenchAdapter(harness=SIMPLE, **options).check_from_row(row(), store).grading_digest
               for options in ({}, {"verdict": "official"})}
    reference = SWEBenchAdapter(harness=SIMPLE).reference_check(
        store=store, instance_id="pkg__pkg-1", repo="pkg/pkg", version="1.0", image=IMAGE, eval_script=eval_script(),
        log_parser="parse_log_pytest", eval_type="pass_and_fail", reference=G, patch_exclude=["tests/test_h.py"])
    assert len(digests | {reference.grading_digest}) == 3
    # a check with a rule this version does not know is never graded
    check = SWEBenchAdapter(harness=SIMPLE).check_from_row(row(), store)
    alien = dataclasses.replace(check, grading={**check.grading, "verdict_rule": {"rule": "lenient"}})
    run = SWEBenchAdapter(harness=SIMPLE).run(alien, store=store, log=log({F2P[0]: "PASSED", P2P[0]: "PASSED"}))
    v = SWEBenchAdapter(harness=SIMPLE).parse(alien, run, store)
    assert v.outcome is Outcome.UNDETERMINABLE and v.final and v.reason.startswith("unknown_verdict_rule")
    # nor is a check built with the superseded options of an earlier logical rule
    stale = dataclasses.replace(check, grading={**check.grading, "verdict_rule": {
        "rule": "logical", "missing_status": "undeterminable", "cross_check": True}})
    v = SWEBenchAdapter(harness=SIMPLE).parse(stale, run, store)
    assert v.outcome is Outcome.UNDETERMINABLE and v.final and "superseded" in v.reason


def test_an_official_fail_is_never_reused_under_the_logical_rule():
    """A producer graded with the official rule records a run that timed out as a fail. Under a logical-rule spec the
    recorded fail has another grading identity: it is re-read under the logical rule (no verdict) and re-executed,
    so an infrastructure error never decides a SWE-ABS conjunction."""
    store = MemoryStore()
    official, logical = SWEBenchAdapter(harness=SIMPLE, verdict="official"), SWEBenchAdapter(harness=SIMPLE)
    e0 = official.version_from_rows([row()], store, name="e0")
    c0 = e0.entries["pkg__pkg-1"]["*"]
    run = official.run(c0, store=store, log=log({P2P[0]: "PASSED"}, extra="Timeout error: 1800 seconds exceeded.\n"),
                       timed_out=True)
    rec = official.record(store=store, task_id="pkg__pkg-1", trajectory_id="t", image=IMAGE, patch="diff fix",
                          runs=[run])
    recorded = grade_recorded(official, rec, c0, store, grader_version=e0.version_id)
    assert recorded.outcome is Outcome.FAIL and recorded.reason == "timeout"
    assert Scorer([official], store).score([ScoreInput(rec, recorded)], e0)[0].action == "reuse"  # same rule: reused
    e1 = logical.conjunction_version([row()], [{"instance_id": "pkg__pkg-1", "test_patch": HARDENED_PATCH,
                                                "FAIL_TO_PASS": F2P, "PASS_TO_PASS": P2P}], store, name="e1")
    backend = FakeEvaluator()
    out = Scorer([logical], store, backend).score([ScoreInput(rec, recorded)], e1)[0]
    assert not out.short_circuit and out.action == "reexec" and out.score.passed
    assert [p.method for p in out.score.parts] == [Method.REEXEC, Method.REEXEC] and len(backend.requests) == 2


def test_an_undeterminable_original_suite_does_not_hide_a_failing_hardened_suite():
    """SWE-ABS's conjunction is monotone: the hardened suite still runs after the original suite's run errored, and
    its logical fail fails the task."""
    store = MemoryStore()
    adapter = SWEBenchAdapter(harness=SIMPLE)
    e1 = adapter.conjunction_version([row()], [{"instance_id": "pkg__pkg-1", "test_patch": HARDENED_PATCH,
                                                "FAIL_TO_PASS": F2P, "PASS_TO_PASS": P2P}], store, name="e1")
    base, hardened = e1.entries["pkg__pkg-1"]["*"].parts
    runs = [adapter.run(base, store=store, log=None, error="pod evicted"),
            adapter.run(hardened, store=store, log=log({F2P[0]: "FAILED", P2P[0]: "PASSED"}, exit_code=1))]
    rec = adapter.record(store=store, task_id="pkg__pkg-1", trajectory_id="t", image=IMAGE, patch="diff fragile fix",
                         runs=runs)
    recorded = grade_recorded(adapter, rec, e1.entries["pkg__pkg-1"]["*"], store, grader_version=e1.version_id)
    assert recorded.outcome is Outcome.FAIL and [p.outcome for p in recorded.parts] == [
        Outcome.UNDETERMINABLE, Outcome.FAIL]
    backend = FakeEvaluator(error=True)
    out = Scorer([adapter], store, backend).score([ScoreInput(rec, recorded)], e1)[0]
    # the recorded hardened fail decides: the errored original suite is not executed again
    assert out.score.outcome is Outcome.FAIL and out.action == "reuse" and backend.requests == []
    assert out.score.skipped == ("tests",) and out.short_circuit


def test_patch_exclusion():
    patch = ("diff --git a/src/mod.py b/src/mod.py\n--- a/src/mod.py\n+++ b/src/mod.py\n@@ -1 +1 @@\n-a\n+b\n"
             "diff --git a/tests/test_mod.py b/tests/test_mod.py\n--- a/tests/test_mod.py\n+++ b/tests/test_mod.py\n"
             "@@ -1 +1 @@\n-x\n+y\n")
    assert exclude_paths(patch, ["tests/test_mod.py"]) == patch.split("diff --git a/tests")[0]
    assert exclude_paths(patch, ["other.py"]) == patch
    assert exclude_paths(patch, ["src/mod.py", "tests/test_mod.py"]) == ""
    renamed = "diff --git a/old.py b/new.py\nsimilarity index 90%\nrename from old.py\nrename to new.py\n"
    assert exclude_paths(renamed, ["old.py"]) == "" and exclude_paths(renamed, ["new.py"]) == ""
    filtered, record = filter_patch(patch, ["tests/test_mod.py"])
    assert record["removed_files"] == ["tests/test_mod.py"] and record["removed_diffs"] == 1
    assert record["kept_diffs"] == 1 and not record["empty_after_filter"] and filtered.startswith("diff --git a/src")
    assert filter_patch(patch, ["src/mod.py", "tests/test_mod.py"])[1]["empty_after_filter"]
    # a plain unified diff splits at ---/+++/@@ headers only; text with no recognisable diff is applied as it is
    plain = "--- a/src/mod.py\n+++ b/src/mod.py\n@@ -1 +1 @@\n-a\n+b\n--- a/t.py\n+++ b/t.py\n@@ -1 +1 @@\n-x\n+y\n"
    assert exclude_paths(plain, ["t.py"]) == plain.split("--- a/t.py")[0]
    headless = "--- a/t.py\n+++ b/t.py\n-x\n+y\n"  # no hunk header: not a per-file diff
    assert exclude_paths(headless, ["t.py"]) == headless
    assert exclude_paths("not a patch at all\n", ["t.py"]) == "not a patch at all\n"
    preamble = "From: someone\n\n" + patch
    assert exclude_paths(preamble, ["src/mod.py", "tests/test_mod.py"]) == ""  # only text that touches no file is left
    quoted = 'diff --git "a/tests/t\\303\\251.py" "b/tests/t\\303\\251.py"\n--- "a/tests/t\\303\\251.py"\n' \
             '+++ "b/tests/t\\303\\251.py"\n@@ -1 +1 @@\n-x\n+y\n'
    assert exclude_paths(quoted, ["tests/t\u00e9.py"]) == ""


    store = MemoryStore()
    adapter = SWEBenchAdapter(harness=SIMPLE)
    e1 = adapter.conjunction_version([row()], [{"instance_id": "pkg__pkg-1", "test_patch": HARDENED_PATCH,
                                                "FAIL_TO_PASS": F2P, "PASS_TO_PASS": P2P,
                                                "patch_exclude": ["tests/test_mod.py"]}], store, name="e1")
    base, hardened = e1.entries["pkg__pkg-1"]["*"].parts
    assert "patch_exclude" in hardened.execution and "patch_exclude" not in base.execution
    plain = adapter.conjunction_version([row()], [{"instance_id": "pkg__pkg-1", "test_patch": HARDENED_PATCH,
                                                   "FAIL_TO_PASS": F2P, "PASS_TO_PASS": P2P}], store, name="e1'")
    assert plain.entries["pkg__pkg-1"]["*"].parts[1].execution_digest != hardened.execution_digest
    rec = adapter.record(store=store, task_id="pkg__pkg-1", trajectory_id="t", image=IMAGE, patch=patch)
    request = adapter.reexec_request(rec, hardened, store)
    assert get_text(store, request.inputs["patch"]) == exclude_paths(patch, ["tests/test_mod.py"])
    assert request.params["patch_exclude"] == hardened.execution["patch_exclude"]
    assert adapter.reexec_request(rec, base, store).inputs["patch"] == rec.artifact("patch").digest


def test_the_paths_a_test_patch_touches():
    assert test_patch_paths(TEST_PATCH) == ["tests/test_mod.py"]
    created = ("diff --git a/tests/test_new.py b/tests/test_new.py\nnew file mode 100644\n--- /dev/null\n"
               "+++ b/tests/test_new.py\n@@ -0,0 +1 @@\n+def test_h(): pass\n"
               "diff --git a/tests/old.py b/tests/moved.py\nsimilarity index 90%\nrename from tests/old.py\n"
               "rename to tests/moved.py\n")
    assert test_patch_paths(created) == ["tests/moved.py", "tests/old.py", "tests/test_new.py"]
    assert test_patch_paths("") == []


# ---------------------------------------------------------------------------------------------- eval scripts


def test_substitute_test_patch():
    script = eval_script()
    new = substitute_test_patch(script, HARDENED_PATCH, original_test_patch=TEST_PATCH)
    assert new == eval_script(HARDENED_PATCH)  # only the heredoc body changed
    assert new.count("edge=True") == 1
    assert touched_files(TEST_PATCH) == ["tests/test_mod.py"]
    other = HARDENED_PATCH.replace("tests/test_mod.py", "tests/test_other.py")
    with pytest.raises(ValueError, match="other files"):
        substitute_test_patch(script, other)
    with pytest.raises(ValueError, match="original"):
        substitute_test_patch(script, HARDENED_PATCH, original_test_patch="diff --git a/x b/x\n")
    with pytest.raises(ValueError, match="heredoc"):
        substitute_test_patch("#!/bin/bash\necho no heredoc\n", HARDENED_PATCH)


# ---------------------------------------------------------------------------------------------- adapter flows


@pytest.fixture
def swe():
    store = MemoryStore()
    adapter = SWEBenchAdapter(harness=SIMPLE)
    return store, adapter


def produce(store, adapter, check, patch, statuses, *, trajectory="traj", applied=True, error=None,
            exit_code="auto"):
    run = adapter.run(check, store=store, log=None if statuses is None else log(statuses, exit_code=exit_code),
                      applied=applied, error=error)
    rec = adapter.record(store=store, task_id="pkg__pkg-1", trajectory_id=trajectory, image=IMAGE, patch=patch,
                         runs=[run])
    return rec, grade_recorded(adapter, rec, check, store)


class FakeEvaluator(CallableBackend):
    """Executes SWE-bench requests by reading the script: the hardened suite fails a patch marked 'fragile'."""

    def __init__(self, *, error=False):
        self.requests = []
        self.error = error
        super().__init__(self._run)

    def _run(self, request, store):
        self.requests.append(request)
        if self.error:
            raise RuntimeError("container did not start")
        patch, script = get_text(store, request.inputs["patch"]), get_text(store, request.inputs["eval_script"])
        fixed = "fix" in patch and not ("fragile" in patch and "edge=True" in script)
        text = log({F2P[0]: "PASSED" if fixed else "FAILED", P2P[0]: "PASSED"}, exit_code=0 if fixed else 1)
        data = text.encode()
        ref = EvidenceRef(role="test_log", digest=store.put(data), size=len(data))
        return ExecResult(request_id=request.request_id, status="ok", outputs=(ref,), facts={"applied": True})


def test_records_and_recorded_verdicts(swe):
    store, adapter = swe
    v = adapter.version_from_rows([row()], store, name="e0")
    check = v.entries["pkg__pkg-1"]["*"]
    assert set(check.execution) == {"image", "eval_script", "apply_cmds"}
    assert set(check.grading) == {"labels", "parser", "grading_code", "eval_type", "verdict_rule"}
    rec, recorded = produce(store, adapter, check, "diff fix", {F2P[0]: "PASSED", P2P[0]: "PASSED"})
    assert rec.world_version == IMAGE and rec.artifact("patch") is not None and recorded.passed
    assert rec.runs[0].facts["has_output_markers"] is True
    empty, recorded_empty = produce(store, adapter, check, "   ", None)
    assert empty.facts["patch_empty"] and recorded_empty.reason == "no_agent_patch" and not recorded_empty.passed
    with pytest.raises(ValueError):
        adapter.check_from_row({**row(), "eval_script": ""}, store)


def test_label_change_reparses_and_script_change_reexecutes(swe):
    store, adapter = swe
    e0 = adapter.version_from_rows([row()], store, name="e0")
    check = e0.entries["pkg__pkg-1"]["*"]
    rec, recorded = produce(store, adapter, check, "diff fix", {F2P[0]: "PASSED", P2P[0]: "FAILED"})
    assert not recorded.passed
    backend = FakeEvaluator()
    scorer = Scorer([adapter], store, backend)
    relabel = adapter.version_from_rows([row(PASS_TO_PASS=[])], store, name="e1")  # a P2P label removed
    out = scorer.score([ScoreInput(rec, recorded)], relabel)[0]
    assert out.action == "reparse" and out.score.passed and backend.requests == []
    rescript = adapter.version_from_rows([row(eval_script=eval_script(extra=" -p no:cacheprovider"))], store,
                                         name="e2")
    out = scorer.score([ScoreInput(rec, recorded)], rescript)[0]
    assert out.action == "reexec" and out.score.passed and len(backend.requests) == 1
    (request,) = backend.requests
    assert request.inputs["patch"] == rec.artifact("patch").digest
    assert request.params["log_parser"] == "parse_log_pytest" and request.execution == rescript.entries[
        "pkg__pkg-1"]["*"].execution


def test_swe_abs_conjunction(swe):
    store, adapter = swe
    base_rows = [row()]
    e0 = adapter.version_from_rows(base_rows, store, name="e0")
    e1 = adapter.conjunction_version(base_rows, [{"instance_id": "pkg__pkg-1", "test_patch": HARDENED_PATCH,
                                                  "FAIL_TO_PASS": F2P, "PASS_TO_PASS": P2P}], store, name="e1")
    spec = e1.entries["pkg__pkg-1"]["*"]
    assert isinstance(spec, AllOf) and [c.name for c in spec.parts] == ["tests", "hardened"]
    assert spec.parts[0] == e0.entries["pkg__pkg-1"]["*"]  # the original suite is the e0 check itself
    check = spec.parts[0]
    passing, p_rec = produce(store, adapter, check, "diff fix", {F2P[0]: "PASSED", P2P[0]: "PASSED"},
                             trajectory="pass")
    fragile, f_rec = produce(store, adapter, check, "diff fix fragile", {F2P[0]: "PASSED", P2P[0]: "PASSED"},
                             trajectory="fragile")
    failing, x_rec = produce(store, adapter, check, "diff nothing", {F2P[0]: "FAILED", P2P[0]: "PASSED"},
                             trajectory="fail")
    errored, e_rec = produce(store, adapter, check, "diff fix", None, trajectory="error", error="pod evicted")
    assert e_rec.outcome is Outcome.UNDETERMINABLE
    backend = FakeEvaluator()
    out = Scorer([adapter], store, backend).score(
        [ScoreInput(passing, p_rec), ScoreInput(fragile, f_rec), ScoreInput(failing, x_rec),
         ScoreInput(errored, e_rec)], e1)
    p, f, x, e = out
    assert p.score.passed and [q.method for q in p.score.parts] == [Method.RECORDED, Method.REEXEC]
    assert not f.score.passed and f.score.reason == "tests_failed"  # rejected by the hardened tests
    assert x.short_circuit and x.action == "reuse" and x.score.skipped == ("hardened",)  # no execution
    assert e.score.passed and [q.method for q in e.score.parts] == [Method.REEXEC, Method.REEXEC]
    kinds = sorted("edge=True" in get_text(store, r.inputs["eval_script"]) for r in backend.requests)
    assert kinds == [False, True, True]  # one base re-run (the errored record), two distinct hardened runs


def test_conjunction_from_explicit_eval_scripts_and_missing_hardened_rows(swe):
    store, adapter = swe
    rows = [row(), row(instance_id="pkg__pkg-2")]
    hardened = [{"instance_id": "pkg__pkg-1", "eval_script": eval_script(HARDENED_PATCH),
                 "FAIL_TO_PASS": F2P, "PASS_TO_PASS": P2P}]
    v = adapter.conjunction_version(rows, hardened, store, name="e1")
    assert isinstance(v.entries["pkg__pkg-1"]["*"], AllOf)
    assert not isinstance(v.entries["pkg__pkg-2"]["*"], AllOf)
    with pytest.raises(ValueError):
        adapter.conjunction_version(rows, [{"instance_id": "pkg__pkg-1", "FAIL_TO_PASS": F2P,
                                            "PASS_TO_PASS": P2P}], store, name="bad")


def test_empty_patch_and_patch_not_applied(swe):
    store, adapter = swe
    e0 = adapter.version_from_rows([row()], store, name="e0")
    check = e0.entries["pkg__pkg-1"]["*"]
    empty, recorded = produce(store, adapter, check, None, None)
    rescript = adapter.version_from_rows([row(eval_script=eval_script(extra=" -x"))], store, name="e2")
    backend = FakeEvaluator()
    out = Scorer([adapter], store, backend).score([ScoreInput(empty, recorded)], rescript)[0]
    assert not out.score.passed and out.score.reason == "no_agent_patch" and backend.requests == []
    stuck, rec2 = produce(store, adapter, check, "diff fix", None, applied=False, trajectory="stuck")
    assert rec2.outcome is Outcome.FAIL and rec2.reason == "patch_did_not_apply"


def test_a_recorded_apply_failure_carries_over_to_another_eval_script(swe):
    """Whether a patch applies depends on the image, the application commands and the patch filter, not on the eval
    script: a recorded application failure decides a suite with another script by reuse (ecc policy), never one with
    another patch filter, and never when the recorded application stopped at an executor error."""
    store, adapter = swe
    e0 = adapter.version_from_rows([row()], store, name="e0")
    check = e0.entries["pkg__pkg-1"]["*"]
    stuck, recorded = produce(store, adapter, check, "diff fix", None, applied=False, trajectory="stuck")
    assert recorded.outcome is Outcome.FAIL and recorded.reason == "patch_did_not_apply"
    rescript = adapter.version_from_rows([row(eval_script=eval_script(extra=" -x"))], store, name="e2")
    backend = FakeEvaluator()
    out = Scorer([adapter], store, backend).score([ScoreInput(stuck, recorded)], rescript)[0]
    assert out.action == "reuse" and out.score.reason == "patch_did_not_apply" and backend.requests == []
    assert out.score.based_on is None  # not the recorded verdict of this suite
    (part,) = out.score.parts  # carried over: the part names the recorded run it comes from, of the other script
    assert part.method.value == "recorded" and part.run == stuck.runs[0]
    assert part.run.execution_digest != rescript.entries["pkg__pkg-1"]["*"].execution_digest
    assert out.score.new_runs() == ()  # provenance, not new evidence
    # the same suite under other labels (a label fix): the application failure carries over too, by reuse
    relabel = adapter.version_from_rows([row(PASS_TO_PASS=[])], store, name="e0-labels")
    out = Scorer([adapter], store, backend).score([ScoreInput(stuck, recorded)], relabel)[0]
    assert out.action == "reuse" and out.score.parts[0].run == stuck.runs[0] and backend.requests == []
    for policy in ("full", "reparse_all"):  # a re-verification executes it
        out = Scorer([adapter], store, backend, policy=policy).score([ScoreInput(stuck, recorded)], rescript)[0]
        assert out.action == "reexec"
    assert len(backend.requests) == 2
    filtered = adapter.conjunction_version([row()], [{"instance_id": "pkg__pkg-1", "test_patch": HARDENED_PATCH,
                                                      "FAIL_TO_PASS": F2P, "PASS_TO_PASS": P2P,
                                                      "patch_exclude": ["tests/test_mod.py"]}], store, name="e1")
    assert adapter.carry_over(stuck, filtered.entries["pkg__pkg-1"]["*"].parts[1]) is None
    stopped_run = adapter.run(check, store=store, log=None, applied=False,
                              application={"applied": False, "infra_error": "git apply --verbose: exit_code=124"})
    stopped = adapter.record(store=store, task_id="pkg__pkg-1", trajectory_id="stopped", image=IMAGE,
                             patch="diff fix", runs=[stopped_run])
    assert adapter.carry_over(stopped, rescript.entries["pkg__pkg-1"]["*"]) is None


def test_hardened_suite_arguments_are_explicit():
    """A reference-set suite names the paths its patch filter removes (SWE-ABS's first step) explicitly; repository
    paths are a collection, never a string (which would be read character by character); a verdict rule is a name or
    a mapping wherever one is taken."""
    store = MemoryStore()
    adapter = SWEBenchAdapter(harness=SIMPLE)
    common = dict(store=store, instance_id="pkg__pkg-1", repo="pkg/pkg", version="1.0", image=IMAGE,
                  eval_script=eval_script(), log_parser="parse_log_pytest", eval_type="pass_and_fail")
    with pytest.raises(TypeError):
        adapter.reference_check(**common, reference=G)  # no patch_exclude: refused, never silently empty
    for call in (lambda: adapter.reference_check(**common, reference=G, patch_exclude="tests/test_h.py"),
                 lambda: adapter.check(**common, fail_to_pass=F2P, pass_to_pass=P2P, patch_exclude="tests/test_h.py"),
                 lambda: filter_patch(HARDENED_PATCH, "tests/test_mod.py"),
                 lambda: exclude_paths(HARDENED_PATCH, "tests/test_mod.py")):
        with pytest.raises(TypeError, match="not a string"):
            call()
    by_mapping = adapter.reference_check(**common, reference=G, patch_exclude=["tests/test_h.py"])
    by_name = adapter.check(**common, fail_to_pass=[], pass_to_pass=G, name=by_mapping.name, rule="reference_set",
                            patch_exclude=["tests/test_h.py"])
    assert by_name == by_mapping
    text = log({G[0]: "PASSED", G[1]: "PASSED", G[2]: "XFAIL"})
    assert classify(text, labels=G_LABELS, rule={"rule": "reference_set"}) == \
        classify(text, labels=G_LABELS, rule="reference_set")
    assert SWEBenchAdapter(harness=SIMPLE, verdict={"rule": "official"}).rule == verdict_rule("official")
    with pytest.raises(ValueError):
        classify(text, labels=G_LABELS, rule={"rule": "lenient"})
    with pytest.raises(ValueError):
        adapter.check(**common, fail_to_pass=F2P, pass_to_pass=P2P, rule="lenient")


def test_a_hardened_row_with_a_reference_set(swe):
    store, adapter = swe
    hardened_script = eval_script(HARDENED_PATCH, extra=" tests/test_h.py")
    h = {"instance_id": "pkg__pkg-1", "eval_script": hardened_script, "test_patch": HARDENED_PATCH, "reference_set": G}
    e1 = adapter.conjunction_version([row()], [h], store, name="e1")
    base, hard = e1.entries["pkg__pkg-1"]["*"].parts
    assert base == adapter.check_from_row(row(), store) and hard.name == "hardened"
    assert hard.grading["verdict_rule"] == {"rule": "reference_set"}
    assert get_json(store, hard.grading["labels"]) == {"FAIL_TO_PASS": [], "PASS_TO_PASS": sorted(G)}
    assert get_json(store, hard.execution["patch_exclude"]) == test_patch_paths(HARDENED_PATCH) == ["tests/test_mod.py"]
    explicit = adapter.conjunction_version([row()], [{**h, "patch_exclude": ["tests/test_h.py"]}], store, name="x")
    assert get_json(store, explicit.entries["pkg__pkg-1"]["*"].parts[1].execution["patch_exclude"]) == [
        "tests/test_h.py"]
    with pytest.raises(ValueError, match="eval_script"):
        adapter.conjunction_version([row()], [{k: v for k, v in h.items() if k != "eval_script"}], store, name="y")
    with pytest.raises(ValueError, match="patch_exclude"):
        adapter.conjunction_version([row()], [{k: v for k, v in h.items() if k != "test_patch"}], store, name="z")
    # the agent's own edit of the hardened test file leaves the patch under test; G decides
    patch = ("diff --git a/src/mod.py b/src/mod.py\n--- a/src/mod.py\n+++ b/src/mod.py\n@@ -1 +1 @@\n-a\n+fix\n"
             "diff --git a/tests/test_mod.py b/tests/test_mod.py\n--- a/tests/test_mod.py\n+++ b/tests/test_mod.py\n"
             "@@ -1 +1 @@\n-x\n+y\n")
    rec, recorded = produce(store, adapter, base, patch, {F2P[0]: "PASSED", P2P[0]: "PASSED"})

    def evaluator(request, store_):
        sent = get_text(store_, request.inputs["patch"])
        assert "tests/test_mod.py" not in sent and "+fix" in sent
        text = log({G[0]: "PASSED", G[1]: "SKIPPED", G[2]: "FAILED"})
        data = text.encode()
        return ExecResult(request_id=request.request_id, status="ok", facts={"applied": True},
                          outputs=(EvidenceRef(role="test_log", digest=store_.put(data), size=len(data)),))

    out = Scorer([adapter], store, CallableBackend(evaluator)).score([ScoreInput(rec, recorded)], e1)[0]
    assert out.action == "reexec" and out.score.outcome is Outcome.FAIL and out.score.reason == "tests_failed"
    assert [p.method for p in out.score.parts] == [Method.RECORDED, Method.REEXEC]


def test_a_target_grader_that_is_not_importable_is_final(swe):
    store, adapter = swe
    e0 = adapter.version_from_rows([row()], store, name="e0")
    check = e0.entries["pkg__pkg-1"]["*"]
    rec, recorded = produce(store, adapter, check, "diff fix", {F2P[0]: "PASSED", P2P[0]: "PASSED"})

    class OtherParser(SimpleHarness):
        def identity(self, log_parser):
            return {"parser": "sha256:" + "f" * 64, "grading_code": "sha256:" + "e" * 64}

    other = SWEBenchAdapter(harness=OtherParser()).version_from_rows([row(PASS_TO_PASS=[])], store, name="other")
    backend = FakeEvaluator()
    out = Scorer([adapter], store, backend).score([ScoreInput(rec, recorded)], other)[0]
    assert out.score.reason == "grader_unavailable" and backend.requests == []


def test_backend_failures_leave_the_member_undeterminable(swe):
    store, adapter = swe
    e0 = adapter.version_from_rows([row()], store, name="e0")
    rec, recorded = produce(store, adapter, e0.entries["pkg__pkg-1"]["*"], "diff fix",
                            {F2P[0]: "PASSED", P2P[0]: "PASSED"})
    rescript = adapter.version_from_rows([row(eval_script=eval_script(extra=" -x"))], store, name="e2")
    backend = FakeEvaluator(error=True)
    out = Scorer([adapter], store, backend, retries=2).score([ScoreInput(rec, recorded)], rescript)[0]
    assert out.score.outcome is Outcome.UNDETERMINABLE and len(backend.requests) == 3
    assert out.score.reason.startswith("infra_error")


# ---------------------------------------------------------------------------------------------- the official harness


@pytest.mark.swebench
@pytest.mark.skipif(not HAS_SWEBENCH, reason="the official swebench harness is not installed")
class TestOfficialHarness:
    def test_identity(self):
        h = OfficialHarness()
        ident = h.identity("parse_log_pytest")
        assert ident["parser"].startswith("sha256:") and ident["grading_code"].startswith("sha256:")
        assert h.identity("parse_log_pytest") == ident and h.apply_cmds_digest().startswith("sha256:")
        assert "git apply --verbose" in h.git_apply_cmds()
        if h.ns.version == "5.0.2":
            # the grading sources of release 5.0.2 (the digest a SWE-bench Verified grader epoch would stamp)
            assert ident["grading_code"] == "sha256:e29ffb8387c5a15b709cb1e98369222d9ebfaab506bda7b0bc99443b1ad555e3"

    def test_verdicts_agree_with_the_official_report(self):
        h = OfficialHarness()
        for statuses, expected in (({F2P[0]: "PASSED", P2P[0]: "PASSED"}, Outcome.PASS),
                                   ({F2P[0]: "FAILED", P2P[0]: "PASSED"}, Outcome.FAIL),
                                   ({F2P[0]: "PASSED", P2P[0]: "ERROR"}, Outcome.FAIL)):
            v = classify(log(statuses, exit_code=0 if expected is Outcome.PASS else 1), harness=h)
            assert v.outcome is expected, (statuses, v)
        v = classify(log({F2P[0]: "PASSED"}), harness=h)  # a listed test without a status fails
        assert v.outcome is Outcome.FAIL and v.reason == "tests_failed"
        v = classify(log({}), harness=h)  # nothing ran: the official reading rejects the log
        assert v.outcome is Outcome.FAIL and v.reason == "log_rejected"

    def test_the_logical_rule_is_the_official_report_on_every_complete_log(self):
        """On a log with its markers and its exit-code record, the logical rule's verdict is exactly the official
        get_eval_report's; the logs it keeps out of verdicts are the infrastructure shapes."""
        h = OfficialHarness()
        good = {F2P[0]: "PASSED", P2P[0]: "PASSED"}
        complete = (log(good), log(good, exit_code=2), log({}), log({F2P[0]: "PASSED"}),
                    log({F2P[0]: "FAILED", P2P[0]: "PASSED"}), log({F2P[0]: "PASSED", P2P[0]: "SKIPPED"}),
                    log({F2P[0]: "SKIPPED", P2P[0]: "PASSED"}), log({F2P[0]: "XFAIL", P2P[0]: "PASSED"}),
                    log(good, extra=f"\n{TESTS_ERROR}\n"), log(good, extra=f"\n{APPLY_PATCH_FAIL}\n"),
                    log({F2P[0]: "PASSED", P2P[0]: "ERROR"}), log({"tests/test_mod.py::other": "PASSED"}))
        for text in complete:
            v = classify(text, harness=h)
            report = h.official_resolved(text, LABELS, "pass_and_fail",
                                         {"instance_id": "pkg__pkg-1", "log_parser": "parse_log_pytest"})
            assert v.outcome is (Outcome.PASS if report else Outcome.FAIL), (text, v)
        for text in (log(good, exit_code=None), log(good, markers=False),
                     log(good, extra="\n\nTimeout error: 1800 seconds exceeded.")):
            assert classify(text, harness=h).outcome is Outcome.UNDETERMINABLE, text

    def test_the_reference_set_rule_with_the_official_parser(self):
        h = OfficialHarness()
        text = log({G[0]: "PASSED", G[1]: "SKIPPED", G[2]: "PASSED"})
        assert classify(text, labels=G_LABELS, rule="reference_set", harness=h).outcome is Outcome.PASS
        text = log({G[0]: "PASSED", G[1]: "XFAIL", G[2]: "PASSED"})
        assert classify(text, labels=G_LABELS, rule="reference_set", harness=h).outcome is Outcome.FAIL

    def test_django_parser(self):
        h = OfficialHarness()
        labels = canonical_labels(["test_fix (app.tests.T)"], ["test_keep (app.tests.T)"])
        text = (f"{START_TEST_OUTPUT}\ntest_fix (app.tests.T) ... ok\ntest_keep (app.tests.T) ... ok\n\n"
                f"Ran 2 tests in 0.01s\n\nOK\n{END_TEST_OUTPUT}\n{TEST_EXIT_CODE}: 0\n")
        v = classify_log(text, labels=labels, log_parser="parse_log_django", eval_type="pass_and_fail",
                         instance={"instance_id": "django__django-1"}, harness=h)
        assert v.outcome is Outcome.PASS
        v = classify_log(text.replace("test_fix (app.tests.T) ... ok", "test_fix (app.tests.T) ... FAIL"),
                         labels=labels, log_parser="parse_log_django", eval_type="pass_and_fail",
                         instance={"instance_id": "django__django-1"}, harness=h)
        assert v.outcome is Outcome.FAIL

    def test_official_rule_matches_the_official_report(self):
        h = OfficialHarness()
        good = {F2P[0]: "PASSED", P2P[0]: "PASSED"}
        cases = (log(good, exit_code=0), log(good, exit_code=2), log({}), log({F2P[0]: "PASSED"}),
                 log(good, markers=False), log({F2P[0]: "FAILED", P2P[0]: "PASSED"}, exit_code=1))
        for text in cases:
            v = official_verdict(text, labels=LABELS, log_parser="parse_log_pytest", eval_type="pass_and_fail",
                                 instance={"instance_id": "pkg__pkg-1"}, harness=h)
            report = h.official_resolved(text, LABELS, "pass_and_fail",
                                         {"instance_id": "pkg__pkg-1", "log_parser": "parse_log_pytest"})
            assert v.outcome is (Outcome.PASS if report else Outcome.FAIL), (text, v)

    def test_adapter_with_the_official_harness(self):
        store = MemoryStore()
        adapter = SWEBenchAdapter()
        v = adapter.version_from_rows([row()], store, name="e0")
        check = v.entries["pkg__pkg-1"]["*"]
        assert check.params["harness"].startswith("swebench/")
        rec, recorded = produce(store, adapter, check, "diff fix", {F2P[0]: "PASSED", P2P[0]: "PASSED"}, exit_code=0)
        assert recorded.passed
        relabel = adapter.version_from_rows([row(FAIL_TO_PASS=F2P + ["tests/test_mod.py::test_new"])], store,
                                            name="e1")
        out = Scorer([adapter], store).score([ScoreInput(rec, recorded)], relabel)[0]
        # the new label names a test the stored log never ran: it counts as failed, decided by re-parsing the log
        assert out.action == "reparse" and out.score.outcome is Outcome.FAIL and out.score.reason == "tests_failed"
