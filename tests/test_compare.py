# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from agent_ecc.compare import Action, Conjunction, Policy, combine, plan_check
from agent_ecc.records import RUN_ERROR, EvidenceRef, Method, Outcome, PartScore, TestRun
from agent_ecc.canonical import digest_bytes
from agent_ecc.versions import AllOf, Check

LOG = EvidenceRef(role="test_log", digest=digest_bytes(b"log"))


def check(run="r1", exp="e1", name="tests"):
    return Check(name=name, execution={"runner": run}, grading={"expected": exp})


def run_of(c, *, status="ok", outputs=(LOG,)):
    return TestRun(check=c.name, execution=c.execution, status=status, outputs=outputs)


def part_of(c, outcome="pass", method="recorded"):
    return PartScore(check=c.name, execution_digest=c.execution_digest, grading_digest=c.grading_digest,
                     outcome=outcome, method=method)


def test_identity_equal_reuses_the_recorded_verdict():
    c = check()
    plan = plan_check(c, runs=[run_of(c)], parts=[part_of(c)], can_reexec=True)
    assert plan.action is Action.REUSE and plan.reason == "identity_equal" and plan.prior is not None


def test_grading_change_reparses_the_recorded_run():
    old, new = check(exp="e1"), check(exp="e2")
    plan = plan_check(new, runs=[run_of(old)], parts=[part_of(old)], can_reexec=True)
    assert plan.action is Action.REPARSE and plan.reason == "grading_changed" and plan.run.execution == old.execution


def test_execution_change_reexecutes_when_the_artifact_is_held():
    old, new = check(run="r1"), check(run="r2")
    plan = plan_check(new, runs=[run_of(old)], parts=[part_of(old)], can_reexec=True)
    assert plan.action is Action.REEXEC and plan.reason == "execution_changed"
    plan = plan_check(new, runs=[run_of(old)], parts=[part_of(old)], can_reexec=False)
    assert plan.action is Action.UNDETERMINABLE and plan.reason == "missing_artifact:execution_changed"


def test_missing_outputs_cannot_be_reparsed():
    old, new = check(exp="e1"), check(exp="e2")
    plan = plan_check(new, runs=[run_of(old)], parts=[part_of(old)], output_available=lambda r: False,
                      can_reexec=True)
    assert plan.action is Action.REEXEC and plan.reason == "output_unavailable"
    # stamps and a verdict kept, but no run at all (evidence level L0+)
    plan = plan_check(new, runs=[], parts=[part_of(old)], can_reexec=False)
    assert plan.action is Action.UNDETERMINABLE and plan.reason == "missing_artifact:output_unavailable"


def test_infrastructure_errors_are_never_verdicts():
    c = check()
    infra_part = part_of(c, outcome="undeterminable", method="undeterminable")
    plan = plan_check(c, runs=[run_of(c, status=RUN_ERROR, outputs=())], parts=[infra_part], can_reexec=True)
    assert plan.action is Action.REEXEC and plan.reason == "recorded_infra_error"
    plan = plan_check(c, runs=[run_of(c, status=RUN_ERROR, outputs=())], parts=[infra_part], can_reexec=True,
                      recorded_infra="undeterminable")
    assert plan.action is Action.UNDETERMINABLE
    plan = plan_check(c, runs=[], parts=[infra_part], can_reexec=True)
    assert plan.action is Action.REEXEC and plan.reason == "recorded_infra_error"


def test_reparse_all_never_trusts_but_reexecutes_only_on_execution_change():
    c = check()
    plan = plan_check(c, runs=[run_of(c)], parts=[part_of(c)], can_reexec=True, policy="reparse_all")
    assert plan.action is Action.REPARSE and plan.reason == "reparse_all"
    other = check(run="r2")
    plan = plan_check(other, runs=[run_of(c)], parts=[part_of(c)], can_reexec=True, policy=Policy.REPARSE_ALL)
    assert plan.action is Action.REEXEC


def test_the_old_full_compatible_name_is_refused_with_guidance():
    """'full_compatible' meant re-parse everything re-parseable; a re-verification arm that re-executes every member
    is the full policy. The old name is refused rather than silently taken for either."""
    with pytest.raises(ValueError, match="reparse_all"):
        Policy("full_compatible")
    with pytest.raises(ValueError):
        plan_check(check(), policy="full_compatible")


def test_full_reexecutes_everything():
    c = check()
    plan = plan_check(c, runs=[run_of(c)], parts=[part_of(c)], can_reexec=True, policy=Policy.FULL)
    assert plan.action is Action.REEXEC and plan.reason == "full"
    plan = plan_check(c, runs=[run_of(c)], parts=[part_of(c)], can_reexec=False, policy=Policy.FULL)
    assert plan.action is Action.UNDETERMINABLE


def test_no_evidence_at_all():
    plan = plan_check(check(), can_reexec=False)
    assert plan.action is Action.UNDETERMINABLE and plan.reason == "missing_artifact:no_run"
    assert plan_check(check(), can_reexec=True).reason == "no_run"


def test_reuse_needs_both_identities_and_a_logical_part():
    c = check()
    assert plan_check(c, parts=[part_of(check(exp="e2"))], runs=[run_of(c)]).action is Action.REPARSE
    assert plan_check(c, parts=[part_of(c, outcome="undeterminable", method="undeterminable")],
                      runs=[run_of(c)]).action is Action.REPARSE


def test_combine_and_conjunction_short_circuit():
    P, F, U = Outcome.PASS, Outcome.FAIL, Outcome.UNDETERMINABLE
    assert combine([P, P]) == (P, 2) and combine([P, F, P]) == (F, 2)
    assert combine([U, F]) == (F, 2) and combine([F, U]) == (F, 1)  # monotone: a logical fail decides
    assert combine([U, P]) == (U, 2) and combine([P, U]) == (U, 2)
    a, b, d = check(name="a"), check(run="r2", name="b"), check(run="r3", name="d")
    conj = Conjunction(AllOf("all", (a, AllOf("inner", (b, d)))))
    assert conj.next_check() == a
    conj.resolve(part_of(a, "fail"))
    assert conj.done and conj.outcome is F and conj.skipped == ("b", "d") and conj.short_circuit
    conj = Conjunction(AllOf("all", (a, b)))
    conj.resolve(part_of(a, "undeterminable", "undeterminable"))
    assert not conj.done and conj.next_check() == b  # an error is not a fail, and does not hide a later one
    conj.resolve(part_of(b))
    assert conj.outcome is U and conj.skipped == () and not conj.short_circuit
    conj = Conjunction(AllOf("all", (a, b)))
    conj.resolve(part_of(a, "undeterminable", "undeterminable"))
    conj.resolve(part_of(b, "fail"))
    assert conj.outcome is F and not conj.short_circuit  # undeterminable AND fail is a fail
    conj = Conjunction(AllOf("all", (a, b)))
    conj.resolve(part_of(a))
    with pytest.raises(ValueError):
        conj.resolve(part_of(a))  # a part must resolve the check it names
    conj.resolve(part_of(b))
    assert conj.outcome is P and conj.skipped == ()
    with pytest.raises(RuntimeError):
        conj.resolve(part_of(b))
    single = Conjunction(a)
    single.resolve(part_of(a, "fail", Method.REPARSE))
    assert single.outcome is F and not single.short_circuit


def test_conjunction_resolves_out_of_order_and_without_the_short_circuit():
    P, F = Outcome.PASS, Outcome.FAIL
    a, b, d = check(name="a"), check(run="r2", name="b"), check(run="r3", name="d")
    conj = Conjunction(AllOf("all", (a, b, d)))
    conj.resolve(part_of(d, "fail"), 2)  # a later check, known for free, fails: the earlier ones are skipped
    assert conj.outcome is F and conj.skipped == ("a", "b") and conj.short_circuit
    conj.resolve(part_of(b), 1)  # free information after the decision is still recorded
    assert conj.outcome is F and [p.check for p in conj.parts] == ["b", "d"] and conj.skipped == ("a",)
    with pytest.raises(ValueError):
        conj.resolve(part_of(b), 1)
    full = Conjunction(AllOf("all", (a, b)), short_circuit=False)
    full.resolve(part_of(a, "fail"))
    assert not full.done and full.next_check() == b  # every check is evaluated
    full.resolve(part_of(b))
    assert full.outcome is F and not full.short_circuit and full.skipped == ()
    assert P is Outcome.PASS
