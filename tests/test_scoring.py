# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from agent_ecc.compare import Action, Policy
from agent_ecc.records import Method, Outcome, PartScore, ScoreRecord
from agent_ecc.scoring import ScoreInput, Scorer, grade_recorded
from agent_ecc.store import MemoryStore
from agent_ecc.versions import GraderVersion

from toy import ToyAdapter, ToyBackend, check, conj, oracle_verdict


@pytest.fixture
def env():
    store = MemoryStore()
    adapter = ToyAdapter()
    backend = ToyBackend()
    return store, adapter, backend, Scorer([adapter], store, backend)


def graded(store, adapter, spec, answer, *, task="t", trajectory=None, version=None, keep_output=True,
           error=False, world="w0"):
    """A record graded under ``spec`` when produced: its runs and its recorded score."""
    from agent_ecc.versions import iter_checks

    runs = [adapter.run(store, c, answer, keep_output=keep_output, error=error) for c in iter_checks(spec)]
    record = adapter.record(store, task=task, trajectory=trajectory or f"{task}-{answer}", answer=answer, runs=runs,
                            world=world)
    return record, grade_recorded(adapter, record, spec, store, grader_version=version)


def version(name, spec, task="t", world="*"):
    return GraderVersion(name=name, entries={task: {world: spec}})


def test_unchanged_grader_reuses_every_verdict(env):
    store, adapter, backend, scorer = env
    c1 = check("t", "plain", "ab", store)
    g1 = version("g1", c1)
    items = [ScoreInput(*graded(store, adapter, c1, a, version=g1.version_id)) for a in ("ab", "ba")]
    out = scorer.score(items, g1)
    assert [o.action for o in out] == ["reuse", "reuse"]
    assert [o.score.outcome for o in out] == [Outcome.PASS, Outcome.FAIL]
    assert all(o.score.method is Method.RECORDED and o.score.based_on == i.prior[0].score_id
               for o, i in zip(out, items))
    assert backend.executed == [] and adapter.parsed == 2  # the two parses were the producer's own grading


def test_grading_change_reparses_and_execution_change_reexecutes(env):
    store, adapter, backend, scorer = env
    c1 = check("t", "plain", "ab", store)
    items = [ScoreInput(*graded(store, adapter, c1, a)) for a in ("ab", "ba", "zz")]
    relabel = version("g2", check("t", "plain", "ba", store))
    out = scorer.score(items, relabel)
    assert [o.action for o in out] == ["reparse"] * 3
    assert [o.score.passed for o in out] == [False, True, False] and backend.executed == []
    rerun = version("g3", check("t", "rev", "ab", store))
    out = scorer.score(items, rerun)
    assert [o.action for o in out] == ["reexec"] * 3
    assert [o.score.passed for o in out] == [False, True, False]
    assert len(backend.executed) == 3 and backend.batches == [3]  # one wave
    assert all(o.score.cost.executions == 1 and o.score.new_runs() for o in out)


def test_identical_requests_are_executed_once_and_shared(env):
    store, adapter, backend, scorer = env
    c1 = check("t", "plain", "ab", store)
    items = [ScoreInput(*graded(store, adapter, c1, "ab", trajectory=f"same-{i}")) for i in range(4)]
    out = scorer.score(items, version("g", check("t", "up", "AB", store)))
    assert len(backend.executed) == 1
    assert sorted(o.score.cost.executions for o in out) == [0, 0, 0, 1]
    assert sorted(o.score.cost.shared for o in out) == [0, 1, 1, 1]
    assert all(o.score.passed for o in out)


def test_infrastructure_errors_are_retried_and_never_become_verdicts(env):
    store, adapter, _, _ = env
    c1 = check("t", "plain", "ab", store)
    item = ScoreInput(*graded(store, adapter, c1, "ab"))
    target = version("g", check("t", "rev", "ba", store))
    request = adapter.reexec_request(item.execution, target.entries["t"]["*"], store)
    flaky = ToyBackend(infra={request.request_id: 2})
    out = Scorer([adapter], store, flaky, retries=2).score([item], target)[0]
    assert out.action == "reexec" and out.score.passed and out.score.parts[0].attempts == 3
    assert len(flaky.executed) == 3 and flaky.batches == [1, 1, 1]
    dead = ToyBackend(always_fail=True)
    out = Scorer([adapter], store, dead, retries=1).score([item], target)[0]
    assert out.action == "undeterminable" and out.score.outcome is Outcome.UNDETERMINABLE
    assert out.score.reason.startswith("infra_error") and len(dead.executed) == 2


def test_unparseable_outputs_escalate_to_reexecution(env):
    store, _, backend, _ = env
    adapter = ToyAdapter(fail_parse_for={"ab"})
    c1 = check("t", "plain", "ab", store)
    item = ScoreInput(*graded(store, adapter, c1, "ab"))
    relabel = version("g", check("t", "plain", "xx", store))
    out = Scorer([adapter], store, backend).score([item], relabel)[0]
    # the recorded output reads as garbled, so the check is executed; the fresh output is garbled too
    assert out.score.outcome is Outcome.UNDETERMINABLE and len(backend.executed) == 3
    assert out.score.parts[0].reason == "garbled_output"


def test_recorded_infra_error_is_reexecuted(env):
    store, adapter, backend, scorer = env
    c1 = check("t", "plain", "ab", store)
    record, recorded = graded(store, adapter, c1, "ab", error=True)
    assert recorded.outcome is Outcome.UNDETERMINABLE
    out = scorer.score([ScoreInput(record, (recorded,))], version("g1", c1))[0]
    assert out.action == "reexec" and out.score.passed and len(backend.executed) == 1
    strict = scorer.with_options(recorded_infra="undeterminable")
    assert strict.score([ScoreInput(record, (recorded,))], version("g1", c1))[0].action == "undeterminable"


def test_a_recorded_undeterminable_verdict_is_final_only_under_the_trainer_compatible_setting(env):
    """A record whose recorded verdict is undeterminable as a whole (its run errored when it was graded) is scored like
    any other by default: executed again, under any target, recorded_infra="undeterminable" stopping only the same
    execution. With recorded_undeterminable="final" (a trainer that calls such a member unrepairable) it is
    undeterminable under every policy and target, nothing executed; a logically recorded record is scored as before."""
    store, adapter, backend, scorer = env
    c1 = check("t", "plain", "ab", store)
    record, recorded = graded(store, adapter, c1, "ab", error=True)
    assert recorded.outcome is Outcome.UNDETERMINABLE
    item = ScoreInput(record, (recorded,))
    same, rerun = version("g1", c1), version("g2", check("t", "rev", "ba", store))
    assert scorer.score([item], same)[0].action == "reexec"
    assert scorer.with_options(recorded_infra="undeterminable").score([item], rerun)[0].action == "reexec"
    executed = len(backend.executed)
    assert executed == 2
    final = scorer.with_options(recorded_undeterminable="final")
    assert final.recorded_undeterminable == "final" and scorer.recorded_undeterminable == "resolve"
    for policy in ("ecc", "reparse_all", "full"):
        for target in (same, rerun):
            out = final.with_options(policy=policy).score([item], target)[0]
            assert out.action == "undeterminable" and out.score.outcome is Outcome.UNDETERMINABLE, policy
            assert out.score.reason == "recorded_undeterminable" and out.score.cost.executions == 0, policy
            assert [p.reason for p in out.score.parts] == ["recorded_undeterminable"]
    assert len(backend.executed) == executed
    # under an allow that forbids executing it is undeterminable as well, never pending
    assert final.with_options(allow="reuse").score([item], rerun)[0].action == "undeterminable"
    ok = ScoreInput(*graded(store, adapter, c1, "ab", trajectory="ok"))
    assert final.score([ok], same)[0].action == "reuse"
    assert final.score([ok], rerun)[0].action == "reexec"


def test_under_the_trainer_compatible_setting_the_evidence_still_decides(env):
    """An evidence-only verdict (an empty answer) comes first: it fails the record even when its recorded verdict is
    undeterminable, as an empty patch does in the trainer whose rule the setting reproduces."""
    store, adapter, backend, scorer = env
    c1 = check("t", "plain", "ab", store)
    empty = adapter.record(store, task="t", trajectory="empty", answer=None, empty=True)
    recorded = ScoreRecord(execution_id=empty.record_id, grader_version=None, outcome=Outcome.UNDETERMINABLE,
                           method=Method.UNDETERMINABLE, reason="boom",
                           parts=(PartScore(check=c1.name, execution_digest=c1.execution_digest,
                                            grading_digest=c1.grading_digest, outcome=Outcome.UNDETERMINABLE,
                                            method=Method.UNDETERMINABLE, reason="boom"),))
    out = scorer.with_options(recorded_undeterminable="final").score([ScoreInput(empty, (recorded,))],
                                                                     version("g", conj("all", c1)))[0]
    assert out.score.outcome is Outcome.FAIL and out.score.reason == "empty_answer" and backend.executed == []


def test_the_trainer_compatible_setting_covers_every_check_of_a_conjunction(env):
    """recorded_infra="undeterminable" leaves only the errored check undeterminable: a further check of the target is
    executed and can still fail the record. recorded_undeterminable="final" decides the record as a whole."""
    store, adapter, backend, scorer = env
    base, hardened = check("t", "plain", "ab", store), check("t", "rev", "xx", store)
    item = ScoreInput(*graded(store, adapter, base, "ab", error=True))
    e1 = version("e1", conj("all", base, hardened))
    out = scorer.with_options(recorded_infra="undeterminable").score([item], e1)[0]
    assert out.action == "reexec" and out.score.outcome is Outcome.FAIL and len(backend.executed) == 1
    out = scorer.with_options(recorded_undeterminable="final").score([item], e1)[0]
    assert out.action == "undeterminable" and out.score.reason == "recorded_undeterminable"
    assert [p.reason for p in out.score.parts] == ["recorded_undeterminable"] * 2 and len(backend.executed) == 1


def test_recorded_undeterminable_takes_only_its_two_settings(env):
    store, adapter, backend, scorer = env
    with pytest.raises(ValueError, match="recorded_undeterminable"):
        Scorer([adapter], store, backend, recorded_undeterminable="unrepairable")
    with pytest.raises(ValueError, match="recorded_undeterminable"):
        scorer.with_options(recorded_undeterminable="undeterminable")


def test_conjunction_short_circuits_only_on_logical_fails(env):
    store, adapter, backend, scorer = env
    base = check("t", "plain", "ab", store)
    hardened = check("t", "rev", "ba", store)
    e1 = version("e1", conj("all", base, hardened))
    passing = ScoreInput(*graded(store, adapter, base, "ab", trajectory="p"))
    failing = ScoreInput(*graded(store, adapter, base, "zz", trajectory="f"))
    errored = ScoreInput(*graded(store, adapter, base, "ab", trajectory="e", error=True))
    out = scorer.score([passing, failing, errored], e1)
    p, f, e = out
    assert f.action == "reuse" and f.short_circuit and f.score.skipped == ("t",) and not f.score.passed
    assert p.action == "reexec" and p.score.passed and [x.method for x in p.score.parts] == [
        Method.RECORDED, Method.REEXEC]
    # the errored base run is executed again (an error never implies a fail), then the hardened suite
    assert e.action == "reexec" and e.score.passed and [x.method for x in e.score.parts] == [
        Method.REEXEC, Method.REEXEC]
    # wave 1 runs p's hardened suite and e's base suite; e's hardened request equals p's (same answer, same
    # check), so it is served from wave 1's result instead of a second wave
    assert backend.batches == [2] and len(backend.executed) == 2
    assert e.score.cost.shared == 1 and e.score.cost.executions == 1


def test_ungoverned_uncovered_and_torn(env):
    store, adapter, backend, scorer = env
    c1 = check("t", "plain", "ab", store)
    rec, recorded = graded(store, adapter, c1, "ab", world="w0")
    other = GraderVersion(name="o", entries={"other-task": c1})
    out = scorer.score([ScoreInput(rec, (recorded,))], other)[0]
    assert out.action == "ungoverned" and out.score.passed and out.score.based_on == recorded.score_id
    assert scorer.score([rec], other)[0].score.reason == "no_recorded_verdict"
    per_world = GraderVersion(name="pw", entries={"t": {"w1": c1}})
    assert scorer.score([ScoreInput(rec, (recorded,))], per_world)[0].score.reason == "unknown_world"
    torn = adapter.record(store, task="t", trajectory="torn", answer="ab", runs=rec.runs, observed=("w0", "w1"))
    assert scorer.score([torn], version("g1", c1))[0].score.reason == "torn_observation"
    alien = rec.__class__(family="unknown", task_id="t", trajectory_id="x")
    assert scorer.score([alien], version("g1", c1))[0].score.reason == "no_adapter"


def test_allow_limits_the_strongest_action(env):
    store, adapter, backend, scorer = env
    c1 = check("t", "plain", "ab", store)
    item = ScoreInput(*graded(store, adapter, c1, "ab"))
    audit = scorer.with_options(allow=Action.REPARSE)
    assert audit.score([item], version("g", check("t", "plain", "ba", store)))[0].action == "reparse"
    pending = audit.score([item], version("g", check("t", "rev", "ba", store)))[0]
    assert pending.action == "pending" and pending.required == "reexec" and pending.score is None
    reuse_only = scorer.with_options(allow=Action.REUSE)
    assert reuse_only.score([item], version("g", c1))[0].action == "reuse"
    assert reuse_only.score([item], version("g", check("t", "plain", "ba", store)))[0].action == "pending"
    assert backend.executed == []


def test_evidence_only_verdicts_are_decided_under_every_policy_and_allowed_action(env):
    """An empty answer fails every check without grader input or execution: it is decided even where only reuse is
    allowed, and it is the recorded verdict, reused, when that verdict has the check's identities."""
    store, adapter, backend, scorer = env
    c1 = check("t", "plain", "ab", store)
    empty = adapter.record(store, task="t", trajectory="t-empty", answer=None, empty=True)
    recorded = grade_recorded(adapter, empty, c1, store)
    assert recorded.outcome is Outcome.FAIL and recorded.reason == "empty_answer"
    item = ScoreInput(empty, (recorded,))
    same, changed = version("g1", c1), version("g2", check("t", "rev", "ba", store))
    for policy in Policy:
        for allow in Action:
            if allow is Action.UNDETERMINABLE:
                continue
            s = scorer.with_options(policy=policy, allow=allow)
            out = s.score([item], same)[0]
            assert out.action == "reuse" and out.score.based_on == recorded.score_id, (policy, allow)
            out = s.score([item], changed)[0]
            assert out.action == "reparse" and out.score.outcome is Outcome.FAIL, (policy, allow)
            assert out.score.reason == "empty_answer"
    assert backend.executed == []


def test_policies(env):
    store, adapter, backend, scorer = env
    c1 = check("t", "plain", "ab", store)
    item = ScoreInput(*graded(store, adapter, c1, "ab"))
    g1 = version("g1", c1)
    assert scorer.with_options(policy=Policy.REPARSE_ALL).score([item], g1)[0].action == "reparse"
    assert backend.executed == []
    full = scorer.with_options(policy="full").score([item], g1)[0]
    assert full.action == "reexec" and full.score.passed and len(backend.executed) == 1


def test_no_backend_and_missing_artifact(env):
    store, adapter, _, _ = env
    c1 = check("t", "plain", "ab", store)
    rec, recorded = graded(store, adapter, c1, "ab")
    target = version("g", check("t", "rev", "ba", store))
    out = Scorer([adapter], store).score([ScoreInput(rec, (recorded,))], target)[0]
    assert out.score.reason.startswith("no_backend") and out.action == "undeterminable"
    bare = adapter.record(store, task="t", trajectory="no-answer", answer=None, runs=rec.runs)
    out = Scorer([adapter], store, ToyBackend()).score([bare], target)[0]
    assert out.score.reason == "missing_artifact:execution_changed"
    empty = adapter.record(store, task="t", trajectory="empty", answer=None, empty=True)
    assert Scorer([adapter], store).score([empty], target)[0].score.reason == "empty_answer"


def test_runs_from_an_earlier_reexecution_are_reparsed_later(env):
    store, adapter, backend, scorer = env
    c1 = check("t", "plain", "ab", store)
    rec, recorded = graded(store, adapter, c1, "ba")
    g2 = version("g2", check("t", "rev", "ab", store))
    s2 = scorer.score([ScoreInput(rec, (recorded,))], g2)[0].score
    assert s2.method is Method.REEXEC and len(backend.executed) == 1
    g3 = version("g3", check("t", "rev", "zz", store))  # same execution as g2, new expected value
    s3 = scorer.score([ScoreInput(rec, (recorded, s2))], g3)[0]
    assert s3.action == "reparse" and len(backend.executed) == 1
    s2_again = scorer.score([ScoreInput(rec, (recorded, s2))], g2)[0]
    assert s2_again.action == "reuse" and s2_again.score.based_on == s2.score_id


def test_shared_cache_spans_calls(env):
    store, adapter, backend, _ = env
    cache = {}
    scorer = Scorer([adapter], store, backend, cache=cache)
    c1 = check("t", "plain", "ab", store)
    target = version("g", check("t", "up", "AB", store))
    scorer.score([ScoreInput(*graded(store, adapter, c1, "ab", trajectory="a"))], target)
    out = scorer.score([ScoreInput(*graded(store, adapter, c1, "ab", trajectory="b"))], target)[0]
    assert len(backend.executed) == 1 and out.score.cost.shared == 1 and out.score.passed and cache


def test_scores_match_the_oracle_under_a_chain_of_versions(env):
    store, adapter, backend, scorer = env
    c1 = check("t", "plain", "ab", store)
    answers = ["ab", "ba", "AB", "abab", "x"]
    items = [ScoreInput(*graded(store, adapter, c1, a)) for a in answers]
    specs = [check("t", "plain", "ba", store), check("t", "up", "AB", store),
             conj("all", check("t", "plain", "ab", store), check("t", "dup", "abab", store)),
             conj("all", check("t", "rev", "ba", store), check("t", "up", "AB", store))]
    for i, spec in enumerate(specs):
        out = scorer.score(items, version(f"g{i}", spec))
        for item, o in zip(items, out):
            answer = store.get(item.execution.artifact("answer").digest).decode()
            assert o.score.passed == oracle_verdict(spec, answer, store)


def test_prior_scores_must_belong_to_the_record(env):
    store, adapter, _, scorer = env
    c1 = check("t", "plain", "ab", store)
    rec_a, score_a = graded(store, adapter, c1, "ab", trajectory="a")
    rec_b, _ = graded(store, adapter, c1, "ab", trajectory="b")
    with pytest.raises(ValueError):
        scorer.score([ScoreInput(rec_b, (score_a,))], version("g", c1))


def test_grade_recorded_short_circuits_and_reports_missing_runs(env):
    store, adapter, _, _ = env
    base, hard = check("t", "plain", "ab", store), check("t", "rev", "ba", store)
    rec = adapter.record(store, task="t", trajectory="x", answer="zz", runs=[adapter.run(store, base, "zz")])
    s = grade_recorded(adapter, rec, conj("all", base, hard), store)
    assert s.outcome is Outcome.FAIL and s.skipped == ("t",) and s.method is Method.RECORDED
    rec2 = adapter.record(store, task="t", trajectory="y", answer="ab", runs=[adapter.run(store, base, "ab")])
    s2 = grade_recorded(adapter, rec2, conj("all", base, hard), store)
    assert s2.outcome is Outcome.UNDETERMINABLE and s2.parts[-1].reason == "not_run"


def test_an_undeterminable_part_and_a_logical_fail_are_a_fail(env):
    """The conjunction is monotone: the first suite errored at grading and cannot be executed again (no artifact),
    the second failed logically: the record fails, it is not undeterminable."""
    store, adapter, backend, scorer = env
    a, b = check("t", "plain", "ab", store), check("t", "rev", "ab", store)
    runs = [adapter.run(store, a, "ab", error=True), adapter.run(store, b, "ab")]
    record = adapter.record(store, task="t", trajectory="u-and-f", answer=None, runs=runs)
    spec = conj("all", a, b)
    recorded = grade_recorded(adapter, record, spec, store)
    assert recorded.outcome is Outcome.FAIL and [p.outcome for p in recorded.parts] == [
        Outcome.UNDETERMINABLE, Outcome.FAIL]
    out = scorer.score([ScoreInput(record, ())], version("e1", spec))[0]
    assert out.score.outcome is Outcome.FAIL and out.action == "reparse" and not out.short_circuit
    assert [p.outcome for p in out.score.parts] == [Outcome.UNDETERMINABLE, Outcome.FAIL]
    # and an undeterminable part beside passes stays undeterminable
    ok = adapter.record(store, task="t", trajectory="u-and-p", answer=None,
                        runs=[adapter.run(store, a, "ab", error=True), adapter.run(store, check("t", "rev", "ba",
                                                                                         store), "ab")])
    out = scorer.score([ok], version("e2", conj("all", a, check("t", "rev", "ba", store))))[0]
    assert out.score.outcome is Outcome.UNDETERMINABLE


def test_a_known_fail_decides_before_anything_is_executed(env):
    """A recorded fail of a later suite decides the conjunction although an earlier suite would need an execution:
    nothing is executed (the short circuit uses everything known for free first)."""
    store, adapter, backend, scorer = env
    a, b = check("t", "rev", "ba", store), check("t", "plain", "zz", store)
    record = adapter.record(store, task="t", trajectory="late-fail", answer="ab", runs=[adapter.run(store, b, "ab")])
    recorded = grade_recorded(adapter, record, b, store)
    out = scorer.score([ScoreInput(record, (recorded,))], version("e1", conj("all", a, b)))[0]
    assert out.score.outcome is Outcome.FAIL and out.action == "reuse" and out.short_circuit
    assert out.score.skipped == ("t",) and backend.executed == []


def test_an_evidence_decided_verdict_overrides_a_disagreeing_recorded_one(env):
    """An empty answer fails every check: a recorded pass for it (a grading bug) is not reused, under any policy."""
    store, adapter, backend, scorer = env
    c1 = check("t", "plain", "ab", store)
    record = adapter.record(store, task="t", trajectory="empty-pass", answer=None,
                            runs=[adapter.run(store, c1, "ab")], empty=True)
    wrong = ScoreRecord(execution_id=record.record_id, grader_version=None, outcome=Outcome.PASS,
                        method=Method.RECORDED, parts=(PartScore(check=c1.name, execution_digest=c1.execution_digest,
                                                                 grading_digest=c1.grading_digest,
                                                                 outcome=Outcome.PASS, method=Method.RECORDED),))
    for policy in Policy:
        out = scorer.with_options(policy=policy).score([ScoreInput(record, (wrong,))], version("g1", c1))[0]
        assert out.score.outcome is Outcome.FAIL and out.action == "reparse" and out.score.reason == "empty_answer"


def test_a_cached_result_without_a_verdict_is_executed_again(env):
    """Only results that gave their check a logical verdict are shared: a garbled output is executed again by the
    next call, even with no retries."""
    store, _, _, _ = env
    adapter = ToyAdapter(fail_parse_for={"\x00garbled"})
    cache: dict = {}
    c1 = check("t", "plain", "ab", store)
    target = version("g2", check("t", "rev", "ab", store))
    record = adapter.record(store, task="t", trajectory="garbled", answer="ba", runs=[adapter.run(store, c1, "ba")])
    first = Scorer([adapter], store, ToyBackend(garble=True), retries=0, cache=cache).score([record], target)[0]
    assert first.score.outcome is Outcome.UNDETERMINABLE and cache == {}
    healthy = ToyBackend()
    second = Scorer([adapter], store, healthy, retries=0, cache=cache).score([record], target)[0]
    assert healthy.executed and second.score.passed and len(cache) == 1
    third = Scorer([adapter], store, ToyBackend(garble=True), retries=0, cache=cache).score([record], target)[0]
    assert third.score.passed and third.score.cost.shared == 1  # the logical result is served


def test_based_on_names_one_score_only_when_every_part_was_taken_from_it(env):
    store, adapter, backend, scorer = env
    a, b = check("t", "plain", "ab", store), check("t", "rev", "ab", store)
    rec, recorded = graded(store, adapter, a, "ab")
    e1 = version("e1", conj("all", a, b))
    evaluated = scorer.score([ScoreInput(rec, (recorded,))], e1)[0].score
    assert evaluated.based_on is None and [p.method for p in evaluated.parts] == [Method.RECORDED, Method.REEXEC]
    again = scorer.score([ScoreInput(rec, (recorded, evaluated))], e1)[0].score
    assert again.method is Method.RECORDED and again.based_on == evaluated.score_id  # the latest score with both
    only_b = scorer.score([ScoreInput(rec, (recorded,))], version("gb", b))[0].score
    mixed = scorer.score([ScoreInput(rec, (recorded, only_b))], e1)[0].score
    assert mixed.method is Method.RECORDED and mixed.based_on is None  # parts from two scores: no single source
    same = scorer.score([ScoreInput(rec, (recorded,))], version("g1", a))[0].score
    assert same.based_on == recorded.score_id
