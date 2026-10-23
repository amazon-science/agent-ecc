# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoring execution records under a grader version (``score_under_version``).

:class:`Scorer` turns (execution record, earlier scores) into a :class:`agent_ecc.records.ScoreRecord` under a target
grader version. Per record it looks the task and observed world up in the version and resolves the checks of the
spec (:class:`agent_ecc.compare.Conjunction`, a monotone three-valued AND), cheapest first:

1. **Without any work**: a verdict the record's evidence decides alone (an empty patch), a recorded verdict whose
   identities equal the check's (:func:`agent_ecc.compare.plan_check`), under the ``ecc`` policy a verdict a recorded
   run of another check carries over (a patch that did not apply under the same image, application commands and patch
   filter), and a check no action can decide.
2. **In process**: recorded outputs re-parsed by the family adapter (every re-parseable check).
3. **Executed**: the remaining checks, one at a time in spec order, through a backend; infrastructure errors are
   retried and never become verdicts.

A record whose recorded verdict is undeterminable as a whole is scored like any other record (its logical parts reused,
the rest re-parsed or executed). With ``recorded_undeterminable="final"`` it stays undeterminable under every version
instead: only a verdict its evidence decides alone (an empty patch) is taken, and nothing is reused, re-parsed or
executed for it, as a trainer that calls such a member unrepairable decides it.

A logical fail stops every step more expensive than the one that found it (the short circuit; ``short_circuit=False``
evaluates every check): a fail known without work means nothing is re-parsed or executed, a re-parsed fail means
nothing is executed, an executed fail ends the executions. An undeterminable check decides nothing, and the checks
after it are still evaluated.

Re-executions of a whole batch are issued in waves: every record that needs an execution now contributes one request,
identical requests (same artifact, same target check) are executed once, and records whose conjunction needs a further
check (a SWE-ABS hardened suite after the original suite passed) contribute to the next wave. Retries of
infrastructure errors also go to the next wave. A result that gave its check a logical verdict is shared with later
waves and, through ``cache``, with later calls; a result without one is never served again (it is executed again).

The scorer is used both for evaluation (score a fixed record set under one version) and inside a seal (score the
members of a batch under the batch's epoch). ``allow`` limits the strongest action it may take: an audit seal re-parses
but never executes; a validate-and-discard seal only reuses. A verdict the record's evidence decides alone (an empty
patch) needs no recomputation and is decided whatever ``allow`` and the policy are: it is the recorded verdict, reused,
when that verdict has the check's identities and agrees, and an in-process decision otherwise.

With ``prepared`` (a view of :class:`agent_ecc.prepare.Preparer` bound to one epoch, which a repair seal gives its
scorer), each request a wave would execute for the first time is offered to the preparation first: a result prepared
for exactly that request under exactly that epoch answers it when it gives the check a logical verdict (a job still
running is waited for); every other request is executed here, at once, while the scorer waits for running jobs. The
scores are then exactly those the same raw results give when the scorer executes the requests itself; only the
execution counts differ (a prepared answer is charged as a shared result, not an execution).
"""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from .canonical import FrozenDict, freeze
from .compare import Action, Conjunction, Policy, plan_check
from .records import (
    RUN_ERROR,
    RUN_OK,
    Cost,
    ExecutionRecord,
    Method,
    Outcome,
    PartScore,
    ScoreRecord,
    TestOutcome,
    TestRun,
)
from .recompute.base import Backend, ExecRequest, ExecResult, _add_transport, _gave_up, infra_result
from .store import ContentStore
from .versions import Check, Epoch, GraderVersion, iter_checks, spec_digest

__all__ = [
    "Adapter",
    "PartVerdict",
    "ScoreInput",
    "Scored",
    "Scorer",
    "grade_recorded",
]


@dataclass(frozen=True)
class PartVerdict:
    """What an adapter reads from a run's outputs under a check: a logical PASS / FAIL, or UNDETERMINABLE when the
    outputs do not hold a logical verdict (an infrastructure error, a missing or truncated log, a parse failure).
    ``source`` is set on a verdict carried over from a recorded run of another check (``Adapter.carry_over``): that
    run, which the scored part names as its provenance."""

    outcome: Outcome
    reason: str | None = None
    tests: tuple[TestOutcome, ...] = ()
    final: bool = False  # an undeterminable verdict that executing again cannot change (e.g. the grader is missing)
    source: TestRun | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", Outcome(self.outcome))
        object.__setattr__(self, "tests", tuple(self.tests))

    @classmethod
    def passed(cls, tests: Sequence[TestOutcome] = (), reason: str | None = None) -> PartVerdict:
        return cls(Outcome.PASS, reason, tuple(tests))

    @classmethod
    def failed(cls, reason: str, tests: Sequence[TestOutcome] = ()) -> PartVerdict:
        return cls(Outcome.FAIL, reason, tuple(tests))

    @classmethod
    def undeterminable(cls, reason: str, tests: Sequence[TestOutcome] = (), *, final: bool = False) -> PartVerdict:
        return cls(Outcome.UNDETERMINABLE, reason, tuple(tests), final)


@runtime_checkable
class Adapter(Protocol):
    """What a task family provides to the scorer.

    * ``parse(check, run, store)``: the verdict the run's outputs give under the check's grading identity. Must be a
      pure function of the check, the run and the objects it reads; must return UNDETERMINABLE rather than a verdict
      when the outputs do not support one.
    * ``reexec_request(record, check, store)``: the request that executes the check on the record's artifact, or None
      when the record does not hold what an execution needs.
    * ``decide_without_execution(record, check)``: a verdict the record's evidence decides alone (for example: an
      empty patch fails every test suite), or None.

    Optionally, ``reexec_run(check, request, result)`` builds the run a re-execution's raw result becomes (the
    evidence later versions re-parse). The default is the check's execution identity with the result's outputs and
    facts; an adapter whose raw outputs depend on grading inputs (an evaluator that compares with an expected answer
    while it runs) stamps them on the run, so that a later check with other grading inputs does not read the outputs
    as its own.

    Optionally, ``carry_over(record, check)`` returns a verdict one of the record's own runs gives a check whose
    execution or grading inputs differ from that run's without executing it (SWE-bench: a patch that did not apply
    under the same image, application commands and patch filter fails under any eval script and any grading inputs),
    with that run as the verdict's ``source``. The scorer asks it only under the ``ecc`` policy, after equal-identity
    reuse, and decides the check as a reuse (method ``recorded``) whose part names the source run (``PartScore.run``).
    """

    family: str

    def parse(self, check: Check, run: TestRun, store: ContentStore) -> PartVerdict: ...

    def reexec_request(self, record: ExecutionRecord, check: Check, store: ContentStore) -> ExecRequest | None: ...

    def decide_without_execution(self, record: ExecutionRecord, check: Check) -> PartVerdict | None: ...


@dataclass(frozen=True)
class ScoreInput:
    """One record to score, with the scores it already has (its recorded verdict and any earlier rescoring)."""

    execution: ExecutionRecord
    prior: tuple[ScoreRecord, ...] = ()

    def __post_init__(self) -> None:
        prior = self.prior
        if isinstance(prior, ScoreRecord):
            prior = (prior,)
        object.__setattr__(self, "prior", tuple(p for p in prior if p is not None))


@dataclass(frozen=True)
class Scored:
    """The scorer's answer for one record: the score under the target version, and a summary of how it was obtained.

    ``action`` is reuse, reparse, reexec, undeterminable, ungoverned, or pending (the strongest action needed exceeded
    ``allow``; ``required`` names it and no score is produced). ``short_circuit`` is true when a logical fail decided
    the conjunction while some of its checks were never evaluated. ``transport`` tallies the requests first charged
    to this record that a transport carried: ``requests``, ``attempts``, ``retried``, ``gave_up`` and ``reasons``."""

    execution_id: str
    score: ScoreRecord | None
    action: str
    required: str | None = None
    short_circuit: bool = False
    transport: Mapping[str, Any] = field(default_factory=FrozenDict, compare=False)


@dataclass
class _Job:
    index: int  # the check's position in the conjunction
    check: Check
    request: ExecRequest
    reason: str


@dataclass
class _Wait:
    state: _State
    job: _Job
    attempts: int = 0


@dataclass
class _State:
    index: int
    item: ScoreInput
    record: ExecutionRecord
    adapter: Any = None
    conj: Conjunction | None = None
    spec_digest: str | None = None
    parts_prior: list = field(default_factory=list)
    part_source: dict = field(default_factory=dict)  # id(prior part) -> the score id it belongs to
    sources: dict = field(default_factory=dict)  # check index -> the score id a reused verdict was taken from
    runs: list = field(default_factory=list)
    planned: bool = False
    queue: list = field(default_factory=list)  # _Job, in check order: the checks left to execute
    retry: _Wait | None = None
    waiting: bool = False
    result: Scored | None = None
    costs: list = field(default_factory=list)  # per check of the conjunction, in order
    version_id: str | None = None
    transport: dict = field(default_factory=dict)  # the transport tally of the requests first charged to it


_METHOD_ACTION = {Method.RECORDED: "reuse", Method.REPARSE: "reparse", Method.REEXEC: "reexec",
                  Method.UNDETERMINABLE: "undeterminable"}
# what a record whose recorded verdict is undeterminable as a whole leads to (Scorer(recorded_undeterminable=...))
RECORDED_UNDETERMINABLE_MODES = ("resolve", "final")
# the reason of every check such a record leaves undeterminable under "final"
RECORDED_UNDETERMINABLE = "recorded_undeterminable"


class Scorer:
    """Scores records under grader versions.

    ``adapters`` maps a family name to its adapter; ``store`` holds evidence; ``backend`` executes re-execution requests
    (without one, a check that needs an execution is undeterminable). ``retries`` is the number of further executions
    after one without a logical verdict, unless the backend's transport gave up on it. ``recorded_infra`` says what a
    recorded run that hit an infrastructure error leads to (``"reexec"`` or ``"undeterminable"``). ``short_circuit``
    (default) stops evaluating a conjunction at its first logical fail; False evaluates every check. ``cache`` may be a
    dict shared between calls: request id -> result, for results that gave their check a logical verdict.

    ``recorded_undeterminable`` says what a record whose recorded verdict is undeterminable as a whole leads to (the
    recorded verdict: the first earlier score that was recorded and not derived from another, otherwise the first
    earlier score; a record without earlier scores has none): ``"resolve"`` (default) scores it like any other record;
    ``"final"`` leaves every check its evidence does not decide alone undeterminable (reason
    ``recorded_undeterminable``), under every policy and before any reuse, re-parse or execution, so the record is
    undeterminable under every version unless an evidence-only verdict fails it. ``"final"`` reproduces a trainer that
    marks such a member unrepairable without trying (a seal then neutralizes the member's group under repair and
    discard; audit trains its recorded reward).

    ``prepared`` is a preparation view bound to one epoch (:meth:`agent_ecc.prepare.Preparer.view`): it is consulted
    only when the version scored is given as that very epoch (an :class:`Epoch` with the view's publication and
    content; never for a bare :class:`GraderVersion`), and only for a request's first execution.
    """

    def __init__(
        self,
        adapters: Mapping[str, Adapter] | Sequence[Adapter],
        store: ContentStore,
        backend: Backend | None = None,
        *,
        policy: Policy | str = Policy.ECC,
        retries: int = 2,
        recorded_infra: str = "reexec",
        allow: Action | str = Action.REEXEC,
        short_circuit: bool = True,
        cache: dict | None = None,
        clock: Callable[[], float] = time.perf_counter,
        recorded_undeterminable: str = "resolve",
        prepared: Any = None,
    ) -> None:
        if not isinstance(adapters, Mapping):
            adapters = {a.family: a for a in adapters}
        if recorded_infra not in ("reexec", "undeterminable"):
            raise ValueError("recorded_infra must be 'reexec' or 'undeterminable'")
        if recorded_undeterminable not in RECORDED_UNDETERMINABLE_MODES:
            raise ValueError("recorded_undeterminable must be 'resolve' or 'final'")
        if retries < 0:
            raise ValueError("retries must be >= 0")
        self.adapters = dict(adapters)
        self.store = store
        self.backend = backend
        self.policy = Policy(policy)
        self.retries = int(retries)
        self.recorded_infra = recorded_infra
        self.allow = Action(allow)
        self.short_circuit = bool(short_circuit)
        self.cache = cache
        self.clock = clock
        self.recorded_undeterminable = recorded_undeterminable
        self.prepared = prepared
        self._cache_lock = threading.Lock()

    def with_options(self, **changes: Any) -> Scorer:
        """A copy of this scorer (same class, adapters, store, backend and cache) with some options changed:
        ``policy``, ``retries``, ``recorded_infra``, ``recorded_undeterminable``, ``allow``, ``short_circuit``,
        ``cache``, ``backend``, ``clock``, ``prepared``."""
        clone = copy.copy(self)
        for name, value in changes.items():
            if name == "policy":
                value = Policy(value)
            elif name == "allow":
                value = Action(value)
            elif name == "retries":
                if int(value) < 0:
                    raise ValueError("retries must be >= 0")
                value = int(value)
            elif name == "recorded_infra":
                if value not in ("reexec", "undeterminable"):
                    raise ValueError("recorded_infra must be 'reexec' or 'undeterminable'")
            elif name == "recorded_undeterminable":
                if value not in RECORDED_UNDETERMINABLE_MODES:
                    raise ValueError("recorded_undeterminable must be 'resolve' or 'final'")
            elif name == "short_circuit":
                value = bool(value)
            elif name not in ("cache", "backend", "clock", "prepared"):
                raise TypeError(f"unknown scorer option {name!r}")
            setattr(clone, name, value)
        return clone

    # ------------------------------------------------------------------------------------------------ public

    def score(self, items: Sequence[ScoreInput | ExecutionRecord], version: GraderVersion | Epoch) -> list[Scored]:
        """Score every item under ``version``; one :class:`Scored` per item, in order."""
        identity = None
        if isinstance(version, Epoch):
            identity = (version.ref.seq, version.ref.version_id)
            version = version.version
        view = self._prepared_view(identity)
        states = [self._start(i, item if isinstance(item, ScoreInput) else ScoreInput(item), version)
                  for i, item in enumerate(items)]
        local: dict[str, ExecResult] = {}  # results that gave their check a logical verdict, in this call
        while True:
            waits: list[_Wait] = []
            for st in states:
                if st.result is None and not st.waiting:
                    self._advance(st, version, waits, local)
            if not waits:
                break
            if view is not None:
                # requests preparation answers are resolved now; the rest were executed here (results by request id)
                waits, results = self._prepared_wave(view, waits, local)
                if not waits:
                    continue
            else:
                unique: dict[str, ExecRequest] = {}
                for w in waits:
                    unique.setdefault(w.job.request.request_id, w.job.request)
                results = self._execute(list(unique.values()))
            charged: set[str] = set()
            for w in waits:
                rid = w.job.request.request_id
                result = results[rid]
                first = rid not in charged
                charged.add(rid)
                if first:
                    _add_transport(w.state.transport, getattr(result, "transport", None))
                if self._on_result(w, result, first) and result.status == RUN_OK:
                    local[rid] = result
                    if self.cache is not None:
                        with self._cache_lock:
                            self.cache[rid] = result
        return [replace(st.result, transport=freeze(st.transport)) if st.transport else st.result  # type: ignore
                for st in states]

    def score_one(self, item: ScoreInput | ExecutionRecord, version: GraderVersion | Epoch) -> Scored:
        return self.score([item], version)[0]

    # ------------------------------------------------------------------------------------------------ internals

    def _start(self, index: int, item: ScoreInput, version: GraderVersion) -> _State:
        record = item.execution
        st = _State(index=index, item=item, record=record, version_id=version.version_id)
        st.adapter = self.adapters.get(record.family)
        if st.adapter is None:
            st.result = self._final_undeterminable(st, version, "no_adapter")
            return st
        if record.torn:
            st.result = self._final_undeterminable(st, version, "torn_observation")
            return st
        lookup = version.lookup(record.task_id, record.world_version)
        if lookup.status == "ungoverned":
            st.result = self._ungoverned(st, version)
            return st
        if lookup.status == "uncovered" or lookup.spec is None:
            st.result = self._final_undeterminable(st, version, "unknown_world")
            return st
        st.conj = Conjunction(lookup.spec, short_circuit=self.short_circuit)
        st.costs = [Cost() for _ in st.conj.checks]
        st.spec_digest = spec_digest(lookup.spec)
        for score in item.prior:
            if score.execution_id != record.record_id:
                raise ValueError(f"prior score {score.score_id} is for another execution")
            for part in score.parts:
                st.parts_prior.append(part)
                st.part_source[id(part)] = score.score_id
            st.runs.extend(score.new_runs())
        st.runs = list(record.runs) + st.runs
        return st

    def _available(self, run: TestRun) -> bool:
        return all(self.store.contains(o.digest) for o in run.outputs)

    def _advance(self, st: _State, version: GraderVersion, waits: list[_Wait], local: dict[str, ExecResult]) -> None:
        conj = st.conj
        assert conj is not None
        if not st.planned:
            st.planned = True
            self._plan(st)
            if st.result is not None:
                return
        while not conj.done and st.queue:
            if st.retry is not None:
                w, st.retry = st.retry, None
                waits.append(w)
                st.waiting = True
                return
            job = st.queue[0]
            cached = local.get(job.request.request_id) or self._cached(job.request.request_id)
            if cached is not None and self._serve(st, job, cached):
                st.queue.pop(0)
                continue
            # not executed yet, or the stored result gives this check no verdict: execute
            waits.append(_Wait(state=st, job=job))
            st.waiting = True
            return
        st.result = self._finish(st)

    def _plan(self, st: _State) -> None:
        """Resolve every check that needs no execution; queue the rest. Sets ``st.result`` when the record is decided
        without execution, or needs an action ``allow`` forbids (pending)."""
        conj, record, adapter = st.conj, st.record, st.adapter
        assert conj is not None
        needs = []  # (index, check, plan, request): checks a re-parse or an execution decides
        # a record whose recorded verdict is undeterminable as a whole, under recorded_undeterminable="final": only
        # its evidence decides; every other check is undeterminable, whatever the policy and the identities
        final = self.recorded_undeterminable == "final" and _recorded_undeterminable(st.item.prior)
        # 1. decided without any work: evidence-only verdicts, recorded verdicts with equal identities, and checks no
        #    action can decide
        for i, check in enumerate(conj.checks):
            static = self._static(adapter, record, check)
            if static is not None:
                self._resolve_static(st, i, check, static)
                continue
            if final:
                conj.resolve(self._part(check, PartVerdict.undeterminable(RECORDED_UNDETERMINABLE, final=True),
                                        Method.UNDETERMINABLE), i)
                continue
            request = self._request(adapter, record, check)
            plan = plan_check(check, runs=st.runs, parts=st.parts_prior, output_available=self._available,
                              can_reexec=request is not None, policy=self.policy,
                              recorded_infra=self.recorded_infra)
            carried = self._carried(adapter, record, check) if plan.action is not Action.REUSE else None
            if plan.action is Action.REUSE:
                assert plan.prior is not None
                self._resolve_reused(st, i, check, plan.prior)
            elif carried is not None:
                # a reuse of another check's recorded run: the part names that run (its provenance; not new evidence)
                conj.resolve(self._part(check, carried, Method.RECORDED, run=carried.source), i)
            elif plan.action is Action.UNDETERMINABLE:
                conj.resolve(self._part(check, PartVerdict.undeterminable(plan.reason), Method.UNDETERMINABLE), i)
            else:
                needs.append((i, check, plan, request))
        if conj.done:
            st.result = self._finish(st)
            return
        if self.allow.rank < Action.REPARSE.rank:
            required = Action.REEXEC if any(p.action is Action.REEXEC for _, _, p, _ in needs) else Action.REPARSE
            st.result = Scored(record.record_id, None, "pending", required=required.value)
            return
        # 2. in process: recorded outputs re-parsed under the target's grading identity
        jobs = []
        for i, check, plan, request in needs:
            if plan.action is Action.REPARSE:
                assert plan.run is not None
                t0 = self.clock()
                verdict = self._parse(adapter, check, plan.run)
                seconds = self.clock() - t0
                self._charge(st, i, Cost(seconds=seconds))
                if verdict.outcome.logical:
                    conj.resolve(self._part(check, verdict, Method.REPARSE, run=plan.run, seconds=seconds), i)
                elif request is None or verdict.final:
                    conj.resolve(self._part(check, verdict, Method.UNDETERMINABLE, run=plan.run, seconds=seconds), i)
                else:
                    jobs.append(_Job(i, check, request, f"reparse_undeterminable:{verdict.reason}"))
            else:
                assert request is not None
                jobs.append(_Job(i, check, request, plan.reason))
        if conj.done:
            st.result = self._finish(st)
            return
        # 3. what is left needs an execution
        if self.allow.rank < Action.REEXEC.rank:
            st.result = Scored(record.record_id, None, "pending", required=Action.REEXEC.value)
            return
        if self.backend is None:
            for job in jobs:
                conj.resolve(self._part(job.check, PartVerdict.undeterminable(f"no_backend:{job.reason}"),
                                        Method.UNDETERMINABLE), job.index)
            st.result = self._finish(st)
            return
        st.queue = jobs

    def _resolve_reused(self, st: _State, index: int, check: Check, prior: PartScore) -> None:
        assert st.conj is not None
        st.conj.resolve(PartScore(check=check.name, execution_digest=check.execution_digest,
                                  grading_digest=check.grading_digest, outcome=prior.outcome, method=Method.RECORDED,
                                  reason=prior.reason, tests=prior.tests), index)
        st.sources[index] = st.part_source.get(id(prior))

    def _resolve_static(self, st: _State, index: int, check: Check, static: PartVerdict) -> None:
        """A verdict the record's evidence decides alone depends on no grader input and needs no execution: it is the
        recorded verdict, reused, when one was produced under the check's identities and agrees, and an in-process
        decision otherwise (a recorded verdict that disagrees with the evidence is not taken)."""
        assert st.conj is not None
        if static.outcome.logical:
            prior = next((p for p in reversed(st.parts_prior)
                          if p.execution_digest == check.execution_digest
                          and p.grading_digest == check.grading_digest and p.outcome is static.outcome), None)
            if prior is not None:
                self._resolve_reused(st, index, check, prior)
                return
            st.conj.resolve(self._part(check, static, Method.REPARSE), index)
            return
        st.conj.resolve(self._part(check, static, Method.UNDETERMINABLE), index)

    def _cached(self, request_id: str) -> ExecResult | None:
        if self.cache is None:
            return None
        with self._cache_lock:
            return self.cache.get(request_id)

    def _serve(self, st: _State, job: _Job, result: ExecResult, *, attempts: int = 1) -> bool:
        """Resolve ``job`` from a result executed earlier (in this call, a cached one, or a prepared one) when it gives
        the check a logical verdict. A result without one is not served: the check is executed again."""
        if result.status != RUN_OK:
            return False
        run, verdict, seconds = self._read_result(st, job, result)
        if not verdict.outcome.logical:
            return False
        assert st.conj is not None
        self._charge(st, job.index, Cost(seconds=seconds, shared=1))
        st.conj.resolve(self._part(job.check, verdict, Method.REEXEC, run=run, attempts=attempts,
                                   seconds=st.costs[job.index].seconds), job.index)
        return True

    # ------------------------------------------------------------------------------------------------ preparation

    def _prepared_view(self, identity: tuple | None) -> Any:
        """The preparation view to consult in this call: only when the version scored is the view's epoch, given as
        an :class:`Epoch` (its publication and content); never for a bare version."""
        view = self.prepared
        if view is None or identity is None:
            return None
        return view if tuple(getattr(view, "identity", ()) or ()) == tuple(identity) else None

    def _serve_prepared(self, waits: list[_Wait], result: ExecResult | None,
                        local: dict[str, ExecResult]) -> list[_Wait]:
        """Resolve every wait whose check a prepared result gives a logical verdict, as a result executed earlier is
        served (no execution charged, no seal-time attempt); the seal's cache records the result, as it records its
        own executions. Returns the waits it did not resolve (to execute here)."""
        if result is None:
            return list(waits)
        left = []
        for w in waits:
            if self._serve(w.state, w.job, result, attempts=0):
                w.state.waiting = False
                w.state.queue.pop(0)
            else:
                left.append(w)
        if len(left) < len(waits):
            rid = waits[0].job.request.request_id
            local[rid] = result
            if self.cache is not None:
                with self._cache_lock:
                    self.cache[rid] = result
        return left

    def _prepared_wave(self, view: Any, waits: list[_Wait],
                       local: dict[str, ExecResult]) -> tuple[list[_Wait], dict[str, ExecResult]]:
        """One wave of executions under preparation.

        A request asked for the first time is offered to the view (:meth:`agent_ecc.prepare.PreparedView.claim`):
        a finished prepared result answers it when it gives the check a logical verdict; a job still running is waited
        for; a job not yet started is taken over (it never runs); anything else is executed here. The requests to
        execute here start at once, in a worker thread, while the scorer waits for the running jobs; a waited-for job
        without a usable result is executed when it ends. Retries of executions made here never consult the view.
        Returns the waits still to apply and the results of the executions made here, by request id."""
        fresh: dict[str, list[_Wait]] = {}
        execute: list[_Wait] = []
        for w in waits:
            if w.attempts == 0:
                fresh.setdefault(w.job.request.request_id, []).append(w)
            else:
                execute.append(w)  # a retry of an execution made here
        claim = view.claim([ws[0].job.request for ws in fresh.values()])
        for rid in claim.now:
            execute.extend(fresh[rid])
        for rid, result in claim.ready.items():
            left = self._serve_prepared(fresh[rid], result, local)
            view.answered(rid, used=not left, waited_s=0.0)
            execute.extend(left)
        results: dict[str, ExecResult] = {}
        lock = threading.Lock()
        workers: list[threading.Thread] = []

        def start(batch: list[_Wait]) -> None:
            """Execute the batch's requests here, now, in a worker thread (each request once)."""
            unique: dict[str, ExecRequest] = {}
            for w in batch:
                unique.setdefault(w.job.request.request_id, w.job.request)
            requests = list(unique.values())
            if not requests:
                return
            view.executing(requests)

            def work() -> None:
                got: dict[str, ExecResult] = {}
                try:
                    got = self._execute(requests)
                finally:
                    view.executed(requests, got)
                    with lock:
                        results.update(got)

            worker = threading.Thread(target=work, name="agent-ecc-seal-execute", daemon=True)
            worker.start()
            workers.append(worker)

        start(execute)  # executed at once, while the running jobs are waited for
        late: list[_Wait] = []
        try:
            for rid, job in claim.running.items():
                result, waited_s = view.wait(job)
                left = self._serve_prepared(fresh[rid], result, local)
                view.answered(rid, used=not left, waited_s=waited_s)
                start(left)  # a waited-for job without a usable result: executed here, when it ends
                late.extend(left)
        finally:
            for worker in workers:
                worker.join()  # the executions started here are applied whatever happened while waiting
        for w in execute + late:
            rid = w.job.request.request_id
            if rid not in results:  # a worker that could not even report (it never does): no verdict
                results[rid] = infra_result(w.job.request, "seal execution did not report")
        return execute + late, results

    def _read_result(self, st: _State, job: _Job, result: ExecResult) -> tuple[TestRun, PartVerdict, float]:
        """The run a raw result becomes, and its verdict under the job's check (with the parse time)."""
        return self.read_result(st.adapter, job.check, job.request, result)

    def read_result(self, adapter: Adapter, check: Check, request: ExecRequest,
                    result: ExecResult) -> tuple[TestRun, PartVerdict, float]:
        """The run a re-execution's raw result becomes (the adapter's ``reexec_run``, or the check's execution
        identity with the result's outputs and facts), its verdict under ``check`` and the parse time. A result
        without outputs, or outputs the adapter cannot turn into a run, give no verdict. The preparer judges its
        results with this, as the scorer does."""
        if result.status != RUN_OK:
            run = TestRun(check=check.name, execution=check.execution, status=RUN_ERROR,
                          facts={"error": result.error or "infra_error"})
            return run, PartVerdict.undeterminable(f"infra_error:{result.error or 'unknown'}"), 0.0
        make = getattr(adapter, "reexec_run", None)
        try:
            run = make(check, request, result) if make is not None else TestRun(
                check=check.name, execution=check.execution, status=RUN_OK, outputs=result.outputs,
                facts=result.facts)
            if not isinstance(run, TestRun):
                raise TypeError("reexec_run returned no TestRun")
        except Exception as error:  # noqa: BLE001 - outputs the adapter cannot turn into evidence decide nothing
            reason = f"reexec_run_error:{type(error).__name__}: {error}"[:300]
            run = TestRun(check=check.name, execution=check.execution, status=RUN_ERROR, facts={"error": reason})
            return run, PartVerdict.undeterminable(reason), 0.0
        t0 = self.clock()
        verdict = self._parse(adapter, check, run)
        return run, verdict, self.clock() - t0

    def _on_result(self, w: _Wait, result: ExecResult, first: bool) -> bool:
        """Apply one execution's result; True when it gave its check a logical verdict."""
        st, job = w.state, w.job
        st.waiting = False
        w.attempts += 1
        self._charge(st, job.index, result.cost if first else Cost(shared=1))
        run, verdict, seconds = self._read_result(st, job, result)
        self._charge(st, job.index, Cost(seconds=seconds))
        assert st.conj is not None
        total = st.costs[job.index].seconds
        if verdict.outcome.logical:
            st.conj.resolve(self._part(job.check, verdict, Method.REEXEC, run=run, attempts=w.attempts,
                                       seconds=total), job.index)
            st.queue.pop(0)
            return True
        # a transport failure is retried by the transport alone (with backoff, in its deadline): never again here
        if w.attempts <= self.retries and not verdict.final and not _gave_up(result):
            st.retry = w
            return False
        st.conj.resolve(self._part(job.check, verdict, Method.UNDETERMINABLE, run=run, attempts=w.attempts,
                                   seconds=total), job.index)
        st.queue.pop(0)
        return False

    def _execute(self, requests: list[ExecRequest]) -> dict[str, ExecResult]:
        assert self.backend is not None
        try:
            results = list(self.backend.execute(requests, self.store))
            if len(results) != len(requests):
                raise RuntimeError(f"backend returned {len(results)} results for {len(requests)} requests")
        except Exception as error:  # noqa: BLE001 - a backend failure is an infrastructure error, never a verdict
            results = [infra_result(r, f"backend: {type(error).__name__}: {error}") for r in requests]
        out = {}
        for request, result in zip(requests, results):
            if result.request_id != request.request_id:
                result = infra_result(request, "backend answered another request")
            out[request.request_id] = result
        return out

    def _request(self, adapter: Adapter, record: ExecutionRecord, check: Check) -> ExecRequest | None:
        try:
            return adapter.reexec_request(record, check, self.store)
        except Exception:  # noqa: BLE001 - an artifact the adapter cannot read cannot be executed
            return None

    def _static(self, adapter: Adapter, record: ExecutionRecord, check: Check) -> PartVerdict | None:
        try:
            return adapter.decide_without_execution(record, check)
        except Exception:  # noqa: BLE001
            return None

    def _carried(self, adapter: Adapter, record: ExecutionRecord, check: Check) -> PartVerdict | None:
        """A logical verdict one of the record's own runs carries over to ``check`` (``ecc`` policy only)."""
        carry = getattr(adapter, "carry_over", None)
        if carry is None or self.policy is not Policy.ECC:
            return None
        try:
            verdict = carry(record, check)
        except Exception:  # noqa: BLE001
            return None
        return verdict if isinstance(verdict, PartVerdict) and verdict.outcome.logical else None

    def _parse(self, adapter: Adapter, check: Check, run: TestRun) -> PartVerdict:
        try:
            verdict = adapter.parse(check, run, self.store)
        except Exception as error:  # noqa: BLE001 - a parse failure is not a verdict
            return PartVerdict.undeterminable(f"parse_error:{type(error).__name__}: {error}"[:300])
        if not isinstance(verdict, PartVerdict):
            return PartVerdict.undeterminable("parse_error:adapter returned no PartVerdict")
        return verdict

    def _charge(self, st: _State, index: int, cost: Cost) -> None:
        """Add to the cost of the check at ``index``."""
        st.costs[index] = st.costs[index] + cost

    def _part(self, check: Check, verdict: PartVerdict, method: Method, *, run: TestRun | None = None,
              attempts: int = 0, seconds: float = 0.0) -> PartScore:
        if not verdict.outcome.logical:
            method = Method.UNDETERMINABLE
        return PartScore(check=check.name, execution_digest=check.execution_digest,
                         grading_digest=check.grading_digest, outcome=verdict.outcome, method=method,
                         reason=verdict.reason, tests=verdict.tests, run=run, attempts=attempts, seconds=seconds)

    def _finish(self, st: _State) -> Scored:
        conj = st.conj
        assert conj is not None and conj.outcome is not None
        parts = tuple(conj.parts)
        outcome = conj.outcome
        if outcome is Outcome.UNDETERMINABLE:
            method = Method.UNDETERMINABLE
            deciding = next((p for p in parts if not p.logical), None)
        else:
            method = max((p.method for p in parts if p.logical), key=lambda m: m.rank)
            deciding = next((p for p in parts if p.outcome is Outcome.FAIL), None)
        reason = deciding.reason if deciding is not None else None
        cost = Cost()
        for part_cost in st.costs:
            cost = cost + part_cost
        # a score reused as a whole: every check with a verdict took it from one earlier score (a check left without
        # one, which cost nothing, decides nothing)
        based_on = None
        if method is Method.RECORDED:
            sources = {st.sources.get(i) for i in range(len(conj.checks))
                       if conj.resolved(i) and conj.parts_by_index[i].logical}
            if len(sources) == 1 and None not in sources:
                based_on = sources.pop()
        score = ScoreRecord(execution_id=st.record.record_id, grader_version=st.version_id, outcome=outcome,
                            method=method, spec_digest=st.spec_digest, reason=reason, parts=parts,
                            skipped=conj.skipped, cost=cost, based_on=based_on)
        return Scored(st.record.record_id, score, _METHOD_ACTION[method], short_circuit=conj.short_circuit)

    def _final_undeterminable(self, st: _State, version: GraderVersion, reason: str) -> Scored:
        score = ScoreRecord(execution_id=st.record.record_id, grader_version=version.version_id,
                            outcome=Outcome.UNDETERMINABLE, method=Method.UNDETERMINABLE, reason=reason)
        return Scored(st.record.record_id, score, "undeterminable")

    def _ungoverned(self, st: _State, version: GraderVersion) -> Scored:
        recorded = _recorded_score(st.item.prior)
        if recorded is None or not recorded.logical:
            return self._final_undeterminable(st, version, "no_recorded_verdict")
        score = ScoreRecord(execution_id=st.record.record_id, grader_version=version.version_id,
                            outcome=recorded.outcome, method=Method.RECORDED, reason="ungoverned",
                            based_on=recorded.score_id)
        return Scored(st.record.record_id, score, "ungoverned")


def _recorded_score(prior: Sequence[ScoreRecord]) -> ScoreRecord | None:
    """The verdict recorded when the record was graded: the first prior score that was recorded and not derived
    from another; otherwise the first prior score."""
    for score in prior:
        if score.method is Method.RECORDED and score.based_on is None:
            return score
    return prior[0] if prior else None


def _recorded_undeterminable(prior: Sequence[ScoreRecord]) -> bool:
    """Whether the record's recorded verdict exists and is undeterminable as a whole."""
    recorded = _recorded_score(prior)
    return recorded is not None and not recorded.logical


def grade_recorded(adapter: Adapter, record: ExecutionRecord, spec, store: ContentStore, *,
                   grader_version: str | None = None, meta: Mapping[str, Any] | None = None) -> ScoreRecord:
    """The verdict a producer records when it grades a trajectory: every check of ``spec`` evaluated on the record's
    own runs (no execution), combined by the monotone three-valued AND. Every check whose run exists is graded, even
    after a logical fail, so the recorded score holds every suite that ran. A check the record has no run for is
    undeterminable (``not_run``), or skipped when another check failed logically."""
    checks = tuple(iter_checks(spec))
    parts: dict[int, PartScore] = {}
    not_run: list[int] = []
    for i, check in enumerate(checks):
        verdict = None
        try:
            verdict = adapter.decide_without_execution(record, check)
        except Exception:  # noqa: BLE001
            verdict = None
        if verdict is None:
            run = next((r for r in reversed(record.runs) if r.execution_digest == check.execution_digest), None)
            if run is None:
                not_run.append(i)
                continue
            try:
                verdict = adapter.parse(check, run, store)
            except Exception as error:  # noqa: BLE001
                verdict = PartVerdict.undeterminable(f"parse_error:{type(error).__name__}")
        method = Method.RECORDED if verdict.outcome.logical else Method.UNDETERMINABLE
        parts[i] = PartScore(check=check.name, execution_digest=check.execution_digest,
                             grading_digest=check.grading_digest, outcome=verdict.outcome, method=method,
                             reason=verdict.reason, tests=verdict.tests)
    failed = any(p.outcome is Outcome.FAIL for p in parts.values())
    if not failed:
        for i in not_run:
            check = checks[i]
            parts[i] = PartScore(check=check.name, execution_digest=check.execution_digest,
                                 grading_digest=check.grading_digest, outcome=Outcome.UNDETERMINABLE,
                                 method=Method.UNDETERMINABLE, reason="not_run")
        not_run = []
    ordered = tuple(parts[i] for i in sorted(parts))
    if failed:
        outcome = Outcome.FAIL
        deciding = next(p for p in ordered if p.outcome is Outcome.FAIL)
    elif all(p.outcome is Outcome.PASS for p in ordered):
        outcome, deciding = Outcome.PASS, None
    else:
        outcome = Outcome.UNDETERMINABLE
        deciding = next(p for p in ordered if not p.logical)
    return ScoreRecord(execution_id=record.record_id, grader_version=grader_version, outcome=outcome,
                       method=Method.RECORDED if outcome.logical else Method.UNDETERMINABLE,
                       spec_digest=spec_digest(spec), reason=deciding.reason if deciding else None,
                       parts=ordered, skipped=tuple(checks[i].name for i in not_run), meta=meta or {})
