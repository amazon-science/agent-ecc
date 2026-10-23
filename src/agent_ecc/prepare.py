# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared re-execution: the seal's re-executions start when a group is admitted; the seal only verifies.

A synchronous repair seal executes, at the seal, every check the batch epoch needs that the records do not answer (a
hardened suite after a grader switch, for instance), and the learner waits for the slowest of them. The contract asks
that every label be validated against the batch epoch *at the seal*; the work behind a label may start earlier. A
:class:`Preparer` does that work in the background:

* **Target.** It follows the epoch pointer (:meth:`Preparer.poll`; every ``poll_s`` with background threads). Its
  target is the epoch the pointer names, identified by its publication (``seq``) and its content (``version_id``).
  Under the startup epoch (the first grader publication, or ``startup``) nothing is prepared.
* **Planning.** When a group is admitted (:meth:`Preparer.admit`), and, when the target changes, for every admitted
  group no seal has decided, each member is planned under the target exactly as the seal will plan it: the seal's own
  scorer (adapters, policy, retries, its treatment of infrastructure errors and of undeterminable recorded verdicts,
  the short circuit), reading the run's cache but never writing it, executing nothing. The first request the plan
  leaves to execute becomes a background job, or joins an identical one. When a job ends with a result that gives its
  check a logical verdict, the members waiting on it are planned again with that result, so a conjunction's next suite
  is prepared in turn, and nothing after a logical fail.
* **Execution.** Jobs run first in, first out, at most ``slots`` at once, through the scorer's backend, each with the
  scorer's retries; the seal's own executions in flight are counted first against the backend's ``capacity``, so a
  seal is never kept waiting for a sandbox by preparation.
* **Binding.** A job is keyed by (the target's publication and content, the request id). The request id is the
  content address of what runs (the task, the check's execution identity, the input objects such as the patch as it
  is applied, the parameters), so a prepared result is bound to the evidence the member's own record holds.

The seal (:class:`agent_ecc.seal.SealSession` with ``prepared=``) reads the pointer once, as always, and plans its
members itself. For each request it would execute, its scorer asks the view bound to the batch epoch
(:meth:`Preparer.view`):

* a finished job of that request under that epoch answers it, if its result gives the check a logical verdict;
* a job still running is waited for (at most ``wait_s``), then used the same way;
* a job not yet started is taken over: the seal executes the request itself, and the job never runs;
* anything else (no job, a job under another epoch, a failed job, a result without a logical verdict) is executed at
  the seal, at once, as a synchronous seal executes it.

The seal then decides every member from these raw results exactly as it decides from its own executions. A prepared
result is never a label: the preparer writes no score the seal reads, never writes the run's cache (the seal records a
result it used, as it records its own), never changes a record, and a result prepared under an epoch the batch does
not use is never read. Labels, statuses, neutralizations, the post-seal filter and the decision are therefore those of
a synchronous seal given the same raw results, and, with deterministic checks (the contract's precondition), those of
the synchronous seal; only the seal's execution counts and its wall time differ.

Failure. The preparer never fails a seal: an error in it disables it, and every request is then executed at the seal;
a member whose planning raises is skipped.

Events (:attr:`Preparer.events`, and the JSON-lines ``log`` when given), one object each, with the time ``t``:
``start``; ``target`` (a new target: whether it is active, the jobs it superseded, the groups swept); ``plan`` (one
member planned: ``job``, ``attached``, ``attached_seal``, ``nothing``, or why it was not: ``skip_decided``,
``skip_in_session``, ``skip_other_epoch_session``, ``stale_target``, ``no_record``, ``plan_error``); ``job`` (a job's
terminal state, once: ``done``, ``failed``, ``superseded``, ``taken``, ``dropped``, ``disabled``, or ``seal_done`` for
an execution a seal made itself, with its members, times, attempts and whether its result was usable); ``use`` (how a
seal answered one request: ``prepared`` (``ready``, ``waited``) or ``sync`` (``no_job``, ``other_identity``,
``taken_over``, ``failed``, ``error``, ``undeterminable``, ``superseded``, ``dropped``, ``disabled``,
``job_unfinished``)); ``deciding``; ``sealed``; ``retired``; ``disabled``.
"""

from __future__ import annotations

import collections
import json
import os
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .compare import Action
from .records import RUN_OK
from .recompute.base import ExecRequest, ExecResult, _gave_up, infra_result
from .scoring import ScoreInput, Scorer
from .seal import Candidate, Member
from .versions import Check, Epoch, iter_checks

__all__ = [
    "Claim",
    "Job",
    "PreparedView",
    "Preparer",
    "epoch_identity",
]

#: job states: waiting for a slot, executing, ended with a result, ended without one, replaced by a publication, taken
#: over by a seal, dropped (every member decided), disabled; a seal's own execution: seal, then seal_done
JOB_STATES = ("queued", "running", "done", "failed", "superseded", "taken", "dropped", "disabled", "seal",
              "seal_done")
DEFERRED = "prepare:deferred"  # what planning's backend answers: nothing is executed while planning


def epoch_identity(epoch: Epoch | None) -> tuple[int, str] | None:
    """An epoch's identity for preparation: its publication ``seq`` and its content ``version_id``. A republication
    of the same content is another identity: a result prepared under one is never read under the other."""
    if epoch is None:
        return None
    return (int(epoch.ref.seq), str(epoch.ref.version_id))


@dataclass(eq=False)
class Job:
    """One prepared execution of one request under one epoch identity (or, ``seal`` true, a seal's own execution of
    it, which members planned meanwhile attach to). ``check`` and ``family`` are those of the member that created it:
    its result is usable when it gives that check a logical verdict (a seal judges it again for its own check)."""

    id: str
    identity: tuple[int, str]
    request: ExecRequest
    check: Check
    family: str
    trigger: str
    state: str = "queued"
    seal: bool = False
    members: list = field(default_factory=list)
    result: ExecResult | None = None
    usable: bool = False
    attempts: int = 0
    error: str | None = None
    taken_by: str | None = None
    t_queued: float = 0.0
    t_start: float | None = None
    t_done: float | None = None
    event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def request_id(self) -> str:
        return self.request.request_id


@dataclass(frozen=True)
class Claim:
    """How a seal's requests stand, at one look: finished results to try (``ready``), jobs to wait for
    (``running``), and requests to execute at the seal now, with why (``now``)."""

    ready: Mapping[str, ExecResult]
    running: Mapping[str, Job]
    now: Mapping[str, str]


class _Capture:
    """Planning's backend: it executes nothing. It records the requests of its first call (the requests a plan
    leaves to execute first) and answers every request as deferred (an infrastructure error, never a verdict)."""

    def __init__(self) -> None:
        self.first: list[ExecRequest] | None = None

    def execute(self, requests: Sequence[ExecRequest], store: Any) -> list[ExecResult]:
        if self.first is None:
            self.first = list(requests)
        return [infra_result(r, DEFERRED) for r in requests]


class _PlanningCache:
    """What planning reads instead of the seal's cache: the seal's cache (read only), and the results this preparer
    holds for the target's identity (its jobs', and those of the seal's own executions it has seen end). Writes are
    dropped: planning never writes the seal's cache."""

    def __init__(self, cache: Mapping[str, ExecResult] | None, preparer: Preparer, identity: tuple[int, str]) -> None:
        self.cache, self.preparer, self.identity = cache, preparer, identity

    def get(self, request_id: str, default: Any = None) -> Any:
        if self.cache is not None:
            hit = self.cache.get(request_id)
            if hit is not None:
                return hit
        with self.preparer.lock:
            job = self.preparer.jobs.get((self.identity, request_id))
            if job is not None and job.state in ("done", "seal_done") and job.result is not None:
                return job.result
        return default

    def __setitem__(self, request_id: str, result: ExecResult) -> None:
        return None


class Preparer:
    """Background preparation of a repair seal's re-executions (see the module documentation).

    ``epochs`` is the registry (its pointer, read by :meth:`poll`, and its publications); ``scorer`` is the scorer the
    seal uses (its adapters, store, backend, policy, retries, options and cache): preparation plans with it and executes
    through its backend. ``capacity`` is how many executions the backend runs at once, for the seal and the preparer
    together; ``slots`` (default ``capacity`` less an eighth) bounds the jobs running at once. ``wait_s`` bounds a
    seal's wait for one running job. ``startup`` is the startup epoch's publication ``seq`` (default: the first grader
    publication). ``log`` is a JSON-lines path for the events; ``keep_events`` (default) also keeps them in
    :attr:`events` (a long run may prefer the log alone). With ``threads`` the preparer starts its pointer poll,
    planner and executor threads now; otherwise the caller drives it (:meth:`poll`, :meth:`plan`, :meth:`run`), or calls
    :meth:`start` later. It is also a context manager that starts and stops its threads."""

    def __init__(self, epochs: Any, scorer: Scorer, *, capacity: int = 16, slots: int | None = None,
                 poll_s: float = 1.0, wait_s: float = 3600.0, startup: int | None = None,
                 log: str | os.PathLike | None = None, keep_events: bool = True, planners: int = 1,
                 threads: bool = False, clock: Callable[[], float] = time.time) -> None:
        if scorer.backend is None:
            raise ValueError("preparation executes through the scorer's backend: the scorer has none")
        self.epochs = epochs
        self.scorer = scorer.with_options(prepared=None)
        self.capacity = max(1, int(capacity))
        self.slots = max(1, int(slots if slots is not None else self.capacity - max(1, self.capacity // 8)))
        self.poll_s = float(poll_s)
        self.wait_s = float(wait_s)
        self.planners = max(1, int(planners))
        self.clock = clock
        self.log_path = os.fspath(log) if log is not None else None
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.jobs: dict[tuple, Job] = {}       # (identity, request id) -> the job (or a seal's own execution)
        self.by_id: dict[str, Job] = {}
        self.queue: collections.deque = collections.deque()  # jobs waiting for a slot, oldest first
        self.items: dict[str, ScoreInput] = {}  # member id -> what the seal will score
        self.groups: collections.OrderedDict = collections.OrderedDict()  # admitted group -> its member ids
        self.sealed: set[str] = set()
        self.retired: set[str] = set()
        self.deciding: dict[str, tuple] = {}    # member id -> (identity, seal id) of the seal deciding it
        self.planned: dict[str, tuple] = {}     # member id -> the identity it was last planned under
        self.plan_queue: collections.deque = collections.deque()
        self.target: Epoch | None = None
        self.startup = None if startup is None else int(startup)
        self.disabled: str | None = None
        self.running = 0
        self.seal_inflight = 0
        self.events: list[dict] = []
        self.keep_events = bool(keep_events)
        self.stopped = False
        self._n = 0
        self._threads: list[threading.Thread] = []
        self._event({"kind": "start", "capacity": self.capacity, "slots": self.slots, "poll_s": self.poll_s,
                     "wait_s": self.wait_s, "planners": self.planners, "startup": self.startup})
        self.poll()  # the target before anything is planned
        if threads:
            self.start()

    # ------------------------------------------------------------------------------------------------ lifecycle

    def start(self) -> Preparer:
        """Start the pointer poll, the planner and the executor threads (once)."""
        with self.lock:
            if self._threads or self.stopped:
                return self
        try:
            self._spawn(self._poll_loop, "agent-ecc-prepare-poll")
            for i in range(self.planners):
                self._spawn(self._plan_loop, f"agent-ecc-prepare-plan-{i}")
            for i in range(self.slots):
                self._spawn(self._exec_loop, f"agent-ecc-prepare-exec-{i}")
        except Exception as error:  # noqa: BLE001 - e.g. no thread can start: nothing is prepared
            self.disable(f"thread start: {type(error).__name__}: {error}")
        return self

    def stop(self, *, join_s: float | None = 5.0) -> None:
        """Stop the threads (running jobs finish; nothing new starts)."""
        with self.lock:
            self.stopped = True
            self.cond.notify_all()
        for t in list(self._threads):
            if t is not threading.current_thread():
                t.join(timeout=join_s)

    def __enter__(self) -> Preparer:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def _spawn(self, fn: Callable[[], None], name: str) -> None:
        t = threading.Thread(target=self._guarded, args=(fn,), name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def _guarded(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as error:  # noqa: BLE001 - preparation never fails the run; the seal executes instead
            self.disable(f"{type(error).__name__}: {error}", traceback.format_exc(limit=8))

    def disable(self, reason: str, detail: str | None = None) -> None:
        """Stop preparing: queued jobs are disabled, and every seal executes everything itself."""
        dropped = []
        with self.lock:
            if self.disabled:
                return
            self.disabled = reason
            for job in list(self.queue):
                job.state, job.error, job.t_done = "disabled", "preparer_disabled", self.clock()
                job.event.set()
                dropped.append(job)
            self.queue.clear()
            self.cond.notify_all()
        self._event({"kind": "disabled", "reason": reason, "detail": detail})
        for job in dropped:
            self._job_event(job)

    # ------------------------------------------------------------------------------------------------ events

    def _event(self, entry: dict) -> None:
        entry = {"t": self.clock(), **entry}
        if self.keep_events:
            with self.lock:
                self.events.append(entry)
        if self.log_path is not None:
            try:
                line = json.dumps(entry, sort_keys=True, default=str) + "\n"
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line)
            except OSError:  # a log that cannot be written never stops preparation
                pass

    def _job_event(self, job: Job, **extra: Any) -> None:
        cost = job.result.cost if job.result is not None else None
        self._event({"kind": "job", "job": job.id, "seal_execution": job.seal, "identity": list(job.identity),
                     "request_id": job.request_id, "task": job.request.task_id, "check": job.check.name,
                     "state": job.state, "trigger": job.trigger, "members": list(job.members),
                     "t_queued": job.t_queued, "t_start": job.t_start, "t_done": job.t_done,
                     "attempts": job.attempts, "usable": job.usable, "error": job.error,
                     "result_status": job.result.status if job.result is not None else None,
                     "container_seconds": cost.container_seconds if cost is not None else None,
                     "taken_by": job.taken_by, **extra})

    def stats(self) -> dict:
        """Counts: jobs by state, seal answers by how and why, background executions and their sandbox seconds."""
        with self.lock:
            jobs = list(self.by_id.values())
            events = list(self.events)
        by_state: dict[str, int] = collections.Counter(j.state for j in jobs if not j.seal)
        uses: dict[str, int] = collections.Counter(f"{e['how']}:{e['why']}" for e in events if e["kind"] == "use")
        background = [j for j in jobs if not j.seal and j.result is not None]
        return {"jobs": dict(sorted(by_state.items())), "uses": dict(sorted(uses.items())),
                "background_attempts": sum(j.attempts for j in background),
                "background_container_seconds": round(sum(j.result.cost.container_seconds for j in background), 6),
                "disabled": self.disabled}

    # ------------------------------------------------------------------------------------------------ the target

    def _identity(self) -> tuple[int, str] | None:
        return epoch_identity(self.target)

    def _active(self) -> bool:
        """Whether the target can differ from what a record was graded under: not under the startup epoch."""
        if self.target is None:
            return False
        if self.startup is None:
            return True
        return int(self.target.ref.seq) != self.startup

    def poll(self) -> bool:
        """Read the pointer. A new target supersedes the queued jobs of every other identity (running ones finish,
        their results bound to their own) and plans again every admitted member no seal has decided. Returns
        whether the target changed."""
        if self.disabled:
            return False
        epoch = self.epochs.read_epoch()
        if epoch is None:
            return False
        if self.startup is None:
            graders = list(self.epochs.publications("grader"))
            if graders:
                self.startup = int(graders[0].seq)
        identity = epoch_identity(epoch)
        superseded, swept = [], []
        with self.lock:
            if identity == self._identity():
                return False
            self.target = epoch
            keep = collections.deque()
            for job in self.queue:
                if job.identity != identity:
                    job.state, job.t_done = "superseded", self.clock()
                    job.event.set()
                    superseded.append(job)
                else:
                    keep.append(job)
            self.queue = keep
            active = self._active()
            if active:
                for gid, mids in self.groups.items():
                    if any(m not in self.sealed and m not in self.retired for m in mids):
                        swept.append(gid)
                        self.plan_queue.append(("publication", gid, list(mids)))
            self.cond.notify_all()
        for job in superseded:
            self._job_event(job, superseded_by=list(identity))
        self._event({"kind": "target", "identity": list(identity), "name": epoch.ref.name,
                     "published_at": epoch.ref.published_at, "active": active,
                     "superseded": [j.id for j in superseded], "swept_groups": len(swept)})
        return True

    def _poll_loop(self) -> None:
        while True:
            with self.lock:
                if self.stopped or self.disabled:
                    return
            self.poll()
            with self.lock:
                if self.stopped or self.disabled:
                    return
                self.cond.wait(timeout=self.poll_s)

    # ------------------------------------------------------------------------------------------------ the run's events

    def admit(self, candidates: Candidate | Iterable[Candidate]) -> int:
        """Groups entered the buffer: each member with an execution record is planned under the current target (by a
        planner thread, or by :meth:`plan`). Returns the members noted. Never blocks on planning."""
        if isinstance(candidates, Candidate):
            candidates = [candidates]
        n = 0
        with self.lock:
            if self.disabled:
                return 0
            for c in candidates:
                mids = []
                for m in c.members:
                    if not isinstance(m, Member) or m.execution is None:
                        continue
                    self.items[m.member_id] = ScoreInput(m.execution, m.prior)
                    mids.append(m.member_id)
                self.groups[c.group_id] = mids
                self.plan_queue.append(("admission", c.group_id, mids))
                n += len(mids)
            self.cond.notify_all()
        return n

    def retire(self, member_ids: Iterable[str]) -> None:
        """Members of groups retired without being decided: never prepared."""
        ids = {str(m) for m in member_ids}
        with self.lock:
            self.retired |= ids
            dropped = self._drop_decided()
        self._event({"kind": "retired", "members": sorted(ids)})
        for job in dropped:
            self._job_event(job)

    def note_deciding(self, member_ids: Iterable[str], epoch: Epoch | None, *, seal_id: str | None = None) -> None:
        """A seal decides these members under ``epoch`` (one decision round). Seals are serialized: a seal that
        starts deciding ends every other seal's marks (the members of a failed round can be prepared again)."""
        identity = epoch_identity(epoch)
        ids = [str(m) for m in member_ids]
        with self.lock:
            for mid, (_, sid) in list(self.deciding.items()):
                if sid != seal_id:
                    del self.deciding[mid]
            for mid in ids:
                self.deciding[mid] = (identity, seal_id)
            dropped = self._drop_decided()
        self._event({"kind": "deciding", "seal_id": seal_id, "identity": list(identity) if identity else None,
                     "members": sorted(ids)})
        for job in dropped:
            self._job_event(job)

    def note_sealed(self, member_ids: Iterable[str], *, seal_id: str | None = None) -> None:
        """A seal finished: the members it decided (handed over or excluded) are never prepared again."""
        ids = {str(m) for m in member_ids}
        with self.lock:
            self.sealed |= ids
            for mid in ids:
                self.deciding.pop(mid, None)
            dropped = self._drop_decided()
        self._event({"kind": "sealed", "seal_id": seal_id, "members": sorted(ids)})
        for job in dropped:
            self._job_event(job)

    def _drop_decided(self) -> list[Job]:
        """Under the lock: queued jobs whose every member is sealed, retired, or being decided under another identity
        than the job's are dropped (a seal under the job's identity takes it over instead)."""
        keep, dropped = collections.deque(), []
        for job in self.queue:
            if job.state == "queued" and job.members and all(
                    m in self.sealed or m in self.retired
                    or (m in self.deciding and self.deciding[m][0] != job.identity) for m in job.members):
                job.state, job.t_done = "dropped", self.clock()
                job.event.set()
                dropped.append(job)
            else:
                keep.append(job)
        self.queue = keep
        return dropped

    # ------------------------------------------------------------------------------------------------ planning

    def plan(self) -> int:
        """Plan every queued admission, sweep and chain item now, in the caller's thread. Returns items planned."""
        n = 0
        while True:
            with self.lock:
                if not self.plan_queue:
                    return n
                item = self.plan_queue.popleft()
            self._plan_group(*item)
            n += 1

    def _plan_loop(self) -> None:
        while True:
            with self.lock:
                while not self.plan_queue and not self.stopped and not self.disabled:
                    self.cond.wait(timeout=1.0)
                if self.stopped or self.disabled:
                    return
                item = self.plan_queue.popleft()
            self._plan_group(*item)

    def _plan_group(self, trigger: str, group_id: str, member_ids: Sequence[str]) -> None:
        for mid in member_ids:
            with self.lock:
                target, active = self.target, self._active()
            if target is None or not active or self.disabled:
                return
            try:
                self._plan_member(mid, group_id, trigger, target)
            except Exception as error:  # noqa: BLE001 - one member never disables preparation
                self._event({"kind": "plan", "member": mid, "group": group_id, "trigger": trigger,
                             "decision": "plan_error", "error": f"{type(error).__name__}: {error}"[:300]})

    def _plan_member(self, mid: str, group_id: str | None, trigger: str, target: Epoch, *,
                     again: bool = False) -> None:
        identity = epoch_identity(target)
        assert identity is not None
        entry: dict = {"kind": "plan", "member": mid, "group": group_id, "trigger": trigger,
                       "identity": list(identity)}
        skip = None
        with self.lock:
            item = self.items.get(mid)
            if mid in self.sealed or mid in self.retired:
                skip = "skip_decided"
            elif mid in self.deciding:
                skip = "skip_in_session" if self.deciding[mid][0] == identity else "skip_other_epoch_session"
            elif not again and self.planned.get(mid) == identity:
                return  # already planned under this target (an admission and a sweep can both name it)
            else:
                self.planned[mid] = identity
        if skip is not None:
            self._event({**entry, "decision": skip})
            return
        if item is None:
            self._event({**entry, "decision": "no_record"})
            return
        nxt = self._next_request(item, target, identity)
        if nxt is None:
            self._event({**entry, "decision": "nothing"})
            return
        request, check, family = nxt
        rid = request.request_id
        with self.lock:
            if self._identity() != identity:
                decision, jid = "stale_target", None  # a publication arrived meanwhile: its sweep plans it again
            elif mid in self.sealed or mid in self.retired or mid in self.deciding:
                decision, jid = "skip_decided", None
            else:
                job = self.jobs.get((identity, rid))
                if job is not None and job.state in ("queued", "running", "done", "seal"):
                    if mid not in job.members:
                        job.members.append(mid)
                    decision, jid = ("attached_seal" if job.seal else "attached"), job.id
                else:
                    self._n += 1
                    job = Job(id=f"p{self._n}", identity=identity, request=request, check=check, family=family,
                              trigger=trigger, members=[mid], t_queued=self.clock())
                    self.jobs[(identity, rid)] = job
                    self.by_id[job.id] = job
                    self.queue.append(job)
                    self.cond.notify_all()
                    decision, jid = "job", job.id
        self._event({**entry, "decision": decision, "job": jid, "request_id": rid, "check": check.name})

    def _next_request(self, item: ScoreInput, target: Epoch,
                      identity: tuple[int, str]) -> tuple[ExecRequest, Check, str] | None:
        """The first request the seal's plan of ``item`` under ``target`` leaves to execute, given the run's cache and
        the results prepared under ``identity`` (None when the plan executes nothing), with the check that asks it."""
        capture = _Capture()
        planner = self.scorer.with_options(backend=capture, cache=_PlanningCache(self.scorer.cache, self, identity),
                                           retries=0, allow=Action.REEXEC, prepared=None)
        planner.score([item], target)
        if not capture.first:
            return None
        request = capture.first[0]
        record = item.execution
        adapter = self.scorer.adapters.get(record.family)
        lookup = target.version.lookup(record.task_id, record.world_version)
        for check in (iter_checks(lookup.spec) if lookup.spec is not None else ()):
            try:
                asked = adapter.reexec_request(record, check, self.scorer.store) if adapter is not None else None
            except Exception:  # noqa: BLE001 - an artifact the adapter cannot read asks nothing
                asked = None
            if asked is not None and asked.request_id == request.request_id:
                return request, check, record.family
        return None

    # ------------------------------------------------------------------------------------------------ execution

    def _room(self) -> bool:
        """Under the lock: a slot is free and the backend has room once the seal's own executions are counted."""
        return self.running < self.slots and self.running + self.seal_inflight < self.capacity

    def take_job(self) -> Job | None:
        """The oldest queued job, marked running; None when there is none or no room."""
        dropped = []
        try:
            with self.lock:
                if self.disabled or not self._room():
                    return None
                dropped = self._drop_decided()
                while self.queue:
                    job = self.queue.popleft()
                    if job.state != "queued":
                        continue
                    job.state, job.t_start = "running", self.clock()
                    self.running += 1
                    return job
                return None
        finally:
            for job in dropped:
                self._job_event(job)

    def run(self, limit: int | None = None) -> int:
        """Execute queued jobs now, in the caller's thread, one at a time (and plan what their results lead to).
        Returns jobs run."""
        n = 0
        while limit is None or n < limit:
            job = self.take_job()
            if job is None:
                return n
            self.run_job(job)
            n += 1
            self.plan()
        return n

    def _exec_loop(self) -> None:
        while True:
            with self.lock:
                while (not self.queue or not self._room()) and not self.stopped and not self.disabled:
                    self.cond.wait(timeout=1.0)
                if self.stopped or self.disabled:
                    return
            job = self.take_job()
            if job is not None:
                self.run_job(job)

    def _execute_once(self, request: ExecRequest) -> ExecResult:
        results = list(self.scorer.backend.execute([request], self.scorer.store))  # type: ignore[union-attr]
        if len(results) != 1 or not isinstance(results[0], ExecResult):
            raise RuntimeError(f"the backend returned {len(results)} results for one request")
        if results[0].request_id != request.request_id:
            raise RuntimeError("the backend answered another request")
        return results[0]

    def _judge(self, job: Job, result: ExecResult) -> tuple[bool, bool]:
        """(usable, final): whether the result gives the job's check a logical verdict, as the scorer reads it, and
        whether executing again cannot change that."""
        adapter = self.scorer.adapters.get(job.family)
        if adapter is None:
            return False, True
        _, verdict, _ = self.scorer.read_result(adapter, job.check, job.request, result)
        return verdict.outcome.logical, verdict.final

    def run_job(self, job: Job) -> None:
        """Execute a running job, with the scorer's retries for results that give no logical verdict. A backend that
        raises (or answers wrongly) fails the job; the seal then executes the request itself."""
        result, error, usable = None, None, False
        try:
            while True:
                result = self._execute_once(job.request)
                job.attempts += 1
                usable, final = self._judge(job, result)
                if usable or final or job.attempts > self.scorer.retries or _gave_up(result):  # as the scorer does
                    break
        except Exception as exc:  # noqa: BLE001 - a failed job is executed at the seal instead
            result, error = None, f"{type(exc).__name__}: {exc}"[:300]
        finally:
            with self.lock:
                self.running -= 1
                job.result, job.error, job.usable, job.t_done = result, error, bool(usable and error is None), \
                    self.clock()
                job.state = "done" if (result is not None and error is None) else "failed"
                job.event.set()
                self.cond.notify_all()
        self._job_event(job)
        if job.usable:
            self._continue(job)

    def _continue(self, job: Job) -> None:
        """Plan again, with the job's result, the members it served: a conjunction's next suite, if any."""
        with self.lock:
            target = self.target
            if epoch_identity(target) != job.identity:
                return  # a publication superseded the job's epoch; its sweep plans the members again
            members = list(job.members)
        for mid in members:
            try:
                self._plan_member(mid, None, "chain", target, again=True)  # type: ignore[arg-type]
            except Exception as error:  # noqa: BLE001
                self._event({"kind": "plan", "member": mid, "trigger": "chain", "decision": "plan_error",
                             "error": f"{type(error).__name__}: {error}"[:300]})

    # ------------------------------------------------------------------------------------------------ the seal

    def view(self, epoch: Epoch, *, seal_id: str | None = None, batch_id: str | None = None) -> PreparedView:
        """The view a seal under ``epoch`` gives its scorer: prepared results of exactly that epoch identity."""
        return PreparedView(self, epoch, seal_id=seal_id, batch_id=batch_id)


class PreparedView:
    """One seal's view of a :class:`Preparer`, bound to the seal's batch epoch (its publication and content). The
    scorer asks it (:meth:`claim`, :meth:`wait`) and tells it what it used (:meth:`answered`) and what the seal
    executes itself (:meth:`executing`, :meth:`executed`)."""

    def __init__(self, preparer: Preparer, epoch: Epoch, *, seal_id: str | None = None,
                 batch_id: str | None = None) -> None:
        identity = epoch_identity(epoch)
        if identity is None:
            raise ValueError("a prepared view is bound to an epoch")
        self.preparer = preparer
        self.identity = identity
        self.seal_id = seal_id
        self.batch_id = batch_id
        self.uses: list[dict] = []
        self._claimed: dict[str, tuple[str, Job]] = {}  # request id -> (ready | running, job)
        self._placeholders: dict[str, Job] = {}
        self._lock = threading.Lock()

    def _use(self, rid: str, how: str, why: str, job: Job | None, waited_s: float = 0.0) -> None:
        use = {"request_id": rid, "how": how, "why": why, "job": job.id if job is not None else None,
               "waited_s": round(float(waited_s), 6)}
        with self._lock:
            self.uses.append(use)
        self.preparer._event({"kind": "use", "seal_id": self.seal_id, "batch_id": self.batch_id,
                              "identity": list(self.identity), **use})

    def _failed(self, where: str, error: BaseException) -> None:
        """An error in the view never fails a seal: preparation is disabled, and the seal executes what it needs."""
        try:
            self.preparer.disable(f"view.{where}: {type(error).__name__}: {error}"[:300],
                                  traceback.format_exc(limit=8))
        except Exception:  # noqa: BLE001
            pass

    def claim(self, requests: Sequence[ExecRequest]) -> Claim:
        """Classify a wave's requests (each asked for the first time): finished results to try, running jobs to wait
        for, and requests to execute at the seal now (a queued job is taken over: it never runs). A request to execute
        now gets the seal's placeholder in the same critical section, so a member planned meanwhile attaches to the
        seal's execution instead of starting a job of its own. Never raises: on an error every request is executed at
        the seal."""
        try:
            return self._claim(requests)
        except Exception as error:  # noqa: BLE001
            self._failed("claim", error)
            return Claim(ready={}, running={}, now={r.request_id: "preparer_error" for r in requests})

    def _claim(self, requests: Sequence[ExecRequest]) -> Claim:
        p = self.preparer
        ready: dict[str, ExecResult] = {}
        running: dict[str, Job] = {}
        now: dict[str, str] = {}
        taken: list[Job] = []
        notes: list[tuple[str, str, Job | None]] = []
        with p.lock:
            for request in requests:
                rid = request.request_id
                if p.disabled:
                    now[rid] = "disabled"
                    notes.append((rid, "disabled", None))
                    continue
                job = p.jobs.get((self.identity, rid))
                if job is None or job.seal:
                    other = any(k == rid and ident != self.identity and not j.seal
                                for (ident, k), j in p.jobs.items())
                    why = "other_identity" if other else "no_job"
                    now[rid] = why
                    notes.append((rid, why, None))
                    self._placeholder(request, job)
                elif job.state == "queued":
                    job.state, job.taken_by, job.t_done = "taken", self.seal_id, p.clock()
                    try:
                        p.queue.remove(job)
                    except ValueError:
                        pass
                    job.event.set()
                    taken.append(job)
                    now[rid] = "taken_over"
                    notes.append((rid, "taken_over", job))
                    self._placeholder(request, job)
                elif job.state == "running":
                    running[rid] = job
                    self._claimed[rid] = ("running", job)
                elif job.state == "done" and job.result is not None and job.result.status == RUN_OK:
                    ready[rid] = job.result
                    self._claimed[rid] = ("ready", job)
                else:
                    why = "error" if job.state == "done" else job.state
                    now[rid] = why
                    notes.append((rid, why, job))
                    self._placeholder(request, job)
        for job in taken:
            p._job_event(job)
        for rid, why, job in notes:
            self._use(rid, "sync", why, job)
        return Claim(ready=ready, running=running, now=now)

    def _placeholder(self, request: ExecRequest, current: Job | None) -> Job:
        """Under the preparer's lock: the seal's own execution of ``request``, registered under its key (replacing a
        finished, failed or taken entry; a seal's earlier placeholder for it is reused)."""
        p = self.preparer
        rid = request.request_id
        with self._lock:
            mine = self._placeholders.get(rid)
        if mine is not None:
            return mine
        p._n += 1
        ph = Job(id=f"s{p._n}", identity=self.identity, request=request,
                 check=current.check if current is not None else Check(name="seal"),
                 family=current.family if current is not None else request.family, trigger="seal", state="seal",
                 seal=True, t_queued=p.clock(), t_start=p.clock(), taken_by=self.seal_id)
        if current is not None and current.seal:
            ph.members = list(current.members)  # members attached to an earlier seal's execution stay attached
        p.jobs[(self.identity, rid)] = ph
        p.by_id[ph.id] = ph
        with self._lock:
            self._placeholders[rid] = ph
        return ph

    def wait(self, job: Job) -> tuple[ExecResult | None, float]:
        """Wait (at most the preparer's ``wait_s``) for a running job; its result if it ended with outputs."""
        t0 = time.monotonic()
        try:
            job.event.wait(timeout=self.preparer.wait_s)
            waited = time.monotonic() - t0
            with self.preparer.lock:
                if job.state == "done" and job.result is not None and job.result.status == RUN_OK:
                    return job.result, waited
        except Exception as error:  # noqa: BLE001
            self._failed("wait", error)
        return None, time.monotonic() - t0

    def answered(self, rid: str, *, used: bool, waited_s: float = 0.0) -> None:
        """Record how a claimed request (ready or running) was answered: by its prepared result, or at the seal."""
        try:
            self._answered(rid, used=used, waited_s=waited_s)
        except Exception as error:  # noqa: BLE001
            self._failed("answered", error)

    def _answered(self, rid: str, *, used: bool, waited_s: float) -> None:
        kind, job = self._claimed.get(rid, ("ready", None))
        if used:
            self._use(rid, "prepared", "waited" if kind == "running" else "ready", job, waited_s)
            return
        if job is None or job.state == "done":
            why = "undeterminable" if (job is not None and job.result is not None
                                       and job.result.status == RUN_OK) else "error"
        elif job.state == "running":
            why = "job_unfinished"
        else:
            why = job.state
        self._use(rid, "sync", why, job, waited_s)

    def executing(self, requests: Sequence[ExecRequest]) -> None:
        """The seal executes these requests itself now: they hold the backend first (counted against the preparer's
        capacity), and members planned meanwhile attach to the seal's execution instead of a new job."""
        try:
            self._executing(requests)
        except Exception as error:  # noqa: BLE001
            self._failed("executing", error)

    def _executing(self, requests: Sequence[ExecRequest]) -> None:
        p = self.preparer
        with p.lock:
            p.seal_inflight += len(requests)
            for request in requests:
                current = p.jobs.get((self.identity, request.request_id))
                if current is not None and current.state in ("queued", "running"):
                    continue  # live preparation the seal did not claim (a retry's request): it is left alone
                self._placeholder(request, current)

    def executed(self, requests: Sequence[ExecRequest], results: Mapping[str, ExecResult]) -> None:
        """The seal's own executions of these requests ended: members attached to them are planned again (they find
        the seal's results in its cache)."""
        try:
            self._executed(requests, results)
        except Exception as error:  # noqa: BLE001
            self._failed("executed", error)

    def _executed(self, requests: Sequence[ExecRequest], results: Mapping[str, ExecResult]) -> None:
        p = self.preparer
        ended = []
        with p.lock:
            p.seal_inflight = max(0, p.seal_inflight - len(requests))
            for request in requests:
                with self._lock:
                    ph = self._placeholders.pop(request.request_id, None)
                if ph is None:
                    continue
                ph.state, ph.t_done = "seal_done", p.clock()
                ph.result = results.get(request.request_id)
                ph.attempts = 1 if ph.result is not None else 0
                ph.event.set()
                ended.append(ph)
            p.cond.notify_all()
        for ph in ended:
            p._job_event(ph)
            if ph.members:
                with p.lock:
                    target = p.target
                if epoch_identity(target) == ph.identity:
                    for mid in list(ph.members):
                        try:
                            p._plan_member(mid, None, "chain", target, again=True)  # type: ignore[arg-type]
                        except Exception:  # noqa: BLE001
                            pass

    def summary(self) -> dict:
        """How every request this seal offered was answered, by how and why, and the time it waited."""
        with self._lock:
            uses = list(self.uses)
        counts = collections.Counter(f"{u['how']}:{u['why']}" for u in uses)
        return {"identity": list(self.identity), "requests": len(uses),
                "prepared": sum(1 for u in uses if u["how"] == "prepared"),
                "sync": sum(1 for u in uses if u["how"] == "sync"), "by_reason": dict(sorted(counts.items())),
                "waited_s": round(sum(u["waited_s"] for u in uses), 6)}
