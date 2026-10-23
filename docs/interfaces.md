# Interfaces

The package, its record formats, and the protocols an integration implements. The rules these interfaces follow are in
[contract.md](contract.md).

## Modules

| Module | Responsibility |
|---|---|
| `agent_ecc.canonical` | Canonical JSON, `sha256:` digests, immutable JSON values (`FrozenDict`). |
| `agent_ecc.store` | Content-addressed stores: `MemoryStore`, `DirectoryStore`, `LocatorStore`, `ChainStore`. |
| `agent_ecc.records` | The four record types and their parts; JSON-lines helpers. |
| `agent_ecc.versions` | Checks, conjunctions, grader and world versions; publication (`MemoryRegistry`, `DirectoryRegistry`). |
| `agent_ecc.compare` | Per-check decisions (`plan_check`) and conjunction evaluation (`Conjunction`). |
| `agent_ecc.evidence` | Evidence levels, dispositions per change type, sufficiency checks, masking for ablations. |
| `agent_ecc.scoring` | `Scorer`: scoring records under a version; the `Adapter` protocol; `grade_recorded`. |
| `agent_ecc.seal` | `SealSession` (with a target pin, `pin=`, and prepared results, `prepared=`), `SealJournal`, `apply_rewards`; the arms of a grader switch (`Arm`, `arm_session`). |
| `agent_ecc.prepare` | Prepared re-execution: `Preparer` (background jobs started at admission) and the view a seal takes results from (`PreparedView`). |
| `agent_ecc.export` | Evaluation manifests; consumption records and training manifests. |
| `agent_ecc.audit` | Audits of a finished run: the epochs its seals read (`audit_epochs`: I3, I4), identical executions with different verdicts (`verdict_conflicts`). |
| `agent_ecc.recompute` | The `Backend` protocol, `RecordedBackend`, `RecordingBackend`, `LocalDockerBackend`, the HTTP service. |
| `agent_ecc.adapters.swebench` | SWE-bench evidence, verdict rules with the official harness, the patch filter, reference sets, SWE-ABS conjunction versions. |
| `agent_ecc.adapters.appworld` | AppWorld evidence, answer re-grading, end-state re-grading with the AppWorld evaluator. |

## The four calls

| Call | In agent-ecc |
|---|---|
| collect a record | an adapter's record builders (`SWEBenchAdapter.record` and `.run`, `AppWorldAdapter.record_from_report`), and `scoring.grade_recorded` for the producer's own verdict |
| score under a version | `Scorer.score(items, version)` -> one `Scored` per item, holding a `ScoreRecord` |
| export an evaluation | `export.evaluation_manifest`, `export.evaluate`, `export.compare_evaluations`, `export.reproduce_evaluation` |
| commit to training | `SealSession(...).assemble(...)` / `.decide(...)` (or `seal.arm_session(arm, ...)` for a comparison arm; `prepared=` to take results a `Preparer` started at admission), `.finish(..., journal=SealJournal(...))`, `seal.apply_rewards`; afterwards `export.consumption_records` / `export.training_manifest`, `audit.audit_epochs` |

## Identity

Every record type is a frozen dataclass whose identity is the digest of its canonical JSON body (all fields except the
identity itself): `ExecutionRecord.record_id`, `ScoreRecord.score_id`, `SealDecision.decision_id`,
`ConsumptionRecord.consumption_id`; likewise `GraderVersion.version_id`, `WorldVersion.version_id`,
`ExecRequest.request_id` and the manifests' `manifest_id`. `to_dict()` adds the identity, `from_dict()` verifies it.
Mappings inside records are frozen; changing a record means making a new one, with a new identity.

## Record formats

All formats are JSON; digests are `sha256:<64 hex>`.

### ExecutionRecord (`agent-ecc/execution/1`)

```json
{"schema": "agent-ecc/execution/1", "family": "swebench", "task_id": "pkg__pkg-1", "trajectory_id": "pkg__pkg-1/a0",
 "world_version": "example.org/img@sha256:...", "world_observations": [],
 "artifacts": [{"role": "patch", "digest": "sha256:...", "size": 912, "media_type": "text/x-diff"}],
 "runs": [{"check": "tests",
           "execution": {"image": "...", "eval_script": "sha256:...", "apply_cmds": "sha256:..."},
           "status": "ok",
           "outputs": [{"role": "test_log", "digest": "sha256:...", "size": 80211, "media_type": "text/plain"}],
           "facts": {"applied": true, "timed_out": false, "has_output_markers": true, "test_exit_code": 1}}],
 "facts": {}, "meta": {}, "record_id": "sha256:..."}
```

* `world_version` is the one version the trajectory resolved at start; `world_observations` lists versions seen at
  later serving events (a record whose observations differ from `world_version` is torn).
* `artifacts` are what the agent produced; `runs` are test executions made when it was graded, each with the execution
  identity of what ran, its raw outputs, and facts the runner observed. A run's `status` is `ok` (outputs captured) or
  `error` (an infrastructure failure; never a verdict). A SWE-bench run's facts may hold `application`, the record of
  its patch application, which decides whether a patch that did not apply is a fail or an infrastructure error
  (`adapters.swebench.apply_infrastructure_error`). Its fields, as executors record them: `attempts` (each with
  `command`, `exit_code`, `error`: the application commands tried in order), `steps` (other commands of the
  application, such as the restore of a pristine tree between two attempts, with the same fields), `reverse_check`
  (the already-applied check: `exit_code`, `error`), `executor_error` (the sandbox executor failed), and `infra_error`
  (a strict application that stopped at an executor error or a timeout). An `error` in any attempt, step or the
  reverse check, an exit code of 124 in any of them, an `executor_error` or an `infra_error` makes it an
  infrastructure error; otherwise the patch genuinely did not apply.
* Evidence roles used by the shipped adapters: `patch`, `test_log` (SWE-bench); `answer`, `end_state`, `tests`,
  `commands` (AppWorld). `evidence.ROLE_LEVELS` maps roles to evidence levels.

### ScoreRecord (`agent-ecc/score/1`)

```json
{"schema": "agent-ecc/score/1", "execution_id": "sha256:...", "grader_version": "sha256:...",
 "spec_digest": "sha256:...", "outcome": "fail", "method": "reexec", "reason": "tests_failed",
 "parts": [{"check": "tests", "execution_digest": "sha256:...", "grading_digest": "sha256:...", "outcome": "pass",
            "method": "recorded", "reason": "resolved", "tests": [], "run": null, "attempts": 0, "seconds": 0.0},
           {"check": "hardened", "execution_digest": "sha256:...", "grading_digest": "sha256:...", "outcome": "fail",
            "method": "reexec", "reason": "tests_failed",
            "tests": [{"name": "tests/test_mod.py::test_edge", "status": "failed", "group": "FAIL_TO_PASS"}],
            "run": {"check": "hardened", "execution": {"...": "..."}, "status": "ok", "outputs": ["..."], "facts": {}},
            "attempts": 1, "seconds": 61.2}],
 "skipped": [], "cost": {"seconds": 61.2, "executions": 1, "container_seconds": 60.8, "shared": 0},
 "based_on": null, "meta": {}, "score_id": "sha256:..."}
```

* `outcome` is `pass`, `fail` or `undeterminable` (the monotone three-valued AND of the parts: any logical fail
  fails); `method` is the strongest method among the logical parts: `recorded` (verdicts reused because both
  identities are equal, verdicts carried over from a recorded run of another check or of other grading inputs by
  `Adapter.carry_over`, or the producer's own), `reparse`, `reexec`, or `undeterminable` for an undeterminable
  outcome.
* `parts` are the evaluated checks in spec order (an undeterminable part beside a logical fail included); `skipped`
  names the checks never evaluated because a logical fail decided the conjunction first. A part's
  `(execution_digest, grading_digest)` is what a later version's check is matched against for reuse.
* `run` in a re-executed part is the new evidence the execution produced; in a re-parsed part, the recorded run whose
  outputs were judged again; in a carried-over part (method `recorded`, the target check's identities), the recorded
  run the verdict was carried over from. Only a re-executed part's run is new evidence for later versions.
* `based_on` is the score a reused verdict was taken from: set only when every evaluated check took its verdict from
  that one score. A seal treats a member as consistent only when `based_on` is its recorded score of the same spec.
* `grader_version` is null for a verdict recorded under a grader that was never published as a version.

### SealDecision (`agent-ecc/seal/1`)

```json
{"schema": "agent-ecc/seal/1", "seal_id": "4c1d...", "batch_id": "step-42",
 "epoch": {"seq": 3, "version_id": "sha256:...", "name": "g2", "published_at": 1790000000.0, "read_at": 1790000123.4},
 "mode": "repair", "policy": "ecc", "strict": true,
 "groups": [{"group_id": "u17", "round": 1, "role": "selected", "source": "selected", "neutralized": false,
             "neutralize_reason": null,
             "members": [{"member_id": "u17/a0", "execution_id": "sha256:...", "action": "reparse",
                          "status": "repaired_flip", "outcome": "pass", "reason": "resolved",
                          "score_id": "sha256:...", "recorded_reward": 0.0, "target_reward": 1.0,
                          "trained_reward": 1.0, "flipped": true, "short_circuit": false,
                          "consistent": false}]}],
 "excluded": [{"group": {"group_id": "u9", "...": "..."}, "reason": "zero_variance_after_seal"}],
 "target_groups": 4, "rounds": 2, "failed": null, "opened_at": 1790000123.4, "closed_at": 1790000124.0,
 "stats": {"statuses": {"repaired_flip": 1}, "executions": 0, "...": "..."}, "decision_id": "sha256:..."}
```

A pinned seal's decision also holds `"target_pin": {"pin": "e0", "resolved": true, "published": {"seq": 5,
"version_id": "sha256:...", "name": "e1", "...": "..."}}`: the pin, whether it resolved, and the epoch the pointer named
at the read (`epoch` is then the pinned epoch). A decision without a pin has no `target_pin` key, so its identity is
the one it had before pins existed. A seal that took prepared results has `stats["prepared"]`: how every request it
would have executed was answered (`requests`, `prepared`, `sync`, `by_reason` such as `prepared:ready`,
`prepared:waited`, `sync:no_job`, `sync:taken_over`, and `waited_s`). A seal whose executions went over a transport
(`HTTPBackend`) has `stats["transport"]`: `requests`, `attempts`, `retried`, `gave_up` and failed attempts by `reasons`,
counted per request (a post that carries n requests counts n times).

Member `action`: `reuse`, `reparse`, `reexec`, `undeterminable`, `ungoverned`, `pending` (audit and discard: a re-parse
or an execution would be needed), `off`, `failed`. `consistent` is true when the member was graded under exactly the
batch epoch's spec (its recorded verdict reused as a whole): only a consistent member trains its recorded reward, and
only consistent (or ungoverned) members keep a discard-mode group. `status` is a compact name derived from these
(`same` / `flip` compare the target verdict with the recorded label): `consistent`; `repaired_reuse_same` /
`repaired_reuse_flip` (decided from earlier verdicts alone, but not the recorded verdict of this spec);
`repaired_short_same` / `repaired_short_flip` (a logical fail decided it while other checks were never evaluated,
nothing executed); `repaired_same` / `repaired_flip` (a re-parse took part); `repaired_exec_same` /
`repaired_exec_flip` (an execution took part); `ungoverned`, `unknown_world`, `torn`, `no_epoch`, `no_record`,
`unrepairable`, `needs_reparse`, `needs_reexec`, `off`, `failed`. Group `role`: `selected`, `backfill`, `filler`.
`decision.shortfall` is `target_groups` minus the groups delivered.

### ConsumptionRecord (`agent-ecc/consumption/1`)

```json
{"schema": "agent-ecc/consumption/1", "batch_id": "step-42", "step": 42, "seal_id": "4c1d...",
 "decision_id": "sha256:...", "epoch": {"seq": 3, "version_id": "sha256:...", "...": "..."}, "completed_at": 1790000190.2,
 "records": [{"member_id": "u17/a0", "group_id": "u17", "execution_id": "sha256:...", "score_id": "sha256:...",
              "reward": 1.0, "sealed_reward": 1.0, "advantage": 0.866, "loss_tokens": 1834, "identity": "verified",
              "shards": ["dp3"]}],
 "issues": {"records_unsealed": [], "records_unconsumed": [], "group_mismatch": [], "reward_mismatch": [],
            "identity_unverified": [], "duplicate_consumption": [], "loss_tokens_mismatch": [],
            "advantage_inconsistent": [], "zero_loss_tokens": [], "epoch_mixed": []},
 "consumption_id": "sha256:..."}
```

The issue lists are `export.ISSUE_KEYS` (each documented there). A `records_unsealed` entry names the seals that did
hand the record over (`sealed_in`: batch id, epoch seq and version); `zero_loss_tokens` lists sealed records consumed
on no loss token; `epoch_mixed` lists records the step consumed that another seal handed over under another epoch.

## Versions and publication

```json
{"schema": "agent-ecc/grader-version/1", "name": "verified+hardened", "family": "swebench",
 "entries": {"pkg__pkg-1": {"*": {"type": "all_of", "name": "all_of", "parts": [
     {"type": "check", "name": "tests", "execution": {"...": "..."}, "grading": {"...": "..."}, "params": {}},
     {"type": "check", "name": "hardened", "execution": {"...": "..."}, "grading": {"...": "..."}, "params": {}}]}}},
 "governed": null, "meta": {}, "version_id": "sha256:..."}
```

* `entries` maps task -> world version -> spec; the world `"*"` stands for every world. `governed` lists the tasks the
  version makes a claim about (null: every task with an entry).
* `Registry`: `publish_grader(version, source=None, require_coverage=True)`, `publish_world(world, source=None)`,
  `read_epoch() -> Epoch | None` (the seal's one read), `resolve_start() -> StartPin` (a trajectory's pins),
  `current_world()`, `publications(kind=None)`, `grader(version_id)`, `epoch_at(t)`.
* `DirectoryRegistry(root)` layout: `objects/sha256/<hex>` (bodies), `publications.jsonl` (one line per publication:
  `{"seq", "kind": "grader"|"world", "version_id", "name", "published_at", "source"}`), `.publish.lock`. It works across
  processes and machines on one shared filesystem. A line a crashed publisher left without its newline is not a
  publication: readers ignore it, and the next publisher marks it aborted (` #aborted-publication`) so that it never
  parses.

## Adapter protocol (`scoring.Adapter`)

```python
class Adapter(Protocol):
    family: str
    def parse(self, check: Check, run: TestRun, store: ContentStore) -> PartVerdict: ...
    def reexec_request(self, record: ExecutionRecord, check: Check, store: ContentStore) -> ExecRequest | None: ...
    def decide_without_execution(self, record: ExecutionRecord, check: Check) -> PartVerdict | None: ...
    # optional
    def reexec_run(self, check: Check, request: ExecRequest, result: ExecResult) -> TestRun: ...
    def carry_over(self, record: ExecutionRecord, check: Check) -> PartVerdict | None: ...
```

Rules for a new adapter:

1. Put every input that can change a verdict into a check's `execution` (it changes what runs) or `grading` (it
   changes how outputs are judged) identity, as digests of canonical content. Anything left out is assumed not to
   matter, and verdicts will be reused across changes of it.
2. Producers must stamp runs with the execution identity of the check that actually ran, and recorded verdicts with
   the identities of the check that graded them (build both from the same check objects).
3. `parse` is a pure function of the check, the run and the objects it reads. It returns `UNDETERMINABLE` when the
   outputs do not support a logical verdict, and marks a verdict `final` when executing again cannot change it.
4. `reexec_request` returns None when the record lacks what an execution needs. Requests must be content-addressed by
   what runs (so identical work is executed once); a backend returns raw outputs, never verdicts.
5. `decide_without_execution` returns a verdict only when the record's evidence decides it alone (an empty patch). The
   scorer makes such decisions under every policy and at every allowed action, before any reuse: a recorded verdict
   that disagrees with the evidence is not taken.
6. `reexec_run` (optional) turns a re-execution's raw result into the run later versions re-parse. The default keeps
   the check's execution identity with the result's outputs and facts. Define it when the raw outputs depend on
   grading inputs: stamp the run with them, so that a check with other grading inputs does not read the outputs as
   its own (the AppWorld adapter stamps the truth the evaluator ran with, and adds the submitted answer).
7. `carry_over` (optional) returns the verdict one of the record's own runs gives a check whose inputs differ from
   that run's, when the inputs that differ cannot change it, with that run as the verdict's `source` (the SWE-bench
   adapter: a patch that did not apply under the same image, application commands and patch filter fails under any
   eval script and any grading inputs). The scorer asks it under the `ecc` policy only, after equal-identity reuse,
   and decides the check by reuse; the part's `run` names the source run.

The SWE-bench adapter's verdict rule is part of each check's grading identity (`grading["verdict_rule"]`, built by
`adapters.swebench.verdict_rule`), and `parse` grades a run by the rule of the check it is given, whatever the
adapter's own options. There are three rules, none with options:

* `{"rule": "logical"}` (the default, `SWEBenchAdapter()`): the official report on a complete log
  (`classify_log`): the status map of the official `get_logs_eval`, then `get_eval_tests_report` and
  `get_resolution_status`. A listed test without a status fails, a log the official reading rejects fails, a patch
  that genuinely does not apply fails. Infrastructure errors are no verdict: a run with status `error`; an application
  attempt that errored in the executor or exited 124 (`apply_infrastructure_error`, read from the run's `application`
  fact); and, by `parse_log`, the harness's timeout marker ending the log, missing output markers, an end marker
  without the exit-code record after it, a parser exception.
* `{"rule": "reference_set"}`: a hardened suite against its reference set G, held as the labels' PASS_TO_PASS with
  FAIL_TO_PASS empty (`SWEBenchAdapter.reference_check`, whose `patch_exclude` is required: SWE-ABS removes the diffs
  of the hardened test patch's paths first; a check whose labels are not a reference set is refused): it
  passes if and only if every test of G is PASSED or SKIPPED (XFAIL fails, OK is PASSED, exact ids, a test of G without
  a status fails). The same infrastructure errors apply. `reference_set(runs)` computes G from two gold runs read by
  `parse_log` (the tests passing in both; invalid on a disagreement, an invalid run or an empty G).
* `{"rule": "official"}` (`SWEBenchAdapter(verdict="official")`): the official harness's own end-to-end report on
  every log, as SWE-bench's evaluation does (`official_verdict`): every log it does not resolve fails, and so does a
  run that timed out. Use it to reproduce verdicts recorded under it; only the other rules keep infrastructure faults
  out of training labels.

The options `missing_status` and `cross_check` of an earlier version are refused (nothing makes a listed test without
a status undeterminable), and so is a check whose rule carries them. Wherever a rule is taken (`classify_log(rule=)`,
`SWEBenchAdapter(verdict=)`, `check(rule=)`, `conjunction_version(hardened_rule=)`), it is a rule name or a
`verdict_rule` mapping. Repository paths (`patch_exclude`, `filter_patch`'s `paths`) are a collection; a string is
refused. `conjunction_version(..., hardened_rule=...)` gives a hardened suite of listed tests its own rule; a hardened
row with a `reference_set` builds a reference-set check from its own `eval_script` (or `hardened_eval_script`). A hardened row's `patch_exclude` (a list of repository paths;
for a reference-set row, by default every path of its `test_patch`, `test_patch_paths`) removes the patch's per-file
diffs of those paths before the suite is applied: the paths are in the hardened check's execution identity
(`patch_exclude`, the digest of the sorted list), the re-execution request carries the filtered patch
(`adapters.swebench.filter_patch` / `exclude_paths`: a git patch splits at its `diff --git` headers, a plain diff at
`---`/`+++`/`@@` headers; a patch with no per-file diff is applied as it is; when nothing is left the patch is empty and
the tests run on the unmodified tree), and a producer runs the suite on the same filtered patch.

The AppWorld adapter's two checks of a task ask one request, which names the whole truth the evaluator runs with
(`execution`: evaluator, data version, state truth, expected answer) and carries the end state and the submitted
answer. The state check passes when every non-answer test passed and the evaluator's own verdict (`success`, or a
graded report's `other_tests_passed`) does not say otherwise. A report with an `error`, or without `success`, holds no
verdict: its tests are recorded as a failed state run.

## Arms (`seal.Arm`, `seal.arm_session`)

```python
session = arm_session("full_compatible", registry, scorer, cache=run_cache)   # reads the epoch pointer once
decision = session.finish(session.assemble(candidates, backfill=...))
```

| Arm | Session |
|---|---|
| `keep` | `mode="audit"`, policy `ecc`: decides what repair would (re-parsing, never executing), trains the recorded labels |
| `discard` | `mode="discard"`, policy `ecc`: a group with a member not graded under exactly the batch epoch's spec is neutralized |
| `ecc` | `mode="repair"`, policy `ecc` |
| `full_compatible` | `mode="repair"`; policy `full` once the batch epoch is not the startup epoch (`startup=`, the first grader publication by default), else `ecc`; needs `cache=` (or the scorer's), one dict for every seal of the run |
| `never` (also `pinned`) | `mode="repair"`, policy `ecc`, `pin="e0"` (or `startup=`'s seq): the batch epoch stays the startup epoch; needs a registry (given an `Epoch`, it seals under it as `ecc` does, without a pin record) |

`SealSession(registry, scorer, pin=...)` pins any seal: `pin="e0"` is the first grader publication, an int a grader
publication's `seq` (`seal.pinned_epoch`). The pointer is read once, as always; the pinned epoch comes from the
publication log and must have been published by that read.

## Prepared re-execution (`prepare.Preparer`)

```python
preparer = Preparer(registry, scorer, capacity=16, threads=True)   # the scorer the seal uses: adapters, backend, cache
preparer.admit(candidate)                       # where the coordinator admits a group into the buffer
...
session = SealSession(registry, scorer, batch_id=str(step), prepared=preparer)   # repair mode
assembly = session.assemble(selected, backfill=...)
decision = session.finish(assembly, journal=journal)   # the members it decided are never prepared again
preparer.retire(member_ids_of_groups_retired_unselected)   # e.g. past the wait bound: never prepared again
preparer.stop()
```

* `Preparer(epochs, scorer, *, capacity=16, slots=None, poll_s=1.0, wait_s=3600.0, startup=None, log=None,
  keep_events=True, planners=1, threads=False)`: `capacity` is how many executions the backend runs at once (the seal's
  own counted first), `slots` the jobs running at once (default `capacity` less an eighth), `wait_s` a seal's longest
  wait for a running job, `startup` the startup epoch's seq (default: the first grader publication), `log` a JSON-lines
  path for the events. `threads=True` (or `start()`, or `with Preparer(...)`) runs the pointer poll, planners and
  executors in threads; without them the caller drives it: `poll()` (read the pointer; a new target supersedes queued
  jobs of other identities and sweeps the admitted members), `plan()` (plan queued items), `run(limit=None)` (execute
  queued jobs).
* `admit(candidates)`, `retire(member_ids)`; a seal given `prepared=` calls `note_deciding` and `note_sealed` itself.
* `view(epoch, seal_id=, batch_id=)` is what a session gives its scorer (`Scorer(prepared=...)`): `claim(requests)`
  (finished results, running jobs to wait for, requests to execute now with why), `wait(job)`, `answered(...)`,
  `executing(requests)` / `executed(requests, results)` (the seal's own executions), `summary()`.
* `events` (and the log): `start`, `target`, `plan`, `job`, `use`, `deciding`, `sealed`, `retired`, `disabled`, each
  with its time `t`. `stats()` counts jobs by state, the seals' answers by how and why, and the background executions.
* `prepare.epoch_identity(epoch)` is `(seq, version_id)`: what binds a prepared result.

## Backends (`recompute`)

```python
class Backend(Protocol):
    def execute(self, requests: Sequence[ExecRequest], store: ContentStore) -> list[ExecResult]: ...
```

One result per request, in order; a failed execution is an `ExecResult` with `status="error"`, not an exception.

* `ExecRequest(family, kind, task_id, execution, inputs, params)`: `inputs` maps a role to an object digest in the
  store. `ExecResult(request_id, status, outputs, facts, cost, error, transport)`: `transport` is how a client obtained
  the result over a network (`attempts`, failed attempts by `reasons`, `gave_up`), recorded only when set and ignored
  by comparisons.
* `RecordedBackend(results)` replays recorded raw results by request id (`from_jsonl(path)`); a request without one is
  an infrastructure error (`not_recorded`). `RecordingBackend(inner, results_path)` records a live backend's results
  for later replay.
* `LocalDockerBackend(workers, timeout_s)` runs SWE-bench requests with the official `run_instance`
  (`agent-ecc[swebench]`); an empty patch (a patch filter removed every diff) runs the eval script on the unmodified
  tree (`run_instance(..., skip_patch=True)`; a `run_instance` without that option reports an infrastructure error). `AppWorldEvaluatorBackend(data_dirs, workers)` re-runs a data version's
  evaluator on recorded end states (`agent-ecc[appworld]`) and reports its `success` and `num_tests`. A data
  version must hold the truth the version's checks name.

### The reference re-verification service

```
python -m agent_ecc.recompute.service --backend {swebench-docker,appworld,recorded} [--port 8770] [--store DIR]
                                      [--workers N] [--backlog 1024] [--timeout S] [--appworld-data VERSION=DIR ...]
                                      [--recorded FILE]
```

The server (`make_server(service, host, port, backlog=1024)`) queues up to `backlog` connections it has not accepted
yet; socketserver's default of 5 drops or resets a burst of concurrent connections before any request is read. The
kernel may cap the queue (`net.core.somaxconn` on Linux).

* `GET /v1/health` -> `{"ok": true, "served": n, "backend": name, "workers": n}`
* `POST /v1/execute` with `{"requests": [ExecRequest...], "objects": {digest: base64}}` ->
  `{"results": [ExecResult...], "objects": {digest: base64}, "seconds": s}`

`HTTPBackend(url, attempts=3, backoff_s=1.0, max_backoff_s=30.0, deadline_s=None)` is the client: it uploads the
input objects requests name and verifies every returned object against its digest before storing it. A post that fails
in transport (a connection reset or refused, a timeout before a response, an HTTP 5xx without results) is sent again,
unchanged, after a full-jitter backoff (`backoff_s`, doubling, at most `max_backoff_s`), for at most `attempts` posts
within `deadline_s` (default `timeout_s`, which also bounds each attempt). An answer that holds results is never sent
again, whatever its verdicts or status; every post carries the same requests, so request ids and cache keys do not
change. Each result's `transport` records the attempts and the failures by reason (`reset`, `refused`, `timeout`,
`http_5xx`). A failure that outlasts the attempts or the deadline is an infrastructure error (`gave_up`), which the
scorer and a preparer do not execute again: the transport is the one place that retries it.

## Learner log (`export.LearnerLine`, `export.StepCompletion`)

One JSON object per consumed sample, written by the learner where its loss consumes the sample, every field read from
that sample's own row:

```json
{"batch_id": "step-42", "member_id": "u17/a0", "group_id": "u17", "step": 42, "reward": 1.0, "normalized": 0.866,
 "advantage": 0.866, "advantage_min": 0.866, "advantage_max": 0.866, "loss_tokens": 1834, "handoff_loss_tokens": 1834,
 "identity": "verified", "reporter": "dp3", "microbatch": 2, "step_part": 0}
```

* `reward` is the reward the sample was handed over with; `normalized` the learner's group-normalized reward;
  `advantage_*` the advantage over the sample's loss tokens; `identity` is `verified` when the tokens trained hash to
  the digest stamped at hand-off, `mismatch` or `unstamped` otherwise.
* `reporter` names the data-parallel shard, `<shard>:cp:<part>` for a context-parallel part (their loss tokens are
  summed). A micro-batch logged twice counts once (the last line per batch, step part, reporter, micro-batch and
  member is kept). A record consumed in more than one (step part, shard, micro-batch) is a duplicate consumption.
* A step completion is `{"batch_id": "step-42", "step": 42, "completed_at": 1790000190.2, "ok": true}`.

## Audits (`audit`)

* `audit_epochs(decisions, epochs, *, tolerance_s=1.0) -> EpochAudit`: per decision (in commit order) its batch epoch,
  what its pointer read named, the epoch in force at the read by the publication log, and a status: `ok`, `race` (the
  read missed only publications made within `tolerance_s` of it), `unexpected` (I4 fails: `problem` is
  `older_than_in_force`, `newer_than_in_force`, `no_epoch`, `not_a_grader_publication` or `not_the_pinned_epoch`) or
  `pinned`. `unexpected`, `decreasing` (I3), `pinned`, `races`, `ok`. Publication times and read times must come from
  comparable clocks.
* `verdict_conflicts(records, scores, versions, adapters, store) -> VerdictConflicts`: `groups` (request, grading
  identity) observed, `repeated` (more than one distinct output), `conflicts` (a pass and a fail among distinct
  outputs, with the task, the check, the runs by outcome and the records), `by_task`, `deterministic`.

## Stores

`ContentStore` is `put(bytes) -> digest`, `get(digest) -> bytes` (verified), `contains(digest)`. `DirectoryStore(root)`
keeps `<root>/sha256/<hex>`, written atomically. `LocatorStore({digest: path})` serves existing files read-only;
`ChainStore(a, b, ...)` reads through several stores and writes to the first.
