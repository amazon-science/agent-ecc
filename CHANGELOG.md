# Changelog

## 0.2.2 (2026-10-23)

* **Transport retries** in the reference service's client (`HTTPBackend`): a post that fails in transport (a
  connection reset or refused, a timeout before a response, an HTTP 5xx without results) is sent again, unchanged,
  after a full-jitter backoff (`backoff_s=1.0`, doubling, at most `max_backoff_s=30.0`), for at most `attempts=3` posts
  within one deadline per post (`deadline_s`, default `timeout_s`, which also bounds each attempt). An answer that
  holds results is never sent again, whatever its verdicts or its status. Every post carries the same requests, so the
  request ids, and the keys a scorer's cache files results under, are those of a single post. A failure that outlasts
  the attempts or the deadline is an infrastructure error, as before: the check is undeterminable and a repair seal
  neutralizes the member's group. Before, the client turned a transport failure into an infrastructure error at once,
  and only the scorer's `retries` executed the request again, without waiting, so retries met the same burst of
  connection resets (a server's small listen backlog) that caused the failure.
* The reference service listens with a **backlog of 1024** (`make_server(..., backlog=1024)`, `--backlog`):
  socketserver's default of 5 let a burst of concurrent connections overflow the queue, which drops or resets them
  before any request is read.
* Transport failures are retried in one place: the scorer, and a preparer's jobs, no longer execute again a request
  whose transport gave up (`retries` still applies to every other result without a logical verdict), so attempts do
  not multiply and one deadline bounds the request.
* `ExecResult.transport` records how a client obtained a result: `attempts`, failed attempts by `reasons` (`reset`,
  `refused`, `timeout`, `http_5xx`) and `gave_up`. A recording holds it only when it is set, so a result no transport
  carried keeps its 0.2.1 form, and comparisons ignore it. `Scored.transport` tallies the requests first charged to a
  record, and a seal whose executions went over a transport has `stats["transport"]`: `requests`, `attempts`,
  `retried`, `gave_up` and `reasons`, counted per request (a post that carries n requests counts n times).
* Tests: a raw-socket service that resets the first connections, one that always resets, answers with results (a pass
  and a fail, under status 200 and 503) that are never sent again, 4xx and 5xx answers, refused connections, timeouts
  and the deadline, the seal's statistics with a kept and a neutralized group, and a recording replayed without a
  second retry; a burst of 20 connections the service has not accepted yet, all served (and, on Linux, one that a
  backlog of 5 cannot hold).
* **Post hoc example: the trainer's evidence binding.** `examples/posthoc` already read a sidecar's evidence only
  when its outcome digest matched, but that digest binds the trajectory, the attempt, the status and the patch's digest
  and nothing else: a changed suite class, base commit, execution stamp, log reference or verdict passed it, and the
  record was scored, exported and compared as the sidecar said. A sidecar is now also unrepairable (no evidence read,
  an undeterminable recorded verdict) when its `evidence_binding`, the trainer's digest of the whole sidecar but its
  volatile fields (`events`, `timing`), is missing, of another version, or no longer matches. Like the outcome digest it
  covers every sidecar, whatever the seal later did with it; when both checks fail the outcome digest is the reason
  kept. `--no-evidence-binding` reads runs whose sidecars record none; `summary.records.evidence_binding` counts the
  sidecars it rejected. It proves a sidecar is unchanged since it was written, not that its grading was right.
* **Scorer: a prior score of another execution is refused on every path.** `Scorer.score` raised for a prior score
  whose `execution_id` is not the record's only for records the version governs; for a task it does not govern (whose
  recorded verdict is taken from the prior score), an unknown world, a torn observation or a family no adapter serves
  the check was skipped. It now runs first, for every item.
* Tests: seven corruptions of a stored sidecar (a suite class flipped, the execution stamps changed, a base-commit
  marker, the verdict changed, a log reference swapped, the binding removed, a binding of another version), each on a
  resolved and on a failed attempt; a sidecar failing both checks; volatile fields; the example's digest against a
  fixed digest computed by the trainer's function; and a foreign prior score on each path the scorer can take.

## 0.2.1 (2026-10-21)

* Example `examples/posthoc`: a sidecar's evidence is read only when its outcome digest (the trainer's capture of the
  trajectory, the attempt and the patch digest) matches the one recomputed from the sidecar; a missing or mismatched
  digest makes the record unrepairable (no evidence, undeterminable under every version; `records.outcome_digest` in
  the summary). `--no-outcome-digest` reads runs whose sidecars record none.
* Tests: the check on a small run directory, with a sidecar naming another record's patch and log.

## 0.2.0 (2026-10-21, first public release with 0.2.1)

* Seal: a **target pin**. `SealSession(pin="e0" | seq)` makes the batch epoch the pinned epoch, read from the
  publication log (`seal.pinned_epoch`; `"e0"` is the first grader publication), whatever the pointer names; the
  pointer is still read once, and the decision records the pin, whether it resolved and the epoch the pointer named
  (`SealDecision.target_pin`, `records.TargetPin`). A pin the log does not hold at the read leaves the batch without
  an epoch (repair neutralizes it). A pinned seal keeps I1 to I3 and relaxes I4's adoption on purpose. A decision
  without a pin has no `target_pin` key, so its identity is unchanged.
* Seal: the **never** arm (`Arm.NEVER`, also `"pinned"`): repair under the `ecc` policy with the batch epoch pinned to
  the startup epoch; a member graded under a later epoch trains the startup spec's verdict from its recorded parts.
* **Prepared re-execution** (`agent_ecc.prepare`): a `Preparer` follows the pointer and, under any epoch but the
  startup one, plans every admitted member as the seal will (the seal's scorer, reading the run's cache, executing
  nothing) and runs the requests the plan leaves in the background, one check at a time, keyed by (the epoch's
  publication and content, the request id); a publication supersedes queued jobs of other epochs and sweeps the
  undecided admitted members. `SealSession(prepared=...)` (repair mode) gives its scorer a view bound to the batch
  epoch (`Scorer(prepared=...)`): a request it would execute for the first time is answered by a finished job of that
  request under that epoch when the result gives the check a logical verdict, waits for a running job, takes over a
  queued one, and is executed at the seal otherwise, at once. Labels and decisions are those of the synchronous seal
  given the same raw results; preparation never writes the cache, never produces a score the seal reads, and never
  changes a record. `decision.stats["prepared"]` counts how each request was answered. `Scorer.read_result` is the one
  reading of a raw result the scorer and the preparer share.
* **Audits** (`agent_ecc.audit`): `audit_epochs` checks a seal journal against the publication log (I4: every seal's
  epoch is the one in force at its read, reads overlapping a publication reported as races, pinned seals checked
  against their pin; I3: epochs never decrease in commit order). `verdict_conflicts` grades every raw output the
  records hold under the given versions and reports identical requests whose outputs got both verdicts under one
  grading identity: the contract's deterministic-scoring precondition, which the seal does not check.
* Consumption: `zero_loss_tokens` (a sealed record consumed on no loss token: removed from the loss before or after the
  hand-off stamp; before it, the stamp and the learner agree on 0 and no other check sees it) and `epoch_mixed` (a
  record another seal handed over under another epoch); a `records_unsealed` entry names the seals that handed the
  record over (`sealed_in`). Consumption records hold the two new issue lists, so their identities differ from 0.1's
  for the same data.
* Example: `examples/prepared`, a grader switch sealed three ways (synchronous ECC, prepared, never) on synthetic data.
* Documentation: the contract (adoption is a choice and a pin relaxes it; prepared re-execution and why it decides no
  label; deterministic scoring and how to check it; what stays with the trainer), the interfaces, the trainer
  integration (preparation at admission, the never arm, the audits, replaying prepared seals, scheduling).
* Tests: the pin and the never arm; prepared re-execution, ported from a trainer's tests, with a property test that
  preparation never changes a label; the audits; ten kinds of seeded consumption faults, each reported in its category
  with no finding outside the records it touched.
* Requires nothing new: the core is still standard library only, on Python 3.10 or later.

## 0.1.0 (not released)

First version.

* Records: `ExecutionRecord`, `ScoreRecord`, `SealDecision`, `ConsumptionRecord`, frozen and content-addressed, with
  canonical JSON and verified round trips; JSON-lines helpers.
* Versions: checks with separate execution and grading identities, ordered conjunctions (`AllOf`), grader and world
  versions, a publication log that is the pointer (in memory or in a shared directory), the epoch-first rule, one
  pointer read per seal.
* Decisions: reuse / reparse / reexec / undeterminable per check under the `ecc`, `reparse_all` and `full`
  policies; logical verdicts only from complete outputs; infrastructure errors retried and never treated as verdicts;
  the conjunction short circuit on logical fails.
* Scoring: batched re-execution in waves, identical requests executed once, a result cache across calls, runs from
  earlier re-executions re-parsed by later versions, `grade_recorded` for producers.
* Seal: `SealSession` with repair, audit, discard and off modes; group neutralization; reward-dependent filtering after
  the decision; backfill under the same epoch through a caller callback; fillers off by default; strict failure
  handling; `SealJournal` (non-decreasing epochs in commit order); `apply_rewards`.
* Export: evaluation manifests (one version per manifest, undeterminable records counted apart), comparison and
  reproduction of manifests; consumption records and training manifests joined from a learner's per-record log and
  step completions.
* Evidence levels L0 to L2', dispositions per change type, sufficiency checks, masking for ablations.
* Backends: replay (`RecordedBackend`), recording, local Docker with the official SWE-bench harness, the HTTP reference
  re-verification service and its client.
* Adapters: SWE-bench (official harness, SWE-ABS conjunction versions, eval-script test-patch substitution) and
  AppWorld (answer re-grading, end-state re-grading with the AppWorld evaluator).
* Example: the record lifecycle on synthetic data. Documentation: the contract, the interfaces, trainer integration.
* Example: the record lifecycle post hoc on a finished trainer run (`examples/posthoc`): execution records, every
  published version, evaluation manifests by re-parsing stored logs at three evidence levels, the trainer's seal
  decisions as `SealDecision` records, the learner's log joined with them, reproduction of the first version's
  result. Its reader of the trainer's run layout lives in the example.
* SWE-bench adapter: `verdict="official"`, the official harness's own report as the verdict rule (every log it does not
  resolve fails), to reproduce verdicts recorded under it; the logical rule stays the default.
* Scorer: a verdict the record's evidence decides alone (an empty patch) is decided under every policy and at every
  allowed action, and is the recorded verdict, reused, when that verdict has the check's identities and agrees.
* Replay: a seal re-decided from its inputs alone (the rounds offered, the publication log and the pointer-read
  time, recorded raw re-execution outputs) gives the same decision; the method for checking agent-ecc against a
  built-in implementation is documented (docs/integrations/trainer.md, section 7). Trainer-specific log readers are
  not part of this package.
* Conjunctions are monotone three-valued ANDs: a logical fail decides whatever the other parts are (undeterminable
  ones included), an undeterminable part decides nothing and does not stop evaluation. The scorer resolves checks
  cheapest first (evidence-only and reused verdicts, then re-parses, then executions in spec order) and stops at a
  known logical fail; `Scorer(short_circuit=False)` evaluates every check. `grade_recorded` grades every suite whose
  run exists.
* Seal: a member is consistent only when its recorded verdict of exactly the batch epoch's spec is reused as a whole
  (`MemberDecision.consistent`); only then does it keep its recorded reward, and only consistent or ungoverned members
  keep a discard-mode group. Other reuses train their target verdict's reward (statuses `repaired_reuse_*`,
  `repaired_short_*`). `SealSession.finish` accepts only groups the session issued, as it issued them, with the
  session's own scores under its epoch.
* Evidence-only verdicts come before any reuse: a recorded verdict that disagrees with the evidence is not taken.
* The scorer shares and caches only results that gave their check a logical verdict; a result without one is
  executed again.
* Policy `full_compatible` is now `reparse_all`; the old name is refused with guidance. (The name `full_compatible`
  now belongs to a comparison arm, `Arm.FULL_COMPATIBLE`, below: repair under policy `full` after the switch.)
* SWE-bench: the verdict rule (`verdict`) is part of every check's grading identity, and a run is graded by its check's
  rule. Hardened suites can exclude paths from the patch under test (`patch_exclude`, in the execution identity;
  `exclude_paths`), and take their own rule.
* AppWorld: both checks of a task ask one request naming the whole truth (evaluator, data version, state truth,
  expected answer); an evaluator re-run is stamped with that truth and carries the submitted answer, so a later
  expected-answer change re-parses the answer instead of reading the old answer test; the state check follows the
  evaluator's own `success` (and a graded report's `other_tests_passed`); an errored report's partial tests are not a
  verdict.
* A publisher that crashed before its line's newline leaves a fragment the next publisher marks aborted: the
  publication log stays readable. A record consumed in two step parts is a duplicate consumption.
* SWE-bench's logical rule (the default) is the official report on a complete log: the status map of the official
  `get_logs_eval`, then `get_eval_tests_report` and `get_resolution_status`. A listed test without a status fails, a
  skipped PASS_TO_PASS test is maintained, a skipped FAIL_TO_PASS test fails, a log the official reading rejects fails.
  Infrastructure errors (never verdicts): a run without outputs; an application attempt that errored in the executor
  or exited 124 (`apply_infrastructure_error`, from the run's `application` record; a genuine apply failure fails);
  and, on the log, in this order (`parse_log`), the harness's timeout marker ending it, missing output markers, an end
  marker without the exit-code record after it, a parser exception. The options `missing_status` and `cross_check`
  are gone, and refused: nothing makes a listed test without a status undeterminable. The official rule is unchanged.
* SWE-bench `reference_set` rule: a hardened suite passes if and only if every test of its reference set G (the labels'
  PASS_TO_PASS, FAIL_TO_PASS empty) is PASSED or SKIPPED (XFAIL fails, OK is PASSED, exact ids, a test of G without a
  status fails). `reference_check`, hardened rows with a `reference_set` in `conjunction_version` (filtered by their
  test patch's paths, `test_patch_paths`), and `reference_set(runs)`: G from two gold runs (the tests passing in both;
  invalid on a disagreement, an invalid run or an empty G).
* SWE-bench patch filter: a plain diff splits only at `---`/`+++`/`@@` headers, a patch with no per-file diff is applied
  as it is, and when nothing is left the tests run on the unmodified tree (`filter_patch` records what was removed);
  the local Docker backend runs such a request with `run_instance(..., skip_patch=True)`.
* Scorer: under the `ecc` policy, an adapter's optional `carry_over(record, check)` decides a check by reuse from one of
  the record's own runs whose inputs differ from the check's. For SWE-bench, a patch that did not apply under the same
  image, application commands and patch filter fails under any eval script and any grading inputs.
* Seal: the arms of a grader switch, `Arm` and `arm_session` (`keep`, `discard`, `full_compatible`, `ecc`). After the
  switch, `full_compatible` re-verifies every member under every target spec (policy `full`, not the former policy of
  that name, which is `reparse_all`), sharing executions across seals through the scorer's cache, keyed by
  `ExecRequest.request_id`.
* Replay: seals of conjunction epochs and of every arm, including seals that share the run's cache, re-decide from
  their recorded raw outputs (`RecordedBackend`, one fresh cache for the run, each seal under the epoch in force at
  its pointer read) to the same decisions; how a trainer's own seal logs are checked against agent-ecc this way is in
  docs/integrations/trainer.md, section 7.
* Seal: a member the seal has no label for (no execution record, no epoch) trains its recorded reward outside repair,
  as every audit member does: under audit (the `keep` arm) the post-seal filter keeps a group whose recorded rewards
  vary instead of turning it into a filler. Repair and discard neutralize its group, as before.
* Local Docker backend: a re-execution's `timed_out` fact comes only from the harness's timeout marker at the end of
  the log (as `parse_log` reads it), never from the words anywhere in it; a complete log in which a test printed them
  is read for its verdict.
* Tests: under `full_compatible` an unrepairable member neutralizes its group; keep trains, and the other arms
  neutralize, a group with a member the seal has no record for; two `full_compatible` seals sharing the run's cache,
  recorded live (`RecordingBackend`) and replayed from the recordings (`RecordedBackend`), decide the same and ask for
  exactly the executions made, each once.
* Scorer: a verdict carried over by `Adapter.carry_over` names the recorded run it comes from (`PartVerdict.source`,
  then the part's `run`), so an audit finds the run that justified it; it stays a reuse (method `recorded`) and is not
  new evidence. The SWE-bench carry-over covers any check with the same image, application commands and patch filter:
  another eval script, or the same suite under other grading inputs (documented as such).
* SWE-bench adapter: `reference_check` requires `patch_exclude` (a direct caller can no longer skip SWE-ABS's patch
  filter by omission); repository paths given as a string are refused instead of being read character by character;
  a verdict rule is a name or a `verdict_rule` mapping wherever one is taken (`classify_log`, `SWEBenchAdapter`,
  `check`, `conjunction_version`).
* Documentation: a member's record is the record of the attempt that produced its sample; an earlier attempt of a
  trajectory run again (a retried generation, a replacement) never decides it. A replay reader finds each member's
  record by the trainer's own rule, under the setting the run was made with, and checks it against the record the
  seal logged it read (docs/integrations/trainer.md, sections 3 and 7).
* License: Apache-2.0 (`LICENSE`, `NOTICE`, `license = "Apache-2.0"` in the package metadata, and a copyright and
  SPDX header on every source file).
* Scorer: `recorded_undeterminable="final"`, a trainer-compatible setting for a record whose recorded verdict is
  undeterminable as a whole: every check its evidence does not decide alone is undeterminable (reason
  `recorded_undeterminable`) under every policy, before any reuse, re-parse or execution, so the record is
  undeterminable under every version unless an evidence-only verdict fails it (repair and discard neutralize its group,
  audit trains its recorded reward), as a trainer that calls such a member unrepairable decides it. The default,
  `"resolve"`, is unchanged: such a record is decided like any other (re-executed when its artifact allows).
