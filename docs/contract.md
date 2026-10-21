# The training-commit contract

This document states what Agent ECC guarantees, in terms of the objects it defines, and the rules that implement the
guarantee. Code references are to the `agent_ecc` package; every rule has tests under `tests/`.

## Terms

* **World version** `w`: an immutable, content-addressed version of everything a trajectory observes while it runs
  (databases, an environment image, tool implementations). A trajectory resolves its world version **once**, when it
  starts, and observes only that version (*pinning*).
* **Check**: an atomic test spec with two identities (`versions.Check`):
  * the **execution identity**: digests of every input that decides *what runs* (an image, an eval script,
    patch-application commands, an evaluator program);
  * the **grading identity**: digests of every input that decides *how the outputs are judged* (test labels, a log
    parser, grading code, an expected answer).
* **Test spec**: a check, or a conjunction `AllOf(spec, ...)` evaluated in order.
* **Grader version** (an **epoch**) `g`: an immutable, content-addressed map (task, world version) -> test spec
  (`versions.GraderVersion`). Its verdict on a record `τ` that observed world `w(τ)` is written `Γ_g(τ; w(τ))`. A task
  the version does not *govern* keeps its recorded verdict under it.
* **Execution record** `τ`: what one trajectory did, by evidence digest, and the test runs made when it was graded
  (`records.ExecutionRecord`). Immutable.
* **Score**: `τ`'s verdict under one grader version, with the method that produced it (`records.ScoreRecord`).
* **Batch** `b`: the groups one learner step trains on. **Seal**: deciding every label of `b`. **Commit**: the learner
  step that consumes `b`. A committed batch is never revised.

## The contract

> Every batch has one target epoch `g_b`, read once when the batch is sealed. Every label the batch trains on equals
> `Γ_{g_b}(τ; w(τ))`, the target epoch's verdict on the record evaluated against the world version the record observed.
> A record whose label cannot be computed under `g_b` neutralizes its group: every reward in the group is set to the
> same value, so the group contributes no policy gradient. A committed batch is never revised.

## Invariants

For every committed record `τ` in batch `b`:

| | Invariant | Enforced by | Checked by |
|---|---|---|---|
| **I1** | `τ` observed exactly one world version, the one its stamp names. | Pinning at the producer; a record whose observations name another version is *torn* (`ExecutionRecord.torn`) and has no verdict under any epoch. | `tests/test_properties.py`, `tests/test_scoring.py` |
| **I2** | If `τ` is trained, its reward is `Γ_{g_b}(τ; w(τ))`. | `SealSession.decide` scores every member under `g_b` only and trains the reward of that verdict (a recorded reward only for the recorded verdict of exactly `g_b`'s spec); undecidable members neutralize their group. | `tests/test_properties.py` (an oracle computes verdicts from the answers alone, and the trained rewards are checked against them) |
| **I3** | All of `b` shares `g_b`, and `g_b` is non-decreasing in commit order. | One session, one read; `SealSession.finish` accepts only groups the session decided, exactly as it handed them out, whose members' scores are the session's own under `g_b`; `SealJournal` refuses a decision whose epoch was published before the last committed one. | `tests/test_seal.py`, `tests/test_properties.py`; afterwards, `audit.audit_epochs` (`decreasing`) |
| **I4** | An epoch published before `b` is sealed is reflected in `g_b`, and no label comes from an epoch newer than `g_b`. | The seal reads the publication log's last grader line when it opens; publications during the seal (for example during backfill) do not reach the batch. A pinned seal relaxes the first half on purpose and records it (below). | `tests/test_properties.py` (`test_a_batch_never_mixes_epochs_under_racing_publications`); afterwards, `audit.audit_epochs` (every seal's epoch is the one in force at its read, `tests/test_audit.py`) |
| **I5** | (Retention) A group is neutralized only if some member cannot be computed under `g_b`, so no record is dropped merely because its world or grader version is old. | Re-parsing and re-execution recompute labels instead of discarding them; neutralization requires an undecidable member. | `tests/test_properties.py` |

I1 to I4 are safety; I5 is retention. The policy-staleness bound (how many policy versions old a trained sample may be)
is a separate constraint, also to be checked at commit, not at admission.

**Adoption is a choice.** I1 to I3 hold for any `g_b`, an old one included; I4 is what makes an operator's adoption of
a new grader version take effect at the next seal. A **target pin** (`SealSession(pin="e0")`, or a grader
publication's `seq`; the `never` arm) relaxes I4's adoption on purpose: the seal still reads the pointer once, then
takes the pinned epoch from the publication log (never from the pointer), and its decision records the pin, whether it
resolved, and the epoch the pointer named (`SealDecision.target_pin`). Later publications still happen and producers
still grade under them; a member graded under a later epoch is decided under the pinned spec (its recorded parts of
that spec reused, nothing executed). A pin the log does not hold at the read (or one published after it) leaves the
batch without an epoch: repair neutralizes it. `audit.audit_epochs` reports pinned seals as pinned, checking their
epoch against their pin and their pointer read against the log, never as I4 violations.

## Publication

* Grader versions and world versions are content-addressed: their identity is the digest of their canonical JSON
  (`canonical.canonical_json`: sorted keys, compact separators, ASCII, finite numbers only).
* A registry (`versions.Registry`) keeps one append-only **publication log**. The log *is* the pointer: the current
  grader version is the last grader publication. Publishing writes the version body first and appends the log line
  second, so any reader that sees a publication can load its body; a publisher that crashed mid-append left a fragment
  that is not a publication. Sequence numbers increase strictly across all publications.
* **Epoch first**: a world version can be published only when the current grader version covers it (has a spec for
  it, for every task it governs), so no trajectory can resolve a world that no published epoch can grade. A grader
  version that does not cover the current world (a fix that defines truth only for a newer world) needs
  `require_coverage=False`; records of the old world then have no verdict and their groups are neutralized.
* A producer pins at trajectory start (`Registry.resolve_start`): the world version it will observe, and the grader
  epoch current then, which grades the trajectory and whose identities its recorded verdict carries.
* A seal reads the pointer **once** (`Registry.read_epoch`); that read is the batch's linearization point. An audit or
  oracle that must know which epoch a past seal used derives it from the log and the read time
  (`Registry.epoch_at`), never from the step index.

## Decision rules

### Per check (`compare.plan_check`)

For each check of the target spec, given the record's runs and earlier verdicts:

| Situation | `ecc` (default) | `reparse_all` | `full` |
|---|---|---|---|
| A logical verdict with equal execution **and** grading identities | **reuse** | reparse | reexec |
| A run with an equal execution identity whose outputs are held | **reparse** | reparse | reexec |
| The execution identity differs, or no usable run | **reexec** | reexec | reexec |
| Re-execution needs an artifact the record does not hold | **undeterminable** | undeterminable | undeterminable |

* **reparse** judges recorded outputs again in-process (a SWE-bench log under new labels; an AppWorld answer under a
  new expected answer). **reexec** executes the target check's tests on the recorded artifact (a stored patch, a
  recorded end state); it never replays the episode.
* Before any of these, an adapter may decide a check from the evidence alone (an empty patch fails every suite). Such
  a verdict depends on no grader input and needs no execution, so it is decided under every policy and in every mode
  (discard included): it is the recorded verdict, reused, when that verdict was produced under the check's identities
  and agrees with the evidence, and an in-process decision (method `reparse`) otherwise. A recorded verdict that
  disagrees with the evidence is never taken.
* Under `ecc`, after equal-identity reuse, an adapter may carry a recorded run's verdict over to a check whose inputs
  differ from that run's when the inputs that differ cannot change it (`Adapter.carry_over`). For SWE-bench: whether a
  patch applies depends on the image, the application commands and the patch filter, not on the eval script or on how
  outputs are graded, so a recorded run whose patch did not apply fails any check with the same image, application
  commands and patch filter (another eval script, or other grading inputs such as a label fix), by reuse. The carried
  part has method `recorded` and the target check's identities, and its `run` names the recorded run it was carried
  over from. Never across another patch filter, never when the check's rule reads the recorded application as an
  infrastructure error, and never under `reparse_all` or `full`.
* Runs created by earlier re-executions are evidence too: a later version whose check has the same execution identity
  re-parses them instead of executing again. An adapter whose raw outputs depend on grading inputs (AppWorld's
  evaluator compares with the expected answer while it runs) stamps them on the run it makes of a re-execution, so a
  later check with other grading inputs does not read them as its own.
* Re-verification baselines are policies: `full` from a switch on is a re-verification of every member under every
  target spec (with the scorer's cache of logical results across seals, if wanted); a re-grade that also evaluates the
  checks after a logical fail turns the short circuit off (`Scorer(short_circuit=False)`).

### Logical verdicts and infrastructure errors

* A verdict is **logical** (a pass or a fail) unless the run hit an infrastructure error. For SWE-bench under the
  logical rule (`adapters.swebench.classify_log`, the default), every run that is not an infrastructure error is
  logical, and its verdict is the official report on the log: the status map of the official `get_logs_eval`, then
  `get_eval_tests_report` and `get_resolution_status`. A listed test without a status fails, a skipped PASS_TO_PASS
  test is maintained, a skipped FAIL_TO_PASS test fails, a complete log the official reading rejects fails, and so does
  a patch that genuinely does not apply. No option makes a listed test without a status undeterminable.
* A hardened suite is graded against a **reference set** G (`verdict="reference_set"`, G held as the labels'
  PASS_TO_PASS): it passes if and only if every test of G is PASSED or SKIPPED in the run (XFAIL is a failure, a
  parser's OK is PASSED, ids are compared exactly, a test of G without a status fails), on the patch without its
  per-file diffs of the hardened test patch's paths (nothing left: the tests run on the unmodified tree). G is the set
  of tests passing in two valid gold runs (`adapters.swebench.reference_set`); runs that disagree, or a run that is not
  valid, leave the gold without G.
* **Infrastructure errors are never verdicts**: an evaluator, container or sandbox failure (a run without outputs); a
  patch-application attempt that errored in the executor or exited with code 124; and, on the log, in this order, the
  harness's timeout marker ending it, no start marker or no end marker after it (an empty or truncated log), the end
  marker without the test exit-code record after it, and a parser exception. They are re-executed (up to `retries`,
  default 2, further executions) and otherwise leave the check **undeterminable**. A recorded run that hit an
  infrastructure error is re-executed as well (`recorded_infra="reexec"`), never read as a fail.
* One exception is explicit: SWE-bench's **official rule** (`verdict="official"`) reproduces the official harness's
  own end-to-end report on every log, under which every log it does not resolve is a fail, a timeout and a log without
  output markers included. It exists to reproduce verdicts recorded under that rule. The rule is part of every check's
  grading identity, so a verdict obtained under it is never reused, or re-read, under the logical rule: there the same
  run is judged again, and a timeout is no verdict.
* A verdict that executing again cannot change (the target's grader code is not the one available) is final:
  undeterminable without retries.

### Conjunctions and the short circuit (`compare.Conjunction`)

`AllOf(a, b, ...)` is a three-valued AND, which is **monotone**:

* a **logical fail** of any part fails the conjunction, whatever the other parts are, undeterminable ones included.
  For SWE-ABS (`e1 = original suite AND hardened suite against G`): a record whose original suite failed logically
  fails `e1` without running the hardened suite, and a record whose original run errored but whose hardened suite
  fails, fails;
* the conjunction **passes** only when every part passes;
* otherwise it is **undeterminable**: an infrastructure error never implies a fail, and never hides a later one.

The scorer resolves the checks cheapest first: everything decided without work (evidence-only verdicts, reused
verdicts, checks no action can decide), then every in-process re-parse, then executions one check at a time in spec
order. A logical fail stops everything more expensive than what found it (the **short circuit**, sound because the
conjunction is monotone): a fail known without work means nothing is re-parsed or executed, a fail found by a
re-parse means nothing is executed, an executed fail ends the executions. The checks never evaluated are listed in
the score's `skipped`. An undeterminable check stops nothing. `Scorer(short_circuit=False)` evaluates every check.

A producer grading with `scoring.grade_recorded` grades every suite whose run exists, even after a logical fail, so
its recorded verdict holds every suite that ran.

### Per record

* A record of a task the target epoch does not govern keeps its recorded verdict (*ungoverned*).
* A governed task without a spec for the record's world has no verdict (*unknown world*).
* A torn record has no verdict (I1).
* A member's trained reward is the reward of its target verdict (`binary_reward` by default). Only a **consistent**
  member keeps its recorded reward (which may be shaped): one graded under exactly the batch epoch's spec, whose
  recorded verdict is reused as a whole (every evaluated check from that one score, of the same spec, with the same
  outcome). A verdict reused from a later score (an evaluation, an earlier seal), assembled from several scores, or
  taken from a larger spec (a rollback) is the target verdict but not the recorded one: it trains its own reward.

### Decisions an implementation could make otherwise

These are deliberate, and a built-in implementation being replaced may differ on them:

* A recorded run that hit an infrastructure error says nothing about a target check with another execution identity:
  the record is re-executed when its artifact is held (with `recorded_infra="undeterminable"`, only a recorded error of
  the same execution identity is final). A record whose recorded verdict is undeterminable as a whole is decided like
  any other (an implementation may call it unrepairable without trying; `Scorer(recorded_undeterminable="final")`
  decides it that way: undeterminable under every version, only an evidence-only verdict taken).
* A patch that did not apply fails every suite with the same image, application commands and patch filter (any eval
  script and any grading inputs, by reuse under `ecc`); it needs no log to be re-parsed.
* A re-parse that gives no verdict (or raises) is escalated to re-execution when the artifact is held.
* Under SWE-bench's official rule a run that timed out fails, as the official report says.
* An ungoverned task keeps its recorded verdict only if that verdict is logical; otherwise it has no label.
* Identical re-execution requests of one batch are executed once, and their logical results are shared (and cached
  across calls with `cache`); results without a verdict are executed again. Requests are identified by
  `ExecRequest.request_id`, which names the patch as it is applied (after the check's patch filter), and the cache is
  consulted before every check that needs an execution. An implementation that keys executions by the raw patch, or
  reads its cache only up to the first miss in a member's chain of suites, executes some requests that agent-ecc
  answers from an earlier execution: with deterministic suites only the cost differs; with a nondeterministic suite the
  later member's label can differ. Prepared re-execution (below) is the same kind of sharing: a prepared result is one
  execution of the identical request, made earlier.
* Prepared jobs execute one check at a time, in spec order, and the next check of a conjunction is prepared only once
  the result before it gives a logical verdict; an implementation that executes a member's whole chain of suites in
  one request prepares the same checks, in one step.

### Deterministic scoring, and how to check it

The contract presumes that a check gives one verdict per execution: re-executing a stored artifact under the same
execution identity reproduces the verdict. **The seal does not check this.** A test whose outcome follows something
the execution identity does not name breaks it: an order that follows the test process's string-hash seed (a
de-duplicated list built from a set, under a runner that draws a fresh seed per process), the network, the clock.
Then a record with consistent provenance can still carry a label that a re-execution under the batch epoch
contradicts, and sharing one execution between identical requests (within a batch, through the cache, or through
preparation) fixes one draw of a coin.

* `audit.verdict_conflicts(records, scores, versions, adapters, store)` finds it after the fact: it grades every raw
  output the records hold (each record's own runs, and the runs its scores' re-executions made) under the versions'
  checks, and reports the requests whose distinct outputs got a pass and a fail under one grading identity.
* To keep it out of training, name what decides the outcome in the check's execution identity (an eval script that
  pins the seed), or leave such a task out of the version until it does; a verdict that depends on the hash seed or
  the network should not train.

## The seal (`seal.SealSession`)

1. **Read once.** The session reads the epoch pointer when it opens. Every member of every group it decides, in every
   round, is decided under that epoch.
2. **Decide.** Each member is scored under `g_b` by the rules above.
3. **Neutralize** (repair mode): a group with a member whose label is undeterminable has every reward set to the
   neutral value (0.0 by default). Every other member trains the reward of its target verdict (a consistent member:
   its recorded reward).
4. **Filter after deciding.** A reward-dependent filter (by default, the GRPO rule: the group's decided rewards must
   have a sample standard deviation above 1e-6) is applied to the decided rewards, never to recorded ones. A group
   that has no variance under the recorded grader but has variance under `g_b` is trained; a neutralized group has no
   variance and is excluded.
5. **Backfill.** Each excluded slot can be refilled through a caller callback with further groups, decided by the
   same session and so under the same `g_b`. A new round starts only within the backfill deadline.
6. **Fillers** are off by default: if backfill cannot fill the batch, it is delivered short and the shortfall is
   recorded. With `fillers="neutral"`, the oldest excluded groups fill the remaining slots, neutralized.
7. **Finish.** The session emits one `SealDecision`, built only from the groups it decided, as it handed them out
   (by `decide`, or by `assemble` with their roles and fillers). Appending it to a `SealJournal` enforces
   non-decreasing epochs in commit order (seals must be serialized: one sealed, uncommitted batch at a time).

Modes: `repair` enforces the contract; `audit` decides what repair would (re-parsing, never executing) and trains the
recorded labels, which makes it the canary for a new epoch; `discard` recomputes nothing and neutralizes every group
holding a member that is not consistent (graded under exactly `g_b`'s spec) or ungoverned; `off` decides nothing.

**Arms** (`seal.Arm`, `seal.arm_session`): the comparison arms of a grader switch are seal configurations, each
reading the epoch pointer once. `keep` is audit; `discard` is discard; `ecc` is repair under the `ecc` policy; `never`
(also `pinned`) is repair under the `ecc` policy with the batch epoch pinned to the startup epoch (I4's adoption
relaxed and recorded, above); `full_compatible` is repair that, once the batch epoch is not the startup epoch (the
first grader publication), re-verifies every member under every target spec under the `full` policy (recorded
verdicts not trusted, the short circuit kept). `full_compatible` shares its executions through the scorer's cache, one dict for every seal of the run,
keyed by the execution request (`ExecRequest.request_id`: the task, the check's execution identity, the input objects
(the patch as it is applied, after the check's patch filter, and the eval script) and the request's parameters
(instance fields, log parser, eval type, the patch filter)): every request a seal executed to a logical verdict
answers the identical request later, the cache being consulted before each check is executed, and identical requests
of a batch are executed once. Under `ecc` and `full_compatible` an undecidable member neutralizes its group.

Strict mode (default): a seal that cannot finish (its first decision raises, or its decision cannot be logged) is
delivered by `SealSession.fail`, which neutralizes every group; with no epoch published, discard mode neutralizes.
Non-strict mode trains the recorded labels in those cases.

## Prepared re-execution (`prepare.Preparer`)

The contract fixes where a label is validated (at the seal, against the batch epoch), not when the work behind it
runs. A repair seal that re-executes at the seal makes the learner wait for its slowest re-execution; a `Preparer`
starts that work when a group is admitted, and the seal only verifies.

1. **Target.** The preparer follows the pointer. Its target is the current epoch, identified by its publication
   (`seq`) and its content (`version_id`); a republication of the same content is another target. Nothing is
   prepared under the startup epoch.
2. **Planning.** Each admitted member (and, at a publication, every admitted member no seal has decided) is planned
   under the target exactly as the seal will plan it: the seal's scorer, its policy, retries and options, reading the
   run's cache, executing nothing. The first request the plan leaves becomes a background job, keyed by (the target's
   identity, `ExecRequest.request_id`), or joins an identical one; when a job's result gives its check a logical
   verdict, the member's next check is planned with it. A member a seal has decided, one being decided under another
   epoch, or one retired is not prepared; queued jobs whose members are all decided are dropped, and a publication
   supersedes every queued job of another identity.
3. **Execution.** Jobs run first in, first out, through the scorer's backend, with the scorer's retries; the seal's
   own executions are counted first against the backend's capacity.
4. **The seal.** It reads the pointer once and plans its members itself. For each request it would execute for the
   first time, it takes a prepared result only if the job's identity is the batch epoch's, its request id is the one
   the seal built from the member's own record (so the patch as applied, the execution identity and the parameters are
   the same), and the result gives the check a logical verdict. A job still running is waited for; a job not yet
   started is taken over (it never runs); anything else is executed at the seal, at once, as without preparation.
5. **Why no label changes.** The seal decides every member from raw results by its own rules, as it decides from its
   own executions: a prepared result is a raw result of the identical request, under the same epoch, judged by the
   seal's own reading. The preparer never writes the run's cache (the seal records a result it used), never produces a
   score the seal reads, never changes a record, and a result prepared under another epoch is never read. Labels,
   statuses, neutralizations, the post-seal filter and the decision are those of the synchronous seal given the same
   raw results (with deterministic checks, those of the synchronous seal), and I1 to I5 hold unchanged; only the
   seal's execution counts and its wall time differ. `tests/test_prepare.py` checks this on random worlds.

A preparer that fails disables itself and every request is executed at the seal; it never fails a seal.

## Consumption (`export.consumption_records`)

A learner step's consumption is established from the learner's side, never from the seal:

* the learner writes one line per consumed sample where the loss reads it (member id, group id, the reward it trained
  with, its group-normalized reward and advantage, its loss tokens, and whether its tokens are the ones handed over),
  and one completion event per finished step;
* a step is **consumed** only if it completed and its records were logged; a step with lines but no completion is
  *incomplete*, a completed step without lines is *unconfirmed* (a gap), a sealed batch with neither is *unconsumed*;
* per consumed step, the join with its seal decision checks that every sealed record was consumed and nothing else,
  once (by one shard, in one micro-batch of one step part; its context-parallel parts are one consumption), in its
  sealed group, with its sealed reward, on verified tokens, with the advantage its group's sealed rewards give, on the
  tokens it was handed over with, and on at least one token (`zero_loss_tokens`: a record removed from the loss before
  the hand-off stamp leaves a stamp and learner lines that agree on 0, which no other check compares with the seal);
* a consumed record the step's seal did not hand over is reported with the seals that did (`sealed_in`: a record
  delivered again), and as `epoch_mixed` when one of them sealed it under another epoch than the step's: the batch
  mixed epochs.

`tests/test_consumption_faults.py` seeds ten kinds of hand-off and learner faults (a stale label delivered or trained,
a record removed from the loss before or after the stamp, a dropped slot, a record consumed twice in a step or again in
a later one, a group sealed under another epoch, advantages from stale statistics, positional attribution, a record
normalized in another group) and checks that each is reported in its category, with nothing reported on records the
fault did not touch.

## Evidence

What a record keeps decides what a later grader version can do with it (`evidence`):

| Change | L0 reward | L0+ stamps | L1 I/O, answer | L2 state, patch + log | L2' snapshot |
|---|---|---|---|---|---|
| Does not touch the task or its tests | rerun | **keep** | keep | keep | keep |
| Expected answer (label-only truth) | rerun | rerun | **reparse** | reparse | reparse |
| World data, episode pinned | rerun | rerun | **reparse** | reparse | reparse |
| State-dependent test or evaluator code | rerun | rerun | replay | **reexec** (on the end state) | reexec |
| Test labels, log parser, grading rules | rerun | rerun | rerun | **reparse** (the log) | reparse |
| Eval script, image, patch application | rerun | rerun | rerun | **reexec** (the patch) | reexec |
| Tool implementation or backend outputs | rerun | rerun | **replay** | replay | replay |
| Tool interface, docs or prompt | neutralize | neutralize | neutralize | neutralize | neutralize |
| Truth defined only for a newer world | neutralize | neutralize | neutralize | neutralize | neutralize |

*rerun*: only a new episode gives a label, so in training the group is neutralized. *replay* (re-running recorded tool
calls against new tools) is outside the core decision rules in this version.

## Not covered by the contract

* Policy staleness: bound it at commit, separately.
* Changes the policy saw (tool interfaces, prompts): a recorded trajectory was generated for another task; its group is
  neutralized, not re-graded.
* Learned graders: expressible (the epoch is the grader's weights and prompt; re-execution is a scoring pass), but no
  adapter ships in this version. Their verdicts must be deterministic for a fixed version before the contract applies.
* Deterministic scoring is a precondition, not something the seal checks (see "Deterministic scoring, and how to check
  it" above).
* Scheduling is the trainer's: when trajectories start under which epoch (a staged release that moves starting
  trajectories onto a new version over time), whether starts pause while old work drains (a version-aware barrier),
  and how long an admitted group may wait before its seal (retirement). agent-ecc provides the pins a starting
  trajectory resolves (`Registry.resolve_start`) and the seal modes such policies use (a barrier's batches are sealed
  in discard mode), not the policies.
