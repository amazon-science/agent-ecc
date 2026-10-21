# Integrating a trainer

This example shows where a fully asynchronous GRPO-style trainer calls Agent ECC. It assumes the usual shape: rollout
workers run episodes in sandboxes and grade them; a coordinator admits completed groups into a buffer; at each step
the trainer selects groups, hands them to the learner, and the learner's loss consumes them. Nothing here depends on a
particular trainer; the lifecycle example ([examples/lifecycle/run.py](../../examples/lifecycle/run.py)) runs the same
calls on synthetic data.

## 1. Publication (operator or controller)

```python
from agent_ecc import DirectoryRegistry

registry = DirectoryRegistry("/shared/run/ecc/registry")    # one per run, on a filesystem every component sees
registry.publish_grader(g0, source="startup")               # epoch first ...
registry.publish_world(world0, source="startup")            # ... then the world it covers
# later, a grader fix: published while training runs; the next seal reads it
registry.publish_grader(g1, source="fix")
```

## 2. Trajectory start (rollout worker)

Pin once, before the sandbox loads anything:

```python
pin = registry.resolve_start()        # pin.world: the world the sandbox serves; pin.epoch: the grader that grades it
```

## 3. Grading (rollout worker)

Record what the trajectory did and the verdict it got under its pinned epoch, and carry the record id on the sample:

```python
version = registry.grader(pin.epoch.version_id)
check = version.lookup(task, pin.world).spec
run = swe.run(check, store=store, log=test_log, applied=patch_applied)
record = swe.record(store=store, task_id=task, trajectory_id=attempt_id, image=pin.world, patch=patch, runs=[run])
recorded = grade_recorded(swe, record, check, store, grader_version=version.version_id)
dump_jsonl(records_path, [record]); dump_jsonl(scores_path, [recorded])
sample.member_id = attempt_id                 # the trainer's own sample, as it travels to the learner
sample.reward = 1.0 if recorded.passed else 0.0
```

A member's record is the record of the attempt that produced its sample. When a trajectory is run again (its
generation retried, or the trajectory replaced), keep one record per attempt and give the member the record of the
attempt whose output the sample holds: an earlier attempt's record, even one that was graded, never decides the
member, and a member whose producing attempt left no record has none (repair and discard neutralize its group).

When the pinned spec is a conjunction (an SWE-ABS epoch: the original suite AND a hardened suite), the producer runs
every suite and records one run per suite, the hardened suite on the patch without its diffs of the hardened test
patch's paths, with the record of each application:

```python
from agent_ecc.adapters.swebench import filter_patch
from agent_ecc.store import get_json, get_text
from agent_ecc.versions import iter_checks

spec = version.lookup(task, pin.world).spec                 # AllOf(original suite, hardened suite)
runs = []
for check in iter_checks(spec):                             # in spec order
    applied = patch
    if check.execution.get("patch_exclude"):                # the hardened suite: the patch filter first
        applied, removed = filter_patch(patch, get_json(store, check.execution["patch_exclude"]))
    out = executor.run(image=check.execution["image"], patch=applied,   # your sandbox; an empty patch runs the
                       eval_script=get_text(store, check.execution["eval_script"]))  # tests on the unmodified tree
    runs.append(swe.run(check, store=store, log=out.log, applied=out.applied, timed_out=out.timed_out,
                        error=out.error, application=out.application))
record = swe.record(store=store, task_id=task, trajectory_id=attempt_id, image=pin.world, patch=patch, runs=runs)
recorded = grade_recorded(swe, record, spec, store, grader_version=version.version_id)   # every suite that ran
```

`out.application` is the application record (`attempts`, `steps`, `reverse_check`, `executor_error`,
`infra_error`; see the interfaces), which tells a patch that genuinely does not apply (a fail) from an application
the executor stopped (no verdict); `out.error` is an infrastructure failure of the run itself. SWE-ABS rows keep
SWE-bench's FAIL_TO_PASS / PASS_TO_PASS lists, which name the original tests, while the hardened eval script runs the
new test files: graded by those labels, every hardened run fails, the gold patch's included. Build the hardened suite
from its reference set G instead (the tests passing in two gold runs, `adapters.swebench.reference_set`):
`conjunction_version(rows, [{"instance_id": ..., "eval_script": ..., "test_patch": ..., "reference_set": G}], ...)`
grades it under the `reference_set` rule and filters the patch by the test patch's paths.

Admission must not drop a group for having no reward variance under its *recorded* rewards: that decision belongs
after the seal (step 4), where the rewards are the batch epoch's. Keep every other admission check where it is.

Admission is also where re-executions can start early (prepared re-execution, optional): give the coordinator a
`Preparer` built with the scorer the seal uses, and note each admitted group. The preparer plans its members under
the current epoch (nothing under the startup epoch; every undecided admitted group again at each publication) and runs
what the seal will need in the background; it decides no label:

```python
from agent_ecc import Preparer

preparer = Preparer(registry, scorer, capacity=16, threads=True)   # once per run; capacity: the backend's sandboxes
...
buffer.add(group)
preparer.admit(candidate(group))                                     # where the coordinator admits the unit
```

## 4. Hand-off (trainer, before each learner step)

```python
from agent_ecc import Candidate, Member, Scorer, SealJournal, SealSession, apply_rewards
from agent_ecc.recompute.service import HTTPBackend

scorer = Scorer([swe], store, backend=HTTPBackend("http://reverify:8770"))   # the re-verification service
journal = SealJournal("/shared/run/ecc/seals.jsonl")

def candidate(group):                         # group: the trainer's unit of samples
    return Candidate(group_id=group.unit_id, handle=group, members=tuple(
        Member(member_id=s.member_id, execution=records[s.member_id], prior=scores[s.member_id],
               recorded_reward=s.reward) for s in group.samples))

session = SealSession(registry, scorer, batch_id=str(step),      # reads the epoch pointer once
                      prepared=preparer)                         # optional: prepared results for this epoch only
try:
    assembly = session.assemble(
        [candidate(g) for g in selected],
        backfill=lambda need, considered, remaining_s: [candidate(g) for g in buffer.take(need, exclude=considered,
                                                                                           wait_s=remaining_s)],
        deadline_s=60.0)
    decision = session.finish(assembly, journal=journal)
    batch = [c.handle for c, _ in assembly.batch]
    retire([c.handle for c, _ in assembly.excluded])           # no gradient under this epoch: do not decide again
    give_back([c.handle for c in assembly.unused])             # offered, never decided
except Exception as error:                                     # strict: an undecided batch trains nothing
    decision = session.fail(error, [candidate(g) for g in selected])
    batch = selected

samples = [s for g in batch for s in g.samples]
apply_rewards(decision, samples, member_id=lambda s: s.member_id,
              set_reward=lambda s, r: setattr(s, "reward", r))  # all or nothing
learner.train(samples)
```

Notes:

* **Serialize seals**: one sealed, uncommitted batch at a time. `SealJournal` refuses a decision whose epoch is older
  than the last committed one (invariant I3).
* **Off the event loop**: in an asynchronous trainer, run the session in a worker thread (`asyncio.to_thread`); a
  backfill callback can wait for newly admitted groups with `asyncio.run_coroutine_threadsafe(...).result(timeout)`.
* **Bound staleness at commit**: policy staleness is a separate bound, and it must be checked here, at hand-off, not
  when a group is admitted (groups can wait in the buffer).
* **Audit first**: `SealSession(..., mode="audit")` computes what a new epoch would change without training on it.
* **The batch may be short**: fillers are off by default; `decision.shortfall` says how many groups are missing.
  Pass `fillers="neutral"` to keep a fixed batch size with zero-advantage groups instead.
* **Prepared results**: with `prepared=`, the session's scorer takes, for each request it would execute, a result the
  preparer made for exactly that request under exactly this batch's epoch, if it gives the check a logical verdict;
  it waits for a job still running, takes over one not yet started, and executes everything else itself, at once.
  `decision.stats["prepared"]` says how each request was answered. Groups the trainer retires without a seal (past its
  wait bound, for instance) can be given to `preparer.retire(...)` so that they are not prepared again.
* **A pinned control**: `arm_session("never", registry, scorer, cache=...)` seals every batch under the startup epoch
  while later versions are published and graded; its decisions record the pin and what the pointer named.

## 5. Consumption (learner)

Where the loss consumes each micro-batch, write one line per sample, every field from that sample's own row (never
joined by position across lists), and one completion per finished step:

```python
line = {"batch_id": str(step), "member_id": row.member_id, "group_id": row.unit_id, "step": step,
        "reward": row.reward, "normalized": row.normalized_reward, "advantage": adv_mean, "advantage_min": adv_min,
        "advantage_max": adv_max, "loss_tokens": int(row.loss_mask.sum()), "handoff_loss_tokens": row.stamped_tokens,
        "identity": "verified" if token_digest(row.tokens) == row.stamped_digest else "mismatch",
        "reporter": f"dp{dp_rank}", "microbatch": microbatch_index}
learner_log.write(json.dumps(line) + "\n")
...
completions.write(json.dumps({"batch_id": str(step), "step": step, "completed_at": time.time(), "ok": True}) + "\n")
```

## 6. Afterwards

```python
from agent_ecc import consumption_records

report = consumption_records(journal.read(), learner_lines, completions)
assert report.manifest.ok            # every consumed record matched its seal decision
```

Each `ConsumptionRecord` names the step, the seal decision, the epoch, and per record the execution and score it
consumed, so any training step can be traced back to the exact verdicts it trained on, and every evaluation manifest
under the same epoch shares those verdicts.

Two audits read the same records:

```python
from agent_ecc import audit_epochs, verdict_conflicts

epochs = audit_epochs(journal.read(), registry)   # I4: every seal read the epoch in force; I3: never decreasing
assert epochs.ok, epochs.unexpected               # pinned seals are listed apart (epochs.pinned), not as violations
nondeterministic = verdict_conflicts(records, scores, registry_versions, [swe], store)
# requests whose identical executions got both verdicts under one grading identity: a test that follows an unpinned
# seed breaks the contract's deterministic-scoring precondition, which the seal does not check
```

## 7. Moving from a built-in implementation

A trainer that already seals batches with its own ECC code can check that agent-ecc decides exactly as it did before
switching, by replaying its logged seals:

1. **A reader** for the trainer's run directory splits every logged seal into what it was given and what it decided,
   and never reads the second as the first. Given: the groups offered in each decision round (member ids, records and
   their recorded verdicts, with the identities that graded them, and the reward each member's sample carried into
   the seal), the publication log with publication times, the time the seal read the epoch pointer, the run's
   configuration (mode, strictness, filtering), and the raw outputs of the re-executions the seal made (logs, run
   facts; never their verdicts). Decided: the epoch, every member's
   decision, label and trained reward, neutralization, the groups excluded, backfilled or used as fillers.
   Each member's record is the one the trainer's seal read: the reader finds it by the trainer's own rule, under the
   setting the run was made with (a trainer that keeps one record per attempt of a retried trajectory may read the
   record named by the member's id, or the latest attempt's), since a record found another way is another input.
   When the seal logs which record it read, the reader checks its own choice against that, member by member.
2. **The replay** builds the published versions with the adapter from the epoch bodies (and checks that every
   identity it derives is the one the body names), derives each seal's epoch from the publication log at the
   pointer-read time (`Registry.epoch_at`), serves the recorded raw outputs with `RecordedBackend` (a request the
   trainer never executed is an infrastructure error, never a verdict), and re-decides the seal with `SealSession`
   in the run's mode (`arm_session` for a comparison arm), the later rounds given back through `backfill`. When the
   trainer's seals share executions through a cache, the replay's scorer gets one cache for the run and each seal is
   also served the raw outputs of the run's earlier seals (its own recording first): a later seal may need what an
   earlier one executed.
3. **The comparison** is per seal and per member: the epoch, the decision type (reuse, evidence, reparse, reexec,
   pending, undeterminable), the label, the trained reward and neutralization, and per batch the groups handed over
   and excluded. Every mismatch is either explained by a check computed from the replay's own inputs (two executions
   of one request whose logs disagree, which agent-ecc executes once) or left unexplained. The standard: zero
   unexplained mismatches, every identity the replay derives equal to the one the epoch names, every record the seal
   logged it read the one the replay read, and no execution asked for that the trainer did not make in that seal (none
   the backend could not serve, none served only by an earlier seal's recording): a request the trainer never made is
   served as an infrastructure error, so its decision can agree while agent-ecc, run live, would have executed it.
4. **Sensitivity checks**: a log whose decisions are scrambled replays to the same decisions (the decisions are not
   inputs); a different verdict rule or a moved publication time is detected.

The frozen configuration must be the one the trainer ran: for SWE-bench, verdicts graded with the official harness's
end-to-end report on every log are reproduced with `SWEBenchAdapter(verdict="official")`, whose rule the recorded
verdicts carry in their grading identity; a trainer that keeps infrastructure errors out of verdicts and otherwise
takes the official report (and grades a hardened suite against a reference set) is reproduced with the `logical` and
`reference_set` rules, which each spec of the version names. A trainer that calls a member whose recorded verdict is
undeterminable as a whole (an infrastructure error at grading that no suite's logical fail outweighs) unrepairable,
without reusing, re-parsing or executing anything for it, is reproduced with
`Scorer(..., recorded_undeterminable="final")`: such a record is undeterminable under every version unless an
evidence-only verdict (an empty patch) fails it, so repair and discard neutralize its group and audit trains its
recorded reward. agent-ecc's default (`"resolve"`) decides it like any other record, re-executing what its artifact
allows, and a replay under the default asks for executions such a trainer never made. `tests/test_replay.py` shows
the replay on synthetic data with agent-ecc's own logs; a reader for a particular trainer's layout belongs with that
trainer. The deliberate differences such a comparison can surface are listed in the contract ([Decisions an
implementation could make otherwise](../contract.md#decisions-an-implementation-could-make-otherwise)).

The usual comparison arms are seal configurations (`seal.Arm`, `seal.arm_session(arm, registry, scorer, cache=...)`,
which reads the epoch pointer once): `keep` is audit (decide, never execute, train the recorded labels); `discard` is
validate-and-discard; `ecc` is repair under the `ecc` policy; `never` is repair with the batch epoch pinned to the
startup epoch; `full_compatible` re-verifies every member under every target spec once the batch epoch is not the
startup epoch (policy `full` from the switch on), sharing its executions through one cache for the run. A full re-grade that also evaluates the checks after a logical fail adds
`short_circuit=False`. `policy="reparse_all"` re-parses every recorded output whose execution identity is unchanged and
re-executes the rest; it is not a re-execution arm.

A seal that took prepared results replays like a synchronous one: its decisions depend only on the raw results of the
requests it asked for, so record the preparer's executions too (give the preparer a scorer whose backend is a
`RecordingBackend`, or record the backend both share) and serve them back with `RecordedBackend`; the replay then
needs no preparer. A trainer whose preparer executes a member's whole chain of suites in one request serves the same
raw outputs per suite.

## 8. What stays with the trainer

agent-ecc decides and records labels; scheduling stays in the trainer. A staged release (starting trajectories moved
onto a new version over time) is the resolver's choice of what `Registry.resolve_start` hands each trajectory; a
version-aware barrier (no new trajectory while old work drains, then the new version) is a gate on trajectory starts,
whose batches are sealed in discard mode; the bound on how long an admitted group may wait for its seal, and the policy
staleness bound, are checks at commit. None of them needs anything from agent-ecc beyond the registry, the seal modes
and the records.
