# Agent ECC

**Version-consistent experience consumption for agent training.** Validate recorded experience against a chosen
verifier version, re-evaluate it when needed, and track what each training step actually consumes.

Agent ECC is the open-source implementation of *Epoch-Consistent Commit*. In asynchronous agent training, trajectories
are generated, graded and buffered long before a learner step consumes them, while the things that decide their
labels keep changing: test labels are fixed, test suites are hardened, evaluators are patched, worlds are refreshed.
A label recorded under one grader and trained after another has been published is a label nobody chose. Agent ECC
makes the choice explicit: every training batch is sealed under **one** grader version, read once; every label the
batch trains on is that version's verdict on what the trajectory actually observed (reused when nothing it depends on
changed, re-parsed or re-executed when something did); a group whose label cannot be decided is neutralized rather
than guessed; and what each learner step actually consumed is established from the learner's own log.

## Layers

* **ECC is the contract and the mechanism**: content-addressed grader and world versions, a publication log that is
  the pointer, the seal that reads it once per batch, and the invariants I1 to I5 ([docs/contract.md](docs/contract.md)).
* **The SDK is how producers, graders and trainers use it**: record types, scoring under a version, evaluation and
  training manifests, the seal session (with re-executions prepared in the background from admission, and target
  pins for a control that never adopts), and audits of a finished run: the epochs its seals read and whether identical
  executions got different verdicts ([docs/interfaces.md](docs/interfaces.md)).
* **A reference re-verification service** does re-grading and re-execution behind a small HTTP API
  (`python -m agent_ecc.recompute.service`).
* **Adapters connect existing systems**: SWE-bench (with the official harness, including SWE-ABS hardened suites),
  AppWorld (answer re-grading and end-state re-grading with the AppWorld evaluator), and a trainer integration
  documented as an example ([docs/integrations/trainer.md](docs/integrations/trainer.md)).

It is a library and an SDK, not a platform: no scheduler, no database, no service you must run. The core package uses
the Python standard library only. Scheduling stays with the trainer (staged releases, barriers that pause starts,
bounds on how long a group may wait): agent-ecc gives it the registry, the seal and the records
([docs/integrations/trainer.md](docs/integrations/trainer.md), section 8).

## Four record types

| Record | What it is |
|---|---|
| `ExecutionRecord` | What one trajectory did: its evidence by digest (a patch, a raw test log, a submitted answer, an end state) and the test runs made when it was graded. Immutable and content-addressed. |
| `ScoreRecord` | One execution's verdict under one grader version: per-test outcomes, the method (`recorded`, `reparse`, `reexec`, `undeterminable`) and its cost. |
| `SealDecision` | What one seal intends to train: the epoch it read once, a decision per member, neutralized groups, groups filtered after the decision, backfill and fillers. |
| `ConsumptionRecord` | What one learner step actually consumed, confirmed by the learner's per-record log and the step's completion, with every mismatch against its seal decision (a record unsealed or unconsumed, consumed twice, in another group, with another reward or advantage, on other tokens or on none, or handed over under another epoch). A seal decision alone is not consumption. |

## Install

Once the package is on PyPI (until then, install a built wheel or a source checkout, below):

```bash
pip install agent-ecc                 # the core: standard library only (Python >= 3.10)
pip install "agent-ecc[swebench]"     # the official SWE-bench harness: log parsing, Docker re-execution
pip install "agent-ecc[appworld]"     # the AppWorld evaluator: end-state re-grading
pip install "agent-ecc[service]"      # the reference re-verification service (standard library server)
```

From a source checkout: `pip install -e ".[test]"` and `python -m pytest`. `python -m build` makes the sdist (with the
documentation, the examples and the tests) and the wheel.

## A tour

A sketch (dataset rows, logs, patches and groups come from your system; the runnable version is below):

```python
from agent_ecc import (MemoryRegistry, MemoryStore, Scorer, ScoreInput, SealSession, SealJournal,
                       evaluation_manifest, training_manifest)
from agent_ecc.adapters.swebench import SWEBenchAdapter
from agent_ecc.recompute.service import HTTPBackend
from agent_ecc.scoring import grade_recorded

store, registry = MemoryStore(), MemoryRegistry()
swe = SWEBenchAdapter()                                      # grades logs with the official harness

# grader versions are content-addressed maps (task, world) -> test spec
g1 = swe.version_from_rows(rows, store, name="verified")
g2 = swe.conjunction_version(rows, hardened_rows, store, name="verified+hardened")   # original AND hardened suite
registry.publish_grader(g1)

# collect: a producer records what a trajectory did, and the verdict it got under the epoch it pinned at start
# (under a conjunction such as g2, one run per suite: docs/integrations/trainer.md, section 3)
check = g1.entries[task]["*"]
run = swe.run(check, store=store, log=test_log, applied=True)
record = swe.record(store=store, task_id=task, trajectory_id=traj, image=image, patch=patch, runs=[run])
recorded = grade_recorded(swe, record, check, store, grader_version=g1.version_id)

# score under any version: reuse, re-parse or re-execute, whichever the identities require
scorer = Scorer([swe], store, backend=HTTPBackend("http://reverify:8770"))
scored = scorer.score([ScoreInput(record, recorded)], g2)

# evaluate: one manifest per version over the same record set, undeterminable records counted apart
manifest = evaluation_manifest(records, [s.score for s in scored], g2)

# train: a seal reads the epoch once, decides every member under it, filters after deciding, backfills
registry.publish_grader(g2)
session = SealSession(registry, scorer, batch_id="step-42")
assembly = session.assemble(groups, backfill=more_groups)    # more_groups(need, considered, remaining_s)
decision = session.finish(assembly, journal=SealJournal("seals.jsonl"))
rewards = decision.rewards()                                 # member id -> reward to train with

# afterwards: what the learner actually consumed, from its own per-record log and step completions
manifest = training_manifest(SealJournal("seals.jsonl").read(), learner_lines, step_completions)
```

The comparison arms of a grader switch are seal configurations: `arm_session(arm, registry, scorer, cache=...)` seals a
batch as `keep` (audit: decide, never execute, train the recorded labels), `discard`, `full_compatible` (after the
switch, re-verify every member under every target spec, sharing executions across seals), `ecc`, or `never` (the batch
epoch pinned to the startup epoch while later versions are published; the decision records the pin and what the
pointer named).

Re-executions can start before the seal. A `Preparer` follows the pointer and plans every admitted group under the
current epoch as the seal will, running what the plan leaves in the background; the seal still reads the pointer once
and decides every member itself, taking a prepared result only for the request it would execute, under its own epoch,
when the result gives the check a logical verdict. Preparation changes when executions run, never a label:

```python
from agent_ecc import Preparer, audit_epochs

preparer = Preparer(registry, scorer, threads=True)          # once per run, with the scorer the seal uses
preparer.admit(candidate)                                     # where groups enter the buffer
session = SealSession(registry, scorer, batch_id="step-43", prepared=preparer)
decision = session.finish(session.assemble(groups, backfill=more_groups), journal=SealJournal("seals.jsonl"))
decision.stats["prepared"]                                    # how each request was answered: prepared, or at the seal

assert audit_epochs(SealJournal("seals.jsonl").read(), registry).ok   # every seal read the epoch in force (I4, I3)
```

The runnable version of this tour, on synthetic data, is [examples/lifecycle](examples/lifecycle/run.py):

```bash
python examples/lifecycle/run.py --out /tmp/lifecycle
```

It grades one trajectory set under g1 and g2 (a label fix, an eval-script change, a hardened suite, unchanged tasks),
writes two evaluation manifests and their difference, re-scores g1 to show the old results are reproducible, then
seals three training steps under g2 and writes the training consumption manifest of the two steps the simulated learner
completed.

A grader switch sealed three ways (synchronous ECC, ECC with prepared re-execution, and `never`) on the same
synthetic data is [examples/prepared](examples/prepared/run.py): the prepared seal executes nothing and decides exactly
what the synchronous seal decides; the pinned one trains the passes the new version retracts:

```bash
python examples/prepared/run.py --out /tmp/prepared
```

The same lifecycle on real records, post hoc, is [examples/posthoc](examples/posthoc/run.py): it reads a finished run
directory of the trainer agent-ecc's SWE-bench integration was extracted from (its layout is in the script; the reader
lives in the example, not in the package), builds the execution records, every published grader version and both
evaluation manifests by re-parsing stored logs (nothing is executed; a check that needs an execution nobody recorded is
undeterminable), turns the trainer's logged seals into seal decisions linked to records and scores, joins the
learner's own per-record log, and reproduces the first version's result from the records as written. A sidecar
whose outcome digest is missing or does not match its trajectory and patch is unrepairable and its evidence is never
read (`--no-outcome-digest` for runs that recorded none):

```bash
python examples/posthoc/run.py --run RUN_DIR --out OUT [--executions LATER_REEXECUTIONS_DIR]
```

## Status

Version 0.2: the record types, the decision rules, the seal (with prepared re-execution and target pins), the
consumption join and the audits are complete and tested; interfaces may still change before 1.0. See
[CHANGELOG.md](CHANGELOG.md).

License: [Apache-2.0](LICENSE); see also [NOTICE](NOTICE).
