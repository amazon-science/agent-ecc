# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The record lifecycle on a finished training run, post hoc: real records, every published grader version, the
seal decisions the trainer made and the learner's own log.

The synthetic example (``examples/lifecycle``) produces its records, seals and learner log itself. This one reads them
from the run directory of a trainer whose SWE-bench integration agent-ecc was extracted from (layout below), builds
everything with agent-ecc, and checks what the lifecycle promises:

1. **Records.** Every graded trajectory attempt the trainer's seal would read becomes an ``ExecutionRecord`` (its
   patch and the suite runs made when it was graded, by digest), and its recorded verdict a ``ScoreRecord``. Every
   evidence object is read once and verified against its digest. A sidecar whose outcome digest is missing or does
   not bind its trajectory to the patch its evidence names is unrepairable: its evidence is never read.
2. **Versions.** Every published grader epoch becomes a ``GraderVersion`` built by the SWE-bench adapter from the
   epoch's own objects; every identity agent-ecc derives is compared with the one the epoch body names. The
   publication log is replayed into a ``DirectoryRegistry`` at the logged times.
3. **Scores.** Every record is scored under every version by re-parsing its stored logs (policy ``reparse_all``:
   no recorded verdict is taken on trust), at three evidence levels: ``grading`` (the runs made at grading only),
   ``run`` (also the re-executions the run's seals recorded) and ``posthoc`` (also re-executions recorded after the
   run, ``--executions``). A check that needs an execution nobody recorded is undeterminable, never guessed; nothing
   is executed.
4. **Evaluation manifests**, one per version over the same record set, at the ``run`` level (the last version also at
   the other two), and the difference between the first and the last.
5. **Seals.** Each logged seal becomes a ``SealDecision``: the trainer's own decision for every member, its epoch as
   the replayed registry derives it from the pointer-read time, each member linked to its execution record and to its
   score under that epoch. The trainer's label is compared with the manifest's verdict, member by member.
6. **Consumption.** The learner's per-record log and step completions are joined with the seal decisions
   (``consumption_records``), and every consumed reward is compared with the verdict of its batch epoch.
7. **Reproduction.** The first version's manifest is scored again from the records as written to disk, under the
   version as loaded from the registry, and compared with the verdicts the run itself logged; the last version's
   outcomes are reproduced from the records and the stored re-execution runs without executing anything.

Run directory layout read (relative to ``--run``; the integration documented in docs/integrations/trainer.md):

    ecc/mode, ecc/publish.log, ecc/epochs/<n>.json   the arm, the publication log, the epoch bodies (conjunction
                                                     entries: a list of test specs per task, each with its labels,
                                                     eval script, verdict rule and patch filter, objects in the
                                                     epoch's store)
    ecc/seal.jsonl                                   one line per sealed batch: the epoch it read and its decisions
    ecc/recompute.jsonl, ecc/recompute-cache.jsonl   the seals' re-executions (suite, spec digest, patch digest, log)
    reval-meta/request_<w>_attempt_<a>[-a<k>].json   one sidecar per graded attempt (schema swe-bench-sidecar/2;
                                                     /1 where grading could not start)
    task-artifacts/<hh>/<hex>                        content-addressed evidence (patches, raw test logs)
    learner-records.jsonl                            the learner's per-record log (one line per consumed sample)
    audit-events/main.jsonl                          the step completions (train_group_step_end)
    knobs.json                                       which sidecar the seal read for a retried attempt

An ``--executions`` directory holds ``ecc/recompute.jsonl``, ``ecc/recompute-cache.jsonl`` and ``task-artifacts/``
in the same format (for example the output of a re-verification made after the run).

Writes under ``--out`` (which must be new or empty): records.jsonl, scores.jsonl, registry/, eval_<version>.json
(and .grading / .posthoc variants of the last version), eval_diff.json, seals.jsonl, learner.jsonl,
completions.jsonl, consumption.jsonl, training.json and summary.json. Nothing under ``--run`` is written.

The official SWE-bench harness must be importable (``pip install "agent-ecc[swebench]"``, or ``--swebench`` naming
its source tree).

Run:  python examples/posthoc/run.py --run RUN_DIR --out OUT [--executions DIR ...] [--name NAME] [--swebench PATH]
      [--no-outcome-digest] (for runs whose sidecars record no outcome digest)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import agent_ecc  # noqa: F401
except ImportError:  # running from a source checkout without installing
    sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from agent_ecc import (  # noqa: E402
    ANY_WORLD,
    AllOf,
    ChainStore,
    DirectoryRegistry,
    DirectoryStore,
    EvaluationManifest,
    Exclusion,
    ExecutionRecord,
    GraderVersion,
    GroupDecision,
    LearnerLine,
    MemberDecision,
    MemoryStore,
    Method,
    Outcome,
    PartScore,
    ScoreInput,
    ScoreRecord,
    Scorer,
    SealDecision,
    StepCompletion,
    TestRun,
    compare_evaluations,
    consumption_records,
    dump_jsonl,
    evaluation_manifest,
    load_jsonl,
    reproduce_evaluation,
)
from agent_ecc.canonical import FrozenDict, digest_bytes, digest_json, thaw  # noqa: E402
from agent_ecc.records import RUN_ERROR, RUN_OK, Cost, EvidenceRef  # noqa: E402
from agent_ecc.recompute.base import ExecResult, RecordedBackend  # noqa: E402
from agent_ecc.versions import iter_checks, spec_digest  # noqa: E402

CONJUNCTION_SCHEMA = "swe-bench-sidecar/2"
SINGLE_SUITE_SCHEMA = "swe-bench-sidecar/1"
KNOB_LATEST_ATTEMPT = "the latest backend attempt's (a backend retry's included)"
BACKEND_ATTEMPTS = 16  # the backend attempts the trainer's sidecar lookup scans
RULES = {"official": "logical", "reference_set": "reference_set"}  # a spec's verdict rule -> agent-ecc's rule
SHARED_FIELDS = ("image", "apply_cmds", "grading_code", "parser", "log_parser", "eval_type")
IDENTITY_KEYS = ("image", "eval_script", "apply_cmds", "patch_exclude", "labels", "parser", "grading_code")
LEVELS = ("grading", "run", "posthoc")
NO_PATCH = "no_agent_patch"
SIDECAR_RE = re.compile(r"^request_(?P<work>.+?)_attempt_(?P<attempt>\d+)(?:-a(?P<k>\d+))?(?P<start>\.start)?\.json$")
# the trainer's member status -> (agent-ecc action, reason, short circuit, consistent)
STATUS = {
    "consistent": ("reuse", None, False, True),
    "repaired_reuse_same": ("reuse", None, False, False),
    "repaired_reuse_flip": ("reuse", None, False, False),
    "repaired_short_same": ("reuse", None, True, False),
    "repaired_short_flip": ("reuse", None, True, False),
    "repaired_same": ("reparse", None, False, False),
    "repaired_flip": ("reparse", None, False, False),
    "repaired_exec_same": ("reexec", None, False, False),
    "repaired_exec_flip": ("reexec", None, False, False),
    "no_patch": ("reparse", NO_PATCH, False, False),
    "ungoverned": ("ungoverned", "ungoverned", False, False),
    "needs_reparse": ("pending", "needs_reparse", False, False),
    "needs_exec": ("pending", "needs_reexec", False, False),
    "needs_exec_audit": ("pending", "needs_reexec", False, False),
    "unrepairable": ("undeterminable", "unrepairable", False, False),
    "unknown_world": ("undeterminable", "unknown_world", False, False),
    "no_meta": ("undeterminable", "no_record", False, False),
    "no_epoch": ("undeterminable", "no_epoch", False, False),
    "torn_live": ("undeterminable", "torn_observation", False, False),
    "off": ("off", None, False, False),
}
SEAL_MODES = {"repair": "repair", "regrade": "repair", "fullcompat": "repair", "audit": "audit", "discard": "discard"}


# ---------------------------------------------------------------------------------------------------------- reading


def read_json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_bytes(paths: list[str], threads: int) -> dict[str, bytes | None]:
    """Read many small files at once (on a parallel file system a cold small-file read is slow; reads in parallel)."""

    def one(path):
        try:
            with open(path, "rb") as f:
                return path, f.read()
        except OSError:
            return path, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        return dict(pool.map(one, sorted(set(paths))))


def artifact_path(root: str, digest: str) -> str:
    hexdigest = str(digest).partition(":")[2]
    return os.path.join(root, "task-artifacts", hexdigest[:2], hexdigest)


def sidecar_choice(names: list[str], lookup: str) -> tuple[dict[str, str | None], int]:
    """record id (``<work>/a<attempt>``) -> the sidecar the trainer's seal reads for it (None when the attempt it reads
    left none), and how many other sidecars (superseded backend attempts) the directory holds. ``latest_attempt``:
    the highest backend attempt below 16, gaps included, that left a sidecar or a start record; ``record``: the
    attempt's own (backend attempt 0)."""
    seen: dict[str, dict[int, set]] = defaultdict(lambda: defaultdict(set))
    for name in names:
        m = SIDECAR_RE.match(name)
        if m is None:
            continue
        rid = f"{m.group('work')}/a{m.group('attempt')}"
        seen[rid][int(m.group("k") or 0)].add("start" if m.group("start") else "sidecar")
    chosen, superseded = {}, 0
    for rid, attempts in seen.items():
        work, _, attempt = rid.partition("/a")
        stem = f"request_{work}_attempt_{attempt}"
        if lookup == "latest_attempt":
            k = max(k for k in attempts if k < BACKEND_ATTEMPTS)
        else:
            k = 0
        name = stem if k == 0 else f"{stem}-a{k}"
        chosen[rid] = f"{name}.json" if "sidecar" in attempts.get(k, set()) else None
        superseded += sum(1 for j, kinds in attempts.items() if j != k and "sidecar" in kinds)
    return chosen, superseded


def member_sidecar_lookup(knobs) -> str:
    if not isinstance(knobs, dict):
        return "record"
    value = (knobs.get("seal") or {}).get("member_sidecar")
    if value is None:
        return "record"
    if value == KNOB_LATEST_ATTEMPT:
        return "latest_attempt"
    raise ValueError(f"knobs.json names a member sidecar lookup this example does not know: {value!r}")


def utc_seconds(stamp: str) -> float:
    return datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


# ---------------------------------------------------------------------------------------------------------- versions


def version_from_body(adapter, store, body: dict, task_meta: dict, issues: list) -> GraderVersion:
    """A published epoch body as a GraderVersion: each task's conjunction entry as AllOf of its specs in order, every
    identity agent-ecc derives compared with the one the body names."""
    if body.get("kind") != "swe_bench":
        raise ValueError(f"epoch {body.get('epoch')} is not a SWE-bench epoch")
    from agent_ecc.adapters.swebench import verdict_rule

    entries = {}
    for task, entry in sorted((body.get("swe") or {}).items()):
        if not isinstance(entry.get("specs"), list) or not entry["specs"]:
            raise ValueError(f"epoch {body.get('epoch')}: {task} is not a conjunction entry (this example reads "
                             f"conjunction epochs only)")
        meta = task_meta.get(task) or {}
        repo = meta.get("repo") or task.partition("__")[0] + "/" + task.partition("__")[2].rsplit("-", 1)[0]
        checks = []
        for i, raw in enumerate(entry["specs"]):
            spec = {**{k: entry.get(k) for k in SHARED_FIELDS}, **raw}
            rule = RULES.get(str(spec.get("verdict_rule") or "official"))
            if rule is None:
                raise ValueError(f"{task}: unknown verdict rule {spec.get('verdict_rule')!r}")
            labels = json.loads(store.get(str(spec["labels"])))
            script = store.get(str(spec["eval_script"])).decode("utf-8")
            paths = json.loads(store.get(str(spec["patch_exclude"]))) if spec.get("patch_exclude") else ()
            check = adapter.check(store=store, instance_id=task, repo=repo, version=str(meta.get("version") or ""),
                                  image=str(spec["image"]), eval_script=script, log_parser=str(spec["log_parser"]),
                                  eval_type=str(spec["eval_type"]), fail_to_pass=labels["FAIL_TO_PASS"],
                                  pass_to_pass=labels["PASS_TO_PASS"], name=str(spec.get("name") or f"s{i}"),
                                  rule=verdict_rule(rule), patch_exclude=paths)
            got = {**{k: check.execution.get(k) for k in ("image", "eval_script", "apply_cmds", "patch_exclude")},
                   **{k: check.grading.get(k) for k in ("labels", "parser", "grading_code")}}
            for k in IDENTITY_KEYS:
                if got.get(k) != (spec.get(k) or None):
                    issues.append({"epoch": body.get("epoch"), "task": task, "spec": spec.get("name"), "key": k})
            checks.append(check)
        entries[task] = {ANY_WORLD: AllOf(name="all_of", parts=tuple(checks))}
    return GraderVersion(name=str(body.get("name")), entries=entries, family="swebench",
                         meta={"content_digest": body.get("content_digest")})


# ---------------------------------------------------------------------------------------------------------- records


def record_from_single_suite(record_id: str, sc: dict) -> tuple[ExecutionRecord, ScoreRecord]:
    """A schema /1 sidecar (one suite). In a conjunction run the producer writes one only when grading could not
    start, e.g. when the agent's patch could not be captured: a record without the evidence a verdict needs, its
    recorded verdict undeterminable."""
    stamps, task = sc.get("stamps") or {}, sc.get("task") or {}
    ev, v = sc.get("evidence") or {}, sc.get("verdict") or {}
    execution = {"image": stamps.get("image"), "eval_script": stamps.get("eval_script"),
                 "apply_cmds": stamps.get("apply_cmds")}
    grading = {"labels": stamps.get("labels"), "parser": stamps.get("parser"),
               "grading_code": stamps.get("grading_code"), "eval_type": task.get("eval_type"),
               "verdict_rule": {"rule": "official"}}
    artifacts, facts, runs = [], {}, []
    patch = ev.get("agent_patch")
    if patch:
        artifacts.append(EvidenceRef(role="patch", digest=patch["digest"], size=patch.get("size"),
                                     media_type="text/x-diff"))
    if v.get("reason") == NO_PATCH:
        facts["patch_empty"] = True
    elif v.get("valid") and ev.get("test_log"):
        log = ev["test_log"]
        runs.append(TestRun(check="tests", execution=execution, status=RUN_OK,
                            outputs=(EvidenceRef(role="test_log", digest=log["digest"], size=log.get("size"),
                                                 media_type="text/plain"),),
                            facts={"applied": bool((ev.get("patch_application") or {}).get("applied", True)),
                                   "timed_out": False}))
    record = ExecutionRecord(family="swebench", task_id=str(sc.get("task_id")), trajectory_id=record_id,
                             world_version=None if stamps.get("image") is None else str(stamps["image"]),
                             artifacts=tuple(artifacts), runs=tuple(runs), facts=facts)
    if not v.get("valid"):
        outcome, why = Outcome.UNDETERMINABLE, str(v.get("reason") or "evaluator_failed")
    else:
        outcome, why = (Outcome.PASS, "resolved") if v.get("resolved") else (Outcome.FAIL, v.get("reason"))
    method = Method.RECORDED if outcome.logical else Method.UNDETERMINABLE
    part = PartScore(check="tests", execution_digest=digest_json(execution), grading_digest=digest_json(grading),
                     outcome=outcome, method=method, reason=why)
    return record, ScoreRecord(execution_id=record.record_id, grader_version=None, outcome=outcome, method=method,
                               reason=why, parts=(part,), meta={"source": "sidecar"})


def outcome_digest(sc: dict) -> str:
    """The outcome digest the trainer's capture records, recomputed from the sidecar's identity fields and the digest
    of the patch its evidence names (the attempt id is the trajectory id)."""
    patch = ((sc.get("evidence") or {}).get("agent_patch") or {}).get("digest")
    value = {"task_id": sc.get("task_id"), "trajectory_id": sc.get("trajectory_id"),
             "attempt_id": sc.get("trajectory_id"), "status": sc.get("execution_status"),
             "artifacts": {"agent_patch": patch} if patch else {}}
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def outcome_digest_check(sc: dict | None) -> str:
    """"ok", "missing" (no digest recorded) or "mismatch": the sidecar's ``outcome_digest`` vs :func:`outcome_digest`.
    Only on "ok" is the sidecar's evidence read; otherwise the record is unrepairable (fails closed)."""
    if not (sc or {}).get("outcome_digest"):
        return "missing"
    return "ok" if outcome_digest(sc) == sc["outcome_digest"] else "mismatch"


def unrepairable_record(record_id: str, sc: dict, check: str) -> tuple[ExecutionRecord, ScoreRecord]:
    """A sidecar that failed the outcome-digest check: a record without its evidence (no patch, no runs, so nothing is
    re-parsed or re-executed for it) and an undeterminable recorded verdict."""
    stamps = sc.get("stamps") or {}
    record = ExecutionRecord(family="swebench", task_id=str(sc.get("task_id")), trajectory_id=record_id,
                             world_version=None if stamps.get("image") is None else str(stamps["image"]),
                             facts={"outcome_digest": check})
    return record, ScoreRecord(execution_id=record.record_id, grader_version=None, outcome=Outcome.UNDETERMINABLE,
                               method=Method.UNDETERMINABLE, reason=f"unrepairable: outcome digest {check}",
                               meta={"source": "sidecar"})


def record_from_sidecar(record_id: str, sc: dict, version_of_epoch: dict) -> tuple[ExecutionRecord, ScoreRecord]:
    """A schema /2 sidecar as an execution record (the patch; one run per suite that ran, with its final raw log and
    patch application record; a suite whose evaluator failed is a run with status error) and its recorded verdict
    (one part per suite, graded by the identities its stamps name)."""
    if sc.get("schema") == SINGLE_SUITE_SCHEMA:
        return record_from_single_suite(record_id, sc)
    stamps, ev, v = sc.get("stamps") or {}, sc.get("evidence") or {}, sc.get("verdict") or {}
    by_name = {st.get("name"): st for st in stamps.get("specs") or []}
    artifacts, facts, runs, parts = [], {}, [], []
    patch = ev.get("agent_patch")
    if patch:
        artifacts.append(EvidenceRef(role="patch", digest=patch["digest"], size=patch.get("size"),
                                     media_type="text/x-diff"))
    if v.get("reason") == NO_PATCH:
        facts["patch_empty"] = True
    for suite in ev.get("suites") or []:
        st = by_name.get(suite.get("name")) or {}
        execution = {"image": st.get("image"), "eval_script": st.get("eval_script"), "apply_cmds": st.get("apply_cmds")}
        if st.get("patch_exclude"):
            execution["patch_exclude"] = st["patch_exclude"]
        rule = RULES.get(str(st.get("verdict_rule") or "official"), str(st.get("verdict_rule")))
        grading = {"labels": st.get("labels"), "parser": st.get("parser"), "grading_code": st.get("grading_code"),
                   "eval_type": st.get("eval_type"), "verdict_rule": {"rule": rule}}
        cls = suite.get("class")
        outcome = Outcome(cls) if cls in ("pass", "fail", "undeterminable") else Outcome.UNDETERMINABLE
        parts.append(PartScore(check=str(suite.get("name")), execution_digest=digest_json(execution),
                               grading_digest=digest_json(grading), outcome=outcome,
                               method=Method.RECORDED if outcome.logical else Method.UNDETERMINABLE,
                               reason=suite.get("reason")))
        if facts.get("patch_empty"):
            continue  # nothing ran: the empty patch decides every suite
        attempts = suite.get("attempts") or []
        final = attempts[-1] if attempts else {}
        log = suite.get("test_log")
        application = suite.get("patch_application")
        run_facts = {"applied": bool(application.get("applied", True)) if application else log is not None,
                     "timed_out": False, "has_output_markers": suite.get("has_output_markers"),
                     "test_exit_code": suite.get("test_exit_code")}
        if application:
            run_facts["application"] = application
        if log:
            runs.append(TestRun(check=str(suite.get("name")), execution=execution, status=RUN_OK,
                                outputs=(EvidenceRef(role="test_log", digest=log["digest"], size=log.get("size"),
                                                     media_type="text/plain"),), facts=run_facts))
        elif application and not application.get("applied", True):
            runs.append(TestRun(check=str(suite.get("name")), execution=execution, status=RUN_OK, facts=run_facts))
        elif suite.get("executions"):
            runs.append(TestRun(check=str(suite.get("name")), execution=execution, status=RUN_ERROR,
                                facts={"error": str(final.get("error") or suite.get("reason") or "evaluator_failed")}))
    record = ExecutionRecord(family="swebench", task_id=str(sc.get("task_id")), trajectory_id=record_id,
                             world_version=None if stamps.get("image") is None else str(stamps["image"]),
                             artifacts=tuple(artifacts), runs=tuple(runs), facts=facts)
    classes = [p.outcome for p in parts]
    if any(o is Outcome.FAIL for o in classes):
        outcome = Outcome.FAIL
    elif classes and all(o is Outcome.PASS for o in classes):
        outcome = Outcome.PASS
    else:
        outcome = Outcome.UNDETERMINABLE
    graded = sc.get("graded_epoch") or {}
    version = version_of_epoch.get(graded.get("epoch")) if graded.get("covered") else None
    recorded = ScoreRecord(execution_id=record.record_id, grader_version=version.version_id if version else None,
                           outcome=outcome, method=Method.RECORDED if outcome.logical else Method.UNDETERMINABLE,
                           reason=v.get("reason") if outcome is not Outcome.PASS else "resolved", parts=tuple(parts),
                           meta={"source": "sidecar"})
    return record, recorded


def logged_suite_class(sc: dict, name: str) -> str | None:
    """The class the run itself logged for one suite when it graded the attempt (an empty patch fails every suite)."""
    verdict = sc.get("verdict") or {}
    if verdict.get("reason") == NO_PATCH:
        return "fail"
    for suite in (sc.get("evidence") or {}).get("suites") or []:
        if suite.get("name") == name:
            return suite.get("class")
    if sc.get("schema") == SINGLE_SUITE_SCHEMA and not verdict.get("valid"):
        return "undeterminable"  # grading never started: the run logged no verdict
    return None


# ---------------------------------------------------------------------------------------------------------- executions


def execution_index(sources: list[tuple[str, str]], spec_names: dict) -> tuple[dict, dict]:
    """Recorded re-executions, by (task, suite name, agent patch digest): [(source, entry), ...] in source order.
    ``sources`` is [(source label, directory)]. A spec digest is mapped to its suite name by what the run's own
    records and the directories' execution logs name; an entry whose spec maps to no name, or to two, is not bound."""
    for _, root in sources:
        for e in read_jsonl(os.path.join(root, "ecc", "recompute.jsonl")):
            if e.get("kind") == "execution" and e.get("suite") and e.get("spec"):
                spec_names.setdefault((e["task"], e["spec"]), set()).add(str(e["suite"]))
    index: dict[tuple, list] = defaultdict(list)
    notes = Counter()
    for label, root in sources:
        for e in read_jsonl(os.path.join(root, "ecc", "recompute-cache.jsonl")):
            names = spec_names.get((e.get("task"), e.get("spec"))) or set()
            if len(names) != 1 or not e.get("patch"):
                notes[f"{label}:unbound"] += 1
                continue
            index[(e["task"], next(iter(names)), e["patch"])].append((label, root, e))
            notes[f"{label}:entries"] += 1
    return index, dict(notes)


def exec_result(request_id: str, root: str, entry: dict) -> ExecResult:
    log = entry.get("log")
    cost = Cost(executions=1)
    if log and log.get("digest"):
        ref = EvidenceRef(role="test_log", digest=log["digest"], size=log.get("size"), media_type="text/plain")
        return ExecResult(request_id=request_id, status=RUN_OK, outputs=(ref,),
                          facts={"applied": True, "timed_out": False}, cost=cost)
    if entry.get("reason") == "patch_did_not_apply":
        return ExecResult(request_id=request_id, status=RUN_OK, facts={"applied": False, "timed_out": False},
                          cost=cost)
    return ExecResult(request_id=request_id, status=RUN_ERROR, error=str(entry.get("reason") or "no_output"),
                      cost=cost)


# ---------------------------------------------------------------------------------------------------------- outputs


def plain(value):
    """JSON for values that hold the records' frozen mappings."""
    if isinstance(value, FrozenDict):
        return thaw(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: str, value) -> str:
    data = (json.dumps(value, indent=1, sort_keys=True, default=plain) + "\n").encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)
    return "sha256:" + hashlib.sha256(data).hexdigest()


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return "sha256:" + hashlib.sha256(f.read()).hexdigest()


def reward_of(outcome: Outcome | None) -> float | None:
    if outcome is None or outcome is Outcome.UNDETERMINABLE:
        return None
    return 1.0 if outcome is Outcome.PASS else 0.0


def reason_class(score: ScoreRecord) -> str:
    """Why a score is undeterminable, in a few classes."""
    if (score.reason or "").startswith("unrepairable"):
        return "unrepairable"
    reasons = [p.reason or "" for p in score.parts if not p.logical] or [score.reason or ""]
    r = reasons[0]
    if r.startswith("no_backend:") or "not_recorded" in r:
        return "needs_execution_not_recorded"
    if r.startswith("infra_error") or r.startswith("apply_infra_error") or r == "timeout":
        return "infrastructure_error"
    return r.split(":")[0] or "unknown"


# ---------------------------------------------------------------------------------------------------------- main


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="the trainer's run directory (read only)")
    ap.add_argument("--out", required=True, help="output directory (new or empty)")
    ap.add_argument("--executions", action="append", default=[],
                    help="a directory of re-executions recorded after the run (repeatable)")
    ap.add_argument("--name", default=None, help="the run's name in the outputs (default: the directory name)")
    ap.add_argument("--epoch-store", default=None, help="the epochs' object store, if not where the bodies say")
    ap.add_argument("--swebench", default=None, help="source tree of the official SWE-bench harness, if not installed")
    ap.add_argument("--threads", type=int, default=64, help="parallel file reads")
    ap.add_argument("--no-outcome-digest", action="store_true",
                    help="skip the outcome-digest check (runs whose sidecars record no outcome digest)")
    args = ap.parse_args(argv)
    if args.swebench:
        sys.path.insert(0, os.path.abspath(args.swebench))
    from agent_ecc.adapters.swebench import SWEBenchAdapter

    root = os.path.abspath(args.run)
    run_name = args.name or os.path.basename(root.rstrip("/"))
    out = os.path.abspath(args.out)
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"--out {out} is not empty")
    os.makedirs(out, exist_ok=True)
    zero = lambda: 0.0  # noqa: E731 - fixed clocks: score ids do not depend on how long parsing took
    started = time.monotonic()
    configuration = {
        "adapter": "SWEBenchAdapter(verdict='official'); each spec names its rule (official -> logical, "
                   "reference_set -> reference_set)",
        "scorer": {"policy": "reparse_all", "retries": 0, "short_circuit": True, "recorded_undeterminable": "resolve",
                   "clock": "fixed"},
        "levels": {"grading": "runs made when each attempt was graded",
                   "run": "plus the re-executions the run's seals recorded",
                   "posthoc": "plus re-executions recorded after the run"},
        "executions_run_here": 0,
    }
    summary: dict = {"run": run_name, "configuration": configuration}

    # ------------------------------------------------------------------------ 1. the run's logs
    mode = open(os.path.join(root, "ecc", "mode"), encoding="utf-8").read().strip()
    publications = read_jsonl(os.path.join(root, "ecc", "publish.log"))
    bodies = {int(p["epoch"]): read_json(os.path.join(root, "ecc", "epochs", f"{int(p['epoch'])}.json"))
              for p in publications}
    knobs = read_json(os.path.join(root, "knobs.json"))
    lookup = member_sidecar_lookup(knobs)
    seals = read_jsonl(os.path.join(root, "ecc", "seal.jsonl"))
    learner_name = (knobs or {}).get("learner_record_log") or "learner-records.jsonl"
    learner = read_jsonl(os.path.join(root, learner_name))
    events = read_jsonl(os.path.join(root, "audit-events", "main.jsonl"))

    names = os.listdir(os.path.join(root, "reval-meta"))
    chosen, superseded = sidecar_choice(names, lookup)
    paths = {rid: os.path.join(root, "reval-meta", f) for rid, f in chosen.items() if f}
    raw = read_bytes(list(paths.values()), args.threads)
    sidecars = {}
    for rid, path in paths.items():
        data = raw.get(path)
        if data is not None:
            sc = json.loads(data)
            if sc.get("schema") not in (CONJUNCTION_SCHEMA, SINGLE_SUITE_SCHEMA):
                raise SystemExit(f"{path}: unknown sidecar schema {sc.get('schema')!r}")
            sidecars[rid] = sc
    del raw
    # a sidecar's evidence is read only if its outcome digest binds its trajectory to the patch it names
    digest_check = {rid: "ok" if args.no_outcome_digest else outcome_digest_check(sc) for rid, sc in sidecars.items()}
    rejected = {rid: c for rid, c in digest_check.items() if c != "ok"}

    # evidence: every object the records, the seals' and the later re-executions name, read once and verified
    sources = [("run", root)] + [(f"posthoc{i}", os.path.abspath(d)) for i, d in enumerate(args.executions)]
    spec_names: dict = {}
    for rid, sc in sidecars.items():
        if rid in rejected:
            continue
        for st in (sc.get("stamps") or {}).get("specs") or []:
            if st.get("spec_digest"):
                spec_names.setdefault((sc.get("task_id"), st["spec_digest"]), set()).add(str(st.get("name")))
    for s in seals:
        for m in list(s.get("members") or []) + list((s.get("postseal") or {}).get("excluded_members") or []):
            for p in ((m.get("swe") or {}).get("plan") or []):
                if p.get("spec_digest") and p.get("name"):
                    spec_names.setdefault((m.get("task"), p["spec_digest"]), set()).add(str(p["name"]))
    index, index_notes = execution_index(sources, spec_names)
    wanted: dict[str, str] = {}
    for rid, sc in sidecars.items():
        if rid in rejected:
            continue
        ev = sc.get("evidence") or {}
        refs = [ev.get("agent_patch")] + [s.get("test_log") for s in ev.get("suites") or []]
        for ref in refs:
            if ref and ref.get("digest"):
                wanted[ref["digest"]] = artifact_path(root, ref["digest"])
    log_source: dict[str, str] = {}  # the log of a recorded re-execution -> the source that recorded it
    for entries in index.values():
        for label, src, e in entries:
            if (e.get("log") or {}).get("digest"):
                wanted.setdefault(e["log"]["digest"], artifact_path(src, e["log"]["digest"]))
                log_source.setdefault(e["log"]["digest"], label)
    raw = read_bytes(list(wanted.values()), args.threads)
    evidence = MemoryStore()
    corrupt, missing, nbytes = [], [], 0
    for digest, path in wanted.items():
        data = raw.get(path)
        if data is None:
            missing.append(digest)
        elif digest_bytes(data) != digest:
            corrupt.append(digest)
        else:
            evidence.put(data)
            nbytes += len(data)
    del raw
    epoch_stores = []
    for body in bodies.values():
        store = str(body.get("store") or "")
        for candidate in (args.epoch_store, store, os.path.normpath(os.path.join(root, store))):
            if candidate and os.path.isdir(candidate):
                if candidate not in epoch_stores:
                    epoch_stores.append(candidate)
                break
        else:
            raise SystemExit(f"epoch {body.get('epoch')}: object store not found (pass --epoch-store)")
    work = MemoryStore()  # objects agent-ecc writes: eval scripts, labels, filtered patches
    store = ChainStore(work, evidence, *[DirectoryStore(p) for p in epoch_stores])

    # ------------------------------------------------------------------------ 2. versions and the registry
    adapter = SWEBenchAdapter(verdict="official")
    task_meta = {}
    for sc in sidecars.values():
        task = sc.get("task") or {}
        task_meta.setdefault(str(sc.get("task_id")), {"repo": task.get("repo"), "version": task.get("version")})
    identity_issues: list = []
    version_of_epoch = {e: version_from_body(adapter, store, b, task_meta, identity_issues) for e, b in bodies.items()}
    now = [0.0]
    registry = DirectoryRegistry(os.path.join(out, "registry"), clock=lambda: now[0])
    seq_of_epoch = {}
    for p in publications:
        now[0] = float(p["t"])
        pub = registry.publish_grader(version_of_epoch[int(p["epoch"])], source=p.get("source"))
        seq_of_epoch[int(p["epoch"])] = pub.seq
    versions = [version_of_epoch[int(p["epoch"])] for p in publications]
    first, last = versions[0], versions[-1]
    summary["versions"] = [{"epoch": int(p["epoch"]), "name": v.name, "version_id": v.version_id,
                            "trainer_content_digest": bodies[int(p["epoch"])].get("content_digest"),
                            "source": p.get("source"), "tasks": len(v.entries),
                            "checks_per_task": sorted({len(list(iter_checks(s[ANY_WORLD])))
                                                       for s in v.entries.values()})}
                           for p, v in zip(publications, versions)]
    summary["identity_differences"] = len(identity_issues)

    # ------------------------------------------------------------------------ 3. records
    records, recorded = [], {}
    by_traj = {}
    for rid in sorted(sidecars, key=lambda r: (int(r.split("/a")[0]) if r.split("/a")[0].isdigit() else 0, r)):
        if rid in rejected:
            record, score = unrepairable_record(rid, sidecars[rid], rejected[rid])
        else:
            record, score = record_from_sidecar(rid, sidecars[rid], version_of_epoch)
        records.append(record)
        recorded[record.record_id] = score
        by_traj[rid] = record
    dump_jsonl(os.path.join(out, "records.jsonl"), records)
    records_sha = file_sha256(os.path.join(out, "records.jsonl"))
    graded = Counter(str((sc.get("graded_epoch") or {}).get("name")) for sc in sidecars.values())
    summary["records"] = {
        "sidecar_lookup": lookup, "attempts_listed": len(chosen), "records": len(records),
        "attempts_without_record": sum(1 for f in chosen.values() if f is None),
        "superseded_backend_attempts": superseded, "by_graded_epoch": dict(sorted(graded.items())),
        "suite_runs": sum(len(r.runs) for r in records),
        "suite_runs_by_name": dict(Counter(run.check for r in records for run in r.runs)),
        "empty_patches": sum(1 for r in records if r.facts.get("patch_empty")),
        "recorded_verdicts": dict(Counter(s.outcome.value for s in recorded.values())),
        "evidence_objects": len(wanted) - len(missing) - len(corrupt), "evidence_bytes": nbytes,
        "evidence_missing": len(missing), "evidence_corrupt": len(corrupt),
        "records_jsonl_sha256": records_sha,
        "outcome_digest": {"checked": not args.no_outcome_digest, "unrepairable": len(rejected),
                           **{k: v for k, v in sorted(Counter(rejected.values()).items())}},
    }

    # ------------------------------------------------------------------------ 4. scores under every version
    def results_for(level: str) -> dict[str, ExecResult]:
        allowed = {"grading": (), "run": ("run",), "posthoc": tuple(lbl for lbl, _ in sources)}[level]
        results = {}
        for record in records:
            patch = record.artifact("patch")
            if patch is None:
                continue
            for version in versions:
                for check in iter_checks(version.entries[record.task_id][ANY_WORLD]):
                    entries = [x for x in index.get((record.task_id, check.name, patch.digest), ())
                               if x[0] in allowed]
                    if not entries:
                        continue
                    request = adapter.reexec_request(record, check, store)
                    if request is None or request.request_id in results:
                        continue
                    _, src, e = entries[0]  # the run's own first, then later sources, in log order
                    results[request.request_id] = exec_result(request.request_id, src, e)
        return results

    def score_all(version: GraderVersion, level: str):
        results = results_for(level)
        backend = RecordedBackend(results) if results else None
        scorer = Scorer([adapter], store, backend, policy="reparse_all", retries=0, clock=zero)
        scored = scorer.score([ScoreInput(r, (recorded[r.record_id],)) for r in records], version)
        return [s.score for s in scored]

    scores: dict[tuple[str, str], list[ScoreRecord]] = {}
    manifests: dict[tuple[str, str], EvaluationManifest] = {}
    for version in versions:
        for level in LEVELS:
            if level != "run" and version is not last:
                continue
            if level == "posthoc" and len(sources) == 1:
                continue
            got = score_all(version, level)
            scores[(version.version_id, level)] = got
            manifests[(version.version_id, level)] = evaluation_manifest(records, got, version)
    all_scores = list(recorded.values()) + [s for ss in scores.values() for s in ss]
    dump_jsonl(os.path.join(out, "scores.jsonl"), all_scores)

    manifest_files = {}
    for (vid, level), m in manifests.items():
        v = next(x for x in versions if x.version_id == vid)
        fname = f"eval_{v.name}.json" if level == "run" else f"eval_{v.name}.{level}.json"
        manifest_files[(vid, level)] = {"file": fname, "manifest_id": m.manifest_id,
                                        "file_sha256": write_json(os.path.join(out, fname), m.to_dict())}
    diff = compare_evaluations(manifests[(first.version_id, "run")], manifests[(last.version_id, "run")])
    write_json(os.path.join(out, "eval_diff.json"), diff)

    # which suites the run's seals executed for which trajectory (a served execution may be an identical request
    # recorded for another trajectory: same patch after the suite's filter, same check)
    rec_by_id = {r.record_id: r for r in records}
    executed_for = set()
    for s in seals:
        for m in list(s.get("members") or []) + list((s.get("postseal") or {}).get("excluded_members") or []):
            for suite in (((m.get("swe") or {}).get("exec") or {}).get("suites") or []):
                if suite.get("log") and not suite.get("skipped"):
                    executed_for.add((str(m["traj"]), str(suite.get("name"))))

    def manifest_summary(vid, level):
        m = manifests[(vid, level)]
        ss = scores[(vid, level)]
        undet = Counter(reason_class(s) for s in ss if not s.logical)
        served, logs = Counter(), set()
        for sc in ss:
            r = rec_by_id[sc.execution_id]
            for p in sc.parts:
                if p.method is Method.REEXEC and p.run is not None and p.run.outputs:
                    logs.add(p.run.outputs[0].digest)
                    src = log_source.get(p.run.outputs[0].digest, "unknown")
                    if src == "run":
                        if (r.trajectory_id, p.check) in executed_for:
                            src += ":executed_for_this_trajectory"
                        elif (r.task_id, p.check, r.artifact("patch").digest) in index:
                            src += ":same_patch_executed_for_another_trajectory"
                        else:
                            src += ":same_patch_after_the_suite_filter_executed_for_another_trajectory"
                    served[src] += 1
        return {**manifest_files[(vid, level)], "level": level, "record_set": m.record_set,
                "records": m.counts["records"], "pass": m.counts["pass"], "fail": m.counts["fail"],
                "undeterminable": m.undeterminable, "pass_rate_decided": m.counts["pass_rate"],
                "by_method": dict(m.counts["by_method"]), "undeterminable_by_reason": dict(undet),
                "short_circuited": sum(1 for s in ss if s.skipped),
                "re_executed_parts": sum(served.values()), "distinct_recorded_executions_used": len(logs),
                "re_executed_parts_by_source": dict(served)}

    summary["evaluations"] = {f"{next(x.name for x in versions if x.version_id == vid)}@{level}":
                              manifest_summary(vid, level) for vid, level in manifests}
    summary["difference"] = {k: diff[k] for k in ("from_version", "to_version", "records", "changed", "flips",
                                                  "pass_to_fail", "fail_to_pass", "became_undeterminable",
                                                  "pass_rate")}

    # ---------------------------------------------------------------------- (a) records unchanged, scores side by side
    reloaded = load_jsonl(os.path.join(out, "records.jsonl"), ExecutionRecord)  # verifies every record id
    record_sets = {m.record_set for m in manifests.values()}
    per_record = defaultdict(set)
    for (vid, level), ss in scores.items():
        if level == "run":
            for s in ss:
                per_record[s.execution_id].add(vid)
    summary["records_unchanged"] = {
        "records_reloaded": len(reloaded),
        "record_ids_equal": [r.record_id for r in reloaded] == [r.record_id for r in records],
        "records_jsonl_sha256_after_scoring": file_sha256(os.path.join(out, "records.jsonl")),
        "same_sha256": file_sha256(os.path.join(out, "records.jsonl")) == records_sha,
        "one_record_set_for_every_manifest": len(record_sets) == 1,
        "records_with_a_score_under_every_version": sum(1 for r in records
                                                        if per_record[r.record_id] == {v.version_id for v in versions}),
        "score_records_written": len(all_scores),
    }

    # ------------------------------------------------------------------------ (b) every last-version score traces back
    def trace(score: ScoreRecord, record: ExecutionRecord, version: GraderVersion) -> str | None:
        """None when the score traces to its record, its version's checks and verified evidence; else why not."""
        if score.grader_version != version.version_id or score.execution_id != record.record_id:
            return "identity"
        spec = version.entries[record.task_id][ANY_WORLD]
        if score.spec_digest != spec_digest(spec):
            return "spec_digest"
        checks = {c.name: c for c in iter_checks(spec)}
        own_runs = {r.run_digest for r in record.runs}
        for p in score.parts:
            c = checks.get(p.check)
            if c is None or (p.execution_digest, p.grading_digest) != (c.execution_digest, c.grading_digest):
                return f"part_identity:{p.check}"
            if not p.logical:
                continue
            if p.reason == NO_PATCH:
                if not record.facts.get("patch_empty"):
                    return "empty_patch_not_in_record"
                continue
            if p.run is None:
                return f"no_evidence:{p.check}"
            if p.method is Method.REPARSE and p.run.run_digest not in own_runs:
                return f"reparsed_run_not_in_record:{p.check}"
            for o in p.run.outputs:
                try:
                    store.get(o.digest)  # verified against its digest
                except Exception:  # noqa: BLE001
                    return f"evidence_unreadable:{p.check}"
        for check_name in score.skipped:
            if check_name not in checks:
                return "skipped_unknown_check"
        return None

    traced = Counter()
    trace_failures = []
    evidence_kinds = Counter()
    for s in scores[(last.version_id, "run")]:
        why = trace(s, rec_by_id[s.execution_id], last)
        traced["traced" if why is None else "not_traced"] += 1
        if why is not None:
            trace_failures.append({"record": rec_by_id[s.execution_id].trajectory_id, "why": why})
        for p in s.parts:
            if not p.logical:
                evidence_kinds["undeterminable part: " + reason_class(s)] += 1
            elif p.reason == NO_PATCH:
                evidence_kinds["empty patch (record facts)"] += 1
            elif p.method is Method.REPARSE:
                evidence_kinds[f"{p.check}: grading-time log re-parsed"] += 1
            elif p.method is Method.REEXEC:
                evidence_kinds[f"{p.check}: recorded seal-time re-execution re-parsed"] += 1
        for check_name in s.skipped:
            evidence_kinds[f"{check_name}: skipped after a logical fail"] += 1
    summary["traceability"] = {"version": last.name, "level": "run", **dict(traced),
                               "failures": trace_failures[:20], "parts_by_evidence": dict(evidence_kinds)}

    # ------------------------------------------------------------------------ 5. the trainer's seal decisions
    score_of = {lvl: {} for lvl in LEVELS}
    for (vid, level), ss in scores.items():
        for s in ss:
            score_of[level][(vid, s.execution_id)] = s
    decisions, label_check, epoch_check = [], Counter(), Counter()
    label_differences = []
    for i, s in enumerate(seals):
        ref = registry.epoch_at(float(s["t_read"]))
        ref = dataclasses.replace(ref, read_at=float(s["t_read"]))
        epoch_check["agree" if ref.name == s.get("epoch_name") and ref.seq == seq_of_epoch.get(s.get("epoch"))
                    else "differ"] += 1
        vid = ref.version_id
        post = s.get("postseal") or {}

        def member(m):
            action, reason, short, consistent = STATUS[m["status"]]
            record = by_traj.get(str(m["traj"]))
            sc = score_of["run"].get((vid, record.record_id)) if record is not None else None
            label = m.get("target_label")
            outcome = None if label is None else (Outcome.PASS if label else Outcome.FAIL)
            target = reward_of(outcome)
            rec = m.get("recorded")
            ours = sc.outcome if sc is not None else None
            if outcome is None:
                label_check["trainer_no_label:" + ("ours_" + ours.value if ours else "no_record")] += 1
            elif ours is None:
                label_check["no_record"] += 1
            elif not ours.logical:
                label_check["ours_undeterminable"] += 1
            elif ours is outcome:
                label_check["agree"] += 1
            else:
                label_check["differ"] += 1
                label_differences.append({"seal": i, "member": m["traj"], "trainer": outcome.value,
                                          "agent_ecc": ours.value})
            return MemberDecision(member_id=str(m["traj"]), execution_id=record.record_id if record else None,
                                  action=action, outcome=outcome, reason=reason,
                                  score_id=sc.score_id if sc is not None else None, recorded_reward=rec,
                                  target_reward=target, trained_reward=m.get("trained"),
                                  flipped=None if target is None or rec is None else float(rec) != target,
                                  short_circuit=short, consistent=consistent)

        def groups(members, role_of):
            units: dict[str, list] = {}
            for m in members:
                units.setdefault(str(m.get("unit") if post else m.get("g")), []).append(m)
            out_groups = []
            for unit, ms in units.items():
                role = role_of(ms[0])
                neutral = any(m.get("neutralized") for m in ms)
                out_groups.append(GroupDecision(group_id=unit, members=tuple(member(m) for m in ms),
                                                round=int(ms[0].get("round") or 1), role=role,
                                                source=ms[0].get("source"), neutralized=neutral,
                                                neutralize_reason=("filler" if role == "filler" else "no_label")
                                                if neutral else None))
            return out_groups

        batch = groups(s.get("members") or [], lambda m: m.get("batch_role") if m.get("batch_role") in
                       ("selected", "backfill", "filler") else "selected")
        excluded = [Exclusion(group=g, reason="filtered_after_decision")
                    for g in groups(post.get("excluded_members") or [], lambda m: "selected")]
        policy = "full" if mode == "fullcompat" and ref.seq != 0 else "ecc"
        decisions.append(SealDecision(
            seal_id=f"{run_name}:{i}", epoch=ref, mode=SEAL_MODES.get(mode, mode), policy=policy,
            strict=bool(s.get("strict", True)), groups=tuple(batch), batch_id=str(s["rollout_id"]),
            excluded=tuple(excluded), target_groups=post.get("target_groups"), rounds=int(post.get("rounds") or 1),
            opened_at=float(s["t"]), stats={"trainer_counts": s.get("counts") or {}, "source": "trainer seal log"}))
    dump_jsonl(os.path.join(out, "seals.jsonl"), decisions)
    by_epoch = Counter(d.epoch.name for d in decisions)
    statuses = Counter()
    for s in seals:
        for m in s.get("members") or []:
            statuses[f"{s.get('epoch_name')}:batch:{m['status']}"] += 1
        for m in (s.get("postseal") or {}).get("excluded_members") or []:
            statuses[f"{s.get('epoch_name')}:excluded:{m['status']}"] += 1
    summary["seals"] = {"seals": len(decisions), "by_epoch": dict(by_epoch), "epoch_from_registry": dict(epoch_check),
                        "members_decided": sum(len(g.members) for d in decisions for g in d.groups)
                        + sum(len(e.group.members) for d in decisions for e in d.excluded),
                        "members_in_batches": sum(len(g.members) for d in decisions for g in d.groups),
                        "groups_in_batches": sum(len(d.groups) for d in decisions),
                        "groups_excluded": sum(len(d.excluded) for d in decisions),
                        "neutralized_groups": sum(d.neutralized_groups for d in decisions),
                        "trainer_statuses": dict(sorted(statuses.items())),
                        "trainer_label_vs_manifest": dict(label_check), "label_differences": label_differences[:20]}

    # ------------------------------------------------------------------------ 6. what the learner consumed
    lines = [LearnerLine(batch_id=str(x["sealed_rollout_id"]), member_id=str(x["record_id"]),
                         group_id=x.get("unit_id"), step=x.get("step"), reward=x.get("raw_reward"),
                         normalized=x.get("reward"), advantage=x.get("advantage"),
                         advantage_min=x.get("advantage_min"), advantage_max=x.get("advantage_max"),
                         loss_tokens=x.get("loss_mask_tokens"), handoff_loss_tokens=x.get("handoff_loss_mask_tokens"),
                         identity=x.get("identity"), reporter=x.get("reporter"), microbatch=x.get("microbatch"))
             for x in learner]
    completions = []
    for e in events:
        if e.get("type") == "train_group_step_end" and e.get("rollout_id") is not None:
            cells = e.get("cell_outcomes") or {}
            ok = all(o == "normal" for outs in cells.values() for o in outs)
            completions.append(StepCompletion(batch_id=str(e["rollout_id"]), step=int(e["rollout_id"]),
                                              completed_at=utc_seconds(e["timestamp"]), ok=ok))
    with open(os.path.join(out, "learner.jsonl"), "w") as f:
        f.writelines(json.dumps(line.to_dict(), sort_keys=True) + "\n" for line in lines)
    with open(os.path.join(out, "completions.jsonl"), "w") as f:
        f.writelines(json.dumps(c.to_dict(), sort_keys=True) + "\n" for c in completions)
    report = consumption_records(decisions, lines, completions)
    dump_jsonl(os.path.join(out, "consumption.jsonl"), report.records)
    training_sha = write_json(os.path.join(out, "training.json"), report.manifest.to_dict())
    tm = report.manifest

    # (c) every consumed reward against the verdict of its batch epoch
    scores_by_id = {s.score_id: s for ss in scores.values() for s in ss}
    decision_of = {d.batch_id: d for d in decisions}
    join = Counter()
    join_by_epoch = Counter()
    mismatches, posthoc_check = [], Counter()
    changed_consumed = Counter()
    first_epoch_rows = Counter()
    for c in report.records:
        d = decision_of.get(c.batch_id)
        groups_by_member = {m.member_id: g for g, m in d.members()} if d else {}
        for row in c.records:
            if row.score_id is None:
                join["no_record"] += 1
                continue
            s = scores_by_id[row.score_id]
            if s.grader_version != c.epoch.version_id:
                join["score_not_under_batch_epoch"] += 1
                continue
            g = groups_by_member.get(row.member_id)
            if g is not None and g.neutralized:
                join["neutralized_group"] += 1
                continue
            want = reward_of(s.outcome)
            key = f"{c.epoch.name}"
            if c.epoch.version_id == first.version_id:
                later = score_of["run"].get((last.version_id, s.execution_id))
                first_epoch_rows["last_version_" + (later.outcome.value if later else "none")] += 1
                if later is not None and later.logical and s.logical and later.outcome is not s.outcome:
                    first_epoch_rows["label_differs_under_last_version"] += 1
            if want is None:
                join["no_verdict_under_batch_epoch"] += 1
                join_by_epoch[key + ":no_verdict"] += 1
                ph = score_of["posthoc"].get((c.epoch.version_id, s.execution_id))
                if ph is not None:
                    pw = reward_of(ph.outcome)
                    if pw is None:
                        posthoc_check["still_undeterminable"] += 1
                    elif row.reward is not None and abs(pw - row.reward) < 1e-9:
                        posthoc_check["consumed_reward_equals_posthoc_verdict"] += 1
                    else:
                        posthoc_check["consumed_reward_differs_from_posthoc_verdict"] += 1
            elif row.reward is not None and abs(want - row.reward) < 1e-9:
                join["equals_batch_epoch_verdict"] += 1
                join_by_epoch[key + ":equal"] += 1
            else:
                join["differs_from_batch_epoch_verdict"] += 1
                join_by_epoch[key + ":differ"] += 1
                mismatches.append({"step": c.step, "member": row.member_id, "consumed": row.reward, "verdict": want})
            # records whose verdict the version change flipped: which version's label did the learner consume?
            if c.epoch.version_id != first.version_id:
                g1 = score_of["run"].get((first.version_id, s.execution_id))
                if g1 is not None and g1.logical and s.logical and g1.outcome is not s.outcome:
                    took = row.reward is not None and abs(reward_of(s.outcome) - row.reward) < 1e-9
                    changed_consumed["consumed_the_batch_epoch_label" if took
                                     else "consumed_the_first_version_label"] += 1
    summary["consumption"] = {
        "learner_lines": len(lines), "step_completions": len(completions),
        "steps_consumed": len(report.records), "records_consumed": tm.records_consumed, "manifest_ok": tm.ok,
        "issues": {k: v for k, v in tm.issues.items() if v}, "labels_consumed_by_action": dict(tm.labels),
        "consumed_by_version": {next(v.name for v in versions if v.version_id == k): n
                                for k, n in tm.by_grader_version.items() if k != "none"},
        "normalization": dict(tm.normalization), "incomplete_batches": list(tm.incomplete_batches),
        "unconfirmed_batches": list(tm.unconfirmed_batches), "unconsumed_seals": list(tm.unconsumed_seals),
        "training_manifest_id": tm.manifest_id, "training_json_sha256": training_sha,
        "join": dict(join), "join_by_epoch": dict(sorted(join_by_epoch.items())), "mismatches": mismatches[:20],
        "no_verdict_checked_posthoc": dict(posthoc_check), "flipped_records_consumed": dict(changed_consumed),
        "consumed_under_first_version": dict(first_epoch_rows),
    }

    # where the records without a last-version verdict were in the run: consumed or decided under which epoch
    consumed_epoch = {}
    for c in report.records:
        for row in c.records:
            consumed_epoch[row.member_id] = c.epoch.name
    decided_epoch = defaultdict(set)
    for d in decisions:
        for g in list(d.groups) + [x.group for x in d.excluded]:
            for m in g.members:
                decided_epoch[m.member_id].add(d.epoch.name)
    fate = {}
    for level in LEVELS:
        if (last.version_id, level) not in scores:
            continue
        counts = Counter()
        for sc in scores[(last.version_id, level)]:
            if sc.logical:
                continue
            traj = rec_by_id[sc.execution_id].trajectory_id
            if traj in consumed_epoch:
                where = f"consumed_under_{consumed_epoch[traj]}"
            elif decided_epoch.get(traj):
                where = "decided_not_consumed_under_" + "+".join(sorted(decided_epoch[traj]))
            else:
                where = "never_sealed"
            counts[f"{reason_class(sc)}:{where}"] += 1
        fate[level] = dict(sorted(counts.items()))
    summary["undeterminable_records_in_the_run"] = {"version": last.name, **fate}

    # ------------------------------------------------------------------------ 7. reproduction
    loaded = load_jsonl(os.path.join(out, "records.jsonl"), ExecutionRecord)
    again_registry = DirectoryRegistry(os.path.join(out, "registry"))
    g1 = again_registry.grader(first.version_id)  # the body as published, verified against its id
    prior = {r.record_id: (recorded[r.record_id],) for r in loaded}
    offline = Scorer([adapter], store, None, policy="reparse_all", retries=0, clock=zero)
    m1 = manifests[(first.version_id, "run")]
    rep1 = reproduce_evaluation(m1, loaded, g1, offline, prior=prior)
    # against what the run itself logged for the first version's suites when it graded each attempt
    check_names = {next(iter_checks(g1.entries[t][ANY_WORLD])).name for t in g1.entries}
    logged = Counter()
    logged_pass = logged_decided = 0
    differ = []
    for r in loaded:
        sc = sidecars[r.trajectory_id]
        if r.trajectory_id in rejected:
            logged["unrepairable"] += 1  # its logged verdict rests on evidence the digest does not bind
            continue
        spec = g1.entries[r.task_id][ANY_WORLD]
        classes = [logged_suite_class(sc, c.name) for c in iter_checks(spec)]
        if any(x is None for x in classes):
            logged["not_logged"] += 1
            continue
        if "fail" in classes:
            want = "fail"
        elif all(x == "pass" for x in classes):
            want = "pass"
        else:
            want = "undeterminable"
        got = m1.outcomes()[r.record_id].value
        logged["equal" if got == want else "differ"] += 1
        if got != want:
            differ.append({"record": r.trajectory_id, "logged": want, "agent_ecc": got})
        if want != "undeterminable":
            logged_decided += 1
            logged_pass += want == "pass"
    m2 = manifests[(last.version_id, "run")]
    g2 = again_registry.grader(last.version_id)
    prior2 = {r.record_id: (recorded[r.record_id], score_of["run"][(last.version_id, r.record_id)]) for r in loaded}
    rep2 = reproduce_evaluation(m2, loaded, g2, offline, prior=prior2)
    summary["reproduction"] = {
        "first_version": {"version": first.name, "reproduced": rep1["reproduced"], "records": rep1["records"],
                          "differ": len(rep1["differ"]), "manifest_id": rep1["manifest_id"],
                          "again_id": rep1["again_id"], "same_manifest_id": rep1["manifest_id"] == rep1["again_id"],
                          "against_run_log": dict(logged), "run_log_differences": differ[:20],
                          "suites_compared": sorted(check_names),
                          "run_logged_pass_rate": (logged_pass / logged_decided) if logged_decided else None,
                          "manifest_pass_rate": m1.counts["pass_rate"]},
        "last_version_without_executing": {"version": last.name, "reproduced": rep2["reproduced"],
                                           "records": rep2["records"], "differ": len(rep2["differ"]),
                                           "manifest_id": rep2["manifest_id"], "again_id": rep2["again_id"]},
    }

    # ------------------------------------------------------------------------ one record, end to end
    example = None
    consumed_at = {line.member_id: line for line in lines}
    for d in decisions:
        if d.epoch.version_id == first.version_id:
            continue
        for g, m in d.members():
            if m.action == "reexec" and m.flipped and m.member_id in consumed_at and m.execution_id:
                r = rec_by_id[m.execution_id]
                s1 = score_of["run"][(first.version_id, r.record_id)]
                s2 = score_of["run"][(last.version_id, r.record_id)]
                line = consumed_at[m.member_id]
                example = {
                    "trajectory": r.trajectory_id, "task": r.task_id, "execution_record": r.record_id,
                    "patch": (r.artifact("patch").digest if r.artifact("patch") else None),
                    "grading_runs": [{"suite": x.check, "log": x.outputs[0].digest if x.outputs else None}
                                     for x in r.runs],
                    "first_version_score": {"score_id": s1.score_id, "outcome": s1.outcome.value,
                                            "method": s1.method.value},
                    "last_version_score": {"score_id": s2.score_id, "outcome": s2.outcome.value,
                                           "method": s2.method.value,
                                           "parts": [{"suite": p.check, "outcome": p.outcome.value,
                                                      "method": p.method.value,
                                                      "log": p.run.outputs[0].digest
                                                      if p.run is not None and p.run.outputs else None}
                                                     for p in s2.parts]},
                    "seal": {"seal_id": d.seal_id, "batch": d.batch_id, "epoch": d.epoch.name,
                             "decision_id": d.decision_id, "action": m.action, "status": m.status,
                             "recorded_reward": m.recorded_reward, "trained_reward": m.trained_reward},
                    "learner": {"step": line.step, "reward": line.reward, "advantage": line.advantage,
                                "identity": line.identity},
                }
                break
        if example:
            break
    summary["example_record"] = example
    summary["wall_seconds"] = round(time.monotonic() - started, 1)
    write_json(os.path.join(out, "summary.json"), summary)
    print(json.dumps(summary, indent=1, sort_keys=True, default=plain))
    return summary


if __name__ == "__main__":
    main()
