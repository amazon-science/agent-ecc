# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SWE-bench: evidence format, log verdicts with the official harness, re-execution requests, grader versions.

Evidence. A SWE-bench execution record holds the agent's patch (role ``patch``) and one run per test suite executed
when it was graded: the run's execution identity (image, eval script, patch-application commands, an optional patch
filter) and its raw test log (role ``test_log``), with facts the runner observed (whether the patch applied and the
record of its application attempts, a timeout, the test exit code). An empty patch is recorded as the fact
``patch_empty``: it fails every suite without being executed.

Checks. A check built from a dataset row has

* execution identity: ``image``, ``eval_script`` (digest of the script object), ``apply_cmds`` (digest of the
  patch-application fallback list), and for a hardened suite ``patch_exclude`` (the digest of the sorted list of
  repository paths whose per-file diffs leave the patch under test, :func:`exclude_paths`);
* grading identity: ``labels`` (digest of the canonical FAIL_TO_PASS / PASS_TO_PASS object), ``parser`` and
  ``grading_code`` (digests of the harness code that turns a log into a verdict), ``eval_type``, and
  ``verdict_rule`` (how a log becomes a verdict, below).

So a label, parser, grading-code or verdict-rule change re-parses the stored log, and an eval-script, image, apply or
patch-filter change re-executes the stored patch.

Verdict rules (:func:`verdict_rule`). Under ``logical`` (the default) and ``reference_set`` a run is a **logical**
verdict unless it hit an infrastructure error, which is never a verdict:

* the run did not produce its outputs (an evaluator, container or sandbox failure: a run with status ``error``);
* a patch-application attempt (or the restore between two, or the already-applied check) errored in the sandbox
  executor or exited with code 124 (:func:`apply_infrastructure_error`, read from the run's application record);
* on the log, in this order (:func:`parse_log`): the harness's timeout marker ends it; it has no start marker, or no
  end marker after it (an empty, missing or truncated log); it has the end marker but not the test exit-code record
  after it (a truncated log); the log parser raised.

Every other run is logical. A patch that genuinely does not apply fails, and so does a trajectory with no patch. The
status map is the official ``get_logs_eval``'s (the log between the output markers, the whole log as its fallback);
when it rejects a complete log the map is empty and the run fails. ``logical`` then takes the official report as the
verdict (``get_eval_tests_report`` and ``get_resolution_status``): a listed test without a status counts as failed, a
skipped PASS_TO_PASS test is maintained and a skipped FAIL_TO_PASS test fails. ``reference_set`` is the verdict of a
hardened suite against a reference set G held as the check's labels (FAIL_TO_PASS empty, PASS_TO_PASS = G): it passes
if and only if every test of G is PASSED or SKIPPED in the run (XFAIL is a failure, a parser's OK is PASSED, ids are
compared exactly, a test of G without a status fails). G itself is the set of tests passing in two valid gold runs
(:func:`reference_set`). No option makes a listed test without a status undeterminable. ``official`` instead takes the
official harness's own end-to-end report as the verdict (every log it does not resolve fails, a timeout included),
which is how SWE-bench's evaluation, and trainers that grade with it, have reported results: use it to reproduce
verdicts recorded under that rule. The rule is part of every check's grading identity, so a verdict obtained under
one rule is never reused, or re-read, under another: an official-rule fail of a run that timed out is re-read under
the logical rule, where it is not a verdict. Parsing uses the official ``swebench`` harness when it is installed
(:class:`OfficialHarness`, extra ``agent-ecc[swebench]``); :class:`SimpleHarness` is a small standard-library
stand-in for tests and examples, with its own identity.

SWE-ABS. :meth:`SWEBenchAdapter.conjunction_version` builds a version whose spec for each task is
``AllOf(original suite, hardened suite)``: it passes only if both pass, and fails if either fails logically (a logical
fail of the original suite decides the task without running the hardened one). An infrastructure error of the
original suite decides nothing: the hardened suite still runs, and fails the task if it fails. A hardened row with a
``reference_set`` is graded against G under ``reference_set``, on the patch without its per-file diffs of the hardened
test patch's paths (:func:`test_patch_paths`); if nothing is left the tests run on the unmodified tree.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import os
import re
import sys
import tempfile
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..canonical import digest_text
from ..records import RUN_ERROR, RUN_OK, EvidenceRef, ExecutionRecord, Outcome, TestOutcome, TestRun
from ..recompute.base import ExecRequest
from ..scoring import PartVerdict
from ..store import ContentStore, get_json, get_text, put_json, put_text
from ..versions import ANY_WORLD, AllOf, Check, GraderVersion, TestSpec

__all__ = [
    "CHECK_HARDENED",
    "CHECK_TESTS",
    "FAMILY",
    "REFERENCE_PASSING",
    "VERDICT_RULES",
    "OfficialHarness",
    "ParsedLog",
    "SWEBenchAdapter",
    "SWEHarness",
    "SimpleHarness",
    "apply_infrastructure_error",
    "canonical_labels",
    "classify_log",
    "exclude_paths",
    "filter_patch",
    "gold_run_validity",
    "official_verdict",
    "parse_log",
    "reference_set",
    "substitute_test_patch",
    "test_patch_paths",
    "touched_files",
    "verdict_rule",
]

FAMILY = "swebench"
ROLE_PATCH = "patch"
ROLE_TEST_LOG = "test_log"
CHECK_TESTS = "tests"
CHECK_HARDENED = "hardened"
REQUEST_KIND = "swebench/eval"

START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
RESET_FAILED = ">>>>> Reset Failed"
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"
TEST_EXIT_CODE = ">>>>> Test Exit Code"
_TIMEOUT_RE = re.compile(r"Timeout error: \d+(\.\d+)? seconds exceeded")
# the harness's own timeout marker as the last thing in a log (the official runner appends it after a blank line)
_TRAILING_TIMEOUT_RE = re.compile(r"\n\nTimeout error: \d+(?:\.\d+)? seconds exceeded\.\s*\Z")
_EXIT_CODE_RE = re.compile(re.escape(TEST_EXIT_CODE) + r":\s*(-?\d+)")
_PASSING = ("PASSED", "XFAIL")
#: the statuses that pass a test of a reference set (SWE-ABS's: XFAIL is a failure; a parser's OK is PASSED)
REFERENCE_PASSING = ("PASSED", "SKIPPED")
_STATUS_ALIASES = {"OK": "PASSED"}
_APPLY_TIMEOUT_EXIT = 124


def canonical_labels(fail_to_pass: Any, pass_to_pass: Any) -> dict:
    """The labels object a check's ``labels`` digest names: both lists, sorted (order does not change a verdict)."""
    return {"FAIL_TO_PASS": sorted(_as_list(fail_to_pass)), "PASS_TO_PASS": sorted(_as_list(pass_to_pass))}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    return [str(v) for v in (value or [])]


LOGICAL, OFFICIAL, REFERENCE_SET = "logical", "official", "reference_set"
VERDICT_RULES = (LOGICAL, OFFICIAL, REFERENCE_SET)
# options of an earlier logical rule: refused, so that no setting makes a listed test without a status undeterminable
_SUPERSEDED_OPTIONS = ("missing_status", "cross_check")


def verdict_rule(verdict: str = LOGICAL) -> dict:
    """How a check turns a run into a verdict, as its grading identity names it (``grading["verdict_rule"]``):
    ``{"rule": "logical"}`` (the official report on a complete log, infrastructure errors no verdict),
    ``{"rule": "reference_set"}`` (a hardened suite against its reference set G) or ``{"rule": "official"}`` (the
    official harness's own end-to-end report on every log). See the module documentation. The rule has no options:
    a listed test without a status always fails."""
    if verdict not in VERDICT_RULES:
        raise ValueError(f"verdict must be one of {VERDICT_RULES}, got {verdict!r}")
    return {"rule": verdict}


def _validated_rule(rule: str | Mapping[str, Any]) -> dict:
    """A verdict rule given by its name (``"logical"``) or as the mapping a check's grading identity holds
    (``{"rule": "logical"}``), validated, as that mapping."""
    if isinstance(rule, str):
        return verdict_rule(rule)
    if not isinstance(rule, Mapping):
        raise ValueError(f"a verdict rule is a name or a mapping, got {type(rule).__name__}")
    kind = rule.get("rule")
    if kind in VERDICT_RULES and set(rule) == {"rule"}:
        return {"rule": kind}
    if kind == LOGICAL and set(rule) & set(_SUPERSEDED_OPTIONS):
        raise ValueError(f"verdict rule {dict(rule)!r} carries the superseded options {list(_SUPERSEDED_OPTIONS)}: "
                         "under the logical rule a listed test without a status fails and the official report "
                         "decides; build the check again")
    raise ValueError(f"unknown verdict rule {dict(rule)!r}")


def _refuse_superseded(options: Mapping[str, Any]) -> None:
    superseded = sorted(set(options) & set(_SUPERSEDED_OPTIONS))
    if superseded:
        raise ValueError(f"{superseded} are no longer options: under the logical rule a listed test without a status "
                         "fails, and the official report decides (the harness's end-to-end report is that report)")
    if options:
        raise TypeError(f"unexpected options {sorted(options)}")


# ---------------------------------------------------------------------------------------------- patch exclusion

_DIFF_GIT = "diff --git "


def _path_list(paths: Iterable[str] | None, what: str) -> list[str]:
    """Repository paths given as a collection; a string (which would be read character by character) is refused."""
    if isinstance(paths, (str, bytes)):
        raise TypeError(f"{what} is a collection of repository paths, not a string: {paths!r:.80}")
    return [str(p) for p in (paths or ())]


def _unquote(path: str) -> str:
    """A path as git prints it: C-quoted when it holds special characters."""
    path = path.strip()
    if len(path) >= 2 and path[0] == '"' and path[-1] == '"':
        body = path[1:-1]
        out = bytearray()
        i = 0
        escapes = {"n": 10, "t": 9, '"': 34, "\\": 92, "a": 7, "b": 8, "f": 12, "r": 13, "v": 11}
        while i < len(body):
            ch = body[i]
            if ch == "\\" and i + 1 < len(body):
                nxt = body[i + 1]
                octal = body[i + 1:i + 4]
                if len(octal) == 3 and all(c in "01234567" for c in octal):
                    out.append(int(octal, 8))
                    i += 4
                    continue
                if nxt in escapes:
                    out.append(escapes[nxt])
                    i += 2
                    continue
            out.extend(ch.encode())
            i += 1
        return out.decode("utf-8", "replace")
    return path


def _strip_prefix(path: str) -> str | None:
    path = _unquote(path.split("\t", 1)[0])
    if path == "/dev/null":
        return None
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def _header_paths(line: str) -> set[str]:
    """The paths of a ``diff --git a/X b/Y`` header (unambiguous when X == Y, or when they are quoted)."""
    rest = line[len(_DIFF_GIT):].rstrip("\n")
    if rest.startswith('"'):
        end = rest.find('"', 1)
        while end != -1 and rest[end - 1] == "\\":
            end = rest.find('"', end + 1)
        if end != -1:
            return {p for p in (_strip_prefix(rest[:end + 1]), _strip_prefix(rest[end + 1:].strip())) if p}
    if rest.startswith("a/"):
        half = (len(rest) - 1) // 2
        left, right = rest[:half], rest[half + 1:]
        if right.startswith("b/") and left[2:] == right[2:]:
            return {left[2:]}
        if " b/" in rest:  # a rename without quoting: the last split (the --- / +++ lines decide too)
            left, _, right = rest.rpartition(" b/")
            return {left[2:], right}
    return set()


def _split_patch(patch: str) -> list[str]:
    """The per-file diffs of a patch, in order. A git patch splits at its ``diff --git`` headers; a patch without them
    (a plain unified diff) at each ``--- `` line followed by a ``+++ `` line and an ``@@ `` line. Text before the first
    diff is a section of its own (it touches no file)."""
    lines = patch.splitlines(keepends=True)
    git = any(line.startswith(_DIFF_GIT) for line in lines)
    sections: list[list[str]] = []
    for i, line in enumerate(lines):
        if git:
            starts = line.startswith(_DIFF_GIT)
        else:
            starts = (line.startswith("--- ") and i + 2 < len(lines) and lines[i + 1].startswith("+++ ")
                      and lines[i + 2].startswith("@@ "))
        if starts or not sections:
            sections.append([line])
        else:
            sections[-1].append(line)
    return ["".join(section) for section in sections]


def _is_diff(section: str) -> bool:
    if section.startswith(_DIFF_GIT):
        return True
    lines = section.split("\n", 3)
    return (len(lines) >= 3 and lines[0].startswith("--- ") and lines[1].startswith("+++ ")
            and lines[2].startswith("@@ "))


def _diff_paths(section: str) -> set[str]:
    """Every repository path one per-file diff touches: old and new names (header, ---/+++, rename and copy lines,
    up to the first hunk)."""
    paths: set[str] = set()
    lines = section.splitlines()
    if lines and lines[0].startswith(_DIFF_GIT):
        paths |= _header_paths(lines[0])
        lines = lines[1:]
    for line in lines:
        if line.startswith("@@"):
            break
        for marker in ("--- ", "+++ "):
            if line.startswith(marker):
                path = _strip_prefix(line[len(marker):])
                if path:
                    paths.add(path)
        for marker in ("rename from ", "rename to ", "copy from ", "copy to "):
            if line.startswith(marker):
                paths.add(_unquote(line[len(marker):]))
    return paths


def filter_patch(patch: str, paths: Iterable[str]) -> tuple[str, dict]:
    """The patch without every per-file diff that touches one of ``paths`` (by its old or its new name), and a record of
    what was removed. A patch with no recognisable per-file diff is kept as it is (it is applied as it is); when only
    text that touches no file is left, the result is empty: nothing to apply, and the tests run on the unmodified
    tree."""
    exclude = set(_path_list(paths, "paths"))
    sections = _split_patch(patch)
    kept, removed = [], []
    for section in sections:
        touched = _diff_paths(section) if _is_diff(section) else set()
        if touched & exclude:
            removed.append(sorted(touched))
        else:
            kept.append(section)
    filtered = "".join(kept)
    if not any(_is_diff(section) for section in sections):
        filtered = patch
    elif not any(_is_diff(section) for section in kept):
        filtered = ""
    record = {"removed_files": sorted({p for touched in removed for p in touched}), "removed_diffs": len(removed),
              "kept_diffs": sum(1 for section in kept if _is_diff(section)), "excluded_paths": len(exclude),
              "patch_digest": digest_text(patch), "filtered_digest": digest_text(filtered),
              "empty_after_filter": not filtered.strip()}
    return filtered, record


def exclude_paths(patch: str, paths: Iterable[str]) -> str:
    """The patch without every per-file diff that touches one of ``paths`` (:func:`filter_patch`)."""
    return filter_patch(patch, paths)[0]


def test_patch_paths(test_patch: str) -> list[str]:
    """The repository paths a test patch touches, old and new names, sorted: the paths whose per-file diffs leave the
    patch under test when a hardened suite runs (``patch_exclude``)."""
    out: set[str] = set()
    for section in _split_patch(test_patch or ""):
        if _is_diff(section):
            out |= _diff_paths(section)
    return sorted(out)


test_patch_paths.__test__ = False  # not a pytest test


# ---------------------------------------------------------------------------------------------- patch application


def apply_infrastructure_error(application: Mapping[str, Any] | None) -> str | None:
    """Why a patch that did not apply is an infrastructure error, not a verdict: an application attempt (or the
    restore of a pristine tree between two, or the already-applied check) that errored in the sandbox executor or
    exited with code 124, or an application that stopped at such an error. None when the patch genuinely does not
    apply (or no record says otherwise).

    ``application`` is a run's application record: ``attempts`` (``command``, ``exit_code``, ``error``),
    ``infra_error`` (a strict application that stopped at an executor error or a timeout), ``executor_error``,
    ``steps`` and ``reverse_check``, as executors record them."""
    if not application:
        return None
    if application.get("infra_error"):
        return str(application["infra_error"])[:300]
    if application.get("executor_error"):
        return f"executor error: {application['executor_error']}"[:300]
    checks = list(application.get("attempts") or []) + list(application.get("steps") or [])
    reverse = application.get("reverse_check")
    if reverse:
        checks.append({"command": "git apply --check --reverse", **dict(reverse)})
    for attempt in checks:
        code = attempt.get("exit_code")
        if attempt.get("error"):
            return f"{attempt.get('command')}: executor error: {attempt.get('error')}"[:300]
        if code is not None and int(code) == _APPLY_TIMEOUT_EXIT:
            return f"{attempt.get('command')}: exit {code}"[:300]
    return None


# ---------------------------------------------------------------------------------------------------------- harnesses


class SWEHarness:
    """What the adapter needs from a SWE-bench grader: identities, and the primitives a log verdict is made of."""

    name = "abstract"

    def identity(self, log_parser: str) -> dict[str, str]:
        """``{"parser": digest, "grading_code": digest}`` for a log parser name."""
        raise NotImplementedError

    def apply_cmds_digest(self) -> str:
        raise NotImplementedError

    def status_map(self, section: str, log: str, log_parser: str, instance: Mapping[str, Any]) -> dict[str, str]:
        """test name -> status (PASSED, FAILED, SKIPPED, ERROR, XFAIL) parsed from the marked section of a log."""
        raise NotImplementedError

    def suite_ran(self, log: str) -> bool:
        """Positive evidence that a test runner executed (used when nothing parsed)."""
        raise NotImplementedError

    def logs_eval(self, log: str, log_parser: str, instance: Mapping[str, Any]) -> tuple[dict[str, str], bool]:
        """The official ``get_logs_eval`` reading of a log: (status map, found). It rejects (``({}, False)``) a log with
        a harness failure marker (patch apply failed, reset failed, tests errored, tests timed out), a log without both
        output markers, a log where nothing parsed and no runner evidence shows the suite ran, and a log whose test
        exit code is non-zero while no test failed. The status map is parsed from the log between the markers, or from
        the whole log when that gives nothing."""
        for code in (APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT):
            if code in log:
                return {}, False
        if not (START_TEST_OUTPUT in log and END_TEST_OUTPUT in log):
            return {}, False
        section = log.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
        status_map = self.status_map(section, log, log_parser, instance)
        if not status_map:
            status_map = self.status_map(log, log, log_parser, instance)  # results printed outside the markers
        if not status_map and not self.suite_ran(log):
            return {}, False
        match = _EXIT_CODE_RE.search(log)
        exit_code = int(match.group(1)) if match else None
        if exit_code not in (None, 0) and status_map and not any(v in ("FAILED", "ERROR")
                                                                  for v in status_map.values()):
            return {}, False
        return dict(status_map), True

    def resolve_case(self, case: str, status_map: Mapping[str, str]) -> str | None:
        return case if case in status_map else None

    def resolved(self, status_map: Mapping[str, str], labels: Mapping[str, Sequence[str]], eval_type: str,
                 instance: Mapping[str, Any]) -> bool:
        raise NotImplementedError

    def official_resolved(self, log: str, labels: Mapping[str, Sequence[str]], eval_type: str,
                          instance: Mapping[str, Any]) -> bool | None:
        """The harness's own end-to-end verdict on the log, for a cross-check (None when not available)."""
        return None


class SimpleHarness(SWEHarness):
    """A small standard-library harness for pytest-style ``-rA`` summaries (``PASSED <test>`` lines). It implements
    the pass-and-fail resolution rule (every FAIL_TO_PASS test passes, every PASS_TO_PASS test passes or is skipped)
    and fail-only suites. It has its own identity, so verdicts it produced are never confused with the official
    harness's. Use it for tests, examples and synthetic data; use :class:`OfficialHarness` for SWE-bench itself."""

    name = "simple/1"

    def identity(self, log_parser: str) -> dict[str, str]:
        return {"parser": digest_text(f"agent-ecc/simple-harness/parser/{log_parser}/1"),
                "grading_code": digest_text("agent-ecc/simple-harness/grading/1")}

    def apply_cmds_digest(self) -> str:
        return digest_text("agent-ecc/simple-harness/apply/1")

    def status_map(self, section: str, log: str, log_parser: str, instance: Mapping[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in section.split("\n"):
            parts = line.replace(" - ", " ").split()
            if len(parts) >= 2 and parts[0] in ("PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL"):
                out[parts[1]] = parts[0]
        return out

    def suite_ran(self, log: str) -> bool:
        return bool(re.search(r"\b[1-9]\d* (passed|failed|error)", log))

    def resolved(self, status_map: Mapping[str, str], labels: Mapping[str, Sequence[str]], eval_type: str,
                 instance: Mapping[str, Any]) -> bool:
        if eval_type == "fail_only":
            return not any(status_map.get(t) == "FAILED"
                           for t in list(labels["FAIL_TO_PASS"]) + list(labels["PASS_TO_PASS"]))
        f2p_ok = all(status_map.get(t) in _PASSING for t in labels["FAIL_TO_PASS"])
        p2p_ok = all(status_map.get(t) in _PASSING + ("SKIPPED",) for t in labels["PASS_TO_PASS"])
        return f2p_ok and p2p_ok


# the files whose contents decide a verdict once a log exists, plus the file holding the patch-application list
GRADING_SOURCES = (
    "swebench/harness/constants/__init__.py",
    "swebench/harness/grading.py",
    "swebench/harness/infra_failure.py",
    "swebench/harness/run_evaluation.py",
    "swebench/harness/utils.py",
    "swebench/types.py",
)


class OfficialHarness(SWEHarness):
    """The official ``swebench`` grading code, imported lazily (install ``agent-ecc[swebench]``).

    The grading modules are imported without running ``swebench.harness``'s package initializer when its runner
    dependencies (Docker, Modal) are absent, so parsing works on a machine that cannot execute. ``expected_version``
    and ``expected_grading_digest`` pin the harness: a mismatch raises instead of grading with other code."""

    def __init__(self, *, expected_version: str | None = None, expected_grading_digest: str | None = None) -> None:
        self.expected_version = expected_version
        self.expected_grading_digest = expected_grading_digest
        self._ns: types.SimpleNamespace | None = None
        self._parser_digests: dict[str, str] = {}

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"swebench/{self.ns.version}"

    @property
    def ns(self) -> types.SimpleNamespace:
        if self._ns is None:
            self._ns = _load_official()
            if self.expected_version is not None and self._ns.version != self.expected_version:
                raise RuntimeError(f"swebench {self._ns.version} is installed; {self.expected_version} is required")
            if self.expected_grading_digest is not None and self.grading_code_digest() != \
                    self.expected_grading_digest:
                raise RuntimeError("the installed swebench grading code is not the pinned one")
        return self._ns

    def grading_code_digest(self) -> str:
        root = self.ns.root
        h = hashlib.sha256()
        for rel in GRADING_SOURCES:
            h.update(rel.encode() + b"\0")
            h.update((root / rel).read_bytes())
            h.update(b"\0")
        return f"sha256:{h.hexdigest()}"

    def parser_digest(self, log_parser: str) -> str:
        if log_parser not in self._parser_digests:
            fn = self.ns.parsers[log_parser]
            source = Path(inspect.getsourcefile(fn)).read_bytes()  # type: ignore[arg-type]
            h = hashlib.sha256()
            h.update(f"{log_parser}\0{fn.__module__}.{fn.__name__}\0".encode())
            h.update(source)
            self._parser_digests[log_parser] = f"sha256:{h.hexdigest()}"
        return self._parser_digests[log_parser]

    def identity(self, log_parser: str) -> dict[str, str]:
        return {"parser": self.parser_digest(log_parser), "grading_code": self.grading_code_digest()}

    def apply_cmds_digest(self) -> str:
        return digest_text(json.dumps(list(self.ns.git_apply_cmds)))

    def git_apply_cmds(self) -> tuple[str, ...]:
        return tuple(self.ns.git_apply_cmds)

    def _spec(self, instance: Mapping[str, Any], labels: Mapping[str, Sequence[str]], log_parser: str,
              eval_type: str):
        return self.ns.TestSpec(instance_id=str(instance.get("instance_id") or "instance"),
                                image=str(instance.get("image") or ""), eval_script_list=[],
                                repo=str(instance.get("repo") or ""), version=str(instance.get("version") or ""),
                                FAIL_TO_PASS=list(labels["FAIL_TO_PASS"]), PASS_TO_PASS=list(labels["PASS_TO_PASS"]),
                                log_parser=log_parser, eval_type=eval_type)

    def status_map(self, section: str, log: str, log_parser: str, instance: Mapping[str, Any]) -> dict[str, str]:
        spec = self._spec(instance, {"FAIL_TO_PASS": [], "PASS_TO_PASS": []}, log_parser,
                          str(instance.get("eval_type") or "pass_and_fail"))
        parser = self.ns.parsers[log_parser]
        status = parser(section, spec)
        if not status:
            status = parser(log, spec)  # some runners print results outside the markers (the official fallback)
        return dict(status)

    def suite_ran(self, log: str) -> bool:
        return bool(self.ns.grading.SUITE_RAN.search(log))

    def logs_eval(self, log: str, log_parser: str, instance: Mapping[str, Any]) -> tuple[dict[str, str], bool]:
        spec = self._spec(instance, {"FAIL_TO_PASS": [], "PASS_TO_PASS": []}, log_parser,
                          str(instance.get("eval_type") or "pass_and_fail"))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_output.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(log)
            status_map, found = self.ns.grading.get_logs_eval(spec, path)
        return dict(status_map), bool(found)

    def resolve_case(self, case: str, status_map: Mapping[str, str]) -> str | None:
        resolve = getattr(self.ns.grading, "_resolve_case", None)
        if resolve is None:
            return case if case in status_map else None
        return resolve(case, dict(status_map))

    def resolved(self, status_map: Mapping[str, str], labels: Mapping[str, Sequence[str]], eval_type: str,
                 instance: Mapping[str, Any]) -> bool:
        g = self.ns.grading
        report = g.get_eval_tests_report(dict(status_map), {"FAIL_TO_PASS": list(labels["FAIL_TO_PASS"]),
                                                            "PASS_TO_PASS": list(labels["PASS_TO_PASS"])},
                                         eval_type=self.ns.constants.EvalType(eval_type))
        return g.get_resolution_status(report) == self.ns.constants.ResolvedStatus.FULL.value

    def official_resolved(self, log: str, labels: Mapping[str, Sequence[str]], eval_type: str,
                          instance: Mapping[str, Any]) -> bool | None:
        spec = self._spec(instance, labels, str(instance["log_parser"]), eval_type)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_output.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(log)
            report = self.ns.grading.get_eval_report(
                test_spec=spec, prediction={"instance_id": spec.instance_id, "model_name_or_path": "agent-ecc",
                                            "model_patch": "recorded"},
                test_log_path=path, include_tests_status=False)
        return bool(report[spec.instance_id]["resolved"])


def _load_official() -> types.SimpleNamespace:
    import swebench  # noqa: F401 - the extra agent-ecc[swebench]

    try:
        importlib.import_module("swebench.harness")
    except ImportError:
        # the package initializer imports the Docker and Modal runners, which grading never uses: bind the package
        # without it so the unmodified grading modules still load
        for name in [m for m in sys.modules if m.startswith("swebench.harness")]:
            del sys.modules[name]
        package = types.ModuleType("swebench.harness")
        package.__path__ = [os.path.join(os.path.dirname(swebench.__file__), "harness")]  # type: ignore[attr-defined]
        package.__package__ = "swebench.harness"
        sys.modules["swebench.harness"] = package
    constants = importlib.import_module("swebench.harness.constants")
    grading = importlib.import_module("swebench.harness.grading")
    log_parsers = importlib.import_module("swebench.harness.log_parsers")
    types_mod = importlib.import_module("swebench.types")
    root = Path(swebench.__file__).resolve().parent.parent
    return types.SimpleNamespace(version=getattr(swebench, "__version__", None), root=root, constants=constants,
                                 grading=grading, parsers=log_parsers.PARSER_REGISTRY, TestSpec=types_mod.TestSpec,
                                 git_apply_cmds=_git_apply_cmds(root / "swebench/harness/run_evaluation.py"))


def _git_apply_cmds(path: Path) -> tuple[str, ...]:
    """``GIT_APPLY_CMDS`` from the official runner, read without importing it (it imports Docker at load time)."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GIT_APPLY_CMDS"
                                                for t in node.targets):
            value = ast.literal_eval(node.value)
            if not value or not all(isinstance(v, str) for v in value):
                raise RuntimeError("GIT_APPLY_CMDS is not a list of commands")
            return tuple(value)
    raise RuntimeError(f"GIT_APPLY_CMDS not found in {path}")


# ---------------------------------------------------------------------------------------------------------- verdicts


def _tests(status_map: Mapping[str, str], labels: Mapping[str, Sequence[str]],
           harness: SWEHarness) -> list[TestOutcome]:
    out = []
    for group in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        for name in labels[group]:
            key = harness.resolve_case(name, status_map)
            status = "missing" if key is None else str(status_map[key]).lower()
            out.append(TestOutcome(name=name, status=status, group=group))
    return out


@dataclass(frozen=True)
class ParsedLog:
    """One log read the way the ``logical`` and ``reference_set`` rules read it (:func:`parse_log`): an infrastructure
    reason (no verdict), or the official status map; ``rejected`` when the official reading rejected a complete log
    (the map is then empty, and the run fails)."""

    infra: str | None
    status_map: Mapping[str, str] = field(default_factory=dict)
    rejected: bool = False
    markers: tuple[str, ...] = ()
    error: str | None = None


def parse_log(log: str | None, *, log_parser: str, instance: Mapping[str, Any], harness: SWEHarness) -> ParsedLog:
    """The infrastructure checks and the official status map of one stored log (no execution), in this order: a
    missing log; the harness's timeout marker ending it; no start marker, or no end marker after it; the end marker
    without the test exit-code record after it; the log parser raising. Otherwise the status map of the official
    reading (:meth:`SWEHarness.logs_eval`), or ``rejected`` with an empty map when it rejects the log."""
    if not log:
        return ParsedLog("missing_log")
    if _TRAILING_TIMEOUT_RE.search(log):
        return ParsedLog("timeout")
    start = log.find(START_TEST_OUTPUT)
    end = log.find(END_TEST_OUTPUT, start + len(START_TEST_OUTPUT)) if start >= 0 else -1
    if start < 0 or end < 0:
        return ParsedLog("missing_output_markers")  # empty or truncated: the script did not reach its end marker
    if not _EXIT_CODE_RE.search(log, end):
        return ParsedLog("no_exit_code_record")  # the end marker without the record that follows it: truncated
    try:
        status_map, found = harness.logs_eval(log, log_parser, instance)
    except Exception as error:  # noqa: BLE001 - a parser that raises decides nothing
        return ParsedLog(f"parser_error:{type(error).__name__}", error=f"{type(error).__name__}: {error}"[:300])
    if not found:
        markers = tuple(code for code in (APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT) if code in log)
        return ParsedLog(None, {}, rejected=True, markers=markers)
    return ParsedLog(None, dict(status_map))


def _reference_status(status: Any) -> str:
    value = str(status).upper()
    return _STATUS_ALIASES.get(value, value)


def _reference_problem(labels: Mapping[str, Sequence[str]]) -> str | None:
    """Why labels are not a reference set G (FAIL_TO_PASS empty, PASS_TO_PASS = G, non-empty), or None."""
    if list(labels.get("FAIL_TO_PASS") or []):
        return "fail_to_pass_not_empty"
    if not list(labels.get("PASS_TO_PASS") or []):
        return "empty_reference_set"
    return None


def _reference_verdict(reference: Sequence[str], parsed: ParsedLog) -> PartVerdict:
    tests, ok = [], True
    for name in reference:
        status = parsed.status_map.get(name)  # ids compared exactly, without prefix matching
        if status is None:
            tests.append(TestOutcome(name=name, status="missing", group="REFERENCE_SET"))
            ok = False
            continue
        normal = _reference_status(status)
        tests.append(TestOutcome(name=name, status=normal.lower(), group="REFERENCE_SET"))
        if normal not in REFERENCE_PASSING:
            ok = False
    if ok:
        return PartVerdict.passed(tests, "resolved")
    return PartVerdict.failed("log_rejected" if parsed.rejected else "tests_failed", tests)


def classify_log(log: str | None, *, labels: Mapping[str, Sequence[str]], log_parser: str, eval_type: str,
                 instance: Mapping[str, Any], harness: SWEHarness,
                 rule: str | Mapping[str, Any] = LOGICAL) -> PartVerdict:
    """The verdict of one raw test log under labels and a verdict rule, ``logical`` or ``reference_set`` (see the
    module documentation): an infrastructure error found by :func:`parse_log` is undeterminable; a log the official
    reading rejects fails; otherwise ``logical`` is the official report on the status map (every FAIL_TO_PASS test
    passes, every PASS_TO_PASS test passes or is skipped; a listed test without a status fails) and ``reference_set``
    passes if and only if every test of G (the labels' PASS_TO_PASS) is PASSED or SKIPPED. ``rule`` is a rule name
    or a :func:`verdict_rule` mapping."""
    if not isinstance(rule, str):
        rule = _validated_rule(rule)["rule"]
    if rule not in (LOGICAL, REFERENCE_SET):
        raise ValueError(f"classify_log reads the logical and reference_set rules, not {rule!r} (official_verdict)")
    if rule == REFERENCE_SET:
        problem = _reference_problem(labels)
        if problem is not None:
            return PartVerdict.undeterminable(f"invalid_reference_set:{problem}", final=True)
    parsed = parse_log(log, log_parser=log_parser, instance={**instance, "eval_type": eval_type}, harness=harness)
    if parsed.infra is not None:
        return PartVerdict.undeterminable(parsed.infra)
    if rule == REFERENCE_SET:
        return _reference_verdict([str(t) for t in labels["PASS_TO_PASS"]], parsed)
    tests = _tests(parsed.status_map, labels, harness)
    if parsed.rejected:
        return PartVerdict.failed("log_rejected", tests)
    try:
        resolved = harness.resolved(parsed.status_map, labels, eval_type, instance)
    except Exception as error:  # noqa: BLE001 - the official report raising over the map decides nothing either
        return PartVerdict.undeterminable(f"grading_error:{type(error).__name__}")
    return PartVerdict.passed(tests, "resolved") if resolved else PartVerdict.failed("tests_failed", tests)


def gold_run_validity(parsed: ParsedLog) -> str | None:
    """Why one hardened run of the gold patch is not a valid gold run (an infrastructure error, a rejected log, an
    empty status map, a test of its map that does not pass), None when it is."""
    if parsed.infra is not None:
        return f"infrastructure:{parsed.infra}"
    if parsed.rejected:
        return "log_rejected"
    if not parsed.status_map:
        return "empty_status_map"
    failing = [t for t, status in parsed.status_map.items() if _reference_status(status) not in REFERENCE_PASSING]
    if failing:
        return f"gold_tests_not_passing:{len(failing)}"
    return None


def reference_set(runs: Sequence[ParsedLog]) -> dict:
    """The reference set G of one instance from its two hardened gold runs: the tests with a passing status in both
    runs, ids compared exactly. Invalid (``reference`` None) when there are not exactly two runs, either run is not a
    valid gold run (:func:`gold_run_validity`), the runs disagree on the pass or fail status of a test they share, or
    G is empty. Returns ``{"valid", "reference", "runs", "disagreements", "reason"}``."""
    runs = list(runs)
    out: dict[str, Any] = {"valid": False, "reference": None, "runs": [], "disagreements": [], "reason": None}
    for parsed in runs:
        out["runs"].append({"invalid": gold_run_validity(parsed), "tests": len(parsed.status_map)})
    if len(runs) != 2:
        out["reason"] = f"expected 2 gold runs, got {len(runs)}"
        return out
    a, b = (dict(r.status_map) for r in runs)

    def passing(status):
        return _reference_status(status) in REFERENCE_PASSING

    out["disagreements"] = sorted(t for t in set(a) & set(b) if passing(a[t]) != passing(b[t]))
    invalid = [r["invalid"] for r in out["runs"] if r["invalid"]]
    if invalid:
        out["reason"] = "invalid_run:" + ",".join(invalid)
        return out
    if out["disagreements"]:
        out["reason"] = f"runs_disagree:{len(out['disagreements'])}"
        return out
    reference = sorted(t for t in set(a) & set(b) if passing(a[t]) and passing(b[t]))
    if not reference:
        out["reason"] = "empty_reference_set"
        return out
    out.update(valid=True, reference=reference)
    return out


def _legacy_classify(log: str | None, *, labels: Mapping[str, Sequence[str]], log_parser: str, eval_type: str,
                     instance: Mapping[str, Any], harness: SWEHarness) -> PartVerdict:
    """The reading of a log the official rule reports its reasons and per-test outcomes with (an earlier logical
    classification, a listed test without a status failing, kept unchanged so that verdicts reproduced under the
    official rule stay exactly as they were)."""
    if not log:
        return PartVerdict.undeterminable("missing_log")
    if APPLY_PATCH_FAIL in log:
        return PartVerdict.failed("patch_did_not_apply")
    if TESTS_TIMEOUT in log or _TIMEOUT_RE.search(log):
        return PartVerdict.undeterminable("timeout")
    if RESET_FAILED in log:
        return PartVerdict.undeterminable("reset_failed")
    if TESTS_ERROR in log:
        return PartVerdict.undeterminable("tests_errored")
    if not (START_TEST_OUTPUT in log and END_TEST_OUTPUT in log):
        return PartVerdict.undeterminable("no_output_markers")
    section = log.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    try:
        status_map = harness.status_map(section, log, log_parser, instance)
    except Exception as error:  # noqa: BLE001 - a parser exception is not a verdict
        return PartVerdict.undeterminable(f"parser_error:{type(error).__name__}")
    if not status_map and not harness.suite_ran(log):
        return PartVerdict.undeterminable("suite_did_not_run")
    match = _EXIT_CODE_RE.search(log)
    exit_code = int(match.group(1)) if match else None
    if exit_code not in (None, 0) and status_map and not any(v in ("FAILED", "ERROR") for v in status_map.values()):
        return PartVerdict.undeterminable("exit_code_contradicts_log")
    tests = _tests(status_map, labels, harness)
    try:
        resolved = harness.resolved(status_map, labels, eval_type, instance)
    except Exception as error:  # noqa: BLE001
        return PartVerdict.undeterminable(f"grading_error:{type(error).__name__}")
    return PartVerdict.passed(tests, "resolved") if resolved else PartVerdict.failed("tests_failed", tests)


def official_verdict(log: str | None, *, labels: Mapping[str, Sequence[str]], log_parser: str, eval_type: str,
                     instance: Mapping[str, Any], harness: SWEHarness) -> PartVerdict:
    """The verdict of one raw test log as the official harness reports it (``verdict="official"``).

    The harness's own end-to-end report decides: a log it resolves passes, every other log fails, including logs
    that are infrastructure errors under the logical rule (a timeout, missing output markers, a log without the
    exit-code record). Only a missing log or a grading exception has no verdict. The fail's reason names what left
    the log unresolved (no output markers, a suite that never ran, a test exit code that contradicts the log, error
    or timeout markers, a listed test without a status), as an earlier reading of the log reports it
    (:func:`_legacy_classify`, kept unchanged so that verdicts reproduced under this rule stay as they were). A
    harness without an end-to-end report (:class:`SimpleHarness`) is judged by that reading, every log it does not
    resolve counted as a fail."""
    if not log:
        return PartVerdict.undeterminable("missing_log")
    logical = _legacy_classify(log, labels=labels, log_parser=log_parser, eval_type=eval_type, instance=instance,
                               harness=harness)
    try:
        official = harness.official_resolved(log, labels, eval_type, {**instance, "log_parser": log_parser})
    except Exception as error:  # noqa: BLE001 - a grading exception is not a verdict
        return PartVerdict.undeterminable(f"grading_error:{type(error).__name__}")
    resolved = logical.outcome is Outcome.PASS if official is None else bool(official)
    if resolved:
        return PartVerdict.passed(logical.tests, "resolved")
    reason = logical.reason if logical.outcome is not Outcome.PASS else None
    return PartVerdict.failed(reason or "tests_failed", logical.tests)


# ---------------------------------------------------------------------------------------------------- eval scripts

_DIFF_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.M)
_HEREDOC_RE = re.compile(r"^git apply (?:-v |--verbose )?- <<'(?P<tag>[A-Za-z0-9_]+)'$")


def touched_files(patch: str) -> list[str]:
    """Every path a unified git diff touches (both sides of a rename)."""
    out: list[str] = []
    for a, b in _DIFF_RE.findall(patch):
        for p in (a, b):
            if p not in out:
                out.append(p)
    return out


def substitute_test_patch(eval_script: str, test_patch: str, *, original_test_patch: str | None = None) -> str:
    """The eval script with the test patch in its heredoc replaced by ``test_patch``.

    Only the heredoc body changes. The script's test-file resets and test directives name the files of the original
    test patch, so the substitution is refused unless ``test_patch`` touches exactly the same files; when
    ``original_test_patch`` is given, the heredoc must hold it. Provide a complete eval script otherwise."""
    lines = eval_script.split("\n")
    starts = [i for i, line in enumerate(lines) if _HEREDOC_RE.match(line)]
    if len(starts) != 1:
        raise ValueError(f"expected one test-patch heredoc in the eval script, found {len(starts)}")
    start = starts[0]
    tag = _HEREDOC_RE.match(lines[start]).group("tag")  # type: ignore[union-attr]
    try:
        end = next(i for i in range(start + 1, len(lines)) if lines[i] == tag)
    except StopIteration:
        raise ValueError("the test-patch heredoc is not terminated") from None
    body = "\n".join(lines[start + 1:end])
    if original_test_patch is not None and body.strip("\n") != original_test_patch.strip("\n"):
        raise ValueError("the eval script's heredoc does not hold the original test patch")
    old_files, new_files = set(touched_files(body)), set(touched_files(test_patch))
    if old_files != new_files:
        raise ValueError(f"the new test patch touches other files than the original ({sorted(new_files ^ old_files)}); "
                         "the script's resets and test directives would not match: provide an eval script")
    if tag in test_patch.split("\n"):
        raise ValueError("the new test patch contains the heredoc delimiter")
    new_body = test_patch[:-1] if test_patch.endswith("\n") else test_patch
    return "\n".join(lines[:start + 1] + new_body.split("\n") + lines[end:])


# ---------------------------------------------------------------------------------------------------------- adapter


class SWEBenchAdapter:
    """The SWE-bench family: build records and versions, parse runs, and request re-executions.

    ``harness`` grades logs (default: :class:`OfficialHarness`, loaded on first use). ``verdict`` is the verdict rule
    of the checks this adapter builds from dataset rows (:func:`verdict_rule`): ``"logical"`` (default) takes the
    official report on a complete log as the verdict and keeps infrastructure errors (timeouts, truncated logs,
    parser exceptions, evaluator failures, application attempts that errored in the executor or exited 124) out of
    verdicts (:func:`classify_log`); ``"official"`` takes the official harness's own report on every log, as
    SWE-bench's evaluation does (:func:`official_verdict`): every log it does not resolve fails, and a run that timed
    out fails. A hardened suite graded against a reference set G is built by :meth:`reference_check` (rule
    ``"reference_set"``). Under every rule a listed test without a status fails; there is no option to make it
    undeterminable (the options ``missing_status`` and ``cross_check`` of an earlier version are refused).

    The rule is part of each check's grading identity, and :meth:`parse` grades a run by the rule of the check it is
    given, whatever the adapter's own options: a verdict is never reused, or re-read, under another rule. (A check
    built without a rule, by hand, is graded by the adapter's.)"""

    family = FAMILY

    def __init__(self, *, harness: SWEHarness | None = None, verdict: str | Mapping[str, Any] = LOGICAL,
                 **superseded: Any) -> None:
        _refuse_superseded(superseded)
        if not isinstance(verdict, str):
            verdict = _validated_rule(verdict)["rule"]
        if verdict == REFERENCE_SET:
            raise ValueError("a reference set is a property of a hardened suite: build it with reference_check (or a "
                             "hardened row's reference_set in conjunction_version)")
        self.rule = verdict_rule(verdict)
        self._harness = harness
        self.verdict = verdict

    @property
    def harness(self) -> SWEHarness:
        if self._harness is None:
            self._harness = OfficialHarness()
        return self._harness

    def rule_of(self, check: Check) -> dict:
        """The verdict rule a check is graded by: its own (``grading["verdict_rule"]``), else the adapter's."""
        rule = check.grading.get("verdict_rule")
        return dict(self.rule) if rule is None else _validated_rule(rule)

    # ------------------------------------------------------------------------------------------- grader versions

    def check(self, *, store: ContentStore, instance_id: str, repo: str, version: str, image: str,
              eval_script: str, log_parser: str, eval_type: str, fail_to_pass: Any, pass_to_pass: Any,
              name: str = CHECK_TESTS, rule: str | Mapping[str, Any] | None = None,
              patch_exclude: Iterable[str] | None = None) -> Check:
        """A check, its objects (the eval script, the labels, the excluded paths) written to ``store``. ``rule`` is
        its verdict rule, a name or a :func:`verdict_rule` mapping (default: the adapter's); ``patch_exclude`` names
        repository paths whose diffs are removed from the patch before it is applied (:func:`exclude_paths`; a
        collection of paths, never a string). Under ``reference_set`` the labels must be a reference set:
        FAIL_TO_PASS empty, G as a non-empty PASS_TO_PASS."""
        labels = canonical_labels(fail_to_pass, pass_to_pass)
        verdict = _validated_rule(rule) if rule is not None else dict(self.rule)
        if verdict["rule"] == REFERENCE_SET and _reference_problem(labels) is not None:
            raise ValueError(f"check {name!r} of {instance_id!r}: a reference_set check holds G as a non-empty "
                             f"PASS_TO_PASS with an empty FAIL_TO_PASS ({_reference_problem(labels)})")
        identity = self.harness.identity(log_parser)
        execution = {"image": str(image), "eval_script": put_text(store, eval_script),
                     "apply_cmds": self.harness.apply_cmds_digest()}
        excluded = sorted(set(_path_list(patch_exclude, "patch_exclude")))
        if excluded:
            execution["patch_exclude"] = put_json(store, excluded)
        return Check(
            name=name,
            execution=execution,
            grading={"labels": put_json(store, labels), "parser": identity["parser"],
                     "grading_code": identity["grading_code"], "eval_type": str(eval_type),
                     "verdict_rule": verdict},
            params={"instance_id": str(instance_id), "repo": str(repo), "version": str(version),
                    "log_parser": str(log_parser), "harness": self.harness.name},
        )

    def reference_check(self, *, store: ContentStore, instance_id: str, repo: str, version: str, image: str,
                        eval_script: str, log_parser: str, eval_type: str, reference: Iterable[str],
                        patch_exclude: Iterable[str], name: str = CHECK_HARDENED) -> Check:
        """A hardened suite graded against its reference set G (rule ``reference_set``): G held as the labels'
        PASS_TO_PASS with FAIL_TO_PASS empty, run on the patch without its per-file diffs of the ``patch_exclude``
        paths. ``patch_exclude`` is required: SWE-ABS removes the diffs of the hardened test patch's paths
        (:func:`test_patch_paths` of it) before the suite runs; pass ``()`` only for a suite that runs the whole
        patch."""
        return self.check(store=store, instance_id=instance_id, repo=repo, version=version, image=image,
                          eval_script=eval_script, log_parser=log_parser, eval_type=eval_type, fail_to_pass=[],
                          pass_to_pass=sorted({str(t) for t in reference}), name=name,
                          rule=verdict_rule(REFERENCE_SET), patch_exclude=patch_exclude)

    def check_from_row(self, row: Mapping[str, Any], store: ContentStore, *, name: str = CHECK_TESTS,
                       image: str | None = None, eval_script: str | None = None) -> Check:
        """A check from a dataset row with ``instance_id, repo, version, image, eval_script, log_parser, eval_type,
        FAIL_TO_PASS, PASS_TO_PASS`` (``image`` and ``eval_script`` may be given separately)."""
        image = image or row.get("image")
        eval_script = eval_script if eval_script is not None else row.get("eval_script")
        missing = [k for k, v in (("image", image), ("eval_script", eval_script)) if not v]
        missing += [k for k in ("instance_id", "repo", "version", "log_parser", "eval_type", "FAIL_TO_PASS",
                                "PASS_TO_PASS") if row.get(k) in (None, "")]
        if missing:
            raise ValueError(f"row {row.get('instance_id')!r} is missing {missing}")
        return self.check(store=store, instance_id=row["instance_id"], repo=row["repo"], version=row["version"],
                          image=image, eval_script=eval_script, log_parser=row["log_parser"],  # type: ignore[arg-type]
                          eval_type=row["eval_type"], fail_to_pass=row["FAIL_TO_PASS"],
                          pass_to_pass=row["PASS_TO_PASS"], name=name)

    def version_from_rows(self, rows: Iterable[Mapping[str, Any]], store: ContentStore, *, name: str,
                          governed: Iterable[str] | None = None,
                          image_of: Callable[[Mapping[str, Any]], str] | None = None,
                          meta: Mapping[str, Any] | None = None) -> GraderVersion:
        """A grader version with one check per row (world-independent: the image is in the execution identity)."""
        entries: dict[str, TestSpec] = {}
        for row in rows:
            image = image_of(row) if image_of is not None else None
            entries[str(row["instance_id"])] = self.check_from_row(row, store, image=image)
        return GraderVersion(name=name, entries={t: {ANY_WORLD: s} for t, s in entries.items()},
                             governed=frozenset(governed) if governed is not None else None, family=FAMILY,
                             meta=meta or {})

    def conjunction_version(self, base_rows: Iterable[Mapping[str, Any]],
                            hardened_rows: Iterable[Mapping[str, Any]], store: ContentStore, *, name: str,
                            hardened_eval_script: Callable[[Mapping[str, Any], Mapping[str, Any]], str] | None = None,
                            hardened_rule: str | Mapping[str, Any] | None = None,
                            governed: Iterable[str] | None = None,
                            image_of: Callable[[Mapping[str, Any]], str] | None = None,
                            meta: Mapping[str, Any] | None = None) -> GraderVersion:
        """A version whose spec per task is ``AllOf(original suite, hardened suite)`` (SWE-ABS style).

        A hardened row with a ``reference_set`` (G, the tests passing in both gold runs, :func:`reference_set`) is
        graded against G under ``reference_set`` (:meth:`reference_check`): it needs its own ``eval_script`` (or
        ``hardened_eval_script``), and runs on the patch without its per-file diffs of the row's ``patch_exclude``
        paths, by default every path of its ``test_patch`` (:func:`test_patch_paths`). Any other hardened row carries
        its own FAIL_TO_PASS / PASS_TO_PASS and either an ``eval_script`` or a ``test_patch`` (with the original in
        ``original_test_patch`` or the base row's ``test_patch``), substituted into the base row's eval script by
        :func:`substitute_test_patch`; ``hardened_eval_script(base_row, hardened_row)`` overrides both; it runs on the
        patch without its diffs of the row's ``patch_exclude`` paths (if any) and is graded by ``hardened_rule`` (a
        rule name or a :func:`verdict_rule` mapping; default: the adapter's rule). Either suite runs on the base row's
        image with the same patch-application commands. Tasks without a hardened row keep the original suite alone."""
        hardened = {str(r["instance_id"]): r for r in hardened_rows}
        entries: dict[str, Mapping[str, TestSpec]] = {}
        for row in base_rows:
            task = str(row["instance_id"])
            image = image_of(row) if image_of is not None else row.get("image")
            base = self.check_from_row(row, store, image=image, name=CHECK_TESTS)
            h = hardened.get(task)
            if h is None:
                entries[task] = {ANY_WORLD: base}
                continue
            if h.get("reference_set") is not None:
                script = hardened_eval_script(row, h) if hardened_eval_script is not None else h.get("eval_script")
                if not script:
                    raise ValueError(f"hardened row {task!r} has a reference set but no eval_script")
                exclude = h.get("patch_exclude")
                if exclude is None:
                    if not h.get("test_patch"):
                        raise ValueError(f"hardened row {task!r} names neither patch_exclude nor its test_patch")
                    exclude = test_patch_paths(h["test_patch"])
                hard = self.reference_check(store=store, instance_id=task, repo=row["repo"], version=row["version"],
                                            image=str(image), eval_script=script, log_parser=row["log_parser"],
                                            eval_type=row["eval_type"], reference=h["reference_set"],
                                            patch_exclude=exclude)
                entries[task] = {ANY_WORLD: AllOf(name="all_of", parts=(base, hard))}
                continue
            if hardened_eval_script is not None:
                script = hardened_eval_script(row, h)
            elif h.get("eval_script"):
                script = h["eval_script"]
            elif h.get("test_patch"):
                original = h.get("original_test_patch") or row.get("test_patch")
                script = substitute_test_patch(row["eval_script"], h["test_patch"], original_test_patch=original)
            else:
                raise ValueError(f"hardened row {task!r} has neither an eval_script nor a test_patch")
            hard = self.check(store=store, instance_id=task, repo=row["repo"], version=row["version"],
                              image=str(image), eval_script=script, log_parser=row["log_parser"],
                              eval_type=row["eval_type"], fail_to_pass=h["FAIL_TO_PASS"],
                              pass_to_pass=h["PASS_TO_PASS"], name=CHECK_HARDENED, rule=hardened_rule,
                              patch_exclude=h.get("patch_exclude"))
            entries[task] = {ANY_WORLD: AllOf(name="all_of", parts=(base, hard))}
        return GraderVersion(name=name, entries=entries,
                             governed=frozenset(governed) if governed is not None else None, family=FAMILY,
                             meta=meta or {})

    # ------------------------------------------------------------------------------------------- records

    def record(self, *, store: ContentStore, task_id: str, trajectory_id: str, image: str, patch: str | None,
               runs: Sequence[TestRun] = (), facts: Mapping[str, Any] | None = None,
               meta: Mapping[str, Any] | None = None) -> ExecutionRecord:
        """An execution record: the patch stored as evidence (or the fact that it was empty) and the runs made when
        the trajectory was graded (see :meth:`run`). The world version is the image the agent worked in."""
        facts = dict(facts or {})
        artifacts = []
        if patch is None or not patch.strip():
            facts["patch_empty"] = True
        else:
            data = patch.encode("utf-8")
            artifacts.append(EvidenceRef(role=ROLE_PATCH, digest=store.put(data), size=len(data),
                                         media_type="text/x-diff"))
        return ExecutionRecord(family=FAMILY, task_id=str(task_id), trajectory_id=str(trajectory_id),
                               world_version=str(image), artifacts=tuple(artifacts), runs=tuple(runs), facts=facts,
                               meta=meta or {})

    def run(self, check: Check, *, store: ContentStore, log: str | None, applied: bool | None = True,
            timed_out: bool = False, error: str | None = None,
            application: Mapping[str, Any] | None = None) -> TestRun:
        """The run of ``check`` a producer made: its raw log stored as evidence. ``applied=False`` records that the
        agent's patch did not apply (a logical fail, unless ``application``, the record of its application attempts,
        shows an attempt that errored in the executor or exited 124: :func:`apply_infrastructure_error`); ``error``
        records an infrastructure failure (no verdict)."""
        facts: dict[str, Any] = {"applied": applied, "timed_out": bool(timed_out)}
        if application is not None:
            facts["application"] = dict(application)
        outputs = []
        if log is not None:
            data = log.encode("utf-8")
            outputs.append(EvidenceRef(role=ROLE_TEST_LOG, digest=store.put(data), size=len(data),
                                       media_type="text/plain"))
            facts["has_output_markers"] = START_TEST_OUTPUT in log and END_TEST_OUTPUT in log
            match = _EXIT_CODE_RE.search(log)
            facts["test_exit_code"] = int(match.group(1)) if match else None
        if error is not None:
            facts["error"] = str(error)[:500]
            return TestRun(check=check.name, execution=check.execution, status=RUN_ERROR, outputs=tuple(outputs),
                           facts=facts)
        return TestRun(check=check.name, execution=check.execution, status=RUN_OK, outputs=tuple(outputs),
                       facts=facts)

    # ------------------------------------------------------------------------------------------- scorer protocol

    def parse(self, check: Check, run: TestRun, store: ContentStore) -> PartVerdict:
        try:
            rule = self.rule_of(check)
        except ValueError as error:
            return PartVerdict.undeterminable(f"unknown_verdict_rule:{error}"[:300], final=True)
        official = rule["rule"] == OFFICIAL
        if run.status == RUN_ERROR:
            return PartVerdict.undeterminable(f"infra_error:{run.facts.get('error') or 'unknown'}")
        if run.facts.get("applied") is False:
            # a patch that genuinely does not apply fails; an application stopped by the executor is no verdict (the
            # official rule reproduces a report that counts every application failure as a fail)
            why = None if official else apply_infrastructure_error(run.facts.get("application"))
            if why is not None:
                return PartVerdict.undeterminable(f"apply_infra_error:{why}"[:300])
            return PartVerdict.failed("patch_did_not_apply")
        if run.facts.get("timed_out"):
            # the official runner reports a run that timed out as not resolved; the logical rules give it no verdict
            return PartVerdict.failed("timeout") if official else PartVerdict.undeterminable("timeout")
        log_parser = str(check.params["log_parser"])
        expected = {k: check.grading.get(k) for k in ("parser", "grading_code")}
        if self.harness.identity(log_parser) != expected:
            # the target's grader is not the importable one: grading here would apply other code
            return PartVerdict.undeterminable("grader_unavailable", final=True)
        ref = run.output(ROLE_TEST_LOG)
        if ref is None:
            return PartVerdict.undeterminable("missing_log")
        log = store.get(ref.digest).decode("utf-8", errors="replace")  # as the runner read it
        labels = get_json(store, str(check.grading["labels"]))
        instance = {"instance_id": check.params.get("instance_id"), "repo": check.params.get("repo"),
                    "version": check.params.get("version"), "image": check.execution.get("image"),
                    "eval_type": check.grading.get("eval_type")}
        if official:
            return official_verdict(log, labels=labels, log_parser=log_parser,
                                    eval_type=str(check.grading["eval_type"]), instance=instance, harness=self.harness)
        return classify_log(log, labels=labels, log_parser=log_parser, eval_type=str(check.grading["eval_type"]),
                            instance=instance, harness=self.harness, rule=rule["rule"])

    def reexec_request(self, record: ExecutionRecord, check: Check, store: ContentStore) -> ExecRequest | None:
        patch = record.artifact(ROLE_PATCH)
        script = check.execution.get("eval_script")
        if patch is None or not store.contains(patch.digest) or not script or not store.contains(str(script)):
            return None
        patch_digest = patch.digest
        params = {"instance_id": check.params.get("instance_id"), "repo": check.params.get("repo"),
                  "version": check.params.get("version"), "image": check.execution.get("image"),
                  "log_parser": check.params.get("log_parser"), "eval_type": check.grading.get("eval_type")}
        excluded = check.execution.get("patch_exclude")
        if excluded:
            # the suite runs the patch without its diffs of the excluded paths: the backend is sent that patch (an
            # empty one when nothing is left: the tests then run on the unmodified tree)
            if not store.contains(str(excluded)):
                return None
            filtered = exclude_paths(get_text(store, patch.digest), get_json(store, str(excluded)))
            patch_digest = put_text(store, filtered)
            params["patch_exclude"] = str(excluded)
        return ExecRequest(family=FAMILY, kind=REQUEST_KIND, task_id=record.task_id, execution=check.execution,
                           inputs={"patch": patch_digest, "eval_script": str(script)}, params=params)

    def decide_without_execution(self, record: ExecutionRecord, check: Check) -> PartVerdict | None:
        if record.facts.get("patch_empty"):
            return PartVerdict.failed("no_agent_patch")
        return None

    def carry_over(self, record: ExecutionRecord, check: Check) -> PartVerdict | None:
        """A verdict a recorded run gives ``check`` without executing it: a patch that did not apply to the same image,
        with the same application commands and the same patch filter, does not apply under any eval script, so the
        check fails whatever its eval script and grading inputs (another suite's run, or this suite's run graded under
        other labels). The verdict's ``source`` is that run. A recorded application that the check's rule reads as an
        infrastructure error carries nothing over."""
        try:
            official = self.rule_of(check)["rule"] == OFFICIAL
        except ValueError:
            return None
        want = {k: v for k, v in check.execution.items() if k != "eval_script"}
        for run in record.runs:
            if run.status != RUN_OK or run.facts.get("applied") is not False:
                continue
            if {k: v for k, v in run.execution.items() if k != "eval_script"} != want:
                continue
            if not official and apply_infrastructure_error(run.facts.get("application")) is not None:
                continue
            return PartVerdict(Outcome.FAIL, "patch_did_not_apply", source=run)
        return None
