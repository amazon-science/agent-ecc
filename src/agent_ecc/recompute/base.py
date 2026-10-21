# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-execution requests, results, and the backend protocol.

A backend executes a check's tests on a recorded artifact and returns **raw outputs** (a test log, per-test outcomes),
never a verdict: verdicts are always derived from outputs by the family adapter, the same code that re-parses recorded
outputs. That is what lets a replay (:class:`RecordedBackend`) reproduce decisions exactly from recorded raw outputs.

Requests are content-addressed. Two members with the same artifact under the same target check produce the same
request id, so a batch executes it once and shares the result.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Protocol, runtime_checkable

from ..canonical import FrozenDict, digest_json, freeze, thaw
from ..records import RUN_ERROR, RUN_OK, Cost, EvidenceRef, _tuple_of
from ..store import ContentStore

__all__ = [
    "Backend",
    "CallableBackend",
    "ExecRequest",
    "ExecResult",
    "RecordedBackend",
    "RecordingBackend",
    "infra_result",
]

SCHEMA_REQUEST = "agent-ecc/exec-request/1"
SCHEMA_RESULT = "agent-ecc/exec-result/1"


@dataclass(frozen=True)
class ExecRequest:
    """Execute one check's tests on one artifact.

    ``execution`` is the target check's execution identity; ``inputs`` maps a role (patch, eval_script, end_state, ...)
    to the digest of an object the backend needs, resolvable in the content store; ``params`` holds small parameters
    (instance fields, names). The request id covers all of them, so equal requests are interchangeable.
    """

    family: str
    kind: str
    task_id: str
    execution: Mapping[str, Any]
    inputs: Mapping[str, str]
    params: Mapping[str, Any] = field(default_factory=FrozenDict)
    schema: str = SCHEMA_REQUEST

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution", freeze(self.execution))
        object.__setattr__(self, "inputs", freeze(self.inputs))
        object.__setattr__(self, "params", freeze(self.params))

    def body(self) -> dict:
        return {"schema": self.schema, "family": self.family, "kind": self.kind, "task_id": self.task_id,
                "execution": thaw(self.execution), "inputs": thaw(self.inputs), "params": thaw(self.params)}

    @cached_property
    def request_id(self) -> str:
        return digest_json(self.body())

    def to_dict(self) -> dict:
        return {**self.body(), "request_id": self.request_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ExecRequest:
        request = cls(family=d["family"], kind=d["kind"], task_id=d["task_id"], execution=d.get("execution") or {},
                      inputs=d.get("inputs") or {}, params=d.get("params") or {},
                      schema=d.get("schema", SCHEMA_REQUEST))
        if d.get("request_id") is not None and d["request_id"] != request.request_id:
            raise ValueError(f"request id {d['request_id']} does not match its content")
        return request


@dataclass(frozen=True)
class ExecResult:
    """The raw result of executing one request. ``status`` is ``ok`` (outputs captured; parsing decides whether they
    hold a logical verdict) or ``error`` (infrastructure: no usable outputs). Outputs are stored in the content store
    and referenced by digest."""

    request_id: str
    status: str
    outputs: tuple[EvidenceRef, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=FrozenDict)
    cost: Cost = field(default_factory=Cost)
    error: str | None = None
    schema: str = SCHEMA_RESULT

    def __post_init__(self) -> None:
        if self.status not in (RUN_OK, RUN_ERROR):
            raise ValueError(f"unknown result status {self.status!r}")
        object.__setattr__(self, "outputs", _tuple_of(EvidenceRef, self.outputs))
        object.__setattr__(self, "facts", freeze(self.facts))
        if not isinstance(self.cost, Cost):
            object.__setattr__(self, "cost", Cost.from_dict(self.cost))

    def to_dict(self) -> dict:
        return {"schema": self.schema, "request_id": self.request_id, "status": self.status,
                "outputs": [o.to_dict() for o in self.outputs], "facts": thaw(self.facts),
                "cost": self.cost.to_dict(), "error": self.error}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ExecResult:
        return cls(request_id=d["request_id"], status=d["status"], outputs=d.get("outputs") or (),
                   facts=d.get("facts") or {}, cost=Cost.from_dict(d.get("cost") or {}), error=d.get("error"),
                   schema=d.get("schema", SCHEMA_RESULT))


def infra_result(request: ExecRequest, error: str, *, seconds: float = 0.0) -> ExecResult:
    """An infrastructure failure: no outputs, never a verdict."""
    return ExecResult(request_id=request.request_id, status=RUN_ERROR, error=str(error)[:500],
                      cost=Cost(seconds=seconds))


@runtime_checkable
class Backend(Protocol):
    """Executes requests. Must return one result per request, in order, and must not raise for a failed execution
    (return an ``error`` result instead). Inputs are read from, and outputs written to, ``store``."""

    def execute(self, requests: Sequence[ExecRequest], store: ContentStore) -> list[ExecResult]: ...


class CallableBackend:
    """A backend from a function ``fn(request, store) -> ExecResult``; exceptions become infrastructure errors."""

    def __init__(self, fn: Callable[[ExecRequest, ContentStore], ExecResult]) -> None:
        self.fn = fn
        self.calls = 0

    def execute(self, requests: Sequence[ExecRequest], store: ContentStore) -> list[ExecResult]:
        out = []
        for request in requests:
            self.calls += 1
            try:
                result = self.fn(request, store)
                if result.request_id != request.request_id:
                    raise ValueError("result answers another request")
            except Exception as error:  # noqa: BLE001 - a failed execution is not a verdict
                result = infra_result(request, f"{type(error).__name__}: {error}")
            out.append(result)
        return out


class RecordedBackend:
    """Replays recorded raw results by request id (for replay and audits). A request with no recorded result is an
    infrastructure error ``not_recorded``: a replay never invents an execution."""

    def __init__(self, results: Iterable[ExecResult] | Mapping[str, ExecResult] = ()) -> None:
        if isinstance(results, Mapping):
            self.results = dict(results)
        else:
            self.results = {r.request_id: r for r in results}
        self.served: list[str] = []
        self.missing: list[str] = []

    @classmethod
    def from_jsonl(cls, path: str | os.PathLike) -> RecordedBackend:
        results = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(ExecResult.from_dict(json.loads(line)))
        return cls(results)

    def execute(self, requests: Sequence[ExecRequest], store: ContentStore) -> list[ExecResult]:
        out = []
        for request in requests:
            result = self.results.get(request.request_id)
            if result is None:
                self.missing.append(request.request_id)
                out.append(infra_result(request, "not_recorded"))
                continue
            for ref in result.outputs:
                if not store.contains(ref.digest):
                    self.missing.append(request.request_id)
                    out.append(infra_result(request, f"recorded output {ref.digest} not in store"))
                    break
            else:
                self.served.append(request.request_id)
                out.append(result)
        return out


class RecordingBackend:
    """Wraps a backend and appends every request and result to JSON-lines files, so the run can later be replayed
    with :class:`RecordedBackend` (outputs stay in the content store)."""

    def __init__(self, inner: Backend, results_path: str | os.PathLike,
                 requests_path: str | os.PathLike | None = None) -> None:
        self.inner = inner
        self.results_path = os.fspath(results_path)
        self.requests_path = os.fspath(requests_path) if requests_path is not None else None
        self._lock = threading.Lock()

    def execute(self, requests: Sequence[ExecRequest], store: ContentStore) -> list[ExecResult]:
        results = self.inner.execute(requests, store)
        with self._lock:
            os.makedirs(os.path.dirname(os.path.abspath(self.results_path)), exist_ok=True)
            with open(self.results_path, "a", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
            if self.requests_path:
                with open(self.requests_path, "a", encoding="utf-8") as f:
                    for request in requests:
                        f.write(json.dumps(request.to_dict(), sort_keys=True) + "\n")
        return results
