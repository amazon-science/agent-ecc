# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-execution backends: the protocol, a replay backend, a recording wrapper, local Docker (SWE-bench) and the HTTP
reference service. Backends return raw outputs; family adapters turn them into verdicts."""

from .base import (
    Backend,
    CallableBackend,
    ExecRequest,
    ExecResult,
    RecordedBackend,
    RecordingBackend,
    infra_result,
)

__all__ = [
    "Backend",
    "CallableBackend",
    "ExecRequest",
    "ExecResult",
    "HTTPBackend",
    "LocalDockerBackend",
    "RecordedBackend",
    "RecordingBackend",
    "ReferenceService",
    "infra_result",
]


def __getattr__(name: str):
    if name == "LocalDockerBackend":
        from .docker import LocalDockerBackend

        return LocalDockerBackend
    if name in ("HTTPBackend", "ReferenceService"):
        from . import service

        return getattr(service, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
