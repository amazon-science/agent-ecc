# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical JSON, content digests and immutable JSON values.

Every record in agent-ecc is identified by the digest of its canonical JSON form, so the canonical form must be stable
across processes, machines and Python versions:

* objects are serialized with sorted keys and no insignificant whitespace;
* output is ASCII (non-ASCII characters are escaped), so the bytes never depend on a terminal or locale;
* only JSON types are accepted: ``dict`` with ``str`` keys, ``list``/``tuple``, ``str``, ``int``, finite ``float``,
  ``bool`` and ``None``. NaN and infinities are rejected, because they have no JSON form.

Digests are written ``sha256:<64 lowercase hex digits>``.

Records hold their JSON-like fields as frozen values (:class:`FrozenDict` and tuples), so a record cannot change after
its digest was computed.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping
from typing import Any

__all__ = [
    "DIGEST_ALGORITHM",
    "FrozenDict",
    "canonical_json",
    "digest_bytes",
    "digest_json",
    "digest_text",
    "freeze",
    "is_digest",
    "thaw",
    "to_jsonable",
]

DIGEST_ALGORITHM = "sha256"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class FrozenDict(Mapping):
    """An immutable, hashable mapping with string keys, iterated in sorted key order."""

    __slots__ = ("_data", "_hash")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        data = dict(*args, **kwargs)
        self._data = {_key(key): freeze(data[key]) for key in sorted(data, key=_key)}
        self._hash: int | None = None

    @classmethod
    def _trusted(cls, data: dict) -> "FrozenDict":
        """Wrap a dict whose keys are str and whose values are already frozen."""
        obj = cls.__new__(cls)
        obj._data = {key: data[key] for key in sorted(data)}
        obj._hash = None
        return obj

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(tuple(self._data.items()))
        return self._hash

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    def __reduce__(self):
        return (FrozenDict, (dict(self._data),))


def freeze(value: Any) -> Any:
    """Return an immutable copy of a JSON-like value (dicts become FrozenDict, lists become tuples)."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, enum.Enum):
        return freeze(value.value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float {value!r} has no canonical JSON form")
        return value
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict._trusted({_key(k): freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if hasattr(value, "to_dict"):
        return freeze(value.to_dict())
    raise TypeError(f"{type(value).__name__} is not a JSON value")


def _key(key: Any) -> str:
    if isinstance(key, enum.Enum):
        key = key.value
    if not isinstance(key, str):
        raise TypeError(f"JSON object keys must be str, got {type(key).__name__}")
    return key


def thaw(value: Any) -> Any:
    """Return a plain (mutable) copy of a frozen JSON value: FrozenDict becomes dict, tuples become lists."""
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw(v) for v in value]
    if isinstance(value, enum.Enum):
        return value.value
    return value


def to_jsonable(value: Any) -> Any:
    """Plain JSON data for a value that may contain records (objects with ``to_dict``), enums and frozen values."""
    if hasattr(value, "to_dict") and not isinstance(value, Mapping):
        return to_jsonable(value.to_dict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Mapping):
        return {_key(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite float {value!r} has no canonical JSON form")
    return value


def canonical_json(value: Any) -> bytes:
    """The canonical JSON bytes of a value: sorted keys, compact separators, ASCII, finite numbers only."""
    return json.dumps(
        to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def digest_bytes(data: bytes) -> str:
    """``sha256:<hex>`` of raw bytes."""
    return f"{DIGEST_ALGORITHM}:{hashlib.sha256(data).hexdigest()}"


def digest_text(text: str) -> str:
    """Digest of a string's UTF-8 bytes."""
    return digest_bytes(text.encode("utf-8"))


def digest_json(value: Any) -> str:
    """Digest of a value's canonical JSON form."""
    return digest_bytes(canonical_json(value))


def is_digest(value: Any) -> bool:
    """True for a well-formed ``sha256:<64 hex>`` string."""
    return isinstance(value, str) and bool(_DIGEST_RE.match(value))
