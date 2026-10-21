# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed stores for evidence and version bodies.

Records never hold evidence bytes or locations, only digests (:class:`agent_ecc.records.EvidenceRef`). A store maps a
digest to the bytes it names and verifies them on every read, so a corrupted or substituted object is an error, never
silently different evidence.

* :class:`MemoryStore`    a dict, for tests and short-lived processes;
* :class:`DirectoryStore` ``<root>/sha256/<hex>``, written atomically (safe on shared filesystems);
* :class:`LocatorStore`   read-only, for evidence that already lives in files (a digest -> path map);
* :class:`ChainStore`     reads through several stores, writes to the first.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

from .canonical import canonical_json, digest_bytes, is_digest

__all__ = [
    "ChainStore",
    "ContentStore",
    "DirectoryStore",
    "EvidenceCorrupt",
    "EvidenceMissing",
    "LocatorStore",
    "MemoryStore",
    "copy_objects",
    "get_json",
    "get_text",
    "put_json",
    "put_text",
]


class EvidenceMissing(KeyError):
    """No store holds the object a digest names."""


class EvidenceCorrupt(ValueError):
    """An object's bytes do not hash to the digest that names it."""


@runtime_checkable
class ContentStore(Protocol):
    """What the library needs from a store: put bytes and get back their digest; get verified bytes by digest."""

    def put(self, data: bytes) -> str: ...

    def get(self, digest: str) -> bytes: ...

    def contains(self, digest: str) -> bool: ...


def _check_digest(digest: str) -> str:
    if not is_digest(digest):
        raise ValueError(f"not a sha256 digest: {digest!r}")
    return digest


def _verified(digest: str, data: bytes) -> bytes:
    if digest_bytes(data) != digest:
        raise EvidenceCorrupt(f"object {digest} does not match its content")
    return data


class MemoryStore:
    """An in-memory content store."""

    def __init__(self, objects: Mapping[str, bytes] | None = None) -> None:
        self._objects: dict[str, bytes] = {}
        self._lock = threading.Lock()
        for data in (objects or {}).values():
            self.put(data)

    def put(self, data: bytes) -> str:
        data = bytes(data)
        digest = digest_bytes(data)
        with self._lock:
            self._objects.setdefault(digest, data)
        return digest

    def get(self, digest: str) -> bytes:
        _check_digest(digest)
        with self._lock:
            data = self._objects.get(digest)
        if data is None:
            raise EvidenceMissing(digest)
        return _verified(digest, data)

    def contains(self, digest: str) -> bool:
        with self._lock:
            return digest in self._objects

    def digests(self) -> list[str]:
        with self._lock:
            return sorted(self._objects)

    def __len__(self) -> int:
        with self._lock:
            return len(self._objects)


class DirectoryStore:
    """Objects as files ``<root>/sha256/<hex>``. Writes go to a unique temporary file and are renamed into place."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = os.fspath(root)

    def path(self, digest: str) -> str:
        algorithm, _, hexdigest = _check_digest(digest).partition(":")
        return os.path.join(self.root, algorithm, hexdigest)

    def put(self, data: bytes) -> str:
        data = bytes(data)
        digest = digest_bytes(data)
        path = self.path(digest)
        if os.path.exists(path):
            return digest
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return digest

    def get(self, digest: str) -> bytes:
        path = self.path(digest)
        try:
            with open(path, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            raise EvidenceMissing(digest) from None
        return _verified(digest, data)

    def contains(self, digest: str) -> bool:
        try:
            return os.path.exists(self.path(digest))
        except ValueError:
            return False


class LocatorStore:
    """Read-only access to evidence that already exists as files, verified against the digests that name it."""

    def __init__(self, locations: Mapping[str, str | os.PathLike] | None = None) -> None:
        self._locations: dict[str, str] = {}
        for digest, path in (locations or {}).items():
            self.add(digest, path)

    def add(self, digest: str, path: str | os.PathLike) -> None:
        self._locations[_check_digest(digest)] = os.fspath(path)

    def put(self, data: bytes) -> str:
        raise TypeError("LocatorStore is read-only; chain it behind a writable store")

    def get(self, digest: str) -> bytes:
        path = self._locations.get(_check_digest(digest))
        if path is None:
            raise EvidenceMissing(digest)
        try:
            with open(path, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            raise EvidenceMissing(digest) from None
        return _verified(digest, data)

    def contains(self, digest: str) -> bool:
        path = self._locations.get(digest)
        return path is not None and os.path.exists(path)


class ChainStore:
    """Reads try each store in order; writes go to the first store."""

    def __init__(self, *stores: ContentStore) -> None:
        if not stores:
            raise ValueError("ChainStore needs at least one store")
        self.stores = tuple(stores)

    def put(self, data: bytes) -> str:
        return self.stores[0].put(data)

    def get(self, digest: str) -> bytes:
        for store in self.stores:
            if store.contains(digest):
                return store.get(digest)
        raise EvidenceMissing(digest)

    def contains(self, digest: str) -> bool:
        return any(store.contains(digest) for store in self.stores)


def put_text(store: ContentStore, text: str) -> str:
    return store.put(text.encode("utf-8"))


def get_text(store: ContentStore, digest: str) -> str:
    return store.get(digest).decode("utf-8")


def put_json(store: ContentStore, value: Any) -> str:
    """Store a value as canonical JSON; the digest is then a content address of the value itself."""
    return store.put(canonical_json(value))


def get_json(store: ContentStore, digest: str) -> Any:
    return json.loads(store.get(digest))


def copy_objects(source: ContentStore, target: ContentStore, digests: Iterable[str]) -> int:
    """Copy objects between stores (verified on read). Returns the number copied."""
    n = 0
    for digest in digests:
        if not target.contains(digest):
            target.put(source.get(digest))
            n += 1
    return n
