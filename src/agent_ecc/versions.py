# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grader and world versions, and their publication.

A **test spec** says how one task is graded. Its atoms are :class:`Check` objects, each with two identities:

* ``execution``: digests of every input that decides *what runs* (an image, an eval script, patch-application commands,
  an evaluator program). If it differs from what a recorded run executed, the recorded outputs say nothing about the
  new check and the tests must be executed again.
* ``grading``: digests of every input that decides *how the outputs are judged* (test labels, a log parser, grading
  code, an expected answer). If only this differs, the recorded outputs can be judged again without executing.

:class:`AllOf` is a conjunction of specs, evaluated in order: the first part that fails logically decides the whole
(a short circuit), and a part without a logical verdict leaves the whole without one.

A **grader version** (an *epoch*) is an immutable, content-addressed map (task, world version) -> test spec. A task the
version does not govern keeps its recorded verdict; a governed task without a spec for a record's world has no verdict.

A **registry** publishes versions in a single append-only publication log, which is also the pointer: the current
grader version is the last grader publication. Publication writes the version body first and appends the log line
second, so a reader that sees a publication can always load its body. The **epoch-first rule**: a world version may be
published only when the current grader version covers it, so no sandbox can resolve a world that no published epoch
can grade. A seal reads the pointer exactly once (:meth:`Registry.read_epoch`).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Union

from .canonical import FrozenDict, canonical_json, digest_json, freeze, thaw
from .records import EpochRef
from .store import ContentStore, DirectoryStore, EvidenceMissing, MemoryStore

__all__ = [
    "ANY_WORLD",
    "AllOf",
    "Check",
    "Epoch",
    "EpochFirstError",
    "DirectoryRegistry",
    "GraderVersion",
    "Lookup",
    "MemoryRegistry",
    "Publication",
    "PublicationError",
    "Registry",
    "StartPin",
    "TestSpec",
    "WorldVersion",
    "iter_checks",
    "spec_digest",
    "spec_from_dict",
    "spec_to_dict",
]

ANY_WORLD = "*"
SCHEMA_GRADER = "agent-ecc/grader-version/1"
SCHEMA_WORLD = "agent-ecc/world-version/1"


# ---------------------------------------------------------------------------------------------------------- specs


@dataclass(frozen=True)
class Check:
    """An atomic test spec. Every input that can change a verdict must appear in ``execution`` (it changes what runs)
    or ``grading`` (it changes how outputs are judged). ``params`` carries what an adapter needs to run or parse the
    check (names, object digests to load); it is part of the spec digest but not of the two identities."""

    name: str
    execution: Mapping[str, Any] = field(default_factory=FrozenDict)
    grading: Mapping[str, Any] = field(default_factory=FrozenDict)
    params: Mapping[str, Any] = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution", freeze(self.execution))
        object.__setattr__(self, "grading", freeze(self.grading))
        object.__setattr__(self, "params", freeze(self.params))

    @cached_property
    def execution_digest(self) -> str:
        return digest_json(self.execution)

    @cached_property
    def grading_digest(self) -> str:
        return digest_json(self.grading)

    def to_dict(self) -> dict:
        return {"type": "check", "name": self.name, "execution": thaw(self.execution),
                "grading": thaw(self.grading), "params": thaw(self.params)}


@dataclass(frozen=True)
class AllOf:
    """A conjunction: passes only if every part passes, evaluated in order with a short circuit on a logical fail."""

    name: str
    parts: tuple[TestSpec, ...]

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        if not parts:
            raise ValueError("AllOf needs at least one part")
        for p in parts:
            if not isinstance(p, (Check, AllOf)):
                raise TypeError(f"AllOf parts must be Check or AllOf, got {type(p).__name__}")
        object.__setattr__(self, "parts", parts)

    def to_dict(self) -> dict:
        return {"type": "all_of", "name": self.name, "parts": [p.to_dict() for p in self.parts]}


TestSpec = Union[Check, AllOf]


def spec_to_dict(spec: TestSpec) -> dict:
    return spec.to_dict()


def spec_from_dict(d: Mapping[str, Any]) -> TestSpec:
    kind = d.get("type")
    if kind == "check":
        return Check(name=d["name"], execution=d.get("execution") or {}, grading=d.get("grading") or {},
                     params=d.get("params") or {})
    if kind == "all_of":
        return AllOf(name=d["name"], parts=tuple(spec_from_dict(p) for p in d["parts"]))
    raise ValueError(f"unknown test spec type {kind!r}")


def spec_digest(spec: TestSpec) -> str:
    return digest_json(spec.to_dict())


def iter_checks(spec: TestSpec) -> Iterator[Check]:
    """The checks of a spec in evaluation order (nested conjunctions flattened; AND is associative)."""
    if isinstance(spec, Check):
        yield spec
    else:
        for part in spec.parts:
            yield from iter_checks(part)


# ---------------------------------------------------------------------------------------------------------- versions


@dataclass(frozen=True)
class Lookup:
    """The result of looking a (task, world) up in a grader version."""

    status: str  # covered | ungoverned | uncovered
    spec: TestSpec | None = None


@dataclass(frozen=True)
class GraderVersion:
    """An immutable grader version (epoch): task -> world version (or ``"*"`` for every world) -> test spec.

    ``governed`` lists the tasks the version makes a claim about; ``None`` means every task with an entry. A record of
    an ungoverned task keeps its recorded verdict under this version.
    """

    name: str
    entries: Mapping[str, Mapping[str, TestSpec]]
    governed: frozenset[str] | None = None
    family: str | None = None
    meta: Mapping[str, Any] = field(default_factory=FrozenDict)
    schema: str = SCHEMA_GRADER

    def __post_init__(self) -> None:
        if self.schema != SCHEMA_GRADER:
            raise ValueError(f"unsupported grader version schema {self.schema!r}")
        entries = {}
        for task, by_world in self.entries.items():
            if isinstance(by_world, (Check, AllOf)):
                by_world = {ANY_WORLD: by_world}
            worlds = {}
            for world, spec in by_world.items():
                if not isinstance(spec, (Check, AllOf)):
                    spec = spec_from_dict(spec)
                worlds[str(world)] = spec
            entries[str(task)] = _SpecMap(worlds)
        object.__setattr__(self, "entries", _SpecMap(entries))
        if self.governed is not None:
            object.__setattr__(self, "governed", frozenset(str(t) for t in self.governed))
        object.__setattr__(self, "meta", freeze(self.meta))

    def body(self) -> dict:
        return {"schema": self.schema, "name": self.name, "family": self.family,
                "entries": {task: {world: spec.to_dict() for world, spec in by_world.items()}
                            for task, by_world in self.entries.items()},
                "governed": sorted(self.governed) if self.governed is not None else None, "meta": thaw(self.meta)}

    @cached_property
    def version_id(self) -> str:
        return digest_json(self.body())

    def to_dict(self) -> dict:
        return {**self.body(), "version_id": self.version_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, verify: bool = True) -> GraderVersion:
        governed = d.get("governed")
        version = cls(name=d["name"], entries=d.get("entries") or {},
                      governed=frozenset(governed) if governed is not None else None, family=d.get("family"),
                      meta=d.get("meta") or {}, schema=d.get("schema", SCHEMA_GRADER))
        if verify and d.get("version_id") is not None and d["version_id"] != version.version_id:
            raise ValueError(f"grader version id {d['version_id']} does not match its content")
        return version

    def governs(self, task: str) -> bool:
        if self.governed is not None:
            return task in self.governed
        return task in self.entries

    def lookup(self, task: str, world: str | None) -> Lookup:
        if not self.governs(task):
            return Lookup("ungoverned")
        by_world = self.entries.get(task)
        if not by_world:
            return Lookup("uncovered")
        spec = by_world.get(world) if world is not None else None
        if spec is None:
            spec = by_world.get(ANY_WORLD)
        return Lookup("covered", spec) if spec is not None else Lookup("uncovered")

    def tasks(self) -> frozenset[str]:
        return frozenset(self.governed) if self.governed is not None else frozenset(self.entries)

    def covers_world(self, world: str) -> bool:
        """True when every governed task has a spec for ``world`` (or for every world)."""
        for task in self.tasks():
            by_world = self.entries.get(task) or {}
            if world not in by_world and ANY_WORLD not in by_world:
                return False
        return True


class _SpecMap(FrozenDict):
    """A FrozenDict whose values are test specs (or maps of them), which freeze() would not accept."""

    def __init__(self, data: Mapping[str, Any]) -> None:  # noqa: D107 - values are specs, already immutable
        self._data = {str(k): data[k] for k in sorted(data)}
        self._hash = None


@dataclass(frozen=True)
class WorldVersion:
    """A world version by content: component name -> digest (databases, images, tool implementations)."""

    name: str
    components: Mapping[str, str]
    meta: Mapping[str, Any] = field(default_factory=FrozenDict)
    schema: str = SCHEMA_WORLD

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", freeze(self.components))
        object.__setattr__(self, "meta", freeze(self.meta))

    def body(self) -> dict:
        return {"schema": self.schema, "name": self.name, "components": thaw(self.components),
                "meta": thaw(self.meta)}

    @cached_property
    def version_id(self) -> str:
        return digest_json(self.body())

    def to_dict(self) -> dict:
        return {**self.body(), "version_id": self.version_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> WorldVersion:
        return cls(name=d["name"], components=d.get("components") or {}, meta=d.get("meta") or {},
                   schema=d.get("schema", SCHEMA_WORLD))


# ---------------------------------------------------------------------------------------------------------- publication


class PublicationError(RuntimeError):
    """A publication was refused or the publication log is unusable."""


class EpochFirstError(PublicationError):
    """A world version would become resolvable before a published grader version covers it."""


@dataclass(frozen=True)
class Publication:
    """One line of the publication log. ``seq`` is strictly increasing across all publications."""

    seq: int
    kind: str  # grader | world
    version_id: str
    name: str | None
    published_at: float
    source: str | None = None

    def to_dict(self) -> dict:
        return {"seq": self.seq, "kind": self.kind, "version_id": self.version_id, "name": self.name,
                "published_at": self.published_at, "source": self.source}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Publication:
        return cls(seq=int(d["seq"]), kind=d["kind"], version_id=d["version_id"], name=d.get("name"),
                   published_at=float(d["published_at"]), source=d.get("source"))

    def ref(self, read_at: float | None = None) -> EpochRef:
        return EpochRef(seq=self.seq, version_id=self.version_id, name=self.name, published_at=self.published_at,
                        read_at=read_at)


@dataclass(frozen=True)
class Epoch:
    """A grader version as one read of the pointer returned it: the publication reference and the version body."""

    ref: EpochRef
    version: GraderVersion


@dataclass(frozen=True)
class StartPin:
    """What a starting trajectory resolves, once: the world it will observe and the grader epoch current then."""

    world: str | None
    epoch: EpochRef | None


class Registry:
    """Publication of grader and world versions over an append-only log. Subclasses provide storage."""

    def __init__(self, clock=time.time) -> None:
        self._clock = clock
        self._bodies: dict[str, GraderVersion] = {}

    # storage, provided by subclasses
    def _lock(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def _read_log(self) -> list[Publication]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _append_log(self, publication: Publication) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _store(self) -> ContentStore:  # pragma: no cover - abstract
        raise NotImplementedError

    # publication
    def publish_grader(self, version: GraderVersion, *, source: str | None = None,
                       require_coverage: bool = True) -> Publication:
        """Publish a grader version: its body first, then the log line that makes it current.

        With ``require_coverage`` (default) the version must cover the current world version, so trajectories that
        resolve the current world can be graded under it; pass False for a fix that defines truth only for a newer
        world (records of the old world then have no verdict and their groups are neutralized)."""
        with self._lock():
            log = self._read_log()
            world = _last(log, "world")
            if require_coverage and world is not None and not version.covers_world(world.version_id):
                raise EpochFirstError(f"grader version {version.name!r} does not cover the current world "
                                      f"{world.version_id}; pass require_coverage=False to publish it anyway")
            digest = self._store().put(canonical_json(version.body()))
            if digest != version.version_id:  # pragma: no cover - canonical form is the identity
                raise PublicationError("stored body does not hash to the version id")
            publication = Publication(seq=_next_seq(log), kind="grader", version_id=version.version_id,
                                      name=version.name, published_at=self._clock(), source=source)
            self._append_log(publication)
            self._bodies[version.version_id] = version
            return publication

    def publish_world(self, world: WorldVersion | str, *, source: str | None = None) -> Publication:
        """Make a world version resolvable. Epoch first: the current grader version must already cover it."""
        version_id = world.version_id if isinstance(world, WorldVersion) else str(world)
        name = world.name if isinstance(world, WorldVersion) else None
        with self._lock():
            log = self._read_log()
            grader = _last(log, "grader")
            if grader is None:
                raise EpochFirstError("no grader version is published; publish one that covers the world first")
            if not self._load(grader.version_id).covers_world(version_id):
                raise EpochFirstError(f"the current grader version {grader.name!r} does not cover world "
                                      f"{version_id}; publish a grader version that covers it first")
            if isinstance(world, WorldVersion):
                self._store().put(canonical_json(world.body()))
            publication = Publication(seq=_next_seq(log), kind="world", version_id=version_id, name=name,
                                      published_at=self._clock(), source=source)
            self._append_log(publication)
            return publication

    # reads
    def read_epoch(self) -> Epoch | None:
        """The one read of the grader pointer a seal makes: the current grader version, or None if none is
        published. The returned epoch carries the read time."""
        log = self._read_log()
        read_at = self._clock()
        last = _last(log, "grader")
        if last is None:
            return None
        return Epoch(ref=last.ref(read_at=read_at), version=self._load(last.version_id))

    def current_world(self) -> str | None:
        last = _last(self._read_log(), "world")
        return last.version_id if last is not None else None

    def resolve_start(self) -> StartPin:
        """What a starting trajectory pins: the current world and the current grader epoch, from one log read."""
        log = self._read_log()
        read_at = self._clock()
        world, grader = _last(log, "world"), _last(log, "grader")
        return StartPin(world=world.version_id if world else None,
                        epoch=grader.ref(read_at=read_at) if grader else None)

    def publications(self, kind: str | None = None) -> list[Publication]:
        log = self._read_log()
        return [p for p in log if kind is None or p.kind == kind]

    def grader(self, version_id: str) -> GraderVersion:
        return self._load(version_id)

    def epoch_at(self, t: float) -> EpochRef | None:
        """The grader epoch in force at time ``t`` according to the log (for audits and oracles that must derive a
        seal's epoch from the publication log and the time the seal read the pointer)."""
        current = None
        for p in self._read_log():
            if p.kind == "grader" and p.published_at <= t:
                current = p
        return current.ref() if current is not None else None

    def _load(self, version_id: str) -> GraderVersion:
        version = self._bodies.get(version_id)
        if version is None:
            try:
                body = json.loads(self._store().get(version_id))
            except EvidenceMissing:
                raise PublicationError(f"grader version {version_id} is in the log but its body is missing") from None
            version = GraderVersion.from_dict(body)
            if version.version_id != version_id:  # pragma: no cover - the store verifies the digest
                raise PublicationError(f"grader version body {version_id} does not match")
            self._bodies[version_id] = version
        return version


def _last(log: Sequence[Publication], kind: str) -> Publication | None:
    for p in reversed(log):
        if p.kind == kind:
            return p
    return None


def _next_seq(log: Sequence[Publication]) -> int:
    return (log[-1].seq + 1) if log else 0


class MemoryRegistry(Registry):
    """A registry in memory, for tests, simulations and single-process trainers."""

    def __init__(self, clock=time.time, store: ContentStore | None = None) -> None:
        super().__init__(clock)
        self._log: list[Publication] = []
        self._mutex = threading.RLock()
        self._objects = store if store is not None else MemoryStore()

    def _lock(self):
        return self._mutex

    def _read_log(self) -> list[Publication]:
        with self._mutex:
            return list(self._log)

    def _append_log(self, publication: Publication) -> None:
        with self._mutex:
            if self._log and publication.seq <= self._log[-1].seq:  # pragma: no cover - guarded by the lock
                raise PublicationError("publication sequence must increase")
            self._log.append(publication)

    def _store(self) -> ContentStore:
        return self._objects


class _FileLock:
    """A cross-process exclusive section that works on shared filesystems: an O_EXCL lock file, considered stale after
    ``stale_s`` seconds (a crashed publisher)."""

    def __init__(self, path: str, timeout_s: float = 60.0, stale_s: float = 30.0) -> None:
        self.path, self.timeout_s, self.stale_s = path, timeout_s, stale_s

    def __enter__(self):
        deadline = time.time() + self.timeout_s
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > self.stale_s:
                        os.remove(self.path)
                        continue
                except OSError:
                    continue
                if time.time() > deadline:
                    raise PublicationError(f"timed out waiting for the publication lock {self.path}") from None
                time.sleep(0.02)

    def __exit__(self, *exc) -> None:
        try:
            os.remove(self.path)
        except OSError:
            pass


# appended after a crashed publisher's fragment by the next publisher: the fragment's line can then never parse
_ABORTED = b" #aborted-publication\n"


class DirectoryRegistry(Registry):
    """A registry in a directory, shareable between processes and machines on one filesystem::

        <root>/objects/sha256/<hex>   version bodies (content-addressed)
        <root>/publications.jsonl     the publication log, one JSON object per line, append-only
        <root>/.publish.lock          held while publishing
    """

    LOG = "publications.jsonl"

    def __init__(self, root: str | os.PathLike, clock=time.time, *, lock_timeout_s: float = 60.0) -> None:
        super().__init__(clock)
        self.root = os.fspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._objects = DirectoryStore(os.path.join(self.root, "objects"))
        self._lock_path = os.path.join(self.root, ".publish.lock")
        self._lock_timeout_s = lock_timeout_s

    @property
    def log_path(self) -> str:
        return os.path.join(self.root, self.LOG)

    def _lock(self):
        return _FileLock(self._lock_path, timeout_s=self._lock_timeout_s)

    def _read_log(self) -> list[Publication]:
        try:
            with open(self.log_path, encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            return []
        out = []
        lines = text.split("\n")
        for line in lines[:-1]:  # the last element has no newline yet: a publication in progress, not published
            if not line.strip():
                continue
            try:
                out.append(Publication.from_dict(json.loads(line)))
            except (ValueError, KeyError, TypeError):
                # the fragment of a publisher that crashed mid-append, terminated (and marked aborted) by the next
                # publisher: its pointer never moved, so it was never published
                continue
        for a, b in zip(out, out[1:]):
            if b.seq <= a.seq:
                raise PublicationError(f"publication log {self.log_path} is not strictly increasing at seq {b.seq}")
        return out

    def _append_log(self, publication: Publication) -> None:
        line = (json.dumps(publication.to_dict(), sort_keys=True) + "\n").encode("utf-8")
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                if f.tell() > 0:
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) != b"\n":
                        # a crashed publisher's fragment: it was never a publication (readers ignored it while it was
                        # unterminated), and it must not become one now, even if it is a whole JSON line. The marker
                        # holds no quote or bracket, so no fragment followed by it parses
                        line = _ABORTED + line
        except FileNotFoundError:
            pass
        fd = os.open(self.log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            view = memoryview(line)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)

    def _store(self) -> ContentStore:
        return self._objects


def versions_differ(a: GraderVersion, b: GraderVersion, tasks: Iterable[str] | None = None) -> list[str]:
    """Tasks whose entries differ between two versions (the set a schedule would declare governed)."""
    names = set(tasks) if tasks is not None else set(a.entries) | set(b.entries)
    out = []
    for task in sorted(names):
        ea = {w: s.to_dict() for w, s in (a.entries.get(task) or {}).items()}
        eb = {w: s.to_dict() for w, s in (b.entries.get(task) or {}).items()}
        if ea != eb:
            out.append(task)
    return out
