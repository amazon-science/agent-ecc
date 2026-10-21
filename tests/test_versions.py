# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing
import os

import pytest

from agent_ecc.canonical import canonical_json
from agent_ecc.versions import (
    ANY_WORLD,
    AllOf,
    Check,
    DirectoryRegistry,
    EpochFirstError,
    GraderVersion,
    MemoryRegistry,
    PublicationError,
    WorldVersion,
    iter_checks,
    spec_digest,
    spec_from_dict,
    versions_differ,
)


def c(name, run="r1", exp="e1"):
    return Check(name=name, execution={"runner": run}, grading={"expected": exp}, params={"note": name})


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        self.t += 1.0
        return self.t


def test_check_identities_separate_execution_from_grading():
    a, b = c("x"), c("y")
    assert a.execution_digest == b.execution_digest and a.grading_digest == b.grading_digest
    assert spec_digest(a) != spec_digest(b)  # names and params are part of the spec, not of the identities
    assert c("x", run="r2").execution_digest != a.execution_digest
    assert c("x", exp="e2").grading_digest != a.grading_digest


def test_spec_round_trip_and_flattening():
    spec = AllOf("outer", (c("a"), AllOf("inner", (c("b"), c("c")))))
    assert [x.name for x in iter_checks(spec)] == ["a", "b", "c"]
    again = spec_from_dict(json.loads(json.dumps(spec.to_dict())))
    assert again == spec and spec_digest(again) == spec_digest(spec)
    with pytest.raises(ValueError):
        AllOf("empty", ())
    with pytest.raises(ValueError):
        spec_from_dict({"type": "or"})


def test_grader_version_lookup_governed_and_worlds():
    v = GraderVersion(name="g", entries={"t1": {"w0": c("a"), "w1": c("b")}, "t2": c("z"), "t3": {"w0": c("q")}},
                      governed={"t1", "t3", "t4"})
    assert v.lookup("t1", "w1").spec == c("b") and v.lookup("t1", "w1").status == "covered"
    assert v.lookup("t1", "w9").status == "uncovered"  # governed, no truth for that world
    assert v.lookup("t2", "w0").status == "ungoverned"  # an entry, but not governed
    assert v.lookup("t4", "w0").status == "uncovered"  # governed without entries
    assert v.lookup("t5", "w0").status == "ungoverned"
    assert v.entries["t2"][ANY_WORLD] == c("z")
    v2 = GraderVersion(name="g", entries={"t2": c("z")})  # governed=None: every task with an entry
    assert v2.lookup("t2", None).status == "covered" and v2.lookup("t9", "w0").status == "ungoverned"
    assert not v.covers_world("w1") and not v.covers_world("w0")  # t4 has no entry at all
    v3 = GraderVersion(name="g", entries={"t1": {"w0": c("a"), "w1": c("b")}, "t2": c("z")})
    assert v3.covers_world("w0") and v3.covers_world("w1") and not v3.covers_world("w2")


def test_grader_version_is_content_addressed():
    a = GraderVersion(name="g", entries={"t": c("a")}, governed={"t"}, meta={"k": [1]})
    b = GraderVersion.from_dict(json.loads(json.dumps(a.to_dict())))
    assert a.version_id == b.version_id and a == b
    assert GraderVersion(name="g", entries={"t": c("a", exp="e2")}).version_id != a.version_id
    d = a.to_dict()
    d["name"] = "other"
    with pytest.raises(ValueError):
        GraderVersion.from_dict(d)
    assert versions_differ(a, GraderVersion(name="h", entries={"t": c("a", exp="e2"), "u": c("u")})) == ["t", "u"]


@pytest.fixture(params=["memory", "directory"])
def registry(request, tmp_path):
    clock = Clock()
    if request.param == "memory":
        return MemoryRegistry(clock=clock)
    return DirectoryRegistry(tmp_path / "registry", clock=clock)


def test_publication_log_is_the_pointer(registry):
    assert registry.read_epoch() is None and registry.current_world() is None
    g1 = GraderVersion(name="g1", entries={"t": {"w0": c("a")}})
    p1 = registry.publish_grader(g1, source="startup")
    e = registry.read_epoch()
    assert e.ref.seq == p1.seq == 0 and e.version == g1 and e.ref.read_at > p1.published_at
    g2 = GraderVersion(name="g2", entries={"t": {"w0": c("a", exp="e2")}})
    p2 = registry.publish_grader(g2, source="fix")
    assert p2.seq == 1 and registry.read_epoch().version.name == "g2"
    assert [p.name for p in registry.publications("grader")] == ["g1", "g2"]
    assert registry.epoch_at(p1.published_at + 0.5).version_id == g1.version_id
    assert registry.epoch_at(p2.published_at).version_id == g2.version_id
    assert registry.epoch_at(0.0) is None
    assert registry.grader(g1.version_id) == g1


def test_epoch_first_rule(registry):
    with pytest.raises(EpochFirstError):
        registry.publish_world("w0")  # no grader yet
    g1 = GraderVersion(name="g1", entries={"t": {"w0": c("a")}})
    registry.publish_grader(g1)
    registry.publish_world("w0")
    with pytest.raises(EpochFirstError):
        registry.publish_world("w1")  # g1 has no truth for w1
    g_new_only = GraderVersion(name="g2", entries={"t": {"w1": c("b")}})
    with pytest.raises(EpochFirstError):
        registry.publish_grader(g_new_only)  # would leave the current world w0 without truth
    registry.publish_grader(g_new_only, require_coverage=False)
    registry.publish_world("w1")
    pin = registry.resolve_start()
    assert pin.world == "w1" and pin.epoch.name == "g2"
    assert [p.kind for p in registry.publications()] == ["grader", "world", "grader", "world"]


def test_world_versions_are_content_addressed(registry):
    w = WorldVersion(name="bundle-1", components={"db": "sha256:" + "1" * 64})
    g = GraderVersion(name="g", entries={"t": {w.version_id: c("a")}})
    registry.publish_grader(g)
    registry.publish_world(w)
    assert registry.current_world() == w.version_id


def test_directory_registry_survives_a_torn_publication(tmp_path):
    reg = DirectoryRegistry(tmp_path / "r", clock=Clock())
    reg.publish_grader(GraderVersion(name="g1", entries={"t": c("a")}))
    with open(reg.log_path, "a") as f:
        f.write('{"seq": 1, "kind": "grader", "version_id"')  # a publisher crashed mid-append
    assert reg.read_epoch().version.name == "g1"  # the fragment was never published
    reg.publish_grader(GraderVersion(name="g2", entries={"t": c("b")}))
    reread = DirectoryRegistry(tmp_path / "r")
    assert [p.name for p in reread.publications()] == ["g1", "g2"]
    assert reread.read_epoch().version.name == "g2"


def test_a_whole_line_left_unterminated_by_a_crash_never_becomes_a_publication(tmp_path):
    """A publisher that crashed after writing a whole JSON line but before its newline left a fragment readers ignore.
    The next publisher (whose sequence number does not count the fragment) marks it aborted: the log stays readable
    and the fragment is never published, not even retroactively."""
    reg = DirectoryRegistry(tmp_path / "r", clock=Clock())
    reg.publish_grader(GraderVersion(name="g1", entries={"t": c("a")}))
    g2 = GraderVersion(name="g2", entries={"t": c("b")})
    reg._store().put(canonical_json(g2.body()))
    fragment = json.dumps({"seq": 1, "kind": "grader", "version_id": g2.version_id, "name": "g2",
                           "published_at": 101.5, "source": "crashed"}, sort_keys=True)  # g1 was published at 101
    with open(reg.log_path, "a") as f:
        f.write(fragment)  # the crash happened before the newline
    assert reg.read_epoch().version.name == "g1"
    reg.publish_grader(GraderVersion(name="g3", entries={"t": c("c")}))
    reread = DirectoryRegistry(tmp_path / "r")
    assert [(p.seq, p.name) for p in reread.publications()] == [(0, "g1"), (1, "g3")]
    assert reread.read_epoch().version.name == "g3" and reread.epoch_at(102.0).name == "g1"


def test_directory_registry_detects_a_missing_body(tmp_path):
    reg = DirectoryRegistry(tmp_path / "r")
    g = GraderVersion(name="g1", entries={"t": c("a")})
    reg.publish_grader(g)
    os.remove(reg._objects.path(g.version_id))
    with pytest.raises(PublicationError):
        DirectoryRegistry(tmp_path / "r").read_epoch()


def _publish_many(root, worker, n):
    reg = DirectoryRegistry(root)
    for i in range(n):
        reg.publish_grader(GraderVersion(name=f"w{worker}-{i}", entries={"t": c("a", exp=f"{worker}-{i}")}))


def test_concurrent_publishers_get_distinct_increasing_sequence_numbers(tmp_path):
    root = str(tmp_path / "shared")
    DirectoryRegistry(root)
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_publish_many, args=(root, w, 5)) for w in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    pubs = DirectoryRegistry(root).publications()
    assert len(pubs) == 20 and [p.seq for p in pubs] == list(range(20))
    assert len({p.version_id for p in pubs}) == 20
