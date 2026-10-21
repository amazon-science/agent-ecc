# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
import pickle

import pytest

from agent_ecc.canonical import FrozenDict, canonical_json, digest_bytes, digest_json, freeze, is_digest, thaw
from agent_ecc.store import (
    ChainStore,
    DirectoryStore,
    EvidenceCorrupt,
    EvidenceMissing,
    LocatorStore,
    MemoryStore,
    get_json,
    put_json,
)


def test_canonical_json_is_key_order_and_whitespace_independent():
    a = {"b": [1, 2.5, None], "a": {"y": True, "x": "é"}}
    b = json.loads(json.dumps(a, indent=4))
    b = {"a": {"x": "é", "y": True}, "b": [1, 2.5, None]}
    assert canonical_json(a) == canonical_json(b)
    assert canonical_json(a) == b'{"a":{"x":"\\u00e9","y":true},"b":[1,2.5,null]}'
    assert digest_json(a) == digest_bytes(canonical_json(a))


def test_canonical_json_rejects_non_json_values():
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})
    with pytest.raises(ValueError):
        freeze(float("inf"))
    with pytest.raises(TypeError):
        freeze({1: "int key"})
    with pytest.raises(TypeError):
        freeze(object())


def test_digest_format():
    d = digest_bytes(b"")
    assert d == "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert is_digest(d) and not is_digest("sha256:abc") and not is_digest(d.upper())


def test_freeze_is_deep_immutable_and_hashable():
    value = freeze({"a": [1, {"b": [2, 3]}], "c": "x"})
    assert isinstance(value, FrozenDict) and isinstance(value["a"], tuple) and isinstance(value["a"][1], FrozenDict)
    assert hash(value) == hash(freeze({"c": "x", "a": [1, {"b": [2, 3]}]}))
    with pytest.raises(TypeError):
        value["a"] = 1  # type: ignore[index]
    assert thaw(value) == {"a": [1, {"b": [2, 3]}], "c": "x"}
    assert value == {"a": (1, {"b": (2, 3)}), "c": "x"}
    assert pickle.loads(pickle.dumps(value)) == value


def test_frozendict_constructor_freezes_values():
    fd = FrozenDict({"z": [1, 2], "a": {"k": [3]}})
    assert list(fd) == ["a", "z"] and fd["z"] == (1, 2) and isinstance(fd["a"], FrozenDict)
    hash(fd)


def test_memory_store_round_trip_and_verification():
    store = MemoryStore()
    digest = store.put(b"hello")
    assert store.contains(digest) and store.get(digest) == b"hello"
    assert store.put(b"hello") == digest and len(store) == 1
    with pytest.raises(EvidenceMissing):
        store.get(digest_bytes(b"other"))
    store._objects[digest] = b"tampered"
    with pytest.raises(EvidenceCorrupt):
        store.get(digest)


def test_directory_store_layout_and_corruption(tmp_path):
    store = DirectoryStore(tmp_path)
    digest = put_json(store, {"k": [1, 2]})
    path = store.path(digest)
    assert path == os.path.join(str(tmp_path), "sha256", digest.split(":")[1])
    assert get_json(store, digest) == {"k": [1, 2]}
    assert not any(name.endswith(".tmp") for name in os.listdir(os.path.dirname(path)))
    with open(path, "wb") as f:
        f.write(b"changed")
    with pytest.raises(EvidenceCorrupt):
        store.get(digest)
    with pytest.raises(ValueError):
        store.get("not-a-digest")
    assert not store.contains("not-a-digest")


def test_locator_and_chain_stores(tmp_path):
    p = tmp_path / "log.txt"
    p.write_bytes(b"raw log")
    digest = digest_bytes(b"raw log")
    locator = LocatorStore({digest: p})
    assert locator.get(digest) == b"raw log"
    with pytest.raises(TypeError):
        locator.put(b"x")
    mem = MemoryStore()
    chain = ChainStore(mem, locator)
    assert chain.get(digest) == b"raw log" and chain.contains(digest)
    other = chain.put(b"new")
    assert mem.contains(other)
    p.write_bytes(b"edited")
    with pytest.raises(EvidenceCorrupt):
        chain.get(digest)
    with pytest.raises(EvidenceMissing):
        chain.get(digest_bytes(b"nowhere"))


def test_copy_objects_between_stores():
    from agent_ecc.store import copy_objects

    a, b = MemoryStore(), MemoryStore()
    digests = [a.put(b"one"), a.put(b"two")]
    assert copy_objects(a, b, digests) == 2 and copy_objects(a, b, digests) == 0
    assert b.get(digests[1]) == b"two"
