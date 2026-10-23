# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reference service's client retries transport failures; the scorer and the seal account for the retries."""

import dataclasses
import json
import socket
import struct
import threading

import pytest

from agent_ecc.canonical import thaw
from agent_ecc.records import Outcome, SealDecision
from agent_ecc.recompute.base import ExecResult, RecordedBackend, RecordingBackend
from agent_ecc.recompute.service import HTTPBackend, ReferenceService
from agent_ecc.scoring import ScoreInput, Scorer, grade_recorded
from agent_ecc.seal import Candidate, Member, SealSession
from agent_ecc.store import MemoryStore
from agent_ecc.versions import GraderVersion, MemoryRegistry

from toy import ToyAdapter, ToyBackend, check

FOREVER = 10 ** 9


def _read_post(conn):
    """The body of one HTTP request read from ``conn``."""
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(65536)
        if not chunk:
            break
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    n = next((int(h.split(b":", 1)[1]) for h in head.split(b"\r\n") if h.lower().startswith(b"content-length:")), 0)
    while len(body) < n:
        chunk = conn.recv(65536)
        if not chunk:
            break
        body += chunk
    return body


class FakeService:
    """A reference service behind a raw socket. It reads each post, then resets the first ``resets`` connections (RST,
    no response), then answers with ``status`` and the service's own answer (or ``reply``), or never (``hang``)."""

    def __init__(self, service=None, *, resets=0, status=200, reply=None, hang=False):
        self.service = service or ReferenceService(ToyBackend(), MemoryStore(), name="toy")
        self.resets, self.status, self.reply, self.hang = resets, status, reply, hang
        self.posts, self.held, self.lock = [], [], threading.Lock()
        self.sock = socket.create_server(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._one, args=(conn,), daemon=True).start()

    def _one(self, conn):
        body = _read_post(conn)
        with self.lock:
            k = len(self.posts)
            self.posts.append(body)
        if k < self.resets:
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            conn.close()
        elif self.hang:
            self.held.append(conn)
        else:
            payload = self.reply if self.reply is not None else self.service.handle(json.loads(body))
            data = json.dumps(payload).encode()
            conn.sendall(b"HTTP/1.1 %d X\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n"
                         % (self.status, len(data)) + data)
            conn.close()

    def close(self):
        for conn in [self.sock] + self.held:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()


@pytest.fixture
def fake():
    made = []

    def make(**kwargs):
        made.append(FakeService(**kwargs))
        return made[-1]

    yield make
    for service in made:
        service.close()


class Mid:
    """A jitter source that draws the middle of its range."""

    def uniform(self, a, b):
        return (a + b) / 2


def world(answer="ab"):
    """One record whose target check changes the runner (a re-execution), and the request it asks."""
    store, adapter = MemoryStore(), ToyAdapter()
    c1, c2 = check("t", "plain", "ab", store), check("t", "up", "AB", store)
    record = adapter.record(store, task="t", trajectory="x0", answer=answer, runs=[adapter.run(store, c1, answer)])
    item = ScoreInput(record, grade_recorded(adapter, record, c1, store))
    return store, adapter, item, GraderVersion(name="g2", entries={"t": c2}), adapter.reexec_request(record, c2, store)


def test_resets_are_retried_with_the_same_post_request_id_and_cache_key(fake):
    store, adapter, item, g2, request = world()
    service, slept, cache = fake(resets=2), [], {}
    (out,) = Scorer([adapter], store, HTTPBackend(service.url, sleep=slept.append, rng=Mid()), cache=cache) \
        .score([item], g2)
    assert out.score.outcome is Outcome.PASS and out.score.parts[0].attempts == 1
    assert len(service.posts) == 3 and len(set(service.posts)) == 1  # the same bytes on every attempt
    assert [r["request_id"] for r in json.loads(service.posts[0])["requests"]] == [request.request_id]
    assert slept == [0.5, 1.0]  # full jitter: U(0, 1 s), then U(0, 2 s)
    assert list(cache) == [request.request_id]
    assert thaw(cache[request.request_id].transport) == {"attempts": 3, "reasons": {"reset": 2}, "gave_up": False}
    assert thaw(out.transport) == {"requests": 1, "attempts": 3, "retried": 1, "gave_up": 0, "reasons": {"reset": 2}}
    healthy = {}
    (same,) = Scorer([adapter], store, HTTPBackend(fake().url), cache=healthy).score([item], g2)
    assert list(healthy) == list(cache) and same.score.same_verdict(out.score)  # same request id, same cache key


def test_a_service_that_always_resets_is_given_up_on_once(fake):
    store, adapter, item, g2, request = world()
    service, slept = fake(resets=FOREVER), []
    (out,) = Scorer([adapter], store, HTTPBackend(service.url, sleep=slept.append), retries=2).score([item], g2)
    assert out.score.outcome is Outcome.UNDETERMINABLE and out.score.parts[0].attempts == 1
    assert out.score.reason.startswith("infra_error:service: ")
    assert len(service.posts) == 3  # three posts in all, not three for each of the scorer's retries
    assert len(slept) == 2 and 0 <= slept[0] <= 1.0 and 0 <= slept[1] <= 2.0
    assert thaw(out.transport) == {"requests": 1, "attempts": 3, "retried": 1, "gave_up": 1, "reasons": {"reset": 3}}
    # an infrastructure error the service answers with is no transport failure: the scorer retries it, as before
    failing = fake(service=ReferenceService(ToyBackend(always_fail=True), MemoryStore()))
    (out,) = Scorer([adapter], store, HTTPBackend(failing.url), retries=2).score([item], g2)
    assert out.score.parts[0].attempts == 3 and len(failing.posts) == 3
    assert thaw(out.transport) == {"requests": 3, "attempts": 3, "retried": 0, "gave_up": 0, "reasons": {}}


@pytest.mark.parametrize("status", [200, 503])
@pytest.mark.parametrize("answer, verdict", [("ab", Outcome.PASS), ("ba", Outcome.FAIL)])
def test_an_answer_with_results_is_never_sent_again(fake, status, answer, verdict):
    store, adapter, item, g2, _ = world(answer)
    service, slept = fake(status=status), []
    (out,) = Scorer([adapter], store, HTTPBackend(service.url, sleep=slept.append)).score([item], g2)
    assert out.score.outcome is verdict and len(service.posts) == 1 and not slept
    assert thaw(out.transport) == {"requests": 1, "attempts": 1, "retried": 0, "gave_up": 0, "reasons": {}}


def test_only_transport_failures_are_retried(fake):
    store, _, _, _, request = world()
    for kwargs, error in [({"status": 404, "reply": {"error": "not found"}}, "service: HTTP 404"),
                          ({"reply": [1, 2]}, "service: malformed answer: list")]:
        service = fake(**kwargs)
        (r,) = HTTPBackend(service.url, sleep=lambda s: None).execute([request], store)
        assert r.status == "error" and r.error.startswith(error) and len(service.posts) == 1
        assert thaw(r.transport) == {"attempts": 1, "reasons": {}, "gave_up": False}
    service = fake(status=500, reply={"error": "overloaded"})  # a 5xx without results is a transport failure
    (r,) = HTTPBackend(service.url, sleep=lambda s: None).execute([request], store)
    assert r.error.startswith("service: HTTP 500") and len(service.posts) == 3
    assert thaw(r.transport) == {"attempts": 3, "reasons": {"http_5xx": 3}, "gave_up": True}


def test_refused_timeouts_and_the_deadline(fake):
    store, _, _, _, request = world()
    s = socket.socket()
    s.bind(("127.0.0.1", 0))  # bound, not listening: every connection is refused
    try:
        (r,) = HTTPBackend(f"http://127.0.0.1:{s.getsockname()[1]}", sleep=lambda x: None).execute([request], store)
    finally:
        s.close()
    assert r.status == "error" and thaw(r.transport) == {"attempts": 3, "reasons": {"refused": 3}, "gave_up": True}
    hung = fake(hang=True)
    (r,) = HTTPBackend(hung.url, timeout_s=0.2, deadline_s=5.0, sleep=lambda x: None).execute([request], store)
    assert thaw(r.transport) == {"attempts": 3, "reasons": {"timeout": 3}, "gave_up": True}
    (r,) = HTTPBackend(hung.url, timeout_s=0.3, sleep=lambda x: None).execute([request], store)
    assert thaw(r.transport) == {"attempts": 1, "reasons": {"timeout": 1}, "gave_up": True}  # it spent the deadline
    slept = []
    (r,) = HTTPBackend(fake(resets=FOREVER).url, deadline_s=1.0, backoff_s=10.0, sleep=slept.append,
                       rng=Mid()).execute([request], store)
    assert thaw(r.transport) == {"attempts": 1, "reasons": {"reset": 1}, "gave_up": True} and not slept
    with pytest.raises(ValueError):
        HTTPBackend(hung.url, attempts=0)


def seal_world():
    """A registry that switched to a version re-executing every member, and one group of two members."""
    store, adapter, registry = MemoryStore(), ToyAdapter(), MemoryRegistry()
    c1 = check("t", "plain", "ab", store)
    g1 = GraderVersion(name="g1", entries={"t": c1})
    registry.publish_grader(g1, source="startup")
    registry.publish_grader(GraderVersion(name="g2", entries={"t": check("t", "up", "AB", store)}), source="switch",
                            require_coverage=False)
    members = []
    for i, answer in enumerate(["ab", "ba"]):
        record = adapter.record(store, task="t", trajectory=f"x{i}", answer=answer,
                                runs=[adapter.run(store, c1, answer)])
        recorded = grade_recorded(adapter, record, c1, store, grader_version=g1.version_id)
        members.append(Member(member_id=f"m{i}", execution=record, prior=recorded,
                              recorded_reward=1.0 if recorded.passed else 0.0))
    return store, adapter, registry, Candidate(group_id="u0", members=tuple(members))


def test_the_seal_keeps_a_group_whose_resets_were_retried_and_records_them(fake):
    store, adapter, registry, group = seal_world()
    service = fake(resets=2)
    session = SealSession(registry, Scorer([adapter], store, HTTPBackend(service.url, batch_size=1,
                                                                         sleep=lambda s: None)))
    (decided,) = session.decide([group])
    decision = session.finish([decided])
    assert not decided.neutralized and decided.rewards() == [1.0, 0.0] and len(service.posts) == 4
    assert thaw(decision.stats["transport"]) == {"requests": 2, "attempts": 4, "retried": 1, "gave_up": 0,
                                                 "reasons": {"reset": 2}}
    assert SealDecision.from_dict(json.loads(json.dumps(decision.to_dict()))) == decision


def test_the_seal_neutralizes_the_group_when_the_transport_gives_up(fake):
    store, adapter, registry, group = seal_world()
    service = fake(resets=FOREVER)
    session = SealSession(registry, Scorer([adapter], store, HTTPBackend(service.url, sleep=lambda s: None)))
    (decided,) = session.decide([group])
    decision = session.finish([decided])
    assert decided.neutralized and decided.neutralize_reason == "undeterminable_member" and len(service.posts) == 3
    assert [m.action for m in decided.members] == ["undeterminable", "undeterminable"]
    assert thaw(decision.stats["transport"]) == {"requests": 2, "attempts": 6, "retried": 2, "gave_up": 2,
                                                 "reasons": {"reset": 6}}
    store, adapter, registry, group = seal_world()  # nothing over a transport: no transport stats
    local = SealSession(registry, Scorer([adapter], store, ToyBackend()))
    assert "transport" not in local.finish(local.decide([group])).stats


def test_transport_is_recorded_and_a_replay_does_not_retry_a_give_up(fake, tmp_path):
    store, adapter, item, g2, request = world()
    recorder = RecordingBackend(HTTPBackend(fake(resets=FOREVER).url, sleep=lambda s: None), tmp_path / "res.jsonl")
    (live,) = Scorer([adapter], store, recorder, retries=2).score([item], g2)
    (line,) = [json.loads(x) for x in (tmp_path / "res.jsonl").read_text().splitlines()]
    assert line["transport"] == {"attempts": 3, "reasons": {"reset": 3}, "gave_up": True}
    replay = RecordedBackend.from_jsonl(tmp_path / "res.jsonl")
    (again,) = Scorer([adapter], store, replay, retries=2).score([item], g2)
    assert again.score.parts[0].attempts == live.score.parts[0].attempts == 1 and replay.served == [request.request_id]
    assert thaw(again.transport) == thaw(live.transport)
    result = ExecResult.from_dict(line)
    bare = dataclasses.replace(result, transport={})
    assert result == bare and ExecResult.from_dict(result.to_dict()).transport == result.transport
    assert "transport" not in bare.to_dict()  # a result no transport carried keeps its earlier form
