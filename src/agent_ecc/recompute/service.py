# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reference re-verification service: re-execution over HTTP, and its client backend.

The service wraps any :class:`agent_ecc.recompute.base.Backend` (local Docker for SWE-bench, the AppWorld evaluator,
a recorded replay) behind two endpoints, using only the standard library::

    GET  /v1/health   -> {"ok": true, "served": n, "backend": name, "workers": n}
    POST /v1/execute  {"requests": [ExecRequest...], "objects": {digest: base64}}
                      -> {"results": [ExecResult...], "objects": {digest: base64}, "seconds": s}

The client (:class:`HTTPBackend`) sends the input objects each request names (unless the service already holds
them), verifies every returned object against its digest, and stores it locally; a transport failure is an
infrastructure error for every request, never a verdict. ``workers`` bounds the batches the service executes at once.

    python -m agent_ecc.recompute.service --port 8770 --store DIR --backend swebench-docker --workers 8
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..canonical import digest_bytes
from ..store import ContentStore, DirectoryStore, MemoryStore
from .base import Backend, ExecRequest, ExecResult, RecordedBackend, infra_result

__all__ = ["HTTPBackend", "ReferenceService", "main", "make_server"]

MAX_BODY = 512 * 1024 * 1024


class ReferenceService:
    """Executes posted batches with ``backend`` against ``store`` (objects uploaded by clients are added to it)."""

    def __init__(self, backend: Backend, store: ContentStore | None = None, *, workers: int = 4,
                 name: str | None = None) -> None:
        self.backend = backend
        self.store = store if store is not None else MemoryStore()
        self.workers = max(1, int(workers))
        self.name = name or type(backend).__name__
        self._slots = threading.BoundedSemaphore(self.workers)
        self._lock = threading.Lock()
        self.served = 0

    def health(self) -> dict:
        return {"ok": True, "served": self.served, "backend": self.name, "workers": self.workers}

    def handle(self, body: dict) -> dict:
        t0 = time.time()
        for digest, data in (body.get("objects") or {}).items():
            raw = base64.b64decode(data)
            if digest_bytes(raw) != digest:
                raise ValueError(f"uploaded object does not match its digest {digest}")
            self.store.put(raw)
        requests = [ExecRequest.from_dict(r) for r in body.get("requests") or []]
        with self._slots:
            results = list(self.backend.execute(requests, self.store))
        if len(results) != len(requests):
            raise RuntimeError("backend returned the wrong number of results")
        objects = {}
        for result in results:
            for ref in result.outputs:
                if ref.digest not in objects:
                    objects[ref.digest] = base64.b64encode(self.store.get(ref.digest)).decode("ascii")
        with self._lock:
            self.served += len(requests)
        return {"results": [r.to_dict() for r in results], "objects": objects, "seconds": time.time() - t0}


def make_server(service: ReferenceService, host: str = "127.0.0.1", port: int = 8770) -> ThreadingHTTPServer:
    """An HTTP server for ``service`` (call ``serve_forever()``; port 0 picks a free port)."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") in ("/v1/health", "/health"):
                self._send(200, service.health())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/execute":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    raise ValueError("request body too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                self._send(200, service.handle(body))
            except Exception as error:  # noqa: BLE001
                self._send(500, {"error": f"{type(error).__name__}: {error}"[:500]})

        def log_message(self, fmt, *args):  # quiet by default
            return

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


class HTTPBackend:
    """A backend that sends requests to a reference service.

    ``send_objects`` uploads the input objects requests name (set False when the service shares the store).
    ``batch_size`` splits large batches into several posts."""

    def __init__(self, url: str, *, timeout_s: float = 3600.0, send_objects: bool = True,
                 batch_size: int = 64) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.send_objects = send_objects
        self.batch_size = max(1, int(batch_size))

    def health(self, timeout_s: float = 5.0) -> dict:
        with urllib.request.urlopen(f"{self.url}/v1/health", timeout=timeout_s) as response:
            return json.loads(response.read())

    def execute(self, requests: Sequence[ExecRequest], store: ContentStore) -> list[ExecResult]:
        out: list[ExecResult] = []
        for i in range(0, len(requests), self.batch_size):
            out.extend(self._post(list(requests[i:i + self.batch_size]), store))
        return out

    def _post(self, requests: list[ExecRequest], store: ContentStore) -> list[ExecResult]:
        t0 = time.time()
        try:
            objects = {}
            if self.send_objects:
                for request in requests:
                    for digest in request.inputs.values():
                        digest = str(digest)
                        if digest not in objects and store.contains(digest):
                            objects[digest] = base64.b64encode(store.get(digest)).decode("ascii")
            body = json.dumps({"requests": [r.to_dict() for r in requests], "objects": objects}).encode("utf-8")
            req = urllib.request.Request(f"{self.url}/v1/execute", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read()[:300].decode("utf-8", "replace")
            return [infra_result(r, f"service: HTTP {error.code}: {detail}", seconds=time.time() - t0)
                    for r in requests]
        except Exception as error:  # noqa: BLE001 - transport failures are infrastructure errors
            return [infra_result(r, f"service: {type(error).__name__}: {error}", seconds=time.time() - t0)
                    for r in requests]
        returned = payload.get("objects") or {}
        results = []
        for request, d in zip(requests, payload.get("results") or []):
            result = ExecResult.from_dict(d)
            try:
                if result.request_id != request.request_id:
                    raise ValueError("the service answered another request")
                for ref in result.outputs:
                    if not store.contains(ref.digest):
                        raw = base64.b64decode(returned[ref.digest])
                        if digest_bytes(raw) != ref.digest:
                            raise ValueError(f"returned object does not match its digest {ref.digest}")
                        store.put(raw)
            except Exception as error:  # noqa: BLE001
                result = infra_result(request, f"service: {error}")
            results.append(result)
        if len(results) != len(requests):
            results.extend(infra_result(r, "service: no result") for r in requests[len(results):])
        return results


def _backend_from_args(args: argparse.Namespace) -> Backend:
    if args.backend == "swebench-docker":
        from .docker import LocalDockerBackend

        return LocalDockerBackend(workers=args.workers, timeout_s=args.timeout)
    if args.backend == "appworld":
        from ..adapters.appworld import AppWorldEvaluatorBackend

        dirs = dict(item.split("=", 1) for item in args.appworld_data)
        return AppWorldEvaluatorBackend(dirs, workers=args.workers)
    if args.backend == "recorded":
        return RecordedBackend.from_jsonl(args.recorded)
    raise SystemExit(f"unknown backend {args.backend!r}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent_ecc.recompute.service", description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--store", default=None, help="a directory store for objects (default: in memory)")
    ap.add_argument("--backend", choices=("swebench-docker", "appworld", "recorded"), required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=1800.0, help="eval-script timeout (swebench-docker)")
    ap.add_argument("--appworld-data", action="append", default=[], metavar="VERSION=DIR")
    ap.add_argument("--recorded", default=None, help="JSON-lines of recorded results (recorded)")
    args = ap.parse_args(argv)
    store = DirectoryStore(args.store) if args.store else MemoryStore()
    service = ReferenceService(_backend_from_args(args), store, workers=args.workers, name=args.backend)
    server = make_server(service, args.host, args.port)
    print(f"[agent-ecc] re-verification service on {args.host}:{server.server_address[1]} "
          f"({args.backend}, {args.workers} workers)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
