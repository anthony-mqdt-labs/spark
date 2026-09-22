"""Filtering proxy: /v1/models answers from spark's roster, all else passes through.

Pins the harness-picker complaint: mlx_lm.server has no --model-discovery, so
its listing advertises every MLX-shaped hub repo (embedders, TTS, partials).
The proxy binds the published port, spawns the real server as its child, and
filters only GET /v1/models — to the served id plus registered ∩ present
mlx-servable ids. Backend death (or persistent unhealth) exits the proxy so
the supervisor restarts the unit; the child is always reaped first.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from spark.config.schema import ModelEntry


def _proxy():
    from spark.shims import openai_filter_proxy as proxy

    return proxy


def _entries():
    return [
        (ModelEntry(id="reg-hub", hf_repo="org/Hub", model_format="mlx",
                    backend="mlx_lm"), object()),
        (ModelEntry(id="reg-store", path="/w/store", model_format="mlx",
                    backend="mlx_lm"), object()),
        (ModelEntry(id="pending", model_format="any", backend=""), object()),
        (ModelEntry(id="gguf", hf_repo="org/G", model_format="gguf",
                    backend="llama_cpp"), object()),
        (ModelEntry(id="prism", hf_repo="org/P", model_format="prism",
                    backend="prism_ml"), object()),
    ]


# Unregistered-but-present weights: downloaded, not registered → excluded
# (the endpoint lists the registered ∩ downloaded intersection).
discovered = ["org/Unregistered-8bit"]


class _Inv:
    def __init__(self, entries):
        self.runnable_registered = entries
        self.runnable_discovered = discovered


def test_allowed_ids_is_served_first_then_runnable_mlx(monkeypatch):
    import spark.inventory

    monkeypatch.setattr(
        spark.inventory, "build_inventory", lambda *a, **k: _Inv(_entries())
    )
    ids = _proxy().allowed_ids("served-model")
    assert ids[0] == "served-model"
    # Hub repo id and store path both addressable; unknown-format and
    # non-mlx formats excluded; unregistered never listed.
    assert "org/Hub" in ids
    assert "/w/store" in ids
    assert "pending" in ids  # format "any" counts as servable
    assert "org/Unregistered-8bit" not in ids
    assert "org/G" not in ids
    assert "org/P" not in ids


def test_allowed_ids_dedupes_served(monkeypatch):
    import spark.inventory

    monkeypatch.setattr(
        spark.inventory, "build_inventory", lambda *a, **k: _Inv(_entries())
    )
    ids = _proxy().allowed_ids("org/Hub")
    assert ids.count("org/Hub") == 1 and ids[0] == "org/Hub"


def test_allowed_ids_degrades_to_served_only(monkeypatch):
    import spark.inventory

    def _boom(*a, **k):
        raise OSError("no disk")

    monkeypatch.setattr(spark.inventory, "build_inventory", _boom)
    assert _proxy().allowed_ids("served-model") == ["served-model"]


def test_models_request_matching():
    proxy = _proxy()

    class H:
        def __init__(self, command, path):
            self.command = command
            self.path = path

    assert proxy._is_models_request(H("GET", "/v1/models"))
    assert proxy._is_models_request(H("GET", "/v1/models?x=1"))
    assert proxy._is_models_request(H("GET", "/v1/models/"))
    assert not proxy._is_models_request(H("POST", "/v1/models"))
    assert not proxy._is_models_request(H("GET", "/v1/chat/completions"))
    assert not proxy._is_models_request(H("GET", "/health"))


# --- adapter ------------------------------------------------------------------
def test_proxied_launch_shape(config, tmp_path):
    from spark.runtimes import get_backend

    assert config.runtimes["mlx_lm"].server.filter_models is True
    entry = ModelEntry(id="m", hf_repo="org/M", model_format="mlx")
    cmd = get_backend("mlx_lm", config).build_launch_cmd(entry, "127.0.0.1", 8095)
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("openai_filter_proxy.py")
    assert ["--host", "127.0.0.1", "--port", "8095"] == cmd[2:6]
    assert "--served-id" in cmd and "org/M" in cmd
    sep = cmd.index("--")
    inner = cmd[sep + 1:]
    assert inner[0] == "mlx_lm.server"
    assert "--model" in inner and "org/M" in inner
    # Backend binds loopback-ephemeral, never the published port.
    assert "--host" in inner
    assert inner[inner.index("--host") + 1] == "127.0.0.1"
    upstream = cmd[cmd.index("--upstream") + 1]
    assert upstream.startswith("http://127.0.0.1:")
    assert not upstream.endswith(":8095")


def test_unfiltered_launch_unchanged(config):
    from spark.runtimes import get_backend

    rt = config.runtimes["mlx_lm"]
    rt.server.filter_models = False
    entry = ModelEntry(id="m", hf_repo="org/M", model_format="mlx")
    cmd = get_backend("mlx_lm", config).build_launch_cmd(entry, "127.0.0.1", 8095)
    assert cmd[0] == "mlx_lm.server"
    assert "--port" in cmd and "8095" in cmd


def test_filter_models_defaults_off():
    from spark.config.schema import ServerSpec

    assert ServerSpec().filter_models is False


# --- live stub integration ------------------------------------------------------
_STUB = """\
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

JUNK = {"object": "list", "data": [
    {"id": "BAAI/bge-small", "object": "model"},
    {"id": "mlx-community/Some-TTS", "object": "model"},
]}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path in ("/v1/models", "/health"):
            self._send(200, JUNK if self.path == "/v1/models" else {"ok": True})
        else:
            self._send(404, {})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b"{}"
        self._send(200, {"echo_path": self.path, "echo": json.loads(body)})

ThreadingHTTPServer(("127.0.0.1", {port}), H).serve_forever()
"""


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run_proxy(tmp_path, served_id, extra_args=None):
    stub_port, pub_port = _free_port(), _free_port()
    stub = tmp_path / "stub_backend.py"
    stub.write_text(_STUB.replace("{port}", str(stub_port)))
    shim = (
        Path(__import__("spark.shims.openai_filter_proxy", fromlist=["x"]).__file__)
    )
    cmd = [
        sys.executable, str(shim),
        "--host", "127.0.0.1", "--port", str(pub_port),
        "--upstream", f"http://127.0.0.1:{stub_port}",
        "--served-id", served_id,
        "--ready-timeout-s", "20",
        "--unhealthy-exit-after", "1",
        "--monitor-interval-s", "0.2",
        *(extra_args or []),
        "--",
        sys.executable, str(stub),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    base = f"http://127.0.0.1:{pub_port}"
    for _ in range(200):
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=2) as r:
                if r.status == 200:
                    break
        except OSError:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"proxy died during startup: {proc.communicate()[0][-2000:]}")
        __import__("time").sleep(0.1)
    else:
        proc.terminate()
        raise RuntimeError("proxy never became ready")
    return proc, base, stub_port


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return r.status, json.loads(r.read())


def test_proxy_filters_listing_but_passes_through(tmp_path, monkeypatch):
    """Hermetic end to end: the proxy subprocess inherits a seeded SPARK_HOME
    registry + HF hub cache, so its listing is fully determined by the test."""
    import json as _json

    from spark.config.paths import SparkPaths
    from spark.registry import save_model

    home = tmp_path / "sparkhome"
    paths = SparkPaths(config_dir=home / "config", data_dir=home / "data",
                       cache_dir=home / "cache").ensure()
    save_model(ModelEntry(id="reg-hub", hf_repo="org/Hub", model_format="mlx",
                          backend="mlx_lm"), paths)
    hub = tmp_path / "hfcache"
    snap = hub / "models--org--Hub" / "snapshots" / ("b" * 40)
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(_json.dumps({"model_type": "qwen3"}))
    (snap / "model.safetensors").write_bytes(b"w" * 16)
    (hub / "models--org--Hub" / "refs").mkdir(parents=True)
    (hub / "models--org--Hub" / "refs" / "main").write_text("b" * 40)
    monkeypatch.setenv("SPARK_HOME", str(home))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    proc, base, _stub_port = _run_proxy(tmp_path, "served-model")
    try:
        status, payload = _get(base, "/v1/models")
        assert status == 200
        ids = [m["id"] for m in payload["data"]]
        # Served first, then the seeded registered ∩ downloaded entry; the
        # stub's junk (embeddings/TTS) is gone.
        assert ids == ["served-model", "org/Hub"]
        # Passthrough: POST body echoed by the stub backend verbatim.
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps({"model": "x", "messages": []}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            echoed = json.loads(r.read())
        assert echoed["echo_path"] == "/v1/chat/completions"
        assert echoed["echo"]["model"] == "x"
        # Health (non-models GET) passes through to the stub.
        status, _ = _get(base, "/health")
        assert status == 200
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def _listeners(port: int) -> list[str]:
    out = subprocess.run(
        ["lsof", "-tiTCP:127.0.0.1", "-sTCP:LISTEN"],
        capture_output=True, text=True,
    )
    # lsof has no per-port listen filter together with -t on all macOS
    # versions; match the port via a second pass.
    holders = subprocess.run(
        ["lsof", "-nP", "-sTCP:LISTEN"],
        capture_output=True, text=True,
    )
    pids = []
    for line in holders.stdout.splitlines():
        if f"127.0.0.1:{port}" in line or f"*:{port}" in line:
            parts = line.split()
            if len(parts) > 1 and parts[1].isdigit():
                pids.append(parts[1])
    return pids


def test_proxy_exits_and_reaps_child_when_backend_dies(tmp_path):
    proc, base, stub_port = _run_proxy(tmp_path, "served-model")
    try:
        stub_pids = _listeners(stub_port)
        assert stub_pids, "stub backend must hold its listening socket"
        for pid in stub_pids:
            subprocess.run(["kill", "-9", pid])
        # Monitor (0.2s interval, exit-after-1) must take the proxy down fast,
        # reaping the child rather than orphaning it.
        rc = proc.wait(timeout=15)
        assert rc != 0
        assert _listeners(stub_port) == [], "orphaned backend still listening"
    finally:
        if proc.poll() is None:
            proc.kill()
