"""Hardening regressions, all at in-process seams: no real server, no network.

* the proxy must answer ``GET /v1/models`` from a cached roster, not rebuild an
  inventory (and walk bytes) on every harness poll;
* every health probe must drain and close its connection;
* ``spark __complete`` must reuse the inventory it already built and cache a
  registry-miss preview instead of rebuilding it on every preview keystroke.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.parse
import urllib.request
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from spark.config.schema import ModelEntry


def _proxy():
    from spark.shims import openai_filter_proxy as proxy

    return proxy


def _fake_inv():
    """One runnable registered entry — enough to exercise roster filtering."""
    return SimpleNamespace(
        runnable_registered=[
            (
                SimpleNamespace(id="reg", hf_repo="org/Reg", path="", model_format="mlx"),
                object(),
            )
        ],
        runnable_discovered=[],
    )


@pytest.fixture
def proxy_state(monkeypatch):
    proxy = _proxy()
    monkeypatch.setattr(proxy, "_STATE", {})
    monkeypatch.setattr(proxy, "_MODELS_CACHE", {"ids": None, "built_at": 0.0})
    return proxy


def _count_inventory(monkeypatch, factory=_fake_inv):
    import spark.inventory

    calls = []

    def _build(*a, **k):
        calls.append(k.get("with_bytes", True))
        return factory()

    monkeypatch.setattr(spark.inventory, "build_inventory", _build)
    return calls


# --- 1. roster cache -----------------------------------------------------------
def test_models_payload_builds_inventory_once_until_ttl(proxy_state, monkeypatch):
    proxy = proxy_state
    calls = _count_inventory(monkeypatch)
    proxy._STATE["served_id"] = "served"

    first = proxy.models_payload()
    second = proxy.models_payload()

    assert [m["id"] for m in first["data"]] == ["served", "org/Reg"]
    assert second == first
    assert len(calls) == 1, "two consecutive model-list requests must build once"

    proxy._MODELS_CACHE["built_at"] -= proxy.ALLOWED_IDS_TTL_S + 1
    third = proxy.models_payload()
    assert third == first
    assert len(calls) == 2, "expiry must trigger exactly one refresh"


def test_refresh_allowed_ids_primes_the_cache(proxy_state, monkeypatch):
    proxy = proxy_state
    calls = _count_inventory(monkeypatch)
    proxy._STATE["served_id"] = "served"

    assert proxy._refresh_allowed_ids()[0] == "served"
    assert len(calls) == 1
    assert [m["id"] for m in proxy.models_payload()["data"]] == ["served", "org/Reg"]
    assert len(calls) == 1, "the ready-time prime must serve later requests"


def test_models_request_is_matched_once_per_request(proxy_state, monkeypatch):
    proxy = proxy_state
    proxy._STATE["upstream_ready"] = True
    proxy._MODELS_CACHE["ids"] = ["served"]
    proxy._MODELS_CACHE["built_at"] = time.monotonic()

    seen = []
    original = proxy._is_models_request

    def _count(handler):
        seen.append(handler.path)
        return original(handler)

    monkeypatch.setattr(proxy, "_is_models_request", _count)

    class H:
        command = "GET"
        path = "/v1/models"
        sent = []

        def _send_json(self, code, payload):
            self.sent.append((code, payload))

    handler = H()
    proxy._Handler._handle(handler)
    assert handler.sent[0][0] == 200
    assert seen == ["/v1/models"]


# --- 2. health probe cleanup ---------------------------------------------------
def _install_fake_connection(monkeypatch, status=200):
    proxy = _proxy()
    events = []

    class Resp:
        def __init__(self):
            self.status = status

        def read(self):
            events.append("resp.read")
            return b"{}"

        def close(self):
            events.append("resp.close")

    class Conn:
        def __init__(self, host, port, timeout=None):
            events.append("conn.init")

        def request(self, method, path):
            events.append("conn.request")

        def getresponse(self):
            return Resp()

        def close(self):
            events.append("conn.close")

    monkeypatch.setattr(proxy.http.client, "HTTPConnection", Conn)
    return proxy, events


def test_probe_drains_and_closes_connection(monkeypatch):
    proxy, events = _install_fake_connection(monkeypatch)
    parts = urllib.parse.urlparse("http://127.0.0.1:9")
    assert proxy._probe_health(parts, "/health") is True
    assert events == ["conn.init", "conn.request", "resp.read", "resp.close", "conn.close"]


def test_probe_closes_even_when_unhealthy(monkeypatch):
    proxy, events = _install_fake_connection(monkeypatch, status=503)
    parts = urllib.parse.urlparse("http://127.0.0.1:9")
    assert proxy._probe_health(parts, "/health") is False
    assert events.count("conn.close") == 1
    assert events.count("resp.close") == 1


def test_wait_ready_closes_its_probe_connection(monkeypatch):
    proxy, events = _install_fake_connection(monkeypatch)
    assert proxy._wait_ready("http://127.0.0.1:9", "/health", 5.0) is True
    assert events.count("conn.close") == 1


# --- 3. completion reuse -------------------------------------------------------
def test_admin_completion_reuses_built_inventory(paths, monkeypatch):
    from spark.cli import completion
    import spark.inventory
    from spark.registry import save_model

    save_model(ModelEntry(id="a-1", hf_repo="org/A", model_format="mlx"), paths)
    monkeypatch.setattr(completion, "resolve_paths", lambda: paths)

    builds = []
    real_build = spark.inventory.build_inventory

    def _build(*a, **k):
        builds.append(1)
        return real_build(*a, **k)

    monkeypatch.setattr(spark.inventory, "build_inventory", _build)
    relists = []
    real_list = completion.list_models

    def _list(*a, **k):
        relists.append(1)
        return real_list(*a, **k)

    monkeypatch.setattr(completion, "list_models", _list)

    result = CliRunner().invoke(completion.complete_command, ["models", "forget"])
    assert result.exit_code == 0
    assert "a-1" in result.output
    assert len(builds) == 1
    assert relists == [], "admin branch must reuse the inventory, not re-read TOMLs"


def _discovered_inv(location):
    return SimpleNamespace(
        runnable_registered=[],
        missing=[],
        external=[],
        unverifiable=[],
        runnable_discovered=[
            SimpleNamespace(
                display_id="org/Disco-8B",
                source="store",
                location=str(location),
                model_format="mlx",
                bytes_present=0,
                kind="llm",
                quant="q4",
            )
        ],
    )


def test_describe_discovered_is_cached_and_skips_full_byte_walk(paths, monkeypatch):
    from spark.cli import completion
    import spark.inventory

    weights = paths.model_store_dir / "org-Disco"
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "model.safetensors").write_bytes(b"x" * 1024)

    calls = []

    def _build(*a, **k):
        calls.append(k.get("with_bytes", True))
        return _discovered_inv(weights)

    monkeypatch.setattr(spark.inventory, "build_inventory", _build)
    monkeypatch.setattr(completion, "resolve_paths", lambda: paths)
    monkeypatch.setattr(completion, "_DESCRIBE_CACHE", {})

    first = CliRunner().invoke(completion.complete_command, ["describe", "org/Disco-8B"])
    second = CliRunner().invoke(completion.complete_command, ["describe", "org/Disco-8B"])
    assert first.exit_code == 0 and second.exit_code == 0
    assert "id:       org/Disco-8B" in first.output
    assert "adopt:    spark adopt org/Disco-8B" in first.output
    assert second.output == first.output
    assert calls == [False], "describe must avoid the byte walk and cache the preview"


def test_describe_cache_invalidated_by_snapshot_mtime(paths, monkeypatch):
    """An in-place snapshot swap must bust the cache even though roots did not change."""
    import os

    from spark.cli import completion
    import spark.inventory

    weights = paths.model_store_dir / "org-Disco"
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "model.safetensors").write_bytes(b"x" * 1024)

    calls = []

    def _build(*a, **k):
        calls.append(k.get("with_bytes", True))
        inv = _discovered_inv(weights)
        if len(calls) > 1:  # the rescanned snapshot has different bytes
            inv.runnable_discovered[0].bytes_present = 2 * 2**30
        return inv

    monkeypatch.setattr(spark.inventory, "build_inventory", _build)
    monkeypatch.setattr(completion, "resolve_paths", lambda: paths)
    monkeypatch.setattr(completion, "_DESCRIBE_CACHE", {})

    def _describe():
        return CliRunner().invoke(completion.complete_command, ["describe", "org/Disco-8B"])

    first = _describe()
    assert first.exit_code == 0
    assert "0.0 GiB" in first.output
    assert _describe().output == first.output
    assert calls == [False], "unchanged location must stay cached"

    st = weights.stat()  # roots are untouched; the snapshot itself was swapped
    os.utime(weights, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    third = _describe()
    assert third.exit_code == 0
    assert "2.0 GiB" in third.output, "stale size survived an in-place snapshot swap"
    assert calls == [False, False]


def test_describe_registered_still_renders(paths, monkeypatch):
    from spark.cli import completion
    from spark.registry import save_model

    save_model(
        ModelEntry(id="reg-1", hf_repo="org/Reg", model_format="mlx", backend="mlx_lm"), paths
    )
    monkeypatch.setattr(completion, "resolve_paths", lambda: paths)
    result = CliRunner().invoke(completion.complete_command, ["describe", "reg-1"])
    assert result.exit_code == 0
    assert "id:       reg-1" in result.output


# --- 4. adopted listen fd (port TOCTOU) ----------------------------------------
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _drive_main(proxy, monkeypatch, port, *, reserved_fd=None):
    """Run main() in a thread with its external seams stubbed; return (binds, body)."""

    class _Child:
        def __init__(self):
            self.gone = False

        def poll(self):
            return 1 if self.gone else None

    child = _Child()
    monkeypatch.setattr(proxy, "_spawn_backend", lambda argv: child)
    monkeypatch.setattr(proxy, "_wait_ready", lambda *a: True)
    monkeypatch.setattr(proxy, "_monitor", lambda *a, **k: None)
    monkeypatch.setattr(proxy, "_terminate_child", lambda c: None)

    def _prime(sid):
        proxy._MODELS_CACHE["ids"] = ["served"]
        proxy._MODELS_CACHE["built_at"] = time.monotonic()
        return ["served"]

    monkeypatch.setattr(proxy, "_refresh_allowed_ids", _prime)

    binds = []
    real_bind = socket.socket.bind

    def _record_bind(self, addr):
        binds.append(addr)
        return real_bind(self, addr)

    monkeypatch.setattr(socket.socket, "bind", _record_bind)
    if reserved_fd is None:
        monkeypatch.delenv("SPARK_LISTEN_FD", raising=False)
    else:
        monkeypatch.setenv("SPARK_LISTEN_FD", str(reserved_fd))

    errors = []

    def _run():
        try:
            proxy.main(
                [
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--upstream",
                    "http://127.0.0.1:9",
                    "--served-id",
                    "served",
                    "--unhealthy-exit-after",
                    "1",
                    "--",
                    "/bin/true",
                ]
            )
        except BaseException as exc:  # noqa: BLE001 - report the thread failure
            errors.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    body = None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=0.5) as r:
                body = json.loads(r.read())
                break
        except OSError:
            time.sleep(0.02)
    child.gone = True
    thread.join(timeout=6)
    assert not thread.is_alive(), "proxy main() did not exit"
    assert not errors, errors
    return binds, body


def test_proxy_adopts_reserved_fd_without_binding(proxy_state, monkeypatch):
    """With SPARK_LISTEN_FD, the proxy adopts spark's listening socket: no bind."""
    from spark.runner.ports import reserve_port

    proxy = proxy_state
    port = _free_port()
    reservation = reserve_port("127.0.0.1", port)
    try:
        binds, body = _drive_main(proxy, monkeypatch, port, reserved_fd=reservation.fileno())
        reservation.detach()  # the adopted socket now owns the fd
    finally:
        reservation.close()
    assert binds == [], f"proxy re-bound the reserved port: {binds}"
    assert [m["id"] for m in body["data"]] == ["served"]


def test_proxy_binds_normally_without_reserved_fd(proxy_state, monkeypatch):
    """Back-compat: without the env var the proxy still binds the published port."""
    proxy = proxy_state
    port = _free_port()
    binds, body = _drive_main(proxy, monkeypatch, port)
    assert ("127.0.0.1", port) in binds
    assert [m["id"] for m in body["data"]] == ["served"]


# --- 5. cleanup behavior -------------------------------------------------------
def test_version_helper_and_empty_render_are_quiet(capsys):
    from spark.cli.context import spark_version
    from spark.cli.render import render_server_log, render_store_issues

    assert isinstance(spark_version(), str)
    render_store_issues([])
    render_server_log("x.log", "   ")
    assert capsys.readouterr().out == ""
