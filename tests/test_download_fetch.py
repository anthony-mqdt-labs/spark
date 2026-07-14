"""Download hardening: guaranteed terminal telemetry, projected-size disk
preflight, and partial/orphan store detection (post-incident 2026-07-13)."""

from __future__ import annotations

import json
import signal
import subprocess

import pytest

from spark.cli import download as dl
from spark.cli.context import SparkCtx
from spark.config.loader import load_config
from spark.config.schema import ModelEntry
from spark.errors import InsufficientDiskError, SparkError
from spark.registry import save_model
from spark.registry.store import scan_store
from spark.telemetry import Telemetry


@pytest.fixture
def ctx(paths) -> SparkCtx:
    return SparkCtx(
        paths=paths,
        config=load_config(),
        secrets=None,  # not used by the fetch/preflight paths under test
        telemetry=Telemetry(paths.log_dir, session_id="test"),
        session_id="test",
    )


def _events(paths) -> list[dict]:
    out = []
    for f in sorted((paths.log_dir / "download").glob("*.jsonl")):
        out += [json.loads(line) for line in f.read_text().splitlines()]
    return out


def _terminal(paths) -> list[dict]:
    names = ("fetch_complete", "fetch_failed", "fetch_interrupted")
    return [e for e in _events(paths) if e["event"] in names]


# --- terminal telemetry on every fetch exit path ---------------------------------
def test_fetch_complete_on_success(ctx, monkeypatch):
    dest = ctx.paths.model_store_dir / "m"

    def fake_run(cmd, check, env):
        dest.mkdir(parents=True)
        (dest / "weights.bin").write_bytes(b"x" * 1024)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(dl.subprocess, "run", fake_run)
    dl._run_fetch(ctx, "hf", "org/repo", "m", dest, None)
    (event,) = _terminal(ctx.paths)
    assert event["event"] == "fetch_complete"
    assert event["bytes_fetched"] == 1024


def test_fetch_failed_on_nonzero_exit(ctx, monkeypatch):
    def fake_run(cmd, check, env):
        raise subprocess.CalledProcessError(2, cmd)

    monkeypatch.setattr(dl.subprocess, "run", fake_run)
    with pytest.raises(SparkError):
        dl._run_fetch(ctx, "hf", "org/repo", "m", ctx.paths.model_store_dir / "m", None)
    (event,) = _terminal(ctx.paths)
    assert event["event"] == "fetch_failed"
    assert event["rc"] == 2


def test_fetch_interrupted_on_ctrl_c(ctx, monkeypatch):
    def fake_run(cmd, check, env):
        raise KeyboardInterrupt

    monkeypatch.setattr(dl.subprocess, "run", fake_run)
    with pytest.raises(KeyboardInterrupt):
        dl._run_fetch(ctx, "hf", "org/repo", "m", ctx.paths.model_store_dir / "m", None)
    (event,) = _terminal(ctx.paths)
    assert event["event"] == "fetch_interrupted"
    assert event["signal"] == "SIGINT"


def test_fetch_interrupted_on_sigterm(ctx, monkeypatch):
    def fake_run(cmd, check, env):
        raise dl._FetchInterrupt(signal.SIGTERM)

    monkeypatch.setattr(dl.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        dl._run_fetch(ctx, "hf", "org/repo", "m", ctx.paths.model_store_dir / "m", None)
    assert exc.value.code == 128 + signal.SIGTERM
    (event,) = _terminal(ctx.paths)
    assert event["event"] == "fetch_interrupted"
    assert event["signal"] == "SIGTERM"


def test_fetch_failed_on_unexpected_error(ctx, monkeypatch):
    def fake_run(cmd, check, env):
        raise OSError("disk gone")

    monkeypatch.setattr(dl.subprocess, "run", fake_run)
    with pytest.raises(OSError):
        dl._run_fetch(ctx, "hf", "org/repo", "m", ctx.paths.model_store_dir / "m", None)
    (event,) = _terminal(ctx.paths)
    assert event["event"] == "fetch_failed"
    assert event["error_type"] == "OSError"


# --- projected-size disk preflight ------------------------------------------------
GIB = 2**30


def _patch_disk(monkeypatch, *, free: int, repo_size: int | None):
    import spark.budget as budget

    monkeypatch.setattr(dl, "_repo_size_bytes", lambda repo, token=None: repo_size)
    monkeypatch.setattr(dl, "free_disk_bytes", lambda p: free)
    monkeypatch.setattr(budget, "free_disk_bytes", lambda p: free)


def test_preflight_blocks_when_projected_crosses_floor(ctx, monkeypatch):
    # 11 GiB free, 5 GiB needed, 10 GiB floor -> 6 GiB left -> refuse.
    _patch_disk(monkeypatch, free=11 * GIB, repo_size=5 * GIB)
    with pytest.raises(InsufficientDiskError):
        dl._disk_preflight(ctx, "org/repo", ctx.paths.model_store_dir / "m", None, force=False)


def test_preflight_allows_small_fetch(ctx, monkeypatch):
    _patch_disk(monkeypatch, free=11 * GIB, repo_size=GIB // 2)
    dl._disk_preflight(ctx, "org/repo", ctx.paths.model_store_dir / "m", None, force=False)


def test_preflight_force_bypasses_gate(ctx, monkeypatch):
    _patch_disk(monkeypatch, free=11 * GIB, repo_size=5 * GIB)
    dl._disk_preflight(ctx, "org/repo", ctx.paths.model_store_dir / "m", None, force=True)
    assert any(e["event"] == "disk_gate_forced" for e in _events(ctx.paths))


def test_preflight_credits_resumable_bytes(ctx, monkeypatch):
    # 4.5 of 5 GiB already on disk -> need 0.5 GiB -> passes the same gate.
    dest = ctx.paths.model_store_dir / "m"
    _patch_disk(monkeypatch, free=11 * GIB, repo_size=5 * GIB)
    monkeypatch.setattr(dl, "_dir_bytes", lambda p: int(4.5 * GIB))
    dl._disk_preflight(ctx, "org/repo", dest, None, force=False)


def test_preflight_unknown_size_falls_back_to_floor(ctx, monkeypatch):
    _patch_disk(monkeypatch, free=11 * GIB, repo_size=None)
    dl._disk_preflight(ctx, "org/repo", ctx.paths.model_store_dir / "m", None, force=False)
    _patch_disk(monkeypatch, free=9 * GIB, repo_size=None)
    with pytest.raises(InsufficientDiskError):
        dl._disk_preflight(ctx, "org/repo", ctx.paths.model_store_dir / "m", None, force=False)


def test_repo_size_none_when_api_omits_sizes(monkeypatch):
    import io
    import urllib.request

    body = json.dumps({"siblings": [{"rfilename": "a", "size": None}]}).encode()

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: FakeResp(body))
    assert dl._repo_size_bytes("org/repo") is None


def test_repo_size_sums_siblings(monkeypatch):
    import io
    import urllib.request

    body = json.dumps(
        {"siblings": [{"rfilename": "a", "size": 100}, {"rfilename": "b", "size": 23}]}
    ).encode()

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: FakeResp(body))
    assert dl._repo_size_bytes("org/repo") == 123


# --- store scan: partial + orphaned dirs -------------------------------------------
def test_scan_store_flags_partial_and_orphan(paths):
    store = paths.model_store_dir

    healthy = store / "healthy"
    healthy.mkdir(parents=True)
    (healthy / "model.safetensors").write_bytes(b"w")
    save_model(ModelEntry(id="healthy", hf_repo="org/healthy", path=str(healthy)), paths)

    partial = store / "partial"
    frag_dir = partial / ".cache" / "huggingface" / "download"
    frag_dir.mkdir(parents=True)
    (frag_dir / "blob.incomplete").write_bytes(b"x" * 10)
    save_model(ModelEntry(id="partial", hf_repo="org/partial", path=str(partial)), paths)

    orphan = store / "orphan"
    orphan.mkdir(parents=True)
    (orphan / "config.json").write_text("{}")

    issues = {i.model_id: i for i in scan_store(paths)}
    assert set(issues) == {"partial", "orphan"}
    assert issues["partial"].kind == "partial"
    assert issues["partial"].registered
    assert issues["partial"].hf_repo == "org/partial"
    assert issues["orphan"].kind == "orphaned"
    assert not issues["orphan"].registered
