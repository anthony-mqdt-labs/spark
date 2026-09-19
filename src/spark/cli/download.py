"""`spark download <hf-repo>` — fetch + register a model.

Phase 1: performs the download (via the `hf` CLI), infers a sensible registry
entry, and marks research as pending. The multi-agent research chain that fills in
optimal per-runtime flags lands in Phase 3 (`spark.research`).

Fetch guarantees (post-incident, 2026-07-13):
- Every fetch emits exactly one terminal telemetry event — ``fetch_complete``,
  ``fetch_failed``, or ``fetch_interrupted`` — whatever kills it (SIGKILL aside).
- Disk preflight is projected-size aware: repo size comes from the HF API
  (stdlib urllib, no new deps), already-downloaded bytes are credited (hf
  resumes), and the gate is ``free - need >= floor``. ``--force`` bypasses it.
"""

from __future__ import annotations

import re
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import click

from ..budget import GIB, check_disk, free_disk_bytes
from ..config.schema import ModelEntry
from ..errors import InsufficientDiskError, SparkError
from ..registry import save_model
from ..registry.store import dir_bytes as _dir_bytes
from .context import SparkCtx, build_context, refresh_catalog
from .render import console

_QUANT_RE = re.compile(r"\b(q[2-8])(?:_[a-z0-9]+)?\b", re.I)
_BIT_RE = re.compile(r"\b([2-8])\s*-?bit\b", re.I)
_PARAM_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*b\b", re.I)


def _slug(repo: str) -> str:
    tail = repo.rstrip("/").split("/")[-1]
    tail = re.sub(r"[^A-Za-z0-9_.-]+", "-", tail).strip("-").lower()
    return tail or "model"


_VLM_RE = re.compile(r"\bvl\b|vision|llava|-vl-|qwen[0-9.]*-vl|internvl|pixtral", re.I)


def _infer(repo: str) -> dict:
    name = repo.lower()
    if "gguf" in name:
        fmt = "gguf"
    elif "mlx" in name:
        fmt = "mlx-vlm" if _VLM_RE.search(name) else "mlx"
    else:
        fmt = "any"
    quant_m = _QUANT_RE.search(name)
    bit_m = _BIT_RE.search(name)
    param_m = _PARAM_RE.search(name)
    if quant_m:
        quant = quant_m.group(1).lower()
    elif bit_m:
        quant = f"q{bit_m.group(1)}"   # '4bit' -> 'q4'
    else:
        quant = ""
    return {
        "model_format": fmt,
        "quant": quant,
        "params_billions": (float(param_m.group(1)) if param_m else None),
    }


# --- disk preflight --------------------------------------------------------------
def _repo_size_bytes(repo: str, token: str | None = None) -> int | None:
    """Total byte size of REPO's files from the HF API, or None if unknown
    (offline, gated without token, or the API omitted a size)."""
    import json
    import urllib.request

    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}?blobs=true",
        headers={"User-Agent": "spark-cli"},
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
    except Exception:
        return None
    siblings = data.get("siblings") or []
    sizes = [s.get("size") for s in siblings]
    if not sizes or any(sz is None for sz in sizes):
        return None
    return sum(sizes)


def _hf_token(ctx: SparkCtx) -> str | None:
    if not ctx.secrets.exists("hf_token"):
        return None
    return ctx.secrets.resolve_env({"HF_TOKEN": "hf_token"}).get("HF_TOKEN")


def _disk_preflight(
    ctx: SparkCtx, repo: str, dest: Path, token: str | None, *, force: bool
) -> None:
    """Gate the fetch on projected need vs free disk. Falls back to a floor-only
    check when the repo size cannot be determined."""
    store = ctx.paths.model_store_dir
    free = free_disk_bytes(store)
    floor = ctx.config.disk.min_free_gib
    total = _repo_size_bytes(repo, token)
    have = _dir_bytes(dest)
    need = max(total - have, 0) if total is not None else None

    ctx.telemetry.info(
        "download", "disk_preflight",
        repo=repo, free_bytes=free, floor_gib=floor,
        repo_bytes=total, resume_bytes=have, need_bytes=need,
    )

    if need is None:
        console.print(
            f"[yellow]![/yellow] repo size unknown (offline or gated) — "
            f"floor-only disk check. free {free / GIB:.1f} GiB · floor {floor:g} GiB"
        )
        blocked = free / GIB <= floor
        detail = f"Only {free / GIB:.1f} GiB free; at/below the {floor:g} GiB floor."
    else:
        chk = check_disk(store, need, ctx.config.disk)
        console.print(
            f"[bold]disk[/bold]: need {need / GIB:.2f} GiB "
            f"(repo {total / GIB:.2f} GiB − resumable {have / GIB:.2f} GiB) · {chk.detail}"
        )
        blocked = not chk.ok
        detail = f"Fetching {need / GIB:.2f} GiB would leave less than the {floor:g} GiB floor ({chk.detail})."

    if not blocked or not ctx.config.disk.enforce:
        return
    if force:
        ctx.telemetry.warn("download", "disk_gate_forced", repo=repo, need_bytes=need)
        console.print("[yellow]disk gate bypassed with --force[/yellow]")
        return
    raise InsufficientDiskError(
        detail,
        remediation=[
            "Free disk space before downloading this model.",
            "Or lower disk.min_free_gib in ~/.config/spark/config.toml.",
            "Or bypass once (at your own risk): spark download --force <repo>",
            "Or register a pointer only: spark download --no-fetch <repo>",
        ],
        context={"free_bytes": free, "need_bytes": need, "floor_gib": floor},
    )


# --- fetch with guaranteed terminal telemetry -------------------------------------
class _FetchInterrupt(BaseException):
    """Raised from a signal handler so an external terminate is loggable."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        self.name = signal.Signals(signum).name
        super().__init__(self.name)


@contextmanager
def _raise_on_terminate():
    """Convert SIGTERM/SIGHUP into an exception during the fetch, so the
    terminal telemetry event is written before the process dies. (SIGKILL is
    uncatchable by design; everything else must leave a trace.)"""

    def handler(signum, frame):
        raise _FetchInterrupt(signum)

    previous: dict[int, object] = {}
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            previous[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):  # non-main thread or unsupported platform
            pass
    try:
        yield
    finally:
        for sig, prev in previous.items():
            signal.signal(sig, prev)


def _run_fetch(
    ctx: SparkCtx, hf: str, repo: str, mid: str, dest: Path, env: dict | None
) -> None:
    """Run `hf download` emitting exactly one terminal telemetry event on every
    exit path: fetch_complete | fetch_failed | fetch_interrupted."""
    bytes_before = _dir_bytes(dest)
    started = time.monotonic()
    ctx.telemetry.info("download", "fetch_start", repo=repo, model_id=mid)
    cmd = [hf, "download", repo, "--local-dir", str(dest)]
    try:
        with _raise_on_terminate():
            # subprocess.run kills the child on any exception, so a raised
            # signal here also reaps hf before we log and exit.
            subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        ctx.telemetry.error(
            "download", "fetch_failed", repo=repo, model_id=mid,
            rc=exc.returncode, wall_s=round(time.monotonic() - started, 1),
        )
        raise SparkError(
            f"Download failed (hf exit {exc.returncode}).",
            code="NET_DOWNLOAD",
            remediation=[
                "Gated/private repo? store a token: spark secret set hf_token",
                "Verify the repo id and your network egress.",
            ],
            cause=exc,
        ) from exc
    except KeyboardInterrupt:
        ctx.telemetry.warn(
            "download", "fetch_interrupted", repo=repo, model_id=mid,
            signal="SIGINT", wall_s=round(time.monotonic() - started, 1),
            bytes_fetched=_dir_bytes(dest) - bytes_before,
        )
        raise
    except _FetchInterrupt as sig:
        ctx.telemetry.warn(
            "download", "fetch_interrupted", repo=repo, model_id=mid,
            signal=sig.name, wall_s=round(time.monotonic() - started, 1),
            bytes_fetched=_dir_bytes(dest) - bytes_before,
        )
        raise SystemExit(128 + sig.signum)
    except BaseException as exc:
        ctx.telemetry.error(
            "download", "fetch_failed", repo=repo, model_id=mid,
            error_type=type(exc).__name__, message=str(exc)[:256],
            wall_s=round(time.monotonic() - started, 1),
        )
        raise
    else:
        wall = time.monotonic() - started
        fetched = _dir_bytes(dest) - bytes_before
        ctx.telemetry.info(
            "download", "fetch_complete", repo=repo, model_id=mid,
            bytes_fetched=fetched, wall_s=round(wall, 1),
            rate_kibps=round(fetched / 1024 / wall, 1) if wall > 0 else None,
        )


@click.command(name="download")
@click.argument("repo")
@click.option("--id", "model_id", default=None, help="Override the model id/slug.")
@click.option("--no-fetch", is_flag=True,
              help="Register a pointer without downloading weights now.")
@click.option("--no-research", is_flag=True,
              help="Skip the agent research step (no LLM calls).")
@click.option("--force", is_flag=True,
              help="Bypass the disk preflight gate (not the memory gates).")
def download_command(
    repo: str, model_id: str | None, no_fetch: bool, no_research: bool, force: bool
):
    """Download REPO (a Hugging Face repo id) and register it."""
    ctx = build_context()
    mid = model_id or _slug(repo)
    inferred = _infer(repo)

    path = ""
    if not no_fetch:
        hf = shutil.which("hf") or shutil.which("huggingface-cli")
        if not hf:
            raise SparkError(
                "Hugging Face CLI not found.",
                code="NET_NO_HF_CLI",
                remediation=["Install it: brew install huggingface-cli  (or: uv tool install huggingface_hub)"],
            )
        dest = ctx.paths.model_store_dir / mid
        # HF token, if present, is injected into env only — never argv/logged.
        token = _hf_token(ctx)
        _disk_preflight(ctx, repo, dest, token, force=force)
        env = None
        if token:
            import os

            env = dict(os.environ)
            env["HF_TOKEN"] = token
        console.print(f"[bold]spark[/bold] downloading [cyan]{repo}[/cyan] → {dest}")
        _run_fetch(ctx, hf, repo, mid, dest, env)
        path = str(dest)

    entry = ModelEntry(
        id=mid,
        hf_repo=repo,
        path=path,
        model_format=inferred["model_format"],
        quant=inferred["quant"],
        params_billions=inferred["params_billions"],
        # Recorded so the launch path can price a re-fetch after the weights are
        # reclaimed: the disk gate needs a real number when nothing is on disk,
        # and the HF API is not consulted at launch time.
        size_bytes=_dir_bytes(Path(path)) if path else None,
        research_status="pending",
    )
    save_model(entry, ctx.paths)
    ctx.telemetry.info("download", "registered", model_id=mid, fetched=not no_fetch)
    refresh_catalog(ctx, model_id=mid)
    console.print(f"[green]✓[/green] registered [cyan]{mid}[/cyan] "
                  f"(format={entry.model_format or 'any'}, quant={entry.quant or '?'})")

    # Research: run the agent chain (unless opted out / disabled). Failure here is
    # non-fatal — the model is already registered and runnable.
    if no_research or not ctx.config.research.enabled:
        console.print(f"[dim]research skipped[/dim] — run later: spark research {mid}")
        return
    from .research import perform_research

    try:
        perform_research(ctx, entry)
    except SparkError as exc:
        ctx.telemetry.warn("download", "research_failed", model_id=mid, code=exc.code)
        console.print(f"[yellow]research unavailable[/yellow] ({exc.code}); model still "
                      f"runnable. Retry: spark research {mid}")
