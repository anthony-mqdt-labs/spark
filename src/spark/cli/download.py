"""`spark download <hf-repo>` — fetch + register a model.

Phase 1: performs the download (via the `hf` CLI), infers a sensible registry
entry, and marks research as pending. The multi-agent research chain that fills in
optimal per-runtime flags lands in Phase 3 (`spark.research`).
"""

from __future__ import annotations

import re
import shutil
import subprocess

import click

from ..budget import GIB, free_disk_bytes
from ..config.schema import ModelEntry
from ..errors import InsufficientDiskError, SparkError
from ..registry import save_model
from .context import build_context
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


@click.command(name="download")
@click.argument("repo")
@click.option("--id", "model_id", default=None, help="Override the model id/slug.")
@click.option("--no-fetch", is_flag=True,
              help="Register a pointer without downloading weights now.")
@click.option("--no-research", is_flag=True,
              help="Skip the agent research step (no LLM calls).")
def download_command(repo: str, model_id: str | None, no_fetch: bool, no_research: bool):
    """Download REPO (a Hugging Face repo id) and register it."""
    ctx = build_context()
    mid = model_id or _slug(repo)
    inferred = _infer(repo)

    # Disk preflight: we cannot know the exact size yet, so warn loudly and apply
    # the floor against current free space.
    free = free_disk_bytes(ctx.paths.model_store_dir)
    floor = ctx.config.disk.min_free_gib
    console.print(
        f"[bold]disk[/bold]: {free / GIB:.1f} GiB free · floor {floor:g} GiB"
    )
    if ctx.config.disk.enforce and free / GIB <= floor and not no_fetch:
        raise InsufficientDiskError(
            f"Only {free / GIB:.1f} GiB free; below the {floor:g} GiB floor.",
            remediation=[
                "Free disk space before downloading a model.",
                "Or register a pointer only: spark download --no-fetch <repo>",
            ],
        )

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
        env = None
        if ctx.secrets.exists("hf_token"):
            import os

            env = dict(os.environ)
            env.update(ctx.secrets.resolve_env({"HF_TOKEN": "hf_token"}))
        console.print(f"[bold]spark[/bold] downloading [cyan]{repo}[/cyan] → {dest}")
        ctx.telemetry.info("download", "fetch_start", repo=repo, model_id=mid)
        cmd = [hf, "download", repo, "--local-dir", str(dest)]
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            ctx.telemetry.error("download", "fetch_failed", repo=repo, rc=exc.returncode)
            raise SparkError(
                f"Download failed (hf exit {exc.returncode}).",
                code="NET_DOWNLOAD",
                remediation=[
                    "Gated/private repo? store a token: spark secret set hf_token",
                    "Verify the repo id and your network egress.",
                ],
                cause=exc,
            ) from exc
        path = str(dest)

    entry = ModelEntry(
        id=mid,
        hf_repo=repo,
        path=path,
        model_format=inferred["model_format"],
        quant=inferred["quant"],
        params_billions=inferred["params_billions"],
        research_status="pending",
    )
    save_model(entry, ctx.paths)
    ctx.telemetry.info("download", "registered", model_id=mid, fetched=not no_fetch)
    console.print(f"[green]✓[/green] registered [cyan]{mid}[/cyan] "
                  f"(format={entry.model_format or 'any'}, quant={entry.quant or '?'})")

    # Research: run the agent chain (unless opted out / disabled). Failure here is
    # non-fatal — the model is already registered and runnable.
    if no_research or not ctx.config.research.enabled:
        console.print(f"[dim]research skipped[/dim] — run later: spark research {mid}")
        return
    from ..errors import SparkError
    from .research import perform_research

    try:
        perform_research(ctx, entry)
    except SparkError as exc:
        ctx.telemetry.warn("download", "research_failed", model_id=mid, code=exc.code)
        console.print(f"[yellow]research unavailable[/yellow] ({exc.code}); model still "
                      f"runnable. Retry: spark research {mid}")
