"""`spark research <model>` — run the agent research chain for a registered model.

Output is staged for review (`spark config review <model>`); it is never applied
automatically."""

from __future__ import annotations

import click

from ..probe.host import load_or_probe
from ..registry import resolve_model
from ..research import run_research, stage
from ..research.prompt import host_summary
from .context import build_context
from .render import console


def perform_research(ctx, entry) -> bool:
    """Run the chain for an entry and stage the result. Returns True on success."""
    if not entry.hf_repo:
        console.print("[yellow]no hf_repo on this model — cannot research[/yellow]")
        return False
    profile = load_or_probe(ctx.config, ctx.paths.host_profile_file)
    available = profile.available_runtimes
    console.print(
        f"[bold]spark[/bold] researching [cyan]{entry.id}[/cyan] "
        f"({entry.hf_repo}) across {len(available)} runtime(s) …"
    )
    result = run_research(entry.hf_repo, profile, ctx.config, available)
    path = stage(
        ctx.paths, entry.id, result.output,
        provider=result.provider, hf_repo=entry.hf_repo,
        host_summary=host_summary(profile, ctx.config),
    )
    ctx.telemetry.info(
        "research", "staged", model=entry.id, provider=result.provider,
        recommended=result.output.recommended_backend,
    )
    console.print(
        f"[green]✓[/green] {result.provider} recommends "
        f"[magenta]{result.output.recommended_backend}[/magenta] · staged → {path.name}"
    )
    console.print(f"  review & accept: [bold]spark config review {entry.id}[/bold]")
    return True


@click.command(name="research")
@click.argument("model")
def research_command(model: str):
    """Research optimal per-runtime config for an already-registered MODEL."""
    ctx = build_context()
    entry = resolve_model(model, ctx.paths)
    perform_research(ctx, entry)
