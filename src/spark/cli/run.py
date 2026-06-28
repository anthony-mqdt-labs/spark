"""`spark <model>` / `spark run <model>` — launch the optimal runtime server."""

from __future__ import annotations

import click

from ..probe.host import load_or_probe
from ..registry import resolve_model
from ..runner import Supervisor
from ..runtimes import get_backend, select_backend_for
from .context import build_context
from .render import console


@click.command(name="run")
@click.argument("model")
@click.option("--backend", "backend_override", default=None,
              help="Force a specific runtime backend.")
@click.option("--force", is_flag=True, help="Bypass memory/disk budget gates.")
@click.option("--reprobe", is_flag=True, help="Re-probe host before launching.")
def run_command(model: str, backend_override: str | None, force: bool, reprobe: bool):
    """Launch the best-optimized inference server for MODEL."""
    ctx = build_context()
    entry = resolve_model(model, ctx.paths)

    profile = load_or_probe(
        ctx.config, ctx.paths.host_profile_file, force=reprobe
    )
    available = profile.available_runtimes
    if not available:
        from ..errors import BackendUnavailableError

        raise BackendUnavailableError(
            "No inference runtimes are available on this host.",
            remediation=["Install one (see `spark doctor` for hints)."],
        )

    backend_name = backend_override or select_backend_for(entry, ctx.config, available)
    if backend_name not in available:
        from ..errors import BackendUnavailableError

        raise BackendUnavailableError(
            f"Requested backend '{backend_name}' is not available.",
            remediation=[f"Available: {', '.join(available)}"],
        )

    backend = get_backend(backend_name, ctx.config)
    ctx.telemetry.info(
        "cli", "run_dispatch", model=entry.id, backend=backend_name,
    )

    console.print(
        f"[bold]spark[/bold] launching [cyan]{entry.id}[/cyan] "
        f"via [magenta]{backend_name}[/magenta] …"
    )
    sup = Supervisor(
        backend, entry, ctx.config, profile, ctx.paths, ctx.secrets, force=force
    )
    result = sup.run()
    console.print(
        f"[green]✓[/green] served [cyan]{entry.id}[/cyan] at "
        f"[bold]{result.endpoint}[/bold] (restarts: {result.restarts})"
    )
