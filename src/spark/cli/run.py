"""`spark <model>` / `spark run <model>` — launch the optimal runtime server."""

from __future__ import annotations

import click

from ..errors import SparkError
from ..probe.host import load_or_probe
from ..registry import resolve_model
from ..runner import Supervisor
from ..runtimes import get_backend, select_backend_for
from .context import build_context
from .render import console, render_server_log


@click.command(name="run")
@click.argument("model")
@click.option("--backend", "backend_override", default=None,
              help="Force a specific runtime backend.")
@click.option("--dflash", is_flag=True,
              help="Enable the model's DFlash speculative-decoding profile (opt-in; "
                   "needs a [dflash] section in the model's registry entry).")
@click.option("--force", is_flag=True, help="Bypass memory/disk budget gates.")
@click.option("--reprobe", is_flag=True, help="Re-probe host before launching.")
def run_command(model: str, backend_override: str | None, dflash: bool,
                force: bool, reprobe: bool):
    """Launch the best-optimized inference server for MODEL."""
    ctx = build_context()
    entry = resolve_model(model, ctx.paths)

    if dflash:
        if entry.dflash is None:
            from ..errors import ModelError

            raise ModelError(
                f"Model '{entry.id}' has no DFlash profile to enable.",
                remediation=[
                    "Add a [dflash] section to its registry entry, or drop --dflash.",
                ],
                context={"model": entry.id},
            )
        entry = entry.with_dflash()

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
        "cli", "run_dispatch", model=entry.id, backend=backend_name, dflash=dflash,
    )

    label = backend_name + (" · dflash" if dflash else "")

    def _on_ready(info) -> None:
        _print_ready(entry.id, label, info)

    sup = Supervisor(
        backend, entry, ctx.config, profile, ctx.paths, ctx.secrets, force=force
    )
    try:
        result = sup.run(on_ready=_on_ready)
    except SparkError:
        # Server output was redirected off the terminal; surface its tail so the
        # failure reason is visible, then let the top-level renderer show the error.
        render_server_log(sup.log_path, sup.log_tail())
        raise
    console.print(f"  [dim]spark stopped {entry.id} (restarts: {result.restarts})[/dim]")


def _print_ready(model_id: str, label: str, info) -> None:
    """Vite-style clean startup banner, printed the moment the server is healthy."""
    from importlib.metadata import version

    try:
        ver = version("spark")
    except Exception:
        ver = "0.0.0"
    console.print()
    console.print(f"  [bold green]spark[/bold green] [dim]v{ver}[/dim]  "
                  f"ready in [bold]{info.elapsed_s:.1f}s[/bold]")
    console.print()
    console.print(f"  [green]➜[/green]  [bold]Local:[/bold]   [cyan]{info.base_url}[/cyan]")
    console.print(f"  [green]➜[/green]  [bold]Model:[/bold]   {model_id} [dim]({label})[/dim]")
    console.print(f"  [green]➜[/green]  [bold]Logs:[/bold]    [dim]{info.log_path}[/dim]")
    console.print()
    console.print("  [dim]press Ctrl+C to stop[/dim]")
    console.print()
