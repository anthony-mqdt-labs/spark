"""`spark config ...` — inspect and validate configuration."""

from __future__ import annotations

import click

from ..config.loader import load_config
from ..config.paths import bundled_config_dir, resolve_paths
from .render import console


@click.group(name="config")
def config_group():
    """Inspect and validate spark configuration."""


@config_group.command(name="validate")
def config_validate():
    """Load + validate the full configuration (raises on error)."""
    cfg = load_config()
    console.print(
        f"[green]✓[/green] configuration valid · "
        f"{len(cfg.runtimes)} runtime(s): {', '.join(sorted(cfg.runtimes))}"
    )


@config_group.command(name="path")
def config_path():
    """Show where spark reads/writes configuration and data."""
    p = resolve_paths()
    console.print(f"[bold]bundled defaults[/bold]: {bundled_config_dir()}")
    console.print(f"[bold]user config[/bold]:      {p.user_config_file}")
    console.print(f"[bold]data dir[/bold]:         {p.data_dir}")
    console.print(f"[bold]models[/bold]:           {p.models_dir}")
    console.print(f"[bold]logs[/bold]:             {p.log_dir}")
    console.print(f"[bold]host profile[/bold]:     {p.host_profile_file}")


@config_group.command(name="show")
def config_show():
    """Print the resolved configuration (no secrets — there are none here)."""
    cfg = load_config()
    console.print_json(cfg.model_dump_json(indent=2))


@config_group.command(name="review")
@click.argument("model")
@click.option("--accept", is_flag=True, help="Apply the staged research to the model.")
@click.option("--reject", is_flag=True, help="Discard the staged research.")
def config_review(model: str, accept: bool, reject: bool):
    """Review (and accept/reject) staged research output for MODEL."""
    import json as _json

    from ..registry import resolve_model, save_model
    from ..research import apply_output_to_entry, delete_staged, load_staged
    from ..research.types import ResearchOutput
    from .context import build_context, refresh_catalog

    ctx = build_context()
    entry = resolve_model(model, ctx.paths)
    staged = load_staged(ctx.paths, entry.id)
    if not staged:
        console.print(f"[yellow]no staged research for '{entry.id}'[/yellow] — run "
                      f"`spark research {entry.id}`")
        return

    console.print(f"[bold]Staged research[/bold] for [cyan]{entry.id}[/cyan] "
                  f"(provider: {staged.get('provider')})")
    console.print_json(_json.dumps(staged.get("output", {}), indent=2))

    if reject:
        delete_staged(ctx.paths, entry.id)
        console.print("[yellow]rejected and discarded[/yellow]")
        return
    if accept:
        output = ResearchOutput.model_validate(staged["output"])
        apply_output_to_entry(entry, output)
        save_model(entry, ctx.paths)
        delete_staged(ctx.paths, entry.id)
        ctx.telemetry.info("research", "accepted", model=entry.id, backend=entry.backend)
        refresh_catalog(ctx, model_id=entry.id)
        console.print(f"[green]✓[/green] applied → backend=[magenta]{entry.backend}[/magenta], "
                      f"quant={entry.quant or '—'}")
        return
    console.print("\n[dim]--accept to apply, --reject to discard[/dim]")


@config_group.command(name="import")
@click.argument("model")
def config_import(model: str):
    """Import a research JSON for MODEL from stdin (manual path)."""
    import json as _json
    import sys

    from ..registry import resolve_model, save_model
    from ..research import apply_output_to_entry
    from ..research.types import ResearchOutput
    from .context import build_context, refresh_catalog

    ctx = build_context()
    entry = resolve_model(model, ctx.paths)
    raw = sys.stdin.read()
    try:
        data = _json.loads(raw)
        # accept either a bare output or a staged-doc wrapper
        output = ResearchOutput.model_validate(data.get("output", data))
    except Exception as exc:
        from ..errors import ConfigError

        raise ConfigError(
            "Could not parse research JSON from stdin.",
            remediation=["Pipe a JSON object matching the research schema.",
                         "See `spark config show` and the research schema."],
            cause=exc,
        ) from exc
    apply_output_to_entry(entry, output)
    save_model(entry, ctx.paths)
    ctx.telemetry.info("research", "imported", model=entry.id, backend=entry.backend)
    refresh_catalog(ctx, model_id=entry.id)
    console.print(f"[green]✓[/green] imported → backend=[magenta]{entry.backend}[/magenta]")
