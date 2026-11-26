"""`spark config ...` — inspect and validate configuration."""

from __future__ import annotations

import click
from pydantic import ValidationError

from ..config.loader import load_config
from ..config.paths import bundled_config_dir, resolve_paths
from .render import console


def _accept_output(*, entry, output, ctx, staged: bool) -> None:
    """The one validate → warn → refuse → save → publish pipeline.

    Used by both ``config review --accept`` (staged output) and ``config import``
    (stdin output). They must not drift: the only differences are the error
    wording, whether the staging file is deleted, and the final message.
    """
    from ..errors import ConfigError
    from ..registry import save_model
    from ..research import delete_staged
    from ..research.validate import validate_accept
    from .context import refresh_catalog

    candidate, errors, warnings = validate_accept(entry, output, ctx.config)
    for w in warnings:
        console.print(f"[yellow]![/yellow] [dim]{w}[/dim]")
    if errors:
        remediation = [f"Drop it: {e}" for e in errors]
        if staged:
            remediation += [
                f"Or hand-edit the entry: {ctx.paths.models_dir / (entry.id + '.toml')}",
                f"Or re-run: spark research {entry.id} "
                "(staging is kept until --accept/--reject)",
            ]
        else:
            remediation.append("Or hand-edit the entry after importing without overrides.")
        source = "Staged" if staged else "Imported"
        raise ConfigError(
            f"{source} research for '{entry.id}' names flags its backend does not have.",
            code="CFG_RESEARCH_FLAGS",
            remediation=remediation,
            context={"model": entry.id, "backend": candidate.backend},
        )

    save_model(candidate, ctx.paths)
    if staged:
        delete_staged(ctx.paths, entry.id)
    ctx.telemetry.info(
        "research", "accepted" if staged else "imported",
        model_id=candidate.id, backend=candidate.backend,
    )
    refresh_catalog(ctx, model_id=candidate.id)
    action = "applied" if staged else "imported"
    quant = f", quant={candidate.quant or '—'}" if staged else ""
    console.print(
        f"[green]✓[/green] {action} → backend=[magenta]{candidate.backend}[/magenta]{quant}"
    )


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

    from ..registry import resolve_model
    from ..research import delete_staged, load_staged
    from ..research.types import ResearchOutput
    from .context import build_context

    if accept and reject:
        raise click.UsageError("--accept and --reject are mutually exclusive.")
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
        try:
            output = ResearchOutput.model_validate(staged.get("output"))
        except ValidationError as exc:
            from ..errors import ConfigError

            raise ConfigError(
                f"Staged research for '{entry.id}' does not match the research schema.",
                code="CFG_RESEARCH_SCHEMA",
                remediation=[
                    f"Discard it: spark config review {entry.id} --reject",
                    f"Then re-run: spark research {entry.id}",
                ],
                context={"model": entry.id},
                cause=exc,
            ) from exc
        _accept_output(entry=entry, output=output, ctx=ctx, staged=True)
        return
    console.print("\n[dim]--accept to apply, --reject to discard[/dim]")


@config_group.command(name="import")
@click.argument("model")
def config_import(model: str):
    """Import a research JSON for MODEL from stdin (manual path)."""
    import json as _json
    import sys

    from ..registry import resolve_model
    from ..research.types import ResearchOutput
    from .context import build_context

    ctx = build_context()
    entry = resolve_model(model, ctx.paths)
    raw = sys.stdin.read()
    try:
        data = _json.loads(raw)
        # accept either a bare output or a staged-doc wrapper
        output = ResearchOutput.model_validate(data.get("output", data))
    except (ValueError, ValidationError) as exc:
        from ..errors import ConfigError

        raise ConfigError(
            "Could not parse research JSON from stdin.",
            remediation=["Pipe a JSON object matching the research schema.",
                         "See `spark config show` and the research schema."],
            cause=exc,
        ) from exc
    _accept_output(entry=entry, output=output, ctx=ctx, staged=False)
