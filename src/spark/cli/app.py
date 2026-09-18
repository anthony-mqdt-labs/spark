"""spark CLI entrypoint.

Root behavior:
- ``spark``            -> list registered models (the completion landing view).
- ``spark <model>``    -> dispatch to ``run`` (unknown tokens are treated as models).
- ``spark <command>``  -> the named subcommand wins over a same-named model.
"""

from __future__ import annotations

import sys

import click

from ..errors import SparkError
from .completion import complete_command, completion_command
from .config_cmd import config_group
from .context import build_context
from .doctor import doctor_command
from .download import download_command
from .render import console, render_error
from .research import research_command
from .run import run_command
from .secret import secret_group


class SparkGroup(click.Group):
    """Group that falls through unknown first tokens to ``run <token>``."""

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            if args and not args[0].startswith("-"):
                run = self.get_command(ctx, "run")
                # Treat the whole arg vector as `run <args...>`.
                return run.name, run, args
            raise


@click.command(name="list")
def list_command():
    """List registered models."""
    from ..registry import list_models
    from ..registry.availability import availability_map
    from ..registry.store import scan_store
    from .render import models_table, render_missing_weights, render_store_issues

    ctx = build_context()
    models = list_models(ctx.paths)
    avail = availability_map(models, ctx.paths, config=ctx.config)
    if models:
        console.print(models_table(models, avail))
    else:
        console.print("[dim]no models registered — try `spark download <hf-repo>`[/dim]")
    render_missing_weights(models, avail)
    render_store_issues(scan_store(ctx.paths))


@click.command(name="forget")
@click.argument("model")
def forget_command(model: str):
    """Remove MODEL from the registry (weights on disk are left alone).

    The counterpart to the availability column: an entry whose weights are gone
    keeps being listed, and would keep being a broken `spark run` target, until
    someone drops it. This drops it — and nothing else.
    """
    from ..registry import delete_model, resolve_model
    from ..registry.availability import resolve_availability

    ctx = build_context()
    entry = resolve_model(model, ctx.paths)  # exact -> alias -> unique prefix
    avail = resolve_availability(entry, ctx.paths, requires_weights=False)
    removed = delete_model(entry.id, ctx.paths)
    if not removed:
        console.print(f"[yellow]no registry entry for {entry.id}[/yellow]")
        return
    ctx.telemetry.info(
        "registry", "forgotten",
        model_id=entry.id, state=avail.state, location=avail.location,
    )
    console.print(f"[green]✓[/green] forgot [cyan]{entry.id}[/cyan] (registry entry removed)")
    if avail.state == "local" and avail.location:
        console.print(
            f"  [dim]weights still on disk ({avail.location}) — reclaim with:[/dim]\n"
            f"  [dim]rm -rf {avail.location}[/dim]"
        )
    else:
        console.print("  [dim]no weights were on disk[/dim]")



@click.group(cls=SparkGroup, invoke_without_command=True)
@click.version_option(package_name="spark", prog_name="spark")
@click.pass_context
def cli(ctx: click.Context):
    """spark — one command for every local LLM runtime."""
    if ctx.invoked_subcommand is None:
        # Bare `spark`: show the models landing view.
        ctx.invoke(list_command)


cli.add_command(run_command)
cli.add_command(download_command)
cli.add_command(research_command)
cli.add_command(doctor_command)
cli.add_command(list_command)
cli.add_command(forget_command)
cli.add_command(secret_group)
cli.add_command(config_group)
cli.add_command(completion_command)
cli.add_command(complete_command)


def main() -> None:
    try:
        cli.standalone_mode = False
        cli()
    except SparkError as exc:
        # Actionable {what/why/do-this-next} rendering + structured log.
        try:
            from ..telemetry import get_telemetry

            get_telemetry().error("cli", "command_failed", **exc.as_log())
        except Exception:
            pass
        render_error(exc)
        sys.exit(exc.exit_code)
    except click.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)
    except click.exceptions.Abort:
        console.print("[yellow]aborted[/yellow]")
        sys.exit(130)
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
