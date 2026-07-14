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
    from ..registry.store import scan_store
    from .render import models_table, render_store_issues

    ctx = build_context()
    models = list_models(ctx.paths)
    if models:
        console.print(models_table(models))
    else:
        console.print("[dim]no models registered — try `spark download <hf-repo>`[/dim]")
    render_store_issues(scan_store(ctx.paths))


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
