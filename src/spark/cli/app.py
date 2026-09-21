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
@click.option("--all", "show_all", is_flag=True,
              help="Show missing entries and non-servable cache in full.")
def list_command(show_all: bool):
    """List what this host can actually serve (disk truth, registry or not)."""
    from ..catalog import write_catalog
    from ..inventory import build_inventory
    from ..registry.store import scan_store
    from .render import (
        render_inventory_notes,
        render_store_issues,
        runnable_table,
    )

    ctx = build_context()
    inv = build_inventory(ctx.paths, ctx.config)
    if inv.runnable_registered or inv.runnable_discovered or inv.external or inv.unverifiable:
        console.print(runnable_table(inv))
    else:
        console.print("[dim]nothing runnable on disk — try `spark download <hf-repo>`[/dim]")
    render_inventory_notes(inv, verbose=show_all)
    render_store_issues(scan_store(ctx.paths))
    # Keep the published roster in step with what the operator just looked at.
    write_catalog(ctx.paths, ctx.config)


@click.command(name="catalog")
@click.option("--json", "as_json", is_flag=True, help="Print the catalog JSON instead of a summary.")
@click.option("--refresh/--no-refresh", default=True, help="Regenerate before printing.")
def catalog_command(as_json: bool, refresh: bool):
    """Print the live model catalog (the machine-readable roster).

    Same data consumers read from `<data>/catalog.json`: every registered model
    with its availability as of now, plus any live instances.
    """
    import json as _json

    from ..catalog import build_catalog, write_catalog

    ctx = build_context()
    if refresh:
        path = write_catalog(ctx.paths, ctx.config)
    else:
        from ..catalog import catalog_path

        path = catalog_path(ctx.paths)
    if as_json:
        console.print_json(_json.dumps(build_catalog(ctx.paths, ctx.config)))
        return
    cat = build_catalog(ctx.paths, ctx.config)
    console.print(f"[bold]catalog[/bold] [dim]{path}[/dim]")
    console.print(f"  generated: {cat['generated_at']}   schema: {cat['schema_version']}")
    console.print(f"  api contract: {cat['api_contract']}")
    live = cat["instances"]
    if live:
        console.print(f"  [green]{len(live)} live instance(s)[/green]")
        for inst in live:
            console.print(
                f"    • [cyan]{inst['model_alias']}[/cyan] → {inst['base_url']} "
                f"[dim]({inst['backend']}, pid {inst['pid']})[/dim]"
            )
            console.print(f"      api id: {inst['model_id']}")
    else:
        console.print("  [dim]no live instances[/dim]")
    missing = [m for m in cat["models"] if not m["availability"]["ok"]]
    console.print(f"  models: {len(cat['models'])} registered, {len(missing)} without weights")
    if cat.get("discovered"):
        console.print(f"  discovered: {len(cat['discovered'])} on-disk model(s) not in the registry")
        for d in cat["discovered"]:
            console.print(f"    • [cyan]{d['id']}[/cyan] [dim]({d['source']})[/dim]")


@click.command(name="adopt")
@click.argument("repo")
@click.option("--id", "model_id", default=None, help="Registry id (default: repo tail slug).")
@click.option("--backend", "backend", default="", help="Pin a runtime backend.")
def adopt_command(repo: str, model_id: str | None, backend: str):
    """Register an on-disk model (hub cache or store path) into the registry.

    REPO is an HF repo id (e.g. a row from `spark list`) or a local store path.
    The weights stay where they are; this only writes the registry entry so the
    model keeps a backend, research provenance, and a short id.
    """
    import re

    from ..catalog import write_catalog
    from ..config.schema import ModelEntry
    from ..inventory import build_inventory
    from ..registry import save_model

    ctx = build_context()
    inv = build_inventory(ctx.paths, ctx.config, with_bytes=False)
    target = repo.strip()
    found = None
    for m in inv.runnable_discovered:
        if m.display_id.lower() == target.lower() or m.location == target:
            found = m
            break
    if found is None:
        from ..errors import ModelNotFoundError

        raise ModelNotFoundError(
            f"Nothing adoptable matches '{repo}'.",
            remediation=[
                "Adoptable models are the unregistered rows of: spark list",
                f"Or download it first: spark download {repo}",
            ],
            context={"query": repo},
        )
    slug = model_id or re.sub(r"[^A-Za-z0-9_.-]+", "-", found.display_id.split("/")[-1]).strip("-").lower()
    entry = ModelEntry(
        id=slug,
        hf_repo=found.repo,
        path=found.location if found.source == "store" else "",
        backend=backend,
        model_format=found.model_format,  # type: ignore[arg-type]
        quant=found.quant,
        research_status="pending",
    )
    save_model(entry, ctx.paths)
    ctx.telemetry.info("registry", "adopted", model_id=slug, source=found.source)
    write_catalog(ctx.paths, ctx.config)
    console.print(f"[green]✓[/green] adopted [cyan]{found.display_id}[/cyan] as [cyan]{slug}[/cyan]")
    console.print(f"  [dim]research next: spark research {slug}[/dim]")


@click.command(name="forget")
@click.argument("model")
def forget_command(model: str):
    """Remove MODEL from the registry (weights on disk are left alone).

    The counterpart to the availability column: an entry whose weights are gone
    keeps being listed, and would keep being a broken `spark run` target, until
    someone drops it. This drops it — and nothing else.
    """
    from ..catalog import write_catalog
    from ..registry import delete_model, resolve_model
    from ..registry.availability import requires_weights_for, resolve_availability

    ctx = build_context()
    entry = resolve_model(model, ctx.paths)  # exact -> alias -> unique prefix
    avail = resolve_availability(
        entry, ctx.paths,
        requires_weights=requires_weights_for(entry, ctx.config),
    )
    removed = delete_model(entry.id, ctx.paths)
    if not removed:
        console.print(f"[yellow]no registry entry for {entry.id}[/yellow]")
        return
    ctx.telemetry.info(
        "registry", "forgotten",
        model_id=entry.id, state=avail.state, location=avail.location,
    )
    write_catalog(ctx.paths, ctx.config)
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
cli.add_command(catalog_command)
cli.add_command(forget_command)
cli.add_command(adopt_command)
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
