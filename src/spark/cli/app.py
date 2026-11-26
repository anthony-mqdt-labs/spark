"""spark CLI entrypoint.

Root behavior:
- ``spark``            -> list registered models (the completion landing view).
- ``spark <model>``    -> dispatch to ``run`` (unknown tokens are treated as models).
- ``spark <command>``  -> the named subcommand wins over a same-named model.
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from pathlib import Path

import click

from ..errors import SparkError
from ..registry.adopt_errors import adopt_miss_error as _adopt_miss_error
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
                if run is None:
                    raise
                # Treat the whole arg vector as `run <args...>`.
                return run.name, run, args
            raise


@click.command(name="list")
@click.option(
    "--all", "show_all", is_flag=True, help="Show missing entries and non-servable cache in full."
)
def list_command(show_all: bool):
    """List what this host can actually serve (disk truth, registry or not)."""
    from ..inventory import build_inventory
    from ..registry.store import scan_store
    from .context import refresh_catalog
    from .render import (
        render_inventory_notes,
        render_store_issues,
        runnable_table,
    )

    invalid: list[tuple[Path, str]] = []

    def _note_invalid(path: Path, detail: str) -> None:
        invalid.append((path, detail))

    ctx = build_context()
    inv = build_inventory(ctx.paths, ctx.config, on_invalid=_note_invalid)
    if inv.runnable_registered or inv.runnable_discovered or inv.external or inv.unverifiable:
        console.print(runnable_table(inv))
    else:
        console.print("[dim]nothing runnable on disk — try `spark download <hf-repo>`[/dim]")
    render_inventory_notes(inv, verbose=show_all)
    for path, detail in invalid:
        # One line per skipped file: a hand-edit typo must not silently hide a model.
        console.print(
            f"[yellow]![/yellow] skipped invalid registry entry {path.name}: "
            f"{detail.splitlines()[0] if detail else 'unknown validation error'}"
        )
    render_store_issues(scan_store(ctx.paths))
    # Keep the published roster in step with what the operator just looked at.
    refresh_catalog(ctx)


@click.command(name="catalog")
@click.option(
    "--json", "as_json", is_flag=True, help="Print the catalog JSON instead of a summary."
)
@click.option("--refresh/--no-refresh", default=True, help="Regenerate before printing.")
def catalog_command(as_json: bool, refresh: bool):
    """Print the live model catalog (the machine-readable roster).

    Same data consumers read from `<data>/catalog.json`: every registered model
    with its availability as of now, plus any live instances.
    """
    from ..catalog import build_catalog, write_catalog

    ctx = build_context()
    if refresh:
        path = write_catalog(ctx.paths, ctx.config)
    else:
        from ..catalog import catalog_path

        path = catalog_path(ctx.paths)
    if as_json:
        console.print_json(json.dumps(build_catalog(ctx.paths, ctx.config)))
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
        console.print(
            f"  discovered: {len(cat['discovered'])} on-disk model(s) not in the registry"
        )
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
    from ..config.schema import ModelEntry
    from ..errors import AlreadyRegisteredError, ConfigError
    from ..inventory import build_inventory, infer_quant_and_params
    from ..registry import list_models, save_model
    from ..registry.models import fuzzy_match
    from .context import refresh_catalog

    ctx = build_context()
    if backend and backend not in ctx.config.runtimes:
        raise ConfigError(
            f"Unknown backend '{backend}'.",
            code="CFG_NOT_FOUND",
            remediation=[f"Known runtimes: {', '.join(sorted(ctx.config.runtimes))}"],
        )
    # adopt is a one-off write: pay for the byte walk once so the registry entry
    # it persists carries a real size for the memory/disk gates.
    inv = build_inventory(ctx.paths, ctx.config, with_bytes=True)
    target = repo.strip()
    # Exact on-disk location first, then the shared fuzzy rule (exact id, unique
    # prefix/substring, unique repo tail) — identical to `run` and `registry`.
    found = next((m for m in inv.runnable_discovered if m.location == target), None)
    if found is None:
        found, _ambiguous = fuzzy_match(
            repo, [(m.display_id, m) for m in inv.runnable_discovered], tails=True
        )
    if found is None:
        raise _adopt_miss_error(repo, inv, ctx)
    slug = (
        model_id
        or re.sub(r"[^A-Za-z0-9_.-]+", "-", found.display_id.split("/")[-1]).strip("-").lower()
    )
    clash = next((m for m in list_models(ctx.paths) if m.id == slug), None)
    if clash is not None:
        raise AlreadyRegisteredError(
            f"Id '{slug}' is already registered ({clash.hf_repo or clash.path}).",
            remediation=[f"Pick another id: spark adopt {repo} --id <new-id>"],
            context={"model": slug},
        )
    quant, params = infer_quant_and_params(found.display_id)
    entry = ModelEntry(
        id=slug,
        hf_repo=found.repo,
        path=found.location if found.source == "store" else "",
        backend=backend,
        model_format=found.model_format,
        quant=found.quant or quant,
        params_billions=params,
        size_bytes=found.bytes_present or None,
        research_status="pending",
    )
    save_model(entry, ctx.paths)
    ctx.telemetry.info("registry", "adopted", model_id=slug, source=found.source)
    refresh_catalog(ctx, model_id=slug)
    console.print(
        f"[green]✓[/green] adopted [cyan]{found.display_id}[/cyan] as [cyan]{slug}[/cyan]"
    )
    console.print(f"  [dim]research next: spark research {slug}[/dim]")


@click.command(name="forget")
@click.argument("model")
def forget_command(model: str):
    """Remove MODEL from the registry (weights on disk are left alone).

    The counterpart to the availability column: an entry whose weights are gone
    keeps being listed, and would keep being a broken `spark run` target, until
    someone drops it. This drops it — and nothing else.
    """
    from ..registry import delete_model, resolve_model
    from .context import refresh_catalog
    from ..registry.availability import requires_weights_for, resolve_availability

    ctx = build_context()
    entry = resolve_model(model, ctx.paths)  # exact -> alias -> unique prefix
    avail = resolve_availability(
        entry,
        ctx.paths,
        requires_weights=requires_weights_for(entry, ctx.config),
    )
    removed = delete_model(entry.id, ctx.paths)
    if not removed:
        console.print(f"[yellow]no registry entry for {entry.id}[/yellow]")
        return
    ctx.telemetry.info(
        "registry",
        "forgotten",
        model_id=entry.id,
        state=avail.state,
        location=avail.location,
    )
    refresh_catalog(ctx, model_id=entry.id)
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
        # Pass the mode to main() rather than mutating the group object: click
        # re-raises SparkError so our {what/why/do-this-next} renderer can own it.
        cli(standalone_mode=False)
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
    except Exception as exc:  # noqa: BLE001 - last line: never a bare traceback
        _render_unexpected(exc)
        sys.exit(70)  # EX_SOFTWARE


def _render_unexpected(exc: Exception) -> None:
    """An exception spark did not type is a spark bug — still leave
    {what, why, do-this-next} and a log record a post-hoc diagnosis can use."""
    from ..redaction import scrub_string

    raw_message = str(exc)
    message = scrub_string(raw_message[:512])
    tb = traceback.extract_tb(exc.__traceback__)
    where = tb[-1] if tb else None
    try:
        from ..telemetry import get_telemetry

        get_telemetry().critical(
            "cli",
            "unexpected_error",
            error_type=type(exc).__name__,
            message=message,
            file=where.filename if where else "",
            line=where.lineno if where else 0,
            context_window=get_telemetry().context_window()[-10:],
        )
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get("SPARK_DEBUG"):
        # A traceback can interpolate exception arguments, including a secret.
        print(scrub_string("".join(traceback.format_exception(exc))))
    render_error(
        SparkError(
            f"Unexpected internal error: {type(exc).__name__}: {scrub_string(raw_message[:200])}",
            code="SPARK_INTERNAL",
            remediation=[
                "Re-run with SPARK_DEBUG=1 for the full traceback.",
                "The failure is logged as `unexpected_error` under the data dir's logs/cli/.",
            ],
        )
    )


if __name__ == "__main__":
    main()
