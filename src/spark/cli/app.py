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


def _adopt_miss_error(repo: str, inv, ctx) -> SparkError:
    """State-aware error for `spark adopt <query>` when nothing was adopted.

    The old message always said "nothing adoptable", even when the query was
    already registered (the common confusion: the `status` column shows
    research state — `pending` — not download state). Report what the query
    actually is: already registered (with backend/research/weights state),
    cached-but-not-servable, a partial store download, an incomplete hub
    snapshot, or genuinely unknown.
    """
    from pathlib import Path

    from ..errors import (
        AlreadyRegisteredError,
        AmbiguousModelError,
        ModelNotFoundError,
    )

    target = repo.strip()
    tl = target.lower()

    registered = [
        (e, a)
        for group in (
            inv.runnable_registered,
            inv.missing,
            inv.external,
            inv.unverifiable,
        )
        for e, a in group
    ]

    def _registry_exact():
        for e, a in registered:
            if e.id.lower() == tl:
                return e, a
            if any(al.lower() == tl for al in e.aliases):
                return e, a
            if (e.hf_repo or "").lower() == tl:
                return e, a
            loc = getattr(a, "location", "") or ""
            if loc and (loc == target or loc.lower() == tl):
                return e, a
            if e.path and (e.path == target or e.path.lower() == tl):
                return e, a
        return None

    hit = _registry_exact()
    if hit is not None:
        entry, avail = hit
        backend = entry.backend or "auto"
        research = entry.research_status
        state = getattr(avail, "state", "?")
        detail = getattr(avail, "detail", "")
        location = getattr(avail, "location", "") or ""
        if state == "missing":
            weights = f"weights missing ({detail})" if detail else "weights missing"
        elif state in ("local", "hub"):
            weights = f"weights present ({state}{f': {location}' if location else ''})"
        elif state == "external":
            weights = f"weights owned by '{entry.backend}' runtime"
        elif state == "unverifiable":
            weights = "entry declares neither path nor hf_repo"
        else:
            weights = f"weights state: {state}"
        message = (
            f"'{repo}' is already registered as '{entry.id}' — nothing to adopt.\n"
            f"backend: {backend} · research: {research} · {weights}"
        )
        if state == "missing":
            remediation = (
                [f"Fetch weights: spark download {entry.hf_repo}"]
                if entry.hf_repo
                else []
            )
            remediation += [f"Drop the entry: spark forget {entry.id}"]
        elif research == "pending":
            # The exact confusion that prompted this: `pending` in `spark list`
            # is research state, not download state. Weights are already here.
            remediation = [
                f"Weights are on disk — no adopt needed. Research next: spark research {entry.id}",
                f"Or just run it: spark run {entry.id}",
            ]
        else:
            remediation = [f"Run it: spark run {entry.id}"]
        return AlreadyRegisteredError(
            message,
            remediation=remediation,
            context={
                "query": repo,
                "model_id": entry.id,
                "backend": entry.backend,
                "research_status": research,
                "availability": state,
                "location": location,
            },
        )

    # Ambiguous registry prefix/substring (adopt takes one model).
    prefix = [e for e, _ in registered if e.id.lower().startswith(tl)]
    substr = [e for e, _ in registered if tl in e.id.lower()]
    candidates = prefix or substr
    if len(candidates) > 1:
        ids = sorted({e.id for e in candidates})
        return AmbiguousModelError(
            f"'{repo}' matches multiple registered models.",
            remediation=[f"Be more specific: {', '.join(ids)}"],
            context={"query": repo, "matches": ids},
        )

    # Cached repos spark cannot serve (embedders, speech). Never adoptable,
    # but the query *is* on disk — say so instead of "nothing matches".
    def _non_servable_hits():
        exact = [m for m in inv.non_servable if m.display_id.lower() == tl]
        if exact:
            return exact
        tails = [
            m
            for m in inv.non_servable
            if m.display_id.lower().split("/")[-1] == tl
        ]
        if tails:
            return tails
        pre = [m for m in inv.non_servable if m.display_id.lower().startswith(tl)]
        if pre:
            return pre
        return [m for m in inv.non_servable if tl in m.display_id.lower()]

    ns_hits = _non_servable_hits()
    if ns_hits:
        m = ns_hits[0]
        others = f" (+{len(ns_hits) - 1} more)" if len(ns_hits) > 1 else ""
        return ModelNotFoundError(
            f"'{repo}' is cached as '{m.display_id}'{others} but spark cannot serve it "
            f"({m.kind}) — {m.reason}. Nothing to adopt.",
            remediation=[
                "Adoptable models are the unregistered runnable rows of: spark list",
                "Inspect the hidden cache: spark list --all",
            ],
            context={"query": repo, "kind": m.kind, "reason": m.reason},
        )

    # Partial / orphaned store dirs own their message (resume, don't adopt).
    try:
        from ..registry.store import scan_store

        issues = scan_store(ctx.paths)
    except OSError:
        issues = []
    for issue in issues:
        if (
            issue.model_id.lower() == tl
            or str(issue.path) == target
            or issue.model_id.lower().split("/")[-1] == tl
        ):
            if issue.incomplete_files:
                resume = (
                    f"spark download {issue.hf_repo}"
                    if issue.hf_repo
                    else f"spark download <hf-repo> --id {issue.model_id}"
                )
                return ModelNotFoundError(
                    f"'{repo}' is an incomplete download in the model store "
                    f"({issue.incomplete_files} resume fragment(s)) — resume it, don't adopt it.",
                    remediation=[
                        f"Resume: {resume}",
                        f"Clean: rm -rf {issue.path}",
                    ],
                    context={"query": repo, "path": str(issue.path)},
                )
            return ModelNotFoundError(
                f"'{repo}' is in the model store but has no servable snapshot "
                f"({issue.path}) — nothing to adopt yet.",
                remediation=[
                    "Adoptable models are the unregistered runnable rows of: spark list",
                    f"Clean: rm -rf {issue.path}",
                ],
                context={"query": repo, "path": str(issue.path)},
            )

    # Hub cache dir exists but no complete snapshot (mid-download / reclaimed).
    if "/" in target:
        try:
            from ..hf_cache import repo_dir, snapshot_dir

            if repo_dir(target).is_dir() and snapshot_dir(target) is None:
                return ModelNotFoundError(
                    f"'{repo}' is in the HF hub cache but has no complete snapshot "
                    f"(no config.json) — interrupted download or reclaimed weights.",
                    remediation=[
                        f"Resume: spark download {target}",
                        "Adoptable models are the unregistered runnable rows of: spark list",
                    ],
                    context={"query": repo},
                )
        except OSError:
            pass
        # Case-insensitive hub hit (hub ids are case-sensitive on disk, but the
        # operator often retypes them lowercased): point at the real row.
        try:
            from ..hf_cache import hf_cache_root

            root = hf_cache_root()
            want = target.replace("/", "--").lower()
            if root.is_dir():
                for d in root.iterdir():
                    if d.name.startswith("models--") and d.name[len("models--"):].lower() == want:
                        real = d.name[len("models--"):].replace("--", "/")
                        return ModelNotFoundError(
                            f"Nothing adoptable matches '{repo}' (case differs from cached '{real}').",
                            remediation=[
                                f"Check the exact row: spark list | grep -i {Path(target).name}",
                                "Adoptable models are the unregistered runnable rows of: spark list",
                            ],
                            context={"query": repo, "cached": real},
                        )
        except OSError:
            pass

    if inv.runnable_discovered:
        sample = ", ".join(m.display_id for m in inv.runnable_discovered[:5])
        more = f" (+{len(inv.runnable_discovered) - 5} more)" if len(inv.runnable_discovered) > 5 else ""
        return ModelNotFoundError(
            f"Nothing adoptable matches '{repo}'.",
            remediation=[
                f"Adoptable now: {sample}{more}",
                "Adoptable models are the unregistered rows of: spark list",
                f"Or download it first: spark download {repo}",
            ],
            context={"query": repo},
        )
    return ModelNotFoundError(
        f"Nothing adoptable matches '{repo}'.",
        remediation=[
            "Adoptable models are the unregistered rows of: spark list",
            f"Or download it first: spark download {repo}",
        ],
        context={"query": repo},
    )


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
    tl = target.lower()
    found = None
    for m in inv.runnable_discovered:
        if m.display_id.lower() == tl or m.location == target:
            found = m
            break
    if found is None:
        # Forgiving match: unique tail slug (`bonsai-8b` for
        # `prism-ml/Ternary-Bonsai-8B-mlx-2bit`), then unique prefix/substring.
        tails = [
            m
            for m in inv.runnable_discovered
            if m.display_id.lower().split("/")[-1] == tl
        ]
        if len(tails) == 1:
            found = tails[0]
        else:
            prefix = [
                m for m in inv.runnable_discovered if m.display_id.lower().startswith(tl)
            ]
            substr = [
                m for m in inv.runnable_discovered if tl in m.display_id.lower()
            ]
            unique = prefix if len(prefix) == 1 else (substr if len(substr) == 1 else [])
            if unique:
                found = unique[0]
    if found is None:
        raise _adopt_miss_error(repo, inv, ctx)
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
