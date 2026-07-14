"""`spark doctor` — probe host + runtimes, report health and budgets."""

from __future__ import annotations

import subprocess
import time

import click

from ..budget import GIB, free_disk_bytes, usable_memory_bytes
from ..config.schema import RuntimeDef
from ..errors import RuntimeBackendError
from ..probe.host import (
    dep_fix_command,
    load_or_probe,
    missing_python_deps,
    split_python_req,
)
from ..registry.store import scan_store
from ..telemetry import get_telemetry
from .context import build_context
from .render import console, render_store_issues, runtimes_table


@click.command(name="doctor")
@click.option("--no-cache", is_flag=True, help="Force a fresh probe.")
@click.option(
    "--fix", "apply_fix", is_flag=True,
    help="Install missing runtime Python deps into each runtime's own tool env "
         "(only runtimes installed via `uv tool install`; the command is shown "
         "before it runs).",
)
def doctor_command(no_cache: bool, apply_fix: bool):
    """Diagnose host capabilities and runtime availability."""
    ctx = build_context()
    profile = load_or_probe(
        ctx.config, ctx.paths.host_profile_file, force=no_cache
    )

    mem_total = profile.total_memory_bytes / GIB
    mem_budget = usable_memory_bytes(profile, ctx.config.memory) / GIB
    free = free_disk_bytes(ctx.paths.data_dir) / GIB

    console.print(
        f"[bold]Host[/bold]: {profile.chip}  ·  {profile.arch}  ·  "
        f"{profile.cpu_count} cores  ·  {mem_total:.0f} GiB unified"
    )
    console.print(
        f"[bold]Budget[/bold]: ~{mem_budget:.1f} GiB usable for weights  ·  "
        f"{free:.1f} GiB free disk"
        + ("  [yellow](low disk)[/yellow]" if free < 20 else "")
    )
    console.print(runtimes_table(profile))
    render_store_issues(scan_store(ctx.paths))

    # Runtime readiness: a runtime can be on PATH yet unable to serve because its
    # own venv is missing inference-time Python deps (declared in [detect]).
    unfixed: list[str] = []
    for name in profile.available_runtimes:
        rt = ctx.config.runtimes.get(name)
        if not rt or not rt.detect.python_requires:
            continue
        missing = missing_python_deps(rt.detect.binary, rt.detect.python_requires)
        if not missing:
            continue
        mods = ", ".join(split_python_req(r)[0] for r in missing)
        fix_argv = dep_fix_command(rt.install_hint, missing)
        if apply_fix and fix_argv:
            if not _fix_python_deps(rt, missing, fix_argv):
                unfixed.append(name)
            continue
        if fix_argv:
            fix = f"{' '.join(fix_argv)}  (or: spark doctor --fix)"
        else:
            with_flags = " ".join(
                f"--with {split_python_req(r)[1]}" for r in missing
            )
            fix = f"add to {name}'s venv: {with_flags}"
        console.print(
            f"[yellow]![/yellow] {name}: missing Python deps "
            f"[red]{mods}[/red] (server starts but model-load "
            f"will fail). Fix: [bold]{fix}[/bold]"
        )

    secret_ok = ctx.secrets.is_available()
    console.print(
        f"[bold]Secrets[/bold]: {ctx.secrets.backend_name} "
        + ("[green]available[/green]" if secret_ok else "[red]unavailable[/red]")
    )
    avail = profile.available_runtimes
    if not avail:
        console.print("[yellow]No runtimes available — install one above.[/yellow]")
    else:
        console.print(f"[green]Ready.[/green] Available: {', '.join(avail)}")

    if unfixed:
        raise RuntimeBackendError(
            f"--fix could not repair Python deps for: {', '.join(unfixed)}",
            code="RT_DEPS_UNFIXED",
            remediation=[
                "scroll up for the installer output",
                "run the printed `uv tool install ... --with ...` command by hand",
                "re-check with `spark doctor`",
            ],
            context={"runtimes": unfixed},
        )


def _fix_python_deps(
    rt: RuntimeDef, missing: list[str], argv: list[str]
) -> bool:
    """Run the derived installer for `rt`, then re-verify. Returns True when the
    runtime's interpreter can import everything afterwards."""
    log = get_telemetry()
    mods = [split_python_req(r)[0] for r in missing]
    console.print(
        f"[yellow]![/yellow] {rt.name}: missing [red]{', '.join(mods)}[/red] — "
        f"running [bold]{' '.join(argv)}[/bold]"
    )
    log.info(
        "cli", "python_deps_fix_started",
        runtime=rt.name, missing=mods, command=argv,
    )
    t0 = time.monotonic()
    try:
        # Inherit stdio so the operator sees uv's resolve/download progress live.
        rc = subprocess.run(argv, check=False).returncode
    except OSError as exc:
        log.error(
            "cli", "python_deps_fix_failed",
            runtime=rt.name, error_type=type(exc).__name__, message=str(exc),
        )
        console.print(f"[red]✗[/red] {rt.name}: could not run installer: {exc}")
        return False
    wall_s = round(time.monotonic() - t0, 1)
    if rc != 0:
        log.error(
            "cli", "python_deps_fix_failed",
            runtime=rt.name, rc=rc, wall_s=wall_s,
        )
        console.print(
            f"[red]✗[/red] {rt.name}: installer exited {rc} — deps unchanged."
        )
        return False
    still = missing_python_deps(rt.detect.binary, rt.detect.python_requires)
    if still:
        mods_left = [split_python_req(r)[0] for r in still]
        log.warn(
            "cli", "python_deps_fix_incomplete",
            runtime=rt.name, still_missing=mods_left, wall_s=wall_s,
        )
        console.print(
            f"[red]✗[/red] {rt.name}: installer succeeded but "
            f"[red]{', '.join(mods_left)}[/red] still not importable — check the "
            f"module:pip mapping in the runtime's [detect].python_requires."
        )
        return False
    log.info(
        "cli", "python_deps_fix_succeeded",
        runtime=rt.name, installed=mods, wall_s=wall_s,
    )
    console.print(
        f"[green]✓[/green] {rt.name}: {', '.join(mods)} now importable "
        f"({wall_s}s)."
    )
    return True
