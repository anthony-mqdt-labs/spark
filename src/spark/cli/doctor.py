"""`spark doctor` — probe host + runtimes, report health and budgets."""

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import replace

import click

from ..budget import GIB, free_disk_bytes, usable_memory_bytes
from ..config.schema import RuntimeDef
from ..errors import RuntimeBackendError
from ..probe.host import (
    HostProfile,
    RuntimeStatus,
    dep_fix_command,
    load_or_probe,
    missing_python_deps,
    split_python_req,
)
from ..registry.store import scan_store
from ..telemetry import get_telemetry
from .context import build_context
from .render import (
    console,
    fmt_size,
    hint,
    render_store_issues,
    runtimes_table,
    section,
    short,
    success,
    warning,
)


@click.command(name="doctor")
@click.option("--no-cache", is_flag=True, help="Force a fresh probe.")
@click.option(
    "--fix",
    "apply_fix",
    is_flag=True,
    help="Install missing runtime Python deps into each runtime's own tool env "
    "(only runtimes installed via `uv tool install`; the command is shown "
    "before it runs).",
)
def doctor_command(no_cache: bool, apply_fix: bool) -> None:
    """Diagnose host capabilities and runtime availability."""
    ctx = build_context()
    profile = load_or_probe(
        ctx.config,
        ctx.paths.host_profile_file,
        max_age_s=ctx.config.probe.max_age_s,
        force=no_cache,
    )

    mem_total = profile.total_memory_bytes
    mem_budget = usable_memory_bytes(profile, ctx.config.memory)
    free = free_disk_bytes(ctx.paths.data_dir)

    console.print(
        f"[bold]Host[/bold]: {profile.chip}  ·  {profile.arch}  ·  "
        f"{profile.cpu_count} cores  ·  {fmt_size(mem_total)} unified"
    )
    console.print(
        f"[bold]Budget[/bold]: ~{fmt_size(mem_budget)} usable for weights  ·  "
        f"{fmt_size(free)} free disk"
    )
    if free < 20 * GIB:
        warning(f"low disk: only {fmt_size(free)} free — downloads may be refused")

    # Display-only copy: truncate the fields that feed the table's `detail`
    # column so long install hints cannot wrap rows into paragraphs. The real
    # profile (cache, repair, availability) is never mutated.
    console.print(runtimes_table(_compact_profile(profile)))
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
            with_flags = " ".join(f"--with {split_python_req(r)[1]}" for r in missing)
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

    _render_next_steps(profile)

    avail = profile.available_runtimes
    if not avail:
        warning("no runtimes available — see Next steps above")
    else:
        success(f"{len(avail)}/{len(profile.runtimes)} runtimes available")
        hint("spark list", "next: see what you can run")

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


#: Installer commands a hint can be shown verbatim as a `→ install <name>: <cmd>`.
_INSTALL_CMDS = frozenset(
    {
        "uv",
        "pip",
        "pip3",
        "pipx",
        "python",
        "python3",
        "brew",
        "apt",
        "apt-get",
        "conda",
        "npm",
        "cargo",
        "go",
        "poetry",
    }
)


def _is_install_command(text: str) -> bool:
    """Whether an install_hint is a runnable command rather than prose advice."""
    try:
        argv = shlex.split(text.strip())
    except ValueError:
        return False
    return bool(argv) and argv[0].rsplit("/", 1)[-1] in _INSTALL_CMDS


def _compact_profile(profile: HostProfile, width: int = 28) -> HostProfile:
    """Display-only copy: clamp fields that feed `runtimes_table`'s detail cell.

    Truncating reason/path/install_hint keeps every row to at most two physical
    lines. The original profile is untouched, so probe caching, availability and
    --fix repair all still see the full values.
    """
    runtimes: dict[str, RuntimeStatus] = {
        name: replace(
            status,
            path=short(status.path, width),
            reason=short(status.reason, width),
            install_hint=short(status.install_hint, width),
        )
        for name, status in profile.runtimes.items()
    }
    return replace(profile, runtimes=runtimes)


def _render_next_steps(profile: HostProfile, *, cap: int = 5) -> None:
    """One actionable line per unavailable runtime, capped for scannability."""
    unavailable = [s for s in profile.runtimes.values() if not s.available]
    if not unavailable:
        return
    section("Next steps")
    for status in unavailable[:cap]:
        if _is_install_command(status.install_hint):
            hint(status.install_hint, f"install {status.name}")
        else:
            detail = status.install_hint or status.reason or "unavailable"
            console.print(f"  • {status.name}: {short(detail, 60)}")
    extra = len(unavailable) - cap
    if extra > 0:
        console.print(f"  • ... and {extra} more")


def _fix_python_deps(rt: RuntimeDef, missing: list[str], argv: list[str]) -> bool:
    """Run the derived installer for `rt`, then re-verify. Returns True when the
    runtime's interpreter can import everything afterwards."""
    log = get_telemetry()
    mods = [split_python_req(r)[0] for r in missing]
    console.print(
        f"[yellow]![/yellow] {rt.name}: missing [red]{', '.join(mods)}[/red] — "
        f"running [bold]{' '.join(argv)}[/bold]"
    )
    log.info(
        "cli",
        "python_deps_fix_started",
        runtime=rt.name,
        missing=mods,
        command=argv,
    )
    t0 = time.monotonic()
    try:
        # Inherit stdio so the operator sees uv's resolve/download progress live.
        rc = subprocess.run(argv, check=False).returncode
    except OSError as exc:
        log.error(
            "cli",
            "python_deps_fix_failed",
            runtime=rt.name,
            error_type=type(exc).__name__,
            message=str(exc),
        )
        console.print(f"[red]✗[/red] {rt.name}: could not run installer: {exc}")
        return False
    wall_s = round(time.monotonic() - t0, 1)
    if rc != 0:
        log.error(
            "cli",
            "python_deps_fix_failed",
            runtime=rt.name,
            rc=rc,
            wall_s=wall_s,
        )
        console.print(f"[red]✗[/red] {rt.name}: installer exited {rc} — deps unchanged.")
        return False
    still = missing_python_deps(rt.detect.binary, rt.detect.python_requires)
    if still:
        mods_left = [split_python_req(r)[0] for r in still]
        log.warn(
            "cli",
            "python_deps_fix_incomplete",
            runtime=rt.name,
            still_missing=mods_left,
            wall_s=wall_s,
        )
        console.print(
            f"[red]✗[/red] {rt.name}: installer succeeded but "
            f"[red]{', '.join(mods_left)}[/red] still not importable — check the "
            f"module:pip mapping in the runtime's [detect].python_requires."
        )
        return False
    log.info(
        "cli",
        "python_deps_fix_succeeded",
        runtime=rt.name,
        installed=mods,
        wall_s=wall_s,
    )
    console.print(f"[green]✓[/green] {rt.name}: {', '.join(mods)} now importable ({wall_s}s).")
    return True
