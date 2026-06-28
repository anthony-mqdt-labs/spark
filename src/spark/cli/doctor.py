"""`spark doctor` — probe host + runtimes, report health and budgets."""

from __future__ import annotations

import click

from ..budget import GIB, free_disk_bytes, usable_memory_bytes
from ..probe.host import load_or_probe
from .context import build_context
from .render import console, runtimes_table


@click.command(name="doctor")
@click.option("--no-cache", is_flag=True, help="Force a fresh probe.")
def doctor_command(no_cache: bool):
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
