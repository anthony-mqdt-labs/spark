"""Rich-based rendering helpers, including actionable error output."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..errors import SparkError

console = Console()
err_console = Console(stderr=True)


def render_error(exc: SparkError) -> None:
    """Render a SparkError as {what / why / do-this-next}."""
    body = [f"[bold red]{exc.message}[/bold red]"]
    if exc.code:
        body.append(f"[dim]code: {exc.code}[/dim]")
    if exc.remediation:
        body.append("")
        body.append("[bold]Try:[/bold]")
        for step in exc.remediation:
            body.append(f"  • {step}")
    err_console.print(
        Panel(
            "\n".join(body),
            title="spark error",
            border_style="red",
            expand=False,
        )
    )


def render_server_log(log_path, tail: str) -> None:
    """On launch failure, show the tail of the child server log (its output was
    redirected off the terminal to keep startup clean)."""
    if not tail.strip():
        return
    err_console.print(
        Panel(
            tail,
            title=f"server log · {log_path}",
            border_style="red",
            expand=False,
        )
    )


def runtimes_table(profile) -> Table:
    t = Table(title="Runtimes", expand=False)
    t.add_column("runtime", style="cyan")
    t.add_column("status")
    t.add_column("version", style="dim")
    t.add_column("detail", style="dim")
    for name, s in profile.runtimes.items():
        status = "[green]available[/green]" if s.available else "[red]unavailable[/red]"
        detail = s.path if s.available else (s.reason or "")
        if not s.available and s.install_hint:
            detail = f"{detail}  →  {s.install_hint}"
        t.add_row(name, status, s.version or "—", detail)
    return t


def _fmt_bytes(n: int) -> str:
    if n >= 2**30:
        return f"{n / 2**30:.1f} GiB"
    return f"{n / 2**20:.0f} MiB"


def render_store_issues(issues) -> None:
    """Warn about partial/orphaned model-store dirs with resume/clean steps."""
    if not issues:
        return
    console.print(
        f"[yellow]![/yellow] [bold]{len(issues)} incomplete or unregistered "
        f"download(s) in the model store[/bold] (wasting disk, invisible to run):"
    )
    for i in issues:
        state = []
        if i.incomplete_files:
            state.append(f"partial ({i.incomplete_files} resume fragment(s))")
        if not i.registered:
            state.append("unregistered")
        console.print(
            f"  • [cyan]{i.model_id}[/cyan] — {', '.join(state)} · "
            f"{_fmt_bytes(i.bytes_present)} on disk · {i.path}"
        )
        resume = (
            f"spark download {i.hf_repo}"
            if i.hf_repo
            else f"spark download <hf-repo> --id {i.model_id}"
        )
        console.print(f"    resume: [bold]{resume}[/bold]  ·  clean: rm -rf {i.path}")


def runnable_table(inv) -> Table:
    """What this host can serve right now: disk truth, registry or not.

    Registered models show their backend; discovered-but-unregistered ones show
    their source (hub/store) and a nudge to `spark adopt`. External runtimes
    (routers, daemons) are listed too — they own their weights.
    """
    t = Table(title="Ready to run", expand=False)
    t.add_column("id", style="cyan")
    t.add_column("src")
    t.add_column("backend")
    t.add_column("size", style="dim")
    t.add_column("status", style="dim")
    for entry, avail in inv.runnable_registered:
        t.add_row(
            entry.id,
            "[green]weights[/green]",
            entry.backend or "auto",
            _fmt_bytes(avail.bytes_present) if avail.bytes_present else "—",
            entry.research_status,
        )
    for entry, _avail in inv.external:
        t.add_row(entry.id, "[magenta]daemon[/magenta]", entry.backend or "—", "—", entry.research_status)
    for entry, _avail in inv.unverifiable:
        t.add_row(entry.id, "[dim]unverified[/dim]", entry.backend or "auto", "—", entry.research_status)
    for m in inv.runnable_discovered:
        t.add_row(
            m.display_id,
            f"[yellow]{m.source}[/yellow]",
            m.model_format,
            _fmt_bytes(m.bytes_present) if m.bytes_present else "—",
            "not registered",
        )
    return t


def render_inventory_notes(inv, *, verbose: bool = False) -> None:
    """One-line pointers for everything that is *not* runnable.

    Missing entries name their fix; unregistered runnable models name `adopt`;
    non-servable cache is a count unless verbose. This keeps the default view
    to signal (what can run) plus pointers, not pages of defects.
    """
    if inv.runnable_discovered:
        console.print(
            f"[dim]{len(inv.runnable_discovered)} on-disk model(s) not in the registry — "
            f"run directly, or keep with: spark adopt <repo>[/dim]"
        )
    if inv.missing:
        console.print(
            f"[yellow]![/yellow] [bold]{len(inv.missing)} registered model(s) have no "
            f"weights on this host[/bold]"
            + (" [dim](--all for details)[/dim]" if not verbose else ":")
        )
        if verbose:
            for entry, avail in inv.missing:
                loc = f" · {avail.location}" if avail.location else ""
                console.print(f"  • [cyan]{entry.id}[/cyan] — {avail.detail}{loc}")
                if entry.hf_repo:
                    console.print(f"    fetch: [bold]spark download {entry.hf_repo}[/bold]")
                console.print(f"    drop the entry: [bold]spark forget {entry.id}[/bold]")
    if inv.non_servable and verbose:
        console.print(
            f"[dim]{len(inv.non_servable)} cached repo(s) spark cannot serve "
            f"(embeddings/speech):[/dim]"
        )
        for m in inv.non_servable:
            console.print(f"  • [dim]{m.display_id} — {m.reason}[/dim]")
    elif inv.non_servable:
        console.print(
            f"[dim]{len(inv.non_servable)} cached repo(s) are not chat models "
            f"(embeddings/speech) — hidden, see --all[/dim]"
        )

