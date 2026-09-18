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


def models_table(models, avail=None) -> Table:
    """The registry listing, with a weight-availability column.

    ``avail`` maps model id -> :class:`~spark.registry.availability.Availability`.
    Omitted (or missing an id) renders as an em dash, so callers that do not have
    the check to hand still get a table.
    """
    t = Table(title="Models", expand=False)
    t.add_column("id", style="cyan")
    t.add_column("avail")
    t.add_column("backend")
    t.add_column("quant", style="dim")
    t.add_column("format", style="dim")
    t.add_column("status", style="dim")
    for m in models:
        t.add_row(
            m.id,
            _avail_cell((avail or {}).get(m.id)),
            m.backend or "—",
            m.quant or "—",
            m.model_format,
            m.research_status,
        )
    return t


def _avail_cell(a) -> str:
    if a is None:
        return "[dim]—[/dim]"
    if a.state in ("local", "hub"):
        colour = "green" if a.state == "local" else "cyan"
        return f"[{colour}]{a.state}[/{colour}] [dim]{_fmt_bytes(a.bytes_present)}[/dim]"
    if a.state == "missing":
        colour = "yellow" if a.downloadable else "red"
        return f"[{colour}]MISSING[/{colour}]"
    return f"[dim]{a.state}[/dim]"


def render_missing_weights(entries, avail) -> None:
    """Warn about registered models whose weights are not on this host.

    The inverse of :func:`render_store_issues`: that one finds disk spark cannot
    see, this one finds entries spark would advertise and then fail to serve.
    """
    gone = [e for e in entries if (avail.get(e.id) is not None and not avail[e.id].ok)]
    if not gone:
        return
    console.print(
        f"[yellow]![/yellow] [bold]{len(gone)} registered model(s) have no "
        f"weights on this host[/bold] (the entry outlived the weights):"
    )
    for e in gone:
        a = avail[e.id]
        console.print(
            f"  • [cyan]{e.id}[/cyan] — {a.detail}"
            + (f" · {a.location}" if a.location else "")
        )
        if e.hf_repo:
            console.print(f"    fetch: [bold]spark download {e.hf_repo}[/bold]")
        console.print(f"    drop the entry: [bold]spark forget {e.id}[/bold]")

