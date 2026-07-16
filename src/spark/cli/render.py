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


def models_table(models) -> Table:
    t = Table(title="Models", expand=False)
    t.add_column("id", style="cyan")
    t.add_column("backend")
    t.add_column("quant", style="dim")
    t.add_column("format", style="dim")
    t.add_column("status", style="dim")
    for m in models:
        t.add_row(m.id, m.backend or "—", m.quant or "—", m.model_format, m.research_status)
    return t
