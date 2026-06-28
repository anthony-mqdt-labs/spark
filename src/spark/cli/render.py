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
