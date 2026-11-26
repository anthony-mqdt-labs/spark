"""Rich-based rendering helpers, including actionable error output."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..errors import SparkError
from ..formatting import fmt_size, short

if TYPE_CHECKING:
    from ..inventory import Inventory
    from ..probe.host import HostProfile
    from ..registry.store import StoreIssue

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


def render_server_log(log_path: str | Path, tail: str) -> None:
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


def runtimes_table(profile: HostProfile) -> Table:
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


# --- shared visual language -----------------------------------------------------
# One glyph vocabulary across every command: ✓ done, ! attention, → next action.
# Keep these helpers small and markup-only so they are safe in tests and in
# non-interactive (piped) terminals.


# `fmt_size` and `short` are imported from the neutral `spark.formatting` module
# above so core code can use the same byte formatting without importing `cli/`.


def success(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def warning(message: str) -> None:
    console.print(f"[yellow]![/yellow] {message}")


def hint(action: str, reason: str) -> None:
    """One actionable next step: '→ reason: command'."""
    console.print(f"[cyan]→[/cyan] {reason}: [bold]{action}[/bold]")


def section(title: str) -> None:
    """A scannable section heading with breathing room above it."""
    console.print()
    console.print(f"[bold]{title}[/bold]")


@contextmanager
def working(message: str) -> Iterator[None]:
    """TTY-aware progress for multi-second operations.

    Animated spinner in an interactive terminal; one stable line when piped so
    logs and completion callers never see control characters.
    """
    if console.is_terminal:
        with console.status(message, spinner="dots"):
            yield
    else:
        console.print(f"[cyan]…[/cyan] {message}")
        yield


def _research_label(status: str) -> str:
    """Compact research label for the list view.

    `registered` means tuned flags are applied; `staged` means a result is
    waiting at `spark config review`; `pending` means research has not run.
    They are distinct states and deserve distinct labels.
    """
    return {
        "registered": "tuned",
        "manual": "manual",
        "pending": "pending",
        "staged": "staged",
    }.get(status, status or "pending")


def render_store_issues(issues: list[StoreIssue]) -> None:
    """Warn about partial/orphaned store dirs, then give one fix pointer.

    One scannable line per issue (truncated reason + size) followed by a single
    `→ fix:` hint that keeps every resume/clean command available.
    """
    if not issues:
        return
    width = getattr(console, "width", 0) or 80
    warning(
        f"{len(issues)} incomplete/unregistered store download(s) — wasting disk, invisible to run"
    )
    fixes: list[str] = []
    for i in issues:
        state = []
        if i.incomplete_files:
            state.append(f"partial ({i.incomplete_files} resume fragment(s))")
        if not i.registered:
            state.append("unregistered")
        # Budget the reason and path together so the bullet stays a single line.
        size = fmt_size(i.bytes_present)
        avail = max(24, width - (10 + len(i.model_id) + len(size)))
        reason = ", ".join(state) or "store dir has no usable snapshot"
        reason_w = min(len(reason), max(24, avail - 20))
        path_w = max(12, avail - reason_w)
        console.print(
            f"  • [cyan]{i.model_id}[/cyan] — {short(reason, reason_w)} · "
            f"{size} on disk · [dim]{short(str(i.path), path_w)}[/dim]"
        )
        fixes.append(
            f"spark download {i.hf_repo}"
            if i.hf_repo
            else f"spark download <hf-repo> --id {i.model_id}"
        )
        fixes.append(f"rm -rf {i.path}")
    hint("  ·  ".join(fixes), "fix")


def runnable_table(inv: Inventory) -> Table:
    """What this host can serve right now: disk truth, registry or not.

    Registered models show their backend; discovered-but-unregistered ones show
    their source (hub/store) and a nudge to `spark adopt`. External runtimes
    (routers, daemons) are listed too — they own their weights.
    """
    t = Table(title="Ready to run", expand=False)
    cols = (
        ("id", "cyan"),
        ("src", ""),
        ("backend", ""),
        ("size", "dim"),
        ("research", "dim"),
    )
    for name, style in cols:
        t.add_column(name, style=style, no_wrap=True, overflow="ellipsis")
    for entry, avail in inv.runnable_registered:
        t.add_row(
            entry.id,
            "[green]weights[/green]",
            entry.backend or "auto",
            fmt_size(avail.bytes_present) if avail.bytes_present else "—",
            _research_label(entry.research_status),
        )
    for entry, _avail in inv.external:
        t.add_row(
            entry.id,
            "[magenta]daemon[/magenta]",
            entry.backend or "—",
            "—",
            _research_label(entry.research_status),
        )
    for entry, _avail in inv.unverifiable:
        t.add_row(
            entry.id,
            "[dim]unverified[/dim]",
            entry.backend or "auto",
            "—",
            _research_label(entry.research_status),
        )
    for m in inv.runnable_discovered:
        t.add_row(
            m.display_id,
            f"[yellow]{m.source}[/yellow]",
            m.model_format,
            fmt_size(m.bytes_present) if m.bytes_present else "—",
            "not registered",
        )
    return t


def render_inventory_notes(inv: Inventory, *, verbose: bool = False) -> None:
    """One-line pointers for everything that is *not* runnable.

    Reasons are ellipsis-truncated to the console width (never wrapped); each
    pointer is an actionable `→` hint. Default = signal + pointer, `--all` = the
    full per-item detail (ids, reasons, fetch/drop commands).
    """
    width = getattr(console, "width", 0) or 80
    w = max(24, width - 6)

    if inv.runnable_discovered:
        warning(f"{len(inv.runnable_discovered)} on-disk model(s) not in the registry")
        hint("spark adopt <repo>", "run directly or keep (backend + research)")
    if inv.missing:
        warning(
            f"{len(inv.missing)} registered model(s) have no weights on this host"
            + ("" if verbose else " (--all for details)")
        )
        if verbose:
            for entry, avail in inv.missing:
                detail = short(avail.detail or "weights missing", max(16, w - len(entry.id) - 4))
                loc = f" [dim]· {short(avail.location, w)}[/dim]" if avail.location else ""
                console.print(f"  • [cyan]{entry.id}[/cyan] — {detail}{loc}")
                if entry.hf_repo:
                    hint(f"spark download {entry.hf_repo}", "fetch")
                hint(f"spark forget {entry.id}", "drop")
    if inv.non_servable and verbose:
        warning(f"{len(inv.non_servable)} cached repo(s) spark cannot serve (embeddings/speech)")
        for m in inv.non_servable:
            reason = short(m.reason or "unsupported model type", max(16, w - len(m.display_id) - 4))
            console.print(f"  • [dim]{m.display_id} — {reason}[/dim]")
    elif inv.non_servable:
        console.print(
            f"[dim]{len(inv.non_servable)} cached repo(s) are not chat models "
            "(embeddings/speech) — hidden, see --all[/dim]"
        )
