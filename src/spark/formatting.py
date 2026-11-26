"""Neutral human-readable formatting shared by core and CLI layers.

Lives at package level (not under `cli/`) so core modules such as
`registry/availability.py` can use it without violating the layering rule.
"""

from __future__ import annotations


def fmt_size(n: int) -> str:
    """Human-sized bytes: a 1 KiB file must not render as '0 MiB'."""
    if not n or n < 0:
        return "—"
    if n < 2**10:
        return f"{n} B"
    if n < 2**20:
        return f"{round(n / 2**10)} KiB"
    if n < 2**30:
        return f"{n / 2**20:.0f} MiB"
    return f"{n / 2**30:.1f} GiB"


def short(text: str, width: int) -> str:
    """One-line ellipsis for table cells and list details."""
    if len(text) <= width:
        return text
    if width <= 1:
        return "…"[:width]
    return text[: width - 1].rstrip() + "…"
