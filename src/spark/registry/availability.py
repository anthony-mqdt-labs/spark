"""Do a registered model's weights actually exist on this host?

The registry is a list of *intentions*: one TOML per model, written by
``spark download`` or hand-edited. The read path never checked that the weights
those entries point at still exist, so an entry outlived its weights silently —
``spark list`` kept advertising models whose weights had been reclaimed, and
``spark run`` then triggered a multi-gigabyte fetch (or a hang) at the first
request instead of failing honestly.

This module is the missing reconciliation: one predicate, consulted by the
listing, the completion feed, and the launch preflight, so all three agree.

States
------
``local``        weights in spark's own store (``entry.path``)
``hub``          weights in the HF hub cache (``entry.hf_repo``, no path)
``missing``      the entry asserts weights that are not on this host
``external``     the runtime owns its weights (router child, ollama daemon,
                 relay) — presence is not spark's to judge
``unverifiable`` the entry asserts nothing (no path, no repo)

Only ``missing`` is a defect, and only when the *entry claims* weights: an
entry with neither ``path`` nor ``hf_repo`` (an ollama tag, say) is left alone
rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config.paths import SparkPaths
from ..config.schema import ModelEntry, SparkConfig
from ..hf_cache import bytes_in, snapshot_dir

LOCAL = "local"
HUB = "hub"
MISSING = "missing"
EXTERNAL = "external"
UNVERIFIABLE = "unverifiable"


def _fmt_bytes(n: int) -> str:
    if n >= 2**30:
        return f"{n / 2**30:.1f} GiB"
    if n >= 2**20:
        return f"{n / 2**20:.0f} MiB"
    return f"{n / 2**10:.0f} KiB"


@dataclass(frozen=True)
class Availability:
    """Where an entry's weights are, or why they are not."""

    state: str
    detail: str
    location: str = ""
    bytes_present: int = 0
    downloadable: bool = False  # hf_repo known -> `spark download` can restore it

    @property
    def ok(self) -> bool:
        """True unless the entry asserts weights that are absent."""
        return self.state != MISSING

    @property
    def label(self) -> str:
        """Compact cell for the model table."""
        if self.state == LOCAL:
            return f"local {_fmt_bytes(self.bytes_present)}"
        if self.state == HUB:
            return f"hub {_fmt_bytes(self.bytes_present)}"
        if self.state == MISSING:
            return "MISSING"
        if self.state == EXTERNAL:
            return "external"
        return "—"

    def describe(self) -> str:
        """One line for the completion preview / warnings."""
        if self.location:
            return f"{self.state} ({self.detail}) · {self.location}"
        return f"{self.state} ({self.detail})"


def requires_weights_for(entry: ModelEntry, config: SparkConfig | None) -> bool:
    """Whether the entry's runtime needs weights on this host.

    Routers, daemon-style runtimes and relays manage their own weights, so their
    presence is not spark's to verify. An entry with no backend yet (freshly
    downloaded, research still pending) is treated as weight-bearing — that is
    the common case and the safe default.
    """
    if config is None or not entry.backend:
        return True
    rt = config.runtimes.get(entry.backend)
    return rt.requires_local_weights if rt else True


def resolve_availability(
    entry: ModelEntry,
    paths: SparkPaths,
    *,
    requires_weights: bool = True,
    with_bytes: bool = True,
) -> Availability:
    """Where ``entry``'s weights live right now, checked against the filesystem.

    ``with_bytes=False`` skips the size walk — used by the shell-completion feed,
    which must stay fast per README §9.3; the state itself is still exact.
    """
    if not requires_weights:
        return Availability(
            state=EXTERNAL,
            detail=f"'{entry.backend}' runtime owns its weights",
        )

    if entry.path:
        p = Path(entry.path).expanduser()
        if p.is_dir() or p.is_file():
            if p.is_dir():
                size = bytes_in(p) if with_bytes else (1 if any(p.iterdir()) else 0)
            else:
                size = p.stat().st_size
            if size > 0:
                return Availability(
                    state=LOCAL, detail="in the spark model store",
                    location=str(p), bytes_present=size,
                    downloadable=bool(entry.hf_repo),
                )
            return Availability(
                state=MISSING, detail="store directory is empty",
                location=str(p), downloadable=bool(entry.hf_repo),
            )
        return Availability(
            state=MISSING, detail="store path does not exist",
            location=str(p), downloadable=bool(entry.hf_repo),
        )

    if entry.hf_repo:
        snap = snapshot_dir(entry.hf_repo)
        if snap is not None:
            return Availability(
                state=HUB, detail="in the HF hub cache",
                location=str(snap), bytes_present=bytes_in(snap) if with_bytes else 0,
                downloadable=True,
            )
        return Availability(
            state=MISSING, detail="not in the HF hub cache",
            location=entry.hf_repo, downloadable=True,
        )

    return Availability(
        state=UNVERIFIABLE,
        detail="entry declares neither path nor hf_repo",
    )


def availability_map(
    entries: list[ModelEntry],
    paths: SparkPaths,
    *,
    config: SparkConfig | None = None,
) -> dict[str, Availability]:
    """Availability for a whole registry, keyed by model id."""
    return {
        e.id: resolve_availability(
            e, paths, requires_weights=requires_weights_for(e, config)
        )
        for e in entries
    }


def missing(entries: list[ModelEntry], avail: dict[str, Availability]) -> list[ModelEntry]:
    """Entries whose weights are absent, in registry order."""
    return [e for e in entries if avail.get(e.id) is not None and not avail[e.id].ok]


__all__ = [
    "Availability",
    "LOCAL",
    "HUB",
    "MISSING",
    "EXTERNAL",
    "UNVERIFIABLE",
    "availability_map",
    "missing",
    "requires_weights_for",
    "resolve_availability",
]
