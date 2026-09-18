"""Hugging Face hub-cache resolution — one rule, two callers.

Two different questions are asked of ``~/.cache/huggingface/hub``, and they must
agree or the machine contradicts itself:

* ``runtimes/omlx.py`` asks *where are the weights oMLX should serve?* — it
  resolves a repo id to its snapshot directory.
* ``registry/availability.py`` asks *do this entry's weights exist at all?* —
  needed for entries that resolve via ``hf_repo`` with ``path=""``, which is how
  a model keeps working after its store copy is reclaimed (the Ornith
  precedent, CONTINUE.md 2026-07-15).

Both go through :func:`snapshot_dir`, so "present in the hub cache" has exactly
one definition.

Note that ``mlx_lm.server`` keeps a *second, independent* view of this same
cache for its ``/v1/models`` endpoint (``scan_cache_dir()`` filtered to
MLX-shaped repos). That view is what agent harnesses read, so a hub-cache
deletion is visible to harnesses but invisible to spark's registry — the
asymmetry README §9 documents. This module describes the cache; it never
mutates it.
"""

from __future__ import annotations

import os
from pathlib import Path


def hf_cache_root() -> Path:
    """The hub cache root, honouring HF's own env overrides."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_dir(repo: str) -> Path:
    """The cache directory for a repo id (``org/name`` -> ``models--org--name``)."""
    return hf_cache_root() / ("models--" + repo.replace("/", "--"))


def snapshot_dir(repo: str) -> Path | None:
    """Resolve a repo id to its local snapshot dir in the hub cache, or None.

    Prefers the commit pinned by ``refs/main``; falls back to the newest
    snapshot that actually contains a ``config.json``.
    """
    if not repo:
        return None
    snaps = repo_dir(repo) / "snapshots"
    if not snaps.is_dir():
        return None
    ref = snaps.parent / "refs" / "main"
    if ref.is_file():
        pinned = snaps / ref.read_text().strip()
        if (pinned / "config.json").is_file():
            return pinned
    valid = [d for d in snaps.iterdir() if (d / "config.json").is_file()]
    return max(valid, key=lambda p: p.stat().st_mtime) if valid else None


def bytes_in(root: Path) -> int:
    """Physical file bytes under ``root`` (0 when absent).

    Follows symlinked snapshots (the hub cache stores blobs once and links
    them), and skips dangling links, which is what a partially-reclaimed
    snapshot looks like.
    """
    total = 0
    if not root.is_dir():
        return 0
    for f in root.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total
