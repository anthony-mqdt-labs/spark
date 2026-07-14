"""Model store scan: detect partial or orphaned download directories.

`hf download --local-dir` stages in-flight blobs as
``.cache/huggingface/download/*.incomplete`` inside the destination; on success
each fragment is moved into place, so any surviving ``*.incomplete`` marks the
snapshot as partial. A store directory with no registry entry pointing at it is
an orphan either way — spark cannot launch it and `spark list` would otherwise
never show it. Both states waste disk silently unless surfaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config.paths import SparkPaths
from .models import list_models

_HF_DL_SUBDIR = Path(".cache") / "huggingface" / "download"


@dataclass(frozen=True)
class StoreIssue:
    path: Path
    model_id: str          # directory name (the id `spark download` used)
    registered: bool
    hf_repo: str           # from the registry entry; "" when unregistered
    incomplete_files: int  # surviving *.incomplete resume fragments
    bytes_present: int

    @property
    def kind(self) -> str:
        return "partial" if self.incomplete_files else "orphaned"


def dir_bytes(root: Path) -> int:
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


def scan_store(paths: SparkPaths) -> list[StoreIssue]:
    """Return every store directory that is partially downloaded and/or not
    referenced by any registry entry. Healthy registered snapshots are skipped."""
    store = paths.model_store_dir
    if not store.is_dir():
        return []

    by_path: dict[Path, object] = {}
    for m in list_models(paths):
        if m.path:
            try:
                by_path[Path(m.path).resolve()] = m
            except OSError:
                continue

    issues: list[StoreIssue] = []
    for d in sorted(store.iterdir()):
        if not d.is_dir():
            continue
        entry = by_path.get(d.resolve())
        dl_dir = d / _HF_DL_SUBDIR
        incomplete = list(dl_dir.glob("*.incomplete")) if dl_dir.is_dir() else []
        if entry is not None and not incomplete:
            continue
        issues.append(
            StoreIssue(
                path=d,
                model_id=d.name,
                registered=entry is not None,
                hf_repo=(getattr(entry, "hf_repo", "") or ""),
                incomplete_files=len(incomplete),
                bytes_present=dir_bytes(d),
            )
        )
    return issues
