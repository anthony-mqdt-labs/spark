"""oMLX backend — production multi-model MLX server (`omlx serve --model-dir`).

oMLX (>=0.4) serves every model it finds as a *subdirectory* of ``--model-dir``
(each subdir a real model: config.json + safetensors). It has no single-model or
positional form and does not fetch weights itself. spark serves one model per
launch, so :meth:`prepare` stages a private model-dir under ``run_dir`` holding a
single symlink named after the model id, and points ``--model-dir`` at it; clients
then address the model by that id.

Weights must already be local: either ``entry.path`` (set by ``spark download``)
or the HF hub cache for ``hf_repo``-only entries. If neither resolves, the launch
aborts with remediation rather than silently serving nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..config.paths import resolve_paths
from ..config.schema import ModelEntry
from ..errors import RuntimeBackendError
from .base import Backend, register


def _hf_cache_snapshot(repo: str) -> Path | None:
    """Resolve a repo id to its local snapshot dir in the HF hub cache, or None."""
    if not repo:
        return None
    if os.environ.get("HF_HUB_CACHE"):
        root = Path(os.environ["HF_HUB_CACHE"])
    elif os.environ.get("HF_HOME"):
        root = Path(os.environ["HF_HOME"]) / "hub"
    else:
        root = Path.home() / ".cache" / "huggingface" / "hub"
    snaps = root / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not snaps.is_dir():
        return None
    # Prefer the commit pinned by refs/main; fall back to newest valid snapshot.
    ref = snaps.parent / "refs" / "main"
    if ref.is_file():
        pinned = snaps / ref.read_text().strip()
        if (pinned / "config.json").is_file():
            return pinned
    valid = [d for d in snaps.iterdir() if (d / "config.json").is_file()]
    return max(valid, key=lambda p: p.stat().st_mtime) if valid else None


@register("omlx")
class OmlxBackend(Backend):
    def _source_dir(self, entry: ModelEntry) -> Path:
        """The on-disk model directory oMLX will serve (config.json + weights)."""
        if entry.path:
            local = Path(entry.path).expanduser()
            if (local / "config.json").is_file():
                return local
        snap = _hf_cache_snapshot(entry.hf_repo)
        if snap is not None:
            return snap
        raise RuntimeBackendError(
            f"oMLX needs local weights for '{entry.id}', but none were found.",
            remediation=[
                f"Download them first: spark download {entry.hf_repo or entry.id}",
                "oMLX `serve` only serves models already on disk (no auto-fetch).",
            ],
        )

    def _model_dir(self, entry: ModelEntry) -> Path:
        # Private per-model dir; its single subdir (a symlink) is the served model.
        return resolve_paths().run_dir / "omlx" / entry.id

    def resolve_model_dir(self, entry: ModelEntry) -> str:
        return str(self._model_dir(entry))

    def prepare(self, entry: ModelEntry, *, secrets=None) -> None:
        src = self._source_dir(entry)
        model_dir = self._model_dir(entry)
        model_dir.mkdir(parents=True, exist_ok=True)
        link = model_dir / entry.id  # subdir name -> oMLX model id
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src, target_is_directory=True)
