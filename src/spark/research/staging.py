"""Stage research output for operator review before it touches the registry."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..config.paths import SparkPaths
from ..config.schema import ModelEntry
from .types import ResearchOutput


def _staging_path(paths: SparkPaths, model_id: str) -> Path:
    return paths.staging_dir / f"{model_id}.json"


def stage(
    paths: SparkPaths, model_id: str, output: ResearchOutput, *,
    provider: str, hf_repo: str, host_summary: dict,
) -> Path:
    paths.staging_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "model_id": model_id,
        "hf_repo": hf_repo,
        "provider": provider,
        "staged_at": time.time(),
        "host_summary": host_summary,
        "output": output.model_dump(),
    }
    path = _staging_path(paths, model_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    import os

    os.replace(tmp, path)
    return path


def load_staged(paths: SparkPaths, model_id: str) -> dict | None:
    path = _staging_path(paths, model_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def delete_staged(paths: SparkPaths, model_id: str) -> None:
    _staging_path(paths, model_id).unlink(missing_ok=True)


def apply_output_to_entry(entry: ModelEntry, output: ResearchOutput) -> ModelEntry:
    """Apply a validated research output to a model entry (in place, returns it)."""
    entry.backend = output.recommended_backend or entry.backend
    rec = output.recommended()
    if rec:
        if rec.quant:
            entry.quant = rec.quant
        if rec.context_length:
            entry.context_length = rec.context_length
        if rec.launch_overrides:
            merged = dict(entry.launch_overrides)
            merged.update(rec.launch_overrides)
            entry.launch_overrides = merged
    entry.research_status = "registered"
    return entry
