"""Model registry: one TOML per model under the data dir.

Provides load/save plus fuzzy resolution (exact id -> alias -> unique
prefix/substring) with actionable errors on miss/ambiguity. TOML is written with a
tiny in-house emitter to avoid a TOML *writer* dependency.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from ..config.paths import SparkPaths, resolve_paths
from ..config.schema import ModelEntry
from ..errors import AmbiguousModelError, ModelError, ModelNotFoundError

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _validate_id(model_id: str) -> str:
    if not _ID_RE.match(model_id or ""):
        raise ModelError(
            f"Invalid model id {model_id!r}.",
            remediation=["Use letters/digits/_/.- (start alphanumeric, <=128 chars)."],
        )
    return model_id


def _entry_path(paths: SparkPaths, model_id: str) -> Path:
    return paths.models_dir / f"{model_id}.toml"


# --- TOML emit (minimal, typed) ------------------------------------------------
def _emit_value(v) -> str:
    import json

    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_emit_value(x) for x in v) + "]"
    raise ModelError(f"Cannot serialize value of type {type(v)!r} to TOML.")


def _dump_entry(entry: ModelEntry) -> str:
    data = entry.model_dump()
    overrides = data.pop("launch_overrides", {}) or {}
    lines = ["# spark model registry entry (auto-managed)."]
    for k, v in data.items():
        if v is None or v == "":
            continue
        if isinstance(v, list) and not v:
            continue
        lines.append(f"{k} = {_emit_value(v)}")
    if overrides:
        lines.append("")
        lines.append("[launch_overrides]")
        for k, v in overrides.items():
            lines.append(f"{k} = {_emit_value(v)}")
    return "\n".join(lines) + "\n"


# --- CRUD ----------------------------------------------------------------------
def save_model(entry: ModelEntry, paths: SparkPaths | None = None) -> Path:
    paths = paths or resolve_paths()
    _validate_id(entry.id)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    path = _entry_path(paths, entry.id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(_dump_entry(entry), encoding="utf-8")
    import os

    os.replace(tmp, path)
    return path


def load_model_file(path: Path) -> ModelEntry:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return ModelEntry.model_validate(data)


def list_models(paths: SparkPaths | None = None) -> list[ModelEntry]:
    paths = paths or resolve_paths()
    if not paths.models_dir.is_dir():
        return []
    out: list[ModelEntry] = []
    for f in sorted(paths.models_dir.glob("*.toml")):
        try:
            out.append(load_model_file(f))
        except Exception:
            continue  # skip malformed entries rather than failing the listing
    return out


def delete_model(model_id: str, paths: SparkPaths | None = None) -> bool:
    paths = paths or resolve_paths()
    path = _entry_path(paths, model_id)
    if path.exists():
        path.unlink()
        return True
    return False


# --- resolution ----------------------------------------------------------------
def resolve_model(query: str, paths: SparkPaths | None = None) -> ModelEntry:
    """Resolve a CLI query to one model: exact id -> alias -> unique
    prefix/substring (case-insensitive)."""
    models = list_models(paths)
    if not models:
        raise ModelNotFoundError(
            "No models are registered yet.",
            remediation=[
                "Register one with: spark download <hf-repo>",
                "Or list models with: spark list",
            ],
        )
    q = query.strip().lower()

    for m in models:  # exact id
        if m.id.lower() == q:
            return m
    for m in models:  # alias
        if any(a.lower() == q for a in m.aliases):
            return m

    prefix = [m for m in models if m.id.lower().startswith(q)]
    if len(prefix) == 1:
        return prefix[0]
    substr = [m for m in models if q in m.id.lower()]
    if len(substr) == 1:
        return substr[0]

    candidates = prefix or substr
    if len(candidates) > 1:
        raise AmbiguousModelError(
            f"'{query}' matches multiple models.",
            remediation=[f"Be more specific: {', '.join(m.id for m in candidates)}"],
            context={"matches": [m.id for m in candidates]},
        )
    raise ModelNotFoundError(
        f"No model matches '{query}'.",
        remediation=[
            "List registered models: spark list",
            f"Download it: spark download <hf-repo-for-{query}>",
        ],
        context={"query": query, "known": [m.id for m in models]},
    )
