"""Model registry: one TOML per model under the data dir.

Provides load/save plus fuzzy resolution (exact id -> alias -> unique
prefix/substring) with actionable errors on miss/ambiguity. TOML is written with a
tiny in-house emitter to avoid a TOML *writer* dependency.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from ..config.paths import SparkPaths, resolve_paths
from ..config.schema import ModelEntry, check_model_id
from ..errors import AmbiguousModelError, ModelError, ModelNotFoundError


def validate_id(model_id: str) -> str:
    """A registry id, or ModelError with the rule (ids name files)."""
    try:
        return check_model_id(model_id)
    except ValueError as exc:
        raise ModelError(
            f"Invalid model id {model_id!r}.",
            remediation=["Use letters/digits/_/.- (start alphanumeric, <=128 chars, no '..')."],
        ) from exc


def _entry_path(paths: SparkPaths, model_id: str) -> Path:
    return paths.models_dir / f"{validate_id(model_id)}.toml"


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


def _dump_table(lines: list[str], header: str, table: dict) -> None:
    """Emit a ``[header]`` block of scalar key/values (skips empties)."""
    lines.append("")
    lines.append(f"[{header}]")
    for k, v in table.items():
        if v is None or v == "":
            continue
        lines.append(f"{k} = {_emit_value(v)}")


def _dump_entry(entry: ModelEntry) -> str:
    data = entry.model_dump()
    overrides = data.pop("launch_overrides", {}) or {}
    dflash = data.pop("dflash", None)  # nested table, emitted after top-level scalars
    lines = ["# spark model registry entry (auto-managed)."]
    for k, v in data.items():
        if v is None or v == "":
            continue
        if isinstance(v, list) and not v:
            continue
        lines.append(f"{k} = {_emit_value(v)}")
    if overrides:
        _dump_table(lines, "launch_overrides", overrides)
    if dflash:
        d_over = dflash.pop("launch_overrides", {}) or {}
        _dump_table(lines, "dflash", dflash)
        if d_over:
            _dump_table(lines, "dflash.launch_overrides", d_over)
    return "\n".join(lines) + "\n"


# --- CRUD ----------------------------------------------------------------------
def save_model(entry: ModelEntry, paths: SparkPaths | None = None) -> Path:
    paths = paths or resolve_paths()
    validate_id(entry.id)
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
            entry = load_model_file(f)
        except Exception as exc:  # noqa: BLE001
            # Skip rather than fail the listing — but say so, or a hand-edit
            # typo makes a model silently vanish from every command.
            from ..telemetry import get_telemetry

            get_telemetry().warn(
                "registry", "entry_unreadable", file=f.name,
                error_type=type(exc).__name__, message=str(exc)[:256],
            )
            continue
        if entry.id != f.stem:
            # Kept (hiding a model is worse), but flagged: forget/save address
            # the file by id, so this entry cannot be dropped by its own name.
            from ..telemetry import get_telemetry

            get_telemetry().warn(
                "registry", "entry_id_mismatch", file=f.name, model_id=entry.id,
            )
        out.append(entry)
    return out


def delete_model(model_id: str, paths: SparkPaths | None = None) -> bool:
    paths = paths or resolve_paths()
    path = _entry_path(paths, model_id)
    if path.exists():
        path.unlink()
        return True
    return False


# --- resolution ----------------------------------------------------------------
from typing import TypeVar

_T = TypeVar("_T")


def fuzzy_match(
    query: str,
    candidates: list[tuple[str, _T]],
    *,
    tails: bool = False,
) -> tuple[_T | None, list[_T]]:
    """One resolution rule for every CLI path: exact -> prefix -> substring ->
    (optionally) unique repo tail. Returns ``(match, ambiguous_candidates)``.

    Callers used to reimplement this with slightly different rules, so a model
    could resolve in `spark run` but not in `spark adopt` or completion.
    """
    q = (query or "").strip().lower()
    if not q:
        return None, []

    def _unique(values: list[_T]) -> list[_T]:
        seen: set[int] = set()
        out: list[_T] = []
        for value in values:
            key = id(value)
            if key not in seen:
                seen.add(key)
                out.append(value)
        return out

    exact = _unique([value for name, value in candidates if name.lower() == q])
    if len(exact) == 1:
        return exact[0], []
    if len(exact) > 1:
        return None, exact
    prefix = _unique([value for name, value in candidates if name.lower().startswith(q)])
    if len(prefix) == 1:
        return prefix[0], []
    substr = _unique([value for name, value in candidates if q in name.lower()])
    if len(substr) == 1:
        return substr[0], []
    if tails:
        tail = _unique([
            value for name, value in candidates
            if name.lower().split("/")[-1].startswith(q)
        ])
        if len(tail) == 1:
            return tail[0], []
    return None, _unique(prefix or substr)


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
    names: list[tuple[str, ModelEntry]] = [(m.id, m) for m in models]
    for m in models:
        names.extend((alias, m) for alias in m.aliases)
    hit, candidates = fuzzy_match(query, names)
    if hit is not None:
        return hit

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
