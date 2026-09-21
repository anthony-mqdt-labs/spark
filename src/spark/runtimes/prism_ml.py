"""PrismML backend — Hadamard-packed checkpoints via the vendor loader + shim.

Stock mlx_lm answers ``Model type prism_hadamard_qwen35 not supported`` for
these packs, so this backend serves them through
:mod:`spark.runtimes.prism_shim` (stdlib HTTP, OpenAI-compatible) executed by
the mlx-lm tool interpreter — the env that already owns mlx + mlx_lm (D1: spark
never imports ML libraries; it drives them as subprocesses).

The shim path is package data resolved here (not expressible in TOML); the
interpreter, ports, and timeouts stay data in ``config/runtimes/prism_ml.toml``.
Weights resolve exactly like oMLX: ``entry.path`` first, else the hub-cache
snapshot for ``entry.hf_repo`` — never fetched, presence is a launch gate.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from ..config.schema import ModelEntry
from ..errors import RuntimeBackendError
from ..hf_cache import snapshot_dir as _hf_cache_snapshot
from .base import Backend, register


def shim_path() -> Path:
    """Filesystem path of the serving shim (editable checkout and wheel both).

    Lives in ``spark.shims`` — a shim-only directory — because a script's own
    directory lands on ``sys.path`` at launch, and ``runtimes/`` contains
    ``mlx_lm.py``, which would shadow the real ``mlx_lm`` package.
    """
    return Path(str(files("spark.shims") / "prism_shim.py"))


@register("prism_ml")
class PrismMlBackend(Backend):
    # Eager loader: /v1/models only answers after the pack is resident, so the
    # health gate alone is an honest readiness signal (like llama_cpp).
    lazy_loads_model = False

    def pack_dir(self, entry: ModelEntry) -> Path:
        """The snapshot dir the shim will load (config.json + vendor runtime)."""
        if entry.path:
            local = Path(entry.path).expanduser()
            if (local / "config.json").is_file():
                return local
        if entry.hf_repo:
            snap = _hf_cache_snapshot(entry.hf_repo)
            if snap is not None and (snap / "config.json").is_file():
                return snap
        raise RuntimeBackendError(
            f"prism_ml needs the pack for '{entry.id}' on disk, but none was found.",
            code="RT_NO_WEIGHTS",
            remediation=[
                f"Download it first: spark download {entry.hf_repo or entry.id}",
                "The shim serves local packs only (never fetches).",
            ],
            context={"model": entry.id},
        )

    def prepare(self, entry: ModelEntry, *, secrets=None) -> None:
        pack = self.pack_dir(entry)  # fail fast with remediation, before spawn
        missing = [
            n for n in ("model.safetensors", "tokenizer.json", "chat_template.jinja")
            if not (pack / n).is_file()
        ]
        if not (pack / "runtime" / "artifact.py").is_file():
            missing.append("runtime/artifact.py (vendor loader)")
        if missing:
            raise RuntimeBackendError(
                f"Pack for '{entry.id}' is incomplete: missing {', '.join(missing)}.",
                code="RT_BAD_PACK",
                remediation=[
                    f"Re-fetch the pack: spark download {entry.hf_repo or entry.id}",
                    "A Hadamard pack needs config.json + model.safetensors + "
                    "tokenizer.json + chat_template.jinja + runtime/artifact.py.",
                ],
                context={"model": entry.id, "pack": str(pack)},
            )

    def resolve_model_ref(self, entry: ModelEntry) -> str:
        # The shim takes a directory, never a repo id: resolve hub entries to
        # their snapshot (the same rule availability uses).
        return str(self.pack_dir(entry))

    def api_model_id(self, entry: ModelEntry) -> str:
        # The shim accepts any request `model` value; publish the registry id
        # rather than the snapshot path the launcher needs.
        return entry.id

    def build_launch_cmd(self, entry: ModelEntry, host: str, port: int) -> list[str]:
        return [
            self.rt.server.binary,
            str(shim_path()),
            "--pack", self.resolve_model_ref(entry),
            "--host", host,
            "--port", str(port),
            "--model-id", entry.id,
            *self._override_flags(entry),
        ]

    def _override_flags(self, entry: ModelEntry) -> list[str]:
        from .base import override_flags

        return override_flags(entry.launch_overrides)
