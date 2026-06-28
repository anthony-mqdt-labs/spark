"""oMLX backend — production MLX server (`omlx serve <model>`)."""

from __future__ import annotations

from ..config.schema import ModelEntry
from .base import Backend, register


@register("omlx")
class OmlxBackend(Backend):
    def resolve_model_ref(self, entry: ModelEntry) -> str:
        # `omlx serve` takes a positional model (local path or HF repo id).
        return entry.path or entry.hf_repo or entry.id
