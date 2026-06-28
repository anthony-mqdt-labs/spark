"""MLX-VLM backend — Apple Silicon native vision-language models."""

from __future__ import annotations

from ..config.schema import ModelEntry
from .base import Backend, register


@register("mlx_vlm")
class MlxVlmBackend(Backend):
    def resolve_model_ref(self, entry: ModelEntry) -> str:
        return entry.path or entry.hf_repo or entry.id
