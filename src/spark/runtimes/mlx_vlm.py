"""MLX-VLM backend — Apple Silicon native vision-language models.

Shares the mlx-flavoured resolution and the lazy-loading readiness contract with
:mod:`spark.runtimes.mlx_lm`; only the console script differs (in config).
"""

from __future__ import annotations

from ..config.schema import ModelEntry
from .base import register
from .mlx_lm import MlxLmBackend


@register("mlx_vlm")
class MlxVlmBackend(MlxLmBackend):
    def resolve_model_ref(self, entry: ModelEntry) -> str:
        return entry.path or entry.hf_repo or entry.id
