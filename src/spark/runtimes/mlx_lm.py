"""MLX-LM backend — the Apple Silicon native text path (Phase 1 primary).

Behaviorally identical to the generic server backend today; it exists as the
seam for MLX-specific concerns (e.g. resolving a local snapshot dir vs. an HF repo
id, adapter paths, MLX env tuning) without leaking those into config."""

from __future__ import annotations

from ..config.schema import ModelEntry
from .base import Backend, register


@register("mlx_lm")
class MlxLmBackend(Backend):
    def resolve_model_ref(self, entry: ModelEntry) -> str:
        # mlx_lm.server accepts a local path OR a HF repo id for --model.
        return entry.path or entry.hf_repo or entry.id
