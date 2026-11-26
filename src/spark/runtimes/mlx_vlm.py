"""MLX-VLM backend — Apple Silicon native vision-language models.

Shares the mlx-flavoured resolution and the lazy-loading readiness contract with
:mod:`spark.runtimes.mlx_lm`; only the console script differs (in config).
"""

from __future__ import annotations

from .base import register
from .mlx_lm import MlxLmBackend


@register("mlx_vlm")
class MlxVlmBackend(MlxLmBackend):
    """Only the console script differs (in config); resolution and readiness
    are inherited unchanged from mlx_lm."""
