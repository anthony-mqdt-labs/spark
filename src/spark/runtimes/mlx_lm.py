"""MLX-LM backend — the Apple Silicon native text path (Phase 1 primary).

Two launch shapes: direct (default) and filtered. `mlx_lm.server` answers
`/v1/models` from its own hub-cache scan with no `--model-discovery` switch,
so harness pickers offer embeddings/speech/partials as chat models. With
`[server] filter_models=true`, the supervised child is
`shims/openai_filter_proxy.py` instead: it binds the published port, spawns
the real server on an ephemeral loopback port, passes everything through, and
answers `GET /v1/models` from spark's inventory (registered ∩ present).
"""

from __future__ import annotations

import socket
import sys
from importlib.resources import files
from pathlib import Path

from ..config.schema import ModelEntry
from .base import Backend, override_flags, register
from .base import _fill as _fill_token


def _proxy_shim() -> Path:
    return Path(str(files("spark.shims") / "openai_filter_proxy.py"))


def _free_loopback_port() -> int:
    """An ephemeral port for the proxied server (same bind-then-release race
    the supervisor's own port pick accepts; loopback-only)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@register("mlx_lm")
class MlxLmBackend(Backend):
    # mlx_lm.server starts listening before the model is resident and loads it
    # on the first request — so spark must verify a real generation, not just a
    # health response, before calling it ready.
    lazy_loads_model = True

    def resolve_model_ref(self, entry: ModelEntry) -> str:
        # mlx_lm.server accepts a local path OR a HF repo id for --model.
        return entry.path or entry.hf_repo or entry.id

    def build_launch_cmd(
        self, entry: ModelEntry, host: str, port: int
    ) -> list[str]:
        if not self.rt.server.filter_models:
            return super().build_launch_cmd(entry, host, port)
        ref = self.resolve_model_ref(entry)
        upstream_port = _free_loopback_port()
        ctx: dict[str, object] = {
            "model_path": ref,
            "model_dir": "",
            "model_id": entry.id,
            "host": "127.0.0.1",
            "port": upstream_port,
        }
        inner = [
            self.rt.server.binary,
            *[_fill_token(tok, ctx) for tok in self.rt.server.args_template],
            *override_flags(entry.launch_overrides),
        ]
        return [
            sys.executable, str(_proxy_shim()),
            "--host", host, "--port", str(port),
            "--upstream", f"http://127.0.0.1:{upstream_port}",
            "--served-id", ref,
            "--",
            *inner,
        ]

