"""llama.cpp backend — GGUF via Metal.

Resolves the model to either a local ``.gguf`` file (``--model``) or a Hugging
Face repo (``-hf user/repo``), which llama-server can fetch directly. The static
Metal flags (-ngl/-fa/--jinja) come from the runtime TOML; only the *model*
argument is computed here."""

from __future__ import annotations

from pathlib import Path

from ..config.schema import ModelEntry
from ..errors import ModelNotFoundError
from .base import Backend, _fill, override_flags, register


@register("llama_cpp")
class LlamaCppBackend(Backend):
    def _model_spec(self, entry: ModelEntry) -> tuple[str, str]:
        """Return (flag, value): either ('--model', /path/to.gguf) or ('-hf', repo)."""
        if entry.path:
            p = Path(entry.path).expanduser()
            if p.is_file() and p.suffix == ".gguf":
                return ("--model", str(p))
            if p.is_dir():
                gg = sorted(p.glob("*.gguf"))
                if gg:
                    # Prefer the first shard of a split model if present.
                    first = next((g for g in gg if "00001-of" in g.name), gg[0])
                    return ("--model", str(first))
        if entry.hf_repo:
            return ("-hf", entry.hf_repo)
        raise ModelNotFoundError(
            f"No GGUF file or HF repo for model '{entry.id}'.",
            remediation=[
                "Re-download a GGUF build: spark download <user>/<repo>-GGUF",
                "Or point the registry entry's path at a local .gguf file.",
            ],
            context={"model": entry.id},
        )

    def build_launch_cmd(self, entry: ModelEntry, host: str, port: int) -> list[str]:
        flag, value = self._model_spec(entry)
        ctx: dict[str, object] = {
            "model_path": value, "model_id": entry.id, "host": host, "port": port,
        }
        out = [self.rt.server.binary]
        skip = False
        for tok in self.rt.server.args_template:
            if skip:
                skip = False
                continue
            if tok == "--model":
                out += [flag, value]   # swap in the resolved model flag/value
                skip = True            # drop the original "{model_path}" token
                continue
            out.append(_fill(tok, ctx))
        return out + override_flags(entry.launch_overrides)
