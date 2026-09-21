"""OpenAI-compatible serving shim for PrismML Hadamard-packed checkpoints.

Runs as a *child process* under the mlx-lm tool interpreter
(``~/.local/share/uv/tools/mlx-lm/bin/python``) — never imported by spark
itself (D1). Stdlib HTTP only; the ML imports (mlx, mlx_lm, jinja2) all live in
that interpreter already.

Serving model: load the pack eagerly at startup through the vendor's bundled
loader (``<pack>/runtime/artifact.py``), render prompts with the snapshot's own
``chat_template.jinja``, generate via ``mlx_lm.generate.stream_generate``, and
answer the three endpoints spark needs:

* ``GET /v1/models`` — health + model roster (supervisor health gate).
* ``GET /health`` — plain liveness.
* ``POST /v1/chat/completions`` — non-streaming OpenAI shape (supervisor
  warmup posts exactly this). ``stream=true`` is refused honestly: there is no
  SSE here.

MLX model calls are not thread-safe, so generations serialize on a lock while
the health endpoints stay concurrent.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Serve a PrismML Hadamard pack.")
    p.add_argument("--pack", required=True, help="Snapshot dir (config.json + weights).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8098)
    p.add_argument("--model-id", default="", help="API model id reported to clients.")
    p.add_argument("--max-tokens-default", type=int, default=256)
    return p.parse_args(argv)


_STATE: dict = {}


def _load(pack: Path, model_id: str) -> None:
    """Import the vendor loader from the (read-only) snapshot and load."""
    import mlx.core as mx  # noqa: F401  (warms the backend early, loud failures first)
    from mlx_lm.sample_utils import make_sampler  # noqa: F401
    from mlx_lm.utils import load_tokenizer

    runtime_dir = pack / "runtime"
    sys.path.insert(0, str(runtime_dir))

    import json

    cfg = json.loads((pack / "config.json").read_text(encoding="utf-8"))
    schema = cfg.get("schema_version")
    if schema == 1:
        from artifact import load_model

        t0 = time.monotonic()
        print(f"[prism] loading v1 pack {pack} …", flush=True)
        model, config = load_model(pack)
        print(f"[prism] weights resident ({time.monotonic() - t0:.1f}s, "
              f"{len(config.get('modules', []))} packed modules)", flush=True)
    elif schema == 2:
        model = _load_v2(pack, cfg)
    else:
        raise ValueError(f"unsupported pack schema_version {schema!r}")

    print("[prism] loading tokenizer …", flush=True)
    tokenizer = load_tokenizer(str(pack))

    template_text = (pack / "chat_template.jinja").read_text(encoding="utf-8")
    import jinja2

    template = jinja2.Environment().from_string(template_text)
    try:
        bos = tokenizer.bos_token
    except AttributeError:
        bos = None
    try:
        eos = tokenizer.eos_token
    except AttributeError:
        eos = None

    _STATE.update(
        model=model, tokenizer=tokenizer, template=template,
        bos=bos, eos=eos, model_id=model_id or pack.name,
        lock=threading.Lock(),
    )
    print("[prism] ready", flush=True)


_V2_PREFIX = "language_model."

# The vendor records carry dtype as a string ("float16"); mlx>=0.32 requires a
# Dtype object in astype(), so the bundled v1 loader's pass-through breaks on
# the embedding path. Map it here instead of touching the read-only snapshot.
def _v2_dtype(name: str):
    import mlx.core as mx

    table = {"float16": mx.float16, "float32": mx.float32, "bfloat16": mx.bfloat16}
    try:
        return table[str(name).lower()]
    except KeyError:
        raise ValueError(f"unsupported pack dtype {name!r}")


def _load_v2(pack: Path, cfg: dict):
    """Load a schema-2 pack (multimodal namespace, text tower only).

    v2 moved tensors under ``language_model.*`` next to a ``vision_tower.*``
    the text-only shim cannot serve (vision input is refused, not silently
    dropped — the model simply has no vision path). Records already use the
    bare ``TextModel`` namespace, so each record maps to exactly one
    ``language_model.<record>.<weight|scales|biases|signs>`` tensor set.
    Structural checks (shapes, dtypes, signs) are the vendor's own
    ``validate_record``; anything unexpected raises before serving.
    """
    import inspect

    import mlx.core as mx
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
    from runtime import Packed  # noqa: F401  (installed onto the model below)
    from artifact import validate_record

    text_cfg = cfg.get("text_config") or {}
    fields = set(inspect.signature(TextModelArgs).parameters)
    dropped = sorted(k for k in text_cfg if k not in fields)
    model = TextModel(TextModelArgs.from_dict(
        {k: v for k, v in text_cfg.items() if k in fields}))
    print(f"[prism] TextModel built ({text_cfg.get('num_hidden_layers')} layers; "
          f"ignored text_config keys: {dropped or 'none'})", flush=True)

    t0 = time.monotonic()
    weights = mx.load(str(pack / "model.safetensors"))
    skipped = sorted({n.split(".")[0] for n in weights if not n.startswith(_V2_PREFIX)})
    print(f"[prism] weight map: {len(weights)} tensors "
          f"({time.monotonic() - t0:.1f}s); skipped namespaces: {skipped}", flush=True)

    records = cfg.get("modules") or []
    for record in records:
        path = record["path"]
        stem = _V2_PREFIX + path
        try:
            arrays = [weights[stem + "." + s] for s in ("weight", "scales", "biases")]
        except KeyError:
            raise ValueError(f"pack record {path!r} has no tensors at {stem}.*")
        block = record.get("block") or 0
        signs = weights.get(stem + ".signs")
        if block and signs is None:
            raise ValueError(f"pack record {path!r}: block transform but no signs")
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        original = getattr(parent, parts[-1])
        validate_record(original, record, arrays, signs)
        setattr(parent, parts[-1],
                Packed(arrays, block, signs, record.get("embedding", False),
                       _v2_dtype(record.get("dtype", "float16"))))
    print(f"[prism] {len(records)} packed modules installed", flush=True)

    items = [(n[len(_V2_PREFIX):], w) for n, w in weights.items()
             if n.startswith(_V2_PREFIX)]
    model.load_weights(items, strict=True)
    model.eval()
    mx.eval(model.parameters())
    print(f"[prism] weights resident ({time.monotonic() - t0:.1f}s)", flush=True)
    return model


def _render_prompt(messages: list) -> str:
    template = _STATE["template"]
    return template.render(
        messages=messages,
        add_generation_prompt=True,
        add_bos_token=True,
        bos_token=_STATE["bos"],
        eos_token=_STATE["eos"],
    )


def _chat(messages: list, max_tokens: int, temperature: float, top_p: float) -> dict:
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt = _render_prompt(messages)
    sampler = make_sampler(temp=temperature or 0.0, top_p=top_p or 0.0)
    started = time.monotonic()
    chunks: list[str] = []
    finish = "length"
    with _STATE["lock"]:
        for resp in stream_generate(
            _STATE["model"], _STATE["tokenizer"], prompt,
            max_tokens=max(1, max_tokens), sampler=sampler,
        ):
            chunks.append(resp.text or "")
            if resp.finish_reason:
                finish = resp.finish_reason
    text = "".join(chunks)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _STATE["model_id"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop" if finish == "stop" else "length",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                  "total_tokens": 0,
                  "wall_s": round(time.monotonic() - started, 2)},
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "prism-shim/1"

    def log_message(self, fmt, *args):  # route access log to stderr (spark log file)
        sys.stderr.write(f"[prism] {self.address_string()} {fmt % args}\n")

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/health", "/v1/models"):
            self._send(200, {
                "object": "list" if self.path == "/v1/models" else "ok",
                "data": [{"id": _STATE["model_id"], "object": "model"}]
                if self.path == "/v1/models" else {"status": "ok"},
            })
        else:
            self._send(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": f"unknown path {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError):
            self._send(400, {"error": "invalid JSON body"})
            return
        if body.get("stream"):
            self._send(400, {"error": "streaming is not supported by this shim"})
            return
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send(400, {"error": "`messages` must be a non-empty list"})
            return
        try:
            result = _chat(
                messages,
                max_tokens=int(body.get("max_tokens") or 256),
                temperature=float(body.get("temperature") or 0.0),
                top_p=float(body.get("top_p") or 0.0),
            )
        except Exception as exc:  # noqa: BLE001 — a failed generation is a 500 with the reason
            self._send(500, {"error": f"generation failed: {type(exc).__name__}: {exc}"})
            return
        self._send(200, result)


def main(argv=None) -> int:
    args = _parse_args(argv)
    pack = Path(args.pack).expanduser()
    for needed in ("config.json", "model.safetensors", "tokenizer.json",
                   "chat_template.jinja"):
        if not (pack / needed).is_file():
            print(f"[prism] FATAL: {pack} is missing {needed}", file=sys.stderr)
            return 2
    if not (pack / "runtime" / "artifact.py").is_file():
        print(f"[prism] FATAL: {pack}/runtime/artifact.py missing "
              f"(not a Hadamard pack?)", file=sys.stderr)
        return 2
    try:
        _load(pack, args.model_id)
    except Exception as exc:  # noqa: BLE001 — startup failure must be loud (log tail)
        print(f"[prism] FATAL: load failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[prism] listening on http://{args.host}:{args.port} "
          f"(model {_STATE['model_id']})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
