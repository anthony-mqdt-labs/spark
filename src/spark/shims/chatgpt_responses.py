"""Chat Completions ⇄ Responses translation for ChatGPT-plan upstreams (spec 0005).

Pure functions, no I/O. The scope is the consumer contract
(SYMBIOTE-LLM-CONTRACT.md): system plus user (or assistant/developer) text
messages, non-streaming. Anything outside it is refused with a 400 that names
the field, so a mistranslation is never silent.

SIWC preview rules applied (developers.openai.com/siwc/…/preview-limitations):
- system-role items are rejected, so system text goes to ``instructions``;
- ``store: false`` and ``stream: true`` are required;
- ``temperature``, ``top_p``, ``max_output_tokens`` and friends are rejected, so
  they are dropped, and the caller logs what was dropped.
"""

from __future__ import annotations

import json
import time

#: Present-and-meaningful → refuse (no faithful translation in this scope).
_REFUSE = (
    "tools",
    "functions",
    "tool_choice",
    "function_call",
    "response_format",
    "audio",
    "modalities",
    "prediction",
)

#: errors-and-recovery page: error.code → HTTP status for the consumer.
_STATUS_BY_CODE = {
    "subscription_sharing_usage_limit_exceeded": 429,
    "subscription_sharing_user_not_eligible": 403,
    "subscription_sharing_unsupported_capability": 400,
    "subscription_sharing_usage_unavailable": 503,
    "subscription_sharing_user_unavailable": 503,
    "subscription_sharing_route_not_supported": 403,
    "subscription_sharing_invalid_user": 401,
    "chatpass_v2_scope_not_authorized": 403,
    "chatpass_v2_invalid_authorization_context": 403,
    "rate_limit_exceeded": 429,
}


class TranslationError(ValueError):
    def __init__(self, param: str, message: str) -> None:
        super().__init__(message)
        self.param = param


def _text(content: object, where: str) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for i, part in enumerate(content):
            if not (isinstance(part, dict) and part.get("type") == "text"):
                raise TranslationError(f"{where}[{i}]", "only text content parts are supported")
            parts.append(str(part.get("text", "")))
        return "".join(parts)
    raise TranslationError(where, "content must be a string or a list of text parts")


def chat_to_responses(body: object) -> tuple[dict, list[str]]:
    """(responses_request, dropped_param_names)."""
    if not isinstance(body, dict):
        raise TranslationError("body", "request body must be a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise TranslationError("model", "model is required")
    if body.get("stream") is True:
        raise TranslationError("stream", "streaming is not supported by this provider yet")
    for key in _REFUSE:
        if body.get(key) not in (None, [], {}):
            raise TranslationError(key, f"'{key}' is not supported by this provider")
    if body.get("n") not in (None, 1):
        raise TranslationError("n", "only n=1 is supported")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise TranslationError("messages", "messages must be a non-empty list")

    instructions: list[str] = []
    items: list[dict] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise TranslationError(f"messages[{i}]", "each message must be an object")
        role = m.get("role")
        text = _text(m.get("content"), f"messages[{i}].content")
        if role == "system":
            instructions.append(text)
        elif role in ("developer", "user", "assistant"):
            items.append({"role": role, "content": text})
        else:
            raise TranslationError(f"messages[{i}].role", f"role {role!r} is not supported")
    if not items:
        raise TranslationError("messages", "at least one user/developer/assistant message needed")

    out: dict = {"model": model, "input": items, "store": False, "stream": True}
    if instructions:
        out["instructions"] = "\n\n".join(instructions)
    consumed = {"model", "messages", "stream", "n", *_REFUSE}
    dropped = sorted(k for k in body if k not in consumed)
    return out, dropped


def _output_text(response: dict) -> str:
    """Text from a final Responses object (fallback when no deltas streamed)."""
    texts = []
    for item in response.get("output") or []:
        if isinstance(item, dict) and item.get("type") == "message":
            for c in item.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "output_text":
                    texts.append(str(c.get("text", "")))
    return "".join(texts)


class ResponseCollector:
    """Folds a Responses SSE stream into one result. Feed decoded lines; the
    result counts only after ``response.completed`` (or ``response.incomplete``)."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.done = False
        self.finish = "stop"
        self.usage: dict = {}
        self.error: dict | None = None
        self._final_text = ""
        self._data: list[str] = []

    @property
    def text(self) -> str:
        return "".join(self.parts) or self._final_text

    def feed_line(self, line: str) -> None:
        line = line.rstrip("\r\n")
        if not line:
            self._dispatch()
        elif line.startswith("data:"):
            self._data.append(line[5:].lstrip())
        # `event:`, `id:`, `retry:` and `:` comments carry nothing we need.

    def close(self) -> None:
        self._dispatch()

    def _dispatch(self) -> None:
        if not self._data:
            return
        raw = "\n".join(self._data)
        self._data = []
        if raw == "[DONE]":
            return
        try:
            ev = json.loads(raw)
        except ValueError:
            return
        if not isinstance(ev, dict):
            return
        kind = ev.get("type")
        if kind == "response.output_text.delta":
            self.parts.append(str(ev.get("delta") or ""))
        elif kind in ("response.completed", "response.incomplete"):
            resp = ev.get("response") if isinstance(ev.get("response"), dict) else {}
            self.usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
            self._final_text = _output_text(resp)
            if kind == "response.incomplete":
                self.finish = "length"
            self.done = True
        elif kind in ("response.failed", "error"):
            if kind == "response.failed":
                resp = ev.get("response") if isinstance(ev.get("response"), dict) else {}
                err = resp.get("error")
            else:
                err = ev.get("error") if isinstance(ev.get("error"), dict) else ev
            self.error = err if isinstance(err, dict) else {"message": str(err)}
            self.done = True


def error_status(err: dict) -> int:
    return _STATUS_BY_CODE.get(str(err.get("code") or ""), 502)


def to_chat_completion(model: str, text: str, usage: dict, finish: str, rid: str) -> dict:
    prompt = int(usage.get("input_tokens") or 0)
    completion = int(usage.get("output_tokens") or 0)
    return {
        "id": f"chatcmpl-{rid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": int(usage.get("total_tokens") or prompt + completion),
        },
    }


def models_payload(ids: list[str]) -> dict:
    return {
        "object": "list",
        "data": [{"id": i, "object": "model", "owned_by": "openai"} for i in ids],
    }
