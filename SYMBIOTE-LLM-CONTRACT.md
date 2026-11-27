# Symbiote → spark: LLM request contract

How Symbiote (`~/Repos/vibes/crashcourse/symbiote`) calls spark. Use this to scope
the OpenAI provider integration (OAuth + Responses API).

Source of truth: `crates/core/src/llm.rs` in the Symbiote repo.

**Status (2026-11-26):** the Symbiote side is done and needs nothing from spark
first:
- 429 now switches Symbiote to offline mode.
- Course compilation gets 8000 tokens and 180 s.

The remaining work is all in spark, and the SIWC specifics section below lists
it.

## Answer: Chat Completions

Symbiote sends **Chat Completions** (`/v1/chat/completions`), not Responses.
The OpenAI provider therefore needs a Chat Completions → Responses translation
layer. Symbiote uses only a small part of the API, so the layer only has to
cover the cases below.

## Endpoints used

### `GET /v1/models`: health check

- **Timeout:** 500 ms. Called at boot and every 60 s while Symbiote is offline,
  to detect recovery.
- **Reads:** `data[0].id`, unless `SYMBIOTE_LLM_MODEL` is set.
- **Requirement:** answer quickly. Ideally spark answers it locally and does not
  forward it to OpenAI.

### `POST /v1/chat/completions`

- **Timeout and output cap** depend on the request:

  | Request | `max_tokens` | Timeout |
  | :--- | :--- | :--- |
  | Coaching replies (hints, answers, reviews, `/mock`) | 700 | 15 s |
  | Course compilation (a new track or slice as JSON, including code files) | 8000 | 180 s |

- **Streaming:** off.

The request always has exactly this shape:

```json
{
  "model": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "temperature": 0.3,
  "max_tokens": 700
}
```

`max_tokens` is 8000 for course compilation.

It never includes `stream`, `tools`, `response_format`, images, or multi-turn
history. Every call is one system message plus one user message, both plain
strings.

### Response fields read

| Field | Use |
| :--- | :--- |
| `choices[0].message.content` | The reply text (required) |
| `usage.completion_tokens` | Optional; logged only |

## Auth

Symbiote sends `Authorization: Bearer <SYMBIOTE_LLM_KEY>` only when that
variable is set. spark can ignore this header, because it injects the real
credential (OAuth token) on the upstream request.

## Error handling on Symbiote's side

| Outcome | Symbiote behaviour |
| :--- | :--- |
| Timeout (500 ms health; 15 s chat; 180 s compile) | Switches to Tier 0 (offline drill mode) |
| 2 consecutive 5xx, **429** or transport errors | Switches to Tier 0 |
| Any other 4xx | Shows the error to the user; does not switch tiers |

Pass a usage-limit 429 (for example `subscription_sharing_usage_limit_exceeded`)
through unchanged. Symbiote treats it as "go offline and keep checking", so
don't rewrite it to a 503.

While offline, Symbiote retries `GET /v1/models` every 60 s and switches back
automatically when it succeeds.

## Translation needed (Chat Completions → Responses)

| Chat Completions | Responses |
| :--- | :--- |
| system message content | `instructions` |
| user message content | `input` |
| `max_tokens` | `max_output_tokens` |
| `temperature` | pass through; drop it for reasoning models, which reject it |
| `model` | pass through, or map |

Response mapping back to Chat Completions:

| Responses | Chat Completions |
| :--- | :--- |
| text of `output[]` items with `type: "message"`, content `type: "output_text"` (concatenated) | `choices[0].message.content` |
| `usage.output_tokens` | `usage.completion_tokens` |

## Gotchas

1. **The 15 s chat timeout** (coaching replies). A slow reasoning model will often exceed it and
   push Symbiote offline. Prefer a fast model, or ask Symbiote to raise the
   timeout.
2. **The 500 ms health timeout.** If `/v1/models` is forwarded to OpenAI, it may
   miss that window and keep Symbiote in Tier 0.

## SIWC (Sign in with ChatGPT) specifics

This section assumes the OpenAI provider uses SIWC ChatGPT-plan usage against
`https://api.openai.com/v1/responses`. Its preview restrictions change the
generic translation above:

| Symbiote sends | SIWC preview | spark must |
| :--- | :--- | :--- |
| `role: "system"` message | System-role messages are rejected | Put it in `instructions`. If `instructions` is refused (unverified; test it in the first manual call), send it as the first `input` item with role `developer`, or `user` if that is refused too. **Never drop it**: it carries Symbiote's behaviour rules. |
| `temperature: 0.3` | Not accepted | Always drop it. |
| `max_tokens` | `max_output_tokens` is not accepted | Drop it. Symbiote enforces reply length itself: Focus replies are clamped to 15 lines on receipt. |
| Non-streaming request | `stream: true` is required | Request with `stream: true` and `store: false`. Read the SSE stream until `response.completed`, concatenate the `output_text` deltas, and return one ordinary non-streamed Chat Completions response. |
| `GET /v1/models` | Catalog; keep entries with `visibility == "list"` | Cache the list and answer locally within 500 ms. |

Errors:

| SIWC error | spark returns to Symbiote |
| :--- | :--- |
| `subscription_sharing_usage_limit_exceeded` (429) | 429 unchanged, so Symbiote goes offline and keeps checking. |
| `subscription_sharing_user_not_eligible` (403) | 403 with a message saying the plan is not eligible. Symbiote shows it and stays online. |
| `subscription_sharing_unsupported_capability` (400) | 400 naming the field spark failed to drop. This is a spark translation bug, so log it. |
| Expired or revoked refresh token | 401 with a re-login remediation. |

Timing: the streamed reply must finish within Symbiote's deadline. That is
15 s for coaching replies and 180 s for course compilation. Pick a model that
answers a ~700-token reply in well under 15 s.

What the first manual check should confirm, using Symbiote's real request
shape:
1. The granted scopes include `chatgpt.tokens.use.direct`.
2. Whether `instructions` is accepted. If not, whether a `developer` input item
   is.
3. The time to the full answer for a 700-token reply, measured against 15 s.

## Raw model or Codex-style agent? Raw model

Symbiote needs a **plain model endpoint**. Running `codex exec` as a child
process does not fit, for these reasons:

1. **Symbiote brings its own system prompt.** Every request carries
   Symbiote's prompt: personas, the 15-line reply limit, the plain coworker
   tone, and "give a nudge, don't solve the slice". Codex adds its own agent
   system prompt on top, which would override or dilute those rules.
2. **Some replies are machine-parsed.** The JIT course compiler and `/mock`
   grading require a reply that is only a JSON object. An agent's narration
   and tool-use output would break that parsing.
3. **An agent can act, and Symbiote must be the only actor.**
   - Symbiote's daemon is the single writer of `.symbiote/` and the only thing
     that runs tests.
   - The product's point is that the learner writes the code. An agent with
     shell and file access in the learner's project could edit the slice it
     is meant to coach, and could do much worse.
   - Symbiote has deliberately deferred sandboxing for LLM-generated code
     (see its notes on the JIT verify gate). Adding an agent with tool access
     would widen exactly that risk.
4. **Latency.** An agent loop takes many seconds per turn. Symbiote expects a
   single answer within 15 s.
5. **Symbiote already handles the tools.** It runs probes, captures failures
   and gathers file context itself, then hands the model a finished prompt.
   It only needs text back.

So the OpenAI provider should forward one request to one model and return one
answer, with no agent loop and no tools. The `codex exec` path may suit other
consumers, but it should not be the endpoint Symbiote is pointed at.

## Why not have Symbiote call `/v1/responses` directly

Switching Symbiote would take about 20 lines. Keeping Chat Completions and
translating in spark is preferred, because every local runtime spark
supervises (`mlx_lm.server`, `llama-server`, `ollama`, and the rest) speaks
Chat Completions. That way Symbiote keeps working with all of them.
