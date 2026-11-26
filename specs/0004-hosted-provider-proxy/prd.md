# 0004 — Hosted provider proxy (OpenRouter, DeepSeek)

> **STATE: implemented, merged to `main`** (2026-11-26). Only the P6 live smoke
> test with real keys remains. Origin: 2026-11-26
> operator request. An app under development must test against spark's
> OpenAI-compatible surface, with spark fronting hosted providers as well as
> local runtimes.

## 1. Problem

spark serves only local runtimes. The `relay` backend is attach-only: it
health-checks an external URL and hands that URL to the consumer, who then
talks to the upstream **directly** (`RelayBackend.openai_base_url`). That is
unusable for a hosted provider:

- the consumer would need the provider API key itself, which puts the key in
  app config, env files, and shell history;
- relay has no auth injection, no TLS upstream policy, no request logging;
- `RELAY-USAGE.md` describes a proxy on `:8080` that the code does not
  implement (the docs have drifted from the code; see §7).

## 2. Goal

`spark run openrouter` / `spark run deepseek` starts a **supervised loopback
proxy**. The consumer points an OpenAI SDK at `http://127.0.0.1:<port>/v1`
with no key. spark reads the key from the Keychain, injects
`Authorization: Bearer …` on the way out, and streams the responses back.

Non-goals: CI/deployment, multi-user auth, packaging (spark is local-only, one
consumer), request rewriting and model aliasing (the consumer sends the
provider's own model ids), and caching or cost accounting.

## 3. Vetting findings (pre-implementation audit)

| # | Finding | Disposition |
|---|---|---|
| V1 | Secrets: Keychain via ctypes, no argv, registered with the redactor, injected into child env only (`ServerSpec.secret_env`). | **Reuse as is.** The key never leaves this path. |
| V2 | The supervisor already does spawn, health, bounded restart, the instance manifest, the catalog, and the reserved-fd port handoff. | **Reuse.** The proxy is just another supervised child. |
| V3 | `openai_filter_proxy.py` holds a stdlib streaming passthrough plus fd adoption. | **Reuse** `_adopted_server` and `_HOP_HEADERS` (it gains a handler parameter). |
| V4 | TLS: the venv Python (3.12.13, OpenSSL 3.5.5) verifies against `/etc/ssl/cert.pem`. Live check returned 200 from `openrouter.ai` and 401 from `api.deepseek.com`, unauthenticated. | **No `certifi`, no new dependency.** |
| V5 | `relay` hands out the upstream URL, and its docs claim a proxy that does not exist. | `RELAY-USAGE.md` was rewritten and verified end to end (2026-11-26 docs pass). Its old model TOML with `[server]` was invalid. |
| V6 | `.claude/settings.local.json` allowlists `Bash(uv run *)` and `Bash(uv run python -c ' *)`, which amounts to arbitrary code execution without a prompt. No `dontAsk`. | Flagged to the operator. Not changed. |
| V7 | `[build-system] requires = ["hatchling"]` is unpinned, unlike the runtime deps. It affects only an editable reinstall. | Flagged. Not changed (local-only distribution). |

## 4. Design

**Data (config is data):** one runtime TOML per provider,
`config/runtimes/{openrouter,deepseek}.toml`. Three new `ServerSpec` fields:

- `provider_base_url`: the upstream OpenAI-compatible base. It must be
  `https://`; plain `http` is accepted only for loopback (tests).
- `provider_auth_check_path`: the path hit with the key before launch
  (`/key` for OpenRouter, `/models` for DeepSeek).
- `provider_timeout_s`: the upstream socket timeout, sized for long reasoning
  streams.

The key arrives as `[server.secret_env] SPARK_PROVIDER_API_KEY = "<vault name>"`.
Adding another OpenAI-compatible provider is TOML only: `get_backend` picks
`ProviderBackend` for any runtime that declares `provider_base_url`.

`ModelEntry.remote_model` (optional) is published as `api_model_id`, the
default model a consumer reading `catalog.json` should send.

**Launch (`ProviderBackend`):**

1. `prepare()` is the fail-fast gate. A missing key fails with
   `SEC_NOT_FOUND` and points to `spark secret set …`. A key rejected
   (401/403) on the auth-check path fails with `SEC_REJECTED`, which is never
   retried, because a bad key will not get better. An unreachable upstream
   fails with `NET_UNREACHABLE`.
2. The child is `shims/provider_proxy.py` under spark's own interpreter. It
   adopts spark's reserved listen fd, so there is no port race.
3. Health is the shim's local `GET /health`. The supervisor never polls the
   paid upstream every 5 s.

**Proxy (`shims/provider_proxy.py`, stdlib only):**

- Forwards only `/v1/<rest>` to `<base>/<rest>`, and only `GET` and `POST`.
  Rejects `..` segments.
- Client `Authorization`, `Cookie`, `Host`, `Accept-Encoding`, and hop headers
  are dropped. spark's bearer is set, and `Accept-Encoding: identity` keeps SSE
  unbuffered.
- **Browser-abuse guards** (any page can POST to loopback):
  - `Host` must be loopback, which defeats DNS rebinding.
  - `POST` must be `application/json`, which forces a CORS preflight.
  - `OPTIONS` gets 405, so a preflight never passes.
- Streams with `read1()`, never buffers (time to first token is preserved).
  Body cap is 64 MiB, and chunked request bodies get 411.
- Upstream failure returns a 502 with an OpenAI-shaped error body. The key is
  never echoed.
- JSONL telemetry (`logs/<runtime>/…`): lifecycle events plus one
  `request_forwarded` per call with method, path, status, model, stream flag,
  byte counts, TTFB, and total ms. Correlated through the parent's
  `SPARK_SESSION_ID`. Values pass through the redactor, and the key is
  registered with it.

## 5. Roadmap (executed in order)

| Phase | Deliverable | Verifier |
|---|---|---|
| P0 | Audit (§3) | this doc |
| P1 | Schema fields, URL validation, `remote_model`, error codes | `tests/test_provider_proxy.py` schema cases |
| P2 | `ProviderBackend`: launch cmd, prepare gate, `get_backend` fallback | backend unit tests (fake upstream, `FakeStore`) |
| P3 | `provider_proxy.py` shim | end-to-end against a local fake upstream: auth injection, header stripping, SSE streaming, guards, 502 |
| P4 | Runtime TOMLs, supervisor integration (`spark run`) | supervisor e2e: spawn, health, request through the proxy, clean stop |
| P5 | README section, operator runbook | manual: `spark secret set` → `spark run` → `curl` |
| P6 | Live smoke once the operator supplies keys | `curl …/v1/chat/completions` per provider |

## 6. Operator runbook

```bash
spark secret set openrouter_api_key       # hidden prompt; never argv
spark secret set deepseek_api_key
# one registry entry per provider (remote_model = default model for consumers)
cat > ~/.local/share/spark/models/openrouter.toml <<'EOF'   # `spark config path` → models; port 8100
id = "openrouter"
backend = "openrouter"
research_status = "manual"
remote_model = "deepseek/deepseek-chat"
EOF
spark run openrouter                       # prints http://127.0.0.1:<port>/v1
```

## 7. Deferred (say the word)

- Local client token on the proxy. Today any same-user process can spend the
  key. That is acceptable for a single-operator host, and the add would be one
  header check.
- Model aliasing (`model = "fast"` → the provider id).
- Upstream connection pooling. Today each request opens one TLS connection.
