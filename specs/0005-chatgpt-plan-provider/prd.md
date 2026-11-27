# 0005 — ChatGPT plan provider (Sign in with ChatGPT → Responses)

> **STATE: implemented.** Origin: 2026-11-26 operator request. The consumer
> contract is `SYMBIOTE-LLM-CONTRACT.md`. Research:
> `openai-oauth-research.md` (session scratchpad), plus verbatim checks of
> developers.openai.com/siwc/token-sharing-open-source/{sign-in,
> profiles-and-sessions, models-and-inference, token-reference,
> preview-limitations, errors-and-recovery}.

## 1. Goal

Symbiote, and other spark consumers, call spark's loopback OpenAI surface. The
requests are billed to the operator's **ChatGPT plan**, not to API per-token
rates.

## 2. Route chosen, and the one refused

- **Chosen: OpenAI's sanctioned "Sign in with ChatGPT" (SIWC) token sharing.**
  The Cookbook says ChatGPT plan usage is "available to open-source projects,
  personal projects that run locally, and selected private apps". spark is a
  personal, local, single-user tool. Inference goes to the public
  `POST https://api.openai.com/v1/responses`.
- **Refused: reusing Codex CLI's client id against `chatgpt.com/backend-api`.**
  OpenAI's docs say "do not point it at ChatGPT's `backend-api` endpoints". The
  route is undocumented, its tokens carry connector scopes, and refreshing
  them would race Codex CLI's own single-use refresh tokens.

## 3. Consumer contract (Symbiote): Chat Completions in, Responses upstream

| Chat Completions in | Responses out |
|---|---|
| `system` message(s) | `instructions`, joined with blank lines |
| `developer` / `user` / `assistant` messages | `input` items with the same role |
| `temperature`, `max_tokens` and other sampling params | dropped. SIWC preview rejects them; Symbiote clamps reply length itself |
| non-streaming | `stream: true` and `store: false` upstream, collected until `response.completed` |
| `stream: true`, `tools`, `functions`, `n > 1`, non-text content | **400** naming the field: outside the contract, refused rather than mistranslated |

Back to Chat Completions: the `output_text` deltas are concatenated into
`choices[0].message.content`. `usage.output_tokens` becomes
`completion_tokens`. `response.incomplete` sets `finish_reason: "length"`.

- `GET /v1/models` is answered locally from a cached catalog
  (`visibility == "list"`), refreshed in the background. The registry entry's
  `remote_model` is listed **first**, because Symbiote reads `data[0].id`. This
  keeps the reply inside Symbiote's 500 ms budget.
- **Errors.**
  - A pre-stream upstream status passes through unchanged. A 429
    `subscription_sharing_usage_limit_exceeded` stays a 429, so Symbiote goes
    offline and keeps re-checking.
  - An in-stream `response.failed` maps by `error.code` to OpenAI's documented
    status (429, 403, 400, 503, or 401).
  - An upstream 401 gets one forced refresh and one retry.
  - A dead session returns a 401 that names `spark login <runtime>`.

## 4. OAuth (verbatim from the SIWC docs)

- **Register / sign in:**
  - Authorize at `https://auth.openai.com/api/accounts/authorize` with
    `client_id=dynamic_agent_client` the first time. The callback returns an
    issued `client_id` (`oaiapp_…`); save that, never `dynamic_agent_client`.
  - Every attempt sends `agent_name_hint` (first registration only),
    `ext_agent_host_id` (stable per host; must be `urn:uuid:<UUIDv4>`, `urn:ietf:params:oauth:jwk-thumbprint:…`, or `did:key:…`, so spark uses `urn:uuid:`; a bare random string is rejected with `invalid_authorize_request`), PKCE S256, `state`, `nonce`,
    `resource=https://api.openai.com/v1`, and the scopes
    `openid profile email offline_access resource.invoke chatgpt.tokens.use.direct`.
  - The callback must be `http://127.0.0.1:<port>/auth/callback`: port 1455 at
    registration, and only the port may vary later.
- **Exchange / refresh:** form-encoded POST to
  `https://auth.openai.com/api/accounts/oauth/token` with the `resource` field.
  The access token lasts 1 h. The refresh token lasts 30 d and rotates on every
  refresh. Honour `earliest_refresh_at`.
- **ID token checks:** `iss`, `aud`, `exp` and `nonce` are checked. The
  signature is not verified: OIDC Core §3.1.3.7(6) allows TLS server validation
  in place of the signature check when the token comes straight from the token
  endpoint. Verifying it would need an RSA dependency.
- **Eligibility:** the granted scopes must include `chatgpt.tokens.use.direct`,
  else the error is `SEC_NOT_ELIGIBLE`.
- **Logout:** revoke at the discovery document's `revocation_endpoint`, then
  clear the tokens. The issued client id and host id are kept, as the docs
  direct.

## 5. Secrets: the refresh token stays in the supervisor (token broker)

SIWC refresh tokens **rotate on every use, and reuse kills the session**
(`refresh_token_reused`). Env injection cannot carry them: a copy goes stale
after the first refresh, and a restart would replay it. The first build let the
proxy child read and write the Keychain itself. That put a 30-day credential in
the process that parses untrusted loopback HTTP, so it was replaced (operator
decision, 2026-11-26) with a **parent-side token broker**:

- `Supervisor` asks the backend for `token_broker(secrets)` after `prepare()`.
  The `TokenBroker` (`spark.secrets.broker`) owns the session. It caches the
  access token and refreshes through `ensure_access_token` under the session
  `flock`, so a concurrent `spark login`, launch gate, or second supervisor
  never spends the same refresh token twice.
- Every spawn, restarts included, gets a fresh anonymous `socketpair`. The
  child's end is passed via `pass_fds` and named in `SPARK_TOKEN_FD`. It has no
  filesystem path, so nothing else can connect.
- **Protocol:** newline-delimited JSON, with requests capped at 16 KiB.
  - `{"op":"get"}` returns `{"ok":true,"token":<access>}`.
  - `{"op":"rejected","token":t}` forces a refresh unless `t` was already
    replaced.
  - A failure returns `{"ok":false,"kind":"auth"|"unavailable","code",
    "message","remediation"}`, which the child re-raises as a typed
    `SparkError`.

  The refresh token never crosses the socket.
- **Child (`provider_proxy` OAuth mode):** `BrokerClient` only. It has no
  Keychain, config, or session code, and a test asserts the source contains
  none. It refuses to start without `SPARK_TOKEN_FD`, and it exits if its
  parent dies, rather than serving with a token nobody refreshes.

**Result:** a compromised proxy leaks at most a ≤1 h access token. The
`CLAUDE.md` rule now holds with no exception: children never read the Keychain.

## 6. Verification

- `tests/test_chatgpt_provider.py` runs against fake OAuth and Responses
  servers. It covers the login flow, state/nonce/issuer checks, eligibility,
  refresh rotation and serialization, translation, the SSE collector, error
  mapping, local `/v1/models`, the 401 retry, and the launch gate. For the
  broker it covers:
  - the protocol, including malformed and oversized requests;
  - that no reply carries a refresh token;
  - a static guard on the child's source;
  - a real supervisor-spawned child that is served and crash-restarted through
    the broker, with `ps eww` showing no token in its environment.
- **Live (operator, P6):**
  1. `spark login openai`.
  2. Confirm the scopes include `chatgpt.tokens.use.direct` (login enforces it).
  3. `spark run openai`, then send one request with Symbiote's exact shape and
     time it against 15 s.
  4. Confirm `instructions` is accepted. If it is refused, switch
     `system → developer` (one line in `chatgpt_responses.py`).

### Live results (2026-11-26, operator account)

- **Sign-in:** registered with an issued client id; `eligible=true`
  (`chatgpt.tokens.use.direct` granted).
- **`spark run openai`:** ready in 0.8 s. `/v1/models` answered locally in
  under 1 ms with 10 plan models.
- **`instructions`:** accepted, and replies honoured the system rules (terse,
  at most 15 lines).
- **Symbiote-shaped requests** (`temperature` and `max_tokens` dropped; no
  retries, errors, or translation gaps):

  | Model | Total | TTFB | Output tokens |
  |---|---|---|---|
  | `gpt-6.1-terra` | 5.1 s | 0.87 s | 206 |
  | `gpt-6.2-sol` | 5.3 s | 0.77 s | 198 |
  | `gpt-6-luna` | 6.6 s | 1.35 s | 274 |

  All run at about 40 tok/s. Coaching replies fit 15 s with room to spare.
  **An 8000-token compile (about 200 s) can exceed Symbiote's 180 s timeout**,
  and the length cannot be capped because SIWC rejects `max_output_tokens`.
  Flagged to the operator.
- **Default model:** `gpt-6-luna` (operator choice). It is listed first in
  `/v1/models` (`data[0].id`).

## 7. Deferred

- Streaming Chat Completions translation (`stream: true` → chunks).
- `tools` translation (SIWC needs namespaced tools).
- ID token signature verification (needs an RSA dependency).
- `id_token_hint` and `login_hint` to skip the account selector.
