# spark

A hardened wrapper that fronts **every local LLM inference runtime** on this
machine behind one command. Type `spark`, fuzzy-tab through available models, hit
enter — the best-optimized runtime server for that model launches. `spark download
<hf-repo>` fetches a model and runs a multi-agent research chain to discover the
optimal per-runtime configuration for *this* host. Hosted OpenAI-compatible APIs
(OpenRouter, DeepSeek) and your **ChatGPT plan** (OpenAI's "Sign in with ChatGPT")
are served the same way, through a loopback proxy that holds the credential
([§3](#common-workflows)).

> **This README is the source of truth for design intent.** It explains not just
> *what* the code does but *why*, and — critically — documents the **integrations
> that live outside this repo** (Homebrew, zsh, Hugging Face, macOS Keychain, the
> agent CLIs). A change that looks self-contained in Python can silently break one
> of those couplings. Read [§9 External Integrations](#9-external-integrations--do-not-break-blind)
> and [§10 Decision Log](#10-decision-log-the-why) before modifying anything.

---

## Table of contents
1. [What spark is (and is not)](#1-what-spark-is-and-is-not)
2. [The host shapes everything](#2-the-host-shapes-everything)
3. [User-facing model (UX + commands)](#3-user-facing-model)
4. [Repository layout](#4-repository-layout)
5. [Architecture & data flow](#5-architecture--data-flow)
6. [Configuration system](#6-configuration-system)
7. [Secrets architecture](#7-secrets-architecture)
8. [Subsystems in depth](#8-subsystems-in-depth)
9. [External integrations — DO NOT break blind](#9-external-integrations--do-not-break-blind)
10. [Decision log (the "why")](#10-decision-log-the-why)
11. [Failure handling & logging contract](#11-failure-handling--logging-contract)
12. [Testing](#12-testing)
13. [Install / run / develop](#13-install--run--develop)
14. [Roadmap & known gaps](#14-roadmap--known-gaps)

---

## 1. What spark is (and is not)

**spark is an orchestrator, not an inference engine.** It does not implement model
loading, tokenization, or generation. It *drives already-installed runtime CLIs*
(`mlx_lm.server`, `mlx_vlm.server`, `omlx`, `llama-server`, `ollama`, plus the
`prism_ml` and `darkcore` children) as subprocesses, choosing the right one per
model and managing its lifecycle. Hosted providers get the same supervision via
spark's own stdlib proxy (`shims/provider_proxy.py`).

The single most important architectural consequence: **spark's own Python
environment never imports `mlx`, `vllm`, `torch`, or `llama_cpp`.** Its runtime
dependencies are four pure-Python packages (`click`, `pydantic`, `platformdirs`,
`rich`). This is deliberate — see [Decision D1](#d1-wrapper-not-reimplementation).

What spark provides on top of the raw runtimes:
- **One UX** over many runtimes, with fuzzy model completion.
- **Per-model, per-host optimal config** discovered by an agent research chain.
- **A hardened process supervisor** (preflight resource gates, health checks,
  bounded crash recovery, graceful shutdown).
- **A secrets vault** that keeps tokens out of files, argv, logs, and LLM prompts —
  and out of consumer apps (the provider proxy injects keys upstream).
- **Structured JSONL telemetry** sufficient to diagnose failures post-hoc.

---

## 2. The host shapes everything

spark was designed on and for a specific class of machine. The detected host:

| Fact | Value | Why it matters |
|---|---|---|
| Chip | Apple **M2** | MLX is the native-optimal runtime; vLLM has no viable path |
| Memory | **16 GB unified** (shared CPU/GPU/OS) | Hard ceiling on model size → memory budgeting is a launch gate, not advice |
| Cores | 8 (4P+4E) | — |
| Arch / OS | arm64 / Darwin | Gates which runtimes are even installable |
| Disk | runs near-full (has been ~95%) | Disk headroom is a **first-class fail state** for downloads |

These facts are *probed at runtime* (see [§8 Probe](#probe-srcsparkprobe)), not
hardcoded — but the defaults in `config/defaults.toml` (memory fraction, disk
floor, backend preference order) were tuned for this profile. On a different host
(e.g. a CUDA Linux box) the same code works; `vllm` would become available and the
preference order would warrant adjustment.

**Runtime availability on this host** (from `spark doctor`):

| Runtime | State | Role |
|---|---|---|
| `mlx_lm` | ✅ installed | **Primary — text** |
| `mlx_vlm` | ✅ installed | **Primary — vision/multimodal** |
| `llama_cpp` | ✅ installed | **#2 — GGUF/Metal**, broadest model coverage |
| `omlx` | ✅ installed | Alt MLX server (LRU multi-model) |
| `ollama` | ✅ installed | Convenience daemon (attach, never own) |
| `prism_ml` | ✅ installed | PrismML Hadamard packs via shim ([D12](#d12-prism_ml-serves-hadamard-packs-through-a-shim-not-a-fork)) |
| `openrouter`, `deepseek` | ✅ always | Hosted APIs via key-injecting proxy ([D19](#d19-hosted-providers-are-a-proxy-not-a-relay)) |
| `openai` | ✅ after `spark login openai` | ChatGPT plan usage via Sign in with ChatGPT ([D20](#d20-chatgpt-plan-usage-goes-through-siwc-never-codexs-private-backend)) |
| `relay` | ✅ always | Attach to an external OpenAI-compatible URL |
| `darkcore` | ⚠ needs `$DARKCORE_ROUTER_DIR` | Patchwork dynamic router child |
| `vllm` | ❌ absent | Registered but **unavailable on Apple Silicon** by design |

---

## 3. User-facing model

```
spark                          # what this host can serve (disk truth, registry or not)
spark <model>                  # launch the best-optimized runtime server for <model>
spark run <model> [--backend X] [--force] [--reprobe] [--dflash]   # same, with flags
spark download <hf-repo>       # fetch + research + register (stages config for review)
spark adopt <repo|path>        # register on-disk weights (hub cache or store) as-is
spark research <model>         # (re-)run the research chain for a registered model
spark config review <model> [--accept|--reject]   # apply/discard staged research
spark config import <model>    # import research JSON from stdin (manual path)
spark doctor [--no-cache] [--fix]   # probe host + runtimes + budgets; --fix installs missing runtime py deps
spark list [--all]             # runnable models (default) / everything (--all)
spark catalog [--json] [--refresh]   # the machine-readable roster (catalog.json)
spark forget <model>           # remove a registry entry (weights untouched)
spark secret set|ls|rm|get <name>                 # macOS Keychain vault
spark login|logout <runtime>   # Sign in with ChatGPT (plan usage) for an OAuth provider
spark config validate|path|show [--json]          # inspect configuration (compact; --json = full)
spark completion zsh           # emit the zsh completion function
spark __complete models [context] | describe <model>   # hidden: completion data
```

**Visual language.** Every command uses the same glyph vocabulary from
`cli/render.py`: `✓` done (green), `!` needs attention (yellow), `→` next action
(cyan), `•` list item (dim). Screens show signal first, with detail behind
`--all` / `--json`; long fields are ellipsized to terminal width rather than
wrapped. Byte sizes use KiB/MiB/GiB (`spark.formatting.fmt_size`) in both the CLI
and completion previews. `-h` is an alias for `--help`, and `spark -h` ends with
a quick-start block.

**The `spark <model>` fallthrough.** `spark` is a `click.Group` subclass
(`cli/app.py:SparkGroup`) whose `resolve_command` first tries normal subcommand
resolution; if the first token isn't a known subcommand and isn't a flag, it
dispatches to `run` treating the token as a model id. So real subcommands win over
same-named models, and any other token is a model to launch.

### Onboarding: first five minutes

```bash
spark doctor          # 1. runtimes present? budgets sane? (fix: brew / uv tool per hints)
spark list            # 2. what this host can serve right now
spark <model>         # 3. launch one (Ctrl+C stops; Tab-completes — see below)
```

Two more one-time steps, only if you need them:

- **Fuzzy Tab completion** (recommended): `spark completion zsh >
  ~/.config/zsh/completions/_spark`, source it per §9.3, start a new shell.
  Then `spark <TAB>` is a fuzzy menu with a live preview pane.
- **Gated/private repos**: `spark secret set hf_token` (reads via prompt, stored
  in the macOS Keychain, never in a file). Without it, gated downloads fail and
  the public model-card fetch still works — research just sees less.

If `spark list` shows nothing runnable, your next step is one of the two
workflows below: `download` (weights not here) or `adopt` (weights already here).

### Common workflows

**Run what's here.** `spark <model>` (or `spark run <model>`) picks the backend
— the entry's pinned one, else the highest-preference available runtime that is
format-compatible — and supervises it (preflight gates, health + warmup
verification, bounded restart, graceful Ctrl+C). Flags: `--backend` to force one,
`--force` to bypass the memory gate and the missing-weights / already-serving
refusals (never the fetch disk floor), `--reprobe` to re-detect the host first,
`--dflash` for the entry's opt-in speculative-decoding profile.

**Register weights already on disk** (hub cache or store — the `not registered`
rows of `spark list`). Three steps, in order — each depends on the previous:

```bash
spark adopt prism-ml/Ternary-Bonsai-8B-mlx-2bit   # 1. write the registry entry (weights untouched)
spark research ternary-bonsai-8b-mlx-2bit         # 2. agent chain discovers optimal flags → stages them
spark config review ternary-bonsai-8b-mlx-2bit --accept   # 3. apply the staged flags (or --reject to discard)
```

The model is runnable after step 1 (inferred defaults); steps 2–3 pin tuned
config. `--id`/`--backend` on `adopt` override the slug and pin a backend.
Note: `research` needs an `hf_repo` on the entry — hub-cache adopts have one;
a store-only adopt without a repo runs fine but skips tuning.

**Fetch a new model.** `spark download <hf-repo>` downloads (resumable, disk
preflight), registers, and researches unless told otherwise: `--no-research`
(skip LLM calls), `--no-fetch` (pointer only — `run` will refuse until weights
exist), `--force` (bypass the disk gate), `--id` (override the slug).

**Clean up.** `spark list --all` shows the defects: entries whose weights are
gone (re-fetch with the printed `spark download …`, or drop with `spark forget
<model>` — registry entry only, weights untouched) and cached repos spark
cannot serve (embeddings/speech, with reasons).

**Use a hosted provider (OpenRouter, DeepSeek).** spark runs a loopback proxy
that holds the API key, so your app talks to `127.0.0.1` with **no key**. One-time
setup per provider, in order:

```bash
# 1. Store the key in the Keychain (hidden prompt; never argv, never a file)
spark secret set openrouter_api_key        # DeepSeek: deepseek_api_key

# 2. Register the provider (one entry per provider; ids/backends must match)
cat > ~/.local/share/spark/models/openrouter.toml <<'TOML'
id = "openrouter"
backend = "openrouter"
model_format = "any"
research_status = "manual"
remote_model = "deepseek/deepseek-chat"    # optional: default id published to consumers
TOML
#    DeepSeek: id/backend = "deepseek", remote_model = "deepseek-chat"

# 3. Run it (Ctrl+C stops). The key is checked BEFORE the proxy starts.
spark run openrouter                       # → http://127.0.0.1:8100/v1   (deepseek: 8101)
```

Point any OpenAI SDK at the printed base URL (the endpoint is also published in
`run/instances/<id>.json`, [§9.8](#98-consumer-endpoint-discovery--the-published-record))
with any placeholder key, and send
the **provider's own model ids**: OpenRouter ids like `deepseek/deepseek-chat`;
DeepSeek ids `deepseek-chat` or `deepseek-reasoner`. Streaming works.

```bash
curl -s 127.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek/deepseek-chat","messages":[{"role":"user","content":"hi"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8100/v1", api_key="unused")  # spark injects the real key
client.chat.completions.create(model="deepseek/deepseek-chat", messages=[{"role": "user", "content": "hi"}])
```

If it fails:

| Error / status | Meaning | Do this |
|---|---|---|
| `SEC_NOT_FOUND` | no key stored | `spark secret set <provider>_api_key` |
| `SEC_REJECTED` | provider returned 401/403 for the key | re-store a valid key; check credit/revocation |
| `NET_UNREACHABLE` | DNS/TLS/timeout/5xx at launch | check network; provider status page |
| HTTP 403 from proxy | `Host` header not loopback | use `127.0.0.1` / `localhost` in the base URL |
| HTTP 415 from proxy | POST without `Content-Type: application/json` | SDKs set this; add it to raw requests |
| HTTP 502 `spark_proxy_error` | provider unreachable for that request | retry; see `logs/<provider>/` |

Per-request logs (model, status, latency; never content or key) land in
`~/.local/share/spark/logs/<provider>/<date>.jsonl`. Adding another
OpenAI-compatible provider is a TOML file in `config/runtimes/` (copy
`openrouter.toml`; §8 Runtimes). Remove one with `spark forget <id>`.

**Use your ChatGPT plan (OpenAI).** Requests are billed to your ChatGPT plan's
usage, not API per-token rates. This is OpenAI's sanctioned "Sign in with
ChatGPT" for "personal projects that run locally"; spark does **not** reuse Codex
CLI's private login ([D20](#d20-chatgpt-plan-usage-goes-through-siwc-never-codexs-private-backend)).

```bash
spark login openai            # browser sign-in; session (rotating tokens) → Keychain
                              # prints the models your plan may use
# pick a FAST model and set it as the default consumers see first:
#   ~/.local/share/spark/models/openai.toml  →  remote_model = "<model>"
spark run openai              # → http://127.0.0.1:8102/v1   (no key needed by clients)
spark logout openai           # revoke + clear tokens (registration kept)
```

Clients send **Chat Completions**. spark translates them to OpenAI's Responses
API under the plan's preview rules:
- system messages become `instructions`;
- `temperature`, `max_tokens` and similar are dropped;
- the upstream stream is collected into one non-streamed reply.

`/v1/models` is answered locally, with your `remote_model` first.
`stream: true`, `tools`, images, and `n > 1` get a 400 naming the field: they
are outside the current translation, and a 400 is better than a silent
mistranslation. Errors:

| Status | Meaning | Do this |
|---|---|---|
| 401 naming `spark login openai` | session expired, revoked, or refresh token reused | `spark login openai` |
| 403 `subscription_sharing_user_not_eligible` | plan not eligible | check the plan in ChatGPT settings |
| 429 `subscription_sharing_usage_limit_exceeded` | plan usage limit hit; passed through unchanged | wait; see chatgpt.com/settings/usage |
| 400 naming a field | request outside the translation contract | drop that field |

The consumer contract lives in `SYMBIOTE-LLM-CONTRACT.md`, the design in
`specs/0005-chatgpt-plan-provider/prd.md`.

**Secrets & config.** `spark secret set|ls|rm|get <name>` (Keychain; `get`
needs `--reveal`). `spark config validate|path|show` to inspect; research
staging lives outside it (`review`/`import`).

---

## 4. Repository layout

```
spark/
├── config/                      # SHIPPED configuration (DATA, not code)
│   ├── defaults.toml            # general/memory/disk/supervisor/research defaults
│   └── runtimes/*.toml          # one RuntimeDef per backend (launch templates)
├── completions/_spark           # generated zsh completion (reference copy)
├── src/spark/
│   ├── cli/                     # thin click layer; delegates to core
│   │   ├── app.py               # root group + SparkGroup fallthrough + main()
│   │   ├── context.py           # SparkCtx: builds paths/config/secrets/telemetry once
│   │   ├── run.py download.py research.py doctor.py secret.py config_cmd.py
│   │   ├── completion.py        # `completion zsh` + hidden `__complete`
│   │   └── render.py            # rich tables + actionable error rendering
│   ├── config/                  # paths.py (platformdirs/XDG), schema.py (pydantic), loader.py
│   ├── secrets/                 # store.py (ABC + redaction), keychain.py (ctypes Security),
│   │                            #   oauth.py (Sign in with ChatGPT: login/refresh/revoke),
│   │                            #   broker.py (supervisor→child access-token socketpair)
│   ├── runtimes/                # base.py (Backend ABC + registry) + per-runtime adapters
│   ├── shims/                   # standalone stdlib servers spark supervises as children:
│   │                            #   openai_filter_proxy (mlx_lm listing), provider_proxy
│   │                            #   (hosted APIs; chatgpt_responses = Chat→Responses
│   │                            #   translation) run on spark's interpreter; prism_shim
│   │                            #   on mlx-lm's. Shim-only dir: a script's dir is on
│   │                            #   sys.path, so runtimes/mlx_lm.py would shadow mlx_lm
│   ├── probe/                   # host.py (capability detection + cached profile)
│   ├── registry/                # models.py (one TOML/model, fuzzy resolve),
│   │                            #   availability.py (entry vs disk), store.py (store
│   │                            #   scan), adopt_errors.py (typed adopt failures)
│   ├── catalog.py               # catalog.json + run/instances publishing (§9.8)
│   ├── inventory.py             # disk-first join: store + hub cache vs registry
│   ├── budget.py                # memory/disk gates + metadata hydration
│   ├── hf_cache.py              # HF cache paths, repo-id and ref validation
│   ├── redaction.py             # shared secret registry + scrubbing (no imports
│   │                            #   from telemetry/ or secrets/ — avoids a cycle)
│   ├── runner/                  # supervisor.py, server_log.py, ports.py
│   ├── research/                # types/guard/prompt/providers/chain/staging/validate
│   ├── telemetry/               # jsonl.py (structured logging; redaction imported)
│   └── errors.py                # typed errors carrying remediation
├── tests/                       # pytest (~670 tests)
├── specs/<NNNN>-<slug>/         # PRDs + status.yaml (0004 hosted providers, 0005 ChatGPT plan)
├── SYMBIOTE-LLM-CONTRACT.md     # what the Symbiote consumer sends/expects (drives 0005)
├── spikes/                      # time-boxed evaluation spikes + the portable
│                                #   human/emotion eval skill (spikes/README.md)
├── bake-offs/                   # comparative model bake-offs — quality evals,
│                                #   distinct from patchwork's throughput benches
├── MODEL-EVAL-2026-07-15.md     # cross-model capability matrix + tactical use/dismissal
├── RELAY-USAGE.md               # relay how-to (attach to an external service)
└── CONTINUE.md                  # session handoff: unfinished work only
```

> **Model evaluation artifacts.** `spikes/`, `bake-offs/`, and the dated
> `MODEL-EVAL-*.md` at root capture *which model to reach for and why* — capability
> matrices, expect/employ guidance, and reproducible test batteries. Start with
> [`MODEL-EVAL-2026-07-15.md`](MODEL-EVAL-2026-07-15.md) for the current
> at-a-glance picture.

**Layering rule:** `cli/` is thin and may import anything; core packages
(`config`, `secrets`, `runtimes`, `probe`, `registry`, `runner`, `research`,
`telemetry`) must not import from `cli`. Business logic receives a validated
`SparkConfig` and never parses TOML or reads env/secrets directly.

---

## 5. Architecture & data flow

### `spark <model>` (launch)
```
build_context()                          # cli/context.py
  → resolve_paths().ensure()             # ~/.config/spark, ~/.local/share/spark
  → init_telemetry(log_dir, session_id)
  → load_config(paths)                   # defaults ⊕ runtimes/*.toml ⊕ user overrides
  → get_secret_store()                   # KeychainStore
resolve_model(query)                     # registry: exact id → alias → unique prefix/substr
load_or_probe(config, host_profile)      # probe: cached host.toml or fresh detection
select_backend_for(entry, cfg, available)# registered backend, else preference order, format-compatible
get_backend(name, cfg)                   # runtimes registry → Backend subclass
Supervisor(backend, entry, cfg, profile, paths, secrets).run()
  → preflight: weights present? mem budget? disk? free port (or fixed daemon port)
      · weights (registry vs. disk): a launch whose entry asserts weights that are
        absent is REFUSED (MODEL_WEIGHTS_MISSING) unless --force — otherwise the
        runtime silently fetches GBs at the first request, behind a health gate
        that has already passed.
      · disk: enforced when the launch can write (absent weights / forced fetch),
        advisory when the weights are already local — a tight disk must not block
        a run that writes nothing.
  → backend.prepare(entry)               # ollama: ensure daemon + pull tag;
                                         #   providers: key stored + accepted upstream
  → attach-mode? if daemon already healthy → monitor without owning
  → build_launch_cmd(entry, host, port)  # template ⊕ resolved model ⊕ override flags
  → child env = os.environ ⊕ extra_env ⊕ resolved secret_env (values injected here only)
  → spawn → wait_ready (poll health_url) → verify_generation → monitor (RSS sampling)
  → on crash: restart with exp backoff up to max_restarts → else RecoveryExhausted
  → SIGINT/SIGTERM: graceful terminate (SIGTERM → grace → SIGKILL)
```

### `spark download <hf-repo>`
```
_slug(repo) + _infer(repo)               # id, model_format (gguf/mlx/mlx-vlm), quant, params
disk preflight (disk.min_free_gib floor)
hf download <repo> --local-dir <store>/<id>     # HF_TOKEN injected to env iff stored
save_model(entry)                        # registry, research_status="pending"
perform_research(ctx, entry)             # unless --no-research / research.enabled=false
  → build_request: host_summary + runtime_names + PUBLIC model card  (NO secrets)
  → ResearchChain(providers).run(request)# claude → hermes → pi, first schema-valid wins
      providers: each runs an agent CLI subprocess; secrets-guard asserts on prompt
  → stage(paths, id, output, provider)   # data/staging/<id>.json  (awaiting review)
spark config review <id> --accept        # apply_output_to_entry → save_model → status=registered
```

### Readiness: "answers HTTP" is not "can generate"

`mlx_lm.server` / `mlx_vlm.server` bind their port **before** the model is
resident, and answer `/v1/models` from an HF-cache scan — so a passing health
probe says nothing about the model. spark therefore issues one minimal
generation (`Backend.warmup`, 1 token) after the health check and before
declaring the server ready (`supervisor.verify_generation`, default on; skipped
for eager loaders and attached daemons). Without it, a runtime whose model load
crashed stays "ready" forever: the process never exits, so the crash-restart
path never fires and every request hangs. Verified live: a dead generator thread
answered `/v1/models` and `/health` with 200 while producing no tokens.

### Weights: disk truth first, registry as overlay

`registry/models/*.toml` is a list of *intentions*; the filesystem owns reality.
`inventory.py` scans both weight locations — the spark store and the HF hub
cache — classifies each snapshot as servable (`llm`: generative weights a
runtime can serve) or cached-but-not-servable (embedders, speech-to-text/TTS,
partial downloads), and joins that against the registry. `spark list`, `spark
run`, and completion all read that join:

* **runnable** — servable weights on disk, registered or not — is the only list
  ever offered. An unregistered-but-present model runs via a synthesized
  ephemeral entry (hub models by repo id, store models by path); `spark adopt`
  promotes it into the registry when it should stay.
* **missing** — an entry asserting absent weights — is a defect to re-fetch or
  `forget`, reported as a pointer, never offered. `spark run` refuses it rather
  than triggering an invisible multi-GB fetch behind a passing health check.
* **non-servable cache** is counted, never offered (`--all` names it with reasons).

`registry/availability.py` remains the per-entry predicate (one entry vs. disk);
`inventory.py` is the fleet-wide join (everything on disk vs. all entries). The
launch preflight, `spark list`, completion, and `catalog.json` (`discovered`
key, additive to the v1 contract) all consult the same join, so they cannot
disagree.


---

## 6. Configuration system

**Principle: configuration is data, not code.** No launch flag, port, or quant
multiplier lives in `src/`. They live in TOML and are validated into pydantic
models (`config/schema.py`, all `extra="forbid"` — an unknown/typo'd key is a hard
error, which also blunts config-injection).

**Precedence (low → high), merged then validated** (`config/loader.py`):
1. `config/defaults.toml` (shipped)
2. `config/runtimes/*.toml` (one `RuntimeDef` per file; filename ≈ runtime name)
3. `~/.config/spark/config.toml` (user overrides; may add/override runtimes)

**Behavior knobs that live in TOML, not Python:** `probe.max_age_s` (host-probe
cache TTL); `supervisor.health_probe_timeout_s`, `instance_heartbeat_s`,
`rss_sample_ticks`, `server_log_roll_mib`; `research.flag_probe_timeout_s`.
Launch flags, ports, quant multipliers and these policy values all live in
`config/`; a Python literal default for them is a bug.

**Path resolution** (`config/paths.py`), highest precedence first:
1. `$SPARK_HOME` → `<root>/{config,data,cache}` (used by tests and isolation)
2. `$XDG_CONFIG_HOME` / `$XDG_DATA_HOME` if set
3. `~/.config/spark` and `~/.local/share/spark` (the operator-expected layout)

`platformdirs` is only the cross-platform cache fallback; primary paths are
XDG-style on purpose (operator is a Linux-leaning engineer on macOS).

**Bundled-config discovery** (`paths.bundled_config_dir`): source checkout →
`<repo>/config`; installed wheel → `spark/_bundled_config` (pyproject
`force-include`). **An editable `uv tool install` keeps using the repo's `config/`**
because `__file__` points into the repo `src/`.

**Data directory** (`~/.local/share/spark/`, mode 0700):
`models/<id>.toml` · `store/<id>/` (downloaded weights) · `staging/<id>.json`
(research awaiting review) · `logs/<component>/<date>.jsonl` · `host.toml`
(probe cache) · `logs/servers/<id>.log` (child stdout/stderr, scrubbed, rolled)
· `catalog.json` + `run/instances/<id>.json` (published roster/endpoints, §9.8)
· `run/locks/` (one flock per published endpoint, so a SIGKILLed publisher's
manifest is detectable without a `ps` fork) · `secrets-index.json` (non-secret
names only).

---

## 7. Secrets architecture

**Threat model (operator's):** assume non-zero chance of same-user malware. So a
secret must never be observable from outside the process that needs it.

Rules enforced in code:
- **Never in a text file.** Backend is the **macOS Keychain**.
- **Never on argv.** `security add-generic-password -w <value>` exposes the value
  in `ps`. We bypass the CLI entirely and call the **Security framework via
  `ctypes`** (`secrets/keychain.py`) — value stays in process memory.
  ([Decision D2](#d2-keychain-via-ctypes-not-the-security-cli).)
- **Never in logs.** `telemetry/jsonl.py` redacts (a) any value registered via
  `spark.redaction.register_secret_value()` — the store registers every value it
  reads — and (b) sensitive key names (`token`, `secret`, `password`,
  `authorization`, …). Registered values of four characters or more are masked
  wherever they appear; shorter values are masked only in credential contexts
  (`hf_token=1`, `Bearer x`). Blindly replacing a standalone `1` destroys every
  version string (`Llama-3.1-8B`), so there is no safe context-free scrub for a
  1–3 character token. The prompt guard still detects short values on word
  boundaries. Do not store tokens shorter than four characters.
- **Never in an LLM prompt.** The research path never calls `secrets.get()`, and
  `research/guard.py:assert_no_secrets()` scans every outbound prompt for
  registered values + credential patterns (`hf_`, `sk-`, `AKIA`, `ghp_`, bearer)
  and aborts on a hit.
- **Only path a secret takes:** vault → child process env at spawn
  (`Supervisor._child_env` via `SecretStore.resolve_env`). The launch event logs
  only which env var names were set, never values.
- **Rotating credentials never reach a child** (spec 0005 §5). A Sign in with
  ChatGPT refresh token is single-use and rotates on every refresh, so env
  injection cannot carry it. Instead the supervisor's `TokenBroker`
  (`secrets/broker.py`) owns the session and hands the proxy child short-lived
  access tokens:
  - The channel is an anonymous `socketpair`, passed via `pass_fds` as
    `SPARK_TOKEN_FD`, with a fresh one for every spawn.
  - The protocol has two operations: `get`, and `rejected` (force a refresh).
  - The network-facing child holds at most a ≤1 h access token, never the
    30-day refresh token, and never reads the Keychain.

  Refreshes run under one `flock` (`run/locks/<session>.oauth.lock`), so
  `spark login`, the launch gate, and any supervisor never spend the same
  refresh token twice.

`SecretStore` is an ABC; `KeychainStore` is the macOS impl. Service name is
`spark`; secret *names* (e.g. `hf_token`) are not secret and are tracked in
`secrets-index.json` for `list_names()` (the Keychain enumeration API is awkward).

CLI hygiene: `spark secret set` reads via `getpass`/`--stdin` (never an arg);
`spark secret get` requires `--reveal` and bypasses rich markup.

---

## 8. Subsystems in depth

### Runtimes (`src/spark/runtimes/`)
`base.py:Backend` is a generic server launcher driven entirely by a `RuntimeDef`.
A registry (`@register("name")`) maps runtime names to `Backend` subclasses;
unknown names fall back to the generic `Backend`. Key methods:
- `build_launch_cmd(entry, host, port)` — fills the TOML `args_template`
  placeholders (`{model_path} {model_id} {host} {port}`) then **appends
  `launch_overrides` as real CLI flags** via `override_flags()`
  (`{"--max-tokens":"2048"}` → `["--max-tokens","2048"]`; `""` → bare flag).
- `secret_env_map()` / `extra_env()` — env for the child.
- `prepare(entry, secrets=...)` — pre-launch hook (default no-op).
- `health_url()` / `openai_base_url()` / `attach_if_running`.
- `select_backend_for()` — registered backend if available, else the first
  `general.preference` entry that is available **and** format-compatible.

Adapters and their quirks:
- **`mlx_lm`** — `mlx_lm.server --model <path|repo> --host --port`; accepts a local
  dir or an HF repo id. Primary text path. Port 8095 (canonical). Runs behind
  `shims/openai_filter_proxy.py` (`[server] filter_models=true`): the proxy binds
  the published port, spawns the real server on an ephemeral loopback port, and
  answers `GET /v1/models` from spark's inventory (registered ∩ present,
  mlx-servable) instead of the server's hub-cache scan — so harness pickers never
  see embeddings/speech. Everything else passes through; still one supervised
  child (proxy exits when the backend dies, taking it down for restart). The
  supervisor reserves the published port with a live listening socket and hands
  that fd to the proxy (`SPARK_LISTEN_FD` + `pass_fds`), so there is no
  check-then-bind race for this path. Backends that bind their own port still
  have a residual window spark cannot close without runtime cooperation.
- **`mlx_vlm`** — `mlx_vlm.server`; vision. Its binary defaults to host `0.0.0.0`,
  so the template **pins `--host 127.0.0.1`** (no LAN exposure). Port 8095 (it and
  `mlx_lm` share the canonical endpoint; a second server drifts within the range).
- **`llama_cpp`** — resolves to a local `.gguf` file (`--model`, picks the
  `00001-of` shard for splits) **or** an HF repo (`-hf user/repo[:quant]`, which
  `llama-server` downloads itself). Static Metal flags from TOML: `-ngl 999 -fa
  auto --jinja`. Health is `/health` (not `/v1/models`). Port 8096.
- **`omlx`** — `omlx serve <model> --host --port --memory-guard safe`. Port 8097.
- **`prism_ml`** — `shims/prism_shim.py` on the mlx-lm tool interpreter serves
  PrismML Hadamard packs (`model_format="prism"`); see D12. Port 8098.
- **`ollama`** — daemon-style. `attach_if_running=true`: if the daemon is already
  healthy, **attach (never spawn/kill the shared daemon)**. `prepare()` requires
  the daemon (`/api/tags`) and `ollama pull`s the tag if missing. OpenAI endpoint
  at `:11434/v1`. Model is addressed by *tag*, not path.
- **`vllm`** — registered with `requires_os=["Linux"]` + `unavailable_reason`; on
  this host `is_available()=False` with actionable guidance. Kept so the config
  travels unchanged to a CUDA host.
- **`relay`** — attach-only: health-checks an external OpenAI-compatible service
  (`relay_base_url`) and publishes **that URL** as the endpoint; it never spawns,
  kills, or proxies anything, and injects no auth. Its URL is a *runtime* setting
  (`[runtimes.relay.server]` in the user config), not a model-entry field. Only
  the runtime named `relay` gets this behaviour, so one target at a time. For an
  API that needs a key, use a hosted-provider runtime instead. How-to:
  `RELAY-USAGE.md`.
- **`openrouter`, `deepseek`** (hosted providers, spec 0004) — any runtime TOML
  with `[server] provider_base_url` gets `ProviderBackend`, so a new
  OpenAI-compatible provider is config only. The supervised child is
  `shims/provider_proxy.py`, a loopback proxy that injects
  `Authorization: Bearer <key>` (Keychain → `secret_env SPARK_PROVIDER_API_KEY`
  → child env). Consumers use `http://127.0.0.1:<port>/v1` with **no key** and
  send the provider's own model ids. `prepare()` checks the key before spawn:
  missing → `SEC_NOT_FOUND`, 401/403 → `SEC_REJECTED` (never retried). Ports
  8100/8101, deliberately above the local range so they never collide with a
  local server. Per-request JSONL lands in `logs/<runtime>/`. Operator steps:
  §3 "Use a hosted provider".
- **`openai`** (ChatGPT plan, spec 0005) — the same proxy in OAuth mode, selected
  by `[server.provider_oauth]`, where every endpoint and scope is TOML.
  - `spark login` runs OpenAI's Sign in with ChatGPT: dynamic registration
    (`dynamic_agent_client` → an issued `oaiapp_…` id), PKCE S256, `state` and
    `nonce`, a `127.0.0.1:1455/auth/callback` listener, and the `iss`/`aud`/
    `exp`/`nonce` ID-token checks. Login refuses an account without the
    `chatgpt.tokens.use.direct` scope (`SEC_NOT_ELIGIBLE`).
  - The proxy translates Chat Completions to `POST /v1/responses` (stream on,
    store off), answers `/v1/models` from a cached catalog, and retries an
    upstream 401 once after a forced refresh.
  - Tokens come from the supervisor's `TokenBroker` over `SPARK_TOKEN_FD` (§7).
    The child never holds the refresh token, and it exits if the supervisor
    dies.
  - Port 8102.
- **`darkcore`** — spawns the patchwork dynamic router under the supervisor
  (`python -m darkcore.server` from its own uv venv, set via `$DARKCORE_ROUTER_DIR`).
  Like relay, the router resolves its own tier weights. Pure-TOML runtime (generic
  `Backend`, no subclass), the precedent spec 0003 builds on. Port 8000.

### Probe (`src/spark/probe/`)
Detects chip/memory/cores + per-runtime presence (`shutil.which`), version
(best-effort; many tools lack `--version` so a regex extracts a version token or
returns ""), and host-gating (`requires_os`/`requires_arch`). Result is cached to
`host.toml` keyed by a content fingerprint of the runtime defs + a 24h TTL;
`spark doctor --no-cache` / `spark <model> --reprobe` force a fresh probe.

### Budget (`src/spark/budget.py`)
`usable_memory = total * memory.usable_fraction (0.65)`. Footprint estimate is
quant-aware (`params_billions * bytes_per_param[quant] * overhead 1.20`) with a
size-based fallback; unknown → no gate (warn-only). Disk check enforces
`disk.min_free_gib` (10) against current free space. Both are launch gates unless
`--force` (memory) / appropriate flags.

### Supervisor (`src/spark/runner/supervisor.py`)
Preflight → `prepare()` → attach-or-spawn → `wait_ready` (poll `health_url` until
2xx, or fail if the process dies / times out) → `monitor` (poll liveness + sample
peak RSS) → on unexpected exit, restart with exponential backoff
(`backoff_base 1s → backoff_max 30s`, `max_restarts 3`) → else
`RecoveryExhaustedError` with remediation. SIGINT/SIGTERM set a stop event →
graceful `terminate` (SIGTERM, `shutdown_grace 10s`, then SIGKILL). Child
stdout/stderr go through a **scrubbing pipe** (registered secrets masked) to
`logs/servers/<id>.log`, rolled at `server_log_roll_mib`; the CLI tails it on
failure. Lazy loaders also pass the warmup gate (§5 Readiness). **Attach mode** (daemons):
uses the fixed `default_port` (does not reassign), and never terminates a process
it did not start.

### Research (`src/spark/research/`)
- `types.py` — `ResearchRequest` (secret-free inputs), `ResearchOutput`
  (`recommended_backend` + per-runtime `RuntimeRec{quant, launch_overrides,
  context_length, rationale}`), pydantic-validated.
- `prompt.py` — `host_summary` (non-secret facts), `fetch_model_card` (public
  `https://huggingface.co/<repo>/raw/main/README.md`, **no auth header**), and the
  JSON-only prompt.
- `guard.py` — `assert_no_secrets()` (see §7).
- `providers.py` — `CliAgentProvider` runs a configured agent CLI; `_extract_json`
  tolerates raw JSON / fenced blocks / JSON embedded in prose (string-aware brace
  matching). `parse="json_envelope"` unwraps `claude --output-format json`'s
  `.result`.
- `chain.py` — tries providers in order, skips unavailable, returns first
  schema-valid result; distinguishes "none available" from "all failed".
- `staging.py` — write/read/delete `staging/<id>.json`; `apply_output_to_entry`.

---

## 9. External integrations — DO NOT break blind

These are the couplings a new agent is most likely to overlook. **None of them are
visible from the Python imports.** Changing related code without accounting for
them will silently break user-facing features.

### 9.1 Homebrew (`/opt/homebrew`)
- `llama.cpp` (`llama-server`, `llama-cli`, …), `omlx`, and `fzf` are installed via
  **Homebrew**. spark detects them by name on PATH; it does **not** install them.
- If `spark doctor` shows one unavailable, the fix is a `brew install` (the
  `install_hint` in the runtime TOML says which). Don't "fix" availability in code.
- Homebrew's bin dir must be on PATH for detection. On Apple Silicon that's
  `/opt/homebrew/bin`.

### 9.2 uv-managed runtime tools (`~/.local/bin`)
- `mlx_lm.server` / `mlx_vlm.server` live in `~/.local/bin` (installed via
  `uv tool install mlx-lm` / `mlx-vlm`), in **their own** isolated environments —
  **not** in spark's environment. spark calls them as subprocesses.
- `~/.local/bin` is also where `uv tool install` puts the **`spark` launcher**
  itself, and where `hermes` lives. It must be on PATH.
- **Do not add `mlx`/`mlx_lm` to spark's `pyproject.toml`.** spark must stay free of
  ML deps ([D1](#d1-wrapper-not-reimplementation)). Those tools are upgraded
  independently by the operator.

### 9.3 zsh completion (the most fragile integration)
The fuzzy `spark <Tab>` menu depends on a chain of files **outside this repo**:
- `~/.zshrc` — has **one appended line**: `[[ -f ~/.config/zsh/spark.zsh ]] &&
  source ~/.config/zsh/spark.zsh`. Removing it disables completion.
- `~/.config/zsh/spark.zsh` — adds `~/.config/zsh/completions` to `fpath`, then
  **explicitly runs `compdef _spark spark`**, then sources fzf-tab and sets the
  preview zstyle. **Why `compdef` instead of relying on compinit:** `~/.zshrc`
  runs `compinit -C` (cached) *before* sourcing spark.zsh, and `-C` does **not**
  rescan `fpath` for new `#compdef` functions — so the function would never bind.
  Explicit `compdef` is cache-proof. (See [D5](#d5-explicit-compdef-not-a-compinit-rebuild).)
- `~/.config/zsh/completions/_spark` — the completion function. It is **generated**
  by `spark completion zsh` (source of truth: `cli/completion.py:_ZSH_COMPLETION`).
  The repo's `completions/_spark` is a reference copy. **If you change subcommands,
  or the completion contract, regenerate both** and re-`compdef`. An already-open
  shell keeps the function it autoloaded: start a new shell (or `unfunction
  _spark; autoload -Uz _spark`) to pick up a regenerated one.
- `~/.config/zsh/plugins/fzf-tab/` — the fzf-tab plugin (git clone). Provides the
  fuzzy popup; requires `fzf` (Homebrew). Without it, completion degrades to the
  native menu (still works).
- The completion function calls `command spark __complete models "$ctx"` to get
   `id<TAB>desc` lines, where `$ctx` is the subcommand being completed (empty at
   position 2, i.e. a bare `spark <TAB>`). The context is what keeps entries with
   missing weights out of launch menus while still naming them for `forget`. The
   preview pane calls `spark __complete describe $word`, which also describes
   unregistered-but-present models (with an `adopt` hint). **`__complete` must stay
   fast, stdout-only, and never error** (it's wrapped in try/except and bypasses
   `build_context`). Breaking it breaks Tab.

### 9.4 Hugging Face
- **Downloads** shell out to the `hf` CLI (`hf download <repo> --local-dir …`),
  detected on PATH (Homebrew `huggingface-cli`/`hf`). spark does not use the
  `huggingface_hub` Python package.
- **Gated/private repos**: if a `hf_token` secret exists, it is injected as
  `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` into the child env of `hf download` and of
  the runtime servers (via each runtime TOML's `[server.secret_env]`). Never argv.
- **Model card** for research is fetched from the **public** raw URL with **no auth
  header** — deliberately, so research can never leak a token. Keep it that way.
- **llama.cpp `-hf`**: `llama-server` itself downloads GGUF from HF when given
  `-hf user/repo[:quant]`. That network call happens inside llama-server, not spark.
- **The hub cache is read by TWO different lists, and they disagree.** spark's
  registry (`models/*.toml`) is one namespace; `mlx_lm.server`'s `/v1/models` is
  another — it scans `scan_cache_dir()` and returns every *cached MLX-shaped repo*
  plus the loaded model if `--model` is an existing path. Agent harnesses read the
  second, spark reads the first, and neither sees the other. Consequences to
  respect when touching either side:
    · a model downloaded by `spark download` into `store/` is **invisible** to
      harnesses (it is not a hub-cache repo), and appears at the endpoint only as
      an absolute path while it is loaded;
    · reclaiming a hub-cache repo removes it from every harness while the registry
      entry keeps advertising it — which is why the registry is reconciled against
      disk (`registry/availability.py`) and `spark list` shows an `avail` column;
    · `hf cache prune` **cannot** delete a repo whose `snapshots/` dir is gone (verified);
      a half-reclaimed repo is invisible to `hf cache ls` and inert to `prune`, so it
      has to be cleaned by hand.

### 9.5 macOS Keychain / Security.framework
- Secrets use `SecKeychainAddGenericPassword` / `…FindGenericPassword` /
  `…ItemDelete` via `ctypes` against
  `/System/Library/Frameworks/Security.framework`. These are **deprecated but
  functional** (the `security` CLI uses them too). If a future macOS removes them,
  migrate to `SecItemAdd`/`SecItemCopyMatching` with `CFDictionary`.
- Items are owned by the creating binary (the tool's Python). The first read from a
  new interpreter path may prompt; the editable `uv tool` path is stable.

### 9.6 Agent CLIs for research (`claude`, `hermes`, `pi`)
- The research chain runs these as subprocesses, configured in
  `config/defaults.toml [[research.providers]]`. A provider is **skipped** if its
  `detect_binary` isn't on PATH — so the chain degrades gracefully.
- **`claude`** is verified: `claude -p --output-format json` reads the prompt on
  **stdin** and returns a JSON envelope whose `.result` holds the model's text
  (hence `parse="json_envelope"`, `envelope_field="result"`).
- **`hermes`** (`~/.local/bin/hermes`): `hermes -t clarify -z <prompt>` (oneshot,
  prompt as argv, prints only the response) — verified live, `parse="raw_json"`.
- **`pi`** (`~/.bun/bin/pi`): `pi -p --no-tools <prompt>`, `parse="raw_json"`. It
  needs pi's own `defaultProvider` reachable (ollama on this host), so it is the
  fallback most often unavailable. Re-verify either if its CLI changes flags.
- Running `claude` from inside spark spawns a **nested** Claude Code session (real
  cost + network). `spark download` runs it by default; `--no-research` opts out.

### 9.7 Networking posture
- Servers bind `general.host = 127.0.0.1` (loopback only) by default. Don't change
  this to `0.0.0.0` without intent — `mlx_vlm.server` defaults to `0.0.0.0` and is
  explicitly pinned back to loopback in its TOML.
- Egress happens only in: `hf download`, llama `-hf`, the public model-card fetch,
  the agent CLIs, and hosted-provider proxies (verified TLS to the TOML's
  `provider_base_url` only), plus `auth.openai.com` for ChatGPT sign-in and token
  refresh. Everything else (probe, launch, health) is local.
- `spark login` briefly listens on `127.0.0.1:1455` (registration), or any free
  loopback port (re-sign-in), for the OAuth callback. It accepts exactly one
  `/auth/callback` with a matching `state`, then closes.
- The provider proxy treats loopback as reachable from any browser tab. It
  requires a loopback `Host` (DNS rebinding) and `application/json` POSTs
  (forces a CORS preflight, which it never answers). It forwards only `GET`/`POST`
  under `/v1/`, and it never forwards the client's `Authorization`/`Cookie`.

### 9.8 Consumer endpoint discovery — the published record

Consumers do not have to guess a URL or scrape the CLI. Two artifacts under the
data dir are the contract:

- `run/instances/<model>.json` — one v1 record per live server, written when the
  server becomes ready, heartbeat-updated while it runs (so a stale file from a
  `SIGKILL`/power-loss is detectable), and deleted on exit. Fields:
  `schema_version=1`, `session_id`, `pid`, `state="ready"`, `base_url`,
  `port`, `api_contract="openai.chat-completions.v1"`, `model_id` (the exact API
  model id a client must send in the request body), `model_alias` (spark registry
  id), `backend`, `health_url`, `started_at`, `updated_at`. Consumer-side reader:
  `banter/rust/crates/backend-spark/src/discovery.rs`, which requires a bounded
  regular non-symlink file ≤64 KiB and cross-checks the recorded pid against the
  process table. **Changing a field name breaks that reader** — it is a published
  contract, not internal state. The directory is `0700` and the records `0600`.
- `catalog.json` — the roster: every registered model with availability as of
  `generated_at` (state, location, bytes, `checked_at`), every servable-but-
  unregistered on-disk model under `discovered`, plus the live instances.
  Refreshed by `spark list`, `spark catalog`, `download`, `adopt`, `forget`, research
  accept/import, and on server ready/exit. Consumers that need a model list (not
  just the live endpoint) read this instead of parsing `spark list`.

The port convention is part of the same contract: spark's own servers prefer
8095 (`general.port_range = [8095, 8099]`; llama.cpp 8096, oMLX 8097, prism_ml
8098), hosted providers 8100/8101/8102 (openrouter/deepseek/openai, outside the
range, so they never take a local server's slot; if busy they drift into it). A taken
preferred port still causes drift, but drift is now logged (`port_drift`) and
published, so a consumer that reads the instance record is unaffected.

---

## 10. Decision log (the "why")

### D1. Wrapper, not reimplementation
spark drives installed runtime CLIs and never imports ML libraries. **Why:** (a)
keeps the dependency/attack surface tiny (4 pure-Python deps) per the operator's
supply-chain stance; (b) decouples spark from the brutal ML-wheel compatibility
treadmill — notably the system Python is 3.14, where many ML wheels lag, but spark
doesn't care because it never imports them; (c) lets each runtime be upgraded
independently. **Consequence:** features are added by config + subprocess
orchestration, not by importing engines.

### D2. Keychain via ctypes, not the `security` CLI
`security add-generic-password` can only take the secret on argv (visible in `ps`)
or interactively from a tty (confirmed: it ignores a piped stdin and prompts
twice). Both are unacceptable under the threat model. The deprecated `SecKeychain*`
C API via ctypes keeps the value in process memory with **zero new dependencies**.
Rejected alternatives: the `keyring` PyPI package (extra deps), an age/sops
encrypted file (still a file on disk; key management problem).
Presence checks (`exists`, delete's lookup) are attribute-only (NULL data
pointers): fetching the value is what raises the Keychain "allow access" dialog,
which would block a non-interactive `spark run` just to learn a token exists.

### D3. venv pinned to Python 3.12 (system is 3.14)
spark's deps are pure-Python and run on 3.14, but the venv pins 3.12 for a stable,
widely-supported interpreter and reproducibility. spark never imports mlx/vllm so
3.14's ML-wheel lag never reaches it — the pin is defensive consistency, not a hard
requirement.

### D4. fzf-tab for completion (operator-approved global deps)
The requested UX ("a fuzzy list that autocompletes") maps to fzf-tab + fzf. The
operator explicitly approved installing them. Native compsys is the zero-dep
fallback and still works if fzf-tab is absent. We isolate **all** shell config in
`~/.config/zsh/spark.zsh` so uninstall = delete that file + one `.zshrc` line.

### D5. Explicit `compdef`, not a compinit rebuild
See [§9.3](#93-zsh-completion-the-most-fragile-integration). Because `~/.zshrc`
runs `compinit -C` before our config loads, rescanning `fpath` won't pick up
`_spark`; we bind it explicitly. This is the single least-obvious shell detail.

### D6. Research output is staged, not auto-applied
An LLM writes the launch flags you'll run. Auto-registering that unreviewed is
risky. So research **stages** to `staging/<id>.json`; `spark config review
--accept` is the human gate. The model is still runnable before review (with
inferred defaults). Operator's added requirement: **multi-agent fallback**
(claude → hermes → pi) so one provider being down doesn't block research.

### D7. Config is data; schema is strict
All tunables live in TOML; pydantic models use `extra="forbid"` so a typo or an
injected key fails loudly rather than silently changing behavior. Launch flags
belong in `config/runtimes/*.toml`, never in `src/`.

### D8. Memory & disk are launch *gates*
On a 16 GB unified-memory box, optimism causes OOM/swap. Preflight refuses launches
that exceed the usable-memory budget (override `--force`) and refuses downloads
that would cross the disk floor. These are conservative on purpose.

### D9. launch_overrides are appended as flags
Discovered during Phase 3 verification: the research agent returns overrides like
`{"--max-tokens":"2048"}` (i.e. *flags*). The original code treated overrides as
template *placeholder values* and silently dropped them. They are now appended as
real CLI flags. If you reintroduce placeholder-style overrides, support both
explicitly.

### D10. Backend selection = registered → preference → format-compatible
A model can pin a `backend`. Otherwise spark walks `general.preference`
(MLX-first on this host) and picks the first available, format-compatible runtime.
Vision models (`model_format="mlx-vlm"`) therefore route to `mlx_vlm` even though
`mlx_lm` is higher preference, because `mlx_lm` isn't format-compatible.

### D11. Research is validated against the installed server at accept time
An LLM invents flags (`--max-kv-size` recommended for `mlx_lm`, where it does
not exist — the failure was a cross-backend leak from `mlx_vlm`) and writes
non-canonical quant tags (`2bit`, which silently disables the memory budget).
`review --accept` / `config import` now probe the recommended backend's real
`--help` and refuse unknown flags, and normalize `Nbit` → `qN` (GGUF tags pass
through untouched). An unprobable surface (absent binary, hanging `--help`) is
a warning, never a refusal — the operator's explicit accept outranks a check
spark could not run. New backends are covered automatically; the check reads
the live binary, not a hardcoded list.

### D12. prism_ml serves Hadamard packs through a shim, not a fork
PrismML's v2 packs need the vendor's `runtime/artifact.py` plus a schema-2
loader (multimodal namespace, string dtypes mlx≥0.32 rejects) that no stock
server has. Rather than forking mlx_lm, spark runs a stdlib-HTTP shim
(`shims/prism_shim.py`, OpenAI-compatible) on the mlx-lm tool interpreter as a
supervised child — D1 intact (subprocess, no new spark deps). The `prism`
model format exists so these packs route to `prism_ml` and can never fall
through to stock `mlx_lm`, which rejects them at load. Text tower only: the
vision weights are skipped with a logged note, and the published API id is the
registry id (the shim ignores the request `model` field).

### D13. mlx_lm's listing is filtered by a sidecar, not by flags
`mlx_lm.server` has no `--model-discovery` switch (`mlx_vlm` does — its template
sets `served`), so its `/v1/models` always reflects the hub-cache scan. The
filter proxy exists for exactly this gap: stdlib HTTP, binds the published port,
spawns the real server as its child on an ephemeral port, filters only
`GET /v1/models` (served id + registered ∩ present mlx-servable ids), and exits
when the backend dies so the supervisor restarts the unit — child reaped first,
never orphaned. Opt-in per backend (`[server] filter_models`, default off);
generation is never gated, only the listing. The proxy **streams** (it never
buffers a response: `stream: true` is SSE, and buffering turns time-to-first-token
into total time) and binds its port before the backend loads, answering 503 until
it is healthy.

### D14. Ids are path components; one rule, enforced at the schema
A model id names four files (registry TOML, staging JSON, instance manifest,
server log), so `ModelEntry.id` is validated in the schema: `[A-Za-z0-9][\w.-]*`,
no `..`. Registry ids are one component. Ephemeral ids for unregistered hub
repos may be `owner/name`, and `id_file_stem()` maps them to `owner--name` (the
HF-cache convention) wherever they become a filename. `launch_overrides` keys
must be `-`-prefixed flags (D9 appends them to argv), and the accept-time check
enforces that even when the `--help` probe cannot run. HF repo ids are validated
before they reach `hf download` argv or the HF API URL (a leading `-` would
otherwise be read as an `hf` option).

### D15. Tuned flags belong to their backend
`launch_overrides` are researched and validated for the entry's pinned backend.
Launching a different one (pinned runtime uninstalled, or `--backend`) drops
them with a printed note instead of handing foreign flags to a server that
would die on argparse through every restart (`entry_for_backend`).
`spark download` of an already-registered repo keeps the tuned entry, and a
different repo whose slug collides with an existing id is refused.

### D16. One live server per alias
Instance records are keyed by alias, and a second server for the same model
doubles resident memory on a 16 GB box. So preflight refuses a launch when a live
record (pid alive) exists for the alias, and names the endpoint already serving.
`--force` overrides, and the heartbeat then re-asserts the survivor's record.
Records from a SIGKILLed spark (dead pid) are not published, and a spark removes
only a record it wrote itself.

### D17. Research agents run without tools
The research prompt embeds a third-party model card, which is untrusted text.
`claude` runs with `--tools "" --strict-mcp-config`, `pi` with `--no-tools`, and
`hermes` with `-t clarify` (its only tool is ask-the-user, which oneshot `-z`
mode strips, leaving zero tools; checked against hermes v0.23.1's own tool
resolver). A prompt-injected card has nothing to act with. The prompt fences the card as data.
The agent subprocess env drops every var that carries a vault secret to a runtime
(`HF_TOKEN` …), matching what the prompt guard enforces on the text. The prism shim renders the snapshot's `chat_template.jinja` in
jinja2's `ImmutableSandboxedEnvironment` (as HF transformers does). A plain
`Environment` gives a hostile repo SSTI, i.e. code execution.

### D18. Restart budget is per incident
`max_restarts` bounds one incident, not the session: a crash after
`supervisor.stable_after_s` (300 s) of healthy uptime gets a fresh budget and
backoff. Without that reset, a server that crashes once a day is permanently out
of restarts by day four. `restarts` in the exit event is the session total. Any exit
path out of `Supervisor.run`, including an unanticipated exception, terminates
the child (no orphaned server holding port and RAM).

### D19. Hosted providers are a proxy, not a relay
`relay` hands the consumer the upstream URL, so the consumer would also need the
key. Providers instead run a supervised loopback proxy that holds the key, which
gets them the same lifecycle, manifest, catalog, and restart policy as a local
server. Health stays local (`/health` on the proxy) so the 5 s probe never bills
the API. A rejected key is checked once, before spawn: no restart can fix it.

### D20. ChatGPT plan usage goes through SIWC, never Codex's private backend
Two routes spend a ChatGPT subscription.

**Codex CLI's login against `chatgpt.com/backend-api/codex`** is what most
community proxies copy. spark refuses it:
- OpenAI's docs say "do not point it at ChatGPT's `backend-api` endpoints";
- it is undocumented, and its tokens carry connector scopes;
- spark refreshing them would race Codex CLI's own single-use refresh tokens.

**"Sign in with ChatGPT" token sharing** is documented. It is offered for
"personal projects that run locally", sends inference to the public
`api.openai.com/v1/responses`, and uses its own issued client id. spark uses it.

Its preview rules shape the translation:
- Responses only, so Chat Completions are translated;
- no system role, so system messages become `instructions`;
- no sampling params, so they are dropped;
- `stream: true` is mandatory, so the stream is collected.

The refresh token lives in the supervisor, not in the proxy that parses
untrusted loopback HTTP. The proxy gets ≤1 h access tokens from a token broker
over an inherited socketpair (§7). An earlier build let the child read the
Keychain; that was replaced, because it put a 30-day credential in the most
exposed process.

The ID token's signature is not verified. OIDC Core §3.1.3.7(6) allows TLS
validation in place of the signature for a token taken directly from the token
endpoint, and verifying it would mean adding an RSA dependency.

---

## 11. Failure handling & logging contract

**Errors** (`errors.py`): every failure is a `SparkError` subclass with a stable
`code`, a `remediation` list, and non-secret `context`. The CLI renders them as
`{what / why / do-this-next}` (`render.py:render_error`) instead of a traceback.
Add new failure modes as typed errors with remediation — don't raise bare
exceptions across module boundaries. Anything untyped that still escapes is
caught in `main()`: logged as `unexpected_error` (type, message, file, line,
`context_window`), rendered as `SPARK_INTERNAL`, exit 70. `SPARK_DEBUG=1` adds the
traceback.

**Logging** (`telemetry/jsonl.py`) implements the repo-wide mandatory contract:
JSONL to `logs/<component>/<YYYY-MM-DD>.jsonl`, one object per line, with required
fields `ts, level, component, event, session_id, pid`. A payload key that
collides with one of these is renamed `payload_<key>` rather than overwriting it
(the child's pid is logged as `child_pid`). Daily rotation; retention from
`general.log_retention_days` (default 7). Secret redaction masks sensitive key
names (`hf_token`, `api_key` …, but not counts like `prompt_tokens`) and every
literal vault value; a ring buffer of recent events attached as
`context_window` on crash events. Logging must never crash the program (IO errors
are swallowed). A well-logged failure should be diagnosable from the file alone —
e.g. supervisor logs `weights_check` (availability state + location),
`disk_check` (with `will_fetch`/`need_bytes`), `process_start`,
`server_ready` (with port/endpoint), `warmup` (the generation-verified readiness
result), `process_crash`, `restart_scheduled`, `process_exit` (wall time, peak RSS).

---

## 12. Testing

`uv run pytest` — ~670 tests, no live LLM/Keychain/network required for the suite (the
Keychain roundtrip test self-skips off macOS; research uses fake providers and a
real-subprocess JSON test that needs no model). Coverage spans: config
merge/precedence, secret redaction + name validation, Keychain roundtrip, probe
parsing/gating, backend launch-cmd construction (incl. llama `-hf` vs local gguf,
override flags), backend selection/routing, registry fuzzy resolution, supervisor
port-pick/backoff/attach/prepare, budget math, download inference, research guard
+ JSON extraction + chain fallback + staging, and the hosted-provider proxy
end to end against loopback fake upstreams (plain and self-signed TLS):
key injection, browser-abuse guards, SSE streaming, stall timeout, redirects,
supervisor launch and crash-restart (`tests/test_provider_proxy.py`).
The ChatGPT-plan provider is tested against a fake OpenAI that implements the
documented protocol (`tests/test_chatgpt_provider.py`). The suite covers:
- PKCE-checked code exchange, registration vs re-login, and forged
  state/nonce/issuer;
- eligibility;
- refresh rotation, `refresh_token_reused`, a 6-thread refresh race that must
  spend the token once, and `earliest_refresh_at`;
- the token broker: the protocol, that no reply carries a refresh token, a
  static guard that the child's code cannot reach the Keychain, and a real
  supervisor-spawned child that is served and crash-restarted through the broker
  with no token in its environment;
- logout/revoke;
- Symbiote's exact request round trip, local `/v1/models` within 500 ms, 429
  pass-through, in-stream failure mapping, and the 401 refresh-retry.

Live verifications performed during development (not in the unit suite): real MLX
model served end-to-end (observed port-fallback + crash recovery), llama.cpp GGUF
via `-hf`, a real `claude` research run → stage → accept, and the provider key
check against the real OpenRouter/DeepSeek APIs (bogus key → `SEC_REJECTED`).

---

## 13. Install / run / develop

```bash
# Global command (recommended) — isolated env, launcher on PATH, editable:
uv tool install --editable ../spark --python 3.12
spark doctor                              # works anywhere, no venv

# After changing pyproject.toml dependencies, refresh the tool env:
uv tool install --editable ../spark --reinstall
uv tool list • uv tool upgrade spark • uv tool uninstall spark

# Dev workflow from the repo:
uv sync                                   # .venv pinned 3.12
uv run pytest
SPARK_HOME=/tmp/sparktest uv run spark …  # isolate config/data/logs for testing
```

`SPARK_HOME` is the test/isolation escape hatch; unset, spark uses
`~/.config/spark` + `~/.local/share/spark`.

Runtime and dev dependencies are **exact pins** (`==`) in `pyproject.toml` with a
matching `uv.lock`; bumping one is a deliberate, vetted change (`uv lock` first).
Ruff (E4/E7/E9/F/B, 100 columns) and pyright are configured in `pyproject.toml`;
run `ruff check src tests`, `ruff format --check src tests`, and
`pyright src/spark` before committing. There is no CI: this is a single-host
personal tool.

---

## 14. Roadmap & known gaps

- **Live-verify `mlx_vlm` / `omlx` / `ollama`** adapters (logic done + unit-tested;
  only `mlx_lm` and `llama_cpp` proven with a real served model).
- **ChatGPT plan provider (spec 0005): live sign-in + smoke** pending. Confirm
  `instructions` is accepted and a ~700-token reply lands under Symbiote's 15 s.
  Deferred: streaming Chat Completions translation, `tools`, ID-token signature
  verification, `id_token_hint` to skip the account picker.
- **Hosted providers: live smoke with real keys** pending (spec 0004 P6). Deferred
  by design: a local client token on the proxy (today any same-user process can
  spend the key), model aliasing, upstream connection pooling.
- **Distribution is local-only by decision**: the editable install *is* the
  deployment; spark is not packaged or published.
- **Store-backed models have no alias-shaped API id.** `mlx_lm.server` resolves
  the request `model` field itself, so a model served from the spark store can
  only be addressed by its absolute path; agent CLIs that validate model ids
  (pi, omp) silently drop such an entry from their pickers. The catalog publishes
  the working id per instance (that is what `model_id` is for) — the gap is that a
  *static* CLI config cannot express it.
- **Port drift is published, not prevented.** 8095 is preferred; if it is taken,
  spark walks the range, logs `port_drift`, and writes the real endpoint to
  `run/instances/`. A consumer that does not read that file still breaks.
- See `CONTINUE.md` for the current session handoff and `specs/` for tracked
  feature work.

---

*spark is a personal tool for a specific Apple-Silicon host. It assumes macOS for
secrets and prefers MLX for inference; the architecture is portable but the
defaults are tuned for this machine.*
