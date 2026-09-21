# spark

A hardened wrapper that fronts **every local LLM inference runtime** on this
machine behind one command. Type `spark`, fuzzy-tab through available models, hit
enter — the best-optimized runtime server for that model launches. `spark download
<hf-repo>` fetches a model and runs a multi-agent research chain to discover the
optimal per-runtime configuration for *this* host.

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
(`mlx_lm.server`, `mlx_vlm.server`, `omlx`, `llama-server`, `ollama`) as
subprocesses, choosing the right one per model and managing its lifecycle.

The single most important architectural consequence: **spark's own Python
environment never imports `mlx`, `vllm`, `torch`, or `llama_cpp`.** Its runtime
dependencies are four pure-Python packages (`click`, `pydantic`, `platformdirs`,
`rich`). This is deliberate — see [Decision D1](#d1-wrapper-not-reimplementation).

What spark provides on top of the raw runtimes:
- **One UX** over many runtimes, with fuzzy model completion.
- **Per-model, per-host optimal config** discovered by an agent research chain.
- **A hardened process supervisor** (preflight resource gates, health checks,
  bounded crash recovery, graceful shutdown).
- **A secrets vault** that keeps tokens out of files, argv, logs, and LLM prompts.
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
| Disk | ran ~95% full (~24 GiB free) | Disk headroom is a **first-class fail state** for downloads |

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
| `ollama` | ✅ installed | Convenience daemon |
| `vllm` | ❌ absent | Registered but **unavailable on Apple Silicon** by design |

---

## 3. User-facing model

```
spark                          # what this host can serve (disk truth, registry or not)
spark <model>                  # launch the best-optimized runtime server for <model>
spark download <hf-repo>       # fetch + research + register (stages config for review)
spark adopt <repo|path>        # register on-disk weights (hub cache or store) as-is
spark research <model>         # (re-)run the research chain for a registered model
spark config review <model> [--accept|--reject]   # apply/discard staged research
spark config import <model>    # import research JSON from stdin (manual path)
spark doctor                   # probe host + runtime availability + budgets
spark list [--all]             # runnable models (default) / everything (--all)
spark catalog [--json]         # the machine-readable roster (catalog.json)
spark forget <model>           # remove a registry entry (weights untouched)
spark secret set|ls|rm|get <name>                 # macOS Keychain vault
spark config validate|path|show                   # inspect configuration
spark completion zsh           # emit the zsh completion function
spark __complete models [context] | describe <model>   # hidden: completion data
```

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
`--force` to bypass memory/disk gates, `--reprobe` to re-detect the host first.

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
│   ├── secrets/                 # store.py (ABC + redaction), keychain.py (ctypes Security)
│   ├── runtimes/                # base.py (Backend ABC + registry) + per-runtime adapters
│   ├── shims/                     # standalone servers run by foreign interpreters
│   │                              #   (shim-only dir: a script's dir is on sys.path,
│   │                              #   so runtimes/mlx_lm.py would shadow real mlx_lm)
│   ├── probe/                   # host.py (capability detection + cached profile)
│   ├── registry/                # models.py (one TOML/model, fuzzy resolve)
│   ├── inventory.py             # disk-first join: store + hub cache vs registry
│   ├── runner/                  # supervisor.py (spawn/health/restart/attach/signals)
│   ├── research/                # types/guard/prompt/providers/chain/staging/validate
│   ├── telemetry/               # jsonl.py (structured logging + redaction)
│   └── errors.py                # typed errors carrying remediation
├── tests/                       # pytest (173 tests)
├── spikes/                      # time-boxed evaluation spikes + the portable
│                                #   human/emotion eval skill (spikes/README.md)
├── bake-offs/                   # comparative model bake-offs — quality evals,
│                                #   distinct from patchwork's throughput benches
└── MODEL-EVAL-2026-07-15.md     # cross-model capability matrix + tactical use/dismissal
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
  → backend.prepare(entry)               # e.g. ollama: ensure daemon + pull tag
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
disk preflight (the 24 GiB wall)
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
(probe cache) · `run/` · `secrets-index.json` (non-secret names only).

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
  `register_secret_value()` — the store registers every value it reads — and (b)
  sensitive key names (`token`, `secret`, `password`, `authorization`, …).
- **Never in an LLM prompt.** The research path never calls `secrets.get()`, and
  `research/guard.py:assert_no_secrets()` scans every outbound prompt for
  registered values + credential patterns (`hf_`, `sk-`, `AKIA`, `ghp_`, bearer)
  and aborts on a hit.
- **Only path a secret takes:** vault → child process env at spawn
  (`Supervisor._child_env` via `SecretStore.resolve_env`). The launch event logs
  only which env var names were set, never values.

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
  dir or an HF repo id. Primary text path. Ports default 8080.
- **`mlx_vlm`** — `mlx_vlm.server`; vision. Its binary defaults to host `0.0.0.0`,
  so the template **pins `--host 127.0.0.1`** (no LAN exposure). Port 8081.
- **`llama_cpp`** — resolves to a local `.gguf` file (`--model`, picks the
  `00001-of` shard for splits) **or** an HF repo (`-hf user/repo[:quant]`, which
  `llama-server` downloads itself). Static Metal flags from TOML: `-ngl 999 -fa
  auto --jinja`. Health is `/health` (not `/v1/models`). Port 8082.
- **`omlx`** — `omlx serve <model> --host --port --memory-guard safe`. Port 8083.
- **`ollama`** — daemon-style. `attach_if_running=true`: if the daemon is already
  healthy, **attach (never spawn/kill the shared daemon)**. `prepare()` requires
  the daemon (`/api/tags`) and `ollama pull`s the tag if missing. OpenAI endpoint
  at `:11434/v1`. Model is addressed by *tag*, not path.
- **`vllm`** — registered with `requires_os=["Linux"]` + `unavailable_reason`; on
  this host `is_available()=False` with actionable guidance. Kept so the config
  travels unchanged to a CUDA host.

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
graceful `terminate` (SIGTERM, `shutdown_grace 10s`, then SIGKILL). Child inherits
stdout/stderr so the operator sees server logs live. **Attach mode** (daemons):
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
- **`hermes`** (`~/.local/bin/hermes`) and **`pi`** (`~/.bun/bin/pi`) are present
  but their command templates are **best-guess** (`-p` + stdin, `parse="raw_json"`).
  If you wire them for real, confirm each CLI's actual flags and JSON behavior and
  fix the TOML — don't assume.
- Running `claude` from inside spark spawns a **nested** Claude Code session (real
  cost + network). `spark download` runs it by default; `--no-research` opts out.

### 9.7 Networking posture
- Servers bind `general.host = 127.0.0.1` (loopback only) by default. Don't change
  this to `0.0.0.0` without intent — `mlx_vlm.server` defaults to `0.0.0.0` and is
  explicitly pinned back to loopback in its TOML.
- Egress happens only in: `hf download`, llama `-hf`, the public model-card fetch,
  and the agent CLIs. Everything else (probe, launch, health) is local.

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
8095 (`general.port_range = [8095, 8099]`; llama.cpp 8096, oMLX 8097). A taken
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

---

## 11. Failure handling & logging contract

**Errors** (`errors.py`): every failure is a `SparkError` subclass with a stable
`code`, a `remediation` list, and non-secret `context`. The CLI renders them as
`{what / why / do-this-next}` (`render.py:render_error`) instead of a traceback.
Add new failure modes as typed errors with remediation — don't raise bare
exceptions across module boundaries.

**Logging** (`telemetry/jsonl.py`) implements the repo-wide mandatory contract:
JSONL to `logs/<component>/<YYYY-MM-DD>.jsonl`, one object per line, with required
fields `ts, level, component, event, session_id, pid`; daily rotation; 7-day
retention; secret redaction; a ring buffer of recent events attached as
`context_window` on crash events. Logging must never crash the program (IO errors
are swallowed). A well-logged failure should be diagnosable from the file alone —
e.g. supervisor logs `weights_check` (availability state + location),
`disk_check` (with `will_fetch`/`need_bytes`), `process_start`,
`server_ready` (with port/endpoint), `warmup` (the generation-verified readiness
result), `process_crash`, `restart_scheduled`, `process_exit` (wall time, peak RSS).

---

## 12. Testing

`uv run pytest` — 221 tests, no live LLM/Keychain required for the suite (the
Keychain roundtrip test self-skips off macOS; research uses fake providers and a
real-subprocess JSON test that needs no model). Coverage spans: config
merge/precedence, secret redaction + name validation, Keychain roundtrip, probe
parsing/gating, backend launch-cmd construction (incl. llama `-hf` vs local gguf,
override flags), backend selection/routing, registry fuzzy resolution, supervisor
port-pick/backoff/attach/prepare, budget math, download inference, research guard
+ JSON extraction + chain fallback + staging.

Live verifications performed during development (not in the unit suite): real MLX
model served end-to-end (observed port-fallback + crash recovery), llama.cpp GGUF
via `-hf`, and a real `claude` research run → stage → accept.

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

---

## 14. Roadmap & known gaps

- **Live-verify `mlx_vlm` / `omlx` / `ollama`** adapters (logic done + unit-tested;
  only `mlx_lm` and `llama_cpp` proven with a real served model).
- **`hermes` / `pi` provider templates** are best-guess; only `claude` is verified.
- **Distribution beyond this host**: editable install needs the repo present; a
  built wheel (`uv build`) would let spark run on machines without the checkout.
- **Store-backed models have no alias-shaped API id.** `mlx_lm.server` resolves
  the request `model` field itself, so a model served from the spark store can
  only be addressed by its absolute path; agent CLIs that validate model ids
  (pi, omp) silently drop such an entry from their pickers. The catalog publishes
  the working id per instance (that is what `model_id` is for) — the gap is that a
  *static* CLI config cannot express it.
- **Port drift is published, not prevented.** 8095 is preferred; if it is taken,
  spark walks the range, logs `port_drift`, and writes the real endpoint to
  `run/instances/`. A consumer that does not read that file still breaks.
- See `CONTINUE.md` for the current session handoff and `~/.claude/plans/
  parallel-wandering-lemur.md` for the original approved plan.

---

*spark is a personal tool for a specific Apple-Silicon host. It assumes macOS for
secrets and prefers MLX for inference; the architecture is portable but the
defaults are tuned for this machine.*
