# CONTINUE.md — spark

Session handoff snapshot. Overwrite at session end / before compaction.

## What spark is
A hardened wrapper that fronts every local LLM inference runtime behind one
`spark` command (Apple Silicon / M2, 16 GB). Wrapper-only: orchestrates installed
runtime CLIs via subprocess; never imports mlx/vllm. Plan:
`~/.claude/plans/parallel-wandering-lemur.md`.

## Status — Phases 1, 2, 3 COMPLETE

### Phase 3 (download research chain) — done, verified
- **Multi-agent fallback chain** (`src/spark/research/`): ordered providers
  (claude → hermes → pi) tried until one returns schema-valid JSON. Providers are
  config-driven (`[research]` in defaults.toml; `ResearchProviderDef`), CLI-based,
  auto-skipped if their detect_binary is absent. All three present on this host.
- **Secrets-guard** (`research/guard.py`): scans every outbound prompt for
  registered secret values + credential patterns (hf_/sk-/AKIA/ghp_/bearer); the
  code path never calls `secrets.get()`. Prompt built from public model card +
  host facts + runtime names only.
- **Staging + review**: results go to `data/staging/<id>.json`; `spark config
  review <id> [--accept|--reject]` applies/discards; `spark config import <id>`
  for the manual path. `apply_output_to_entry` sets backend/quant/context/overrides
  + status=registered.
- **launch_overrides are appended as real CLI flags** (was a placeholder-only bug;
  fixed in `runtimes/base.py:override_flags`, applied in base + llama adapters).
- **CLI**: `spark research <model>`; download runs the chain by default
  (`--no-research` to skip; non-fatal on failure). Completion updated.
- **Live e2e**: real `claude` agent researched the qwen MLX model → recommended
  mlx_lm/q4/32K with per-runtime rationale → staged → accepted → launch cmd shows
  `--max-tokens 2048`. No secrets in research logs. 76 tests pass.

## Status — Phase 1 + Phase 2 COMPLETE

### Phase 2 (live backends + fuzzy completion) — done, verified
- **Adapters** (`src/spark/runtimes/`): `llama_cpp` (local .gguf OR `-hf` repo +
  Metal -ngl/-fa/--jinja), `mlx_vlm`, `omlx` (serve positional), `ollama`
  (prepare()=daemon-check + pull, attach-mode). All registered.
- **Backend hooks**: `ServerSpec.attach_if_running`, `Backend.prepare()`,
  `Backend.openai_base_url()`; supervisor attach-mode (never spawns/kills a shared
  daemon) + calls prepare() before launch.
- **Routing**: vision models (model_format `mlx-vlm`) route to `mlx_vlm`;
  download inference detects vlm/quant-bit-style/params.
- **fzf-tab installed + wired**: cloned to `~/.config/zsh/plugins/fzf-tab`,
  `_spark` in `~/.config/zsh/completions`, all spark shell config in
  `~/.config/zsh/spark.zsh` (sourced from `~/.zshrc` via one appended line).
  Registered via explicit `compdef _spark spark` (cache-proof). `$_comps[spark]`
  confirmed. Preview pane = `spark __complete describe`.
- **Live e2e**: llama.cpp served `Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M` via `-hf` on
  port 8082, real chat completion, graceful shutdown. 60 tests pass.

## Status — Phase 1 (walking skeleton) COMPLETE
Delivered and tested end to end:
- **Scaffold**: uv project, venv pinned 3.12.13, src layout, `spark` console script,
  lean pinned deps (click, pydantic, platformdirs, rich), `git init` done (no commits).
- **Config** (`src/spark/config/`): layered TOML loader (defaults → runtimes/*.toml →
  user override), pydantic schema with `extra=forbid`. Shipped `config/` tree.
- **Secrets** (`src/spark/secrets/`): macOS Keychain via **ctypes Security framework**
  (no argv/file/log/LLM exposure) + non-secret names index. Real roundtrip verified.
- **Telemetry** (`src/spark/telemetry/`): JSONL, required fields, rotation+retention,
  secret redaction (literal values + sensitive key names).
- **Probe** (`src/spark/probe/host.py`): host + per-runtime detection, cached host.toml.
- **Runtimes** (`src/spark/runtimes/`): Backend ABC + registry + generic launcher;
  `mlx_lm` adapter. 6 runtime defs (mlx_lm/mlx_vlm/llama_cpp/omlx/ollama available;
  vllm registered-unavailable).
- **Registry** (`src/spark/registry/models.py`): one TOML/model, fuzzy resolve.
- **Supervisor** (`src/spark/runner/supervisor.py`): preflight (mem/disk/port gates),
  secret env injection, health-wait, bounded restart+backoff, graceful SIGINT.
- **CLI** (`src/spark/cli/`): `spark`, `spark <model>`, run/download/doctor/list/
  secret/config/completion + hidden `__complete`. fzf-tab-ready zsh completion.
- **Tests**: `tests/` — 48 passing (`uv run pytest`).

## Verification done
- `spark doctor` → correct M2/16GB/disk + runtime table.
- Keychain set/ls/get/rm via CLI + real roundtrip test.
- `spark download --no-fetch` registers; `spark list`; completion data emit.
- E2E: real MLX model served via `mlx_lm.server`, OpenAI endpoint hit, graceful
  shutdown. (See session log.)

## Global install — DONE
`spark` is a global command via `uv tool install --editable .` → launcher at
`~/.local/bin/spark` (on PATH). **Editable**: repo source edits take effect with no
reinstall. Globally it uses real dirs (`~/.config/spark`, `~/.local/share/spark`)
with bundled config from the repo. The global registry starts empty (test models
lived in a throwaway `SPARK_HOME`).
- Refresh after pyproject dep changes: `uv tool install --editable ../spark --reinstall`
- `uv tool list` · `uv tool upgrade spark` · `uv tool uninstall spark`

## Optional loose ends (none blocking)
- **Live-verify adapters — ALL DONE.** mlx_lm, llama_cpp, **ollama**, **mlx_vlm**,
  and **omlx** all proven with a real served model + graceful shutdown.
  - **omlx — DONE + REWORKED (2026-06-28).** Live-verify caught a real bug: spark's
    template assumed `omlx serve <positional repo>`, but omlx 0.4.4 dropped that —
    `serve` now only discovers models from subdirs of `--model-dir`. Reworked the
    adapter (`runtimes/omlx.py`): resolves local weights (`entry.path` from
    `spark download`, else the HF hub-cache snapshot), stages a private one-model
    dir under `run_dir/omlx/<id>/` with a symlink named `<id>`, and serves via
    `--model-dir`. Added a `{model_dir}` placeholder + default `resolve_model_dir`
    in `base.py`; updated `omlx.toml`; +3 tests (79 total). Verified: HF-cache
    snapshot resolution → staged symlink → `omlx serve --model-dir …` → model served
    as clean id `qwen05` → real completion (`PONG!`) → graceful SIGINT/reap. NB: omlx
    `serve` does NOT auto-fetch — weights must be local first (adapter aborts with
    remediation otherwise); the bounded-restart path was also exercised (the old
    broken template surfaced `RUN_RECOVERY_EXHAUSTED` after 4 attempts).
  - **mlx_vlm — DONE (live-verified 2026-06-28)**. Registered an isolated SPARK_HOME
    entry (`backend=mlx_vlm`, `model_format=mlx-vlm`,
    `hf_repo=mlx-community/Qwen2-VL-2B-Instruct-4bit`, ~1.2 GB), ran `spark run`.
    Verified the **spawn** path (vs ollama attach): memory gate (est 1.1 GiB vs
    10.4 budget) → disk gate → `secret_env_resolved` (no hf_token, skipped, public
    model) → `process_start` → `server_ready` on :8081 → real **vision** completion
    (red PNG in → text out, 127 tok/s, peak 1.32 GiB) → graceful SIGINT
    (`server_ready → signal_received → shutdown_initiated → process_exit`, child
    server reaped). No spark code change needed.
    - **PREREQUISITE (environment, not spark):** mlx_vlm's Qwen2-VL processor needs
      **PyTorch + Torchvision**, absent from a bare `uv tool install mlx-vlm`. Without
      them the server starts but model-load 500s ("Qwen2VLVideoProcessor requires the
      Torchvision library"). Fixed once on this host:
      `uv tool install mlx-vlm --with torch --with torchvision` (torch 2.12.1,
      torchvision 0.27.1). Worth surfacing in `spark doctor` as a mlx_vlm readiness
      check.
  - **ollama — DONE (live-verified 2026-06-28)**. Registered an isolated SPARK_HOME
    entry (`backend=ollama`, `launch_overrides.ollama_tag=qwen3.5:0.8b-mlx`, the
    smallest already-cached tag), ran `spark run`. Verified: preflight (disk gate) →
    `prepare()` daemon health check + tag-present (no pull) → **attach-mode** to the
    running daemon at `:11434/v1` (never spawned/killed it) → real OpenAI chat
    completion (`PONG`) → graceful SIGINT (`attached → signal_received → detached`).
    No code change needed. NB: qwen3.5:0.8b-mlx is a *thinking* model — empty
    `content` at low `max_tokens` (burns the budget on `reasoning`); needs ~512 tok.
    NB: spark telemetry lands in `$SPARK_HOME/data/logs/{cli,probe,supervisor}/`.
- ~~**hermes/pi provider command templates**~~ — DONE (commit 29a8717). Corrected to
  each CLI's real one-shot interface: hermes `-z/--oneshot {prompt}` (live-verified,
  clean JSON, exit 0); pi `-p/--print {prompt}` text mode + raw_json. Both now
  `prompt_via=arg`. Note: pi's configured backend is local **ollama** (`defaultProvider`
  in `~/.pi/agent/settings.json`), currently down — pi research only works when that
  daemon is up; spark owns pi's flags, not pi's backend auth.
- **Distribution beyond this host**: `uv build` a wheel / publish if spark should run
  on machines without the repo checked out (editable install needs the repo present).
- **Git**: repo initialized, no commits yet (awaiting operator go-ahead).
- **Tune `mlx_lm.server` chat-template / sampling defaults** per-model via research
  `launch_overrides` once more models are registered.

## Run it
```
# Global (anywhere, no venv):
spark doctor
spark download <hf-repo>            # fetch + agent research → stage for review
spark <model>                      # launch best-optimized runtime
spark config review <model> --accept

# Dev (from the repo):
cd ../spark
uv run pytest                      # 76 tests
SPARK_HOME=<dir> uv run spark ...  # isolate config/data/logs for testing
```

## Gotchas
- System Python is 3.14 (ML wheels lag) — spark venv is pinned 3.12; spark never
  imports mlx, so this never bites.
- Keychain uses deprecated-but-functional SecKeychain* APIs (the `security` CLI uses
  them too). If a future macOS removes them, migrate to SecItem* (CFDictionary).
- mlx_lm/mlx_vlm have no `--version`; probe extracts a version token or shows none.
