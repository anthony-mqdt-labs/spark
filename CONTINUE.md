# CONTINUE.md — spark

Session handoff snapshot. Overwrite at session end / before compaction.

## 2026-07-14 — `spark doctor --fix` (auto-remedy missing runtime Python deps)

**Committed 30bb9f4**, tested (105 tests, was 96), live-verified. The mlx_vlm
torch/torchvision prerequisite is now self-healing:

- `spark doctor --fix` installs missing `[detect].python_requires` modules into
  the runtime's OWN tool env. The command is derived from config
  (`install_hint` + `python_requires`), shown before it runs, and only offered
  for `uv tool install` hints — spark never guesses at other installers
  (`dep_fix_command` in probe/host.py returns None for brew/curl hints).
- `python_requires` entries now support `module:pip-package` syntax for cases
  where import name ≠ PyPI name (e.g. `PIL:pillow`); mapping is always declared
  in TOML, never guessed (supply-chain rule).
- After install, doctor re-verifies via the runtime's interpreter; success →
  green ✓ + `python_deps_fix_succeeded`; installer failure or still-missing →
  ✗ + telemetry, and doctor exits 1 with `RT_DEPS_UNFIXED` remediation panel.
- Live-verified both paths in an isolated SPARK_HOME with a throwaway pycowsay
  tool env (installed, fixed with `--with six`, then a bogus package to prove
  the failure path; tool uninstalled after). Real mlx-vlm env untouched — it
  already has torch/torchvision, so doctor shows no warning on this host.
- Files: cli/doctor.py (flag + `_fix_python_deps`), probe/host.py
  (`split_python_req`, `dep_fix_command`, `missing_python_deps` returns raw
  reqs), config/schema.py + config/runtimes/mlx_vlm.toml comments,
  tests/test_doctor_fix.py (new), tests/test_probe.py (+5).

## 2026-07-14 — doctor versions column fixed (was mostly "—")

**Committed 1bc9795**, tested (109 tests), live-verified: all 5 available runtimes now
show a version in `spark doctor`. Three root causes, three fixes (probe/host.py):

1. **mlx_lm/mlx_vlm have no --version** (argparse usage banner only). New
   `[detect].version_package` config field: read the dist version via
   `importlib.metadata` in the runtime's OWN interpreter (`_version_from_package`;
   reuses `_interpreter_for`). Fast — never imports mlx. Tried before
   version_args when set. Declared in mlx_lm.toml + mlx_vlm.toml.
2. **ollama with daemon down** prints only `Warning: client version is 0.32.0`,
   which the banner filter skipped. `_detect_version` now does a second pass
   over skipped banner lines accepting a strictly numeric dotted token only
   (never the loose `version: <word>`).
3. **Stale cache**: llama_cpp detection worked live, but the day-long host-profile
   cache served old empty results, and nothing busted it on code changes. Added
   `_PROBE_SCHEMA_VERSION` to the cache fingerprint — bump it whenever detection
   logic changes shape/behavior.

## Triage 2026-07-13 — RESOLVED same day: all 4 bugs fixed; download blocked by HF, not spark

All four triage bugs fixed, tested (96 tests, was 79), live-verified.
**Committed 93db25d** (2026-07-14).

1. **Telemetry gap — FIXED.** `_run_fetch` (cli/download.py) guarantees exactly
   one terminal event per fetch: `fetch_complete` (bytes/wall_s/rate) /
   `fetch_failed` (rc or error_type) / `fetch_interrupted` (SIGINT caught;
   SIGTERM/SIGHUP via temporary handler → log → exit 128+n). Live-verified
   twice by SIGTERMing real wedged downloads.
2. **Partial downloads visible — FIXED.** New `registry/store.py::scan_store`;
   `spark list` + `spark doctor` flag partial/orphaned store dirs with
   resume/clean hints (`render_store_issues` in cli/render.py).
3. **Disk preflight — FIXED.** Projected-size gate: repo size from the HF API
   (stdlib urllib, token optional), resume-bytes credit, `free − need ≥ floor`
   via `budget.check_disk`; `--force` bypass (logs `disk_gate_forced`); loud
   floor-only fallback when size unknown. Live-verified: correctly refused the
   5.57 GiB Ornith fetch at 10.2 GiB free / 10 GiB floor (exit 75, RUN_DISK).
4. **June silent deaths — DIAGNOSED, two layers.** (a) The silence: Ctrl-C path
   had no telemetry (bug 1). LuLu exonerated — a denial would have exited
   non-zero, the one path that *was* logged. (b) The actual killer, reproduced
   live today: **HF's Xet CAS bridge denies large-blob transfers from the
   Proton VPN shared exit IP** — 403 AccessDenied on any repo, with or without
   a valid token, while API + small files work. With hf_xet installed the
   client doesn't error, it retry-wedges at a deterministic byte offset
   (2,677,991,959 twice today). `HF_HUB_DISABLE_XET=1` surfaces the honest 403.
   Wiki: `bugs/spark-infra-hf-cas-bridge-403-stall-001.md` and
   `bugs/spark-python-silent-download-death-001.md`. Also learned:
   `*.incomplete` fragments are never reused across runs (random per-run
   suffix) — every retry restarts the blob from 0; stale fragments are dead
   weight.

**Ornith download status:** operator approved `--force`; 3 attempts made.
`store/ornith-1.0-9b-4bit/` has shard 2 complete (600 MB) + all metadata;
shard 1 (5.35 GB) blocked by the 403 above. **Not registered** (registration
only follows a successful fetch); shows in `spark list` as an orphaned store
dir, as designed. Unblock options (operator calls): change VPN exit / toggle
VPN, or wait out the IP block, then rerun
`spark download mlx-community/Ornith-1.0-9B-4bit` (disk now ~21 GiB free —
gate passes without --force). Consider `HF_HUB_DISABLE_XET=1` so a re-block
fails fast instead of wedging. June's orphan
(`store/ornith-1.0-9b-4bit-mtp-mlx-serve/`, 203 MB, superseded giaki3003 repo)
still present — safe to `rm -rf` once confirmed unwanted.

**Context:** quorum (`../quorum`) needs spark healthy for its Seam #1
(real inference for its society of minds). Note for that use: a 9B model is
likely wrong-sized anyway — quorum wants several *small* minds (0.5–3B) on this
16 GB M2, not one large one.

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
      torchvision 0.27.1). **Now surfaced in `spark doctor`** (commit 09c9b1c):
      runtimes declare `[detect].python_requires` in their TOML; doctor resolves the
      runtime's OWN interpreter (console-script shebang) and find_spec-checks the
      modules there, printing the `--with` fix if any are missing. Data-driven, so
      adding a dep check to another runtime is a one-line TOML edit, no code.
      **2026-07-14: `spark doctor --fix` now applies that fix automatically** (see
      top of this file).
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
- ~~**Distribution beyond this host**~~ — **DECIDED: NO (2026-06-28).** spark is
  local-only; never package/publish. The editable install (`uv tool install
  --editable`) + repo checkout IS the deployment model. Don't propose wheels/PyPI.
  Only reinstall trigger is a pyproject dep change (`--reinstall`).
- ~~**Git**: no commits~~ — DONE. 17 commits on `main`, clean history (scaffold →
  per-module → fixes). No remote; nothing pushed (intentional, local-only).
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
- mlx_lm/mlx_vlm have no `--version`; probe reads their dist version from the
  runtime's own interpreter via `[detect].version_package` (2026-07-14).
