# CONTINUE.md — spark

Handoff for an agent resuming **unfinished** work. Completed history is not kept
here: `git log`, `README.md`, and `../bugs/` are the record. Anything that is done
gets deleted from this file rather than annotated. Overwrite at session end.

**State, 2026-11-26.** `main` is the only branch. The hardening passes and spec
0004 (hosted providers: OpenRouter and DeepSeek behind a key-injecting loopback
proxy) are all merged into `main`. 626 tests pass.

## Do next — concrete, unblocked

1. **Live smoke test of the hosted providers (spec 0004, P6).** It needs the
   operator's keys:
   - `spark secret set openrouter_api_key` and `spark secret set deepseek_api_key`
   - then `spark run openrouter` (port 8100) and `spark run deepseek` (port 8101)
   - then one `curl …/v1/chat/completions` each.

   The registry entries already exist in
   `~/.local/share/spark/models/{openrouter,deepseek}.toml`. On success, set
   `specs/0004-*/status.yaml` `remaining: []`.

## Awaiting an operator decision

- **Four entries assert weights that are gone:** `minicpm5-2b-8bit`,
  `ornith-1.5-9b-mlx-6bit`, `ternary-bonsai-2-27b-mlx-2bit`, and
  `ternary-bonsai-8b-mlx-2bit` (`spark list --all`, checked 2026-11-26). That
  leaves only darkcore and the two providers runnable. `spark list --all` prints
  the fetch and forget command for each. Re-fetching or dropping them is the
  operator's call, not an agent's. Banter's persisted endpoint names
  `minicpm5-2b-8bit`, so that model's fate tracks banter's.
- **pi/omp cannot see store-backed models.** Both drop an id that is an absolute
  path. The durable workaround is in place: `catalog.json` publishes the working
  id per live instance. Open question: move such models to the hub cache, or
  leave them.
- **DFlash is wired but parked.** It is an opt-in profile and a net loss at 4-bit
  on this 16 GB box. Worth revisiting on a ≥24 GB machine.
- **Hosted-provider follow-ups, deferred by design** (spec 0004 §7):
  - a local client token on the proxy;
  - model aliasing;
  - upstream connection pooling.

## Deliberately gated — do not implement without the operator

- **spec 0001 `model-fleet-api`** (`wip-research`): sparkd as the machine's model
  fleet authority. Spikes S1–S6 plus four open questions gate any `design.md`.
- **spec 0002 `afm-on-device-runtime`** (`wip-research`): Apple Foundation Models
  as a supervised runtime. Spikes A1–A4 gate it.
- **spec 0003 `sdcpp-image-runtime`** (`wip-research`): proposal only. Spikes
  S1–S6 gate it.

## Environment blockers

- **The ollama daemon is down** (`curl 127.0.0.1:11434/api/version` fails,
  2026-11-26), so the `pi` research provider is unavailable. claude and hermes
  still serve the research chain.
- **`darkcore` needs `$DARKCORE_ROUTER_DIR`** set to the router's project root.
  `spark doctor` reports it unavailable until then.

## Adjacent repos

- **banter:** spec 0028 is still `in-flight`, and its
  recorded follow-up stands: the Rust `backend-spark` adapter does not yet read the
  persisted endpoint record (`python/.banter-endpoint.json`). It keeps its
  environment contract and the 8095 default. See
  `banter/specs/0028-*/{status.yaml,runbook.md}`.

## Orientation — only what a resuming agent needs

- spark supervises already-installed local runtimes, and proxies hosted
  OpenAI-compatible APIs through `shims/provider_proxy.py`. It serves no
  inference itself.
- `README.md` is design truth:
  - §3 has the operator workflows, including hosted providers;
  - §5 is the availability contract;
  - §9 lists the external integrations that break silently;
  - §10 is the decision log (D19 covers providers).
- Hermes skills (not Claude Code ones) carry some hard-won detail:
  `mlx-inference` and `tui-keybinding-debugging` in `~/.hermes/profiles/dev/skills/`.
  `spark-operations` survives only in an archive (penny profile) and may be stale.
- Tests: `cd ~/Repos/vibes/spark && .venv/bin/python -m pytest`. Isolate with
  `SPARK_HOME=<dir>`. The editable install makes source edits live. Only a
  pyproject dependency change needs `uv tool install --editable ../spark --reinstall`.
- Ports:
  - Consumers point at 8095: nine Hermes configs (`mlx-local`), the pi/omp/opencode
    providers, and banter's env plus its Rust default.
  - Hosted providers prefer 8100 (OpenRouter) and 8101 (DeepSeek), outside the
    local range [8095, 8099].
