# CONTINUE.md — spark

Handoff for an agent resuming **unfinished** work. Completed history is not kept
here: `git log`, `README.md`, and `../bugs/` are the record. Anything that is done
gets deleted from this file rather than annotated. Overwrite at session end.

**State, 2026-09-18.** `main` is clean, one commit ahead of origin (`f4e30bb`,
completion policy). Nothing running; 8095 free. 173 tests pass; two pre-existing
F401s stand in `runtimes/base.py` and `runtimes/ollama.py`.

## Do next — concrete, unblocked

1. **Pin `minicpm5-2b-8bit`'s backend.** Its `research_status` is `pending`, so
   the registry carries no backend and `spark run` picks one at launch. A staged
   result (claude, 2026-09-18) recommends mlx_lm / q8 / ctx 32768, no overrides:
   `spark config review minicpm5-2b-8bit --accept`. Worth doing now — this is the
   model banter's persisted endpoint names.
2. **Reject the stale `ornith-1.0-9b-4bit` staged result.** It recommends
   mlx_vlm, but the entry is deliberately `manual`/mlx_lm (inference-bench #002:
   18.2 tok/s, and the bench chose it for this host), with DFlash opt-in via
   `spark run ornith --dflash`. Accepting it would regress that call:
   `spark config review ornith-1.0-9b-4bit --reject`.
3. **Push `f4e30bb`** on the operator's word.

## Awaiting an operator decision

- **Four entries assert weights that are gone** — `minicpm5-1b-8bit`,
  `ornith-1.0-9b-4bit`, `qwen3.5-0.8b-optiq-4bit`, `ternary-bonsai-27b-mlx-2bit`.
  `spark list` names the fetch and forget command for each. Re-fetch or drop is
  the operator's call, not an agent's. `minicpm5-1b-8bit` is coupled to banter's
  default model, so its fate tracks banter's.
- **pi/omp cannot see store-backed models.** Both drop an id that is an absolute
  path, so a store model (`minicpm5-2b-8bit`) never appears in their pickers while
  hub-cache ids do. The durable workaround is in place — `catalog.json` publishes
  the working id per live instance. Open question: move that model to the hub
  cache, or leave it.
- **DFlash is wired but parked.** Opt-in profile; a net loss at 4-bit on this
  17 GB box, worth revisiting on a ≥24 GB machine.

## Deliberately gated — do not implement without the operator

- **spec 0001 `model-fleet-api`** (`wip-research`): sparkd as the machine's model
  fleet authority. Spikes S1–S6 plus four open questions gate any `design.md`.
- **spec 0002 `afm-on-device-runtime`** (`wip-research`): Apple Foundation Models
  as a supervised runtime. Spikes A1–A4 gate it.

## Environment blockers

- **ollama daemon is down** (`curl 127.0.0.1:11434/api/version` fails), so the
  `pi` research provider is unavailable; claude and hermes still serve the chain.

## Adjacent repos touched by the same work

- **banter:** four commits unpushed (`8825dd8`, `824b0c8`, `ffc9cda`, `df0d765`);
  spec 0028 is `in-flight`. Its recorded follow-up: the Rust `backend-spark`
  adapter does not yet read the persisted endpoint record
  (`python/.banter-endpoint.json`) — it keeps its environment contract and the
  8095 default. See `banter/specs/0028-*/{status.yaml,runbook.md}`.

## Orientation — only what a resuming agent needs

- spark supervises already-installed local runtimes and serves no inference
  itself. `README.md` is design truth (§5 the availability contract, §9 the
  external integrations that break silently).
- Skills carrying the hard-won parts: `spark-operations` (the three namespaces,
  ports, the completion contract, the `run/` manifests), `mlx-inference`,
  `tui-keybinding-debugging` (pty verification for anything terminal-facing).
- Tests: `cd ~/Repos/vibes/spark && .venv/bin/python -m pytest`. Isolate with
  `SPARK_HOME=<dir>`. The editable install means source edits are live — only a
  pyproject dependency change needs
  `uv tool install --editable ../spark --reinstall`.
- Consumers currently point at 8095: nine Hermes configs (`mlx-local`),
  pi/omp/opencode providers, and banter's env plus Rust default.
