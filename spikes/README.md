# spikes

Time-boxed **evaluation spikes**: bounded investigations that reduce uncertainty
about whether a model, runtime, or technique earns a place in spark. A spike is
allowed to be scrappy and is judged by the *decision* it produces, not by shipped
code.

A typical spike runs: technology evaluation → **compatibility gating / preflight**
(fail fast at the cheapest gate, before paying download + load cost) → runtime
bring-up → **performance characterization** on the target host → a verdict.

## Conventions

- One file per spike: `YYYY-MM-DD-<slug>.md`, ending in an explicit
  **go / no-go / conditional** verdict.
- If a spike graduates into a head-to-head, the comparison lives in `bake-offs/`
  and the spike links to it.
- Don't duplicate artifacts. If the spike produced a benchmark result or a
  registry entry, **link** it (single source of truth).

## Logged spikes

- **2026-07-15 — Bonsai-27B edge bring-up.** PrismML Ternary-Bonsai-27B evaluated
  for local inference on the M2 target. Preflight found the MLX repo is stock
  2-bit affine quant of a `qwen3_5` arch (no PrismML fork / custom kernel needed —
  that path is GGUF-only), so plain `mlx_lm` serves it. Brought up via the
  `ternary-bonsai-27b-mlx-2bit` registry entry; characterized at 9.55 tok/s /
  7.86 GB peak. **Verdict: go, conditional** — fits and runs, but sits at the
  memory edge of a 16 GB box. Artifacts (no separate spike doc, to avoid
  duplication):
  - Benchmark: `patchwork/experiments/inference-bench` **result #003**
  - Head-to-head follow-on: [`bake-offs/2026-07-15-bonsai-27b-vs-ornith-9b.md`](../bake-offs/2026-07-15-bonsai-27b-vs-ornith-9b.md)
  - Registry: `~/.local/share/spark/models/ternary-bonsai-27b-mlx-2bit.toml`
