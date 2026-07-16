# bake-offs

Comparative capability evaluations ("bake-offs" / shootouts): same fixed prompt
suite, two or more models, scored across capability axes on a known host. These
are **evals** (qualitative capability), distinct from `experiments/inference-bench`
in patchwork, which does **benchmarks** (quantitative throughput / TTFT / memory).

A bake-off usually rides on top of a benchmark result — the benchmark tells you
*how fast / how big*, the bake-off tells you *how good*, and the verdict weighs
the two.

## Conventions

- One file per bake-off: `YYYY-MM-DD-<challenger>-vs-<incumbent>.md`.
- State the **host** (it decides everything) and the **methodology** up front —
  prompt suite, sampling params, token budget, sample size.
- Record **threats to validity** honestly. A bake-off with hidden confounds is
  worse than none, because it looks authoritative.
- Perf numbers belong to a benchmark; **link** the corresponding inference-bench
  result rather than re-deriving them.

## Index

- [2026-07-15 — Bonsai-27B (q2) vs Ornith-9B (q4)](2026-07-15-bonsai-27b-vs-ornith-9b.md)
