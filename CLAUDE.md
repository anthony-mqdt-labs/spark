# CLAUDE.md — spark

Inherits all conventions from the parent `../CLAUDE.md` (operator
profile, supply-chain sensitivity, PII hygiene, mandatory JSONL logging, bug wiki).
This file records spark-specific rules.

## What spark is
A wrapper that orchestrates already-installed LLM inference runtime CLIs
(`mlx_lm.server`, `mlx_vlm.server`, `omlx`, `llama-server`, `ollama`) via
subprocess. spark's own venv is intentionally tiny and **never imports
mlx/vllm/llama** — that keeps it off the system-Python wheel-lag treadmill.

## Hard rules
- **Config is data, not code.** Launch flags, runtime defs, defaults, and the
  model registry live in TOML (`config/` shipped; user data dir at runtime). Code
  reads a validated `SparkConfig`. Never hardcode a runtime flag in `src/`.
- **Secrets never touch disk-as-text, argv, logs, or LLM prompts.** Backend is the
  macOS Keychain via `security`. Secret values are read with getpass/stdin, passed
  to child processes via env only, and run through the telemetry redaction filter.
- **Every fail state surfaces {what, why, do-this-next}** via a `SparkError`
  subclass carrying `remediation`. The supervisor attempts bounded restart with
  backoff before terminal failure.
- **Lean pinned deps.** Adding a dependency requires a threat-model justification,
  not convenience. No native build / postinstall packages.

## Layout
`src/spark/{cli,config,secrets,runtimes,probe,registry,runner,research,telemetry}`
plus `errors.py`. See `~/.claude/plans/parallel-wandering-lemur.md` for the plan.

## Host facts (this machine)
Apple M2 · 16 GB unified · 8 cores · tight disk. MLX is the native-optimal path;
`vllm` is unavailable on Apple Silicon (registered as an unavailable backend).
