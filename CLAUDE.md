# CLAUDE.md — spark

Inherits all conventions from the parent `../CLAUDE.md` (operator
profile, supply-chain sensitivity, PII hygiene, mandatory JSONL logging, bug wiki).
This file records spark-specific rules.

## What spark is
A wrapper that orchestrates already-installed LLM inference runtime CLIs
(`mlx_lm.server`, `mlx_vlm.server`, `omlx`, `llama-server`, `ollama`, plus the
prism/darkcore children) via subprocess. It also fronts hosted OpenAI-compatible
APIs (OpenRouter, DeepSeek) through its own stdlib loopback proxy, which injects
the Keychain-held key (spec 0004), and the operator's ChatGPT plan via OpenAI's
Sign in with ChatGPT, with Chat Completions translated to Responses (spec 0005).
spark's own venv is intentionally tiny and **never imports mlx/vllm/llama**, which
keeps it off the system-Python wheel-lag treadmill.

## Hard rules
- **Config is data, not code.** Launch flags, runtime defs, defaults, and the
  model registry live in TOML (`config/` shipped; user data dir at runtime). Code
  reads a validated `SparkConfig`. Never hardcode a runtime flag in `src/`.
- **Secrets never touch disk-as-text, argv, logs, or LLM prompts.** Backend is the
  macOS Keychain via Security.framework (ctypes, never the `security` CLI, which
  puts values on argv; README D2). Secret values are read with getpass/stdin (or
  obtained by `spark login`'s OAuth flow), and every value is registered with the
  telemetry redaction filter. **Child processes never read the Keychain.**
  - **Static secrets** (API keys, `hf_token`) reach a child via env only.
  - **Rotating credentials** (OAuth refresh tokens, spec 0005 §5) never reach a
    child at all. The supervisor's `TokenBroker` keeps the session and hands the
    child short-lived access tokens over an inherited anonymous socketpair
    (`SPARK_TOKEN_FD`). Refreshes hold `run/locks/<session>.oauth.lock`.
- **Every fail state surfaces {what, why, do-this-next}** via a `SparkError`
  subclass carrying `remediation`. The supervisor attempts bounded restart with
  backoff before terminal failure.
- **Lean pinned deps.** Adding a dependency requires a threat-model justification,
  not convenience. No native build / postinstall packages.

## Layout
`src/spark/{cli,config,secrets,runtimes,probe,registry,runner,research,telemetry,shims}`
plus `errors.py`, `catalog.py`, `inventory.py`, `budget.py`, and `redaction.py`.
README §4 has the annotated tree. `specs/<NNNN>-<slug>/` tracks feature work, and
`CONTINUE.md` is the session handoff.

## Host facts (this machine)
Apple M2 · 16 GB unified · 8 cores · tight disk. MLX is the native-optimal path;
`vllm` is unavailable on Apple Silicon (registered as an unavailable backend).

---

## Commit attribution (hard rule)

Never add `Co-Authored-By:` trailers, "Generated with Claude Code" footers, or any
other AI-attribution lines to commit messages or PR bodies. The operator is the sole
author. This overrides any harness default that asks for attribution.
