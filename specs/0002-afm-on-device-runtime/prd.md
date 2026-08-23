# 0002 — AFM On-Device Runtime (Apple Foundation Models)

> **STATE: WIP / RESEARCH (feature proposal).** Not a design, not an
> implementation. This document records a potential feature — wiring Apple's
> on-device Foundation Model (AFM 3 Core) into spark as a supervised runtime —
> with the verified host facts, the gating unknowns, and the spikes that must
> run before any design.md. Do not implement from this document; run the
> spikes.
>
> Origin: 2026-08-23 session (operator + agent hardware investigation of
> macOS 26.5.2's installed Apple Intelligence model assets). Precedent: the
> `darkcore` runtime seam (config/runtimes/darkcore.toml) — a supervised
> child that speaks OpenAI-compatible HTTP with a `/health` endpoint, driven
> by the generic `Backend`, `model_format = "any"`.

---

## 1. Problem statement

spark fronts every local inference runtime on this machine except the one
Apple ships with the OS. macOS 26.5.2 preinstalls **AFM 3 Core** — Apple's
~3B dense third-generation on-device language model — as system assets
(`com_apple_MobileAsset_UAF_FM_GenerativeModels`, ~2.34 GB, flagged
"system+, never remove"), plus a ~5.3 GB coding model
(`UAF_FM_CodeLM`, Xcode predictive completion). The runtime daemons
(modelmanagerd, assetsubscriptiond, modelcatalogd, generativeexperiencesd)
are alive on this host.

But there is **no CLI** to call it (checked /usr/bin, /usr/local/bin, the
Xcode 26.6 toolchain — nothing; `FoundationModels.framework` is a v1.0
stub whose real code lives in the dyld shared cache), and spark has no
runtime that reaches it. The only public API is the Swift
FoundationModels framework (`SystemLanguageModel`, `LanguageModelSession`).

Feature: expose AFM 3 Core as a first-class spark runtime — an external
Swift bridge binary (FoundationModels → OpenAI-compatible HTTP on
loopback) supervised by spark exactly like `darkcore`. Net effect: `spark
afm` serves a private, zero-download, OS-managed model; no weights on
disk, no HF, no network.

## 2. What already exists (do not rebuild)

| Piece | Where | Reuse |
|---|---|---|
| Supervised-child runtime pattern | `config/runtimes/darkcore.toml` (spawns `python -m darkcore.server`, `/health`, OpenAI-compatible, `model_format="any"`, env-resolved binary) | The exact shape an AFM bridge needs — including "no Backend subclass required" (darkcore has none) |
| Lifecycle | `src/spark/runner/supervisor.py` (port pick, health-wait, restart+backoff, graceful SIGINT) | unchanged |
| Registry + budget | `src/spark/registry/`, `src/spark/budget.py` | OS-managed weights need a memory-gate treatment (darkcore note: the gate can't see the router's weights — same principle) |
| Swift toolchain | Xcode 26.6 (build 17F113) installed; `swiftc` compiles a `FoundationModels` import (verified 2026-08-23) | bridge build environment ready |
| OS runtime stack | modelmanagerd / modelcatalogd / generativeexperiencesd etc. running | model is loadable once opted in |

## 3. Committed intent (stable even while WIP)

1. **A supervised bridge, not a spark-native implementation.** A small Swift
   executable wraps `LanguageModelSession` and serves
   `/v1/models`, `/v1/chat/completions`, `/health` on 127.0.0.1 — the darkcore
   server contract. spark spawns it via a `config/runtimes/afm.toml`
   (`binary = <bridge path>`, `args_template = ["--host","{host}","--port","{port}"]`,
   `openai_compatible = true`, `health_path = "/health"`, generous
   `ready_timeout_s` for cold model load) plus a registry entry
   (`models/afm.toml`: `backend = "afm"`, `model_format = "any"`,
   `params_billions = 3.0`).
2. **Zero new Python in spark core.** The bridge is an external binary —
   the wrapper-not-reimplementation invariant (D1) holds; spark's venv stays
   at four pure-Python deps.
3. **Loopback only, no secrets.** Same posture as every other runtime.
4. **Dormant-by-default.** Nothing changes until the operator enables Apple
   Intelligence; the runtime is additive and can sit unregistered.

## 4. Verified host facts (2026-08-23) — the gates live here

1. **Apple Intelligence opt-in is FALSE.** generativeexperiencesd logs
   "Fetched value for opt-in status: false" / "GMS Unavailable".
   `SystemLanguageModel.availability` stays `.unavailable` regardless of
   code until the operator flips it on (System Settings > Apple Intelligence
   & Siri). **This gates everything.**
2. **Model identity.** Installed = **AFM 3 Core** (~3B dense, gen-3). The
   interesting sibling — **AFM 3 Core Advanced** (20B sparse,
   Instruction-Following Pruning, ~1–4B active) — is gated to M3+/12GB-class
   hardware and is **not available on this M2**. This feature is therefore
   wiring the weak sibling; say that plainly in any design doc.
3. **No CLI, but a compiling SDK.** No Apple-shipped FoundationModels CLI on
   this machine; `swiftc` links the module fine (probe compiled 2026-08-23).
4. **Context size: not officially published for gen-3.** Lineage evidence:
   gen-1 (arXiv:2407.21075) context-lengthening to 32,768-token sequences;
   gen-2 (arXiv:2507.13575) "continued training on sequences containing up
   to 65K tokens". Expect 32K–65K class; the runtime truth is
   `SystemLanguageModel.contextSize` — one of spike A1's outputs.
5. **Consent gate.** A live FoundationModels invocation trips a system
   consent prompt (observed 2026-08-23 — an unentitled probe was blocked).
   First real run needs operator presence at the prompt.
6. **Ability ceiling.** Published gen-2 on-device numbers (Table 1,
   arXiv:2507.13575): MMLU 67.85 / MMMLU 60.60 / MGSM 74.91 vs
   Qwen-2.5-3B 66.37/56.53/64.80, Gemma-3-4B 62.81/56.71/74.74,
   Qwen-3-4B 75.10/66.52/82.97. AFM ≈ top of the 3B class — **below the
   Qwen-3-4B spark already serves via mlx_lm**. Value is privacy +
   zero-download + OS-managed, not quality.

## 5. Spikes & experiments (run before design.md)

- [ ] **A1 — Opt-in + availability.** Operator enables Apple Intelligence;
      run a scratch entitled Swift app that prints
      `SystemLanguageModel.default.availability` and `.contextSize`.
      Gates everything downstream; answers the context question for real.
- [ ] **A2 — Entitlement/signing.** Build the minimal bridge CLI and codesign
      with `com.apple.developer.intelligence` (ad-hoc first; Xcode dev cert
      fallback if ad-hoc is rejected). Verify the model actually loads and
      responds. This is the highest-uncertainty technical risk.
- [ ] **A3 — Bridge server + quality spot-check.** Full OpenAI-compatible
      HTTP loop; measure TTFT, tokens/sec, context ceiling (long-prompt
      probe), and a small battery (summarize / classify / extract) vs
      `mlx_lm` Qwen-3-4B on this host. Decide whether the gap is
      acceptable for any real routing.
- [ ] **A4 — Spark wiring dry run.** `config/runtimes/afm.toml` + registry
      entry; `spark run afm` health-wait, restart behavior, and the
      memory-gate treatment for an OS-managed model (no weights path —
      confirm `params_billions` budgeting is sane, not gamed).

## 6. Open questions (blocking design.md)

1. Bridge residency: standalone repo (darkcore precedent: the router is a
   sibling project) vs a `bridge/` dir inside this repo?
2. Registry shape: first-class model entry (`spark afm` routes like any
   model) vs a special system runtime the operator invokes explicitly?
3. Routing policy: AFM is a 3B utility model on a box that already serves
   Qwen-3-4B. Is there a real use case (privacy-critical tasks, zero-disk
   fallback, battery-frugal quick tasks), or does this stay a spike?
4. Does the FM_CodeLM (~5.3 GB, Xcode coding model) deserve a separate
   runtime later, or is that Xcode's private concern?

## 7. Non-goals (v1)

- No AFM 3 Core Advanced (20B) support — not eligible on M2; revisit only on
  M3+/M4-class hardware.
- No FM_Visual / diffusion (not installed on this host).
- No PCC / server models (AFM 3 Cloud etc. are Apple-side cloud, outside
  spark's job).
- No attempt to read protected asset weights directly
  (/System/Library/AssetsV2 UAF_FM_* trees are System-Policy denied — the
  framework is the only sanctioned path).

## 8. References

- Apple MLR: "Introducing the Third Generation of Apple's Foundation Models"
  (2026-06) — AFM 3 family, IFP, QAT compression
- arXiv:2407.21075 (gen-1 on-device: 32K lengthening, GQA/SwiGLU/RoPE, 2-bit
  QAT) · arXiv:2507.13575 (gen-2 on-device: 65K sequences, KV-cache sharing,
  Table 1 benchmarks)
- `config/runtimes/darkcore.toml` — the supervised-child seam this feature
  copies
- `src/spark/runner/supervisor.py`, `src/spark/budget.py` — reuse surface
- Host inventory: `/System/Library/AssetsV2/com_apple_MobileAsset_UAF_FM_*`
  (listable via mobileassetd CacheDelete logs; dirs themselves are
  System-Policy protected)
