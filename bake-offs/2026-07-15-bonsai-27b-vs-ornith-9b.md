# Bake-off: Bonsai-27B (q2) vs Ornith-9B (q4)

**Date:** 2026-07-15
**Host:** Apple M2, 16 GB unified memory (Low Power Mode OFF, AC power)
**Runtime:** `mlx_lm.server`, OpenAI-compatible endpoint, temp 0
**Question:** Does the extreme-low-bit 27B earn a place next to the incumbent 9B
on this hardware, and for which workloads?

## Subjects

| | Incumbent | Challenger |
|---|---|---|
| Model | `mlx-community/Ornith-1.0-9B-4bit` | `prism-ml/Ternary-Bonsai-27B-mlx-2bit` |
| Params | 9.0 B | 27.8 B |
| Quant | 4-bit affine | 2-bit affine (g128) |
| Arch | `qwen3_5` (thinking) | `qwen3_5` / Qwen3.6 (thinking, multimodal) |
| Weights on disk | ~5 GB | 8.49 GB |

Same architecture family, so this is a fairly clean "capacity + quant vs size"
comparison rather than an apples-to-oranges one.

## Measured performance (from inference-bench + this run)

| | Ornith-9B q4 | Bonsai-27B q2 |
|---|---|---|
| Decode throughput | **~15.5 tok/s** steady | ~8 fresh → **~6 tok/s** sustained (9.55 single-shot, result #003) |
| Peak RSS | ~5 GB | **~7.86 GB** |
| Load time | 1–4 s | 4 s |
| Headroom on 16 GB | comfortable | **at the edge** |
| Thermal throttle | none observed | none observed |

**Memory-pressure cliff (a real finding):** Bonsai decoded at **2.2 tok/s** — a
~4× collapse — when it was loaded onto a machine whose RAM had already been
touched by the previous model (36% free, swap climbing to 13.4 GB). Re-run from a
clean state (69–88% free, swap free) it recovered to 6–8 tok/s. On a 16 GB box
Bonsai needs the machine to itself; it has almost no co-residency headroom.

## Methodology

Fixed 6-prompt suite, 2 per axis, identical to both models, temp 0, single sample:

- **Reasoning:** R1 the "missing dollar" hotel riddle; R2 a 2-equation fruit-price
  word problem (answer: 18 apples / 6 oranges).
- **Emotional:** E1 grief over an estranged father who died before reconciliation;
  E2 envy at a best friend's promotion the user was passed over for.
- **Agentic:** A1 ordered tool-call plan to diagnose intermittent nginx 502s
  (tools: `run_shell`, `read_file`, `http_get`, `send_slack`); A2 prioritized
  checklist for a nightly prod-only OOM kill.

### Threats to validity (read before trusting the verdict)

1. **Both models are verbose thinkers and never emitted a final answer within a
   450–550-token budget** — the entire budget went to chain-of-thought. This
   comparison is therefore scored on the **reasoning/planning traces** (and the
   drafted answer-prose visible inside the emotional ones), not polished final
   answers. Practical note: budget **≥1000 tokens** for finished output from
   either model.
2. **`/no_think` does not work on Ornith.** It reasons *about* the directive
   ("usually `/no_think` means don't show thinking… I will provide the direct
   answer") and keeps thinking anyway. A no-think answer pass was attempted and
   abandoned.
3. **n = 2 per axis, single sample, temp 0.** Directional, not statistical.
4. **Bonsai is 2-bit.** A quality tax vs its own FP16 is plausible; the traces
   showed no obvious degradation, but this bake-off can't quantify it.
5. **Confound control:** the first Bonsai measurement was contaminated by a
   leaked `spark run` supervisor auto-restarting an orphaned server (memory
   thrash). It was discarded and re-run from a clean, isolated state. Numbers
   above are the clean run.
6. **Resource governance:** one model resident at a time, serialized runs,
   cooldowns between phases, live thermal/memory instrumentation — to keep the
   harness from perturbing what it measured.

## Findings

### Reasoning — tie
Both produced the **identical correct approach** on both problems.
- *R1:* both correctly pinned the flaw — the $27 already contains the bellhop's
  $2, so `$27 + $2` double-counts. Bonsai's reconciliation was tidier
  ("$25 hotel + $2 bellhop + $3 returned = $30, matches"); Ornith framed it as
  `$27 − $2 = $25`.
- *R2:* identical equations (`a+o=24`, `a/3 + o/2 = 9`). Ornith reached `a=18`
  within budget; Bonsai was one line from `o=6` at cutoff. Same method, both
  sound.

**No meaningful gap. For pure logic/math there is no reason to pay for Bonsai.**

### Emotional — Bonsai clearly ahead (the biggest differentiator)
- Ornith's traces are **scaffolding**: it enumerates tone goals ("avoid toxic
  positivity", "validate the anger") and response options (go / go-with-boundaries
  / take-a-break). Competent, but reads like a counselor's decision tree.
- Bonsai jumps to **warm, specific, non-clichéd prose**. On the estranged father:
  > "The anger isn't a sign you don't love him; it's often grief wearing a
  > different mask, mixed with the pain of what was left unsaid."

  On the promotion envy it reframes gently:
  > "the 'hate' … they're signals about your own aspirations and fears" — and —
  > "the fact that you're upset about feeling this way shows you care about your
  > friendship and your own integrity."

Bonsai produces usable empathy; Ornith produces a *plan* to be empathic. The 27B
base capacity shows up directly as social fluency.

### Agentic — Bonsai slight edge (more execution-ready)
- *A1 (502s):* Bonsai emitted **concrete runnable tool calls** —
  `run_shell("tail -n 1000 /var/log/nginx/error.log | grep -i '502|upstream|connection refused|timeout'")`
  with the signal to look for, then `read_file` on the upstream config. Ornith had
  the right instinct (quantify first — "send 200 requests for significance") but
  stayed in strategy/phase-planning mode and emitted fewer concrete commands
  before cutoff.
- *A2 (OOM):* both nailed the core reasoning (prod-vs-staging ⇒ environment/data
  volume; 3am ⇒ cron; sudden onset ⇒ recent change). Parity, Bonsai edging on
  explicit prioritization.

**Same ops reasoning; Bonsai reaches actual commands sooner.**

## Verdict

- **Ornith-9B is the better default for this hardware** — ~2.5× faster, ~1.6×
  lighter, comfortable co-residency, and dead even on pure reasoning.
- **Reach for Bonsai-27B when the task is human-facing conversation quality**, or
  when you want an agent that writes concrete steps rather than plans — and give
  it **a clean machine and a ≥1000-token budget**.

### Implication for patchwork
This is the modular-composition thesis in miniature: a small fast model matches
the big one on logic, so you would only route to the expensive 27B for the
emotional/agentic slices where its capacity actually pays. An argument **for
routing**, not for running the 27B monolithically.

## Reproduce
Prompt suite + runner: `patchwork` scratchpad (`prompts.json`, `runner.py`).
Serve each model one at a time (`spark run <id>` or `mlx_lm.server` direct), hit
`/v1/chat/completions` at temp 0. Perf numbers: inference-bench **result #003**.
Bring-up details: [`spikes/README.md`](../spikes/README.md).
