# Model Evaluation — Tactical Reference (2026-07-15)

Cross-model capability + tactical-use guide for the four models exercised today,
**all currently in the HF cache and all runnable on stock `mlx_lm`** (no fork).
Host: Apple M2, 16 GB. Sampling: `temperature = 0`.

> **Methodology & honesty note.** The two **small Bonsai** models (1.7B, 8B) were
> scored on **completed final answers** (≥1000-token budget). The **27B** and
> **Ornith-9B** figures come from the earlier **bake-off**, where both are heavy
> "thinking" models that exhausted a 450–550-token budget on chain-of-thought and
> never emitted a final answer — so they are graded on their **reasoning/planning
> traces** and the drafted prose inside them (marked `*`). Sample size is n=1 per
> cell: **directional, not statistical**. Prompts R1/R2/A1/A2 and the emotional
> battery E1/E2 are defined in `spikes/human-emotion-eval-skill.md`.

## Capability matrix

Legend: **★ Strong** · **✔ Pass** · **~ Partial** · **✗ Fail**

| Model | R1 — trap reasoning | R2 — procedural algebra | A1 — tool-use plan | A2 — clue-driven triage | Emotional / human |
|---|---|---|---|---|---|
| **Bonsai 1.7B** (ternary) | ✗ incoherent; wrong ("dollar went to the clerk") | ✔ correct + verified | ✗ broken tool calls, dup steps | ~ right format, misses every clue | ✗ generic template / 9-pt listicle |
| **Bonsai 8B** (ternary) | ✔ correct; names the false "$27+$2" (1 self-corrected slip) | ✔ correct + verified | ★ correct tool semantics, usable | ~ format + some substance (cgroup, contention); misses top clues | ~ competent; reads the cue; usable script but listicle-y |
| **Ornith 9B** (4-bit) `*` | ✔ spots double-count `*` | ✔ reaches a=18 `*` | ~ right instinct, stays meta `*` | ★ nails prod-vs-staging, cron, recent-change `*` | ~ structured scaffolding / "decision tree" `*` |
| **Bonsai 27B** (ternary) `*` | ★ clean full ledger `*` | ✔ correct setup `*` | ★ concrete runnable commands `*` | ★ nails the clues `*` | ★ warm integrated prose; names the specific crux `*` |

`*` graded on reasoning traces (final answer not emitted within budget).

**Reading the matrix:**
- **Procedure is size-robust** — every model passes R2 (mechanical algebra).
- **Insight is scale-dependent** — trap-spotting (R1) and correct tool semantics
  (A1) collapse below ~8B; the 1.7B produces the *form* of an answer filled with
  nonsense or broken calls.
- **~8B is the competence threshold** on every axis. Below it: form without
  substance and outright failures. At/above it: correct, if not always crisp.
- **Same-tier flavor split (Ornith-9B vs Bonsai-8B):** Ornith reasons better about
  *clues* (A2) but over-thinks and doesn't finish; Bonsai-8B emits *concrete,
  correct tool calls* (A1) and returns completed answers. Pick by whether you need
  deliberation or execution.

## Runtime & cost profile (measured)

| Model | Params | Quant | Cache size | Peak RAM | Decode tok/s | Default behavior |
|---|---|---|---|---|---|---|
| **Bonsai 1.7B** | 1.7B | ternary (MLX 2-bit) | 0.47 GB | <1 GB | **~74** | goes straight to answer (instruct-like) |
| **Bonsai 8B** | 8B | ternary (MLX 2-bit) | 2.2 GB | ~2.5 GB | **~27** | goes straight to answer |
| **Ornith 9B** | 9B | 4-bit affine | 5.6 GB | ~5 GB | ~15.5 | **heavy chain-of-thought** |
| **Bonsai 27B** | 27.8B | ternary (MLX 2-bit) | 7.9 GB | ~7.9 GB | ~6–9 | **heavy chain-of-thought** |

**Behavioral tell that matters operationally:** the small Bonsai ternary models
answer directly and finish inside a normal budget → **low latency to usable
output**. Ornith and the 27B *think* — they need **≥1000 completion tokens or they
return an empty answer**, and they cost more tokens/time per reply. Do not point a
tight `max_tokens` at a thinking model.

---

## Expect / Employ

### Bonsai 1.7B ternary — *the cheap mechanical pass* (0.47 GB · ~74 tok/s)
**Expect:** fast and reliable on **well-specified, procedural, verifiable** work —
algebra, format/structure, template-filling, following an explicit recipe. **Fails**
at anything needing judgment: trap-spotting, tool orchestration, emotional
attunement, using situational clues. It is **confidently wrong** (R1 produced
authoritative-looking nonsense).
**Employ for:** high-throughput low-stakes transforms — classification, extraction,
tagging, format/JSON shaping, simple math, draft scaffolding, autocomplete-style
suggestions, a cheap **router / first-pass** in a pipeline. **Never** the final
authority on correctness; **never** as a tool-calling agent or in a human-facing
emotional role.

### Bonsai 8B ternary — *the local workhorse* (2.2 GB · ~27 tok/s)
**Expect:** a genuinely competent generalist — correct procedural **and**
trap-spotting reasoning, **usable tool-use plans with correct semantics**, and
helpful (if listicle-y) human-facing replies. **Weaknesses:** pads/repeats, soft
prioritization, doesn't always grab the sharpest clue, lacks the 27B's crisp insight
and warmth.
**Employ for:** the **default local model** on this hardware — coding/ops assistant,
diagnostic planning, structured reasoning, RAG answerer, first-line support chat.
Small and fast enough to co-reside with other work. Escalate to the 27B only for
high-stakes emotional conversation, clue-critical tasks, or when you need
concrete-first agentic output.

### Ornith 9B (4-bit) — *the deliberate reasoner* (5.6 GB · ~15.5 tok/s)
**Expect:** strong **reasoning about a problem** — its traces exploited the A2 clues
better than any small model and it spots the R1 trap. But it **over-thinks**: it
burns budget on chain-of-thought and, under a normal cap, may not emit a final
answer; its agentic output stays at the "strategy/phases" altitude rather than
concrete commands; emotionally it defaults to a decision-tree.
**Employ for:** offline/batch reasoning where deliberation quality matters and
latency doesn't — analysis, triage planning, "think this through" tasks — with a
**generous token budget**. Less suited to interactive or tool-emitting agent loops
where you want the answer *now*.

### Bonsai 27B ternary — *the quality ceiling* (7.9 GB · ~6–9 tok/s)
**Expect:** best across every axis — clean trap reasoning, concrete agentic output,
and the only model that reliably **names the specific human crux** and writes warm
integrated prose rather than a checklist. Costs: slowest, largest, and a **thinker**
(needs the big budget).
**Employ for:** the escalation tier — high-stakes human-facing conversation,
tasks where a subtle clue is the whole game, or when output quality outranks
latency/footprint. **On a clean machine only** (see below).

---

## Other tactical notes (use / dismissal)

- **Memory-edge on the 27B.** At ~7.9 GB peak it sits at the edge of a 16 GB box.
  If another workload has already touched RAM it **thrashes to swap and collapses
  ~4× (to 2.2 tok/s)**. Give it a clean machine, or serve it alone. The 1.7B/8B
  have ample headroom and co-reside comfortably.
- **The listicle reflex** is a small-model tell: on open-ended "what do I do?"
  prompts, sub-8B models turn distress/ambiguity into a numbered checklist instead
  of engaging it. Watch for it as a quality regression signal.
- **Padding / repetition** at 8B (e.g. the same health-check restated 4×, every A2
  item labeled "most likely") — prune or post-process if you need tight output.
- **Confidently-wrong risk scales inversely with size.** The 1.7B is the dangerous
  one: fluent, structured, and wrong. Gate its output behind verification.
- **Suggested tiering (routing story):** `1.7B mechanical first-pass → 8B/9B
  workhorse → 27B escalation for nuance/insight`. The router's real job is
  detecting "judgment vs execution" — execution stays cheap, judgment escalates.
- **Deleted / not available:** `Bonsai-8B-mlx-1bit` was fetched and **deleted**;
  it never ran because **stock MLX rejects `bits=1`** (supported: 2,3,4,5,6,8) — it
  requires PrismML's `mlx` fork. If 1-bit behavior ever matters, that's a
  fork-install + security-review task, not a stock run.
- **All four cached models are text-in/text-out on stock `mlx_lm`.** The 27B repo
  is multimodal (`ForConditionalGeneration`, vision tower present) but was run
  text-only; image input would route through `mlx-vlm`, untested here.

_Source runs: patchwork `experiments/inference-bench` result #003 (27B perf);
`bake-offs/2026-07-15-bonsai-27b-vs-ornith-9b.md` (27B vs Ornith); this session's
1.7B/8B emotional + reasoning/agentic runs. Prompts + rubric:
`spikes/human-emotion-eval-skill.md`._
