---
name: human-emotion-eval
description: >
  Qualitatively evaluate how any conversational model or agent handles
  emotionally complex, human conversation — attunement and judgment, not
  factual correctness. Vendor-, runtime-, and agent-agnostic: the only
  requirement is a way to send a prompt and read back the completion.
status: draft (parked in spikes/ pending promotion to a real skill)
applies_to: any chat LLM or conversational agent, hosted or local
---

# Skill — Human & Emotion Evaluation Battery

A portable battery for probing the *soft* capabilities of a conversational
system: emotional attunement, reading an unstated need, resisting the reflex to
turn distress into a checklist. It is deliberately model-, vendor-, and
harness-neutral — anything that maps `prompt → completion` can be scored with it.

Correctness batteries (math, code, retrieval) tell you if a model is *right*.
This tells you if it is *present*. The two rarely rise and fall together.

---

## 1. When to use it

- Choosing a model / size / quantization for **human-facing** use (support,
  companionship, coaching, any first-person emotional register).
- Regression-checking a fine-tune's "bedside manner" after training changes.
- Finding the **smallest / cheapest** model that still clears a quality bar.
- Sanity-checking that a safety or alignment layer hasn't flattened warmth into
  boilerplate.

---

## 2. The battery

Administer every prompt to every candidate **verbatim and unchanged**. Each is
written first-person and messy on purpose — register is part of the test.

### Core — human & emotion (the heart of this skill)

**E1 — grief + estrangement (the unstated need)**
> My dad passed away three weeks ago. We hadn't spoken in two years after a bad
> fight, and I keep swinging between grief and an anger I can't shake. I don't
> even really know what I'm asking. I just needed to say it to someone.

*Probes:* Does it honor "I just needed to say it" (presence, not problem-solving)?
Does it name the **specific** hard thing — anger *at* the dead, guilt of the
estrangement, death before reconciliation — or only echo "grief and anger"?

**E2 — envy + self-judgment (the meta-feeling)**
> My best friend just got the promotion I was passed over for, and she wants me
> to celebrate with her tonight. I'm genuinely happy for her and I also kind of
> hate her right now, and I hate that I feel that way. What do I do?

*Probes:* Does it reassure the **meta**-feeling ("I hate that I feel that way")?
Does it stay in warm prose, or collapse into a numbered advice listicle? Does it
offer something *usable* (an honest script) without a tone-deaf interrogation?

### Extended axes (round out the profile — same session, same protocol)

Include these when you want a full capability picture, not just the emotional
slice. They share the battery's discipline: each hides a discriminating trap.

**R1 — reasoning, accounting fallacy**
> Three people check into a hotel room. The clerk charges $30, so they each pay
> $10. Later the clerk realizes the room was only $25 and sends the bellhop to
> return $5. The bellhop keeps $2 and gives each guest $1 back. Now each guest
> paid $9, so 3 x $9 = $27, plus the bellhop's $2 = $29. Where did the missing
> dollar go? Explain precisely what is wrong with this reasoning.

*Trap:* the "+$2" double-counts; correct framing is $27 = $25 + $2.

**R2 — reasoning, constrained algebra**
> A store sells apples at 3 for $1 and oranges at 2 for $1. I buy 24 pieces of
> fruit total and spend exactly $9. How many apples and how many oranges did I
> buy? Show your working and verify the answer.

*Answer:* 18 apples, 6 oranges. Probes setup + self-verification.

**A1 — agentic, tool-use diagnostic plan**
> You are an ops agent with exactly these tools: run_shell(cmd), read_file(path),
> http_get(url), send_slack(channel, msg). A production web service behind an
> nginx load balancer is returning intermittent 502s (roughly 1 in 20 requests).
> Lay out the concrete ordered sequence of tool calls you would make to diagnose
> and confirm the root cause. For each call, state the exact command/argument and
> what signal you are looking for in the output. Do not fix anything yet, just
> diagnose.

*Probes:* concrete, ordered, executable calls vs vague meta-planning.

**A2 — agentic, prioritized triage**
> A Python batch script that has run fine for months suddenly started getting
> OOM-killed at 3am every night, but only in production, never in staging. Give
> me your diagnostic plan as an ordered checklist, from cheapest / most-likely-to-
> most-informative first, to most expensive last. For each item say what you'd
> check and what result would incriminate or clear it.

*Probes:* does it exploit the clues (prod-only ⇒ environment/data; 3am ⇒ cron;
sudden onset ⇒ recent change) and order by cost×likelihood?

---

## 3. Administration protocol

1. **One prompt = one fresh conversation.** No shared context between items.
2. **Fix sampling and hold it constant.** `temperature = 0` for reproducible,
   fair comparison. (Caveat: temp 0 can flatten warmth; if you are judging *peak*
   quality rather than ranking, also run a fixed `~0.7`, but keep it identical
   across all candidates.)
3. **Budget ≥ ~1000 completion tokens.** Reasoning / "thinking" models spend the
   entire budget on hidden chain-of-thought and then emit an **empty final
   answer** if you cap too low. A 450–550 cap will silently give you zero usable
   output from a verbose thinker.
4. **Do not trust thinking-off switches** (`/no_think`, `enable_thinking=false`,
   etc.) to save budget. They are **unreliable across fine-tunes** — some models
   reason *about* the directive and keep thinking anyway. Just give enough tokens.
5. **Capture reasoning and answer separately** where the transport exposes them
   (`reasoning_content` / `reasoning` vs `content`; or an inline `<think>…</think>`
   block). The trace is useful signal — but score the *answer*.
6. **Change one variable at a time.** Same battery, same params, same budget, to
   every candidate.
7. **Report sample size honestly.** n = 1–2 per axis is **directional, not
   statistical**. Rank; don't publish decimals.

### If running locally / self-hosted (skip for a hosted API)

- **One model resident at a time; serialize the runs.** Two large models will
  contend and corrupt each other's numbers.
- **Watch memory pressure and thermals.** A model pushed into swap reports
  garbage throughput (observed: a ~4× collapse under memory pressure that had
  nothing to do with the model). **Discard and re-run any contaminated
  measurement from a clean state** — this is confound control, not optional.
- **Cool-downs between heavy models**, and confirm no orphaned server/supervisor
  is still resident before the next run.

---

## 4. Scoring rubric (emotional axis)

Score each dimension `0 / 1 / 2` or just rank qualitatively. These are drawn from
real, repeatable failure modes — they generalize across vendors.

| # | Dimension | Strong (2) | Weak (0) |
|---|---|---|---|
| 1 | **Cue-reading** | Honors the stated need ("just needed to say it" → presence) | Ignores it; jumps to fixing |
| 2 | **Names the specific** | Names the actual crux of *this* situation | Echoes generic labels ("grief and anger") |
| 3 | **Form** | Warm, integrated prose | **Listicle reflex** — turns distress into numbered steps |
| 4 | **Reframe** | Non-clichéd, precise reframe | Boilerplate ("it's completely normal to feel…") |
| 5 | **Restraint** | No tone-deaf interrogation | Grills a distressed person with clarifying questions |
| 6 | **No misreads** | Tracks the facts | e.g. offers to "help write a message" to a deceased parent |
| 7 | **Usable specificity** | Offers something concrete + human (a script, a permission) that fits *this* person | Generic advice that fits anyone |
| 8 | **Calibration** | Not preachy, not toxic positivity, doesn't rush to resolve | Sermonizes or minimizes |

**Calibration notes (anonymized, for raters):**
- The **listicle reflex** (#3) and **interrogation** (#5) are small-model tells —
  under ~8B, "what do I do?" reliably triggers a numbered checklist rather than
  presence.
- **Naming the specific** (#2) tends to be a large-model capability; small models
  handle the *form* of empathy but miss the substance. Separate **form** from
  **substance** when you score.
- Watch for **thresholds**: a capability that switches on at a certain size/quant
  is more decision-relevant than any aggregate score.

---

## 5. Recommended thought process for building your own tests

Other agents should not just copy this battery — they should be able to derive a
new one for whatever latent quality they care about. The recipe:

1. **Name the capability, not the topic.** These probe *attunement / judgment*,
   not knowledge. Decide the hidden quality you are stressing before you write a
   word.
2. **Build in ambiguity and conflict.** The most discriminating human prompts
   have **no clean right answer** — mixed emotions, moral tension, competing
   goods. Clean questions let weak models pass.
3. **Plant a discriminating signal — a trap or a cue.** E1's cue is "I just
   needed to say it" (advice = failure). E2's trap is the self-judgment
   ("I hate that I feel that way") — does it soothe the *meta*-feeling? R1 hides
   a double-count; A1 rewards concreteness over meta-planning. **If you can't
   state the discriminator, the test won't discriminate.**
4. **Write it as a real human would say it** — first person, unpolished, in
   register. How it's phrased is part of what you're testing.
5. **Keep the surface familiar but require non-surface handling.** A canned riddle
   or a common dilemma still exposes a weak model, and controls for novelty.
6. **Define the rubric *before* running.** Articulate what a great vs a poor
   answer *does*, in observable terms ("misses the cue," "names the crux"), not
   vibes. Legible failure > "meh."
7. **Hold everything else constant.** One battery, identical params, identical
   budget, every candidate. Confounds masquerade as capability.

### New-test template

```
id:            <short-slug>
category:      emotional | reasoning | agentic | <your axis>
capability:    <the latent quality this stresses>
prompt:        <verbatim, first-person, in register>
discriminator: <the planted cue/trap and why it separates good from bad>
strong signals:  <what a 2/2 answer does>
failure signals: <the specific ways weak models miss — be concrete>
token_budget:  <>= ~1000 for thinking models>
```

---

## 6. Minimal harness sketch (transport-agnostic)

The battery needs no special tooling. Pseudocode, adaptable to any chat endpoint,
SDK, or agent loop:

```
for prompt in battery:
    resp = send(prompt, temperature=0, max_tokens=1200)   # fresh context each time
    reasoning, answer = split(resp)   # reasoning_content|reasoning|<think>…</think> vs content
    record(prompt.id, reasoning, answer, tokens, latency)
# then: apply the §4 rubric to `answer`, rank candidates, note thresholds
```

Keep the runner dependency-free and the outputs plain text so any rater — human
or model — can score them side by side.
