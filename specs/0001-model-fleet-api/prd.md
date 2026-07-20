# 0001 — Model Fleet API (sparkd)

> **STATE: WIP / RESEARCH.** This spec is deliberately incomplete. It frames
> the problem, fixes the seam contracts we're confident about, and enumerates
> the spikes that must run before a design.md is worth writing. Do not
> implement from this document; implement the spikes.
>
> Origin: patchwork routing session 2026-07-18 (operator + agent gap
> analysis). Cross-repo context: `../patchwork/` specs 0001–0003,
> `docs/routing-architecture.md`, and the seam-5 precedent
> (`spark darkcore-router` supervising the router as a child).

---

## 1. Problem statement

Spark today is a *launcher*: one foreground invocation, one supervised model
server, dies with the TTY. The darkcore router today is a *self-hoster*: its
`ModelPool` loads all three tiers in-process via mlx-lm and privately manages
the residency dance (T0+T1 co-resident; 27B alone; evict/rewarm).

We want spark to become the machine's **model fleet authority**: a resident
daemon that serves a *selection* of models as OpenAI-compatible endpoints and
**communicates those endpoints to consumers** — the router first, other
agents later. The router then consumes tiers as HTTP endpoints instead of
in-process loads, through its existing control surface.

Why this split is right (and why it echoes the patchwork architecture): one
process should own "what is resident on this box" (spark, which already has
`budget.py`, the registry, and the supervisor), and consumers should be dumb
about it. Two processes both believing they own memory on a 16 GB box is how
the S4 eviction saga happens at fleet scale.

## 2. What already exists (do not rebuild)

| Piece | Where | Reuse |
|---|---|---|
| Per-model OpenAI servers | `config/runtimes/*.toml` (`mlx_lm.server` et al.) | "serve model M on port N" is solved |
| Supervisor primitives | `src/spark/runner/supervisor.py` (port pick, health-wait, reap, monitor) | lifts into the job table |
| Model registry | `src/spark/registry/` | becomes `GET /models` |
| Memory model | `src/spark/budget.py` | becomes the scheduler's gatekeeper |
| OpenAI proxy pattern | `src/spark/runtimes/relay.py` (tested) | template for router-side HTTP tier client |
| Child-process precedent | `spark darkcore-router` (seam 5) | inverts: spark serves, router consumes |
| Router-side knob | darkcore control surface: tier roster | endpoint handoff channel (§4) |

## 3. Committed intent (stable even while WIP)

1. **A resident daemon** (`sparkd` / `spark serve --fleet`), loopback-only,
   holding a job table of N supervised model servers.
2. **Control API** (shape TBD by S1): list models, start/stop/ensure a model,
   report endpoints + health. Deep health aggregates child `/health`.
3. **Endpoint handoff via the consumer's own config channel, not a new
   discovery protocol.** For darkcore: an actuator writes the tier roster
   through `set_config` (versioned, journaled, conflict-safe, single-writer).
   The router learns endpoints the way it learns everything else.
4. **Budget-arbitrated residency.** The API is `ensure(model)` /
   `release(model)` with budget consultation — never "start these N blind."
   Spark owns eviction; consumers request, never assume.
5. **Security posture:** loopback bind only; no secrets in argv (existing
   `secret_env` pattern); per-child log routing per the vibes logging
   standard; endpoint list is not sensitive but is also not exposed off-host.

## 4. The hard problem (name it, don't hide it)

**Residency negotiation.** The router's cascade needs T2 *now* mid-climb;
spark may have T0/T1/embedder resident and something else warm for another
consumer. Who waits, who evicts, what's the contract? Options to explore in
S4: (a) synchronous `ensure()` that blocks until resident (router keeps its
current semantics, latency moves into spark); (b) lease/priority model
(router holds a lease per climb); (c) spark exposes eviction *events* and the
router keeps its rewarm-at-evicting-route's-tail pattern against them.
Patchwork Exp 4 economics and the embedder-eviction findings (BENCH-REPORT
v0.2-mlx: mmap is mmap, mlx bought no immunity) are the empirical base, but
they were measured **in-process** — separate server processes change the
numbers (S2).

## 5. Spikes & experiments (run these before design.md)

- [ ] **S1 — Daemon skeleton spike.** Minimal `sparkd`: job table of 2
      supervised `mlx_lm.server` children (T0+T1), `GET /endpoints`,
      start/stop. Goal: find out what breaks lifting Supervisor from
      one-attached-child to N-detached (signals, log fan-out, orphan reaping,
      port collisions on restart). Timebox: a day. Artifact:
      `spikes/fleet-daemon.md` + throwaway branch.
- [ ] **S2 — Cross-process residency economics.** Re-run the patchwork Exp 4
      matrix with tiers as *separate server processes*: T0+T1 co-residency
      throughput, T2 load-with-others-parked, evict/rewarm timing, and the
      embedder's fate during a T2 climb. In-process numbers do NOT transfer
      (separate Metal contexts, separate mmaps). Artifact: results table in
      `spikes/`, compared against patchwork `swap-econ.results.jsonl`.
- [ ] **S3 — HTTP tier latency tax.** Route the patchwork battery with one
      tier behind `mlx_lm.server` HTTP vs in-process `stream_generate`:
      per-request overhead, streaming behavior under the router's SSE seam
      (comment keep-alives, verify-then-emit), timeout/retry semantics on a
      wedged child. Expected ~ms vs generation-seconds, but *measure*, and
      find the failure modes (the router currently never sees a dead tier —
      HTTP makes that a real state).
- [ ] **S4 — Residency negotiation paper spike.** Write the contract options
      from §4 against S2's numbers; pick one; pressure-test against the
      three consumers we know about (router cascade, ad-hoc `spark` CLI use,
      future 0003-orchestrator sampling). Paper only — this decision shapes
      everything and should not be prototyped first.
- [ ] **S5 — Roster actuation dry run.** Hand-write a tier roster update
      through darkcore `set_config` pointing one tier at an S1 endpoint;
      verify conflict behavior, journal entry, and router pickup without
      restart. Proves the handoff channel end-to-end before any glue code.
- [ ] **S6 — Embedder as first fleet tenant** (promoted from open question 5,
      operator 2026-07-18). Serve the mlx bge-small embedder
      (`patchwork/experiments/router/darkcore/embedder_mlx.py`) as a sparkd
      endpoint (`POST /embed`, ~33 M params — no OpenAI server exists for
      encoder-only, so this is a ~50-line loopback wrapper) and point the
      router's predictor at it behind a fallback-to-in-process switch. Why
      the embedder makes the ideal first tenant: it is the component whose
      residency keeps being violated by forces outside its own process
      (BENCH-REPORT v0.2-mlx: the 27B evicts it, mlx or not), it is tiny
      (cheap to be wrong about), and it exercises the full ensure/release
      contract — spark must keep it warm *through* a T2 climb, which is the
      exact negotiation S4 must solve, at 1/200th the weight of a real tier.
      Measures: embed latency via HTTP vs in-process (baseline 9.4 ms warm,
      fixtures/embedder-parity timings), post-T2-climb latency (does spark
      ownership beat the 22.31 ms rewarm?), and parity (the frozen reference
      vectors gate the served path exactly as they gated the port).
      **Sequencing: run S6 right after S1** — it is the smallest end-to-end
      proof of the whole idea, and an S6 failure is the cheapest possible
      falsification of fleet serving on this box.

## 6. Router-side work (tracked here, implemented in patchwork)

- HTTP tier backend behind the `Tier` boundary
  (`patchwork/experiments/router/plans/generalized-router-interfaces.md`) —
  this spec is that plan's first concrete customer. `generate(query, context,
  budget) -> Attempt` over an OpenAI client; residency calls per S4's chosen
  contract.
- `ModelPool` keeps the in-process path as the zero-config default —
  dark-operability is a patchwork invariant; a router with no sparkd present
  must still work.

## 7. Open questions (blocking design.md)

1. API transport: bare HTTP+JSON vs reusing the OpenAI-ish conventions
   consumers already speak? (Leaning bare+boring; it's a control API.)
2. Does sparkd own the *router's* process too (extend seam 5), or is the
   router a peer consumer? (Peer keeps planes separate; owner simplifies
   ops. Genuinely unsettled.)
3. Persistence: does the job table survive sparkd restart (state file +
   re-adopt children) or is restart = fleet restart? (Re-adoption is the
   operationally right answer and the implementation-expensive one.)
4. Multi-consumer fairness: first-come or priority classes? (Defer until a
   second real consumer exists; note it in S4.)
5. ~~Does the embedder belong in the fleet as a served endpoint?~~ —
   PROMOTED to spike S6 (operator, 2026-07-18): the embedder is the first
   fleet tenant; its eviction becomes spark's problem. The residual open
   part: if S6 succeeds, does the in-process embedder path stay as the
   dark-operable fallback forever (leaning yes — same invariant as tiers)?

## 8. Non-goals (v1)

- Off-host serving, TLS, authn — loopback only.
- Autoscaling, GPU partitioning, quantization management (registry's job).
- Replacing the interactive `spark` CLI — it stays; the daemon is additive.
- Any scheduling smarter than budget-gated ensure/release + one policy from
  S4. No bandit/RL residency policies; that is 0003-orchestrator territory
  if it ever exists.

## 9. References

- `../patchwork/docs/routing-architecture.md` — planes, control
  surface as hard interface (the pattern this spec extends to model serving)
- `../patchwork/experiments/router/BENCH-REPORT.md` — v0.2-mlx
  (embedder eviction persists), the S4 saga
- `../patchwork/experiments/router/swap_econ.py` + results —
  in-process residency economics baseline for S2
- `src/spark/runner/supervisor.py`, `src/spark/budget.py`,
  `src/spark/runtimes/relay.py` — the reuse surface
- README §9–10 — external couplings + decision log (binding on any daemon
  work; the zsh/Homebrew/Keychain integrations must not break)
