# 0003 — sd.cpp Image Runtime (stable-diffusion.cpp as a supervised spark runtime)

> **STATE: WIP / RESEARCH (feature proposal).** Not a design, not an
> implementation. This document records a potential feature — wiring
> `stable-diffusion.cpp` (`sd-server`) into spark as a supervised runtime so
> image generation becomes a first-class spark model — with the verified
> upstream facts, the contract seam it forces open, and the spikes that must
> run before any design.md. Do not implement from this document; run the
> spikes.
>
> Origin: 2026-09-23 session (operator + agent, after pulling
> Qwen-Image-2.1 GGUF weights on this host). Precedent: the `darkcore` runtime
> seam (`config/runtimes/darkcore.toml`) — a supervised child that speaks
> OpenAI-compatible HTTP, driven by the generic `Backend`.
>
> Operator decision: **spark development is deferred.** This spec exists so the
> seam is written down while the evidence is fresh. The interim workflow is
> agent-side (see §9).

---

## 1. Problem statement

spark fronts every local inference runtime on this machine, and all of them
answer the same question: text in, text out. `sd-server` answers a different
one — text in, **image** out — while being structurally identical to the
runtimes spark already supervises: one long-running child process, one HTTP
socket, one readiness condition.

That structural similarity is the whole reason this is worth a spec rather than
a shrug. The friction is not technical; it is contractual. spark publishes a
machine-readable endpoint manifest whose `api_contract` is a single hardcoded
constant, `"openai.chat-completions.v1"` (`src/spark/catalog.py:49`). An image
endpoint is not a chat-completions endpoint, and the manifest's field names are
an interface ("Renaming a manifest field silently breaks every consumer that
validates it", README §9.8). Admitting a second protocol is therefore a
deliberate interface change, not a config edit.

Secondary friction, and the reason this is gated on spikes rather than
scheduled: the model is **not one artifact**. A working Qwen-Image-2.1 pipeline
needs three files — the denoiser, a VAE, and a text encoder — where every
existing spark registry entry assumes one. And on a 16 GB host whose measured
idle state is already 6.2 GiB into swap, D8's launch gate may correctly refuse
the whole thing (see §4).

## 2. What already exists (do not rebuild)

- **`stable-diffusion.cpp` ships a server, not just a CLI.** `examples/server/`
  contains `main.cpp`, `routes_openai.cpp`, `routes_sdapi.cpp`,
  `routes_sdcpp.cpp`, `async_jobs.cpp`. `docs/qwen_image_2.1.md` documents the
  model; `examples/server/api.md` (1351 lines) documents the HTTP surface.
- **Day-0 support for Qwen-Image-2.1**, announced 2026/09/20 in the upstream
  README. This is an upstream concern, not something spark builds.
- **spark's generic `Backend`.** `base.py:Backend` is a server launcher driven
  entirely by a `RuntimeDef`; unknown runtime names fall back to the generic
  subclass, so a new runtime is **data, not code** (CLAUDE.md: "Config is data,
  not code").
- **`launch_overrides` (D9).** Appended as real CLI flags
  (`{"--vae":"/p/vae.safetensors"}` → `["--vae","/p/vae.safetensors"]`). This
  is the existing escape hatch for the multi-artifact problem.
- **`darkcore` precedent.** A supervised child speaking OpenAI-compatible HTTP
  with a `/health` endpoint and `model_format = "any"`, requiring no `Backend`
  subclass. This is the shape the sd.cpp entry would take.
- **The weights are already on this host** (see §4) — fetching is not the
  blocker.

## 3. Committed intent (stable even while WIP)

1. **Wrapping, not reimplementing.** spark supervises `sd-server` as a child
   process. It never links, imports, or reimplements diffusion code — same rule
   as D1.
2. **Runtime def stays TOML.** `config/runtimes/sd_cpp.toml` is the deliverable;
   `src/` changes only if a spike proves TOML insufficient.
3. **The contract change is explicit and versioned.** No image endpoint is
   published under `openai.chat-completions.v1`.
4. **The launch gate stays honest.** If the memory gate refuses on a 16 GB host,
   that is the feature working. It is not a bug to be worked around with a
   runtime-specific exemption, unless a spike justifies the exemption in writing.
5. **No consumer, no ship.** A producer with no reader is plumbing to nowhere.
   Ship the consumer first (see §9, and S5).

## 4. Verified host facts (2026-09-23) — the gates live here

**Upstream (read from source, not assumed):**

- `sd-server` HTTP surface: OpenAI `POST /v1/images/generations`,
  `POST /v1/images/edits`, `GET /v1/models`; A1111 `POST /sdapi/v1/txt2img`,
  `img2img`, plus `loras`/`samplers`/`sd-models`; native
  `POST /sdcpp/v1/img_gen`, `GET /sdcpp/v1/jobs/{id}`,
  `POST /sdcpp/v1/jobs/{id}/cancel`, `POST /sdcpp/v1/vid_gen`.
- Launch shape: `--diffusion-model <gguf|safetensors> --vae <safetensors>
  --llm <gguf|safetensors>` plus `--diffusion-fa`, `--offload-to-cpu`,
  `--cfg-scale`. Transport: `--listen-ip`, `--listen-port` (default 1234). Web
  UI served at `/`.
- **Prefix cache is on by default, FP32, and costs ~4 GiB per condition** at a
  4096-token prefix, *in addition to weights and working buffers*. At the
  documented `--cfg-scale 6.0` both positive and negative conditions are cached
  — ~8 GiB of overhead. It can be disabled with
  `--model-args qwen_image_2_1_prefix_cache=false`. The runner releases the
  caches on OOM and retries the full sequence once, so degradation is graceful.
- Resolutions must be divisible by 32. Alpha is prompt-driven and survives only
  in `.png`/`.webp`.
- The release CI builds top-level (`SD_STANDALONE` ON → `SD_BUILD_EXAMPLES`
  ON) and zips all of `build/bin/*`, so the macOS arm64 artifact should contain
  **both** `sd-cli` and `sd-server`. Unverified — see S1.

**This host:**

- Apple M2, 16 GB unified, tight disk. Measured 2026-09-23: free RAM 0.08 GiB,
  3.98 GiB compressed, swap **6220 MB of 7168 MB used**, boot volume 96% full
  with 19 GiB free. Blender alone held 6399 MB; a VM held 2276 MB.
- Qwen-Image-2.1 weights already present on the external volume
  (`/Volumes/Games/Models/models--unsloth--Qwen-Image-2.1-GGUF/`, 12 quants,
  60.38 GiB, checksum-verified). VAE and text encoder are **not** yet local.
- Realistic runnable set on this host: denoiser 2.95 GiB (Q3_K_M) + encoder
  2.90 GiB (UD-IQ2_M) + VAE 0.63 GiB ≈ 6.48 GiB of weights. With the prefix
  cache off, that is the difference between "runs" and "swaps to death".

## 5. Spikes & experiments (run before design.md)

- **S1 — Artifact inventory. ✅ SETTLED 2026-09-23.** Listed the macOS arm64
  release zip (`master-908-88411ef`) without installing: it contains **both**
  `sd-cli` (3.5 MB) and `sd-server` (5.0 MB), plus `libstable-diffusion.dylib`
  (76 MB). No build step is required on this host. Both are installed at
  `~/.local/opt/sdcpp/` via wrapper scripts (not symlinks — `@rpath/$ORIGIN`
  must still resolve the dylib).
- **S2 — Generic-Backend dry run.** Hand-write `config/runtimes/sd_cpp.toml`
  and launch under the generic `Backend` with a registry entry. Prove the
  `args_template` can express `--listen-ip/--listen-port` and that
  `openai_compatible = true` does not mislead the supervisor.
- **S3 — Readiness semantics.** spark's launch gate does a *real* generation for
  LLM runtimes. For a diffusion server that means rendering an image — tens of
  seconds. Determine whether `GET /sdcpp/v1/capabilities` or `/v1/models`
  suffices, given README §8's warning that a 200 on `/v1/models` does not mean
  the generator works.
- **S4 — Memory profile.** Measure peak RSS for the 6.48 GiB set at 1024×1024
  and 2048×2048, prefix cache on and off, with the host otherwise idle. These
  are the numbers D8's gate needs; without them any gate is a guess.
- **S5 — Consumer identification.** Establish who actually calls
  `/v1/images/generations`. Candidates: an agent tool (Hermes), a future
  ComfyUI-style front end (operator's stated later direction), or nothing.
- **S6 — Contract shape.** Decide and write down: a second `api_contract` value
  (`openai.images.v1`), a per-instance contract field, or images excluded from
  the endpoint manifest entirely.

## 6. Open questions (blocking design.md)

1. Is `api_contract` one value per spark, or per published instance? The
   constant at `catalog.py:49` currently implies the former.
2. Does the registry grow multi-artifact entries, or is `launch_overrides`
   carrying `--vae`/`--llm` acceptable long-term? (D9 makes it work today; it is
   not obviously the right final home for two of three required files.)
3. What is the cheap-but-sufficient readiness probe for a diffusion server?
4. Does the image endpoint belong in spark's endpoint manifest at all, or is
   spark the wrong layer — i.e. is this a *model* to supervise, or a *tool* to
   invoke?
5. Does D8's memory gate need a per-runtime threshold, given that a diffusion
   pipeline's working set is dominated by activations and caches rather than
   weights?

## 7. Non-goals (v1)

- No ComfyUI-equivalent UI, no photo editor, no node graph. Operator's stated
  direction is to build those later; this spec covers only the supervised
  runtime seam.
- No LoRA, no upscalers, no AnimateDiff/video (`/sdcpp/v1/vid_gen` exists
  upstream and is out of scope).
- No multi-host/RPC distribution (`docs/rpc.md` upstream).
- No changes to `specs/0001-model-fleet-api`, which remains WIP/RESEARCH and
  is not to be implemented from its document.

## 8. References

| Source | What it settles |
|---|---|
| upstream `README.md` | day-0 Qwen-Image-2.1 support, 2026/09/20 |
| upstream `docs/qwen_image_2.1.md` | launch command, 3-file requirement, prefix-cache cost, α-channel and resolution rules |
| upstream `examples/server/api.md` | the three HTTP surfaces and their routes |
| upstream `examples/server/README.md` | `sd-server` launch flags, default port 1234 |
| upstream `CMakeLists.txt` | `SD_STANDALONE` → `SD_BUILD_EXAMPLES`: why both binaries should ship |
| spark `README.md` §8 | `Backend`/`RuntimeDef` contract, `launch_overrides`, readiness caveat |
| spark `README.md` §9.8 | the published record — manifest fields are an interface |
| spark `src/spark/catalog.py:49` | the single `API_CONTRACT` constant |
| spark `specs/0002` §7 | diffusion previously scoped out: "No FM_Visual / diffusion" |

## 9. Interim workflow (not spark)

Until this spec graduates, image generation on this host is **agent-side and
deliberately outside spark**: an agent-level tool that invokes `sd-cli`
one-shot, with no server, no supervised endpoint, and no contract to change.

That placement is not a stopgap for convenience — it is the correct layer under
§3.5 and §6.4. A one-shot invocation needs none of spark's machinery: no
lifetime management, no readiness probe, no published manifest. spark earns its
keep only when the image endpoint has a consumer and a contract.

If the agent-side tool proves the need, S5 and S6 produce the evidence this
spec needs to argue for the runtime.
