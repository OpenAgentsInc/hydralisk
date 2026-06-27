# DeepSpec / DSpark Speculative-Decoding Adoption Audit (Hydralisk)

- Date: 2026-06-27
- Scope: assess the extent to which Hydralisk could benefit from adopting
  DeepSpec / DSpark style draft-model speculative decoding.
- Repo role (per `AGENTS.md`): Hydralisk is the standalone OpenAgents
  Python/NVIDIA inference lane — vLLM, SGLang, TensorRT-LLM, model profiles,
  evals, public-safe receipts. It consumes engines and checkpoints; it is not
  Psionic-native. That makes it the lane that can adopt DSpark **fastest**
  (drop-in checkpoint / engine flag) but **not** the lane that would reimplement
  the algorithm. The Rust-native implementation question is covered in the
  parallel Psionic audit.

## Source material

- Reference clone: `projects/repos/deepspec/` (DeepSeek-AI, read-only).
- Paper: `projects/repos/deepspec/DSpark_paper.pdf` — *"DSpark:
  Confidence-Scheduled Speculative Decoding with Semi-Autoregressive
  Generation."*
- DeepSpec open-sources **Eagle3**, **DFlash**, and **DSpark** drafters plus a
  train/eval pipeline, and publishes **DSpark draft checkpoints for
  DeepSeek-V4-Flash (preview) and DeepSeek-V4-Pro (preview)**.

### What DSpark is (one paragraph)

A draft model proposes a block of `γ` tokens; the target verifies the whole
block in one pass and accepts the longest distribution-consistent prefix
(lossless). DSpark adds (1) **semi-autoregressive generation** — a parallel
DFlash-style backbone for first-token capacity plus a cheap sequential head
(default: low-rank **Markov head**, `r=256`) that injects intra-block token
dependency to stop suffix decay, costing only **+0.2–1.3%** round latency; and
(2) **confidence-scheduled verification** — a confidence head (STS-calibrated)
feeding a **Hardware-Aware Prefix Scheduler** that picks a *per-request* verify
length to maximize `Θ = τ·SPS(B)`, so target batch capacity is spent only on
high-expected-return tokens. Headline production result: vs an **MTP-1**
baseline in DeepSeek-V4 serving, **+60–85%** per-user speed on V4-Flash and
**+57–78%** on V4-Pro at matched throughput, and it removes the throughput
cliff under strict-SLA / high-concurrency tiers.

## Hydralisk current state — already shipping speculative decoding

Unlike Psionic, Hydralisk **already runs production speculative decoding** via
vLLM's native MTP path. This is the most important framing fact: the question is
not "should we adopt spec decoding" (we have) but "where does DSpark beat what
we run today."

- **MTP-2 live on GLM-5.2-REAP-504B (private speed canary).**
  - Launch: `scripts/launch-glm-52-reap-504b-b12x-gce.sh:29-30,145-157`
    (`MTP=1`, `NUM_SPECULATIVE_TOKENS=2`, injects vLLM
    `--speculative-config {"method":"mtp","num_speculative_tokens":2}`).
  - Profile: `profiles/glm-5.2-reap-504b-b12x-g4.json:545,570-579` records
    `mtp: true`, `numSpeculativeTokens: 2`, and that **`min_p` must be omitted**
    because "vLLM speculative decoding rejects min_p/logit_bias."
  - Evidence: `docs/evidence/2026-06-25-glm-52-reap-504b-mtp2-speed-gate.md` —
    MTP-2 improved decode throughput by **~+31–34%** (33.86→44.25 tok/s @160,
    35.19→47.23 tok/s @512), TTFT 0.251s→0.289s.
  - Config plumbing: `hydralisk/serve/config.py:92,177`
    (`speculative_decoding` field, `HYDRALISK_SPECULATIVE_DECODING` env),
    serialized into receipts at `hydralisk/serve/receipts.py:206-207`.
- **EAGLE configured but blocked** on GLM-5.2-FP8 (SGLang):
  `profiles/glm-5.2-fp8-sglang.json:164-169` (`algorithm: EAGLE`,
  `draftTokens: 6`) — `not_reached_on_g4_load_smoke` (DSA kernel / arch block).
- **No speculative decoding on two lanes that could use it:**
  - `gpt-oss-20b` on L4 (live public) — no MTP/draft config.
  - `deepseek-v4-flash-gce-preflight.json` / `deepseek-v4-fable-adapter-g4.json`
    — **no `mtp`/`speculative`/`draft` keys at all** despite being the
    DeepSeek-V4-Flash lane.
- **Single-flight today:** `max_num_seqs=2`, `singleflight=1` for the 504B lane,
  so the *concurrency* regime DSpark's scheduler targets is not yet exercised.

## Benefit assessment

**Overall: HIGH and unusually concrete**, because Hydralisk already serves the
exact model DSpark ships drop-in checkpoints for, and already has the spec-decode
config/receipt plumbing.

### 1. DeepSeek-V4-Flash lane — the direct, highest-confidence win

Hydralisk has a DeepSeek-V4-Flash profile (`profiles/deepseek-v4-flash-*.json`)
with **no speculative decoding configured**, while DSpark publishes a
**DeepSeek-V4-Flash draft checkpoint** and reports **+60–85% per-user speed vs
MTP-1** on exactly that model in production. This is the cleanest opportunity in
the workspace: a released, model-matched drafter for a lane we already intend to
serve. Subject to engine support for loading the DSpark drafter (vLLM/SGLang
EAGLE-style external draft, or DeepSeek's own serving path), this is a
benchmark-and-adopt task, not a research task.

### 2. GLM-5.2-504B — train a DSpark drafter to beat MTP-2

Our current GLM win is vLLM **MTP** (the model's own multi-token-prediction
heads). DSpark's whole premise is beating MTP: offline it raises accepted length
~16–30% over DFlash/Eagle3, and in production it beat MTP-1 by 60–85%. There is
**no released DSpark checkpoint for GLM-5.2**, so capturing this means *training*
a DFlash/DSpark drafter for GLM via DeepSpec — heavier, and the right owner for
"train a Rust-native drafter" is Psionic, but DeepSpec's Python pipeline
(`projects/repos/deepspec/{train.py,eval.py}`) is usable here as a reference
backfill if we want a vLLM/SGLang-loadable EAGLE-style draft quickly.

### 3. gpt-oss-20b (L4) — enable *any* spec decoding

This lane has none today. Even vanilla EAGLE/Eagle3 (DeepSpec ships Eagle3) or
MTP-if-available would plausibly add 20–35% decode throughput on a latency-
sensitive public lane. Low risk, good evidence value.

### 4. Confidence-scheduled verification — the idea that maps onto our pain

DSpark's hardware-aware prefix scheduler exists to stop spec decoding from
*degrading* throughput under concurrency by wasting verify capacity on
low-confidence suffix tokens. Our profiles already show the symptoms this
targets: GLM 504B falls off a cliff at concurrency >1 (held to single-flight),
and we already omit `min_p` to keep MTP legal. The scheduler is the part of
DSpark most relevant to our cost/SLA story — **but** it was deployed inside
DeepSeek's own serving stack (HAI-LLM), not vanilla vLLM/SGLang. Adopting it
means either waiting for engine support or running DeepSeek's serving path; it
should not be on the near-term Hydralisk roadmap as a from-scratch build.

## Honest caveats / constraints

- **Engine support gates everything.** DSpark drafters and especially the
  confidence scheduler need the serving engine to load an external draft model
  and (for the scheduler) to support dynamic per-request verify lengths. vLLM
  MTP and EAGLE are supported; a bespoke DSpark drafter + scheduler may not be
  without DeepSeek's stack. Confirm loadability before promising numbers.
- **`min_p` / sampling incompatibility is real** and already bit us
  (`profiles/glm-5.2-reap-504b-b12x-g4.json:578-579`). Any new spec lane must
  re-validate sampling-parameter compatibility and keep a non-spec fallback.
- **No checkpoint for GLM/gpt-oss** — those require training (heavy; target-
  response regeneration is ~38 TB for Qwen3-4B per DeepSpec's data README).
- **AGENTS boundary:** weights/checkpoints/engines are not committed here
  (`AGENTS.md` Invariants). Adoption work stays as profiles, launch scripts,
  evals, and receipts; drafter *training* belongs in the Rust substrate
  (Psionic) or as referenced external artifacts, not vendored into this repo.
- **Single-flight masks the scheduler's value.** Until we run real concurrency,
  the confidence scheduler has little to reclaim; prioritize drafter quality
  (`τ`) over the scheduler.

## Recommendation (ordered by effort-adjusted payoff)

1. **DeepSeek-V4-Flash drop-in DSpark drafter.** Verify the released
   DeepSeek-V4-Flash DSpark checkpoint can be loaded by our intended engine,
   add a profile + launch config, and run the standard speed gate vs the
   no-spec baseline. Highest confidence, model-matched, smallest lift.
2. **Enable spec decoding on gpt-oss-20b (L4).** Try MTP/EAGLE; gate with a
   speed-gate evidence doc like the GLM MTP-2 gate. Low risk.
3. **Benchmark a DSpark/DFlash drafter for GLM-5.2 vs current MTP-2.** Use
   DeepSpec (`train.py`/`eval.py`) as reference to produce an EAGLE-style draft
   loadable by vLLM/SGLang; only pursue if engine support is confirmed.
   Coordinate the "owned, Rust-native drafter" track with Psionic.
4. **Defer the confidence-scheduled verifier** until (a) we run real
   concurrency and (b) the serving engine supports dynamic per-request verify
   length. Track it as the eventual answer to our concurrency throughput cliff.

Bottom line: Hydralisk doesn't need convincing that spec decoding pays — it's
already +31–34% on GLM via MTP-2. DSpark's concrete value here is (a) a
model-matched, released drafter for the DeepSeek-V4-Flash lane that currently
has none, and (b) a credible path to beat MTP on GLM, with the hardware-aware
scheduler as a longer-horizon answer to the concurrency cliff.

## Key references

- Paper: `projects/repos/deepspec/DSpark_paper.pdf`
- DeepSpec pipeline: `projects/repos/deepspec/{README.md,train.py,eval.py}`
- Live MTP-2 evidence: `docs/evidence/2026-06-25-glm-52-reap-504b-mtp2-speed-gate.md`
- GLM profile spec section: `profiles/glm-5.2-reap-504b-b12x-g4.json:545,570-579`
- EAGLE (blocked) profile: `profiles/glm-5.2-fp8-sglang.json:164-169`
- DeepSeek-V4-Flash lane (no spec config): `profiles/deepseek-v4-flash-gce-preflight.json`
- gpt-oss-20b lane (no spec config): see `profiles/` / L4 runbook
- Config/receipt plumbing: `hydralisk/serve/config.py:92,177`, `hydralisk/serve/receipts.py:206-207`
- Launch injection: `scripts/launch-glm-52-reap-504b-b12x-gce.sh:29-30,145-157`
