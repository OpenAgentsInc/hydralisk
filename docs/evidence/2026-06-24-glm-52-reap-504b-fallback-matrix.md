# GLM-5.2 REAP fallback matrix

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/90

Profile:
[`profiles/glm-5.2-reap-504b-b12x-g4.json`](../../profiles/glm-5.2-reap-504b-b12x-g4.json)

Public-safety boundary: this packet contains model identifiers, hardware lane
names, gate requirements, and claim language only. It contains no secrets, raw
prompts, raw responses, private source, hidden reasoning traces, weights,
checkpoints, compiled engines, profiler dumps, or raw benchmark output.

## Source facts

- Primary model: `0xSero/GLM-5.2-504B`
- Primary model card: <https://huggingface.co/0xSero/GLM-5.2-504B>
- The 504B model card describes this as the flagship GLM-5.2 REAP/NVFP4 cut,
  keeps 168 routed experts per layer, and recommends vLLM with
  `modelopt_fp4`, `fp8` KV cache, and a long-context serving target.
- Model-size fallback: `0xSero/GLM-5.2-NVFP4-REAP-469B`
- 469B model card:
  <https://huggingface.co/0xSero/GLM-5.2-NVFP4-REAP-469B>
- The 469B card identifies it as REAP-pruned GLM-5.2 NVFP4, retaining 61.4% of
  the original parameters, with 156 routed experts per layer.

## Rule

Do not silently change the public claim. Every serving or eval claim must name:

- exact model repository and revision;
- exact GPU family and count;
- whether the host is a standalone 4x G4 shape or an 8x fallback host;
- whether all 8 GPUs are used or only four selected GPUs are used;
- context/concurrency ceiling;
- sampler settings;
- benchmark version and denominator definitions for eval claims.

The current live result remains the 504B model using four selected G4 GPUs
inside an admitted 8x fallback host. That does not prove standalone 4x
`g4-standard-192` capacity.

## Fallback order

| Order | Lane | Trigger | Claim wording |
| ---: | --- | --- | --- |
| 1 | 504B on 4x G4 standalone | Default path; retry when `g4-standard-192` capacity is available | `504B REAP/NVFP4 on standalone 4x RTX PRO 6000 G4` |
| 2 | 504B on 8x G4 fallback | 4x G4 capacity unavailable, or explicit 8-GPU runtime experiment needed | `504B REAP/NVFP4 on 8x RTX PRO 6000 G4 fallback` |
| 3 | 469B on 4x G4 | 504B fails admission, load, smoke, tuning, or eval on the required 4x G4 envelope | `469B REAP/NVFP4 on 4x RTX PRO 6000 G4` |
| 4 | 504B on H200/B200 | G4 kernel/runtime/context blocker needs premium-hardware proof | `504B REAP/NVFP4 on H200/B200; not the accessible G4 claim` |

## Lane requirements

### 1. Primary 504B 4x G4

Status: `capacity_blocked_for_standalone_4x`, with 4-GPU runtime envelope
tested on the admitted 8x host.

Trigger to use:

- Always first for the accessibility claim.
- Retry when GCE can admit `g4-standard-192` with 4 x `nvidia-rtx-pro-6000`.

Required gates:

- Admission: create a fresh standalone 4x G4 host and record zone, machine
  type, accelerator, provisioning model, disk, and private IP.
- Staging: prove the pinned 504B revision is complete on the host or durable
  attached disk.
- Launch: use the b12x/vLLM recipe with TP=4, DCP=4, `modelopt_fp4`,
  `kv-cache-dtype=fp8`, `B12X_MLA_SPARSE`, `b12x`, and GLM parser flags.
- Smoke: `/v1/models` ready and a public-safe completion receipt.
- Tuning: 250K context, `max_num_seqs=2`, `max_num_batched_tokens=4096`, and
  MTP-2/no-`min_p` admitted by the 2026-06-25 speed gate.
- Eval: Terminal-Bench receipt with task statuses, denominator definitions,
  and sanitized error classes only.

### 2. Same 504B model on 8x G4

Status: `capacity_fallback_admitted`; current host is
`hydralisk-glm52-reap-504b-g4-8g-b-20260624214500`.

Trigger to use:

- 4x standalone G4 capacity remains unavailable.
- 4x runtime is healthy but operational needs require more scheduler headroom.
- A deliberate 8-GPU experiment is needed to test full-context concurrency or
  longer context.

Required gates:

- Admission: record that this is `g4-standard-384`, not a 4x host.
- Staging: reuse the 504B staged checkpoint only if the manifest still matches
  the pinned revision and shard count.
- Launch: if using only GPUs `0,1,2,3`, keep the 4-GPU claim boundary. If using
  all eight GPUs, write a new launch receipt with `GPU_DEVICES=0,1,2,3,4,5,6,7`
  and `TP_SIZE=8`.
- Smoke: separate 8-GPU smoke if all eight GPUs are used.
- Tuning: separate context/concurrency tuning; do not inherit 4-GPU ceilings.
- Eval: separate Terminal-Bench receipt and denominator definitions.

Public language must say `8x G4 fallback` whenever all eight GPUs, the 8x host,
or 8x scheduler capacity is material to the result.

### 3. 469B REAP model on 4x G4

Status: `not_started`.

Trigger to use:

- 504B cannot pass a required gate on the primary 4x G4 envelope.
- 504B can serve only with an 8x or premium-hardware condition that would defeat
  the accessible-hardware claim.
- Cost, context, or reliability pressure makes the smaller model the better
  public candidate.

Required gates:

- Profile: create a distinct `glm-5.2-reap-469b` profile and aliases; do not
  reuse the 504B profile ID or `openagents/glm-5.2-reap-504b` alias.
- Admission: fresh 4x G4 admission evidence, or an explicit capacity-blocked
  note if only the existing 8x fallback host is used for early runtime proof.
- Staging: pin the exact 469B revision, shard count, total bytes, and
  repository metadata.
- Launch: b12x/vLLM recipe with model-specific routed expert count and DSA
  index pattern verified from the 469B config.
- Smoke: public-safe `/v1/models` and completion receipt.
- Tuning: repeat context, concurrency, MTP, batch-token, and sampler gates; do
  not copy 504B limits.
- Eval: separate Terminal-Bench receipt; never compare 469B results as if they
  were 504B.

Public language must say `469B` in the model name and claim.

### 4. Premium H200/B200 proof lane

Status: `not_started_for_reap_504b`.

Trigger to use:

- G4 has a kernel, memory, context, or scheduler blocker that may be hardware
  specific.
- We need to distinguish model/runtime correctness from G4 availability.
- We need a long-context proof beyond the current 250K G4 ceiling.

Required gates:

- Admission: H200/B200 quota, machine type, GPU count, zone, and cost/blast
  radius must be recorded.
- Staging: pinned 504B revision and complete manifest on that host class.
- Launch: vLLM/b12x or another explicitly pinned runtime compatible with the
  hardware.
- Smoke: public-safe health/models/completion receipt.
- Tuning: separate context/concurrency/MTP receipts; do not backport the result
  to a G4 claim.
- Eval: separate Terminal-Bench receipt and denominator definitions.

Public language must say this is a premium-hardware proof and not the accessible
4x RTX PRO 6000 claim.

## Evidence templates

Fallback lanes should reuse these packet shapes:

- Profile/evidence contract:
  [`2026-06-24-glm-52-reap-504b-profile.md`](2026-06-24-glm-52-reap-504b-profile.md)
- Admission:
  [`2026-06-24-glm-52-reap-504b-g4-admission.md`](2026-06-24-glm-52-reap-504b-g4-admission.md)
- Staging:
  [`2026-06-24-glm-52-reap-504b-staging.md`](2026-06-24-glm-52-reap-504b-staging.md)
- Launch:
  [`2026-06-24-glm-52-reap-504b-b12x-launch-profile.md`](2026-06-24-glm-52-reap-504b-b12x-launch-profile.md)
- Smoke:
  [`2026-06-24-glm-52-reap-504b-load-smoke.md`](2026-06-24-glm-52-reap-504b-load-smoke.md)
- Private endpoint:
  [`2026-06-24-glm-52-reap-504b-private-endpoint.md`](2026-06-24-glm-52-reap-504b-private-endpoint.md)
- Tuning:
  [`2026-06-24-glm-52-reap-504b-tuning.md`](2026-06-24-glm-52-reap-504b-tuning.md)
- Terminal-Bench:
  [`2026-06-24-glm-52-reap-504b-terminal-bench-20.md`](2026-06-24-glm-52-reap-504b-terminal-bench-20.md)

Each fallback result should create new evidence files with the lane name in the
filename rather than editing a 504B/4x G4 result to mean something else.

## Stop conditions

Stop and create a new lane instead of mutating the primary profile when:

- the model repository changes from 504B to 469B;
- the GPU count changes from four selected GPUs to all eight GPUs;
- the accelerator family changes from RTX PRO 6000 to H200/B200;
- context/concurrency ceilings change because of different hardware;
- benchmark agent, retry policy, or denominator definitions change.
