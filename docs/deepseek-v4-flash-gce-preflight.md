# DeepSeek-V4-Flash GCE preflight

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/5

Profile:
[`profiles/deepseek-v4-flash-gce-preflight.json`](../profiles/deepseek-v4-flash-gce-preflight.json)

Evidence:
[`docs/evidence/2026-06-24-deepseek-v4-flash-gce-preflight.md`](evidence/2026-06-24-deepseek-v4-flash-gce-preflight.md)

Load-smoke evidence:
[`docs/evidence/2026-06-24-deepseek-v4-flash-gce-load-smoke.md`](evidence/2026-06-24-deepseek-v4-flash-gce-load-smoke.md)

Backend-matrix evidence:
[`docs/evidence/2026-06-24-deepseek-v4-flash-g4-backend-matrix.md`](evidence/2026-06-24-deepseek-v4-flash-g4-backend-matrix.md)

Scaled-mm probe evidence:
[`docs/evidence/2026-06-24-deepseek-v4-flash-scaled-mm-g4-probe.md`](evidence/2026-06-24-deepseek-v4-flash-scaled-mm-g4-probe.md)

E8M0 upcast evidence:
[`docs/evidence/2026-06-24-deepseek-v4-flash-e8m0-upcast-g4.md`](evidence/2026-06-24-deepseek-v4-flash-e8m0-upcast-g4.md)

o_proj evidence:
[`docs/evidence/2026-06-24-deepseek-v4-flash-o-proj-g4.md`](evidence/2026-06-24-deepseek-v4-flash-o-proj-g4.md)

Grouped o_proj RHS evidence:
[`docs/evidence/2026-06-24-deepseek-v4-flash-o-proj-group-rhs-g4.md`](evidence/2026-06-24-deepseek-v4-flash-o-proj-group-rhs-g4.md)

Grouped o_proj RHS scale-mode evidence:
[`docs/evidence/2026-06-24-deepseek-v4-flash-o-proj-rhs-scale-g4.md`](evidence/2026-06-24-deepseek-v4-flash-o-proj-rhs-scale-g4.md)

Provider-stack evidence:
[`docs/evidence/2026-06-24-deepseek-v4-flash-provider-stack-g4.md`](evidence/2026-06-24-deepseek-v4-flash-provider-stack-g4.md)

Published-recipe GCE admission evidence:
[`docs/evidence/2026-06-24-deepseek-v4-flash-published-recipe-gce-admission.md`](evidence/2026-06-24-deepseek-v4-flash-published-recipe-gce-admission.md)

NVFP4 G4 evidence:
[`docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-g4-probe.md`](evidence/2026-06-24-deepseek-v4-flash-nvfp4-g4-probe.md)

NVFP4 SM120 G4 evidence:
[`docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-sm120-g4-probe.md`](evidence/2026-06-24-deepseek-v4-flash-nvfp4-sm120-g4-probe.md)

NVFP4 private-egress G4 evidence:
[`docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-private-egress-g4.md`](evidence/2026-06-24-deepseek-v4-flash-nvfp4-private-egress-g4.md)

NVFP4 no-Xet G4 evidence:
[`docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-no-xet-g4.md`](evidence/2026-06-24-deepseek-v4-flash-nvfp4-no-xet-g4.md)

NVFP4 Triton-linear G4 evidence:
[`docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-triton-g4.md`](evidence/2026-06-24-deepseek-v4-flash-nvfp4-triton-g4.md)

## Decision

Start in Hydralisk, not Psionic.

DeepSeek-V4-Flash on our Google estate is first a CUDA/Python admission and
runtime-shape question. We need to prove the model artifact, engine, GPU
memory, host RAM, and kernel support before Psionic should spend Rust-runtime
work on the same serving behavior. The Psionic target is the useful result of
this lane: expert repack, hot expert pool, overlap/prefetch behavior, and any
accept/reject evidence from real NVIDIA hosts.

This is not a Khala product route and not a public model selector.

## Model facts to validate

- Model: `deepseek-ai/DeepSeek-V4-Flash`
- Reported scale: 284B total parameters, 13B active parameters.
- Context window: 1,048,576 tokens.
- Architecture in the local GGUF: `deepseek4`.
- Local artifact:
  `~/Downloads/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`
- Local artifact size: 86,720,111,200 bytes.
- Official quantization path: FP4 + FP8 mixed.
- Local GGUF quantization label:
  `IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8`.

The local GGUF is useful for admission and parser validation. It is not by
itself a public serving revision. Before any serving claim, pin either the
upstream HF revision or the local artifact digest and the exact engine/container
that loads it.

## First hard thing

Run the public-safe preflight:

```bash
uv run hydralisk-deepseek-v4-preflight \
  --gguf ~/Downloads/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --output-dir .hydralisk/deepseek-v4-flash-preflight-20260624 \
  --collect-gcloud
```

The command:

- reads only the GGUF header and metadata;
- does not read tensor payloads or load weights;
- optionally captures sanitized live GPU instance inventory from `gcloud`;
- classifies the current Google GPU lanes against a low-context smoke reserve;
- writes JSON and Markdown under ignored `.hydralisk/`.

Copy only the public-safe Markdown summary into `docs/evidence/` after review.
Do not commit the generated JSON if it contains local paths you do not want in
history.

## Current lane read

The live single-H100 host is rejected for DeepSeek work because it is already
reserved for GPT-OSS 120B.

On 2026-06-24, Google admitted the first fresh DeepSeek probe:
`g4-standard-96` in `us-central1-b` with 2 x RTX PRO 6000 Blackwell GPUs. That
changed the blocker from quota/capacity to vLLM kernel compatibility. The host
loaded enough of `deepseek-ai/DeepSeek-V4-Flash` to resolve
`DeepseekV4ForCausalLM`, use tensor parallel size 2, use FP8 KV, cache roughly
149 GB of model data, and compile TileLang kernels after pinning CUDA 12.9.
It did not reach `/v1/models`.

The first blocking signature was:

```text
RuntimeError: dispatch_scaled_mm,
/workspace/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp:17
```

Disabling DeepGEMM, UE8M0, TMA-aligned scales, and block-scale FP8 FlashInfer
did not avoid the same `cutlass_scaled_mm` failure in vLLM `0.23.0`.
The scaled-mm microprobe narrowed that failure:

- vLLM reports CUTLASS FP8 support for SM120, but direct CUTLASS FP8
  `cutlass_scaled_mm` fails even for tiny matrices.
- The Triton block-scaled FP8 path passes with ordinary float32 scales.
- The Triton path fails on `float8_e8m0fnu` scale tensors, matching the
  full-model `--linear-backend triton` error.

The next implementation step is to test an E8M0 scale upcast patch or wrapper
for vLLM's CUDA Triton block-scaled FP8 path, then retry the full model with
Triton linear kernels and expert parallel enabled.

That E8M0 patch was validated on the microprobe: the Triton E8M0 case now
passes. The full-model smoke still stops before `/v1/models`, but now at
DeepSeek's NVIDIA `o_proj` DeepGEMM `fp8_einsum` layout assertion:

```text
RuntimeError: Assertion error
(/workspace/.deps/deepgemm-src/csrc/apis/../jit_kernels/impls/../heuristics/../../utils/layout.hpp:39):
t.dim() == N
```

The grouped `o_proj` RHS patch moved the failure from `layout.hpp:39:
t.dim() == N` to `layout.hpp:59: Unknown SF transformation`. DeepGEMM now sees
`rhs_weight` as `[4,1024,4096]` and `rhs_scale` as `[4,8,32]`.

The grouped RHS scale-mode probe then tested raw E8M0, fp32-upcasted scales,
DeepGEMM's transform helper on E8M0, and the same helper after fp32 upcast. None
reached `/v1/models`. `deepgemm_transform` fails on DeepGEMM's dtype assertion,
while `fp32` and `deepgemm_transform_fp32` still fail with
`layout.hpp:59: Unknown SF transformation`.

The provider-stack probe then built a clean container derived from
`vllm/vllm-openai:latest`, installed CUDA development libraries, ran vLLM's
`tools/install_deepgemm.sh` helper with `UV_SYSTEM_PYTHON=1`, and confirmed
that vLLM `0.23.0`, Torch `2.11.0+cu130`, and DeepGEMM import successfully on
both RTX PRO 6000 GPUs. That removed the patched Python venv as the explanation
for failure.

The clean provider stack still failed before `/v1/models`, now back at the
original stock vLLM CUTLASS block-scaled FP8 error:

```text
RuntimeError: dispatch_scaled_mm,
/workspace/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp:17
```

The provider card supplied with this investigation confirms the shape of the
published recipe: vLLM 0.20+, DeepGEMM installed through vLLM's helper,
FP8 KV cache, block size 256, expert parallel enabled, and DeepSeek V4 parser
flags. It also says tensor parallelism must match GPU count to avoid replicated
dense-layer OOM. The advertised NVIDIA deployment shapes are 8 x H100,
8 x H200, 8 x B200, 4 x GB200 NVL4, or a DGX Station class single-GPU path.

The published-recipe GCE admission probe then checked those Google shapes
directly. The project can see matching catalog entries and machine types for
H100, H100 Mega, H200, B200, and GB200 candidates in the sampled zones, but
every candidate blocked on missing regional quota metrics. The probed regions
exposed L4 GPU quota only. No create attempt was made because `ATTEMPT_CREATE=0`
by default and the candidates were quota-blocked before create.

The NVFP4 Blackwell variant was then probed on a fresh
`g4-standard-96` host with 2 x RTX PRO 6000. The model revision
`nvidia/DeepSeek-V4-Flash-NVFP4@e3cd60e7de98e9867116860d522499a728de1cf9`
is public and ungated. The clean vLLM/DeepGEMM image built successfully,
DeepGEMM imported on both GPUs, and the failure moved away from the native-FP8
`dispatch_scaled_mm` path. Stock vLLM still did not reach `/v1/models`.

The NVFP4 blocker is now vLLM's MoE backend selector:

```text
NotImplementedError:
No NvFp4 MoE backend supports the deployment configuration.
```

An explicit backend matrix showed that FlashInfer TRTLLM is the only plausible
clamp-capable path for this model config, and it rejects the current G4 device:

```text
flashinfer_trtllm  blocked: kernel does not support current device cuda
other backends     blocked: swiglu_limit=10.0 and no SwiGLU clamp support
```

The next issue should stop treating the two-card G4 stock-vLLM path as a
near-serving lane. The published-recipe path is administratively blocked until
we obtain H100/H200/B200/GB200 quota. The Google hardware admitted today is
G4 RTX PRO 6000, so the next executable lane is a custom G4 implementation.
The immediate next hard thing is to isolate and validate the FlashInfer TRTLLM
NVFP4 device support gate for RTX PRO 6000 before retrying the full model.

The provider inventory pasted into the investigation reinforces the same target
shape: DeepSeek-V4-Flash wants vLLM `0.20.0+`, DeepGEMM, FP8 KV cache, block
size `256`, expert parallel, DeepSeek V4 parser flags, and tensor parallelism
matching visible GPU count. It also keeps the real published-serving target in
view: 8 x H100, 8 x H200, 8 x B200, 4 x GB200 NVL4, or a DGX Station class
single-GPU path. The admitted 2 x RTX PRO 6000 G4 lane remains a compatibility
probe, not a published-recipe shape.

The SM120 gate probe then built a derived NVFP4 image with an explicit
default-off local patch allowing `is_device_capability_family(120)` in the
FlashInfer TRTLLM NVFP4 gate. That moved the lane past the immediate
`flashinfer_trtllm` backend selector rejection and into vLLM `0.23.0` startup
with the pinned NVFP4 model revision. It still did not reach `/v1/models`.

The new blocker is artifact access on the private-only G4 host:

```text
dns huggingface.co 3.170.185.25
dns_error cdn-lfs.huggingface.co gaierror [Errno -2] Name or service not known
fetch_error URLError <urlopen error [Errno 101] Network is unreachable>
NET_RC=124
```

This means the next useful step is to configure private egress for the G4
subnet or pre-stage the pinned HF snapshot under
`/var/lib/hydralisk/huggingface`, then rerun the same SM120 probe. It does not
yet prove or disprove weight load, memory fit, or the actual FlashInfer TRTLLM
kernel on RTX PRO 6000.

The private-egress probe then created a Cloud Router/NAT pair for the `default`
network in `us-central1`:

```text
hydralisk-default-router-us-central1
hydralisk-default-nat-us-central1
```

The G4 VM kept no external IP, but the derived container could fetch the pinned
model config over private outbound egress. The rerun advanced into real load:
vLLM resolved `DeepseekV4ForCausalLM`, used tensor/expert parallel world size
2, placed 128 of 256 experts on each rank, selected
`FLASHINFER_TRTLLM`, and held about `80781 MiB` on each RTX PRO 6000. The HF
cache grew to about `95G`.

It still did not reach `/v1/models`. The process appeared wedged with no vLLM
log writes after `08:19:03Z`, no Xet log writes after `08:21:44Z`, defunct
worker processes, and the parent process still alive holding GPU memory. The
next useful step is to make snapshot acquisition deterministic: use an
out-of-band `HF_TOKEN`, disable Xet if supported by the installed HF stack, or
pre-stage the complete pinned snapshot before starting vLLM.

The no-Xet probe then set `HF_HUB_DISABLE_XET=1` on the private G4 host. That
run reused Cloud NAT, fetched the pinned config, avoided fresh Xet log traffic,
grew the Hugging Face cache to about `163G`, and reached a deterministic vLLM
exception instead of hanging with defunct workers. vLLM again resolved
`DeepseekV4ForCausalLM`, enabled expert parallelism, selected
`FLASHINFER_TRTLLM`, and used the FP8 indexer cache, but it failed during
`determine_available_memory` on the non-expert FP8 linear path:

```text
RuntimeError: dispatch_scaled_mm,
/workspace/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp:17
```

The useful next step is no longer Hugging Face transfer. It is to force or patch
dense FP8 linear layers away from the failing CUTLASS block-scaled-mm dispatch,
preferably toward the Triton path that passed the earlier microprobe, while
keeping NVFP4 experts on `flashinfer_trtllm`.

The Triton-linear probe then did exactly that with
`VLLM_LINEAR_BACKEND=triton`, `VLLM_E8M0_TRITON_UPCAST=1`,
`ALLOW_NVFP4_SM120=1`, `HF_HUB_DISABLE_XET=1`, and
`MOE_BACKEND=flashinfer_trtllm`. It removed the CUTLASS
`dispatch_scaled_mm` blocker and moved the failure to DeepSeek V4's NVIDIA
`o_proj` path:

```text
RuntimeError: Assertion error
(csrc/apis/../jit_kernels/impls/../heuristics/../../utils/layout.hpp:39):
t.dim() == N
```

The stack now passes through `vllm/models/deepseek_v4/nvidia/flashmla.py`,
`vllm/models/deepseek_v4/nvidia/ops/o_proj.py`, and
`vllm/utils/deep_gemm.py`. The next hard thing is a public-safe shape trace and
minimal fallback for `deep_gemm_fp8_o_proj`, while preserving the
FlashInfer TRTLLM NVFP4 MoE path.

The first viable lanes are:

1. `g4-standard-96`, 2 x RTX PRO 6000. Google admitted this lane; it clears
   the low-context all-GPU memory preflight and is directionally aligned with
   Blackwell FP4 work, but stock vLLM `0.23.0` currently fails in CUTLASS
   FP8 scaled-mm before readiness.
2. `g4-standard-48`, 1 x RTX PRO 6000, for the cheapest offload/prefetch
   validation. It does not clear conservative all-GPU memory once runtime and
   KV reserve are included, but it is the right place to test expert repack plus
   a hot expert cache.
3. `a3-highgpu-2g`, 2 x H100, if capacity admits it. Hopper is mature, but this
   does not validate the Blackwell-specific FP4 direction as directly.

Multi-L4 is deprioritized. Aggregate memory can look plausible on paper, but
bandwidth and interconnect make it a poor first target for this MoE path, and
active Khala/GPT-OSS L4 hosts must not be disturbed.

## Load-smoke ladder

After this preflight passes and a fresh GPU host is admitted:

1. Pin model revision or local artifact digest.
2. Pin engine version and container image digest.
3. Start with no public ingress and no OpenAgents product routing.
4. For `g4-standard-96` or `a3-highgpu-2g`, attempt an all-GPU low-context load
   smoke with vLLM first.
5. For `g4-standard-48`, attempt only an offload/prefetch validation:
   separate skeleton and expert weights, hold a bounded warm expert pool in
   VRAM, stream cold experts from host RAM, and measure whether prefetch hides
   the PCIe slack.
6. Emit a public-safe receipt with GPU memory, engine version, model revision,
   context cap, usage, latency, and blocker state.
7. Only then evaluate whether Psionic should implement the same scheduling
   behavior natively.

The reproducible smoke runner is:

```bash
scripts/smoke-deepseek-v4-gce.sh
```

Useful knobs:

```bash
ISSUE_NUMBER=7
VLLM_USE_DEEP_GEMM=0
VLLM_USE_DEEP_GEMM_E8M0=0
VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=0
VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
VLLM_LINEAR_BACKEND=triton
VLLM_ENABLE_EXPERT_PARALLEL=1
FORCE_PYTHON_VLLM=1
REUSE_PYTHON_VENV=1
HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=blackwell
HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE=1
HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS=1
HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE=fp32
```

Use `TARGET_INSTANCE`, `TARGET_ZONE`, and `TARGET_GPU_COUNT` only for a fresh
`hydralisk-deepseek-v4-*` probe. The script refuses arbitrary existing host
names so it does not accidentally target Khala/GPT-OSS product hosts.

## Stop conditions

Stop and record a blocker if:

- the host lacks enough disk for the artifact and engine cache;
- `gcloud` admits only a product host or a host already serving another model;
- the engine cannot parse `deepseek4` or the official HF artifact;
- CUDA/kernel support rejects RTX PRO 6000 or H100 for this model path;
- the smoke cannot produce public-safe usage, latency, memory, and blocker
  receipts.

The 2026-06-24 G4 smoke is currently stopped on the CUDA/kernel support
condition above. More random flag trials on the same host are not the next
useful step; the useful split is now an 8-GPU H100/H200/B200 allocation that
matches the published recipe, or a custom expert-prefetch/offload route for
RTX PRO 6000. The published-recipe allocation is currently blocked by missing
H100/H100 Mega/H200/B200/GB200 quota in this project. The NVFP4 stock-vLLM
route first stopped on vLLM/FlashInfer backend support for this exact G4
device/configuration. The patched SM120 probe got past that device gate,
Cloud NAT got past private config fetch, and `HF_HUB_DISABLE_XET=1` got past
the Xet/vLLM transfer wedge. The active blocker is now vLLM's dense FP8
`cutlass_scaled_mm` dispatch on SM120 before readiness. After
`VLLM_LINEAR_BACKEND=triton` plus the E8M0 upcast patch, that CUTLASS blocker
is cleared and the lane stops on DeepGEMM `o_proj` layout handling before
readiness.

## Promotion boundary

DeepSeek-V4-Flash should not become a public OpenAgents model name from this
work. Any later user-facing capability must stay behind the product-owned Khala
identity unless the product repos explicitly add a new public promise.
