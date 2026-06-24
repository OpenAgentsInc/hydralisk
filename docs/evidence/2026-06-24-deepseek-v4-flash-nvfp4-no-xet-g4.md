# DeepSeek-V4-Flash NVFP4 no-Xet G4 probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/18

## Target

- Project: `openagentsgemini`
- Instance: `hydralisk-deepseek-v4-nvfp4-g4-2g-b-20260624073921`
- Zone: `us-central1-b`
- Machine: `g4-standard-96`
- GPUs: 2 x `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- VM external IP: none
- Private egress: `hydralisk-default-nat-us-central1`
- Model: `nvidia/DeepSeek-V4-Flash-NVFP4`
- Model revision: `e3cd60e7de98e9867116860d522499a728de1cf9`
- Derived image:
  `hydralisk-deepseek-v4-nvfp4-sm120-vllm:20260624083125`
- MoE backend: `flashinfer_trtllm`
- Tensor parallel size: `2`

## Script change

The provider-stack and NVFP4 G4 probe scripts now expose explicit snapshot
acquisition knobs:

```text
HF_HUB_DISABLE_XET
HF_XET_HIGH_PERFORMANCE
HF_XET_NUM_CONCURRENT_RANGE_GETS
```

The defaults preserve prior behavior:

```text
HF_HUB_DISABLE_XET=0
HF_XET_HIGH_PERFORMANCE=0
HF_XET_NUM_CONCURRENT_RANGE_GETS=default
```

For this run, `HF_HUB_DISABLE_XET=1` was set and recorded in the engine file.
No token was passed and no model artifact was committed.

## Snapshot result

The run reused the private Cloud NAT path and fetched the pinned model config:

```text
FETCH	https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4/resolve/e3cd60e7de98e9867116860d522499a728de1cf9/config.json	200	6873
NETWORK_RC	0
```

The no-Xet run behaved differently from the prior Xet run:

- The old Xet log did not update.
- The cache grew from roughly `99G` to `163G`.
- Recent cache files were ordinary Hugging Face Hub blobs rather than active
  Xet log traffic.
- The run completed far enough to produce a vLLM root-cause exception instead
  of hanging with defunct workers.

## Model-load result

The model again reached real load:

```text
Resolved architecture: DeepseekV4ForCausalLM
Using max model len 4096
world_size=2
Expert parallelism is enabled
Local/global number of experts: 128/256
Detected ModelOpt NVFP4 checkpoint
Using 'FLASHINFER_TRTLLM' NvFp4 MoE backend
Using FP8 indexer cache for Lightning Indexer
```

It did not reach `/v1/models`:

```text
READY	0
```

## Blocker

The deterministic no-Xet run exposes the older stock-vLLM FP8 CUTLASS blocker
again during `determine_available_memory`:

```text
RuntimeError: dispatch_scaled_mm,
/workspace/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp:17
```

The stack passes through the FP8 linear path:

```text
vllm/model_executor/layers/quantization/fp8.py
vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py
vllm/model_executor/kernels/linear/scaled_mm/cutlass.py
torch.ops._C.cutlass_scaled_mm
```

This means issue #18 answered the snapshot question: disabling Xet gets the
private G4 run past the previous transfer/load wedge. The next blocker is no
longer acquisition; it is the stock CUTLASS FP8 scaled-mm path on RTX PRO 6000
SM120 while loading the NVFP4 model's non-expert FP8 layers.

## Decision

Keep the snapshot knobs. Use `HF_HUB_DISABLE_XET=1` for the next G4 probe unless
there is a specific reason to test Xet again.

The next hard thing should target the FP8 scaled-mm path, not Hugging Face
transfer:

1. Force or patch the non-expert FP8 linear backend away from CUTLASS toward the
   Triton path that passed the earlier microprobe, or
2. Disable the failing CUTLASS block-scaled-mm dispatch for SM120 in the derived
   probe image, or
3. Patch vLLM's backend selector for this model/config so NVFP4 experts still
   use FlashInfer TRTLLM while dense FP8 layers use the working Triton path.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
