# FlashInfer DSV4 FMHA SM120 source audit

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/53

## Summary

Issue #53 inspected the installed FlashInfer/TRTLLM FMHA source and cubin
inventory behind the direct issue #52 repro. This is not a safe one-line SM120
allowlist patch.

The FlashInfer package defines `kSM_120`, but the DSV4 TRTLLM FMHA runner
guard admits only SM100 and SM103:

```text
/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/trtllm/fmha/fmhaRunner.cuh:37:
FLASHINFER_CHECK(mSM == kSM_100 || mSM == kSM_103, "Unsupported architecture");
```

The FMHA compatibility helper also only special-cases SM100-family kernels for
SM100 and SM103:

```cpp
constexpr bool isSMCompatible(int gpuSM, int kernelSM) {
  if (gpuSM == kSM_103) {
    return kernelSM == kSM_100f || kernelSM == kSM_103;
  } else if (gpuSM == kSM_100) {
    return kernelSM == kSM_100f || kernelSM == kSM_100;
  }

  return gpuSM == kernelSM;
}
```

The installed cubin inventory confirms the runner is not merely missing an
SM120 admission branch:

```json
{
  "trtllmGenFmhaCount": 13452,
  "sm100Count": 11560,
  "sm100aCount": 312,
  "sm103Count": 1892,
  "sm120Count": 0
}
```

## Command

```bash
TARGET_INSTANCE=hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036 \
TARGET_ZONE=us-central1-b \
bash scripts/audit-flashinfer-dsv4-fmha-sm120-gce.sh
```

## Environment

```text
targetInstance=hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036
targetZone=us-central1-b
image=hydralisk-deepseek-v4-b12x-g4-vllm:20260624150453
flashinfer=0.6.12
torch=2.11.0+cu130
cuda=13.0
```

## Decision

Hydralisk should not ship a derived image that only changes the guard to accept
SM120. The package has no SM120 TRTLLM-gen FMHA cubins for this path, and the
compatibility helper would not select SM100-family FMHA kernels for SM120.

The replacement contract is now one of:

- provide an SM120-built TRTLLM-gen DSV4 FMHA kernel and dispatch metadata,
  then rerun `scripts/probe-flashinfer-dsv4-fmha-gce.sh`;
- implement a correctness-first DeepSeek V4 attention fallback for G4 that
  supports sparse MLA decode on SM120, then rerun the full DSV4 model smoke.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
