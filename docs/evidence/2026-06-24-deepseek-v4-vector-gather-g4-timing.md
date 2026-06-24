# DeepSeek V4 vector-gather G4 timing

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/61

## Summary

Issue #61 replaced the inner Python head/candidate loop in the Hydralisk
DeepSeek V4 sparse MLA fallback with a vectorized gather plus batched softmax
path. The goal was to test whether the 8 x G4 lane could move from "executes"
to "maybe worth a Khala readiness gate."

The target remained the private spot VM:

```text
instance=hydralisk-deepseek-v4-b12x-g4-8g-b-20260624155352
project=openagentsgemini
zone=us-central1-b
machine=g4-standard-384
accelerator=nvidia-rtx-pro-6000 x8
gpu=NVIDIA RTX PRO 6000 Blackwell Server Edition
driver=580.159.03
cuda=13.0
externalIp=<none>
```

## Image

```text
baseImage=hydralisk-deepseek-v4-b12x-g4-vllm-issue60-metadata:20260624170016
baseImageDigest=sha256:ace20aadf862812912b4b63ff9bd19f8b04d335680f2983a1200ebf6eaf47418
derivedImage=hydralisk-deepseek-v4-b12x-g4-vllm-issue60-vector-v3:20260624v3vector2
derivedImageDigest=sha256:a43653081fe01ab53901ecdff7415d2d8c40a6ea76b5c923aaff1b0e0e661451
```

The generated provider-stack patch upgraded
`vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` to:

```text
HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK_VECTOR_GATHER_V3
```

## Standard Smoke

The v3 image reached `/v1/models` and completed the same public-safe
OpenAI-compatible smoke request:

```json
{
  "id": "chatcmpl-a5b843d2355ac9cb",
  "model": "nvidia/DeepSeek-V4-Flash-NVFP4",
  "usage": {
    "prompt_tokens": 9,
    "total_tokens": 12,
    "completion_tokens": 3,
    "prompt_tokens_details": null
  },
  "finish_reason": "stop"
}
```

No prompt text or response text is committed in this receipt.

## Timing

The prior issue #60 image was measured on the same host with the same
`max_model_len=2048`, `max_num_seqs=1`, eager-mode, and SWA-only sparse-indexer
configuration:

| Path | TTFT | Decode after first token | End-to-end output rate |
| --- | ---: | ---: | ---: |
| issue #60 cache-layout v2 fallback | 13.115 s | 0.885 tok/s | 0.665 tok/s |
| issue #61 vector-gather v3 fallback | 0.317 s | 11.229 tok/s | 10.396 tok/s |

The full v3 timing run:

```json
{
  "image": "hydralisk-deepseek-v4-b12x-g4-vllm-issue60-vector-v3:20260624v3vector2",
  "ready": true,
  "server_start_to_ready_wall_s": 116,
  "runs": [
    {
      "label": "warmup_nonstream_8",
      "mode": "nonstream",
      "max_tokens": 8,
      "elapsed_s": 30.383387,
      "prompt_tokens": 28,
      "completion_tokens": 8,
      "total_tokens": 36,
      "output_tokens_per_s_full_elapsed": 0.263302,
      "finish_reason": "length"
    },
    {
      "label": "timed_stream_32",
      "mode": "stream",
      "max_tokens": 32,
      "elapsed_s": 3.078189,
      "time_to_first_sse_s": 0.317321,
      "time_to_first_delta_s": 0.317342,
      "decode_window_s": 2.760798,
      "prompt_tokens": 28,
      "completion_tokens": 32,
      "total_tokens": 60,
      "chunks_with_delta": 32,
      "output_tokens_per_s_after_first_delta": 11.228637,
      "output_tokens_per_s_full_elapsed": 10.395723,
      "finish_reason": "length"
    },
    {
      "label": "timed_nonstream_32",
      "mode": "nonstream",
      "max_tokens": 32,
      "elapsed_s": 3.053477,
      "prompt_tokens": 28,
      "completion_tokens": 32,
      "total_tokens": 60,
      "output_tokens_per_s_full_elapsed": 10.479856,
      "finish_reason": "length"
    }
  ]
}
```

## Interpretation

This is the first DeepSeek V4 G4 result that is plausibly worth a Khala
readiness gate. The warm path now has sub-second TTFT and roughly 10-11 output
tokens per second for a 32-token single-request smoke.

It is not yet enough to justify adding DeepSeek V4 Flash to Khala:

- startup is still about 116 seconds to `/v1/models`;
- the first post-ready warmup request still pays roughly 30 seconds of JIT and
  shape work;
- `max_model_len` is still only 2048;
- `max_num_seqs=1`, so no useful concurrency has been tested;
- `HYDRALISK_DEEPSEEK_INDEXER_SWA_ONLY=1` still bypasses the compressed sparse
  top-k indexer path;
- no repeated-run stability, quality, or long-context gate has passed.

The next useful gate is a resident-server readiness harness: warm the known
shapes, run repeated public-safe streaming requests, record p50/p95 TTFT and
decode throughput, and run a small answer-quality gate without committing raw
prompts or responses.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
