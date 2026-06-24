# DeepSeek-V4-Flash B12x eager-mode G4 smoke

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/39

## Summary

The issue #39 probe added an eager-mode launch option to the DeepSeek-V4
provider-stack harness and retried the pinned NVFP4 model on the private
8 x RTX PRO 6000 G4 host:

```text
instance hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036
zone     us-central1-b
model    nvidia/DeepSeek-V4-Flash-NVFP4
revision e3cd60e7de98e9867116860d522499a728de1cf9
backend  flashinfer_b12x
tp       8
eager    true
```

`--enforce-eager` moved the full-model smoke past the prior vLLM
cudagraph-memory-profiling path where DeepGEMM MLA metadata rejected SM120:

```text
vllm/v1/attention/backends/mla/indexer.py build
vllm/utils/deep_gemm.py get_paged_mqa_logits_metadata
RuntimeError: Assertion error (csrc/apis/attention.hpp:219): Unsupported architecture
```

With eager mode enabled, vLLM completed engine initialization, exposed
`/v1/models`, and advertised the pinned model with `max_model_len=2048`.

This is not generation readiness. A tiny public-safe chat completion then
failed on first forward in DeepSeek V4's NVIDIA FlashMLA sparse prefill path:

```text
vllm/models/deepseek_v4/attention.py attention_impl
vllm/models/deepseek_v4/nvidia/flashmla.py _forward_prefill
vllm/third_party/flashmla/flash_mla_interface.py flash_mla_sparse_fwd
RuntimeError: Sparse Attention Forward Kernel is only supported on SM90a and SM100f architectures.
```

## Harness Changes

The provider-stack script now has three explicit smoke controls:

- `VLLM_ENFORCE_EAGER=1` adds `--enforce-eager`.
- `COMPLETION_TIMEOUT_SECONDS` bounds the public-safe completion smoke.
- `CONTAINER_START_TIMEOUT_SECONDS` gives `docker run` a startup grace before
  readiness checks treat a missing container as failed.

The readiness loop now watches Docker container state instead of only the
local `docker run` wrapper process. That avoids an earlier false negative
where the script wrote `READY 0` while the vLLM server came up moments later.

## Provider Inventory Cross-Check

The provider inventory pasted during the investigation matches the recipe
Hydralisk is adapting: vLLM `0.20.0+`, DeepGEMM, FP8 KV cache, block size
`256`, DeepSeek V4 tokenizer/tool/reasoning parsers, and tensor parallelism
matching visible GPU count. It also reinforces that the published happy path
is H200/B200/GB-class hardware; G4 SM120 remains a compatibility lane.

One useful detail from that inventory is that published H200 prefill examples
also use `--enforce-eager`. That makes eager mode a reasonable probe setting,
not just a local hack.

## Blocker

The next hard blocker is now FlashMLA sparse prefill support for SM120. The
model is loaded enough for `/v1/models`, and the B12x MoE/o_proj path can
advance to first forward, but generation cannot proceed while
`flash_mla_sparse_fwd` rejects RTX PRO 6000 Blackwell Server Edition.

The next issue should inspect DeepSeek V4's FlashMLA integration and either:

- select an existing non-FlashMLA attention backend that can run on SM120;
- patch the architecture gate only if the kernel actually works on SM120; or
- implement a correctness-first fallback prefill path for tiny-batch G4 smoke.

## Tooling Footnote

After the runtime evidence above was captured, local `gcloud` auth token
refresh began failing non-interactively with `Reauthentication failed. cannot
prompt during non-interactive execution.` A final scripted rerun could not be
completed in this turn. The private G4 host was confirmed clean before that
reauth failure: no Docker containers were running and all eight GPUs reported
`0 MiB` used.

## Public Safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
