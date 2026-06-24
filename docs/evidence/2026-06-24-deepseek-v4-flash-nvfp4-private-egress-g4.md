# DeepSeek-V4-Flash NVFP4 private-egress G4 probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/17

## Target

- Project: `openagentsgemini`
- Instance: `hydralisk-deepseek-v4-nvfp4-g4-2g-b-20260624073921`
- Zone: `us-central1-b`
- Machine: `g4-standard-96`
- GPUs: 2 x `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- VM network: `default`
- VM subnet: `default` in `us-central1`
- VM internal IP: `10.128.0.28`
- VM external IP: none
- Model: `nvidia/DeepSeek-V4-Flash-NVFP4`
- Model revision: `e3cd60e7de98e9867116860d522499a728de1cf9`
- Derived image:
  `hydralisk-deepseek-v4-nvfp4-sm120-vllm:20260624081828`
- MoE backend: `flashinfer_trtllm`
- Tensor parallel size: `2`

## Private egress change

Issue #16 proved the SM120 patch could advance into vLLM startup, but the
private-only VM could not fetch Hugging Face artifacts. The fix was to keep the
VM private and add outbound-only Cloud NAT to the `default` VPC in `us-central1`.

Created resources:

```text
Cloud Router: hydralisk-default-router-us-central1
Network: default
Region: us-central1
Cloud NAT: hydralisk-default-nat-us-central1
NAT IP allocation: AUTO_ONLY
Source subnetwork ranges: ALL_SUBNETWORKS_ALL_IP_RANGES
Logging: enabled, ALL
```

The existing `oa-nat-router-us-central1` / `oa-nat-us-central1` resources for
the `oa-lightning` network were left untouched.

The G4 VM still has no external IP after the NAT change:

```text
networkInterfaces[0].networkIP: 10.128.0.28
```

## Artifact-access result

The same derived container can now fetch the pinned model config over the
private outbound path:

```text
DNS	huggingface.co	3.170.185.33
DNS_ERROR	cdn-lfs.huggingface.co	gaierror: [Errno -2] Name or service not known
DNS	google.com	173.194.206.101
FETCH	https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4/resolve/e3cd60e7de98e9867116860d522499a728de1cf9/config.json	200	6873
NETWORK_RC	0
```

Direct `cdn-lfs.huggingface.co` DNS still fails, but this model's active fetch
path uses Hugging Face's Xet/CAS URLs under `us.gcp.cdn.hf.co`, and the Xet log
showed successful HTTP `206` range responses after the Cloud NAT change.

## Model-load result

The rerun advanced farther than issue #16:

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

During the run, both GPUs held model memory:

```text
GPU 0 memory.used: 80781 MiB / 97887 MiB
GPU 1 memory.used: 80781 MiB / 97887 MiB
```

The Hugging Face cache grew to roughly `95G`.

## Blocker

The model still did not reach `/v1/models`:

```text
READY	0
curl: (7) Failed to connect to 127.0.0.1 port 8000
```

The run was stopped after it appeared wedged:

```text
Current time: 2026-06-24T08:24:51Z
vLLM log mtime: 2026-06-24 08:19:03Z
Xet log mtime: 2026-06-24 08:21:44Z
Cache size: 95G
Parent vLLM process: alive
Worker processes: defunct
```

The vLLM log did not emit a Python exception before the stop. The public-safe
read is that private artifact egress is fixed, but the first full model-load
attempt stalled in the Hugging Face Xet / vLLM load path before readiness.

## Decision

Do not claim serving readiness.

The next hard thing is to make the model snapshot deterministic before another
kernel probe:

1. Retry with an explicit `HF_TOKEN` supplied out-of-band to raise Hub/Xet
   limits, or
2. Disable Xet for the download path if the installed Hugging Face stack
   supports it, or
3. Pre-stage the complete pinned snapshot under
   `/var/lib/hydralisk/huggingface` before starting vLLM.

After the snapshot is complete, rerun the same patched SM120 probe and determine
whether the next blocker is memory fit, FlashInfer TRTLLM runtime support, or
readiness.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
