# GLM-5.2 504B REAP private 4-GPU load smoke

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/86

Profile:
[`profiles/glm-5.2-reap-504b-b12x-g4.json`](../../profiles/glm-5.2-reap-504b-b12x-g4.json)

Launcher:
[`scripts/launch-glm-52-reap-504b-b12x-gce.sh`](../../scripts/launch-glm-52-reap-504b-b12x-gce.sh)

Public receipt:
[`docs/evidence/2026-06-24-glm-52-reap-504b-load-smoke-receipt.json`](2026-06-24-glm-52-reap-504b-load-smoke-receipt.json)

Public-safety boundary: this packet contains run IDs, launch settings, prompt
hashes, response hashes, token counts, timings, and resource envelopes only. It
contains no bearer tokens, model-provider credentials, raw prompts, raw
responses, private source, hidden reasoning traces, weights, checkpoints,
compiled engines, profiler dumps, or raw model logs.

## Result

PASS, with a claim boundary:

- The model loaded and served on 4 selected RTX PRO 6000 GPUs
  (`CUDA_VISIBLE_DEVICES=0,1,2,3`).
- The physical host was the already-admitted 8x fallback
  `g4-standard-384`, because standalone 4x `g4-standard-192` admission was
  capacity-blocked in issue #83.
- Therefore this proves the 4-GPU memory/runtime envelope on G4 hardware, but
  does not yet prove GCE can currently allocate the exact 4x machine shape.

## Host and runtime

- Instance:
  `hydralisk-glm52-reap-504b-g4-8g-b-20260624214500`
- Zone: `us-central1-b`
- Machine type: `g4-standard-384`
- Active GPUs for this smoke: `0,1,2,3`
- Model: `0xSero/GLM-5.2-504B`
- Model revision: `cb6b1e0451b9d560cda864f84187869c9a679712`
- Runtime image:
  `voipmonitor/vllm@sha256:ce23a9b075bd7138ce3b12ee29609b98606e5050e2def4a29bbb917ad96e5997`
- vLLM:
  `0.11.2.dev279+black.benediction.b12xpr11.cu132.20260608`
- Torch: `2.12.0+cu132`
- FlashInfer Python: `0.6.12+cu132`

## Launch envelope

- Run ID: `20260624224500`
- Tensor parallel size: 4
- Decode-context parallel size: 4
- Quantization: `modelopt_fp4`
- KV cache dtype: `fp8`
- Attention backend: `B12X_MLA_SPARSE`
- MoE backend: `b12x`
- Max model length: 32,768 tokens
- Max sequences: 1
- Max batched tokens: 4096
- GPU memory utilization: 0.95
- Host/port: `127.0.0.1:8000`
- MTP: disabled

## Startup

- Container started: `2026-06-24T22:17:30Z`
- vLLM server listening: `2026-06-24T22:20:54Z`
- `/v1/models` ready check: `2026-06-24T22:21:10Z`
- Engine init reported by vLLM: 107.32 seconds
- Compilation during engine init: 39.06 seconds
- CUDA graph capture: 30 seconds, 0.19 GiB per worker
- KV cache: 485,376 tokens reported by vLLM
- Maximum concurrency reported for 32,768 tokens/request: 14.81x

## Completion smoke

The final smoke used a synthetic prompt and stored only hashes and aggregate
metrics. GLM thinking was disabled for the direct API smoke with:

```json
{"chat_template_kwargs":{"enable_thinking":false}}
```

Request parameters:

- `temperature=0`
- `max_tokens=64`
- `min_p=0.05`
- `repetition_penalty=1.05`
- `stream=true`

Observed response:

- HTTP status: 200
- Finish reason: `stop`
- Prompt SHA-256:
  `5f8103cbebaac77e42161be89a636352dafbb3ceda5efef0aa6484d67233dfe2`
- Visible completion SHA-256:
  `deb72954879f318cd0fcb41355e82f54fbed51947d68e71b465fd31aba03f166`
- Visible completion characters: 18
- Prompt tokens: 22
- Completion tokens: 9
- Total tokens: 31
- First visible token latency: 0.2376 seconds
- Total request time: 0.4610 seconds
- Completion-token throughput for the tiny smoke: 19.52 tokens/s

Before disabling thinking, the endpoint also returned HTTP 200 and generated
tokens, but the 24-token and 128-token caps were consumed by non-visible
deltas. Those attempts are not counted as completion-smoke passes because they
did not produce visible final content.

## Resource envelope after smoke

Selected GPU memory:

| GPU | Used MiB | Free MiB | Total MiB |
| --- | ---: | ---: | ---: |
| 0 | 93333 | 3918 | 97887 |
| 1 | 93335 | 3916 | 97887 |
| 2 | 93331 | 3920 | 97887 |
| 3 | 93331 | 3920 | 97887 |

Host/container envelope:

- Container host RAM: `13.08GiB / 1.384TiB`
- Host `MemAvailable`: `1451299520 kB`
- Swap: disabled / 0 kB
- Root disk: `1.5T`, `338G` used, `1.1T` available
- Docker block I/O at envelope capture: `0B / 332MB`
- Container process count: 1052

## NCCL launch fix

Two earlier attempts failed before weight load:

- `20260624222500`: PyNCCL communicator init failed with
  `RuntimeError: NCCL error: unhandled system error`.
- `20260624223500`: after adding `SYS_NICE` and `NCCL_DEBUG=INFO`, NCCL showed
  an empty `NCCL_GRAPH_FILE` path and failed while opening the XML graph file.

The passing run unsets `NCCL_GRAPH_FILE` inside the container entrypoint before
starting vLLM:

```bash
bash -lc "unset NCCL_GRAPH_FILE; exec $launch_command"
```

That moved NCCL communicator init to `Init COMPLETE` for ranks 0-3 and allowed
the model load, CUDA graph capture, `/v1/models`, and completion smoke to pass.

## Next gates

- Issue #87 may expose this already-running service through a private,
  fail-closed endpoint.
- Issue #88 should tune the memory/concurrency envelope. The current 32K,
  `max_num_seqs=1` profile has about 3.9 GiB free per selected GPU after the
  tiny smoke.
- Any public claim must continue to say this was a private smoke on four GPUs
  within the 8x fallback host until standalone 4x G4 admission succeeds.
