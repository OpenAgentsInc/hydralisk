# DeepSeek-V4-Fable merged checkpoint G4 preflight

Date: 2026-06-24T20:08:00Z

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/78

Depends on: https://github.com/OpenAgentsInc/hydralisk/issues/77

Profile: `profiles/deepseek-v4-fable-adapter-g4.json`

Status: `go_for_staging_high_runtime_risk`

## Decision

- Stage merged checkpoint on G4: `true`
- Use current adapter path for semantic canary: `false`
- Use current base DeepSeek-V4 public/Khala route: `false`
- Public aliases allowed: `false`
- MPP public sale allowed: `false`
- Next step: `stage_merged_checkpoint_on_existing_g4_boot_disk_or_attached_pd`

## Target host

- Project: `openagentsgemini`
- Instance: `hydralisk-deepseek-v4-b12x-g4-8g-b-20260624155352`
- Zone: `us-central1-b`
- Status: `RUNNING`
- Machine type: `g4-standard-384`
- Provisioning model: `SPOT`
- Termination action: `DELETE`
- Max run duration: `21600` seconds
- Network: private IP only, no external IP
- SSH path: IAP tunnel
- Service account scopes include `devstorage.read_only`

## GPU and disk inventory

- GPUs: `8 x NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Per-GPU memory reported by `nvidia-smi`: `97887 MiB`
- GPU memory in use during preflight: `0 MiB` on all 8 GPUs
- Boot disk: `900 GB` persistent disk
- Filesystem available on `/`: `663 GB`
- Filesystem used on `/`: `210 GB`
- Docker images present: `37.94 GB`
- Running Docker containers: `0`

The merged checkpoint tensor payload is `298425334924` bytes, about `278 GiB`.
It fits on the current boot disk with roughly `365 GB` remaining after staging,
before download scratch and logs.

## Network path

The private G4 host can reach Hugging Face metadata and shard URLs directly:

- `model.safetensors.index.json` HEAD: HTTP `200`
- Index content length: `2733424`
- Index ETag: `5bfe52de335aacddf0f0f005fc63483675029fc3`
- First shard HEAD: HTTP `200`
- First shard content length: `6434269642`
- First shard ETag:
  `f26562cc043d473c187b6c7e1a103ddb80f1888e63cad602a22392f1ca72c4b6`
- First shard `accept-ranges`: `bytes`

Range support means issue #79 should use resumable downloads and per-file
verification rather than a single all-or-nothing transfer.

## Artifact shape

Upstream metadata at
`Chunjiang-Intelligence/DeepSeek-v4-Fable@999909137c15e0b5539fee887431824fa7cb5b10`:

- `model.safetensors.index.json` entries: `35020`
- Merged shard count: `47`
- Total tensor payload bytes: `298425334924`
- Config `model_type`: `deepseek_v4`
- Config `torch_dtype`: `bfloat16`
- Config `quantization_config.quant_method`: `fp8`
- Config `quantization_config.fmt`: `e4m3`
- Context window in config: `1048576`
- Hidden size: `4096`
- Layers: `43`
- Routed experts: `256`
- Shared experts: `1`

The checkpoint key families are canonical Fable/DeepSeek keys such as
`layers.0.attn.wq_a.weight`, `layers.0.attn.wkv.weight`, and
`layers.0.ffn.experts.*.w{1,2,3}.weight`.

The current Hydralisk runtime includes a `WeightsMapper` that maps
`layers.` -> `model.layers.`, and the NVIDIA DeepSeek-V4 loader has stacked
mapping for:

- `attn.wq_a` -> `attn.fused_wqa_wkv` shard `0`
- `attn.wkv` -> `attn.fused_wqa_wkv` shard `1`
- `compressor.wkv` -> `compressor.fused_wkv_wgate` shard `0`
- `compressor.wgate` -> `compressor.fused_wkv_wgate` shard `1`
- `w1`/`w3` expert and shared-expert families into gate/up fused paths
- `w2` into down-projection paths

Therefore the checkpoint naming is not an immediate preflight no-go. The real
risk is runtime compatibility for the FP8 merged artifact on the patched G4
SM120 lane.

## Cost and blast radius

- Existing boot disk staging adds no new provisioned-disk resource because the
  900 GB boot disk already exists.
- Boot disk staging is fragile: this is a Spot instance with termination action
  `DELETE`, so staging can disappear with the VM.
- A separate attached persistent disk would add provisioned GB-month cost but
  would reduce the blast radius from Spot deletion and make retry/canary work
  less wasteful.
- Google documents Persistent Disk as durable network block storage and bills
  provisioned disk space by size over time:
  <https://cloud.google.com/compute/disks-image-pricing>

Transfer time depends on Hugging Face CDN throughput. For a `298.4 GB`
payload, rough wall-clock bounds are:

| Sustained transfer | Estimated time |
| ---: | ---: |
| `10 MB/s` | about `8.3 h` |
| `25 MB/s` | about `3.3 h` |
| `50 MB/s` | about `1.7 h` |
| `100 MB/s` | about `0.8 h` |

Because the host is Spot with a 6-hour max run duration setting, issue #79
should use resumable downloads and write a manifest as each shard completes.

## Preflight result

Staging is operationally sane if treated as a private experiment, not as a
serving claim.

Use one of these staging locations:

1. Fast path: existing boot disk under an ignored path such as
   `/opt/hydralisk/models/deepseek-v4-fable-merged`, accepting Spot-loss risk;
2. Safer path: attach a separate durable persistent disk and stage there.

Do not expose Fable publicly. Do not route through Khala or MPP. The next
honest issue is staging the merged checkpoint with resumable verification, then
running a private load-only canary.

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains tensor values: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
- Contains target details: false
- Contains full third-party source: false
