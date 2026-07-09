# OAV-3: Streaming TTS seam — Chirp 3 HD interim + CosyVoice Sarah clone

Date: 2026-07-09
Lane: openagents#8613 (epic #8610), spec
`openagents/docs/sarah/2026-07-09-owned-avatar-video-pipeline-spec.md` §3/§8.

Hydralisk owns Sarah's TTS behind one seam so the OAV-2 avatar render service
and future `apps/sarah` surfaces can swap backends without contract changes.

## 1. The seam

`hydralisk/tts/seam.py`:

```python
adapter.synthesize_stream(text, voice_ref) -> AsyncIterator[bytes]
```

- PCM contract (every adapter, no exceptions): **16-bit signed LE, 24,000 Hz,
  mono**. Adapters convert internally if their native output differs.
- `instrument_stream(adapter, text, voice_ref)` wraps any adapter with
  time-to-first-chunk and total-wall-time measurement
  (`SynthesisMetrics`: charsIn, msToFirstChunk, totalMs, chunksOut, bytesOut,
  audioSecondsOut, errorCode). Metrics never store the synthesized text.

HTTP service (`hydralisk/tts/service.py`, entry point `hydralisk-tts`,
default `127.0.0.1:8022`), following the Hydralisk proxy conventions:

- `GET /health` — public-safe (`unarmed` without a bearer token).
- `GET /hydralisk/tts/v1/capabilities` — adapter ref, default voice, PCM
  contract, receipt schema.
- `POST /hydralisk/tts/v1/synthesize` — bearer-authed
  (`HYDRALISK_TTS_BEARER_TOKEN`; fail-closed 503 when unarmed); JSON
  `{"text": "...", "voiceId"?: "...", "languageCode"?: "..."}` in; chunked
  `audio/L16;rate=24000;channels=1` out; `x-hydralisk-tts-run-ref` /
  `x-hydralisk-tts-receipt-ref` headers point at the receipt.
- `GET /hydralisk/tts/v1/receipts/{run_ref}` and
  `GET /hydralisk/tts/v1/metrics` — public-safe.
- Receipts are JSONL (`HYDRALISK_TTS_RECEIPT_PATH`, default
  `.hydralisk/tts-receipts.jsonl`), schema
  `hydralisk.tts.run_receipt.v1`: charsIn, msToFirstChunk, totalMs, bytes,
  blockers. The log refuses lines carrying `text`/`promptText`/token keys.

Adapter selection: `HYDRALISK_TTS_ADAPTER=chirp3hd` (default) or `cosyvoice`.

## 2. Chirp 3 HD interim adapter (managed, live today)

`hydralisk/tts/chirp.py` — Google Cloud TTS `streaming_synthesize` via
`TextToSpeechAsyncClient`, raw PCM at 24 kHz requested in the streaming config.
Optional dependency: `uv sync --extra tts-chirp`
(`google-cloud-texttospeech`).

**Interim voice decision:** `en-US-Chirp3-HD-Sulafat` — Google's "warm"
en-US female Chirp 3 HD voice, matching the warm/professional Sarah brief.
Override per-request via `voiceId` or per-deploy via
`HYDRALISK_TTS_CHIRP_VOICE`. Chirp 3 **Instant Custom Voice** (managed
cloning) is allow-list gated at Google; it is intentionally NOT a dependency
of this lane — the owned clone path is CosyVoice (§3).

Auth: Application Default Credentials. On the operator Mac the
`oa-mvp-automation` service-account key works as-is (no extra IAM grant was
needed; `texttospeech.googleapis.com` is enabled on `openagentsgemini`).

### Measured latency (live, 2026-07-09)

Environment: operator macOS (Darwin 25.4, Apple Silicon), residential network
→ Google Cloud TTS, project `openagentsgemini`, voice
`en-US-Chirp3-HD-Sulafat`, 89-char utterance, PCM 24 kHz out.

Adapter-direct (`instrument_stream`, no HTTP):

| run | msToFirstChunk | totalMs | audio out |
|---|---:|---:|---:|
| cold (client channel setup) | 2527 | 3245 | 5.64 s |
| warm | 174 | 1125 | 5.48 s |
| warm | 185 | 929 | 5.64 s |

End-to-end through `hydralisk-tts` (bearer-authed HTTP, receipts JSONL):

| run | msToFirstChunk | totalMs | bytesOut | audio out |
|---|---:|---:|---:|---:|
| 1 (fresh client) | 664 | 1525 | 295,680 | 6.16 s |
| 2 | 323 | 1137 | 295,680 | 6.16 s |
| 3 | 163 | 981 | 295,680 | 6.16 s |
| 4 | 170 | 1014 | 307,200 | 6.40 s |

Warm steady state is **~165–185 ms to first PCM chunk**, comfortably inside
the spec §4 "TTS first packet ~150 ms"-class budget (measured from the
operator Mac; a GCE-resident consumer should shave transport). Output audio
was decode-verified with ffmpeg (5.6–6.4 s speech, max_volume −3.3 dB).
Auth checks: no/bad bearer → 401; unarmed service → 503 fail-closed.

## 3. CosyVoice self-hosted adapter (owned clone lane)

`hydralisk/tts/cosyvoice.py` — CosyVoice `inference_zero_shot(..., stream=True)`
behind the same seam. The blocking model generator is bridged onto the event
loop through a bounded queue; output tensors are converted (and linearly
resampled if the model is not 24 kHz native) to the seam PCM contract.
Heavy imports are lazy and injectable, so `uv run pytest` stays CPU-safe with
a fake model.

Fail-closed configuration (adapter refuses to construct without an owned
reference):

```bash
export HYDRALISK_TTS_ADAPTER=cosyvoice
export HYDRALISK_TTS_COSYVOICE_CHECKOUT=/opt/CosyVoice   # repo w/ deps installed
export HYDRALISK_TTS_COSYVOICE_MODEL_DIR=pretrained_models/CosyVoice2-0.5B
export HYDRALISK_TTS_COSYVOICE_PROMPT_WAV=/opt/sarah/sarah_voice_ref_v1.wav
export HYDRALISK_TTS_COSYVOICE_PROMPT_TEXT="What's the most repetitive, annoying work happening in your business this week?"
```

The clone voice is served as `voiceId=sarah-cosyvoice-clone-v1`; the adapter
rejects foreign voice ids rather than silently serving the wrong voice.

### 3.1 Sarah voice reference (owned)

Published at `gs://openagentsgemini-oa-artifacts/sarah-avatar/voice-ref/`
(rebuildable via `scripts/extract-sarah-voice-ref.sh`):

- `sarah_voice_ref_v1.wav` (primary): clip 4
  (`v2_traced-4c7b99f9-ed57-41b1-85f2-5cecf550b1e7.mp4`) 0.00–5.40 s,
  24 kHz mono s16le, mean −22.7 dB. Exact transcript (Google STT
  `latest_long`, confidence **1.00**):
  *"What's the most repetitive, annoying work happening in your business this
  week?"*
- `sarah_voice_ref_v1_long.wav` (alternate, 11.5 s): clip 4 utterance +
  clip 0 (`v2_1783572761179-360477728.mp4`) 0.00–6.10 s. Two takes spliced;
  prefer the single-take primary unless a GPU A/B shows the longer prompt
  wins.

**Honesty note / deviation from the footage README:** the README's "loudest
clip" (`v2_traced-df7d3b47…`, mean −11.1 dB) contains **no recognizable
speech** per Google STT (two models, zero results, 0 s billed) — its loudness
is non-speech audio. The references above come from the clips where STT
confirms clean Sarah speech at 0.96–1.00 confidence. The character is fully
synthetic; the voice is ours to define and own.

### 3.2 GPU deploy (sarah-avatar-gpu-1)

Host (provisioned by OAV-1): GCE `g2-standard-8`, 1× NVIDIA L4,
`us-central1-b`, project `openagentsgemini`. CosyVoice checkout at
`~/CosyVoice`, venv `~/venvs/cosyvoice` (torch 2.3.1+cu121, CUDA available),
model `pretrained_models/CosyVoice2-0.5B` (ModelScope snapshot).

Steps (repeatable):

1. `gcloud compute ssh sarah-avatar-gpu-1 --zone us-central1-b --tunnel-through-iap`
2. Voice ref: `gsutil cp gs://openagentsgemini-oa-artifacts/sarah-avatar/voice-ref/sarah_voice_ref_v1.wav ~/oav3/`
3. Install hydralisk + serve:
   `uv sync && HYDRALISK_TTS_ADAPTER=cosyvoice HYDRALISK_TTS_COSYVOICE_CHECKOUT=$HOME/CosyVoice \
   HYDRALISK_TTS_COSYVOICE_PROMPT_WAV=$HOME/oav3/sarah_voice_ref_v1.wav \
   HYDRALISK_TTS_COSYVOICE_PROMPT_TEXT="What's the most repetitive, annoying work happening in your business this week?" \
   HYDRALISK_TTS_BEARER_TOKEN=<out-of-band> uv run hydralisk-tts --host 127.0.0.1 --port 8022`
   (raw service stays localhost; front with the same Caddy/HTTPS pattern as
   the Khala lanes if remote access is needed).
4. Latency gate: run the streaming measurement (§3.3) and require warm
   msToFirstChunk in the ~150 ms class before wiring OAV-2 to this adapter.

### 3.3 Measured clone latency (L4)

Measured live on `sarah-avatar-gpu-1` (1× L4, torch 2.3.1+cu121,
`CosyVoice2-0.5B` HF snapshot, stock PyTorch — no fp16/JIT/TRT/vLLM
acceleration), streaming zero-shot with the primary Sarah reference,
89-char utterance, 2026-07-09:

Per-call zero-shot (prompt processed every call):

| run | msToFirstChunk | totalMs | audio out | RTF |
|---|---:|---:|---:|---:|
| 0 (first inference, triton autotune) | 6839 | 12360 | 6.56 s | 1.884 |
| 1 | 8221 | 8229 | 5.96 s | 1.381 |
| 2 | 6980 | 8185 | 7.84 s | 1.044 |
| 3 | 6025 | 6942 | 7.00 s | 0.992 |

Precomputed speaker (`add_zero_shot_spk("sarah")` once, 1.86 s, then
`zero_shot_spk_id="sarah"` per call — the production shape):

| run | msToFirstChunk | totalMs | audio out | RTF |
|---|---:|---:|---:|---:|
| 0 | 3053 | 7303 | 6.88 s | 1.061 |
| 1 | 5836 | 6774 | 7.64 s | 0.887 |
| 2 | 5379 | 6169 | 7.08 s | 0.871 |
| 3 | 5377 | 6274 | 7.04 s | 0.891 |
| 4 | 5117 | 5123 | 6.32 s | 0.811 |

**Honest read:** the clone works, but the stock PyTorch path on L4 is not a
real-time streaming lane. Warm RTF is ~0.81–0.89 (just under real time) and
time-to-first-chunk is 3–5.8 s; on several runs the "stream" delivered nearly
all audio in one late chunk. The published ~150 ms first-packet figure
assumes CosyVoice's accelerated bidirectional-streaming deployment (fp16 +
JIT/TensorRT flow, vLLM-served LLM — see `vllm_example.py` and
`runtime/triton_trtllm` in the CosyVoice repo), not this baseline. Until that
acceleration lane is built and measured, **Chirp 3 HD remains the live
low-latency path** and CosyVoice is the offline/clone-quality lane (fine for
OAV-5 pre-rendered takes today).

Clone-quality note (measured proxy, not a human listen): the L4 clone sample
of an unseen sentence STT-transcribes back to exactly the input text at 0.92
confidence, 24 kHz output, levels consistent with the reference
(mean −19.3 dB). Sample at `/tmp/oav3_sarah_clone_sample.wav` on the host;
human A/B against the Chirp interim still pending.

Next steps for the CosyVoice lane (in order): `fp16=True` + `load_jit` /
`load_trt` on the flow model, vLLM for the CosyVoice LLM, then re-gate against
the ~150 ms-class target before pointing OAV-2 at `HYDRALISK_TTS_ADAPTER=cosyvoice`.

## 4. What OAV-2 consumes

Either adapter, same call:

```
POST /hydralisk/tts/v1/synthesize   {"text": "<sarah reply chunk>"}
→ chunked PCM s16le 24 kHz mono; first chunk gates the first inpainted frame
```

Switching the render service from the Chirp interim to the owned CosyVoice
clone is one env flip (`HYDRALISK_TTS_ADAPTER`) or one base-URL change; the
PCM contract, auth shape, receipts, and instrumentation are identical.
