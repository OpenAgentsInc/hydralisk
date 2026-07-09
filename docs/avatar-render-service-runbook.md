# Sarah Avatar Render Service Runbook (OAV-2)

Status: service code landed and CPU-tested; GPU deployment pending the
`sarah-avatar-gpu-1` host that the OAV-1 lane is provisioning. Nothing in
this runbook has been executed against a live GPU yet unless an evidence
doc under `docs/evidence/` says otherwise.

Spec: `openagents` repo,
`docs/sarah/2026-07-09-owned-avatar-video-pipeline-spec.md` (§2
architecture, §8 OAV-2). Epic: openagents#8610, lane: openagents#8612.

## What this service is

`hydralisk-avatar` renders the Sarah avatar in real time from our own
catalogued footage:

- idle → loops clips 6/3; listening → clips 4/5; speaking → frames from
  talking clips 8/0 re-lipped by MuseTalk 256x256 mouth inpainting with
  preprocessed per-clip references; interrupt → crossfade to listening.
- Egress is WebRTC (aiortc): one video track (720p24 BGR compositing) plus
  the TTS audio track (PCM mono 24 kHz).
- Control is a bearer-authed HTTP + WebSocket API mirroring the LiveAvatar
  LITE cycle (`agent.speak` base64 PCM chunks sharing an `event_id`,
  `agent.speak_end`, `agent.interrupt`, `agent.start_listening`,
  `agent.stop_listening`, `agent.keepalive`). This is the seam the
  apps/sarah OAV-4 integration calls.
- Per-session JSONL receipts (session ref, minutes, frames, interrupts,
  utterances) land in `HYDRALISK_AVATAR_RECEIPT_DIR`, public-safe.

The frame scheduler, mirror-index clip looping, silence-passthrough
compositing, MuseTalk integration shape, and WebRTC track pacing are
ported from the Apache-2.0 [LiveTalking](https://github.com/lipku/livetalking)
project.

## Control API (the OAV-4 contract)

Base port: `8020` (keep it localhost/private; front with Caddy per
`deploy/caddy/hydralisk.Caddyfile.example` if a public HTTPS origin is
needed, same posture as the LLM proxy).

| Surface | Auth | Purpose |
| --- | --- | --- |
| `GET /healthz` | none | liveness, active sessions, webrtc availability |
| `GET /avatar/capabilities` | none | public-safe manifest: protocol id, control events, clip/state routing, audio format, MuseTalk blockers |
| `POST /avatar/sessions` | bearer | create session → `sessionRef`, `controlPath`, `webrtc.offerPath` |
| `GET /avatar/sessions/{ref}` | bearer | status: state, frames, interrupts, utterances |
| `POST /avatar/sessions/{ref}/stop` | bearer | stop + summary receipt |
| `POST /avatar/sessions/{ref}/webrtc` | bearer | body `{sdp, type}` offer → answer |
| `WS /avatar/sessions/{ref}/control` | bearer header or `?token=` | the `agent.*` turn cycle |

Audio format is king: `agent.speak` audio must be base64 16-bit
little-endian **mono PCM at 24 kHz**, all chunks of one utterance sharing
one `event_id`. Wrong format renders garbled with no error (the LITE docs
say the same). The server pushes `session.state`, `agent.speak_started`,
`agent.speak_ended`, `session.keepalive_ack`, `session.error`, and
`session.stopped` events back over the socket.

Renderer selection is env-gated and fail-closed: without CUDA + weights +
preprocessed references the service still runs (CPU no-op renderer,
synthetic frames) and `GET /avatar/capabilities` lists the exact
`musetalkBlockers`. That is the CI posture; the GPU host must show an
empty blocker list before a session is considered production-real.

## GPU host bring-up (sarah-avatar-gpu-1)

Host: one NVIDIA L4 (GCP `g2-standard-8`), per the spec's one-L4-per-
concurrent-stream sizing. The OAV-1 lane owns provisioning of
`sarah-avatar-gpu-1`; do not create a competing host. Until it is
available, everything below is staged but unexecuted.

1. Base install mirrors `deploy/gce/install-hydralisk-l4.sh` (driver,
   uv, repo checkout at `/opt/hydralisk`).
2. Clone the LiveTalking reference and fetch MuseTalk weights:

   ```bash
   git clone https://github.com/lipku/livetalking /opt/livetalking
   cd /opt/livetalking
   # follow its README to download models/: musetalk 1.5, whisper, vae, unet
   pip install -r requirements.txt   # torch/cu12x wheel per its README
   ```

3. Stage footage + preprocessed references:

   ```bash
   gcloud storage cp -r \
     gs://openagentsgemini-oa-artifacts/sarah-avatar/footage /opt/sarah/footage
   # OAV-1's offline proof produces the per-clip references; layout:
   #   /opt/sarah/avatar-data/clip{0,3,4,5,6,8}/
   #     full_imgs/  coords.pkl  latents.pt  mask/  mask_coords.pkl
   ```

   The per-clip directories follow LiveTalking's `data/avatars/<id>`
   layout exactly (its `genavatar_musetalk` tooling produces them), one
   directory per catalogued clip index. Only the clips routed by the
   state machine are required: 6/3 (idle), 4/5 (listening), 8/0
   (speaking).

4. Install the service env (no secrets in tracked files):

   ```bash
   sudo tee /etc/hydralisk/avatar.env >/dev/null <<'EOF'
   HYDRALISK_AVATAR_BEARER_TOKEN=<mint one>
   HYDRALISK_AVATAR_RENDERER=auto
   HYDRALISK_AVATAR_MUSETALK_REPO=/opt/livetalking
   HYDRALISK_AVATAR_DATA_DIR=/opt/sarah/avatar-data
   HYDRALISK_AVATAR_FOOTAGE_DIR=/opt/sarah/footage
   HYDRALISK_AVATAR_RECEIPT_DIR=/var/lib/hydralisk/avatar-receipts
   EOF
   sudo chmod 600 /etc/hydralisk/avatar.env
   ```

5. Run under systemd (pattern of `deploy/systemd/hydralisk-proxy.service`,
   `ExecStart=uv run hydralisk-avatar`, `EnvironmentFile=` the env above),
   with the venv synced as `uv sync --extra avatar` (pulls aiortc) plus
   the GPU stack from the LiveTalking requirements. GPU deps are
   intentionally not pinned in `pyproject.toml` — the L4 host owns its
   torch/cu12x wheel set, and CI never installs them.

6. Scale-to-zero: same posture as the GLM lanes — the VM stops when idle;
   session mint from the OAV-4 seam is the wake trigger (Cloud Run
   watchdog or MIG-min-0 pattern per the existing GCE runbooks).

## Smoke (record as evidence when the GPU host exists)

```bash
TOKEN=... HOST=http://127.0.0.1:8020
# 1. capabilities show zero musetalkBlockers
curl -s $HOST/avatar/capabilities | jq .rendererBackends
# 2. create a session
REF=$(curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  $HOST/avatar/sessions | jq -r .sessionRef)
# 3. stream a WAV as agent.speak chunks over the control socket
uv run python - <<'PY'
# ~40 lines: open ws, chunk wav at 24k into base64 agent.speak frames,
# send speak_end, watch for agent.speak_ended; see tests/test_avatar_service.py
PY
# 4. confirm frames + receipt
curl -s -H "Authorization: Bearer $TOKEN" $HOST/avatar/sessions/$REF
curl -s -X POST -H "Authorization: Bearer $TOKEN" $HOST/avatar/sessions/$REF/stop
cat /var/lib/hydralisk/avatar-receipts/$REF.jsonl
```

Success bar (spec §4): first inpainted frame ≤ 300 ms after the first
audio chunk clears the ~200 ms jitter buffer; sustained 24 fps; interrupt
crossfades to listening within `HYDRALISK_AVATAR_CROSSFADE_FRAMES` (6
frames = 250 ms).

## Honest status

- CPU-safe surface (state machine, scheduler timing, protocol parsing,
  PCM assembly, receipts, HTTP/WS API, renderer fallback) is fully
  unit-tested (`uv run pytest tests/test_avatar_*.py`).
- The MuseTalk backend and WebRTC egress are implemented but **untested on
  GPU** — no `sarah-avatar-gpu-1` was available when this landed. The
  first live smoke must be recorded as a `docs/evidence/` entry before
  claiming the lane real, per the repo's fail-closed invariants.
- Known MuseTalk failure modes to watch in the first smoke (spec §7):
  teeth smearing, single-frame jitter, chin seam under rotation.
