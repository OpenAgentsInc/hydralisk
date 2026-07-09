# OAV-2 MuseTalk GPU Smoke — Sarah Avatar Render Service

Date: 2026-07-09
Lane: openagents#8612 (epic openagents#8610)
Hydralisk commits under test: `80383251250f` and `78faf3c371e3`
Host: `sarah-avatar-gpu-1` (GCE `g2-standard-8`, us-central1-b, 1x NVIDIA L4)

## Why this smoke exists

The first OAV-2 host smoke proved the control API, state machine, WebRTC
egress, and receipts on the target L4, but it honestly ran `cpu-noop` because
the MuseTalk weights and per-clip references were not yet staged. This pass
closes that gap: the host ran with `HYDRALISK_AVATAR_RENDERER=musetalk`,
`musetalkBlockers: []`, LiveTalking/MuseTalk weights, and preprocessed
references for clips 0, 3, 4, 5, 6, and 8.

## Fixes landed before the passing smoke

- `80383251250f` adapts Hydralisk to the LiveTalking checkout on the host,
  whose `Audio2Feature.feature2chunks()` requires `batch_size`.
- `78faf3c371e3` keeps the WebRTC stream at 24fps on one L4 by running
  MuseTalk every other speaking frame and holding the last inpainted frame
  between GPU passes. Without that real-time stride, MuseTalk no longer
  crashed, but speaking phases delivered only about 12fps.

## Host state

Public-safe capability check:

```json
{
  "schema": "hydralisk.avatar.capabilities.v1",
  "protocol": "hydralisk.avatar.control.v1",
  "video": {
    "fps": 24,
    "honestFps": false,
    "targetFps": 24,
    "width": 1280,
    "height": 720,
    "codecPath": "webrtc"
  },
  "webrtcAvailable": true,
  "rendererBackends": {
    "requested": "musetalk",
    "musetalkBlockers": []
  },
  "publicSafe": true
}
```

Reference layout:

- `~/avatar-data/clip0` → `coords.pkl`, `mask_coords.pkl`, `latents.pt`,
  384 `full_imgs`, 384 masks
- `~/avatar-data/clip3` → `coords.pkl`, `mask_coords.pkl`, `latents.pt`,
  384 `full_imgs`, 384 masks
- `~/avatar-data/clip4` → `coords.pkl`, `mask_coords.pkl`, `latents.pt`,
  384 `full_imgs`, 384 masks
- `~/avatar-data/clip5` → `coords.pkl`, `mask_coords.pkl`, `latents.pt`,
  384 `full_imgs`, 384 masks
- `~/avatar-data/clip6` → `coords.pkl`, `mask_coords.pkl`, `latents.pt`,
  384 `full_imgs`, 384 masks
- `~/avatar-data/clip8` → `coords.pkl`, `mask_coords.pkl`, `latents.pt`,
  370 `full_imgs`, 370 masks

Dependency note: GPU packages are host-managed, per the runbook. After pulling
Hydralisk code, use an editable install (`uv pip install -e .`) rather than
`uv sync`, because `uv sync` removes the manually staged torch/cv2/MuseTalk
runtime stack that is intentionally not pinned in `pyproject.toml`.

## Real-session simulator

Command shape:

```bash
HYDRALISK_AVATAR_BEARER_TOKEN=<from systemd env> \
  hydralisk-avatar-sim \
  --base https://34.63.208.229.sslip.io \
  --min-fps 18 \
  --max-gap 1.0 \
  --idle-seconds 10
```

Result:

```json
{
  "sessionId": "hydralisk-avatar-0b7ce40a41cd4bfba530b8cf8b8a0a64",
  "phases": [
    {
      "phase": "connect",
      "seconds": 2.1,
      "frames": 49,
      "fps": 23.31,
      "maxInterFrameGapSeconds": 0.084
    },
    {
      "phase": "utterance_1",
      "seconds": 10.43,
      "frames": 250,
      "fps": 23.96,
      "maxInterFrameGapSeconds": 0.746
    },
    {
      "phase": "between_turns",
      "seconds": 5.0,
      "frames": 120,
      "fps": 24.0,
      "maxInterFrameGapSeconds": 0.072
    },
    {
      "phase": "utterance_2",
      "seconds": 10.44,
      "frames": 251,
      "fps": 24.03,
      "maxInterFrameGapSeconds": 0.17
    },
    {
      "phase": "idle_10s",
      "seconds": 10.0,
      "frames": 240,
      "fps": 24.0,
      "maxInterFrameGapSeconds": 0.068
    }
  ],
  "totalFramesReceived": 910,
  "serverStatus": {
    "framesRendered": 914,
    "speakingFrames": 318,
    "utterances": 2,
    "state": "idle",
    "stopReason": null
  },
  "failures": [],
  "pass": true
}
```

## Verdict

OAV-2 is real on the target GPU host: the service mints a MuseTalk-backed
session, negotiates browser-faithful WebRTC, survives two sentence-streamed
speaking turns plus an idle tail, sustains the simulator's frame-rate gate, and
returns to idle with no render error.
