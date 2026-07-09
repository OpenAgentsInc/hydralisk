# OAV-2 Avatar Render Service — First Host Smoke (sarah-avatar-gpu-1)

Date: 2026-07-09
Lane: openagents#8612 (epic #8610)
Commit under test: `68b0ce8` (hydralisk main)
Host: `sarah-avatar-gpu-1` (GCE `g2-standard-8`, us-central1-b, NVIDIA L4,
driver working — provisioned by the OAV-1 lane)

## What was run

Full test suite on the host, then one live end-to-end control-cycle smoke
against `uv run hydralisk-avatar` (port 8020, random bearer token, receipts
to a scratch dir):

1. `GET /healthz` → ok, `webrtcAvailable: true` (aiortc extra installed).
2. `GET /avatar/capabilities` → fail-closed MuseTalk blockers reported:
   `torch_missing`, `opencv_missing`, `musetalk_repo_unset`,
   `avatar_data_unset` (the OAV-1 weight/reference staging has not landed
   on the host yet).
3. `POST /avatar/sessions` → session created, `renderer: cpu-noop`,
   WebRTC egress active.
4. Control WebSocket: streamed a generated 1 s 440 Hz wav (24 kHz mono
   PCM16) as `agent.speak` chunks under one `event_id`, then
   `agent.speak_end`.
5. Observed `agent.speak_started` → `session.state: speaking` →
   `agent.speak_ended` → `session.state: idle` (drain-based return).
6. Status + stop: `framesRendered: 28`, `speakingFrames: 24` over the
   ~1.1 s session at 720p24; per-session JSONL receipt written with
   `avatar.session_started` + `avatar.session_summary`
   (`schema: hydralisk.avatar.session_receipt.v1`, `minutes: 1`,
   `utterances: 1`, `interrupts: 0`, `publicSafe: true`).

Result: **PASS** for the control API, state machine, scheduler pacing,
audio assembly, egress push path, and receipts on the real target host.

Test suite on host: `175 passed, 1 skipped` (skip is the pre-existing
vLLM-checkout-dependent test, unrelated to the avatar lane).

## What was NOT smoked (honest gap)

- The **MuseTalk GPU inpainting path is untested**: no torch/cv2, no
  LiveTalking checkout, no MuseTalk weights, and no preprocessed per-clip
  references were on the host yet (that staging belongs to the OAV-1
  offline-proof lane). The renderer therefore ran `cpu-noop` synthetic
  frames.
- No browser-side WebRTC answer was negotiated (no client on the host);
  the aiortc peer/track/`av.VideoFrame` conversion path was exercised by
  the render loop pushes, not by a full SDP handshake.

Next gate before calling openagents#8612 done: stage weights + references
per `docs/avatar-render-service-runbook.md`, rerun this smoke with
`musetalkBlockers: []` and `renderer: musetalk`, and record first-frame
latency against the spec's ≤ 300 ms budget.
