"""Sarah avatar render service (OAV-2).

Owned real-time avatar rendering for the Sarah surface: an idle/listening/
speaking state machine over the catalogued footage clips, a MuseTalk-class
mouth-inpainting backend (env-gated on GPU + weights), WebRTC egress, and a
bearer-authed HTTP + WebSocket control API mirroring the LiveAvatar LITE turn
cycle (`agent.speak` PCM chunks / `agent.speak_end` / `agent.interrupt` /
listening states / keepalive).

The frame scheduler, mirror-index clip looping, silence-passthrough
compositing, and WebRTC track pacing are ported from the Apache-2.0
LiveTalking project (https://github.com/lipku/livetalking) — credit where the
production patterns came from. Everything here imports, unit-tests, and runs a
CPU no-op renderer without a GPU; the MuseTalk backend activates only when
CUDA, weights, and preprocessed avatar references are present.
"""

from hydralisk.avatar.state import AvatarState, AvatarStateMachine

__all__ = ["AvatarState", "AvatarStateMachine"]
