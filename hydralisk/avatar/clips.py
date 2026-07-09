"""Sarah footage clip catalog and clip-cycle selection.

Clip metadata comes from the footage README
(`gs://openagentsgemini-oa-artifacts/sarah-avatar/footage/README.md`): ten
8-second H.264 1280x720 @ 24 fps clips with catalogued expression states.
State-to-clip routing follows the OAV spec (§2): idle loops clips 6/3,
listening uses clips 4/5, speaking re-lips the strong talking candidates
(clips 8/0) with MuseTalk.

`mirror_index` is ported from LiveTalking (`utils/image.py`, Apache-2.0,
https://github.com/lipku/livetalking): ping-pong looping over a frame cycle
so the loop point never jump-cuts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hydralisk.avatar.state import AvatarState


CLIP_DURATION_SECONDS = 8.0
CLIP_FPS = 24
FRAMES_PER_CLIP = int(CLIP_DURATION_SECONDS * CLIP_FPS)  # 192


class ClipRole(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    FLAIR = "flair"


@dataclass(frozen=True)
class ClipSpec:
    index: int
    filename: str
    role: ClipRole
    note: str
    mean_db: float


CLIP_CATALOG: tuple[ClipSpec, ...] = (
    ClipSpec(
        0,
        "v2_1783572761179-360477728.mp4",
        ClipRole.SPEAKING,
        "Neutral/engaged framing; ends neutral. Talking candidate.",
        -18.6,
    ),
    ClipSpec(
        1,
        "v2_traced-11963824-fda9-4485-ab77-e7b9f8c68566.mp4",
        ClipRole.SPEAKING,
        "Neutral start; ends mid-speech with mouth open.",
        -25.0,
    ),
    ClipSpec(
        2,
        "v2_traced-17fc6c67-b5d3-4ad8-a99f-bf74ee5b7195.mp4",
        ClipRole.SPEAKING,
        "Neutral start; ends speaking with mouth open.",
        -21.7,
    ),
    ClipSpec(
        3,
        "v2_traced-47fc42d3-158b-40e1-8da0-ca8e7fddc09f.mp4",
        ClipRole.IDLE,
        "Neutral/basic framing; ends neutral. Good filler/idle candidate.",
        -21.3,
    ),
    ClipSpec(
        4,
        "v2_traced-4c7b99f9-ed57-41b1-85f2-5cecf550b1e7.mp4",
        ClipRole.LISTENING,
        "Ends with a large smile. More animated/listening-like.",
        -23.2,
    ),
    ClipSpec(
        5,
        "v2_traced-6d38289c-5a5d-4c73-8e8f-cdefbce77a36.mp4",
        ClipRole.LISTENING,
        "Ends with subtle smile. Good transition clip.",
        -19.8,
    ),
    ClipSpec(
        6,
        "v2_traced-947babe3-9108-420b-b7f4-18463ac20443.mp4",
        ClipRole.IDLE,
        "Neutral/basic framing; ends neutral. Best quiet idle candidate.",
        -46.9,
    ),
    ClipSpec(
        7,
        "v2_traced-df47de0a-9102-4fee-a094-d9cc5e49d994.mp4",
        ClipRole.FLAIR,
        "Strong hologram/UI overlay; intro/outro flair only.",
        -22.0,
    ),
    ClipSpec(
        8,
        "v2_traced-df7d3b47-8156-4a45-aa68-d92cbf9401f3.mp4",
        ClipRole.SPEAKING,
        "Neutral/basic framing; loudest clip. Best talking candidate.",
        -11.1,
    ),
    ClipSpec(
        9,
        "v2_traced-f276a1fc-3009-4257-bf88-8f2874bd984e.mp4",
        ClipRole.FLAIR,
        "Circular UI/ring framing; less disruptive flair.",
        -23.6,
    ),
)


# State → ordered clip indices (spec §2: idle 6/3, listening 4/5,
# speaking 8/0 as the strong talking candidates for re-lipping).
STATE_CLIP_CYCLE: dict[AvatarState, tuple[int, ...]] = {
    AvatarState.IDLE: (6, 3),
    AvatarState.LISTENING: (4, 5),
    AvatarState.SPEAKING: (8, 0),
}


def clip_by_index(index: int) -> ClipSpec:
    return CLIP_CATALOG[index]


def mirror_index(size: int, index: int) -> int:
    """Ping-pong index over a cycle of `size` frames.

    Ported from LiveTalking `utils/image.py` (Apache-2.0).
    """
    if size <= 0:
        raise ValueError("mirror_index size must be positive")
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    return size - res - 1


@dataclass(frozen=True)
class ClipFrameRef:
    """A concrete (clip, frame) position inside a state's clip cycle."""

    clip_index: int
    frame_index: int


class ClipCatalog:
    """Maps (state, monotonically-increasing cursor) → (clip, frame).

    Each state's clips are treated as one concatenated frame cycle and
    looped with `mirror_index`, so playback ping-pongs and never jump-cuts
    at the loop boundary.
    """

    def __init__(
        self,
        *,
        frames_per_clip: int = FRAMES_PER_CLIP,
        state_cycles: dict[AvatarState, tuple[int, ...]] | None = None,
    ) -> None:
        if frames_per_clip <= 0:
            raise ValueError("frames_per_clip must be positive")
        self.frames_per_clip = frames_per_clip
        self.state_cycles = dict(state_cycles or STATE_CLIP_CYCLE)
        for state, clips in self.state_cycles.items():
            if not clips:
                raise ValueError(f"state {state} has an empty clip cycle")

    def cycle_length(self, state: AvatarState) -> int:
        return len(self.state_cycles[state]) * self.frames_per_clip

    def frame_for(self, state: AvatarState, cursor: int) -> ClipFrameRef:
        if cursor < 0:
            raise ValueError("cursor must be non-negative")
        clips = self.state_cycles[state]
        cycle = self.cycle_length(state)
        position = mirror_index(cycle, cursor)
        clip_slot, frame_index = divmod(position, self.frames_per_clip)
        return ClipFrameRef(
            clip_index=clips[clip_slot], frame_index=frame_index
        )
