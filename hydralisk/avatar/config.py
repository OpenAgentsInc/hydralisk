from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_flag(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on", "ready"}


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if value is None:
        return default
    return float(value)


@dataclass(frozen=True)
class AvatarSettings:
    """Environment-driven settings for the avatar render service.

    All GPU-facing paths are optional: with none of them set the service
    imports, serves the control API, and renders through the CPU no-op
    backend so tests and CI pass on any machine.
    """

    bearer_token: str | None = None
    allow_insecure_dev: bool = False

    # Output stream geometry (the footage is 1280x720 @ 24 fps).
    # SQ-4 (#8621): when the L4 cannot sustain 24, set HYDRALISK_AVATAR_FPS=20
    # and HYDRALISK_AVATAR_HONEST_FPS_LABEL=1 so capabilities report the real
    # paced fps instead of advertising 24 while delivering ~20.
    fps: int = 24
    honest_fps_label: bool = False
    width: int = 1280
    height: int = 720

    # Control-plane audio: LITE-cycle PCM is 16-bit mono 24 kHz.
    sample_rate: int = 24000
    # MuseTalk whisper features run at 16 kHz.
    feature_sample_rate: int = 16000

    # Renderer backend: "auto" | "cpu" | "musetalk".
    renderer_backend: str = "auto"

    # Footage + preprocessed avatar references (per-clip frames/coords/
    # masks/latents produced by the OAV-1 preprocessing pass).
    footage_dir: Path | None = None
    avatar_data_dir: Path | None = None
    # A LiveTalking checkout whose MuseTalk modules the GPU backend reuses.
    musetalk_repo: Path | None = None
    musetalk_batch_size: int = 4

    # Session policy: one L4 serves one stream in v1.
    max_sessions: int = 1
    keepalive_timeout_seconds: float = 60.0
    crossfade_frames: int = 6
    # Jitter buffer before lips move (~200 ms at 24 fps is ~5 frames).
    jitter_buffer_frames: int = 5

    receipt_dir: Path = Path(".hydralisk/avatar-receipts")

    # Public HTTPS base URL of this service (Caddy front), used to build the
    # absolute capability-URL `webrtc.offer_url` returned by the OAV-4 compat
    # mint. None keeps the offer_url relative.
    public_base_url: str | None = None

    gpu_name: str = "NVIDIA L4"
    gpu_class: str = "l4"
    gpu_count: int = 1

    @property
    def frame_period_seconds(self) -> float:
        return 1.0 / self.fps

    @property
    def audio_chunks_per_frame(self) -> int:
        return 2

    @property
    def chunk_samples(self) -> int:
        # Two audio chunks per video frame, LiveTalking-style
        # (sample_rate // (fps * 2)).
        return self.sample_rate // (self.fps * self.audio_chunks_per_frame)


def load_avatar_settings() -> AvatarSettings:
    def _path(name: str) -> Path | None:
        value = _env(name)
        return Path(value) if value else None

    return AvatarSettings(
        bearer_token=_env("HYDRALISK_AVATAR_BEARER_TOKEN"),
        allow_insecure_dev=_env_flag("HYDRALISK_AVATAR_ALLOW_INSECURE_DEV"),
        fps=_env_int("HYDRALISK_AVATAR_FPS", 24),
        honest_fps_label=_env_flag("HYDRALISK_AVATAR_HONEST_FPS_LABEL"),
        width=_env_int("HYDRALISK_AVATAR_WIDTH", 1280),
        height=_env_int("HYDRALISK_AVATAR_HEIGHT", 720),
        sample_rate=_env_int("HYDRALISK_AVATAR_SAMPLE_RATE", 24000),
        feature_sample_rate=_env_int(
            "HYDRALISK_AVATAR_FEATURE_SAMPLE_RATE", 16000
        ),
        renderer_backend=_env("HYDRALISK_AVATAR_RENDERER", "auto") or "auto",
        footage_dir=_path("HYDRALISK_AVATAR_FOOTAGE_DIR"),
        avatar_data_dir=_path("HYDRALISK_AVATAR_DATA_DIR"),
        musetalk_repo=_path("HYDRALISK_AVATAR_MUSETALK_REPO"),
        musetalk_batch_size=_env_int("HYDRALISK_AVATAR_MUSETALK_BATCH_SIZE", 4),
        max_sessions=_env_int("HYDRALISK_AVATAR_MAX_SESSIONS", 1),
        keepalive_timeout_seconds=_env_float(
            "HYDRALISK_AVATAR_KEEPALIVE_TIMEOUT_SECONDS", 60.0
        ),
        crossfade_frames=_env_int("HYDRALISK_AVATAR_CROSSFADE_FRAMES", 6),
        jitter_buffer_frames=_env_int("HYDRALISK_AVATAR_JITTER_FRAMES", 5),
        receipt_dir=Path(
            _env("HYDRALISK_AVATAR_RECEIPT_DIR", ".hydralisk/avatar-receipts")
            or ".hydralisk/avatar-receipts"
        ),
        public_base_url=_env("HYDRALISK_AVATAR_PUBLIC_BASE_URL"),
        gpu_name=_env("HYDRALISK_AVATAR_GPU_NAME", "NVIDIA L4") or "NVIDIA L4",
        gpu_class=_env("HYDRALISK_AVATAR_GPU_CLASS", "l4") or "l4",
        gpu_count=_env_int("HYDRALISK_AVATAR_GPU_COUNT", 1),
    )
