"""MuseTalk mouth-inpainting backend (env-gated; GPU only).

This module never imports torch/cv2 at module import time: everything heavy
is deferred so the whole avatar package imports and unit-tests on machines
without CUDA. `musetalk_blockers` reports exactly why the backend is not
active, matching Hydralisk's fail-closed posture.

The runtime path reuses the Apache-2.0 LiveTalking MuseTalk integration
(https://github.com/lipku/livetalking, `avatars/musetalk_avatar.py`):

- a one-time preprocessing pass over our ten catalogued clips produces the
  per-clip references (`full_imgs/`, `coords.pkl`, `latents.pt`, `mask/`,
  `mask_coords.pkl`) LiveTalking's `load_avatar` expects — this is the
  "preprocess once, cheap forever" property the OAV spec relies on;
- at render time, silent frames are raw clip passthrough; speaking frames
  run whisper features over the consumed PCM, one U-Net pass per 256x256
  mouth crop, VAE decode, and paste-back into the untouched 720p frame.

Deployment layout (see docs/avatar-render-service-runbook.md):

    HYDRALISK_AVATAR_MUSETALK_REPO   → a LiveTalking checkout (provides the
                                       MuseTalk model/util modules on sys.path)
    HYDRALISK_AVATAR_DATA_DIR        → preprocessed per-clip references,
                                       one subdirectory per clip index
    weights under <repo>/models      → MuseTalk 1.5 + whisper + vae + unet
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
from typing import Any

import numpy as np

from hydralisk.avatar.audio import pcm_int16_to_float32, resample_pcm
from hydralisk.avatar.clips import STATE_CLIP_CYCLE
from hydralisk.avatar.config import AvatarSettings
from hydralisk.avatar.scheduler import FrameJob
from hydralisk.avatar.state import AvatarState


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def musetalk_blockers(settings: AvatarSettings) -> list[dict[str, str]]:
    """Why the MuseTalk backend cannot activate right now (empty = ready)."""
    blockers: list[dict[str, str]] = []

    if not _module_available("torch"):
        blockers.append(
            {
                "code": "torch_missing",
                "message": "torch is not installed in this environment.",
            }
        )
    else:
        import torch  # noqa: PLC0415 — gated import by design

        if not torch.cuda.is_available():
            blockers.append(
                {
                    "code": "cuda_unavailable",
                    "message": "torch is installed but CUDA is not available.",
                }
            )

    if not _module_available("cv2"):
        blockers.append(
            {
                "code": "opencv_missing",
                "message": "opencv-python (cv2) is not installed.",
            }
        )

    repo = settings.musetalk_repo
    if repo is None:
        blockers.append(
            {
                "code": "musetalk_repo_unset",
                "message": "HYDRALISK_AVATAR_MUSETALK_REPO is not set "
                "(needs a LiveTalking checkout).",
            }
        )
    elif not (Path(repo) / "avatars" / "musetalk").is_dir():
        blockers.append(
            {
                "code": "musetalk_repo_invalid",
                "message": "musetalk repo does not look like a LiveTalking "
                "checkout (missing avatars/musetalk).",
            }
        )
    elif not (Path(repo) / "models" / "musetalk").is_dir():
        blockers.append(
            {
                "code": "musetalk_weights_missing",
                "message": "MuseTalk weights are not present under "
                "<musetalk_repo>/models/musetalk.",
            }
        )

    data_dir = settings.avatar_data_dir
    if data_dir is None:
        blockers.append(
            {
                "code": "avatar_data_unset",
                "message": "HYDRALISK_AVATAR_DATA_DIR is not set (needs the "
                "preprocessed per-clip references).",
            }
        )
    else:
        required_clips = sorted(
            {index for clips in STATE_CLIP_CYCLE.values() for index in clips}
        )
        missing = [
            str(index)
            for index in required_clips
            if not (Path(data_dir) / f"clip{index}" / "coords.pkl").exists()
        ]
        if missing:
            blockers.append(
                {
                    "code": "avatar_data_incomplete",
                    "message": "preprocessed references missing for clips: "
                    + ", ".join(missing),
                }
            )

    return blockers


def _feature2chunks(
    audio_processor: Any,
    *,
    feature_array: np.ndarray,
    fps: int,
    batch_size: int = 1,
) -> list[Any]:
    """Call LiveTalking's drifting feature2chunks API.

    The upstream LiveTalking/MuseTalk surface has shipped with both
    `feature2chunks(feature_array, fps)` and
    `feature2chunks(feature_array, fps, batch_size, ...)`. Hydralisk renders
    one avatar frame at a time, so a batch size of 1 is the correct adapter
    value when the checkout requires it.
    """
    feature2chunks = audio_processor.feature2chunks
    try:
        params = inspect.signature(feature2chunks).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive for C shims
        params = {}

    kwargs: dict[str, Any] = {"feature_array": feature_array, "fps": fps}
    if "batch_size" in params:
        kwargs["batch_size"] = batch_size
    return feature2chunks(**kwargs)


class MuseTalkRenderer:
    """Real-time mouth inpainting over the catalogued clip library.

    Construction fails loudly if `musetalk_blockers` is non-empty; use
    `select_renderer` for the fail-toward-CPU path.
    """

    backend = "musetalk"

    def __init__(self, settings: AvatarSettings) -> None:
        blockers = musetalk_blockers(settings)
        if blockers:
            raise RuntimeError(
                "MuseTalk backend unavailable: "
                + "; ".join(b["message"] for b in blockers)
            )
        self.settings = settings
        self._repo = str(Path(settings.musetalk_repo).resolve())  # type: ignore[arg-type]
        self._data_dir = Path(settings.avatar_data_dir)  # type: ignore[arg-type]
        self._models: Any = None
        self._clips: dict[int, dict[str, Any]] = {}
        self._np = np

    def start(self) -> None:
        if self._repo not in sys.path:
            sys.path.insert(0, self._repo)

        import torch  # noqa: PLC0415

        utils = importlib.import_module("avatars.musetalk.utils.utils")
        audio2feature = importlib.import_module(
            "avatars.musetalk.whisper.audio2feature"
        )
        self._blending = importlib.import_module("avatars.musetalk.myutil")
        self._cv2 = importlib.import_module("cv2")

        vae, unet, pe = utils.load_all_model()
        device = torch.device("cuda")
        pe = pe.half().to(device)
        vae.vae = vae.vae.half().to(device)
        unet.model = unet.model.half().to(device)
        audio_processor = audio2feature.Audio2Feature(
            model_path=str(Path(self._repo) / "models" / "whisper")
        )
        self._models = {
            "vae": vae,
            "unet": unet,
            "pe": pe,
            "timesteps": torch.tensor([0], device=device),
            "audio_processor": audio_processor,
            "device": device,
        }

        for clips in STATE_CLIP_CYCLE.values():
            for index in clips:
                if index not in self._clips:
                    self._clips[index] = self._load_clip_references(index)

    def _load_clip_references(self, index: int) -> dict[str, Any]:
        # Mirrors LiveTalking load_avatar() over our per-clip layout.
        import glob
        import pickle

        import torch  # noqa: PLC0415

        clip_dir = self._data_dir / f"clip{index}"
        with open(clip_dir / "coords.pkl", "rb") as f:
            coords = pickle.load(f)
        with open(clip_dir / "mask_coords.pkl", "rb") as f:
            mask_coords = pickle.load(f)
        latents = torch.load(clip_dir / "latents.pt")

        def _read_sorted(pattern: str) -> list[np.ndarray]:
            paths = sorted(
                glob.glob(str(clip_dir / pattern)),
                key=lambda p: int(Path(p).stem),
            )
            return [self._cv2.imread(p) for p in paths]

        return {
            "frames": _read_sorted("full_imgs/*.[jpJP][pnPN]*[gG]"),
            "masks": _read_sorted("mask/*.[jpJP][pnPN]*[gG]"),
            "coords": coords,
            "mask_coords": mask_coords,
            "latents": latents,
        }

    def render(self, job: FrameJob) -> np.ndarray:
        refs = self._clips[job.clip.clip_index]
        frame_index = job.clip.frame_index % len(refs["frames"])
        base = refs["frames"][frame_index]

        if not job.speaking or job.state is not AvatarState.SPEAKING:
            return base

        # SQ-4: refuse to lip-sync over a placeholder / invalid face bbox.
        # Prefer identity passthrough over a crash or a garbage mouth crop.
        try:
            x1, y1, x2, y2 = (int(v) for v in refs["coords"][frame_index])
            if x2 <= x1 or y2 <= y1:
                return base
        except (TypeError, ValueError, IndexError):
            return base

        import torch  # noqa: PLC0415

        models = self._models
        try:
            pcm = np.concatenate(job.audio_chunks)
            pcm16k = resample_pcm(
                pcm, self.settings.sample_rate, self.settings.feature_sample_rate
            )
            audio_float = pcm_int16_to_float32(pcm16k)
            whisper_feature = models["audio_processor"].audio2feat_from_array(
                audio_float
            ) if hasattr(models["audio_processor"], "audio2feat_from_array") else (
                models["audio_processor"].audio2feat(audio_float)
            )
            chunks = _feature2chunks(
                models["audio_processor"],
                feature_array=whisper_feature,
                fps=self.settings.fps,
            )
            feature = np.stack([chunks[0]]) if chunks else None
            if feature is None:
                return base

            with torch.no_grad():
                latent = refs["latents"][frame_index % len(refs["latents"])]
                audio_batch = torch.from_numpy(feature).to(
                    device=models["unet"].device, dtype=models["unet"].model.dtype
                )
                audio_batch = models["pe"](audio_batch)
                latent_batch = latent.to(dtype=models["unet"].model.dtype)
                pred_latents = models["unet"].model(
                    latent_batch,
                    models["timesteps"],
                    encoder_hidden_states=audio_batch,
                ).sample
                pred = models["vae"].decode_latents(pred_latents)

            return self._paste_back(pred[0], refs, frame_index)
        except Exception:
            # Sustained GPU faults are handled by the session watchdog; a single
            # bad frame must never kill the tick.
            return base

    def _paste_back(
        self, pred_frame: np.ndarray, refs: dict[str, Any], idx: int
    ) -> np.ndarray:
        # LiveTalking MuseReal.paste_back_frame (Apache-2.0).
        # SQ-4 (#8621): placeholder / undetected-face bboxes must fail closed
        # to the untouched source frame — never crash the render loop on a
        # zero-size crop (the owner-observed MuseTalk placeholder-bbox class).
        ori_frame = refs["frames"][idx].copy()
        try:
            x1, y1, x2, y2 = (int(v) for v in refs["coords"][idx])
        except (TypeError, ValueError, IndexError):
            return ori_frame
        if x2 <= x1 or y2 <= y1:
            return ori_frame
        try:
            res_frame = self._cv2.resize(
                pred_frame.astype(np.uint8), (x2 - x1, y2 - y1)
            )
            return self._blending.get_image_blending(
                ori_frame,
                res_frame,
                refs["coords"][idx],
                refs["masks"][idx],
                refs["mask_coords"][idx],
            )
        except Exception:
            # Any blend/resize fault → silent identity frame (last-good class).
            return ori_frame

    def close(self) -> None:
        self._models = None
        self._clips = {}
