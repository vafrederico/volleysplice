"""Pinned, frozen DINOv2 frame embeddings for the Track T experiment.

The extractor is deliberately separate from the temporal model.  It never
downloads code or weights, records the exact local repository/checkpoint
identity, and publishes a cache only after every timestamp has a validated
embedding.  The cache is an experiment artifact, not ground truth and never
contains label-derived values.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .features import VideoError, probe_video


DINO_CACHE_SCHEMA_VERSION = 1
DINO_MODEL_NAME = "dinov2_vits14"
DINO_REPOSITORY = "facebookresearch/dinov2"
DINO_EMBEDDING_DIMENSION = 384
DINO_TOKEN_COUNT = 10
DINO_PATCH_SIZE = 14
DEFAULT_DINO_SAMPLE_FPS = 4.0
# The 224/336 no-label sweep selected the largest qualified input on the
# measured 10-GB RTX 3080.  Keep this frozen until a declared resource or
# accuracy ablation changes the configuration.
DEFAULT_DINO_INPUT_SIZE = 336
DEFAULT_DINO_BATCH_SIZE = 8
DEFAULT_DINO_CHUNK_SIZE = 256
IMAGENET_RGB_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_RGB_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class DinoEmbeddingError(RuntimeError):
    """Raised when a frozen DINO extractor or cache is unsafe to use."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_id(recording_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", recording_id).strip("-.")
    return value or "recording"


@dataclass(frozen=True)
class DinoExtractorConfig:
    """All preprocessing and cache-shaping choices that affect embeddings."""

    sample_fps: float = DEFAULT_DINO_SAMPLE_FPS
    input_size: int = DEFAULT_DINO_INPUT_SIZE
    batch_size: int = DEFAULT_DINO_BATCH_SIZE
    chunk_size: int = DEFAULT_DINO_CHUNK_SIZE
    patch_size: int = DINO_PATCH_SIZE

    def validate(self) -> None:
        if not math.isfinite(self.sample_fps) or self.sample_fps <= 0 or self.sample_fps > 4:
            raise ValueError("sample_fps must be finite and in (0, 4]")
        if self.input_size < self.patch_size or self.input_size % self.patch_size:
            raise ValueError("input_size must be a positive multiple of patch_size")
        if self.batch_size < 1 or self.chunk_size < 1:
            raise ValueError("batch_size and chunk_size must be positive")
        if self.patch_size != DINO_PATCH_SIZE:
            raise ValueError("Track T requires the DINOv2 ViT-S/14 patch size")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampleFps": float(self.sample_fps),
            "inputSize": int(self.input_size),
            "batchSize": int(self.batch_size),
            "chunkSize": int(self.chunk_size),
            "patchSize": int(self.patch_size),
            "letterbox": "preserve-aspect-ratio-to-square",
            "normalization": {
                "mean": [float(item) for item in IMAGENET_RGB_MEAN],
                "std": [float(item) for item in IMAGENET_RGB_STD],
            },
        }

    @property
    def config_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class DinoCache:
    path: Path
    timestamps: np.ndarray
    tokens: np.ndarray
    metadata: Mapping[str, Any]


def sample_timestamps(duration: float, sample_fps: float = DEFAULT_DINO_SAMPLE_FPS) -> np.ndarray:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be finite and positive")
    count = max(1, int(math.ceil(duration * sample_fps - 1e-9)))
    values = np.arange(count, dtype=np.float64) / sample_fps
    return np.ascontiguousarray(values[values < duration + 1e-9])


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - exercised in environment setup
        raise VideoError(
            "OpenCV is required for DINOv2 extraction; install analysis/requirements.txt"
        ) from error
    return cv2


def _crop_frame(
    frame: np.ndarray,
    roi: tuple[float, float, float, float] | None,
) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("frame must be a nonempty BGR image")
    if roi is None:
        return frame
    x, y, width, height = roi
    frame_height, frame_width = frame.shape[:2]
    left = min(frame_width - 1, max(0, int(round(x * frame_width))))
    top = min(frame_height - 1, max(0, int(round(y * frame_height))))
    right = min(frame_width, max(left + 1, int(round((x + width) * frame_width))))
    bottom = min(frame_height, max(top + 1, int(round((y + height) * frame_height))))
    result = frame[top:bottom, left:right]
    if not result.size:
        raise ValueError("the declared ROI produced an empty frame")
    return result


def letterbox_frame(
    frame: np.ndarray,
    input_size: int = DEFAULT_DINO_INPUT_SIZE,
) -> np.ndarray:
    """Convert a BGR frame to a square RGB uint8 DINO input."""

    if input_size < DINO_PATCH_SIZE or input_size % DINO_PATCH_SIZE:
        raise ValueError("input_size must be a positive multiple of 14")
    cv2 = _cv2()
    cropped = _crop_frame(frame, None)
    height, width = cropped.shape[:2]
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, min(input_size, int(round(width * scale))))
    resized_height = max(1, min(input_size, int(round(height * scale))))
    rgb = np.ascontiguousarray(cropped[:, :, ::-1])
    resized = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    left = (input_size - resized_width) // 2
    top = (input_size - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def preprocess_frames(
    frames: Sequence[np.ndarray],
    rois: Sequence[tuple[float, float, float, float] | None] | None = None,
    *,
    input_size: int = DEFAULT_DINO_INPUT_SIZE,
) -> np.ndarray:
    """Return normalized RGB frames in NCHW float32 layout."""

    if not frames:
        raise ValueError("at least one frame is required")
    if rois is None:
        rois = (None,) * len(frames)
    if len(rois) != len(frames):
        raise ValueError("frames and rois must have equal lengths")
    rows: list[np.ndarray] = []
    for frame, roi in zip(frames, rois, strict=True):
        image = letterbox_frame(_crop_frame(frame, roi), input_size)
        values = image.astype(np.float32) / np.float32(255.0)
        values = (values - IMAGENET_RGB_MEAN) / IMAGENET_RGB_STD
        rows.append(np.ascontiguousarray(values.transpose(2, 0, 1)))
    result = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
    if not np.isfinite(result).all():
        raise DinoEmbeddingError("DINO preprocessing produced non-finite values")
    return result


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise DinoEmbeddingError(
            "PyTorch is required for DINOv2; install a CUDA wheel selected for this WSL host"
        ) from error
    return torch


def _patch_pool(torch: Any, patch_tokens: Any, input_size: int) -> Any:
    if patch_tokens.ndim != 3:
        raise DinoEmbeddingError("DINO patch tokens must have shape [batch, patches, dim]")
    batch, patch_count, dimension = patch_tokens.shape
    grid = int(math.isqrt(int(patch_count)))
    if grid * grid != patch_count:
        raise DinoEmbeddingError(
            f"DINO patch count {patch_count} is not a square for input size {input_size}"
        )
    if grid != input_size // DINO_PATCH_SIZE:
        raise DinoEmbeddingError("DINO patch grid does not match the configured input size")
    values = patch_tokens.transpose(1, 2).reshape(batch, dimension, grid, grid)
    pooled = torch.nn.functional.adaptive_avg_pool2d(values, (3, 3))
    return pooled.reshape(batch, dimension, 9).transpose(1, 2)


def _extract_feature_tokens(model: Any, torch: Any, inputs: Any, input_size: int) -> Any:
    if hasattr(model, "forward_features"):
        output = model.forward_features(inputs)
    else:
        output = model(inputs)
    if isinstance(output, Mapping):
        class_token = output.get("x_norm_clstoken", output.get("x_prenorm_clstoken"))
        patch_tokens = output.get("x_norm_patchtokens", output.get("x_prenorm_patchtokens"))
    elif isinstance(output, (tuple, list)) and len(output) >= 2:
        class_token, patch_tokens = output[0], output[1]
    else:
        raise DinoEmbeddingError(
            "the pinned DINOv2 model did not expose class and patch tokens"
        )
    if class_token is None or patch_tokens is None:
        raise DinoEmbeddingError("DINOv2 output is missing normalized class/patch tokens")
    if class_token.ndim != 2 or class_token.shape[-1] != DINO_EMBEDDING_DIMENSION:
        raise DinoEmbeddingError("DINO class token is not [batch, 384]")
    if patch_tokens.shape[-1] != DINO_EMBEDDING_DIMENSION:
        raise DinoEmbeddingError("DINO patch token dimension is not 384")
    regions = _patch_pool(torch, patch_tokens, input_size)
    result = torch.cat((class_token.unsqueeze(1), regions), dim=1)
    if result.shape[1:] != (DINO_TOKEN_COUNT, DINO_EMBEDDING_DIMENSION):
        raise DinoEmbeddingError("DINO pooled representation is not [batch, 10, 384]")
    return result


class DinoV2Backbone:
    """A frozen local DINOv2 model with a deterministic pooled representation."""

    def __init__(
        self,
        model: Any,
        *,
        repository: str | Path,
        repository_commit: str,
        checkpoint: str | Path,
        checkpoint_sha256: str,
        device: str = "cpu",
    ) -> None:
        torch = _torch()
        if not _COMMIT_RE.fullmatch(repository_commit.lower()):
            raise ValueError("repository_commit must be a full 40-character hexadecimal commit")
        if not _SHA256_RE.fullmatch(checkpoint_sha256.lower()):
            raise ValueError("checkpoint_sha256 must be a 64-character hexadecimal digest")
        self.torch = torch
        self.model = model.to(device).eval()
        self.device = str(device)
        self.repository = str(Path(repository).expanduser().resolve())
        self.repository_commit = repository_commit.lower()
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.checkpoint_sha256 = checkpoint_sha256.lower()
        self.embedding_dimension = DINO_EMBEDDING_DIMENSION
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def identity(self) -> dict[str, Any]:
        return {
            "modelName": DINO_MODEL_NAME,
            "repository": self.repository,
            "repositoryCommit": self.repository_commit,
            "checkpoint": self.checkpoint,
            "checkpointSha256": self.checkpoint_sha256,
            "embeddingDimension": self.embedding_dimension,
            "tokenCount": DINO_TOKEN_COUNT,
            "patchSize": DINO_PATCH_SIZE,
        }

    def embed_frames(
        self,
        frames: Sequence[np.ndarray],
        rois: Sequence[tuple[float, float, float, float] | None] | None = None,
        *,
        input_size: int = DEFAULT_DINO_INPUT_SIZE,
    ) -> np.ndarray:
        values = preprocess_frames(frames, rois, input_size=input_size)
        inputs = self.torch.from_numpy(values).to(self.device)
        with self.torch.inference_mode():
            tokens = _extract_feature_tokens(self.model, self.torch, inputs, input_size)
        result = tokens.detach().to("cpu").float().numpy()
        if result.shape != (len(frames), DINO_TOKEN_COUNT, DINO_EMBEDDING_DIMENSION):
            raise DinoEmbeddingError("DINO output shape changed after extraction")
        if not np.isfinite(result).all():
            raise DinoEmbeddingError("DINO output contains non-finite values")
        return np.ascontiguousarray(result, dtype=np.float32)


def _load_checkpoint_state(path: Path, torch: Any) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older supported torch versions
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "teacher", "student"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    if not isinstance(payload, Mapping) or not payload:
        raise DinoEmbeddingError("DINO checkpoint does not contain a state dictionary")
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        name = key
        for prefix in ("module.", "student.", "teacher."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        normalized[name] = value
    return normalized


def load_pinned_dinov2(
    repository: str | Path,
    *,
    repository_commit: str,
    checkpoint: str | Path,
    checkpoint_sha256: str,
    device: str = "cuda",
) -> DinoV2Backbone:
    """Load DINOv2 only from an explicitly pinned local checkout and weight file."""

    repo = Path(repository).expanduser().resolve()
    weights = Path(checkpoint).expanduser().resolve()
    if not repo.is_dir() or not (repo / "hubconf.py").is_file():
        raise DinoEmbeddingError(f"local DINOv2 repository with hubconf.py is required: {repo}")
    if not weights.is_file():
        raise DinoEmbeddingError(f"local DINOv2 checkpoint does not exist: {weights}")
    if not _COMMIT_RE.fullmatch(repository_commit.lower()):
        raise ValueError("repository_commit must be a full 40-character hexadecimal commit")
    actual_sha256 = sha256_file(weights)
    if actual_sha256 != checkpoint_sha256.lower():
        raise DinoEmbeddingError(
            f"DINO checkpoint SHA-256 mismatch: expected {checkpoint_sha256}, got {actual_sha256}"
        )
    torch = _torch()
    try:
        model = torch.hub.load(
            str(repo),
            DINO_MODEL_NAME,
            source="local",
            pretrained=False,
        )
    except Exception as error:
        raise DinoEmbeddingError(f"could not load DINOv2 from local checkout {repo}: {error}") from error
    state = _load_checkpoint_state(weights, torch)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise DinoEmbeddingError(
            "pinned DINOv2 checkpoint did not load all model parameters: "
            f"missing={list(missing)[:8]}"
        )
    if unexpected and any(not str(item).startswith(("head.", "mask_token")) for item in unexpected):
        raise DinoEmbeddingError(f"pinned DINOv2 checkpoint has unexpected parameters: {list(unexpected)[:8]}")
    return DinoV2Backbone(
        model,
        repository=repo,
        repository_commit=repository_commit,
        checkpoint=weights,
        checkpoint_sha256=actual_sha256,
        device=device,
    )


def qualify_backbone(
    backbone: DinoV2Backbone,
    frame: np.ndarray,
    *,
    input_sizes: Sequence[int] = (224, 336),
    rois: Sequence[tuple[float, float, float, float] | None] | None = None,
) -> dict[str, Any]:
    """Run the no-label deterministic/memory/resource probe for one frame."""

    torch = backbone.torch
    if not input_sizes:
        raise ValueError("at least one input size is required")
    rows: list[dict[str, Any]] = []
    for input_size in input_sizes:
        if input_size < DINO_PATCH_SIZE or input_size % DINO_PATCH_SIZE:
            raise ValueError("resource-sweep input sizes must be multiples of 14")
        if torch.cuda.is_available() and str(backbone.device).startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        output = backbone.embed_frames((frame,), rois, input_size=input_size)
        if torch.cuda.is_available() and str(backbone.device).startswith("cuda"):
            torch.cuda.synchronize()
            allocated = int(torch.cuda.max_memory_allocated())
            reserved = int(torch.cuda.max_memory_reserved())
        else:
            allocated = reserved = 0
        rows.append(
            {
                "inputSize": int(input_size),
                "shape": list(output.shape),
                "finite": bool(np.isfinite(output).all()),
                "wallSeconds": round(time.perf_counter() - started, 6),
                "maxMemoryAllocatedBytes": allocated,
                "maxMemoryReservedBytes": reserved,
            }
        )
    first = backbone.embed_frames((frame,), rois, input_size=int(input_sizes[0]))
    second = backbone.embed_frames((frame,), rois, input_size=int(input_sizes[0]))
    difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
    return {
        "backbone": backbone.identity(),
        "torch": {
            "version": getattr(torch, "__version__", "unknown"),
            "cudaAvailable": bool(torch.cuda.is_available()),
            "device": str(backbone.device),
        },
        "determinism": {
            "maxAbsoluteDifference": float(np.max(difference)),
            "meanAbsoluteDifference": float(np.mean(difference)),
            "tolerance": 1e-5,
            "passed": bool(np.max(difference) <= 1e-5),
        },
        "runs": rows,
    }


def cache_path_for(
    cache_dir: str | Path,
    recording_id: str,
    recording_content_sha256: str,
    config: DinoExtractorConfig,
    backbone: DinoV2Backbone,
) -> Path:
    if not _SHA256_RE.fullmatch(recording_content_sha256.lower()):
        raise ValueError("recording_content_sha256 must be a 64-character SHA-256 digest")
    identity = {
        "recordingId": recording_id,
        "recordingContentSha256": recording_content_sha256.lower(),
        "extractorConfig": config.to_dict(),
        "extractorConfigSha256": config.config_sha256,
        "backbone": backbone.identity(),
    }
    digest = _canonical_sha256(identity)
    return (
        Path(cache_dir).expanduser().resolve()
        / f"{_safe_id(recording_id)}-{digest[:20]}"
        / "cache.npz"
    )


def _write_npz_no_replace(
    destination: Path,
    *,
    timestamps: np.ndarray,
    tokens: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing DINO cache: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                timestamps=np.asarray(timestamps, dtype=np.float64),
                tokens=np.asarray(tokens, dtype=np.float16),
                metadata_json=np.asarray(json.dumps(dict(metadata), sort_keys=True, allow_nan=False)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_chunk_no_replace(path: Path, timestamps: np.ndarray, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                timestamps=np.asarray(timestamps, dtype=np.float64),
                tokens=np.asarray(tokens, dtype=np.float16),
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            temporary.replace(path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def _load_chunk(path: Path, expected_length: int) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
            tokens = np.asarray(payload["tokens"], dtype=np.float32)
    except (OSError, KeyError, ValueError):
        return None
    if (
        timestamps.ndim != 1
        or len(timestamps) != expected_length
        or tokens.shape != (expected_length, DINO_TOKEN_COUNT, DINO_EMBEDDING_DIMENSION)
        or not np.isfinite(timestamps).all()
        or not np.isfinite(tokens).all()
    ):
        return None
    return np.ascontiguousarray(timestamps), np.ascontiguousarray(tokens)


def load_dino_cache(
    path: str | Path,
    *,
    recording_id: str | None = None,
    recording_content_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_backbone: Mapping[str, Any] | None = None,
) -> DinoCache:
    resolved = Path(path).expanduser().resolve()
    try:
        with np.load(resolved, allow_pickle=False) as payload:
            timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
            tokens = np.asarray(payload["tokens"], dtype=np.float32)
            metadata = json.loads(str(payload["metadata_json"].item()))
    except (OSError, KeyError, ValueError, json.JSONDecodeError, AttributeError) as error:
        raise DinoEmbeddingError(f"invalid DINO cache {resolved}: {error}") from error
    if not isinstance(metadata, dict):
        raise DinoEmbeddingError(f"DINO cache metadata is not an object: {resolved}")
    if (
        metadata.get("schemaVersion") != DINO_CACHE_SCHEMA_VERSION
        or metadata.get("completed") is not True
        or timestamps.ndim != 1
        or not len(timestamps)
        or tokens.shape != (len(timestamps), DINO_TOKEN_COUNT, DINO_EMBEDDING_DIMENSION)
        or not np.isfinite(timestamps).all()
        or not np.isfinite(tokens).all()
        or (len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0))
    ):
        raise DinoEmbeddingError(f"DINO cache structure is invalid: {resolved}")
    if recording_id is not None and metadata.get("recordingId") != recording_id:
        raise DinoEmbeddingError(f"DINO cache recording ID mismatch: {resolved}")
    if recording_content_sha256 is not None and metadata.get("recordingContentSha256") != recording_content_sha256:
        raise DinoEmbeddingError(f"DINO cache source SHA-256 mismatch: {resolved}")
    if expected_config_sha256 is not None and metadata.get("extractorConfigSha256") != expected_config_sha256:
        raise DinoEmbeddingError(f"DINO cache extractor config mismatch: {resolved}")
    if expected_backbone is not None and metadata.get("backbone") != dict(expected_backbone):
        raise DinoEmbeddingError(f"DINO cache backbone identity mismatch: {resolved}")
    return DinoCache(
        path=resolved,
        timestamps=np.ascontiguousarray(timestamps),
        tokens=np.ascontiguousarray(tokens),
        metadata=metadata,
    )


def extract_recording_cache(
    recording: Any,
    backbone: DinoV2Backbone,
    config: DinoExtractorConfig,
    cache_dir: str | Path,
    *,
    progress: Any | None = None,
) -> tuple[DinoCache, str]:
    """Extract one recording, resuming validated chunks and never replacing a cache."""

    config.validate()
    if not getattr(recording, "content_sha256", None):
        raise DinoEmbeddingError(f"{recording.id}: manifest content SHA-256 is required")
    destination = cache_path_for(
        cache_dir,
        recording.id,
        str(recording.content_sha256),
        config,
        backbone,
    )
    if destination.exists():
        return (
            load_dino_cache(
                destination,
                recording_id=recording.id,
                recording_content_sha256=recording.content_sha256,
                expected_config_sha256=config.config_sha256,
                expected_backbone=backbone.identity(),
            ),
            "reused",
        )
    metadata = probe_video(recording.video)
    timestamps = sample_timestamps(metadata.duration, config.sample_fps)
    partial_dir = destination.parent / "chunks"
    partial_dir.mkdir(parents=True, exist_ok=True)
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(recording.video))
    if not capture.isOpened():
        raise VideoError(f"cannot open video: {recording.video}")
    chunks: list[np.ndarray] = []
    try:
        for start in range(0, len(timestamps), config.chunk_size):
            end = min(len(timestamps), start + config.chunk_size)
            chunk_path = partial_dir / f"{start:08d}-{end:08d}.npz"
            cached = _load_chunk(chunk_path, end - start) if chunk_path.exists() else None
            if cached is not None and np.array_equal(cached[0], timestamps[start:end]):
                chunks.append(cached[1])
                if progress is not None:
                    progress(f"{recording.id}: resumed DINO chunk {start}:{end}")
                continue
            frames: list[np.ndarray] = []
            for timestamp in timestamps[start:end]:
                frame_index = min(
                    max(metadata.frame_count - 1, 0),
                    max(0, int(round(float(timestamp) * metadata.fps))),
                )
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None or not frame.size:
                    raise VideoError(
                        f"{recording.id}: cannot decode DINO sample at {timestamp:.3f}s"
                    )
                frames.append(frame)
            rows: list[np.ndarray] = []
            for batch_start in range(0, len(frames), config.batch_size):
                batch_frames = frames[batch_start : batch_start + config.batch_size]
                rows.append(
                    backbone.embed_frames(
                        batch_frames,
                        (recording.roi,) * len(batch_frames),
                        input_size=config.input_size,
                    )
                )
            values = np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float32)
            _write_chunk_no_replace(chunk_path, timestamps[start:end], values)
            chunks.append(values)
            if progress is not None:
                progress(f"{recording.id}: embedded {end}/{len(timestamps)} timestamps")
    finally:
        capture.release()
    tokens = np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)
    if tokens.shape != (len(timestamps), DINO_TOKEN_COUNT, DINO_EMBEDDING_DIMENSION):
        raise DinoEmbeddingError(f"{recording.id}: extracted DINO tensor has invalid shape {tokens.shape}")
    cache_metadata: dict[str, Any] = {
        "schemaVersion": DINO_CACHE_SCHEMA_VERSION,
        "kind": "volleycut-frozen-dinov2-vits14-cache",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "recordingId": recording.id,
        "sourceVideoPath": str(Path(recording.video).expanduser().resolve()),
        "recordingContentSha256": recording.content_sha256,
        "video": metadata.to_dict(),
        "analysisTimestamps": {
            "sampleFps": float(config.sample_fps),
            "count": len(timestamps),
            "sha256": hashlib.sha256(np.ascontiguousarray(timestamps).tobytes()).hexdigest(),
            "cadence": "exact 4 fps grid; last partial tick omitted",
        },
        "roi": list(recording.roi) if recording.roi is not None else None,
        "roiRule": "declared court ROI when present; otherwise full proxy frame",
        "preprocessing": config.to_dict(),
        "extractorConfigSha256": config.config_sha256,
        "backbone": backbone.identity(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": getattr(backbone.torch, "__version__", "unknown"),
            "device": backbone.device,
        },
        "array": {
            "shape": list(tokens.shape),
            "layout": "timestamp,token,dimension",
            "dtype": "float16",
            "classTokenIndex": 0,
            "regionTokenIndexes": list(range(1, DINO_TOKEN_COUNT)),
        },
        "labelsUsed": False,
        "completed": True,
        "chunks": {
            "chunkSize": int(config.chunk_size),
            "count": int(math.ceil(len(timestamps) / config.chunk_size)),
            "directory": str(partial_dir),
        },
    }
    _write_npz_no_replace(
        destination,
        timestamps=timestamps,
        tokens=tokens,
        metadata=cache_metadata,
    )
    return load_dino_cache(
        destination,
        recording_id=recording.id,
        recording_content_sha256=recording.content_sha256,
        expected_config_sha256=config.config_sha256,
        expected_backbone=backbone.identity(),
    ), "created"
