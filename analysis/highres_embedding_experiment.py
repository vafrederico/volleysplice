"""Frozen high-resolution appearance features and nested development study.

The warm audiovisual cache intentionally uses small frames.  This module adds
one explicitly frozen, CPU-only appearance path at a much higher input
resolution.  It extracts the penultimate MobileNetV2 representation from four
predeclared rectangle-ROI crops, L2-normalizes it, and reduces it with a
content-addressed deterministic signed projection before writing a cache.

The development study reconstructs a frozen *binary* transition/court feature
report as its control.  High-resolution summaries append exactly once at the
warm-cache timestamps; they are not multiplied over the baseline context
offsets.  Protected recordings are never prepared by the development runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import FEATURE_VERSION, DecoderConfig, FeatureConfig, TrainingConfig
from .feature_experiments import (
    DEFAULT_OBJECTIVE_MARGIN,
    DEFAULT_SIGN_CONSISTENCY,
    DEVELOPMENT_SPLITS,
    FeatureExperimentError,
    FeatureSet,
    _aggregate_artifacts,
    _fit,
    _fold_report,
    _model_fingerprint,
    _protected_summary,
    _run_candidate_fold,
    _seed,
    _select_decoder,
    _subset_prepared,
    build_fold_plan,
    objective,
    sha256_file,
)
from .features import VideoError, percentile_rank_values, probe_video
from .multistate_feature_study import (
    _load_frozen_upstream_report,
    reconstruct_frozen_upstream_features,
)
from .pipeline import (
    PreparedRecording,
    _evaluate_prepared_probabilities,
    _manifest_digest,
    _prepare_many,
)
from .schema import DatasetManifest, Recording, load_manifest
from .transition_feature_experiment import (
    DerivedFeatureBlock,
    _multi_iou_metrics,
    _paired_candidate_comparison,
    _signature_sha256,
    append_feature_blocks,
)


HIGHRES_CACHE_SCHEMA_VERSION = 1
HIGHRES_STUDY_SCHEMA_VERSION = 1
DEFAULT_BACKBONE_LAYER = "onnx_node!GlobalAveragePool_97"
DEFAULT_BACKBONE_OUTPUT_KIND = "penultimate-global-average-pool"
DEFAULT_SOURCE_DIMENSION = 1280
DEFAULT_PROJECTION_DIMENSION = 64
DEFAULT_PROJECTION_SEED = 20260812
DEFAULT_SAMPLE_FPS = 1.0
DEFAULT_SHORT_WINDOW_SECONDS = 2.0
DEFAULT_LONG_WINDOW_SECONDS = 8.0
DEFAULT_INPUT_SIZE = 224
DEFAULT_RESIZE_SIZE = 256
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)
SUPPORTED_UPSTREAM_KINDS = frozenset(
    {
        "volleycut-transition-feature-experiment-development",
        "volleycut-court-relative-feature-study-development",
        "volleycut-multistate-feature-study-development",
    }
)


@dataclass(frozen=True)
class CropSpec:
    """A normalized child rectangle relative to the recording ROI."""

    name: str
    x: float
    y: float
    width: float
    height: float
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "relativeTo": "recording ROI (or full frame when ROI is absent)",
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "interpretation": self.interpretation,
        }


CROP_SPECS = (
    CropSpec(
        "full_roi",
        0.0,
        0.0,
        1.0,
        1.0,
        "complete declared court rectangle",
    ),
    CropSpec(
        "near_endline_third",
        0.0,
        2.0 / 3.0,
        1.0,
        1.0 / 3.0,
        (
            "near/bottom ROI third for the centered-behind-endline capture contract; "
            "not assumed to be the currently serving side"
        ),
    ),
    CropSpec(
        "far_court_half",
        0.0,
        0.0,
        1.0,
        0.5,
        (
            "far/top ROI half; no serving/receiving role is inferred from its position"
        ),
    ),
    CropSpec(
        "net_strip_proxy",
        0.0,
        0.40,
        1.0,
        0.20,
        "middle horizontal strip; a rectangle proxy rather than calibrated net geometry",
    ),
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def signed_projection_matrix(
    source_dimension: int,
    output_dimension: int = DEFAULT_PROJECTION_DIMENSION,
    seed: int = DEFAULT_PROJECTION_SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a version-independent deterministic Rademacher JL projection.

    Bits come from counter-mode SHA-256 rather than a NumPy PRNG, so regenerating
    the matrix is independent of the installed NumPy random implementation.
    Columns have unit L2 norm.  This is a signed random projection, not a learned
    reduction and not an orthogonal PCA fit on the corpus.
    """

    if source_dimension < 1 or output_dimension < 1:
        raise ValueError("projection dimensions must be positive")
    needed = source_dimension * output_dimension
    bits: list[np.ndarray] = []
    produced = 0
    counter = 0
    while produced < needed:
        digest = hashlib.sha256(
            f"volleycut-highres-rademacher-v1\0{seed}\0{counter}".encode("ascii")
        ).digest()
        unpacked = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))
        bits.append(unpacked)
        produced += len(unpacked)
        counter += 1
    selected = np.concatenate(bits)[:needed]
    signs = np.where(selected == 0, -1.0, 1.0).astype(np.float32)
    matrix = signs.reshape(source_dimension, output_dimension)
    matrix /= np.float32(math.sqrt(source_dimension))
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    matrix_sha256 = hashlib.sha256(matrix.astype("<f4", copy=False).tobytes()).hexdigest()
    declaration = {
        "algorithm": "sha256-counter-rademacher-jl-v1",
        "sourceDimension": source_dimension,
        "outputDimension": output_dimension,
        "seed": seed,
        "scale": "1/sqrt(sourceDimension)",
        "matrixSha256": matrix_sha256,
    }
    declaration["specSha256"] = _canonical_sha256(declaration)
    return matrix, declaration


@dataclass(frozen=True)
class ExtractionSpec:
    backbone_path: Path
    backbone_sha256: str
    layer: str
    output_kind: str
    source_dimension: int
    projection_dimension: int
    projection_seed: int
    projection_matrix_sha256: str
    projection_spec_sha256: str
    sample_fps: float
    input_size: int
    opencv_version: str
    dnn_backend: str
    dnn_target: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "extractor": "opencv-dnn-mobilenetv2-2022apr-v1",
            "runtime": {
                "opencvVersion": self.opencv_version,
                "dnnBackend": self.dnn_backend,
                "dnnTarget": self.dnn_target,
            },
            "backbone": {
                "path": str(self.backbone_path),
                "sha256": self.backbone_sha256,
                "layer": self.layer,
                "outputKind": self.output_kind,
                "sourceDimension": self.source_dimension,
            },
            "preprocessing": {
                "inputSize": [self.input_size, self.input_size],
                "resizeSize": [
                    int(round(self.input_size * DEFAULT_RESIZE_SIZE / DEFAULT_INPUT_SIZE)),
                    int(round(self.input_size * DEFAULT_RESIZE_SIZE / DEFAULT_INPUT_SIZE)),
                ],
                "scale": "1/255",
                "colorOrder": "RGB",
                "meanRgb": list(IMAGENET_RGB_MEAN),
                "stdRgb": list(IMAGENET_RGB_STD),
                "crop": "center crop after square resize",
                "resizePolicy": (
                    "OpenCV Zoo MobileNet demo parity: BGR-to-RGB, square resize "
                    "from 224 to 256 scale, then centered model-size crop"
                ),
                "representationNormalization": "per-crop L2 before projection",
            },
            "sampling": {"fps": self.sample_fps, "timestampOriginSeconds": 0.0},
            "crops": [item.to_dict() for item in CROP_SPECS],
            "projection": {
                "algorithm": "sha256-counter-rademacher-jl-v1",
                "outputDimension": self.projection_dimension,
                "seed": self.projection_seed,
                "matrixSha256": self.projection_matrix_sha256,
                "specSha256": self.projection_spec_sha256,
            },
        }

    @property
    def config_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


class OpenCvFrozenBackbone:
    """OpenCV DNN wrapper that exposes and verifies one frozen ONNX layer."""

    def __init__(
        self,
        backbone_path: str | Path,
        *,
        layer: str = DEFAULT_BACKBONE_LAYER,
        expected_source_dimension: int = DEFAULT_SOURCE_DIMENSION,
        input_size: int = DEFAULT_INPUT_SIZE,
        cv2_module: Any | None = None,
    ) -> None:
        if expected_source_dimension < 1 or input_size < 1:
            raise ValueError("backbone dimensions must be positive")
        path = Path(backbone_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"frozen backbone does not exist: {path}; it is never downloaded implicitly"
            )
        if cv2_module is None:
            try:
                import cv2 as cv2_module
            except ImportError as error:
                raise VideoError(
                    "OpenCV is required for high-resolution embedding extraction"
                ) from error
        self.cv2 = cv2_module
        self.path = path
        self.sha256 = sha256_file(path)
        self.layer = layer
        self.expected_source_dimension = expected_source_dimension
        self.input_size = input_size
        self.net = self.cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(self.cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(self.cv2.dnn.DNN_TARGET_CPU)
        layer_names = tuple(str(item) for item in self.net.getLayerNames())
        if layer not in layer_names:
            raise VideoError(
                f"requested backbone layer {layer!r} is absent; available tail: "
                f"{layer_names[-8:]}"
            )
        probe = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        output = self._forward_one(probe)
        if output.shape != (expected_source_dimension,):
            raise VideoError(
                f"backbone layer {layer!r} produced {output.shape}, expected "
                f"({expected_source_dimension},)"
            )

    def _forward_one(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError("backbone input must be a nonempty BGR image")
        # Match the pinned model's official OpenCV Zoo preprocessing exactly:
        # BGR -> RGB, resize to 256 square, center-crop 224, then ImageNet
        # channel normalization.  For non-default test sizes the same 256/224
        # resize ratio is retained.
        resize_size = int(
            round(self.input_size * DEFAULT_RESIZE_SIZE / DEFAULT_INPUT_SIZE)
        )
        resize_size = max(self.input_size, resize_size)
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        resized = self.cv2.resize(
            rgb,
            (resize_size, resize_size),
            interpolation=self.cv2.INTER_LINEAR,
        )
        offset = (resize_size - self.input_size) // 2
        cropped = resized[
            offset : offset + self.input_size,
            offset : offset + self.input_size,
        ]
        normalized = cropped.astype(np.float32) / np.float32(255.0)
        normalized = (
            normalized - np.asarray(IMAGENET_RGB_MEAN, dtype=np.float32)
        ) / np.asarray(IMAGENET_RGB_STD, dtype=np.float32)
        blob = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])
        self.net.setInput(blob)
        output = np.asarray(self.net.forward(self.layer), dtype=np.float32).reshape(-1)
        if output.shape != (self.expected_source_dimension,) or not np.isfinite(output).all():
            raise VideoError(
                f"backbone layer {self.layer!r} returned an invalid frozen representation"
            )
        return output

    def project_crops(
        self,
        crops: Sequence[np.ndarray],
        projection: np.ndarray,
    ) -> np.ndarray:
        if projection.shape[0] != self.expected_source_dimension:
            raise ValueError("projection source dimension differs from backbone output")
        rows: list[np.ndarray] = []
        # The pinned 2022apr graph contains a fixed-batch Reshape before Gemm.
        # OpenCV shape-infers that tail even when an intermediate output is
        # requested, so crops intentionally run as individual batch-1 forwards.
        for crop in crops:
            representation = self._forward_one(crop)
            norm = float(np.linalg.norm(representation))
            normalized = representation / norm if norm > 1e-12 else representation
            rows.append(normalized @ projection)
        result = np.ascontiguousarray(np.vstack(rows), dtype=np.float32)
        if not np.isfinite(result).all():
            raise VideoError("projected backbone representations contain non-finite values")
        return result


def build_extraction_spec(
    backbone: OpenCvFrozenBackbone,
    *,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    projection_dimension: int = DEFAULT_PROJECTION_DIMENSION,
    projection_seed: int = DEFAULT_PROJECTION_SEED,
) -> tuple[ExtractionSpec, np.ndarray]:
    if not math.isfinite(sample_fps) or sample_fps <= 0 or sample_fps > 4:
        raise ValueError("embedding sample FPS must be finite and in (0, 4]")
    projection, declaration = signed_projection_matrix(
        backbone.expected_source_dimension,
        projection_dimension,
        projection_seed,
    )
    output_kind = (
        DEFAULT_BACKBONE_OUTPUT_KIND
        if backbone.layer == DEFAULT_BACKBONE_LAYER
        else "explicit-intermediate-layer"
    )
    return (
        ExtractionSpec(
            backbone_path=backbone.path,
            backbone_sha256=backbone.sha256,
            layer=backbone.layer,
            output_kind=output_kind,
            source_dimension=backbone.expected_source_dimension,
            projection_dimension=projection_dimension,
            projection_seed=projection_seed,
            projection_matrix_sha256=str(declaration["matrixSha256"]),
            projection_spec_sha256=str(declaration["specSha256"]),
            sample_fps=float(sample_fps),
            input_size=backbone.input_size,
            opencv_version=str(getattr(backbone.cv2, "__version__", "test-double")),
            dnn_backend="DNN_BACKEND_OPENCV",
            dnn_target="DNN_TARGET_CPU",
        ),
        projection,
    )


def _pixel_rectangle(
    width: int,
    height: int,
    rectangle: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    x, y, item_width, item_height = rectangle
    left = min(width - 1, max(0, int(round(x * width))))
    top = min(height - 1, max(0, int(round(y * height))))
    right = min(width, max(left + 1, int(round((x + item_width) * width))))
    bottom = min(height, max(top + 1, int(round((y + item_height) * height))))
    return left, top, right, bottom


def crop_views(
    frame: np.ndarray,
    roi: tuple[float, float, float, float] | None,
) -> tuple[np.ndarray, ...]:
    """Return the four declared BGR crop views in stable declaration order."""

    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("frame must be a nonempty BGR image")
    height, width = frame.shape[:2]
    outer = roi if roi is not None else (0.0, 0.0, 1.0, 1.0)
    left, top, right, bottom = _pixel_rectangle(width, height, outer)
    base = frame[top:bottom, left:right]
    result: list[np.ndarray] = []
    for spec in CROP_SPECS:
        child_left, child_top, child_right, child_bottom = _pixel_rectangle(
            base.shape[1],
            base.shape[0],
            (spec.x, spec.y, spec.width, spec.height),
        )
        crop = base[child_top:child_bottom, child_left:child_right]
        if not crop.size:
            raise VideoError(f"crop {spec.name!r} is empty")
        result.append(crop)
    return tuple(result)


def sample_times(duration: float, sample_fps: float) -> np.ndarray:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample FPS must be finite and positive")
    count = max(1, int(math.ceil(duration * sample_fps - 1e-9)))
    times = np.arange(count, dtype=np.float64) / sample_fps
    return times[times < duration + 1e-9]


def cache_path_for(
    cache_dir: str | Path,
    recording: Recording,
    spec: ExtractionSpec,
) -> Path:
    digest = recording.content_sha256
    if not isinstance(digest, str) or len(digest) != 64:
        raise FeatureExperimentError(
            f"{recording.id}: manifest content SHA-256 is required for high-res caching"
        )
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", recording.id).strip("-.") or "recording"
    identity = _canonical_sha256(
        {
            "recordingId": recording.id,
            "recordingContentSha256": digest,
            "extractorConfigSha256": spec.config_sha256,
        }
    )
    return Path(cache_dir).expanduser().resolve() / f"{safe_id}-{identity[:20]}.npz"


@dataclass(frozen=True)
class HighresCache:
    path: Path
    times: np.ndarray
    values: np.ndarray
    crop_names: tuple[str, ...]
    metadata: Mapping[str, Any]


def load_highres_cache(
    path: str | Path,
    *,
    recording: Recording | None = None,
    spec: ExtractionSpec | None = None,
) -> HighresCache:
    resolved = Path(path).expanduser().resolve()
    try:
        with np.load(resolved, allow_pickle=False) as payload:
            times = payload["times"].astype(np.float64, copy=False)
            values = payload["values"].astype(np.float32, copy=False)
            crop_names = tuple(str(item) for item in payload["crop_names"])
            metadata = json.loads(str(payload["metadata_json"].item()))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise FeatureExperimentError(f"invalid high-resolution cache {resolved}: {error}") from error
    if (
        not isinstance(metadata, dict)
        or metadata.get("schemaVersion") != HIGHRES_CACHE_SCHEMA_VERSION
        or times.ndim != 1
        or not len(times)
        or values.ndim != 3
        or values.shape[0] != len(times)
        or values.shape[1] != len(crop_names)
        or tuple(crop_names) != tuple(item.name for item in CROP_SPECS)
        or not np.isfinite(times).all()
        or not np.isfinite(values).all()
        or (len(times) > 1 and not np.all(np.diff(times) > 0))
    ):
        raise FeatureExperimentError(f"high-resolution cache structure is invalid: {resolved}")
    if recording is not None and (
        metadata.get("recordingId") != recording.id
        or metadata.get("recordingContentSha256") != recording.content_sha256
    ):
        raise FeatureExperimentError(
            f"high-resolution cache does not match recording {recording.id}: {resolved}"
        )
    if spec is not None and (
        metadata.get("extractorConfigSha256") != spec.config_sha256
        or metadata.get("extractor") != spec.to_dict()
        or values.shape[2] != spec.projection_dimension
    ):
        raise FeatureExperimentError(
            f"high-resolution cache extractor identity differs from the frozen spec: {resolved}"
        )
    return HighresCache(
        path=resolved,
        times=np.ascontiguousarray(times),
        values=np.ascontiguousarray(values),
        crop_names=crop_names,
        metadata=metadata,
    )


def _write_cache_no_replace(
    destination: Path,
    *,
    times: np.ndarray,
    values: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {destination}")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                times=times.astype(np.float64, copy=False),
                values=values.astype(np.float32, copy=False),
                crop_names=np.asarray([item.name for item in CROP_SPECS]),
                metadata_json=np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails rather than replacing a
        # cache another process may have completed concurrently.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def extract_recording_highres_cache(
    recording: Recording,
    backbone: OpenCvFrozenBackbone,
    spec: ExtractionSpec,
    projection: np.ndarray,
    cache_dir: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[HighresCache, str]:
    """Extract or reuse one exact cache; never replace an existing file."""

    destination = cache_path_for(cache_dir, recording, spec)
    if destination.exists():
        return load_highres_cache(destination, recording=recording, spec=spec), "reused"
    if backbone.sha256 != spec.backbone_sha256 or backbone.layer != spec.layer:
        raise FeatureExperimentError("loaded backbone differs from the extraction spec")
    if projection.shape != (spec.source_dimension, spec.projection_dimension):
        raise FeatureExperimentError("projection matrix differs from the extraction spec")
    matrix_sha256 = hashlib.sha256(
        np.ascontiguousarray(projection, dtype="<f4").tobytes()
    ).hexdigest()
    if matrix_sha256 != spec.projection_matrix_sha256:
        raise FeatureExperimentError("projection matrix content hash differs from its spec")

    metadata = probe_video(recording.video)
    times = sample_times(metadata.duration, spec.sample_fps)
    cv2 = backbone.cv2
    capture = cv2.VideoCapture(str(recording.video))
    if not capture.isOpened():
        raise VideoError(f"cannot open video: {recording.video}")
    rows: list[np.ndarray] = []
    try:
        for index, timestamp in enumerate(times):
            frame_index = min(
                max(metadata.frame_count - 1, 0),
                max(0, int(round(float(timestamp) * metadata.fps))),
            )
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None or not frame.size:
                raise VideoError(
                    f"{recording.id}: cannot decode sample at {timestamp:.3f}s "
                    f"(frame {frame_index})"
                )
            rows.append(
                backbone.project_crops(crop_views(frame, recording.roi), projection)
            )
            if progress is not None and (
                index == 0 or (index + 1) % 300 == 0 or index + 1 == len(times)
            ):
                progress(
                    f"{recording.id}: embedded {index + 1}/{len(times)} timestamps "
                    f"({len(CROP_SPECS)} crops each)"
                )
    finally:
        capture.release()
    values = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
    cache_metadata = {
        "schemaVersion": HIGHRES_CACHE_SCHEMA_VERSION,
        "kind": "volleycut-frozen-highres-projected-representations",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "recordingId": recording.id,
        "recordingContentSha256": recording.content_sha256,
        "videoDurationSeconds": metadata.duration,
        "videoWidth": metadata.width,
        "videoHeight": metadata.height,
        "videoFps": metadata.fps,
        "extractorConfigSha256": spec.config_sha256,
        "extractor": spec.to_dict(),
        "array": {
            "shape": list(values.shape),
            "layout": "timestamp,crop,projected-dimension",
            "dtype": "float32",
        },
        "labelsUsed": False,
    }
    _write_cache_no_replace(
        destination,
        times=times,
        values=values,
        metadata=cache_metadata,
    )
    return load_highres_cache(destination, recording=recording, spec=spec), "created"


def _interpolate_projected_values(
    cache: HighresCache,
    target_times: np.ndarray,
) -> np.ndarray:
    if target_times.ndim != 1 or not len(target_times):
        raise ValueError("target times must be a nonempty vector")
    flat = cache.values.reshape(len(cache.times), -1).astype(np.float64, copy=False)
    if len(cache.times) == 1:
        return np.repeat(flat, len(target_times), axis=0)
    columns = [
        np.interp(target_times, cache.times, flat[:, index])
        for index in range(flat.shape[1])
    ]
    return np.column_stack(columns)


def _centered_time_mean(
    times: np.ndarray,
    values: np.ndarray,
    window_seconds: float,
) -> np.ndarray:
    if (
        times.ndim != 1
        or values.ndim != 2
        or len(times) != len(values)
        or not len(times)
        or not math.isfinite(window_seconds)
        or window_seconds <= 0
    ):
        raise ValueError("invalid centered temporal summary input")
    half = window_seconds / 2.0
    starts = np.searchsorted(times, times - half, side="left")
    ends = np.searchsorted(times, times + half, side="right")
    cumulative = np.vstack(
        (np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0))
    )
    counts = np.maximum(ends - starts, 1).astype(np.float64)
    return (cumulative[ends] - cumulative[starts]) / counts[:, None]


def _rank_nonconstant(values: np.ndarray) -> np.ndarray:
    ranked = percentile_rank_values(values)
    for index in range(values.shape[1]):
        if float(np.ptp(values[:, index])) < 1e-12:
            ranked[:, index] = values[:, index]
    return ranked.astype(np.float32, copy=False)


def highres_feature_block(
    cache: HighresCache,
    target_times: np.ndarray,
    *,
    short_window_seconds: float = DEFAULT_SHORT_WINDOW_SECONDS,
    long_window_seconds: float = DEFAULT_LONG_WINDOW_SECONDS,
) -> DerivedFeatureBlock:
    """Align projected crops and create compact short/long mean summaries."""

    if (
        not math.isfinite(short_window_seconds)
        or not math.isfinite(long_window_seconds)
        or short_window_seconds < 1.0
        or short_window_seconds > 2.0
        or long_window_seconds < 6.0
        or long_window_seconds > 10.0
        or short_window_seconds >= long_window_seconds
    ):
        raise ValueError(
            "short window must be 1-2s, long window 6-10s, and short < long"
        )
    aligned = _interpolate_projected_values(cache, target_times)
    short = _centered_time_mean(target_times, aligned, short_window_seconds)
    long = _centered_time_mean(target_times, aligned, long_window_seconds)
    raw = np.concatenate((short, long), axis=1)
    values = _rank_nonconstant(raw)
    projection_dimension = cache.values.shape[2]
    flat_names = tuple(
        f"highres/{crop_name}/signed_projection_{dimension:03d}"
        for crop_name in cache.crop_names
        for dimension in range(projection_dimension)
    )

    def token(value: float) -> str:
        return f"{value:g}".replace(".", "p")

    names = tuple(
        f"{name}/centered_mean_w{token(window)}s"
        for window in (short_window_seconds, long_window_seconds)
        for name in flat_names
    )
    if values.shape != (len(target_times), len(names)) or not np.isfinite(values).all():
        raise FeatureExperimentError("high-resolution temporal feature block is invalid")
    per_window = len(flat_names)
    return DerivedFeatureBlock(
        values=np.ascontiguousarray(values),
        names=names,
        groups={
            f"highres:centered_mean_w{token(short_window_seconds)}s": tuple(
                range(per_window)
            ),
            f"highres:centered_mean_w{token(long_window_seconds)}s": tuple(
                range(per_window, 2 * per_window)
            ),
        },
        definitions={
            "alignment": (
                "Each projected 1-fps crop stream is linearly interpolated to the "
                "warm-cache timestamp grid; endpoints use nearest-value hold."
            ),
            "temporalSummaries": {
                "kind": "centered mean (offline-only symmetric context)",
                "shortWindowSeconds": short_window_seconds,
                "longWindowSeconds": long_window_seconds,
            },
            "normalization": (
                "Every nonconstant summary column is percentile-ranked within its "
                "recording; constants retain their extractor value."
            ),
            "appendPolicy": (
                "The high-resolution block appends once to the frozen binary substrate "
                "and is not contextualized again."
            ),
        },
    )


@dataclass(frozen=True)
class HighresStudyPrepared:
    prepared: tuple[PreparedRecording, ...]
    candidates: tuple[FeatureSet, ...]
    control_names: tuple[str, ...]
    highres_names: tuple[str, ...]
    cache_files: Mapping[str, Mapping[str, Any]]
    definitions: Mapping[str, Any]


def _validate_binary_upstream(
    manifest: DatasetManifest,
    report: Mapping[str, Any],
) -> None:
    kind = report.get("kind")
    if kind not in SUPPORTED_UPSTREAM_KINDS:
        raise FeatureExperimentError(
            f"high-resolution study requires a frozen binary transition/court report, got {kind!r}"
        )
    if (
        kind == "volleycut-multistate-feature-study-development"
        and report.get("promotionDecision", {}).get("selectedArchitecture")
        != "binary_control"
    ):
        raise FeatureExperimentError(
            "the frozen step-3 winner is multistate; this first high-resolution "
            "study intentionally requires its frozen step-2 binary report as the "
            "architecture-scoped control"
        )
    if (
        report.get("freezeStatus") != "frozen-development-selection"
        or report.get("testLabelsUsed") is not False
        or report.get("manifestFileSha256") != sha256_file(manifest.path)
        or report.get("manifestSnapshotSha256") != _manifest_digest(manifest)
        or report.get("recordingContentSha256")
        != {item.id: item.content_sha256 for item in manifest.recordings}
        or report.get("featureVersion") != FEATURE_VERSION
    ):
        raise FeatureExperimentError(
            "frozen binary upstream identity does not match the current manifest/code"
        )
    inner_limit = report.get("selectionProtocol", {}).get("innerFoldLimit")
    if inner_limit not in (None, 0):
        raise FeatureExperimentError(
            "high-resolution study requires a full nested upstream report, not a smoke run"
        )
    if report.get("selectionProtocol", {}).get("folds") != [
        item.to_dict() for item in build_fold_plan(manifest.recordings)
    ]:
        raise FeatureExperimentError("upstream source-group folds changed")


def _load_highres_upstream_report(
    path: str | Path,
) -> tuple[Path, dict[str, Any], tuple[str, ...]]:
    """Accept step 3 directly only when its frozen winner remains binary."""

    resolved = Path(path).expanduser().resolve()
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureExperimentError(
            f"cannot read upstream feature/architecture report {resolved}: {error}"
        ) from error
    if not isinstance(report, dict):
        raise FeatureExperimentError("upstream report must contain one JSON object")
    if report.get("kind") != "volleycut-multistate-feature-study-development":
        return _load_frozen_upstream_report(resolved)
    finalization = report.get("finalizationPlan")
    names = finalization.get("featureNames") if isinstance(finalization, dict) else None
    if (
        report.get("freezeStatus") != "frozen-development-selection"
        or report.get("testLabelsUsed") is not False
        or not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise FeatureExperimentError(
            "step-3 report must be a frozen unopened-test architecture selection "
            "with a unique final feature signature"
        )
    expected = finalization.get("featureSignatureSha256")
    frozen = tuple(names)
    if expected is not None and expected != _signature_sha256(frozen):
        raise FeatureExperimentError("step-3 frozen feature signature hash is invalid")
    return resolved, report, frozen


def prepare_highres_study_candidates(
    prepared: Sequence[PreparedRecording],
    feature_config: FeatureConfig,
    frozen_names: Sequence[str],
    spec: ExtractionSpec,
    cache_dir: str | Path,
    *,
    short_window_seconds: float = DEFAULT_SHORT_WINDOW_SECONDS,
    long_window_seconds: float = DEFAULT_LONG_WINDOW_SECONDS,
) -> HighresStudyPrepared:
    if not prepared:
        raise FeatureExperimentError("high-resolution study data is empty")
    controls = reconstruct_frozen_upstream_features(
        prepared, feature_config, frozen_names
    )
    control_names = controls[0].contextual_names
    augmented: list[PreparedRecording] = []
    cache_files: dict[str, Mapping[str, Any]] = {}
    first_block: DerivedFeatureBlock | None = None
    for control in controls:
        cache_path = cache_path_for(cache_dir, control.recording, spec)
        cache = load_highres_cache(
            cache_path,
            recording=control.recording,
            spec=spec,
        )
        block = highres_feature_block(
            cache,
            control.sequence.times,
            short_window_seconds=short_window_seconds,
            long_window_seconds=long_window_seconds,
        )
        if first_block is None:
            first_block = block
        elif block.names != first_block.names or block.groups != first_block.groups:
            raise FeatureExperimentError(
                "high-resolution feature signatures differ across recordings"
            )
        augmented.append(append_feature_blocks(control, block))
        cache_files[control.recording.id] = {
            "path": str(cache.path),
            "fileSha256": sha256_file(cache.path),
            "extractorConfigSha256": cache.metadata["extractorConfigSha256"],
            "recordingContentSha256": cache.metadata["recordingContentSha256"],
            "samples": len(cache.times),
        }
    assert first_block is not None
    control_count = len(control_names)
    all_count = len(augmented[0].contextual_names)
    candidates = (
        FeatureSet(
            "frozen_binary_control",
            tuple(range(control_count)),
            ("frozen_binary_upstream",),
            "frozen upstream binary feature architecture",
        ),
        FeatureSet(
            "control_plus_frozen_highres",
            tuple(range(all_count)),
            ("frozen_binary_upstream", "frozen_highres_appearance"),
            "frozen projected MobileNetV2 crop summaries",
        ),
    )
    return HighresStudyPrepared(
        prepared=tuple(augmented),
        candidates=candidates,
        control_names=control_names,
        highres_names=first_block.names,
        cache_files=cache_files,
        definitions={
            **dict(first_block.definitions),
            "extractor": spec.to_dict(),
            "extractorConfigSha256": spec.config_sha256,
            "rawSourceRepresentation": (
                "1280-value penultimate global-average-pool output, not class logits"
                if spec.output_kind == DEFAULT_BACKBONE_OUTPUT_KIND
                else spec.output_kind
            ),
            "projectedDimensionsPerCrop": spec.projection_dimension,
            "cropCount": len(CROP_SPECS),
            "featureCount": len(first_block.names),
        },
    )


def _highres_code_provenance() -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    tracked = (
        package / "highres_embedding_experiment.py",
        package / "multistate_feature_study.py",
        package / "transition_feature_experiment.py",
        package / "court_relative_feature_study.py",
        package / "feature_experiments.py",
        package / "features.py",
        package / "model.py",
        package / "pipeline.py",
        package / "metrics.py",
        package / "decoder.py",
        package.parent / "scripts" / "extract-highres-embeddings.py",
        package.parent / "scripts" / "evaluate-highres-embeddings.py",
    )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package.parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=package.parent,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        head, dirty = "unknown", True
    return {
        "gitHead": head,
        "gitDirty": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "filesSha256": {
            str(path.relative_to(package.parent)): sha256_file(path)
            for path in tracked
        },
    }


def run_development_highres_study(
    manifest: DatasetManifest,
    prepared: Sequence[PreparedRecording],
    *,
    upstream_report_path: str | Path,
    spec: ExtractionSpec,
    cache_dir: str | Path,
    short_window_seconds: float = DEFAULT_SHORT_WINDOW_SECONDS,
    long_window_seconds: float = DEFAULT_LONG_WINDOW_SECONDS,
    objective_margin: float = DEFAULT_OBJECTIVE_MARGIN,
    sign_consistency: float = DEFAULT_SIGN_CONSISTENCY,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run nested LOGO comparison without preparing any protected recording."""

    started = time.perf_counter()
    upstream_path, upstream, frozen_names = _load_highres_upstream_report(
        upstream_report_path
    )
    _validate_binary_upstream(manifest, upstream)
    if objective_margin < 0 or not 0.5 < sign_consistency <= 1.0:
        raise ValueError("invalid objective margin or sign consistency")
    development_rows = tuple(
        item for item in manifest.recordings if item.split in DEVELOPMENT_SPLITS
    )
    expected_ids = {item.id for item in development_rows}
    if {item.recording.id for item in prepared} != expected_ids or any(
        item.recording.split not in DEVELOPMENT_SPLITS for item in prepared
    ):
        raise FeatureExperimentError(
            "prepared recordings must contain exactly train+validation development rows"
        )
    feature_config = FeatureConfig.from_dict(dict(upstream["featureConfig"]))
    training_config = TrainingConfig(**dict(upstream["trainingConfig"]))
    decoder_config = DecoderConfig.from_dict(dict(upstream["baseDecoderConfig"]))
    feature_config.validate()
    training_config.validate()
    decoder_config.validate()
    study = prepare_highres_study_candidates(
        prepared,
        feature_config,
        frozen_names,
        spec,
        cache_dir,
        short_window_seconds=short_window_seconds,
        long_window_seconds=long_window_seconds,
    )
    folds = build_fold_plan(manifest.recordings)
    artifacts_by_candidate: dict[str, list[Any]] = {}
    candidate_reports: dict[str, Any] = {}
    signature = study.prepared[0].contextual_names
    total = len(study.candidates) * len(folds)
    completed = 0
    for candidate in study.candidates:
        artifacts: list[Any] = []
        for fold in folds:
            completed += 1
            if progress is not None:
                progress(
                    f"High-res experiment {completed}/{total}: {candidate.name}, "
                    f"hold out {fold.held_out_group}"
                )
            artifacts.append(
                _run_candidate_fold(
                    study.prepared,
                    fold,
                    candidate,
                    feature_config=feature_config,
                    training_config=training_config,
                    decoder_config=decoder_config,
                    inner_fold_limit=None,
                )
            )
        artifacts_by_candidate[candidate.name] = artifacts
        aggregated = _aggregate_artifacts(artifacts)
        candidate_names = tuple(signature[index] for index in candidate.indexes)
        candidate_reports[candidate.name] = {
            "featureSet": {
                "name": candidate.name,
                "featureCount": len(candidate_names),
                "featureNames": list(candidate_names),
                "featureSignatureSha256": _signature_sha256(candidate_names),
                "families": list(candidate.families),
                "interpretationSubject": candidate.interpretation_subject,
            },
            "oof": aggregated,
            "multiIouMetrics": _multi_iou_metrics(aggregated["aggregate"]),
            "outcomeSlices": aggregated["aggregate"]["outcomeSlices"],
            "outerFolds": [_fold_report(item) for item in artifacts],
        }

    control_name = "frozen_binary_control"
    candidate_name = "control_plus_frozen_highres"
    comparison = _paired_candidate_comparison(
        artifacts_by_candidate[control_name],
        artifacts_by_candidate[candidate_name],
        subject="frozen projected MobileNetV2 crop summaries",
        margin=objective_margin,
        sign_consistency=sign_consistency,
    )
    helpful = comparison["classification"]["classification"] == "helpful"
    selected_name = candidate_name if helpful else control_name
    selected_artifacts = artifacts_by_candidate[selected_name]
    pooled_prepared = [
        item for artifact in selected_artifacts for item in artifact.held_prepared
    ]
    pooled_probabilities = [
        values for artifact in selected_artifacts for values in artifact.probabilities
    ]
    frozen_decoder, decoder_selection = _select_decoder(
        pooled_prepared, pooled_probabilities, decoder_config
    )
    epoch_cap = max(
        1,
        int(round(float(np.median([item.epoch_cap for item in selected_artifacts])))),
    )
    selected_set = next(
        item for item in study.candidates if item.name == selected_name
    )
    selected_names = tuple(signature[index] for index in selected_set.indexes)
    upstream_sha256 = sha256_file(upstream_path)
    return {
        "schemaVersion": HIGHRES_STUDY_SCHEMA_VERSION,
        "kind": "volleycut-frozen-highres-embedding-study-development",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "freezeStatus": "frozen-development-selection",
        "assessmentRole": "development-only-source-group-out-of-fold-selection",
        "testLabelsUsed": False,
        "testRecordingsPrepared": False,
        "dataset": manifest.name,
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "manifestSnapshotSha256": _manifest_digest(manifest),
        "recordingContentSha256": {
            item.id: item.content_sha256 for item in manifest.recordings
        },
        "development": {
            "splits": sorted(DEVELOPMENT_SPLITS),
            "recordingIds": sorted(expected_ids),
            "sourceGroups": sorted({item.source_group for item in development_rows}),
        },
        "protected": _protected_summary(manifest.recordings),
        "featureVersion": FEATURE_VERSION,
        "featureConfig": feature_config.to_dict(),
        "trainingConfig": training_config.to_dict(),
        "baseDecoderConfig": decoder_config.to_dict(),
        "architectureScope": {
            "control": "same-feature binary logistic plus hysteresis decoder",
            "upstreamKind": upstream["kind"],
            "reason": (
                "Step 3 selected the binary control, so that frozen architecture is used."
                if upstream["kind"]
                == "volleycut-multistate-feature-study-development"
                else "A multistate architecture would require reconstructing four state "
                "heads; this first high-resolution ablation isolates appearance features "
                "on the frozen step-2-compatible binary substrate."
            ),
        },
        "upstreamBinaryReport": {
            "path": str(upstream_path),
            "sha256": upstream_sha256,
            "kind": upstream["kind"],
            "featureCount": len(study.control_names),
            "featureSignatureSha256": _signature_sha256(study.control_names),
            "selectionFrozenBeforeHighresStudy": True,
        },
        "highresFeatureDefinitions": study.definitions,
        "highresCaches": dict(study.cache_files),
        "selectionProtocol": {
            "outer": "leave one development sourceGroup out",
            "inner": "leave one remaining sourceGroup out",
            "innerFoldLimit": None,
            "folds": [item.to_dict() for item in folds],
            "decoder": "selected from pooled inner out-of-fold probabilities",
            "objective": {"eventF1": 0.55, "timeIoU": 0.30, "liveTimeRecall": 0.15},
            "objectiveMargin": objective_margin,
            "signConsistency": sign_consistency,
        },
        "metricProtocol": {
            "headlineIoU": 0.5,
            "reportedEventIoUThresholds": [0.3, 0.5, 0.7],
            "outcomeSlices": [
                "all",
                "shortAtMost3Seconds",
                "ace",
                "serviceFault",
                "ordinaryLong",
            ],
        },
        "candidates": candidate_reports,
        "pairedComparisonAgainstControl": comparison,
        "selectedCandidateForRetrospectiveTest": selected_name,
        "candidateSelection": {
            "rule": (
                "High-resolution features promote only when paired source-group objective "
                "deltas have mean and median at least +0.01 and improve at least 3 of 4 "
                "groups; otherwise retain the frozen binary control."
            ),
            "selected": selected_name,
            "highresPromoted": helpful,
            "testMetricsConsulted": False,
        },
        "finalizationPlan": {
            "candidate": selected_name,
            "featureCount": len(selected_names),
            "featureNames": list(selected_names),
            "featureSignatureSha256": _signature_sha256(selected_names),
            "epochCap": epoch_cap,
            "seed": _seed(
                training_config.seed,
                "highres-final-refit",
                upstream_sha256,
                selected_name,
            ),
            "decoder": frozen_decoder.to_dict(),
            "decoderSelectionOnDevelopmentOof": decoder_selection,
            "fitRows": "all train+validation development recordings",
            "checkpointSelection": "minimum training loss up to frozen epoch cap",
        },
        "guardrails": [
            "The extractor never downloads a model and every cache pins its backbone SHA-256.",
            "Only train and validation recordings are prepared during development.",
            "Source groups, not recordings, define every inner and outer fold.",
            "The projection is frozen from a declared seed and matrix content hash.",
            "High-resolution summaries append once and are never multiplied by context offsets.",
            "No labels, outcomes, confidence, environment, game metadata, or player counts enter features.",
            "Existing cache/report paths are reused only after identity validation and never overwritten.",
        ],
        "provenance": _highres_code_provenance(),
        "processing": {
            "wallClockSeconds": round(time.perf_counter() - started, 3),
            "featureCacheNote": (
                "Warm audiovisual and frozen high-resolution caches are read; raw video "
                "is not decoded by this development comparison."
            ),
        },
        "limitations": [
            "Only four independent development source groups are available.",
            "Rectangle crop locations are proxies, not a calibrated court homography.",
            "The pinned ONNX graph requires batch-1 forwards in OpenCV due to its fixed Reshape.",
            "One-fps representations can miss subsecond contact appearance.",
            "Centered temporal means use future context and are offline-cutter features.",
            "This first study is architecture-scoped to the frozen binary step-2-compatible control.",
            "Paired evidence classifications are practical gates, not significance tests.",
        ],
    }


def _load_frozen_highres_study(
    path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Load a full nested development study that explicitly promoted high-res.

    This is intentionally the first retrospective operation.  A failed feature
    gate therefore stops before the manifest, upstream report, caches, or test
    annotations are read.
    """

    resolved = Path(path).expanduser().resolve()
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureExperimentError(
            f"cannot read high-resolution development report {resolved}: {error}"
        ) from error
    if not isinstance(report, dict):
        raise FeatureExperimentError(
            "study report is not a frozen unopened-test high-resolution development selection"
        )
    finalization = report.get("finalizationPlan")
    protocol = report.get("selectionProtocol")
    selection = report.get("candidateSelection")
    comparison = report.get("pairedComparisonAgainstControl")
    candidates = report.get("candidates")
    if (
        report.get("schemaVersion") != HIGHRES_STUDY_SCHEMA_VERSION
        or report.get("kind")
        != "volleycut-frozen-highres-embedding-study-development"
        or report.get("freezeStatus") != "frozen-development-selection"
        or report.get("testLabelsUsed") is not False
        or report.get("testRecordingsPrepared") is not False
        or not isinstance(finalization, dict)
        or not isinstance(protocol, dict)
        or not isinstance(selection, dict)
    ):
        raise FeatureExperimentError(
            "study report is not a frozen unopened-test high-resolution development selection"
        )
    if protocol.get("innerFoldLimit") not in (None, 0):
        raise FeatureExperimentError(
            "retrospective test requires a full nested high-resolution report, not a smoke run"
        )
    if selection.get("highresPromoted") is not True:
        raise FeatureExperimentError(
            "retrospective test is gated off because high-resolution features did not "
            "pass development"
        )
    classification = comparison.get("classification") if isinstance(comparison, dict) else None
    if (
        not isinstance(comparison, dict)
        or not isinstance(candidates, dict)
        or not isinstance(classification, dict)
        or classification.get("classification") != "helpful"
        or "frozen_binary_control" not in candidates
        or "control_plus_frozen_highres" not in candidates
    ):
        raise FeatureExperimentError(
            "high-resolution promotion flag is inconsistent with paired development evidence"
        )
    selected = report.get("selectedCandidateForRetrospectiveTest")
    if (
        selected != "control_plus_frozen_highres"
        or selection.get("selected") != selected
        or finalization.get("candidate") != selected
    ):
        raise FeatureExperimentError(
            "promoted high-resolution candidate and frozen finalization plan disagree"
        )
    return resolved, report


def _validate_extraction_spec_content(spec: ExtractionSpec) -> None:
    """Verify that a supplied spec still resolves to its pinned model/projection."""

    if not spec.backbone_path.is_file():
        raise FeatureExperimentError(
            f"frozen high-resolution backbone is missing: {spec.backbone_path}"
        )
    if sha256_file(spec.backbone_path) != spec.backbone_sha256:
        raise FeatureExperimentError("frozen high-resolution backbone content changed")
    matrix, declaration = signed_projection_matrix(
        spec.source_dimension,
        spec.projection_dimension,
        spec.projection_seed,
    )
    del matrix
    if (
        declaration["matrixSha256"] != spec.projection_matrix_sha256
        or declaration["specSha256"] != spec.projection_spec_sha256
    ):
        raise FeatureExperimentError("frozen high-resolution projection identity changed")


def _validate_frozen_highres_identity(
    manifest: DatasetManifest,
    development: Mapping[str, Any],
    spec: ExtractionSpec,
    *,
    short_window_seconds: float,
    long_window_seconds: float,
) -> tuple[Path, dict[str, Any], tuple[str, ...]]:
    """Validate every identity frozen before protected-split access."""

    if development.get("manifestFileSha256") != sha256_file(manifest.path):
        raise FeatureExperimentError("high-resolution report does not match the manifest")
    if development.get("manifestSnapshotSha256") != _manifest_digest(manifest):
        raise FeatureExperimentError(
            "recording snapshots changed after the high-resolution study was frozen"
        )
    if development.get("recordingContentSha256") != {
        item.id: item.content_sha256 for item in manifest.recordings
    }:
        raise FeatureExperimentError(
            "recording content identities changed after the high-resolution study"
        )
    if development.get("featureVersion") != FEATURE_VERSION:
        raise FeatureExperimentError(
            "feature implementation version changed after the high-resolution study"
        )
    protocol = development.get("selectionProtocol")
    if not isinstance(protocol, dict) or protocol.get("folds") != [
        item.to_dict() for item in build_fold_plan(manifest.recordings)
    ]:
        raise FeatureExperimentError(
            "source-group folds changed after the high-resolution study"
        )
    current_provenance = _highres_code_provenance()
    frozen_provenance = development.get("provenance")
    if (
        not isinstance(frozen_provenance, dict)
        or frozen_provenance.get("filesSha256")
        != current_provenance.get("filesSha256")
    ):
        raise FeatureExperimentError(
            "high-resolution experiment code changed after development"
        )

    _validate_extraction_spec_content(spec)
    definitions = development.get("highresFeatureDefinitions")
    temporal = definitions.get("temporalSummaries") if isinstance(definitions, dict) else None
    short_window = temporal.get("shortWindowSeconds") if isinstance(temporal, dict) else None
    long_window = temporal.get("longWindowSeconds") if isinstance(temporal, dict) else None
    if (
        not isinstance(definitions, dict)
        or definitions.get("extractor") != spec.to_dict()
        or definitions.get("extractorConfigSha256") != spec.config_sha256
        or not isinstance(temporal, dict)
        or not isinstance(short_window, (int, float))
        or isinstance(short_window, bool)
        or not isinstance(long_window, (int, float))
        or isinstance(long_window, bool)
        or float(short_window) != float(short_window_seconds)
        or float(long_window) != float(long_window_seconds)
    ):
        raise FeatureExperimentError(
            "high-resolution extractor or temporal-summary spec changed after development"
        )

    upstream_info = development.get("upstreamBinaryReport")
    if not isinstance(upstream_info, dict) or not isinstance(
        upstream_info.get("path"), str
    ):
        raise FeatureExperimentError("frozen high-resolution upstream identity is missing")
    upstream_path, upstream, frozen_names = _load_highres_upstream_report(
        upstream_info["path"]
    )
    if sha256_file(upstream_path) != upstream_info.get("sha256"):
        raise FeatureExperimentError(
            "frozen binary upstream report changed after high-resolution development"
        )
    _validate_binary_upstream(manifest, upstream)
    if (
        upstream_info.get("kind") != upstream.get("kind")
        or upstream_info.get("featureCount") != len(frozen_names)
        or upstream_info.get("featureSignatureSha256")
        != _signature_sha256(frozen_names)
        or upstream_info.get("selectionFrozenBeforeHighresStudy") is not True
    ):
        raise FeatureExperimentError(
            "frozen binary upstream feature identity is inconsistent"
        )
    if (
        development.get("featureConfig") != upstream.get("featureConfig")
        or development.get("trainingConfig") != upstream.get("trainingConfig")
        or development.get("baseDecoderConfig") != upstream.get("baseDecoderConfig")
    ):
        raise FeatureExperimentError(
            "high-resolution report model configuration differs from its upstream"
        )
    return upstream_path, upstream, frozen_names


def _require_preextracted_highres_caches(
    recordings: Sequence[Recording],
    cache_dir: str | Path,
    spec: ExtractionSpec,
) -> None:
    missing = [
        item.id
        for item in recordings
        if not cache_path_for(cache_dir, item, spec).is_file()
    ]
    if missing:
        raise FeatureExperimentError(
            "retrospective test requires separately pre-extracted high-resolution "
            f"caches; missing recording IDs: {missing}"
        )


def _load_test_cache_extraction_index(
    path: str | Path,
    manifest: DatasetManifest,
    recordings: Sequence[Recording],
    cache_dir: str | Path,
    spec: ExtractionSpec,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Validate the explicit protected-cache extraction receipt and its hashes."""

    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureExperimentError(
            f"cannot read protected high-resolution extraction index {resolved}: {error}"
        ) from error
    rows = payload.get("recordings") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schemaVersion") != 1
        or payload.get("kind")
        != "volleycut-frozen-highres-cache-extraction-index"
        or payload.get("manifestFileSha256") != sha256_file(manifest.path)
        or payload.get("protectedSplitAcknowledged") is not True
        or payload.get("labelsUsed") is not False
        or payload.get("extractorConfigSha256") != spec.config_sha256
        or payload.get("extractor") != spec.to_dict()
        or not isinstance(rows, list)
    ):
        raise FeatureExperimentError(
            "protected high-resolution extraction index identity is invalid"
        )
    selected_splits = payload.get("selectedSplits")
    if not isinstance(selected_splits, list) or "test" not in selected_splits:
        raise FeatureExperimentError(
            "protected high-resolution extraction index did not acknowledge the test split"
        )
    expected_by_id = {item.id: item for item in recordings}
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("recordingId"), str):
            raise FeatureExperimentError(
                "protected high-resolution extraction index has an invalid recording row"
            )
        recording_id = row["recordingId"]
        if recording_id not in expected_by_id:
            continue
        if recording_id in indexed:
            raise FeatureExperimentError(
                f"{recording_id}: duplicate protected high-resolution extraction row"
            )
        recording = expected_by_id[recording_id]
        expected_path = cache_path_for(cache_dir, recording, spec)
        try:
            indexed_path = Path(row.get("cache", "")).expanduser().resolve()
        except (OSError, TypeError):
            indexed_path = Path("")
        if (
            row.get("split") != "test"
            or row.get("sourceGroup") != recording.source_group
            or row.get("recordingContentSha256") != recording.content_sha256
            or indexed_path != expected_path
            or not isinstance(row.get("cacheFileSha256"), str)
            or not isinstance(row.get("samples"), int)
            or isinstance(row.get("samples"), bool)
            or row["samples"] < 1
        ):
            raise FeatureExperimentError(
                f"{recording_id}: protected high-resolution extraction identity changed"
            )
        indexed[recording_id] = {
            "path": str(expected_path),
            "fileSha256": row["cacheFileSha256"],
            "extractorConfigSha256": spec.config_sha256,
            "recordingContentSha256": recording.content_sha256,
            "samples": row["samples"],
        }
    if set(indexed) != set(expected_by_id):
        raise FeatureExperimentError(
            "protected high-resolution extraction index does not cover every test recording"
        )
    return resolved, indexed


def _validated_highres_cache_inventory(
    recordings: Sequence[Recording],
    cache_dir: str | Path,
    spec: ExtractionSpec,
    *,
    frozen_inventory: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Read and hash exact caches without extracting or replacing anything."""

    expected_ids = {item.id for item in recordings}
    if frozen_inventory is not None and set(frozen_inventory) != expected_ids:
        raise FeatureExperimentError(
            "development high-resolution cache inventory recording IDs changed"
        )
    verified: dict[str, dict[str, Any]] = {}
    for recording in recordings:
        path = cache_path_for(cache_dir, recording, spec)
        if not path.is_file():
            raise FeatureExperimentError(
                f"{recording.id}: high-resolution cache was not pre-extracted: {path}"
            )
        cache = load_highres_cache(path, recording=recording, spec=spec)
        if (
            cache.metadata.get("kind")
            != "volleycut-frozen-highres-projected-representations"
            or cache.metadata.get("labelsUsed") is not False
        ):
            raise FeatureExperimentError(
                f"{recording.id}: high-resolution cache provenance is unsafe"
            )
        row = {
            "path": str(path),
            "fileSha256": sha256_file(path),
            "extractorConfigSha256": spec.config_sha256,
            "recordingContentSha256": recording.content_sha256,
            "samples": len(cache.times),
        }
        if frozen_inventory is not None:
            frozen = frozen_inventory.get(recording.id)
            if not isinstance(frozen, dict):
                raise FeatureExperimentError(
                    f"{recording.id}: frozen development cache record is invalid"
                )
            try:
                frozen_path = str(Path(frozen.get("path", "")).expanduser().resolve())
            except (OSError, TypeError):
                frozen_path = ""
            if frozen_path != str(path) or any(
                frozen.get(key) != row[key]
                for key in (
                    "fileSha256",
                    "extractorConfigSha256",
                    "recordingContentSha256",
                    "samples",
                )
            ):
                raise FeatureExperimentError(
                    f"{recording.id}: development high-resolution cache identity changed"
                )
        verified[recording.id] = row
    return verified


def run_retrospective_highres_test(
    manifest_path: str | Path,
    development_report_path: str | Path,
    warm_cache_dir: str | Path,
    highres_cache_dir: str | Path,
    *,
    spec: ExtractionSpec,
    test_cache_index_path: str | Path,
    short_window_seconds: float = DEFAULT_SHORT_WINDOW_SECONDS,
    long_window_seconds: float = DEFAULT_LONG_WINDOW_SECONDS,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fit a promoted frozen high-res candidate, then open test exactly once."""

    report_path, development = _load_frozen_highres_study(development_report_path)
    manifest = load_manifest(manifest_path)
    upstream_path, upstream, frozen_names = _validate_frozen_highres_identity(
        manifest,
        development,
        spec,
        short_window_seconds=short_window_seconds,
        long_window_seconds=long_window_seconds,
    )

    feature_config = FeatureConfig.from_dict(dict(development["featureConfig"]))
    training_config = TrainingConfig(**dict(development["trainingConfig"]))
    finalization = development["finalizationPlan"]
    epoch_cap = int(finalization.get("epochCap", 0))
    if epoch_cap < 1:
        raise FeatureExperimentError("frozen high-resolution epoch cap is invalid")
    decoder = DecoderConfig.from_dict(dict(finalization["decoder"]))
    seed = int(finalization["seed"])
    feature_config.validate()
    training_config.validate()
    decoder.validate()

    development_rows = tuple(
        item for item in manifest.recordings if item.split in DEVELOPMENT_SPLITS
    )
    test_rows = manifest.for_split("test")
    if not development_rows or not test_rows:
        raise FeatureExperimentError("retrospective test requires development and test rows")
    test_cache_index, frozen_test_cache_inventory = (
        _load_test_cache_extraction_index(
            test_cache_index_path,
            manifest,
            test_rows,
            highres_cache_dir,
            spec,
        )
    )
    frozen_cache_inventory = development.get("highresCaches")
    if not isinstance(frozen_cache_inventory, dict):
        raise FeatureExperimentError("development high-resolution cache inventory is missing")
    development_cache_inventory = _validated_highres_cache_inventory(
        development_rows,
        highres_cache_dir,
        spec,
        frozen_inventory=frozen_cache_inventory,
    )
    # Absence fails before the expensive final fit.  Cache extraction remains a
    # separate, explicitly acknowledged command; this evaluator never decodes
    # protected video to create a high-resolution cache.
    _require_preextracted_highres_caches(test_rows, highres_cache_dir, spec)

    if progress is not None:
        progress("Preparing frozen development features; protected test remains closed")
    development_raw = _prepare_many(
        development_rows,
        feature_config,
        warm_cache_dir,
        progress=progress,
    )
    development_study = prepare_highres_study_candidates(
        development_raw,
        feature_config,
        frozen_names,
        spec,
        highres_cache_dir,
        short_window_seconds=short_window_seconds,
        long_window_seconds=long_window_seconds,
    )
    selected_spec = next(
        (
            item
            for item in development_study.candidates
            if item.name == "control_plus_frozen_highres"
        ),
        None,
    )
    if selected_spec is None:
        raise FeatureExperimentError("promoted high-resolution candidate is not implemented")
    selected_development = _subset_prepared(
        development_study.prepared, selected_spec.indexes
    )
    selected_names = selected_development[0].contextual_names
    if (
        list(selected_names) != finalization.get("featureNames")
        or len(selected_names) != finalization.get("featureCount")
        or _signature_sha256(selected_names)
        != finalization.get("featureSignatureSha256")
    ):
        raise FeatureExperimentError(
            "frozen high-resolution feature signature changed after development"
        )
    fit_config = replace(
        training_config,
        epochs=epoch_cap,
        patience=max(training_config.patience, epoch_cap + 1),
        seed=seed,
    )
    if progress is not None:
        progress("Fitting the frozen promoted high-resolution candidate on development")
    model = _fit(
        selected_development,
        (),
        feature_config=feature_config,
        decoder=decoder,
        config=fit_config,
        seed=seed,
    )
    model.decoder = decoder

    if progress is not None:
        progress("Opening the protected retrospective test once with selection frozen")
    test_cache_inventory = _validated_highres_cache_inventory(
        test_rows,
        highres_cache_dir,
        spec,
        frozen_inventory=frozen_test_cache_inventory,
    )
    test_raw = _prepare_many(
        test_rows,
        feature_config,
        warm_cache_dir,
        progress=progress,
    )
    test_study = prepare_highres_study_candidates(
        test_raw,
        feature_config,
        frozen_names,
        spec,
        highres_cache_dir,
        short_window_seconds=short_window_seconds,
        long_window_seconds=long_window_seconds,
    )
    test_spec = next(
        item
        for item in test_study.candidates
        if item.name == "control_plus_frozen_highres"
    )
    selected_test = _subset_prepared(test_study.prepared, test_spec.indexes)
    if any(item.contextual_names != selected_names for item in selected_test):
        raise FeatureExperimentError(
            "test high-resolution feature signature differs from frozen development"
        )
    probabilities = [model.predict(item.contextual_values) for item in selected_test]
    per_recording, aggregate = _evaluate_prepared_probabilities(
        selected_test, probabilities, decoder
    )
    aggregate["objective"] = objective(aggregate)
    current_provenance = _highres_code_provenance()
    return {
        "schemaVersion": HIGHRES_STUDY_SCHEMA_VERSION,
        "kind": "volleycut-frozen-highres-embedding-study-retrospective-test",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "assessmentRole": "explicit-single-source-retrospective-regression-test-access",
        "testLabelsOpened": True,
        "selectionLockedBeforeTest": True,
        "highresPromotionGatePassed": True,
        "developmentReport": str(report_path),
        "developmentReportSha256": sha256_file(report_path),
        "upstreamBinaryReport": str(upstream_path),
        "upstreamBinaryReportSha256": sha256_file(upstream_path),
        "upstreamBinaryReportKind": upstream["kind"],
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "manifestSnapshotSha256": _manifest_digest(manifest),
        "recordingContentSha256": {
            item.id: item.content_sha256 for item in manifest.recordings
        },
        "selectedCandidate": "control_plus_frozen_highres",
        "featureCount": len(selected_names),
        "featureSignatureSha256": _signature_sha256(selected_names),
        "extractorConfigSha256": spec.config_sha256,
        "extractor": spec.to_dict(),
        "verifiedCaches": {
            "development": development_cache_inventory,
            "test": test_cache_inventory,
            "testExtractionIndex": str(test_cache_index),
            "testExtractionIndexSha256": sha256_file(test_cache_index),
            "testCachesPreextracted": True,
            "cacheExtractionPerformedByEvaluator": False,
        },
        "training": {
            "recordingIds": [item.recording.id for item in selected_development],
            "sourceGroups": sorted(
                {item.recording.source_group for item in selected_development}
            ),
            "splits": sorted(DEVELOPMENT_SPLITS),
            "testRowsUsedForFitting": False,
            "epochCap": epoch_cap,
            "seed": seed,
            "modelFingerprint": _model_fingerprint(model),
        },
        "decoder": decoder.to_dict(),
        "test": {
            "aggregate": aggregate,
            "multiIouMetrics": _multi_iou_metrics(aggregate),
            "outcomeSlices": aggregate["outcomeSlices"],
            "recordings": per_recording,
        },
        "provenance": current_provenance,
        "guardrails": [
            "The full nested development report promoted high-resolution features before test access.",
            "Manifest, recording content, code, upstream report, extractor spec, and development cache hashes were revalidated.",
            "Candidate, feature signature, epoch cap, seed, and decoder were frozen in development.",
            "Only train+validation rows fit the final model.",
            "Protected high-resolution caches had to be extracted separately with explicit acknowledgement.",
            "Test metrics cannot revise the frozen candidate or extractor.",
        ],
        "warning": (
            "This protected split is a retrospective regression check, not external "
            "generalization evidence. Do not revise this feature bank from its result."
        ),
    }
