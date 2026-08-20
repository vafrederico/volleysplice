"""Pinned quantized person localization for side-switch feature research.

The detector is the Apache-2.0 OpenCV Zoo block-quantized MediaPipe pose
detector.  It is deliberately wrapped here instead of importing OpenCV Zoo's
large generated anchor table so extraction and a later Android port share a
small, deterministic contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from analysis.artifacts import atomic_write_text


DETECTOR_SCHEMA_VERSION = 1
MODEL_ID = "opencv-zoo-mediapipe-person-int8bq-2023mar"
MODEL_FILENAME = "model.onnx"
UPSTREAM_COMMIT = "47534e27c9851bb1128ccc0102f1145e27f23f98"
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    f"{UPSTREAM_COMMIT}/models/person_detection_mediapipe/"
    "person_detection_mediapipe_2023mar_int8bq.onnx"
)
MODEL_SIZE_BYTES = 3_482_053
MODEL_SHA256 = "c5ed8c00c028b98e5d2c55b920a6e975af6c4cd538cfeea7c054f4fbbd8b9075"
LICENSE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/"
    f"{UPSTREAM_COMMIT}/models/person_detection_mediapipe/LICENSE"
)
LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
INPUT_SIZE = 224
SCORE_THRESHOLD = 0.20
NMS_THRESHOLD = 0.30
MAXIMUM_DETECTIONS = 6
TILE_COVERAGE = 0.62


class PersonDetectorError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlayerDetection:
    x: float
    y: float
    width: float
    height: float
    hip_x: float
    hip_y: float
    shoulder_x: float
    shoulder_y: float
    score: float

    @property
    def center_x(self) -> float:
        return self.x + 0.5 * self.width

    @property
    def center_y(self) -> float:
        return self.y + 0.5 * self.height

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "hipX": self.hip_x,
            "hipY": self.hip_y,
            "shoulderX": self.shoulder_x,
            "shoulderY": self.shoulder_y,
            "score": self.score,
        }


@dataclass(frozen=True)
class PersonDetectionResult:
    detections: tuple[PlayerDetection, ...]
    raw_candidates: int
    inference_milliseconds: float


@dataclass(frozen=True)
class _Tile:
    x: int
    y: int
    width: int
    height: int
    owner_column: int
    owner_row: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download_verified(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int | None = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", suffix=".download", dir=destination.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as output, opener(url) as response:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if expected_size is not None and size != expected_size:
            raise PersonDetectorError(
                f"downloaded {destination.name} has {size} bytes; expected {expected_size}"
            )
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise PersonDetectorError(
                f"downloaded {destination.name} SHA-256 {actual} does not match pin"
            )
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite detector artifact: {destination}")
        temporary.replace(destination)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def install_pinned_model(destination: str | Path) -> Path:
    """Install the exact quantized detector and its license without overwrites."""

    model_dir = Path(destination).expanduser().resolve()
    if model_dir.exists() and (not model_dir.is_dir() or any(model_dir.iterdir())):
        raise FileExistsError(f"detector destination is not empty: {model_dir}")
    created = not model_dir.exists()
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILENAME
    license_path = model_dir / "LICENSE"
    metadata_path = model_dir / "model.json"
    try:
        _download_verified(
            MODEL_URL,
            model_path,
            expected_sha256=MODEL_SHA256,
            expected_size=MODEL_SIZE_BYTES,
        )
        _download_verified(
            LICENSE_URL,
            license_path,
            expected_sha256=LICENSE_SHA256,
        )
        metadata = {
            "schemaVersion": DETECTOR_SCHEMA_VERSION,
            "modelId": MODEL_ID,
            "purpose": "quantized side-switch player localization",
            "createdAt": datetime.now(UTC).isoformat(),
            "upstream": {
                "repository": "https://github.com/opencv/opencv_zoo",
                "commit": UPSTREAM_COMMIT,
                "modelUrl": MODEL_URL,
            },
            "artifact": {
                "file": MODEL_FILENAME,
                "sizeBytes": MODEL_SIZE_BYTES,
                "sha256": MODEL_SHA256,
                "precision": "block-quantized int8, block size 64",
            },
            "license": {
                "spdx": "Apache-2.0",
                "file": license_path.name,
                "sha256": LICENSE_SHA256,
                "sourceUrl": LICENSE_URL,
            },
            "runtime": {
                "engine": "OpenCV DNN",
                "opencvVersionAtInstall": cv2.__version__,
                "target": "CPU",
            },
            "input": {
                "shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
                "color": "RGB",
                "normalization": "value / 127.5 - 1",
                "letterbox": "centered, zero after normalization",
            },
            "output": {
                "boxesAndLandmarks": [1, 2254, 12],
                "scores": [1, 2254, 1],
                "landmarkOrder": [
                    "hipCenter",
                    "fullBodyScalePoint",
                    "shoulderCenter",
                    "upperBodyScalePoint",
                ],
            },
        }
        atomic_write_text(
            metadata_path, json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
    except Exception:
        for path in (model_path, license_path, metadata_path):
            path.unlink(missing_ok=True)
        if created:
            try:
                model_dir.rmdir()
            except OSError:
                pass
        raise
    return model_dir


def load_model_metadata(model_dir: str | Path) -> dict[str, Any]:
    directory = Path(model_dir).expanduser().resolve()
    metadata_path = directory / "model.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PersonDetectorError(f"cannot load detector metadata: {error}") from error
    if not isinstance(payload, dict) or payload.get("schemaVersion") != DETECTOR_SCHEMA_VERSION:
        raise PersonDetectorError("unsupported person-detector metadata schema")
    if payload.get("modelId") != MODEL_ID:
        raise PersonDetectorError("unexpected person-detector model identity")
    model_path = directory / MODEL_FILENAME
    if (
        not model_path.is_file()
        or model_path.stat().st_size != MODEL_SIZE_BYTES
        or sha256_file(model_path) != MODEL_SHA256
    ):
        raise PersonDetectorError("person-detector bytes do not match the pin")
    license_path = directory / "LICENSE"
    if not license_path.is_file() or sha256_file(license_path) != LICENSE_SHA256:
        raise PersonDetectorError("person-detector license does not match the pin")
    return payload


def mediapipe_anchors() -> np.ndarray:
    """Generate the exact 2,254 center anchors used by the pinned graph."""

    anchors: list[tuple[float, float]] = []
    for grid_size, repeats in ((28, 2), (14, 2), (7, 6)):
        for y in range(grid_size):
            center_y = (y + 0.5) / grid_size
            for x in range(grid_size):
                center_x = (x + 0.5) / grid_size
                anchors.extend([(center_x, center_y)] * repeats)
    result = np.asarray(anchors, dtype=np.float32)
    if result.shape != (2254, 2):
        raise AssertionError(f"unexpected MediaPipe anchor shape: {result.shape}")
    return result


_ANCHORS = mediapipe_anchors()


def detector_tiles(width: int, height: int) -> tuple[_Tile, ...]:
    if width < 2 or height < 2:
        raise ValueError("detector frame must be at least 2x2")
    tile_width = min(width, max(1, round(width * TILE_COVERAGE)))
    tile_height = min(height, max(1, round(height * TILE_COVERAGE)))
    return tuple(
        _Tile(
            x=0 if column == 0 else width - tile_width,
            y=0 if row == 0 else height - tile_height,
            width=tile_width,
            height=tile_height,
            owner_column=column,
            owner_row=row,
        )
        for row in range(2)
        for column in range(2)
    )


def _preprocess(frame: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("detector input must be a non-empty BGR image")
    height, width = frame.shape[:2]
    ratio = INPUT_SIZE / max(height, width)
    resized_width = max(1, int(width * ratio))
    resized_height = max(1, int(height * ratio))
    resized = cv2.resize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    resized = resized / 127.5 - 1.0
    padded = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    left = (INPUT_SIZE - resized_width) // 2
    top = (INPUT_SIZE - resized_height) // 2
    padded[top : top + resized_height, left : left + resized_width] = resized
    blob = np.transpose(padded, (2, 0, 1))[None]
    return blob, ratio, left / ratio, top / ratio


def _iou(left: PlayerDetection, right: PlayerDetection) -> float:
    x0 = max(left.x, right.x)
    y0 = max(left.y, right.y)
    x1 = min(left.x + left.width, right.x + right.width)
    y1 = min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def _same_player(
    left: PlayerDetection,
    right: PlayerDetection,
    *,
    frame_width: int,
    frame_height: int,
) -> bool:
    hip_distance = math.hypot(
        (left.hip_x - right.hip_x) / frame_width,
        (left.hip_y - right.hip_y) / frame_height,
    )
    return _iou(left, right) >= NMS_THRESHOLD or hip_distance < 0.10


def _decode_tile(
    geometry: np.ndarray,
    raw_scores: np.ndarray,
    *,
    tile: _Tile,
    frame_width: int,
    frame_height: int,
    pad_x: float,
    pad_y: float,
) -> tuple[list[PlayerDetection], int]:
    if geometry.shape != (1, 2254, 12) or raw_scores.shape != (1, 2254, 1):
        raise PersonDetectorError(
            f"unexpected person-detector outputs: {geometry.shape}, {raw_scores.shape}"
        )
    logits = np.clip(raw_scores[0, :, 0].astype(np.float64), -100.0, 100.0)
    scores = 1.0 / (1.0 + np.exp(-logits))
    indexes = np.flatnonzero(scores >= SCORE_THRESHOLD)
    raw_count = int(len(indexes))
    if not raw_count:
        return [], 0
    scale = float(max(tile.width, tile.height))
    values = geometry[0, indexes].astype(np.float64)
    landmarks = values[:, 4:].reshape(-1, 4, 2) / INPUT_SIZE
    landmarks += _ANCHORS[indexes, None, :]
    landmarks *= scale
    landmarks -= np.asarray([pad_x, pad_y])

    result: list[PlayerDetection] = []
    for local_index, score in enumerate(scores[indexes]):
        points = landmarks[local_index]
        hip, body_scale, shoulder, upper_scale = points
        if not np.isfinite(points).all():
            continue
        torso = float(np.linalg.norm(hip - shoulder))
        body_radius = float(np.linalg.norm(body_scale - hip))
        upper_radius = float(np.linalg.norm(upper_scale - shoulder))
        if torso < max(4.0, 0.012 * max(tile.width, tile.height)):
            continue
        if torso > 0.48 * max(tile.width, tile.height):
            continue
        if not 0.75 <= body_radius / torso <= 4.5:
            continue
        if not 0.35 <= upper_radius / torso <= 3.5:
            continue
        if (hip[1] - shoulder[1]) / torso < -0.25:
            continue

        hip_global = hip + np.asarray([tile.x, tile.y])
        shoulder_global = shoulder + np.asarray([tile.x, tile.y])
        owner_column = 0 if hip_global[0] < frame_width * 0.5 else 1
        owner_row = 0 if hip_global[1] < frame_height * 0.5 else 1
        if owner_column != tile.owner_column or owner_row != tile.owner_row:
            continue
        if not (
            -0.03 * frame_width <= hip_global[0] <= 1.03 * frame_width
            and -0.03 * frame_height <= hip_global[1] <= 1.03 * frame_height
            and -0.05 * frame_width <= shoulder_global[0] <= 1.05 * frame_width
            and -0.05 * frame_height <= shoulder_global[1] <= 1.05 * frame_height
        ):
            continue

        center_x = 0.5 * (hip_global[0] + shoulder_global[0])
        half_width = 0.42 * torso
        top = min(hip_global[1], shoulder_global[1]) - 0.18 * torso
        bottom = max(hip_global[1], shoulder_global[1]) + 0.12 * torso
        left = max(0.0, center_x - half_width)
        right = min(float(frame_width), center_x + half_width)
        top = max(0.0, top)
        bottom = min(float(frame_height), bottom)
        if right - left < 3.0 or bottom - top < 3.0:
            continue
        result.append(
            PlayerDetection(
                x=float(left),
                y=float(top),
                width=float(right - left),
                height=float(bottom - top),
                hip_x=float(hip_global[0]),
                hip_y=float(hip_global[1]),
                shoulder_x=float(shoulder_global[0]),
                shoulder_y=float(shoulder_global[1]),
                score=float(score),
            )
        )
    return result, raw_count


class QuantizedPersonDetector:
    def __init__(self, model_dir: str | Path, *, opencv_threads: int = 6) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.metadata = load_model_metadata(self.model_dir)
        if opencv_threads < 1:
            raise ValueError("OpenCV thread count must be positive")
        cv2.setNumThreads(opencv_threads)
        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.model_dir / MODEL_FILENAME))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.output_names = tuple(self.net.getUnconnectedOutLayersNames())
        except cv2.error as error:
            raise PersonDetectorError(f"cannot load quantized person detector: {error}") from error
        if len(self.output_names) != 2:
            raise PersonDetectorError(f"unexpected detector output names: {self.output_names}")

    def detect(self, frame: np.ndarray) -> PersonDetectionResult:
        height, width = frame.shape[:2]
        candidates: list[PlayerDetection] = []
        raw_candidates = 0
        inference_milliseconds = 0.0
        for tile in detector_tiles(width, height):
            crop = frame[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
            blob, _, pad_x, pad_y = _preprocess(crop)
            self.net.setInput(blob)
            started = time.perf_counter()
            outputs = self.net.forward(self.output_names)
            inference_milliseconds += (time.perf_counter() - started) * 1000.0
            geometry, scores = outputs
            decoded, raw_count = _decode_tile(
                geometry,
                scores,
                tile=tile,
                frame_width=width,
                frame_height=height,
                pad_x=pad_x,
                pad_y=pad_y,
            )
            candidates.extend(decoded)
            raw_candidates += raw_count
        selected: list[PlayerDetection] = []
        for candidate in sorted(candidates, key=lambda value: value.score, reverse=True):
            if all(
                not _same_player(
                    candidate,
                    prior,
                    frame_width=width,
                    frame_height=height,
                )
                for prior in selected
            ):
                selected.append(candidate)
            if len(selected) == MAXIMUM_DETECTIONS:
                break
        return PersonDetectionResult(
            detections=tuple(selected),
            raw_candidates=raw_candidates,
            inference_milliseconds=inference_milliseconds,
        )


def detector_identity(model_dir: str | Path) -> dict[str, Any]:
    directory = Path(model_dir).expanduser().resolve()
    metadata = load_model_metadata(directory)
    return {
        "modelId": MODEL_ID,
        "directory": str(directory),
        "metadataPath": str(directory / "model.json"),
        "metadataSha256": sha256_file(directory / "model.json"),
        "modelPath": str(directory / MODEL_FILENAME),
        "modelSha256": MODEL_SHA256,
        "modelSizeBytes": MODEL_SIZE_BYTES,
        "licensePath": str(directory / "LICENSE"),
        "licenseSha256": LICENSE_SHA256,
        "upstreamCommit": metadata["upstream"]["commit"],
    }


__all__ = [
    "INPUT_SIZE",
    "LICENSE_SHA256",
    "MAXIMUM_DETECTIONS",
    "MODEL_ID",
    "MODEL_SHA256",
    "MODEL_SIZE_BYTES",
    "PersonDetectionResult",
    "PlayerDetection",
    "QuantizedPersonDetector",
    "SCORE_THRESHOLD",
    "TILE_COVERAGE",
    "detector_identity",
    "detector_tiles",
    "install_pinned_model",
    "load_model_metadata",
    "mediapipe_anchors",
]
