from __future__ import annotations

import hashlib
import copy
import json
import math
import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .artifacts import atomic_write_text
from .ball_annotation import validate_ball_annotation_task
from .features import probe_video


BALL_DETECTOR_SCHEMA_VERSION = 1
MODEL_ID = "opencv-zoo-yolox-s-2022nov"
MODEL_FILENAME = "model.onnx"
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    "47534e27c9851bb1128ccc0102f1145e27f23f98/models/"
    "object_detection_yolox/object_detection_yolox_2022nov.onnx"
)
MODEL_SIZE_BYTES = 35_858_002
MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
UPSTREAM_COMMIT = "47534e27c9851bb1128ccc0102f1145e27f23f98"
LICENSE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/"
    f"{UPSTREAM_COMMIT}/models/object_detection_yolox/LICENSE"
)
LICENSE_SHA256 = "0ec3668d3274bcf29e8a29e9576d5a2cd96fc78d3c5bec4387355a796e5d9088"
SPORTS_BALL_CLASS_INDEX = 32
INPUT_SIZE = 640
STRIDES = (8, 16, 32)


class BallDetectorError(RuntimeError):
    """Raised when a detector artifact or suggestion run is unsafe to use."""


@dataclass(frozen=True)
class BallDetection:
    x: float
    y: float
    width: float
    height: float
    score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "score": self.score,
        }


@dataclass(frozen=True)
class FrameDetections:
    detections: tuple[BallDetection, ...]
    maximum_raw_score: float
    raw_candidates_above_floor: int
    inference_milliseconds: float


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int | None = None,
    opener: Callable[[str], Any] = urllib.request.urlopen,
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
            raise BallDetectorError(
                f"downloaded {destination.name} has {size} bytes; expected {expected_size}"
            )
        actual_digest = digest.hexdigest()
        if actual_digest != expected_sha256:
            raise BallDetectorError(
                f"downloaded {destination.name} SHA-256 {actual_digest} does not match "
                f"{expected_sha256}"
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
    """Install the pinned Apache-2.0 OpenCV Zoo model without overwriting files."""

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
            "schemaVersion": BALL_DETECTOR_SCHEMA_VERSION,
            "modelId": MODEL_ID,
            "purpose": "unvalidated volleyball-ball annotation proposals",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "upstream": {
                "repository": "https://github.com/opencv/opencv_zoo",
                "commit": UPSTREAM_COMMIT,
                "modelUrl": MODEL_URL,
            },
            "artifact": {
                "file": MODEL_FILENAME,
                "sizeBytes": MODEL_SIZE_BYTES,
                "sha256": MODEL_SHA256,
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
                "size": [INPUT_SIZE, INPUT_SIZE],
                "color": "RGB",
                "letterbox": "top-left, value 114",
                "scale": 1.0,
                "mean": None,
            },
            "output": {
                "shape": [1, 8400, 85],
                "strides": list(STRIDES),
                "classIndex": SPORTS_BALL_CLASS_INDEX,
                "className": "sports ball",
                "score": "objectness * class probability",
            },
        }
        atomic_write_text(
            metadata_path,
            json.dumps(metadata, indent=2, allow_nan=False) + "\n",
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
        raise BallDetectorError(f"cannot load detector metadata {metadata_path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schemaVersion") != BALL_DETECTOR_SCHEMA_VERSION:
        raise BallDetectorError("unsupported detector metadata schema")
    if payload.get("modelId") != MODEL_ID:
        raise BallDetectorError(f"unexpected detector modelId: {payload.get('modelId')!r}")
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("file") != MODEL_FILENAME:
        raise BallDetectorError("detector artifact metadata is invalid")
    model_path = directory / MODEL_FILENAME
    if not model_path.is_file():
        raise BallDetectorError(f"detector artifact is missing: {model_path}")
    if model_path.stat().st_size != MODEL_SIZE_BYTES or sha256_file(model_path) != MODEL_SHA256:
        raise BallDetectorError("detector artifact size or SHA-256 does not match the pin")
    license_path = directory / "LICENSE"
    if not license_path.is_file() or sha256_file(license_path) != LICENSE_SHA256:
        raise BallDetectorError("detector license is missing or does not match the pin")
    return payload


def _grids_and_strides() -> tuple[np.ndarray, np.ndarray]:
    grids: list[np.ndarray] = []
    expanded: list[np.ndarray] = []
    for stride in STRIDES:
        size = INPUT_SIZE // stride
        xv, yv = np.meshgrid(np.arange(size), np.arange(size))
        grid = np.stack((xv, yv), axis=2).reshape(-1, 2).astype(np.float32)
        grids.append(grid)
        expanded.append(np.full((len(grid), 1), float(stride), dtype=np.float32))
    return np.concatenate(grids), np.concatenate(expanded)


_GRIDS, _EXPANDED_STRIDES = _grids_and_strides()


def letterbox_rgb(frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.size == 0:
        raise ValueError("frame must be a non-empty HxWx3 BGR image")
    height, width = frame_bgr.shape[:2]
    ratio = min(INPUT_SIZE / height, INPUT_SIZE / width)
    resized_width = max(1, int(width * ratio))
    resized_height = max(1, int(height * ratio))
    resized = cv2.resize(
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114.0, dtype=np.float32)
    padded[:resized_height, :resized_width] = resized
    return np.transpose(padded, (2, 0, 1))[None], ratio


def decode_sports_ball_output(
    output: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    letterbox_ratio: float,
    score_floor: float,
    nms_threshold: float,
    maximum_detections: int,
) -> tuple[tuple[BallDetection, ...], float, int]:
    if output.shape == (1, 8400, 85):
        rows = output[0].astype(np.float32, copy=True)
    elif output.shape == (8400, 85):
        rows = output.astype(np.float32, copy=True)
    else:
        raise BallDetectorError(f"unexpected YOLOX output shape: {output.shape}")
    if (
        image_width < 1
        or image_height < 1
        or not math.isfinite(letterbox_ratio)
        or letterbox_ratio <= 0
    ):
        raise ValueError("invalid image dimensions or letterbox ratio")
    if not 0 < score_floor <= 1 or not 0 <= nms_threshold <= 1:
        raise ValueError("score_floor must be in (0, 1] and nms_threshold in [0, 1]")
    if maximum_detections < 1:
        raise ValueError("maximum_detections must be positive")

    scores = rows[:, 4] * rows[:, 5 + SPORTS_BALL_CLASS_INDEX]
    maximum_raw_score = max(0.0, float(np.max(scores)))
    selected = np.flatnonzero(scores >= score_floor)
    raw_count = int(len(selected))
    if not raw_count:
        return (), maximum_raw_score, 0

    centers = (rows[selected, :2] + _GRIDS[selected]) * _EXPANDED_STRIDES[selected]
    sizes = np.exp(np.clip(rows[selected, 2:4], -20.0, 20.0)) * _EXPANDED_STRIDES[selected]
    top_left = centers - sizes / 2.0
    boxes = np.concatenate((top_left, sizes), axis=1) / float(letterbox_ratio)
    candidate_scores = scores[selected]
    nms_boxes = boxes.tolist()
    keep = cv2.dnn.NMSBoxes(
        nms_boxes,
        candidate_scores.tolist(),
        float(score_floor),
        float(nms_threshold),
    )
    if len(keep) == 0:
        return (), maximum_raw_score, raw_count
    indexes = np.asarray(keep).reshape(-1)
    indexes = indexes[np.argsort(candidate_scores[indexes])[::-1]][:maximum_detections]
    detections: list[BallDetection] = []
    for index in indexes:
        x, y, width, height = (float(item) for item in boxes[int(index)])
        x0 = min(max(x, 0.0), float(image_width))
        y0 = min(max(y, 0.0), float(image_height))
        x1 = min(max(x + width, 0.0), float(image_width))
        y1 = min(max(y + height, 0.0), float(image_height))
        if x1 <= x0 or y1 <= y0:
            continue
        detections.append(
            BallDetection(
                x=x0,
                y=y0,
                width=x1 - x0,
                height=y1 - y0,
                score=float(candidate_scores[int(index)]),
            )
        )
    return tuple(detections), maximum_raw_score, raw_count


class YoloXBallDetector:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        score_floor: float = 0.01,
        nms_threshold: float = 0.5,
        maximum_detections: int = 20,
        opencv_threads: int = 6,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.metadata = load_model_metadata(self.model_dir)
        if not 0 < score_floor <= 1:
            raise ValueError("score_floor must be in (0, 1]")
        if not 0 <= nms_threshold <= 1:
            raise ValueError("nms_threshold must be in [0, 1]")
        if maximum_detections < 1 or opencv_threads < 1:
            raise ValueError("maximum detections and OpenCV threads must be positive")
        self.score_floor = float(score_floor)
        self.nms_threshold = float(nms_threshold)
        self.maximum_detections = int(maximum_detections)
        cv2.setNumThreads(int(opencv_threads))
        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.model_dir / MODEL_FILENAME))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except cv2.error as error:
            raise BallDetectorError(f"cannot load YOLOX ONNX artifact: {error}") from error

    def detect(self, frame_bgr: np.ndarray) -> FrameDetections:
        blob, ratio = letterbox_rgb(frame_bgr)
        self.net.setInput(blob)
        start = time.perf_counter()
        output = self.net.forward()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        detections, maximum_raw_score, raw_count = decode_sports_ball_output(
            output,
            image_width=frame_bgr.shape[1],
            image_height=frame_bgr.shape[0],
            letterbox_ratio=ratio,
            score_floor=self.score_floor,
            nms_threshold=self.nms_threshold,
            maximum_detections=self.maximum_detections,
        )
        return FrameDetections(
            detections=detections,
            maximum_raw_score=maximum_raw_score,
            raw_candidates_above_floor=raw_count,
            inference_milliseconds=elapsed_ms,
        )


def _read_task(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        task = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise BallDetectorError(f"cannot load ball annotation task {path}: {error}") from error
    try:
        validated = validate_ball_annotation_task(
            task,
            task_path=path,
            verify_images=True,
        )
    except ValueError as error:
        raise BallDetectorError(f"invalid ball annotation task {path}: {error}") from error
    immutable = validated["immutable"]
    rows = immutable["frames"]
    if not rows:
        raise BallDetectorError("ball annotation task contains no frames")
    return task, rows


def infer_annotation_task(
    task_path: str | Path,
    model_dir: str | Path,
    output_path: str | Path,
    *,
    score_floor: float = 0.01,
    nms_threshold: float = 0.5,
    maximum_detections: int = 20,
    opencv_threads: int = 6,
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    source = Path(task_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite detector suggestions: {destination}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when supplied")
    task, rows = _read_task(source)
    if limit is not None:
        rows = rows[:limit]
    for frame in task["immutable"]["frames"]:
        image = frame.get("image")
        image_name = image.get("path") if isinstance(image, dict) else None
        if not isinstance(image_name, str):
            raise BallDetectorError("task frame image.path must be a string")
        source_image = (source.parent / image_name).resolve()
        destination_image = (destination.parent / image_name).resolve()
        if source_image != destination_image:
            raise BallDetectorError(
                "suggestion output must preserve immutable relative image paths; "
                "write it in a sibling directory with the same depth as the task directory"
            )
    detector = YoloXBallDetector(
        model_dir,
        score_floor=score_floor,
        nms_threshold=nms_threshold,
        maximum_detections=maximum_detections,
        opencv_threads=opencv_threads,
    )
    root = source.parent
    suggestion_frames: dict[str, dict[str, Any]] = {}
    times: list[float] = []
    detections_total = 0
    frames_with_detection = 0
    for index, frame in enumerate(rows):
        frame_id = frame.get("id")
        image = frame.get("image")
        if not isinstance(frame_id, str) or not isinstance(image, dict):
            raise BallDetectorError("task frame is missing id or image metadata")
        image_name = image.get("path")
        if not isinstance(image_name, str):
            raise BallDetectorError("task frame image.path must be a string")
        image_path = Path(image_name)
        if not image_path.is_absolute():
            image_path = (root / image_path).resolve()
        if not image_path.is_file():
            raise BallDetectorError(f"task image is missing: {image_path}")
        expected_image_sha = image.get("sha256")
        if not isinstance(expected_image_sha, str) or sha256_file(image_path) != expected_image_sha:
            raise BallDetectorError(f"task image SHA-256 does not match: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise BallDetectorError(f"cannot decode task image: {image_path}")
        result = detector.detect(image)
        height, width = image.shape[:2]
        suggestions = [
            {
                "confidence": item.score,
                "bbox": {
                    "x": item.x / width,
                    "y": item.y / height,
                    "width": item.width / width,
                    "height": item.height / height,
                },
            }
            for item in result.detections
        ]
        frames_with_detection += bool(suggestions)
        detections_total += len(suggestions)
        times.append(result.inference_milliseconds)
        suggestion_frames[frame_id] = {
            "ballPresenceProbability": max(
                (item["confidence"] for item in suggestions), default=0.0
            ),
            "detections": suggestions,
            "diagnostics": {
                "maximumRawSportsBallScore": result.maximum_raw_score,
                "rawCandidatesAboveFloor": result.raw_candidates_above_floor,
                "inferenceMilliseconds": result.inference_milliseconds,
            },
        }
        if progress is not None and ((index + 1) % 50 == 0 or index + 1 == len(rows)):
            progress(f"Detected {index + 1}/{len(rows)} task frames")
    timing = np.asarray(times, dtype=np.float64)
    model_path = detector.model_dir / MODEL_FILENAME
    enriched = copy.deepcopy(task)
    enriched["suggestions"] = {
        "status": "partial" if len(rows) < len(task["immutable"]["frames"]) else "complete",
        "model": {
            "modelId": MODEL_ID,
            "modelPath": str(model_path),
            "modelSha256": MODEL_SHA256,
            "modelMetadataSha256": sha256_file(detector.model_dir / "model.json"),
            "opencvVersion": cv2.__version__,
            "backend": "OpenCV DNN CPU",
            "sourceTask": {
                "pathHint": str(source),
                "sha256": sha256_file(source),
            },
            "settings": {
                "scoreFloor": score_floor,
                "nmsThreshold": nms_threshold,
                "maximumDetections": maximum_detections,
                "sportsBallClassIndex": SPORTS_BALL_CLASS_INDEX,
                "opencvThreads": opencv_threads,
            },
            "warning": "unreviewed proposals are not annotation truth",
        },
        "summary": {
            "taskFrames": len(task["immutable"]["frames"]),
            "processedFrames": len(rows),
            "limitedRun": limit is not None,
            "framesWithDetection": frames_with_detection,
            "detections": detections_total,
            "medianInferenceMilliseconds": float(np.median(timing)),
            "p90InferenceMilliseconds": float(np.percentile(timing, 90)),
            "meanInferenceMilliseconds": float(np.mean(timing)),
        },
        "frames": suggestion_frames,
    }
    validate_ball_annotation_task(enriched, task_path=source, verify_images=True)
    written = atomic_write_text(
        destination,
        json.dumps(enriched, indent=2, allow_nan=False) + "\n",
    )
    validate_ball_annotation_task(written, verify_images=True)
    return written


def build_suggestion_index(
    pilot_index_path: str | Path,
    suggestions_directory: str | Path,
    output_path: str | Path,
) -> Path:
    """Bind every complete proposal task to its immutable Stage-A task and detector."""

    pilot_path = Path(pilot_index_path).expanduser().resolve()
    suggestions_dir = Path(suggestions_directory).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite suggestion index: {destination}")
    try:
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise BallDetectorError(f"cannot load pilot index {pilot_path}: {error}") from error
    if (
        not isinstance(pilot, dict)
        or pilot.get("artifactType") != "volleycut-ball-presence-pilot-index"
        or pilot.get("developmentOnly") is not True
        or not isinstance(pilot.get("tasks"), list)
    ):
        raise BallDetectorError("pilot index is not a development-only ball-presence index")
    rows: list[dict[str, Any]] = []
    detector_signature: str | None = None
    detector_payload: dict[str, Any] | None = None
    total_frames = total_detected = total_detections = 0
    for indexed in pilot["tasks"]:
        if not isinstance(indexed, dict) or not isinstance(indexed.get("task"), str):
            raise BallDetectorError("pilot index contains an invalid task row")
        suggestion_path = suggestions_dir / Path(indexed["task"]).name
        try:
            task = validate_ball_annotation_task(
                suggestion_path,
                verify_images=True,
            )
        except ValueError as error:
            raise BallDetectorError(
                f"invalid detector proposal task {suggestion_path}: {error}"
            ) from error
        if task["immutable"]["taskId"] != indexed.get("taskId"):
            raise BallDetectorError(
                f"proposal taskId does not match pilot index: {suggestion_path}"
            )
        suggestions = task["suggestions"]
        if suggestions["status"] != "complete":
            raise BallDetectorError(f"proposal task is not complete: {suggestion_path}")
        model = suggestions["model"]
        source_task = model.get("sourceTask") if isinstance(model, dict) else None
        if (
            not isinstance(source_task, dict)
            or source_task.get("sha256") != indexed.get("initialTaskSha256")
        ):
            raise BallDetectorError(
                f"proposal source-task SHA does not match pilot index: {suggestion_path}"
            )
        stable_model = copy.deepcopy(model)
        stable_model.pop("modelPath", None)
        stable_model.pop("sourceTask", None)
        signature = hashlib.sha256(
            json.dumps(
                stable_model,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if detector_signature is None:
            detector_signature = signature
            detector_payload = stable_model
        elif signature != detector_signature:
            raise BallDetectorError("proposal tasks use inconsistent detector configurations")
        summary = suggestions.get("summary")
        if not isinstance(summary, dict):
            raise BallDetectorError(f"proposal task has no summary: {suggestion_path}")
        frames = int(summary.get("processedFrames", -1))
        detected = int(summary.get("framesWithDetection", -1))
        detections = int(summary.get("detections", -1))
        if frames != indexed.get("frameCount") or min(detected, detections) < 0:
            raise BallDetectorError(f"proposal summary is inconsistent: {suggestion_path}")
        total_frames += frames
        total_detected += detected
        total_detections += detections
        rows.append(
            {
                "recordingId": indexed.get("recordingId"),
                "taskId": indexed.get("taskId"),
                "sourceTaskSha256": indexed.get("initialTaskSha256"),
                "proposal": str(suggestion_path),
                "proposalSha256": sha256_file(suggestion_path),
                "frameCount": frames,
                "framesWithDetection": detected,
                "detections": detections,
                "medianInferenceMilliseconds": summary.get(
                    "medianInferenceMilliseconds"
                ),
                "p90InferenceMilliseconds": summary.get("p90InferenceMilliseconds"),
            }
        )
    if detector_signature is None or detector_payload is None:
        raise BallDetectorError("pilot index contains no proposal tasks")
    report = {
        "schemaVersion": 1,
        "kind": "volleycut-ball-detector-suggestion-index",
        "status": "unreviewed-proposals-not-ground-truth",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "pilotIndex": {
            "path": str(pilot_path),
            "sha256": sha256_file(pilot_path),
            "manifestSha256": pilot.get("manifest", {}).get("sha256"),
            "round": pilot.get("round"),
        },
        "detector": detector_payload,
        "detectorConfigurationSha256": detector_signature,
        "summary": {
            "recordings": len(rows),
            "frames": total_frames,
            "framesWithDetection": total_detected,
            "framesWithDetectionFraction": total_detected / max(1, total_frames),
            "detections": total_detections,
            "timingWarning": (
                "per-task inference timing may reflect unrelated host contention and is not "
                "an acceptance benchmark"
            ),
        },
        "tasks": rows,
    }
    return atomic_write_text(
        destination,
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )


def _visual_observability(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(np.var(cv2.Laplacian(gray, cv2.CV_32F)))
    focus = laplacian_variance / (laplacian_variance + 100.0)
    exposure = max(
        0.0,
        1.0 - float(np.mean(gray <= 12)) - float(np.mean(gray >= 243)),
    )
    contrast = min(1.0, float(np.std(gray)) / 30.0)
    return float(np.sqrt(max(0.0, focus * exposure * contrast)))


def _atomic_savez(destination: Path, **arrays: np.ndarray) -> Path:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite detector sidecar: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=".npz", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite detector sidecar: {destination}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def infer_video_sidecar(
    video_path: str | Path,
    model_dir: str | Path,
    output_path: str | Path,
    *,
    recording_id: str,
    source_group: str,
    split: str,
    expected_video_sha256: str | None = None,
    detector_fps: float = 15.0,
    score_floor: float = 0.01,
    nms_threshold: float = 0.5,
    maximum_detections: int = 20,
    opencv_threads: int = 6,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Create an immutable high-rate score sidecar; no detector threshold is selected here."""

    source = Path(video_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise BallDetectorError(f"video is missing: {source}")
    if not recording_id or not source_group or split not in {"train", "validation", "test", "challenge"}:
        raise ValueError("recording_id/source_group/split provenance is invalid")
    metadata = probe_video(source)
    if (
        not math.isfinite(detector_fps)
        or detector_fps < 8.0
        or detector_fps > metadata.fps + 1e-6
    ):
        raise ValueError(
            "detector_fps must be at least 8 (2x the 4 fps model) and no greater than source FPS"
        )
    source_digest = sha256_file(source)
    if expected_video_sha256 is not None and source_digest != expected_video_sha256:
        raise BallDetectorError(
            f"video SHA-256 {source_digest} does not match manifest {expected_video_sha256}"
        )
    detector = YoloXBallDetector(
        model_dir,
        score_floor=score_floor,
        nms_threshold=nms_threshold,
        maximum_detections=maximum_detections,
        opencv_threads=opencv_threads,
    )
    sample_count = max(1, int(math.ceil(metadata.duration * detector_fps - 1e-9)))
    requested_times = np.arange(sample_count, dtype=np.float64) / detector_fps
    target_frames = np.rint(requested_times * metadata.fps).astype(np.int64)
    target_frames = np.clip(target_frames, 0, max(0, metadata.frame_count - 1))
    if np.any(target_frames[1:] == target_frames[:-1]):
        raise BallDetectorError(
            "detector sampling maps multiple rows to one frame; lower detector_fps"
        )
    times = requested_times
    if len(target_frames) == 0:
        raise BallDetectorError("requested sidecar range contains no source frames")

    frame_available = np.zeros(len(target_frames), dtype=np.bool_)
    observability = np.zeros(len(target_frames), dtype=np.float32)
    best_score = np.zeros(len(target_frames), dtype=np.float32)
    candidate_count = np.zeros(len(target_frames), dtype=np.int16)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise BallDetectorError(f"cannot open video: {source}")
    current_frame = int(target_frames[0])
    capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    try:
        for index, target_frame in enumerate(target_frames):
            while current_frame < int(target_frame):
                if not capture.grab():
                    break
                current_frame += 1
            ok, frame = capture.read()
            current_frame += 1
            if not ok or frame is None:
                raise BallDetectorError(
                    f"cannot decode source frame {int(target_frame)} for {recording_id}"
                )
            try:
                result = detector.detect(frame)
            except cv2.error as error:
                raise BallDetectorError(
                    f"detector failed on source frame {int(target_frame)}: {error}"
                ) from error
            frame_available[index] = True
            observability[index] = _visual_observability(frame)
            candidate_count[index] = len(result.detections)
            if result.detections:
                best_score[index] = max(item.score for item in result.detections)
            if progress is not None and (
                (index + 1) % 250 == 0 or index + 1 == len(target_frames)
            ):
                progress(f"Detected {index + 1}/{len(target_frames)} video frames")
    finally:
        capture.release()

    sidecar_metadata = {
        "schemaVersion": BALL_DETECTOR_SCHEMA_VERSION,
        "kind": "volleycut-ball-presence-sidecar",
        "status": "available",
        "unavailableReason": None,
        "recordingId": recording_id,
        "sourceGroup": source_group,
        "split": split,
        "sourceVideo": {
            "contentSha256": source_digest,
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "frameCount": metadata.frame_count,
            "durationSeconds": metadata.duration,
            "timeBase": "video-start-seconds",
        },
        "detector": {
            "id": MODEL_ID,
            "artifactSha256": MODEL_SHA256,
            "sampleFps": detector_fps,
            "scoreFloor": score_floor,
            "nmsThreshold": nms_threshold,
            "maximumDetections": maximum_detections,
            "opencvVersion": cv2.__version__,
            "implementationSha256": sha256_file(Path(__file__).resolve()),
            "sampleFrameRule": "round(sampleIndex * sourceFps / sampleFps)",
            "roi": None,
        },
    }
    metadata_json = json.dumps(sidecar_metadata, sort_keys=True, allow_nan=False)
    return _atomic_savez(
        destination,
        metadata_json=np.asarray(metadata_json),
        times=times,
        frame_available=frame_available,
        observability=observability,
        best_score=best_score,
        candidate_count=candidate_count,
    )
