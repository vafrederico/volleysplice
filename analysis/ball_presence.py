from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


BALL_PRESENCE_SCHEMA_VERSION = 1
BALL_PRESENCE_KIND = "volleycut-ball-presence-sidecar"
BALL_PRESENCE_FEATURE_NAMES = (
    "ball_detector_available",
    "ball_observation_fraction",
    "ball_observability_quality",
    "ball_presence_probability",
    "ball_detected_fraction",
    "ball_candidate_count_normalized",
    "ball_seconds_since_detection",
    "ball_quality_gated_presence",
)

_SIDECAR_KEYS = {
    "metadata_json",
    "times",
    "frame_available",
    "observability",
    "best_score",
    "candidate_count",
}
_ROOT_METADATA_KEYS = {
    "schemaVersion",
    "kind",
    "status",
    "unavailableReason",
    "recordingId",
    "sourceGroup",
    "split",
    "sourceVideo",
    "detector",
}
_SOURCE_METADATA_KEYS = {
    "contentSha256",
    "durationSeconds",
    "width",
    "height",
    "fps",
    "frameCount",
    "timeBase",
}
_DETECTOR_METADATA_KEYS = {
    "id",
    "artifactSha256",
    "sampleFps",
    "scoreFloor",
    "nmsThreshold",
    "maximumDetections",
    "opencvVersion",
    "implementationSha256",
    "sampleFrameRule",
    "roi",
}
_ROI_KEYS = {"x", "y", "width", "height"}
_STATUSES = {"available", "unavailable"}


class BallPresenceError(ValueError):
    """Raised when a ball-presence sidecar is unsafe to use."""


@dataclass(frozen=True)
class BallPresenceSidecar:
    path: Path
    artifact_sha256: str
    status: str
    unavailable_reason: str | None
    recording_id: str
    source_group: str
    split: str
    source_video_sha256: str
    source_duration_seconds: float
    source_width: int
    source_height: int
    detector_id: str
    detector_artifact_sha256: str | None
    sample_fps: float
    score_floor: float
    nms_threshold: float
    maximum_detections: int
    opencv_version: str
    implementation_sha256: str
    roi: tuple[float, float, float, float] | None
    times: np.ndarray
    frame_available: np.ndarray
    observability: np.ndarray
    best_score: np.ndarray
    candidate_count: np.ndarray


@dataclass(frozen=True)
class BallPresenceFeatures:
    times: np.ndarray
    values: np.ndarray
    names: tuple[str, ...]
    analysis_fps: float
    sidecar_sha256: str
    sidecar_status: str
    detector_id: str
    detector_artifact_sha256: str | None
    detection_threshold: float


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_exact_keys(value: dict[str, Any], expected: set[str], where: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        raise BallPresenceError(
            f"{where} keys do not match schema; missing={missing}, unknown={unknown}"
        )


def _sha256(value: Any, where: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        suffix = " or null" if nullable else ""
        raise BallPresenceError(f"{where} must be a hexadecimal SHA-256{suffix}")
    return value.lower()


def _positive_float(value: Any, where: str) -> float:
    if not _is_number(value) or float(value) <= 0:
        raise BallPresenceError(f"{where} must be a positive finite number")
    return float(value)


def _positive_int(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BallPresenceError(f"{where} must be a positive integer")
    return value


def _normalized_roi(
    value: Any, where: str
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise BallPresenceError(f"{where} must be a normalized rectangle or null")
    _require_exact_keys(value, _ROI_KEYS, where)
    if any(not _is_number(value[key]) for key in ("x", "y", "width", "height")):
        raise BallPresenceError(f"{where} coordinates must be finite numbers")
    x, y, width, height = (
        float(value[key]) for key in ("x", "y", "width", "height")
    )
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > 1
        or y + height > 1
    ):
        raise BallPresenceError(f"{where} must lie within the normalized video frame")
    return x, y, width, height


def _json_object(value: np.ndarray) -> dict[str, Any]:
    if value.shape != ():
        raise BallPresenceError("metadata_json must be a scalar JSON string")
    raw = value.item()
    if not isinstance(raw, str):
        raise BallPresenceError("metadata_json must be a scalar JSON string")

    def reject_constant(token: str) -> None:
        raise BallPresenceError(f"metadata_json contains non-finite constant {token}")

    try:
        parsed = json.loads(raw, parse_constant=reject_constant)
    except (json.JSONDecodeError, TypeError) as error:
        raise BallPresenceError(f"metadata_json is invalid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise BallPresenceError("metadata_json root must be an object")
    return parsed


def _one_dimensional(value: np.ndarray, name: str) -> np.ndarray:
    if value.ndim != 1:
        raise BallPresenceError(f"{name} must be a one-dimensional array")
    return value


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _validate_metadata(
    metadata: dict[str, Any],
) -> tuple[
    str,
    str | None,
    str,
    str,
    str,
    str,
    float,
    int,
    int,
    str,
    str | None,
    float,
    float,
    float,
    int,
    str,
    str,
    tuple[float, float, float, float] | None,
]:
    _require_exact_keys(metadata, _ROOT_METADATA_KEYS, "metadata")
    if metadata["schemaVersion"] != BALL_PRESENCE_SCHEMA_VERSION:
        raise BallPresenceError(
            "metadata.schemaVersion must be "
            f"{BALL_PRESENCE_SCHEMA_VERSION}"
        )
    if metadata["kind"] != BALL_PRESENCE_KIND:
        raise BallPresenceError(f"metadata.kind must be {BALL_PRESENCE_KIND!r}")
    status = metadata["status"]
    if status not in _STATUSES:
        raise BallPresenceError(f"metadata.status must be one of {sorted(_STATUSES)}")
    unavailable_reason = metadata["unavailableReason"]
    if status == "available":
        if unavailable_reason is not None:
            raise BallPresenceError(
                "metadata.unavailableReason must be null when status is available"
            )
    elif not isinstance(unavailable_reason, str) or not unavailable_reason.strip():
        raise BallPresenceError(
            "metadata.unavailableReason must be a non-empty string when unavailable"
        )

    recording_id = metadata["recordingId"]
    if not isinstance(recording_id, str) or not recording_id.strip():
        raise BallPresenceError("metadata.recordingId must be a non-empty string")
    if recording_id != recording_id.strip():
        raise BallPresenceError("metadata.recordingId must not have surrounding whitespace")
    source_group = metadata["sourceGroup"]
    if not isinstance(source_group, str) or not source_group.strip():
        raise BallPresenceError("metadata.sourceGroup must be a non-empty string")
    if source_group != source_group.strip():
        raise BallPresenceError("metadata.sourceGroup must not have surrounding whitespace")
    split = metadata["split"]
    if split not in {"train", "validation", "test", "challenge"}:
        raise BallPresenceError("metadata.split is invalid")

    source = metadata["sourceVideo"]
    if not isinstance(source, dict):
        raise BallPresenceError("metadata.sourceVideo must be an object")
    _require_exact_keys(source, _SOURCE_METADATA_KEYS, "metadata.sourceVideo")
    source_sha256 = _sha256(
        source["contentSha256"], "metadata.sourceVideo.contentSha256"
    )
    assert source_sha256 is not None
    duration = _positive_float(
        source["durationSeconds"], "metadata.sourceVideo.durationSeconds"
    )
    width = _positive_int(source["width"], "metadata.sourceVideo.width")
    height = _positive_int(source["height"], "metadata.sourceVideo.height")
    source_fps = _positive_float(source["fps"], "metadata.sourceVideo.fps")
    source_frame_count = _positive_int(
        source["frameCount"], "metadata.sourceVideo.frameCount"
    )
    derived_duration = source_frame_count / source_fps
    if not math.isclose(
        duration,
        derived_duration,
        rel_tol=0.0,
        abs_tol=max(1e-6, 1e-6 * duration),
    ):
        raise BallPresenceError(
            "metadata.sourceVideo.durationSeconds must equal frameCount / fps"
        )
    if source["timeBase"] != "video-start-seconds":
        raise BallPresenceError(
            "metadata.sourceVideo.timeBase must be 'video-start-seconds'"
        )

    detector = metadata["detector"]
    if not isinstance(detector, dict):
        raise BallPresenceError("metadata.detector must be an object")
    _require_exact_keys(detector, _DETECTOR_METADATA_KEYS, "metadata.detector")
    detector_id = detector["id"]
    if not isinstance(detector_id, str) or not detector_id.strip():
        raise BallPresenceError("metadata.detector.id must be a non-empty string")
    if detector_id != detector_id.strip():
        raise BallPresenceError("metadata.detector.id must not have surrounding whitespace")
    detector_sha256 = _sha256(
        detector["artifactSha256"],
        "metadata.detector.artifactSha256",
        nullable=True,
    )
    if status == "available" and detector_sha256 is None:
        raise BallPresenceError(
            "metadata.detector.artifactSha256 is required when status is available"
        )
    if status == "unavailable" and detector_sha256 is not None:
        raise BallPresenceError(
            "metadata.detector.artifactSha256 must be null when status is unavailable"
        )
    sample_fps = _positive_float(
        detector["sampleFps"], "metadata.detector.sampleFps"
    )
    if sample_fps > 120:
        raise BallPresenceError("metadata.detector.sampleFps must not exceed 120")
    if sample_fps > source_fps + 1e-9:
        raise BallPresenceError(
            "metadata.detector.sampleFps must not exceed source video fps"
        )
    score_floor = detector["scoreFloor"]
    if not _is_number(score_floor) or not 0 < float(score_floor) <= 1:
        raise BallPresenceError(
            "metadata.detector.scoreFloor must satisfy 0 < scoreFloor <= 1"
        )
    nms_threshold = detector["nmsThreshold"]
    if not _is_number(nms_threshold) or not 0 <= float(nms_threshold) <= 1:
        raise BallPresenceError("metadata.detector.nmsThreshold must be in [0, 1]")
    maximum_detections = _positive_int(
        detector["maximumDetections"], "metadata.detector.maximumDetections"
    )
    opencv_version = detector["opencvVersion"]
    if not isinstance(opencv_version, str) or not opencv_version.strip():
        raise BallPresenceError("metadata.detector.opencvVersion must be a non-empty string")
    implementation_sha256 = _sha256(
        detector["implementationSha256"], "metadata.detector.implementationSha256"
    )
    assert implementation_sha256 is not None
    if detector["sampleFrameRule"] != "round(sampleIndex * sourceFps / sampleFps)":
        raise BallPresenceError("metadata.detector.sampleFrameRule is unsupported")
    roi = _normalized_roi(detector["roi"], "metadata.detector.roi")
    return (
        status,
        unavailable_reason,
        recording_id,
        source_group,
        split,
        source_sha256,
        duration,
        width,
        height,
        detector_id,
        detector_sha256,
        sample_fps,
        float(score_floor),
        float(nms_threshold),
        maximum_detections,
        opencv_version,
        implementation_sha256,
        roi,
    )


def _validate_arrays(
    *,
    status: str,
    duration: float,
    sample_fps: float,
    score_floor: float,
    maximum_detections: int,
    times: np.ndarray,
    frame_available: np.ndarray,
    observability: np.ndarray,
    best_score: np.ndarray,
    candidate_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = {
        "times": _one_dimensional(times, "times"),
        "frame_available": _one_dimensional(frame_available, "frame_available"),
        "observability": _one_dimensional(observability, "observability"),
        "best_score": _one_dimensional(best_score, "best_score"),
        "candidate_count": _one_dimensional(candidate_count, "candidate_count"),
    }
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise BallPresenceError("sidecar arrays must have identical lengths")
    if frame_available.dtype != np.bool_:
        raise BallPresenceError("frame_available must have boolean dtype")
    if not np.issubdtype(candidate_count.dtype, np.integer):
        raise BallPresenceError("candidate_count must have an integer dtype")
    for name in ("times", "observability", "best_score"):
        if not np.issubdtype(arrays[name].dtype, np.number):
            raise BallPresenceError(f"{name} must have a numeric dtype")

    times_value = times.astype(np.float64, copy=True)
    available_value = frame_available.astype(np.bool_, copy=True)
    observability_value = observability.astype(np.float32, copy=True)
    score_value = best_score.astype(np.float32, copy=True)
    count_value = candidate_count.astype(np.int32, copy=True)

    if status == "unavailable":
        if len(times_value) != 0:
            raise BallPresenceError("an unavailable sidecar must contain empty arrays")
        return tuple(
            _readonly(value)
            for value in (
                times_value,
                available_value,
                observability_value,
                score_value,
                count_value,
            )
        )  # type: ignore[return-value]

    expected_count = max(1, int(math.ceil(duration * sample_fps - 1e-9)))
    if len(times_value) != expected_count:
        raise BallPresenceError(
            "available sidecar must contain the complete fixed sampling grid: "
            f"expected {expected_count} rows, found {len(times_value)}"
        )
    if not all(
        np.isfinite(value).all()
        for value in (times_value, observability_value, score_value)
    ):
        raise BallPresenceError("sidecar arrays contain non-finite values")
    expected_times = np.arange(expected_count, dtype=np.float64) / sample_fps
    cadence_tolerance = max(1e-6, 0.02 / sample_fps)
    if not np.allclose(
        times_value, expected_times, rtol=0.0, atol=cadence_tolerance
    ):
        raise BallPresenceError(
            "times must be the fixed video-start sampling grid implied by sampleFps"
        )
    if times_value[-1] >= duration + cadence_tolerance:
        raise BallPresenceError("sidecar sampling grid extends past the source duration")
    if np.any((observability_value < 0) | (observability_value > 1)):
        raise BallPresenceError("observability values must lie in [0, 1]")
    if np.any((score_value < 0) | (score_value > 1)):
        raise BallPresenceError("best_score values must lie in [0, 1]")
    if np.any(count_value < 0):
        raise BallPresenceError("candidate_count values must be non-negative")
    if np.any(count_value > maximum_detections):
        raise BallPresenceError(
            "candidate_count exceeds metadata.detector.maximumDetections"
        )
    unavailable = ~available_value
    if (
        np.any(observability_value[unavailable] != 0)
        or np.any(score_value[unavailable] != 0)
        or np.any(count_value[unavailable] != 0)
    ):
        raise BallPresenceError(
            "unavailable frames must have zero observability, score, and candidate count"
        )
    if np.any((count_value == 0) & (score_value != 0)):
        raise BallPresenceError("best_score must be zero when candidate_count is zero")
    if np.any((count_value > 0) & (score_value < score_floor)):
        raise BallPresenceError(
            "best_score must reach scoreFloor when candidate_count is positive"
        )

    return tuple(
        _readonly(value)
        for value in (
            times_value,
            available_value,
            observability_value,
            score_value,
            count_value,
        )
    )  # type: ignore[return-value]


def load_ball_presence_sidecar(
    path: str | Path,
    *,
    expected_recording_id: str,
    expected_source_group: str,
    expected_split: str,
    expected_video_sha256: str,
    expected_duration_seconds: float,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_detector_id: str | None = None,
    expected_detector_sha256: str | None = None,
    expected_detector_sample_fps: float | None = None,
    expected_score_floor: float | None = None,
    expected_nms_threshold: float | None = None,
    expected_maximum_detections: int | None = None,
    expected_implementation_sha256: str | None = None,
    expected_sidecar_sha256: str | None = None,
    expected_roi: tuple[float, float, float, float] | None = None,
) -> BallPresenceSidecar:
    """Load an immutable high-rate detector sidecar after provenance validation.

    A missing path is an error. Detector unavailability must instead be represented by
    a valid sidecar with ``status='unavailable'`` so it cannot be confused with a
    successful detector run that happened to find no ball.
    """

    supplied = Path(path).expanduser().resolve()
    try:
        artifact_sha256 = sha256_file(supplied)
    except OSError as error:
        raise BallPresenceError(f"cannot read ball-presence sidecar {supplied}: {error}") from error
    if expected_sidecar_sha256 is not None:
        expected_artifact = _sha256(
            expected_sidecar_sha256, "expected_sidecar_sha256"
        )
        if artifact_sha256 != expected_artifact:
            raise BallPresenceError("ball-presence sidecar SHA-256 does not match")

    try:
        with np.load(supplied, allow_pickle=False) as artifact:
            if set(artifact.files) != _SIDECAR_KEYS:
                missing = sorted(_SIDECAR_KEYS - set(artifact.files))
                unknown = sorted(set(artifact.files) - _SIDECAR_KEYS)
                raise BallPresenceError(
                    "sidecar keys do not match schema; "
                    f"missing={missing}, unknown={unknown}"
                )
            metadata = _json_object(artifact["metadata_json"])
            raw_arrays = {
                name: artifact[name].copy()
                for name in _SIDECAR_KEYS
                if name != "metadata_json"
            }
    except BallPresenceError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise BallPresenceError(f"cannot decode ball-presence sidecar {supplied}: {error}") from error

    (
        status,
        unavailable_reason,
        recording_id,
        source_group,
        split,
        video_sha256,
        duration,
        width,
        height,
        detector_id,
        detector_sha256,
        sample_fps,
        score_floor,
        nms_threshold,
        maximum_detections,
        opencv_version,
        implementation_sha256,
        roi,
    ) = _validate_metadata(metadata)
    expected_video = _sha256(expected_video_sha256, "expected_video_sha256")
    if recording_id != expected_recording_id:
        raise BallPresenceError(
            f"recording id mismatch: expected {expected_recording_id!r}, found {recording_id!r}"
        )
    if source_group != expected_source_group:
        raise BallPresenceError(
            f"source group mismatch: expected {expected_source_group!r}, found {source_group!r}"
        )
    if split != expected_split:
        raise BallPresenceError(
            f"split mismatch: expected {expected_split!r}, found {split!r}"
        )
    if video_sha256 != expected_video:
        raise BallPresenceError("source video SHA-256 does not match the recording")
    if not _is_number(expected_duration_seconds) or expected_duration_seconds <= 0:
        raise BallPresenceError("expected_duration_seconds must be positive and finite")
    duration_tolerance = max(1e-3, 1e-6 * float(expected_duration_seconds))
    if abs(duration - float(expected_duration_seconds)) > duration_tolerance:
        raise BallPresenceError(
            "source duration does not match the recording: "
            f"expected {expected_duration_seconds:g}, found {duration:g}"
        )
    if expected_width is not None and width != expected_width:
        raise BallPresenceError(
            f"source width mismatch: expected {expected_width}, found {width}"
        )
    if expected_height is not None and height != expected_height:
        raise BallPresenceError(
            f"source height mismatch: expected {expected_height}, found {height}"
        )
    if expected_detector_id is not None and detector_id != expected_detector_id:
        raise BallPresenceError(
            f"detector id mismatch: expected {expected_detector_id!r}, found {detector_id!r}"
        )
    if expected_detector_sha256 is not None:
        expected_detector = _sha256(
            expected_detector_sha256, "expected_detector_sha256"
        )
        if detector_sha256 != expected_detector:
            raise BallPresenceError("detector artifact SHA-256 does not match")
    expected_numbers = (
        (expected_detector_sample_fps, sample_fps, "detector sample FPS"),
        (expected_score_floor, score_floor, "detector score floor"),
        (expected_nms_threshold, nms_threshold, "detector NMS threshold"),
    )
    for expected_value, actual_value, label in expected_numbers:
        if expected_value is not None and (
            not _is_number(expected_value)
            or not math.isclose(
                float(expected_value), actual_value, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise BallPresenceError(f"{label} does not match")
    if (
        expected_maximum_detections is not None
        and expected_maximum_detections != maximum_detections
    ):
        raise BallPresenceError("detector maximum detections does not match")
    if expected_implementation_sha256 is not None:
        expected_implementation = _sha256(
            expected_implementation_sha256, "expected_implementation_sha256"
        )
        if implementation_sha256 != expected_implementation:
            raise BallPresenceError("detector implementation SHA-256 does not match")
    if expected_roi is not None:
        normalized_expected_roi = _normalized_roi(
            dict(zip(("x", "y", "width", "height"), expected_roi, strict=True)),
            "expected_roi",
        )
        if roi is None or not np.allclose(
            np.asarray(roi), np.asarray(normalized_expected_roi), rtol=0.0, atol=1e-9
        ):
            raise BallPresenceError("detector ROI does not match the expected ROI")

    times, frame_available, observability, best_score, candidate_count = (
        _validate_arrays(
            status=status,
            duration=duration,
            sample_fps=sample_fps,
            score_floor=score_floor,
            maximum_detections=maximum_detections,
            times=raw_arrays["times"],
            frame_available=raw_arrays["frame_available"],
            observability=raw_arrays["observability"],
            best_score=raw_arrays["best_score"],
            candidate_count=raw_arrays["candidate_count"],
        )
    )
    return BallPresenceSidecar(
        path=supplied,
        artifact_sha256=artifact_sha256,
        status=status,
        unavailable_reason=unavailable_reason,
        recording_id=recording_id,
        source_group=source_group,
        split=split,
        source_video_sha256=video_sha256,
        source_duration_seconds=duration,
        source_width=width,
        source_height=height,
        detector_id=detector_id,
        detector_artifact_sha256=detector_sha256,
        sample_fps=sample_fps,
        score_floor=score_floor,
        nms_threshold=nms_threshold,
        maximum_detections=maximum_detections,
        opencv_version=opencv_version,
        implementation_sha256=implementation_sha256,
        roi=roi,
        times=times,
        frame_available=frame_available,
        observability=observability,
        best_score=best_score,
        candidate_count=candidate_count,
    )


def _validate_target_times(
    times: Sequence[float] | np.ndarray,
    *,
    duration: float,
    analysis_fps: float,
    detector_fps: float,
) -> np.ndarray:
    if not _is_number(analysis_fps) or analysis_fps <= 0:
        raise BallPresenceError("analysis_fps must be positive and finite")
    if detector_fps + 1e-9 < 2.0 * float(analysis_fps):
        raise BallPresenceError(
            "ball-presence sampling must be at least twice the model analysis FPS"
        )
    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise BallPresenceError("target_times must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise BallPresenceError("target_times must be finite and strictly increasing")
    expected_count = max(1, int(math.ceil(duration * float(analysis_fps) - 1e-9)))
    if len(values) != expected_count:
        raise BallPresenceError(
            "target_times do not cover the full source duration at analysis_fps: "
            f"expected {expected_count} rows, found {len(values)}"
        )
    expected = np.arange(expected_count, dtype=np.float64) / float(analysis_fps)
    alignment_tolerance = max(1e-6, 0.5 / detector_fps + 1e-6)
    if not np.allclose(values, expected, rtol=0.0, atol=alignment_tolerance):
        raise BallPresenceError(
            "target_times are not aligned to the video-start analysis grid"
        )
    return values


def aggregate_ball_presence(
    sidecar: BallPresenceSidecar,
    target_times: Sequence[float] | np.ndarray,
    *,
    analysis_fps: float = 4.0,
    detection_threshold: float,
    seconds_since_cap: float = 10.0,
    candidate_count_cap: float = 3.0,
) -> BallPresenceFeatures:
    """Aggregate high-rate detections into centered model-time features.

    Window statistics use the centered 4-fps window. ``seconds_since_detection`` is
    deliberately causal: detections later than the target timestamp never reduce it.
    All channels are zero for an explicitly unavailable detector. When the detector
    ran but found no ball, availability remains one and seconds-since reaches its cap.
    """

    if not _is_number(seconds_since_cap) or seconds_since_cap <= 0:
        raise BallPresenceError("seconds_since_cap must be positive and finite")
    if not _is_number(candidate_count_cap) or candidate_count_cap <= 0:
        raise BallPresenceError("candidate_count_cap must be positive and finite")
    if (
        not _is_number(detection_threshold)
        or not sidecar.score_floor <= float(detection_threshold) <= 1
    ):
        raise BallPresenceError(
            "detection_threshold must be in [sidecar score floor, 1]"
        )
    times = _validate_target_times(
        target_times,
        duration=sidecar.source_duration_seconds,
        analysis_fps=float(analysis_fps),
        detector_fps=sidecar.sample_fps,
    )
    output = np.zeros(
        (len(times), len(BALL_PRESENCE_FEATURE_NAMES)), dtype=np.float32
    )
    if sidecar.status == "available":
        output[:, 0] = 1.0
        detected_mask = (
            sidecar.frame_available
            & (sidecar.best_score >= float(detection_threshold))
        )
        detected_times = sidecar.times[detected_mask]
        half_width = 0.5 / float(analysis_fps)
        for row, timestamp in enumerate(times):
            left = int(
                np.searchsorted(sidecar.times, timestamp - half_width, side="left")
            )
            right = int(
                np.searchsorted(sidecar.times, timestamp + half_width, side="left")
            )
            if right > left:
                frame_available = sidecar.frame_available[left:right]
                output[row, 1] = float(np.mean(frame_available))
                if np.any(frame_available):
                    observability = sidecar.observability[left:right][frame_available]
                    scores = sidecar.best_score[left:right][frame_available]
                    counts = sidecar.candidate_count[left:right][frame_available]
                    output[row, 2] = float(np.mean(observability))
                    output[row, 3] = float(np.max(scores))
                    output[row, 4] = float(
                        np.mean(scores >= float(detection_threshold))
                    )
                    output[row, 5] = float(
                        np.mean(
                            np.clip(
                                counts.astype(np.float32) / candidate_count_cap,
                                0.0,
                                1.0,
                            )
                        )
                    )
                    output[row, 7] = float(np.max(scores * observability))
            prior = int(np.searchsorted(detected_times, timestamp, side="right")) - 1
            output[row, 6] = (
                min(float(seconds_since_cap), timestamp - float(detected_times[prior]))
                if prior >= 0
                else float(seconds_since_cap)
            )
    if not np.isfinite(output).all():
        raise BallPresenceError("ball-presence aggregation produced non-finite values")
    return BallPresenceFeatures(
        times=_readonly(times.astype(np.float64, copy=True)),
        values=_readonly(output),
        names=BALL_PRESENCE_FEATURE_NAMES,
        analysis_fps=float(analysis_fps),
        sidecar_sha256=sidecar.artifact_sha256,
        sidecar_status=sidecar.status,
        detector_id=sidecar.detector_id,
        detector_artifact_sha256=sidecar.detector_artifact_sha256,
        detection_threshold=float(detection_threshold),
    )


def contextualize_ball_presence(
    features: BallPresenceFeatures,
    context_offsets_seconds: Sequence[float],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Add temporal offsets without percentile-ranking absolute detector states."""

    offsets = tuple(float(item) for item in context_offsets_seconds)
    if (
        not offsets
        or 0.0 not in offsets
        or any(not math.isfinite(item) for item in offsets)
    ):
        raise BallPresenceError(
            "context offsets must be finite, non-empty, and include zero"
        )
    if features.values.shape != (
        len(features.times),
        len(BALL_PRESENCE_FEATURE_NAMES),
    ) or features.names != BALL_PRESENCE_FEATURE_NAMES:
        raise BallPresenceError("ball-presence feature signature is invalid")
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for offset in offsets:
        targets = features.times + offset
        right = np.searchsorted(features.times, targets, side="left")
        right = np.clip(right, 0, len(features.times) - 1)
        left = np.clip(right - 1, 0, len(features.times) - 1)
        choose_left = (
            np.abs(features.times[left] - targets)
            <= np.abs(features.times[right] - targets)
        )
        nearest = np.where(choose_left, left, right)
        nearest = np.where(targets <= features.times[0], 0, nearest)
        nearest = np.where(targets >= features.times[-1], len(features.times) - 1, nearest)
        blocks.append(features.values[nearest])
        prefix = f"t{offset:+g}s/"
        names.extend(prefix + name for name in features.names)
    values = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise BallPresenceError("contextual ball-presence features are non-finite")
    return values, tuple(names)
