from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .features import VideoMetadata, probe_video
from .schema import ENVIRONMENTS, DatasetManifest, Interval, Recording, load_manifest


BALL_ANNOTATION_SCHEMA_VERSION = 1
BALL_ANNOTATION_TASK_TYPE = "volleycut-ball-presence-frame-annotation"
BALL_SAMPLING_POLICY_ID = "six-strata-3s-exact-proxy-frames-v1"
SAMPLE_FPS = 15
WINDOW_SECONDS = 3
FRAMES_PER_WINDOW = SAMPLE_FPS * WINDOW_SECONDS
MIN_CENTER_SEPARATION_SECONDS = 5
STRATA = (
    "serve_window",
    "mid_live_a",
    "mid_live_b",
    "end_transition",
    "ordinary_dead",
    "high_motion_dead",
)
ANNOTATION_POLICY_ID = "primary-court-volleyball-state-v1"
PRIMARY_BALL_STATES = {
    "localizable",
    "fully_occluded",
    "out_of_frame",
    "indeterminate",
}
OBJECT_ROLES = {"primary-court", "other-court", "unknown"}
OBJECT_VISIBILITIES = {"clear", "motion-blurred", "partially-occluded"}
SAFE_RECORDING_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
ANNOTATION_POLICY = {
    "id": ANNOTATION_POLICY_ID,
    "target": (
        "Annotate an actual volleyball whenever it is visible and localizable on the "
        "primary court, including while it is held, retrieved, or dead between rallies."
    ),
    "roleRule": (
        "The primary-court role records camera/court association, not an inference that "
        "the ball is live or participating in the labeled rally."
    ),
    "stateRule": (
        "Use localizable only when the primary-court ball can be boxed; otherwise mark it "
        "fully_occluded, out_of_frame, or indeterminate."
    ),
    "negativeRule": (
        "Scoreboards, logos, watermarks, and all other graphical overlays are never "
        "annotation objects."
    ),
    "adjacentCourtRule": (
        "Visible volleyballs from adjacent courts are objects with role other-court, not "
        "primary-court positives."
    ),
    "trackingRule": (
        "Reuse an object id across frames in the same three-second window when identity can "
        "be followed reliably; do not force continuity through ambiguity."
    ),
}


class BallAnnotationError(ValueError):
    """Raised when a ball-presence task cannot be prepared or safely loaded."""


Progress = Callable[[str], None]


@dataclass(frozen=True)
class SamplingWindow:
    requested_stratum: str
    actual_source: str
    start_sample_index: int
    reference: dict[str, Any]
    fallback_reason: str | None = None

    @property
    def end_sample_index(self) -> int:
        return self.start_sample_index + FRAMES_PER_WINDOW

    @property
    def start_seconds(self) -> float:
        return self.start_sample_index / SAMPLE_FPS

    @property
    def end_seconds(self) -> float:
        return self.end_sample_index / SAMPLE_FPS

    @property
    def center_seconds(self) -> float:
        return (self.start_sample_index + FRAMES_PER_WINDOW / 2) / SAMPLE_FPS

    def to_dict(self, order: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": f"{order + 1:02d}-{self.requested_stratum}",
            "requestedStratum": self.requested_stratum,
            "actualSource": self.actual_source,
            "startSampleIndex": self.start_sample_index,
            "startSeconds": _rounded(self.start_seconds),
            "endSeconds": _rounded(self.end_seconds),
            "centerSeconds": _rounded(self.center_seconds),
            "reference": copy.deepcopy(self.reference),
        }
        if self.fallback_reason is not None:
            payload["fallbackReason"] = self.fallback_reason
        return payload


@dataclass(frozen=True)
class _Candidate:
    start_sample_index: int
    actual_source: str
    reference: dict[str, Any]


def _rounded(value: float) -> float:
    return round(float(value), 9)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _sample_step(metadata: VideoMetadata) -> int:
    ratio = metadata.fps / SAMPLE_FPS
    step = int(round(ratio))
    if step < 1 or not math.isclose(ratio, step, rel_tol=0.0, abs_tol=1e-6):
        raise BallAnnotationError(
            f"proxy frame rate {metadata.fps:g} is not an integer multiple of {SAMPLE_FPS} fps"
        )
    if metadata.frame_count < step * FRAMES_PER_WINDOW:
        raise BallAnnotationError("proxy is too short for one three-second annotation window")
    return step


def _total_sample_frames(metadata: VideoMetadata, step: int) -> int:
    return (metadata.frame_count - 1) // step + 1


def _quantized_window_start(anchor_seconds: float, total_samples: int) -> int:
    raw_start = (float(anchor_seconds) - WINDOW_SECONDS / 2) * SAMPLE_FPS
    start = int(math.floor(raw_start + 0.5))
    return min(max(0, start), total_samples - FRAMES_PER_WINDOW)


def _window_times(start_sample_index: int) -> tuple[float, float]:
    return (
        start_sample_index / SAMPLE_FPS,
        (start_sample_index + FRAMES_PER_WINDOW) / SAMPLE_FPS,
    )


def _overlaps(start: float, end: float, intervals: Iterable[Interval]) -> bool:
    return any(start < interval.end and interval.start < end for interval in intervals)


def _fits(start_sample_index: int, interval_start: float, interval_end: float) -> bool:
    start, end = _window_times(start_sample_index)
    return start + 1e-9 >= interval_start and end <= interval_end + 1e-9


def _hard_negatives(recording: Recording, duration: float) -> tuple[dict[str, Any], ...]:
    value = recording.raw.get("hardNegatives", [])
    if not isinstance(value, list):
        raise BallAnnotationError(f"{recording.id}.hardNegatives must be an array")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        where = f"{recording.id}.hardNegatives[{index}]"
        if (
            not isinstance(item, dict)
            or not _is_number(item.get("start"))
            or not _is_number(item.get("end"))
        ):
            raise BallAnnotationError(f"{where} must contain numeric start and end")
        start, end = float(item["start"]), float(item["end"])
        category = item.get("category")
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or end > duration + 1e-6
            or not isinstance(category, str)
            or not category.strip()
        ):
            raise BallAnnotationError(f"{where} is outside the proxy or has no category")
        result.append({"start": start, "end": end, "category": category.strip()})
    return tuple(result)


def _candidate_rank(seed: str, stratum: str, candidate: _Candidate) -> str:
    payload = {
        "seed": seed,
        "stratum": stratum,
        "startSampleIndex": candidate.start_sample_index,
        "actualSource": candidate.actual_source,
        "reference": candidate.reference,
    }
    return _canonical_sha256(payload)


def _is_separated(candidate: _Candidate, selected: Iterable[SamplingWindow]) -> bool:
    minimum = MIN_CENTER_SEPARATION_SECONDS * SAMPLE_FPS
    return all(
        abs(candidate.start_sample_index - window.start_sample_index) >= minimum
        for window in selected
    )


def _deduplicate(candidates: Iterable[_Candidate]) -> list[_Candidate]:
    unique: dict[tuple[int, str], _Candidate] = {}
    for candidate in candidates:
        key = (candidate.start_sample_index, _canonical_sha256(candidate.reference))
        unique.setdefault(key, candidate)
    return list(unique.values())


def _ranked_choice(
    candidates: Iterable[_Candidate],
    *,
    seed: str,
    stratum: str,
    selected: Iterable[SamplingWindow],
) -> _Candidate | None:
    eligible = [item for item in _deduplicate(candidates) if _is_separated(item, selected)]
    if not eligible:
        return None
    return min(eligible, key=lambda item: _candidate_rank(seed, stratum, item))


def _grid_candidates(total_samples: int) -> list[int]:
    maximum = total_samples - FRAMES_PER_WINDOW
    return list(range(0, maximum + 1, SAMPLE_FPS))


def _dead_candidates(
    recording: Recording,
    total_samples: int,
    hard_negatives: tuple[dict[str, Any], ...],
    *,
    include_hard_negatives: bool,
) -> list[_Candidate]:
    result: list[_Candidate] = []
    hard_intervals = tuple(
        Interval(float(item["start"]), float(item["end"])) for item in hard_negatives
    )
    for start_sample_index in _grid_candidates(total_samples):
        start, end = _window_times(start_sample_index)
        if _overlaps(start, end, recording.rallies) or _overlaps(
            start, end, recording.ignored_intervals
        ):
            continue
        if not include_hard_negatives and _overlaps(start, end, hard_intervals):
            continue
        result.append(
            _Candidate(
                start_sample_index,
                "ordinary_dead",
                {"intervalType": "dead_time"},
            )
        )
    return result


def _timeout_candidates(
    total_samples: int,
    hard_negatives: tuple[dict[str, Any], ...],
) -> list[_Candidate]:
    result: list[_Candidate] = []
    for hard_negative_index, item in enumerate(hard_negatives):
        if item["category"].lower().replace("-", "") != "timeout":
            continue
        for start_sample_index in _grid_candidates(total_samples):
            if _fits(start_sample_index, float(item["start"]), float(item["end"])):
                result.append(
                    _Candidate(
                        start_sample_index,
                        "timeout_hard_negative",
                        {
                            "hardNegativeIndex": hard_negative_index,
                            "category": item["category"],
                            "intervalStartSeconds": _rounded(float(item["start"])),
                            "intervalEndSeconds": _rounded(float(item["end"])),
                        },
                    )
                )
    return result


def _uniform_candidates(total_samples: int) -> list[_Candidate]:
    return [
        _Candidate(index, "uniform_fallback", {"intervalType": "uniform"})
        for index in _grid_candidates(total_samples)
    ]


def _as_window(
    requested_stratum: str,
    candidate: _Candidate,
    *,
    fallback_reason: str | None = None,
) -> SamplingWindow:
    return SamplingWindow(
        requested_stratum=requested_stratum,
        actual_source=candidate.actual_source,
        start_sample_index=candidate.start_sample_index,
        reference=copy.deepcopy(candidate.reference),
        fallback_reason=fallback_reason,
    )


def select_sampling_windows(
    recording: Recording,
    metadata: VideoMetadata,
    *,
    manifest_sha256: str,
    round_index: int,
    motion_scores: Mapping[int, float] | None = None,
) -> tuple[SamplingWindow, ...]:
    """Select the fixed six-window development-only annotation protocol.

    ``motion_scores`` maps 15 fps window start indices to mean normalized frame
    difference. It is required only when no eligible timeout hard negative exists.
    """
    if not _valid_sha256(manifest_sha256):
        raise BallAnnotationError("manifest_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 1:
        raise BallAnnotationError("round_index must be a positive integer")
    if recording.split not in {"train", "validation"}:
        raise BallAnnotationError(
            f"{recording.id} is split {recording.split!r}; ball-pilot sampling is development-only"
        )
    step = _sample_step(metadata)
    total_samples = _total_sample_frames(metadata, step)
    duration = metadata.frame_count / metadata.fps
    hard_negatives = _hard_negatives(recording, duration)
    seed = hashlib.sha256(
        f"{manifest_sha256}\0{recording.id}\0{round_index}".encode("utf-8")
    ).hexdigest()
    selected: list[SamplingWindow] = []
    uniform = _uniform_candidates(total_samples)

    def select_or_fallback(
        stratum: str,
        candidates: Iterable[_Candidate],
        reason: str,
    ) -> _Candidate:
        candidate = _ranked_choice(
            candidates,
            seed=seed,
            stratum=stratum,
            selected=selected,
        )
        if candidate is not None:
            selected.append(_as_window(stratum, candidate))
            return candidate
        fallback = _ranked_choice(
            uniform,
            seed=seed,
            stratum=f"{stratum}:fallback",
            selected=selected,
        )
        if fallback is None:
            raise BallAnnotationError(
                f"{recording.id} cannot fit six windows with centers at least "
                f"{MIN_CENTER_SEPARATION_SECONDS} seconds apart"
            )
        selected.append(_as_window(stratum, fallback, fallback_reason=reason))
        return fallback

    serve_candidates = [
        _Candidate(
            _quantized_window_start(rally.start, total_samples),
            "rally_serve_contact",
            {
                "rallyIndex": index,
                "rallyStartSeconds": _rounded(rally.start),
                "rallyEndSeconds": _rounded(rally.end),
            },
        )
        for index, rally in enumerate(recording.rallies)
    ]
    select_or_fallback(
        "serve_window",
        serve_candidates,
        "no rally start window satisfied the bounds and separation constraints",
    )

    midpoint_candidates: list[_Candidate] = []
    for index, rally in enumerate(recording.rallies):
        start = _quantized_window_start((rally.start + rally.end) / 2, total_samples)
        if _fits(start, rally.start, rally.end):
            midpoint_candidates.append(
                _Candidate(
                    start,
                    "rally_mid_live",
                    {
                        "rallyIndex": index,
                        "rallyStartSeconds": _rounded(rally.start),
                        "rallyEndSeconds": _rounded(rally.end),
                    },
                )
            )
    mid_a = select_or_fallback(
        "mid_live_a",
        midpoint_candidates,
        "no fully-live rally midpoint satisfied the bounds and separation constraints",
    )
    first_mid_rally = mid_a.reference.get("rallyIndex") if mid_a.actual_source == "rally_mid_live" else None
    distinct_midpoints = [
        item
        for item in midpoint_candidates
        if first_mid_rally is None or item.reference.get("rallyIndex") != first_mid_rally
    ]
    select_or_fallback(
        "mid_live_b",
        distinct_midpoints,
        "no second distinct-rally midpoint satisfied the bounds and separation constraints",
    )

    end_candidates = [
        _Candidate(
            _quantized_window_start(rally.end, total_samples),
            "rally_end_transition",
            {
                "rallyIndex": index,
                "rallyStartSeconds": _rounded(rally.start),
                "rallyEndSeconds": _rounded(rally.end),
            },
        )
        for index, rally in enumerate(recording.rallies)
    ]
    select_or_fallback(
        "end_transition",
        end_candidates,
        "no rally end transition satisfied the bounds and separation constraints",
    )

    ordinary_dead = _dead_candidates(
        recording,
        total_samples,
        hard_negatives,
        include_hard_negatives=False,
    )
    select_or_fallback(
        "ordinary_dead",
        ordinary_dead,
        "no non-rally window outside ignored and hard-negative intervals was eligible",
    )

    timeouts = _timeout_candidates(total_samples, hard_negatives)
    timeout = _ranked_choice(
        timeouts,
        seed=seed,
        stratum="high_motion_dead:timeout",
        selected=selected,
    )
    if timeout is not None:
        selected.append(_as_window("high_motion_dead", timeout))
    else:
        high_motion_candidates = [
            item
            for item in _dead_candidates(
                recording,
                total_samples,
                hard_negatives,
                include_hard_negatives=True,
            )
            if _is_separated(item, selected)
            and motion_scores is not None
            and item.start_sample_index in motion_scores
            and math.isfinite(float(motion_scores[item.start_sample_index]))
        ]
        if high_motion_candidates:
            best_score = max(float(motion_scores[item.start_sample_index]) for item in high_motion_candidates)  # type: ignore[index]
            tied = [
                item
                for item in high_motion_candidates
                if math.isclose(
                    float(motion_scores[item.start_sample_index]),  # type: ignore[index]
                    best_score,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ]
            candidate = min(
                tied,
                key=lambda item: _candidate_rank(seed, "high_motion_dead:motion", item),
            )
            reference = dict(candidate.reference)
            reference["motionMeanAbsDiff"] = _rounded(best_score)
            selected.append(
                SamplingWindow(
                    requested_stratum="high_motion_dead",
                    actual_source="highest_frame_difference_dead",
                    start_sample_index=candidate.start_sample_index,
                    reference=reference,
                    fallback_reason=(
                        "no eligible timeout hard negative; used the highest frame-difference "
                        "dead window"
                    ),
                )
            )
        else:
            reason = (
                "no eligible timeout hard negative or scored dead window; used a uniform window"
            )
            fallback = _ranked_choice(
                uniform,
                seed=seed,
                stratum="high_motion_dead:fallback",
                selected=selected,
            )
            if fallback is None:
                raise BallAnnotationError(
                    f"{recording.id} cannot fit the high-motion fallback window"
                )
            selected.append(_as_window("high_motion_dead", fallback, fallback_reason=reason))

    if tuple(window.requested_stratum for window in selected) != STRATA:
        raise AssertionError("internal error: ball annotation strata are incomplete")
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            if abs(left.center_seconds - right.center_seconds) < MIN_CENTER_SEPARATION_SECONDS - 1e-9:
                raise AssertionError("internal error: selected window centers are too close")
    return tuple(selected)


def motion_candidate_start_indices(
    recording: Recording,
    metadata: VideoMetadata,
) -> tuple[int, ...]:
    """Return dead-window starts that may need a frame-difference score."""
    step = _sample_step(metadata)
    total_samples = _total_sample_frames(metadata, step)
    hard_negatives = _hard_negatives(recording, metadata.frame_count / metadata.fps)
    return tuple(
        item.start_sample_index
        for item in _dead_candidates(
            recording,
            total_samples,
            hard_negatives,
            include_hard_negatives=True,
        )
    )


def has_usable_timeout(recording: Recording, metadata: VideoMetadata) -> bool:
    step = _sample_step(metadata)
    total_samples = _total_sample_frames(metadata, step)
    hard_negatives = _hard_negatives(recording, metadata.frame_count / metadata.fps)
    return bool(_timeout_candidates(total_samples, hard_negatives))


def compute_frame_difference_scores(
    video: str | Path,
    metadata: VideoMetadata,
    window_start_indices: Iterable[int],
    *,
    motion_fps: int = 3,
    resize_width: int = 160,
    progress: Progress | None = None,
) -> dict[int, float]:
    """Score candidate windows with a low-resolution, whole-frame difference scan."""
    if motion_fps < 1 or SAMPLE_FPS % motion_fps != 0:
        raise BallAnnotationError("motion_fps must be a positive divisor of 15")
    if resize_width < 32:
        raise BallAnnotationError("resize_width must be at least 32 pixels")
    source_stride_ratio = metadata.fps / motion_fps
    source_stride = int(round(source_stride_ratio))
    if not math.isclose(source_stride_ratio, source_stride, rel_tol=0.0, abs_tol=1e-6):
        raise BallAnnotationError("proxy frame rate is incompatible with the motion scan rate")
    starts = tuple(sorted(set(int(item) for item in window_start_indices)))
    if not starts:
        return {}

    try:
        import cv2
    except ImportError as error:
        raise BallAnnotationError("OpenCV is required to score hard-negative motion") from error
    video_path = Path(video).expanduser().resolve()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise BallAnnotationError(f"cannot open proxy for motion scan: {video_path}")
    timeline: list[tuple[float, float]] = []
    previous = None
    try:
        for source_index in range(metadata.frame_count):
            if not capture.grab():
                if source_index < metadata.frame_count - 1:
                    raise BallAnnotationError(
                        f"proxy decode stopped at frame {source_index} during motion scan"
                    )
                break
            if source_index % source_stride != 0:
                continue
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                raise BallAnnotationError(
                    f"cannot retrieve proxy frame {source_index} during motion scan"
                )
            height = max(18, int(round(frame.shape[0] * resize_width / frame.shape[1])))
            gray = cv2.cvtColor(
                cv2.resize(frame, (resize_width, height), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY,
            )
            if previous is not None:
                score = float(cv2.absdiff(gray, previous).mean() / 255.0)
                timeline.append((source_index / metadata.fps, score))
            previous = gray
            if progress and source_index and source_index % 30000 == 0:
                progress(f"motion scan decoded {source_index}/{metadata.frame_count} frames")
    finally:
        capture.release()
    if not timeline:
        raise BallAnnotationError(f"motion scan produced no frame differences for {video_path}")

    result: dict[int, float] = {}
    for start_sample_index in starts:
        start, end = _window_times(start_sample_index)
        values = [score for timestamp, score in timeline if start <= timestamp < end]
        if values:
            result[start_sample_index] = sum(values) / len(values)
    return result


def _initial_annotation() -> dict[str, Any]:
    return {
        "status": "unreviewed",
        "primaryBallState": None,
        "objects": [],
        "notes": "",
    }


def _read_normalization_provenance(video: Path) -> dict[str, Any] | None:
    path = video.with_suffix(video.suffix + ".provenance.json")
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BallAnnotationError(f"cannot read proxy provenance {path}: {error}") from error
    source = payload.get("source")
    if not isinstance(source, dict):
        raise BallAnnotationError(f"proxy provenance has no source object: {path}")
    result = {
        "sidecarFilename": path.name,
        "sidecarSha256": _sha256_file(path),
        "originalSource": {
            "filename": source.get("filename"),
            "sizeBytes": source.get("sizeBytes"),
            "sha256": source.get("sha256"),
        },
    }
    return result


def _extract_exact_frames(
    recording: Recording,
    metadata: VideoMetadata,
    windows: tuple[SamplingWindow, ...],
    *,
    image_root: Path,
    task_path: Path,
    progress: Progress | None = None,
) -> list[dict[str, Any]]:
    try:
        import cv2
    except ImportError as error:
        raise BallAnnotationError("OpenCV is required to extract annotation frames") from error
    step = _sample_step(metadata)
    wanted: dict[int, tuple[int, int]] = {}
    for window_index, window in enumerate(windows):
        for offset in range(FRAMES_PER_WINDOW):
            sample_index = window.start_sample_index + offset
            source_index = sample_index * step
            if source_index in wanted:
                raise BallAnnotationError("selected windows unexpectedly share a source frame")
            wanted[source_index] = (window_index, offset)
    maximum_source_index = max(wanted)
    image_root.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(recording.video))
    if not capture.isOpened():
        raise BallAnnotationError(f"cannot open proxy for exact-frame extraction: {recording.video}")
    frames: list[dict[str, Any]] = []
    try:
        for source_index in range(maximum_source_index + 1):
            if not capture.grab():
                raise BallAnnotationError(
                    f"proxy decode stopped before requested source frame {source_index}"
                )
            target = wanted.get(source_index)
            if target is None:
                continue
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                raise BallAnnotationError(f"cannot retrieve source frame {source_index}")
            window_index, offset = target
            window = windows[window_index]
            window_id = f"{window_index + 1:02d}-{window.requested_stratum}"
            directory = image_root / window_id
            directory.mkdir(parents=True, exist_ok=True)
            frame_id = f"f{source_index:09d}"
            image_path = directory / f"{frame_id}.png"
            parameters = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            if not cv2.imwrite(str(image_path), frame, parameters):
                raise BallAnnotationError(f"cannot write annotation image {image_path}")
            height, width = frame.shape[:2]
            relative_image = os.path.relpath(image_path, task_path.parent)
            frames.append(
                {
                    "id": frame_id,
                    "windowId": window_id,
                    "sampleOffset": offset,
                    "sourceFrameIndex": source_index,
                    "sourceTimestampSeconds": _rounded(source_index / metadata.fps),
                    "image": {
                        "path": relative_image,
                        "sha256": _sha256_file(image_path),
                        "width": int(width),
                        "height": int(height),
                        "format": "png",
                    },
                }
            )
            if progress and len(frames) % 90 == 0:
                progress(
                    f"{recording.id}: extracted {len(frames)}/{len(wanted)} exact proxy frames"
                )
    finally:
        capture.release()
    if len(frames) != len(wanted):
        raise BallAnnotationError(
            f"extracted {len(frames)} of {len(wanted)} requested frames for {recording.id}"
        )
    return sorted(frames, key=lambda item: (item["windowId"], item["sampleOffset"]))


def _immutable_digest(immutable: Mapping[str, Any]) -> str:
    unsigned = dict(immutable)
    unsigned.pop("taskId", None)
    unsigned.pop("digestSha256", None)
    return _canonical_sha256(unsigned)


def _build_task(
    manifest: DatasetManifest,
    manifest_sha256: str,
    recording: Recording,
    metadata: VideoMetadata,
    windows: tuple[SamplingWindow, ...],
    frames: list[dict[str, Any]],
    *,
    round_index: int,
) -> dict[str, Any]:
    if recording.content_sha256 is None:
        raise BallAnnotationError(f"{recording.id} has no verified proxy SHA-256")
    normalization = _read_normalization_provenance(recording.video)
    immutable: dict[str, Any] = {
        "manifest": {
            "name": manifest.name,
            "filename": manifest.path.name,
            "pathHint": str(manifest.path),
            "sha256": manifest_sha256,
        },
        "recording": {
            "id": recording.id,
            "split": recording.split,
            "sourceGroup": recording.source_group,
            "environment": recording.environment,
        },
        "source": {
            "proxy": {
                "filename": recording.video.name,
                "pathHint": str(recording.video),
                "sizeBytes": recording.video.stat().st_size,
                "sha256": recording.content_sha256,
                "width": metadata.width,
                "height": metadata.height,
                "fps": _rounded(metadata.fps),
                "frameCount": metadata.frame_count,
                "durationSeconds": _rounded(metadata.frame_count / metadata.fps),
            },
            "normalizationProvenance": normalization,
        },
        "sampling": {
            "policyId": BALL_SAMPLING_POLICY_ID,
            "round": round_index,
            "sampleFps": SAMPLE_FPS,
            "windowSeconds": WINDOW_SECONDS,
            "framesPerWindow": FRAMES_PER_WINDOW,
            "minimumCenterSeparationSeconds": MIN_CENTER_SEPARATION_SECONDS,
            "seedMaterial": "manifest SHA-256 + recording id + round",
            "frameRule": (
                "sourceFrameIndex is authoritative; timestamps are CFR frame-index/fps "
                "derivatives from the normalized proxy"
            ),
        },
        "annotationPolicy": copy.deepcopy(ANNOTATION_POLICY),
        "windows": [window.to_dict(index) for index, window in enumerate(windows)],
        "frames": frames,
    }
    digest = _immutable_digest(immutable)
    immutable["taskId"] = f"ball-presence-{digest[:24]}"
    immutable["digestSha256"] = digest
    annotations = {item["id"]: _initial_annotation() for item in frames}
    return {
        "schemaVersion": BALL_ANNOTATION_SCHEMA_VERSION,
        "taskType": BALL_ANNOTATION_TASK_TYPE,
        "immutable": immutable,
        "suggestions": {
            "status": "empty",
            "model": None,
            "frames": {},
        },
        "annotations": {
            "review": {
                "status": "unreviewed",
                "annotator": None,
                "reviewedAt": None,
                "notes": "",
            },
            "frames": annotations,
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _recording_artifact_paths(pilot_root: Path, recording_id: str) -> tuple[Path, Path]:
    """Resolve one recording's outputs without permitting path components."""

    if (
        not isinstance(recording_id, str)
        or SAFE_RECORDING_ID_PATTERN.fullmatch(recording_id) is None
    ):
        raise BallAnnotationError(
            "ball-pilot recording ids may contain only ASCII letters, digits, '_' and '-'"
        )
    root = pilot_root.resolve()
    tasks_root = (root / "tasks").resolve()
    images_root = (root / "images").resolve()
    task_path = (tasks_root / f"{recording_id}.ball-presence.json").resolve()
    image_root = (images_root / recording_id).resolve()
    if task_path.parent != tasks_root or image_root.parent != images_root:
        raise BallAnnotationError("ball-pilot recording output escaped its artifact root")
    return task_path, image_root


def prepare_ball_presence_pilot(
    manifest_path: str | Path,
    output_directory: str | Path,
    *,
    round_index: int = 1,
    expected_recordings: int | None = 8,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Create development-only ball-presence tasks without touching test video frames."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing ball pilot: {output}")
    if expected_recordings is not None and (
        not isinstance(expected_recordings, int)
        or isinstance(expected_recordings, bool)
        or expected_recordings < 1
    ):
        raise BallAnnotationError("expected_recordings must be a positive integer or null")
    manifest_sha256 = _sha256_file(manifest_file)
    manifest = load_manifest(manifest_file)
    recordings = tuple(
        item for item in manifest.recordings if item.split in {"train", "validation"}
    )
    excluded = tuple(
        {"id": item.id, "split": item.split}
        for item in manifest.recordings
        if item.split not in {"train", "validation"}
    )
    if expected_recordings is not None and len(recordings) != expected_recordings:
        raise BallAnnotationError(
            f"expected {expected_recordings} development recordings, found {len(recordings)}"
        )
    if not recordings:
        raise BallAnnotationError("manifest contains no train/validation recordings")
    for recording in recordings:
        if SAFE_RECORDING_ID_PATTERN.fullmatch(recording.id) is None:
            raise BallAnnotationError(
                f"unsafe ball-pilot recording id {recording.id!r}; use only ASCII "
                "letters, digits, '_' and '-'"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", suffix=".tmp", dir=output.parent)
    )
    task_rows: list[dict[str, Any]] = []
    try:
        for recording_index, recording in enumerate(recordings):
            if progress:
                progress(
                    f"[{recording_index + 1}/{len(recordings)}] prepare {recording.id}"
                )
            metadata = probe_video(recording.video)
            _sample_step(metadata)
            windows = select_sampling_windows(
                recording,
                metadata,
                manifest_sha256=manifest_sha256,
                round_index=round_index,
            )
            if windows[-1].actual_source != "timeout_hard_negative":
                starts = motion_candidate_start_indices(recording, metadata)
                motion_scores = compute_frame_difference_scores(
                    recording.video,
                    metadata,
                    starts,
                    progress=progress,
                )
                windows = select_sampling_windows(
                    recording,
                    metadata,
                    manifest_sha256=manifest_sha256,
                    round_index=round_index,
                    motion_scores=motion_scores,
                )
            task_path, image_root = _recording_artifact_paths(temporary, recording.id)
            frames = _extract_exact_frames(
                recording,
                metadata,
                windows,
                image_root=image_root,
                task_path=task_path,
                progress=progress,
            )
            task = _build_task(
                manifest,
                manifest_sha256,
                recording,
                metadata,
                windows,
                frames,
                round_index=round_index,
            )
            _write_json(task_path, task)
            validate_ball_annotation_task(task, task_path=task_path, verify_images=True)
            task_rows.append(
                {
                    "recordingId": recording.id,
                    "split": recording.split,
                    "task": os.path.relpath(task_path, temporary),
                    "taskId": task["immutable"]["taskId"],
                    "initialTaskSha256": _sha256_file(task_path),
                    "frameCount": len(frames),
                }
            )
        index = {
            "schemaVersion": 1,
            "artifactType": "volleycut-ball-presence-pilot-index",
            "manifest": {
                "pathHint": str(manifest.path),
                "sha256": manifest_sha256,
            },
            "samplingPolicyId": BALL_SAMPLING_POLICY_ID,
            "round": round_index,
            "developmentOnly": True,
            "excludedRecordings": list(excluded),
            "recordingCount": len(task_rows),
            "windowCount": len(task_rows) * len(STRATA),
            "frameCount": sum(int(item["frameCount"]) for item in task_rows),
            "tasks": task_rows,
        }
        _write_json(temporary / "index.json", index)
        temporary.replace(output)
        return {
            "output": str(output),
            "index": str(output / "index.json"),
            **{key: index[key] for key in ("recordingCount", "windowCount", "frameCount")},
            "excludedRecordings": list(excluded),
        }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _require_keys(value: Any, keys: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BallAnnotationError(f"{where} must be an object")
    missing = keys - set(value)
    if missing:
        raise BallAnnotationError(f"{where} is missing {sorted(missing)}")
    return value


def _validate_bbox(value: Any, where: str) -> None:
    bbox = _require_keys(value, {"x", "y", "width", "height"}, where)
    values = tuple(bbox[key] for key in ("x", "y", "width", "height"))
    if any(not _is_number(item) or not math.isfinite(float(item)) for item in values):
        raise BallAnnotationError(f"{where} coordinates must be finite numbers")
    x, y, width, height = (float(item) for item in values)
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
        raise BallAnnotationError(f"{where} must be a normalized rectangle inside the image")


def validate_ball_annotation_task(
    value: Any,
    *,
    task_path: str | Path | None = None,
    verify_images: bool = False,
) -> dict[str, Any]:
    """Validate task structure, immutable digest, labels, suggestions, and optional images."""
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        try:
            task = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BallAnnotationError(f"cannot read ball annotation task {path}: {error}") from error
        if task_path is None:
            task_path = path
    else:
        task = value
    root = _require_keys(
        task,
        {"schemaVersion", "taskType", "immutable", "suggestions", "annotations"},
        "task",
    )
    if root["schemaVersion"] != BALL_ANNOTATION_SCHEMA_VERSION:
        raise BallAnnotationError(
            f"task.schemaVersion must be {BALL_ANNOTATION_SCHEMA_VERSION}"
        )
    if root["taskType"] != BALL_ANNOTATION_TASK_TYPE:
        raise BallAnnotationError(f"task.taskType must be {BALL_ANNOTATION_TASK_TYPE!r}")
    immutable = _require_keys(
        root["immutable"],
        {
            "taskId",
            "digestSha256",
            "manifest",
            "recording",
            "source",
            "sampling",
            "annotationPolicy",
            "windows",
            "frames",
        },
        "task.immutable",
    )
    digest = _immutable_digest(immutable)
    if immutable["digestSha256"] != digest:
        raise BallAnnotationError("task immutable provenance digest does not match its contents")
    if immutable["taskId"] != f"ball-presence-{digest[:24]}":
        raise BallAnnotationError("task immutable taskId does not match its provenance digest")
    recording = _require_keys(
        immutable["recording"],
        {"id", "split", "sourceGroup", "environment"},
        "immutable.recording",
    )
    if not isinstance(recording["id"], str) or not recording["id"].strip():
        raise BallAnnotationError("immutable.recording.id must be a non-empty string")
    if recording["split"] not in {"train", "validation"}:
        raise BallAnnotationError(
            "immutable.recording.split must be train or validation for this development pilot"
        )
    if (
        not isinstance(recording["sourceGroup"], str)
        or not recording["sourceGroup"].strip()
    ):
        raise BallAnnotationError(
            "immutable.recording.sourceGroup must be a non-empty string"
        )
    if recording["environment"] not in ENVIRONMENTS:
        raise BallAnnotationError("immutable.recording.environment is invalid")
    manifest = _require_keys(
        immutable["manifest"], {"name", "filename", "pathHint", "sha256"}, "immutable.manifest"
    )
    if not _valid_sha256(manifest["sha256"]):
        raise BallAnnotationError("immutable.manifest.sha256 must be a SHA-256 digest")
    source = _require_keys(immutable["source"], {"proxy", "normalizationProvenance"}, "immutable.source")
    proxy = _require_keys(
        source["proxy"],
        {
            "filename",
            "pathHint",
            "sizeBytes",
            "sha256",
            "width",
            "height",
            "fps",
            "frameCount",
            "durationSeconds",
        },
        "immutable.source.proxy",
    )
    if not _valid_sha256(proxy["sha256"]):
        raise BallAnnotationError("immutable.source.proxy.sha256 must be a SHA-256 digest")
    if (
        not isinstance(proxy["width"], int)
        or isinstance(proxy["width"], bool)
        or proxy["width"] <= 0
        or not isinstance(proxy["height"], int)
        or isinstance(proxy["height"], bool)
        or proxy["height"] <= 0
        or not isinstance(proxy["frameCount"], int)
        or isinstance(proxy["frameCount"], bool)
        or proxy["frameCount"] <= 0
        or not isinstance(proxy["sizeBytes"], int)
        or isinstance(proxy["sizeBytes"], bool)
        or proxy["sizeBytes"] <= 0
        or not _is_number(proxy["fps"])
        or float(proxy["fps"]) <= 0
        or not _is_number(proxy["durationSeconds"])
        or float(proxy["durationSeconds"]) <= 0
    ):
        raise BallAnnotationError("immutable.source.proxy numeric metadata is invalid")
    proxy_fps = float(proxy["fps"])
    source_step = int(round(proxy_fps / SAMPLE_FPS))
    if source_step < 1 or not math.isclose(
        proxy_fps / SAMPLE_FPS,
        source_step,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise BallAnnotationError("immutable source fps is incompatible with exact 15 fps frames")
    if not math.isclose(
        float(proxy["durationSeconds"]),
        proxy["frameCount"] / proxy_fps,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise BallAnnotationError("immutable source duration does not match frameCount/fps")
    total_sample_frames = (proxy["frameCount"] - 1) // source_step + 1
    sampling = _require_keys(
        immutable["sampling"],
        {
            "policyId",
            "round",
            "sampleFps",
            "windowSeconds",
            "framesPerWindow",
            "minimumCenterSeparationSeconds",
            "seedMaterial",
            "frameRule",
        },
        "immutable.sampling",
    )
    if (
        sampling["policyId"] != BALL_SAMPLING_POLICY_ID
        or sampling["sampleFps"] != SAMPLE_FPS
        or sampling["windowSeconds"] != WINDOW_SECONDS
        or sampling["framesPerWindow"] != FRAMES_PER_WINDOW
        or sampling["minimumCenterSeparationSeconds"] != MIN_CENTER_SEPARATION_SECONDS
    ):
        raise BallAnnotationError("immutable sampling policy constants do not match the schema")
    if immutable["annotationPolicy"] != ANNOTATION_POLICY:
        raise BallAnnotationError(
            f"immutable.annotationPolicy must match {ANNOTATION_POLICY_ID!r}"
        )
    windows = immutable["windows"]
    if not isinstance(windows, list) or len(windows) != len(STRATA):
        raise BallAnnotationError(f"immutable.windows must contain exactly {len(STRATA)} windows")
    centers: list[float] = []
    window_ids: set[str] = set()
    window_starts: dict[str, int] = {}
    for index, (window, expected_stratum) in enumerate(zip(windows, STRATA)):
        where = f"immutable.windows[{index}]"
        row = _require_keys(
            window,
            {
                "id",
                "requestedStratum",
                "actualSource",
                "startSampleIndex",
                "startSeconds",
                "endSeconds",
                "centerSeconds",
                "reference",
            },
            where,
        )
        if row["requestedStratum"] != expected_stratum:
            raise BallAnnotationError(f"{where}.requestedStratum must be {expected_stratum!r}")
        if not isinstance(row["id"], str) or row["id"] in window_ids:
            raise BallAnnotationError(f"{where}.id must be a unique string")
        window_ids.add(row["id"])
        if (
            not isinstance(row["startSampleIndex"], int)
            or isinstance(row["startSampleIndex"], bool)
            or row["startSampleIndex"] < 0
            or row["startSampleIndex"] + FRAMES_PER_WINDOW > total_sample_frames
        ):
            raise BallAnnotationError(
                f"{where}.startSampleIndex must fit a complete window inside the proxy"
            )
        expected_start = row["startSampleIndex"] / SAMPLE_FPS
        expected_end = (row["startSampleIndex"] + FRAMES_PER_WINDOW) / SAMPLE_FPS
        expected_center = (row["startSampleIndex"] + FRAMES_PER_WINDOW / 2) / SAMPLE_FPS
        seconds = (row["startSeconds"], row["endSeconds"], row["centerSeconds"])
        if any(not _is_number(item) for item in seconds) or not all(
            math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-8)
            for actual, expected in zip(seconds, (expected_start, expected_end, expected_center))
        ):
            raise BallAnnotationError(
                f"{where} start/end/center seconds do not match its sample index"
            )
        centers.append(expected_center)
        window_starts[row["id"]] = row["startSampleIndex"]
        if row["actualSource"] == "uniform_fallback" and not isinstance(
            row.get("fallbackReason"), str
        ):
            raise BallAnnotationError(f"{where} uniform fallback must record fallbackReason")
    for left_index, left in enumerate(centers):
        for right in centers[left_index + 1 :]:
            if abs(left - right) < MIN_CENTER_SEPARATION_SECONDS - 1e-8:
                raise BallAnnotationError("immutable window centers violate minimum separation")

    frames = immutable["frames"]
    expected_frames = len(STRATA) * FRAMES_PER_WINDOW
    if not isinstance(frames, list) or len(frames) != expected_frames:
        raise BallAnnotationError(
            f"immutable.frames must contain exactly {expected_frames} frame records"
        )
    frame_ids: set[str] = set()
    task_file = Path(task_path).expanduser().resolve() if task_path is not None else None
    artifact_root = task_file.parent.parent if task_file is not None else None
    per_window: dict[str, list[int]] = {item: [] for item in window_ids}
    for index, frame in enumerate(frames):
        where = f"immutable.frames[{index}]"
        row = _require_keys(
            frame,
            {
                "id",
                "windowId",
                "sampleOffset",
                "sourceFrameIndex",
                "sourceTimestampSeconds",
                "image",
            },
            where,
        )
        frame_id = row["id"]
        if not isinstance(frame_id, str) or not frame_id or frame_id in frame_ids:
            raise BallAnnotationError(f"{where}.id must be a unique non-empty string")
        frame_ids.add(frame_id)
        if row["windowId"] not in window_ids:
            raise BallAnnotationError(f"{where}.windowId is unknown")
        if (
            not isinstance(row["sampleOffset"], int)
            or isinstance(row["sampleOffset"], bool)
            or not 0 <= row["sampleOffset"] < FRAMES_PER_WINDOW
        ):
            raise BallAnnotationError(f"{where}.sampleOffset is invalid")
        per_window[row["windowId"]].append(row["sampleOffset"])
        expected_source_index = (
            window_starts[row["windowId"]] + row["sampleOffset"]
        ) * source_step
        if (
            not isinstance(row["sourceFrameIndex"], int)
            or isinstance(row["sourceFrameIndex"], bool)
            or row["sourceFrameIndex"] != expected_source_index
            or row["sourceFrameIndex"] >= proxy["frameCount"]
        ):
            raise BallAnnotationError(
                f"{where}.sourceFrameIndex does not match window start, offset, and fps"
            )
        expected_timestamp = expected_source_index / proxy_fps
        if not _is_number(row["sourceTimestampSeconds"]) or not math.isclose(
            float(row["sourceTimestampSeconds"]),
            expected_timestamp,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise BallAnnotationError(
                f"{where}.sourceTimestampSeconds does not match sourceFrameIndex/fps"
            )
        image = _require_keys(
            row["image"], {"path", "sha256", "width", "height", "format"}, f"{where}.image"
        )
        if not isinstance(image["path"], str) or not image["path"]:
            raise BallAnnotationError(f"{where}.image.path must be a non-empty string")
        if not _valid_sha256(image["sha256"]) or image["format"] != "png":
            raise BallAnnotationError(f"{where}.image must describe a SHA-256-pinned PNG")
        if image["width"] != proxy["width"] or image["height"] != proxy["height"]:
            raise BallAnnotationError(f"{where}.image dimensions do not match the proxy")
        if verify_images:
            if task_file is None or artifact_root is None:
                raise BallAnnotationError("task_path is required when verify_images is true")
            image_path = (task_file.parent / image["path"]).resolve()
            try:
                image_path.relative_to(artifact_root)
            except ValueError as error:
                raise BallAnnotationError(f"{where}.image.path escapes the pilot directory") from error
            if not image_path.is_file() or _sha256_file(image_path) != image["sha256"]:
                raise BallAnnotationError(f"{where}.image file is missing or has changed")
    if any(sorted(offsets) != list(range(FRAMES_PER_WINDOW)) for offsets in per_window.values()):
        raise BallAnnotationError("each immutable window must contain every 15 fps sample offset once")

    suggestions = _require_keys(root["suggestions"], {"status", "model", "frames"}, "suggestions")
    if suggestions["status"] not in {"empty", "partial", "complete"}:
        raise BallAnnotationError("suggestions.status must be empty, partial, or complete")
    suggestion_frames = suggestions["frames"]
    if not isinstance(suggestion_frames, dict) or any(key not in frame_ids for key in suggestion_frames):
        raise BallAnnotationError("suggestions.frames must be keyed only by immutable frame ids")
    if suggestions["status"] == "empty" and (suggestions["model"] is not None or suggestion_frames):
        raise BallAnnotationError("empty suggestions must not contain a model or frame proposals")
    if suggestions["status"] != "empty" and not isinstance(suggestions["model"], dict):
        raise BallAnnotationError("non-empty suggestions require separate model provenance")
    if suggestions["status"] == "complete" and set(suggestion_frames) != frame_ids:
        raise BallAnnotationError("complete suggestions require a proposal row for every frame")
    for frame_id, suggestion in suggestion_frames.items():
        where = f"suggestions.frames[{frame_id!r}]"
        row = _require_keys(suggestion, {"ballPresenceProbability", "detections"}, where)
        probability = row["ballPresenceProbability"]
        if not _is_number(probability) or not 0 <= float(probability) <= 1:
            raise BallAnnotationError(f"{where}.ballPresenceProbability must be in [0,1]")
        if not isinstance(row["detections"], list):
            raise BallAnnotationError(f"{where}.detections must be an array")
        for detection_index, detection in enumerate(row["detections"]):
            detection_where = f"{where}.detections[{detection_index}]"
            detection_row = _require_keys(detection, {"confidence", "bbox"}, detection_where)
            if set(detection_row) != {"confidence", "bbox"}:
                raise BallAnnotationError(
                    f"{detection_where} must remain a role-free detector proposal"
                )
            confidence = detection_row["confidence"]
            if not _is_number(confidence) or not 0 <= float(confidence) <= 1:
                raise BallAnnotationError(f"{detection_where}.confidence must be in [0,1]")
            _validate_bbox(detection_row["bbox"], f"{detection_where}.bbox")

    annotations = _require_keys(root["annotations"], {"review", "frames"}, "annotations")
    review = _require_keys(
        annotations["review"],
        {"status", "annotator", "reviewedAt", "notes"},
        "annotations.review",
    )
    if review["status"] not in {"unreviewed", "in_progress", "complete"}:
        raise BallAnnotationError("annotations.review.status is invalid")
    if not isinstance(review["notes"], str):
        raise BallAnnotationError("annotations.review.notes must be a string")
    annotation_frames = annotations["frames"]
    if not isinstance(annotation_frames, dict) or set(annotation_frames) != frame_ids:
        raise BallAnnotationError("annotations.frames must contain every immutable frame id exactly once")
    reviewed_count = 0
    for frame_id, annotation in annotation_frames.items():
        where = f"annotations.frames[{frame_id!r}]"
        row = _require_keys(
            annotation,
            {"status", "primaryBallState", "objects", "notes"},
            where,
        )
        if row["status"] not in {"unreviewed", "reviewed"}:
            raise BallAnnotationError(f"{where}.status is invalid")
        state = row["primaryBallState"]
        if row["status"] == "unreviewed" and state is not None:
            raise BallAnnotationError(f"{where} cannot have a label before review")
        if row["status"] == "reviewed" and state not in PRIMARY_BALL_STATES:
            raise BallAnnotationError(f"{where}.primaryBallState is invalid")
        objects = row["objects"]
        if not isinstance(objects, list):
            raise BallAnnotationError(f"{where}.objects must be an array")
        if row["status"] == "unreviewed" and objects:
            raise BallAnnotationError(f"{where} cannot contain objects before review")
        object_ids: set[str] = set()
        primary_count = 0
        for object_index, item in enumerate(objects):
            object_where = f"{where}.objects[{object_index}]"
            object_row = _require_keys(
                item,
                {"id", "category", "role", "bbox", "visibility", "truncated"},
                object_where,
            )
            if set(object_row) != {
                "id",
                "category",
                "role",
                "bbox",
                "visibility",
                "truncated",
            }:
                raise BallAnnotationError(f"{object_where} has unsupported fields")
            object_id = object_row["id"]
            if not isinstance(object_id, str) or not object_id.strip() or object_id in object_ids:
                raise BallAnnotationError(f"{object_where}.id must be unique within the frame")
            object_ids.add(object_id)
            if object_row["category"] != "volleyball":
                raise BallAnnotationError(f"{object_where}.category must be 'volleyball'")
            if object_row["role"] not in OBJECT_ROLES:
                raise BallAnnotationError(f"{object_where}.role is invalid")
            if object_row["visibility"] not in OBJECT_VISIBILITIES:
                raise BallAnnotationError(f"{object_where}.visibility is invalid")
            if not isinstance(object_row["truncated"], bool):
                raise BallAnnotationError(f"{object_where}.truncated must be boolean")
            _validate_bbox(object_row["bbox"], f"{object_where}.bbox")
            primary_count += object_row["role"] == "primary-court"
        if state == "localizable" and primary_count != 1:
            raise BallAnnotationError(
                f"{where} localizable state requires exactly one primary-court object"
            )
        if state in PRIMARY_BALL_STATES - {"localizable"} and primary_count != 0:
            raise BallAnnotationError(
                f"{where} non-localizable state cannot contain a primary-court object"
            )
        if not isinstance(row["notes"], str):
            raise BallAnnotationError(f"{where}.notes must be a string")
        if row["status"] == "reviewed":
            reviewed_count += 1
    if review["status"] == "complete" and reviewed_count != len(frame_ids):
        raise BallAnnotationError("a complete review requires every frame to be reviewed")
    if review["status"] == "unreviewed":
        if reviewed_count != 0 or review["annotator"] is not None or review["reviewedAt"] is not None:
            raise BallAnnotationError(
                "an unreviewed task cannot contain reviewed frames or reviewer metadata"
            )
    elif review["status"] == "in_progress":
        if not 0 < reviewed_count < len(frame_ids):
            raise BallAnnotationError(
                "an in-progress review requires some, but not all, frames to be reviewed"
            )
        if not isinstance(review["annotator"], str) or not review["annotator"].strip():
            raise BallAnnotationError("an in-progress review requires an annotator")
        if review["reviewedAt"] is not None:
            raise BallAnnotationError("reviewedAt must remain null until review is complete")
    else:
        if not isinstance(review["annotator"], str) or not review["annotator"].strip():
            raise BallAnnotationError("a complete review requires an annotator")
        reviewed_at = review["reviewedAt"]
        if not isinstance(reviewed_at, str) or not reviewed_at.strip():
            raise BallAnnotationError("a complete review requires reviewedAt")
        try:
            parsed_reviewed_at = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise BallAnnotationError("annotations.review.reviewedAt must be ISO-8601") from error
        if parsed_reviewed_at.tzinfo is None:
            raise BallAnnotationError("annotations.review.reviewedAt must include a timezone")
    return root


def assert_immutable_provenance_unchanged(original: Any, updated: Any) -> None:
    """Reject review/import updates that alter any source, window, or image provenance."""
    old = validate_ball_annotation_task(original)
    new = validate_ball_annotation_task(updated)
    if old["immutable"] != new["immutable"]:
        raise BallAnnotationError("immutable source/image provenance changed")
