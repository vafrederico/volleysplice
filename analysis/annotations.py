from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .artifacts import atomic_write_text
from .features import probe_video
from .schema import (
    ANNOTATION_POLICY_ID,
    ENVIRONMENTS,
    SPLITS,
    Interval,
    ManifestError,
    _intervals_overlap,
    _read_game,
    _read_intervals,
    _read_roi,
    _sha256_file,
    load_manifest,
)


LABEL_SCHEMA_VERSION = 1
LABEL_KIND = "volleycut-rally-labels"
LABEL_STATUSES = {"not-started", "in-progress", "complete"}
HARD_NEGATIVE_CATEGORIES = {
    "adjacent-court",
    "camera-motion",
    "celebration",
    "foreground-crossing",
    "setup-between-points",
    "timeout",
    "warmup",
    "other",
}


@dataclass(frozen=True)
class LabelDocument:
    path: Path
    payload: dict[str, Any]
    video: Path
    recording_id: str
    source_group: str
    split: str
    environment: str
    duration: float
    rallies: tuple[Interval, ...]
    ignored_intervals: tuple[Interval, ...]
    hard_negatives: tuple[Interval, ...]
    warnings: tuple[str, ...]


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ManifestError(f"{where} must be a non-empty trimmed string")
    return value


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read label document {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError("label document root must be an object")
    if payload.get("schemaVersion") != LABEL_SCHEMA_VERSION:
        raise ManifestError(f"label schemaVersion must be {LABEL_SCHEMA_VERSION}")
    if payload.get("kind") != LABEL_KIND:
        raise ManifestError(f"label kind must be {LABEL_KIND!r}")
    return payload


def _validate_bounds(
    intervals: Sequence[Interval],
    duration: float,
    where: str,
) -> None:
    for index, interval in enumerate(intervals):
        if interval.end > duration + 1e-6:
            raise ManifestError(
                f"{where}[{index}].end exceeds recording duration "
                f"({interval.end:.3f}s > {duration:.3f}s)"
            )


def load_label_document(
    path: str | Path,
    *,
    require_complete: bool = True,
    require_video: bool = True,
) -> LabelDocument:
    label_path = Path(path).expanduser().resolve()
    payload = _read_payload(label_path)
    recording = payload.get("recording")
    if not isinstance(recording, dict):
        raise ManifestError("recording must be an object")
    recording_id = _nonempty_string(recording.get("id"), "recording.id")
    source_group = _nonempty_string(recording.get("sourceGroup"), "recording.sourceGroup")
    split = recording.get("split")
    if split not in SPLITS:
        raise ManifestError(f"recording.split must be one of {sorted(SPLITS)}")
    environment = recording.get("environment", "unknown")
    if environment not in ENVIRONMENTS:
        raise ManifestError(f"recording.environment must be one of {sorted(ENVIRONMENTS)}")

    video_value = recording.get("video")
    if not isinstance(video_value, str) or not video_value:
        raise ManifestError("recording.video must be a non-empty path string")
    video = Path(video_value).expanduser()
    if not video.is_absolute():
        video = (label_path.parent / video).resolve()
    else:
        video = video.resolve()
    if require_video and not video.is_file():
        raise ManifestError(f"recording.video does not exist: {video}")
    content_sha256 = recording.get("contentSha256")
    if (
        not isinstance(content_sha256, str)
        or len(content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in content_sha256.lower())
    ):
        raise ManifestError("recording.contentSha256 must be a SHA-256 hex digest")
    if require_video and _sha256_file(video) != content_sha256.lower():
        raise ManifestError("recording video does not match recording.contentSha256")

    duration_value = recording.get("durationSeconds")
    if (
        not isinstance(duration_value, (int, float))
        or isinstance(duration_value, bool)
        or not math.isfinite(float(duration_value))
        or float(duration_value) <= 0
    ):
        raise ManifestError("recording.durationSeconds must be a positive finite number")
    duration = float(duration_value)
    if require_video:
        actual_duration = probe_video(video).duration
        if abs(actual_duration - duration) > max(0.1, 1 / 24):
            raise ManifestError(
                "recording.durationSeconds does not match the referenced video "
                f"({duration:.3f}s != {actual_duration:.3f}s)"
            )

    annotation_policy = payload.get("annotationPolicy")
    if not isinstance(annotation_policy, dict) or annotation_policy.get("id") != ANNOTATION_POLICY_ID:
        raise ManifestError(f"annotationPolicy.id must be {ANNOTATION_POLICY_ID!r}")
    annotation = payload.get("annotation")
    if not isinstance(annotation, dict):
        raise ManifestError("annotation must be an object")
    status = annotation.get("status")
    if status not in LABEL_STATUSES:
        raise ManifestError(f"annotation.status must be one of {sorted(LABEL_STATUSES)}")
    if require_complete:
        if status != "complete":
            raise ManifestError("annotation.status must be 'complete'")
        _nonempty_string(annotation.get("annotator"), "annotation.annotator")
        if annotation.get("continuousVideoReviewed") is not True:
            raise ManifestError("annotation.continuousVideoReviewed must be true")
        _nonempty_string(annotation.get("reviewedAt"), "annotation.reviewedAt")

    _read_game(recording.get("game"), "recording")
    _read_roi(recording.get("roi"), "recording")
    capture = recording.get("capture", {})
    if not isinstance(capture, dict):
        raise ManifestError("recording.capture must be an object")
    rallies = _read_intervals(payload.get("rallies"), "labels")
    ignored = _read_intervals(payload.get("ignoredIntervals", []), "labels", "ignoredIntervals")
    hard_negatives = _read_intervals(payload.get("hardNegatives", []), "labels", "hardNegatives")
    _validate_bounds(rallies, duration, "rallies")
    _validate_bounds(ignored, duration, "ignoredIntervals")
    _validate_bounds(hard_negatives, duration, "hardNegatives")
    if _intervals_overlap(rallies, ignored):
        raise ManifestError("ignoredIntervals must not overlap rallies")
    if _intervals_overlap(rallies, hard_negatives):
        raise ManifestError("hardNegatives must not overlap rallies")
    if _intervals_overlap(ignored, hard_negatives):
        raise ManifestError("hardNegatives must not overlap ignoredIntervals")
    for index, row in enumerate(payload.get("hardNegatives", [])):
        category = row.get("category") if isinstance(row, dict) else None
        if category not in HARD_NEGATIVE_CATEGORIES:
            raise ManifestError(
                f"hardNegatives[{index}].category must be one of "
                f"{sorted(HARD_NEGATIVE_CATEGORIES)}"
            )

    warnings: list[str] = []
    game = recording.get("game", {})
    if game.get("playersPerTeam") is None:
        warnings.append("players per team are unknown")
    if game.get("targetPoints") is None:
        warnings.append("target points are unknown")
    if not rallies:
        warnings.append("no rallies are labeled; confirm this is an intentional hard-negative recording")
    if ignored:
        warnings.append(f"{len(ignored)} ignored intervals will be excluded from training and scoring")
    return LabelDocument(
        path=label_path,
        payload=payload,
        video=video,
        recording_id=recording_id,
        source_group=source_group,
        split=str(split),
        environment=str(environment),
        duration=duration,
        rallies=rallies,
        ignored_intervals=ignored,
        hard_negatives=hard_negatives,
        warnings=tuple(warnings),
    )


def create_label_draft(
    video_path: str | Path,
    destination: str | Path,
    *,
    recording_id: str,
    source_group: str,
    split: str,
    environment: str,
    players_per_team: int | None = None,
    target_points: int | None = None,
    format_name: str | None = None,
    roi: tuple[float, float, float, float] | None = None,
    capture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(destination).expanduser().resolve()
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise ManifestError(f"video does not exist: {video}")
    if split not in SPLITS:
        raise ManifestError(f"split must be one of {sorted(SPLITS)}")
    if environment not in ENVIRONMENTS:
        raise ManifestError(f"environment must be one of {sorted(ENVIRONMENTS)}")
    _nonempty_string(recording_id, "recording id")
    _nonempty_string(source_group, "source group")
    metadata = probe_video(video)
    relative_video = os.path.relpath(video, output.parent)
    game = {
        "playersPerTeam": players_per_team,
        "targetPoints": target_points,
        "format": format_name,
    }
    _read_game(game, "recording")
    roi_payload = None
    if roi is not None:
        roi_payload = {"x": roi[0], "y": roi[1], "width": roi[2], "height": roi[3]}
        _read_roi(roi_payload, "recording")
    payload: dict[str, Any] = {
        "schemaVersion": LABEL_SCHEMA_VERSION,
        "kind": LABEL_KIND,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "recording": {
            "id": recording_id,
            "video": relative_video,
            "videoFilename": video.name,
            "contentSha256": _sha256_file(video),
            "durationSeconds": round(metadata.duration, 6),
            "sourceGroup": source_group,
            "split": split,
            "environment": environment,
            "game": game,
            "capture": capture or {},
            "roi": roi_payload,
        },
        "annotationPolicy": {
            "id": ANNOTATION_POLICY_ID,
            "rallyStart": "serve-ball contact",
            "rallyEnd": "first instant live play has ended",
            "intervalConvention": "half-open [start,end) seconds on this normalized video",
        },
        "annotation": {
            "status": "not-started",
            "annotator": "",
            "continuousVideoReviewed": False,
            "reviewedAt": None,
            "notes": "",
        },
        "rallies": [],
        "ignoredIntervals": [],
        "hardNegatives": [],
    }
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def build_manifest_from_labels(
    label_paths: Sequence[str | Path],
    destination: str | Path,
    *,
    name: str,
    require_videos: bool = True,
) -> dict[str, Any]:
    if not label_paths:
        raise ManifestError("at least one completed label document is required")
    output = Path(destination).expanduser().resolve()
    documents = [
        load_label_document(path, require_complete=True, require_video=require_videos)
        for path in label_paths
    ]
    ids: set[str] = set()
    group_splits: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for document in sorted(documents, key=lambda item: item.recording_id):
        if document.recording_id in ids:
            raise ManifestError(f"duplicate recording id: {document.recording_id}")
        ids.add(document.recording_id)
        old_split = group_splits.setdefault(document.source_group, document.split)
        if old_split != document.split:
            raise ManifestError(
                f"sourceGroup {document.source_group!r} crosses {old_split!r} "
                f"and {document.split!r} splits"
            )
        recording = document.payload["recording"]
        rows.append(
            {
                "id": document.recording_id,
                "video": os.path.relpath(document.video, output.parent),
                "split": document.split,
                "sourceGroup": document.source_group,
                "environment": document.environment,
                "game": recording.get("game", {}),
                "consent": {
                    "analyze": True,
                    "train": document.split in {"train", "validation"},
                },
                "capture": recording.get("capture", {}),
                "roi": recording.get("roi"),
                "rallies": document.payload["rallies"],
                "ignoredIntervals": document.payload.get("ignoredIntervals", []),
                "hardNegatives": document.payload.get("hardNegatives", []),
                "annotation": document.payload["annotation"],
            }
        )
    payload = {
        "schemaVersion": 1,
        "name": name,
        "annotationPolicy": {
            "id": ANNOTATION_POLICY_ID,
            "rallyStart": "serve-ball contact",
            "rallyEnd": "first instant live play has ended",
            "intervalConvention": "half-open [start,end) seconds on the normalized video",
        },
        "recordings": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".validate-label-manifest-",
        suffix=".json",
        dir=output.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        load_manifest(temporary_name, require_videos=require_videos)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload
