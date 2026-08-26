from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
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
    "celebration-huddle",
    "foreground-crossing",
    "model-false-positive",
    "random-dead-control",
    "setup-between-points",
    "timeout",
    "walking-ball-retrieval",
    "warmup",
    "other",
}
TERMINAL_CUES = {
    "ball-down-or-out",
    "whistle-or-stoppage",
    "no-recovery",
    "unobservable",
}
END_OBSERVABILITY_VALUES = {
    "observable",
    "partially-observable",
    "unobservable",
}
COURT_CORNER_NAMES = ("nearLeft", "nearRight", "farLeft", "farRight")
NET_ANCHOR_NAMES = ("left", "right")
SERVICE_ZONE_ANCHOR_NAMES = ("near", "far")
PLAYER_TRACKLET_WINDOWS = {"serve", "rally-end"}
PLAYER_TEAMS = {"team-a", "team-b", "unknown"}
PLAYER_COURT_SIDES = {"near", "far", "outside", "unknown"}
PLAYER_STATES = {"ready", "playing", "jumping", "stand-down", "walking"}
SERVING_SIDES = {"near", "far", "review"}


@dataclass(frozen=True)
class ServeMarker:
    time: float
    side: str
    notes: str | None
    origin: str | None
    model_side: str | None
    model_confidence: float | None
    model_id: str | None
    rally_id: str | None


@dataclass(frozen=True)
class SideSwitch:
    time: float
    notes: str | None
    origin: str | None
    model_confidence: float | None
    model_id: str | None
    model_event_id: str | None


@dataclass(frozen=True)
class RallyTransitionAnnotation:
    receiver_reaction_time: float | None
    collective_stand_down_time: float | None
    terminal_cue: str | None
    end_observability: str | None
    start_confidence: float | None
    end_confidence: float | None
    verified_immediate_result: bool | None


@dataclass(frozen=True)
class PlayerTrackletObservation:
    time: float
    footpoint: dict[str, float] | None
    box: dict[str, float] | None
    state: str | None


@dataclass(frozen=True)
class PlayerTracklet:
    rally_index: int
    track_id: str
    window: str
    team: str
    court_side: str
    observations: tuple[PlayerTrackletObservation, ...]
    notes: str | None


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
    serve_markers: tuple[ServeMarker, ...]
    side_switches: tuple[SideSwitch, ...]
    court_geometry: dict[str, Any] | None
    rally_transitions: tuple[RallyTransitionAnnotation, ...]
    player_tracklets: tuple[PlayerTracklet, ...]
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


def _optional_model_metadata(
    row: dict[str, Any], where: str
) -> tuple[str | None, float | None, str | None]:
    origin = row.get("origin")
    if origin is not None and origin not in {"model", "manual"}:
        raise ManifestError(f"{where}.origin must be 'model' or 'manual'")
    confidence = row.get("modelConfidence")
    if confidence is not None and (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not math.isfinite(float(confidence))
        or not 0 <= float(confidence) <= 1
    ):
        raise ManifestError(f"{where}.modelConfidence must be in [0, 1]")
    model_id = row.get("modelId")
    if model_id is not None and not isinstance(model_id, str):
        raise ManifestError(f"{where}.modelId must be a string")
    return origin, float(confidence) if confidence is not None else None, model_id


def _read_serve_markers(value: Any, duration: float) -> tuple[ServeMarker, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ManifestError("serveMarkers must be an array")
    markers: list[ServeMarker] = []
    previous_time = -1.0
    for index, row in enumerate(value):
        where = f"serveMarkers[{index}]"
        if not isinstance(row, dict):
            raise ManifestError(f"{where} must be an object")
        raw_time = row.get("time")
        if (
            not isinstance(raw_time, (int, float))
            or isinstance(raw_time, bool)
            or not math.isfinite(float(raw_time))
        ):
            raise ManifestError(f"{where}.time must be finite")
        marker_time = float(raw_time)
        if marker_time < 0 or marker_time > duration or marker_time <= previous_time:
            raise ManifestError(
                "serveMarkers must be strictly ordered points within the recording duration"
            )
        side = row.get("side")
        if side not in SERVING_SIDES:
            raise ManifestError(f"{where}.side must be one of {sorted(SERVING_SIDES)}")
        model_side = row.get("modelSide")
        if model_side is not None and model_side not in SERVING_SIDES:
            raise ManifestError(f"{where}.modelSide must be one of {sorted(SERVING_SIDES)}")
        notes = row.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise ManifestError(f"{where}.notes must be a string when present")
        rally_id = row.get("rallyId")
        if rally_id is not None and not isinstance(rally_id, str):
            raise ManifestError(f"{where}.rallyId must be a string")
        origin, confidence, model_id = _optional_model_metadata(row, where)
        markers.append(
            ServeMarker(
                time=marker_time,
                side=str(side),
                notes=notes,
                origin=origin,
                model_side=str(model_side) if model_side is not None else None,
                model_confidence=confidence,
                model_id=model_id,
                rally_id=rally_id,
            )
        )
        previous_time = marker_time
    return tuple(markers)


def _read_side_switches(value: Any, duration: float) -> tuple[SideSwitch, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ManifestError("sideSwitches must be an array")
    markers: list[SideSwitch] = []
    previous_time = -1.0
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ManifestError(f"sideSwitches[{index}] must be an object")
        raw_time = row.get("time")
        if (
            not isinstance(raw_time, (int, float))
            or isinstance(raw_time, bool)
            or not math.isfinite(float(raw_time))
        ):
            raise ManifestError(f"sideSwitches[{index}].time must be finite")
        marker_time = float(raw_time)
        if marker_time < 0 or marker_time > duration or marker_time <= previous_time:
            raise ManifestError(
                "sideSwitches must be strictly ordered points within the recording duration"
            )
        notes = row.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise ManifestError(f"sideSwitches[{index}].notes must be a string when present")
        origin, confidence, model_id = _optional_model_metadata(
            row, f"sideSwitches[{index}]"
        )
        model_event_id = row.get("modelEventId")
        if model_event_id is not None and not isinstance(model_event_id, str):
            raise ManifestError(
                f"sideSwitches[{index}].modelEventId must be a string"
            )
        markers.append(
            SideSwitch(
                time=marker_time,
                notes=notes,
                origin=origin,
                model_confidence=confidence,
                model_id=model_id,
                model_event_id=model_event_id,
            )
        )
        previous_time = marker_time
    return tuple(markers)


def _read_normalized_point(value: Any, where: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ManifestError(f"{where} must be an object")
    if set(value) != {"x", "y"}:
        raise ManifestError(f"{where} must contain only x and y")
    coordinates: dict[str, float] = {}
    for axis in ("x", "y"):
        raw = value.get(axis)
        if (
            not isinstance(raw, (int, float))
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
            or not 0 <= float(raw) <= 1
        ):
            raise ManifestError(f"{where}.{axis} must be a finite normalized coordinate")
        coordinates[axis] = float(raw)
    return coordinates


def _read_normalized_box(value: Any, where: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != {"x", "y", "width", "height"}:
        raise ManifestError(f"{where} must contain only x, y, width, and height")
    result: dict[str, float] = {}
    for field in ("x", "y", "width", "height"):
        raw = value[field]
        if (
            not isinstance(raw, (int, float))
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
        ):
            raise ManifestError(f"{where}.{field} must be finite")
        result[field] = float(raw)
    if (
        result["x"] < 0
        or result["y"] < 0
        or result["width"] <= 0
        or result["height"] <= 0
        or result["x"] + result["width"] > 1
        or result["y"] + result["height"] > 1
    ):
        raise ManifestError(f"{where} must be a positive normalized frame box")
    return result


def _read_player_tracklets(
    value: Any,
    rallies: Sequence[Interval],
    duration: float,
) -> tuple[PlayerTracklet, ...]:
    if not isinstance(value, list) or len(value) != len(rallies):
        raise ManifestError("rallies must remain an array while reading player tracklets")
    tracklets: list[PlayerTracklet] = []
    for rally_index, (row, rally) in enumerate(zip(value, rallies, strict=True)):
        if not isinstance(row, dict):
            raise ManifestError(f"rallies[{rally_index}] must be an object")
        raw_tracklets = row.get("playerTracklets", [])
        if not isinstance(raw_tracklets, list):
            raise ManifestError(f"rallies[{rally_index}].playerTracklets must be an array")
        seen_keys: set[tuple[str, str]] = set()
        for tracklet_index, raw_tracklet in enumerate(raw_tracklets):
            where = f"rallies[{rally_index}].playerTracklets[{tracklet_index}]"
            if not isinstance(raw_tracklet, dict):
                raise ManifestError(f"{where} must be an object")
            allowed = {"trackId", "window", "team", "courtSide", "observations", "notes"}
            unknown = set(raw_tracklet) - allowed
            if unknown:
                raise ManifestError(f"{where} has unrecognized fields: {sorted(unknown)}")
            track_id = raw_tracklet.get("trackId")
            if (
                not isinstance(track_id, str)
                or track_id != track_id.strip()
                or re.fullmatch(r"[A-Z]{0,2}[0-9]{1,3}", track_id) is None
            ):
                raise ManifestError(
                    f"{where}.trackId must be an anonymous token such as P1 or A02"
                )
            window = raw_tracklet.get("window")
            if window not in PLAYER_TRACKLET_WINDOWS:
                raise ManifestError(
                    f"{where}.window must be one of {sorted(PLAYER_TRACKLET_WINDOWS)}"
                )
            team = raw_tracklet.get("team")
            if team not in PLAYER_TEAMS:
                raise ManifestError(f"{where}.team must be one of {sorted(PLAYER_TEAMS)}")
            court_side = raw_tracklet.get("courtSide")
            if court_side not in PLAYER_COURT_SIDES:
                raise ManifestError(
                    f"{where}.courtSide must be one of {sorted(PLAYER_COURT_SIDES)}"
                )
            notes = raw_tracklet.get("notes")
            if notes is not None and not isinstance(notes, str):
                raise ManifestError(f"{where}.notes must be a string when present")
            unique_key = (str(window), track_id)
            if unique_key in seen_keys:
                raise ManifestError(
                    f"{where} duplicates track {track_id!r} in the same boundary window"
                )
            seen_keys.add(unique_key)
            raw_observations = raw_tracklet.get("observations")
            if not isinstance(raw_observations, list) or not raw_observations:
                raise ManifestError(
                    f"{where}.observations must contain at least one labeled frame"
                )
            if window == "serve":
                window_start = max(0.0, rally.start - 2.0)
                window_end = min(duration, rally.start + 3.0)
            else:
                window_start = max(0.0, rally.end - 3.0)
                window_end = min(duration, rally.end + 2.0)
            observations: list[PlayerTrackletObservation] = []
            previous_time = -1.0
            for observation_index, raw_observation in enumerate(raw_observations):
                observation_where = f"{where}.observations[{observation_index}]"
                if not isinstance(raw_observation, dict):
                    raise ManifestError(f"{observation_where} must be an object")
                observation_unknown = set(raw_observation) - {
                    "time",
                    "footpoint",
                    "box",
                    "state",
                }
                if observation_unknown:
                    raise ManifestError(
                        f"{observation_where} has unrecognized fields: "
                        f"{sorted(observation_unknown)}"
                    )
                observation_time = _optional_finite_number(
                    raw_observation.get("time"), f"{observation_where}.time"
                )
                if (
                    observation_time is None
                    or observation_time < window_start
                    or observation_time > window_end
                    or observation_time <= previous_time
                ):
                    raise ManifestError(
                        f"{observation_where}.time must be strictly ordered inside "
                        "its boundary window"
                    )
                previous_time = observation_time
                footpoint = (
                    _read_normalized_point(
                        raw_observation["footpoint"], f"{observation_where}.footpoint"
                    )
                    if "footpoint" in raw_observation
                    else None
                )
                box = (
                    _read_normalized_box(
                        raw_observation["box"], f"{observation_where}.box"
                    )
                    if "box" in raw_observation
                    else None
                )
                if footpoint is None and box is None:
                    raise ManifestError(
                        f"{observation_where} must contain a footpoint or box"
                    )
                state = raw_observation.get("state")
                if state is not None and state not in PLAYER_STATES:
                    raise ManifestError(
                        f"{observation_where}.state must be one of {sorted(PLAYER_STATES)}"
                    )
                observations.append(
                    PlayerTrackletObservation(
                        time=observation_time,
                        footpoint=footpoint,
                        box=box,
                        state=state,
                    )
                )
            tracklets.append(
                PlayerTracklet(
                    rally_index=rally_index,
                    track_id=track_id,
                    window=str(window),
                    team=str(team),
                    court_side=str(court_side),
                    observations=tuple(observations),
                    notes=notes,
                )
            )
    return tuple(tracklets)


def _read_point_group(
    value: Any,
    where: str,
    names: tuple[str, ...],
    *,
    require_pair: bool,
) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict):
        raise ManifestError(f"{where} must be an object")
    unknown = set(value) - set(names)
    if unknown:
        raise ManifestError(f"{where} has unrecognized anchors: {sorted(unknown)}")
    points = {
        name: _read_normalized_point(value[name], f"{where}.{name}")
        for name in names
        if name in value
    }
    if require_pair and points and len(points) != len(names):
        raise ManifestError(f"{where} must include both {', '.join(names)} anchors when completed")
    return points


def _read_court_geometry(
    value: Any,
    *,
    require_complete: bool,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ManifestError("recording.courtGeometry must be an object")
    allowed = {"corners", "netAnchors", "serviceZoneAnchors"}
    unknown = set(value) - allowed
    if unknown:
        raise ManifestError(
            f"recording.courtGeometry has unrecognized fields: {sorted(unknown)}"
        )
    corners = _read_point_group(
        value.get("corners"),
        "recording.courtGeometry.corners",
        COURT_CORNER_NAMES,
        require_pair=require_complete,
    )
    if require_complete and len(corners) != len(COURT_CORNER_NAMES):
        raise ManifestError(
            "recording.courtGeometry.corners must include nearLeft, nearRight, "
            "farLeft, and farRight when completed"
        )
    result: dict[str, Any] = {"corners": corners}
    for field, names in (
        ("netAnchors", NET_ANCHOR_NAMES),
        ("serviceZoneAnchors", SERVICE_ZONE_ANCHOR_NAMES),
    ):
        if field in value:
            result[field] = _read_point_group(
                value[field],
                f"recording.courtGeometry.{field}",
                names,
                require_pair=require_complete,
            )
    return result


def _optional_finite_number(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ManifestError(f"{where} must be a finite number when present")
    return float(value)


def _read_rally_transitions(
    value: Any,
    rallies: Sequence[Interval],
    duration: float,
) -> tuple[RallyTransitionAnnotation, ...]:
    if not isinstance(value, list) or len(value) != len(rallies):
        raise ManifestError("rallies must remain an array while reading transition annotations")
    annotations: list[RallyTransitionAnnotation] = []
    for index, (row, rally) in enumerate(zip(value, rallies, strict=True)):
        if not isinstance(row, dict):
            raise ManifestError(f"rallies[{index}] must be an object")
        notes = row.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise ManifestError(f"rallies[{index}].notes must be a string when present")
        reaction = _optional_finite_number(
            row.get("receiverReactionTime"),
            f"rallies[{index}].receiverReactionTime",
        )
        if reaction is not None and not (
            rally.start <= reaction <= min(rally.end, rally.start + 5.0, duration)
        ):
            raise ManifestError(
                f"rallies[{index}].receiverReactionTime must be within five seconds "
                "after rally start"
            )
        stand_down = _optional_finite_number(
            row.get("collectiveStandDownTime"),
            f"rallies[{index}].collectiveStandDownTime",
        )
        if stand_down is not None and not (
            max(rally.start, rally.end - 5.0)
            <= stand_down
            <= min(duration, rally.end + 5.0)
        ):
            raise ManifestError(
                f"rallies[{index}].collectiveStandDownTime must be within five seconds "
                "of rally end"
            )
        if reaction is not None and stand_down is not None and reaction > stand_down:
            raise ManifestError(
                f"rallies[{index}].receiverReactionTime must not follow "
                "collectiveStandDownTime"
            )
        terminal_cue = row.get("terminalCue")
        if terminal_cue is not None and terminal_cue not in TERMINAL_CUES:
            raise ManifestError(
                f"rallies[{index}].terminalCue must be one of {sorted(TERMINAL_CUES)}"
            )
        end_observability = row.get("endObservability")
        if end_observability is not None and end_observability not in END_OBSERVABILITY_VALUES:
            raise ManifestError(
                f"rallies[{index}].endObservability must be one of "
                f"{sorted(END_OBSERVABILITY_VALUES)}"
            )
        confidence_values: dict[str, float | None] = {}
        for json_field, result_field in (
            ("startConfidence", "start_confidence"),
            ("endConfidence", "end_confidence"),
        ):
            confidence = _optional_finite_number(
                row.get(json_field), f"rallies[{index}].{json_field}"
            )
            if confidence is not None and not 0 <= confidence <= 1:
                raise ManifestError(f"rallies[{index}].{json_field} must be between 0 and 1")
            confidence_values[result_field] = confidence
        immediate = row.get("verifiedImmediateResult")
        if immediate is not None and not isinstance(immediate, bool):
            raise ManifestError(
                f"rallies[{index}].verifiedImmediateResult must be boolean when present"
            )
        annotations.append(
            RallyTransitionAnnotation(
                receiver_reaction_time=reaction,
                collective_stand_down_time=stand_down,
                terminal_cue=terminal_cue,
                end_observability=end_observability,
                start_confidence=confidence_values["start_confidence"],
                end_confidence=confidence_values["end_confidence"],
                verified_immediate_result=immediate,
            )
        )
    return tuple(annotations)


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
    rally_transitions = _read_rally_transitions(payload.get("rallies"), rallies, duration)
    player_tracklets = _read_player_tracklets(payload.get("rallies"), rallies, duration)
    ignored = _read_intervals(payload.get("ignoredIntervals", []), "labels", "ignoredIntervals")
    hard_negatives = _read_intervals(payload.get("hardNegatives", []), "labels", "hardNegatives")
    serve_markers = _read_serve_markers(payload.get("serveMarkers", []), duration)
    side_switches = _read_side_switches(payload.get("sideSwitches", []), duration)
    if require_complete and any(marker.side == "review" for marker in serve_markers):
        raise ManifestError(
            "completed labels must resolve every serving-side marker to near or far"
        )
    court_geometry = _read_court_geometry(
        recording.get("courtGeometry"),
        require_complete=require_complete,
    )
    _validate_bounds(rallies, duration, "rallies")
    _validate_bounds(ignored, duration, "ignoredIntervals")
    _validate_bounds(hard_negatives, duration, "hardNegatives")
    if require_complete:
        for index, (previous, current) in enumerate(
            zip(rallies, rallies[1:], strict=False),
            start=1,
        ):
            if current.start <= previous.end:
                raise ManifestError(
                    "completed rallies must have positive dead time between them "
                    f"(rallies {index} and {index + 1} touch at {current.start:.3f}s)"
                )
    if _intervals_overlap(rallies, hard_negatives):
        raise ManifestError("hardNegatives must not overlap rallies")
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
        serve_markers=serve_markers,
        side_switches=side_switches,
        court_geometry=court_geometry,
        rally_transitions=rally_transitions,
        player_tracklets=player_tracklets,
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
        "serveMarkers": [],
        "sideSwitches": [],
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
                **(
                    {"courtGeometry": recording["courtGeometry"]}
                    if "courtGeometry" in recording
                    else {}
                ),
                "rallies": document.payload["rallies"],
                "ignoredIntervals": document.payload.get("ignoredIntervals", []),
                "hardNegatives": document.payload.get("hardNegatives", []),
                "serveMarkers": document.payload.get("serveMarkers", []),
                "sideSwitches": document.payload.get("sideSwitches", []),
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


def freeze_label_snapshot(
    label_paths: Sequence[str | Path],
    destination: str | Path,
    *,
    annotator: str,
    drop_touching_duplicate_tails: bool = False,
    require_videos: bool = True,
) -> tuple[LabelDocument, ...]:
    """Freeze reviewed drafts as an immutable, fully validated snapshot directory."""
    if not label_paths:
        raise ManifestError("at least one reviewed label document is required")
    reviewer = _nonempty_string(annotator, "annotator")
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing snapshot: {output}")
    documents = [
        load_label_document(path, require_complete=False, require_video=require_videos)
        for path in label_paths
    ]
    ids: set[str] = set()
    for document in documents:
        if document.recording_id in ids:
            raise ManifestError(f"duplicate recording id: {document.recording_id}")
        ids.add(document.recording_id)
        if document.payload["annotation"].get("continuousVideoReviewed") is not True:
            raise ManifestError(
                f"{document.recording_id}: continuousVideoReviewed must be true before freezing"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent)
    )
    reviewed_at = datetime.now(timezone.utc).isoformat()
    frozen: list[LabelDocument] = []
    ledger_rows: list[dict[str, Any]] = []
    try:
        for document in sorted(documents, key=lambda item: item.recording_id):
            # JSON round-tripping makes an independent, JSON-safe copy without sharing nested state.
            payload = json.loads(json.dumps(document.payload, allow_nan=False))
            transformations: list[dict[str, Any]] = []
            if drop_touching_duplicate_tails:
                cleaned_rallies: list[dict[str, Any]] = []
                for source_index, rally in enumerate(payload["rallies"]):
                    previous = cleaned_rallies[-1] if cleaned_rallies else None
                    is_duplicate_tail = (
                        previous is not None
                        and abs(float(rally["start"]) - float(previous["end"])) < 1e-9
                        and rally.get("tags", []) == previous.get("tags", [])
                        and rally.get("notes") == previous.get("notes")
                    )
                    if is_duplicate_tail:
                        transformations.append(
                            {
                                "type": "drop-touching-duplicate-tail",
                                "sourceRallyIndex": source_index,
                                "removed": rally,
                                "reason": (
                                    "zero dead-time gap and metadata identical to the preceding "
                                    "rally; deterministic split-shortcut artifact"
                                ),
                            }
                        )
                    else:
                        cleaned_rallies.append(rally)
                payload["rallies"] = cleaned_rallies
            payload["recording"]["video"] = os.path.relpath(document.video, output)
            payload["annotation"] = {
                **payload["annotation"],
                "status": "complete",
                "annotator": reviewer,
                "continuousVideoReviewed": True,
                "reviewedAt": reviewed_at,
            }
            if payload["annotation"].get("notes") == (
                "Blind audiovisual AI prelabel. Validate every serve-contact and dead-ball "
                "boundary before marking this recording complete."
            ):
                payload["annotation"]["notes"] = (
                    "Human-verified from a blind audiovisual AI prelabel; per-rally AI "
                    "provenance tags are retained for comparison."
                )
            snapshot_path = staging / document.path.name
            atomic_write_text(
                snapshot_path,
                json.dumps(payload, indent=2, allow_nan=False) + "\n",
            )
            frozen.append(
                load_label_document(
                    snapshot_path,
                    require_complete=True,
                    require_video=require_videos,
                )
            )
            ledger_rows.append(
                {
                    "recordingId": document.recording_id,
                    "sourceDraft": str(document.path),
                    "sourceDraftSha256": hashlib.sha256(document.path.read_bytes()).hexdigest(),
                    "snapshotFile": snapshot_path.name,
                    "snapshotSha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                    "video": str(document.video),
                    "videoContentSha256": document.payload["recording"]["contentSha256"],
                    "transformations": transformations,
                }
            )
        ledger_path = staging / "snapshot.json"
        atomic_write_text(
            ledger_path,
            json.dumps(
                {
                    "schemaVersion": 1,
                    "kind": "volleycut-completed-label-snapshot",
                    "createdAt": reviewed_at,
                    "annotator": reviewer,
                    "recordings": ledger_rows,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
        )
        for artifact in staging.iterdir():
            artifact.chmod(0o444)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing snapshot: {output}")
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return tuple(
        load_label_document(
            output / document.path.name,
            require_complete=True,
            require_video=require_videos,
        )
        for document in frozen
    )
