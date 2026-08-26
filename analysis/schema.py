from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA_VERSION = 1
ANNOTATION_POLICY_ID = "serve-contact-to-dead-ball-v1"
SPLITS = {"train", "validation", "test", "challenge"}
ENVIRONMENTS = {"indoor", "beach", "grass", "broadcast", "unknown"}


class ManifestError(ValueError):
    """Raised when annotations or split provenance are unsafe to use."""


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"start": self.start, "end": self.end}
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload


@dataclass(frozen=True)
class Recording:
    id: str
    video: Path
    split: str
    source_group: str
    environment: str
    game: dict[str, Any]
    rallies: tuple[Interval, ...]
    ignored_intervals: tuple[Interval, ...]
    roi: tuple[float, float, float, float] | None
    capture: dict[str, Any]
    consent: dict[str, bool]
    content_sha256: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class DatasetManifest:
    path: Path
    name: str
    recordings: tuple[Recording, ...]
    raw: dict[str, Any]

    def for_split(self, split: str) -> tuple[Recording, ...]:
        return tuple(recording for recording in self.recordings if recording.split == split)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _read_intervals(
    value: Any,
    prefix: str,
    field: str = "rallies",
) -> tuple[Interval, ...]:
    if not isinstance(value, list):
        raise ManifestError(f"{prefix}.{field} must be an array")
    intervals: list[Interval] = []
    previous_end = -1.0
    for index, item in enumerate(value):
        where = f"{prefix}.{field}[{index}]"
        if not isinstance(item, dict) or not _is_number(item.get("start")) or not _is_number(item.get("end")):
            raise ManifestError(f"{where} must contain numeric start and end")
        start, end = float(item["start"]), float(item["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ManifestError(f"{where} must satisfy 0 <= start < end")
        if start < previous_end:
            raise ManifestError(f"{where} overlaps or is not ordered")
        raw_tags = item.get("tags", [])
        if (
            not isinstance(raw_tags, list)
            or any(not isinstance(tag, str) or not tag.strip() for tag in raw_tags)
        ):
            raise ManifestError(f"{where}.tags must be an array of non-empty strings")
        intervals.append(
            Interval(start=start, end=end, tags=tuple(dict.fromkeys(raw_tags)))
        )
        previous_end = end
    return tuple(intervals)


def _intervals_overlap(left: Iterable[Interval], right: Iterable[Interval]) -> bool:
    right_rows = tuple(right)
    return any(
        first.start < second.end and second.start < first.end
        for first in left
        for second in right_rows
    )


def _read_roi(value: Any, prefix: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ManifestError(f"{prefix}.roi must be an object")
    keys = ("x", "y", "width", "height")
    if any(not _is_number(value.get(key)) for key in keys):
        raise ManifestError(f"{prefix}.roi must contain numeric x, y, width and height")
    x, y, width, height = (float(value[key]) for key in keys)
    if (
        not all(math.isfinite(item) for item in (x, y, width, height))
        or x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > 1
        or y + height > 1
    ):
        raise ManifestError(f"{prefix}.roi must be a normalized rectangle inside the frame")
    return x, y, width, height


def _read_game(value: Any, prefix: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ManifestError(f"{prefix}.game must be an object")
    game = dict(value)
    players_per_team = game.get("playersPerTeam")
    if players_per_team is not None and (
        not isinstance(players_per_team, int)
        or isinstance(players_per_team, bool)
        or not 1 <= players_per_team <= 6
    ):
        raise ManifestError(f"{prefix}.game.playersPerTeam must be an integer from 1 to 6 or null")
    target_points = game.get("targetPoints")
    if target_points is not None and (
        not isinstance(target_points, int)
        or isinstance(target_points, bool)
        or not 1 <= target_points <= 100
    ):
        raise ManifestError(f"{prefix}.game.targetPoints must be an integer from 1 to 100 or null")
    for key in ("format", "scoringRule"):
        item = game.get(key)
        if item is not None and (not isinstance(item, str) or not item.strip()):
            raise ManifestError(f"{prefix}.game.{key} must be a non-empty string or null")
    return game


def _provenance_digests(video: Path, prefix: str) -> tuple[str | None, str | None]:
    sidecar = video.with_suffix(video.suffix + ".provenance.json")
    if not sidecar.is_file():
        return None, None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        normalized = payload["normalized"]["sha256"]
        normalized_size = payload["normalized"]["sizeBytes"]
        source = payload["source"]["sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ManifestError(f"{prefix}: invalid normalization provenance {sidecar}: {error}") from error
    for label, digest in (("normalized", normalized), ("source", source)):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise ManifestError(f"{prefix}: invalid {label} SHA-256 in {sidecar}")
    if not isinstance(normalized_size, int) or normalized_size != video.stat().st_size:
        raise ManifestError(f"{prefix}: normalization provenance size does not match {video}")
    return normalized.lower(), source.lower()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path, *, require_videos: bool = True) -> DatasetManifest:
    manifest_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read manifest {manifest_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ManifestError("manifest root must be an object")
    if raw.get("schemaVersion") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(f"manifest schemaVersion must be {MANIFEST_SCHEMA_VERSION}")
    annotation_policy = raw.get("annotationPolicy")
    if not isinstance(annotation_policy, dict) or annotation_policy.get("id") != ANNOTATION_POLICY_ID:
        raise ManifestError(
            f"manifest annotationPolicy.id must be {ANNOTATION_POLICY_ID!r}"
        )
    rows = raw.get("recordings")
    if not isinstance(rows, list) or not rows:
        raise ManifestError("manifest recordings must be a non-empty array")

    recordings: list[Recording] = []
    ids: set[str] = set()
    groups: dict[str, str] = {}
    resolved_videos: dict[Path, tuple[str, str]] = {}
    normalized_digests: dict[str, str] = {}
    for index, row in enumerate(rows):
        prefix = f"recordings[{index}]"
        if not isinstance(row, dict):
            raise ManifestError(f"{prefix} must be an object")
        recording_id = row.get("id")
        if not isinstance(recording_id, str) or not recording_id.strip():
            raise ManifestError(f"{prefix}.id must be a non-empty string")
        if recording_id != recording_id.strip():
            raise ManifestError(f"{prefix}.id must not have leading or trailing whitespace")
        if recording_id in ids:
            raise ManifestError(f"duplicate recording id: {recording_id}")
        ids.add(recording_id)

        video_value = row.get("video")
        if not isinstance(video_value, str) or not video_value:
            raise ManifestError(f"{prefix}.video must be a non-empty path string")
        video = Path(video_value).expanduser()
        if not video.is_absolute():
            video = (manifest_path.parent / video).resolve()
        else:
            video = video.resolve()
        if require_videos and not video.is_file():
            raise ManifestError(f"{prefix}.video does not exist: {video}")
        content_sha256: str | None = None
        if require_videos:
            claimed_digest, _ = _provenance_digests(video, prefix)
            normalized_digest = _sha256_file(video)
            if claimed_digest is not None and claimed_digest != normalized_digest:
                raise ManifestError(f"{prefix}: normalized video SHA-256 does not match its provenance")
            content_sha256 = normalized_digest
            previous_id = normalized_digests.setdefault(normalized_digest, recording_id)
            if previous_id != recording_id:
                raise ManifestError(
                    f"normalized video content is duplicated by {previous_id!r} and {recording_id!r}"
                )

        split = row.get("split")
        if split not in SPLITS:
            raise ManifestError(f"{prefix}.split must be one of {sorted(SPLITS)}")
        source_group = row.get("sourceGroup")
        if not isinstance(source_group, str) or not source_group.strip():
            raise ManifestError(f"{prefix}.sourceGroup is required to prevent split leakage")
        if source_group != source_group.strip():
            raise ManifestError(f"{prefix}.sourceGroup must not have leading or trailing whitespace")
        old_split = groups.setdefault(source_group, split)
        if old_split != split:
            raise ManifestError(
                f"sourceGroup {source_group!r} crosses {old_split!r} and {split!r} splits"
            )
        previous = resolved_videos.setdefault(video, (split, recording_id))
        if previous != (split, recording_id):
            raise ManifestError(
                f"video {video} is reused by recordings {previous[1]!r} and {recording_id!r}"
            )

        environment = row.get("environment", "unknown")
        if environment not in ENVIRONMENTS:
            raise ManifestError(f"{prefix}.environment must be one of {sorted(ENVIRONMENTS)}")
        game = _read_game(row.get("game"), prefix)
        rallies = _read_intervals(row.get("rallies"), prefix)
        ignored_intervals = _read_intervals(
            row.get("ignoredIntervals", []),
            prefix,
            "ignoredIntervals",
        )
        roi = _read_roi(row.get("roi"), prefix)
        capture = row.get("capture", {})
        consent = row.get("consent", {})
        if not isinstance(capture, dict) or not isinstance(consent, dict):
            raise ManifestError(f"{prefix}.capture and consent must be objects")
        if consent.get("analyze") is not True:
            raise ManifestError(f"{prefix}.consent.analyze must be true")
        if split in {"train", "validation"} and consent.get("train") is not True:
            raise ManifestError(f"{prefix}.consent.train must be true for {split} data")

        recordings.append(
            Recording(
                id=recording_id,
                video=video,
                split=split,
                source_group=source_group,
                environment=environment,
                game=game,
                rallies=rallies,
                ignored_intervals=ignored_intervals,
                roi=roi,
                capture=dict(capture),
                consent={str(key): bool(item) for key, item in consent.items()},
                content_sha256=content_sha256,
                raw=dict(row),
            )
        )

    return DatasetManifest(
        path=manifest_path,
        name=str(raw.get("name", manifest_path.stem)),
        recordings=tuple(recordings),
        raw=raw,
    )


def labels_for_times(times: "Any", intervals: Iterable[Interval]) -> "Any":
    """Return float labels using half-open [start, end) annotation intervals."""
    import numpy as np

    result = np.zeros(len(times), dtype=np.float32)
    for interval in intervals:
        result[(times >= interval.start) & (times < interval.end)] = 1.0
    return result


def mask_for_times(times: "Any", ignored_intervals: Iterable[Interval]) -> "Any":
    """Return a boolean mask that excludes ambiguous/censored half-open intervals."""
    import numpy as np

    result = np.ones(len(times), dtype=np.bool_)
    for interval in ignored_intervals:
        result[(times >= interval.start) & (times < interval.end)] = False
    return result


def manifest_warnings(manifest: DatasetManifest) -> list[str]:
    warnings: list[str] = []
    present = {recording.split for recording in manifest.recordings}
    for split in ("train", "validation", "test"):
        if split not in present:
            warnings.append(f"no recordings assigned to the {split!r} split")
    for recording in manifest.recordings:
        if recording.roi is None:
            warnings.append(f"{recording.id}: no court ROI; the full frame will be analyzed")
        if not recording.capture:
            warnings.append(f"{recording.id}: capture-placement metadata is missing")
        else:
            if recording.capture.get("stationary") is not True:
                warnings.append(f"{recording.id}: camera is not confirmed stationary")
            if recording.capture.get("fullCourtVisible") is not True:
                warnings.append(f"{recording.id}: full court is not confirmed visible")
            if recording.capture.get("serviceAreasVisible") is not True:
                warnings.append(f"{recording.id}: both service areas are not confirmed visible")
            if recording.capture.get("position") != "centered-behind-endline":
                warnings.append(f"{recording.id}: capture is outside the supported centered end-line view")
        if recording.game.get("playersPerTeam") is None:
            warnings.append(f"{recording.id}: players per team are unknown")
        if recording.game.get("targetPoints") is None:
            warnings.append(f"{recording.id}: game target points are unknown")
        if recording.ignored_intervals:
            warnings.append(
                f"{recording.id}: {len(recording.ignored_intervals)} ambiguous/censored intervals "
                "will be excluded from fitting and scoring"
            )
        if not recording.rallies:
            warnings.append(f"{recording.id}: zero rallies (valid only for an intentional hard negative)")
    return warnings
