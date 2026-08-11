from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping, Sequence

from .artifacts import atomic_write_text
from .ball_annotation import (
    BALL_ANNOTATION_TASK_TYPE,
    STRATA,
    assert_immutable_provenance_unchanged,
    validate_ball_annotation_task,
)


BALL_DETECTOR_EVALUATION_SCHEMA_VERSION = 1
BALL_DETECTOR_EVALUATION_KIND = "volleycut-ball-detector-development-evaluation"
SOL_REVIEW_PROVENANCE_SCHEMA_VERSION = 1
SOL_REVIEW_PROVENANCE_KIND = "volleycut-detector-blind-sol-ball-review"
SOL_PREPARATION_RECEIPT_KIND = "volleycut-sol-ball-review-preparation-receipt"
SIZE_BINS_PIXELS = (
    ("tiny_lt_8px", 0.0, 8.0),
    ("small_8_to_lt_16px", 8.0, 16.0),
    ("medium_16_to_lt_32px", 16.0, 32.0),
    ("large_ge_32px", 32.0, math.inf),
)


class BallDetectorEvaluationError(ValueError):
    """Raised when reviewed pilot tasks cannot support a valid evaluation."""


@dataclass(frozen=True)
class ProtocolRequirements:
    """Minimum independent coverage required to promote a development threshold."""

    min_recordings: int = 4
    min_source_groups: int = 3
    min_windows: int = 24
    min_reviewed_frames: int = 1080
    min_primary_positive_frames: int = 30
    min_any_ball_positive_frames: int = 30
    min_ball_free_frames: int = 100
    min_primary_positive_windows: int = 6
    min_any_ball_positive_windows: int = 6
    min_ball_free_windows: int = 6
    min_primary_positive_source_groups: int = 2
    min_ball_free_source_groups: int = 2

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0


@dataclass(frozen=True)
class TruthObject:
    bbox: Box
    role: str
    visibility: str
    truncated: bool


@dataclass(frozen=True)
class Detection:
    bbox: Box
    confidence: float


@dataclass(frozen=True)
class EvaluationFrame:
    key: str
    task_id: str
    recording_id: str
    source_group: str
    environment: str
    split: str
    window_key: str
    window_id: str
    stratum: str
    width: int
    height: int
    primary_ball_state: str
    truth_objects: tuple[TruthObject, ...]
    ball_presence_probability: float
    detections: tuple[Detection, ...]
    proposal_exposure: str = "not_shown"

    @property
    def primary_truth(self) -> tuple[TruthObject, ...]:
        return tuple(item for item in self.truth_objects if item.role == "primary-court")

    @property
    def any_ball_truth(self) -> tuple[TruthObject, ...]:
        return self.truth_objects


@dataclass(frozen=True)
class LoadedEvaluation:
    frames: tuple[EvaluationFrame, ...]
    tasks: tuple[dict[str, Any], ...]
    detector: dict[str, Any]
    detector_signature_sha256: str
    manifest_sha256: str
    exposure_summary: dict[str, int]


@dataclass(frozen=True)
class SolAnnotationFrame:
    key: str
    task_id: str
    primary_ball_state: str
    objects: tuple[TruthObject, ...]

    @property
    def primary_objects(self) -> tuple[TruthObject, ...]:
        return tuple(item for item in self.objects if item.role == "primary-court")


@dataclass(frozen=True)
class LoadedSolReviews:
    frames: tuple[SolAnnotationFrame, ...]
    tasks: tuple[dict[str, Any], ...]


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


def _valid_nonzero_sha256(value: Any) -> bool:
    return _valid_sha256(value) and value != "0" * 64


def _read_task_with_hash(path: Path) -> tuple[dict[str, Any], str]:
    try:
        encoded = path.read_bytes()
        task = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BallDetectorEvaluationError(f"cannot read annotation task {path}: {error}") from error
    if not isinstance(task, dict):
        raise BallDetectorEvaluationError(f"annotation task must be a JSON object: {path}")
    return task, hashlib.sha256(encoded).hexdigest()


def _proposal_exposure_status(annotation: Mapping[str, Any]) -> str:
    """Return the persisted human-review exposure class for one frame."""

    exposure = annotation.get("proposalExposure")
    if exposure is None:
        return "not_recorded"
    if exposure not in {"not_shown", "shown_before_label_finalized"}:
        raise BallDetectorEvaluationError(
            "annotation proposalExposure must be 'not_shown' or "
            "'shown_before_label_finalized'"
        )
    return str(exposure)


def _human_exposure_summary(task: Mapping[str, Any]) -> dict[str, int]:
    counts = {
        "not_shown": 0,
        "shown_before_label_finalized": 0,
        "not_recorded": 0,
    }
    for annotation in task["annotations"]["frames"].values():
        counts[_proposal_exposure_status(annotation)] += 1
    return counts


def _require_exact_mapping_keys(
    value: Any,
    expected: set[str],
    where: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise BallDetectorEvaluationError(
            f"{where} must contain exactly {sorted(expected)}"
        )
    return value


def _validate_completed_human_review_structure(task: Mapping[str, Any]) -> None:
    """Reject hidden proposal channels before issuing blind-merge provenance."""

    _require_exact_mapping_keys(
        task,
        {"schemaVersion", "taskType", "immutable", "suggestions", "annotations"},
        "reviewed-label task root",
    )
    _require_exact_mapping_keys(
        task["immutable"],
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
        "reviewed-label immutable root",
    )
    suggestions = _require_exact_mapping_keys(
        task["suggestions"],
        {"status", "model", "frames"},
        "reviewed-label suggestions",
    )
    if suggestions != {"status": "empty", "model": None, "frames": {}}:
        raise BallDetectorEvaluationError(
            "reviewed-label task must have exact empty suggestions"
        )
    annotations = _require_exact_mapping_keys(
        task["annotations"],
        {"review", "frames"},
        "reviewed-label annotations",
    )
    review = _require_exact_mapping_keys(
        annotations["review"],
        {"status", "annotator", "reviewedAt", "notes"},
        "reviewed-label review metadata",
    )
    if review["status"] != "complete":
        raise BallDetectorEvaluationError("reviewed-label task review must be complete")
    frames = annotations["frames"]
    if not isinstance(frames, dict):
        raise BallDetectorEvaluationError("reviewed-label frames must be an object")
    for frame_id, annotation in frames.items():
        frame_row = _require_exact_mapping_keys(
            annotation,
            {
                "status",
                "primaryBallState",
                "objects",
                "notes",
                "proposalExposure",
            },
            f"reviewed-label frame {frame_id!r}",
        )
        if frame_row["proposalExposure"] not in {
            "not_shown",
            "shown_before_label_finalized",
        }:
            raise BallDetectorEvaluationError(
                f"reviewed-label frame {frame_id!r} proposalExposure is invalid"
            )
        for object_index, object_row in enumerate(frame_row["objects"]):
            object_mapping = _require_exact_mapping_keys(
                object_row,
                {"id", "category", "role", "bbox", "visibility", "truncated"},
                f"reviewed-label frame {frame_id!r} object {object_index}",
            )
            _require_exact_mapping_keys(
                object_mapping["bbox"],
                {"x", "y", "width", "height"},
                f"reviewed-label frame {frame_id!r} object {object_index} bbox",
            )


def _validate_pristine_sol_source_structure(task: Mapping[str, Any]) -> None:
    if set(task) != {
        "schemaVersion",
        "taskType",
        "immutable",
        "suggestions",
        "annotations",
    }:
        raise BallDetectorEvaluationError(
            "Sol source task root must be canonical and contain no extra proposal channels"
        )
    immutable = task["immutable"]
    if set(immutable) != {
        "taskId",
        "digestSha256",
        "manifest",
        "recording",
        "source",
        "sampling",
        "annotationPolicy",
        "windows",
        "frames",
    }:
        raise BallDetectorEvaluationError(
            "Sol source immutable root has unsupported fields or proposal channels"
        )
    exact_mappings = (
        (
            immutable["manifest"],
            {"name", "filename", "pathHint", "sha256"},
            "immutable.manifest",
        ),
        (
            immutable["recording"],
            {"id", "split", "sourceGroup", "environment"},
            "immutable.recording",
        ),
        (
            immutable["source"],
            {"proxy", "normalizationProvenance"},
            "immutable.source",
        ),
        (
            immutable["source"]["proxy"],
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
        ),
        (
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
        ),
    )
    for value, allowed, where in exact_mappings:
        if not isinstance(value, dict) or set(value) != allowed:
            raise BallDetectorEvaluationError(f"Sol source {where} fields are not canonical")
    normalization = immutable["source"]["normalizationProvenance"]
    if normalization is not None:
        if not isinstance(normalization, dict) or set(normalization) != {
            "sidecarFilename",
            "sidecarSha256",
            "originalSource",
        }:
            raise BallDetectorEvaluationError(
                "Sol source normalization provenance fields are not canonical"
            )
        original = normalization["originalSource"]
        if not isinstance(original, dict) or set(original) != {
            "filename",
            "sizeBytes",
            "sha256",
        }:
            raise BallDetectorEvaluationError(
                "Sol source original-source provenance fields are not canonical"
            )
    reference_fields = {
        "rallyIndex",
        "rallyStartSeconds",
        "rallyEndSeconds",
        "intervalType",
        "hardNegativeIndex",
        "category",
        "intervalStartSeconds",
        "intervalEndSeconds",
        "motionMeanAbsDiff",
    }
    window_fields = {
        "id",
        "requestedStratum",
        "actualSource",
        "startSampleIndex",
        "startSeconds",
        "endSeconds",
        "centerSeconds",
        "reference",
    }
    for index, window in enumerate(immutable["windows"]):
        allowed_window_fields = window_fields | (
            {"fallbackReason"} if "fallbackReason" in window else set()
        )
        if not isinstance(window, dict) or set(window) != allowed_window_fields:
            raise BallDetectorEvaluationError(
                f"Sol source immutable.windows[{index}] fields are not canonical"
            )
        reference = window["reference"]
        if (
            not isinstance(reference, dict)
            or not set(reference).issubset(reference_fields)
            or any(isinstance(value, (dict, list)) for value in reference.values())
        ):
            raise BallDetectorEvaluationError(
                f"Sol source immutable.windows[{index}].reference is not canonical"
            )
    frame_fields = {
        "id",
        "windowId",
        "sampleOffset",
        "sourceFrameIndex",
        "sourceTimestampSeconds",
        "image",
    }
    image_fields = {"path", "sha256", "width", "height", "format"}
    for index, frame in enumerate(immutable["frames"]):
        if not isinstance(frame, dict) or set(frame) != frame_fields:
            raise BallDetectorEvaluationError(
                f"Sol source immutable.frames[{index}] fields are not canonical"
            )
        if not isinstance(frame["image"], dict) or set(frame["image"]) != image_fields:
            raise BallDetectorEvaluationError(
                f"Sol source immutable.frames[{index}].image fields are not canonical"
            )
    if set(task["suggestions"]) != {"status", "model", "frames"}:
        raise BallDetectorEvaluationError("Sol source suggestions fields are not canonical")
    annotations = task["annotations"]
    if set(annotations) != {"review", "frames"}:
        raise BallDetectorEvaluationError("Sol source annotations fields are not canonical")
    review = annotations["review"]
    if set(review) != {"status", "annotator", "reviewedAt", "notes"} or review[
        "notes"
    ] != "":
        raise BallDetectorEvaluationError(
            "Sol source review must have exact fields and blank notes"
        )
    for frame_id, annotation in annotations["frames"].items():
        if set(annotation) != {"status", "primaryBallState", "objects", "notes"}:
            raise BallDetectorEvaluationError(
                f"Sol source frame {frame_id} has non-canonical mutable fields"
            )
        if annotation["notes"] != "":
            raise BallDetectorEvaluationError(
                f"Sol source frame {frame_id} notes must be blank"
            )


def _validate_sol_completed_structure(task: Mapping[str, Any]) -> None:
    if set(task) != {
        "schemaVersion",
        "taskType",
        "immutable",
        "suggestions",
        "annotations",
        "solReviewProvenance",
    }:
        raise BallDetectorEvaluationError(
            "completed Sol task root has unsupported fields or proposal channels"
        )
    if set(task["suggestions"]) != {"status", "model", "frames"}:
        raise BallDetectorEvaluationError("completed Sol suggestions fields are not canonical")
    annotations = task["annotations"]
    if set(annotations) != {"review", "frames"} or set(annotations["review"]) != {
        "status",
        "annotator",
        "reviewedAt",
        "notes",
    }:
        raise BallDetectorEvaluationError("completed Sol annotations fields are not canonical")
    for frame_id, annotation in annotations["frames"].items():
        if set(annotation) != {"status", "primaryBallState", "objects", "notes"}:
            raise BallDetectorEvaluationError(
                f"completed Sol frame {frame_id} has unsupported fields"
            )


def _validate_initial_source_index_binding(
    task: Mapping[str, Any],
    task_sha256: str,
    pilot_index_path: str | Path,
    *,
    expected_index_sha256: str | None = None,
) -> dict[str, Any]:
    index_path = Path(pilot_index_path).expanduser().resolve()
    try:
        encoded = index_path.read_bytes()
        index = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BallDetectorEvaluationError(
            f"cannot read Sol preparation pilot index {index_path}: {error}"
        ) from error
    index_sha = hashlib.sha256(encoded).hexdigest()
    if expected_index_sha256 is not None and index_sha != expected_index_sha256:
        raise BallDetectorEvaluationError(
            "Sol preparation pilot index SHA-256 does not match its receipt"
        )
    if (
        not isinstance(index, dict)
        or index.get("schemaVersion") != 1
        or index.get("artifactType") != "volleycut-ball-presence-pilot-index"
        or index.get("developmentOnly") is not True
    ):
        raise BallDetectorEvaluationError("Sol preparation pilot index is invalid")
    for field in ("recordingCount", "windowCount", "frameCount"):
        if (
            not isinstance(index.get(field), int)
            or isinstance(index[field], bool)
            or index[field] < 1
        ):
            raise BallDetectorEvaluationError(
                f"Sol preparation pilot index {field} is invalid"
            )
    immutable = task["immutable"]
    manifest = index.get("manifest")
    if (
        not isinstance(manifest, dict)
        or manifest.get("sha256") != immutable["manifest"]["sha256"]
        or index.get("samplingPolicyId") != immutable["sampling"]["policyId"]
        or index.get("round") != immutable["sampling"]["round"]
    ):
        raise BallDetectorEvaluationError(
            "Sol source manifest/sampling provenance differs from the pilot index"
        )
    rows = index.get("tasks")
    if not isinstance(rows, list):
        raise BallDetectorEvaluationError("Sol preparation pilot index tasks are invalid")
    normalized_rows: list[Mapping[str, Any]] = []
    recording_ids: set[str] = set()
    task_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BallDetectorEvaluationError(
                f"Sol preparation pilot index tasks[{row_index}] is invalid"
            )
        required_row = {
            "recordingId",
            "taskId",
            "frameCount",
            "initialTaskSha256",
        }
        if not required_row.issubset(row):
            raise BallDetectorEvaluationError(
                f"Sol preparation pilot index tasks[{row_index}] lacks binding fields"
            )
        if (
            not isinstance(row["recordingId"], str)
            or not row["recordingId"]
            or row["recordingId"] in recording_ids
            or not isinstance(row["taskId"], str)
            or not row["taskId"]
            or row["taskId"] in task_ids
            or not isinstance(row["frameCount"], int)
            or isinstance(row["frameCount"], bool)
            or row["frameCount"] < 1
            or not _valid_nonzero_sha256(row["initialTaskSha256"])
            or row.get("split", "train") not in {"train", "validation"}
        ):
            raise BallDetectorEvaluationError(
                f"Sol preparation pilot index tasks[{row_index}] provenance is invalid"
            )
        recording_ids.add(row["recordingId"])
        task_ids.add(row["taskId"])
        normalized_rows.append(row)
    if (
        len(normalized_rows) != index["recordingCount"]
        or sum(int(row["frameCount"]) for row in normalized_rows)
        != index["frameCount"]
        or index["windowCount"] != index["recordingCount"] * len(STRATA)
    ):
        raise BallDetectorEvaluationError(
            "Sol preparation pilot index aggregate counts are inconsistent"
        )
    recording_id = immutable["recording"]["id"]
    matches = [
        row
        for row in normalized_rows
        if row.get("recordingId") == recording_id
    ]
    if len(matches) != 1:
        raise BallDetectorEvaluationError(
            "Sol source recording must occur exactly once in the pilot index"
        )
    row = matches[0]
    expected = {
        "taskId": immutable["taskId"],
        "frameCount": len(immutable["frames"]),
        "initialTaskSha256": task_sha256,
    }
    mismatches = [field for field, value in expected.items() if row.get(field) != value]
    if row.get("split", immutable["recording"]["split"]) != immutable["recording"][
        "split"
    ]:
        mismatches.append("split")
    if mismatches:
        raise BallDetectorEvaluationError(
            f"Sol source does not exactly match its indexed initial task: {mismatches}"
        )
    return {
        "pathHint": str(index_path),
        "sha256": index_sha,
        "recordingId": recording_id,
        "taskId": immutable["taskId"],
        "initialTaskSha256": task_sha256,
        "samplingPolicyId": index["samplingPolicyId"],
        "round": index["round"],
    }


def _sol_preparation_receipt_path(task_path: Path) -> Path:
    return task_path.with_name(f"{task_path.name}.preparation-receipt.json")


def _validate_blind_merge_provenance(
    task: Mapping[str, Any],
    merged_path: Path,
) -> dict[str, Any]:
    provenance = task.get("blindMergeProvenance")
    required = {
        "schemaVersion",
        "kind",
        "createdAt",
        "reviewedLabels",
        "detectorProposals",
        "humanReviewExposure",
        "immutableDigestSha256",
        "implementationSha256",
    }
    if not isinstance(provenance, dict) or set(provenance) != required:
        raise BallDetectorEvaluationError(
            f"{merged_path} must contain exact blindMergeProvenance fields"
        )
    if provenance["schemaVersion"] != 1:
        raise BallDetectorEvaluationError("blindMergeProvenance schemaVersion must be 1")
    if provenance["kind"] != "volleycut-ball-presence-blind-review-merge":
        raise BallDetectorEvaluationError("blindMergeProvenance kind is invalid")
    if provenance["immutableDigestSha256"] != task["immutable"]["digestSha256"]:
        raise BallDetectorEvaluationError(
            "blindMergeProvenance immutable digest does not match the merged task"
        )
    if not _valid_sha256(provenance["implementationSha256"]):
        raise BallDetectorEvaluationError(
            "blindMergeProvenance implementationSha256 is invalid"
        )
    created_at = provenance["createdAt"]
    if not isinstance(created_at, str):
        raise BallDetectorEvaluationError("blindMergeProvenance createdAt is invalid")
    try:
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise BallDetectorEvaluationError(
            "blindMergeProvenance createdAt must be ISO-8601"
        ) from error
    if parsed_created_at.tzinfo is None:
        raise BallDetectorEvaluationError(
            "blindMergeProvenance createdAt must include a timezone"
        )

    source_specs = (
        (
            "reviewedLabels",
            "complete annotations with empty suggestions",
            "complete",
            "empty",
        ),
        (
            "detectorProposals",
            "complete suggestions with unreviewed annotations",
            "unreviewed",
            "complete",
        ),
    )
    source_tasks: dict[str, dict[str, Any]] = {}
    source_summary: dict[str, Any] = {}
    for name, required_state, review_status, suggestion_status in source_specs:
        source = provenance[name]
        if not isinstance(source, dict) or set(source) != {
            "pathHint",
            "sha256",
            "requiredState",
        }:
            raise BallDetectorEvaluationError(
                f"blindMergeProvenance.{name} fields are invalid"
            )
        if source["requiredState"] != required_state:
            raise BallDetectorEvaluationError(
                f"blindMergeProvenance.{name}.requiredState is invalid"
            )
        if not _valid_sha256(source["sha256"]):
            raise BallDetectorEvaluationError(
                f"blindMergeProvenance.{name}.sha256 is invalid"
            )
        if not isinstance(source["pathHint"], str) or not source["pathHint"]:
            raise BallDetectorEvaluationError(
                f"blindMergeProvenance.{name}.pathHint is invalid"
            )
        source_path = Path(source["pathHint"]).expanduser().resolve()
        if source_path == merged_path:
            raise BallDetectorEvaluationError(
                f"blindMergeProvenance.{name} cannot point to the merged task"
            )
        source_raw, source_sha = _read_task_with_hash(source_path)
        if source_sha != source["sha256"]:
            raise BallDetectorEvaluationError(
                f"blindMergeProvenance.{name} source SHA-256 does not match"
            )
        try:
            validated = validate_ball_annotation_task(
                source_raw,
                task_path=source_path,
                verify_images=True,
            )
        except ValueError as error:
            raise BallDetectorEvaluationError(
                f"blindMergeProvenance.{name} source task is invalid: {error}"
            ) from error
        if name == "reviewedLabels":
            _validate_completed_human_review_structure(validated)
        if validated["annotations"]["review"]["status"] != review_status:
            raise BallDetectorEvaluationError(
                f"blindMergeProvenance.{name} review state changed"
            )
        if validated["suggestions"]["status"] != suggestion_status:
            raise BallDetectorEvaluationError(
                f"blindMergeProvenance.{name} suggestion state changed"
            )
        try:
            assert_immutable_provenance_unchanged(task, validated)
        except ValueError as error:
            raise BallDetectorEvaluationError(
                f"blindMergeProvenance.{name} immutable provenance differs: {error}"
            ) from error
        source_tasks[name] = validated
        source_summary[name] = {
            "pathHint": str(source_path),
            "sha256": source_sha,
            "requiredState": required_state,
        }

    if source_tasks["reviewedLabels"]["annotations"] != task["annotations"]:
        raise BallDetectorEvaluationError(
            "merged annotations do not exactly match the SHA-pinned reviewed-label source"
        )
    if source_tasks["detectorProposals"]["suggestions"] != task["suggestions"]:
        raise BallDetectorEvaluationError(
            "merged suggestions do not exactly match the SHA-pinned detector-proposal source"
        )
    exposure_counts = _human_exposure_summary(source_tasks["reviewedLabels"])
    expected_exposure = {
        "field": "annotations.frames[*].proposalExposure",
        "notShownFrameCount": exposure_counts["not_shown"],
        "shownBeforeLabelFinalizedFrameCount": exposure_counts[
            "shown_before_label_finalized"
        ],
        "notRecordedFrameCount": exposure_counts["not_recorded"],
        "qualityEligibleFrameCount": exposure_counts["not_shown"],
        "blanketBlindnessClaim": False,
    }
    if provenance["humanReviewExposure"] != expected_exposure:
        raise BallDetectorEvaluationError(
            "blindMergeProvenance humanReviewExposure does not match reviewed labels"
        )
    model = source_tasks["detectorProposals"]["suggestions"]["model"]
    source_task = model.get("sourceTask") if isinstance(model, dict) else None
    if (
        not isinstance(source_task, dict)
        or not _valid_sha256(source_task.get("sha256"))
        or not isinstance(source_task.get("pathHint"), str)
        or not source_task["pathHint"]
    ):
        raise BallDetectorEvaluationError(
            "detector proposal model must retain its initial sourceTask path and SHA-256"
        )
    initial_path = Path(source_task["pathHint"]).expanduser().resolve()
    initial_file_verified = False
    if initial_path.is_file():
        initial_sha = _sha256_file(initial_path)
        if initial_sha != source_task["sha256"]:
            raise BallDetectorEvaluationError(
                "detector model sourceTask file exists but no longer matches its SHA-256"
            )
        initial_file_verified = True
    return {
        "schemaVersion": provenance["schemaVersion"],
        "kind": provenance["kind"],
        "createdAt": provenance["createdAt"],
        "immutableDigestSha256": provenance["immutableDigestSha256"],
        "implementationSha256": provenance["implementationSha256"],
        "humanReviewExposure": copy.deepcopy(expected_exposure),
        **source_summary,
        "initialTask": {
            "pathHint": str(initial_path),
            "sha256": source_task["sha256"],
            "fileVerified": initial_file_verified,
        },
    }


def _validate_sol_review_provenance(
    task: Mapping[str, Any],
    task_path: Path,
    *,
    required_review_status: str,
) -> dict[str, Any]:
    """Verify that a Sol label file descends from a detector-empty source task."""

    provenance = task.get("solReviewProvenance")
    required = {
        "schemaVersion",
        "kind",
        "preparedAt",
        "sourceTask",
        "preparationReceipt",
        "reviewer",
        "detectorSuggestionsAbsent",
        "immutableDigestSha256",
        "implementationSha256",
    }
    if not isinstance(provenance, dict) or set(provenance) != required:
        raise BallDetectorEvaluationError(
            f"{task_path} must contain exact solReviewProvenance fields"
        )
    _validate_sol_completed_structure(task)
    if provenance["schemaVersion"] != SOL_REVIEW_PROVENANCE_SCHEMA_VERSION:
        raise BallDetectorEvaluationError("solReviewProvenance schemaVersion is invalid")
    if provenance["kind"] != SOL_REVIEW_PROVENANCE_KIND:
        raise BallDetectorEvaluationError("solReviewProvenance kind is invalid")
    if provenance["detectorSuggestionsAbsent"] is not True:
        raise BallDetectorEvaluationError(
            "solReviewProvenance must affirm detectorSuggestionsAbsent"
        )
    if provenance["immutableDigestSha256"] != task["immutable"]["digestSha256"]:
        raise BallDetectorEvaluationError(
            "solReviewProvenance immutable digest does not match the task"
        )
    if not _valid_nonzero_sha256(provenance["implementationSha256"]):
        raise BallDetectorEvaluationError(
            "solReviewProvenance implementationSha256 is invalid"
        )
    prepared_at = provenance["preparedAt"]
    if not isinstance(prepared_at, str):
        raise BallDetectorEvaluationError("solReviewProvenance preparedAt is invalid")
    try:
        parsed_prepared_at = datetime.fromisoformat(prepared_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise BallDetectorEvaluationError(
            "solReviewProvenance preparedAt must be ISO-8601"
        ) from error
    if parsed_prepared_at.tzinfo is None:
        raise BallDetectorEvaluationError(
            "solReviewProvenance preparedAt must include a timezone"
        )

    reviewer = provenance["reviewer"]
    if not isinstance(reviewer, dict) or set(reviewer) != {
        "kind",
        "agentId",
        "modelId",
        "runId",
    }:
        raise BallDetectorEvaluationError("solReviewProvenance reviewer fields are invalid")
    if reviewer["kind"] != "detector-blind-sol-agent":
        raise BallDetectorEvaluationError("solReviewProvenance reviewer kind is invalid")
    for field in ("agentId", "modelId", "runId"):
        if not isinstance(reviewer[field], str) or not reviewer[field].strip():
            raise BallDetectorEvaluationError(
                f"solReviewProvenance reviewer {field} is invalid"
            )

    source = provenance["sourceTask"]
    if not isinstance(source, dict) or set(source) != {
        "pathHint",
        "sha256",
        "requiredState",
    }:
        raise BallDetectorEvaluationError(
            "solReviewProvenance sourceTask fields are invalid"
        )
    if source["requiredState"] != "unreviewed annotations with empty suggestions":
        raise BallDetectorEvaluationError(
            "solReviewProvenance sourceTask requiredState is invalid"
        )
    if not _valid_sha256(source["sha256"]):
        raise BallDetectorEvaluationError(
            "solReviewProvenance sourceTask sha256 is invalid"
        )
    if not isinstance(source["pathHint"], str) or not source["pathHint"]:
        raise BallDetectorEvaluationError(
            "solReviewProvenance sourceTask pathHint is invalid"
        )
    source_path = Path(source["pathHint"]).expanduser().resolve()
    if source_path == task_path:
        raise BallDetectorEvaluationError(
            "solReviewProvenance sourceTask cannot point to the Sol review copy"
        )
    if source_path.parent.parent != task_path.parent.parent:
        raise BallDetectorEvaluationError(
            "Sol review and source task must remain in sibling-depth directories"
        )
    source_raw, source_sha = _read_task_with_hash(source_path)
    if source_sha != source["sha256"]:
        raise BallDetectorEvaluationError(
            "solReviewProvenance source task SHA-256 does not match"
        )
    try:
        validated_source = validate_ball_annotation_task(
            source_raw,
            task_path=source_path,
            verify_images=True,
        )
    except ValueError as error:
        raise BallDetectorEvaluationError(
            f"solReviewProvenance source task is invalid: {error}"
        ) from error
    _validate_pristine_sol_source_structure(validated_source)
    if validated_source["annotations"]["review"]["status"] != "unreviewed":
        raise BallDetectorEvaluationError(
            "Sol source task annotations were not detector-blind and unreviewed"
        )
    if validated_source["suggestions"]["status"] != "empty":
        raise BallDetectorEvaluationError(
            "Sol source task did not have empty detector suggestions"
        )
    if "blindMergeProvenance" in validated_source or "solReviewProvenance" in validated_source:
        raise BallDetectorEvaluationError(
            "Sol source task must be an original detector-empty sampling task"
        )
    if task["suggestions"]["status"] != "empty":
        raise BallDetectorEvaluationError(
            "Sol review task must keep detector suggestions empty"
        )
    if task["annotations"]["review"]["status"] != required_review_status:
        raise BallDetectorEvaluationError(
            f"Sol review task must have {required_review_status} annotations"
        )
    try:
        assert_immutable_provenance_unchanged(task, validated_source)
    except ValueError as error:
        raise BallDetectorEvaluationError(
            f"Sol review immutable provenance differs from its source: {error}"
        ) from error
    for frame in task["immutable"]["frames"]:
        relative_image = Path(frame["image"]["path"])
        if (source_path.parent / relative_image).resolve() != (
            task_path.parent / relative_image
        ).resolve():
            raise BallDetectorEvaluationError(
                "Sol review no longer resolves to the source task's immutable images"
            )
    if required_review_status == "unreviewed" and task["annotations"] != validated_source[
        "annotations"
    ]:
        raise BallDetectorEvaluationError(
            "prepared Sol review annotations must exactly match the empty source task"
        )

    receipt_pointer = provenance["preparationReceipt"]
    if not isinstance(receipt_pointer, dict) or set(receipt_pointer) != {
        "pathHint",
        "sha256",
    }:
        raise BallDetectorEvaluationError(
            "solReviewProvenance preparationReceipt fields are invalid"
        )
    if not _valid_nonzero_sha256(receipt_pointer["sha256"]):
        raise BallDetectorEvaluationError(
            "solReviewProvenance preparationReceipt sha256 is invalid"
        )
    if not isinstance(receipt_pointer["pathHint"], str) or not receipt_pointer[
        "pathHint"
    ]:
        raise BallDetectorEvaluationError(
            "solReviewProvenance preparationReceipt pathHint is invalid"
        )
    receipt_path = Path(receipt_pointer["pathHint"]).expanduser().resolve()
    if receipt_path != _sol_preparation_receipt_path(task_path):
        raise BallDetectorEvaluationError(
            "Sol preparation receipt is not at the deterministic sibling path"
        )
    try:
        receipt_encoded = receipt_path.read_bytes()
        receipt = json.loads(receipt_encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BallDetectorEvaluationError(
            f"cannot read Sol preparation receipt {receipt_path}: {error}"
        ) from error
    receipt_sha = hashlib.sha256(receipt_encoded).hexdigest()
    if receipt_sha != receipt_pointer["sha256"]:
        raise BallDetectorEvaluationError("Sol preparation receipt SHA-256 does not match")
    receipt_fields = {
        "schemaVersion",
        "kind",
        "preparedAt",
        "outputTask",
        "sourceTask",
        "pilotIndex",
        "reviewer",
        "detectorSuggestionsAbsent",
        "immutableDigestSha256",
        "implementationSha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != receipt_fields:
        raise BallDetectorEvaluationError("Sol preparation receipt fields are invalid")
    if receipt["schemaVersion"] != 1 or receipt["kind"] != SOL_PREPARATION_RECEIPT_KIND:
        raise BallDetectorEvaluationError("Sol preparation receipt kind/schema is invalid")
    expected_receipt_fields = {
        "preparedAt": provenance["preparedAt"],
        "sourceTask": provenance["sourceTask"],
        "reviewer": provenance["reviewer"],
        "detectorSuggestionsAbsent": True,
        "immutableDigestSha256": provenance["immutableDigestSha256"],
        "implementationSha256": provenance["implementationSha256"],
    }
    for field, expected_value in expected_receipt_fields.items():
        if receipt[field] != expected_value:
            raise BallDetectorEvaluationError(
                f"Sol preparation receipt {field} differs from task provenance"
            )
    if receipt["outputTask"] != {
        "pathHint": str(task_path),
        "requiredState": "unreviewed annotations with empty suggestions",
    }:
        raise BallDetectorEvaluationError(
            "Sol preparation receipt outputTask binding is invalid"
        )
    pilot_pointer = receipt["pilotIndex"]
    if not isinstance(pilot_pointer, dict) or set(pilot_pointer) != {
        "pathHint",
        "sha256",
    }:
        raise BallDetectorEvaluationError(
            "Sol preparation receipt pilotIndex fields are invalid"
        )
    if (
        not isinstance(pilot_pointer["pathHint"], str)
        or not pilot_pointer["pathHint"]
        or not _valid_nonzero_sha256(pilot_pointer["sha256"])
    ):
        raise BallDetectorEvaluationError(
            "Sol preparation receipt pilotIndex provenance is invalid"
        )
    index_binding = _validate_initial_source_index_binding(
        validated_source,
        source_sha,
        pilot_pointer["pathHint"],
        expected_index_sha256=pilot_pointer["sha256"],
    )
    if required_review_status == "complete":
        reviewed_at = task["annotations"]["review"]["reviewedAt"]
        try:
            parsed_reviewed_at = datetime.fromisoformat(
                str(reviewed_at).replace("Z", "+00:00")
            )
        except ValueError as error:
            raise BallDetectorEvaluationError(
                "completed Sol review reviewedAt must be ISO-8601"
            ) from error
        if parsed_reviewed_at.tzinfo is None or parsed_reviewed_at < parsed_prepared_at:
            raise BallDetectorEvaluationError(
                "completed Sol review reviewedAt must be timezone-aware and not precede "
                "preparation"
            )
    return {
        "schemaVersion": provenance["schemaVersion"],
        "kind": provenance["kind"],
        "preparedAt": provenance["preparedAt"],
        "detectorSuggestionsAbsent": True,
        "immutableDigestSha256": provenance["immutableDigestSha256"],
        "implementationSha256": provenance["implementationSha256"],
        "preparationReceipt": {
            "pathHint": str(receipt_path),
            "sha256": receipt_sha,
        },
        "pilotIndex": index_binding,
        "sourceTask": {
            "pathHint": str(source_path),
            "sha256": source_sha,
            "requiredState": source["requiredState"],
        },
        "reviewer": copy.deepcopy(reviewer),
    }


def prepare_detector_blind_sol_review(
    source_task_path: str | Path,
    output_path: str | Path,
    *,
    pilot_index_path: str | Path,
    agent_id: str,
    model_id: str,
    run_id: str,
) -> Path:
    """Create an unreviewed Sol copy whose SHA-pinned source contains no proposals."""

    source_path = Path(source_task_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source_path == destination:
        raise BallDetectorEvaluationError("source task and Sol review output must differ")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Sol review task: {destination}")
    receipt_path = _sol_preparation_receipt_path(destination)
    if receipt_path.exists():
        raise FileExistsError(
            f"refusing to overwrite Sol preparation receipt: {receipt_path}"
        )
    if source_path.parent.parent != destination.parent.parent:
        raise BallDetectorEvaluationError(
            "source task and Sol review output must use sibling-depth directories under "
            "one pilot root"
        )
    for field, value in (
        ("agent_id", agent_id),
        ("model_id", model_id),
        ("run_id", run_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise BallDetectorEvaluationError(f"{field} must be a non-empty string")

    source_raw, source_sha = _read_task_with_hash(source_path)
    try:
        source = validate_ball_annotation_task(
            source_raw,
            task_path=source_path,
            verify_images=True,
        )
    except ValueError as error:
        raise BallDetectorEvaluationError(f"invalid Sol source task: {error}") from error
    _validate_pristine_sol_source_structure(source)
    split = source["immutable"]["recording"]["split"]
    if split not in {"train", "validation"}:
        raise BallDetectorEvaluationError(
            f"Sol review preparation is development-only; split {split!r} is forbidden"
        )
    if source["annotations"]["review"]["status"] != "unreviewed":
        raise BallDetectorEvaluationError("Sol source annotations must be unreviewed")
    if source["suggestions"]["status"] != "empty":
        raise BallDetectorEvaluationError(
            "Sol source must have empty detector suggestions"
        )
    if "blindMergeProvenance" in source or "solReviewProvenance" in source:
        raise BallDetectorEvaluationError(
            "Sol source must be an original detector-empty sampling task"
        )
    index_binding = _validate_initial_source_index_binding(
        source,
        source_sha,
        pilot_index_path,
    )
    for frame in source["immutable"]["frames"]:
        relative_image = Path(frame["image"]["path"])
        if (source_path.parent / relative_image).resolve() != (
            destination.parent / relative_image
        ).resolve():
            raise BallDetectorEvaluationError(
                "Sol output does not preserve immutable image paths; choose a sibling "
                "directory at the same depth as the source"
            )

    prepared_at = datetime.now(timezone.utc).isoformat()
    implementation_sha = _sha256_file(Path(__file__).resolve())
    source_provenance = {
        "pathHint": str(source_path),
        "sha256": source_sha,
        "requiredState": "unreviewed annotations with empty suggestions",
    }
    reviewer_provenance = {
        "kind": "detector-blind-sol-agent",
        "agentId": agent_id.strip(),
        "modelId": model_id.strip(),
        "runId": run_id.strip(),
    }
    receipt = {
        "schemaVersion": 1,
        "kind": SOL_PREPARATION_RECEIPT_KIND,
        "preparedAt": prepared_at,
        "outputTask": {
            "pathHint": str(destination),
            "requiredState": "unreviewed annotations with empty suggestions",
        },
        "sourceTask": source_provenance,
        "pilotIndex": {
            "pathHint": index_binding["pathHint"],
            "sha256": index_binding["sha256"],
        },
        "reviewer": reviewer_provenance,
        "detectorSuggestionsAbsent": True,
        "immutableDigestSha256": source["immutable"]["digestSha256"],
        "implementationSha256": implementation_sha,
    }
    receipt_text = json.dumps(
        receipt,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    receipt_sha = hashlib.sha256(receipt_text.encode("utf-8")).hexdigest()
    prepared = copy.deepcopy(source)
    prepared["solReviewProvenance"] = {
        "schemaVersion": SOL_REVIEW_PROVENANCE_SCHEMA_VERSION,
        "kind": SOL_REVIEW_PROVENANCE_KIND,
        "preparedAt": prepared_at,
        "sourceTask": source_provenance,
        "preparationReceipt": {
            "pathHint": str(receipt_path),
            "sha256": receipt_sha,
        },
        "reviewer": reviewer_provenance,
        "detectorSuggestionsAbsent": True,
        "immutableDigestSha256": source["immutable"]["digestSha256"],
        "implementationSha256": implementation_sha,
    }
    try:
        prepared = validate_ball_annotation_task(
            prepared,
            task_path=destination,
            verify_images=True,
        )
    except ValueError as error:
        raise BallDetectorEvaluationError(
            f"prepared Sol review task is invalid: {error}"
        ) from error
    receipt_created = False
    task_created = False
    try:
        atomic_write_text(receipt_path, receipt_text)
        receipt_created = True
        _validate_sol_review_provenance(
            prepared,
            destination,
            required_review_status="unreviewed",
        )
        written = atomic_write_text(
            destination,
            json.dumps(prepared, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        )
        task_created = True
        validate_ball_annotation_task(written, verify_images=True)
        return written
    except BaseException:
        if task_created:
            destination.unlink(missing_ok=True)
        if receipt_created:
            receipt_path.unlink(missing_ok=True)
        raise


def merge_blind_review_with_detector_suggestions(
    reviewed_labels_path: str | Path,
    detector_proposals_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Merge labels and proposals only after an independently blind review is complete."""

    reviewed_path = Path(reviewed_labels_path).expanduser().resolve()
    proposal_path = Path(detector_proposals_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if len({reviewed_path, proposal_path, destination}) != 3:
        raise BallDetectorEvaluationError(
            "reviewed labels, detector proposals, and merged output must be distinct files"
        )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite merged task: {destination}")
    artifact_roots = {
        reviewed_path.parent.parent,
        proposal_path.parent.parent,
        destination.parent.parent,
    }
    if len(artifact_roots) != 1:
        raise BallDetectorEvaluationError(
            "reviewed labels, detector proposals, and output must use sibling-depth "
            "directories under one pilot root"
        )

    reviewed_raw, reviewed_sha = _read_task_with_hash(reviewed_path)
    proposal_raw, proposal_sha = _read_task_with_hash(proposal_path)
    try:
        reviewed = validate_ball_annotation_task(
            reviewed_raw,
            task_path=reviewed_path,
            verify_images=True,
        )
        proposals = validate_ball_annotation_task(
            proposal_raw,
            task_path=proposal_path,
            verify_images=True,
        )
    except ValueError as error:
        raise BallDetectorEvaluationError(f"cannot validate blind-merge inputs: {error}") from error
    _validate_completed_human_review_structure(reviewed)
    if proposals["suggestions"]["status"] != "complete":
        raise BallDetectorEvaluationError("proposal task suggestions must be complete")
    if proposals["annotations"]["review"]["status"] != "unreviewed":
        raise BallDetectorEvaluationError(
            "proposal task annotations must remain unreviewed and separate from labels"
        )
    try:
        assert_immutable_provenance_unchanged(reviewed, proposals)
    except ValueError as error:
        raise BallDetectorEvaluationError(
            f"reviewed labels and detector proposals have different immutable provenance: {error}"
        ) from error

    for frame in reviewed["immutable"]["frames"]:
        relative_image = Path(frame["image"]["path"])
        resolved = {
            (reviewed_path.parent / relative_image).resolve(),
            (proposal_path.parent / relative_image).resolve(),
            (destination.parent / relative_image).resolve(),
        }
        if len(resolved) != 1:
            raise BallDetectorEvaluationError(
                "merged output does not preserve the exact immutable image paths; choose a "
                "sibling directory at the same depth as both inputs"
            )

    merged = copy.deepcopy(reviewed)
    merged["suggestions"] = copy.deepcopy(proposals["suggestions"])
    review_exposure = _human_exposure_summary(reviewed)
    merged["blindMergeProvenance"] = {
        "schemaVersion": 1,
        "kind": "volleycut-ball-presence-blind-review-merge",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "reviewedLabels": {
            "pathHint": str(reviewed_path),
            "sha256": reviewed_sha,
            "requiredState": "complete annotations with empty suggestions",
        },
        "detectorProposals": {
            "pathHint": str(proposal_path),
            "sha256": proposal_sha,
            "requiredState": "complete suggestions with unreviewed annotations",
        },
        "humanReviewExposure": {
            "field": "annotations.frames[*].proposalExposure",
            "notShownFrameCount": review_exposure["not_shown"],
            "shownBeforeLabelFinalizedFrameCount": review_exposure[
                "shown_before_label_finalized"
            ],
            "notRecordedFrameCount": review_exposure["not_recorded"],
            "qualityEligibleFrameCount": review_exposure["not_shown"],
            "blanketBlindnessClaim": False,
        },
        "immutableDigestSha256": reviewed["immutable"]["digestSha256"],
        "implementationSha256": _sha256_file(Path(__file__).resolve()),
    }
    try:
        validate_ball_annotation_task(
            merged,
            task_path=destination,
            verify_images=True,
        )
    except ValueError as error:
        raise BallDetectorEvaluationError(f"merged task is invalid: {error}") from error
    _validate_blind_merge_provenance(merged, destination)
    written = atomic_write_text(
        destination,
        json.dumps(merged, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    validate_ball_annotation_task(written, verify_images=True)
    return written


def _box(value: Mapping[str, Any]) -> Box:
    return Box(*(float(value[key]) for key in ("x", "y", "width", "height")))


def _discover_task_paths(inputs: Iterable[str | Path]) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for supplied in inputs:
        path = Path(supplied).expanduser().resolve()
        if path.is_file():
            paths.add(path)
        elif path.is_dir():
            paths.update(item.resolve() for item in path.rglob("*.json") if item.is_file())
        else:
            raise BallDetectorEvaluationError(f"task input does not exist: {path}")
    task_paths: list[Path] = []
    for path in sorted(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BallDetectorEvaluationError(f"cannot read JSON input {path}: {error}") from error
        if isinstance(payload, dict) and payload.get("taskType") == BALL_ANNOTATION_TASK_TYPE:
            task_paths.append(path)
    if not task_paths:
        raise BallDetectorEvaluationError("no ball-presence annotation tasks found")
    return tuple(task_paths)


def _stable_detector_payload(model: Mapping[str, Any]) -> dict[str, Any]:
    """Remove per-task/path fields while retaining detector identity and settings."""

    stable = copy.deepcopy(dict(model))
    stable.pop("sourceTask", None)
    stable.pop("modelPath", None)
    return stable


def load_completed_reviewed_tasks(inputs: Iterable[str | Path]) -> LoadedEvaluation:
    """Load complete development reviews with complete, embedded detector suggestions."""

    frames: list[EvaluationFrame] = []
    task_provenance: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    detector_payload: dict[str, Any] | None = None
    detector_signature: str | None = None
    manifest_sha: str | None = None
    exposure_summary = {
        "not_shown": 0,
        "shown_before_label_finalized": 0,
        "not_recorded": 0,
    }

    for path in _discover_task_paths(inputs):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            task = validate_ball_annotation_task(raw, task_path=path, verify_images=False)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise BallDetectorEvaluationError(f"invalid annotation task {path}: {error}") from error
        blind_merge = _validate_blind_merge_provenance(task, path)
        immutable = task["immutable"]
        task_id = immutable["taskId"]
        if task_id in task_ids:
            raise BallDetectorEvaluationError(f"duplicate immutable task id: {task_id}")
        task_ids.add(task_id)
        recording = immutable["recording"]
        split = recording["split"]
        if split not in {"train", "validation"}:
            raise BallDetectorEvaluationError(
                f"{task_id} is split {split!r}; detector threshold evaluation is development-only"
            )
        review = task["annotations"]["review"]
        if review["status"] != "complete":
            raise BallDetectorEvaluationError(f"{task_id} review is not complete")
        suggestions = task["suggestions"]
        if suggestions["status"] != "complete":
            raise BallDetectorEvaluationError(f"{task_id} detector suggestions are not complete")
        stable_model = _stable_detector_payload(suggestions["model"])
        current_signature = _canonical_sha256(stable_model)
        if detector_signature is None:
            detector_signature = current_signature
            detector_payload = stable_model
        elif detector_signature != current_signature:
            raise BallDetectorEvaluationError(
                "tasks contain inconsistent detector artifacts or inference settings"
            )
        current_manifest_sha = immutable["manifest"]["sha256"]
        if manifest_sha is None:
            manifest_sha = current_manifest_sha
        elif manifest_sha != current_manifest_sha:
            raise BallDetectorEvaluationError("tasks were sampled from different manifests")

        windows = {item["id"]: item for item in immutable["windows"]}
        labels = task["annotations"]["frames"]
        proposals = suggestions["frames"]
        task_exposure_summary = _human_exposure_summary(task)
        for frame_row in immutable["frames"]:
            frame_id = frame_row["id"]
            label = labels[frame_id]
            if label["status"] != "reviewed":
                raise BallDetectorEvaluationError(
                    f"{task_id}/{frame_id} is not reviewed despite complete review status"
                )
            exposure = _proposal_exposure_status(label)
            exposure_summary[exposure] += 1
            if exposure != "not_shown":
                continue
            proposal = proposals[frame_id]
            window = windows[frame_row["windowId"]]
            image = frame_row["image"]
            truth = tuple(
                TruthObject(
                    bbox=_box(item["bbox"]),
                    role=item["role"],
                    visibility=item["visibility"],
                    truncated=bool(item["truncated"]),
                )
                for item in label["objects"]
            )
            detections = tuple(
                Detection(
                    bbox=_box(item["bbox"]),
                    confidence=float(item["confidence"]),
                )
                for item in proposal["detections"]
            )
            presence_probability = float(proposal["ballPresenceProbability"])
            expected_probability = max(
                (item.confidence for item in detections), default=0.0
            )
            if not math.isclose(
                presence_probability,
                expected_probability,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise BallDetectorEvaluationError(
                    f"{task_id}/{frame_id} ballPresenceProbability is not the maximum "
                    "stored detection confidence"
                )
            frames.append(
                EvaluationFrame(
                    key=f"{task_id}/{frame_id}",
                    task_id=task_id,
                    recording_id=recording["id"],
                    source_group=str(recording["sourceGroup"]),
                    environment=(
                        "unknown"
                        if recording["environment"] is None
                        else str(recording["environment"])
                    ),
                    split=split,
                    window_key=f"{task_id}/{frame_row['windowId']}",
                    window_id=frame_row["windowId"],
                    stratum=window["requestedStratum"],
                    width=int(image["width"]),
                    height=int(image["height"]),
                    primary_ball_state=label["primaryBallState"],
                    truth_objects=truth,
                    ball_presence_probability=presence_probability,
                    detections=detections,
                    proposal_exposure=exposure,
                )
            )
        task_provenance.append(
            {
                "taskId": task_id,
                "pathHint": str(path),
                "fileSha256": _sha256_file(path),
                "immutableDigestSha256": immutable["digestSha256"],
                "recordingId": recording["id"],
                "split": split,
                "sourceGroup": recording["sourceGroup"],
                "environment": recording["environment"],
                "proxySha256": immutable["source"]["proxy"]["sha256"],
                "reviewedAt": review["reviewedAt"],
                "frameCount": len(immutable["frames"]),
                "windowCount": len(immutable["windows"]),
                "samplingPolicyId": immutable["sampling"]["policyId"],
                "round": immutable["sampling"]["round"],
                "initialTaskSha256": blind_merge["initialTask"]["sha256"],
                "blindMergeProvenance": blind_merge,
                "humanReviewExposure": task_exposure_summary,
            }
        )
    if detector_payload is None or detector_signature is None or manifest_sha is None:
        raise AssertionError("task discovery returned no loadable tasks")
    if not frames:
        raise BallDetectorEvaluationError(
            "no quality-eligible human frames remain: every frame was assisted or lacks "
            "a persisted proposalExposure audit"
        )
    return LoadedEvaluation(
        frames=tuple(frames),
        tasks=tuple(sorted(task_provenance, key=lambda item: item["recordingId"])),
        detector=detector_payload,
        detector_signature_sha256=detector_signature,
        manifest_sha256=manifest_sha,
        exposure_summary=exposure_summary,
    )


def _discover_sol_task_paths(inputs: Iterable[str | Path]) -> tuple[Path, ...]:
    direct: set[Path] = set()
    discovered: set[Path] = set()
    for supplied in inputs:
        path = Path(supplied).expanduser().resolve()
        if path.is_file():
            direct.add(path)
            discovered.add(path)
        elif path.is_dir():
            for candidate in path.rglob("*.json"):
                if not candidate.is_file():
                    continue
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise BallDetectorEvaluationError(
                        f"cannot read JSON input {candidate}: {error}"
                    ) from error
                if (
                    isinstance(payload, dict)
                    and payload.get("taskType") == BALL_ANNOTATION_TASK_TYPE
                    and "solReviewProvenance" in payload
                ):
                    discovered.add(candidate.resolve())
        else:
            raise BallDetectorEvaluationError(f"Sol task input does not exist: {path}")
    if not discovered:
        raise BallDetectorEvaluationError("no provenance-bound Sol review tasks found")
    for path in direct:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BallDetectorEvaluationError(f"cannot read Sol task {path}: {error}") from error
        if not isinstance(payload, dict) or "solReviewProvenance" not in payload:
            raise BallDetectorEvaluationError(
                f"explicit Sol input lacks solReviewProvenance: {path}"
            )
    return tuple(sorted(discovered))


def load_completed_sol_reviews(
    inputs: Iterable[str | Path],
    human_evaluation: LoadedEvaluation,
) -> LoadedSolReviews:
    """Load one detector-empty, complete Sol review for every human-review task."""

    expected = {str(task["taskId"]): task for task in human_evaluation.tasks}
    seen: set[str] = set()
    frames: list[SolAnnotationFrame] = []
    task_provenance: list[dict[str, Any]] = []
    preparation_index_sha: str | None = None
    for path in _discover_sol_task_paths(inputs):
        raw, file_sha = _read_task_with_hash(path)
        try:
            task = validate_ball_annotation_task(
                raw,
                task_path=path,
                verify_images=True,
            )
        except ValueError as error:
            raise BallDetectorEvaluationError(f"invalid Sol review task {path}: {error}") from error
        if "blindMergeProvenance" in task:
            raise BallDetectorEvaluationError(
                "Sol review task must remain separate from the human/detector merged task"
            )
        provenance = _validate_sol_review_provenance(
            task,
            path,
            required_review_status="complete",
        )
        current_index_sha = provenance["pilotIndex"]["sha256"]
        if preparation_index_sha is None:
            preparation_index_sha = current_index_sha
        elif current_index_sha != preparation_index_sha:
            raise BallDetectorEvaluationError(
                "Sol review tasks were prepared from inconsistent pilot indexes"
            )
        immutable = task["immutable"]
        task_id = str(immutable["taskId"])
        if task_id in seen:
            raise BallDetectorEvaluationError(f"duplicate Sol review task id: {task_id}")
        seen.add(task_id)
        human_task = expected.get(task_id)
        if human_task is None:
            raise BallDetectorEvaluationError(
                f"Sol review task {task_id} has no matching human-review task"
            )
        if immutable["recording"]["split"] not in {"train", "validation"}:
            raise BallDetectorEvaluationError("Sol comparison is development-only")
        if immutable["digestSha256"] != human_task["immutableDigestSha256"]:
            raise BallDetectorEvaluationError(
                f"Sol review task {task_id} immutable digest differs from human truth"
            )
        if immutable["recording"]["id"] != human_task["recordingId"]:
            raise BallDetectorEvaluationError(
                f"Sol review task {task_id} recording differs from human truth"
            )
        if len(immutable["frames"]) != int(human_task["frameCount"]):
            raise BallDetectorEvaluationError(
                f"Sol review task {task_id} frame count differs from human truth"
            )
        source_sha = provenance["sourceTask"]["sha256"]
        if source_sha != human_task["initialTaskSha256"]:
            raise BallDetectorEvaluationError(
                f"Sol review task {task_id} was not prepared from the exact indexed "
                "detector-empty source task"
            )
        review = task["annotations"]["review"]
        if review["annotator"] != provenance["reviewer"]["agentId"]:
            raise BallDetectorEvaluationError(
                f"Sol review task {task_id} annotator must equal its provenance agentId"
            )
        annotations = task["annotations"]["frames"]
        for frame_row in immutable["frames"]:
            frame_id = frame_row["id"]
            annotation = annotations[frame_id]
            if annotation["status"] != "reviewed":
                raise BallDetectorEvaluationError(
                    f"Sol review task {task_id}/{frame_id} is not reviewed"
                )
            objects = tuple(
                TruthObject(
                    bbox=_box(item["bbox"]),
                    role=item["role"],
                    visibility=item["visibility"],
                    truncated=bool(item["truncated"]),
                )
                for item in annotation["objects"]
            )
            frames.append(
                SolAnnotationFrame(
                    key=f"{task_id}/{frame_id}",
                    task_id=task_id,
                    primary_ball_state=annotation["primaryBallState"],
                    objects=objects,
                )
            )
        task_provenance.append(
            {
                "taskId": task_id,
                "pathHint": str(path),
                "fileSha256": file_sha,
                "immutableDigestSha256": immutable["digestSha256"],
                "recordingId": immutable["recording"]["id"],
                "frameCount": len(immutable["frames"]),
                "reviewedAt": review["reviewedAt"],
                "annotator": review["annotator"],
                "solReviewProvenance": provenance,
            }
        )
    missing = sorted(set(expected) - seen)
    if missing:
        raise BallDetectorEvaluationError(
            f"Sol review inputs do not cover every human task; missing task ids: {missing}"
        )
    return LoadedSolReviews(
        frames=tuple(frames),
        tasks=tuple(sorted(task_provenance, key=lambda item: item["recordingId"])),
    )


def bbox_iou(first: Box, second: Box) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union > 0 else 0.0


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _classification_counts(truth: Sequence[bool], predicted: Sequence[bool]) -> dict[str, Any]:
    true_positive = sum(actual and estimate for actual, estimate in zip(truth, predicted))
    false_positive = sum(not actual and estimate for actual, estimate in zip(truth, predicted))
    false_negative = sum(actual and not estimate for actual, estimate in zip(truth, predicted))
    true_negative = len(truth) - true_positive - false_positive - false_negative
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "frames": len(truth),
        "positiveFrames": true_positive + false_negative,
        "predictedPositiveFrames": true_positive + false_positive,
        "truePositive": true_positive,
        "falsePositive": false_positive,
        "falseNegative": false_negative,
        "trueNegative": true_negative,
        "precision": _ratio(true_positive, true_positive + false_positive),
        "recall": _ratio(true_positive, true_positive + false_negative),
        "f1": 2 * true_positive / denominator if denominator else None,
    }


def _maximum_cardinality_matches(
    predictions: Sequence[Detection],
    truths: Sequence[TruthObject],
    qualifies: Callable[[Detection, TruthObject], bool],
    quality: Callable[[Detection, TruthObject], float],
) -> tuple[tuple[int, int], ...]:
    edges = [
        sorted(
            (truth_index for truth_index, truth in enumerate(truths) if qualifies(prediction, truth)),
            key=lambda truth_index: quality(prediction, truths[truth_index]),
            reverse=True,
        )
        for prediction in predictions
    ]
    truth_to_prediction: dict[int, int] = {}

    def assign(prediction_index: int, visited: set[int]) -> bool:
        for truth_index in edges[prediction_index]:
            if truth_index in visited:
                continue
            visited.add(truth_index)
            previous = truth_to_prediction.get(truth_index)
            if previous is None or assign(previous, visited):
                truth_to_prediction[truth_index] = prediction_index
                return True
        return False

    order = sorted(
        range(len(predictions)),
        key=lambda index: (-predictions[index].confidence, index),
    )
    for prediction_index in order:
        assign(prediction_index, set())
    return tuple(
        sorted(
            (prediction_index, truth_index)
            for truth_index, prediction_index in truth_to_prediction.items()
        )
    )


def _truth_for(frame: EvaluationFrame, target: str) -> tuple[TruthObject, ...]:
    if target == "primary":
        return frame.primary_truth
    if target == "any":
        return frame.any_ball_truth
    raise ValueError(f"unknown ball target: {target}")


def _is_evaluable(frame: EvaluationFrame, target: str) -> bool:
    if target == "primary":
        return frame.primary_ball_state != "indeterminate"
    if target == "any":
        return bool(frame.any_ball_truth) or frame.primary_ball_state != "indeterminate"
    raise ValueError(f"unknown ball target: {target}")


def _predictions_at(frame: EvaluationFrame, threshold: float) -> tuple[Detection, ...]:
    return tuple(item for item in frame.detections if item.confidence >= threshold)


def _box_point_metrics(
    frames: Sequence[EvaluationFrame],
    threshold: float,
    *,
    target: str,
    minimum_iou: float,
) -> dict[str, Any]:
    predicted_count = truth_count = matched_count = 0
    for frame in frames:
        if not _is_evaluable(frame, target):
            continue
        predictions = _predictions_at(frame, threshold)
        truths = _truth_for(frame, target)
        matches = _maximum_cardinality_matches(
            predictions,
            truths,
            lambda prediction, truth: bbox_iou(prediction.bbox, truth.bbox) >= minimum_iou,
            lambda prediction, truth: bbox_iou(prediction.bbox, truth.bbox),
        )
        predicted_count += len(predictions)
        truth_count += len(truths)
        matched_count += len(matches)
    denominator = 2 * matched_count + (predicted_count - matched_count) + (truth_count - matched_count)
    return {
        "minimumIou": minimum_iou,
        "truthBoxes": truth_count,
        "predictedBoxes": predicted_count,
        "matchedBoxes": matched_count,
        "precision": _ratio(matched_count, predicted_count),
        "recall": _ratio(matched_count, truth_count),
        "f1": 2 * matched_count / denominator if denominator else None,
    }


def _center_distance_pixels(
    prediction: Detection,
    truth: TruthObject,
    frame: EvaluationFrame,
) -> float:
    predicted_x, predicted_y = prediction.bbox.center
    truth_x, truth_y = truth.bbox.center
    return math.hypot(
        (predicted_x - truth_x) * frame.width,
        (predicted_y - truth_y) * frame.height,
    )


def _annotated_diameter_pixels(truth: TruthObject, frame: EvaluationFrame) -> float:
    return max(truth.bbox.width * frame.width, truth.bbox.height * frame.height)


def _center_match_metrics(
    frames: Sequence[EvaluationFrame],
    threshold: float,
    *,
    target: str,
    truth_filter: Callable[[TruthObject, EvaluationFrame], bool] | None = None,
) -> dict[str, Any]:
    truth_count = matched_count = 0
    frames_with_truth = 0
    for frame in frames:
        if not _is_evaluable(frame, target):
            continue
        truths = tuple(
            item
            for item in _truth_for(frame, target)
            if truth_filter is None or truth_filter(item, frame)
        )
        if truths:
            frames_with_truth += 1
        predictions = _predictions_at(frame, threshold)
        matches = _maximum_cardinality_matches(
            predictions,
            truths,
            lambda prediction, truth: _center_distance_pixels(prediction, truth, frame)
            <= max(4.0, _annotated_diameter_pixels(truth, frame)) + 1e-9,
            lambda prediction, truth: -_center_distance_pixels(prediction, truth, frame),
        )
        truth_count += len(truths)
        matched_count += len(matches)
    return {
        "truthBoxes": truth_count,
        "framesWithTruth": frames_with_truth,
        "matchedTruthBoxes": matched_count,
        "recall": _ratio(matched_count, truth_count),
        "toleranceRule": "center distance <= max(4 px, max annotated box dimension in px)",
    }


def _ranked_average_precision(
    frames: Sequence[EvaluationFrame],
    *,
    target: str,
    minimum_iou: float,
) -> dict[str, Any]:
    eligible_frames = tuple(frame for frame in frames if _is_evaluable(frame, target))
    truths_by_frame = {frame.key: _truth_for(frame, target) for frame in eligible_frames}
    truth_count = sum(len(items) for items in truths_by_frame.values())
    ranked: list[tuple[float, str, int, Detection]] = []
    for frame in eligible_frames:
        ranked.extend(
            (item.confidence, frame.key, index, item)
            for index, item in enumerate(frame.detections)
        )
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched_truth: set[tuple[str, int]] = set()
    true_flags: list[int] = []
    false_flags: list[int] = []
    for _, frame_key, _, prediction in ranked:
        candidates = [
            (bbox_iou(prediction.bbox, truth.bbox), truth_index)
            for truth_index, truth in enumerate(truths_by_frame[frame_key])
            if (frame_key, truth_index) not in matched_truth
        ]
        overlap, truth_index = max(candidates, default=(0.0, -1))
        if overlap >= minimum_iou:
            matched_truth.add((frame_key, truth_index))
            true_flags.append(1)
            false_flags.append(0)
        else:
            true_flags.append(0)
            false_flags.append(1)
    if truth_count == 0:
        average_precision: float | None = None
    elif not ranked:
        average_precision = 0.0
    else:
        cumulative_true = 0
        cumulative_false = 0
        recalls: list[float] = []
        precisions: list[float] = []
        for true_flag, false_flag in zip(true_flags, false_flags):
            cumulative_true += true_flag
            cumulative_false += false_flag
            recalls.append(cumulative_true / truth_count)
            precisions.append(cumulative_true / (cumulative_true + cumulative_false))
        recall_points = [0.0, *recalls, 1.0]
        precision_envelope = [0.0, *precisions, 0.0]
        for index in range(len(precision_envelope) - 2, -1, -1):
            precision_envelope[index] = max(
                precision_envelope[index], precision_envelope[index + 1]
            )
        average_precision = sum(
            (recall_points[index] - recall_points[index - 1]) * precision_envelope[index]
            for index in range(1, len(recall_points))
            if recall_points[index] != recall_points[index - 1]
        )
    return {
        "minimumIou": minimum_iou,
        "truthBoxes": truth_count,
        "rankedDetections": len(ranked),
        "averagePrecision": average_precision,
        "method": (
            "global confidence ranking, greedy unmatched same-frame truth assignment, "
            "all-point interpolated precision envelope"
        ),
        "scopeCaveat": "ranked only over proposals retained above the inference score floor",
    }


def _brier(frames: Sequence[EvaluationFrame], *, target: str) -> float | None:
    eligible = tuple(frame for frame in frames if _is_evaluable(frame, target))
    if not eligible:
        return None
    return fmean(
        (frame.ball_presence_probability - float(bool(_truth_for(frame, target)))) ** 2
        for frame in eligible
    )


def _presence_metrics(
    frames: Sequence[EvaluationFrame], threshold: float, *, target: str
) -> dict[str, Any]:
    eligible = tuple(frame for frame in frames if _is_evaluable(frame, target))
    truth = [bool(_truth_for(frame, target)) for frame in eligible]
    predictions = [frame.ball_presence_probability >= threshold for frame in eligible]
    result = _classification_counts(truth, predictions)
    result["excludedIndeterminateFrames"] = len(frames) - len(eligible)
    return result


def _truly_ball_free_frames(
    frames: Sequence[EvaluationFrame],
) -> tuple[EvaluationFrame, ...]:
    return tuple(
        frame
        for frame in frames
        if not frame.any_ball_truth and frame.primary_ball_state == "out_of_frame"
    )


def evaluate_frames_at_threshold(
    frames: Sequence[EvaluationFrame],
    threshold: float,
    *,
    include_ranked_average_precision: bool = True,
) -> dict[str, Any]:
    """Evaluate one operating point; inputs are exact reviewed frames, not iid samples."""

    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be finite, greater than zero, and at most one")
    ball_free = _truly_ball_free_frames(frames)
    false_detections = sum(len(_predictions_at(frame, threshold)) for frame in ball_free)
    frames_with_false_detection = sum(bool(_predictions_at(frame, threshold)) for frame in ball_free)
    result: dict[str, Any] = {
        "threshold": threshold,
        "framePresence": {
            "primaryLocalizable": _presence_metrics(
                frames, threshold, target="primary"
            ),
            "anyAnnotatedRealBall": _presence_metrics(frames, threshold, target="any"),
        },
        "calibration": {
            "primaryLocalizableBrier": _brier(frames, target="primary"),
            "anyAnnotatedRealBallBrier": _brier(frames, target="any"),
            "probabilityField": "suggestions.frames[*].ballPresenceProbability",
        },
        "localization": {},
        "falseDetectionsOnTrulyBallFreeFrames": {
            "ballFreeFrames": len(ball_free),
            "falseDetections": false_detections,
            "framesWithFalseDetection": frames_with_false_detection,
            "falseDetectionsPer1000BallFreeFrames": (
                false_detections * 1000.0 / len(ball_free) if ball_free else None
            ),
            "framesWithFalseDetectionPer1000BallFreeFrames": (
                frames_with_false_detection * 1000.0 / len(ball_free) if ball_free else None
            ),
        },
    }
    for target, label in (("primary", "primaryLocalizable"), ("any", "anyAnnotatedRealBall")):
        localization: dict[str, Any] = {
            "atIou025": _box_point_metrics(
                frames, threshold, target=target, minimum_iou=0.25
            ),
            "atIou050": _box_point_metrics(
                frames, threshold, target=target, minimum_iou=0.50
            ),
            "centerMatch": _center_match_metrics(frames, threshold, target=target),
        }
        if include_ranked_average_precision:
            localization["rankedAveragePrecision"] = {
                "atIou025": _ranked_average_precision(
                    frames, target=target, minimum_iou=0.25
                ),
                "atIou050": _ranked_average_precision(
                    frames, target=target, minimum_iou=0.50
                ),
            }
        result["localization"][label] = localization
    return result


def _threshold_candidates(
    frames: Sequence[EvaluationFrame], thresholds: Sequence[float] | None
) -> tuple[float, ...]:
    if thresholds is None:
        values = {
            frame.ball_presence_probability
            for frame in frames
            if frame.ball_presence_probability > 0
        }
        values.add(1.0)
    else:
        values = {float(item) for item in thresholds}
    if not values or any(not math.isfinite(item) or not 0 < item <= 1 for item in values):
        raise ValueError("thresholds must contain finite values in (0, 1]")
    return tuple(sorted(values, reverse=True))


def _sweep_row(frames: Sequence[EvaluationFrame], threshold: float) -> dict[str, Any]:
    ball_free = _truly_ball_free_frames(frames)
    false_detections = sum(len(_predictions_at(frame, threshold)) for frame in ball_free)
    false_frames = sum(bool(_predictions_at(frame, threshold)) for frame in ball_free)
    return {
        "threshold": threshold,
        "primaryLocalizable": _presence_metrics(frames, threshold, target="primary"),
        "anyAnnotatedRealBall": _presence_metrics(frames, threshold, target="any"),
        "falseDetectionsOnTrulyBallFreeFrames": {
            "ballFreeFrames": len(ball_free),
            "falseDetections": false_detections,
            "framesWithFalseDetection": false_frames,
            "falseDetectionsPer1000BallFreeFrames": (
                false_detections * 1000.0 / len(ball_free) if ball_free else None
            ),
            "framesWithFalseDetectionPer1000BallFreeFrames": (
                false_frames * 1000.0 / len(ball_free) if ball_free else None
            ),
        },
    }


def _selection_key(row: Mapping[str, Any], target: str) -> tuple[float, ...]:
    selected = row[target]
    alternate = row[
        "anyAnnotatedRealBall" if target == "primaryLocalizable" else "primaryLocalizable"
    ]

    def value(item: Any) -> float:
        return -1.0 if item is None else float(item)

    false_rate = row["falseDetectionsOnTrulyBallFreeFrames"][
        "falseDetectionsPer1000BallFreeFrames"
    ]
    false_detection_preference = (
        float("-inf") if false_rate is None else -float(false_rate)
    )
    return (
        value(selected["f1"]),
        value(selected["recall"]),
        value(selected["precision"]),
        value(alternate["f1"]),
        false_detection_preference,
        float(row["threshold"]),
    )


def _diagnostic_best(
    sweep: Sequence[Mapping[str, Any]], target: str
) -> Mapping[str, Any]:
    if not sweep:
        raise ValueError("cannot select from an empty threshold sweep")
    return max(sweep, key=lambda item: _selection_key(item, target))


def _precision_constrained_selection(
    sweep: Sequence[Mapping[str, Any]],
    *,
    minimum_primary_precision: float = 0.85,
) -> tuple[Mapping[str, Any], str]:
    qualified = [
        row
        for row in sweep
        if row["primaryLocalizable"]["precision"] is not None
        and float(row["primaryLocalizable"]["precision"])
        >= minimum_primary_precision
    ]
    if not qualified:
        return _diagnostic_best(sweep, "primaryLocalizable"), "diagnostic-fallback"

    def constrained_key(row: Mapping[str, Any]) -> tuple[float, ...]:
        metrics = row["primaryLocalizable"]
        false_rate = row["falseDetectionsOnTrulyBallFreeFrames"][
            "falseDetectionsPer1000BallFreeFrames"
        ]
        return (
            float(metrics["recall"]),
            -1.0 if metrics["f1"] is None else float(metrics["f1"]),
            float(metrics["precision"]),
            float("-inf") if false_rate is None else -float(false_rate),
            float(row["threshold"]),
        )

    return max(qualified, key=constrained_key), "primary-precision-constrained"


def _pool_classification_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    true_positive = sum(int(row["truePositive"]) for row in rows)
    false_positive = sum(int(row["falsePositive"]) for row in rows)
    false_negative = sum(int(row["falseNegative"]) for row in rows)
    true_negative = sum(int(row["trueNegative"]) for row in rows)
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "frames": sum(int(row["frames"]) for row in rows),
        "positiveFrames": true_positive + false_negative,
        "predictedPositiveFrames": true_positive + false_positive,
        "truePositive": true_positive,
        "falsePositive": false_positive,
        "falseNegative": false_negative,
        "trueNegative": true_negative,
        "precision": _ratio(true_positive, true_positive + false_positive),
        "recall": _ratio(true_positive, true_positive + false_negative),
        "f1": 2 * true_positive / denominator if denominator else None,
        "excludedIndeterminateFrames": sum(
            int(row["excludedIndeterminateFrames"]) for row in rows
        ),
    }


def _pool_box_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    truth = sum(int(row["truthBoxes"]) for row in rows)
    predicted = sum(int(row["predictedBoxes"]) for row in rows)
    matched = sum(int(row["matchedBoxes"]) for row in rows)
    denominator = 2 * matched + (predicted - matched) + (truth - matched)
    return {
        "minimumIou": rows[0]["minimumIou"],
        "truthBoxes": truth,
        "predictedBoxes": predicted,
        "matchedBoxes": matched,
        "precision": _ratio(matched, predicted),
        "recall": _ratio(matched, truth),
        "f1": 2 * matched / denominator if denominator else None,
    }


def _pool_center_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    truth = sum(int(row["truthBoxes"]) for row in rows)
    matched = sum(int(row["matchedTruthBoxes"]) for row in rows)
    return {
        "truthBoxes": truth,
        "framesWithTruth": sum(int(row["framesWithTruth"]) for row in rows),
        "matchedTruthBoxes": matched,
        "recall": _ratio(matched, truth),
        "toleranceRule": rows[0]["toleranceRule"],
    }


def _pool_false_detection_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ball_free = sum(int(row["ballFreeFrames"]) for row in rows)
    false_detections = sum(int(row["falseDetections"]) for row in rows)
    false_frames = sum(int(row["framesWithFalseDetection"]) for row in rows)
    return {
        "ballFreeFrames": ball_free,
        "falseDetections": false_detections,
        "framesWithFalseDetection": false_frames,
        "falseDetectionsPer1000BallFreeFrames": (
            false_detections * 1000.0 / ball_free if ball_free else None
        ),
        "framesWithFalseDetectionPer1000BallFreeFrames": (
            false_frames * 1000.0 / ball_free if ball_free else None
        ),
    }


def _pool_operating_metrics(
    rows: Sequence[Mapping[str, Any]],
    frames: Sequence[EvaluationFrame],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot pool an empty operating-metric sequence")
    localization: dict[str, Any] = {}
    for target in ("primaryLocalizable", "anyAnnotatedRealBall"):
        localization[target] = {
            "atIou025": _pool_box_metrics(
                [row["localization"][target]["atIou025"] for row in rows]
            ),
            "atIou050": _pool_box_metrics(
                [row["localization"][target]["atIou050"] for row in rows]
            ),
            "centerMatch": _pool_center_metrics(
                [row["localization"][target]["centerMatch"] for row in rows]
            ),
        }
    return {
        "threshold": "selected independently within each source-group fold",
        "framePresence": {
            target: _pool_classification_metrics(
                [row["framePresence"][target] for row in rows]
            )
            for target in ("primaryLocalizable", "anyAnnotatedRealBall")
        },
        "calibration": {
            "primaryLocalizableBrier": _brier(frames, target="primary"),
            "anyAnnotatedRealBallBrier": _brier(frames, target="any"),
            "probabilityField": "suggestions.frames[*].ballPresenceProbability",
        },
        "localization": localization,
        "rankedAveragePrecision": {
            "reported": False,
            "reason": (
                "ranked AP is threshold-independent and is reported at the all-development "
                "operating point, not pooled across unequal fold thresholds"
            ),
        },
        "falseDetectionsOnTrulyBallFreeFrames": _pool_false_detection_metrics(
            [row["falseDetectionsOnTrulyBallFreeFrames"] for row in rows]
        ),
    }


def _source_group_leave_one_out(
    frames: Sequence[EvaluationFrame],
    thresholds: Sequence[float] | None,
) -> dict[str, Any]:
    groups = sorted({frame.source_group for frame in frames})
    if len(groups) < 2:
        return {
            "available": False,
            "reason": "at least two source groups are required",
            "folds": [],
            "pooledOutOfFold": None,
            "groupedOutOfFold": {"sourceGroup": {}, "environment": {}},
        }
    folds: list[dict[str, Any]] = []
    held_metrics: list[dict[str, Any]] = []
    source_metrics: dict[str, Any] = {}
    environment_metric_parts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    environment_frames: dict[str, list[EvaluationFrame]] = defaultdict(list)
    for held_group in groups:
        training = tuple(frame for frame in frames if frame.source_group != held_group)
        held = tuple(frame for frame in frames if frame.source_group == held_group)
        training_sweep = [
            _sweep_row(training, threshold)
            for threshold in _threshold_candidates(training, thresholds)
        ]
        selected, selection_status = _precision_constrained_selection(training_sweep)
        selected_threshold = float(selected["threshold"])
        metrics = evaluate_frames_at_threshold(
            held,
            selected_threshold,
            include_ranked_average_precision=False,
        )
        held_metrics.append(metrics)
        source_metrics[held_group] = metrics
        environments = sorted({frame.environment for frame in held})
        for environment in environments:
            subset = tuple(frame for frame in held if frame.environment == environment)
            environment_metric_parts[environment].append(
                evaluate_frames_at_threshold(
                    subset,
                    selected_threshold,
                    include_ranked_average_precision=False,
                )
            )
            environment_frames[environment].extend(subset)
        folds.append(
            {
                "heldOutSourceGroup": held_group,
                "trainingSourceGroups": [group for group in groups if group != held_group],
                "trainingFrames": len(training),
                "heldOutFrames": len(held),
                "selectionStatus": selection_status,
                "selectedThreshold": selected_threshold,
                "trainingSelectionMetrics": selected,
                "heldOutMetrics": metrics,
            }
        )
    environment_metrics = {
        environment: _pool_operating_metrics(
            metric_rows,
            environment_frames[environment],
        )
        for environment, metric_rows in sorted(environment_metric_parts.items())
    }
    return {
        "available": True,
        "selectionRule": (
            "on all non-held source groups require primary frame precision >= 0.85 and "
            "maximize recall; use the diagnostic F1 optimum only when no threshold qualifies"
        ),
        "folds": folds,
        "pooledOutOfFold": _pool_operating_metrics(held_metrics, frames),
        "groupedOutOfFold": {
            "sourceGroup": source_metrics,
            "environment": environment_metrics,
        },
    }


def _out_of_fold_quality_gate(loo: Mapping[str, Any]) -> dict[str, Any]:
    if not loo["available"]:
        return {
            "passes": False,
            "criteria": [
                _criterion("sourceGroupLeaveOneOutAvailable", False, True, False)
            ],
            "failures": ["sourceGroupLeaveOneOutAvailable"],
        }
    pooled = loo["pooledOutOfFold"]["framePresence"]["primaryLocalizable"]
    source_recalls = {
        group: metrics["framePresence"]["primaryLocalizable"]["recall"]
        for group, metrics in loo["groupedOutOfFold"]["sourceGroup"].items()
    }
    environment_recalls = {
        environment: metrics["framePresence"]["primaryLocalizable"]["recall"]
        for environment, metrics in loo["groupedOutOfFold"]["environment"].items()
    }
    pooled_precision = pooled["precision"]
    pooled_recall = pooled["recall"]
    source_passes = bool(source_recalls) and all(
        value is not None and float(value) >= 0.35 for value in source_recalls.values()
    )
    environment_passes = bool(environment_recalls) and all(
        value is not None and float(value) >= 0.45
        for value in environment_recalls.values()
    )
    criteria = [
        _criterion(
            "pooledPrimaryPrecision",
            pooled_precision,
            ">= 0.85",
            pooled_precision is not None and float(pooled_precision) >= 0.85,
        ),
        _criterion(
            "pooledPrimaryRecall",
            pooled_recall,
            ">= 0.60",
            pooled_recall is not None and float(pooled_recall) >= 0.60,
        ),
        _criterion(
            "everySourceGroupPrimaryRecall",
            source_recalls,
            ">= 0.35 for every source group",
            source_passes,
        ),
        _criterion(
            "everyEnvironmentPrimaryRecall",
            environment_recalls,
            ">= 0.45 for every environment",
            environment_passes,
        ),
    ]
    failures = [item["name"] for item in criteria if not item["passes"]]
    return {
        "passes": not failures,
        "criteria": criteria,
        "failures": failures,
        "meaning": "out-of-fold detector quality only; downstream utility is separate",
    }


def _threshold_freeze_gate(
    coverage_gate: Mapping[str, Any],
    quality_gate: Mapping[str, Any],
    all_development_selection_status: str,
    loo: Mapping[str, Any],
) -> dict[str, Any]:
    fold_statuses = {
        fold["heldOutSourceGroup"]: fold["selectionStatus"]
        for fold in loo.get("folds", [])
    }
    every_fold_constrained = bool(fold_statuses) and all(
        status == "primary-precision-constrained"
        for status in fold_statuses.values()
    )
    criteria = [
        _criterion(
            "coverageSufficient",
            coverage_gate["coverageSufficient"],
            True,
            coverage_gate["coverageSufficient"] is True,
        ),
        _criterion(
            "outOfFoldQualityPassed",
            quality_gate["passes"],
            True,
            quality_gate["passes"] is True,
        ),
        _criterion(
            "allDevelopmentSelectionConstrained",
            all_development_selection_status,
            "primary-precision-constrained",
            all_development_selection_status == "primary-precision-constrained",
        ),
        _criterion(
            "everyFoldSelectionConstrained",
            fold_statuses,
            "primary-precision-constrained for every source-group fold",
            every_fold_constrained,
        ),
    ]
    failures = [item["name"] for item in criteria if not item["passes"]]
    return {
        "passes": not failures,
        "criteria": criteria,
        "failures": failures,
        "meaning": (
            "No all-development threshold is frozen when any development or source-group "
            "selection had to fall back from the primary-precision constraint."
        ),
    }


def _grouped_metrics(
    frames: Sequence[EvaluationFrame],
    threshold: float,
    key: Callable[[EvaluationFrame], str],
) -> dict[str, Any]:
    grouped: dict[str, list[EvaluationFrame]] = defaultdict(list)
    for frame in frames:
        grouped[key(frame)].append(frame)
    return {
        name: evaluate_frames_at_threshold(items, threshold)
        for name, items in sorted(grouped.items())
    }


def _size_bin(truth: TruthObject, frame: EvaluationFrame) -> str:
    diameter = _annotated_diameter_pixels(truth, frame)
    for name, lower, upper in SIZE_BINS_PIXELS:
        if lower <= diameter < upper:
            return name
    raise AssertionError("size bins do not cover a finite box diameter")


def _positive_object_slices(
    frames: Sequence[EvaluationFrame],
    threshold: float,
    *,
    dimension: str,
    target: str,
) -> dict[str, Any]:
    if dimension == "visibility":
        category = lambda truth, frame: truth.visibility
    elif dimension == "size":
        category = _size_bin
    else:
        raise ValueError(f"unknown positive-object slice dimension: {dimension}")
    categories = sorted(
        {
            category(truth, frame)
            for frame in frames
            for truth in _truth_for(frame, target)
        }
    )
    result: dict[str, Any] = {}
    for name in categories:
        truth_filter = lambda truth, frame, selected=name: category(truth, frame) == selected
        truth_count = sum(
            truth_filter(truth, frame)
            for frame in frames
            for truth in _truth_for(frame, target)
        )
        frames_with_truth = sum(
            any(truth_filter(truth, frame) for truth in _truth_for(frame, target))
            for frame in frames
        )
        iou_metrics: dict[str, Any] = {}
        for field, minimum_iou in (("atIou025", 0.25), ("atIou050", 0.50)):
            matched = 0
            for frame in frames:
                truths = tuple(
                    truth
                    for truth in _truth_for(frame, target)
                    if truth_filter(truth, frame)
                )
                matched += len(
                    _maximum_cardinality_matches(
                        _predictions_at(frame, threshold),
                        truths,
                        lambda prediction, truth, cutoff=minimum_iou: bbox_iou(
                            prediction.bbox, truth.bbox
                        )
                        >= cutoff,
                        lambda prediction, truth: bbox_iou(prediction.bbox, truth.bbox),
                    )
                )
            iou_metrics[field] = {
                "minimumIou": minimum_iou,
                "matchedTruthBoxes": matched,
                "truthBoxes": truth_count,
                "recall": _ratio(matched, truth_count),
            }
        result[name] = {
            "truthBoxes": truth_count,
            "framesWithTruth": frames_with_truth,
            **iou_metrics,
            "centerMatch": _center_match_metrics(
                frames,
                threshold,
                target=target,
                truth_filter=truth_filter,
            ),
        }
    return result


_MACRO_FIELDS = {
    "primaryPresencePrecision": ("framePresence", "primaryLocalizable", "precision"),
    "primaryPresenceRecall": ("framePresence", "primaryLocalizable", "recall"),
    "primaryPresenceF1": ("framePresence", "primaryLocalizable", "f1"),
    "anyRealBallPresencePrecision": ("framePresence", "anyAnnotatedRealBall", "precision"),
    "anyRealBallPresenceRecall": ("framePresence", "anyAnnotatedRealBall", "recall"),
    "anyRealBallPresenceF1": ("framePresence", "anyAnnotatedRealBall", "f1"),
    "primaryBrier": ("calibration", "primaryLocalizableBrier"),
    "anyRealBallBrier": ("calibration", "anyAnnotatedRealBallBrier"),
    "primaryBoxRecallIou025": ("localization", "primaryLocalizable", "atIou025", "recall"),
    "primaryBoxRecallIou050": ("localization", "primaryLocalizable", "atIou050", "recall"),
    "primaryCenterMatchRecall": ("localization", "primaryLocalizable", "centerMatch", "recall"),
    "anyRealBallBoxRecallIou025": (
        "localization",
        "anyAnnotatedRealBall",
        "atIou025",
        "recall",
    ),
    "anyRealBallBoxRecallIou050": (
        "localization",
        "anyAnnotatedRealBall",
        "atIou050",
        "recall",
    ),
    "anyRealBallCenterMatchRecall": (
        "localization",
        "anyAnnotatedRealBall",
        "centerMatch",
        "recall",
    ),
    "falseDetectionsPer1000BallFreeFrames": (
        "falseDetectionsOnTrulyBallFreeFrames",
        "falseDetectionsPer1000BallFreeFrames",
    ),
}


def _nested(metrics: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = metrics
    for key in path:
        value = value[key]
    return value


def _macro_summary(
    frames: Sequence[EvaluationFrame],
    threshold: float,
    key: Callable[[EvaluationFrame], str],
    *,
    unit_name: str,
) -> dict[str, Any]:
    grouped: dict[str, list[EvaluationFrame]] = defaultdict(list)
    for frame in frames:
        grouped[key(frame)].append(frame)
    unit_metrics = [
        evaluate_frames_at_threshold(
            items,
            threshold,
            include_ranked_average_precision=False,
        )
        for _, items in sorted(grouped.items())
    ]
    means: dict[str, float | None] = {}
    eligible: dict[str, int] = {}
    for name, path in _MACRO_FIELDS.items():
        values = [_nested(metrics, path) for metrics in unit_metrics]
        finite = [float(value) for value in values if value is not None]
        means[name] = fmean(finite) if finite else None
        eligible[name] = len(finite)
    return {
        "unit": unit_name,
        "units": len(grouped),
        "mean": means,
        "eligibleUnitsByMetric": eligible,
        "undefinedUnitMetricsAreExcluded": True,
    }


def _sol_prediction_frames(
    human_frames: Sequence[EvaluationFrame],
    sol_by_key: Mapping[str, SolAnnotationFrame],
    *,
    target: str,
) -> tuple[EvaluationFrame, ...]:
    transformed: list[EvaluationFrame] = []
    for human in human_frames:
        sol = sol_by_key.get(human.key)
        if sol is None:
            raise BallDetectorEvaluationError(
                f"Sol review is missing quality-eligible frame {human.key}"
            )
        sol_objects = sol.primary_objects if target == "primary" else sol.objects
        detections = tuple(Detection(item.bbox, 1.0) for item in sol_objects)
        transformed.append(
            EvaluationFrame(
                key=human.key,
                task_id=human.task_id,
                recording_id=human.recording_id,
                source_group=human.source_group,
                environment=human.environment,
                split=human.split,
                window_key=human.window_key,
                window_id=human.window_id,
                stratum=human.stratum,
                width=human.width,
                height=human.height,
                primary_ball_state=human.primary_ball_state,
                truth_objects=human.truth_objects,
                ball_presence_probability=float(bool(sol_objects)),
                detections=detections,
                proposal_exposure=human.proposal_exposure,
            )
        )
    return tuple(transformed)


def _sol_fixed_point_metrics(
    human_frames: Sequence[EvaluationFrame],
    sol_by_key: Mapping[str, SolAnnotationFrame],
) -> dict[str, Any]:
    primary_frames = _sol_prediction_frames(human_frames, sol_by_key, target="primary")
    any_frames = _sol_prediction_frames(human_frames, sol_by_key, target="any")
    primary = evaluate_frames_at_threshold(
        primary_frames,
        0.5,
        include_ranked_average_precision=False,
    )
    any_ball = evaluate_frames_at_threshold(
        any_frames,
        0.5,
        include_ranked_average_precision=False,
    )
    eligible_sol = [sol_by_key[frame.key] for frame in human_frames]
    objects = [item for frame in eligible_sol for item in frame.objects]
    state_counts = {
        state: sum(frame.primary_ball_state == state for frame in eligible_sol)
        for state in (
            "localizable",
            "fully_occluded",
            "out_of_frame",
            "indeterminate",
        )
    }
    role_counts = {
        role: sum(item.role == role for item in objects)
        for role in ("primary-court", "other-court", "unknown")
    }
    role_assigned = role_counts["primary-court"] + role_counts["other-court"]
    localization: dict[str, Any] = {
        "primaryLocalizable": copy.deepcopy(
            primary["localization"]["primaryLocalizable"]
        ),
        "anyAnnotatedRealBall": copy.deepcopy(
            any_ball["localization"]["anyAnnotatedRealBall"]
        ),
    }
    for target_metrics in localization.values():
        target_metrics["rankedAveragePrecision"] = {
            "reported": False,
            "reason": (
                "Sol annotations contain no calibrated per-box confidence ranking; these "
                "are fixed-point boxes, so calling this AP would be incorrect"
            ),
        }
    return {
        "operatingPoint": "one deterministic annotation set; no confidence threshold",
        "decisionRules": {
            "primaryLocalizable": (
                "positive when Sol supplied a primary-court object; unknown-role objects "
                "are intentionally not promoted to primary"
            ),
            "anyAnnotatedRealBall": "positive when Sol supplied any volleyball object",
        },
        "framePresence": {
            "primaryLocalizable": primary["framePresence"]["primaryLocalizable"],
            "anyAnnotatedRealBall": any_ball["framePresence"][
                "anyAnnotatedRealBall"
            ],
        },
        "localization": localization,
        "falseDetectionsOnTrulyBallFreeFrames": any_ball[
            "falseDetectionsOnTrulyBallFreeFrames"
        ],
        "annotationCoverage": {
            "qualityEligibleHumanFrames": len(human_frames),
            "solPrimaryBallStateCounts": state_counts,
            "solObjects": len(objects),
            "roleCounts": role_counts,
            "roleAssignedObjects": role_assigned,
            "roleAssignmentRate": _ratio(role_assigned, len(objects)),
            "framesWithUnknownRoleObject": sum(
                any(item.role == "unknown" for item in frame.objects)
                for frame in eligible_sol
            ),
        },
        "calibration": {
            "reported": False,
            "brierReported": False,
            "reason": (
                "the current Sol review schema stores categorical states and boxes but no "
                "calibrated frame probability"
            ),
        },
    }


_SOL_MACRO_FIELDS = {
    name: path
    for name, path in _MACRO_FIELDS.items()
    if name not in {"primaryBrier", "anyRealBallBrier"}
}


def _sol_grouped_metrics(
    frames: Sequence[EvaluationFrame],
    sol_by_key: Mapping[str, SolAnnotationFrame],
    key: Callable[[EvaluationFrame], str],
) -> dict[str, Any]:
    grouped: dict[str, list[EvaluationFrame]] = defaultdict(list)
    for frame in frames:
        grouped[key(frame)].append(frame)
    return {
        name: _sol_fixed_point_metrics(items, sol_by_key)
        for name, items in sorted(grouped.items())
    }


def _sol_macro_summary(
    frames: Sequence[EvaluationFrame],
    sol_by_key: Mapping[str, SolAnnotationFrame],
    key: Callable[[EvaluationFrame], str],
    *,
    unit_name: str,
) -> dict[str, Any]:
    grouped: dict[str, list[EvaluationFrame]] = defaultdict(list)
    for frame in frames:
        grouped[key(frame)].append(frame)
    unit_metrics = [
        _sol_fixed_point_metrics(items, sol_by_key)
        for _, items in sorted(grouped.items())
    ]
    means: dict[str, float | None] = {}
    eligible: dict[str, int] = {}
    for name, path in _SOL_MACRO_FIELDS.items():
        values = [_nested(metrics, path) for metrics in unit_metrics]
        finite = [float(value) for value in values if value is not None]
        means[name] = fmean(finite) if finite else None
        eligible[name] = len(finite)
    return {
        "unit": unit_name,
        "units": len(grouped),
        "mean": means,
        "eligibleUnitsByMetric": eligible,
        "undefinedUnitMetricsAreExcluded": True,
    }


def _sol_comparison(
    human: LoadedEvaluation,
    sol: LoadedSolReviews | None,
) -> dict[str, Any]:
    if sol is None:
        return {
            "available": False,
            "reason": "no --sol-task inputs were supplied",
        }
    sol_by_key = {frame.key: frame for frame in sol.frames}
    overall = _sol_fixed_point_metrics(human.frames, sol_by_key)
    primary_frames = _sol_prediction_frames(human.frames, sol_by_key, target="primary")
    any_frames = _sol_prediction_frames(human.frames, sol_by_key, target="any")
    identities = sorted(
        {
            (
                task["solReviewProvenance"]["reviewer"]["agentId"],
                task["solReviewProvenance"]["reviewer"]["modelId"],
                task["solReviewProvenance"]["reviewer"]["runId"],
            )
            for task in sol.tasks
        }
    )
    return {
        "available": True,
        "population": (
            "the same proposal-unexposed human-truth frames used for detector quality metrics"
        ),
        "confidence": {
            "available": False,
            "thresholdSweepReported": False,
            "brierReported": False,
            "rankedAveragePrecisionReported": False,
            "caveat": (
                "Sol boxes have no calibrated scores. Precision/recall/F1, IoU matching, "
                "and center recall are one fixed operating point; AP and Brier are not "
                "defined and are not imitated with unit scores."
            ),
        },
        "roleHandling": (
            "primary metrics use only Sol objects explicitly labeled primary-court; "
            "unknown-role objects participate only in any-real-ball metrics"
        ),
        "overallMicro": overall,
        "grouped": {
            "sourceGroup": _sol_grouped_metrics(
                human.frames, sol_by_key, lambda frame: frame.source_group
            ),
            "environment": _sol_grouped_metrics(
                human.frames, sol_by_key, lambda frame: frame.environment
            ),
            "stratum": _sol_grouped_metrics(
                human.frames, sol_by_key, lambda frame: frame.stratum
            ),
            "visibility": {
                "primaryLocalizable": _positive_object_slices(
                    primary_frames,
                    0.5,
                    dimension="visibility",
                    target="primary",
                ),
                "anyAnnotatedRealBall": _positive_object_slices(
                    any_frames,
                    0.5,
                    dimension="visibility",
                    target="any",
                ),
            },
            "size": {
                "primaryLocalizable": _positive_object_slices(
                    primary_frames,
                    0.5,
                    dimension="size",
                    target="primary",
                ),
                "anyAnnotatedRealBall": _positive_object_slices(
                    any_frames,
                    0.5,
                    dimension="size",
                    target="any",
                ),
            },
        },
        "macro": {
            "byWindow": _sol_macro_summary(
                human.frames,
                sol_by_key,
                lambda frame: frame.window_key,
                unit_name="three-second annotation window",
            ),
            "byRecording": _sol_macro_summary(
                human.frames,
                sol_by_key,
                lambda frame: frame.recording_id,
                unit_name="recording",
            ),
            "bySourceGroup": _sol_macro_summary(
                human.frames,
                sol_by_key,
                lambda frame: frame.source_group,
                unit_name="source group",
            ),
            "correlationWarning": (
                "The 15 fps frames within each three-second window are temporally "
                "correlated; frame-micro results are not independent trials."
            ),
        },
        "provenance": {
            "tasks": list(sol.tasks),
            "reviewerRuns": [
                {"agentId": agent, "modelId": model, "runId": run}
                for agent, model, run in identities
            ],
            "binding": (
                "each Sol task immutable digest and detector-empty source SHA exactly match "
                "the corresponding human/detector task and receipt-verified indexed initial "
                "task"
            ),
            "attestationCaveat": (
                "the receipt and hashes prove artifact integrity, chronology, and that the "
                "prepared input file had no proposal channel; without a trusted signed run "
                "attestation they cannot prove the model received no out-of-band detector "
                "information"
            ),
            "allSampledFramesLabeled": True,
            "qualityEligibleHumanFrames": len(human.frames),
            "humanFramesExcludedForProposalExposure": (
                human.exposure_summary["shown_before_label_finalized"]
                + human.exposure_summary["not_recorded"]
            ),
        },
    }


def _criterion(name: str, observed: Any, required: Any, passes: bool) -> dict[str, Any]:
    return {"name": name, "observed": observed, "required": required, "passes": passes}


def _read_pilot_index(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    index_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BallDetectorEvaluationError(f"cannot read pilot index {index_path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("artifactType") != "volleycut-ball-presence-pilot-index":
        raise BallDetectorEvaluationError(f"not a ball-presence pilot index: {index_path}")
    result = copy.deepcopy(payload)
    result["pathHint"] = str(index_path)
    result["fileSha256"] = _sha256_file(index_path)
    return result


def _validate_pilot_index_binding(
    loaded: LoadedEvaluation,
    pilot_index: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schemaVersion",
        "artifactType",
        "manifest",
        "samplingPolicyId",
        "round",
        "developmentOnly",
        "recordingCount",
        "windowCount",
        "frameCount",
        "tasks",
    }
    missing = required - set(pilot_index)
    if missing:
        raise BallDetectorEvaluationError(
            f"pilot index is missing exact-binding fields: {sorted(missing)}"
        )
    if (
        pilot_index["schemaVersion"] != 1
        or pilot_index["artifactType"] != "volleycut-ball-presence-pilot-index"
        or pilot_index["developmentOnly"] is not True
    ):
        raise BallDetectorEvaluationError(
            "pilot index schemaVersion/artifactType/developmentOnly provenance is invalid"
        )
    if (
        not isinstance(pilot_index["samplingPolicyId"], str)
        or not pilot_index["samplingPolicyId"]
        or not isinstance(pilot_index["round"], int)
        or isinstance(pilot_index["round"], bool)
        or pilot_index["round"] < 1
    ):
        raise BallDetectorEvaluationError(
            "pilot index samplingPolicyId or round is invalid"
        )
    for field in ("recordingCount", "windowCount", "frameCount"):
        if (
            not isinstance(pilot_index[field], int)
            or isinstance(pilot_index[field], bool)
            or pilot_index[field] < 1
        ):
            raise BallDetectorEvaluationError(f"pilot index {field} is invalid")
    manifest = pilot_index["manifest"]
    if (
        not isinstance(manifest, dict)
        or manifest.get("sha256") != loaded.manifest_sha256
    ):
        raise BallDetectorEvaluationError("pilot index manifest SHA-256 does not match tasks")

    tasks_by_recording = {item["recordingId"]: item for item in loaded.tasks}
    if len(tasks_by_recording) != len(loaded.tasks):
        raise BallDetectorEvaluationError(
            "evaluation tasks contain duplicate recording identifiers"
        )
    indexed_rows = pilot_index["tasks"]
    if not isinstance(indexed_rows, list):
        raise BallDetectorEvaluationError("pilot index tasks must be an array")
    index_by_recording: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(indexed_rows):
        if not isinstance(row, dict):
            raise BallDetectorEvaluationError(f"pilot index tasks[{index}] is not an object")
        recording_id = row.get("recordingId")
        if not isinstance(recording_id, str) or not recording_id:
            raise BallDetectorEvaluationError(
                f"pilot index tasks[{index}].recordingId is invalid"
            )
        if recording_id in index_by_recording:
            raise BallDetectorEvaluationError(
                f"pilot index repeats recordingId {recording_id!r}"
            )
        for field in ("taskId", "frameCount", "initialTaskSha256"):
            if field not in row:
                raise BallDetectorEvaluationError(
                    f"pilot index tasks[{index}] is missing {field}"
                )
        if not isinstance(row["taskId"], str) or not row["taskId"]:
            raise BallDetectorEvaluationError(
                f"pilot index tasks[{index}].taskId is invalid"
            )
        if (
            not isinstance(row["frameCount"], int)
            or isinstance(row["frameCount"], bool)
            or row["frameCount"] < 1
        ):
            raise BallDetectorEvaluationError(
                f"pilot index tasks[{index}].frameCount is invalid"
            )
        if not _valid_sha256(row["initialTaskSha256"]):
            raise BallDetectorEvaluationError(
                f"pilot index tasks[{index}].initialTaskSha256 is invalid"
            )
        index_by_recording[recording_id] = row
    if set(index_by_recording) != set(tasks_by_recording):
        raise BallDetectorEvaluationError(
            "pilot index recording set does not exactly match evaluation tasks"
        )

    task_bindings: list[dict[str, Any]] = []
    for recording_id, task in sorted(tasks_by_recording.items()):
        indexed = index_by_recording[recording_id]
        comparisons = {
            "taskId": (task["taskId"], indexed["taskId"]),
            "frameCount": (task["frameCount"], indexed["frameCount"]),
            "initialTaskSha256": (
                task["initialTaskSha256"],
                indexed["initialTaskSha256"],
            ),
        }
        mismatches = [
            field for field, (actual, expected) in comparisons.items() if actual != expected
        ]
        if indexed.get("split", task["split"]) != task["split"]:
            mismatches.append("split")
        if mismatches:
            raise BallDetectorEvaluationError(
                f"pilot index binding mismatch for {recording_id}: {mismatches}"
            )
        task_bindings.append(
            {
                "recordingId": recording_id,
                "taskId": task["taskId"],
                "frameCount": task["frameCount"],
                "initialTaskSha256": task["initialTaskSha256"],
                "initialTaskFileVerified": task["blindMergeProvenance"][
                    "initialTask"
                ]["fileVerified"],
                "proposalSourceSha256": task["blindMergeProvenance"][
                    "detectorProposals"
                ]["sha256"],
                "reviewedSourceSha256": task["blindMergeProvenance"][
                    "reviewedLabels"
                ]["sha256"],
            }
        )

    sampling_policies = {item["samplingPolicyId"] for item in loaded.tasks}
    rounds = {item["round"] for item in loaded.tasks}
    if sampling_policies != {pilot_index["samplingPolicyId"]}:
        raise BallDetectorEvaluationError(
            "pilot index samplingPolicyId does not exactly match every task"
        )
    if rounds != {pilot_index["round"]}:
        raise BallDetectorEvaluationError(
            "pilot index round does not exactly match every task"
        )
    aggregate = {
        "recordingCount": len(loaded.tasks),
        "windowCount": sum(int(item["windowCount"]) for item in loaded.tasks),
        "frameCount": sum(int(item["frameCount"]) for item in loaded.tasks),
    }
    for field, actual in aggregate.items():
        if pilot_index[field] != actual:
            raise BallDetectorEvaluationError(
                f"pilot index {field}={pilot_index[field]!r} does not match {actual}"
            )
    return {
        "exact": True,
        "samplingPolicyId": pilot_index["samplingPolicyId"],
        "round": pilot_index["round"],
        **aggregate,
        "taskBindings": task_bindings,
        "initialTaskHashChain": (
            "index initialTaskSha256 == detector model sourceTask.sha256 inside the "
            "SHA-pinned proposal source named by blindMergeProvenance"
        ),
    }


def _coverage_gate(
    loaded: LoadedEvaluation,
    requirements: ProtocolRequirements,
    pilot_index: Mapping[str, Any] | None,
    pilot_index_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    frames = loaded.frames
    recordings = {frame.recording_id for frame in frames}
    source_groups = {frame.source_group for frame in frames}
    windows = {frame.window_key for frame in frames}
    strata = {frame.stratum for frame in frames}
    primary_positive = [frame for frame in frames if frame.primary_truth]
    any_positive = [frame for frame in frames if frame.any_ball_truth]
    ball_free = list(_truly_ball_free_frames(frames))
    primary_positive_windows = {frame.window_key for frame in primary_positive}
    any_positive_windows = {frame.window_key for frame in any_positive}
    ball_free_windows = {frame.window_key for frame in ball_free}
    primary_positive_groups = {frame.source_group for frame in primary_positive}
    ball_free_groups = {frame.source_group for frame in ball_free}
    criteria = [
        _criterion(
            "recordings",
            len(recordings),
            requirements.min_recordings,
            len(recordings) >= requirements.min_recordings,
        ),
        _criterion(
            "sourceGroups",
            len(source_groups),
            requirements.min_source_groups,
            len(source_groups) >= requirements.min_source_groups,
        ),
        _criterion(
            "windows",
            len(windows),
            requirements.min_windows,
            len(windows) >= requirements.min_windows,
        ),
        _criterion(
            "reviewedFrames",
            len(frames),
            requirements.min_reviewed_frames,
            len(frames) >= requirements.min_reviewed_frames,
        ),
        _criterion(
            "requiredStrata",
            sorted(strata),
            list(STRATA),
            set(STRATA).issubset(strata),
        ),
        _criterion(
            "primaryPositiveFrames",
            len(primary_positive),
            requirements.min_primary_positive_frames,
            len(primary_positive) >= requirements.min_primary_positive_frames,
        ),
        _criterion(
            "anyBallPositiveFrames",
            len(any_positive),
            requirements.min_any_ball_positive_frames,
            len(any_positive) >= requirements.min_any_ball_positive_frames,
        ),
        _criterion(
            "trulyBallFreeFrames",
            len(ball_free),
            requirements.min_ball_free_frames,
            len(ball_free) >= requirements.min_ball_free_frames,
        ),
        _criterion(
            "primaryPositiveWindows",
            len(primary_positive_windows),
            requirements.min_primary_positive_windows,
            len(primary_positive_windows) >= requirements.min_primary_positive_windows,
        ),
        _criterion(
            "anyBallPositiveWindows",
            len(any_positive_windows),
            requirements.min_any_ball_positive_windows,
            len(any_positive_windows) >= requirements.min_any_ball_positive_windows,
        ),
        _criterion(
            "ballFreeWindows",
            len(ball_free_windows),
            requirements.min_ball_free_windows,
            len(ball_free_windows) >= requirements.min_ball_free_windows,
        ),
        _criterion(
            "primaryPositiveSourceGroups",
            len(primary_positive_groups),
            requirements.min_primary_positive_source_groups,
            len(primary_positive_groups)
            >= requirements.min_primary_positive_source_groups,
        ),
        _criterion(
            "ballFreeSourceGroups",
            len(ball_free_groups),
            requirements.min_ball_free_source_groups,
            len(ball_free_groups) >= requirements.min_ball_free_source_groups,
        ),
    ]
    if pilot_index is None:
        criteria.append(
            _criterion(
                "pilotIndexCoverage",
                "not supplied",
                "supply the immutable pilot index and cover every indexed development task",
                False,
            )
        )
    else:
        if pilot_index_binding is None:
            raise AssertionError("pilot index binding must be validated before coverage")
        criteria.append(
            _criterion(
                "pilotIndexExactBinding",
                pilot_index_binding["exact"],
                True,
                pilot_index_binding["exact"] is True,
            )
        )
    failures = [item["name"] for item in criteria if not item["passes"]]
    return {
        "coverageSufficient": not failures,
        "criteria": criteria,
        "failures": failures,
        "meaning": (
            "Coverage only; it cannot promote or freeze a threshold without out-of-fold "
            "quality and does not make correlated frames independent."
        ),
    }


def evaluate_ball_detector_tasks(
    task_inputs: Iterable[str | Path],
    *,
    sol_task_inputs: Iterable[str | Path] | None = None,
    thresholds: Sequence[float] | None = None,
    selection_target: str = "primary",
    requirements: ProtocolRequirements | None = None,
    pilot_index_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate reviewed development proposals and optionally promote a threshold."""

    if selection_target not in {"primary", "any"}:
        raise ValueError("selection_target must be 'primary' or 'any'")
    requirements = requirements or ProtocolRequirements()
    loaded = load_completed_reviewed_tasks(task_inputs)
    loaded_sol = (
        load_completed_sol_reviews(sol_task_inputs, loaded)
        if sol_task_inputs is not None
        else None
    )
    pilot_index = _read_pilot_index(pilot_index_path)
    if loaded_sol is not None and pilot_index is not None:
        mismatched_sol_indexes = [
            task["taskId"]
            for task in loaded_sol.tasks
            if task["solReviewProvenance"]["pilotIndex"]["sha256"]
            != pilot_index["fileSha256"]
        ]
        if mismatched_sol_indexes:
            raise BallDetectorEvaluationError(
                "Sol preparation receipts do not match the supplied evaluation pilot "
                f"index for task ids: {mismatched_sol_indexes}"
            )
    pilot_index_binding = (
        _validate_pilot_index_binding(loaded, pilot_index)
        if pilot_index is not None
        else None
    )
    threshold_values = _threshold_candidates(loaded.frames, thresholds)
    sweep = [_sweep_row(loaded.frames, threshold) for threshold in threshold_values]
    target_field = (
        "primaryLocalizable" if selection_target == "primary" else "anyAnnotatedRealBall"
    )
    diagnostic_best = _diagnostic_best(sweep, target_field)
    diagnostic_threshold = float(diagnostic_best["threshold"])
    all_development_selection, all_development_status = (
        _precision_constrained_selection(sweep)
    )
    candidate_threshold = float(all_development_selection["threshold"])
    gate = _coverage_gate(
        loaded,
        requirements,
        pilot_index,
        pilot_index_binding,
    )
    source_group_loo = _source_group_leave_one_out(loaded.frames, thresholds)
    quality_gate = _out_of_fold_quality_gate(source_group_loo)
    freeze_gate = _threshold_freeze_gate(
        gate,
        quality_gate,
        all_development_status,
        source_group_loo,
    )
    frozen_threshold = candidate_threshold if freeze_gate["passes"] else None
    promoted_threshold = None
    downstream_gate = {
        "complete": False,
        "eligible": False,
        "promoted": False,
        "components": {
            "coverage": {
                "available": True,
                "passes": gate["coverageSufficient"],
            },
            "outOfFoldFrameQuality": {
                "available": source_group_loo["available"],
                "passes": quality_gate["passes"],
            },
            "thresholdFreeze": {
                "available": True,
                "passes": freeze_gate["passes"],
            },
            "falseTrackGate": {
                "available": False,
                "passes": False,
                "reason": "no reviewed temporal false-track benchmark is available",
            },
            "throughputGate": {
                "available": False,
                "passes": False,
                "reason": "no end-to-end full-recording throughput acceptance gate is available",
            },
        },
        "missingRequiredGates": ["falseTrackGate", "throughputGate"],
        "meaning": (
            "Even a frozen detector threshold is not eligible for downstream model promotion "
            "until temporal false-track and throughput gates are measured and pass."
        ),
    }
    operating = evaluate_frames_at_threshold(loaded.frames, candidate_threshold)
    sol_comparison = _sol_comparison(loaded, loaded_sol)
    grouped = {
        "sourceGroup": _grouped_metrics(
            loaded.frames, candidate_threshold, lambda frame: frame.source_group
        ),
        "environment": _grouped_metrics(
            loaded.frames, candidate_threshold, lambda frame: frame.environment
        ),
        "stratum": _grouped_metrics(
            loaded.frames, candidate_threshold, lambda frame: frame.stratum
        ),
        "visibility": {
            "primaryLocalizable": _positive_object_slices(
                loaded.frames,
                candidate_threshold,
                dimension="visibility",
                target="primary",
            ),
            "anyAnnotatedRealBall": _positive_object_slices(
                loaded.frames,
                candidate_threshold,
                dimension="visibility",
                target="any",
            ),
        },
        "size": {
            "primaryLocalizable": _positive_object_slices(
                loaded.frames,
                candidate_threshold,
                dimension="size",
                target="primary",
            ),
            "anyAnnotatedRealBall": _positive_object_slices(
                loaded.frames,
                candidate_threshold,
                dimension="size",
                target="any",
            ),
        },
    }
    macro = {
        "byWindow": _macro_summary(
            loaded.frames,
            candidate_threshold,
            lambda frame: frame.window_key,
            unit_name="three-second annotation window",
        ),
        "byRecording": _macro_summary(
            loaded.frames,
            candidate_threshold,
            lambda frame: frame.recording_id,
            unit_name="recording",
        ),
        "bySourceGroup": _macro_summary(
            loaded.frames,
            candidate_threshold,
            lambda frame: frame.source_group,
            unit_name="source group",
        ),
        "correlationWarning": (
            "The 15 fps frames within each three-second window are temporally correlated. "
            "Frame-micro metrics describe sampled frames, not independent trials; window, "
            "recording, and source-group macro summaries are therefore reported separately."
        ),
    }
    implementation_path = Path(__file__).resolve()
    return {
        "schemaVersion": BALL_DETECTOR_EVALUATION_SCHEMA_VERSION,
        "kind": BALL_DETECTOR_EVALUATION_KIND,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "population": (
            "complete manually reviewed train/validation ball-pilot tasks, restricted to "
            "frames whose persisted proposalExposure is not_shown"
        ),
        "targetDefinitions": {
            "primaryLocalizable": (
                "positive exactly when primaryBallState is localizable and the reviewed frame "
                "contains its one primary-court volleyball box"
            ),
            "anyAnnotatedRealBall": (
                "positive when the reviewed frame contains any visible localizable volleyball "
                "object, regardless of primary-court, other-court, or unknown role"
            ),
            "trulyBallFree": (
                "primaryBallState is out_of_frame and there is no reviewed volleyball object "
                "of any role; fully_occluded and indeterminate frames are excluded"
            ),
        },
        "metricDefinitions": {
            "pointBoxMatching": "maximum-cardinality one-to-one matching within each frame",
            "boxThresholds": [0.25, 0.50],
            "centerMatchTolerance": (
                "max(4 px, annotated diameter), with diameter defined as the maximum annotated "
                "box dimension in source-proxy pixels"
            ),
            "brier": "mean squared error of stored frame ballPresenceProbability",
            "presenceProbability": (
                "validated as the maximum stored detection confidence, or zero when no "
                "proposal was retained"
            ),
            "indeterminateFrames": (
                "excluded from a target when absence is not reviewable; frames with an "
                "annotated non-primary ball remain evaluable for the any-real-ball target"
            ),
            "rankedAveragePrecision": (
                "true confidence-ranked all-point interpolated AP over stored proposals; "
                "not point precision and not a threshold average"
            ),
            "visibilityAndSizeSlices": (
                "positive-object recall slices; precision is intentionally omitted because "
                "false detections have no ground-truth visibility or size category"
            ),
            "sizeBinsPixels": [
                {
                    "name": name,
                    "minimumInclusive": lower,
                    "maximumExclusive": upper if math.isfinite(upper) else None,
                }
                for name, lower, upper in SIZE_BINS_PIXELS
            ],
        },
        "provenance": {
            "implementation": {
                "pathHint": str(implementation_path),
                "sha256": _sha256_file(implementation_path),
                "pythonVersion": platform.python_version(),
            },
            "manifestSha256": loaded.manifest_sha256,
            "detectorConfiguration": loaded.detector,
            "detectorConfigurationSha256": loaded.detector_signature_sha256,
            "tasks": list(loaded.tasks),
            "humanProposalExposure": {
                **loaded.exposure_summary,
                "eligibleValue": "not_shown",
                "excludedValues": [
                    "shown_before_label_finalized",
                    "not_recorded",
                ],
                "meaning": (
                    "blindMergeProvenance records how separate files were combined; it does "
                    "not assert that every human frame was proposal-blind"
                ),
            },
            "pilotIndex": pilot_index,
            "pilotIndexBinding": pilot_index_binding,
        },
        "coverage": {
            "recordings": len({frame.recording_id for frame in loaded.frames}),
            "sourceGroups": len({frame.source_group for frame in loaded.frames}),
            "windows": len({frame.window_key for frame in loaded.frames}),
            "frames": len(loaded.frames),
            "allHumanReviewedFrames": sum(loaded.exposure_summary.values()),
            "excludedAssistedHumanFrames": loaded.exposure_summary[
                "shown_before_label_finalized"
            ],
            "excludedHumanFramesWithoutExposureAudit": loaded.exposure_summary[
                "not_recorded"
            ],
            "primaryEvaluableFrames": sum(
                _is_evaluable(frame, "primary") for frame in loaded.frames
            ),
            "anyRealBallEvaluableFrames": sum(
                _is_evaluable(frame, "any") for frame in loaded.frames
            ),
            "splits": sorted({frame.split for frame in loaded.frames}),
        },
        "protocolRequirements": asdict(requirements),
        "protocolGate": gate,
        "sourceGroupLeaveOneOut": source_group_loo,
        "outOfFoldQualityGate": quality_gate,
        "thresholdFreezeGate": freeze_gate,
        "downstreamEligibility": downstream_gate,
        "thresholdSelection": {
            "diagnosticSelectionTarget": target_field,
            "diagnosticSelectionRule": (
                "maximize frame F1, then recall, precision, alternate-target F1, minimize false "
                "detections per 1000 truly ball-free frames, then prefer the higher threshold"
            ),
            "allDevelopmentFreezeSelectionRule": (
                "require primary frame precision >= 0.85, then maximize recall; use the "
                "diagnostic primary-F1 optimum when no threshold qualifies"
            ),
            "sweepPolicy": (
                "explicit supplied thresholds"
                if thresholds is not None
                else "every unique positive stored frame probability plus 1.0"
            ),
            "thresholdsEvaluated": len(sweep),
            "diagnosticBestThreshold": diagnostic_threshold,
            "candidateAllDevelopmentThreshold": candidate_threshold,
            "candidateAllDevelopmentSelectionStatus": all_development_status,
            "candidateAllDevelopmentMetrics": all_development_selection,
            "frozenAllDevelopmentThreshold": frozen_threshold,
            "promotedThreshold": promoted_threshold,
            "status": (
                "frozen-pending-downstream-gates"
                if frozen_threshold is not None
                else "diagnostic-only-not-frozen"
            ),
            "freezeRequires": (
                "coverage, OOF quality, constrained all-development selection, and constrained "
                "selection in every source-group fold"
            ),
            "promotionBlockedBy": ["falseTrackGate", "throughputGate"],
        },
        "thresholdSweep": sweep,
        "operatingPoint": {
            "threshold": candidate_threshold,
            "status": (
                "frozen-development-threshold-pending-downstream-gates"
                if frozen_threshold is not None
                else "all-development-candidate-not-frozen"
            ),
            "overallMicro": operating,
            "grouped": grouped,
            "macro": macro,
        },
        "solComparison": sol_comparison,
        "warnings": [
            (
                "All threshold selection and metrics are development-only; no "
                "test/challenge task is accepted."
            ),
            (
                "Detector suggestions remain model outputs even after a human independently "
                "reviews the labels."
            ),
            (
                "Human frames exposed to Sol or detector proposals before the label was "
                "finalized are excluded from detector selection and both detector/Sol "
                "quality metrics. Completed human inputs lacking the exposure field are "
                "rejected before blind merge."
            ),
            (
                "Coverage alone never freezes or promotes a threshold; source-group-held-out "
                "quality is required before an all-development threshold is frozen."
            ),
            (
                "False-track and throughput gates are unavailable, so downstream promotion "
                "is explicitly incomplete and false even if a threshold is frozen."
            ),
        ],
    }
