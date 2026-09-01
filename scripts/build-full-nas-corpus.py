#!/usr/bin/env python3
"""Build the deduplicated corpus manifest used by the side diagnostics.

The manifest joins the current completed labels, intake labeling workspace,
blind intake prelabels, and the raw-no-backup model-feedback videos.  It does
not mutate any label document.  Candidate-only rally windows are retained for
manual review, but downstream reports keep them out of gold metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis.artifacts import atomic_write_text
from analysis.exported_project_dataset import (
    load_exported_project_dataset,
    sha256_file as exported_sha256,
)
from analysis.nas_video_corpus import MANIFEST_KIND


DEFAULT_COMPLETED_ROOT = Path(
    "/mnt/freenas/volleycut/labeling-v1-2026-08-09/completed/full-v1"
)
DEFAULT_INTAKE_ROOT = Path("/mnt/freenas/volleycut/intake-2026-08-13")
DEFAULT_RAW_NO_BACKUP_ROOT = Path("/mnt/freenas/volleycut-raw-no-backup")
DEFAULT_EXPORTED_PROJECT_ROOT = Path(
    "/mnt/freenas/volleycut/exported-project-datasets"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{where} must be numeric")
    result = float(value)
    if result < 0:
        raise ValueError(f"{where} must not be negative")
    return result


def resolve_video(label_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label_path} has no recording.video")
    video = Path(value).expanduser()
    return (label_path.parent / video).resolve() if not video.is_absolute() else video.resolve()


def source_ignored_intervals(
    value: Any, *, duration: float, where: str
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{where} must be an array")
    result: list[dict[str, Any]] = []
    previous_end = -1.0
    for index, row in enumerate(value):
        item_where = f"{where}[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{item_where} must be an object")
        start = finite(row.get("start"), f"{item_where}.start")
        end = finite(row.get("end"), f"{item_where}.end")
        reason = row.get("reason")
        if (
            start < 0
            or start >= end
            or end > duration
            or start < previous_end
            or not isinstance(reason, str)
            or not reason
        ):
            raise ValueError(f"{item_where} is invalid or overlaps")
        result.append({"start": start, "end": end, "reason": reason})
        previous_end = end
    return result


def label_record(
    path: Path,
    *,
    source_type: str,
    priority: int,
    target_status: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recording = payload["recording"]
    recording_id = str(recording["id"])
    duration = finite(
        recording["durationSeconds"], f"{recording_id}.durationSeconds"
    )
    rallies = []
    for index, rally in enumerate(payload.get("rallies", []), start=1):
        notes = rally.get("notes")
        if notes is not None and not isinstance(notes, str):
            notes = None
        tags = rally.get("tags", [])
        rallies.append(
            {
                "index": index,
                "start": finite(rally["start"], f"{recording_id}.rallies[{index}].start"),
                "end": finite(rally["end"], f"{recording_id}.rallies[{index}].end"),
                "notes": notes,
                "tags": [tag for tag in tags if isinstance(tag, str)],
            }
        )
    switches = [
        {"time": finite(row["time"], f"{recording_id}.sideSwitches.time")}
        for row in payload.get("sideSwitches", [])
    ]
    serve_markers = [
        {
            "time": finite(row["time"], f"{recording_id}.serveMarkers.time"),
            "side": row.get("side"),
            "origin": row.get("origin"),
            **({"modelSide": row["modelSide"]} if "modelSide" in row else {}),
            **(
                {"modelConfidence": row["modelConfidence"]}
                if "modelConfidence" in row
                else {}
            ),
            **({"modelId": row["modelId"]} if "modelId" in row else {}),
            **({"rallyId": row["rallyId"]} if "rallyId" in row else {}),
        }
        for row in payload.get("serveMarkers", [])
    ]
    annotation = payload.get("annotation", {})
    return {
        "recordingId": recording_id,
        "environment": str(recording.get("environment", "unknown")),
        "sourceGroup": str(recording.get("sourceGroup", "unknown")),
        "split": str(recording.get("split", "unknown")),
        "sourceType": source_type,
        "targetStatus": target_status,
        "labelPath": str(path.resolve()),
        "labelSha256": sha256(path),
        "videoPath": str(resolve_video(path, recording.get("video"))),
        "videoFilename": str(recording.get("videoFilename", "")),
        "durationSeconds": duration,
        "roi": recording.get("roi"),
        "ignoredIntervals": source_ignored_intervals(
            payload.get("ignoredIntervals"),
            duration=duration,
            where=f"{recording_id}.ignoredIntervals",
        ),
        "rallies": rallies,
        "serveMarkers": serve_markers,
        "sideSwitches": switches,
        "candidateSource": {
            "kind": "label-document",
            "annotationStatus": annotation.get("status"),
            "continuousVideoReviewed": annotation.get("continuousVideoReviewed"),
        },
        "priority": priority,
    }


def add_label_directory(
    records: dict[str, dict[str, Any]],
    directory: Path,
    *,
    source_type: str,
    priority: int,
    target_status: str,
) -> int:
    added = 0
    for path in sorted(directory.glob("*.labels.json")):
        record = label_record(
            path,
            source_type=source_type,
            priority=priority,
            target_status=target_status,
        )
        current = records.get(record["recordingId"])
        if current is None or record["priority"] < current["priority"]:
            records[record["recordingId"]] = record
            added += 1
    return added


def feedback_record(video_path: Path, feedback_path: Path) -> dict[str, Any]:
    payload = json.loads(feedback_path.read_text(encoding="utf-8"))
    source = payload["source"]
    inference = payload["initialInference"]
    media = source.get("media", {})
    duration = finite(media["duration"], f"{video_path.name}.duration")
    roi = source.get("featureRoi")
    ranges = []
    for index, row in enumerate(inference.get("ranges", []), start=1):
        if row.get("included", True) is False:
            continue
        start = finite(row["start"], f"{video_path.name}.ranges[{index}].start")
        end = finite(row["end"], f"{video_path.name}.ranges[{index}].end")
        if end <= start:
            continue
        ranges.append(
            {
                "index": len(ranges) + 1,
                "start": start,
                "end": end,
                "notes": None,
                "tags": ["candidate-model-rally"],
            }
        )
    recording_id = f"raw-no-backup-{video_path.stem}"
    return {
        "recordingId": recording_id,
        "environment": "unknown",
        "sourceGroup": "volleycut-raw-no-backup",
        "split": "non-training",
        "sourceType": "raw-no-backup-model-feedback",
        "targetStatus": "candidate-only",
        "labelPath": str(feedback_path.resolve()),
        "labelSha256": sha256(feedback_path),
        "videoPath": str(video_path.resolve()),
        "videoFilename": video_path.name,
        "durationSeconds": duration,
        "roi": roi,
        "ignoredIntervals": source_ignored_intervals(
            payload.get("corrections", {}).get("ignoredIntervals"),
            duration=duration,
            where=f"{video_path.name}.corrections.ignoredIntervals",
        ),
        "rallies": ranges,
        "sideSwitches": [],
        "candidateSource": {
            "kind": "model-feedback-initial-inference",
            "feedbackPath": str(feedback_path.resolve()),
            "analysisId": source.get("analysisId"),
            "modelId": inference.get("modelId"),
            "probabilityModelId": inference.get("probabilityModelId"),
            "components": inference.get("components", []),
            "ensembleAlgorithmVersion": inference.get("ensembleAlgorithmVersion"),
            "gameWindow": source.get("gameWindow"),
            "featureRoi": roi,
            "initialRangeCount": len(ranges),
            "finalExportIntervalCount": len(payload.get("finalExportIntervals", [])),
            "ignoredIntervalCount": len(
                payload.get("corrections", {}).get("ignoredIntervals", [])
            ),
        },
        "priority": 30,
    }


def add_raw_no_backup(records: dict[str, dict[str, Any]], root: Path) -> int:
    added = 0
    for video_path in sorted(root.glob("*.mp4")):
        feedback_path = root / f"{video_path.stem}.model-feedback.json"
        if not feedback_path.is_file():
            continue
        record = feedback_record(video_path, feedback_path)
        if record["recordingId"] not in records:
            records[record["recordingId"]] = record
            added += 1
    return added


def exported_project_record(
    dataset_path: Path,
    dataset: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    reference_path = dataset_path.parent / str(row["referencePath"])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if exported_sha256(reference_path) != str(row["referenceSha256"]):
        raise ValueError(f"exported-project reference changed: {reference_path}")
    annotations = reference["annotations"]
    rallies = [
        {
            "index": index,
            "id": item["id"],
            "start": float(item["start"]),
            "end": float(item["end"]),
            "notes": None,
            "tags": ["human-retained-export-core", "weak-rally-coverage"],
        }
        for index, item in enumerate(annotations["associationCoreRanges"], start=1)
    ]
    serves = [
        {
            "id": item["id"],
            "time": float(item["time"]),
            "rawTime": float(item["rawTime"]),
            "side": item.get("side"),
            "origin": item.get("origin"),
            "rawRallyId": item.get("rawRallyId"),
            "alignedRangeId": item.get("alignedRangeId"),
            "coverageRangeIds": item.get("coverageRangeIds", []),
            "alignment": item.get("alignment"),
            "wasTimeAdjusted": item.get("wasTimeAdjusted") is True,
            "ignorePreviousPoint": item.get("ignorePreviousPoint") is True,
            **({"modelSide": item["modelSide"]} if "modelSide" in item else {}),
            **(
                {"modelConfidence": item["modelConfidence"]}
                if "modelConfidence" in item
                else {}
            ),
        }
        for item in annotations["serveEvents"]
    ]
    switches = [dict(item) for item in annotations["sideSwitches"]]
    inference_index_path = dataset_path.parent / str(
        dataset.get("regeneratedInferenceIndexPath", "")
    )
    regenerated: dict[str, Any] | None = None
    if inference_index_path.is_file():
        inference_index = json.loads(inference_index_path.read_text(encoding="utf-8"))
        match = next(
            (
                item
                for item in inference_index.get("recordings", [])
                if item.get("recordingId") == reference["recordingId"]
            ),
            None,
        )
        if match is not None:
            regenerated = {
                "indexPath": str(inference_index_path.resolve()),
                "indexSha256": sha256(inference_index_path),
                **match,
            }
    return {
        "recordingId": reference["recordingId"],
        "environment": reference["environment"],
        "sourceGroup": reference["sourceGroup"],
        "split": reference["split"],
        "sourceType": "human-reviewed-model-feedback-export",
        "targetStatus": dataset.get("targetStatus", "reviewed-export-coverage"),
        "labelPath": str(reference_path.resolve()),
        "labelSha256": str(row["referenceSha256"]),
        "videoPath": reference["videoPath"],
        "videoFilename": reference["videoFilename"],
        "videoSha256": reference["videoSha256"],
        "durationSeconds": float(annotations["durationSeconds"]),
        "roi": reference.get("roi"),
        "ignoredIntervals": annotations["ignoredIntervals"],
        "rallies": rallies,
        "serveMarkers": serves,
        "sideSwitches": switches,
        "candidateSource": {
            "kind": "human-reviewed-model-feedback-export",
            "datasetPath": str(dataset_path.resolve()),
            "datasetSha256": exported_sha256(dataset_path),
            "feedbackPath": reference["feedbackPath"],
            "feedbackSha256": reference["feedbackSha256"],
            "annotationScope": reference["annotationScope"],
            "normalization": annotations["normalization"],
            "rawRetainedCoreRangeCount": len(annotations["retainedCoreRanges"]),
            "normalizedCoverageRangeCount": len(rallies),
            "serveEventCount": len(serves),
            "sideSwitchCount": len(switches),
            "regeneratedInference": regenerated,
        },
        "priority": 5,
    }


def add_exported_project_datasets(
    records: dict[str, dict[str, Any]], root: Path
) -> tuple[int, list[Path]]:
    added = 0
    paths: list[Path] = []
    if not root.is_dir():
        return added, paths
    for dataset_path in sorted(root.glob("*/dataset.json")):
        dataset = load_exported_project_dataset(dataset_path)
        paths.append(dataset_path.resolve())
        for row in dataset["records"]:
            record = exported_project_record(dataset_path, dataset, row)
            current = records.get(record["recordingId"])
            if current is None or record["priority"] < current["priority"]:
                records[record["recordingId"]] = record
                added += 1
    return added, paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-root", type=Path, default=DEFAULT_COMPLETED_ROOT)
    parser.add_argument("--intake-root", type=Path, default=DEFAULT_INTAKE_ROOT)
    parser.add_argument("--raw-no-backup-root", type=Path, default=DEFAULT_RAW_NO_BACKUP_ROOT)
    parser.add_argument(
        "--exported-project-root",
        type=Path,
        default=DEFAULT_EXPORTED_PROJECT_ROOT,
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {output}")
    completed_root = arguments.completed_root.expanduser().resolve()
    intake_root = arguments.intake_root.expanduser().resolve()
    raw_root = arguments.raw_no_backup_root.expanduser().resolve()
    exported_root = arguments.exported_project_root.expanduser().resolve()
    records: dict[str, dict[str, Any]] = {}
    roots: list[dict[str, Any]] = []

    added_completed = add_label_directory(
        records,
        completed_root,
        source_type="completed-labels",
        priority=0,
        target_status="gold",
    )
    roots.append({"path": str(completed_root), "kind": "completed-labels"})
    added_intake = add_label_directory(
        records,
        intake_root / "labels" / "full",
        source_type="intake-labels",
        priority=10,
        target_status="reviewed-draft",
    )
    roots.append({"path": str(intake_root / "labels" / "full"), "kind": "intake-labels"})
    added_prelabels = add_label_directory(
        records,
        intake_root / "blind-sol" / "prelabels",
        source_type="intake-blind-prelabel",
        priority=20,
        target_status="candidate-only",
    )
    roots.append({"path": str(intake_root / "blind-sol" / "prelabels"), "kind": "intake-blind-prelabel"})
    added_raw = add_raw_no_backup(records, raw_root)
    roots.append({"path": str(raw_root), "kind": "raw-no-backup-model-feedback"})
    added_exported, exported_paths = add_exported_project_datasets(
        records, exported_root
    )
    roots.extend(
        {"path": str(path), "kind": "human-reviewed-model-feedback-export"}
        for path in exported_paths
    )

    for record in records.values():
        video = Path(record["videoPath"])
        if not video.is_file():
            raise FileNotFoundError(f"{record['recordingId']} video does not exist: {video}")
        if record["durationSeconds"] <= 0:
            raise ValueError(f"{record['recordingId']} has no positive duration")
    ordered = sorted(records.values(), key=lambda row: row["recordingId"])
    for record in ordered:
        record.pop("priority", None)
    payload = {
        "schemaVersion": 1,
        "kind": MANIFEST_KIND,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "scope": "all-discovered-deduplicated-NAS-video-sources",
        "notes": [
            "Candidate-only rows are for manual validation and are excluded from gold metrics.",
            "Every consumer must treat ignoredIntervals as outside the training and evaluation universe.",
            "The raw source directory is named volleycut-raw-no-backup (singular) on this NAS.",
            "Human-reviewed exported-project rows preserve rally coverage separately from serve events and are not frame-exact gold by default.",
        ],
        "roots": roots,
        "records": ordered,
    }
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for record in ordered:
        by_type[record["sourceType"]] = by_type.get(record["sourceType"], 0) + 1
        by_status[record["targetStatus"]] = by_status.get(record["targetStatus"], 0) + 1
    print(
        json.dumps(
            {
                "manifest": str(output),
                "recordings": len(ordered),
                "rallies": sum(len(row["rallies"]) for row in ordered),
                "added": {
                    "completed": added_completed,
                    "intakeLabels": added_intake,
                    "intakePrelabels": added_prelabels,
                    "rawNoBackup": added_raw,
                    "exportedProjects": added_exported,
                },
                "sourceTypes": by_type,
                "targetStatuses": by_status,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
