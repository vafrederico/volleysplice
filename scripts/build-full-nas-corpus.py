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
from analysis.nas_video_corpus import MANIFEST_KIND


DEFAULT_COMPLETED_ROOT = Path(
    "/mnt/freenas/volleycut/labeling-v1-2026-08-09/completed/full-v1"
)
DEFAULT_INTAKE_ROOT = Path("/mnt/freenas/volleycut/intake-2026-08-13")
DEFAULT_RAW_NO_BACKUP_ROOT = Path("/mnt/freenas/volleycut-raw-no-backup")


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
        "durationSeconds": finite(
            recording["durationSeconds"], f"{recording_id}.durationSeconds"
        ),
        "roi": recording.get("roi"),
        "rallies": rallies,
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
        "durationSeconds": finite(media["duration"], f"{video_path.name}.duration"),
        "roi": roi,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-root", type=Path, default=DEFAULT_COMPLETED_ROOT)
    parser.add_argument("--intake-root", type=Path, default=DEFAULT_INTAKE_ROOT)
    parser.add_argument("--raw-no-backup-root", type=Path, default=DEFAULT_RAW_NO_BACKUP_ROOT)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {output}")
    completed_root = arguments.completed_root.expanduser().resolve()
    intake_root = arguments.intake_root.expanduser().resolve()
    raw_root = arguments.raw_no_backup_root.expanduser().resolve()
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
            "The raw source directory is named volleycut-raw-no-backup (singular) on this NAS.",
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
