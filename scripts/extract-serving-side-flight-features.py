#!/usr/bin/env python3
"""Extract development-only multi-resolution flight-motion grid features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import cv2

from analysis.artifacts import atomic_write_text
from analysis.serving_side import crop_roi
from analysis.serving_side_flight import (
    FEATURE_VERSION,
    OFFSETS_SECONDS,
    extract_motion_sequence,
    feature_names,
    summarize_motion,
)
from analysis.serving_side_specialist import (
    apply_review_corrections,
    reviewed_rallies,
)
from analysis.side_switch_appearance import read_frame


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_REPORT = (
    ROOT
    / "reports/serving-side/serving-side-existing-label-variants-full-nas-v2.json"
)
DEFAULT_DECISIONS = (
    ROOT / "reports/serving-side/serving-side-review-decisions-full-nas-v1.json"
)
DEFAULT_CORRECTIONS = (
    ROOT / "reports/serving-side/serving-side-result-label-corrections-v1.json"
)
DEFAULT_V2_DATASET = ROOT / "features/serving-side-v2/development.json"
DEFAULT_OUTPUT = ROOT / "features/serving-side-flight-v1/development.json"
REPORT_SHA256 = "611d698961534740b3b07624a2ade1d574de4e06b8bf5ce701b641b4690fc35b"
DECISIONS_SHA256 = "1066edcf9579d3c92157a8504dbe5d798023ebb3ff4605e0dad9e451c97080b4"
RESOLUTIONS = ((192, 108), (384, 216), (640, 360))
GRIDS = ((3, 3), (4, 4), (4, 6))
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    REPOSITORY_ROOT / "analysis/serving_side_flight.py",
    Path(__file__).resolve(),
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bytes(path: Path) -> tuple[Mapping[str, Any], bytes]:
    content = path.read_bytes()
    payload = json.loads(content)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload, content


def _load(path: Path) -> Mapping[str, Any]:
    return _load_bytes(path)[0]


def _roi(file_record: Mapping[str, Any]) -> tuple[float, float, float, float]:
    source = _load(Path(str(file_record["path"])))
    recording = source.get("recording")
    recording = recording if isinstance(recording, Mapping) else {}
    value = recording.get("roi")
    if not isinstance(value, Mapping):
        candidate = file_record.get("candidateSource")
        candidate = candidate if isinstance(candidate, Mapping) else {}
        value = candidate.get("featureRoi")
    if isinstance(value, Mapping):
        result = tuple(float(value[name]) for name in ("x", "y", "width", "height"))
    else:
        result = (0.0, 0.0, 1.0, 1.0)
    if (
        len(result) != 4
        or result[2] <= 0
        or result[3] <= 0
        or not all(math.isfinite(item) for item in result)
    ):
        raise ValueError(f"invalid ROI for {file_record['recordingId']}")
    return result  # type: ignore[return-value]


def _configuration_name(
    width: int, height: int, grid_rows: int, grid_columns: int
) -> str:
    return f"{width}x{height}-r{grid_rows}c{grid_columns}"


def extract(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.limit_per_recording is not None and args.limit_per_recording < 1:
        raise ValueError("limit-per-recording must be positive")
    report_path = args.serving_report.resolve()
    decisions_path = args.decisions.resolve()
    corrections_path = args.corrections.resolve()
    v2_path = args.v2_dataset.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite flight features: {output_path}")
    report, report_bytes = _load_bytes(report_path)
    decisions, decision_bytes = _load_bytes(decisions_path)
    corrections, correction_bytes = _load_bytes(corrections_path)
    report_hash = _sha256_bytes(report_bytes)
    decision_hash = _sha256_bytes(decision_bytes)
    correction_hash = _sha256_bytes(correction_bytes)
    if report_hash != REPORT_SHA256 or decision_hash != DECISIONS_SHA256:
        raise ValueError("review sources do not match the frozen serving-side hashes")
    effective, correction_counts = apply_review_corrections(
        report,
        decisions,
        corrections,
        base_decision_sha256=decision_hash,
    )
    reviewed, review_counts = reviewed_rallies(report, effective)
    protected_source_groups = sorted(
        {row.source_group for row in reviewed if row.split == "test"}
    )
    wanted = [
        row
        for row in reviewed
        if row.split != "test" and row.source_group not in protected_source_groups
    ]
    if not wanted or any(row.split == "test" for row in wanted):
        raise ValueError("flight extraction must contain development rows only")
    if args.limit_per_recording is not None:
        selected = []
        for recording_id in sorted({row.recording_id for row in wanted}):
            selected.extend(
                [row for row in wanted if row.recording_id == recording_id][
                    : args.limit_per_recording
                ]
            )
        wanted = selected

    v2 = _load(v2_path)
    if (
        v2.get("scope") != "development"
        or v2.get("sources", {}).get("servingSideReport", {}).get("sha256")
        != report_hash
        or v2.get("sources", {}).get("reviewDecisions", {}).get("sha256")
        != decision_hash
    ):
        raise ValueError("v2 baseline dataset does not bind the same development labels")
    raw_v2_rows = v2.get("rows")
    if not isinstance(raw_v2_rows, list):
        raise ValueError("v2 baseline dataset rows are invalid")
    v2_by_id = {
        str(row["rallyId"]): row
        for row in raw_v2_rows
        if isinstance(row, Mapping) and isinstance(row.get("rallyId"), str)
    }
    if any(row.rally_id not in v2_by_id for row in wanted):
        raise ValueError("v2 baseline does not cover every flight row")

    labels = report.get("labels")
    files = labels.get("files") if isinstance(labels, Mapping) else None
    if not isinstance(files, list):
        raise ValueError("serving-side report has no file catalog")
    file_by_id = {
        str(item["recordingId"]): item
        for item in files
        if isinstance(item, Mapping) and isinstance(item.get("recordingId"), str)
    }
    configurations = [
        {
            "name": _configuration_name(width, height, rows, columns),
            "resize": {"width": width, "height": height},
            "grid": {"rows": rows, "columns": columns},
            "featureNames": list(feature_names(rows, columns)),
        }
        for width, height in RESOLUTIONS
        for rows, columns in GRIDS
    ]
    cv2.setNumThreads(1)

    def extract_recording(recording_id: str) -> list[dict[str, Any]]:
        metadata = file_by_id.get(recording_id)
        if metadata is None:
            raise ValueError(f"missing source metadata for {recording_id}")
        video_path = Path(str(metadata["videoPath"])).resolve()
        duration = float(metadata["durationSeconds"])
        roi = _roi(metadata)
        recording_rows = sorted(
            (row for row in wanted if row.recording_id == recording_id),
            key=lambda row: (float(row.rally["start"]), row.rally_id),
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        recording_output: list[dict[str, Any]] = []
        try:
            for index, row in enumerate(recording_rows, start=1):
                anchor = float(row.rally["start"])
                timestamps = [
                    min(max(0.0, anchor + offset), max(0.0, duration - 0.01))
                    for offset in OFFSETS_SECONDS
                ]
                source_frames = [
                    crop_roi(read_frame(capture, timestamp), roi)
                    for timestamp in timestamps
                ]
                configuration_features: dict[str, dict[str, float]] = {}
                for width, height in RESOLUTIONS:
                    frames = [
                        cv2.resize(
                            frame,
                            (width, height),
                            interpolation=cv2.INTER_AREA,
                        )
                        for frame in source_frames
                    ]
                    motions = extract_motion_sequence(frames)
                    for grid_rows, grid_columns in GRIDS:
                        name = _configuration_name(
                            width, height, grid_rows, grid_columns
                        )
                        configuration_features[name] = summarize_motion(
                            motions, grid_rows, grid_columns
                        )
                baseline = v2_by_id[row.rally_id]
                recording_output.append(
                    {
                        "rallyId": row.rally_id,
                        "recordingId": recording_id,
                        "environment": row.environment,
                        "sourceGroup": row.source_group,
                        "sourceSplit": row.split,
                        "sourceType": row.source_type,
                        "targetStatus": row.target_status,
                        "serveAnchor": anchor,
                        "decision": row.decision,
                        "label": row.label,
                        "v2RecordingRankFeatures": baseline[
                            "recordingRankFeatures"
                        ],
                        "configurations": configuration_features,
                    }
                )
                if index % 10 == 0 or index == len(recording_rows):
                    print(
                        f"{recording_id}: {index}/{len(recording_rows)}",
                        flush=True,
                    )
        finally:
            capture.release()
        return recording_output

    recording_ids = sorted({row.recording_id for row in wanted})
    output_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for recording_output in executor.map(extract_recording, recording_ids):
            output_rows.extend(recording_output)
    output_rows.sort(
        key=lambda row: (
            str(row["recordingId"]),
            float(row["serveAnchor"]),
            str(row["rallyId"]),
        )
    )
    if _sha256(corrections_path) != correction_hash:
        raise RuntimeError("correction overlay changed during flight extraction")
    if args.limit_per_recording is None and len(output_rows) != len(wanted):
        raise AssertionError("flight extraction did not cover every development row")
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-flight-feature-development-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "scope": "development",
        "featureVersion": FEATURE_VERSION,
        "offsetsSeconds": list(OFFSETS_SECONDS),
        "v2FeatureNames": v2.get("featureNames"),
        "configurations": configurations,
        "rows": output_rows,
        "counts": {
            "rows": len(output_rows),
            "recordings": len({row["recordingId"] for row in output_rows}),
            "sourceGroups": len({row["sourceGroup"] for row in output_rows}),
            "near": sum(row["label"] for row in output_rows),
            "far": sum(1 - row["label"] for row in output_rows),
        },
        "reviewCounts": review_counts,
        "correctionCounts": correction_counts,
        "dataPolicy": {
            "protectedTestIncluded": False,
            "protectedSourceGroupsExcluded": protected_source_groups,
            "unclear": "excluded",
            "notServeCorrections": "excluded",
            "sideCorrections": "applied",
            "serveAnchor": "authoritative rally.start from the reviewed source",
            "limitPerRecording": args.limit_per_recording,
        },
        "sources": {
            "servingSideReport": {
                "path": str(report_path),
                "sha256": report_hash,
            },
            "reviewDecisions": {
                "path": str(decisions_path),
                "sha256": decision_hash,
            },
            "humanLabelCorrections": {
                "path": str(corrections_path),
                "sha256": correction_hash,
            },
            "v2DevelopmentDataset": {
                "path": str(v2_path),
                "sha256": _sha256(v2_path),
            },
            "files": [
                {
                    "recordingId": recording_id,
                    "videoPath": str(file_by_id[recording_id]["videoPath"]),
                    "labelPath": str(file_by_id[recording_id]["path"]),
                    "labelSha256": str(file_by_id[recording_id]["sha256"]),
                }
                for recording_id in sorted(
                    {row["recordingId"] for row in output_rows}
                )
            ],
            "implementation": [
                {
                    "path": str(path.relative_to(REPOSITORY_ROOT)),
                    "sha256": _sha256(path),
                }
                for path in IMPLEMENTATION_PATHS
            ],
        },
    }
    atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    print(f"wrote {output_path} ({len(output_rows)} rows)")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--v2-dataset", type=Path, default=DEFAULT_V2_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-per-recording", type=int)
    parser.add_argument("--workers", type=int, default=4)
    return parser


if __name__ == "__main__":
    extract(_parser().parse_args())
