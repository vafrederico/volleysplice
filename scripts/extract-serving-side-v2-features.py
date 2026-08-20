#!/usr/bin/env python3
"""Extract serving-side v2 short-window features into an immutable NAS artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.serving_side import crop_roi
from analysis.serving_side_specialist import apply_review_corrections, reviewed_rallies
from analysis.serving_side_v2 import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    OFFSETS_SECONDS,
    extract_window_features,
    service_zone_masks,
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
DEFAULT_DEVELOPMENT = ROOT / "features/serving-side-v2/development.json"
DEFAULT_PROTECTED = ROOT / "features/serving-side-v2/protected-test.json"
REPORT_SHA256 = "611d698961534740b3b07624a2ade1d574de4e06b8bf5ce701b641b4690fc35b"
DECISIONS_SHA256 = "1066edcf9579d3c92157a8504dbe5d798023ebb3ff4605e0dad9e451c97080b4"
RESIZE = (192, 108)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    REPOSITORY_ROOT / "analysis/serving_side_specialist.py",
    REPOSITORY_ROOT / "analysis/serving_side_v2.py",
    Path(__file__).resolve(),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _roi_and_geometry(file_record: Mapping[str, Any]) -> tuple[tuple[float, float, float, float], Mapping[str, Any] | None]:
    source_path = Path(str(file_record["path"]))
    source = _load(source_path)
    recording = source.get("recording")
    recording = recording if isinstance(recording, Mapping) else {}
    roi_value = recording.get("roi")
    if not isinstance(roi_value, Mapping):
        candidate = file_record.get("candidateSource")
        candidate = candidate if isinstance(candidate, Mapping) else {}
        roi_value = candidate.get("featureRoi")
    if isinstance(roi_value, Mapping):
        roi = tuple(float(roi_value[name]) for name in ("x", "y", "width", "height"))
    else:
        roi = (0.0, 0.0, 1.0, 1.0)
    if roi[2] <= 0 or roi[3] <= 0 or not all(math.isfinite(value) for value in roi):
        raise ValueError(f"invalid ROI for {file_record['recordingId']}")
    geometry = recording.get("courtGeometry")
    return roi, geometry if isinstance(geometry, Mapping) else None


def _rank_features(rows: list[dict[str, Any]]) -> None:
    for recording_id in sorted({str(row["recordingId"]) for row in rows}):
        recording_rows = [row for row in rows if row["recordingId"] == recording_id]
        count = len(recording_rows)
        for name in FEATURE_NAMES:
            values = np.asarray([row["courtFlowFeatures"][name] for row in recording_rows])
            order = np.argsort(values, kind="stable")
            sorted_values = values[order]
            ranks = np.zeros(count, dtype=np.float64)
            start = 0
            while start < count:
                end = start + 1
                while end < count and sorted_values[end] == sorted_values[start]:
                    end += 1
                rank = 0.5 if count == 1 else ((start + end - 1) / 2.0) / (count - 1)
                ranks[order[start:end]] = rank
                start = end
            for row, rank in zip(recording_rows, ranks, strict=True):
                row["recordingRankFeatures"][name] = float(rank)


def extract(args: argparse.Namespace) -> dict[str, Any]:
    report_path = args.serving_report.resolve()
    decisions_path = args.decisions.resolve()
    output = args.output.resolve() if args.output else (
        DEFAULT_PROTECTED if args.protected_test else DEFAULT_DEVELOPMENT
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feature artifact: {output}")
    report_hash, decisions_hash = _sha256(report_path), _sha256(decisions_path)
    if report_hash != REPORT_SHA256 or decisions_hash != DECISIONS_SHA256:
        raise ValueError("review sources do not match the frozen serving-side v1 hashes")
    report, decisions = _load(report_path), _load(decisions_path)
    corrections_path = args.corrections.resolve()
    correction_source = None
    correction_counts = {"applied": 0, "near": 0, "far": 0, "notServe": 0}
    if corrections_path.exists():
        correction_hash = _sha256(corrections_path)
        decisions, correction_counts = apply_review_corrections(
            report,
            decisions,
            _load(corrections_path),
            base_decision_sha256=decisions_hash,
        )
        correction_source = {
            "path": str(corrections_path),
            "sha256": correction_hash,
        }
    reviewed, review_counts = reviewed_rallies(report, decisions)
    review_counts["notServe"] = correction_counts["notServe"]
    review_counts["missing"] -= correction_counts["notServe"]
    wanted = [row for row in reviewed if (row.split == "test") == args.protected_test]
    if not wanted:
        raise ValueError("selected extraction scope has no clear reviewed rows")
    if not args.protected_test and any(row.split == "test" for row in wanted):
        raise AssertionError("protected test leaked into development extraction")

    files = report.get("labels", {}).get("files", [])
    file_by_id = {
        str(item["recordingId"]): item for item in files if isinstance(item, Mapping)
    }
    output_rows: list[dict[str, Any]] = []
    geometry_counts: dict[str, int] = {}
    for recording_id in sorted({row.recording_id for row in wanted}):
        metadata = file_by_id[recording_id]
        video_path = Path(str(metadata["videoPath"])).resolve()
        roi, geometry = _roi_and_geometry(metadata)
        selected = [row for row in wanted if row.recording_id == recording_id]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        try:
            for index, row in enumerate(selected, start=1):
                anchor = float(row.rally["start"])
                duration = float(metadata["durationSeconds"])
                times = [min(max(0.0, anchor + offset), max(0.0, duration - 0.01)) for offset in OFFSETS_SECONDS]
                frames = [
                    cv2.resize(crop_roi(read_frame(capture, timestamp), roi), RESIZE, interpolation=cv2.INTER_AREA)
                    for timestamp in times
                ]
                masks, provenance = service_zone_masks(
                    RESIZE[1], RESIZE[0], roi=roi, court_geometry=geometry
                )
                features = extract_window_features(frames, masks)
                geometry_counts[provenance] = geometry_counts.get(provenance, 0) + 1
                output_rows.append(
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
                        "geometryProvenance": provenance,
                        "roi": list(roi),
                        "oldFeatures": dict(row.rally.get("features", {})),
                        "courtFlowFeatures": features,
                        "recordingRankFeatures": {},
                    }
                )
                if index % 20 == 0 or index == len(selected):
                    print(f"{recording_id}: {index}/{len(selected)}", flush=True)
        finally:
            capture.release()
    _rank_features(output_rows)
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-v2-feature-dataset",
        "createdAt": datetime.now(UTC).isoformat(),
        "scope": "protected-test" if args.protected_test else "development",
        "featureVersion": FEATURE_VERSION,
        "featureNames": list(FEATURE_NAMES),
        "offsetsSeconds": list(OFFSETS_SECONDS),
        "resize": {"width": RESIZE[0], "height": RESIZE[1]},
        "reviewCounts": review_counts,
        "correctionCounts": correction_counts,
        "rows": output_rows,
        "counts": {
            "rows": len(output_rows),
            "recordings": len({row["recordingId"] for row in output_rows}),
            "near": sum(row["label"] for row in output_rows),
            "far": sum(1 - row["label"] for row in output_rows),
            "geometryProvenance": geometry_counts,
        },
        "dataPolicy": {
            "unclear": "excluded",
            "notServeCorrections": "excluded",
            "protectedTestIncluded": args.protected_test,
            "recordingRanks": "computed from unlabeled feature values within each recording",
            "fallbackGeometry": "top/bottom 32% of the recording ROI; camera-relative, not metric court calibration",
        },
        "sources": {
            "servingSideReport": {"path": str(report_path), "sha256": report_hash},
            "reviewDecisions": {"path": str(decisions_path), "sha256": decisions_hash},
            "humanLabelCorrections": correction_source,
            "files": [
                {
                    "recordingId": recording_id,
                    "videoPath": str(file_by_id[recording_id]["videoPath"]),
                    "labelPath": str(file_by_id[recording_id]["path"]),
                    "labelSha256": str(file_by_id[recording_id]["sha256"]),
                }
                for recording_id in sorted({row.recording_id for row in wanted})
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
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output} ({len(output_rows)} rows)")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protected-test", action="store_true")
    return parser


if __name__ == "__main__":
    extract(_parser().parse_args())
