#!/usr/bin/env python3
"""Extract selective source-resolution track-patch features on development videos."""

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
from analysis.serving_side_flight import OFFSETS_SECONDS, extract_motion_sequence
from analysis.serving_side_selective_highres import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    PATCH_SIZE,
    SOURCE_PATCH_FRACTION,
    extract_selective_highres_features,
)
from analysis.side_switch_appearance import read_frame


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_TRAJECTORY = ROOT / "features/serving-side-trajectory-v1/development.json"
DEFAULT_OUTPUT = ROOT / "features/serving-side-selective-highres-v1/development.json"
LOW_RESIZE = (192, 108)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    REPOSITORY_ROOT / "analysis/serving_side_flight.py",
    REPOSITORY_ROOT / "analysis/serving_side_trajectory.py",
    REPOSITORY_ROOT / "analysis/serving_side_selective_highres.py",
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


def _recording_metadata(
    file_record: Mapping[str, Any],
) -> tuple[float, tuple[float, float, float, float]]:
    label_path = Path(str(file_record["path"])).resolve()
    source = _load(label_path)
    recording = source.get("recording")
    recording = recording if isinstance(recording, Mapping) else {}
    raw_roi = recording.get("roi")
    if not isinstance(raw_roi, Mapping):
        candidate = file_record.get("candidateSource")
        candidate = candidate if isinstance(candidate, Mapping) else {}
        raw_roi = candidate.get("featureRoi")
    if isinstance(raw_roi, Mapping):
        roi = tuple(float(raw_roi[name]) for name in ("x", "y", "width", "height"))
    else:
        roi = (0.0, 0.0, 1.0, 1.0)
    duration = float(file_record["durationSeconds"])
    if (
        not math.isfinite(duration)
        or duration <= 0
        or roi[2] <= 0
        or roi[3] <= 0
        or not all(math.isfinite(value) for value in roi)
    ):
        raise ValueError(f"invalid recording metadata: {label_path}")
    return duration, roi  # type: ignore[return-value]


def extract(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.limit_per_recording is not None and args.limit_per_recording < 1:
        raise ValueError("limit-per-recording must be positive")
    trajectory_path = args.trajectory_dataset.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite selective features: {output_path}")
    trajectory_hash = _sha256(trajectory_path)
    trajectory = _load(trajectory_path)
    if (
        trajectory.get("kind")
        != "volleycut-serving-side-trajectory-feature-development-v1"
        or trajectory.get("scope") != "development"
        or trajectory.get("dataPolicy", {}).get("protectedTestIncluded") is not False
        or trajectory.get("dataPolicy", {}).get("limitPerRecording") is not None
    ):
        raise ValueError("selective extraction requires the complete trajectory bank")
    raw_rows = trajectory.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("trajectory bank has no rows")
    rows = [row for row in raw_rows if isinstance(row, Mapping)]
    if len(rows) != len(raw_rows) or any(row.get("sourceSplit") == "test" for row in rows):
        raise ValueError("trajectory bank contains invalid or protected rows")
    if args.limit_per_recording is not None:
        selected_rows: list[Mapping[str, Any]] = []
        for recording_id in sorted({str(row["recordingId"]) for row in rows}):
            selected_rows.extend(
                [row for row in rows if str(row["recordingId"]) == recording_id][
                    : args.limit_per_recording
                ]
            )
        rows = selected_rows

    sources = trajectory.get("sources")
    sources = sources if isinstance(sources, Mapping) else {}
    report_source = sources.get("servingSideReport")
    if not isinstance(report_source, Mapping):
        raise ValueError("trajectory bank lacks its serving-side report")
    report_path = Path(str(report_source["path"])).resolve()
    report_hash = _sha256(report_path)
    if report_hash != report_source.get("sha256"):
        raise ValueError("serving-side report changed since trajectory extraction")
    report = _load(report_path)
    labels = report.get("labels")
    files = labels.get("files") if isinstance(labels, Mapping) else None
    if not isinstance(files, list):
        raise ValueError("serving-side report has no file catalog")
    file_by_id = {
        str(item["recordingId"]): item
        for item in files
        if isinstance(item, Mapping) and isinstance(item.get("recordingId"), str)
    }
    mutable_sources = []
    for name in ("humanLabelCorrections", "sourceQualityExclusions"):
        source = sources.get(name)
        if not isinstance(source, Mapping):
            raise ValueError(f"trajectory bank lacks {name}")
        path = Path(str(source["path"])).resolve()
        digest = _sha256(path)
        if digest != source.get("sha256"):
            raise ValueError(f"{name} changed since trajectory extraction")
        mutable_sources.append((name, path, digest))

    cv2.setNumThreads(1)

    def extract_recording(recording_id: str) -> list[dict[str, Any]]:
        metadata = file_by_id.get(recording_id)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"missing source metadata for {recording_id}")
        video_path = Path(str(metadata["videoPath"])).resolve()
        duration, roi = _recording_metadata(metadata)
        recording_rows = sorted(
            (row for row in rows if str(row["recordingId"]) == recording_id),
            key=lambda row: (float(row["serveAnchor"]), str(row["rallyId"])),
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        output: list[dict[str, Any]] = []
        try:
            for index, row in enumerate(recording_rows, start=1):
                anchor = float(row["serveAnchor"])
                timestamps = [
                    min(max(0.0, anchor + offset), max(0.0, duration - 0.01))
                    for offset in OFFSETS_SECONDS
                ]
                source_frames = [
                    crop_roi(read_frame(capture, timestamp), roi)
                    for timestamp in timestamps
                ]
                low_frames = [
                    cv2.resize(
                        frame,
                        LOW_RESIZE,
                        interpolation=cv2.INTER_AREA,
                    )
                    for frame in source_frames
                ]
                motions = extract_motion_sequence(low_frames)
                output.append(
                    {
                        "rallyId": row["rallyId"],
                        "recordingId": recording_id,
                        "environment": row["environment"],
                        "sourceGroup": row["sourceGroup"],
                        "sourceSplit": row["sourceSplit"],
                        "serveAnchor": anchor,
                        "decision": row["decision"],
                        "label": row["label"],
                        "selectiveHighresFeatures": extract_selective_highres_features(
                            source_frames, low_frames, motions
                        ),
                    }
                )
                if index % 10 == 0 or index == len(recording_rows):
                    print(f"{recording_id}: {index}/{len(recording_rows)}", flush=True)
        finally:
            capture.release()
        return output

    recording_ids = sorted({str(row["recordingId"]) for row in rows})
    output_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for recording_rows in executor.map(extract_recording, recording_ids):
            output_rows.extend(recording_rows)
    output_rows.sort(key=lambda row: str(row["rallyId"]))
    if len(output_rows) != len(rows):
        raise AssertionError("selective extraction did not cover its selected rows")
    if _sha256(trajectory_path) != trajectory_hash or _sha256(report_path) != report_hash:
        raise RuntimeError("a frozen selective feature source changed during extraction")
    for name, path, digest in mutable_sources:
        if _sha256(path) != digest:
            raise RuntimeError(f"{name} changed during selective extraction")

    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-selective-highres-feature-development-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "scope": "development",
        "featureVersion": FEATURE_VERSION,
        "featureNames": list(FEATURE_NAMES),
        "offsetsSeconds": list(OFFSETS_SECONDS),
        "lowResolutionLocator": {"width": LOW_RESIZE[0], "height": LOW_RESIZE[1]},
        "patch": {
            "sourceFractionOfShortEdge": SOURCE_PATCH_FRACTION,
            "normalizedWidth": PATCH_SIZE,
            "normalizedHeight": PATCH_SIZE,
            "selectors": ["strongest persistent track", "strongest compact/fast track"],
        },
        "rows": output_rows,
        "counts": {
            "rows": len(output_rows),
            "recordings": len(recording_ids),
            "sourceGroups": len({row["sourceGroup"] for row in output_rows}),
            "near": sum(int(row["label"]) for row in output_rows),
            "far": sum(1 - int(row["label"]) for row in output_rows),
        },
        "dataPolicy": {
            "protectedTestIncluded": False,
            "humanCorrectedAnchorsUsed": False,
            "humanVisibilityUsed": False,
            "limitPerRecording": args.limit_per_recording,
            "sourceResolutionUsage": (
                "decoded source frames; high-detail computation restricted to selected patches"
            ),
        },
        "sources": {
            "trajectoryDevelopmentDataset": {
                "path": str(trajectory_path),
                "sha256": trajectory_hash,
            },
            "centerFlightDataset": sources["centerFlightDataset"],
            "servingSideReport": report_source,
            "humanLabelCorrections": sources["humanLabelCorrections"],
            "sourceQualityExclusions": sources["sourceQualityExclusions"],
            "files": [
                {
                    "recordingId": recording_id,
                    "videoPath": str(file_by_id[recording_id]["videoPath"]),
                    "labelPath": str(file_by_id[recording_id]["path"]),
                    "labelSha256": str(file_by_id[recording_id]["sha256"]),
                }
                for recording_id in recording_ids
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
    atomic_write_text(output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output_path} ({len(output_rows)} rows)")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--trajectory-dataset", type=Path, default=DEFAULT_TRAJECTORY)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--workers", type=int, default=6)
    result.add_argument("--limit-per-recording", type=int)
    return result


if __name__ == "__main__":
    extract(parser().parse_args())
