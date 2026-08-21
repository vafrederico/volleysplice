#!/usr/bin/env python3
"""Extract inference-safe early/center/late flight windows on development data."""

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
from analysis.serving_side_exclusions import (
    load_source_quality_exclusions,
    samples_touch_source_exclusion,
)
from analysis.serving_side_flight import (
    OFFSETS_SECONDS,
    extract_motion_sequence,
    feature_names,
    summarize_motion,
)
from analysis.side_switch_appearance import read_frame


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_CENTER = ROOT / "features/serving-side-flight-v3/development.json"
DEFAULT_OUTPUT = ROOT / "features/serving-side-temporal-flight-v1/development.json"
WINDOW_SHIFTS_SECONDS = (-1.0, 1.0)
RESIZE = (192, 108)
GRID = (4, 6)
CONFIGURATION = "192x108-r4c6"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    REPOSITORY_ROOT / "analysis/serving_side_flight.py",
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
    path = Path(str(file_record["path"])).resolve()
    payload = _load(path)
    recording = payload.get("recording")
    recording = recording if isinstance(recording, Mapping) else {}
    duration = float(file_record["durationSeconds"])
    raw_roi = recording.get("roi")
    if not isinstance(raw_roi, Mapping):
        candidate = file_record.get("candidateSource")
        candidate = candidate if isinstance(candidate, Mapping) else {}
        raw_roi = candidate.get("featureRoi")
    if isinstance(raw_roi, Mapping):
        roi = tuple(float(raw_roi[name]) for name in ("x", "y", "width", "height"))
    else:
        roi = (0.0, 0.0, 1.0, 1.0)
    if (
        not math.isfinite(duration)
        or duration <= 0
        or roi[2] <= 0
        or roi[3] <= 0
        or not all(math.isfinite(value) for value in roi)
    ):
        raise ValueError(f"invalid recording metadata: {path}")
    return duration, roi  # type: ignore[return-value]


def _shift_name(shift: float) -> str:
    return "minus1" if shift < 0 else "plus1"


def extract(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.workers < 1:
        raise ValueError("workers must be positive")
    center_path = args.center_dataset.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite temporal features: {output_path}")
    center_hash = _sha256(center_path)
    center = _load(center_path)
    if (
        center.get("kind") != "volleycut-serving-side-flight-feature-development-v1"
        or center.get("scope") != "development"
        or center.get("dataPolicy", {}).get("protectedTestIncluded") is not False
        or center.get("dataPolicy", {}).get("limitPerRecording") is not None
    ):
        raise ValueError("temporal extraction requires a complete development flight dataset")
    configurations = center.get("configurations")
    selected_configuration = next(
        (
            item
            for item in configurations
            if isinstance(item, Mapping) and item.get("name") == CONFIGURATION
        ),
        None,
    ) if isinstance(configurations, list) else None
    if (
        not isinstance(selected_configuration, Mapping)
        or selected_configuration.get("resize") != {"width": RESIZE[0], "height": RESIZE[1]}
        or selected_configuration.get("grid") != {"rows": GRID[0], "columns": GRID[1]}
    ):
        raise ValueError(f"center dataset lacks {CONFIGURATION}")
    raw_rows = center.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("center dataset has no rows")
    rows = [row for row in raw_rows if isinstance(row, Mapping)]
    if len(rows) != len(raw_rows):
        raise ValueError("center dataset contains invalid rows")

    sources = center.get("sources")
    sources = sources if isinstance(sources, Mapping) else {}
    raw_files = sources.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("center dataset has no source file catalog")
    file_by_id = {
        str(item["recordingId"]): item
        for item in raw_files
        if isinstance(item, Mapping) and isinstance(item.get("recordingId"), str)
    }
    report_source = sources.get("servingSideReport")
    if not isinstance(report_source, Mapping):
        raise ValueError("center dataset lacks its serving-side report")
    report_path = Path(str(report_source["path"])).resolve()
    if _sha256(report_path) != report_source.get("sha256"):
        raise ValueError("serving-side report changed since center extraction")
    report = _load(report_path)
    report_files = report.get("labels", {}).get("files")
    if not isinstance(report_files, list):
        raise ValueError("serving-side report has no file catalog")
    report_file_by_id = {
        str(item["recordingId"]): item
        for item in report_files
        if isinstance(item, Mapping) and isinstance(item.get("recordingId"), str)
    }
    exclusion_source = sources.get("sourceQualityExclusions")
    if not isinstance(exclusion_source, Mapping):
        raise ValueError("center dataset lacks source exclusions")
    exclusion_path = Path(str(exclusion_source["path"])).resolve()
    exclusion_hash = _sha256(exclusion_path)
    if exclusion_hash != exclusion_source.get("sha256"):
        raise ValueError("source exclusion artifact changed since center extraction")
    exclusions = load_source_quality_exclusions(exclusion_path)

    wanted = []
    temporal_quality_excluded = 0
    for row in rows:
        anchor = float(row["serveAnchor"])
        sample_times = [
            anchor + shift + offset
            for shift in (-1.0, 0.0, 1.0)
            for offset in OFFSETS_SECONDS
        ]
        if samples_touch_source_exclusion(
            exclusions, str(row["recordingId"]), sample_times
        ):
            temporal_quality_excluded += 1
        else:
            wanted.append(row)
    if not wanted:
        raise ValueError("temporal extraction has no eligible development rows")

    cv2.setNumThreads(1)
    names = feature_names(*GRID)

    def extract_recording(recording_id: str) -> list[dict[str, Any]]:
        source = file_by_id.get(recording_id)
        metadata = report_file_by_id.get(recording_id)
        if not isinstance(source, Mapping) or not isinstance(metadata, Mapping):
            raise ValueError(f"missing source file for {recording_id}")
        video_path = Path(str(metadata["videoPath"])).resolve()
        duration, roi = _recording_metadata(metadata)
        recording_rows = sorted(
            (row for row in wanted if row["recordingId"] == recording_id),
            key=lambda row: (float(row["serveAnchor"]), str(row["rallyId"])),
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        output: list[dict[str, Any]] = []
        try:
            for index, row in enumerate(recording_rows, start=1):
                anchor = float(row["serveAnchor"])
                shifted: dict[str, dict[str, float]] = {}
                for shift in WINDOW_SHIFTS_SECONDS:
                    timestamps = [
                        min(
                            max(0.0, anchor + shift + offset),
                            max(0.0, duration - 0.01),
                        )
                        for offset in OFFSETS_SECONDS
                    ]
                    frames = [
                        cv2.resize(
                            crop_roi(read_frame(capture, timestamp), roi),
                            RESIZE,
                            interpolation=cv2.INTER_AREA,
                        )
                        for timestamp in timestamps
                    ]
                    shifted[_shift_name(shift)] = summarize_motion(
                        extract_motion_sequence(frames), *GRID
                    )
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
                        "shiftedFlightFeatures": shifted,
                    }
                )
                if index % 10 == 0 or index == len(recording_rows):
                    print(f"{recording_id}: {index}/{len(recording_rows)}", flush=True)
        finally:
            capture.release()
        return output

    output_rows: list[dict[str, Any]] = []
    recording_ids = sorted({str(row["recordingId"]) for row in wanted})
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for recording_rows in executor.map(extract_recording, recording_ids):
            output_rows.extend(recording_rows)
    output_rows.sort(key=lambda row: str(row["rallyId"]))
    if len(output_rows) != len(wanted):
        raise AssertionError("temporal extraction did not cover every eligible row")
    if _sha256(exclusion_path) != exclusion_hash:
        raise RuntimeError("source exclusions changed during temporal extraction")
    if _sha256(center_path) != center_hash:
        raise RuntimeError("center flight dataset changed during temporal extraction")

    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-temporal-flight-feature-development-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "scope": "development",
        "configuration": {
            "name": CONFIGURATION,
            "resize": {"width": RESIZE[0], "height": RESIZE[1]},
            "grid": {"rows": GRID[0], "columns": GRID[1]},
            "featureNames": list(names),
        },
        "centerWindowOffsetsSeconds": list(OFFSETS_SECONDS),
        "additionalWindowShiftsSeconds": list(WINDOW_SHIFTS_SECONDS),
        "rows": output_rows,
        "counts": {
            "rows": len(output_rows),
            "recordings": len(recording_ids),
            "sourceGroups": len({row["sourceGroup"] for row in output_rows}),
            "near": sum(int(row["label"]) for row in output_rows),
            "far": sum(1 - int(row["label"]) for row in output_rows),
            "additionalSourceQualityExcluded": temporal_quality_excluded,
        },
        "dataPolicy": {
            "protectedTestIncluded": False,
            "humanCorrectedAnchorsUsed": False,
            "windowSelectionAtInference": "all three fixed shifts are available uniformly",
            "sourceQualityIntervals": "any early, center, or late frame touching an exclusion removes the row",
        },
        "sources": {
            "centerFlightDataset": {
                "path": str(center_path),
                "sha256": center_hash,
            },
            "humanLabelCorrections": sources.get("humanLabelCorrections"),
            "sourceQualityExclusions": {
                "path": str(exclusion_path),
                "sha256": exclusion_hash,
            },
            "files": [file_by_id[recording_id] for recording_id in recording_ids],
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
    result.add_argument("--center-dataset", type=Path, default=DEFAULT_CENTER)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--workers", type=int, default=6)
    return result


if __name__ == "__main__":
    extract(parser().parse_args())
