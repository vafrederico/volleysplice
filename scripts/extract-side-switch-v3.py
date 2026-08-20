#!/usr/bin/env python3
"""Extract low-resolution, detector-free side-switch v3 features."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_appearance import read_frame
from analysis.side_switch_training_policy import validate_side_switch_fit_recordings
from analysis.side_switch_v3 import (
    FEATURE_ARTIFACT_KIND,
    FEATURE_ARTIFACT_SCHEMA_VERSION,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    FRAMES_PER_RALLY,
    FROZEN_RECORDING_SPLIT,
    RECORDING_ROLE,
    summarize_sequence,
    visual_features,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_APPEARANCE = ROOT / "reports/side-switch/appearance-diagnostic-full-nas-v1.json"
DEFAULT_DECISIONS = (
    ROOT / "reports/side-switch/appearance-review-decisions-full-nas-v1.json"
)
DEFAULT_MANIFEST = ROOT / "reports/full-nas-video-corpus-v1.json"
DEFAULT_OUTPUT = ROOT / "reports/side-switch/side-switch-v3-features.json"
EXPECTED_SHA256 = {
    "appearance": "dc947698ab72c8c41e602d9359a437d845759ae1956698b0d08c41796d2899cb",
    "decisions": "6f75df5621fcfe0bd8370f3b015226ab04cf0456842fb5a259b3544aa382abe4",
    "manifest": "c177f2936dc1ea7f2953cf94a6b9720cd6d4bf4e652a4c7409499374c9fe1e24",
}
GAP_PATTERN = re.compile(r":(?:candidate-gap|control):(\d+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _gap_order(event: Mapping[str, Any], rallies: Sequence[Mapping[str, Any]]) -> int:
    match = GAP_PATTERN.search(str(event["eventId"]))
    if match is not None:
        return int(match.group(1))
    timestamp = float(event["transitionTime"])
    for index, (before, after) in enumerate(zip(rallies, rallies[1:]), start=1):
        if float(before["end"]) - 1e-6 <= timestamp <= float(after["start"]) + 1e-6:
            return index
    raise ValueError(f"side-switch event is outside every rally gap: {event['eventId']}")


def _sample_times(rally: Mapping[str, Any]) -> tuple[float, ...]:
    start = float(rally["start"])
    end = float(rally["end"])
    duration = end - start
    if duration <= 0:
        raise ValueError("rally duration must be positive")
    window = min(duration * 0.90, 2.4)
    first = start + min(0.12, duration * 0.05)
    last = min(end - min(0.08, duration * 0.03), first + window)
    if last <= first:
        return (start + duration * 0.25, start + duration * 0.75)
    return tuple(float(value) for value in np.linspace(first, last, FRAMES_PER_RALLY))


def _crop_and_resize(frame: np.ndarray, roi: Mapping[str, Any]) -> np.ndarray:
    height, width = frame.shape[:2]
    left = max(0, min(width - 1, round(float(roi.get("x", 0.0)) * width)))
    top = max(0, min(height - 1, round(float(roi.get("y", 0.0)) * height)))
    right = max(
        left + 1,
        min(width, round((float(roi.get("x", 0.0)) + float(roi.get("width", 1.0))) * width)),
    )
    bottom = max(
        top + 1,
        min(height, round((float(roi.get("y", 0.0)) + float(roi.get("height", 1.0))) * height)),
    )
    return cv2.resize(
        frame[top:bottom, left:right],
        (FRAME_WIDTH, FRAME_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def _collapse_rows(
    appearance: Mapping[str, Any],
    decisions: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_decisions = decisions.get("decisions")
    raw_events = appearance.get("events")
    if not isinstance(raw_decisions, Mapping) or not isinstance(raw_events, list):
        raise ValueError("appearance or decisions schema is invalid")
    grouped: dict[tuple[str, int], list[tuple[Mapping[str, Any], str]]] = defaultdict(list)
    for event in raw_events:
        if not isinstance(event, Mapping):
            continue
        recording_id = str(event.get("recordingId", ""))
        if recording_id not in RECORDING_ROLE:
            continue
        decision = raw_decisions.get(str(event.get("eventId", "")))
        if decision not in {"switch", "no-switch", "unclear"}:
            raise ValueError(f"missing or invalid decision for {event.get('eventId')}")
        gap_order = _gap_order(event, records[recording_id]["rallies"])
        grouped[(recording_id, gap_order)].append((event, str(decision)))

    rows: list[dict[str, Any]] = []
    duplicate_markers = 0
    unclear_gaps = 0
    invalidated_exact_markers = 0
    for (recording_id, gap_order), members in sorted(grouped.items()):
        member_decisions = [decision for _, decision in members]
        exact_members = [event for event, _ in members if event.get("source") == "sideSwitches"]
        duplicate_markers += max(0, len(exact_members) - 1)
        if "switch" in member_decisions:
            decision = "switch"
            label = 1
        elif "unclear" in member_decisions:
            unclear_gaps += 1
            continue
        else:
            decision = "no-switch"
            label = 0
        if exact_members and label == 0:
            invalidated_exact_markers += len(exact_members)
        representative = min(
            (event for event, _ in members),
            key=lambda event: float(event["transitionTime"]),
        )
        record = records[recording_id]
        rows.append(
            {
                "eventId": f"{recording_id}:gap:{gap_order}",
                "sourceEventIds": sorted(str(event["eventId"]) for event, _ in members),
                "recordingId": recording_id,
                "role": RECORDING_ROLE[recording_id],
                "environment": record.get("environment", "unknown"),
                "sourceGroup": record.get("sourceGroup", "unknown"),
                "sourceType": record.get("sourceType", "unknown"),
                "targetStatus": record.get("targetStatus", "unknown"),
                "gapOrder": gap_order,
                "gapStart": float(record["rallies"][gap_order - 1]["end"]),
                "gapEnd": float(record["rallies"][gap_order]["start"]),
                "transitionTime": float(representative["transitionTime"]),
                "decision": decision,
                "label": label,
                "exactMarkerTimes": sorted(
                    float(event["transitionTime"]) for event in exact_members
                ),
            }
        )
    return rows, {
        "sourceEvents": sum(len(value) for value in grouped.values()),
        "collapsedRows": len(rows),
        "duplicateExactMarkersCollapsed": duplicate_markers,
        "unclearGapsExcluded": unclear_gaps,
        "reviewInvalidatedExactMarkers": invalidated_exact_markers,
    }


def extract(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "appearance": args.appearance.expanduser().resolve(),
        "decisions": args.decisions.expanduser().resolve(),
        "manifest": args.manifest.expanduser().resolve(),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v3 feature artifact: {output}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"v3 source identity changed: {hashes}")
    appearance = _load(paths["appearance"])
    decisions = _load(paths["decisions"])
    manifest = _load(paths["manifest"])
    raw_records = manifest.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("manifest has no records list")
    records = {
        str(record["recordingId"]): record
        for record in raw_records
        if isinstance(record, Mapping) and str(record.get("recordingId", "")) in RECORDING_ROLE
    }
    if set(records) != set(RECORDING_ROLE):
        raise ValueError("manifest does not cover the frozen v3 split")
    validate_side_switch_fit_recordings(list(FROZEN_RECORDING_SPLIT["train"]))
    rows, collapse_audit = _collapse_rows(appearance, decisions, records)
    rows_by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_recording[str(row["recordingId"])].append(row)

    extraction_audit: dict[str, Any] = {}
    for role in ("train", "validation", "evaluation"):
        for recording_id in FROZEN_RECORDING_SPLIT[role]:
            recording_rows = sorted(
                rows_by_recording[recording_id], key=lambda row: int(row["gapOrder"])
            )
            record = records[recording_id]
            rallies = record["rallies"]
            needed_rallies = sorted(
                {
                    rally_index
                    for row in recording_rows
                    for rally_index in (int(row["gapOrder"]), int(row["gapOrder"]) + 1)
                }
            )
            video_path = Path(str(record["videoPath"])).resolve()
            if not video_path.is_file():
                raise FileNotFoundError(f"missing v3 source video: {video_path}")
            print(
                f"Extracting v3 {role} {recording_id}: "
                f"{len(recording_rows)} gaps, {len(needed_rallies)} rally summaries",
                file=sys.stderr,
                flush=True,
            )
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"could not open v3 source video: {video_path}")
            summaries: dict[int, Any] = {}
            errors: dict[int, str] = {}
            try:
                for rally_number in needed_rallies:
                    rally = rallies[rally_number - 1]
                    try:
                        frames = [
                            _crop_and_resize(read_frame(capture, timestamp), record["roi"])
                            for timestamp in _sample_times(rally)
                        ]
                        summaries[rally_number] = summarize_sequence(frames)
                    except (RuntimeError, ValueError, cv2.error) as error:
                        errors[rally_number] = str(error)
            finally:
                capture.release()
            for row in recording_rows:
                gap_order = int(row["gapOrder"])
                before = summaries.get(gap_order)
                after = summaries.get(gap_order + 1)
                if before is None or after is None:
                    row["status"] = "frame-error"
                    row["error"] = errors.get(gap_order) or errors.get(gap_order + 1)
                    row["features"] = {}
                else:
                    row["status"] = "ok"
                    row["features"] = {
                        name: round(float(value), 8)
                        for name, value in visual_features(before, after).items()
                    }
            extraction_audit[recording_id] = {
                "role": role,
                "videoPath": str(video_path),
                "reviewedGaps": len(recording_rows),
                "rallySummaries": len(summaries),
                "frameErrors": len(errors),
                "roi": record["roi"],
            }

    payload = {
        "schemaVersion": FEATURE_ARTIFACT_SCHEMA_VERSION,
        "kind": FEATURE_ARTIFACT_KIND,
        "createdAt": datetime.now(UTC).isoformat(),
        "profile": {
            "frameShape": [FRAME_HEIGHT, FRAME_WIDTH],
            "framesPerRally": FRAMES_PER_RALLY,
            "sampleRegion": "first min(90% of rally, 2.4 seconds)",
            "proposalDetector": None,
            "operations": "resize, HSV histograms, frame differences, Laplacian, Canny",
            "labelResolution": (
                "review decision determines visible switch; duplicate exact markers in "
                "one gap collapse; unclear gaps are excluded"
            ),
        },
        "frozenRecordingSplit": {
            role: list(recording_ids)
            for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
        },
        "collapseAudit": collapse_audit,
        "extractionAudit": extraction_audit,
        "rows": rows,
        "sources": {
            name: {"path": str(paths[name]), "sha256": hashes[name]}
            for name in paths
        },
    }
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance", type=Path, default=DEFAULT_APPEARANCE)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    payload = extract(_parser().parse_args())
    counts = {
        role: sum(row["role"] == role for row in payload["rows"])
        for role in FROZEN_RECORDING_SPLIT
    }
    print(json.dumps({"rows": counts, "collapseAudit": payload["collapseAudit"]}, indent=2))


if __name__ == "__main__":
    main()
