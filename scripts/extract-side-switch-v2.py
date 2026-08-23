#!/usr/bin/env python3
"""Extract immutable, decision-free side-switch v2 features for the frozen split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_appearance import create_hog, read_frame
from analysis.side_switch_v2 import (
    FEATURE_ARTIFACT_KIND,
    FEATURE_ARTIFACT_SCHEMA_VERSION,
    FROZEN_RECORDING_SPLIT,
    RECORDING_ROLE,
    add_interactions,
    bind_recording_normalization,
    bind_recording_orientation,
    calibrate_side_divider,
    derived_existing_features,
    extract_court_players,
    refine_side_divider,
    side_frame_from_players,
    side_pair_features,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_APPEARANCE_REPORT = (
    ROOT / "reports/side-switch/appearance-diagnostic-full-nas-v1.json"
)
DEFAULT_CORPUS_MANIFEST = ROOT / "reports/full-nas-video-corpus-v1.json"
DEFAULT_OUTPUT = ROOT / "reports/side-switch/side-switch-v2-features.json"
EXPECTED_APPEARANCE_SHA256 = (
    "dc947698ab72c8c41e602d9359a437d845759ae1956698b0d08c41796d2899cb"
)
RALLY_ORDER_PATTERN = re.compile(r":candidate-gap:(\d+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _segment_times(start: float, end: float, count: int) -> tuple[float, ...]:
    if end <= start:
        return ()
    return tuple(float(value) for value in np.linspace(start, end, count))


def _sample_times(
    event: Mapping[str, Any],
    *,
    samples_per_side: int,
    flank_seconds: float,
    edge_margin_seconds: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    gap_start = float(event["gapStart"])
    gap_end = float(event["gapEnd"])
    transition = float(event["transitionTime"])
    left = gap_start + edge_margin_seconds
    right = gap_end - edge_margin_seconds
    return (
        _segment_times(
            max(left, transition - flank_seconds),
            min(right, transition - edge_margin_seconds),
            samples_per_side,
        ),
        _segment_times(
            max(left, transition + edge_margin_seconds),
            min(right, transition + flank_seconds),
            samples_per_side,
        ),
    )


def _round_optional(value: float | None, digits: int = 8) -> float | None:
    return None if value is None else round(float(value), digits)


def _rally_order(event_id: str) -> int:
    match = RALLY_ORDER_PATTERN.search(event_id)
    if match is None:
        raise ValueError(f"raw candidate event has no rally order: {event_id}")
    return int(match.group(1))


def _recording_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("corpus manifest has no records list")
    result = {
        str(record["recordingId"]): record
        for record in records
        if isinstance(record, Mapping)
        and str(record.get("recordingId", "")) in RECORDING_ROLE
    }
    missing = sorted(set(RECORDING_ROLE) - set(result))
    if missing:
        raise ValueError(f"corpus manifest is missing frozen recordings: {missing}")
    return result


def extract(args: argparse.Namespace) -> dict[str, Any]:
    appearance_path = args.appearance_report.expanduser().resolve()
    manifest_path = args.corpus_manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite v2 feature artifact: {output_path}")
    if args.samples_per_side < 3:
        raise ValueError("frame-consistency extraction requires at least 3 samples per side")
    if args.flank_seconds <= args.edge_margin_seconds:
        raise ValueError("flank seconds must exceed edge margin seconds")

    appearance_sha = _sha256(appearance_path)
    if args.enforce_source_hash and appearance_sha != EXPECTED_APPEARANCE_SHA256:
        raise ValueError(
            "appearance source hash changed; inspect and update the frozen extraction contract"
        )
    appearance = _load_json(appearance_path)
    manifest = _load_json(manifest_path)
    records = _recording_map(manifest)
    source_events = appearance.get("events")
    if not isinstance(source_events, list):
        raise ValueError("appearance report has no events list")
    events = [
        event
        for event in source_events
        if isinstance(event, Mapping)
        and str(event.get("recordingId", "")) in RECORDING_ROLE
    ]
    if not events:
        raise ValueError("appearance report has no frozen raw candidate events")
    if any(event.get("label") is not None for event in events):
        raise ValueError("v2 source candidates unexpectedly expose labels")
    by_recording: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        by_recording.setdefault(str(event["recordingId"]), []).append(event)
    if set(by_recording) != set(RECORDING_ROLE):
        raise ValueError("appearance candidates do not cover the exact frozen split")

    rows: list[dict[str, Any]] = []
    moments: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}
    geometry_audit: dict[str, Any] = {}
    hog = create_hog()
    for role in ("train", "validation", "evaluation"):
        for recording_id in FROZEN_RECORDING_SPLIT[role]:
            recording_events = sorted(
                by_recording[recording_id], key=lambda event: float(event["transitionTime"])
            )
            video_path = Path(str(records[recording_id]["videoPath"])).resolve()
            if not video_path.is_file():
                raise FileNotFoundError(f"video does not exist: {video_path}")
            print(
                f"Extracting {role} {recording_id}: {len(recording_events)} candidates",
                file=sys.stderr,
                flush=True,
            )
            event_times: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
            unique_times: set[float] = set()
            for event in recording_events:
                sampled = _sample_times(
                    event,
                    samples_per_side=args.samples_per_side,
                    flank_seconds=args.flank_seconds,
                    edge_margin_seconds=args.edge_margin_seconds,
                )
                if not sampled[0] or not sampled[1]:
                    raise ValueError(f"insufficient v2 window for {event['eventId']}")
                event_times[str(event["eventId"])] = sampled
                unique_times.update((*sampled[0], *sampled[1]))
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"could not open video: {video_path}")
            players_by_time: dict[float, Any] = {}
            try:
                for index, timestamp in enumerate(sorted(unique_times), start=1):
                    frame = read_frame(capture, timestamp)
                    players_by_time[timestamp] = extract_court_players(frame, hog)
                    if index % 40 == 0 or index == len(unique_times):
                        print(
                            f"  frames {index}/{len(unique_times)}",
                            file=sys.stderr,
                            flush=True,
                        )
            finally:
                capture.release()

            recording_divider, recording_geometry = calibrate_side_divider(
                list(players_by_time.values())
            )
            local_sources: dict[str, int] = {}
            local_dividers: list[float] = []
            for event in recording_events:
                event_id = str(event["eventId"])
                before_times, after_times = event_times[event_id]
                player_frames = [
                    players_by_time[timestamp]
                    for timestamp in (*before_times, *after_times)
                ]
                divider, local_geometry = refine_side_divider(
                    player_frames, recording_divider
                )
                local_sources[str(local_geometry["source"])] = (
                    local_sources.get(str(local_geometry["source"]), 0) + 1
                )
                local_dividers.append(divider)
                before_frames = [
                    side_frame_from_players(players_by_time[timestamp], timestamp, divider)
                    for timestamp in before_times
                ]
                after_frames = [
                    side_frame_from_players(players_by_time[timestamp], timestamp, divider)
                    for timestamp in after_times
                ]
                side_features, before_moment, after_moment = side_pair_features(
                    before_frames, after_frames
                )
                combined = {
                    **derived_existing_features(event),
                    **side_features,
                }
                combined = add_interactions(combined)
                row = {
                    "eventId": event_id,
                    "recordingId": recording_id,
                    "role": role,
                    "sourceGroup": event.get("sourceGroup"),
                    "sourceType": event.get("sourceType"),
                    "targetStatus": event.get("targetStatus"),
                    "rallyOrder": _rally_order(event_id),
                    "transitionTime": event.get("transitionTime"),
                    "gapStart": event.get("gapStart"),
                    "gapEnd": event.get("gapEnd"),
                    "gapSecondsContextOnly": event.get("gapSeconds"),
                    "beforeTimes": [round(value, 5) for value in before_times],
                    "afterTimes": [round(value, 5) for value in after_times],
                    "geometry": {
                        "divider": round(divider, 8),
                        "source": local_geometry["source"],
                        "playerCount": local_geometry["playerCount"],
                    },
                    "features": {
                        name: _round_optional(value) for name, value in combined.items()
                    },
                }
                rows.append(row)
                moments[event_id] = (before_moment, after_moment)
            geometry_audit[recording_id] = {
                "recordingCalibration": recording_geometry,
                "localCalibrationSources": local_sources,
                "minimumLocalDivider": min(local_dividers),
                "maximumLocalDivider": max(local_dividers),
                "meanLocalDivider": float(np.mean(local_dividers)),
            }

    orientation_binding = bind_recording_orientation(rows, moments)
    normalization = bind_recording_normalization(rows)
    for row in rows:
        row["features"] = {
            name: _round_optional(value) for name, value in row["features"].items()
        }
        row["normalizedFeatures"] = {
            name: _round_optional(value)
            for name, value in row["normalizedFeatures"].items()
        }
        row["beforeOrientation"] = _round_optional(row["beforeOrientation"])
        row["afterOrientation"] = _round_optional(row["afterOrientation"])

    created_at = datetime.now(UTC).isoformat()
    payload = {
        "schemaVersion": FEATURE_ARTIFACT_SCHEMA_VERSION,
        "kind": FEATURE_ARTIFACT_KIND,
        "createdAt": created_at,
        "sources": {
            "appearanceReport": {
                "path": str(appearance_path),
                "sha256": appearance_sha,
                "kind": appearance.get("kind"),
                "createdAt": appearance.get("createdAt"),
            },
            "corpusManifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
                "kind": manifest.get("kind"),
                "createdAt": manifest.get("createdAt"),
            },
            "reviewDecisions": None,
        },
        "frozenSplit": {
            role: list(recording_ids)
            for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
        },
        "protocol": {
            "samplesPerSide": args.samples_per_side,
            "flankSeconds": args.flank_seconds,
            "edgeMarginSeconds": args.edge_margin_seconds,
            "personProposal": "opencv-hog-default-people-detector-v1",
            "detectorMaximumWidth": 1280,
            "courtProposalBoundsNormalized": {
                "left": 0.12,
                "right": 0.88,
                "topFoot": 0.38,
                "bottomFoot": 0.92,
            },
            "geometryCalibration": (
                "unlabeled recording-wide weighted two-means with local shrinkage; "
                "0.68 normalized-height fallback"
            ),
            "normalization": (
                "per-recording median and 1.4826*MAD over complete candidate sequence; "
                "IQR then unit fallback; clipped to [-10,10]"
            ),
            "orientationBinding": (
                "unlabeled per-recording first principal component of color moments"
            ),
            "gapDurationPolicy": "retained as audit context only; excluded from every learned feature set",
        },
        "summary": {
            "events": len(rows),
            "byRole": {
                role: sum(row["role"] == role for row in rows)
                for role in ("train", "validation", "evaluation")
            },
            "recordings": len(RECORDING_ROLE),
        },
        "geometryCalibration": geometry_audit,
        "orientationBinding": orientation_binding,
        "recordingNormalization": normalization,
        "events": rows,
    }
    atomic_write_text(output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-report", type=Path, default=DEFAULT_APPEARANCE_REPORT)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-side", type=int, default=4)
    parser.add_argument("--flank-seconds", type=float, default=8.0)
    parser.add_argument("--edge-margin-seconds", type=float, default=0.75)
    parser.add_argument(
        "--no-enforce-source-hash",
        action="store_false",
        dest="enforce_source_hash",
        help="permit an appearance report other than the frozen full-NAS source",
    )
    return parser


def main() -> None:
    payload = extract(_parser().parse_args())
    print(json.dumps({"events": payload["summary"]["events"]}, indent=2))


if __name__ == "__main__":
    main()
