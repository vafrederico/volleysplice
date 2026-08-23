#!/usr/bin/env python3
"""Extract score-compatible V5-state features for the complete candidate union."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_appearance import read_frame
from analysis.side_switch_full_union_features import (
    FRAMES_PER_SEQUENCE,
    candidate_windows,
    whole_rally_sample_times,
)
from analysis.side_switch_production_replay import (
    load_frozen_heads,
    numeric_array,
    prepared_from_feedback,
    replay_trace,
)
from analysis.side_switch_production_state import (
    PRODUCTION_MODEL_SOURCES,
    PRODUCTION_STATE_FEATURE_NAMES,
    TimeRange,
    gap_state_features,
    merge_production_components,
    rally_evidence,
    rally_evidence_payload,
)
from analysis.side_switch_v4 import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    CourtGeometry,
    summarize_sequence,
    visual_features,
)
from analysis.side_switch_v5 import (
    VISUAL_FEATURE_NAMES,
    player_features,
    summarize_player_sequence,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
INTAKE = Path("/mnt/freenas/volleycut/intake-2026-08-13")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_MANIFEST = ROOT / "reports/full-nas-video-corpus-v1.json"
DEFAULT_CANDIDATES = REPORTS / "side-switch-candidate-union-v1-evaluation.json"
DEFAULT_V5_FEATURES = REPORTS / "side-switch-v5-player-orientation-features.json"
DEFAULT_STATE_FEATURES = REPORTS / "side-switch-v5-production-state-v1-features.json"
DEFAULT_OUTPUT = REPORTS / "side-switch-full-union-v5-state-features-v1.json"
V2_BUNDLE = INTAKE / "experiments/environment-specialists-v2/models/all-labels/bundle.json"
OLD_MODEL_ROOT = (
    Path("/mnt/freenas/volleycut")
    / "labeling-v1-2026-08-09-no-beach-2026-08-12/models"
)
OLD_HEAD_NAMES = {
    "rally": "full-audiovisual-audio-normalized-v3",
    "serve": "serve-specialist-audio-normalized-v5",
    "dead": "dead-state-transition-audio-normalized-v5-no-legacy-final",
}
EXPECTED_SHA256 = {
    "manifest": "c177f2936dc1ea7f2953cf94a6b9720cd6d4bf4e652a4c7409499374c9fe1e24",
    "candidates": "c4717a6e056b659fc7c63541a2eac13451518b0720dd7362dfb49643688edbc2",
    "v5Features": "4406fe47de0326c1257b8d9cf353116a9495f82d2e43c92e9721b1d6764ba5db",
    "stateFeatures": "c86b8ef7427dd8e1347d726c7d2326f18f22a6eb9fb16a0eec685c0fde0c9f36",
}
PARITY_TOLERANCE = 1e-8
PRODUCTION_REPLAY_TOLERANCE = 5e-6


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _geometry(payload: Mapping[str, Any]) -> CourtGeometry:
    return CourtGeometry(
        net_y_ratio=float(payload["netYRatio"]),
        confidence=float(payload["confidence"]),
        detected_frames=int(payload["detectedFrames"]),
        sampled_frames=int(payload["sampledFrames"]),
    )


def _crop_and_resize(frame: np.ndarray, roi: Mapping[str, Any]) -> np.ndarray:
    height, width = frame.shape[:2]
    left = max(0, min(width - 1, round(float(roi.get("x", 0.0)) * width)))
    top = max(0, min(height - 1, round(float(roi.get("y", 0.0)) * height)))
    right = max(
        left + 1,
        min(
            width,
            round(
                (float(roi.get("x", 0.0)) + float(roi.get("width", 1.0)))
                * width
            ),
        ),
    )
    bottom = max(
        top + 1,
        min(
            height,
            round(
                (float(roi.get("y", 0.0)) + float(roi.get("height", 1.0)))
                * height
            ),
        ),
    )
    return cv2.resize(
        frame[top:bottom, left:right],
        (FRAME_WIDTH, FRAME_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def _rounded(values: Mapping[str, float]) -> dict[str, float]:
    return {name: round(float(value), 8) for name, value in values.items()}


def _window_key(start: float, end: float) -> tuple[float, float]:
    return round(start, 9), round(end, 9)


def _initial_probability_parity(
    feedback: Mapping[str, Any], trace: Any
) -> dict[str, float]:
    initial = feedback["initialInference"]
    rows = len(trace.times)
    result: dict[str, float] = {}
    for output_name, payload_name in (
        ("rally", "rally"),
        ("serve", "serve"),
        ("deadState", "deadState"),
    ):
        expected = numeric_array(
            initial["probabilities"][payload_name], "float32", (rows,)
        )
        actual = {
            "rally": trace.rally_scores,
            "serve": trace.serve_scores,
            "deadState": trace.dead_state_scores,
        }[output_name]["all-labels-v2"]
        result[output_name] = float(np.max(np.abs(expected - actual)))
    return result


def extract(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "manifest": args.manifest.expanduser().resolve(),
        "candidates": args.candidates.expanduser().resolve(),
        "v5Features": args.v5_features.expanduser().resolve(),
        "stateFeatures": args.state_features.expanduser().resolve(),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite full-union features: {output}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"full-union feature source identity changed: {hashes}")

    manifest = _load(paths["manifest"])
    candidate_artifact = _load(paths["candidates"])
    v5_artifact = _load(paths["v5Features"])
    state_artifact = _load(paths["stateFeatures"])
    selected = candidate_artifact["selected"]["metrics"]["4.0"]
    recording_ids = tuple(candidate_artifact["scope"]["recordingIds"])
    records = {
        str(record["recordingId"]): record
        for record in manifest["records"]
        if str(record.get("recordingId")) in recording_ids
    }
    if set(records) != set(recording_ids):
        raise ValueError("manifest does not cover the full-union scope")

    old_rows = {
        (
            str(row["recordingId"]),
            round(float(row["gapStart"]), 6),
            round(float(row["gapEnd"]), 6),
        ): row
        for row in state_artifact["rows"]
        if str(row.get("recordingId")) in recording_ids
    }
    if len(old_rows) != 352:
        raise ValueError("frozen raw-phone parity universe changed")

    old_heads, v2_heads = load_frozen_heads(V2_BUNDLE, OLD_MODEL_ROOT, OLD_HEAD_NAMES)
    if v2_heads.rally.feature_names != old_heads.rally.feature_names:
        raise ValueError("production replay signature changed")

    rows: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    parity_differences: list[float] = []
    parity_rows = 0
    for recording_number, recording_id in enumerate(recording_ids, start=1):
        record = records[recording_id]
        candidates = list(selected["byRecording"][recording_id]["candidateInventory"])
        feedback_path = Path(str(record["labelPath"])).resolve()
        feedback = _load(feedback_path)
        expected_feedback = candidate_artifact["feedbackSources"][recording_id]
        initial_hash = _json_sha256(feedback["initialInference"])
        if initial_hash != expected_feedback["initialInferenceSha256"]:
            raise ValueError(f"initial inference changed for {recording_id}")
        prepared = prepared_from_feedback(record, feedback, v2_heads.rally)
        trace = replay_trace(prepared, old_heads, v2_heads)
        probability_parity = _initial_probability_parity(feedback, trace)
        if max(probability_parity.values()) > PRODUCTION_REPLAY_TOLERANCE:
            raise ValueError(f"production replay parity failed for {recording_id}")

        ranges = list(feedback["initialInference"]["ranges"])
        source_ranges = [
            TimeRange(float(value["start"]), float(value["end"])) for value in ranges
        ]
        components = merge_production_components(trace.ranges)
        expected_components = feedback["initialInference"]["ranges"]
        if len(components) != len(expected_components):
            raise ValueError(f"production range replay count drifted for {recording_id}")
        maximum_boundary_error = max(
            (
                difference
                for component, expected in zip(
                    components, expected_components, strict=True
                )
                for difference in (
                    abs(component.start - float(expected["start"])),
                    abs(component.end - float(expected["end"])),
                )
            ),
            default=0.0,
        )
        if maximum_boundary_error > 1e-6:
            raise ValueError(f"production range replay drifted for {recording_id}")
        evidences = [rally_evidence(value, trace, components) for value in source_ranges]
        range_index = {str(value["id"]): index for index, value in enumerate(ranges)}
        if len(range_index) != len(ranges):
            raise ValueError(f"duplicate production range IDs for {recording_id}")

        geometry_payload = v5_artifact["extractionAudit"][recording_id]["courtGeometry"]
        geometry = _geometry(geometry_payload)
        window_by_candidate: dict[str, tuple[Any, Any, int | None]] = {}
        unique_windows: dict[tuple[float, float], Any] = {}
        for candidate in candidates:
            before, after, boundary_index = candidate_windows(candidate, ranges)
            window_by_candidate[str(candidate["eventId"])] = (
                before,
                after,
                boundary_index,
            )
            unique_windows[_window_key(before.start, before.end)] = before
            unique_windows[_window_key(after.start, after.end)] = after

        print(
            f"[{recording_number}/{len(recording_ids)}] Extracting {recording_id}: "
            f"{len(candidates)} candidates, {len(unique_windows)} unique sequences",
            file=sys.stderr,
            flush=True,
        )
        capture = cv2.VideoCapture(str(record["videoPath"]))
        if not capture.isOpened():
            raise RuntimeError(f"could not open source video: {record['videoPath']}")
        v4_summaries: dict[tuple[float, float], Any] = {}
        v5_summaries: dict[tuple[float, float], Any] = {}
        try:
            for key, window in sorted(unique_windows.items()):
                frames = [
                    _crop_and_resize(read_frame(capture, timestamp), record["roi"])
                    for timestamp in whole_rally_sample_times(window.start, window.end)
                ]
                v4_summaries[key] = summarize_sequence(frames, geometry)
                v5_summaries[key] = summarize_player_sequence(frames, geometry)
        finally:
            capture.release()

        kind_counts: dict[str, int] = {}
        for candidate in candidates:
            event_id = str(candidate["eventId"])
            kind = str(candidate["kind"])
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            before_window, after_window, boundary_index = window_by_candidate[event_id]
            before_key = _window_key(before_window.start, before_window.end)
            after_key = _window_key(after_window.start, after_window.end)
            v4_values = visual_features(
                v4_summaries[before_key], v4_summaries[after_key]
            )
            visual_values = player_features(
                v5_summaries[before_key], v5_summaries[after_key], v4_values
            )
            if boundary_index is not None:
                before_evidence = evidences[boundary_index]
                after_evidence = evidences[boundary_index + 1]
                production_context_eligible = True
                evidence_contract = "adjacent decoded ranges"
            else:
                internal_index = range_index[str(candidate["sourceRangeId"])]
                before_evidence = evidences[internal_index]
                after_evidence = evidences[internal_index]
                production_context_eligible = False
                evidence_contract = "same containing decoded range on both flanks"
            state_values, suppression = gap_state_features(
                before_evidence,
                after_evidence,
                float(candidate["gapStart"]),
                float(candidate["gapEnd"]),
                trace,
            )
            if suppression:
                raise ValueError("suppression unexpectedly entered full-union extraction")
            features = {**_rounded(visual_values), **_rounded(state_values)}
            if tuple(features) != (*VISUAL_FEATURE_NAMES, *PRODUCTION_STATE_FEATURE_NAMES):
                raise ValueError("full-union feature signature drifted")

            parity_key = (
                recording_id,
                round(float(candidate["gapStart"]), 6),
                round(float(candidate["gapEnd"]), 6),
            )
            old_row = old_rows.get(parity_key)
            parity_maximum = None
            if old_row is not None:
                old_features = old_row["features"]
                differences = [
                    abs(float(features[name]) - float(old_features[name]))
                    for name in features
                ]
                parity_maximum = max(differences, default=0.0)
                parity_differences.append(parity_maximum)
                parity_rows += 1
                if parity_maximum > PARITY_TOLERANCE:
                    raise ValueError(
                        f"legacy feature parity failed for {event_id}: {parity_maximum}"
                    )

            rows.append(
                {
                    **dict(candidate),
                    "status": "ok",
                    "features": features,
                    "comparisonWindows": {
                        "before": before_window.to_dict(),
                        "after": after_window.to_dict(),
                        "framesPerWindow": FRAMES_PER_SEQUENCE,
                    },
                    "productionStateContext": {
                        "eligibleForFrozenProductionContextHead": production_context_eligible,
                        "evidenceContract": evidence_contract,
                        "before": rally_evidence_payload(before_evidence),
                        "after": rally_evidence_payload(after_evidence),
                    },
                    "legacyParityMaximumAbsoluteDifference": parity_maximum,
                }
            )

        audits[recording_id] = {
            "candidates": len(candidates),
            "candidateKinds": kind_counts,
            "uniqueComparisonSequences": len(unique_windows),
            "decodedFrames": len(unique_windows) * FRAMES_PER_SEQUENCE,
            "productionSamples": len(trace.times),
            "productionProbabilityMaximumAbsoluteDifference": probability_parity,
            "productionProbabilityTolerance": PRODUCTION_REPLAY_TOLERANCE,
            "productionRangeMaximumBoundaryErrorSeconds": maximum_boundary_error,
            "feedbackPath": str(feedback_path),
            "feedbackWholeFileSha256": _sha256(feedback_path),
            "feedbackInitialInferenceSha256": initial_hash,
            "feedbackFeaturesSha256": _json_sha256(feedback["features"]),
            "videoPath": str(record["videoPath"]),
            "roi": record["roi"],
            "courtGeometry": geometry.to_dict(),
        }

    if len(rows) != 704 or parity_rows != 352:
        raise ValueError(
            f"full-union counts changed: rows={len(rows)}, parityRows={parity_rows}"
        )
    maximum_parity = max(parity_differences, default=math.inf)
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-full-union-v5-state-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "scope": {
            "status": "opened-development-only",
            "recordingIds": list(recording_ids),
            "recordings": len(recording_ids),
            "candidates": len(rows),
            "adjacentRallyBoundaries": sum(
                row["kind"] == "adjacent-rally-boundary" for row in rows
            ),
            "internalDeadStatePeaks": sum(
                row["kind"] == "internal-dead-state-peak" for row in rows
            ),
        },
        "profile": {
            "name": "FULL-UNION-V5-STATE42",
            "featureNames": [*VISUAL_FEATURE_NAMES, *PRODUCTION_STATE_FEATURE_NAMES],
            "primaryFrozenHeadFeatureNames": [
                *VISUAL_FEATURE_NAMES,
                *PRODUCTION_STATE_FEATURE_NAMES[:10],
            ],
            "frameShape": [FRAME_HEIGHT, FRAME_WIDTH],
            "framesPerComparisonWindow": FRAMES_PER_SEQUENCE,
            "boundarySampling": "frozen whole-rally 8%-to-92% contract",
            "internalSampling": (
                "fixed three-second flanks [t-4,t-1] and [t+1,t+4] within the "
                "same decoded range"
            ),
            "internalProductionEvidence": (
                "same containing decoded-range evidence on both sides; frozen "
                "production-context head is declared ineligible"
            ),
            "labelIndependentExtraction": True,
            "onDeviceCompatibility": (
                "256x144 frame operations plus existing F104 production-head replay"
            ),
        },
        "parityAudit": {
            "legacyRowsExpected": 352,
            "legacyRowsMatched": parity_rows,
            "tolerance": PARITY_TOLERANCE,
            "maximumAbsoluteFeatureDifference": maximum_parity,
            "passed": maximum_parity <= PARITY_TOLERANCE,
        },
        "extractionAudit": audits,
        "rows": rows,
        "productionModels": state_artifact["productionModels"],
        "sources": {
            name: {"path": str(paths[name]), "sha256": hashes[name]}
            for name in paths
        },
        "limitations": [
            "All 11 raw-phone recordings are opened development scope, not an untouched test split.",
            "Internal candidates do not have two distinct decoded-rally serve anchors, so the frozen production-context head is not directly comparable there.",
            "Feature extraction alone does not select proposals or estimate precision.",
        ],
    }
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--v5-features", type=Path, default=DEFAULT_V5_FEATURES)
    parser.add_argument("--state-features", type=Path, default=DEFAULT_STATE_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    payload = extract(_parser().parse_args())
    print(
        json.dumps(
            {
                "output": str(DEFAULT_OUTPUT),
                "rows": payload["scope"]["candidates"],
                "parityAudit": payload["parityAudit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
