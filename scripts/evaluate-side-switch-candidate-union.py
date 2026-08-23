#!/usr/bin/env python3
"""Evaluate a high-recall side-switch union over complete production traces."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_candidate_union import (
    CandidateUnionConfig,
    boundary_candidates,
    candidate_union,
)
from analysis.side_switch_full_video import (
    event_metric_counts,
    monotonic_interval_match,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_MANIFEST = ROOT / "reports/full-nas-video-corpus-v1.json"
DEFAULT_FULL_AUDIT = REPORTS / "side-switch-full-video-marker-evaluation-2026-08-21-r2.json"
DEFAULT_V5_FEATURES = REPORTS / "side-switch-v5-player-orientation-features.json"
DEFAULT_CONTINUITY = REPORTS / "side-switch-continuity-verifier-v1-evaluation.json"
DEFAULT_OUTPUT = REPORTS / "side-switch-candidate-union-v1-evaluation.json"
EXPECTED_SHA256 = {
    "manifest": "c177f2936dc1ea7f2953cf94a6b9720cd6d4bf4e652a4c7409499374c9fe1e24",
    "fullAudit": "142d6617aed7f20b51cdcdfc9b3e61c8c76beaf79a31fc2d7ee54fb0fdf27b89",
    "v5Features": "4406fe47de0326c1257b8d9cf353116a9495f82d2e43c92e9721b1d6764ba5db",
    "continuity": "19f1379971db2fbd80cdaf09c8f16d784607ed4892579fb9156a742d6ac0a11d",
}
TARGET_RECALL = 0.90
GRID = tuple(
    CandidateUnionConfig(signal, threshold, separation)
    for signal in ("deadState", "maxDeadOrInverseRally")
    for separation in (6.0, 10.0, 14.0)
    for threshold in (0.70, 0.80, 0.90, 0.95, 0.98)
)


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


def _array(payload: Mapping[str, Any], data_type: str) -> np.ndarray:
    if (
        payload.get("encoding") != "base64"
        or payload.get("byteOrder") != "little-endian"
        or payload.get("dataType") != data_type
    ):
        raise ValueError("production trace array encoding changed")
    shape = tuple(int(value) for value in payload.get("shape", []))
    if len(shape) != 1:
        raise ValueError("production trace arrays must be vectors")
    dtype = "<f8" if data_type == "float64" else "<f4"
    values = np.frombuffer(base64.b64decode(str(payload["data"])), dtype=dtype)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError("production trace array shape or values are invalid")
    return values.astype(np.float64)


def _metrics(
    candidates_by_recording: Mapping[str, Sequence[Mapping[str, Any]]],
    markers_by_recording: Mapping[str, Sequence[Mapping[str, Any]]],
    padding_seconds: float,
    *,
    include_inventory: bool,
) -> dict[str, Any]:
    per_video: dict[str, Any] = {}
    for recording_id, markers in markers_by_recording.items():
        candidates = list(candidates_by_recording[recording_id])
        match = monotonic_interval_match(candidates, markers, padding_seconds)
        counts = event_metric_counts(match)
        per_video[recording_id] = {
            "humanEvents": len(markers),
            "candidates": len(candidates),
            "boundaryCandidates": sum(
                value["kind"] == "adjacent-rally-boundary" for value in candidates
            ),
            "internalPeakCandidates": sum(
                value["kind"] == "internal-dead-state-peak" for value in candidates
            ),
            **counts,
            "missedHumanTimes": [
                float(markers[index]["time"])
                for index in match.unmatched_marker_indices
            ],
        }
        if include_inventory:
            per_video[recording_id]["candidateInventory"] = candidates
    true_positives = sum(int(value["truePositives"]) for value in per_video.values())
    false_positives = sum(int(value["falsePositives"]) for value in per_video.values())
    false_negatives = sum(int(value["falseNegatives"]) for value in per_video.values())
    candidates = true_positives + false_positives
    human_events = true_positives + false_negatives
    return {
        "paddingSeconds": padding_seconds,
        "recordings": len(per_video),
        "humanEvents": human_events,
        "candidates": candidates,
        "boundaryCandidates": sum(
            int(value["boundaryCandidates"]) for value in per_video.values()
        ),
        "internalPeakCandidates": sum(
            int(value["internalPeakCandidates"]) for value in per_video.values()
        ),
        "coveredHumanEvents": true_positives,
        "uncoveredHumanEvents": false_negatives,
        "candidateRecall": true_positives / human_events if human_events else 0.0,
        "candidateHitRate": true_positives / candidates if candidates else 0.0,
        "averageCandidatesPerVideo": candidates / len(per_video),
        "averageInternalPeaksPerVideo": sum(
            int(value["internalPeakCandidates"]) for value in per_video.values()
        )
        / len(per_video),
        "byRecording": per_video,
    }


def _select_config(
    config_metrics: Mapping[str, Mapping[str, Any]], recording_ids: Sequence[str]
) -> str:
    eligible: list[tuple[float, float, str]] = []
    human_events = sum(
        int(next(iter(config_metrics.values()))["byRecording"][recording_id]["humanEvents"])
        for recording_id in recording_ids
    )
    for config_id, metrics in config_metrics.items():
        candidates = sum(
            int(metrics["byRecording"][recording_id]["candidates"])
            for recording_id in recording_ids
        )
        true_positives = sum(
            int(metrics["byRecording"][recording_id]["truePositives"])
            for recording_id in recording_ids
        )
        recall = true_positives / human_events
        if recall >= TARGET_RECALL:
            eligible.append((candidates / len(recording_ids), -recall, config_id))
    if not eligible:
        raise ValueError("no candidate-union configuration reaches target recall")
    return min(eligible)[2]


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "manifest": args.manifest.expanduser().resolve(),
        "fullAudit": args.full_audit.expanduser().resolve(),
        "v5Features": args.v5_features.expanduser().resolve(),
        "continuity": args.continuity.expanduser().resolve(),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate-union artifact: {output}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"candidate-union source identity changed: {hashes}")
    manifest = _load(paths["manifest"])
    full_audit = _load(paths["fullAudit"])
    v5_features = _load(paths["v5Features"])
    continuity = _load(paths["continuity"])
    recording_ids = tuple(str(value) for value in full_audit["scope"]["recordingIds"])
    markers_by_recording = {
        recording_id: [
            {"time": float(value)}
            for value in full_audit["scope"]["humanEventsByRecording"][recording_id]
        ]
        for recording_id in recording_ids
    }
    records = {
        str(record["recordingId"]): record
        for record in manifest.get("records", [])
        if str(record.get("recordingId")) in recording_ids
    }
    if set(records) != set(recording_ids):
        raise ValueError("manifest does not cover candidate-union recordings")

    traces: dict[str, dict[str, Any]] = {}
    feedback_sources: dict[str, Any] = {}
    for recording_id in recording_ids:
        record = records[recording_id]
        feedback_path = Path(str(record["labelPath"])).resolve()
        feedback_hash = _sha256(feedback_path)
        feedback = _load(feedback_path)
        initial = feedback["initialInference"]
        if (
            initial.get("probabilityModelId") != "model-1ca43e38eefc"
            or initial.get("ensembleAlgorithmVersion")
            != "overlap-union-disagreement-v1"
        ):
            raise ValueError(f"production trace identity changed for {recording_id}")
        ranges = list(initial["ranges"])
        manifest_rallies = list(record["rallies"])
        if len(ranges) != len(manifest_rallies) or any(
            abs(float(source[key]) - float(rally[key])) > 1e-9
            for source, rally in zip(ranges, manifest_rallies, strict=True)
            for key in ("start", "end")
        ):
            raise ValueError(f"manifest ranges diverge from feedback for {recording_id}")
        times = _array(initial["timestamps"], "float64")
        rally = _array(initial["probabilities"]["rally"], "float32")
        dead = _array(initial["probabilities"]["deadState"], "float32")
        if times.shape != rally.shape or times.shape != dead.shape:
            raise ValueError(f"production probability shapes diverge for {recording_id}")
        traces[recording_id] = {
            "ranges": ranges,
            "times": times,
            "rally": rally,
            "dead": dead,
        }
        feedback_sources[recording_id] = {
            "path": str(feedback_path),
            "sha256": feedback_hash,
            "manifestSha256": str(record["labelSha256"]),
            "wholeFileMatchesManifest": feedback_hash == str(record["labelSha256"]),
            "initialInferenceSha256": _json_sha256(initial),
            "analysisFps": float(feedback["features"]["analysisFps"]),
            "samples": len(times),
            "probabilityModelId": str(initial["probabilityModelId"]),
            "ensembleAlgorithmVersion": str(initial["ensembleAlgorithmVersion"]),
        }

    boundary_by_recording = {
        recording_id: boundary_candidates(
            recording_id, traces[recording_id]["ranges"]
        )
        for recording_id in recording_ids
    }
    boundary_metrics = {
        str(padding): _metrics(
            boundary_by_recording,
            markers_by_recording,
            padding,
            include_inventory=False,
        )
        for padding in (0.0, 4.0)
    }
    candidates_by_config: dict[str, dict[str, list[dict[str, Any]]]] = {}
    grid_results: dict[str, Any] = {}
    for config in GRID:
        values = {
            recording_id: candidate_union(
                recording_id,
                traces[recording_id]["ranges"],
                traces[recording_id]["times"],
                traces[recording_id]["rally"],
                traces[recording_id]["dead"],
                config,
            )
            for recording_id in recording_ids
        }
        candidates_by_config[config.identifier] = values
        grid_results[config.identifier] = {
            "config": config.to_dict(),
            **_metrics(values, markers_by_recording, 4.0, include_inventory=False),
        }
    selected_id = (
        str(args.fixed_config_id)
        if args.fixed_config_id is not None
        else _select_config(grid_results, recording_ids)
    )
    if selected_id not in candidates_by_config:
        raise ValueError(f"unknown fixed candidate-union config: {selected_id}")
    selected_config = next(config for config in GRID if config.identifier == selected_id)
    selected_candidates = candidates_by_config[selected_id]
    selected_metrics = {
        str(padding): _metrics(
            selected_candidates,
            markers_by_recording,
            padding,
            include_inventory=True,
        )
        for padding in (0.0, 4.0)
    }

    loo_selections: dict[str, str] = {}
    loo_candidates: dict[str, list[dict[str, Any]]] = {}
    for held_out in recording_ids:
        fit_ids = [value for value in recording_ids if value != held_out]
        config_id = (
            selected_id
            if args.fixed_config_id is not None
            else _select_config(grid_results, fit_ids)
        )
        loo_selections[held_out] = config_id
        loo_candidates[held_out] = candidates_by_config[config_id][held_out]
    loo_metrics = _metrics(
        loo_candidates, markers_by_recording, 4.0, include_inventory=False
    )

    v5_rows = [
        row
        for row in v5_features.get("rows", [])
        if str(row.get("recordingId")) in recording_ids
    ]
    exact_v5_windows = {
        (
            str(row["recordingId"]),
            round(float(row["gapStart"]), 6),
            round(float(row["gapEnd"]), 6),
        )
        for row in v5_rows
    }
    scoreable_candidates = [
        candidate
        for values in selected_candidates.values()
        for candidate in values
        if (
            str(candidate["recordingId"]),
            round(float(candidate["gapStart"]), 6),
            round(float(candidate["gapEnd"]), 6),
        )
        in exact_v5_windows
    ]
    continuity_primary = continuity["primary"]["result"]
    continuity_variant = next(
        value for value in continuity["variants"] if value["name"] == "player"
    )
    compatible_output = continuity_variant["vetoOnly"]
    if (
        compatible_output["proposals"],
        compatible_output["truePositives"],
        compatible_output["falsePositives"],
    ) != (57, 25, 32):
        raise ValueError("candidate union continuity source changed")

    payload = {
        "schemaVersion": 1,
        "kind": (
            "volleycut-side-switch-candidate-union-fixed-evaluation-v1"
            if args.fixed_config_id is not None
            else "volleycut-side-switch-candidate-union-evaluation-v1"
        ),
        "createdAt": datetime.now(UTC).isoformat(),
        "scope": {
            "status": "opened-development-only",
            "recordingIds": list(recording_ids),
            "recordings": len(recording_ids),
            "humanMarkers": sum(map(len, markers_by_recording.values())),
        },
        "target": {
            "candidateRecall": TARGET_RECALL,
            "selectionRule": "minimum average candidates per video reaching target on fit recordings; tie by higher recall then stable config ID",
            "primaryPaddingSeconds": 4.0,
        },
        "existingCandidateUniverse": full_audit["candidateUniverseCoverage"],
        "allAdjacentBoundaries": boundary_metrics,
        "grid": grid_results,
        "selected": {
            "configId": selected_id,
            "config": selected_config.to_dict(),
            "selectionScope": (
                "preregistered fixed configuration from the v1 opened-development grid"
                if args.fixed_config_id is not None
                else "all opened development recordings"
            ),
            "metrics": selected_metrics,
        },
        "leaveOneRecordingOut": {
            "selectedConfigByHeldOutRecording": loo_selections,
            "metrics": loo_metrics,
        },
        "downstreamCompatibility": {
            "selectedUnionCandidates": selected_metrics["4.0"]["candidates"],
            "candidatesWithExactFrozenV5FeatureWindow": len(scoreable_candidates),
            "candidatesNeedingNewAppearanceFeatures": (
                selected_metrics["4.0"]["candidates"] - len(scoreable_candidates)
            ),
            "compatibleFrozenContinuityOutput": {
                key: compatible_output[key]
                for key in (
                    "proposals",
                    "truePositives",
                    "falsePositives",
                    "falseNegatives",
                    "precision",
                    "recall",
                    "f1",
                )
            },
            "conclusion": "The frozen V5 winner and continuity veto can score only exact legacy windows; new boundary and internal-peak features must be extracted before the high-recall union can change final output.",
            "continuityPrimaryIdentity": continuity_primary,
        },
        "runtime": {
            "newModelInference": False,
            "newVideoDecode": False,
            "inputs": "existing 4 Hz dead-state probabilities and decoded production ranges",
            "candidateLogic": "linear scan plus score-ranked within-range NMS",
        },
        "feedbackSources": feedback_sources,
        "sources": {
            name: {"path": str(paths[name]), "sha256": hashes[name]}
            for name in paths
        },
        "limitations": [
            "All marker truth is opened development and configuration selection is not independent confirmation.",
            "Candidate hit rate is not model precision; candidates are an internal high-recall universe.",
            "Four events remain outside the selected union at ±4 seconds.",
            "Only legacy windows have the frozen V5 feature signature, so final decoder output is unchanged in this loop.",
        ],
    }
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--full-audit", type=Path, default=DEFAULT_FULL_AUDIT)
    parser.add_argument("--v5-features", type=Path, default=DEFAULT_V5_FEATURES)
    parser.add_argument("--continuity", type=Path, default=DEFAULT_CONTINUITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fixed-config-id",
        help="emit one fixed grid configuration instead of selecting by target recall",
    )
    parser.add_argument(
        "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    payload = run(_parser().parse_args())
    print(
        json.dumps(
            {
                "boundaries": payload["allAdjacentBoundaries"]["4.0"],
                "selectedConfig": payload["selected"]["config"],
                "selected": {
                    key: payload["selected"]["metrics"]["4.0"][key]
                    for key in (
                        "candidates",
                        "boundaryCandidates",
                        "internalPeakCandidates",
                        "coveredHumanEvents",
                        "uncoveredHumanEvents",
                        "candidateRecall",
                        "candidateHitRate",
                        "averageCandidatesPerVideo",
                    )
                },
                "leaveOneRecordingOut": {
                    "configIds": sorted(
                        set(
                            payload["leaveOneRecordingOut"][
                                "selectedConfigByHeldOutRecording"
                            ].values()
                        )
                    ),
                    "metrics": payload["leaveOneRecordingOut"]["metrics"],
                },
                "downstreamCompatibility": payload["downstreamCompatibility"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
