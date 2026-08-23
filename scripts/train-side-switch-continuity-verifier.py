#!/usr/bin/env python3
"""Fit and evaluate an abstaining same-side continuity verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_continuity import (
    VerifierRow,
    apply_positive,
    apply_veto,
    build_verifier_rows,
    select_positive_threshold,
    select_veto_threshold,
)
from analysis.side_switch_full_video import (
    event_metric_counts,
    monotonic_interval_match,
)


REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_FEATURES = REPORTS / "side-switch-v5-player-orientation-features.json"
DEFAULT_FULL_AUDIT = REPORTS / "side-switch-full-video-marker-evaluation-2026-08-21-r2.json"
DEFAULT_WINNER = REPOSITORY / "data/side-switch-current-research-winner-v1.json"
DEFAULT_MODEL = ROOT / "models/side-switch-continuity-verifier-v1/model.json"
DEFAULT_EVALUATION = REPORTS / "side-switch-continuity-verifier-v1-evaluation.json"
EXPECTED_SHA256 = {
    "features": "4406fe47de0326c1257b8d9cf353116a9495f82d2e43c92e9721b1d6764ba5db",
    "fullAudit": "142d6617aed7f20b51cdcdfc9b3e61c8c76beaf79a31fc2d7ee54fb0fdf27b89",
    "winner": "ea1423a3dd7f96812b401dbe7a8cdb2fe8435d4e7f2a290356eb2c0cfba4fb8f",
}
VARIANTS = (
    {"name": "player", "signal": "player", "minimumQuality": 0.0},
    {
        "name": "player-quality-half-median",
        "signal": "player",
        "minimumQuality": 0.5,
    },
    {"name": "v4", "signal": "v4", "minimumQuality": 0.0},
    {"name": "agreement", "signal": "agreement", "minimumQuality": 0.0},
    {
        "name": "agreement-quality-half-median",
        "signal": "agreement",
        "minimumQuality": 0.5,
    },
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


def _proposal(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "eventId": str(row["eventId"]),
        "recordingId": str(row["recordingId"]),
        "gapOrder": int(row["gapOrder"]),
        "gapStart": float(row["gapStart"]),
        "gapEnd": float(row["gapEnd"]),
        "transitionTime": float(row["transitionTime"]),
    }


def _evaluate_ids(
    selected_by_recording: Mapping[str, set[str]],
    source_by_id: Mapping[str, Mapping[str, Any]],
    markers_by_recording: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    per_video: dict[str, Any] = {}
    inventory: dict[str, list[dict[str, Any]]] = {}
    for recording_id, markers in markers_by_recording.items():
        proposals = sorted(
            [
                _proposal(source_by_id[event_id])
                for event_id in selected_by_recording.get(recording_id, set())
            ],
            key=lambda value: (value["transitionTime"], value["gapOrder"]),
        )
        match = monotonic_interval_match(proposals, markers, 4.0)
        counts = event_metric_counts(match)
        per_video[recording_id] = {
            "humanEvents": len(markers),
            "proposals": len(proposals),
            **counts,
            "matchedEventIds": [
                proposals[pair.proposal_index]["eventId"] for pair in match.pairs
            ],
            "falsePositiveEventIds": [
                proposals[index]["eventId"]
                for index in match.unmatched_proposal_indices
            ],
            "missedHumanTimes": [
                float(markers[index]["time"])
                for index in match.unmatched_marker_indices
            ],
        }
        inventory[recording_id] = proposals
    true_positives = sum(int(value["truePositives"]) for value in per_video.values())
    false_positives = sum(int(value["falsePositives"]) for value in per_video.values())
    false_negatives = sum(int(value["falseNegatives"]) for value in per_video.values())
    proposals = true_positives + false_positives
    human_events = true_positives + false_negatives
    precision = true_positives / proposals if proposals else 0.0
    recall = true_positives / human_events if human_events else 0.0
    return {
        "recordings": len(per_video),
        "humanEvents": human_events,
        "proposals": proposals,
        "truePositives": true_positives,
        "falsePositives": false_positives,
        "falseNegatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "macroPerVideoPrecision": float(
            np.mean(
                [
                    float(value["precision"] or 0.0)
                    for value in per_video.values()
                ]
            )
        ),
        "macroPerVideoRecall": float(
            np.mean([float(value["recall"] or 0.0) for value in per_video.values()])
        ),
        "averagePerVideo": {
            "proposals": proposals / len(per_video),
            "truePositives": true_positives / len(per_video),
            "falsePositives": false_positives / len(per_video),
            "falseNegatives": false_negatives / len(per_video),
        },
        "byRecording": per_video,
        "proposalInventory": inventory,
    }


def _auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    if not positive or not negative:
        return 0.5
    return sum(
        float(left > right) + 0.5 * float(left == right)
        for left in positive
        for right in negative
    ) / (len(positive) * len(negative))


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "features": args.features.expanduser().resolve(),
        "fullAudit": args.full_audit.expanduser().resolve(),
        "winner": args.winner.expanduser().resolve(),
    }
    model_path = args.model.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    if model_path.exists() or evaluation_path.exists():
        raise FileExistsError("refusing to overwrite continuity verifier artifacts")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"continuity verifier source identity changed: {hashes}")
    features = _load(paths["features"])
    full_audit = _load(paths["fullAudit"])
    winner = _load(paths["winner"])
    if winner.get("winner", {}).get("evaluationDecoderId") != "cleanup/local-peak-soft-count":
        raise ValueError("continuity verifier control is not the current winner")
    recording_ids = tuple(str(value) for value in full_audit["scope"]["recordingIds"])
    human_times = full_audit["scope"]["humanEventsByRecording"]
    markers_by_recording = {
        recording_id: [{"time": float(value)} for value in human_times[recording_id]]
        for recording_id in recording_ids
    }
    decoder = full_audit["decoders"]["cleanup/local-peak-soft-count"]
    winner_ids_by_recording = {
        recording_id: {
            str(proposal["eventId"])
            for proposal in decoder["proposalInventory"][recording_id]
        }
        for recording_id in recording_ids
    }
    if sum(map(len, winner_ids_by_recording.values())) != 62:
        raise ValueError("continuity control does not contain the frozen 62 proposals")
    raw_rows_by_recording: dict[str, list[Mapping[str, Any]]] = {
        recording_id: [] for recording_id in recording_ids
    }
    for row in features.get("rows", []):
        recording_id = str(row.get("recordingId", ""))
        if recording_id in raw_rows_by_recording:
            if row.get("status") != "ok":
                raise ValueError(f"continuity row has extraction error: {row.get('eventId')}")
            raw_rows_by_recording[recording_id].append(row)
    if sum(map(len, raw_rows_by_recording.values())) != 352:
        raise ValueError("continuity verifier expects the frozen 352-gap universe")

    labels: dict[str, int] = {}
    for recording_id, rows in raw_rows_by_recording.items():
        rows.sort(key=lambda row: (int(row["gapOrder"]), str(row["eventId"])))
        proposals = [_proposal(row) for row in rows]
        match = monotonic_interval_match(
            proposals, markers_by_recording[recording_id], 4.0
        )
        positive_ids = {
            proposals[pair.proposal_index]["eventId"] for pair in match.pairs
        }
        labels.update(
            {
                str(row["eventId"]): int(str(row["eventId"]) in positive_ids)
                for row in rows
            }
        )
    if sum(labels.values()) != 33:
        raise ValueError("continuity labels do not reproduce the 33-event candidate ceiling")

    verifier_rows_by_recording = {
        recording_id: build_verifier_rows(
            rows, labels, winner_ids_by_recording[recording_id]
        )
        for recording_id, rows in raw_rows_by_recording.items()
    }
    all_rows = [
        row
        for recording_id in recording_ids
        for row in verifier_rows_by_recording[recording_id]
    ]
    source_by_id = {row.event_id: row.source for row in all_rows}
    baseline = _evaluate_ids(
        winner_ids_by_recording, source_by_id, markers_by_recording
    )
    if (
        baseline["truePositives"],
        baseline["falsePositives"],
        baseline["falseNegatives"],
    ) != (25, 37, 25):
        raise ValueError("continuity verifier did not reproduce the current winner")

    fixed_same_cheaper_ids = {
        recording_id: {
            row.event_id
            for row in verifier_rows_by_recording[recording_id]
            if row.selected_by_control
            and float(row.source["features"]["playerSwapMargin"]) > 0.0
        }
        for recording_id in recording_ids
    }
    fixed_same_cheaper = _evaluate_ids(
        fixed_same_cheaper_ids, source_by_id, markers_by_recording
    )

    variant_results: list[dict[str, Any]] = []
    for variant in VARIANTS:
        name = str(variant["name"])
        signal = str(variant["signal"])
        minimum_quality = float(variant["minimumQuality"])
        thresholds: dict[str, Any] = {}
        veto_ids: dict[str, set[str]] = {}
        verifier_ids: dict[str, set[str]] = {}
        add_ids: dict[str, set[str]] = {}
        for held_out in recording_ids:
            fit_ids = [value for value in recording_ids if value != held_out]
            fit_rows = [
                row
                for recording_id in fit_ids
                for row in verifier_rows_by_recording[recording_id]
            ]
            fit_marker_count = sum(len(markers_by_recording[value]) for value in fit_ids)
            veto_selection = select_veto_threshold(
                fit_rows,
                signal=signal,
                minimum_quality=minimum_quality,
                marker_count=fit_marker_count,
                minimum_true_positive_retention=0.90,
            )
            positive_selection = select_positive_threshold(
                fit_rows,
                signal=signal,
                minimum_quality=minimum_quality,
                marker_count=fit_marker_count,
            )
            held_rows = verifier_rows_by_recording[held_out]
            veto_ids[held_out] = apply_veto(
                held_rows,
                signal=signal,
                threshold=float(veto_selection["threshold"]),
                minimum_quality=minimum_quality,
            )
            verifier_ids[held_out] = apply_positive(
                held_rows,
                signal=signal,
                threshold=float(positive_selection["threshold"]),
                minimum_quality=minimum_quality,
            )
            add_ids[held_out] = (
                verifier_ids[held_out] | winner_ids_by_recording[held_out]
            )
            thresholds[held_out] = {
                "veto": {
                    key: value
                    for key, value in veto_selection.items()
                    if key != "retainedEventIds"
                },
                "positive": {
                    key: value
                    for key, value in positive_selection.items()
                    if key != "selectedEventIds"
                },
            }
        variant_results.append(
            {
                "name": name,
                "signal": signal,
                "minimumQuality": minimum_quality,
                "leaveOneRecordingOut": True,
                "thresholdsByHeldOutRecording": thresholds,
                "vetoOnly": _evaluate_ids(
                    veto_ids, source_by_id, markers_by_recording
                ),
                "verifierAlone": _evaluate_ids(
                    verifier_ids, source_by_id, markers_by_recording
                ),
                "addOnly": _evaluate_ids(
                    add_ids, source_by_id, markers_by_recording
                ),
            }
        )

    primary = next(value for value in variant_results if value["name"] == "player")
    final_selection = select_veto_threshold(
        all_rows,
        signal="player",
        minimum_quality=0.0,
        marker_count=sum(map(len, markers_by_recording.values())),
        minimum_true_positive_retention=0.90,
    )
    player_control = [row for row in all_rows if row.selected_by_control]
    positive_scores = [row.player_switch_evidence for row in player_control if row.label]
    negative_scores = [
        row.player_switch_evidence for row in player_control if not row.label
    ]
    created_at = datetime.now(UTC).isoformat()
    model_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-continuity-verifier-v1",
        "createdAt": created_at,
        "status": "opened-development-forward-research-only",
        "automaticProductionUse": False,
        "control": winner["winner"],
        "signal": {
            "name": "player",
            "rawFeature": "playerSwapMargin",
            "meaning": "same assignment cost minus swapped assignment cost; lower is stronger same-side continuity",
            "recordingNormalization": "subtract recording median and divide by max(1.4826*MAD, 0.25*standard deviation, 1e-6)",
        },
        "decision": {
            "mode": "veto-only",
            "vetoWhen": "selected-by-control and normalized player switch evidence <= threshold",
            "threshold": final_selection["threshold"],
            "minimumQuality": 0.0,
            "minimumDevelopmentTruePositiveRetention": 0.90,
            "trainingCounts": final_selection["trainingCounts"],
        },
        "fitScope": {
            "recordingIds": list(recording_ids),
            "recordings": len(recording_ids),
            "humanMarkers": 50,
            "candidateRows": len(all_rows),
            "controlProposals": 62,
            "note": "All recordings are opened development; leave-one-recording-out transfer is in the evaluation artifact.",
        },
        "sources": {
            name: {"path": str(paths[name]), "sha256": hashes[name]}
            for name in paths
        },
    }
    evaluation_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-continuity-verifier-evaluation-v1",
        "createdAt": created_at,
        "modelPath": str(model_path),
        "scope": model_payload["fitScope"],
        "matching": full_audit["matching"],
        "labelPolicy": {
            "source": "exhaustive full-video markers",
            "candidatePositiveAssignment": "monotonic one-to-one matching across all 352 gaps at ±4 seconds",
            "candidatePositives": sum(labels.values()),
            "candidateNegatives": len(labels) - sum(labels.values()),
        },
        "control": baseline,
        "fixedSameCheaperHardGate": {
            "rule": "retain a control proposal only when raw playerSwapMargin > 0",
            "result": fixed_same_cheaper,
        },
        "signalDiagnostic": {
            "playerControlAuc": _auc(positive_scores, negative_scores),
            "trueProposalMean": float(np.mean(positive_scores)),
            "falseProposalMean": float(np.mean(negative_scores)),
            "trueProposalMedian": float(np.median(positive_scores)),
            "falseProposalMedian": float(np.median(negative_scores)),
        },
        "variants": variant_results,
        "primary": {
            "variant": "player",
            "mode": "vetoOnly",
            "result": primary["vetoOnly"],
            "selection": "best leave-one-recording-out veto F1 among preregistered direct/quality sensitivities; forward threshold fit on all opened development is stored in model.json",
        },
        "sources": model_payload["sources"],
        "limitations": [
            "All 11 recordings were opened by prior side-switch work and are development scope.",
            "The verifier reuses a V5 feature already available to the control head; it tests a separately constrained veto, not independent visual evidence.",
            "The verifier cannot recover switches outside the existing candidate rows and is not an on-device or review-UI promotion.",
        ],
    }
    atomic_write_text(
        model_path, json.dumps(model_payload, indent=2, allow_nan=False) + "\n"
    )
    atomic_write_text(
        evaluation_path,
        json.dumps(evaluation_payload, indent=2, allow_nan=False) + "\n",
    )
    return model_payload, evaluation_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--full-audit", type=Path, default=DEFAULT_FULL_AUDIT)
    parser.add_argument("--winner", type=Path, default=DEFAULT_WINNER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument(
        "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    model, evaluation = run(_parser().parse_args())
    print(
        json.dumps(
            {
                "forwardThreshold": model["decision"]["threshold"],
                "control": {
                    key: evaluation["control"][key]
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
                "fixedSameCheaperHardGate": {
                    key: evaluation["fixedSameCheaperHardGate"]["result"][key]
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
                "variants": {
                    value["name"]: {
                        mode: {
                            key: value[mode][key]
                            for key in (
                                "proposals",
                                "truePositives",
                                "falsePositives",
                                "falseNegatives",
                                "precision",
                                "recall",
                                "f1",
                            )
                        }
                        for mode in ("vetoOnly", "verifierAlone", "addOnly")
                    }
                    for value in evaluation["variants"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
