#!/usr/bin/env python3
"""Freeze and evaluate the cadence-first side-switch v3 specialist."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_training_policy import validate_side_switch_fit_recordings
from analysis.side_switch_v3 import (
    FEATURE_ARTIFACT_KIND,
    FROZEN_RECORDING_SPLIT,
    MODEL_KIND,
    MODEL_SCHEMA_VERSION,
    DecoderSettings,
    V3Event,
    average_precision,
    decode_all,
    event_metrics,
    fingerprint,
    fit_model,
    grouped_cross_fit,
    labels_for,
    matrix_for,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_FEATURES = ROOT / "reports/side-switch/side-switch-v3-features.json"
DEFAULT_MODEL_DIR = ROOT / "models/side-switch-specialist-v3-reanchored-capped6"
DEFAULT_EVALUATION = (
    ROOT
    / "reports/side-switch/side-switch-specialist-v3-reanchored-capped6-evaluation.json"
)
EXPECTED_FEATURE_SHA256 = (
    "e50bad8020d0e1992b65617bd2a8d2ce7f9a5aa407fb0834c19e0921e35c05fa"
)
L2_GRID = (0.01, 0.1, 1.0, 10.0)
MARGIN_GRID = (1, 2, 3, 4)
DISTANCE_PENALTY_GRID = (0.0, 0.1, 0.25, 0.5)


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


def _load_features(path: Path, enforce_source_hash: bool) -> Mapping[str, Any]:
    if enforce_source_hash and _sha256(path) != EXPECTED_FEATURE_SHA256:
        raise ValueError("v3 feature artifact identity changed")
    payload = _load(path)
    if payload.get("kind") != FEATURE_ARTIFACT_KIND:
        raise ValueError("feature artifact is not side-switch v3")
    if payload.get("frozenRecordingSplit") != {
        role: list(recording_ids)
        for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
    }:
        raise ValueError("v3 feature artifact does not match the frozen split")
    return payload


def _events(payload: Mapping[str, Any], roles: set[str]) -> list[V3Event]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("v3 feature artifact has no rows")
    events = [
        V3Event(
            event_id=str(row["eventId"]),
            recording_id=str(row["recordingId"]),
            role=str(row["role"]),
            gap_order=int(row["gapOrder"]),
            label=int(row["label"]),
            row=row,
        )
        for row in rows
        if isinstance(row, Mapping) and str(row.get("role")) in roles
    ]
    if not events:
        raise ValueError(f"v3 feature artifact has no events for roles {sorted(roles)}")
    if any(event.label not in {0, 1} for event in events):
        raise ValueError("v3 event labels must be binary")
    return events


def _counts(events: Sequence[V3Event]) -> dict[str, Any]:
    return {
        "rows": len(events),
        "switch": sum(event.label for event in events),
        "noSwitch": sum(1 - event.label for event in events),
        "recordings": len({event.recording_id for event in events}),
        "recordingIds": sorted({event.recording_id for event in events}),
        "sourceGroups": sorted(
            {str(event.row.get("sourceGroup", "unknown")) for event in events}
        ),
    }


def _cadence_predictions(
    events: Sequence[V3Event], margin: int
) -> np.ndarray:
    return decode_all(
        events,
        np.full(len(events), 0.5, dtype=np.float64),
        0.5,
        DecoderSettings(candidate_margin=margin, distance_penalty=1.0),
        force_each_opportunity=True,
    )


def _metric_sensitivity(
    events: Sequence[V3Event], predictions: np.ndarray
) -> dict[str, Any]:
    return {
        str(tolerance): event_metrics(events, predictions, tolerance=tolerance)
        for tolerance in (0, 1, 2)
    }


def _by_recording(
    events: Sequence[V3Event], predictions: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for recording_id in sorted({event.recording_id for event in events}):
        indexes = [
            index for index, event in enumerate(events) if event.recording_id == recording_id
        ]
        subset = [events[index] for index in indexes]
        result[recording_id] = _metric_sensitivity(subset, predictions[indexes])
    return result


def _decoder_candidates(
    events: Sequence[V3Event], probabilities: np.ndarray
) -> list[dict[str, Any]]:
    thresholds = sorted({float(value) for value in probabilities}, reverse=True)
    thresholds.insert(0, math.nextafter(thresholds[0], math.inf))
    candidates: list[dict[str, Any]] = []
    for margin in MARGIN_GRID:
        for distance_penalty in DISTANCE_PENALTY_GRID:
            settings = DecoderSettings(margin, distance_penalty)
            for threshold in thresholds:
                predictions = decode_all(events, probabilities, threshold, settings)
                metrics = event_metrics(events, predictions, tolerance=0)
                candidates.append(
                    {
                        "threshold": threshold,
                        "settings": settings.to_dict(),
                        "metrics": metrics,
                    }
                )
    return candidates


def _decoder_rank(candidate: Mapping[str, Any]) -> tuple[float, float, float, int, float, float]:
    metrics = candidate["metrics"]
    settings = candidate["settings"]
    return (
        float(metrics["f1"] or 0.0),
        float(metrics["precision"] or 0.0),
        float(metrics["recall"] or 0.0),
        -int(settings["candidateMargin"]),
        -float(settings["distancePenalty"]),
        float(candidate["threshold"]),
    )


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    features_path = args.features.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    if model_dir.exists():
        raise FileExistsError(f"refusing to reuse v3 model directory: {model_dir}")
    features = _load_features(features_path, args.enforce_source_hash)
    train_events = _events(features, {"train"})
    validation_events = _events(features, {"validation"})
    validate_side_switch_fit_recordings(
        sorted({event.recording_id for event in train_events})
    )

    leaderboard: list[dict[str, Any]] = []
    for l2 in L2_GRID:
        probabilities, cross_fit = grouped_cross_fit(train_events, l2)
        leaderboard.append(
            {
                "l2": l2,
                "outOfFoldAveragePrecision": average_precision(
                    labels_for(train_events), probabilities
                ),
                "folds": cross_fit["folds"],
            }
        )
    leaderboard.sort(
        key=lambda row: (
            float(row["outOfFoldAveragePrecision"] or 0.0),
            -float(row["l2"]),
        ),
        reverse=True,
    )
    selected_l2 = float(leaderboard[0]["l2"])
    model = fit_model(train_events, selected_l2)
    validation_probabilities = model.predict_proba(matrix_for(validation_events))
    decoder_candidates = _decoder_candidates(
        validation_events, validation_probabilities
    )
    selected_decoder = max(decoder_candidates, key=_decoder_rank)
    model = replace(model, threshold=float(selected_decoder["threshold"]))
    decoder = DecoderSettings.from_dict(selected_decoder["settings"])
    validation_predictions = decode_all(
        validation_events, validation_probabilities, model.threshold, decoder
    )
    cadence_validation = {
        str(margin): _metric_sensitivity(
            validation_events, _cadence_predictions(validation_events, margin)
        )
        for margin in MARGIN_GRID
    }
    deployable = {"classifier": model.to_dict(), "decoder": decoder.to_dict()}
    created_at = datetime.now(UTC).isoformat()
    feature_source = {
        "path": str(features_path),
        "sha256": _sha256(features_path),
        "createdAt": features.get("createdAt"),
    }
    model_payload = {
        "schemaVersion": MODEL_SCHEMA_VERSION,
        "kind": MODEL_KIND,
        "createdAt": created_at,
        "fingerprint": fingerprint(deployable),
        **deployable,
        "selection": {
            "modelFamily": "fixed 12-input low-resolution linear logistic ranker",
            "l2": "maximum recording-held-out training average precision",
            "decoder": (
                "maximum exact-gap validation event F1; precision, recall, smaller "
                "margin/penalty, then threshold tie-break"
            ),
            "l2Grid": list(L2_GRID),
            "candidateMarginGrid": list(MARGIN_GRID),
            "distancePenaltyGrid": list(DISTANCE_PENALTY_GRID),
            "l2Leaderboard": leaderboard,
            "selectedDecoder": selected_decoder,
        },
        "counts": {
            "train": _counts(train_events),
            "validation": _counts(validation_events),
        },
        "validation": {
            "visualRowAveragePrecision": average_precision(
                labels_for(validation_events), validation_probabilities
            ),
            "selectedVisual": _metric_sensitivity(
                validation_events, validation_predictions
            ),
            "cadenceOnlyByMargin": cadence_validation,
        },
        "dataPolicy": {
            "oneRecordingOneSet": True,
            "startPointTotal": 0,
            "cadencePoints": 7,
            "maximumOpportunities": 6,
            "reanchorOnSelectedSwitch": True,
            "fitExclusions": ["beach-source-02"],
            "evaluationLabelsOpenedDuringSelection": False,
            "confirmationStatus": "retrospective; labels were used by historical v1/v2 research",
        },
        "sources": {"features": feature_source},
    }
    dataset_payload = {
        "schemaVersion": 3,
        "kind": "volleycut-side-switch-specialist-development-dataset-v3",
        "createdAt": created_at,
        "modelFingerprint": model_payload["fingerprint"],
        "featureNames": list(model.feature_names),
        "counts": model_payload["counts"],
        "rows": [event.row for event in (*train_events, *validation_events)],
        "sources": model_payload["sources"],
    }
    atomic_write_text(
        model_dir / "model.json",
        json.dumps(model_payload, indent=2, allow_nan=False) + "\n",
    )
    atomic_write_text(
        model_dir / "dataset-development.json",
        json.dumps(dataset_payload, indent=2, allow_nan=False) + "\n",
    )
    return model_payload


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    features_path = args.features.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_path = args.evaluation_output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite v3 evaluation: {output_path}")
    features = _load_features(features_path, args.enforce_source_hash)
    model_payload = _load(model_dir / "model.json")
    if model_payload.get("kind") != MODEL_KIND:
        raise ValueError("model artifact is not side-switch v3")
    deployable = {
        "classifier": model_payload["classifier"],
        "decoder": model_payload["decoder"],
    }
    if model_payload.get("fingerprint") != fingerprint(deployable):
        raise ValueError("v3 model fingerprint is invalid")
    if model_payload.get("sources", {}).get("features", {}).get("sha256") != _sha256(
        features_path
    ):
        raise ValueError("v3 model and feature artifact do not match")
    from analysis.side_switch_v3 import V3Model

    model = V3Model.from_dict(model_payload["classifier"])
    decoder = DecoderSettings.from_dict(model_payload["decoder"])
    events = _events(features, {"evaluation"})
    probabilities = model.predict_proba(matrix_for(events))
    selected_predictions = decode_all(
        events, probabilities, model.threshold, decoder
    )
    cadence_by_margin: dict[str, Any] = {}
    visual_by_margin: dict[str, Any] = {}
    cadence_predictions: dict[int, np.ndarray] = {}
    visual_predictions: dict[int, np.ndarray] = {}
    for margin in MARGIN_GRID:
        cadence = _cadence_predictions(events, margin)
        visual = decode_all(
            events,
            probabilities,
            model.threshold,
            DecoderSettings(margin, decoder.distance_penalty),
        )
        cadence_predictions[margin] = cadence
        visual_predictions[margin] = visual
        cadence_by_margin[str(margin)] = _metric_sensitivity(events, cadence)
        visual_by_margin[str(margin)] = _metric_sensitivity(events, visual)
    payload = {
        "schemaVersion": 3,
        "kind": "volleycut-side-switch-specialist-evaluation-v3",
        "createdAt": datetime.now(UTC).isoformat(),
        "modelFingerprint": model_payload["fingerprint"],
        "modelPath": str(model_dir / "model.json"),
        "developmentDatasetPath": str(model_dir / "dataset-development.json"),
        "evaluationPath": str(output_path),
        "counts": _counts(events),
        "selectedDecoder": {
            "threshold": model.threshold,
            **decoder.to_dict(),
        },
        "visualRowAveragePrecision": average_precision(
            labels_for(events), probabilities
        ),
        "selectedVisual": {
            "overall": _metric_sensitivity(events, selected_predictions),
            "byRecording": _by_recording(events, selected_predictions),
        },
        "cadenceOnlyByMargin": cadence_by_margin,
        "visualByMargin": visual_by_margin,
        "plusMinusFour": {
            "cadenceOnly": cadence_by_margin["4"],
            "cadencePlusVisual": visual_by_margin["4"],
            "overlapPolicy": (
                "monotonic one-to-one assignment; each gap and each seven-point "
                "opportunity can be selected at most once"
            ),
        },
        "knownLimitations": {
            "candidateConditioned": (
                "raw-phone metrics cover reviewed rally-detector gaps; unproposed gaps "
                "have no review label"
            ),
            "retrospectiveConfirmation": (
                "the raw recordings were opened by historical side-switch research; "
                "this is not a pristine promotion test"
            ),
            "scoreProxy": (
                "rally gap order is a noisy point-total proxy because of re-dos and "
                "missed/extra rally detections"
            ),
        },
        "predictions": [
            {
                "eventId": event.event_id,
                "recordingId": event.recording_id,
                "gapOrder": event.gap_order,
                "decision": event.row.get("decision"),
                "score": float(score),
                "selectedPrediction": bool(selected),
                "cadenceByMargin": {
                    str(margin): bool(cadence_predictions[margin][index])
                    for margin in MARGIN_GRID
                },
                "visualByMargin": {
                    str(margin): bool(visual_predictions[margin][index])
                    for margin in MARGIN_GRID
                },
            }
            for index, (event, score, selected) in enumerate(
                zip(events, probabilities, selected_predictions, strict=True)
            )
        ],
        "sources": model_payload["sources"],
    }
    atomic_write_text(output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "evaluate"):
        child = commands.add_parser(command)
        child.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
        child.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
        child.add_argument(
            "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
        )
        if command == "evaluate":
            child.add_argument(
                "--evaluation-output", type=Path, default=DEFAULT_EVALUATION
            )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "freeze":
        payload = freeze(args)
        print(
            json.dumps(
                {
                    "fingerprint": payload["fingerprint"],
                    "l2": payload["classifier"]["l2"],
                    "threshold": payload["classifier"]["threshold"],
                    "decoder": payload["decoder"],
                    "validation": payload["validation"],
                },
                indent=2,
            )
        )
    else:
        payload = evaluate(args)
        print(
            json.dumps(
                {
                    "selected": payload["selectedVisual"]["overall"],
                    "cadenceOnlyByMargin": payload["cadenceOnlyByMargin"],
                    "visualByMargin": payload["visualByMargin"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
