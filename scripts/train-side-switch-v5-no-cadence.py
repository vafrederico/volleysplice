#!/usr/bin/env python3
"""Freeze and evaluate V5 production-state variants without cadence decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_no_cadence import (
    IndependentGapDecoderSettings,
    independent_gap_predictions,
    prediction_structure,
    select_independent_threshold,
)
from analysis.side_switch_production_state import STATE_GATE_FEATURE_NAMES
from analysis.side_switch_training_policy import validate_side_switch_fit_recordings
from analysis.side_switch_v3 import (
    FROZEN_RECORDING_SPLIT,
    V3Event,
    average_precision,
    event_metrics,
)
from analysis.side_switch_v5 import VISUAL_FEATURE_NAMES as V5_FEATURE_NAMES
from analysis.side_switch_v6 import (
    V6Model,
    fingerprint,
    fit_model,
    grouped_cross_fit,
    labels_for,
    matrix_for,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_FEATURES = (
    ROOT / "reports/side-switch/side-switch-v5-production-state-v1-features.json"
)
DEFAULT_CADENCE_MODEL = ROOT / "models/side-switch-v5-production-state-v1/model.json"
DEFAULT_CADENCE_EVALUATION = (
    ROOT / "reports/side-switch/side-switch-v5-production-state-v1-evaluation.json"
)
DEFAULT_MODEL_DIR = ROOT / "models/side-switch-v5-no-cadence-v1"
DEFAULT_EVALUATION = (
    ROOT / "reports/side-switch/side-switch-v5-no-cadence-v1-evaluation.json"
)
EXPECTED_FEATURE_SHA256 = (
    "c86b8ef7427dd8e1347d726c7d2326f18f22a6eb9fb16a0eec685c0fde0c9f36"
)
EXPECTED_CADENCE_MODEL_SHA256 = (
    "0fcbde8ed4deb5e3d805a336861918f9f52d5ad753a0f6ef1f6ee1bdda25a7c0"
)
EXPECTED_CADENCE_EVALUATION_SHA256 = (
    "7b243b5140fdbf08fb476d0679af314196744137588fd2e30f2f92e3d9bf3391"
)
FEATURE_KIND = "volleycut-side-switch-production-state-augmented-features-v1"
MODEL_KIND = "volleycut-side-switch-v5-no-cadence-specialist-v1"
EVALUATION_KIND = "volleycut-side-switch-v5-no-cadence-evaluation-v1"
DATASET_KIND = "volleycut-side-switch-v5-no-cadence-development-dataset-v1"
L2_GRID = (0.01, 0.1, 1.0, 10.0)
FEATURE_PROFILES = {
    "original:base": tuple(V5_FEATURE_NAMES),
    "original:state-gate": (*V5_FEATURE_NAMES, *STATE_GATE_FEATURE_NAMES),
}


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


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _load_features(path: Path) -> Mapping[str, Any]:
    if _sha256(path) != EXPECTED_FEATURE_SHA256:
        raise ValueError("V5 production-state feature artifact identity changed")
    payload = _load(path)
    expected_split = {
        role: list(recording_ids)
        for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
    }
    if (
        payload.get("kind") != FEATURE_KIND
        or payload.get("family") != "v5"
        or payload.get("appearanceMode") != "original"
        or payload.get("frozenRecordingSplit") != expected_split
    ):
        raise ValueError("V5 production-state feature contract changed")
    declared = tuple(payload.get("profile", {}).get("featureNames", []))
    expected = (
        *V5_FEATURE_NAMES,
        *STATE_GATE_FEATURE_NAMES,
        *payload["profile"]["serveAnchorFeatureNames"],
    )
    # The artifact declares base + state + serve banks; both no-cadence profiles
    # intentionally use frozen ordered prefixes/subsets from that artifact.
    if declared != expected:
        raise ValueError("V5 production-state feature signature changed")
    if payload.get("suppressionQuarantine", {}).get("eligibleForModelInput") is not False:
        raise ValueError("suppression quarantine is missing")
    return payload


def _events(payload: Mapping[str, Any], roles: set[str]) -> list[V3Event]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("feature artifact has no rows")
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
    if not events or any(event.label not in {0, 1} for event in events):
        raise ValueError(f"invalid events for roles {sorted(roles)}")
    return events


def _counts(events: Sequence[V3Event]) -> dict[str, Any]:
    return {
        "rows": len(events),
        "switch": sum(event.label for event in events),
        "noSwitch": sum(1 - event.label for event in events),
        "recordings": len({event.recording_id for event in events}),
        "recordingIds": sorted({event.recording_id for event in events}),
    }


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
            index
            for index, event in enumerate(events)
            if event.recording_id == recording_id
        ]
        result[recording_id] = _metric_sensitivity(
            [events[index] for index in indexes], predictions[indexes]
        )
    return result


def _head_parity(
    refitted: Mapping[str, Any], inherited: Mapping[str, Any]
) -> dict[str, Any]:
    if tuple(refitted["featureNames"]) != tuple(inherited["featureNames"]):
        raise ValueError("refitted and inherited feature signatures diverged")
    maximum_delta = 0.0
    for name in ("impute", "mean", "scale", "weights"):
        maximum_delta = max(
            maximum_delta,
            float(
                np.max(
                    np.abs(
                        np.asarray(refitted[name], dtype=np.float64)
                        - np.asarray(inherited[name], dtype=np.float64)
                    )
                )
            ),
        )
    for name in ("bias", "l2"):
        maximum_delta = max(
            maximum_delta, abs(float(refitted[name]) - float(inherited[name]))
        )
    if maximum_delta > 1e-12:
        raise ValueError(
            f"no-cadence refit changed the frozen classifier head: {maximum_delta}"
        )
    return {
        "parametersCompared": ["impute", "mean", "scale", "weights", "bias", "l2"],
        "maximumAbsoluteDifference": maximum_delta,
        "exactWithinTolerance": True,
        "excludedField": "threshold is decoder-specific and selected again on validation",
    }


def _fit_candidate(
    train_events: Sequence[V3Event],
    validation_events: Sequence[V3Event],
    feature_names: Sequence[str],
    inherited: Mapping[str, Any],
) -> dict[str, Any]:
    names = tuple(feature_names)
    leaderboard: list[dict[str, Any]] = []
    for l2 in L2_GRID:
        probabilities, cross_fit = grouped_cross_fit(train_events, l2, names)
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
    classifier = fit_model(train_events, selected_l2, names)
    parity = _head_parity(classifier.to_dict(), inherited["classifier"])
    validation_probabilities = classifier.predict_proba(
        matrix_for(validation_events, names)
    )
    threshold_selection, predictions = select_independent_threshold(
        validation_events, validation_probabilities
    )
    classifier = replace(
        classifier, threshold=float(threshold_selection["threshold"])
    )
    decoder = IndependentGapDecoderSettings()
    deployable = {
        "classifier": classifier.to_dict(),
        "decoder": decoder.to_dict(),
    }
    return {
        "featureNames": list(names),
        "fingerprint": fingerprint(deployable),
        **deployable,
        "l2Leaderboard": leaderboard,
        "headParityWithCadenceVariant": parity,
        "thresholdSelection": threshold_selection,
        "validation": {
            "rowAveragePrecision": average_precision(
                labels_for(validation_events), validation_probabilities
            ),
            "overall": _metric_sensitivity(validation_events, predictions),
            "predictionStructure": prediction_structure(
                validation_events, predictions
            ),
        },
    }


def _candidate_rank(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    exact = candidate["validation"]["overall"]["0"]
    return (
        float(exact["f1"] or 0.0),
        float(exact["precision"] or 0.0),
        float(exact["recall"] or 0.0),
        float(candidate["validation"]["rowAveragePrecision"] or 0.0),
        -len(candidate["featureNames"]),
    )


def freeze(args: argparse.Namespace) -> Mapping[str, Any]:
    features_path = args.features.expanduser().resolve()
    cadence_model_path = args.cadence_model.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    if model_dir.exists():
        raise FileExistsError(f"refusing to reuse model directory: {model_dir}")
    if _sha256(cadence_model_path) != EXPECTED_CADENCE_MODEL_SHA256:
        raise ValueError("cadence-model identity changed")
    features = _load_features(features_path)
    cadence_model = _load(cadence_model_path)
    if (
        cadence_model.get("kind")
        != "volleycut-side-switch-production-state-specialist-v1"
        or cadence_model.get("family") != "v5"
    ):
        raise ValueError("cadence-model contract changed")
    train_events = _events(features, {"train"})
    validation_events = _events(features, {"validation"})
    validate_side_switch_fit_recordings(
        sorted({event.recording_id for event in train_events})
    )

    candidates: dict[str, Any] = {}
    for candidate_id, names in FEATURE_PROFILES.items():
        inherited = cadence_model["candidates"].get(candidate_id)
        if not isinstance(inherited, Mapping):
            raise ValueError(f"cadence model is missing {candidate_id}")
        candidates[candidate_id] = _fit_candidate(
            train_events, validation_events, names, inherited
        )
        exact = candidates[candidate_id]["validation"]["overall"]["0"]
        print(
            f"Fitted no-cadence {candidate_id}: "
            f"F1={exact['f1']:.4f}, predictions={exact['predictedEvents']}",
            flush=True,
        )

    selected_id = max(candidates, key=lambda value: _candidate_rank(candidates[value]))
    selected = candidates[selected_id]
    created_at = datetime.now(UTC).isoformat()
    sources = {
        "features": {
            "path": str(features_path),
            "sha256": _sha256(features_path),
            "createdAt": features.get("createdAt"),
        },
        "cadenceModel": {
            "path": str(cadence_model_path),
            "sha256": _sha256(cadence_model_path),
            "selectedCandidate": cadence_model.get("selectedCandidate"),
        },
        "productionState": features.get("sources", {}).get("productionState"),
    }
    payload = {
        "schemaVersion": 1,
        "kind": MODEL_KIND,
        "createdAt": created_at,
        "family": "v5",
        "selectedCandidate": selected_id,
        "fingerprint": selected["fingerprint"],
        "classifier": selected["classifier"],
        "decoder": selected["decoder"],
        "candidates": candidates,
        "selection": {
            "status": "exploratory validation selection; not a production promotion",
            "featureProfiles": list(FEATURE_PROFILES),
            "l2": "maximum recording-held-out training average precision",
            "threshold": "maximum exact-gap validation F1, precision, recall, then higher threshold",
            "candidate": "maximum validation exact F1, precision, recall, row AP, then fewer inputs",
            "l2Grid": list(L2_GRID),
            "evaluationLabelsOpenedDuringSelection": False,
        },
        "counts": {
            "train": _counts(train_events),
            "validation": _counts(validation_events),
        },
        "dataPolicy": {
            "frozenRecordingSplit": {
                role: list(recording_ids)
                for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
            },
            "fitExclusions": ["beach-source-02"],
            "oneRecordingOneSet": True,
            "startPointTotal": 0,
            "cadencePoints": None,
            "candidateMargin": None,
            "minimumGapSeparation": 0,
            "maximumPredictions": None,
            "reanchorOnSelectedSwitch": False,
            "scoredUniverse": "every frozen reviewed inter-rally gap",
            "suppressionModelInput": False,
        },
        "productionModels": features.get("productionModels"),
        "suppressionQuarantine": features.get("suppressionQuarantine"),
        "sources": sources,
    }
    dataset = {
        "schemaVersion": 1,
        "kind": DATASET_KIND,
        "createdAt": created_at,
        "family": "v5",
        "selectedCandidate": selected_id,
        "modelFingerprint": selected["fingerprint"],
        "counts": payload["counts"],
        "eventIdentity": [
            {
                "eventId": event.event_id,
                "recordingId": event.recording_id,
                "role": event.role,
                "gapOrder": event.gap_order,
                "label": event.label,
            }
            for event in (*train_events, *validation_events)
        ],
        "sources": sources,
    }
    _write(model_dir / "model.json", payload)
    _write(model_dir / "dataset-development.json", dataset)
    return payload


def _delta(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "predictedEvents": int(current["predictedEvents"])
        - int(baseline["predictedEvents"]),
        "truePositives": int(current["truePositives"])
        - int(baseline["truePositives"]),
        "falsePositives": int(current["falsePositives"])
        - int(baseline["falsePositives"]),
        "falseNegatives": int(current["falseNegatives"])
        - int(baseline["falseNegatives"]),
        "precision": float(current["precision"] or 0.0)
        - float(baseline["precision"] or 0.0),
        "recall": float(current["recall"] or 0.0)
        - float(baseline["recall"] or 0.0),
        "f1": float(current["f1"] or 0.0) - float(baseline["f1"] or 0.0),
    }


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    features_path = args.features.expanduser().resolve()
    cadence_evaluation_path = args.cadence_evaluation.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output = args.evaluation_output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    if _sha256(cadence_evaluation_path) != EXPECTED_CADENCE_EVALUATION_SHA256:
        raise ValueError("cadence-evaluation identity changed")
    features = _load_features(features_path)
    model_path = model_dir / "model.json"
    model = _load(model_path)
    if model.get("kind") != MODEL_KIND or model.get("family") != "v5":
        raise ValueError("no-cadence model contract changed")
    if model["sources"]["features"]["sha256"] != _sha256(features_path):
        raise ValueError("no-cadence model feature binding changed")
    if (
        model["sources"]["cadenceModel"]["sha256"]
        != EXPECTED_CADENCE_MODEL_SHA256
    ):
        raise ValueError("no-cadence model cadence-control binding changed")
    cadence_evaluation = _load(cadence_evaluation_path)
    if (
        cadence_evaluation.get("kind")
        != "volleycut-side-switch-production-state-evaluation-v1"
        or cadence_evaluation.get("modelSha256")
        != EXPECTED_CADENCE_MODEL_SHA256
    ):
        raise ValueError("cadence evaluation contract changed")
    evaluation_events = _events(features, {"evaluation"})

    results: dict[str, Any] = {}
    scored: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for candidate_id, candidate in model["candidates"].items():
        names = tuple(candidate["featureNames"])
        deployable = {
            "classifier": candidate["classifier"],
            "decoder": candidate["decoder"],
        }
        if candidate.get("fingerprint") != fingerprint(deployable):
            raise ValueError(f"candidate fingerprint changed: {candidate_id}")
        classifier = V6Model.from_dict(candidate["classifier"], expected_names=names)
        IndependentGapDecoderSettings.from_dict(candidate["decoder"])
        probabilities = classifier.predict_proba(matrix_for(evaluation_events, names))
        predictions = independent_gap_predictions(probabilities, classifier.threshold)
        overall = _metric_sensitivity(evaluation_events, predictions)
        cadence_overall = cadence_evaluation["candidates"][candidate_id]["overall"]
        results[candidate_id] = {
            "rowAveragePrecision": average_precision(
                labels_for(evaluation_events), probabilities
            ),
            "overall": overall,
            "byRecording": _by_recording(evaluation_events, predictions),
            "predictionStructure": prediction_structure(
                evaluation_events, predictions
            ),
            "cadenceBaseline": cadence_overall,
            "deltaFromCadence": {
                tolerance: _delta(overall[tolerance], cadence_overall[tolerance])
                for tolerance in ("0", "1", "2")
            },
        }
        scored[candidate_id] = (probabilities, predictions)

    selected_id = str(model["selectedCandidate"])
    if model.get("fingerprint") != model["candidates"][selected_id]["fingerprint"]:
        raise ValueError("selected model fingerprint changed")
    selected_probabilities, selected_predictions = scored[selected_id]
    payload = {
        "schemaVersion": 1,
        "kind": EVALUATION_KIND,
        "createdAt": datetime.now(UTC).isoformat(),
        "family": "v5",
        "modelPath": str(model_path),
        "modelSha256": _sha256(model_path),
        "selectedCandidate": selected_id,
        "counts": _counts(evaluation_events),
        "candidates": results,
        "knownLimitations": {
            "retrospectiveConfirmation": (
                "the evaluation labels were opened by earlier V1-V6 research; this is not a pristine test"
            ),
            "candidateConditioned": (
                "metrics cover the frozen reviewed gap inventory, not every possible full-video gap"
            ),
            "independentDuplicates": (
                "the no-cadence decoder intentionally applies no spacing or local-maximum suppression"
            ),
            "selectionLock": (
                "original:state-gate won validation and remains selected even if another profile is stronger retrospectively"
            ),
        },
        "predictions": [
            {
                "eventId": event.event_id,
                "recordingId": event.recording_id,
                "gapOrder": event.gap_order,
                "decision": event.row.get("decision"),
                "label": event.label,
                "selectedScore": float(selected_probabilities[index]),
                "selectedPrediction": bool(selected_predictions[index]),
                "candidateScores": {
                    candidate_id: float(values[0][index])
                    for candidate_id, values in scored.items()
                },
                "candidatePredictions": {
                    candidate_id: bool(values[1][index])
                    for candidate_id, values in scored.items()
                },
            }
            for index, event in enumerate(evaluation_events)
        ],
        "sources": {
            **model["sources"],
            "cadenceEvaluation": {
                "path": str(cadence_evaluation_path),
                "sha256": _sha256(cadence_evaluation_path),
            },
        },
        "productionModels": model.get("productionModels"),
        "suppressionQuarantine": model.get("suppressionQuarantine"),
    }
    _write(output, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    evaluate_parser = commands.add_parser("evaluate")
    for child in (freeze_parser, evaluate_parser):
        child.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
        child.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    freeze_parser.add_argument(
        "--cadence-model", type=Path, default=DEFAULT_CADENCE_MODEL
    )
    evaluate_parser.add_argument(
        "--cadence-evaluation", type=Path, default=DEFAULT_CADENCE_EVALUATION
    )
    evaluate_parser.add_argument(
        "--evaluation-output", type=Path, default=DEFAULT_EVALUATION
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = freeze(args) if args.command == "freeze" else evaluate(args)
    if args.command == "freeze":
        selected = payload["candidates"][payload["selectedCandidate"]]
        summary = {
            "selectedCandidate": payload["selectedCandidate"],
            "fingerprint": payload["fingerprint"],
            "validation": selected["validation"],
        }
    else:
        selected = payload["candidates"][payload["selectedCandidate"]]
        summary = {
            "selectedCandidate": payload["selectedCandidate"],
            "retrospective": selected,
        }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
