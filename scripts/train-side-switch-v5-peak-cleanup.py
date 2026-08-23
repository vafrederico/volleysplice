#!/usr/bin/env python3
"""Freeze and evaluate cadence-free V5 peak/count/context cleanup variants."""

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
from analysis.side_switch_no_cadence import prediction_structure
from analysis.side_switch_peak_cleanup import (
    PeakCleanupSettings,
    combine_soft_context,
    decode_peak_cleanup,
    selected_time_structure,
)
from analysis.side_switch_production_state import PRODUCTION_STATE_FEATURE_NAMES
from analysis.side_switch_training_policy import validate_side_switch_fit_recordings
from analysis.side_switch_v3 import (
    FROZEN_RECORDING_SPLIT,
    V3Event,
    average_precision,
    event_metrics,
)
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
DEFAULT_NO_CADENCE_MODEL = ROOT / "models/side-switch-v5-no-cadence-v1/model.json"
DEFAULT_NO_CADENCE_EVALUATION = (
    ROOT / "reports/side-switch/side-switch-v5-no-cadence-v1-evaluation.json"
)
DEFAULT_CADENCE_EVALUATION = (
    ROOT / "reports/side-switch/side-switch-v5-production-state-v1-evaluation.json"
)
DEFAULT_MODEL_DIR = ROOT / "models/side-switch-v5-peak-cleanup-v1"
DEFAULT_EVALUATION = (
    ROOT / "reports/side-switch/side-switch-v5-peak-cleanup-v1-evaluation.json"
)

EXPECTED_FEATURE_SHA256 = (
    "c86b8ef7427dd8e1347d726c7d2326f18f22a6eb9fb16a0eec685c0fde0c9f36"
)
EXPECTED_NO_CADENCE_MODEL_SHA256 = (
    "5c89d60c865812f3c42ab1ffa4265d19d1ff9c3de82ea8ac7633850e7a5e9d17"
)
EXPECTED_NO_CADENCE_EVALUATION_SHA256 = (
    "c249f17b9b77471609c651656b7ca3cfc66d18168380a7ba5b450008e9bdcfde"
)
EXPECTED_CADENCE_EVALUATION_SHA256 = (
    "7b243b5140fdbf08fb476d0679af314196744137588fd2e30f2f92e3d9bf3391"
)
FEATURE_KIND = "volleycut-side-switch-production-state-augmented-features-v1"
NO_CADENCE_MODEL_KIND = "volleycut-side-switch-v5-no-cadence-specialist-v1"
MODEL_KIND = "volleycut-side-switch-v5-peak-cleanup-specialist-v1"
DATASET_KIND = "volleycut-side-switch-v5-peak-cleanup-development-dataset-v1"
EVALUATION_KIND = "volleycut-side-switch-v5-peak-cleanup-evaluation-v1"
L2_GRID = (0.01, 0.1, 1.0, 10.0)
FREE_PREDICTIONS_PER_RECORDING = 6
PEAK_TIME_GRID_SECONDS = (0.0, 30.0, 60.0)
COUNT_PENALTY_GRID = (0.25, 0.5, 1.0)
PRODUCTION_CONTEXT_WEIGHT_GRID = (0.25, 0.5, 1.0)
PRODUCTION_CONTEXT_FEATURE_NAMES = tuple(
    name
    for name in PRODUCTION_STATE_FEATURE_NAMES
    if name != "productionGapDurationSeconds"
)


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
    if any(name not in declared for name in PRODUCTION_CONTEXT_FEATURE_NAMES):
        raise ValueError("production-context feature signature changed")
    if (
        payload.get("suppressionQuarantine", {}).get("eligibleForModelInput")
        is not False
    ):
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


def _fit_production_context_head(
    train_events: Sequence[V3Event], validation_events: Sequence[V3Event]
) -> tuple[V6Model, dict[str, Any], np.ndarray]:
    leaderboard: list[dict[str, Any]] = []
    for l2 in L2_GRID:
        probabilities, cross_fit = grouped_cross_fit(
            train_events, l2, PRODUCTION_CONTEXT_FEATURE_NAMES
        )
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
    classifier = fit_model(
        train_events, selected_l2, PRODUCTION_CONTEXT_FEATURE_NAMES
    )
    validation_probabilities = classifier.predict_proba(
        matrix_for(validation_events, PRODUCTION_CONTEXT_FEATURE_NAMES)
    )
    report = {
        "featureNames": list(PRODUCTION_CONTEXT_FEATURE_NAMES),
        "excludesRawGapDuration": True,
        "l2Leaderboard": leaderboard,
        "selectedL2": selected_l2,
        "validationRowAveragePrecision": average_precision(
            labels_for(validation_events), validation_probabilities
        ),
    }
    return classifier, report, validation_probabilities


def _mechanism_name(peak: bool, count: bool, context: bool) -> str:
    parts = ["local-peak" if peak else "independent"]
    if count:
        parts.append("soft-count")
    if context:
        parts.append("production-context")
    if not peak and not count and not context:
        parts.append("control")
    return "-".join(parts)


def _mechanism_grid() -> dict[str, tuple[PeakCleanupSettings, ...]]:
    result: dict[str, tuple[PeakCleanupSettings, ...]] = {}
    for peak in (False, True):
        for count in (False, True):
            for context in (False, True):
                name = _mechanism_name(peak, count, context)
                peak_values = (
                    tuple((2, seconds) for seconds in PEAK_TIME_GRID_SECONDS)
                    if peak
                    else ((0, 0.0),)
                )
                penalty_values = COUNT_PENALTY_GRID if count else (0.0,)
                context_values = (
                    PRODUCTION_CONTEXT_WEIGHT_GRID if context else (0.0,)
                )
                result[name] = tuple(
                    PeakCleanupSettings(
                        minimum_gap_separation=gap_separation,
                        minimum_time_separation_seconds=time_separation,
                        free_predictions_per_recording=(
                            FREE_PREDICTIONS_PER_RECORDING
                        ),
                        count_penalty_logit=penalty,
                        production_context_weight=context_weight,
                    )
                    for gap_separation, time_separation in peak_values
                    for penalty in penalty_values
                    for context_weight in context_values
                )
    return result


def _thresholds(probabilities: np.ndarray) -> tuple[float, ...]:
    values = sorted({float(value) for value in probabilities}, reverse=True)
    if not values:
        raise ValueError("decoder selection needs scored validation gaps")
    above_maximum = min(1.0, math.nextafter(values[0], math.inf))
    return tuple(dict.fromkeys((above_maximum, *values)))


def _complexity(settings: PeakCleanupSettings) -> int:
    return sum(
        (
            settings.minimum_gap_separation > 0,
            settings.minimum_time_separation_seconds > 0.0,
            settings.count_penalty_logit > 0.0,
            settings.production_context_weight > 0.0,
        )
    )


def _selection_rank(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = candidate["metrics"]
    settings = PeakCleanupSettings.from_dict(candidate["decoder"])
    return (
        float(metrics["f1"] or 0.0),
        float(metrics["precision"] or 0.0),
        float(metrics["recall"] or 0.0),
        -float(_complexity(settings)),
        -float(metrics["predictedEvents"]),
        float(candidate["threshold"]),
        -settings.production_context_weight,
        -settings.count_penalty_logit,
        -settings.minimum_time_separation_seconds,
    )


def _select_settings_threshold(
    events: Sequence[V3Event],
    primary_probabilities: np.ndarray,
    context_probabilities: np.ndarray,
    settings: PeakCleanupSettings,
) -> dict[str, Any]:
    combined = combine_soft_context(
        primary_probabilities,
        context_probabilities,
        settings.production_context_weight,
    )
    best: dict[str, Any] | None = None
    for threshold in _thresholds(combined):
        predictions, _ = decode_peak_cleanup(
            events,
            primary_probabilities,
            context_probabilities,
            threshold,
            settings,
        )
        candidate = {
            "threshold": threshold,
            "decoder": settings.to_dict(),
            "metrics": event_metrics(events, predictions, tolerance=0),
        }
        if best is None or _selection_rank(candidate) > _selection_rank(best):
            best = candidate
    if best is None:
        raise ValueError("decoder selection produced no candidate")
    return best


def _select_mechanism(
    events: Sequence[V3Event],
    primary_probabilities: np.ndarray,
    context_probabilities: np.ndarray,
    settings_grid: Sequence[PeakCleanupSettings],
) -> dict[str, Any]:
    leaderboard = [
        _select_settings_threshold(
            events, primary_probabilities, context_probabilities, settings
        )
        for settings in settings_grid
    ]
    leaderboard.sort(key=_selection_rank, reverse=True)
    selected = leaderboard[0]
    settings = PeakCleanupSettings.from_dict(selected["decoder"])
    predictions, combined = decode_peak_cleanup(
        events,
        primary_probabilities,
        context_probabilities,
        float(selected["threshold"]),
        settings,
    )
    return {
        **selected,
        "validation": {
            "rowAveragePrecision": average_precision(
                labels_for(events), combined
            ),
            "overall": _metric_sensitivity(events, predictions),
            "predictionStructure": prediction_structure(events, predictions),
            "selectedTimeStructure": selected_time_structure(events, predictions),
        },
        "settingsLeaderboard": leaderboard,
    }


def _deployable_fingerprint(
    primary_classifier: Mapping[str, Any],
    context_classifier: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> str:
    return fingerprint(
        {
            "primaryClassifier": primary_classifier,
            "productionContextClassifier": context_classifier,
            "threshold": candidate["threshold"],
            "decoder": candidate["decoder"],
        }
    )


def freeze(args: argparse.Namespace) -> Mapping[str, Any]:
    features_path = args.features.expanduser().resolve()
    no_cadence_model_path = args.no_cadence_model.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    if model_dir.exists():
        raise FileExistsError(f"refusing to reuse model directory: {model_dir}")
    if _sha256(no_cadence_model_path) != EXPECTED_NO_CADENCE_MODEL_SHA256:
        raise ValueError("no-cadence model identity changed")
    features = _load_features(features_path)
    no_cadence_model = _load(no_cadence_model_path)
    if (
        no_cadence_model.get("kind") != NO_CADENCE_MODEL_KIND
        or no_cadence_model.get("selectedCandidate") != "original:state-gate"
    ):
        raise ValueError("no-cadence model contract changed")

    train_events = _events(features, {"train"})
    validation_events = _events(features, {"validation"})
    validate_side_switch_fit_recordings(
        sorted({event.recording_id for event in train_events})
    )

    primary_payload = no_cadence_model["classifier"]
    primary_names = tuple(primary_payload["featureNames"])
    primary_classifier = V6Model.from_dict(
        primary_payload, expected_names=primary_names
    )
    primary_validation_probabilities = primary_classifier.predict_proba(
        matrix_for(validation_events, primary_names)
    )
    context_classifier, context_report, context_validation_probabilities = (
        _fit_production_context_head(train_events, validation_events)
    )

    mechanisms: dict[str, Any] = {}
    for mechanism_name, settings_grid in _mechanism_grid().items():
        selected = _select_mechanism(
            validation_events,
            primary_validation_probabilities,
            context_validation_probabilities,
            settings_grid,
        )
        selected["fingerprint"] = _deployable_fingerprint(
            primary_classifier.to_dict(), context_classifier.to_dict(), selected
        )
        mechanisms[mechanism_name] = selected
        exact = selected["validation"]["overall"]["0"]
        print(
            f"Selected {mechanism_name}: F1={exact['f1']:.4f}, "
            f"P={exact['precision']:.4f}, R={exact['recall']:.4f}, "
            f"predictions={exact['predictedEvents']}",
            flush=True,
        )

    selected_mechanism = max(
        mechanisms, key=lambda name: _selection_rank(mechanisms[name])
    )
    selected = mechanisms[selected_mechanism]
    created_at = datetime.now(UTC).isoformat()
    sources = {
        "features": {
            "path": str(features_path),
            "sha256": _sha256(features_path),
            "createdAt": features.get("createdAt"),
        },
        "noCadenceModel": {
            "path": str(no_cadence_model_path),
            "sha256": _sha256(no_cadence_model_path),
            "selectedCandidate": no_cadence_model.get("selectedCandidate"),
            "fingerprint": no_cadence_model.get("fingerprint"),
        },
        "productionState": features.get("sources", {}).get("productionState"),
    }
    payload = {
        "schemaVersion": 1,
        "kind": MODEL_KIND,
        "createdAt": created_at,
        "family": "v5",
        "selectedMechanism": selected_mechanism,
        "fingerprint": selected["fingerprint"],
        "primaryClassifier": primary_classifier.to_dict(),
        "productionContextClassifier": context_classifier.to_dict(),
        "productionContextFit": context_report,
        "threshold": selected["threshold"],
        "decoder": selected["decoder"],
        "mechanisms": mechanisms,
        "selection": {
            "status": "exploratory validation selection; not a production promotion",
            "primaryHead": "frozen validation-selected no-cadence V5-state head",
            "contextHead": (
                "19 production rally/dead/serve context inputs; raw gap duration "
                "excluded"
            ),
            "contextL2": "maximum recording-held-out training average precision",
            "mechanismAndThreshold": (
                "maximum validation exact F1, precision, recall, then lower complexity"
            ),
            "mechanismFamilies": list(mechanisms),
            "peakTimeGridSeconds": list(PEAK_TIME_GRID_SECONDS),
            "softCountFreePredictions": FREE_PREDICTIONS_PER_RECORDING,
            "countPenaltyLogitGrid": list(COUNT_PENALTY_GRID),
            "productionContextWeightGrid": list(
                PRODUCTION_CONTEXT_WEIGHT_GRID
            ),
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
            "reanchorOnSelectedSwitch": False,
            "hardMaximumPredictions": None,
            "scoredUniverse": "every frozen reviewed inter-rally gap",
            "suppressionModelInput": False,
            "productionContextHardGate": False,
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
        "selectedMechanism": selected_mechanism,
        "modelFingerprint": selected["fingerprint"],
        "counts": payload["counts"],
        "eventIdentity": [
            {
                "eventId": event.event_id,
                "recordingId": event.recording_id,
                "role": event.role,
                "gapOrder": event.gap_order,
                "transitionTime": float(event.row["transitionTime"]),
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
    no_cadence_evaluation_path = args.no_cadence_evaluation.expanduser().resolve()
    cadence_evaluation_path = args.cadence_evaluation.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output = args.evaluation_output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    if (
        _sha256(no_cadence_evaluation_path)
        != EXPECTED_NO_CADENCE_EVALUATION_SHA256
    ):
        raise ValueError("no-cadence evaluation identity changed")
    if _sha256(cadence_evaluation_path) != EXPECTED_CADENCE_EVALUATION_SHA256:
        raise ValueError("cadence evaluation identity changed")

    features = _load_features(features_path)
    model_path = model_dir / "model.json"
    model = _load(model_path)
    if model.get("kind") != MODEL_KIND or model.get("family") != "v5":
        raise ValueError("peak-cleanup model contract changed")
    if (
        model.get("sources", {}).get("features", {}).get("sha256")
        != _sha256(features_path)
        or model.get("sources", {}).get("noCadenceModel", {}).get("sha256")
        != EXPECTED_NO_CADENCE_MODEL_SHA256
    ):
        raise ValueError("peak-cleanup model source binding changed")

    no_cadence_evaluation = _load(no_cadence_evaluation_path)
    cadence_evaluation = _load(cadence_evaluation_path)
    no_cadence_selected = str(no_cadence_evaluation["selectedCandidate"])
    cadence_selected = str(cadence_evaluation["selectedCandidate"])
    no_cadence_baseline = no_cadence_evaluation["candidates"][
        no_cadence_selected
    ]["overall"]
    cadence_baseline = cadence_evaluation["candidates"][cadence_selected][
        "overall"
    ]

    evaluation_events = _events(features, {"evaluation"})
    primary_payload = model["primaryClassifier"]
    primary_names = tuple(primary_payload["featureNames"])
    primary_classifier = V6Model.from_dict(
        primary_payload, expected_names=primary_names
    )
    context_payload = model["productionContextClassifier"]
    context_classifier = V6Model.from_dict(
        context_payload, expected_names=PRODUCTION_CONTEXT_FEATURE_NAMES
    )
    primary_probabilities = primary_classifier.predict_proba(
        matrix_for(evaluation_events, primary_names)
    )
    context_probabilities = context_classifier.predict_proba(
        matrix_for(evaluation_events, PRODUCTION_CONTEXT_FEATURE_NAMES)
    )

    results: dict[str, Any] = {}
    scored: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mechanism_name, candidate in model["mechanisms"].items():
        settings = PeakCleanupSettings.from_dict(candidate["decoder"])
        expected_fingerprint = _deployable_fingerprint(
            primary_payload, context_payload, candidate
        )
        if candidate.get("fingerprint") != expected_fingerprint:
            raise ValueError(f"mechanism fingerprint changed: {mechanism_name}")
        predictions, combined = decode_peak_cleanup(
            evaluation_events,
            primary_probabilities,
            context_probabilities,
            float(candidate["threshold"]),
            settings,
        )
        overall = _metric_sensitivity(evaluation_events, predictions)
        results[mechanism_name] = {
            "fingerprint": candidate["fingerprint"],
            "threshold": candidate["threshold"],
            "decoder": candidate["decoder"],
            "rowAveragePrecision": average_precision(
                labels_for(evaluation_events), combined
            ),
            "overall": overall,
            "byRecording": _by_recording(evaluation_events, predictions),
            "predictionStructure": prediction_structure(
                evaluation_events, predictions
            ),
            "selectedTimeStructure": selected_time_structure(
                evaluation_events, predictions
            ),
            "noCadenceBaseline": no_cadence_baseline,
            "cadenceBaseline": cadence_baseline,
            "deltaFromNoCadence": {
                tolerance: _delta(overall[tolerance], no_cadence_baseline[tolerance])
                for tolerance in ("0", "1", "2")
            },
            "deltaFromCadence": {
                tolerance: _delta(overall[tolerance], cadence_baseline[tolerance])
                for tolerance in ("0", "1", "2")
            },
        }
        scored[mechanism_name] = (combined, predictions)

    selected_mechanism = str(model["selectedMechanism"])
    if model.get("fingerprint") != model["mechanisms"][selected_mechanism].get(
        "fingerprint"
    ):
        raise ValueError("selected model fingerprint changed")
    selected_scores, selected_predictions = scored[selected_mechanism]
    payload = {
        "schemaVersion": 1,
        "kind": EVALUATION_KIND,
        "createdAt": datetime.now(UTC).isoformat(),
        "family": "v5",
        "modelPath": str(model_path),
        "modelSha256": _sha256(model_path),
        "selectedMechanism": selected_mechanism,
        "counts": _counts(evaluation_events),
        "primaryRowAveragePrecision": average_precision(
            labels_for(evaluation_events), primary_probabilities
        ),
        "productionContextOnlyRowAveragePrecision": average_precision(
            labels_for(evaluation_events), context_probabilities
        ),
        "mechanisms": results,
        "predictions": [
            {
                "eventId": event.event_id,
                "recordingId": event.recording_id,
                "gapOrder": event.gap_order,
                "transitionTime": float(event.row["transitionTime"]),
                "decision": event.row.get("decision"),
                "label": event.label,
                "primaryScore": float(primary_probabilities[index]),
                "productionContextScore": float(context_probabilities[index]),
                "selectedScore": float(selected_scores[index]),
                "selectedPrediction": bool(selected_predictions[index]),
                "mechanismScores": {
                    mechanism_name: float(values[0][index])
                    for mechanism_name, values in scored.items()
                },
                "mechanismPredictions": {
                    mechanism_name: bool(values[1][index])
                    for mechanism_name, values in scored.items()
                },
            }
            for index, event in enumerate(evaluation_events)
        ],
        "knownLimitations": {
            "retrospectiveConfirmation": (
                "the raw-phone labels were opened by earlier side-switch research; "
                "this is not a pristine test"
            ),
            "historicalValidationBias": (
                "historical validation positives are strongly cadence-aligned, so "
                "decoder selection may favor count/location regularity that does "
                "not transfer"
            ),
            "candidateConditioned": (
                "metrics cover the frozen reviewed gap inventory, not exhaustive "
                "full-video truth"
            ),
            "productionContext": (
                "the auxiliary head is side-switch-trained soft evidence over frozen "
                "production outputs; it is not an independently trained switch "
                "detector"
            ),
            "selectionLock": (
                "the validation-selected mechanism stays selected even if another "
                "locked mechanism is stronger retrospectively"
            ),
        },
        "sources": {
            **model["sources"],
            "noCadenceEvaluation": {
                "path": str(no_cadence_evaluation_path),
                "sha256": _sha256(no_cadence_evaluation_path),
            },
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
        "--no-cadence-model", type=Path, default=DEFAULT_NO_CADENCE_MODEL
    )
    evaluate_parser.add_argument(
        "--no-cadence-evaluation",
        type=Path,
        default=DEFAULT_NO_CADENCE_EVALUATION,
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
        selected = payload["mechanisms"][payload["selectedMechanism"]]
        summary = {
            "selectedMechanism": payload["selectedMechanism"],
            "fingerprint": payload["fingerprint"],
            "decoder": selected["decoder"],
            "threshold": selected["threshold"],
            "validation": selected["validation"],
            "productionContextFit": payload["productionContextFit"],
        }
    else:
        selected = payload["mechanisms"][payload["selectedMechanism"]]
        summary = {
            "selectedMechanism": payload["selectedMechanism"],
            "retrospective": selected,
            "productionContextOnlyRowAveragePrecision": payload[
                "productionContextOnlyRowAveragePrecision"
            ],
        }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
