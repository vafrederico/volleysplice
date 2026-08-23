#!/usr/bin/env python3
"""Train and evaluate V5/V6 production-state and serve-grounded variants."""

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
from analysis.side_switch_production_state import (
    PRODUCTION_STATE_FEATURE_NAMES,
    SERVE_ANCHOR_FEATURE_NAMES,
    STATE_GATE_FEATURE_NAMES,
)
from analysis.side_switch_training_policy import validate_side_switch_fit_recordings
from analysis.side_switch_v3 import (
    FROZEN_RECORDING_SPLIT,
    DecoderSettings,
    V3Event,
    average_precision,
    decode_all as decode_cadence_all,
    event_metrics,
)
from analysis.side_switch_v5 import (
    OrientationDecoderSettings,
    VISUAL_FEATURE_NAMES as V5_FEATURE_NAMES,
    decode_all,
)
from analysis.side_switch_v6 import (
    VISUAL_FEATURE_NAMES as V6_FEATURE_NAMES,
    V6Model,
    fingerprint,
    fit_model,
    grouped_cross_fit,
    labels_for,
    matrix_for,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULTS = {
    "v5": {
        "original": ROOT
        / "reports/side-switch/side-switch-v5-production-state-v1-features.json",
        "grounded": ROOT
        / "reports/side-switch/side-switch-v5-serve-grounded-production-state-v1-features.json",
        "model": ROOT / "models/side-switch-v5-production-state-v1",
        "evaluation": ROOT
        / "reports/side-switch/side-switch-v5-production-state-v1-evaluation.json",
        "baseline": ROOT
        / "reports/side-switch/side-switch-specialist-v5-player-orientation-evaluation.json",
        "attribution": ROOT
        / "reports/side-switch/side-switch-v5-production-state-v1-attribution.json",
    },
    "v6": {
        "original": ROOT
        / "reports/side-switch/side-switch-v6-production-state-v1-features.json",
        "grounded": ROOT
        / "reports/side-switch/side-switch-v6-serve-grounded-production-state-v1-features.json",
        "model": ROOT / "models/side-switch-v6-production-state-v1",
        "evaluation": ROOT
        / "reports/side-switch/side-switch-v6-production-state-v1-evaluation.json",
        "baseline": ROOT
        / "reports/side-switch/side-switch-specialist-v6-detected-adaptive-evaluation.json",
        "attribution": ROOT
        / "reports/side-switch/side-switch-v6-production-state-v1-attribution.json",
    },
}
MODEL_KIND = "volleycut-side-switch-production-state-specialist-v1"
MODEL_SCHEMA_VERSION = 1
FEATURE_KIND = "volleycut-side-switch-production-state-augmented-features-v1"
L2_GRID = (0.01, 0.1, 1.0, 10.0)
MARGIN_GRID = (1, 2, 3, 4)
FROZEN_COMPARISON_DECODER = OrientationDecoderSettings(
    candidate_margin=1,
    distance_penalty=0.25,
    orientation_weight=0.0,
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


def _base_names(family: str) -> tuple[str, ...]:
    return V5_FEATURE_NAMES if family == "v5" else V6_FEATURE_NAMES


def _feature_profiles(family: str) -> dict[str, tuple[str, ...]]:
    base = _base_names(family)
    return {
        "base": base,
        "state-gate": (*base, *STATE_GATE_FEATURE_NAMES),
        "serve-anchor": (*base, *SERVE_ANCHOR_FEATURE_NAMES),
        "combined": (*base, *PRODUCTION_STATE_FEATURE_NAMES),
    }


def _load_features(
    path: Path, family: str, appearance_mode: str
) -> Mapping[str, Any]:
    payload = _load(path)
    if payload.get("kind") != FEATURE_KIND:
        raise ValueError(f"feature artifact has the wrong kind: {path}")
    if payload.get("family") != family or payload.get("appearanceMode") != appearance_mode:
        raise ValueError(f"feature artifact identity does not match {family}/{appearance_mode}")
    frozen = {
        role: list(recording_ids)
        for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
    }
    if payload.get("frozenRecordingSplit") != frozen:
        raise ValueError("feature artifact does not match the frozen split")
    declared = tuple(payload.get("profile", {}).get("featureNames", []))
    expected = (*_base_names(family), *PRODUCTION_STATE_FEATURE_NAMES)
    if declared != expected:
        raise ValueError("augmented feature signature drifted")
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
        raise ValueError("feature events are missing or non-binary")
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


def _decoder_candidates(
    events: Sequence[V3Event], probabilities: np.ndarray
) -> list[dict[str, Any]]:
    thresholds = sorted({float(value) for value in probabilities}, reverse=True)
    thresholds.insert(0, math.nextafter(thresholds[0], math.inf))
    candidates: list[dict[str, Any]] = []
    # V5 and V6 independently selected this same decoder geometry.  Holding it
    # fixed isolates the production-state/appearance change; only the classifier
    # threshold is reselected on validation for each feature view.
    settings = FROZEN_COMPARISON_DECODER
    for threshold in thresholds:
        predictions = decode_all(events, probabilities, threshold, settings)
        candidates.append(
            {
                "threshold": threshold,
                "settings": settings.to_dict(),
                "metrics": event_metrics(events, predictions, tolerance=0),
            }
        )
    return candidates


def _decoder_rank(
    candidate: Mapping[str, Any]
) -> tuple[float, float, float, int, float, float, float]:
    metrics = candidate["metrics"]
    settings = candidate["settings"]
    return (
        float(metrics["f1"] or 0.0),
        float(metrics["precision"] or 0.0),
        float(metrics["recall"] or 0.0),
        -int(settings["candidateMargin"]),
        -float(settings["distancePenalty"]),
        -float(settings["orientationWeight"]),
        float(candidate["threshold"]),
    )


def _fit_candidate(
    train_events: Sequence[V3Event],
    validation_events: Sequence[V3Event],
    feature_names: Sequence[str],
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
    validation_probabilities = classifier.predict_proba(
        matrix_for(validation_events, names)
    )
    selected_decoder = max(
        _decoder_candidates(validation_events, validation_probabilities),
        key=_decoder_rank,
    )
    classifier = replace(
        classifier, threshold=float(selected_decoder["threshold"])
    )
    decoder = OrientationDecoderSettings.from_dict(
        selected_decoder["settings"]
    )
    predictions = decode_all(
        validation_events,
        validation_probabilities,
        classifier.threshold,
        decoder,
    )
    return {
        "classifier": classifier,
        "decoder": decoder,
        "probabilities": validation_probabilities,
        "predictions": predictions,
        "l2Leaderboard": leaderboard,
        "selectedDecoder": selected_decoder,
    }


def _candidate_rank(
    candidate_id: str, candidate: Mapping[str, Any]
) -> tuple[float, float, float, float, int, int, str]:
    exact = candidate["validation"]["metrics"]["0"]
    appearance, profile = candidate_id.split(":", 1)
    return (
        float(exact["f1"] or 0.0),
        float(exact["precision"] or 0.0),
        float(exact["recall"] or 0.0),
        float(candidate["validation"]["rowAveragePrecision"] or 0.0),
        -len(candidate["featureNames"]),
        1 if appearance == "original" and profile == "base" else 0,
        candidate_id,
    )


def _fixed_gates(
    events: Sequence[V3Event], predictions: np.ndarray
) -> dict[str, Any]:
    minimum_support = np.asarray(
        [
            float(event.row["features"]["productionMinimumAdjacentSupportCount"])
            for event in events
        ]
    )
    minimum_serve = np.asarray(
        [
            float(
                event.row["features"][
                    "productionMinimumAdjacentServeSupportCount"
                ]
            )
            for event in events
        ]
    )
    masks = {
        "both-models-on-both-adjacent-rallies": minimum_support >= 2.0,
        "serve-head-on-both-adjacent-rallies": minimum_serve >= 1.0,
        "both-models-and-serve-heads": (minimum_support >= 2.0)
        & (minimum_serve >= 1.0),
    }
    return {
        name: _metric_sensitivity(events, predictions & mask)
        for name, mask in masks.items()
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    family = args.family
    original_path = args.original_features.expanduser().resolve()
    grounded_path = args.grounded_features.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    if model_dir.exists():
        raise FileExistsError(f"refusing to reuse model directory: {model_dir}")
    original = _load_features(original_path, family, "original")
    grounded = _load_features(grounded_path, family, "serve-grounded")
    payloads = {"original": original, "serve-grounded": grounded}
    identities = {
        appearance: [
            (
                str(row["eventId"]),
                str(row["recordingId"]),
                str(row["role"]),
                int(row["gapOrder"]),
                int(row["label"]),
            )
            for row in payload["rows"]
        ]
        for appearance, payload in payloads.items()
    }
    if identities["original"] != identities["serve-grounded"]:
        raise ValueError("original and serve-grounded event identity diverged")

    profiles = _feature_profiles(family)
    candidate_payloads: dict[str, Any] = {}
    for appearance, feature_payload in payloads.items():
        train_events = _events(feature_payload, {"train"})
        validation_events = _events(feature_payload, {"validation"})
        validate_side_switch_fit_recordings(
            sorted({event.recording_id for event in train_events})
        )
        for profile_name, feature_names in profiles.items():
            candidate_id = f"{appearance}:{profile_name}"
            fitted = _fit_candidate(
                train_events, validation_events, feature_names
            )
            classifier: V6Model = fitted["classifier"]
            decoder: OrientationDecoderSettings = fitted["decoder"]
            predictions = fitted["predictions"]
            deployable = {
                "classifier": classifier.to_dict(),
                "decoder": decoder.to_dict(),
            }
            candidate_payloads[candidate_id] = {
                "appearanceMode": appearance,
                "featureProfile": profile_name,
                "featureNames": list(feature_names),
                "fingerprint": fingerprint(deployable),
                **deployable,
                "l2Leaderboard": fitted["l2Leaderboard"],
                "selectedDecoder": fitted["selectedDecoder"],
                "validation": {
                    "rowAveragePrecision": average_precision(
                        labels_for(validation_events), fitted["probabilities"]
                    ),
                    "metrics": _metric_sensitivity(
                        validation_events, predictions
                    ),
                    "fixedGateDiagnostics": _fixed_gates(
                        validation_events, predictions
                    ),
                },
            }
            print(
                f"Fitted {family} {candidate_id}: "
                f"F1={candidate_payloads[candidate_id]['validation']['metrics']['0']['f1']:.4f}",
                flush=True,
            )

    selected_id = max(
        candidate_payloads,
        key=lambda candidate_id: _candidate_rank(
            candidate_id, candidate_payloads[candidate_id]
        ),
    )
    selected = candidate_payloads[selected_id]
    created_at = datetime.now(UTC).isoformat()
    train_counts = _counts(_events(original, {"train"}))
    validation_counts = _counts(_events(original, {"validation"}))
    sources = {
        "originalFeatures": {
            "path": str(original_path),
            "sha256": _sha256(original_path),
            "createdAt": original.get("createdAt"),
        },
        "serveGroundedFeatures": {
            "path": str(grounded_path),
            "sha256": _sha256(grounded_path),
            "createdAt": grounded.get("createdAt"),
        },
        "productionState": original.get("sources", {}).get(
            "productionState"
        ),
    }
    model_payload = {
        "schemaVersion": MODEL_SCHEMA_VERSION,
        "kind": MODEL_KIND,
        "createdAt": created_at,
        "family": family,
        "selectedCandidate": selected_id,
        "fingerprint": selected["fingerprint"],
        "classifier": selected["classifier"],
        "decoder": selected["decoder"],
        "candidates": candidate_payloads,
        "selection": {
            "status": "exploratory validation selection; not a production promotion",
            "candidateProfiles": list(profiles),
            "appearanceModes": list(payloads),
            "l2": "maximum recording-held-out training average precision",
            "decoder": (
                "hold the common frozen V5/V6 decoder at candidate margin 1, "
                "distance penalty 0.25, and orientation weight 0; select only the "
                "classifier threshold by exact-gap validation event F1, precision, "
                "recall, then threshold tie-break"
            ),
            "candidate": (
                "maximum exact-gap validation event F1, precision, recall, row AP, "
                "then fewer inputs; the frozen original base wins an exact tie"
            ),
            "l2Grid": list(L2_GRID),
            "heldDecoder": FROZEN_COMPARISON_DECODER.to_dict(),
            "reportedCandidateMarginSensitivity": list(MARGIN_GRID),
            "evaluationLabelsOpenedDuringSelection": False,
        },
        "counts": {"train": train_counts, "validation": validation_counts},
        "dataPolicy": {
            "frozenRecordingSplit": {
                role: list(recording_ids)
                for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
            },
            "fitExclusions": ["beach-source-02"],
            "oneRecordingOneSet": True,
            "startPointTotal": 0,
            "cadencePoints": 7,
            "maximumOpportunities": 6,
            "reanchorOnSelectedSwitch": True,
            "productionEvidence": (
                "fixed shipped rally/serve/dead-state heads; no side-switch labels "
                "enter production-state feature extraction"
            ),
            "suppressionModelInput": False,
            "suppressionReason": original["suppressionQuarantine"]["reason"],
        },
        "productionModels": original.get("productionModels"),
        "suppressionQuarantine": original.get("suppressionQuarantine"),
        "sources": sources,
    }
    dataset_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-production-state-development-dataset-v1",
        "createdAt": created_at,
        "family": family,
        "selectedCandidate": selected_id,
        "modelFingerprint": selected["fingerprint"],
        "counts": model_payload["counts"],
        "eventIdentity": [
            {
                "eventId": row[0],
                "recordingId": row[1],
                "role": row[2],
                "gapOrder": row[3],
                "label": row[4],
            }
            for row in identities["original"]
            if row[2] in {"train", "validation"}
        ],
        "sources": sources,
    }
    _write(model_dir / "model.json", model_payload)
    _write(model_dir / "dataset-development.json", dataset_payload)
    return model_payload


def _candidate_evaluation(
    candidate: Mapping[str, Any], events: Sequence[V3Event]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    names = tuple(str(value) for value in candidate["featureNames"])
    deployable = {
        "classifier": candidate["classifier"],
        "decoder": candidate["decoder"],
    }
    if candidate.get("fingerprint") != fingerprint(deployable):
        raise ValueError("candidate fingerprint is invalid")
    classifier = V6Model.from_dict(
        candidate["classifier"], expected_names=names
    )
    decoder = OrientationDecoderSettings.from_dict(candidate["decoder"])
    probabilities = classifier.predict_proba(matrix_for(events, names))
    predictions = decode_all(
        events, probabilities, classifier.threshold, decoder
    )
    return (
        {
            "rowAveragePrecision": average_precision(
                labels_for(events), probabilities
            ),
            "overall": _metric_sensitivity(events, predictions),
            "byRecording": _by_recording(events, predictions),
            "fixedGateDiagnostics": _fixed_gates(events, predictions),
        },
        probabilities,
        predictions,
    )


def _suppression_diagnostic(
    events: Sequence[V3Event], overlap_ids: set[str]
) -> dict[str, Any]:
    def summarize(indexes: Sequence[int]) -> dict[str, Any]:
        labels = np.asarray([events[index].label for index in indexes])
        mean_scores = np.asarray(
            [
                float(
                    events[index].row["productionStateContext"][
                        "suppressionDiagnostic"
                    ]["suppressionGapMeanScore"]
                )
                for index in indexes
            ]
        )
        peak_scores = np.asarray(
            [
                float(
                    events[index].row["productionStateContext"][
                        "suppressionDiagnostic"
                    ]["suppressionGapPeakScore"]
                )
                for index in indexes
            ]
        )
        positive = labels == 1
        negative = labels == 0
        return {
            "rows": len(indexes),
            "switchRows": int(np.sum(positive)),
            "meanScoreAveragePrecision": average_precision(labels, mean_scores),
            "peakScoreAveragePrecision": average_precision(labels, peak_scores),
            "meanScoreByLabel": {
                "switch": float(np.mean(mean_scores[positive])) if np.any(positive) else None,
                "unmarked": float(np.mean(mean_scores[negative])) if np.any(negative) else None,
            },
            "peakScoreByLabel": {
                "switch": float(np.mean(peak_scores[positive])) if np.any(positive) else None,
                "unmarked": float(np.mean(peak_scores[negative])) if np.any(negative) else None,
            },
        }

    all_indexes = list(range(len(events)))
    clean = [
        index
        for index, event in enumerate(events)
        if event.recording_id not in overlap_ids
    ]
    contaminated = [
        index
        for index, event in enumerate(events)
        if event.recording_id in overlap_ids
    ]
    return {
        "eligibleForModelInputOrSelection": False,
        "allRetrospective": summarize(all_indexes),
        "recordingDisjointFromSuppressionFit": summarize(clean),
        "recordingOverlapWithSuppressionFit": summarize(contaminated),
    }


def _baseline_payload(path: Path, family: str) -> dict[str, Any]:
    payload = _load(path)
    selected_key = (
        "selectedPersistentOrientation"
        if family == "v5"
        else "selectedDetectedAdaptive"
    )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "visualRowAveragePrecision": payload["visualRowAveragePrecision"],
        "overall": payload[selected_key]["overall"],
    }


def attribute(args: argparse.Namespace) -> dict[str, Any]:
    """Post-hoc attribution only; never changes the frozen selected candidate."""

    family = args.family
    original_path = args.original_features.expanduser().resolve()
    output = args.attribution_output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite attribution: {output}")
    payload = _load_features(original_path, family, "original")
    train_events = _events(payload, {"train"})
    validation_events = _events(payload, {"validation"})
    evaluation_events = _events(payload, {"evaluation"})
    base = _base_names(family)
    head_gate_names = tuple(
        name
        for name in STATE_GATE_FEATURE_NAMES
        if name != "productionGapDurationSeconds"
    )
    profiles = {
        "base": base,
        "gap-duration-only": (*base, "productionGapDurationSeconds"),
        "production-head-state-only": (*base, *head_gate_names),
        "state-gate-combined": (*base, *STATE_GATE_FEATURE_NAMES),
    }
    results: dict[str, Any] = {}
    for profile_name, names in profiles.items():
        fitted = _fit_candidate(train_events, validation_events, names)
        classifier: V6Model = fitted["classifier"]
        decoder: OrientationDecoderSettings = fitted["decoder"]
        validation_predictions = fitted["predictions"]
        evaluation_probabilities = classifier.predict_proba(
            matrix_for(evaluation_events, names)
        )
        evaluation_predictions = decode_all(
            evaluation_events,
            evaluation_probabilities,
            classifier.threshold,
            decoder,
        )
        deployable = {
            "classifier": classifier.to_dict(),
            "decoder": decoder.to_dict(),
        }
        results[profile_name] = {
            "featureNames": list(names),
            "fingerprint": fingerprint(deployable),
            **deployable,
            "l2Leaderboard": fitted["l2Leaderboard"],
            "validation": {
                "rowAveragePrecision": average_precision(
                    labels_for(validation_events), fitted["probabilities"]
                ),
                "overall": _metric_sensitivity(
                    validation_events, validation_predictions
                ),
            },
            "retrospective": {
                "rowAveragePrecision": average_precision(
                    labels_for(evaluation_events), evaluation_probabilities
                ),
                "overall": _metric_sensitivity(
                    evaluation_events, evaluation_predictions
                ),
            },
        }
    attribution = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-production-state-attribution-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "family": family,
        "status": (
            "post-hoc attribution after the frozen retrospective evaluation was "
            "opened; these rows cannot replace the validation-selected model"
        ),
        "question": (
            "Does the V5/V6 state-gate result come from production head outputs, "
            "the raw candidate-gap duration, or their combination?"
        ),
        "heldDecoder": FROZEN_COMPARISON_DECODER.to_dict(),
        "profiles": results,
        "sources": {
            "originalFeatures": {
                "path": str(original_path),
                "sha256": _sha256(original_path),
            }
        },
    }
    _write(output, attribution)
    return attribution


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    family = args.family
    original_path = args.original_features.expanduser().resolve()
    grounded_path = args.grounded_features.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output = args.evaluation_output.expanduser().resolve()
    baseline_path = args.baseline_evaluation.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    original = _load_features(original_path, family, "original")
    grounded = _load_features(grounded_path, family, "serve-grounded")
    payloads = {"original": original, "serve-grounded": grounded}
    model_payload = _load(model_dir / "model.json")
    if model_payload.get("kind") != MODEL_KIND or model_payload.get("family") != family:
        raise ValueError("model artifact identity is invalid")
    expected_sources = {
        "originalFeatures": _sha256(original_path),
        "serveGroundedFeatures": _sha256(grounded_path),
    }
    for source_name, digest in expected_sources.items():
        if model_payload["sources"][source_name]["sha256"] != digest:
            raise ValueError(f"model source mismatch: {source_name}")

    evaluations: dict[str, Any] = {}
    score_rows: dict[str, tuple[np.ndarray, np.ndarray, Sequence[V3Event]]] = {}
    for candidate_id, candidate in model_payload["candidates"].items():
        appearance = str(candidate["appearanceMode"])
        events = _events(payloads[appearance], {"evaluation"})
        summary, probabilities, predictions = _candidate_evaluation(
            candidate, events
        )
        evaluations[candidate_id] = summary
        score_rows[candidate_id] = (probabilities, predictions, events)

    selected_id = str(model_payload["selectedCandidate"])
    selected_candidate = model_payload["candidates"][selected_id]
    selected_probabilities, selected_predictions, selected_events = score_rows[
        selected_id
    ]
    selected_classifier = V6Model.from_dict(
        selected_candidate["classifier"],
        expected_names=selected_candidate["featureNames"],
    )
    selected_decoder = OrientationDecoderSettings.from_dict(
        selected_candidate["decoder"]
    )
    margin_sensitivity = {
        str(margin): _metric_sensitivity(
            selected_events,
            decode_all(
                selected_events,
                selected_probabilities,
                selected_classifier.threshold,
                replace(selected_decoder, candidate_margin=margin),
            ),
        )
        for margin in MARGIN_GRID
    }
    cadence_sensitivity = {}
    for margin in MARGIN_GRID:
        cadence = decode_cadence_all(
            selected_events,
            np.full(len(selected_events), 0.5, dtype=np.float64),
            0.5,
            DecoderSettings(candidate_margin=margin, distance_penalty=1.0),
            force_each_opportunity=True,
        )
        cadence_sensitivity[str(margin)] = _metric_sensitivity(
            selected_events, cadence
        )

    overlap_ids = set(
        str(value)
        for value in model_payload["suppressionQuarantine"][
            "overlappingRecordingIds"
        ]
    )
    suppression = _suppression_diagnostic(selected_events, overlap_ids)
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-production-state-evaluation-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "family": family,
        "modelPath": str(model_dir / "model.json"),
        "modelSha256": _sha256(model_dir / "model.json"),
        "selectedCandidate": selected_id,
        "counts": _counts(selected_events),
        "baseline": _baseline_payload(baseline_path, family),
        "candidates": evaluations,
        "selectedCandidateMarginSensitivity": margin_sensitivity,
        "cadenceOnlyByMargin": cadence_sensitivity,
        "plusMinusFour": {
            "selected": margin_sensitivity["4"],
            "cadenceOnly": cadence_sensitivity["4"],
        },
        "suppressionDiagnostic": suppression,
        "knownLimitations": {
            "retrospectiveConfirmation": (
                "the raw evaluation labels were used by prior V1-V6 research; this "
                "is not a pristine protected test"
            ),
            "candidateConditioned": (
                "metrics cover reviewed rally-detector gaps; unproposed gaps remain unlabeled"
            ),
            "historicalProductionOverlap": (
                "the production rally heads were fitted on several historical side-switch "
                "development videos, but never on the side-switch target"
            ),
            "suppressionLeakage": (
                "suppression scores are quarantined diagnostics because that head's target "
                "included side switches and its fitting recordings overlap every split"
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
                    for candidate_id, values in score_rows.items()
                },
                "candidatePredictions": {
                    candidate_id: bool(values[1][index])
                    for candidate_id, values in score_rows.items()
                },
            }
            for index, event in enumerate(selected_events)
        ],
        "sources": model_payload["sources"],
        "productionModels": model_payload["productionModels"],
        "suppressionQuarantine": model_payload["suppressionQuarantine"],
    }
    _write(output, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "evaluate", "attribute"):
        child = commands.add_parser(command)
        child.add_argument("--family", choices=("v5", "v6"), required=True)
        child.add_argument("--original-features", type=Path)
        child.add_argument("--grounded-features", type=Path)
        child.add_argument("--model-dir", type=Path)
        if command == "evaluate":
            child.add_argument("--evaluation-output", type=Path)
            child.add_argument("--baseline-evaluation", type=Path)
        if command == "attribute":
            child.add_argument("--attribution-output", type=Path)
    return parser


def _resolve_defaults(args: argparse.Namespace) -> None:
    defaults = DEFAULTS[args.family]
    if args.original_features is None:
        args.original_features = defaults["original"]
    if args.grounded_features is None:
        args.grounded_features = defaults["grounded"]
    if args.model_dir is None:
        args.model_dir = defaults["model"]
    if args.command == "evaluate":
        if args.evaluation_output is None:
            args.evaluation_output = defaults["evaluation"]
        if args.baseline_evaluation is None:
            args.baseline_evaluation = defaults["baseline"]
    if args.command == "attribute" and args.attribution_output is None:
        args.attribution_output = defaults["attribution"]


def main() -> None:
    args = _parser().parse_args()
    _resolve_defaults(args)
    if args.command == "freeze":
        payload = freeze(args)
    elif args.command == "evaluate":
        payload = evaluate(args)
    else:
        payload = attribute(args)
    if args.command == "freeze":
        print(
            json.dumps(
                {
                    "family": args.family,
                    "selectedCandidate": payload["selectedCandidate"],
                    "fingerprint": payload["fingerprint"],
                    "validation": payload["candidates"][
                        payload["selectedCandidate"]
                    ]["validation"],
                },
                indent=2,
                allow_nan=False,
            )
        )
    elif args.command == "evaluate":
        selected = payload["candidates"][payload["selectedCandidate"]]
        print(
            json.dumps(
                {
                    "family": args.family,
                    "selectedCandidate": payload["selectedCandidate"],
                    "rowAveragePrecision": selected["rowAveragePrecision"],
                    "overall": selected["overall"],
                    "suppressionDiagnostic": payload["suppressionDiagnostic"],
                },
                indent=2,
                allow_nan=False,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "family": args.family,
                    "profiles": {
                        name: {
                            "validation": value["validation"],
                            "retrospective": value["retrospective"],
                        }
                        for name, value in payload["profiles"].items()
                    },
                },
                indent=2,
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
