#!/usr/bin/env python3
"""Freeze and evaluate the reviewed side-switch v2 specialist.

The ``freeze`` phase performs recording-grouped training selection and validation
tuning, then writes an immutable model without scoring evaluation recordings.
The ``evaluate`` phase loads that frozen model and opens the four preregistered
evaluation recordings once.
"""

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
from analysis.side_switch_v2 import (
    FEATURE_ARTIFACT_KIND,
    FEATURE_SETS,
    FROZEN_RECORDING_SPLIT,
    MODEL_KIND,
    MODEL_SCHEMA_VERSION,
    DecoderSettings,
    V2Event,
    V2Model,
    binary_metrics,
    decode_sequence,
    event_vector,
    fingerprint,
    fit_model,
    grouped_cross_fit,
    labels_for,
    matrix_for,
    select_threshold,
    with_threshold,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_FEATURES = ROOT / "reports/side-switch/side-switch-v2-features.json"
DEFAULT_DECISIONS = (
    ROOT / "reports/side-switch/appearance-review-decisions-full-nas-v1.json"
)
DEFAULT_MODEL_DIR = ROOT / "models/side-switch-specialist-v2"
DEFAULT_EVALUATION = (
    ROOT / "reports/side-switch/side-switch-specialist-v2-evaluation.json"
)
EXPECTED_DECISION_MAP_SHA256 = (
    "5a59727692470ab4984a29d0d48138525e1b5f28b56b3fab630c77070ce1dbd7"
)
L2_GRID = (0.01, 0.1, 1.0, 10.0)


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


def _decision_map_sha256(payload: Mapping[str, Any]) -> str:
    decisions = payload.get("decisions")
    if not isinstance(decisions, Mapping):
        raise ValueError("review decision payload has no decisions map")
    encoded = json.dumps(
        decisions, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reviewed_events(
    feature_payload: Mapping[str, Any],
    decisions_payload: Mapping[str, Any],
    roles: set[str],
) -> tuple[list[V2Event], dict[str, int]]:
    rows = feature_payload.get("events")
    decisions = decisions_payload.get("decisions")
    if not isinstance(rows, list) or not isinstance(decisions, Mapping):
        raise ValueError("invalid feature or review-decision schema")
    events: list[V2Event] = []
    counts = {"switch": 0, "no-switch": 0, "unclear": 0}
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("role")) not in roles:
            continue
        event_id = str(row.get("eventId", ""))
        decision = decisions.get(event_id)
        if decision not in counts:
            raise ValueError(f"missing or invalid review decision for {event_id}")
        counts[str(decision)] += 1
        if decision == "unclear":
            continue
        events.append(
            V2Event(
                event_id=event_id,
                recording_id=str(row["recordingId"]),
                role=str(row["role"]),
                rally_order=int(row["rallyOrder"]),
                decision=str(decision),
                label=1 if decision == "switch" else 0,
                row=row,
            )
        )
    return events, counts


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(
        (np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0))
        / comparisons.size
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    positives = int(np.sum(ranked))
    if not positives:
        return None
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def _metric_bundle(
    events: Sequence[V2Event], scores: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    labels = labels_for(events).astype(np.int64)
    return {
        **binary_metrics(labels, predicted),
        "rocAuc": _rank_auc(labels, scores),
        "averagePrecision": _average_precision(labels, scores),
        "meanSwitchScore": (
            float(np.mean(scores[labels == 1])) if np.any(labels == 1) else None
        ),
        "meanNoSwitchScore": (
            float(np.mean(scores[labels == 0])) if np.any(labels == 0) else None
        ),
    }


def _by_recording(
    events: Sequence[V2Event], scores: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for recording_id in sorted({event.recording_id for event in events}):
        indices = np.asarray(
            [i for i, event in enumerate(events) if event.recording_id == recording_id]
        )
        result[recording_id] = _metric_bundle(
            [events[i] for i in indices], scores[indices], predicted[indices]
        )
    return result


def _decode_all(
    events: Sequence[V2Event],
    probabilities: np.ndarray,
    threshold: float,
    settings: DecoderSettings,
    *,
    calculate_scores: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    predicted = np.zeros(len(events), dtype=bool)
    scores = np.zeros(len(events), dtype=np.float64)
    sequence_scores: dict[str, float] = {}
    for recording_id in sorted({event.recording_id for event in events}):
        indices = np.asarray(
            [i for i, event in enumerate(events) if event.recording_id == recording_id]
        )
        recording_events = [events[i] for i in indices]
        order = np.argsort([event.rally_order for event in recording_events])
        ordered_indices = indices[order]
        ordered_events = [events[i] for i in ordered_indices]
        decoded, marginal, sequence_score = decode_sequence(
            ordered_events,
            probabilities[ordered_indices],
            threshold,
            settings,
            calculate_scores=calculate_scores,
        )
        predicted[ordered_indices] = decoded
        scores[ordered_indices] = marginal
        sequence_scores[recording_id] = sequence_score
    return predicted, scores, sequence_scores


def _candidate_rank(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["outOfFold"]["selectedThresholdMetrics"]
    return (
        float(metrics["f1"] or 0.0),
        float(metrics["precision"] or 0.0),
        float(metrics["recall"] or 0.0),
        -len(FEATURE_SETS[str(candidate["featureSet"])]),
        float(candidate["l2"]),
    )


def _decoder_grid() -> list[DecoderSettings]:
    return [
        DecoderSettings(spacing, close_penalty, orientation, extra_penalty)
        for spacing in (0, 4, 7, 10)
        for close_penalty in (0.0, 1.0, 2.0)
        for orientation in (0.0, 0.15, 0.3)
        for extra_penalty in (0.0, 0.35)
        if spacing > 0 or close_penalty == 0.0
    ]


def _decoder_rank(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["metrics"]
    settings = candidate["settings"]
    complexity = sum(
        float(settings[name]) > 0
        for name in (
            "minimumSpacingRallies",
            "closeSwitchPenalty",
            "orientationWeight",
            "extraSwitchPenalty",
        )
    )
    return (
        float(metrics["f1"] or 0.0),
        float(metrics["precision"] or 0.0),
        float(metrics["recall"] or 0.0),
        -complexity,
        -float(settings["orientationWeight"]),
    )


def _validate_sources(
    features_path: Path,
    decisions_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    features = _load_json(features_path)
    decisions = _load_json(decisions_path)
    if features.get("kind") != FEATURE_ARTIFACT_KIND:
        raise ValueError("feature artifact is not side-switch v2")
    expected_split = {
        role: list(recordings) for role, recordings in FROZEN_RECORDING_SPLIT.items()
    }
    if features.get("frozenSplit") != expected_split:
        raise ValueError("feature artifact does not carry the frozen split")
    decision_sha = _sha256(decisions_path)
    decision_map_sha = _decision_map_sha256(decisions)
    if decision_map_sha != EXPECTED_DECISION_MAP_SHA256:
        raise ValueError("review decision mapping changed from the frozen v2 contract")
    appearance = features.get("sources", {}).get("appearanceReport", {})
    if (
        decisions.get("reportKind") != appearance.get("kind")
        or decisions.get("reportCreatedAt") != appearance.get("createdAt")
    ):
        raise ValueError("review decisions do not belong to the feature source report")
    sources = {
        "features": {
            "path": str(features_path),
            "sha256": _sha256(features_path),
            "kind": features.get("kind"),
            "createdAt": features.get("createdAt"),
        },
        "reviewDecisions": {
            "path": str(decisions_path),
            "sha256": decision_sha,
            "decisionMapSha256": decision_map_sha,
            "savedAt": decisions.get("savedAt"),
        },
    }
    return features, decisions, sources


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    features_path = args.features.expanduser().resolve()
    decisions_path = args.decisions.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    if model_dir.exists():
        raise FileExistsError(f"refusing to reuse v2 model directory: {model_dir}")
    features, decisions, sources = _validate_sources(features_path, decisions_path)

    # Only development decisions enter this phase. Evaluation rows are not even
    # materialized as V2Event objects until the separate evaluate command.
    train_events, train_review = _reviewed_events(features, decisions, {"train"})
    validation_events, validation_review = _reviewed_events(
        features, decisions, {"validation"}
    )
    if {event.recording_id for event in train_events} != set(
        FROZEN_RECORDING_SPLIT["train"]
    ):
        raise ValueError("training events do not cover the frozen recordings")
    if {event.recording_id for event in validation_events} != set(
        FROZEN_RECORDING_SPLIT["validation"]
    ):
        raise ValueError("validation events do not cover the frozen recordings")

    candidates: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        for l2 in L2_GRID:
            scores, audit = grouped_cross_fit(train_events, feature_set, l2)
            candidates.append(
                {
                    "featureSet": feature_set,
                    "l2": l2,
                    "outOfFold": {
                        **audit,
                        "scoreMetricsAtSelectedThreshold": _metric_bundle(
                            train_events,
                            scores,
                            scores >= audit["selectedThresholdMetrics"]["threshold"],
                        ),
                    },
                }
            )
    candidates.sort(key=_candidate_rank, reverse=True)
    selected = candidates[0]
    model = fit_model(
        train_events, str(selected["featureSet"]), float(selected["l2"])
    )
    validation_probabilities = model.predict_proba(
        matrix_for(validation_events, model.feature_set)
    )
    threshold_selection = select_threshold(
        labels_for(validation_events), validation_probabilities
    )
    model = with_threshold(model, float(threshold_selection["threshold"]))
    static_validation = _metric_bundle(
        validation_events,
        validation_probabilities,
        validation_probabilities >= model.threshold,
    )

    decoder_candidates: list[dict[str, Any]] = []
    for settings in _decoder_grid():
        predicted, _, _ = _decode_all(
            validation_events,
            validation_probabilities,
            model.threshold,
            settings,
            calculate_scores=False,
        )
        decoder_candidates.append(
            {
                "settings": settings.to_dict(),
                "metrics": binary_metrics(labels_for(validation_events), predicted),
            }
        )
    decoder_candidates.sort(key=_decoder_rank, reverse=True)
    decoder = DecoderSettings.from_dict(decoder_candidates[0]["settings"])
    decoder_predictions, decoder_scores, sequence_scores = _decode_all(
        validation_events,
        validation_probabilities,
        model.threshold,
        decoder,
        calculate_scores=True,
    )
    decoder_validation = _metric_bundle(
        validation_events, decoder_scores, decoder_predictions
    )

    deployable = {"classifier": model.to_dict(), "decoder": decoder.to_dict()}
    model_fingerprint = fingerprint(deployable)
    created_at = datetime.now(UTC).isoformat()
    model_payload = {
        "schemaVersion": MODEL_SCHEMA_VERSION,
        "kind": MODEL_KIND,
        "createdAt": created_at,
        "fingerprint": model_fingerprint,
        "classifier": model.to_dict(),
        "decoder": decoder.to_dict(),
        "selection": {
            "featureFamilyAndL2": "maximum train recording-grouped OOF F1",
            "staticThreshold": "maximum validation F1; precision then recall tie-break",
            "decoderSettings": "maximum validation F1; precision then recall and simplicity tie-break",
            "l2Grid": list(L2_GRID),
            "selectedClassifierCandidate": selected,
            "staticValidationThreshold": threshold_selection,
            "staticValidationMetrics": static_validation,
            "decoderValidationMetrics": decoder_validation,
            "decoderValidationSequenceScores": sequence_scores,
            "classifierLeaderboard": candidates,
            "decoderLeaderboard": decoder_candidates,
        },
        "dataPolicy": {
            "trainRecordingIds": list(FROZEN_RECORDING_SPLIT["train"]),
            "validationRecordingIds": list(FROZEN_RECORDING_SPLIT["validation"]),
            "evaluationRecordingIds": list(FROZEN_RECORDING_SPLIT["evaluation"]),
            "unclear": "excluded",
            "gapDuration": "not in any learned feature set",
            "normalization": "unlabeled complete-sequence per recording",
            "evaluationExposure": "no evaluation V2Event was materialized during freeze",
            "indoorPolicy": "fixed no-switch and outside specialist ranking metrics",
        },
        "counts": {
            "train": {**train_review, "usable": len(train_events)},
            "validation": {**validation_review, "usable": len(validation_events)},
        },
        "sources": sources,
    }
    dataset = {
        "schemaVersion": 2,
        "kind": "volleycut-side-switch-specialist-development-dataset-v2",
        "createdAt": created_at,
        "modelFingerprint": model_fingerprint,
        "featureSet": model.feature_set,
        "featureNames": list(model.feature_names),
        "sources": sources,
        "rows": [
            {
                "eventId": event.event_id,
                "recordingId": event.recording_id,
                "role": event.role,
                "rallyOrder": event.rally_order,
                "decision": event.decision,
                "label": event.label,
                "features": [
                    None if not math.isfinite(value) else float(value)
                    for value in event_vector(event.row, model.feature_set)
                ],
            }
            for event in (*train_events, *validation_events)
        ],
    }
    atomic_write_text(
        model_dir / "model.json",
        json.dumps(model_payload, indent=2, allow_nan=False) + "\n",
    )
    atomic_write_text(
        model_dir / "dataset-development.json",
        json.dumps(dataset, indent=2, allow_nan=False) + "\n",
    )
    return model_payload


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    features_path = args.features.expanduser().resolve()
    decisions_path = args.decisions.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_path = args.evaluation_output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite v2 evaluation: {output_path}")
    model_payload = _load_json(model_dir / "model.json")
    features, decisions, sources = _validate_sources(features_path, decisions_path)
    if model_payload.get("kind") != MODEL_KIND:
        raise ValueError("model artifact is not side-switch v2")
    if model_payload.get("sources") != sources:
        raise ValueError("frozen model sources do not match evaluation inputs")
    deployable = {
        "classifier": model_payload["classifier"],
        "decoder": model_payload["decoder"],
    }
    if model_payload.get("fingerprint") != fingerprint(deployable):
        raise ValueError("frozen v2 model fingerprint is invalid")
    model = V2Model.from_dict(model_payload["classifier"])
    decoder = DecoderSettings.from_dict(model_payload["decoder"])

    # Evaluation decisions are first materialized here, after all choices are frozen.
    events, review_counts = _reviewed_events(features, decisions, {"evaluation"})
    if {event.recording_id for event in events} != set(
        FROZEN_RECORDING_SPLIT["evaluation"]
    ):
        raise ValueError("evaluation events do not cover the frozen recordings")
    probabilities = model.predict_proba(matrix_for(events, model.feature_set))
    static_predictions = probabilities >= model.threshold
    decoder_predictions, decoder_scores, sequence_scores = _decode_all(
        events,
        probabilities,
        model.threshold,
        decoder,
        calculate_scores=True,
    )
    created_at = datetime.now(UTC).isoformat()
    payload = {
        "schemaVersion": 2,
        "kind": "volleycut-side-switch-specialist-evaluation-v2",
        "createdAt": created_at,
        "modelFingerprint": model_payload["fingerprint"],
        "modelPath": str(model_dir / "model.json"),
        "developmentDatasetPath": str(model_dir / "dataset-development.json"),
        "evaluationPath": str(output_path),
        "reviewCounts": {**review_counts, "usable": len(events)},
        "frozenEvaluationRecordingIds": list(FROZEN_RECORDING_SPLIT["evaluation"]),
        "staticClassifier": {
            "threshold": model.threshold,
            "overall": _metric_bundle(events, probabilities, static_predictions),
            "byRecording": _by_recording(events, probabilities, static_predictions),
        },
        "temporalDecoder": {
            "settings": decoder.to_dict(),
            "overall": _metric_bundle(events, decoder_scores, decoder_predictions),
            "byRecording": _by_recording(events, decoder_scores, decoder_predictions),
            "sequenceScores": sequence_scores,
        },
        "indoorPolicy": (
            "fixed no-switch; no indoor recording is included in specialist metrics"
        ),
        "knownLimitations": {
            "candidateConditioned": (
                "metrics apply only to generated inter-rally candidates, not missed switches"
            ),
            "sourceGroup": (
                "all recordings share volleycut-raw-no-backup; recording-held-out only"
            ),
            "confirmationSet": (
                "raw labels and pooled v1 behavior informed v2 hypotheses; not pristine test data"
            ),
        },
        "predictions": [
            {
                "eventId": event.event_id,
                "recordingId": event.recording_id,
                "rallyOrder": event.rally_order,
                "decision": event.decision,
                "staticScore": float(static_score),
                "staticPrediction": "switch" if static_prediction else "no-switch",
                "decoderScore": float(decoder_score),
                "decoderPrediction": "switch" if decoder_prediction else "no-switch",
            }
            for event, static_score, static_prediction, decoder_score, decoder_prediction in zip(
                events,
                probabilities,
                static_predictions,
                decoder_scores,
                decoder_predictions,
                strict=True,
            )
        ],
        "sources": sources,
    }
    atomic_write_text(output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
        child.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
        child.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
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
                    "modelFingerprint": payload["fingerprint"],
                    "featureSet": payload["classifier"]["featureSet"],
                    "l2": payload["classifier"]["l2"],
                    "threshold": payload["classifier"]["threshold"],
                    "decoder": payload["decoder"],
                },
                indent=2,
            )
        )
    else:
        payload = evaluate(args)
        print(
            json.dumps(
                {
                    "static": payload["staticClassifier"]["overall"],
                    "decoder": payload["temporalDecoder"]["overall"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
