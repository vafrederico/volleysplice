#!/usr/bin/env python3
"""Evaluate imbalance priors and label-free recording calibration for side switches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_full_union_ranker import (
    DERIVED_FEATURE_NAMES,
    UnionDecoderSettings,
    add_derived_features,
    calibrate_recording_scores,
    decode_ranked_candidates,
    fit_weighted_logistic,
)
from analysis.side_switch_full_video import event_metric_counts, monotonic_interval_match
from analysis.side_switch_production_state import STATE_GATE_FEATURE_NAMES
from analysis.side_switch_v3 import V3Event, average_precision
from analysis.side_switch_v5 import VISUAL_FEATURE_NAMES
from analysis.side_switch_v6 import matrix_for


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_FEATURES = REPORTS / "side-switch-full-union-v5-state-features-v1.json"
DEFAULT_FULL_AUDIT = REPORTS / "side-switch-full-video-marker-evaluation-2026-08-21-r2.json"
DEFAULT_CONTROL = REPORTS / "side-switch-full-union-ranker-v1-evaluation.json"
DEFAULT_MODEL = ROOT / "models/side-switch-full-union-calibrated-v1/model.json"
DEFAULT_EVALUATION = REPORTS / "side-switch-full-union-calibrated-v1-evaluation.json"
EXPECTED_SHA256 = {
    "features": "9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551",
    "fullAudit": "142d6617aed7f20b51cdcdfc9b3e61c8c76beaf79a31fc2d7ee54fb0fdf27b89",
    "control": "e332b03b20387ed0d401c3fab37de31cdaafef76420f9045923c1c4c53919ccf",
}
PRIMARY_FEATURE_NAMES = (*VISUAL_FEATURE_NAMES, *STATE_GATE_FEATURE_NAMES)
FEATURE_GROUPS = {
    "primary32": PRIMARY_FEATURE_NAMES,
    "union34": (*PRIMARY_FEATURE_NAMES, *DERIVED_FEATURE_NAMES),
}
L2 = 0.1
PADDING = 4.0
THRESHOLD_QUANTILES = 65
DECODER = UnionDecoderSettings(
    minimum_index_separation=2,
    free_predictions_per_recording=6,
    count_penalty_logit=0.5,
)


@dataclass(frozen=True)
class Variant:
    objective: str
    class_balance_exponent: float
    calibration: str
    feature_group: str

    @property
    def identifier(self) -> str:
        return f"{self.objective}-{self.calibration}-{self.feature_group}"

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_GROUPS[self.feature_group]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "objective": self.objective,
            "classBalanceExponent": self.class_balance_exponent,
            "calibration": self.calibration,
            "featureGroup": self.feature_group,
            "featureNames": list(self.feature_names),
        }


VARIANTS = tuple(
    Variant(objective, exponent, calibration, feature_group)
    for objective, exponent, calibration in (
        ("natural", 0.0, "raw"),
        ("sqrt-balanced", 0.5, "raw"),
        ("balanced", 1.0, "raw"),
        ("balanced", 1.0, "robust-logit"),
        ("balanced", 1.0, "percentile"),
    )
    for feature_group in FEATURE_GROUPS
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
        "kind": str(row["kind"]),
        "gapStart": float(row["gapStart"]),
        "gapEnd": float(row["gapEnd"]),
        "transitionTime": float(row["transitionTime"]),
    }


def _labels(
    rows: Sequence[Mapping[str, Any]],
    markers: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    result = {str(row["eventId"]): 0 for row in rows}
    for recording_id, recording_markers in markers.items():
        recording_rows = [
            row for row in rows if str(row["recordingId"]) == recording_id
        ]
        match = monotonic_interval_match(
            [_proposal(row) for row in recording_rows], recording_markers, PADDING
        )
        for pair in match.pairs:
            result[str(recording_rows[pair.proposal_index]["eventId"])] = 1
    return result


def _events(rows: Sequence[Mapping[str, Any]], labels: Mapping[str, int]) -> list[V3Event]:
    ordinal: dict[str, int] = {}
    for recording_id in sorted({str(row["recordingId"]) for row in rows}):
        ordered = sorted(
            (row for row in rows if str(row["recordingId"]) == recording_id),
            key=lambda row: (float(row["transitionTime"]), str(row["eventId"])),
        )
        ordinal.update(
            {str(row["eventId"]): index for index, row in enumerate(ordered, 1)}
        )
    return [
        V3Event(
            str(row["eventId"]),
            str(row["recordingId"]),
            "opened-development",
            ordinal[str(row["eventId"])],
            int(labels[str(row["eventId"])]),
            row,
        )
        for row in rows
    ]


def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    predictions: np.ndarray,
    markers: Mapping[str, Sequence[Mapping[str, Any]]],
    padding: float,
    *,
    inventory: bool = False,
) -> dict[str, Any]:
    selected = np.asarray(predictions, dtype=bool)
    per_video: dict[str, Any] = {}
    for recording_id, recording_markers in markers.items():
        proposals = sorted(
            [
                _proposal(row)
                for index, row in enumerate(rows)
                if selected[index] and str(row["recordingId"]) == recording_id
            ],
            key=lambda row: (row["transitionTime"], row["eventId"]),
        )
        match = monotonic_interval_match(proposals, recording_markers, padding)
        counts = event_metric_counts(match)
        per_video[recording_id] = {
            "humanEvents": len(recording_markers),
            "proposals": len(proposals),
            **counts,
            "missedHumanTimes": [
                float(recording_markers[index]["time"])
                for index in match.unmatched_marker_indices
            ],
        }
        if inventory:
            per_video[recording_id]["proposalInventory"] = proposals
    tp = sum(int(value["truePositives"]) for value in per_video.values())
    fp = sum(int(value["falsePositives"]) for value in per_video.values())
    fn = sum(int(value["falseNegatives"]) for value in per_video.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "paddingSeconds": padding,
        "proposals": tp + fp,
        "truePositives": tp,
        "falsePositives": fp,
        "falseNegatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "macroPerVideoPrecision": float(
            np.mean([float(value["precision"] or 0.0) for value in per_video.values()])
        ),
        "macroPerVideoRecall": float(
            np.mean([float(value["recall"] or 0.0) for value in per_video.values()])
        ),
        "averagePerVideo": {
            "proposals": (tp + fp) / len(per_video),
            "truePositives": tp / len(per_video),
            "falsePositives": fp / len(per_video),
            "falseNegatives": fn / len(per_video),
        },
        "byRecording": per_video,
    }


def _crossfit_raw(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int],
    variant: Variant,
) -> np.ndarray:
    events = _events(rows, labels)
    scores = np.full(len(rows), np.nan)
    recording_ids = sorted({str(row["recordingId"]) for row in rows})
    for held_id in recording_ids:
        fit_indexes = [
            index for index, row in enumerate(rows) if str(row["recordingId"]) != held_id
        ]
        held_indexes = [
            index for index, row in enumerate(rows) if str(row["recordingId"]) == held_id
        ]
        model = fit_weighted_logistic(
            [events[index] for index in fit_indexes],
            L2,
            variant.feature_names,
            variant.class_balance_exponent,
        )
        scores[held_indexes] = model.predict_proba(
            matrix_for([events[index] for index in held_indexes], variant.feature_names)
        )
    if not np.isfinite(scores).all():
        raise ValueError("weighted cross-fit left rows unscored")
    return scores


def _thresholds(scores: np.ndarray) -> tuple[float, ...]:
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, THRESHOLD_QUANTILES))
    above = min(1.0, math.nextafter(float(np.max(scores)), math.inf))
    return tuple(sorted({above, *(float(value) for value in quantiles)}, reverse=True))


def _rank(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = candidate["metrics"]
    variant = candidate["variant"]
    calibration_complexity = {"raw": 0, "robust-logit": 1, "percentile": 1}[
        variant["calibration"]
    ]
    return (
        float(metrics["f1"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        -float(metrics["proposals"]),
        -float(calibration_complexity),
        -float(len(variant["featureNames"])),
        float(candidate["threshold"]),
    )


def _select_threshold(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    markers: Mapping[str, Sequence[Mapping[str, Any]]],
    variant: Variant,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in _thresholds(scores):
        predictions = decode_ranked_candidates(rows, scores, threshold, DECODER)
        candidate = {
            "variant": variant.to_dict(),
            "threshold": threshold,
            "metrics": _evaluate(rows, predictions, markers, PADDING),
        }
        if best is None or _rank(candidate) > _rank(best):
            best = candidate
    if best is None:
        raise AssertionError("calibration threshold selection failed")
    return best


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "features": args.features.expanduser().resolve(),
        "fullAudit": args.full_audit.expanduser().resolve(),
        "control": args.control.expanduser().resolve(),
    }
    model_path = args.model.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    if model_path.exists() or evaluation_path.exists():
        raise FileExistsError("refusing to overwrite calibrated side-switch artifacts")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"calibration source identity changed: {hashes}")
    features = _load(paths["features"])
    full_audit = _load(paths["fullAudit"])
    control = _load(paths["control"])
    rows = [add_derived_features(row) for row in features["rows"]]
    recording_ids = tuple(features["scope"]["recordingIds"])
    markers = {
        recording_id: [
            {"time": float(value)}
            for value in full_audit["scope"]["humanEventsByRecording"][recording_id]
        ]
        for recording_id in recording_ids
    }
    labels = _labels(rows, markers)
    if sum(labels.values()) != 46:
        raise ValueError("calibration label universe changed")

    nested_scores = np.full(len(rows), np.nan)
    nested_predictions = np.zeros(len(rows), dtype=bool)
    variant_predictions = {
        variant.identifier: np.zeros(len(rows), dtype=bool) for variant in VARIANTS
    }
    variant_scores = {
        variant.identifier: np.full(len(rows), np.nan) for variant in VARIANTS
    }
    outer_folds: list[dict[str, Any]] = []
    for fold_number, held_id in enumerate(recording_ids, 1):
        fit_indexes = [
            index for index, row in enumerate(rows) if str(row["recordingId"]) != held_id
        ]
        held_indexes = [
            index for index, row in enumerate(rows) if str(row["recordingId"]) == held_id
        ]
        fit_rows = [rows[index] for index in fit_indexes]
        held_rows = [rows[index] for index in held_indexes]
        fit_markers = {key: value for key, value in markers.items() if key != held_id}
        print(
            f"[{fold_number}/{len(recording_ids)}] imbalance/calibration held={held_id}",
            flush=True,
        )
        raw_fit_cache: dict[tuple[float, str], np.ndarray] = {}
        raw_held_cache: dict[tuple[float, str], np.ndarray] = {}
        selections: list[dict[str, Any]] = []
        for variant in VARIANTS:
            cache_key = (variant.class_balance_exponent, variant.feature_group)
            if cache_key not in raw_fit_cache:
                raw_fit_cache[cache_key] = _crossfit_raw(fit_rows, labels, variant)
                fit_events = _events(fit_rows, labels)
                held_events = _events(held_rows, labels)
                classifier = fit_weighted_logistic(
                    fit_events,
                    L2,
                    variant.feature_names,
                    variant.class_balance_exponent,
                )
                raw_held_cache[cache_key] = classifier.predict_proba(
                    matrix_for(held_events, variant.feature_names)
                )
            fit_scores = calibrate_recording_scores(
                fit_rows, raw_fit_cache[cache_key], variant.calibration
            )
            held_scores = calibrate_recording_scores(
                held_rows, raw_held_cache[cache_key], variant.calibration
            )
            selected = _select_threshold(fit_rows, fit_scores, fit_markers, variant)
            held_predictions = decode_ranked_candidates(
                held_rows, held_scores, float(selected["threshold"]), DECODER
            )
            variant_scores[variant.identifier][held_indexes] = held_scores
            variant_predictions[variant.identifier][held_indexes] = held_predictions
            selections.append(selected)
        selections.sort(key=_rank, reverse=True)
        winner = selections[0]
        winner_id = str(winner["variant"]["id"])
        nested_scores[held_indexes] = variant_scores[winner_id][held_indexes]
        nested_predictions[held_indexes] = variant_predictions[winner_id][held_indexes]
        outer_folds.append(
            {
                "heldRecordingId": held_id,
                "selected": winner,
                "leaderboard": selections,
            }
        )
    if not np.isfinite(nested_scores).all() or any(
        not np.isfinite(values).all() for values in variant_scores.values()
    ):
        raise ValueError("outer calibration evaluation left rows unscored")

    label_vector = np.asarray([labels[str(row["eventId"])] for row in rows])
    variant_results: dict[str, Any] = {}
    for variant in VARIANTS:
        predictions = variant_predictions[variant.identifier]
        variant_results[variant.identifier] = {
            "variant": variant.to_dict(),
            "rowAveragePrecision": average_precision(
                label_vector, variant_scores[variant.identifier]
            ),
            "strict": _evaluate(rows, predictions, markers, 0.0),
            "primary": _evaluate(rows, predictions, markers, PADDING, inventory=True),
        }

    full_raw_cache: dict[tuple[float, str], np.ndarray] = {}
    full_candidates: list[dict[str, Any]] = []
    for variant in VARIANTS:
        cache_key = (variant.class_balance_exponent, variant.feature_group)
        if cache_key not in full_raw_cache:
            full_raw_cache[cache_key] = _crossfit_raw(rows, labels, variant)
        scores = calibrate_recording_scores(
            rows, full_raw_cache[cache_key], variant.calibration
        )
        candidate = _select_threshold(rows, scores, markers, variant)
        candidate["rowAveragePrecision"] = average_precision(label_vector, scores)
        full_candidates.append(candidate)
    full_candidates.sort(key=_rank, reverse=True)
    selected_full = full_candidates[0]
    selected_variant = next(
        variant
        for variant in VARIANTS
        if variant.identifier == selected_full["variant"]["id"]
    )
    final_events = _events(rows, labels)
    final_classifier = fit_weighted_logistic(
        final_events,
        L2,
        selected_variant.feature_names,
        selected_variant.class_balance_exponent,
    )

    opposite_events = [replace(event, label=1 - event.label) for event in final_events]
    opposite_classifier = fit_weighted_logistic(
        opposite_events,
        L2,
        selected_variant.feature_names,
        selected_variant.class_balance_exponent,
    )
    matrix = matrix_for(final_events, selected_variant.feature_names)
    positive_scores = final_classifier.predict_proba(matrix)
    no_switch_scores = opposite_classifier.predict_proba(matrix)
    complement_error = float(np.max(np.abs(positive_scores - (1.0 - no_switch_scores))))
    if complement_error > 1e-10:
        raise ValueError("opposite-head symmetry control failed")

    created_at = datetime.now(UTC).isoformat()
    model_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-full-union-calibrated-v1",
        "createdAt": created_at,
        "status": "research-only-opened-development",
        "classifier": final_classifier.to_dict(),
        "classBalanceExponent": selected_variant.class_balance_exponent,
        "calibration": selected_variant.calibration,
        "threshold": float(selected_full["threshold"]),
        "decoder": DECODER.to_dict(),
        "selection": selected_full,
        "sources": {
            name: {"path": str(paths[name]), "sha256": hashes[name]}
            for name in paths
        },
    }
    evaluation_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-full-union-calibration-evaluation-v1",
        "createdAt": created_at,
        "scope": {
            **features["scope"],
            "humanMarkers": sum(map(len, markers.values())),
            "positiveCandidateLabels": int(np.sum(label_vector)),
            "status": "opened-development-only",
        },
        "protocol": {
            "outer": "leave one recording out from variant and threshold selection",
            "inner": "grouped leave-one-recording-out probabilities on outer-fit recordings",
            "fixedL2": L2,
            "fixedDecoder": DECODER.to_dict(),
            "thresholdQuantiles": THRESHOLD_QUANTILES,
            "variants": [variant.to_dict() for variant in VARIANTS],
        },
        "control": control["metrics"]["nestedOuterLoo"]["withoutContinuity"],
        "nestedVariantSelection": {
            "rowAveragePrecision": average_precision(label_vector, nested_scores),
            "strict": _evaluate(rows, nested_predictions, markers, 0.0),
            "primary": _evaluate(
                rows, nested_predictions, markers, PADDING, inventory=True
            ),
        },
        "fixedVariantOuterResults": variant_results,
        "outerFolds": outer_folds,
        "fullDevelopmentSelection": {
            "selected": selected_full,
            "leaderboard": full_candidates,
        },
        "oppositeHeadControl": {
            "question": "Does separately predicting no-switch add information?",
            "maximumComplementAbsoluteError": complement_error,
            "conclusion": "No under the same binary features and symmetric weighted logistic objective; the no-switch head is the numerical complement of the switch head.",
        },
        "sources": model_payload["sources"],
        "limitations": [
            "All 11 recordings are opened development rather than an untouched test split.",
            "Recording calibration uses the complete candidate score distribution, so streaming use would need an explicitly different causal contract.",
            "This loop fixes the Loop-4-selected L2 and dominant decoder to isolate objective and calibration effects.",
        ],
    }
    model_path.parent.mkdir(parents=True, exist_ok=False)
    atomic_write_text(model_path, json.dumps(model_payload, indent=2, allow_nan=False) + "\n")
    atomic_write_text(
        evaluation_path,
        json.dumps(evaluation_payload, indent=2, allow_nan=False) + "\n",
    )
    return model_payload, evaluation_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--full-audit", type=Path, default=DEFAULT_FULL_AUDIT)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
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
                "selection": model["selection"],
                "nested": evaluation["nestedVariantSelection"],
                "oppositeHead": evaluation["oppositeHeadControl"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
