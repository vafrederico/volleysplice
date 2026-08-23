#!/usr/bin/env python3
"""Evaluate within-recording pairwise ranking for rare side-switch events."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_full_union_ranker import (
    DERIVED_FEATURE_NAMES,
    UnionDecoderSettings,
    add_derived_features,
    decode_ranked_candidates,
    fit_weighted_logistic,
    fit_within_recording_pairwise_logistic,
)
from analysis.side_switch_full_video import event_metric_counts, monotonic_interval_match
from analysis.side_switch_production_state import STATE_GATE_FEATURE_NAMES
from analysis.side_switch_v3 import V3Event, average_precision
from analysis.side_switch_v5 import VISUAL_FEATURE_NAMES
from analysis.side_switch_v6 import matrix_for


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPORTS / "side-switch-full-union-v5-state-features-v1.json"
DEFAULT_FULL_AUDIT = REPORTS / "side-switch-full-video-marker-evaluation-2026-08-21-r2.json"
DEFAULT_CONTROL = REPORTS / "side-switch-hard-negative-mining-v1-evaluation.json"
DEFAULT_WINNER = REPOSITORY / "data/side-switch-current-research-winner-v1.json"
DEFAULT_MODEL = ROOT / "models/side-switch-pairwise-ranking-v1/model.json"
DEFAULT_EVALUATION = REPORTS / "side-switch-pairwise-ranking-v1-evaluation.json"
EXPECTED_SHA256 = {
    "features": "9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551",
    "fullAudit": "142d6617aed7f20b51cdcdfc9b3e61c8c76beaf79a31fc2d7ee54fb0fdf27b89",
    "control": "e67088b36d177d68c24587efcb186eab4201b5294532be1fdf978ef13aaacc4b",
    "winner": "163c7844267bc48410e89f86bd5cbf0a58b3145c1ec691932e4e3374dae7286c",
}
FEATURE_NAMES = (
    *VISUAL_FEATURE_NAMES,
    *STATE_GATE_FEATURE_NAMES,
    *DERIVED_FEATURE_NAMES,
)
L2 = 0.1
CLASS_BALANCE_EXPONENT = 0.5
HARD_NEGATIVES_PER_RECORDING = 2
HARD_NEGATIVE_MULTIPLIER = 2.0
PADDING = 4.0
THRESHOLD_QUANTILES = 65
DECODER = UnionDecoderSettings(
    minimum_index_separation=2,
    free_predictions_per_recording=6,
    count_penalty_logit=0.5,
)


@dataclass(frozen=True)
class Variant:
    identifier: str
    pairwise_strength: float
    pairwise_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "pairwiseStrength": self.pairwise_strength,
            "pairwiseMarginLogit": self.pairwise_margin,
        }


VARIANTS = (
    Variant("pointwise-control", 0.0, 0.0),
    Variant("pairwise-lambda0.25", 0.25, 0.0),
    Variant("pairwise-lambda0.5", 0.5, 0.0),
    Variant("pairwise-lambda1", 1.0, 0.0),
    Variant("pairwise-lambda2", 2.0, 0.0),
    Variant("pairwise-lambda0.5-margin0.5", 0.5, 0.5),
    Variant("pairwise-lambda1-margin0.5", 1.0, 0.5),
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


def _labels(rows: Sequence[Mapping[str, Any]], markers: Mapping[str, Any]) -> dict[str, int]:
    result = {str(row["eventId"]): 0 for row in rows}
    for recording_id, truth in markers.items():
        local = [row for row in rows if str(row["recordingId"]) == recording_id]
        match = monotonic_interval_match([_proposal(row) for row in local], truth, PADDING)
        for pair in match.pairs:
            result[str(local[pair.proposal_index]["eventId"])] = 1
    return result


def _events(rows: Sequence[Mapping[str, Any]], labels: Mapping[str, int]) -> list[V3Event]:
    order: dict[str, int] = {}
    for recording_id in sorted({str(row["recordingId"]) for row in rows}):
        local = sorted(
            (row for row in rows if str(row["recordingId"]) == recording_id),
            key=lambda row: (float(row["transitionTime"]), str(row["eventId"])),
        )
        order.update({str(row["eventId"]): index for index, row in enumerate(local, 1)})
    return [
        V3Event(
            str(row["eventId"]),
            str(row["recordingId"]),
            "opened-development",
            order[str(row["eventId"])],
            labels[str(row["eventId"])],
            row,
        )
        for row in rows
    ]


def _fit(events: Sequence[V3Event], variant: Variant) -> tuple[Any, dict[str, Any]]:
    initial = fit_weighted_logistic(
        events,
        L2,
        FEATURE_NAMES,
        CLASS_BALANCE_EXPONENT,
    )
    initial_scores = initial.predict_proba(matrix_for(events, FEATURE_NAMES))
    multipliers = np.ones(len(events), dtype=np.float64)
    selected_ids: dict[str, list[str]] = {}
    for recording_id in sorted({event.recording_id for event in events}):
        negatives = [
            index
            for index, event in enumerate(events)
            if event.recording_id == recording_id and event.label == 0
        ]
        chosen = sorted(
            negatives,
            key=lambda index: (-float(initial_scores[index]), events[index].event_id),
        )[:HARD_NEGATIVES_PER_RECORDING]
        multipliers[chosen] = HARD_NEGATIVE_MULTIPLIER
        selected_ids[recording_id] = [events[index].event_id for index in chosen]
    model = fit_within_recording_pairwise_logistic(
        events,
        L2,
        FEATURE_NAMES,
        CLASS_BALANCE_EXPONENT,
        variant.pairwise_strength,
        variant.pairwise_margin,
        multipliers,
    )
    return model, {
        "selectedHardNegatives": int(np.sum(multipliers > 1.0)),
        "byRecording": selected_ids,
    }


def _crossfit(
    rows: Sequence[Mapping[str, Any]], labels: Mapping[str, int], variant: Variant
) -> np.ndarray:
    events = _events(rows, labels)
    scores = np.full(len(rows), np.nan)
    for held_id in sorted({event.recording_id for event in events}):
        fit = [index for index, event in enumerate(events) if event.recording_id != held_id]
        held = [index for index, event in enumerate(events) if event.recording_id == held_id]
        model, _ = _fit([events[index] for index in fit], variant)
        scores[held] = model.predict_proba(
            matrix_for([events[index] for index in held], FEATURE_NAMES)
        )
    if not np.isfinite(scores).all():
        raise ValueError("pairwise cross-fit left rows unscored")
    return scores


def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    predictions: np.ndarray,
    markers: Mapping[str, Any],
    padding: float,
    *,
    inventory: bool = False,
) -> dict[str, Any]:
    per_video: dict[str, Any] = {}
    for recording_id, truth in markers.items():
        proposals = sorted(
            [
                _proposal(row)
                for row, selected in zip(rows, predictions, strict=True)
                if selected and str(row["recordingId"]) == recording_id
            ],
            key=lambda row: (row["transitionTime"], row["eventId"]),
        )
        match = monotonic_interval_match(proposals, truth, padding)
        counts = event_metric_counts(match)
        per_video[recording_id] = {
            "humanEvents": len(truth),
            "proposals": len(proposals),
            **counts,
            "missedHumanTimes": [
                float(truth[index]["time"]) for index in match.unmatched_marker_indices
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
        "byRecording": per_video,
    }


def _thresholds(scores: np.ndarray) -> tuple[float, ...]:
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, THRESHOLD_QUANTILES))
    above = min(1.0, math.nextafter(float(np.max(scores)), math.inf))
    return tuple(sorted({above, *(float(value) for value in quantiles)}, reverse=True))


def _rank(value: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = value["metrics"]
    variant = value["variant"]
    return (
        float(metrics["f1"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        -float(metrics["proposals"]),
        -float(variant["pairwiseStrength"]),
        -float(variant["pairwiseMarginLogit"]),
        float(value["threshold"]),
    )


def _select(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    markers: Mapping[str, Any],
    variant: Variant,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in _thresholds(scores):
        predictions = decode_ranked_candidates(rows, scores, threshold, DECODER)
        value = {
            "variant": variant.to_dict(),
            "threshold": threshold,
            "metrics": _evaluate(rows, predictions, markers, PADDING),
        }
        if best is None or _rank(value) > _rank(best):
            best = value
    if best is None:
        raise AssertionError("pairwise threshold selection failed")
    return best


def _metric_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in ("proposals", "truePositives", "falsePositives", "falseNegatives")
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "features": args.features.expanduser().resolve(),
        "fullAudit": args.full_audit.expanduser().resolve(),
        "control": args.control.expanduser().resolve(),
        "winner": args.winner.expanduser().resolve(),
    }
    model_path = args.model.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    if model_path.exists() or evaluation_path.exists():
        raise FileExistsError("refusing to overwrite pairwise-ranking artifacts")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"pairwise-ranking source identity changed: {hashes}")
    features = _load(paths["features"])
    full_audit = _load(paths["fullAudit"])
    control = _load(paths["control"])
    winner = _load(paths["winner"])
    if winner.get("winner", {}).get("id") != "side-switch-hard-negative-mining-v1/union34-top2-x2":
        raise ValueError("pairwise control is not the promoted hard-negative winner")
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
        raise ValueError("pairwise label universe changed")

    variant_scores = {
        variant.identifier: np.full(len(rows), np.nan) for variant in VARIANTS
    }
    variant_predictions = {
        variant.identifier: np.zeros(len(rows), dtype=bool) for variant in VARIANTS
    }
    nested_scores = np.full(len(rows), np.nan)
    nested_predictions = np.zeros(len(rows), dtype=bool)
    outer_folds: list[dict[str, Any]] = []
    for fold_number, held_id in enumerate(recording_ids, 1):
        print(f"[{fold_number}/{len(recording_ids)}] pairwise held={held_id}", flush=True)
        fit_indexes = [
            index for index, row in enumerate(rows) if str(row["recordingId"]) != held_id
        ]
        held_indexes = [
            index for index, row in enumerate(rows) if str(row["recordingId"]) == held_id
        ]
        fit_rows = [rows[index] for index in fit_indexes]
        held_rows = [rows[index] for index in held_indexes]
        fit_events = _events(fit_rows, labels)
        held_events = _events(held_rows, labels)
        fit_markers = {key: value for key, value in markers.items() if key != held_id}
        selections: list[dict[str, Any]] = []
        for variant in VARIANTS:
            fit_scores = _crossfit(fit_rows, labels, variant)
            held_model, mining_audit = _fit(fit_events, variant)
            held_scores = held_model.predict_proba(matrix_for(held_events, FEATURE_NAMES))
            selected = _select(fit_rows, fit_scores, fit_markers, variant)
            held_predictions = decode_ranked_candidates(
                held_rows, held_scores, float(selected["threshold"]), DECODER
            )
            variant_scores[variant.identifier][held_indexes] = held_scores
            variant_predictions[variant.identifier][held_indexes] = held_predictions
            selections.append({**selected, "miningAudit": mining_audit})
        selections.sort(key=_rank, reverse=True)
        selected = selections[0]
        selected_id = str(selected["variant"]["id"])
        nested_scores[held_indexes] = variant_scores[selected_id][held_indexes]
        nested_predictions[held_indexes] = variant_predictions[selected_id][held_indexes]
        outer_folds.append(
            {
                "heldRecordingId": held_id,
                "selected": selected,
                "leaderboard": selections,
            }
        )
    if not np.isfinite(nested_scores).all() or any(
        not np.isfinite(scores).all() for scores in variant_scores.values()
    ):
        raise ValueError("pairwise outer evaluation left rows unscored")

    labels_array = np.asarray([labels[str(row["eventId"])] for row in rows])
    fixed_results: dict[str, Any] = {}
    for variant in VARIANTS:
        fixed_results[variant.identifier] = {
            "variant": variant.to_dict(),
            "rowAveragePrecision": average_precision(
                labels_array, variant_scores[variant.identifier]
            ),
            "strict": _evaluate(
                rows, variant_predictions[variant.identifier], markers, 0.0
            ),
            "primary": _evaluate(
                rows,
                variant_predictions[variant.identifier],
                markers,
                PADDING,
                inventory=True,
            ),
        }
    source_control = control["fixedVariantOuterResults"]["union34-top2-x2"]
    for key in ("strict", "primary"):
        if _metric_identity(fixed_results["pointwise-control"][key]) != _metric_identity(
            source_control[key]
        ):
            raise ValueError(f"promoted pointwise control failed {key} metric parity")

    full_candidates: list[dict[str, Any]] = []
    for variant in VARIANTS:
        scores = _crossfit(rows, labels, variant)
        selected = _select(rows, scores, markers, variant)
        selected["rowAveragePrecision"] = average_precision(labels_array, scores)
        full_candidates.append(selected)
    full_candidates.sort(key=_rank, reverse=True)
    selected_full = full_candidates[0]
    selected_variant = next(
        variant
        for variant in VARIANTS
        if variant.identifier == selected_full["variant"]["id"]
    )
    final_model, final_mining_audit = _fit(_events(rows, labels), selected_variant)
    created_at = datetime.now(UTC).isoformat()
    model_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-pairwise-ranking-v1",
        "createdAt": created_at,
        "status": "research-only-opened-development",
        "classifier": final_model.to_dict(),
        "objective": {
            "featureProfile": "FULL-UNION-V5-STATE42/union34",
            "classBalanceExponent": CLASS_BALANCE_EXPONENT,
            "hardNegativesPerRecording": HARD_NEGATIVES_PER_RECORDING,
            "hardNegativeMultiplier": HARD_NEGATIVE_MULTIPLIER,
            **selected_variant.to_dict(),
            "pairConstruction": "all positive-negative pairs within each fit recording",
            "pairWeighting": "equal total pair weight per eligible recording",
        },
        "hardNegativeMiningAudit": final_mining_audit,
        "threshold": float(selected_full["threshold"]),
        "decoder": DECODER.to_dict(),
        "selection": selected_full,
        "sources": {
            name: {"path": str(paths[name]), "sha256": hashes[name]} for name in paths
        },
    }
    evaluation_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-pairwise-ranking-evaluation-v1",
        "createdAt": created_at,
        "scope": {
            **features["scope"],
            "humanMarkers": 50,
            "positiveCandidateLabels": int(np.sum(labels_array)),
            "status": "opened-development-only",
        },
        "protocol": {
            "outer": "leave one recording out from pairwise variant and threshold selection",
            "inner": "grouped leave-one-recording-out probabilities on each outer-fit scope",
            "fixedBase": {
                "featureNames": list(FEATURE_NAMES),
                "l2": L2,
                "classBalanceExponent": CLASS_BALANCE_EXPONENT,
                "hardNegativesPerRecording": HARD_NEGATIVES_PER_RECORDING,
                "hardNegativeMultiplier": HARD_NEGATIVE_MULTIPLIER,
                "decoder": DECODER.to_dict(),
            },
            "pairwiseLoss": (
                "recording-balanced mean softplus(margin - positiveLogit + negativeLogit)"
            ),
            "variants": [variant.to_dict() for variant in VARIANTS],
        },
        "promotedControl": source_control,
        "nestedVariantSelection": {
            "rowAveragePrecision": average_precision(labels_array, nested_scores),
            "strict": _evaluate(rows, nested_predictions, markers, 0.0),
            "primary": _evaluate(
                rows, nested_predictions, markers, PADDING, inventory=True
            ),
        },
        "fixedVariantOuterResults": fixed_results,
        "outerFolds": outer_folds,
        "fullDevelopmentSelection": {
            "selected": selected_full,
            "leaderboard": full_candidates,
        },
        "sources": model_payload["sources"],
        "limitations": [
            "All 11 recordings are opened development rather than an untouched test split.",
            "Pairwise variants and strength were compared on this opened scope.",
            "The fixed candidate union still leaves four human events outside its +/-4-second universe.",
            "This is a pairwise logistic AUC surrogate, not the PESG optimizer from the cited deep-AUC work.",
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
                "selection": {
                    key: value
                    for key, value in model["selection"].items()
                    if key != "metrics"
                },
                "nested": {
                    "rowAveragePrecision": evaluation["nestedVariantSelection"][
                        "rowAveragePrecision"
                    ],
                    "strict": {
                        key: value
                        for key, value in evaluation["nestedVariantSelection"]["strict"].items()
                        if key != "byRecording"
                    },
                    "primary": {
                        key: value
                        for key, value in evaluation["nestedVariantSelection"]["primary"].items()
                        if key != "byRecording"
                    },
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
