#!/usr/bin/env python3
"""Evaluate a nested soft penalty for weak internal side-switch peak candidates."""

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
from analysis.side_switch_full_union_ranker import (
    DERIVED_FEATURE_NAMES,
    UnionDecoderSettings,
    add_derived_features,
    decode_ranked_candidates,
    fit_weighted_logistic,
    penalize_internal_candidates,
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
DEFAULT_CONTROL = REPORTS / "side-switch-full-union-calibrated-v1-evaluation.json"
DEFAULT_MODEL = ROOT / "models/side-switch-internal-peak-penalty-v1/model.json"
DEFAULT_EVALUATION = REPORTS / "side-switch-internal-peak-penalty-v1-evaluation.json"
EXPECTED_SHA256 = {
    "features": "9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551",
    "fullAudit": "142d6617aed7f20b51cdcdfc9b3e61c8c76beaf79a31fc2d7ee54fb0fdf27b89",
    "control": "fd8ca4a8695e289cc53ec58c2042819c466bf2bbf13f0b573ac87e891dd6343f",
}
PRIMARY = (*VISUAL_FEATURE_NAMES, *STATE_GATE_FEATURE_NAMES)
FEATURE_GROUPS = {"primary32": PRIMARY, "union34": (*PRIMARY, *DERIVED_FEATURE_NAMES)}
PENALTIES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
DECODER = UnionDecoderSettings(
    minimum_index_separation=2,
    free_predictions_per_recording=6,
    count_penalty_logit=0.5,
)
L2 = 0.1
CLASS_BALANCE_EXPONENT = 0.5
PADDING = 4.0


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


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
    labels = {str(row["eventId"]): 0 for row in rows}
    for recording_id, truth in markers.items():
        local = [row for row in rows if str(row["recordingId"]) == recording_id]
        match = monotonic_interval_match([_proposal(row) for row in local], truth, PADDING)
        for pair in match.pairs:
            labels[str(local[pair.proposal_index]["eventId"])] = 1
    return labels


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


def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    predictions: np.ndarray,
    markers: Mapping[str, Any],
    padding: float,
) -> dict[str, Any]:
    per_video: dict[str, Any] = {}
    total_internal = 0
    matched_internal = 0
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
        local_internal = sum(row["kind"] == "internal-dead-state-peak" for row in proposals)
        local_matched_internal = sum(
            proposals[pair.proposal_index]["kind"] == "internal-dead-state-peak"
            for pair in match.pairs
        )
        total_internal += local_internal
        matched_internal += local_matched_internal
        per_video[recording_id] = {
            "humanEvents": len(truth),
            "proposals": len(proposals),
            "internalProposals": local_internal,
            "internalTruePositives": local_matched_internal,
            **counts,
        }
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
        "internalProposals": total_internal,
        "internalTruePositives": matched_internal,
        "macroPerVideoPrecision": float(
            np.mean([float(value["precision"] or 0.0) for value in per_video.values()])
        ),
        "macroPerVideoRecall": float(
            np.mean([float(value["recall"] or 0.0) for value in per_video.values()])
        ),
        "byRecording": per_video,
    }


def _crossfit(
    rows: Sequence[Mapping[str, Any]], labels: Mapping[str, int], names: Sequence[str]
) -> np.ndarray:
    events = _events(rows, labels)
    result = np.full(len(rows), np.nan)
    for held_id in sorted({str(row["recordingId"]) for row in rows}):
        fit = [index for index, row in enumerate(rows) if str(row["recordingId"]) != held_id]
        held = [index for index, row in enumerate(rows) if str(row["recordingId"]) == held_id]
        model = fit_weighted_logistic(
            [events[index] for index in fit], L2, names, CLASS_BALANCE_EXPONENT
        )
        result[held] = model.predict_proba(
            matrix_for([events[index] for index in held], names)
        )
    return result


def _thresholds(scores: np.ndarray) -> tuple[float, ...]:
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, 65))
    above = min(1.0, math.nextafter(float(np.max(scores)), math.inf))
    return tuple(sorted({above, *(float(value) for value in quantiles)}, reverse=True))


def _rank(value: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = value["metrics"]
    return (
        float(metrics["f1"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        -float(metrics["proposals"]),
        -float(value["internalPenaltyLogit"]),
        -float(len(value["featureNames"])),
        float(value["threshold"]),
    )


def _select(
    rows: Sequence[Mapping[str, Any]],
    raw_by_group: Mapping[str, np.ndarray],
    markers: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for group, raw in raw_by_group.items():
        for penalty in PENALTIES:
            scores = penalize_internal_candidates(rows, raw, penalty)
            best: dict[str, Any] | None = None
            for threshold in _thresholds(scores):
                predictions = decode_ranked_candidates(rows, scores, threshold, DECODER)
                value = {
                    "featureGroup": group,
                    "featureNames": list(FEATURE_GROUPS[group]),
                    "internalPenaltyLogit": penalty,
                    "threshold": threshold,
                    "metrics": _evaluate(rows, predictions, markers, PADDING),
                }
                if best is None or _rank(value) > _rank(best):
                    best = value
            if best is None:
                raise AssertionError("internal penalty selection failed")
            candidates.append(best)
    candidates.sort(key=_rank, reverse=True)
    return candidates


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "features": args.features.expanduser().resolve(),
        "fullAudit": args.full_audit.expanduser().resolve(),
        "control": args.control.expanduser().resolve(),
    }
    model_path = args.model.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    if model_path.exists() or evaluation_path.exists():
        raise FileExistsError("refusing to overwrite internal-penalty artifacts")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"internal-penalty source identity changed: {hashes}")
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
    label_vector = np.asarray([labels[str(row["eventId"])] for row in rows])
    nested_scores = np.full(len(rows), np.nan)
    nested_predictions = np.zeros(len(rows), dtype=bool)
    no_penalty_predictions = np.zeros(len(rows), dtype=bool)
    outer_folds: list[dict[str, Any]] = []
    for fold_number, held_id in enumerate(recording_ids, 1):
        print(f"[{fold_number}/{len(recording_ids)}] internal penalty held={held_id}", flush=True)
        fit_indexes = [index for index, row in enumerate(rows) if str(row["recordingId"]) != held_id]
        held_indexes = [index for index, row in enumerate(rows) if str(row["recordingId"]) == held_id]
        fit_rows = [rows[index] for index in fit_indexes]
        held_rows = [rows[index] for index in held_indexes]
        fit_markers = {key: value for key, value in markers.items() if key != held_id}
        fit_events = _events(fit_rows, labels)
        held_events = _events(held_rows, labels)
        fit_raw: dict[str, np.ndarray] = {}
        held_raw: dict[str, np.ndarray] = {}
        for group, names in FEATURE_GROUPS.items():
            fit_raw[group] = _crossfit(fit_rows, labels, names)
            model = fit_weighted_logistic(
                fit_events, L2, names, CLASS_BALANCE_EXPONENT
            )
            held_raw[group] = model.predict_proba(matrix_for(held_events, names))
        leaderboard = _select(fit_rows, fit_raw, fit_markers)
        selected = leaderboard[0]
        no_penalty = next(
            value
            for value in leaderboard
            if float(value["internalPenaltyLogit"]) == 0.0
        )
        held_scores = penalize_internal_candidates(
            held_rows,
            held_raw[str(selected["featureGroup"])],
            float(selected["internalPenaltyLogit"]),
        )
        held_predictions = decode_ranked_candidates(
            held_rows, held_scores, float(selected["threshold"]), DECODER
        )
        nested_scores[held_indexes] = held_scores
        nested_predictions[held_indexes] = held_predictions
        no_penalty_scores = held_raw[str(no_penalty["featureGroup"])]
        no_penalty_predictions[held_indexes] = decode_ranked_candidates(
            held_rows,
            no_penalty_scores,
            float(no_penalty["threshold"]),
            DECODER,
        )
        outer_folds.append(
            {
                "heldRecordingId": held_id,
                "selected": selected,
                "matchedNoPenaltyControl": no_penalty,
                "leaderboard": leaderboard,
            }
        )
    if not np.isfinite(nested_scores).all():
        raise ValueError("nested internal penalty left rows unscored")

    full_raw = {
        group: _crossfit(rows, labels, names) for group, names in FEATURE_GROUPS.items()
    }
    full_leaderboard = _select(rows, full_raw, markers)
    selected_full = full_leaderboard[0]
    selected_group = str(selected_full["featureGroup"])
    full_scores = penalize_internal_candidates(
        rows, full_raw[selected_group], float(selected_full["internalPenaltyLogit"])
    )
    selected_full["rowAveragePrecision"] = average_precision(label_vector, full_scores)
    final_model = fit_weighted_logistic(
        _events(rows, labels),
        L2,
        FEATURE_GROUPS[selected_group],
        CLASS_BALANCE_EXPONENT,
    )
    created_at = datetime.now(UTC).isoformat()
    model_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-internal-peak-penalty-v1",
        "createdAt": created_at,
        "status": "research-only-opened-development",
        "classifier": final_model.to_dict(),
        "classBalanceExponent": CLASS_BALANCE_EXPONENT,
        "internalPenaltyLogit": selected_full["internalPenaltyLogit"],
        "threshold": selected_full["threshold"],
        "decoder": DECODER.to_dict(),
        "selection": selected_full,
        "sources": {
            name: {"path": str(paths[name]), "sha256": hashes[name]}
            for name in paths
        },
    }
    evaluation_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-internal-peak-penalty-evaluation-v1",
        "createdAt": created_at,
        "scope": {**features["scope"], "humanMarkers": 50, "status": "opened-development-only"},
        "protocol": {
            "outer": "leave one recording out from feature-view, penalty, and threshold selection",
            "inner": "grouped leave-one-recording-out probabilities on outer-fit recordings",
            "classBalanceExponent": CLASS_BALANCE_EXPONENT,
            "l2": L2,
            "penaltyGridLogit": list(PENALTIES),
            "decoder": DECODER.to_dict(),
            "hardInternalGate": False,
        },
        "control": control["nestedVariantSelection"],
        "nestedNoPenaltyControl": {
            "strict": _evaluate(rows, no_penalty_predictions, markers, 0.0),
            "primary": _evaluate(rows, no_penalty_predictions, markers, PADDING),
        },
        "nested": {
            "rowAveragePrecision": average_precision(label_vector, nested_scores),
            "strict": _evaluate(rows, nested_predictions, markers, 0.0),
            "primary": _evaluate(rows, nested_predictions, markers, PADDING),
        },
        "outerFolds": outer_folds,
        "fullDevelopmentSelection": {
            "selected": selected_full,
            "leaderboard": full_leaderboard,
        },
        "sources": model_payload["sources"],
        "limitations": [
            "Only three of the 46 positive candidate labels are internal peaks.",
            "All recordings are opened development rather than an untouched test split.",
            "A type penalty can suppress false internal candidates but cannot create evidence for the four events outside the union.",
        ],
    }
    model_path.parent.mkdir(parents=True, exist_ok=False)
    atomic_write_text(model_path, json.dumps(model_payload, indent=2, allow_nan=False) + "\n")
    atomic_write_text(evaluation_path, json.dumps(evaluation_payload, indent=2, allow_nan=False) + "\n")
    return model_payload, evaluation_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--full-audit", type=Path, default=DEFAULT_FULL_AUDIT)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument(
        "--enforce-source-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    model, evaluation = run(_parser().parse_args())
    print(
        json.dumps(
            {"selection": model["selection"], "nested": evaluation["nested"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
