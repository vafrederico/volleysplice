#!/usr/bin/env python3
"""Evaluate a recording-level reliability head over the side-switch winner."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_full_union_ranker import add_derived_features, decode_ranked_candidates
from analysis.side_switch_recording_reliability import (
    RELIABILITY_FEATURE_NAMES,
    adjust_scores,
    fit_reliability_head,
    optimal_threshold_offset,
    recording_summary,
    summary_matrix,
)
from analysis.side_switch_v3 import average_precision
from analysis.side_switch_v6 import matrix_for
from analysis.side_switch_winner_experiment import (
    DECODER,
    FEATURE_NAMES,
    PADDING,
    candidate_labels,
    crossfit_scores,
    evaluate,
    events,
    fit_promoted_head,
    metric_rank,
    select_threshold,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPORTS / "side-switch-full-union-v5-state-features-v1.json"
DEFAULT_FULL_AUDIT = REPORTS / "side-switch-full-video-marker-evaluation-2026-08-21-r2.json"
DEFAULT_CONTROL = REPORTS / "side-switch-hard-negative-mining-v1-evaluation.json"
DEFAULT_WINNER = REPOSITORY / "data/side-switch-current-research-winner-v1.json"
DEFAULT_MODEL = ROOT / "models/side-switch-recording-reliability-v1/model.json"
DEFAULT_EVALUATION = REPORTS / "side-switch-recording-reliability-v1-evaluation.json"
EXPECTED_SHA256 = {
    "features": "9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551",
    "fullAudit": "142d6617aed7f20b51cdcdfc9b3e61c8c76beaf79a31fc2d7ee54fb0fdf27b89",
    "control": "e67088b36d177d68c24587efcb186eab4201b5294532be1fdf978ef13aaacc4b",
    "winner": "163c7844267bc48410e89f86bd5cbf0a58b3145c1ec691932e4e3374dae7286c",
}


@dataclass(frozen=True)
class Variant:
    identifier: str
    strength: float
    ridge: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "offsetStrength": self.strength,
            "ridge": self.ridge,
        }


VARIANTS = (
    Variant("pointwise-control", 0.0, 1.0),
    Variant("reliability-half-ridge1", 0.5, 1.0),
    Variant("reliability-one-ridge1", 1.0, 1.0),
    Variant("reliability-half-ridge4", 0.5, 4.0),
    Variant("reliability-one-ridge4", 1.0, 4.0),
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


def _local(
    rows: Sequence[Mapping[str, Any]], scores: np.ndarray, recording_id: str
) -> tuple[list[Mapping[str, Any]], np.ndarray]:
    indexes = [
        index for index, row in enumerate(rows) if str(row["recordingId"]) == recording_id
    ]
    return [rows[index] for index in indexes], scores[indexes]


def _reliability_training_data(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    markers: Mapping[str, Any],
    base_threshold: float,
) -> tuple[list[str], list[dict[str, float]], np.ndarray]:
    recording_ids = sorted(markers)
    summaries = []
    targets = []
    for recording_id in recording_ids:
        local_rows, local_scores = _local(rows, scores, recording_id)
        summaries.append(recording_summary(local_rows, local_scores, base_threshold))
        targets.append(
            optimal_threshold_offset(
                local_rows,
                local_scores,
                {recording_id: markers[recording_id]},
                base_threshold,
            )
        )
    return recording_ids, summaries, np.asarray(targets, dtype=np.float64)


def _loo_offsets(
    recording_ids: Sequence[str],
    summaries: Sequence[Mapping[str, float]],
    targets: np.ndarray,
    ridge: float,
) -> dict[str, float]:
    offsets: dict[str, float] = {}
    for held_index, recording_id in enumerate(recording_ids):
        fit_indexes = [index for index in range(len(recording_ids)) if index != held_index]
        head = fit_reliability_head(
            [summaries[index] for index in fit_indexes], targets[fit_indexes], ridge
        )
        offsets[recording_id] = float(
            head.predict(summary_matrix([summaries[held_index]]))[0]
        )
    return offsets


def _variant_rank(value: Mapping[str, Any]) -> tuple[float, ...]:
    variant = value["variant"]
    return (
        *metric_rank(value["metrics"], float(value["baseThreshold"]))[:-1],
        -float(variant["offsetStrength"]),
        -float(variant["ridge"]),
    )


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
        raise FileExistsError("refusing to overwrite reliability artifacts")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"recording-reliability source identity changed: {hashes}")
    features = _load(paths["features"])
    full_audit = _load(paths["fullAudit"])
    control = _load(paths["control"])
    winner = _load(paths["winner"])
    if winner.get("winner", {}).get("id") != "side-switch-hard-negative-mining-v1/union34-top2-x2":
        raise ValueError("reliability control is not the promoted winner")
    rows = [add_derived_features(row) for row in features["rows"]]
    recording_ids = tuple(features["scope"]["recordingIds"])
    markers = {
        recording_id: [
            {"time": float(value)}
            for value in full_audit["scope"]["humanEventsByRecording"][recording_id]
        ]
        for recording_id in recording_ids
    }
    labels = candidate_labels(rows, markers)
    if sum(labels.values()) != 46:
        raise ValueError("reliability label universe changed")

    fixed_scores = {
        variant.identifier: np.full(len(rows), np.nan) for variant in VARIANTS
    }
    fixed_predictions = {
        variant.identifier: np.zeros(len(rows), dtype=bool) for variant in VARIANTS
    }
    nested_scores = np.full(len(rows), np.nan)
    nested_predictions = np.zeros(len(rows), dtype=bool)
    outer_folds: list[dict[str, Any]] = []
    for fold_number, held_id in enumerate(recording_ids, 1):
        print(f"[{fold_number}/{len(recording_ids)}] reliability held={held_id}", flush=True)
        fit_indexes = [
            index for index, row in enumerate(rows) if str(row["recordingId"]) != held_id
        ]
        held_indexes = [
            index for index, row in enumerate(rows) if str(row["recordingId"]) == held_id
        ]
        fit_rows = [rows[index] for index in fit_indexes]
        held_rows = [rows[index] for index in held_indexes]
        fit_markers = {key: value for key, value in markers.items() if key != held_id}
        fit_scores = crossfit_scores(fit_rows, labels)
        base_selection = select_threshold(fit_rows, fit_scores, fit_markers)
        base_threshold = float(base_selection["threshold"])
        held_model, mining_audit = fit_promoted_head(events(fit_rows, labels))
        held_scores = held_model.predict_proba(
            matrix_for(events(held_rows, labels), FEATURE_NAMES)
        )
        train_ids, summaries, targets = _reliability_training_data(
            fit_rows, fit_scores, fit_markers, base_threshold
        )
        held_summary = recording_summary(held_rows, held_scores, base_threshold)
        selections: list[dict[str, Any]] = []
        for variant in VARIANTS:
            if variant.strength == 0:
                fit_offsets = {recording_id: 0.0 for recording_id in train_ids}
                held_offset = 0.0
                head = None
            else:
                fit_offsets = _loo_offsets(train_ids, summaries, targets, variant.ridge)
                head = fit_reliability_head(summaries, targets, variant.ridge)
                held_offset = float(head.predict(summary_matrix([held_summary]))[0])
            adjusted_fit = adjust_scores(
                fit_rows, fit_scores, fit_offsets, variant.strength
            )
            adjusted_held = adjust_scores(
                held_rows,
                held_scores,
                {held_id: held_offset},
                variant.strength,
            )
            fit_predictions = decode_ranked_candidates(
                fit_rows, adjusted_fit, base_threshold, DECODER
            )
            held_predictions = decode_ranked_candidates(
                held_rows, adjusted_held, base_threshold, DECODER
            )
            fixed_scores[variant.identifier][held_indexes] = adjusted_held
            fixed_predictions[variant.identifier][held_indexes] = held_predictions
            selections.append(
                {
                    "variant": variant.to_dict(),
                    "baseThreshold": base_threshold,
                    "metrics": evaluate(fit_rows, fit_predictions, fit_markers, PADDING),
                    "heldPredictedOffset": held_offset,
                    "fitTargetOffsets": dict(zip(train_ids, targets.tolist(), strict=True)),
                    "fitLooPredictedOffsets": fit_offsets,
                    "reliabilityHead": head.to_dict() if head is not None else None,
                    "miningAudit": mining_audit,
                }
            )
        selections.sort(key=_variant_rank, reverse=True)
        selected = selections[0]
        selected_id = str(selected["variant"]["id"])
        nested_scores[held_indexes] = fixed_scores[selected_id][held_indexes]
        nested_predictions[held_indexes] = fixed_predictions[selected_id][held_indexes]
        outer_folds.append(
            {
                "heldRecordingId": held_id,
                "selected": selected,
                "leaderboard": selections,
            }
        )
    if not np.isfinite(nested_scores).all() or any(
        not np.isfinite(scores).all() for scores in fixed_scores.values()
    ):
        raise ValueError("reliability outer evaluation left rows unscored")

    labels_array = np.asarray([labels[str(row["eventId"])] for row in rows])
    fixed_results: dict[str, Any] = {}
    for variant in VARIANTS:
        fixed_results[variant.identifier] = {
            "variant": variant.to_dict(),
            "rowAveragePrecision": average_precision(
                labels_array, fixed_scores[variant.identifier]
            ),
            "strict": evaluate(rows, fixed_predictions[variant.identifier], markers, 0.0),
            "primary": evaluate(
                rows,
                fixed_predictions[variant.identifier],
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
            raise ValueError(f"reliability control failed {key} metric parity")

    full_scores = crossfit_scores(rows, labels)
    full_base = select_threshold(rows, full_scores, markers)
    full_threshold = float(full_base["threshold"])
    train_ids, summaries, targets = _reliability_training_data(
        rows, full_scores, markers, full_threshold
    )
    full_candidates: list[dict[str, Any]] = []
    for variant in VARIANTS:
        offsets = (
            {recording_id: 0.0 for recording_id in train_ids}
            if variant.strength == 0
            else _loo_offsets(train_ids, summaries, targets, variant.ridge)
        )
        adjusted = adjust_scores(rows, full_scores, offsets, variant.strength)
        predictions = decode_ranked_candidates(rows, adjusted, full_threshold, DECODER)
        full_candidates.append(
            {
                "variant": variant.to_dict(),
                "baseThreshold": full_threshold,
                "metrics": evaluate(rows, predictions, markers, PADDING),
                "rowAveragePrecision": average_precision(labels_array, adjusted),
                "looPredictedOffsets": offsets,
            }
        )
    full_candidates.sort(key=_variant_rank, reverse=True)
    selected_full = full_candidates[0]
    selected_variant = next(
        variant
        for variant in VARIANTS
        if variant.identifier == selected_full["variant"]["id"]
    )
    final_classifier, mining_audit = fit_promoted_head(events(rows, labels))
    final_head = (
        None
        if selected_variant.strength == 0
        else fit_reliability_head(summaries, targets, selected_variant.ridge)
    )
    created_at = datetime.now(UTC).isoformat()
    model_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-recording-reliability-v1",
        "createdAt": created_at,
        "status": "research-only-opened-development",
        "classifier": final_classifier.to_dict(),
        "hardNegativeMiningAudit": mining_audit,
        "threshold": full_threshold,
        "decoder": DECODER.to_dict(),
        "reliability": {
            **selected_variant.to_dict(),
            "featureNames": list(RELIABILITY_FEATURE_NAMES),
            "head": final_head.to_dict() if final_head is not None else None,
            "fitTargetOffsets": dict(zip(train_ids, targets.tolist(), strict=True)),
        },
        "selection": selected_full,
        "sources": {
            name: {"path": str(paths[name]), "sha256": hashes[name]} for name in paths
        },
    }
    evaluation_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-recording-reliability-evaluation-v1",
        "createdAt": created_at,
        "scope": {
            **features["scope"],
            "humanMarkers": 50,
            "positiveCandidateLabels": int(np.sum(labels_array)),
            "status": "opened-development-only",
        },
        "protocol": {
            "outer": "leave one recording out from reliability variant and threshold selection",
            "base": "promoted union34/top-2/2x hard-negative head and decoder",
            "target": "per-video optimal threshold-logit offset from inner-crossfit base scores",
            "reliabilityFit": "ridge head with recording LOO predictions inside every outer fit",
            "features": list(RELIABILITY_FEATURE_NAMES),
            "variants": [variant.to_dict() for variant in VARIANTS],
        },
        "promotedControl": source_control,
        "nestedVariantSelection": {
            "rowAveragePrecision": average_precision(labels_array, nested_scores),
            "strict": evaluate(rows, nested_predictions, markers, 0.0),
            "primary": evaluate(
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
            "Only ten recording targets train each outer reliability head.",
            "The retained artifact has camera/alignment/palette quality proxies but no direct blur scalar.",
            "Reliability offsets can suppress false proposals but cannot recover four events outside the candidate union.",
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
                "selection": model["selection"],
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
