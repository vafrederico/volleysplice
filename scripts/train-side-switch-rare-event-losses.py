#!/usr/bin/env python3
"""Compare focal and effective-number losses for the side-switch winner."""

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
from analysis.side_switch_rare_event_losses import (
    fit_effective_number_logistic,
    fit_focal_logistic,
)
from analysis.side_switch_v3 import V3Event, average_precision
from analysis.side_switch_v6 import matrix_for
from analysis.side_switch_winner_experiment import (
    CLASS_BALANCE_EXPONENT,
    DECODER,
    FEATURE_NAMES,
    L2,
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
DEFAULT_MODEL = ROOT / "models/side-switch-rare-event-losses-v1/model.json"
DEFAULT_EVALUATION = REPORTS / "side-switch-rare-event-losses-v1-evaluation.json"
EXPECTED_SHA256 = {
    "features": "9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551",
    "fullAudit": "142d6617aed7f20b51cdcdfc9b3e61c8c76beaf79a31fc2d7ee54fb0fdf27b89",
    "control": "e67088b36d177d68c24587efcb186eab4201b5294532be1fdf978ef13aaacc4b",
    "winner": "163c7844267bc48410e89f86bd5cbf0a58b3145c1ec691932e4e3374dae7286c",
}


@dataclass(frozen=True)
class Variant:
    identifier: str
    objective: str
    parameter: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "objective": self.objective,
            "parameter": self.parameter,
        }


VARIANTS = (
    Variant("pointwise-control", "square-root-bce", 0.5),
    Variant("focal-gamma1", "focal", 1.0),
    Variant("focal-gamma2", "focal", 2.0),
    Variant("effective-beta0.9", "effective-number", 0.9),
    Variant("effective-beta0.99", "effective-number", 0.99),
    Variant("effective-beta0.999", "effective-number", 0.999),
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


def _fitter(variant: Variant):
    if variant.objective == "square-root-bce":
        return None
    if variant.objective == "focal":
        return lambda fit_events, multipliers: fit_focal_logistic(
            fit_events,
            L2,
            FEATURE_NAMES,
            CLASS_BALANCE_EXPONENT,
            variant.parameter,
            multipliers,
        )
    if variant.objective == "effective-number":
        return lambda fit_events, multipliers: fit_effective_number_logistic(
            fit_events,
            L2,
            FEATURE_NAMES,
            variant.parameter,
            multipliers,
        )
    raise ValueError(f"unsupported rare-event objective: {variant.objective}")


def _variant_rank(value: Mapping[str, Any]) -> tuple[float, ...]:
    variant = value["variant"]
    return (
        *metric_rank(value["metrics"], float(value["threshold"]))[:-1],
        float(value["rowAveragePrecision"]),
        1.0 if variant["objective"] == "square-root-bce" else 0.0,
        -float(variant["parameter"]),
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
        raise FileExistsError("refusing to overwrite rare-event-loss artifacts")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"rare-event-loss source identity changed: {hashes}")
    features = _load(paths["features"])
    full_audit = _load(paths["fullAudit"])
    control = _load(paths["control"])
    winner = _load(paths["winner"])
    if winner.get("winner", {}).get("id") != "side-switch-hard-negative-mining-v1/union34-top2-x2":
        raise ValueError("rare-event-loss control is not the promoted winner")
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
        raise ValueError("rare-event-loss label universe changed")

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
        print(f"[{fold_number}/{len(recording_ids)}] rare-loss held={held_id}", flush=True)
        fit_indexes = [
            index for index, row in enumerate(rows) if str(row["recordingId"]) != held_id
        ]
        held_indexes = [
            index for index, row in enumerate(rows) if str(row["recordingId"]) == held_id
        ]
        fit_rows = [rows[index] for index in fit_indexes]
        held_rows = [rows[index] for index in held_indexes]
        fit_markers = {key: value for key, value in markers.items() if key != held_id}
        fit_events = events(fit_rows, labels)
        held_events = events(held_rows, labels)
        selections: list[dict[str, Any]] = []
        for variant in VARIANTS:
            fitter = _fitter(variant)
            fit_scores = crossfit_scores(fit_rows, labels, fitter)
            selection = select_threshold(fit_rows, fit_scores, fit_markers)
            held_model, mining_audit = fit_promoted_head(fit_events, fitter)
            held_scores = held_model.predict_proba(matrix_for(held_events, FEATURE_NAMES))
            held_predictions = decode_ranked_candidates(
                held_rows, held_scores, float(selection["threshold"]), DECODER
            )
            fixed_scores[variant.identifier][held_indexes] = held_scores
            fixed_predictions[variant.identifier][held_indexes] = held_predictions
            selections.append(
                {
                    "variant": variant.to_dict(),
                    "threshold": float(selection["threshold"]),
                    "metrics": selection["metrics"],
                    "rowAveragePrecision": average_precision(
                        np.asarray([labels[str(row["eventId"])] for row in fit_rows]),
                        fit_scores,
                    ),
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
        raise ValueError("rare-event-loss outer evaluation left rows unscored")

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
            raise ValueError(f"rare-event-loss control failed {key} metric parity")

    full_candidates: list[dict[str, Any]] = []
    for variant in VARIANTS:
        fitter = _fitter(variant)
        scores = crossfit_scores(rows, labels, fitter)
        selection = select_threshold(rows, scores, markers)
        full_candidates.append(
            {
                "variant": variant.to_dict(),
                "threshold": float(selection["threshold"]),
                "metrics": selection["metrics"],
                "rowAveragePrecision": average_precision(labels_array, scores),
            }
        )
    full_candidates.sort(key=_variant_rank, reverse=True)
    selected_full = full_candidates[0]
    selected_variant = next(
        variant
        for variant in VARIANTS
        if variant.identifier == selected_full["variant"]["id"]
    )
    final_model, mining_audit = fit_promoted_head(
        events(rows, labels), _fitter(selected_variant)
    )
    created_at = datetime.now(UTC).isoformat()
    model_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-rare-event-losses-v1",
        "createdAt": created_at,
        "status": "research-only-opened-development",
        "classifier": final_model.to_dict(),
        "hardNegativeMiningAudit": mining_audit,
        "objective": selected_variant.to_dict(),
        "threshold": float(selected_full["threshold"]),
        "decoder": DECODER.to_dict(),
        "selection": selected_full,
        "sources": {
            name: {"path": str(paths[name]), "sha256": hashes[name]} for name in paths
        },
    }
    evaluation_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-rare-event-losses-evaluation-v1",
        "createdAt": created_at,
        "scope": {
            **features["scope"],
            "humanMarkers": 50,
            "positiveCandidateLabels": int(np.sum(labels_array)),
            "status": "opened-development-only",
        },
        "protocol": {
            "outer": "leave one recording out from objective and threshold selection",
            "base": "promoted union34/top-2/2x hard-negative model and fixed decoder",
            "focal": "exact binary focal loss optimized by deterministic BFGS",
            "effectiveNumber": "class weights (1-beta)/(1-beta^classCount)",
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
            "Objective and parameter selection are noisy with only 46 positive candidate rows.",
            "No loss can recover four markers outside the retained candidate union.",
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
