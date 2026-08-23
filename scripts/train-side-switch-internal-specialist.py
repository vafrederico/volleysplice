#!/usr/bin/env python3
"""Evaluate a separate internal-candidate representation and side-switch head."""

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
from analysis.side_switch_internal_specialist import (
    FEATURE_GROUPS,
    crossfit_internal_scores,
    enrich_internal_candidates,
    fit_internal_head,
    select_internal_candidates,
)
from analysis.side_switch_v3 import V3Event, average_precision
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
    thresholds,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_BASE_FEATURES = REPORTS / "side-switch-full-union-v5-state-features-v1.json"
DEFAULT_EXPANDED_FEATURES = (
    REPORTS / "side-switch-full-union-expanded-v5-state-features-v1.json"
)
DEFAULT_FULL_AUDIT = REPORTS / "side-switch-full-video-marker-evaluation-2026-08-21-r2.json"
DEFAULT_CONTROL = REPORTS / "side-switch-hard-negative-mining-v1-evaluation.json"
DEFAULT_WINNER = REPOSITORY / "data/side-switch-current-research-winner-v1.json"
DEFAULT_MODEL = ROOT / "models/side-switch-internal-specialist-v1/model.json"
DEFAULT_EVALUATION = REPORTS / "side-switch-internal-specialist-v1-evaluation.json"
EXPECTED_SHA256 = {
    "baseFeatures": "9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551",
    "expandedFeatures": "fe6563d083e49911c1d1c6f79a8c6dae4326774bf40a0041e157fcb2610fc919",
    "fullAudit": "142d6617aed7f20b51cdcdfc9b3e61c8c76beaf79a31fc2d7ee54fb0fdf27b89",
    "control": "e67088b36d177d68c24587efcb186eab4201b5294532be1fdf978ef13aaacc4b",
    "winner": "163c7844267bc48410e89f86bd5cbf0a58b3145c1ec691932e4e3374dae7286c",
}


@dataclass(frozen=True)
class Variant:
    identifier: str
    policy: str
    feature_group: str | None = None
    l2: float | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return () if self.feature_group is None else FEATURE_GROUPS[self.feature_group]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "policy": self.policy,
            "featureGroup": self.feature_group,
            "featureNames": list(self.feature_names),
            "l2": self.l2,
        }


VARIANTS = (
    Variant("promoted-control", "promoted-control"),
    Variant("boundary-only", "boundary-only"),
    *(
        Variant(f"internal-{group}-l2-{l2:g}", "internal-specialist", group, l2)
        for group in FEATURE_GROUPS
        for l2 in (0.1, 1.0)
    ),
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


def _master_rows(
    base_rows: Sequence[Mapping[str, Any]], expanded_rows: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    values = {str(row["eventId"]): row for row in expanded_rows}
    values.update(
        {
            str(row["eventId"]): row
            for row in base_rows
            if str(row["eventId"]) not in values
        }
    )
    return sorted(
        values.values(),
        key=lambda row: (
            str(row["recordingId"]),
            float(row["transitionTime"]),
            str(row["eventId"]),
        ),
    )


def _map_predictions(
    source_rows: Sequence[Mapping[str, Any]],
    source_predictions: np.ndarray,
    target_index: Mapping[str, int],
    target: np.ndarray,
    *,
    boundaries_only: bool = False,
) -> None:
    for row, selected in zip(source_rows, source_predictions, strict=True):
        if not selected or (
            boundaries_only and str(row["kind"]) != "adjacent-rally-boundary"
        ):
            continue
        target[target_index[str(row["eventId"])]] = True


def _map_internal_predictions(
    rows: Sequence[Mapping[str, Any]],
    predictions: np.ndarray,
    target_index: Mapping[str, int],
    target: np.ndarray,
) -> None:
    for row, selected in zip(rows, predictions, strict=True):
        if selected:
            target[target_index[str(row["eventId"])]] = True


def _subset(
    rows: Sequence[Mapping[str, Any]], recording_ids: set[str]
) -> tuple[list[Mapping[str, Any]], list[int]]:
    indexes = [
        index for index, row in enumerate(rows) if str(row["recordingId"]) in recording_ids
    ]
    return [rows[index] for index in indexes], indexes


def _variant_rank(value: Mapping[str, Any]) -> tuple[float, ...]:
    variant = value["variant"]
    ap = value.get("internalRowAveragePrecision")
    raw_threshold = value.get("internalThreshold")
    threshold = 1.0 if raw_threshold is None else float(raw_threshold)
    policy_rank = {"promoted-control": 2.0, "boundary-only": 1.0}.get(
        str(variant["policy"]), 0.0
    )
    return (
        *metric_rank(value["metrics"], threshold)[:-1],
        float(ap) if ap is not None else 0.0,
        policy_rank,
        -float(variant["l2"] or 0.0),
    )


def _select_internal_threshold(
    master_rows: Sequence[Mapping[str, Any]],
    boundary_predictions: np.ndarray,
    internal_rows: Sequence[Mapping[str, Any]],
    internal_scores: np.ndarray,
    markers: Mapping[str, Any],
) -> dict[str, Any]:
    master_index = {str(row["eventId"]): index for index, row in enumerate(master_rows)}
    boundary_rows = [
        row
        for row, selected in zip(master_rows, boundary_predictions, strict=True)
        if selected
    ]
    best: dict[str, Any] | None = None
    for threshold in thresholds(internal_scores):
        internal_predictions = select_internal_candidates(
            internal_rows, internal_scores, threshold, boundary_rows
        )
        combined = boundary_predictions.copy()
        _map_internal_predictions(
            internal_rows, internal_predictions, master_index, combined
        )
        metrics = evaluate(master_rows, combined, markers, PADDING)
        value = {"threshold": threshold, "metrics": metrics}
        if best is None or metric_rank(metrics, threshold) > metric_rank(
            best["metrics"], float(best["threshold"])
        ):
            best = value
    if best is None:
        raise AssertionError("internal threshold selection failed")
    return best


def _metric_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in ("proposals", "truePositives", "falsePositives", "falseNegatives")
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "baseFeatures": args.base_features.expanduser().resolve(),
        "expandedFeatures": args.expanded_features.expanduser().resolve(),
        "fullAudit": args.full_audit.expanduser().resolve(),
        "control": args.control.expanduser().resolve(),
        "winner": args.winner.expanduser().resolve(),
    }
    model_path = args.model.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    if model_path.exists() or evaluation_path.exists():
        raise FileExistsError("refusing to overwrite internal-specialist artifacts")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"internal-specialist source identity changed: {hashes}")
    base_features = _load(paths["baseFeatures"])
    expanded_features = _load(paths["expandedFeatures"])
    full_audit = _load(paths["fullAudit"])
    control = _load(paths["control"])
    winner = _load(paths["winner"])
    if winner.get("winner", {}).get("id") != "side-switch-hard-negative-mining-v1/union34-top2-x2":
        raise ValueError("internal-specialist control is not the promoted winner")
    base_rows = [add_derived_features(row) for row in base_features["rows"]]
    expanded_rows = [add_derived_features(row) for row in expanded_features["rows"]]
    master_rows = _master_rows(base_rows, expanded_rows)
    master_index = {str(row["eventId"]): index for index, row in enumerate(master_rows)}
    raw_internal = [
        row for row in expanded_rows if str(row["kind"]) == "internal-dead-state-peak"
    ]
    internal_rows = enrich_internal_candidates(raw_internal, expanded_rows)
    recording_ids = tuple(base_features["scope"]["recordingIds"])
    markers = {
        recording_id: [
            {"time": float(value)}
            for value in full_audit["scope"]["humanEventsByRecording"][recording_id]
        ]
        for recording_id in recording_ids
    }
    base_labels = candidate_labels(base_rows, markers)
    expanded_labels = candidate_labels(expanded_rows, markers)
    if sum(base_labels.values()) != 46 or sum(expanded_labels.values()) != 50:
        raise ValueError("internal-specialist label universe changed")
    internal_events = events(internal_rows, expanded_labels)
    if sum(event.label for event in internal_events) != 8:
        raise ValueError("internal-specialist positive count changed")

    fixed_predictions = {
        variant.identifier: np.zeros(len(master_rows), dtype=bool) for variant in VARIANTS
    }
    fixed_internal_scores = {
        variant.identifier: np.full(len(internal_rows), np.nan)
        for variant in VARIANTS
        if variant.policy == "internal-specialist"
    }
    nested_predictions = np.zeros(len(master_rows), dtype=bool)
    outer_folds: list[dict[str, Any]] = []
    for fold_number, held_id in enumerate(recording_ids, 1):
        print(f"[{fold_number}/{len(recording_ids)}] internal held={held_id}", flush=True)
        fit_ids = set(recording_ids) - {held_id}
        held_ids = {held_id}
        base_fit_rows, _ = _subset(base_rows, fit_ids)
        base_held_rows, _ = _subset(base_rows, held_ids)
        master_fit_rows, master_fit_indexes = _subset(master_rows, fit_ids)
        master_held_rows, master_held_indexes = _subset(master_rows, held_ids)
        internal_fit_rows, internal_fit_indexes = _subset(internal_rows, fit_ids)
        internal_held_rows, internal_held_indexes = _subset(internal_rows, held_ids)
        fit_markers = {key: value for key, value in markers.items() if key != held_id}

        base_fit_scores = crossfit_scores(base_fit_rows, base_labels)
        base_selection = select_threshold(base_fit_rows, base_fit_scores, fit_markers)
        base_fit_predictions = decode_ranked_candidates(
            base_fit_rows,
            base_fit_scores,
            float(base_selection["threshold"]),
            DECODER,
        )
        base_model, base_mining_audit = fit_promoted_head(events(base_fit_rows, base_labels))
        base_held_scores = base_model.predict_proba(
            matrix_for(events(base_held_rows, base_labels), FEATURE_NAMES)
        )
        base_held_predictions = decode_ranked_candidates(
            base_held_rows,
            base_held_scores,
            float(base_selection["threshold"]),
            DECODER,
        )

        local_fit_index = {
            str(row["eventId"]): index for index, row in enumerate(master_fit_rows)
        }
        local_held_index = {
            str(row["eventId"]): index for index, row in enumerate(master_held_rows)
        }
        control_fit = np.zeros(len(master_fit_rows), dtype=bool)
        control_held = np.zeros(len(master_held_rows), dtype=bool)
        boundary_fit = np.zeros(len(master_fit_rows), dtype=bool)
        boundary_held = np.zeros(len(master_held_rows), dtype=bool)
        _map_predictions(base_fit_rows, base_fit_predictions, local_fit_index, control_fit)
        _map_predictions(base_held_rows, base_held_predictions, local_held_index, control_held)
        _map_predictions(
            base_fit_rows,
            base_fit_predictions,
            local_fit_index,
            boundary_fit,
            boundaries_only=True,
        )
        _map_predictions(
            base_held_rows,
            base_held_predictions,
            local_held_index,
            boundary_held,
            boundaries_only=True,
        )

        selections = [
            {
                "variant": VARIANTS[0].to_dict(),
                "metrics": evaluate(master_fit_rows, control_fit, fit_markers, PADDING),
                "internalThreshold": None,
                "internalRowAveragePrecision": None,
                "baseThreshold": float(base_selection["threshold"]),
                "baseMiningAudit": base_mining_audit,
            },
            {
                "variant": VARIANTS[1].to_dict(),
                "metrics": evaluate(master_fit_rows, boundary_fit, fit_markers, PADDING),
                "internalThreshold": None,
                "internalRowAveragePrecision": None,
                "baseThreshold": float(base_selection["threshold"]),
                "baseMiningAudit": base_mining_audit,
            },
        ]
        held_outputs = {
            "promoted-control": control_held,
            "boundary-only": boundary_held,
        }
        fit_internal_events = events(internal_fit_rows, expanded_labels)
        held_internal_events = events(internal_held_rows, expanded_labels)
        fit_internal_labels = np.asarray([event.label for event in fit_internal_events])
        for variant in VARIANTS[2:]:
            internal_scores = crossfit_internal_scores(
                fit_internal_events, variant.feature_names, float(variant.l2)
            )
            internal_model, internal_audit = fit_internal_head(
                fit_internal_events, variant.feature_names, float(variant.l2)
            )
            held_scores = internal_model.predict_proba(
                matrix_for(held_internal_events, variant.feature_names)
            )
            selected = _select_internal_threshold(
                master_fit_rows,
                boundary_fit,
                internal_fit_rows,
                internal_scores,
                fit_markers,
            )
            boundary_held_rows = [
                row
                for row, selected_boundary in zip(
                    master_held_rows, boundary_held, strict=True
                )
                if selected_boundary
            ]
            held_internal_predictions = select_internal_candidates(
                internal_held_rows,
                held_scores,
                float(selected["threshold"]),
                boundary_held_rows,
            )
            combined_held = boundary_held.copy()
            _map_internal_predictions(
                internal_held_rows,
                held_internal_predictions,
                local_held_index,
                combined_held,
            )
            fixed_internal_scores[variant.identifier][internal_held_indexes] = held_scores
            held_outputs[variant.identifier] = combined_held
            selections.append(
                {
                    "variant": variant.to_dict(),
                    "metrics": selected["metrics"],
                    "internalThreshold": float(selected["threshold"]),
                    "internalRowAveragePrecision": average_precision(
                        fit_internal_labels, internal_scores
                    ),
                    "baseThreshold": float(base_selection["threshold"]),
                    "baseMiningAudit": base_mining_audit,
                    "internalMiningAudit": internal_audit,
                }
            )
        selections.sort(key=_variant_rank, reverse=True)
        selected = selections[0]
        selected_id = str(selected["variant"]["id"])
        for variant in VARIANTS:
            fixed_predictions[variant.identifier][master_held_indexes] = held_outputs[
                variant.identifier
            ]
        nested_predictions[master_held_indexes] = held_outputs[selected_id]
        outer_folds.append(
            {
                "heldRecordingId": held_id,
                "selected": selected,
                "leaderboard": selections,
            }
        )
    if any(
        not np.isfinite(scores).all() for scores in fixed_internal_scores.values()
    ):
        raise ValueError("internal-specialist outer evaluation left scores missing")

    fixed_results: dict[str, Any] = {}
    internal_label_array = np.asarray([event.label for event in internal_events])
    for variant in VARIANTS:
        fixed_results[variant.identifier] = {
            "variant": variant.to_dict(),
            "internalRowAveragePrecision": (
                None
                if variant.policy != "internal-specialist"
                else average_precision(
                    internal_label_array, fixed_internal_scores[variant.identifier]
                )
            ),
            "strict": evaluate(master_rows, fixed_predictions[variant.identifier], markers, 0.0),
            "primary": evaluate(
                master_rows,
                fixed_predictions[variant.identifier],
                markers,
                PADDING,
                inventory=True,
            ),
        }
    source_control = control["fixedVariantOuterResults"]["union34-top2-x2"]
    for key in ("strict", "primary"):
        if _metric_identity(fixed_results["promoted-control"][key]) != _metric_identity(
            source_control[key]
        ):
            raise ValueError(f"internal-specialist control failed {key} metric parity")

    base_scores = crossfit_scores(base_rows, base_labels)
    base_selection = select_threshold(base_rows, base_scores, markers)
    base_predictions = decode_ranked_candidates(
        base_rows, base_scores, float(base_selection["threshold"]), DECODER
    )
    control_predictions = np.zeros(len(master_rows), dtype=bool)
    boundary_predictions = np.zeros(len(master_rows), dtype=bool)
    _map_predictions(base_rows, base_predictions, master_index, control_predictions)
    _map_predictions(
        base_rows,
        base_predictions,
        master_index,
        boundary_predictions,
        boundaries_only=True,
    )
    full_candidates = [
        {
            "variant": VARIANTS[0].to_dict(),
            "metrics": evaluate(master_rows, control_predictions, markers, PADDING),
            "internalThreshold": None,
            "internalRowAveragePrecision": None,
        },
        {
            "variant": VARIANTS[1].to_dict(),
            "metrics": evaluate(master_rows, boundary_predictions, markers, PADDING),
            "internalThreshold": None,
            "internalRowAveragePrecision": None,
        },
    ]
    full_internal_outputs: dict[str, np.ndarray] = {
        "promoted-control": control_predictions,
        "boundary-only": boundary_predictions,
    }
    for variant in VARIANTS[2:]:
        scores = crossfit_internal_scores(
            internal_events, variant.feature_names, float(variant.l2)
        )
        selected = _select_internal_threshold(
            master_rows,
            boundary_predictions,
            internal_rows,
            scores,
            markers,
        )
        selected_internal = select_internal_candidates(
            internal_rows,
            scores,
            float(selected["threshold"]),
            [
                row
                for row, selected_boundary in zip(
                    master_rows, boundary_predictions, strict=True
                )
                if selected_boundary
            ],
        )
        combined = boundary_predictions.copy()
        _map_internal_predictions(internal_rows, selected_internal, master_index, combined)
        full_internal_outputs[variant.identifier] = combined
        full_candidates.append(
            {
                "variant": variant.to_dict(),
                "metrics": selected["metrics"],
                "internalThreshold": float(selected["threshold"]),
                "internalRowAveragePrecision": average_precision(
                    internal_label_array, scores
                ),
            }
        )
    full_candidates.sort(key=_variant_rank, reverse=True)
    selected_full = full_candidates[0]
    selected_variant = next(
        variant
        for variant in VARIANTS
        if variant.identifier == selected_full["variant"]["id"]
    )
    final_base_model, base_mining_audit = fit_promoted_head(events(base_rows, base_labels))
    final_internal_model = None
    final_internal_audit = None
    if selected_variant.policy == "internal-specialist":
        final_internal_model, final_internal_audit = fit_internal_head(
            internal_events,
            selected_variant.feature_names,
            float(selected_variant.l2),
        )
    created_at = datetime.now(UTC).isoformat()
    model_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-internal-specialist-v1",
        "createdAt": created_at,
        "status": "research-only-opened-development",
        "baseClassifier": final_base_model.to_dict(),
        "baseHardNegativeMiningAudit": base_mining_audit,
        "baseThreshold": float(base_selection["threshold"]),
        "baseDecoder": DECODER.to_dict(),
        "internalBranch": {
            **selected_variant.to_dict(),
            "threshold": selected_full["internalThreshold"],
            "minimumSeparationSeconds": 10.0,
            "boundaryDuplicateSeconds": 4.0,
            "classifier": (
                final_internal_model.to_dict() if final_internal_model is not None else None
            ),
            "hardNegativeMiningAudit": final_internal_audit,
        },
        "selection": selected_full,
        "sources": {
            name: {"path": str(paths[name]), "sha256": hashes[name]} for name in paths
        },
    }
    evaluation_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-internal-specialist-evaluation-v1",
        "createdAt": created_at,
        "scope": {
            "recordingIds": list(recording_ids),
            "recordings": len(recording_ids),
            "baseCandidates": len(base_rows),
            "expandedCandidates": len(expanded_rows),
            "masterCandidates": len(master_rows),
            "expandedInternalCandidates": len(internal_rows),
            "expandedInternalPositiveLabels": int(np.sum(internal_label_array)),
            "humanMarkers": 50,
            "candidateCoverage": 1.0,
            "status": "opened-development-only",
        },
        "protocol": {
            "outer": "leave one recording out from branch, feature view, L2, and threshold selection",
            "base": "exact promoted 704-candidate union34/top-2/2x head and decoder",
            "boundaryPolicy": "retain selected base boundaries; replace base internal proposals",
            "internal": "separate 228-row head with transition/range/serve/peak geometry",
            "internalNmsSeconds": 10.0,
            "boundaryDuplicateSeconds": 4.0,
            "variants": [variant.to_dict() for variant in VARIANTS],
        },
        "promotedControl": source_control,
        "nestedVariantSelection": {
            "strict": evaluate(master_rows, nested_predictions, markers, 0.0),
            "primary": evaluate(
                master_rows, nested_predictions, markers, PADDING, inventory=True
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
            "Only eight expanded internal candidates are positive, concentrated in four recordings.",
            "The expanded generator was chosen after inspecting the four original candidate misses.",
            "The representation reuses extracted windows and range geometry; it does not decode new motion/state frames.",
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
    parser.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    parser.add_argument("--expanded-features", type=Path, default=DEFAULT_EXPANDED_FEATURES)
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
