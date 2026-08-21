#!/usr/bin/env python3
"""Evaluate leakage-safe visibility/contact-aware serving-side mixtures."""

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
from analysis.serving_side_development_eval import (
    candidate_rank,
    configuration_matrix,
    group_metrics,
    labels,
    matrix,
    prediction_rows,
    tied_recording_ranks,
)
from analysis.serving_side_review_slices import rescore_reviewed_predictions
from analysis.serving_side_specialist import binary_metrics, select_threshold
from analysis.serving_side_v2 import LogisticModel, fit_logistic
from analysis.serving_side_weighted_logistic import fit_weighted_logistic


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_CENTER = ROOT / "features/serving-side-flight-v3/development.json"
DEFAULT_PATCHES = ROOT / "features/serving-side-selective-highres-v2/development.json"
DEFAULT_QUALITY = ROOT / "reports/serving-side/serving-side-quality-v1-development.json"
DEFAULT_REVIEW_SLICES = (
    ROOT / "reports/serving-side/serving-side-flight-v2-reviewed-slices-v1.json"
)
DEFAULT_OUTPUT = ROOT / "reports/serving-side/serving-side-quality-mixture-v1-development.json"
FLIGHT_CONFIGURATION = "192x108-r4c6"
PATCH_CONFIGURATION = "patch-20pct-96"
SIDE_L2 = 0.1
BLEND_ALPHAS = (0.25, 0.50, 0.75, 1.00)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    REPOSITORY_ROOT / "analysis/serving_side_development_eval.py",
    REPOSITORY_ROOT / "analysis/serving_side_review_slices.py",
    REPOSITORY_ROOT / "analysis/serving_side_weighted_logistic.py",
    Path(__file__).resolve(),
)


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _quality_target(target: str, row: Mapping[str, Any]) -> int | None:
    annotation = row.get("annotation")
    annotation = annotation if isinstance(annotation, Mapping) else {}
    if target == "server-visible":
        value = annotation.get("serverVisibility")
        if value == "visible":
            return 1
        if value in ("partial", "offscreen"):
            return 0
        return None
    if target == "contact-on-anchor":
        value = annotation.get("contactTiming")
        if value == "on-anchor":
            return 1
        if value in ("before-anchor", "after-anchor"):
            return 0
        return None
    raise ValueError(f"unknown quality target: {target}")


def _nested_quality_probabilities(
    *,
    target: str,
    outer_group: str,
    all_rows: Sequence[Mapping[str, Any]],
    all_values: np.ndarray,
    reviewed_rows: Sequence[Mapping[str, Any]],
    row_index: Mapping[str, int],
    l2: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    eligible = [
        row for row in reviewed_rows if _quality_target(target, row) is not None
    ]
    groups = sorted({str(row["sourceGroup"]) for row in all_rows})
    probabilities = np.full(len(all_rows), np.nan, dtype=np.float64)
    audits = []
    for prediction_group in groups:
        excluded = {outer_group, prediction_group}
        if prediction_group == outer_group:
            excluded = {outer_group}
        training = [
            row for row in eligible if str(row["sourceGroup"]) not in excluded
        ]
        training_indices = np.asarray(
            [row_index[str(row["rallyId"])] for row in training]
        )
        truth = np.asarray(
            [_quality_target(target, row) for row in training], dtype=np.int64
        )
        weights = np.asarray([float(row["weight"]) for row in training])
        prediction_indices = np.asarray(
            [
                index
                for index, row in enumerate(all_rows)
                if row["sourceGroup"] == prediction_group
            ]
        )
        model = fit_weighted_logistic(
            all_values[training_indices], truth, weights, l2=l2
        )
        probabilities[prediction_indices] = model.predict_proba(
            all_values[prediction_indices]
        )
        audits.append(
            {
                "predictionSourceGroup": prediction_group,
                "excludedAnnotationSourceGroups": sorted(excluded),
                "trainingReviewedRows": len(training),
                "predictionRows": len(prediction_indices),
            }
        )
    if not np.isfinite(probabilities).all():
        raise AssertionError("nested quality gating left side rows unscored")
    return probabilities, audits


def _audit(
    rows: Sequence[Mapping[str, Any]], probabilities: np.ndarray
) -> dict[str, Any]:
    truth = labels(rows)
    threshold_selection = select_threshold(truth, probabilities)
    threshold = float(threshold_selection["threshold"])
    predicted = probabilities >= threshold
    by_source_group = group_metrics(rows, predicted, "sourceGroup")
    balanced = [
        float(metrics["balancedAccuracy"])
        for metrics in by_source_group.values()
        if metrics["balancedAccuracy"] is not None
    ]
    return {
        "thresholdSelection": threshold_selection,
        "pooledMetrics": binary_metrics(truth, predicted),
        "sourceGroupMacroBalancedAccuracy": float(np.mean(balanced)),
        "worstSourceGroupBalancedAccuracy": float(np.min(balanced)),
        "bySourceGroup": by_source_group,
        "byEnvironment": group_metrics(rows, predicted, "environment"),
        "predictionDigest": hashlib.sha256(probabilities.tobytes()).hexdigest(),
    }


def _paired(
    truth: np.ndarray, candidate: np.ndarray, baseline: np.ndarray
) -> dict[str, int]:
    return {
        "bothCorrect": int(np.sum((candidate == truth) & (baseline == truth))),
        "candidateOnlyCorrect": int(
            np.sum((candidate == truth) & (baseline != truth))
        ),
        "baselineOnlyCorrect": int(
            np.sum((candidate != truth) & (baseline == truth))
        ),
        "bothWrong": int(np.sum((candidate != truth) & (baseline != truth))),
    }


def _model_dict(model: LogisticModel, names: Sequence[str]) -> dict[str, Any]:
    parameters = model.to_dict()
    return {
        "fingerprint": hashlib.sha256(
            json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "featureNames": list(names),
        "parameters": parameters,
    }


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    paths = {
        "center": args.center.resolve(),
        "patches": args.patches.resolve(),
        "quality": args.quality.resolve(),
        "reviewSlices": args.review_slices.resolve(),
    }
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite quality mixture: {output_path}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    center = _load(paths["center"])
    patches = _load(paths["patches"])
    quality = _load(paths["quality"])
    review = _load(paths["reviewSlices"])
    if (
        center.get("dataPolicy", {}).get("protectedTestIncluded") is not False
        or patches.get("dataPolicy", {}).get("protectedTestIncluded") is not False
        or quality.get("selection", {}).get("protectedTest")
        != "not loaded, scored, or used"
        or quality.get("selection", {}).get("humanLabelsAtInference") is not False
    ):
        raise ValueError("quality mixture requires inference-safe development artifacts")
    if quality.get("selection", {}).get("patchConfiguration") != PATCH_CONFIGURATION:
        raise ValueError("quality mixture patch configuration is not frozen at 20%")
    visibility_selected = quality["targets"]["server-visible"]["selectedCandidate"]
    contact_selected = quality["targets"]["contact-on-anchor"]["selectedCandidate"]
    if (
        visibility_selected["featureFamily"] != "patch-absolute"
        or float(visibility_selected["l2"]) != 1.0
        or contact_selected["featureFamily"] != "v2-rank"
        or float(contact_selected["l2"]) != 0.1
    ):
        raise ValueError("quality mixture inputs differ from the selected gates")
    raw_center_rows = center.get("rows")
    raw_patch_rows = patches.get("rows")
    reviewed_rows = review.get("reviewedPredictions")
    if not all(
        isinstance(rows, list) and rows
        for rows in (raw_center_rows, raw_patch_rows, reviewed_rows)
    ):
        raise ValueError("quality mixture parent rows are unavailable")
    rows = [row for row in raw_center_rows if isinstance(row, Mapping)]
    patch_by_id = {
        str(row["rallyId"]): row for row in raw_patch_rows if isinstance(row, Mapping)
    }
    if len(rows) != len(patch_by_id) or any(
        row.get("sourceSplit") == "test" for row in rows
    ):
        raise ValueError("quality mixture rows are incomplete or protected")
    joined = []
    for row in rows:
        rally_id = str(row["rallyId"])
        patch = patch_by_id.get(rally_id)
        if patch is None or any(
            row[field] != patch[field]
            for field in ("recordingId", "sourceGroup", "environment", "label")
        ):
            raise ValueError(f"quality mixture identities differ for {rally_id}")
        joined.append(
            {
                **row,
                "patchFeatures": patch["configurations"][PATCH_CONFIGURATION],
            }
        )
    rows = joined
    row_index = {str(row["rallyId"]): index for index, row in enumerate(rows)}

    v2_names = [str(name) for name in center["v2FeatureNames"]]
    flight_configuration = next(
        item
        for item in center["configurations"]
        if item["name"] == FLIGHT_CONFIGURATION
    )
    flight_names = [str(name) for name in flight_configuration["featureNames"]]
    patch_configuration = next(
        item
        for item in patches["configurations"]
        if item["name"] == PATCH_CONFIGURATION
    )
    patch_names = [str(name) for name in patch_configuration["featureNames"]]
    v2 = matrix(rows, "v2RecordingRankFeatures", v2_names)
    flight_rank = tied_recording_ranks(
        rows, configuration_matrix(rows, FLIGHT_CONFIGURATION, flight_names)
    )
    patch_absolute = matrix(rows, "patchFeatures", patch_names)
    baseline_values = np.column_stack((v2, flight_rank))
    baseline_names = [
        *(f"v2:{name}" for name in v2_names),
        *(f"flight:{name}" for name in flight_names),
    ]
    groups = sorted({str(row["sourceGroup"]) for row in rows})
    truth = labels(rows)
    candidate_names = [
        "fixed-flight-baseline",
        "baseline-plus-visible-probability",
        "baseline-plus-contact-probability",
        "baseline-plus-both-quality-probabilities",
        *(f"visibility-specialist-blend-{alpha:.2f}" for alpha in BLEND_ALPHAS),
    ]
    probabilities = {
        name: np.full(len(rows), np.nan, dtype=np.float64)
        for name in candidate_names
    }
    visible_probabilities = np.full(len(rows), np.nan, dtype=np.float64)
    contact_probabilities = np.full(len(rows), np.nan, dtype=np.float64)
    fold_audits = []
    for outer_group in groups:
        print(f"outer source group: {outer_group}", flush=True)
        held = np.asarray(
            [index for index, row in enumerate(rows) if row["sourceGroup"] == outer_group]
        )
        train = np.asarray(
            [index for index, row in enumerate(rows) if row["sourceGroup"] != outer_group]
        )
        nested_visible, visible_audit = _nested_quality_probabilities(
            target="server-visible",
            outer_group=outer_group,
            all_rows=rows,
            all_values=patch_absolute,
            reviewed_rows=reviewed_rows,
            row_index=row_index,
            l2=1.0,
        )
        nested_contact, contact_audit = _nested_quality_probabilities(
            target="contact-on-anchor",
            outer_group=outer_group,
            all_rows=rows,
            all_values=v2,
            reviewed_rows=reviewed_rows,
            row_index=row_index,
            l2=0.1,
        )
        visible_probabilities[held] = nested_visible[held]
        contact_probabilities[held] = nested_contact[held]
        baseline_model = fit_logistic(
            baseline_values[train], truth[train], l2=SIDE_L2
        )
        baseline_held = baseline_model.predict_proba(baseline_values[held])
        probabilities["fixed-flight-baseline"][held] = baseline_held

        append_specs = {
            "baseline-plus-visible-probability": (nested_visible,),
            "baseline-plus-contact-probability": (nested_contact,),
            "baseline-plus-both-quality-probabilities": (
                nested_visible,
                nested_contact,
            ),
        }
        for name, additions in append_specs.items():
            train_values = np.column_stack(
                (baseline_values[train], *(addition[train] for addition in additions))
            )
            held_values = np.column_stack(
                (baseline_values[held], *(addition[held] for addition in additions))
            )
            model = fit_logistic(train_values, truth[train], l2=SIDE_L2)
            probabilities[name][held] = model.predict_proba(held_values)

        visible_specialist = fit_weighted_logistic(
            v2[train], truth[train], nested_visible[train], l2=SIDE_L2
        )
        degraded_specialist = fit_weighted_logistic(
            flight_rank[train],
            truth[train],
            1.0 - nested_visible[train],
            l2=SIDE_L2,
        )
        specialist_held = (
            nested_visible[held] * visible_specialist.predict_proba(v2[held])
            + (1.0 - nested_visible[held])
            * degraded_specialist.predict_proba(flight_rank[held])
        )
        for alpha in BLEND_ALPHAS:
            name = f"visibility-specialist-blend-{alpha:.2f}"
            probabilities[name][held] = (
                (1.0 - alpha) * baseline_held + alpha * specialist_held
            )
        fold_audits.append(
            {
                "heldSourceGroup": outer_group,
                "trainRows": len(train),
                "heldRows": len(held),
                "visibleGateFits": visible_audit,
                "contactGateFits": contact_audit,
            }
        )
    if not all(np.isfinite(values).all() for values in probabilities.values()):
        raise AssertionError("nested mixture left candidate rows unscored")
    if not np.isfinite(visible_probabilities).all() or not np.isfinite(
        contact_probabilities
    ).all():
        raise AssertionError("nested mixture left held quality probabilities unscored")

    feature_counts = {
        "fixed-flight-baseline": len(baseline_names),
        "baseline-plus-visible-probability": len(baseline_names) + 1,
        "baseline-plus-contact-probability": len(baseline_names) + 1,
        "baseline-plus-both-quality-probabilities": len(baseline_names) + 2,
        **{
            f"visibility-specialist-blend-{alpha:.2f}": (
                len(v2_names) + len(flight_names) + len(patch_names)
            )
            for alpha in BLEND_ALPHAS
        },
    }
    leaderboard = [
        {
            "featureFamily": name,
            "featureCount": feature_counts[name],
            "sideL2": SIDE_L2,
            "evaluation": _audit(rows, values),
        }
        for name, values in probabilities.items()
    ]
    leaderboard.sort(key=candidate_rank, reverse=True)
    selected = leaderboard[0]
    baseline = next(
        item for item in leaderboard if item["featureFamily"] == "fixed-flight-baseline"
    )
    selected_name = str(selected["featureFamily"])
    baseline_threshold = float(
        baseline["evaluation"]["thresholdSelection"]["threshold"]
    )
    baseline_choices = probabilities["fixed-flight-baseline"] >= baseline_threshold
    review_slices_by_candidate = {}
    predictions_by_candidate = {}
    paired_by_candidate = {}
    for candidate in leaderboard:
        name = str(candidate["featureFamily"])
        threshold = float(
            candidate["evaluation"]["thresholdSelection"]["threshold"]
        )
        predictions = prediction_rows(
            rows, probabilities[name], threshold, detailed=True
        )
        rescored = rescore_reviewed_predictions(reviewed_rows, predictions)
        predictions_by_candidate[name] = predictions
        review_slices_by_candidate[name] = {
            "overall": rescored["overall"],
            "slices": rescored["slices"],
        }
        paired_by_candidate[name] = _paired(
            truth, probabilities[name] >= threshold, baseline_choices
        )

    final_visible_by_id = {
        str(row["rallyId"]): float(row["probabilityPositive"])
        for row in quality["targets"]["server-visible"]["allRowInference"]
    }
    final_contact_by_id = {
        str(row["rallyId"]): float(row["probabilityPositive"])
        for row in quality["targets"]["contact-on-anchor"]["allRowInference"]
    }
    final_visible = np.asarray(
        [final_visible_by_id[str(row["rallyId"])] for row in rows]
    )
    final_contact = np.asarray(
        [final_contact_by_id[str(row["rallyId"])] for row in rows]
    )
    selected_threshold = float(
        selected["evaluation"]["thresholdSelection"]["threshold"]
    )
    final_bundle: dict[str, Any] = {
        "featureFamily": selected_name,
        "threshold": selected_threshold,
        "trainingRows": len(rows),
        "qualityModelFingerprints": {
            target: quality["targets"][target]["finalModel"]["fingerprint"]
            for target in ("server-visible", "contact-on-anchor")
        },
    }
    if selected_name == "fixed-flight-baseline":
        model = replace(
            fit_logistic(baseline_values, truth, l2=SIDE_L2),
            threshold=selected_threshold,
        )
        final_bundle["sideModel"] = _model_dict(model, baseline_names)
    elif selected_name.startswith("baseline-plus-"):
        additions = []
        addition_names = []
        if "visible" in selected_name or "both" in selected_name:
            additions.append(final_visible)
            addition_names.append("quality:probabilityServerVisible")
        if "contact" in selected_name or "both" in selected_name:
            additions.append(final_contact)
            addition_names.append("quality:probabilityContactOnAnchor")
        values = np.column_stack((baseline_values, *additions))
        model = replace(
            fit_logistic(values, truth, l2=SIDE_L2), threshold=selected_threshold
        )
        final_bundle["sideModel"] = _model_dict(
            model, [*baseline_names, *addition_names]
        )
    else:
        alpha = float(selected_name.rsplit("-", 1)[1])
        baseline_model = fit_logistic(baseline_values, truth, l2=SIDE_L2)
        visible_model = fit_weighted_logistic(
            v2, truth, final_visible, l2=SIDE_L2
        )
        degraded_model = fit_weighted_logistic(
            flight_rank, truth, 1.0 - final_visible, l2=SIDE_L2
        )
        final_bundle.update(
            {
                "blendAlpha": alpha,
                "baselineModel": _model_dict(baseline_model, baseline_names),
                "visibleSpecialist": _model_dict(
                    visible_model, [f"v2:{name}" for name in v2_names]
                ),
                "degradedVisibilitySpecialist": _model_dict(
                    degraded_model,
                    [f"flight:{name}" for name in flight_names],
                ),
            }
        )
    if any(_sha256(path) != hashes[name] for name, path in paths.items()):
        raise RuntimeError("a frozen quality-mixture source changed during evaluation")

    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-quality-mixture-development-evaluation-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "selection": {
            "primaryMetric": "mean source-group balanced accuracy",
            "tieBreaks": [
                "pooled balanced accuracy",
                "pooled macro-F1",
                "worst source-group balanced accuracy",
                "fewer features",
            ],
            "crossValidation": (
                "outer leave-one-source-group-out side evaluation with nested quality "
                "fits excluding outer and prediction source groups"
            ),
            "protectedTest": "not loaded, scored, or used",
            "humanLabelsAtInference": False,
            "selectedCandidate": selected,
            "baselineCandidate": baseline,
            "offscreenSpecialistTrusted": False,
        },
        "candidateContract": {
            "sideL2": SIDE_L2,
            "blendAlphas": list(BLEND_ALPHAS),
            "visibleSpecialist": "v2 recording-rank body/court motion features",
            "degradedVisibilitySpecialist": (
                "192x108-r4c6 concentrated flight-motion recording ranks"
            ),
            "visibilityGate": "20% absolute regional patch features, L2 1.0",
            "contactGate": "v2 recording-rank features, L2 0.1",
        },
        "leaderboard": leaderboard,
        "pairedAgainstBaselineByCandidate": paired_by_candidate,
        "reviewedSliceEstimatesByCandidate": review_slices_by_candidate,
        "predictionsByCandidate": predictions_by_candidate,
        "nestedHeldQualityProbabilities": [
            {
                "rallyId": row["rallyId"],
                "recordingId": row["recordingId"],
                "sourceGroup": row["sourceGroup"],
                "probabilityServerVisible": float(visible),
                "probabilityContactOnAnchor": float(contact),
            }
            for row, visible, contact in zip(
                rows, visible_probabilities, contact_probabilities, strict=True
            )
        ],
        "foldAudits": fold_audits,
        "finalModelBundle": final_bundle,
        "counts": center.get("counts"),
        "limitations": {
            "offscreenFarObserved": 0,
            "qualitySignals": "weak and uneven; side guardrails determine selection",
        },
        "sources": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        }
        | {
            "implementation": [
                {
                    "path": str(path.relative_to(REPOSITORY_ROOT)),
                    "sha256": _sha256(path),
                }
                for path in IMPLEMENTATION_PATHS
            ]
        },
    }
    atomic_write_text(output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"selected {selected_name}")
    print(f"wrote {output_path}")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--center", type=Path, default=DEFAULT_CENTER)
    result.add_argument("--patches", type=Path, default=DEFAULT_PATCHES)
    result.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    result.add_argument("--review-slices", type=Path, default=DEFAULT_REVIEW_SLICES)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    evaluate(parser().parse_args())
