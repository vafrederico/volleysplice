#!/usr/bin/env python3
"""Evaluate predeclared selective high-resolution patch-size ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.serving_side_development_eval import (
    candidate_rank,
    configuration_matrix,
    cross_fit,
    labels,
    matrix,
    prediction_rows,
    tied_recording_ranks,
)
from analysis.serving_side_review_slices import rescore_reviewed_predictions
from analysis.serving_side_v2 import fit_logistic


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_DATASET = ROOT / "features/serving-side-selective-highres-v2/development.json"
DEFAULT_REFERENCE_V1 = (
    ROOT / "features/serving-side-selective-highres-v1/development.json"
)
DEFAULT_REVIEW_SLICES = (
    ROOT / "reports/serving-side/serving-side-flight-v2-reviewed-slices-v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports/serving-side/serving-side-selective-highres-v2-ablation-development.json"
)
FLIGHT_CONFIGURATION = "192x108-r4c6"
FIXED_L2 = 0.1
REFERENCE_CONFIGURATION = "patch-20pct-96"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    REPOSITORY_ROOT / "analysis/serving_side_development_eval.py",
    REPOSITORY_ROOT / "analysis/serving_side_review_slices.py",
    REPOSITORY_ROOT / "analysis/serving_side_selective_highres.py",
    REPOSITORY_ROOT / "analysis/serving_side_v2.py",
    REPOSITORY_ROOT / "scripts/extract-serving-side-selective-highres-features.py",
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


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    dataset_path = args.dataset.resolve()
    reference_path = args.reference_v1.resolve()
    review_path = args.review_slices.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite patch ablation: {output_path}")
    dataset_hash = _sha256(dataset_path)
    dataset = _load(dataset_path)
    if (
        dataset.get("kind")
        != "volleycut-serving-side-selective-highres-ablation-development-v1"
        or dataset.get("scope") != "development"
        or dataset.get("dataPolicy", {}).get("protectedTestIncluded") is not False
        or dataset.get("dataPolicy", {}).get("humanCorrectedAnchorsUsed") is not False
        or dataset.get("dataPolicy", {}).get("humanVisibilityUsed") is not False
        or dataset.get("dataPolicy", {}).get("limitPerRecording") is not None
    ):
        raise ValueError("patch ablation requires the complete development artifact")
    raw_rows = dataset.get("rows")
    configurations = dataset.get("configurations")
    if not isinstance(raw_rows, list) or not raw_rows or not isinstance(
        configurations, list
    ):
        raise ValueError("patch ablation rows or configurations are unavailable")
    ablation_rows = [row for row in raw_rows if isinstance(row, Mapping)]
    if len(ablation_rows) != len(raw_rows) or any(
        row.get("sourceSplit") == "test" for row in ablation_rows
    ):
        raise ValueError("patch ablation contains invalid or protected rows")
    configuration_by_name = {
        str(item["name"]): item
        for item in configurations
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    expected_names = (
        "patch-12pct-96",
        "patch-20pct-96",
        "patch-30pct-96",
    )
    if tuple(configuration_by_name) != expected_names:
        raise ValueError("patch ablation differs from the predeclared configurations")

    center_source = dataset.get("sources", {}).get("centerFlightDataset")
    if not isinstance(center_source, Mapping):
        raise ValueError("patch ablation lacks its center feature source")
    center_path = Path(str(center_source["path"])).resolve()
    if _sha256(center_path) != center_source.get("sha256"):
        raise ValueError("center feature bank changed since patch extraction")
    center = _load(center_path)
    if (
        center.get("kind") != "volleycut-serving-side-flight-feature-development-v1"
        or center.get("dataPolicy", {}).get("protectedTestIncluded") is not False
        or center.get("dataPolicy", {}).get("limitPerRecording") is not None
    ):
        raise ValueError("patch ablation requires the complete development center bank")
    center_by_id = {
        str(row["rallyId"]): row
        for row in center.get("rows", [])
        if isinstance(row, Mapping) and isinstance(row.get("rallyId"), str)
    }
    if len(center_by_id) != len(ablation_rows):
        raise ValueError("center and patch feature banks differ in row count")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ablation in ablation_rows:
        rally_id = str(ablation["rallyId"])
        if rally_id in seen:
            raise ValueError(f"duplicate patch-ablation row: {rally_id}")
        seen.add(rally_id)
        center_row = center_by_id.get(rally_id)
        if center_row is None:
            raise ValueError(f"center feature bank lacks {rally_id}")
        if any(
            ablation[field] != center_row[field]
            for field in (
                "recordingId",
                "sourceGroup",
                "environment",
                "sourceSplit",
                "serveAnchor",
                "decision",
                "label",
            )
        ):
            raise ValueError(f"patch and center identities differ for {rally_id}")
        rows.append({**center_row, "patchConfigurations": ablation["configurations"]})

    reference_hash = _sha256(reference_path)
    reference = _load(reference_path)
    reference_rows = {
        str(row["rallyId"]): row
        for row in reference.get("rows", [])
        if isinstance(row, Mapping) and isinstance(row.get("rallyId"), str)
    }
    if len(reference_rows) != len(rows):
        raise ValueError("v1 reference and v2 ablation differ in row count")
    for row in rows:
        rally_id = str(row["rallyId"])
        if (
            row["patchConfigurations"][REFERENCE_CONFIGURATION]
            != reference_rows[rally_id]["selectiveHighresFeatures"]
        ):
            raise ValueError(f"20% refactor parity failed for {rally_id}")

    v2_names = center.get("v2FeatureNames")
    flight_configurations = center.get("configurations")
    flight_configuration = next(
        (
            item
            for item in flight_configurations
            if isinstance(item, Mapping)
            and item.get("name") == FLIGHT_CONFIGURATION
        ),
        None,
    ) if isinstance(flight_configurations, list) else None
    flight_names = (
        flight_configuration.get("featureNames")
        if isinstance(flight_configuration, Mapping)
        else None
    )
    patch_names = configuration_by_name[REFERENCE_CONFIGURATION].get("featureNames")
    if not all(
        isinstance(names, list) and names
        for names in (v2_names, flight_names, patch_names)
    ):
        raise ValueError("patch ablation feature contracts are unavailable")
    v2_names = [str(name) for name in v2_names]
    flight_names = [str(name) for name in flight_names]
    patch_names = [str(name) for name in patch_names]
    v2 = matrix(rows, "v2RecordingRankFeatures", v2_names)
    flight_raw = configuration_matrix(rows, FLIGHT_CONFIGURATION, flight_names)
    flight_rank = tied_recording_ranks(rows, flight_raw)
    baseline_values = np.column_stack((v2, flight_rank))
    baseline_names = [
        *(f"v2:{name}" for name in v2_names),
        *(f"flight:{name}" for name in flight_names),
    ]

    baseline_audit, baseline_probabilities = cross_fit(
        rows, baseline_values, FIXED_L2
    )
    baseline = {
        "configuration": None,
        "featureFamily": "v2-plus-fixed-flight-rank",
        "featureCount": len(baseline_names),
        "l2": FIXED_L2,
        "evaluation": baseline_audit,
    }
    candidates = []
    values_by_name: dict[str, np.ndarray] = {}
    names_by_name: dict[str, list[str]] = {}
    probabilities_by_name: dict[str, np.ndarray] = {}
    for name in expected_names:
        patch_raw = np.asarray(
            [
                [float(row["patchConfigurations"][name][feature]) for feature in patch_names]
                for row in rows
            ],
            dtype=np.float64,
        )
        if not np.isfinite(patch_raw).all():
            raise ValueError(f"configuration {name} has non-finite features")
        values = np.column_stack((baseline_values, patch_raw))
        names = [*baseline_names, *(f"selective:{feature}" for feature in patch_names)]
        print(f"evaluating {name} / absolute regional features / l2={FIXED_L2}", flush=True)
        audit, probabilities = cross_fit(rows, values, FIXED_L2)
        candidate = {
            "configuration": name,
            "sourceFractionOfShortEdge": configuration_by_name[name][
                "sourceFractionOfShortEdge"
            ],
            "normalizedPatchSize": configuration_by_name[name]["normalizedWidth"],
            "featureFamily": "v2-plus-fixed-flight-plus-selective-highres-absolute",
            "featureCount": len(names),
            "l2": FIXED_L2,
            "evaluation": audit,
        }
        candidates.append(candidate)
        values_by_name[name] = values
        names_by_name[name] = names
        probabilities_by_name[name] = probabilities
    leaderboard = sorted([baseline, *candidates], key=candidate_rank, reverse=True)
    selected = leaderboard[0]
    best_patch = max(candidates, key=candidate_rank)
    baseline_macro = float(baseline_audit["sourceGroupMacroBalancedAccuracy"])
    baseline_worst = float(baseline_audit["worstSourceGroupBalancedAccuracy"])
    comparisons = [
        {
            "configuration": candidate["configuration"],
            "macroBalancedAccuracyDelta": (
                float(candidate["evaluation"]["sourceGroupMacroBalancedAccuracy"])
                - baseline_macro
            ),
            "pooledBalancedAccuracyDelta": (
                float(candidate["evaluation"]["pooledMetrics"]["balancedAccuracy"])
                - float(baseline_audit["pooledMetrics"]["balancedAccuracy"])
            ),
            "worstSourceGroupBalancedAccuracyDelta": (
                float(candidate["evaluation"]["worstSourceGroupBalancedAccuracy"])
                - baseline_worst
            ),
        }
        for candidate in candidates
    ]
    directionally_stable = all(
        comparison["macroBalancedAccuracyDelta"] > 0
        and comparison["worstSourceGroupBalancedAccuracyDelta"] >= 0
        for comparison in comparisons
    )

    truth = labels(rows)
    baseline_threshold = float(baseline_audit["thresholdSelection"]["threshold"])
    baseline_choices = baseline_probabilities >= baseline_threshold
    review = _load(review_path)
    reviewed_rows = review.get("reviewedPredictions")
    if not isinstance(reviewed_rows, list):
        raise ValueError("review-slice artifact has no frozen sample")
    predictions_by_configuration = {}
    slices_by_configuration = {}
    paired_by_configuration = {}
    models_by_configuration = {}
    for candidate in candidates:
        name = str(candidate["configuration"])
        probabilities = probabilities_by_name[name]
        threshold = float(
            candidate["evaluation"]["thresholdSelection"]["threshold"]
        )
        predictions = prediction_rows(rows, probabilities, threshold, detailed=True)
        choices = probabilities >= threshold
        reviewed = rescore_reviewed_predictions(reviewed_rows, predictions)
        model = replace(
            fit_logistic(values_by_name[name], truth, l2=FIXED_L2),
            threshold=threshold,
        )
        parameters = model.to_dict()
        predictions_by_configuration[name] = predictions
        slices_by_configuration[name] = {
            "overall": reviewed["overall"],
            "slices": reviewed["slices"],
        }
        paired_by_configuration[name] = _paired(
            truth, choices, baseline_choices
        )
        models_by_configuration[name] = {
            "fingerprint": hashlib.sha256(
                json.dumps(
                    parameters, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "featureNames": names_by_name[name],
            "parameters": parameters,
            "trainingRows": len(rows),
        }
    if _sha256(dataset_path) != dataset_hash:
        raise RuntimeError("patch ablation artifact changed during evaluation")

    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-selective-highres-ablation-evaluation-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "predeclaration": {
            "configurationOrder": list(expected_names),
            "fixedL2": FIXED_L2,
            "fixedNormalizedPatchSize": 96,
            "onlyVariable": "source crop fraction of short edge",
            "robustnessRule": (
                "every patch size must improve source-group macro balanced accuracy "
                "and must not regress worst-source-group balanced accuracy"
            ),
        },
        "selection": {
            "primaryMetric": "mean source-group balanced accuracy",
            "tieBreaks": [
                "pooled balanced accuracy",
                "pooled macro-F1",
                "worst source-group balanced accuracy",
                "fewer features",
            ],
            "crossValidation": "leave-one-source-group-out development predictions",
            "protectedTest": "not loaded, scored, or used",
            "selectedCandidate": selected,
            "baselineCandidate": baseline,
            "bestPatchCandidate": best_patch,
            "directionallyStableAcrossPatchSizes": directionally_stable,
            "eligibleForNextStage": bool(
                directionally_stable and selected.get("configuration") is not None
            ),
        },
        "leaderboard": leaderboard,
        "comparisonsAgainstBaseline": comparisons,
        "pairedAgainstBaselineByConfiguration": paired_by_configuration,
        "reviewedSliceEstimatesByConfiguration": slices_by_configuration,
        "modelsByConfiguration": models_by_configuration,
        "predictionsByConfiguration": predictions_by_configuration,
        "counts": dataset.get("counts"),
        "sources": {
            "ablationDevelopmentDataset": {
                "path": str(dataset_path),
                "sha256": dataset_hash,
            },
            "reference20PercentDataset": {
                "path": str(reference_path),
                "sha256": reference_hash,
                "parity": "all row features exactly equal",
            },
            "centerFlightDataset": center_source,
            "reviewedSlices": {
                "path": str(review_path),
                "sha256": _sha256(review_path),
            },
            "humanLabelCorrections": dataset.get("sources", {}).get(
                "humanLabelCorrections"
            ),
            "sourceQualityExclusions": dataset.get("sources", {}).get(
                "sourceQualityExclusions"
            ),
            "implementation": [
                {
                    "path": str(path.relative_to(REPOSITORY_ROOT)),
                    "sha256": _sha256(path),
                }
                for path in IMPLEMENTATION_PATHS
            ],
        },
    }
    atomic_write_text(output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        f"best patch {best_patch['configuration']}; "
        f"directionally stable={directionally_stable}"
    )
    print(f"wrote {output_path}")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    result.add_argument("--reference-v1", type=Path, default=DEFAULT_REFERENCE_V1)
    result.add_argument("--review-slices", type=Path, default=DEFAULT_REVIEW_SLICES)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    evaluate(parser().parse_args())
