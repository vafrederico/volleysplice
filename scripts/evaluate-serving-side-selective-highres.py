#!/usr/bin/env python3
"""Evaluate selective source-resolution patch features on development data."""

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
DEFAULT_DATASET = ROOT / "features/serving-side-selective-highres-v1/development.json"
DEFAULT_REVIEW_SLICES = (
    ROOT / "reports/serving-side/serving-side-flight-v2-reviewed-slices-v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/serving-side/serving-side-selective-highres-v1-development.json"
)
CONFIGURATION = "192x108-r4c6"
L2_GRID = (0.01, 0.1, 1.0, 10.0)
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
    review_path = args.review_slices.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite selective evaluation: {output_path}")
    dataset_hash = _sha256(dataset_path)
    dataset = _load(dataset_path)
    if (
        dataset.get("kind")
        != "volleycut-serving-side-selective-highres-feature-development-v1"
        or dataset.get("scope") != "development"
        or dataset.get("dataPolicy", {}).get("protectedTestIncluded") is not False
        or dataset.get("dataPolicy", {}).get("humanCorrectedAnchorsUsed") is not False
        or dataset.get("dataPolicy", {}).get("humanVisibilityUsed") is not False
        or dataset.get("dataPolicy", {}).get("limitPerRecording") is not None
    ):
        raise ValueError("selective selection requires the complete development artifact")
    raw_selective_rows = dataset.get("rows")
    if not isinstance(raw_selective_rows, list) or not raw_selective_rows:
        raise ValueError("selective dataset has no rows")
    selective_rows = [row for row in raw_selective_rows if isinstance(row, Mapping)]
    if len(selective_rows) != len(raw_selective_rows) or any(
        row.get("sourceSplit") == "test" for row in selective_rows
    ):
        raise ValueError("selective dataset contains invalid or protected rows")

    trajectory_source = dataset.get("sources", {}).get(
        "trajectoryDevelopmentDataset"
    )
    center_source = dataset.get("sources", {}).get("centerFlightDataset")
    if not isinstance(trajectory_source, Mapping) or not isinstance(
        center_source, Mapping
    ):
        raise ValueError("selective dataset lacks frozen parent sources")
    trajectory_path = Path(str(trajectory_source["path"])).resolve()
    center_path = Path(str(center_source["path"])).resolve()
    if _sha256(trajectory_path) != trajectory_source.get("sha256"):
        raise ValueError("trajectory feature bank changed since selective extraction")
    if _sha256(center_path) != center_source.get("sha256"):
        raise ValueError("center feature bank changed since selective extraction")
    center = _load(center_path)
    if (
        center.get("kind") != "volleycut-serving-side-flight-feature-development-v1"
        or center.get("dataPolicy", {}).get("protectedTestIncluded") is not False
        or center.get("dataPolicy", {}).get("limitPerRecording") is not None
    ):
        raise ValueError("selective evaluation requires the complete center bank")
    center_by_id = {
        str(row["rallyId"]): row
        for row in center.get("rows", [])
        if isinstance(row, Mapping) and isinstance(row.get("rallyId"), str)
    }
    if len(center_by_id) != len(selective_rows):
        raise ValueError("center and selective feature banks differ in row count")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for selective in selective_rows:
        rally_id = str(selective["rallyId"])
        if rally_id in seen:
            raise ValueError(f"duplicate selective row: {rally_id}")
        seen.add(rally_id)
        center_row = center_by_id.get(rally_id)
        if center_row is None:
            raise ValueError(f"center feature bank lacks {rally_id}")
        if any(
            selective[field] != center_row[field]
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
            raise ValueError(f"selective and center identities differ for {rally_id}")
        rows.append(
            {
                **center_row,
                "selectiveHighresFeatures": selective["selectiveHighresFeatures"],
            }
        )

    selective_names = dataset.get("featureNames")
    v2_names = center.get("v2FeatureNames")
    configurations = center.get("configurations")
    configuration = next(
        (
            item
            for item in configurations
            if isinstance(item, Mapping) and item.get("name") == CONFIGURATION
        ),
        None,
    ) if isinstance(configurations, list) else None
    flight_names = (
        configuration.get("featureNames")
        if isinstance(configuration, Mapping)
        else None
    )
    if not all(
        isinstance(names, list) and names
        for names in (selective_names, v2_names, flight_names)
    ):
        raise ValueError("selective evaluation feature contracts are unavailable")
    selective_names = [str(name) for name in selective_names]
    v2_names = [str(name) for name in v2_names]
    flight_names = [str(name) for name in flight_names]
    persistent_indices = [
        index for index, name in enumerate(selective_names) if name.startswith("persistent:")
    ]
    compact_indices = [
        index for index, name in enumerate(selective_names) if name.startswith("compact:")
    ]
    if not persistent_indices or not compact_indices:
        raise ValueError("selective feature selectors are unavailable")

    v2 = matrix(rows, "v2RecordingRankFeatures", v2_names)
    flight_raw = configuration_matrix(rows, CONFIGURATION, flight_names)
    selective_raw = matrix(rows, "selectiveHighresFeatures", selective_names)
    flight_rank = tied_recording_ranks(rows, flight_raw)
    selective_rank = tied_recording_ranks(rows, selective_raw)
    baseline_values = np.column_stack((v2, flight_rank))
    v2_labels = [f"v2:{name}" for name in v2_names]
    flight_labels = [f"flight:{name}" for name in flight_names]
    selective_labels = [f"selective:{name}" for name in selective_names]
    baseline_labels = [*v2_labels, *flight_labels]
    families = {
        "v2-plus-fixed-flight-rank": (baseline_values, baseline_labels),
        "selective-absolute": (selective_raw, selective_labels),
        "selective-recording-rank": (selective_rank, selective_labels),
        "v2-plus-selective-rank": (
            np.column_stack((v2, selective_rank)),
            [*v2_labels, *selective_labels],
        ),
        "fixed-flight-plus-selective-rank": (
            np.column_stack((flight_rank, selective_rank)),
            [*flight_labels, *selective_labels],
        ),
        "v2-plus-fixed-flight-plus-persistent-highres-rank": (
            np.column_stack((baseline_values, selective_rank[:, persistent_indices])),
            [
                *baseline_labels,
                *(selective_labels[index] for index in persistent_indices),
            ],
        ),
        "v2-plus-fixed-flight-plus-compact-highres-rank": (
            np.column_stack((baseline_values, selective_rank[:, compact_indices])),
            [
                *baseline_labels,
                *(selective_labels[index] for index in compact_indices),
            ],
        ),
        "v2-plus-fixed-flight-plus-selective-highres-rank": (
            np.column_stack((baseline_values, selective_rank)),
            [*baseline_labels, *selective_labels],
        ),
        "v2-plus-fixed-flight-plus-selective-highres-absolute": (
            np.column_stack((baseline_values, selective_raw)),
            [*baseline_labels, *selective_labels],
        ),
    }

    leaderboard: list[dict[str, Any]] = []
    probabilities_by_key: dict[tuple[str, float], np.ndarray] = {}
    for family, (values, names) in families.items():
        for l2 in L2_GRID:
            print(f"evaluating {family} / l2={l2}", flush=True)
            audit, probabilities = cross_fit(rows, values, l2)
            leaderboard.append(
                {
                    "featureFamily": family,
                    "featureCount": len(names),
                    "l2": l2,
                    "evaluation": audit,
                }
            )
            probabilities_by_key[(family, l2)] = probabilities
    leaderboard.sort(key=candidate_rank, reverse=True)
    selected = leaderboard[0]
    baseline = max(
        (
            item
            for item in leaderboard
            if item["featureFamily"] == "v2-plus-fixed-flight-rank"
        ),
        key=candidate_rank,
    )
    best_selective = max(
        (
            item
            for item in leaderboard
            if "selective" in str(item["featureFamily"])
            or "highres" in str(item["featureFamily"])
        ),
        key=candidate_rank,
    )
    selected_key = (str(selected["featureFamily"]), float(selected["l2"]))
    baseline_key = (str(baseline["featureFamily"]), float(baseline["l2"]))
    best_key = (
        str(best_selective["featureFamily"]),
        float(best_selective["l2"]),
    )
    truth = labels(rows)

    def threshold_for(candidate: Mapping[str, Any]) -> float:
        return float(candidate["evaluation"]["thresholdSelection"]["threshold"])

    def model_payload(
        key: tuple[str, float], candidate: Mapping[str, Any]
    ) -> dict[str, Any]:
        values, names = families[key[0]]
        threshold = threshold_for(candidate)
        model = replace(
            fit_logistic(values, truth, l2=key[1]), threshold=threshold
        )
        parameters = model.to_dict()
        fingerprint = hashlib.sha256(
            json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        top_weights = sorted(
            (
                {
                    "feature": name,
                    "weight": float(weight),
                    "absoluteWeight": abs(float(weight)),
                }
                for name, weight in zip(names, model.weights, strict=True)
            ),
            key=lambda item: (-item["absoluteWeight"], item["feature"]),
        )[:30]
        return {
            "fingerprint": fingerprint,
            "featureFamily": key[0],
            "featureNames": names,
            "parameters": parameters,
            "topWeights": top_weights,
            "trainingRows": len(rows),
        }

    selected_probabilities = probabilities_by_key[selected_key]
    baseline_probabilities = probabilities_by_key[baseline_key]
    best_probabilities = probabilities_by_key[best_key]
    selected_predictions = prediction_rows(
        rows,
        selected_probabilities,
        threshold_for(selected),
        detailed=True,
    )
    best_predictions = prediction_rows(
        rows,
        best_probabilities,
        threshold_for(best_selective),
        detailed=True,
    )
    selected_choices = selected_probabilities >= threshold_for(selected)
    baseline_choices = baseline_probabilities >= threshold_for(baseline)
    best_choices = best_probabilities >= threshold_for(best_selective)

    review = _load(review_path)
    reviewed_rows = review.get("reviewedPredictions")
    if not isinstance(reviewed_rows, list):
        raise ValueError("review-slice artifact has no frozen reviewed sample")
    selected_review = rescore_reviewed_predictions(
        reviewed_rows, selected_predictions
    )
    best_review = rescore_reviewed_predictions(reviewed_rows, best_predictions)
    if _sha256(dataset_path) != dataset_hash:
        raise RuntimeError("selective feature artifact changed during evaluation")

    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-selective-highres-development-evaluation-v1",
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
                "leave-one-source-group-out over correction-clean development rows"
            ),
            "protectedTest": "not loaded, scored, or used",
            "humanVisibility": "reported after selection; never used as an input",
            "selectedCandidate": selected,
            "fixedFlightBaselineCandidate": baseline,
            "bestSelectiveCandidate": best_selective,
        },
        "leaderboard": leaderboard,
        "pairedSelectedAgainstBaseline": _paired(
            truth, selected_choices, baseline_choices
        ),
        "pairedBestSelectiveAgainstBaseline": _paired(
            truth, best_choices, baseline_choices
        ),
        "reviewedSliceEstimate": {
            "overall": selected_review["overall"],
            "slices": selected_review["slices"],
        },
        "bestSelectiveReviewedSliceEstimate": {
            "overall": best_review["overall"],
            "slices": best_review["slices"],
        },
        "finalModel": model_payload(selected_key, selected),
        "bestSelectiveModel": model_payload(best_key, best_selective),
        "selectedPredictions": selected_predictions,
        "bestSelectivePredictions": best_predictions,
        "counts": dataset.get("counts"),
        "sources": {
            "selectiveDevelopmentDataset": {
                "path": str(dataset_path),
                "sha256": dataset_hash,
            },
            "trajectoryDevelopmentDataset": trajectory_source,
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
    print(f"selected {selected_key[0]} / l2={selected_key[1]}")
    print(f"best selective {best_key[0]} / l2={best_key[1]}")
    print(f"wrote {output_path}")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    result.add_argument("--review-slices", type=Path, default=DEFAULT_REVIEW_SLICES)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    evaluate(parser().parse_args())
