"""Population-weighted serving-side metrics from reviewed errors and controls."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


VISIBILITY_VALUES = ("visible", "partial", "offscreen", "unclear")
CONTACT_VALUES = ("on-anchor", "before-anchor", "after-anchor", "unclear")


def _confidence_band(row: Mapping[str, Any]) -> str:
    probability_near = float(row["probabilityNear"])
    confidence = probability_near if row["prediction"] == "near" else 1 - probability_near
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError(f"invalid confidence for {row.get('rallyId')}")
    return "low" if confidence < 0.7 else "medium" if confidence < 0.9 else "high"


def control_stratum(row: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(row["environment"]),
            str(row["sourceGroup"]),
            str(row["decision"]),
            _confidence_band(row),
        )
    )


def _divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def weighted_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    confusion = {
        human: {predicted: 0.0 for predicted in ("near", "far")}
        for human in ("near", "far")
    }
    for row in rows:
        confusion[str(row["human"])][str(row["prediction"])] += float(row["weight"])
    near_near = confusion["near"]["near"]
    near_far = confusion["near"]["far"]
    far_near = confusion["far"]["near"]
    far_far = confusion["far"]["far"]
    estimated_rows = near_near + near_far + far_near + far_far
    near_recall = _divide(near_near, near_near + near_far)
    far_recall = _divide(far_far, far_far + far_near)
    balanced_accuracy = (
        (near_recall + far_recall) / 2
        if near_recall is not None and far_recall is not None
        else None
    )
    return {
        "observedRows": len(rows),
        "estimatedPopulationRows": estimated_rows,
        "estimatedCorrectRows": near_near + far_far,
        "confusion": confusion,
        "accuracy": _divide(near_near + far_far, estimated_rows),
        "balancedAccuracy": balanced_accuracy,
        "nearPrecision": _divide(near_near, near_near + far_near),
        "nearRecall": near_recall,
        "farPrecision": _divide(far_far, far_far + near_far),
        "farRecall": far_recall,
    }


def rescore_reviewed_predictions(
    reviewed_rows: Sequence[Mapping[str, Any]],
    candidate_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply a new candidate to a frozen two-phase review sample."""
    prediction_by_id: dict[str, str] = {}
    for row in candidate_predictions:
        rally_id = row.get("rallyId")
        prediction = row.get("prediction")
        if (
            not isinstance(rally_id, str)
            or rally_id in prediction_by_id
            or prediction not in ("near", "far")
        ):
            raise ValueError("candidate predictions must have unique IDs and valid sides")
        prediction_by_id[rally_id] = str(prediction)
    rescored: list[dict[str, Any]] = []
    for source in reviewed_rows:
        rally_id = str(source["rallyId"])
        if rally_id not in prediction_by_id:
            raise ValueError(f"candidate lacks reviewed prediction: {rally_id}")
        rescored.append({**source, "prediction": prediction_by_id[rally_id]})
    slices: dict[str, Any] = {}
    for field, values in (
        ("serverVisibility", VISIBILITY_VALUES),
        ("contactTiming", CONTACT_VALUES),
    ):
        slices[field] = {
            value: weighted_metrics(
                [row for row in rescored if row["annotation"].get(field) == value]
            )
            for value in values
            if any(row["annotation"].get(field) == value for row in rescored)
        }
    return {
        "overall": weighted_metrics(rescored),
        "slices": slices,
        "reviewedPredictions": rescored,
    }


def build_review_slice_report(
    predictions: Sequence[Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
    cohort_rows: Sequence[Mapping[str, Any]],
    corrections: Mapping[str, str],
) -> dict[str, Any]:
    prediction_by_id: dict[str, dict[str, Any]] = {}
    for source in predictions:
        rally_id = source.get("rallyId")
        if not isinstance(rally_id, str) or rally_id in prediction_by_id:
            raise ValueError("prediction rally IDs must be unique strings")
        human = corrections.get(rally_id, str(source["decision"]))
        if human == "not-serve":
            continue
        if human not in ("near", "far") or source.get("prediction") not in ("near", "far"):
            raise ValueError(f"invalid side decision for {rally_id}")
        prediction_by_id[rally_id] = {
            **source,
            "decision": human,
            "correct": human == source["prediction"],
        }

    correct = [row for row in prediction_by_id.values() if row["correct"]]
    errors = [row for row in prediction_by_id.values() if not row["correct"]]
    cohort_ids = {
        str(row["rallyId"]): str(row["stratumKey"])
        for row in cohort_rows
    }
    population_by_stratum = Counter(control_stratum(row) for row in correct)
    sampled_by_stratum = Counter(
        control_stratum(row) for row in correct if row["rallyId"] in cohort_ids
    )
    missing_strata = sorted(
        key for key, population in population_by_stratum.items()
        if population and not sampled_by_stratum[key]
    )
    if missing_strata:
        raise ValueError(f"correct-control sample misses current strata: {missing_strata}")

    reviewed: list[dict[str, Any]] = []
    for row in errors:
        rally_id = str(row["rallyId"])
        if rally_id not in annotations:
            raise ValueError(f"current error lacks an annotation: {rally_id}")
        reviewed.append(
            {
                "rallyId": rally_id,
                "recordingId": row["recordingId"],
                "environment": row["environment"],
                "sourceGroup": row["sourceGroup"],
                "human": row["decision"],
                "prediction": row["prediction"],
                "samplingRole": "error-census",
                "stratumKey": None,
                "weight": 1.0,
                "annotation": dict(annotations[rally_id]),
            }
        )
    for row in correct:
        rally_id = str(row["rallyId"])
        if rally_id not in cohort_ids:
            continue
        if rally_id not in annotations:
            raise ValueError(f"correct control lacks an annotation: {rally_id}")
        stratum = control_stratum(row)
        if cohort_ids[rally_id] != stratum:
            raise ValueError(f"control stratum changed for {rally_id}")
        reviewed.append(
            {
                "rallyId": rally_id,
                "recordingId": row["recordingId"],
                "environment": row["environment"],
                "sourceGroup": row["sourceGroup"],
                "human": row["decision"],
                "prediction": row["prediction"],
                "samplingRole": "correct-control",
                "stratumKey": stratum,
                "weight": population_by_stratum[stratum] / sampled_by_stratum[stratum],
                "annotation": dict(annotations[rally_id]),
            }
        )
    reviewed.sort(key=lambda row: str(row["rallyId"]))
    expected_rows = len(prediction_by_id)
    overall = weighted_metrics(reviewed)
    if not math.isclose(
        float(overall["estimatedPopulationRows"]), expected_rows, abs_tol=1e-8
    ):
        raise AssertionError("review weights do not recover the evaluation population")

    slices: dict[str, Any] = {}
    for field, values in (
        ("serverVisibility", VISIBILITY_VALUES),
        ("contactTiming", CONTACT_VALUES),
    ):
        slices[field] = {
            value: weighted_metrics(
                [row for row in reviewed if row["annotation"].get(field) == value]
            )
            for value in values
            if any(row["annotation"].get(field) == value for row in reviewed)
        }
    exact_offsets = [
        float(row["annotation"]["correctedServeAnchorSeconds"])
        for row in reviewed
        if row["annotation"].get("correctedServeAnchorSeconds") is not None
    ]
    return {
        "counts": {
            "evaluationRows": expected_rows,
            "errors": len(errors),
            "correctPopulation": len(correct),
            "reviewedErrors": sum(row["samplingRole"] == "error-census" for row in reviewed),
            "reviewedControls": sum(row["samplingRole"] == "correct-control" for row in reviewed),
            "reviewedRows": len(reviewed),
            "unscopedAnnotations": len(set(annotations) - {row["rallyId"] for row in reviewed}),
            "exactCorrectedAnchors": len(exact_offsets),
        },
        "sampling": {
            "design": "all current errors plus stratified sample of current correct predictions",
            "populationByStratum": dict(sorted(population_by_stratum.items())),
            "sampledByStratum": dict(sorted(sampled_by_stratum.items())),
        },
        "overall": overall,
        "slices": slices,
        "reviewedPredictions": reviewed,
    }
