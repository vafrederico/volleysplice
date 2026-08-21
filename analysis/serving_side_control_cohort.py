"""Deterministic stratified controls for serving-side visibility review."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence


CONTROL_COHORT_KIND = "volleycut-serving-side-flight-correct-control-cohort-v1"
CONFIDENCE_BANDS = (("low", 0.0, 0.7), ("medium", 0.7, 0.9), ("high", 0.9, 1.0))


def _confidence_band(confidence: float) -> str:
    if not 0 <= confidence <= 1 or not math.isfinite(confidence):
        raise ValueError("prediction confidence must be finite and inside [0, 1]")
    if confidence < 0.7:
        return "low"
    if confidence < 0.9:
        return "medium"
    return "high"


def _stable_score(experiment_sha256: str, rally_id: str) -> str:
    return hashlib.sha256(f"{experiment_sha256}\0{rally_id}".encode()).hexdigest()


def _stratum(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    prediction = row.get("prediction")
    probability_near = float(row.get("probabilityNear"))
    if prediction == "near":
        confidence = probability_near
    elif prediction == "far":
        confidence = 1.0 - probability_near
    else:
        raise ValueError(f"prediction {row.get('rallyId')} is not near or far")
    environment = row.get("environment")
    source_group = row.get("sourceGroup")
    human = row.get("decision")
    if not all(isinstance(value, str) and value for value in (environment, source_group)):
        raise ValueError(f"prediction {row.get('rallyId')} lacks source metadata")
    if human not in ("near", "far"):
        raise ValueError(f"prediction {row.get('rallyId')} lacks a human side")
    return environment, source_group, human, _confidence_band(confidence)


def _allocate(
    populations: Mapping[tuple[str, str, str, str], int], target_rows: int
) -> dict[tuple[str, str, str, str], int]:
    keys = sorted(populations)
    if target_rows < len(keys):
        raise ValueError(
            f"target rows {target_rows} cannot cover all {len(keys)} non-empty strata"
        )
    allocation = {key: 1 for key in keys}
    remaining = target_rows - len(keys)
    capacities = {key: populations[key] - 1 for key in keys}
    total_capacity = sum(capacities.values())
    if remaining > total_capacity:
        raise ValueError("control target exceeds the correct prediction population")
    if not remaining:
        return allocation
    ideals = {
        key: remaining * capacities[key] / total_capacity
        for key in keys
    }
    for key in keys:
        extra = min(capacities[key], math.floor(ideals[key]))
        allocation[key] += extra
    leftover = target_rows - sum(allocation.values())
    ranked = sorted(
        keys,
        key=lambda key: (
            -(ideals[key] - math.floor(ideals[key])),
            key,
        ),
    )
    for key in ranked:
        if not leftover:
            break
        if allocation[key] >= populations[key]:
            continue
        allocation[key] += 1
        leftover -= 1
    if leftover or sum(allocation.values()) != target_rows:
        raise AssertionError("stratified control allocation did not reach its target")
    return allocation


def build_correct_control_cohort(
    predictions: Sequence[Mapping[str, Any]],
    *,
    experiment_sha256: str,
    target_rows: int,
) -> dict[str, Any]:
    """Select stable correct controls with coverage across every non-empty stratum."""

    if not isinstance(target_rows, int) or target_rows <= 0:
        raise ValueError("target rows must be a positive integer")
    grouped: dict[
        tuple[str, str, str, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    seen: set[str] = set()
    for row in predictions:
        rally_id = row.get("rallyId")
        if not isinstance(rally_id, str) or not rally_id or rally_id in seen:
            raise ValueError("prediction rally IDs must be non-empty and unique")
        seen.add(rally_id)
        human = row.get("decision")
        prediction = row.get("prediction")
        correct = row.get("correct")
        if correct is not (human == prediction):
            raise ValueError(f"prediction {rally_id} has an inconsistent outcome")
        if correct:
            grouped[_stratum(row)].append(row)
    population_rows = sum(len(rows) for rows in grouped.values())
    target = min(target_rows, population_rows)
    allocation = _allocate({key: len(rows) for key, rows in grouped.items()}, target)
    selected: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    for key in sorted(grouped):
        environment, source_group, human, confidence_band = key
        rows = sorted(
            grouped[key],
            key=lambda row: (
                _stable_score(experiment_sha256, str(row["rallyId"])),
                str(row["rallyId"]),
            ),
        )
        sampled_rows = allocation[key]
        population = len(rows)
        stratum_key = "|".join(key)
        strata.append(
            {
                "key": stratum_key,
                "environment": environment,
                "sourceGroup": source_group,
                "humanSide": human,
                "confidenceBand": confidence_band,
                "populationRows": population,
                "sampledRows": sampled_rows,
                "samplingWeight": population / sampled_rows,
            }
        )
        selected.extend(
            {
                "rallyId": str(row["rallyId"]),
                "stratumKey": stratum_key,
                "samplingWeight": population / sampled_rows,
            }
            for row in rows[:sampled_rows]
        )
    selected.sort(key=lambda row: row["rallyId"])
    if len(selected) != target:
        raise AssertionError("control cohort selection has the wrong size")
    return {
        "kind": CONTROL_COHORT_KIND,
        "sampling": {
            "algorithm": "minimum-one-then-proportional-largest-remainder-v1",
            "population": "correct out-of-source-group development predictions",
            "targetRows": target_rows,
            "populationRows": population_rows,
            "sampledRows": len(selected),
            "stratumFields": [
                "environment",
                "sourceGroup",
                "humanSide",
                "confidenceBand",
            ],
            "confidenceBands": [
                {"name": name, "minimumInclusive": lower, "maximumExclusive": upper}
                for name, lower, upper in CONFIDENCE_BANDS
            ],
            "strata": strata,
        },
        "rows": selected,
    }
