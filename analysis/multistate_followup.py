"""Leakage-safe diagnostics and follow-up experiments for the multistate study.

The first-stage runner in this module reconstructs the exact outer-fold models
frozen by :mod:`analysis.multistate_feature_study`.  It writes reusable OOF
probabilities for development recordings only, then reports state calibration,
serve rejection, decoded durations, and label-only boundary oracles.  Oracle
results are diagnostic upper bounds; they are never selectable inference
candidates.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .artifacts import atomic_write_text
from .config import FEATURE_VERSION, DecoderConfig, FeatureConfig, TrainingConfig
from .decoder import decode_probabilities
from .feature_experiments import (
    DEFAULT_OBJECTIVE_MARGIN,
    DEFAULT_SIGN_CONSISTENCY,
    DEVELOPMENT_SPLITS,
    FeatureExperimentError,
    _fit,
    _model_fingerprint,
    _prepared_for_groups,
    _seed,
    build_fold_plan,
    classify_paired_deltas,
    objective,
    sha256_file,
)
from .metrics import (
    aggregate_evaluations,
    aggregate_outcome_slices,
    evaluate_intervals,
    ordered_interval_matches,
    outcome_slice_metrics,
    interval_iou,
)
from .multistate import STATE_ORDER, MultistateState, decode_multistate
from .multistate_feature_study import (
    _clip_ignored_predictions,
    _signature_sha256,
    estimate_fold_decoder,
    fit_state_models,
    reconstruct_frozen_upstream_features,
    state_log_scores,
    state_targets,
)
from .pipeline import PreparedRecording, _manifest_digest
from .schema import DatasetManifest, Interval


FOLLOWUP_SCHEMA_VERSION = 1
EXPECTED_STUDY_KIND = "volleycut-multistate-feature-study-development"
DIAGNOSTIC_KIND = "volleycut-multistate-oof-diagnostics-development"
OOF_CACHE_KIND = "volleycut-multistate-oof-cache"
CALIBRATION_BINS = 10
SERVE_WINDOWS_SECONDS = (0.5, 1.0, 2.0)
HYBRID_MODES = (
    "binary_noop",
    "boundary_snap",
    "short_rescue",
    "snap_and_rescue",
)
SNAP_MAX_DISPLACEMENT_SECONDS = 0.75
SNAP_MIN_IOU = 0.50
SNAP_MIN_SMALLER_COVERAGE = 0.70
SHORT_RESCUE_MIN_SECONDS = 0.50
SHORT_RESCUE_MAX_SECONDS = 3.00
HYBRID_STUDY_KIND = "volleycut-conservative-multistate-hybrid-development"


@dataclass(frozen=True)
class HybridPredictionPayload:
    item: PreparedRecording
    binary_intervals: tuple[Interval, ...]
    multistate_intervals: tuple[Interval, ...]
    state_probabilities: np.ndarray
    decoded_states: tuple[MultistateState, ...]
    smoothed_binary: np.ndarray


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureExperimentError(f"cannot read {label} {resolved}: {error}") from error
    if not isinstance(payload, dict):
        raise FeatureExperimentError(f"{label} must contain one JSON object")
    return resolved, payload


def _validate_development_study(
    manifest: DatasetManifest,
    report_path: str | Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], tuple[str, ...]]:
    path, report = _load_json(report_path, "multistate development report")
    if (
        report.get("kind") != EXPECTED_STUDY_KIND
        or report.get("freezeStatus") != "frozen-development-selection"
        or report.get("testLabelsUsed") is not False
        or report.get("selectionProtocol", {}).get("innerFoldLimit") not in (None, 0)
    ):
        raise FeatureExperimentError(
            "diagnostics require the full nested frozen development report with test closed"
        )
    if report.get("manifestFileSha256") != sha256_file(manifest.path):
        raise FeatureExperimentError("multistate report does not match the manifest file")
    if report.get("manifestSnapshotSha256") != _manifest_digest(manifest):
        raise FeatureExperimentError("manifest snapshots changed after multistate freeze")
    if report.get("recordingContentSha256") != {
        item.id: item.content_sha256 for item in manifest.recordings
    }:
        raise FeatureExperimentError("recording identities changed after multistate freeze")
    if report.get("featureVersion") != FEATURE_VERSION:
        raise FeatureExperimentError("feature version changed after multistate freeze")
    if report.get("selectionProtocol", {}).get("folds") != [
        item.to_dict() for item in build_fold_plan(manifest.recordings)
    ]:
        raise FeatureExperimentError("source-group folds changed after multistate freeze")

    upstream_payload = report.get("upstream")
    if not isinstance(upstream_payload, dict):
        raise FeatureExperimentError("multistate report has no frozen upstream payload")
    upstream_path, upstream = _load_json(
        upstream_payload.get("report", ""), "upstream development report"
    )
    if sha256_file(upstream_path) != upstream_payload.get("reportSha256"):
        raise FeatureExperimentError("frozen upstream report content changed")
    raw_names = upstream_payload.get("featureNames")
    if (
        not isinstance(raw_names, list)
        or not raw_names
        or any(not isinstance(name, str) or not name for name in raw_names)
    ):
        raise FeatureExperimentError("multistate report has no frozen feature signature")
    names = tuple(raw_names)
    if _signature_sha256(names) != upstream_payload.get("featureSignatureSha256"):
        raise FeatureExperimentError("frozen feature signature hash is invalid")
    if upstream.get("featureConfig") != report.get("featureConfig"):
        raise FeatureExperimentError("upstream and multistate feature configs disagree")
    return path, report, upstream_path, upstream, names


def _calibration(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 1 or scores.shape != truth.shape or not len(truth):
        raise ValueError("calibration arrays must be aligned non-empty vectors")
    if not np.isin(truth, (0.0, 1.0)).all():
        raise ValueError("calibration labels must be binary")
    scores = np.clip(scores, 1e-8, 1.0 - 1e-8)
    edges = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    indexes = np.minimum(
        np.searchsorted(edges, scores, side="right") - 1,
        CALIBRATION_BINS - 1,
    )
    bins = []
    ece = 0.0
    for index in range(CALIBRATION_BINS):
        selected = indexes == index
        count = int(np.sum(selected))
        if not count:
            continue
        mean_score = float(np.mean(scores[selected]))
        observed = float(np.mean(truth[selected]))
        ece += count / len(truth) * abs(mean_score - observed)
        bins.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "meanProbability": mean_score,
                "observedRate": observed,
            }
        )
    return {
        "samples": len(truth),
        "prevalence": float(np.mean(truth)),
        "meanProbability": float(np.mean(scores)),
        "brier": float(np.mean(np.square(scores - truth))),
        "logLoss": float(
            -np.mean(truth * np.log(scores) + (1.0 - truth) * np.log1p(-scores))
        ),
        "expectedCalibrationError": float(ece),
        "bins": bins,
    }


def state_classification_report(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 1 or scores.shape != (len(truth), len(STATE_ORDER)):
        raise ValueError("state targets and probabilities must be aligned")
    if not len(truth) or np.any((truth < 0) | (truth >= len(STATE_ORDER))):
        raise ValueError("state targets must be non-empty valid state indexes")
    predictions = np.argmax(scores, axis=1)
    confusion = np.zeros((len(STATE_ORDER), len(STATE_ORDER)), dtype=np.int64)
    np.add.at(confusion, (truth, predictions), 1)
    states: dict[str, Any] = {}
    for state in STATE_ORDER:
        index = int(state)
        true_positive = int(confusion[index, index])
        predicted = int(np.sum(confusion[:, index]))
        actual = int(np.sum(confusion[index, :]))
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        states[state.name] = {
            "actualSamples": actual,
            "predictedSamples": predicted,
            "truePositiveSamples": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
            "oneVsRestCalibration": _calibration(
                (truth == index).astype(np.float64), scores[:, index]
            ),
        }
    return {
        "samples": len(truth),
        "accuracy": float(np.mean(predictions == truth)),
        "confusionRowsTruthColumnsPrediction": confusion.tolist(),
        "states": states,
    }


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": len(array),
        "meanSeconds": float(np.mean(array)),
        "p10Seconds": float(np.percentile(array, 10)),
        "p25Seconds": float(np.percentile(array, 25)),
        "p50Seconds": float(np.percentile(array, 50)),
        "p75Seconds": float(np.percentile(array, 75)),
        "p90Seconds": float(np.percentile(array, 90)),
        "minimumSeconds": float(np.min(array)),
        "maximumSeconds": float(np.max(array)),
        "bins": {
            "atMost1Second": int(np.sum(array <= 1.0)),
            "over1To3Seconds": int(np.sum((array > 1.0) & (array <= 3.0))),
            "over3To8Seconds": int(np.sum((array > 3.0) & (array <= 8.0))),
            "over8Seconds": int(np.sum(array > 8.0)),
        },
    }


def _state_run_durations(
    times: np.ndarray, states: Sequence[MultistateState]
) -> dict[str, list[float]]:
    result = {state.name: [] for state in STATE_ORDER}
    if not len(times):
        return result
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.25
    start = 0
    for index in range(1, len(states) + 1):
        if index < len(states) and states[index] == states[start]:
            continue
        result[states[start].name].append((index - start) * step)
        start = index
    return result


def _predicted_multistate_intervals(
    item: PreparedRecording,
    states_decode: Any,
) -> list[Interval]:
    return _clip_ignored_predictions(
        [Interval(value.start, value.end) for value in states_decode.intervals],
        item.recording.ignored_intervals,
    )


def _sample_index_at_or_after(times: np.ndarray, value: float) -> int:
    index = int(np.searchsorted(times, value, side="left"))
    return min(max(index, 0), len(times) - 1)


def _sample_index_nearest(times: np.ndarray, value: float) -> int:
    insertion = int(np.searchsorted(times, value, side="left"))
    candidates = [min(max(insertion, 0), len(times) - 1)]
    if insertion > 0:
        candidates.append(insertion - 1)
    return min(candidates, key=lambda index: (abs(float(times[index]) - value), index))


def _runner_up_margin(probabilities: np.ndarray, state: MultistateState) -> float:
    selected = float(probabilities[int(state)])
    runner_up = max(
        float(probabilities[int(other)]) for other in STATE_ORDER if other != state
    )
    return selected - runner_up


def _interval_overlap_seconds(first: Interval, second: Interval) -> float:
    return max(0.0, min(first.end, second.end) - max(first.start, second.start))


def _edge_gap_seconds(first: Interval, second: Interval) -> float:
    if first.end <= second.start:
        return second.start - first.end
    if second.end <= first.start:
        return first.start - second.end
    return 0.0


def _mutual_best_pairs(
    binary: Sequence[Interval], multistate: Sequence[Interval]
) -> dict[int, int]:
    if not binary or not multistate:
        return {}
    overlaps = np.asarray(
        [[interval_iou(left, right) for right in multistate] for left in binary],
        dtype=np.float64,
    )
    binary_best = np.argmax(overlaps, axis=1)
    multistate_best = np.argmax(overlaps, axis=0)
    pairs: dict[int, int] = {}
    for binary_index, multistate_index in enumerate(binary_best):
        if int(multistate_best[multistate_index]) != binary_index:
            continue
        left = binary[binary_index]
        right = multistate[int(multistate_index)]
        smaller = min(left.end - left.start, right.end - right.start)
        if (
            overlaps[binary_index, multistate_index] >= SNAP_MIN_IOU
            and smaller > 0
            and _interval_overlap_seconds(left, right) / smaller
            >= SNAP_MIN_SMALLER_COVERAGE
        ):
            pairs[binary_index] = int(multistate_index)
    return pairs


def _decoded_live_indexes(
    times: np.ndarray,
    states: Sequence[MultistateState],
    interval: Interval,
) -> np.ndarray:
    values = np.asarray(states, dtype=np.int64)
    return np.flatnonzero(
        (times >= interval.start)
        & (times < interval.end)
        & (values == int(MultistateState.LIVE))
    )


def _snap_intervals(
    binary: Sequence[Interval],
    multistate: Sequence[Interval],
    *,
    times: np.ndarray,
    state_probabilities: np.ndarray,
    decoded_states: Sequence[MultistateState],
) -> tuple[list[Interval], set[int], dict[str, int]]:
    pairs = _mutual_best_pairs(binary, multistate)
    output: list[Interval] = []
    used_multistate: set[int] = set()
    summary = {
        "paired": len(pairs),
        "startSnaps": 0,
        "endSnaps": 0,
        "rejectedByLongInvariant": 0,
        "rejectedByShortInvariant": 0,
    }
    for binary_index, base in enumerate(binary):
        multistate_index = pairs.get(binary_index)
        if multistate_index is None:
            output.append(base)
            continue
        candidate = multistate[multistate_index]
        live_indexes = _decoded_live_indexes(times, decoded_states, candidate)
        if not len(live_indexes) or float(
            np.mean(state_probabilities[live_indexes, int(MultistateState.LIVE)])
        ) < 0.40:
            output.append(base)
            continue
        start_index = _sample_index_nearest(times, candidate.start)
        end_index = _sample_index_at_or_after(times, candidate.end)
        last_live_index = int(live_indexes[-1])
        start_eligible = (
            abs(candidate.start - base.start) <= SNAP_MAX_DISPLACEMENT_SECONDS
            and state_probabilities[start_index, int(MultistateState.SERVE)] >= 0.40
            and _runner_up_margin(
                state_probabilities[start_index], MultistateState.SERVE
            )
            >= 0.10
        )
        end_eligible = (
            abs(candidate.end - base.end) <= SNAP_MAX_DISPLACEMENT_SECONDS
            and state_probabilities[end_index, int(MultistateState.DEAD)] >= 0.40
            and _runner_up_margin(
                state_probabilities[end_index], MultistateState.DEAD
            )
            >= 0.10
            and state_probabilities[last_live_index, int(MultistateState.LIVE)]
            >= 0.35
        )
        refined = Interval(
            candidate.start if start_eligible else base.start,
            candidate.end if end_eligible else base.end,
        )
        if refined.end <= refined.start:
            output.append(base)
            continue
        base_duration = base.end - base.start
        if base_duration > 3.0:
            retained = _interval_overlap_seconds(base, refined) / base_duration
            if retained < 0.90 or interval_iou(base, refined) < 0.85:
                summary["rejectedByLongInvariant"] += 1
                output.append(base)
                continue
        elif interval_iou(base, refined) < 0.60:
            summary["rejectedByShortInvariant"] += 1
            output.append(base)
            continue
        used_multistate.add(multistate_index)
        summary["startSnaps"] += int(start_eligible and refined.start != base.start)
        summary["endSnaps"] += int(end_eligible and refined.end != base.end)
        output.append(refined)
    return output, used_multistate, summary


def _flank_mean(
    times: np.ndarray,
    smoothed_binary: np.ndarray,
    lower: float,
    upper: float,
) -> float | None:
    selected = (times >= lower) & (times < upper)
    return float(np.mean(smoothed_binary[selected])) if np.any(selected) else None


def _has_consecutive_support(values: np.ndarray, minimum: float, run: int) -> bool:
    above = np.asarray(values >= minimum, dtype=np.int8)
    if len(above) < run:
        return False
    return bool(np.any(np.convolve(above, np.ones(run, dtype=np.int8), "valid") == run))


def _short_rescues(
    base_intervals: Sequence[Interval],
    multistate: Sequence[Interval],
    used_multistate: set[int],
    *,
    times: np.ndarray,
    state_probabilities: np.ndarray,
    decoded_states: Sequence[MultistateState],
    smoothed_binary: np.ndarray,
    ignored: Sequence[Interval],
) -> tuple[list[Interval], dict[str, int]]:
    candidates: list[tuple[tuple[float, float, float, float], Interval]] = []
    for index, interval in enumerate(multistate):
        duration = interval.end - interval.start
        if index in used_multistate or not (
            SHORT_RESCUE_MIN_SECONDS <= duration <= SHORT_RESCUE_MAX_SECONDS
        ):
            continue
        if any(_interval_overlap_seconds(interval, row) > 0 for row in ignored):
            continue
        if any(interval_iou(interval, row) >= 0.10 for row in base_intervals):
            continue
        if any(_edge_gap_seconds(interval, row) < 0.75 for row in base_intervals):
            continue
        live_indexes = _decoded_live_indexes(times, decoded_states, interval)
        if not len(live_indexes):
            continue
        selected = (times >= interval.start) & (times < interval.end)
        if not np.any(selected):
            continue
        start_index = _sample_index_nearest(times, interval.start)
        end_index = _sample_index_at_or_after(times, interval.end)
        serve_margin = _runner_up_margin(
            state_probabilities[start_index], MultistateState.SERVE
        )
        dead_margin = _runner_up_margin(
            state_probabilities[end_index], MultistateState.DEAD
        )
        mean_live = float(
            np.mean(state_probabilities[live_indexes, int(MultistateState.LIVE)])
        )
        local_binary = smoothed_binary[selected]
        if not (
            state_probabilities[start_index, int(MultistateState.SERVE)] >= 0.50
            and serve_margin >= 0.15
            and mean_live >= 0.40
            and state_probabilities[end_index, int(MultistateState.DEAD)] >= 0.50
            and dead_margin >= 0.15
            and float(np.max(local_binary)) >= 0.80
            and float(np.mean(local_binary)) >= 0.55
            and _has_consecutive_support(local_binary, 0.60, 2)
        ):
            continue
        left = _flank_mean(
            times, smoothed_binary, interval.start - 1.0, interval.start
        )
        right = _flank_mean(
            times, smoothed_binary, interval.end, interval.end + 1.0
        )
        if any(value is not None and value > 0.35 for value in (left, right)):
            continue
        candidates.append(
            (
                (
                    min(serve_margin, dead_margin),
                    mean_live,
                    float(np.max(local_binary)),
                    -interval.start,
                ),
                interval,
            )
        )
    cap = min(2, max(1, int(math.ceil(0.05 * len(base_intervals)))))
    accepted: list[Interval] = []
    for _, interval in sorted(candidates, key=lambda row: row[0], reverse=True):
        if len(accepted) >= cap:
            break
        if any(_edge_gap_seconds(interval, row) < 0.75 for row in accepted):
            continue
        accepted.append(interval)
    return accepted, {
        "eligibleBeforeNms": len(candidates),
        "accepted": len(accepted),
        "cap": cap,
    }


def compose_conservative_hybrid(
    binary: Sequence[Interval],
    multistate: Sequence[Interval],
    *,
    mode: str,
    times: np.ndarray,
    state_probabilities: np.ndarray,
    decoded_states: Sequence[MultistateState],
    smoothed_binary: np.ndarray,
    ignored: Sequence[Interval] = (),
) -> tuple[list[Interval], dict[str, Any]]:
    """Compose one of four predeclared binary-preserving hybrid modes."""

    if mode not in HYBRID_MODES:
        raise ValueError(f"unknown conservative hybrid mode: {mode}")
    base = list(binary)
    used: set[int] = set()
    snap_summary = {
        "paired": 0,
        "startSnaps": 0,
        "endSnaps": 0,
        "rejectedByLongInvariant": 0,
        "rejectedByShortInvariant": 0,
    }
    if mode in {"boundary_snap", "snap_and_rescue"}:
        base, used, snap_summary = _snap_intervals(
            binary,
            multistate,
            times=times,
            state_probabilities=state_probabilities,
            decoded_states=decoded_states,
        )
    rescue_summary = {"eligibleBeforeNms": 0, "accepted": 0, "cap": 0}
    rescues: list[Interval] = []
    if mode in {"short_rescue", "snap_and_rescue"}:
        rescues, rescue_summary = _short_rescues(
            base,
            multistate,
            used,
            times=times,
            state_probabilities=state_probabilities,
            decoded_states=decoded_states,
            smoothed_binary=smoothed_binary,
            ignored=ignored,
        )
    output = sorted([*base, *rescues], key=lambda interval: (interval.start, interval.end))
    return output, {
        "mode": mode,
        "binaryIntervalsIn": len(binary),
        "multistateIntervalsIn": len(multistate),
        "outputIntervals": len(output),
        "snap": snap_summary,
        "rescue": rescue_summary,
    }


def _hybrid_payload(
    item: PreparedRecording,
    binary_probabilities: np.ndarray,
    binary_decoder: DecoderConfig,
    state_scores: np.ndarray,
    state_decoder: Any,
) -> HybridPredictionPayload:
    analysis_fps = (
        1.0 / float(np.median(np.diff(item.sequence.times)))
        if len(item.sequence.times) > 1
        else 1.0
    )
    decoded_binary, smoothed = decode_probabilities(
        item.sequence.times,
        binary_probabilities,
        item.sequence.metadata.duration,
        binary_decoder,
        analysis_fps,
    )
    binary = _clip_ignored_predictions(
        [Interval(value.start, value.end) for value in decoded_binary],
        item.recording.ignored_intervals,
    )
    decoded_state = decode_multistate(item.sequence.times, state_scores, state_decoder)
    multistate = _predicted_multistate_intervals(item, decoded_state)
    return HybridPredictionPayload(
        item=item,
        binary_intervals=tuple(binary),
        multistate_intervals=tuple(multistate),
        state_probabilities=np.exp(state_scores),
        decoded_states=decoded_state.states,
        smoothed_binary=smoothed,
    )


def evaluate_hybrid_payloads(
    payloads: Sequence[HybridPredictionPayload], mode: str
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("hybrid evaluation requires prediction payloads")
    rows: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    for payload in payloads:
        predictions, summary = compose_conservative_hybrid(
            payload.binary_intervals,
            payload.multistate_intervals,
            mode=mode,
            times=payload.item.sequence.times,
            state_probabilities=payload.state_probabilities,
            decoded_states=payload.decoded_states,
            smoothed_binary=payload.smoothed_binary,
            ignored=payload.item.recording.ignored_intervals,
        )
        if len(predictions) < len(payload.binary_intervals):
            raise FeatureExperimentError("hybrid deleted a binary proposal")
        if mode in {"binary_noop", "boundary_snap"} and len(predictions) != len(
            payload.binary_intervals
        ):
            raise FeatureExperimentError("non-rescue hybrid changed proposal count")
        rows.append(_evaluated_row(payload.item, predictions))
        operations.append(
            {
                "recordingId": payload.item.recording.id,
                "sourceGroup": payload.item.recording.source_group,
                **summary,
            }
        )
    aggregate = _aggregate_rows(rows)
    groups = sorted({row["sourceGroup"] for row in rows})
    by_group = {
        group: _aggregate_rows(
            [row for row in rows if row["sourceGroup"] == group]
        )
        for group in groups
    }
    return {
        "mode": mode,
        "aggregate": aggregate,
        "macroSourceGroup": {
            key: float(np.mean([metrics[key] for metrics in by_group.values()]))
            for key in ("objective", "eventF1", "timeIoU", "liveTimeRecall")
        },
        "bySourceGroup": by_group,
        "recordings": rows,
        "operations": operations,
    }


def _strict_slice_count(metrics: Mapping[str, Any], name: str) -> int:
    values = metrics["outcomeSlices"][name]
    return int(round(int(values.get("rallies", 0)) * float(values.get("strictMatchRecall", 0.0))))


def _finite_delta(
    candidate: Mapping[str, Any], control: Mapping[str, Any], key: str
) -> float:
    left = candidate.get(key)
    right = control.get(key)
    if left is None or right is None:
        return 0.0 if left == right else -math.inf
    return float(left) - float(right)


def _hybrid_inner_gate(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    if mode == "binary_noop":
        return {
            "eligible": True,
            "checks": {"exactNoop": candidate["aggregate"] == control["aggregate"]},
            "reason": "always-available recall floor",
        }
    control_aggregate = control["aggregate"]
    candidate_aggregate = candidate["aggregate"]
    groups = sorted(control["bySourceGroup"])
    objective_deltas = [
        candidate["bySourceGroup"][group]["objective"]
        - control["bySourceGroup"][group]["objective"]
        for group in groups
    ]
    control_ordinary = control_aggregate["outcomeSlices"]["ordinaryLong"]
    candidate_ordinary = candidate_aggregate["outcomeSlices"]["ordinaryLong"]
    checks = {
        "aggregateLiveRecallLossAtMostOnePoint": (
            candidate_aggregate["liveTimeRecall"]
            >= control_aggregate["liveTimeRecall"] - 0.01
        ),
        "eachSourceLiveRecallLossAtMostThreePoints": all(
            candidate["bySourceGroup"][group]["liveTimeRecall"]
            >= control["bySourceGroup"][group]["liveTimeRecall"] - 0.03
            for group in groups
        ),
        "ordinaryLongStrictMatchesPreserved": (
            _strict_slice_count(candidate_aggregate, "ordinaryLong")
            >= _strict_slice_count(control_aggregate, "ordinaryLong")
        ),
        "ordinaryLongOverlapPreserved": (
            candidate_ordinary["anyOverlapRecall"]
            >= control_ordinary["anyOverlapRecall"] - 0.01
        ),
        "ordinaryLongCoveragePreserved": (
            candidate_ordinary["meanCoverage"]
            >= control_ordinary["meanCoverage"] - 0.01
        ),
        "eachSourceLosesAtMostOneOrdinaryLongStrictMatch": all(
            _strict_slice_count(candidate["bySourceGroup"][group], "ordinaryLong")
            >= _strict_slice_count(control["bySourceGroup"][group], "ordinaryLong") - 1
            for group in groups
        ),
        "eventPrecisionLossAtMostOnePoint": (
            candidate_aggregate["eventPrecision"]
            >= control_aggregate["eventPrecision"] - 0.01
        ),
        "deadSecondsRetainedAtMostTwoPercentHigher": (
            candidate_aggregate["deadSecondsRetained"]
            <= control_aggregate["deadSecondsRetained"] * 1.02 + 1e-9
        ),
        "eachSourceEventF1LossAtMostThreePoints": all(
            candidate["bySourceGroup"][group]["eventF1"]
            >= control["bySourceGroup"][group]["eventF1"] - 0.03
            for group in groups
        ),
        "macroObjectiveGainAtLeastOnePoint": (
            candidate["macroSourceGroup"]["objective"]
            >= control["macroSourceGroup"]["objective"] + 0.01
        ),
        "medianSourceObjectiveNonnegative": float(np.median(objective_deltas)) >= 0.0,
        "atLeastTwoInnerSourcesNonnegative": sum(value >= 0.0 for value in objective_deltas)
        >= 2,
    }
    if mode in {"boundary_snap", "snap_and_rescue"}:
        start_delta = _finite_delta(candidate_aggregate, control_aggregate, "startBoundaryMaeSeconds")
        end_delta = _finite_delta(candidate_aggregate, control_aggregate, "endBoundaryMaeSeconds")
        checks.update(
            {
                "combinedBoundaryMaeImprovesAtLeast005Seconds": start_delta + end_delta
                <= -0.05,
                "neitherBoundaryMaeWorsensOver005Seconds": start_delta <= 0.05
                and end_delta <= 0.05,
            }
        )
    if mode in {"short_rescue", "snap_and_rescue"}:
        checks.update(
            {
                "shortStrictMatchesImprove": (
                    _strict_slice_count(candidate_aggregate, "shortAtMost3Seconds")
                    >= _strict_slice_count(control_aggregate, "shortAtMost3Seconds") + 1
                ),
                "serviceFaultStrictMatchesDoNotDecline": (
                    _strict_slice_count(candidate_aggregate, "serviceFault")
                    >= _strict_slice_count(control_aggregate, "serviceFault")
                ),
            }
        )
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "sourceObjectiveDeltas": dict(zip(groups, objective_deltas, strict=True)),
    }


def select_inner_hybrid(
    reports: Mapping[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if tuple(reports) != HYBRID_MODES:
        raise ValueError("hybrid reports must follow the frozen candidate order")
    control = reports["binary_noop"]
    gates = {
        mode: _hybrid_inner_gate(control, reports[mode], mode=mode)
        for mode in HYBRID_MODES
    }
    selected = "binary_noop"
    for mode in HYBRID_MODES[1:]:
        if not gates[mode]["eligible"]:
            continue
        selected_objective = reports[selected]["aggregate"]["objective"]
        candidate_objective = reports[mode]["aggregate"]["objective"]
        if candidate_objective > selected_objective + 0.005:
            selected = mode
    return selected, {
        "candidateOrderSimplestFirst": list(HYBRID_MODES),
        "tieMargin": 0.005,
        "selectedMode": selected,
        "candidates": {
            mode: {
                "gate": gates[mode],
                "metrics": reports[mode]["aggregate"],
                "macroSourceGroup": reports[mode]["macroSourceGroup"],
                "operations": reports[mode]["operations"],
            }
            for mode in HYBRID_MODES
        },
    }


def _frozen_inner_by_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {
        str(row["validationSourceGroup"]): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("validationSourceGroup"), str)
    }
    if len(result) != len(rows):
        raise FeatureExperimentError("frozen inner-fold rows are invalid or duplicated")
    return result


def _assert_fingerprint(actual: str, expected: Any, label: str) -> None:
    if actual != expected:
        raise FeatureExperimentError(f"{label} model fingerprint did not reproduce")


def _assert_metrics(
    actual: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    for metric in ("eventF1", "timeIoU", "liveTimeRecall", "objective"):
        if not math.isclose(
            float(actual[metric]), float(expected[metric]), abs_tol=1e-10
        ):
            raise FeatureExperimentError(f"{label} metric did not reproduce: {metric}")


def _plain_multistate_report(
    payloads: Sequence[HybridPredictionPayload],
) -> dict[str, Any]:
    rows = [
        _evaluated_row(payload.item, payload.multistate_intervals)
        for payload in payloads
    ]
    return {
        "aggregate": _aggregate_rows(rows),
        "recordings": rows,
    }


def _run_hybrid_outer_fold(
    selected: Sequence[PreparedRecording],
    fold: Any,
    *,
    feature_config: FeatureConfig,
    training_config: TrainingConfig,
    base_decoder_config: DecoderConfig,
    control_frozen: Mapping[str, Any],
    state_frozen: Mapping[str, Any],
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    outer_training = _prepared_for_groups(selected, fold.training_groups)
    held = _prepared_for_groups(selected, (fold.held_out_group,))
    if not outer_training or not held:
        raise FeatureExperimentError("hybrid outer fold is empty")
    binary_decoder = DecoderConfig.from_dict(dict(control_frozen["decoder"]))
    transition_bonus = float(
        state_frozen["innerSelection"]["selectedTransitionBonus"]
    )
    control_inner = _frozen_inner_by_group(
        control_frozen["innerSelection"]["foldsUsed"]
    )
    state_inner = _frozen_inner_by_group(
        state_frozen["innerSelection"]["foldsUsed"]
    )
    inner_payloads: list[HybridPredictionPayload] = []
    inner_reproduction: list[dict[str, Any]] = []
    for index, inner in enumerate(fold.inner_folds, start=1):
        if progress is not None:
            progress(
                f"Hybrid {fold.held_out_group}: inner {index}/{len(fold.inner_folds)} "
                f"validate {inner.validation_group}"
            )
        training = _prepared_for_groups(outer_training, inner.training_groups)
        validation = _prepared_for_groups(
            outer_training, (inner.validation_group,)
        )
        binary_model = _fit(
            training,
            validation,
            feature_config=feature_config,
            decoder=base_decoder_config,
            config=training_config,
            seed=_seed(
                training_config.seed,
                "inner",
                fold.held_out_group,
                inner.validation_group,
            ),
        )
        binary_fingerprint = _model_fingerprint(binary_model)
        _assert_fingerprint(
            binary_fingerprint,
            control_inner[inner.validation_group]["modelFingerprint"],
            f"{fold.held_out_group}/{inner.validation_group} binary inner",
        )
        state_bundle = fit_state_models(
            training,
            validation,
            feature_config=feature_config,
            training_config=training_config,
            seed_parts=(
                "multistate-inner",
                fold.held_out_group,
                inner.validation_group,
            ),
        )
        state_fingerprints = {
            state.name: _model_fingerprint(model)
            for state, model in zip(STATE_ORDER, state_bundle.models, strict=True)
        }
        expected_heads = state_inner[inner.validation_group]["heads"]
        for state in STATE_ORDER:
            _assert_fingerprint(
                state_fingerprints[state.name],
                expected_heads[state.name]["modelFingerprint"],
                f"{fold.held_out_group}/{inner.validation_group}/{state.name} inner",
            )
        state_decoder, prior_summary = estimate_fold_decoder(
            training, transition_bonus=transition_bonus
        )
        for item in validation:
            inner_payloads.append(
                _hybrid_payload(
                    item,
                    binary_model.predict(item.contextual_values),
                    binary_decoder,
                    state_log_scores(state_bundle, item.contextual_values),
                    state_decoder,
                )
            )
        inner_reproduction.append(
            {
                **inner.to_dict(),
                "binaryModelFingerprint": binary_fingerprint,
                "stateModelFingerprints": state_fingerprints,
                "durationAndTransitionPriorEstimate": prior_summary,
            }
        )
    inner_reports = {
        mode: evaluate_hybrid_payloads(inner_payloads, mode) for mode in HYBRID_MODES
    }
    selected_mode, inner_selection = select_inner_hybrid(inner_reports)

    if progress is not None:
        progress(
            f"Hybrid {fold.held_out_group}: refit outer models; selected {selected_mode}"
        )
    binary_epoch_cap = int(control_frozen["innerSelection"]["selectedEpochCap"])
    binary_refit_config = replace(
        training_config,
        epochs=binary_epoch_cap,
        patience=max(training_config.patience, binary_epoch_cap + 1),
    )
    binary_model = _fit(
        outer_training,
        (),
        feature_config=feature_config,
        decoder=binary_decoder,
        config=binary_refit_config,
        seed=_seed(training_config.seed, "outer-refit", fold.held_out_group),
    )
    binary_fingerprint = _model_fingerprint(binary_model)
    _assert_fingerprint(
        binary_fingerprint,
        control_frozen["outerRefit"]["modelFingerprint"],
        f"{fold.held_out_group} binary outer",
    )
    state_epoch_caps = {
        state: int(state_frozen["innerSelection"]["selectedEpochCaps"][state.name])
        for state in STATE_ORDER
    }
    state_bundle = fit_state_models(
        outer_training,
        (),
        feature_config=feature_config,
        training_config=training_config,
        seed_parts=("multistate-outer-refit", fold.held_out_group),
        epoch_caps=state_epoch_caps,
    )
    state_fingerprints = {
        state.name: _model_fingerprint(model)
        for state, model in zip(STATE_ORDER, state_bundle.models, strict=True)
    }
    if state_fingerprints != state_frozen["outerRefit"]["modelFingerprints"]:
        raise FeatureExperimentError(
            f"{fold.held_out_group} state outer fingerprints did not reproduce"
        )
    state_decoder, outer_prior_summary = estimate_fold_decoder(
        outer_training, transition_bonus=transition_bonus
    )
    outer_payloads = [
        _hybrid_payload(
            item,
            binary_model.predict(item.contextual_values),
            binary_decoder,
            state_log_scores(state_bundle, item.contextual_values),
            state_decoder,
        )
        for item in held
    ]
    control_report = evaluate_hybrid_payloads(outer_payloads, "binary_noop")
    candidate_report = evaluate_hybrid_payloads(outer_payloads, selected_mode)
    multistate_report = _plain_multistate_report(outer_payloads)
    _assert_metrics(
        control_report["aggregate"],
        control_frozen["metrics"],
        f"{fold.held_out_group} binary outer",
    )
    _assert_metrics(
        multistate_report["aggregate"],
        state_frozen["metrics"],
        f"{fold.held_out_group} multistate outer",
    )
    return {
        "heldOutSourceGroup": fold.held_out_group,
        "trainingSourceGroups": list(fold.training_groups),
        "innerReproduction": inner_reproduction,
        "innerSelection": inner_selection,
        "outerRefit": {
            "selectedMode": selected_mode,
            "binaryEpochCap": binary_epoch_cap,
            "binaryDecoder": binary_decoder.to_dict(),
            "binaryModelFingerprint": binary_fingerprint,
            "stateEpochCaps": {
                state.name: state_epoch_caps[state] for state in STATE_ORDER
            },
            "stateModelFingerprints": state_fingerprints,
            "transitionBonus": transition_bonus,
            "durationAndTransitionPriorEstimate": outer_prior_summary,
        },
        "binaryControl": control_report,
        "conservativeHybrid": candidate_report,
        "multistateReproduction": multistate_report,
    }


def _aggregate_outer_fold_key(
    outer_folds: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Any]:
    rows = [
        row
        for fold in outer_folds
        for row in fold[key]["recordings"]
    ]
    aggregate = _aggregate_rows(rows)
    by_group = {
        str(fold["heldOutSourceGroup"]): fold[key]["aggregate"]
        for fold in outer_folds
    }
    return {
        "aggregate": aggregate,
        "macroSourceGroup": {
            metric: float(np.mean([row[metric] for row in by_group.values()]))
            for metric in ("objective", "eventF1", "timeIoU", "liveTimeRecall")
        },
        "bySourceGroup": by_group,
        "recordings": rows,
    }


def _hybrid_paired_comparison(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    objective_margin: float,
    sign_consistency: float,
) -> dict[str, Any]:
    groups = sorted(control["bySourceGroup"])
    rows = []
    for group in groups:
        baseline = control["bySourceGroup"][group]
        hybrid = candidate["bySourceGroup"][group]
        rows.append(
            {
                "sourceGroup": group,
                "controlObjective": baseline["objective"],
                "hybridObjective": hybrid["objective"],
                "deltaObjectiveHybridMinusControl": hybrid["objective"]
                - baseline["objective"],
                "deltaEventF1": hybrid["eventF1"] - baseline["eventF1"],
                "deltaTimeIoU": hybrid["timeIoU"] - baseline["timeIoU"],
                "deltaLiveTimeRecall": hybrid["liveTimeRecall"]
                - baseline["liveTimeRecall"],
                "deltaOrdinaryLongStrictMatches": _strict_slice_count(
                    hybrid, "ordinaryLong"
                )
                - _strict_slice_count(baseline, "ordinaryLong"),
            }
        )
    deltas = [row["deltaObjectiveHybridMinusControl"] for row in rows]
    return {
        "deltaDirection": "positive means conservative hybrid improves over binary",
        "classification": classify_paired_deltas(
            deltas, margin=objective_margin, sign_consistency=sign_consistency
        ),
        "meanObjectiveDelta": float(np.mean(deltas)),
        "medianObjectiveDelta": float(np.median(deltas)),
        "positiveSourceGroups": sum(value > 0 for value in deltas),
        "pairedSourceGroups": rows,
    }


def _hybrid_promotion_gate(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    paired: Mapping[str, Any],
    selected_modes: Sequence[str],
) -> dict[str, Any]:
    baseline = control["aggregate"]
    hybrid = candidate["aggregate"]
    groups = sorted(control["bySourceGroup"])
    ordinary_control = baseline["outcomeSlices"]["ordinaryLong"]
    ordinary_hybrid = hybrid["outcomeSlices"]["ordinaryLong"]
    checks = {
        "pairedObjectiveEvidenceHelpful": paired["classification"]["classification"]
        == "helpful",
        "meanAndMedianObjectiveGainAtLeastOnePoint": paired["meanObjectiveDelta"]
        >= 0.01
        and paired["medianObjectiveDelta"] >= 0.01,
        "atLeastThreeOfFourSourcesImprove": paired["positiveSourceGroups"] >= 3,
        "aggregateLiveRecallLossAtMostOnePoint": hybrid["liveTimeRecall"]
        >= baseline["liveTimeRecall"] - 0.01,
        "eachSourceLiveRecallLossAtMostThreePoints": all(
            candidate["bySourceGroup"][group]["liveTimeRecall"]
            >= control["bySourceGroup"][group]["liveTimeRecall"] - 0.03
            for group in groups
        ),
        "ordinaryLongStrictMatchesPreserved": _strict_slice_count(
            hybrid, "ordinaryLong"
        )
        >= _strict_slice_count(baseline, "ordinaryLong"),
        "atLeastThreeSourcesPreserveOrdinaryLongStrict": sum(
            _strict_slice_count(candidate["bySourceGroup"][group], "ordinaryLong")
            >= _strict_slice_count(control["bySourceGroup"][group], "ordinaryLong")
            for group in groups
        )
        >= 3,
        "noSourceLosesOverOneOrdinaryLongStrict": all(
            _strict_slice_count(candidate["bySourceGroup"][group], "ordinaryLong")
            >= _strict_slice_count(control["bySourceGroup"][group], "ordinaryLong") - 1
            for group in groups
        ),
        "ordinaryLongOverlapAndCoveragePreserved": ordinary_hybrid[
            "anyOverlapRecall"
        ]
        >= ordinary_control["anyOverlapRecall"] - 0.01
        and ordinary_hybrid["meanCoverage"] >= ordinary_control["meanCoverage"] - 0.01,
        "eventPrecisionDoesNotWorsen": hybrid["eventPrecision"]
        >= baseline["eventPrecision"],
        "deadSecondsRetainedDoesNotWorsen": hybrid["deadSecondsRetained"]
        <= baseline["deadSecondsRetained"] + 1e-9,
        "noSourceEventF1LossOverThreePoints": all(
            candidate["bySourceGroup"][group]["eventF1"]
            >= control["bySourceGroup"][group]["eventF1"] - 0.03
            for group in groups
        ),
    }
    if any(mode in {"boundary_snap", "snap_and_rescue"} for mode in selected_modes):
        start_delta = _finite_delta(hybrid, baseline, "startBoundaryMaeSeconds")
        end_delta = _finite_delta(hybrid, baseline, "endBoundaryMaeSeconds")
        checks.update(
            {
                "selectedSnapImprovesCombinedBoundaryMae": start_delta + end_delta
                <= -0.05,
                "selectedSnapDoesNotWorsenEitherBoundaryMae": start_delta <= 0.05
                and end_delta <= 0.05,
            }
        )
    if any(mode in {"short_rescue", "snap_and_rescue"} for mode in selected_modes):
        checks.update(
            {
                "selectedRescueAddsAtLeastTwoShortStrictMatches": _strict_slice_count(
                    hybrid, "shortAtMost3Seconds"
                )
                >= _strict_slice_count(baseline, "shortAtMost3Seconds") + 2,
                "selectedRescuePreservesServiceFaultStrictMatches": _strict_slice_count(
                    hybrid, "serviceFault"
                )
                >= _strict_slice_count(baseline, "serviceFault"),
            }
        )
    return {
        "promoteConservativeHybrid": all(checks.values()),
        "checks": checks,
        "selectedModesByOuterFold": list(selected_modes),
        "deltaObjective": hybrid["objective"] - baseline["objective"],
        "deltaLiveTimeRecall": hybrid["liveTimeRecall"]
        - baseline["liveTimeRecall"],
        "strictMatchCounts": {
            name: {
                "control": _strict_slice_count(baseline, name),
                "hybrid": _strict_slice_count(hybrid, name),
            }
            for name in ("shortAtMost3Seconds", "serviceFault", "ordinaryLong")
        },
    }


def run_conservative_hybrid_study(
    manifest: DatasetManifest,
    prepared: Sequence[PreparedRecording],
    *,
    multistate_report_path: str | Path,
    objective_margin: float = DEFAULT_OBJECTIVE_MARGIN,
    sign_consistency: float = DEFAULT_SIGN_CONSISTENCY,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the fixed four-mode hybrid with nested source-group selection."""

    started = time.perf_counter()
    report_path, frozen, upstream_path, _upstream, frozen_names = (
        _validate_development_study(manifest, multistate_report_path)
    )
    development_ids = {
        item.id for item in manifest.recordings if item.split in DEVELOPMENT_SPLITS
    }
    if {item.recording.id for item in prepared} != development_ids or any(
        item.recording.split not in DEVELOPMENT_SPLITS for item in prepared
    ):
        raise FeatureExperimentError(
            "hybrid study requires exactly train+validation prepared recordings"
        )
    feature_config = FeatureConfig.from_dict(frozen["featureConfig"])
    training_config = TrainingConfig(**frozen["trainingConfig"])
    base_decoder_config = DecoderConfig.from_dict(frozen["baseDecoderConfig"])
    selected = reconstruct_frozen_upstream_features(
        prepared, feature_config, frozen_names
    )
    folds = build_fold_plan(manifest.recordings)
    control_frozen = {
        row["heldOutSourceGroup"]: row
        for row in frozen["sameFeatureBinaryControl"]["outerFolds"]
    }
    state_frozen = {
        row["heldOutSourceGroup"]: row
        for row in frozen["serveAnchoredMultistate"]["outerFolds"]
    }
    outer_rows = []
    for index, fold in enumerate(folds, start=1):
        if progress is not None:
            progress(
                f"Conservative hybrid outer {index}/{len(folds)}: hold out "
                f"{fold.held_out_group}"
            )
        outer_rows.append(
            _run_hybrid_outer_fold(
                selected,
                fold,
                feature_config=feature_config,
                training_config=training_config,
                base_decoder_config=base_decoder_config,
                control_frozen=control_frozen[fold.held_out_group],
                state_frozen=state_frozen[fold.held_out_group],
                progress=progress,
            )
        )
    control_oof = _aggregate_outer_fold_key(outer_rows, "binaryControl")
    hybrid_oof = _aggregate_outer_fold_key(outer_rows, "conservativeHybrid")
    _assert_metrics(
        control_oof["aggregate"],
        frozen["sameFeatureBinaryControl"]["oof"]["aggregate"],
        "pooled binary OOF",
    )
    selected_modes = [
        row["innerSelection"]["selectedMode"] for row in outer_rows
    ]
    paired = _hybrid_paired_comparison(
        control_oof,
        hybrid_oof,
        objective_margin=objective_margin,
        sign_consistency=sign_consistency,
    )
    gate = _hybrid_promotion_gate(
        control_oof, hybrid_oof, paired, selected_modes
    )
    final_mode = min(
        HYBRID_MODES,
        key=lambda mode: (-selected_modes.count(mode), HYBRID_MODES.index(mode)),
    )
    return {
        "schemaVersion": FOLLOWUP_SCHEMA_VERSION,
        "kind": HYBRID_STUDY_KIND,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "freezeStatus": "frozen-development-selection",
        "assessmentRole": "full-nested-development-source-group-oof-selection",
        "testLabelsUsed": False,
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "manifestSnapshotSha256": _manifest_digest(manifest),
        "recordingContentSha256": {
            item.id: item.content_sha256 for item in manifest.recordings
        },
        "multistateDevelopmentReport": str(report_path),
        "multistateDevelopmentReportSha256": sha256_file(report_path),
        "upstreamDevelopmentReport": str(upstream_path),
        "upstreamDevelopmentReportSha256": sha256_file(upstream_path),
        "featureVersion": FEATURE_VERSION,
        "featureConfig": feature_config.to_dict(),
        "trainingConfig": training_config.to_dict(),
        "featureNames": list(frozen_names),
        "featureSignatureSha256": _signature_sha256(frozen_names),
        "selectionProtocol": {
            "outer": "leave one development sourceGroup out",
            "inner": "leave one remaining sourceGroup out",
            "innerFoldLimit": None,
            "candidateModes": list(HYBRID_MODES),
            "candidateScope": "fixed 2x2 snap/rescue family; no numeric search",
            "snapRules": {
                "mutualBestMinimumIoU": SNAP_MIN_IOU,
                "minimumSmallerIntervalCoverage": SNAP_MIN_SMALLER_COVERAGE,
                "maximumBoundaryDisplacementSeconds": SNAP_MAX_DISPLACEMENT_SECONDS,
                "longBinaryMinimumCoverage": 0.90,
                "longBinaryMinimumIoU": 0.85,
            },
            "rescueRules": {
                "durationSeconds": [
                    SHORT_RESCUE_MIN_SECONDS,
                    SHORT_RESCUE_MAX_SECONDS,
                ],
                "maximumPerRecording": 2,
                "requiresIndependentBinarySupport": True,
                "requiresIsolatedFlanks": True,
            },
            "objectiveMargin": objective_margin,
            "signConsistency": sign_consistency,
            "folds": [fold.to_dict() for fold in folds],
        },
        "sameFeatureBinaryControl": {"oof": control_oof},
        "conservativeHybrid": {
            "oof": hybrid_oof,
            "outerFolds": outer_rows,
        },
        "pairedComparison": paired,
        "promotionDecision": {
            **gate,
            "selectedArchitecture": (
                "conservative_multistate_hybrid"
                if gate["promoteConservativeHybrid"]
                else "binary_control"
            ),
            "testMetricsConsulted": False,
        },
        "finalizationPlan": {
            "architecture": (
                "conservative_multistate_hybrid"
                if gate["promoteConservativeHybrid"]
                else "binary_control"
            ),
            "hybridMode": final_mode,
            "hybridModeSelection": (
                "most frequent outer-fold selection; exact ties prefer frozen simpler order"
            ),
            "sourceOuterSelections": dict(
                zip(
                    [fold.held_out_group for fold in folds],
                    selected_modes,
                    strict=True,
                )
            ),
            "binaryControl": frozen["finalizationPlan"]["binaryControl"],
            "multistate": frozen["finalizationPlan"]["multistate"],
            "featureNames": list(frozen_names),
            "featureSignatureSha256": _signature_sha256(frozen_names),
        },
        "guardrails": [
            "Only train and validation recordings are prepared.",
            "Every binary and state model fingerprint reproduces the frozen nested study.",
            "Hybrid mode is selected on inner OOF predictions independently in every outer fold.",
            "No hybrid mode deletes a binary proposal; long proposals have structural coverage and IoU floors.",
            "No-op is always available and must reproduce frozen binary metrics exactly.",
            "Protected test remains closed unless every final promotion check passes.",
        ],
        "provenance": _code_provenance(),
        "processing": {
            "wallClockSeconds": round(time.perf_counter() - started, 3),
            "preparedSplits": sorted(DEVELOPMENT_SPLITS),
            "protectedSplitsPrepared": False,
        },
    }


def _predicted_binary_intervals(
    item: PreparedRecording,
    probabilities: np.ndarray,
    decoder: DecoderConfig,
) -> list[Interval]:
    analysis_fps = (
        1.0 / float(np.median(np.diff(item.sequence.times)))
        if len(item.sequence.times) > 1
        else 1.0
    )
    decoded, _ = decode_probabilities(
        item.sequence.times,
        probabilities,
        item.sequence.metadata.duration,
        decoder,
        analysis_fps,
    )
    return _clip_ignored_predictions(
        [Interval(value.start, value.end) for value in decoded],
        item.recording.ignored_intervals,
    )


def _oracle_replace_boundaries(
    truth: Sequence[Interval],
    predictions: Sequence[Interval],
    *,
    replace_start: bool,
    replace_end: bool,
) -> list[Interval]:
    """Replace boundaries only for chronologically matched overlapping events."""

    replacements = {
        prediction_index: truth_index
        for truth_index, prediction_index, _ in ordered_interval_matches(
            truth, predictions, minimum_iou=1e-12
        )
    }
    result: list[Interval] = []
    for prediction_index, prediction in enumerate(predictions):
        truth_index = replacements.get(prediction_index)
        if truth_index is None:
            result.append(prediction)
            continue
        actual = truth[truth_index]
        start = actual.start if replace_start else prediction.start
        end = actual.end if replace_end else prediction.end
        if end > start:
            result.append(Interval(start=start, end=end))
    return result


def _force_gold_state_anchors(
    item: PreparedRecording,
    log_scores: np.ndarray,
    *,
    force_serves: bool,
    force_ends: bool,
    force_all_states: bool = False,
) -> np.ndarray:
    """Return label-forced emissions for development-only oracle diagnosis.

    A forced row assigns zero log score to its annotated state and negative
    infinity to every alternative. Serve rows come from ``state_targets``.
    End rows use the first sample at or after each annotated end, matching the
    decoder's half-open interval convention. ``force_all_states`` is a gold-path
    sanity ceiling, not a candidate.
    """

    scores = np.asarray(log_scores, dtype=np.float64).copy()
    if scores.shape != (len(item.sequence.times), len(STATE_ORDER)):
        raise ValueError("oracle scores must align with the prepared recording")
    targets = state_targets(item)
    forced: dict[int, MultistateState] = {}
    if force_all_states:
        forced.update(
            (index, MultistateState(int(target)))
            for index, target in enumerate(targets)
        )
    else:
        if force_serves:
            forced.update(
                (int(index), MultistateState.SERVE)
                for index in np.flatnonzero(targets == int(MultistateState.SERVE))
            )
        if force_ends:
            times = item.sequence.times
            for rally in item.recording.rallies:
                candidates = np.flatnonzero(times >= rally.end)
                if len(candidates):
                    forced[int(candidates[0])] = MultistateState.DEAD
    for index, state in forced.items():
        scores[index, :] = -math.inf
        scores[index, int(state)] = 0.0
    return scores


def _evaluated_row(
    item: PreparedRecording, predictions: Sequence[Interval]
) -> dict[str, Any]:
    row = evaluate_intervals(item.recording.rallies, predictions)
    row["outcomeSlices"] = outcome_slice_metrics(item.recording.rallies, predictions)
    row["id"] = item.recording.id
    row["sourceGroup"] = item.recording.source_group
    row["environment"] = item.recording.environment
    return row


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_evaluations(rows)
    aggregate["outcomeSlices"] = aggregate_outcome_slices(
        [row["outcomeSlices"] for row in rows]
    )
    aggregate["objective"] = objective(aggregate)
    return aggregate


def _outcome_name(rally: Interval) -> str:
    tags = set(rally.tags)
    if "service-fault" in tags:
        return "serviceFault"
    if "ace" in tags:
        return "ace"
    if rally.end - rally.start <= 3.0:
        return "otherShort"
    return "ordinaryLong"


def _serve_rows(
    item: PreparedRecording,
    probabilities: np.ndarray,
    decoded_states: Sequence[MultistateState],
    predictions: Sequence[Interval],
) -> list[dict[str, Any]]:
    times = item.sequence.times
    valid = np.flatnonzero(item.sample_mask)
    decoded_serves = np.flatnonzero(
        np.asarray(decoded_states, dtype=np.int64) == int(MultistateState.SERVE)
    )
    strict_truth = {
        match[0]
        for match in ordered_interval_matches(item.recording.rallies, predictions, 0.5)
    }
    overlap_truth = {
        match[0]
        for match in ordered_interval_matches(
            item.recording.rallies, predictions, minimum_iou=1e-12
        )
    }
    rows = []
    for index, rally in enumerate(item.recording.rallies):
        eligible = valid[times[valid] < rally.end]
        if not len(eligible):
            continue
        serve_index = int(eligible[np.argmin(np.abs(times[eligible] - rally.start))])
        nearest_distance = (
            float(np.min(np.abs(times[decoded_serves] - rally.start)))
            if len(decoded_serves)
            else math.inf
        )
        rows.append(
            {
                "recordingId": item.recording.id,
                "sourceGroup": item.recording.source_group,
                "outcome": _outcome_name(rally),
                "durationSeconds": rally.end - rally.start,
                "serveSampleTime": float(times[serve_index]),
                "serveSampleStateProbabilities": {
                    state.name: float(probabilities[serve_index, int(state)])
                    for state in STATE_ORDER
                },
                "serveSampleArgmaxState": STATE_ORDER[
                    int(np.argmax(probabilities[serve_index]))
                ].name,
                "nearestDecodedServeDistanceSeconds": (
                    nearest_distance if math.isfinite(nearest_distance) else None
                ),
                "decodedServeWithin": {
                    f"{window:g}Seconds": nearest_distance <= window
                    for window in SERVE_WINDOWS_SECONDS
                },
                "anyOverlap": index in overlap_truth,
                "strictMatch": index in strict_truth,
            }
        )
    return rows


def _summarize_serve_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def summarize(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
        count = len(selected)
        if not count:
            return {"rallies": 0}
        serve_probabilities = [
            row["serveSampleStateProbabilities"]["SERVE"] for row in selected
        ]
        return {
            "rallies": count,
            "serveArgmaxCount": sum(
                row["serveSampleArgmaxState"] == "SERVE" for row in selected
            ),
            "meanServeProbabilityAtTrueAnchor": float(
                np.mean(serve_probabilities)
            ),
            "decodedServeWithin": {
                f"{window:g}Seconds": sum(
                    row["decodedServeWithin"][f"{window:g}Seconds"]
                    for row in selected
                )
                for window in SERVE_WINDOWS_SECONDS
            },
            "anyOverlapCount": sum(row["anyOverlap"] for row in selected),
            "strictMatchCount": sum(row["strictMatch"] for row in selected),
        }

    groups = sorted({row["sourceGroup"] for row in rows})
    outcomes = sorted({row["outcome"] for row in rows})
    return {
        "aggregate": summarize(rows),
        "bySourceGroup": {
            group: summarize([row for row in rows if row["sourceGroup"] == group])
            for group in groups
        },
        "byOutcome": {
            outcome: summarize([row for row in rows if row["outcome"] == outcome])
            for outcome in outcomes
        },
    }


def _cache_name(recording_id: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in recording_id
    )
    return f"{safe}-{hashlib.sha256(recording_id.encode()).hexdigest()[:10]}.npz"


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OOF cache: {path}")
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()


def _code_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    tracked = (
        root / "analysis" / "multistate_followup.py",
        root / "scripts" / "evaluate-multistate-followup.py",
        root / "analysis" / "multistate_feature_study.py",
        root / "analysis" / "multistate.py",
        root / "analysis" / "feature_experiments.py",
        root / "analysis" / "model.py",
        root / "analysis" / "metrics.py",
    )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        head = None
        dirty = None
    return {
        "gitHead": head,
        "gitDirty": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "filesSha256": {
            str(path.relative_to(root)): sha256_file(path)
            for path in tracked
            if path.is_file()
        },
    }


def run_multistate_oof_diagnostics(
    manifest: DatasetManifest,
    prepared: Sequence[PreparedRecording],
    *,
    multistate_report_path: str | Path,
    cache_output: str | Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Recreate frozen development OOF predictions and emit diagnostics/cache."""

    started = time.perf_counter()
    report_path, frozen, upstream_path, _upstream, frozen_names = (
        _validate_development_study(manifest, multistate_report_path)
    )
    development_ids = {
        item.id for item in manifest.recordings if item.split in DEVELOPMENT_SPLITS
    }
    if {item.recording.id for item in prepared} != development_ids or any(
        item.recording.split not in DEVELOPMENT_SPLITS for item in prepared
    ):
        raise FeatureExperimentError(
            "diagnostics require exactly the train+validation prepared recordings"
        )
    feature_config = FeatureConfig.from_dict(frozen["featureConfig"])
    training_config = TrainingConfig(**frozen["trainingConfig"])
    feature_config.validate()
    training_config.validate()
    selected = reconstruct_frozen_upstream_features(
        prepared, feature_config, frozen_names
    )
    if any(item.contextual_names != frozen_names for item in selected):
        raise FeatureExperimentError("reconstructed feature signature changed")

    cache_dir = Path(cache_output).expanduser().resolve()
    if cache_dir.exists():
        raise FileExistsError(f"refusing to overwrite OOF cache directory: {cache_dir}")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_context = tempfile.TemporaryDirectory(
        prefix=f".{cache_dir.name}-staging-", dir=cache_dir.parent
    )
    staging_dir = Path(staging_context.name)
    folds = build_fold_plan(manifest.recordings)
    state_reports = {
        row["heldOutSourceGroup"]: row
        for row in frozen["serveAnchoredMultistate"]["outerFolds"]
    }
    control_reports = {
        row["heldOutSourceGroup"]: row
        for row in frozen["sameFeatureBinaryControl"]["outerFolds"]
    }
    if set(state_reports) != {fold.held_out_group for fold in folds} or set(
        control_reports
    ) != set(state_reports):
        raise FeatureExperimentError("frozen outer-fold reports are incomplete")

    state_arrays_by_group: dict[str, list[np.ndarray]] = {}
    target_arrays_by_group: dict[str, list[np.ndarray]] = {}
    serve_detail_rows: list[dict[str, Any]] = []
    duration_rows: dict[str, dict[str, list[float]]] = {}
    baseline_rows: list[dict[str, Any]] = []
    oracle_start_rows: list[dict[str, Any]] = []
    oracle_end_rows: list[dict[str, Any]] = []
    oracle_both_rows: list[dict[str, Any]] = []
    forced_serve_rows: list[dict[str, Any]] = []
    forced_end_rows: list[dict[str, Any]] = []
    forced_both_rows: list[dict[str, Any]] = []
    gold_state_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for fold_index, fold in enumerate(folds, start=1):
        group = fold.held_out_group
        if progress is not None:
            progress(f"OOF diagnostic {fold_index}/{len(folds)}: refit {group}")
        training = _prepared_for_groups(selected, fold.training_groups)
        held = _prepared_for_groups(selected, (group,))
        state_frozen = state_reports[group]
        epoch_caps = {
            state: int(state_frozen["innerSelection"]["selectedEpochCaps"][state.name])
            for state in STATE_ORDER
        }
        state_bundle = fit_state_models(
            training,
            (),
            feature_config=feature_config,
            training_config=training_config,
            seed_parts=("multistate-outer-refit", group),
            epoch_caps=epoch_caps,
        )
        actual_state_fingerprints = {
            state.name: _model_fingerprint(model)
            for state, model in zip(STATE_ORDER, state_bundle.models, strict=True)
        }
        if actual_state_fingerprints != state_frozen["outerRefit"][
            "modelFingerprints"
        ]:
            raise FeatureExperimentError(f"state model reproduction failed for {group}")
        bonus = float(state_frozen["innerSelection"]["selectedTransitionBonus"])
        state_decoder, prior_summary = estimate_fold_decoder(
            training, transition_bonus=bonus
        )

        control_frozen = control_reports[group]
        control_decoder = DecoderConfig.from_dict(control_frozen["decoder"])
        control_epoch_cap = int(control_frozen["innerSelection"]["selectedEpochCap"])
        control_fit_config = replace(
            training_config,
            epochs=control_epoch_cap,
            patience=max(training_config.patience, control_epoch_cap + 1),
        )
        control_model = _fit(
            training,
            (),
            feature_config=feature_config,
            decoder=control_decoder,
            config=control_fit_config,
            seed=_seed(training_config.seed, "outer-refit", group),
        )
        control_fingerprint = _model_fingerprint(control_model)
        if control_fingerprint != control_frozen["outerRefit"]["modelFingerprint"]:
            raise FeatureExperimentError(f"binary model reproduction failed for {group}")

        group_baseline_rows: list[dict[str, Any]] = []
        group_control_rows: list[dict[str, Any]] = []
        group_duration = {
            "truthIntervals": [],
            "multistateIntervals": [],
            "binaryIntervals": [],
            **{f"decoded{state.name}Runs": [] for state in STATE_ORDER},
        }
        state_arrays_by_group[group] = []
        target_arrays_by_group[group] = []
        for item in held:
            raw_state_probabilities = np.column_stack(
                [model.predict(item.contextual_values) for model in state_bundle.models]
            ).astype(np.float64)
            log_scores = state_log_scores(state_bundle, item.contextual_values)
            state_probabilities = np.exp(log_scores)
            targets = state_targets(item)
            decoded = decode_multistate(item.sequence.times, log_scores, state_decoder)
            multistate_predictions = _predicted_multistate_intervals(item, decoded)
            control_probabilities = control_model.predict(item.contextual_values)
            control_predictions = _predicted_binary_intervals(
                item, control_probabilities, control_decoder
            )
            baseline_row = _evaluated_row(item, multistate_predictions)
            control_row = _evaluated_row(item, control_predictions)
            group_baseline_rows.append(baseline_row)
            group_control_rows.append(control_row)
            baseline_rows.append(baseline_row)

            oracle_start = _oracle_replace_boundaries(
                item.recording.rallies,
                multistate_predictions,
                replace_start=True,
                replace_end=False,
            )
            oracle_end = _oracle_replace_boundaries(
                item.recording.rallies,
                multistate_predictions,
                replace_start=False,
                replace_end=True,
            )
            oracle_both = _oracle_replace_boundaries(
                item.recording.rallies,
                multistate_predictions,
                replace_start=True,
                replace_end=True,
            )
            oracle_start_rows.append(_evaluated_row(item, oracle_start))
            oracle_end_rows.append(_evaluated_row(item, oracle_end))
            oracle_both_rows.append(_evaluated_row(item, oracle_both))
            forced_decodes = {
                "serve": decode_multistate(
                    item.sequence.times,
                    _force_gold_state_anchors(
                        item, log_scores, force_serves=True, force_ends=False
                    ),
                    state_decoder,
                ),
                "end": decode_multistate(
                    item.sequence.times,
                    _force_gold_state_anchors(
                        item, log_scores, force_serves=False, force_ends=True
                    ),
                    state_decoder,
                ),
                "both": decode_multistate(
                    item.sequence.times,
                    _force_gold_state_anchors(
                        item, log_scores, force_serves=True, force_ends=True
                    ),
                    state_decoder,
                ),
                "gold": decode_multistate(
                    item.sequence.times,
                    _force_gold_state_anchors(
                        item,
                        log_scores,
                        force_serves=False,
                        force_ends=False,
                        force_all_states=True,
                    ),
                    state_decoder,
                ),
            }
            forced_rows_by_name = {
                "serve": forced_serve_rows,
                "end": forced_end_rows,
                "both": forced_both_rows,
                "gold": gold_state_rows,
            }
            for name, forced_decode in forced_decodes.items():
                forced_predictions = _predicted_multistate_intervals(
                    item, forced_decode
                )
                forced_rows_by_name[name].append(
                    _evaluated_row(item, forced_predictions)
                )
            serve_detail_rows.extend(
                _serve_rows(
                    item,
                    state_probabilities,
                    decoded.states,
                    multistate_predictions,
                )
            )
            group_duration["truthIntervals"].extend(
                rally.end - rally.start for rally in item.recording.rallies
            )
            group_duration["multistateIntervals"].extend(
                interval.end - interval.start for interval in multistate_predictions
            )
            group_duration["binaryIntervals"].extend(
                interval.end - interval.start for interval in control_predictions
            )
            run_durations = _state_run_durations(item.sequence.times, decoded.states)
            for state in STATE_ORDER:
                group_duration[f"decoded{state.name}Runs"].extend(
                    run_durations[state.name]
                )

            valid_targets = targets[item.sample_mask]
            valid_probabilities = state_probabilities[item.sample_mask]
            target_arrays_by_group[group].append(valid_targets)
            state_arrays_by_group[group].append(valid_probabilities)
            cache_name = _cache_name(item.recording.id)
            cache_path = staging_dir / cache_name
            _write_npz(
                cache_path,
                times=np.asarray(item.sequence.times, dtype=np.float64),
                sample_mask=np.asarray(item.sample_mask, dtype=np.bool_),
                state_targets=np.asarray(targets, dtype=np.int8),
                raw_state_head_probabilities=np.asarray(
                    raw_state_probabilities, dtype=np.float32
                ),
                state_probabilities=np.asarray(state_probabilities, dtype=np.float32),
                multistate_states=np.asarray(decoded.states, dtype=np.int8),
                binary_probabilities=np.asarray(control_probabilities, dtype=np.float32),
            )
            cache_rows.append(
                {
                    "recordingId": item.recording.id,
                    "split": item.recording.split,
                    "sourceGroup": group,
                    "contentSha256": item.recording.content_sha256,
                    "file": cache_name,
                    "fileSha256": sha256_file(cache_path),
                    "samples": len(item.sequence.times),
                }
            )

        expected_state = state_frozen["metrics"]
        actual_state = _aggregate_rows(group_baseline_rows)
        expected_control = control_frozen["metrics"]
        actual_control = _aggregate_rows(group_control_rows)
        for label, expected, actual in (
            ("multistate", expected_state, actual_state),
            ("binary", expected_control, actual_control),
        ):
            for metric in ("eventF1", "timeIoU", "liveTimeRecall", "objective"):
                if not math.isclose(
                    float(expected[metric]), float(actual[metric]), abs_tol=1e-10
                ):
                    raise FeatureExperimentError(
                        f"{label} metric reproduction failed for {group}: {metric}"
                    )
        duration_rows[group] = group_duration
        fold_rows.append(
            {
                "heldOutSourceGroup": group,
                "trainingSourceGroups": list(fold.training_groups),
                "stateModelFingerprints": actual_state_fingerprints,
                "stateHeadClassBalance": {
                    state.name: {
                        key: model.training_summary[key]
                        for key in (
                            "positiveSamples",
                            "negativeSamples",
                            "positiveClassWeight",
                            "negativeClassWeight",
                        )
                    }
                    for state, model in zip(
                        STATE_ORDER, state_bundle.models, strict=True
                    )
                },
                "binaryModelFingerprint": control_fingerprint,
                "selectedStateEpochCaps": {
                    state.name: epoch_caps[state] for state in STATE_ORDER
                },
                "selectedTransitionBonus": bonus,
                "durationAndTransitionPriorEstimate": prior_summary,
                "binaryEpochCap": control_epoch_cap,
                "binaryDecoder": control_decoder.to_dict(),
                "reproducedMetrics": {
                    "multistate": actual_state,
                    "binary": actual_control,
                },
            }
        )

    all_targets = np.concatenate(
        [values for group in sorted(target_arrays_by_group) for values in target_arrays_by_group[group]]
    )
    all_probabilities = np.concatenate(
        [values for group in sorted(state_arrays_by_group) for values in state_arrays_by_group[group]],
        axis=0,
    )
    calibration_by_group = {
        group: state_classification_report(
            np.concatenate(target_arrays_by_group[group]),
            np.concatenate(state_arrays_by_group[group], axis=0),
        )
        for group in sorted(state_arrays_by_group)
    }
    cache_index = {
        "schemaVersion": FOLLOWUP_SCHEMA_VERSION,
        "kind": OOF_CACHE_KIND,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "assessmentRole": "development-only-source-group-out-of-fold-reusable-scores",
        "protectedSplitsIncluded": False,
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "manifestSnapshotSha256": _manifest_digest(manifest),
        "multistateDevelopmentReport": str(report_path),
        "multistateDevelopmentReportSha256": sha256_file(report_path),
        "upstreamDevelopmentReport": str(upstream_path),
        "upstreamDevelopmentReportSha256": sha256_file(upstream_path),
        "featureVersion": FEATURE_VERSION,
        "featureConfig": feature_config.to_dict(),
        "trainingConfig": training_config.to_dict(),
        "featureNames": list(frozen_names),
        "featureSignatureSha256": _signature_sha256(frozen_names),
        "outerFolds": fold_rows,
        "recordings": cache_rows,
        "provenance": _code_provenance(),
    }
    staging_index_path = atomic_write_text(
        staging_dir / "oof-cache.json",
        json.dumps(cache_index, indent=2, allow_nan=False) + "\n",
    )
    oracle_aggregates = {
        "unalteredMultistate": _aggregate_rows(baseline_rows),
        "oracleServeBoundaryOnly": _aggregate_rows(oracle_start_rows),
        "oracleEndBoundaryOnly": _aggregate_rows(oracle_end_rows),
        "oracleBothBoundaries": _aggregate_rows(oracle_both_rows),
        "forcedGoldServeState": _aggregate_rows(forced_serve_rows),
        "forcedGoldEndDeadState": _aggregate_rows(forced_end_rows),
        "forcedGoldServeAndEndStates": _aggregate_rows(forced_both_rows),
        "goldStatePathSanityCeiling": _aggregate_rows(gold_state_rows),
    }
    gold_metrics = oracle_aggregates["goldStatePathSanityCeiling"]
    if not (
        gold_metrics["trueRallies"]
        == gold_metrics["predictedRallies"]
        == gold_metrics["matchedRallies"]
    ):
        raise FeatureExperimentError(
            "gold state path failed the graph/path-extraction sanity assertion"
        )
    baseline_objective = oracle_aggregates["unalteredMultistate"]["objective"]
    for name, metrics in oracle_aggregates.items():
        metrics["deltaObjectiveVsUnaltered"] = metrics["objective"] - baseline_objective
        metrics["oracleUsesDevelopmentLabels"] = name != "unalteredMultistate"
    staging_dir.replace(cache_dir)
    cache_index_path = cache_dir / staging_index_path.name

    return {
        "schemaVersion": FOLLOWUP_SCHEMA_VERSION,
        "kind": DIAGNOSTIC_KIND,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "freezeStatus": "diagnostic-only-no-inference-candidate",
        "assessmentRole": "development-only-source-group-out-of-fold-diagnosis",
        "testLabelsUsed": False,
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "manifestSnapshotSha256": _manifest_digest(manifest),
        "recordingContentSha256": {
            item.id: item.content_sha256 for item in manifest.recordings
        },
        "multistateDevelopmentReport": str(report_path),
        "multistateDevelopmentReportSha256": sha256_file(report_path),
        "oofCacheIndex": str(cache_index_path),
        "oofCacheIndexSha256": sha256_file(cache_index_path),
        "stateClassificationAndCalibration": {
            "aggregate": state_classification_report(
                all_targets, all_probabilities
            ),
            "bySourceGroup": calibration_by_group,
            "note": (
                "Argmax precision/recall and one-vs-rest calibration use only each "
                "recording's outer-fold model. Independently balanced sigmoid heads "
                "are normalized exactly as in the frozen study."
            ),
        },
        "serveRejection": {
            **_summarize_serve_rows(serve_detail_rows),
            "rows": serve_detail_rows,
            "definition": (
                "A rejected true serve has no decoded SERVE state within the named "
                "absolute time window; strict/overlap status comes from unaltered output."
            ),
        },
        "decodedDurations": {
            "aggregate": {
                key: _quantiles(
                    [
                        value
                        for group in sorted(duration_rows)
                        for value in duration_rows[group][key]
                    ]
                )
                for key in next(iter(duration_rows.values()))
            },
            "bySourceGroup": {
                group: {key: _quantiles(values) for key, values in payload.items()}
                for group, payload in sorted(duration_rows.items())
            },
        },
        "boundaryOracleAblations": {
            **oracle_aggregates,
            "protocol": (
                "For chronological prediction/truth pairs with any positive overlap, "
                "replace only the named boundary with its development annotation. "
                "Counts and unmatched events are preserved. These are diagnostic upper "
                "bounds and must never be treated as deployable inference evidence. "
                "The forced-state rows instead constrain the decoder emission at every "
                "annotated SERVE and/or first post-end DEAD sample, so they can recover "
                "events rejected by the original path. The gold-state row forces every "
                "development target and serves only as a graph/path-extraction sanity ceiling."
            ),
        },
        "foldReproduction": fold_rows,
        "provenance": _code_provenance(),
        "processing": {
            "wallClockSeconds": round(time.perf_counter() - started, 3),
            "preparedSplits": sorted(DEVELOPMENT_SPLITS),
            "protectedSplitsPrepared": False,
        },
        "guardrails": [
            "The source report must be full nested, frozen, and unopened-test.",
            "Every state and binary model fingerprint must reproduce exactly.",
            "Every aggregate fold metric must reproduce the frozen report.",
            "Only source-group outer-fold predictions enter calibration and diagnostics.",
            "Oracle boundaries use development labels only and cannot be promoted.",
            "The OOF cache is assessment-only; selectable follow-ups need inner-OOF tuning inside every outer fold.",
            "Raw balanced-head sigmoid scores are retained for fold-local calibration; normalized scores alone are insufficient.",
        ],
    }
