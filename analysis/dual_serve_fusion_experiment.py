from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .artifacts import atomic_write_text
from .decoder import DecodedInterval
from .metrics import (
    aggregate_evaluations,
    aggregate_outcome_slices,
    evaluate_intervals,
    interval_iou,
    ordered_interval_matches,
    outcome_slice_metrics,
)
from .model import LogisticModel, ModelError, load_model
from .pipeline import (
    PreparedRecording,
    _manifest_digest,
    _prepare_many,
    assessment_role_for_split,
)
from .schema import Interval, Recording, load_manifest
from .serve_experiment import (
    PairedPrediction,
    _clip_ignored,
    _prediction_inputs,
    _validate_model_pair,
)
from .version import __version__


EXPERIMENT_ID = "dual-serve-v4-v5-fusion-v1"
BOUNDARY_ACTIONS = {"keep-v4", "v5", "intersection"}


@dataclass(frozen=True)
class BoundarySelectorConfig:
    action: str = "keep-v4"
    minimum_pair_iou: float = 0.0
    maximum_start_delta_seconds: float = 1.0
    minimum_v5_end_earlier_seconds: float = 0.0
    minimum_both_confidence: float = 0.0

    def validate(self) -> None:
        if self.action not in BOUNDARY_ACTIONS:
            raise ValueError(f"unsupported boundary action: {self.action!r}")
        values = (
            self.minimum_pair_iou,
            self.maximum_start_delta_seconds,
            self.minimum_v5_end_earlier_seconds,
            self.minimum_both_confidence,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("boundary selector thresholds must be finite and non-negative")
        if self.minimum_pair_iou > 1.0 or self.minimum_both_confidence > 1.0:
            raise ValueError("boundary IoU and confidence thresholds cannot exceed one")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "minimumPairIoU": self.minimum_pair_iou,
            "maximumStartDeltaSeconds": self.maximum_start_delta_seconds,
            "minimumV5EndEarlierSeconds": self.minimum_v5_end_earlier_seconds,
            "minimumBothConfidence": self.minimum_both_confidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BoundarySelectorConfig:
        try:
            result = cls(
                action=str(payload["action"]),
                minimum_pair_iou=float(payload["minimumPairIoU"]),
                maximum_start_delta_seconds=float(
                    payload["maximumStartDeltaSeconds"]
                ),
                minimum_v5_end_earlier_seconds=float(
                    payload["minimumV5EndEarlierSeconds"]
                ),
                minimum_both_confidence=float(payload["minimumBothConfidence"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid boundary selector: {error}") from error
        result.validate()
        return result


@dataclass(frozen=True)
class OverlapComponent:
    v4: tuple[DecodedInterval, ...]
    v5: tuple[DecodedInterval, ...]


@dataclass(frozen=True)
class DualPrediction:
    recording: Recording
    v4: PairedPrediction
    v5: PairedPrediction
    v4_live_scores: np.ndarray
    v4_serve_scores: np.ndarray
    v5_live_scores: np.ndarray
    v5_serve_scores: np.ndarray


def overlap_components(
    v4: Sequence[DecodedInterval], v5: Sequence[DecodedInterval]
) -> list[OverlapComponent]:
    """Return strict-overlap connected components; touching intervals stay separate."""
    nodes = sorted(
        [(item.start, item.end, 0, item) for item in v4]
        + [(item.start, item.end, 1, item) for item in v5],
        key=lambda row: (row[0], row[1], row[2]),
    )
    components: list[OverlapComponent] = []
    current: list[tuple[float, float, int, DecodedInterval]] = []
    current_end = -math.inf
    for node in nodes:
        if current and node[0] >= current_end:
            components.append(
                OverlapComponent(
                    tuple(row[3] for row in current if row[2] == 0),
                    tuple(row[3] for row in current if row[2] == 1),
                )
            )
            current = []
            current_end = -math.inf
        current.append(node)
        current_end = max(current_end, node[1])
    if current:
        components.append(
            OverlapComponent(
                tuple(row[3] for row in current if row[2] == 0),
                tuple(row[3] for row in current if row[2] == 1),
            )
        )
    return components


def apply_boundary_selector(
    v4: Sequence[DecodedInterval],
    v5: Sequence[DecodedInterval],
    config: BoundarySelectorConfig,
) -> tuple[list[DecodedInterval], int]:
    """Refine only unambiguous one-to-one overlap components; default to v4."""
    config.validate()
    result: list[DecodedInterval] = []
    changed = 0
    for component in overlap_components(v4, v5):
        if len(component.v4) != 1 or len(component.v5) != 1:
            result.extend(component.v4)
            continue
        original, candidate = component.v4[0], component.v5[0]
        eligible = (
            config.action != "keep-v4"
            and interval_iou(original, candidate) + 1e-12
            >= config.minimum_pair_iou
            and abs(candidate.start - original.start)
            <= config.maximum_start_delta_seconds + 1e-12
            and original.end - candidate.end + 1e-12
            >= config.minimum_v5_end_earlier_seconds
            and min(original.confidence, candidate.confidence) + 1e-12
            >= config.minimum_both_confidence
        )
        if not eligible:
            result.append(original)
            continue
        if config.action == "v5":
            selected = candidate
        else:
            selected = DecodedInterval(
                max(original.start, candidate.start),
                min(original.end, candidate.end),
                min(original.confidence, candidate.confidence),
            )
        if selected.end <= selected.start:
            result.append(original)
            continue
        result.append(selected)
        changed += 1
    return sorted(result, key=lambda item: (item.start, item.end)), changed


def add_only_candidates(
    v4: Sequence[DecodedInterval], v5: Sequence[DecodedInterval]
) -> list[DecodedInterval]:
    return [
        candidate
        for candidate in v5
        if not any(
            candidate.start < original.end and original.start < candidate.end
            for original in v4
        )
    ]


def _clip_decoded(
    intervals: Sequence[DecodedInterval], ignored: Sequence[Interval]
) -> list[DecodedInterval]:
    fragments = list(intervals)
    for blocked in ignored:
        clipped: list[DecodedInterval] = []
        for item in fragments:
            if item.end <= blocked.start or item.start >= blocked.end:
                clipped.append(item)
                continue
            if item.start < blocked.start:
                clipped.append(
                    DecodedInterval(item.start, blocked.start, item.confidence)
                )
            if item.end > blocked.end:
                clipped.append(DecodedInterval(blocked.end, item.end, item.confidence))
        fragments = clipped
    return fragments


def _report(
    predictions: Sequence[DualPrediction],
    values: dict[str, Sequence[DecodedInterval]],
) -> dict[str, Any]:
    recordings: list[dict[str, Any]] = []
    for prediction in predictions:
        scored = _clip_ignored(
            values[prediction.recording.id], prediction.recording.ignored_intervals
        )
        metrics = evaluate_intervals(prediction.recording.rallies, scored)
        metrics["outcomeSlices"] = outcome_slice_metrics(
            prediction.recording.rallies, scored
        )
        metrics.update(
            {
                "id": prediction.recording.id,
                "environment": prediction.recording.environment,
            }
        )
        recordings.append(metrics)
    aggregate = aggregate_evaluations(recordings)
    aggregate["outcomeSlices"] = aggregate_outcome_slices(
        [item["outcomeSlices"] for item in recordings]
    )
    return {"aggregate": aggregate, "recordings": recordings}


def _slice_matches(aggregate: dict[str, Any], name: str) -> int:
    row = aggregate["outcomeSlices"][name]
    return round(float(row.get("strictMatchRecall", 0.0)) * int(row["rallies"]))


def _selector_grid() -> tuple[BoundarySelectorConfig, ...]:
    candidates = [BoundarySelectorConfig()]
    for action in ("v5", "intersection"):
        for pair_iou in (0.3, 0.5):
            for end_earlier in (0.5, 1.0):
                for confidence in (0.8, 0.9):
                    candidates.append(
                        BoundarySelectorConfig(
                            action=action,
                            minimum_pair_iou=pair_iou,
                            maximum_start_delta_seconds=1.0,
                            minimum_v5_end_earlier_seconds=end_earlier,
                            minimum_both_confidence=confidence,
                        )
                    )
    return tuple(candidates)


def _candidate_summary(
    config: BoundarySelectorConfig,
    report: dict[str, Any],
    changed: int,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    aggregate = report["aggregate"]
    guardrails = {
        "predictionCountUnchanged": aggregate["predictedRallies"]
        == baseline["predictedRallies"],
        "precisionNotLower": aggregate["eventPrecision"] + 1e-12
        >= baseline["eventPrecision"],
        "ordinaryLongStrictMatchesNotLower": _slice_matches(
            aggregate, "ordinaryLong"
        )
        >= _slice_matches(baseline, "ordinaryLong"),
    }
    return {
        "config": config.to_dict(),
        "changedComponents": changed,
        "guardrails": guardrails,
        "feasible": all(guardrails.values()),
        "metrics": {
            key: aggregate[key]
            for key in (
                "predictedRallies",
                "matchedRallies",
                "eventPrecision",
                "eventRecall",
                "eventF1",
                "timeIoU",
                "liveTimeRecall",
                "liveTimePrecision",
                "deadSecondsRetained",
            )
        },
        "strictMatches": {
            name: _slice_matches(aggregate, name)
            for name in ("shortAtMost3Seconds", "ace", "serviceFault", "ordinaryLong")
        },
    }


def _selection_rank(row: dict[str, Any], index: int) -> tuple[Any, ...]:
    metrics = row["metrics"]
    slices = row["strictMatches"]
    return (
        bool(row["feasible"]),
        float(metrics["eventF1"]),
        int(slices["shortAtMost3Seconds"]),
        int(slices["serviceFault"]),
        float(metrics["timeIoU"]),
        -index,
    )


def _boundary_variants(
    predictions: Sequence[DualPrediction],
    *,
    selected_config: BoundarySelectorConfig | None = None,
) -> tuple[dict[str, Any], BoundarySelectorConfig, list[dict[str, Any]]]:
    baseline_values = {
        item.recording.id: _clip_decoded(
            item.v4.composed, item.recording.ignored_intervals
        )
        for item in predictions
    }
    baseline = _report(predictions, baseline_values)
    rows: list[dict[str, Any]] = []
    configs = _selector_grid()
    for config in configs:
        values: dict[str, Sequence[DecodedInterval]] = {}
        changed = 0
        for item in predictions:
            v4 = _clip_decoded(item.v4.composed, item.recording.ignored_intervals)
            v5 = _clip_decoded(item.v5.composed, item.recording.ignored_intervals)
            selected, count = apply_boundary_selector(
                v4, v5, config
            )
            values[item.recording.id] = selected
            changed += count
        rows.append(
            _candidate_summary(
                config, _report(predictions, values), changed, baseline["aggregate"]
            )
        )
    if selected_config is None:
        selected_index = max(
            range(len(rows)), key=lambda index: _selection_rank(rows[index], index)
        )
        selected_config = configs[selected_index]
    else:
        selected_config.validate()
    selected_values: dict[str, Sequence[DecodedInterval]] = {}
    selected_changed = 0
    for item in predictions:
        v4 = _clip_decoded(item.v4.composed, item.recording.ignored_intervals)
        v5 = _clip_decoded(item.v5.composed, item.recording.ignored_intervals)
        selected, count = apply_boundary_selector(
            v4, v5, selected_config
        )
        selected_values[item.recording.id] = selected
        selected_changed += count
    selected = _report(predictions, selected_values)
    selected["changedComponents"] = selected_changed
    return selected, selected_config, rows


def _full_boundary_ablation(
    predictions: Sequence[DualPrediction], action: str
) -> dict[str, Any]:
    values: dict[str, Sequence[DecodedInterval]] = {}
    changed = 0
    for item in predictions:
        selected: list[DecodedInterval] = []
        v4 = _clip_decoded(item.v4.composed, item.recording.ignored_intervals)
        v5 = _clip_decoded(item.v5.composed, item.recording.ignored_intervals)
        for component in overlap_components(v4, v5):
            if len(component.v4) != 1 or len(component.v5) != 1:
                selected.extend(component.v4)
                continue
            original, candidate = component.v4[0], component.v5[0]
            if action == "v5":
                refined = candidate
            elif action == "intersection":
                refined = DecodedInterval(
                    max(original.start, candidate.start),
                    min(original.end, candidate.end),
                    min(original.confidence, candidate.confidence),
                )
            else:
                raise ValueError(f"unsupported full boundary ablation: {action!r}")
            if refined.end <= refined.start:
                selected.append(original)
            else:
                selected.append(refined)
                changed += 1
        values[item.recording.id] = sorted(
            selected, key=lambda interval: (interval.start, interval.end)
        )
    report = _report(predictions, values)
    report["changedComponents"] = changed
    return report


def _score_window(
    prepared: PreparedRecording,
    scores: np.ndarray,
    start: float,
    end: float,
) -> tuple[float | None, float | None]:
    mask = (prepared.sequence.times >= start) & (prepared.sequence.times < end)
    if not np.any(mask):
        return None, None
    selected = scores[mask]
    return float(np.mean(selected)), float(np.max(selected))


def _candidate_evidence(
    prediction: DualPrediction, candidate: DecodedInterval
) -> dict[str, Any]:
    truth = prediction.recording.rallies
    best_index = max(
        range(len(truth)),
        key=lambda index: interval_iou(candidate, truth[index]),
        default=None,
    )
    best_iou = interval_iou(candidate, truth[best_index]) if best_index is not None else 0.0
    v4_mean, v4_max = _score_window(
        prediction.v4.prepared,
        prediction.v4_live_scores,
        candidate.start,
        candidate.end,
    )
    v5_mean, v5_max = _score_window(
        prediction.v5.prepared,
        prediction.v5_live_scores,
        candidate.start,
        candidate.end,
    )
    _, v4_serve = _score_window(
        prediction.v4.prepared,
        prediction.v4_serve_scores,
        candidate.start - 1.0,
        candidate.start + 1.0,
    )
    _, v5_serve = _score_window(
        prediction.v5.prepared,
        prediction.v5_serve_scores,
        candidate.start - 1.0,
        candidate.start + 1.0,
    )
    edges = [
        abs(candidate.start - edge)
        for interval in prediction.v4.composed
        for edge in (interval.start, interval.end)
    ]
    baseline_truth = {
        truth_index
        for truth_index, _, _ in ordered_interval_matches(
            truth, prediction.v4.composed, 0.5
        )
    }
    return {
        "recordingId": prediction.recording.id,
        "start": candidate.start,
        "end": candidate.end,
        "durationSeconds": candidate.end - candidate.start,
        "v5CompositeConfidence": candidate.confidence,
        "v4ServeRawMaxWithin1Second": v4_serve,
        "v5ServeRawMaxWithin1Second": v5_serve,
        "v4LiveRawMean": v4_mean,
        "v4LiveRawMax": v4_max,
        "v5LiveRawMean": v5_mean,
        "v5LiveRawMax": v5_max,
        "nearestV4BoundarySeconds": min(edges) if edges else None,
        "bestTruthIoU": best_iou,
        "novelStrictMatch": best_index is not None
        and best_iou >= 0.5
        and best_index not in baseline_truth,
        "bestTruth": (
            {
                "index": best_index,
                "start": truth[best_index].start,
                "end": truth[best_index].end,
                "durationSeconds": truth[best_index].end - truth[best_index].start,
                "tags": list(truth[best_index].tags),
            }
            if best_index is not None and best_iou > 0.0
            else None
        ),
    }


def _add_only_audit(
    predictions: Sequence[DualPrediction], *, select_policy: bool
) -> dict[str, Any]:
    baseline_values = {
        item.recording.id: _clip_decoded(
            item.v4.composed, item.recording.ignored_intervals
        )
        for item in predictions
    }
    baseline = _report(predictions, baseline_values)
    evidence: list[dict[str, Any]] = []
    candidate_rows: dict[
        str, list[tuple[DecodedInterval, dict[str, Any]]]
    ] = {}
    for item in predictions:
        v4 = _clip_decoded(item.v4.composed, item.recording.ignored_intervals)
        v5 = _clip_decoded(item.v5.composed, item.recording.ignored_intervals)
        uncovered = add_only_candidates(v4, v5)
        rows = [
            (candidate, _candidate_evidence(item, candidate))
            for candidate in uncovered
        ]
        candidate_rows[item.recording.id] = rows
        evidence.extend(row for _, row in rows)

    gates: list[tuple[str, Callable[[dict[str, Any]], bool], dict[str, Any]]] = []
    for threshold in (0.0, 0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0):
        gates.append(
            (
                "minimum-v5-composite-confidence",
                lambda row, threshold=threshold: float(
                    row["v5CompositeConfidence"]
                )
                + 1e-12
                >= threshold,
                {"minimumV5CompositeConfidence": threshold},
            )
        )
    for threshold in (0.8, 0.9, 0.95):
        gates.append(
            (
                "minimum-both-serve-raw-max-within-1s",
                lambda row, threshold=threshold: min(
                    float(row["v4ServeRawMaxWithin1Second"]),
                    float(row["v5ServeRawMaxWithin1Second"]),
                )
                + 1e-12
                >= threshold,
                {"minimumBothServeRawMaxWithin1Second": threshold},
            )
        )
    for threshold in (0.0, 0.1, 0.2, 0.25, 0.3):
        gates.append(
            (
                "minimum-v5-minus-v4-live-raw-max",
                lambda row, threshold=threshold: float(row["v5LiveRawMax"])
                - float(row["v4LiveRawMax"])
                + 1e-12
                >= threshold,
                {"minimumV5MinusV4LiveRawMax": threshold},
            )
        )
    sweep: list[dict[str, Any]] = []
    for family, accepts, parameters in gates:
        values = {
            item.recording.id: tuple(
                sorted(
                    (
                        *baseline_values[item.recording.id],
                        *(
                            candidate
                            for candidate, row in candidate_rows[item.recording.id]
                            if accepts(row)
                        ),
                    ),
                    key=lambda interval: (interval.start, interval.end),
                )
            )
            for item in predictions
        }
        report = _report(predictions, values)
        sweep.append(
            {
                "gateFamily": family,
                "parameters": parameters,
                "admitted": sum(
                    accepts(row) for row in evidence
                ),
                "novelStrictMatches": sum(
                    row["novelStrictMatch"] and accepts(row) for row in evidence
                ),
                "metrics": {
                    key: report["aggregate"][key]
                    for key in (
                        "predictedRallies",
                        "matchedRallies",
                        "eventPrecision",
                        "eventRecall",
                        "eventF1",
                    )
                },
            }
        )
    viable = [
        row
        for row in sweep
        if row["metrics"]["matchedRallies"]
        > baseline["aggregate"]["matchedRallies"]
        and row["metrics"]["eventF1"]
        > baseline["aggregate"]["eventF1"] + 1e-12
    ]
    if select_policy:
        selected_policy = "disabled" if not viable else "unsupported-candidate"
        selection_reason = (
            "Validation contained no score gate with a novel strict match and higher "
            "event F1; the deterministic no-op avoids extra false rallies."
            if not viable
            else (
                "A score gate passed validation, but deployable add-only replay is "
                "not implemented."
            )
        )
    else:
        selected_policy = "audit-only"
        selection_reason = "The add-only policy is frozen in the validation decision."
    return {
        "selectedPolicy": selected_policy,
        "selectionReason": selection_reason,
        "baseline": baseline,
        "candidateEvidence": evidence,
        "scoreGateSweep": sweep,
    }


def _validate_recording_pair(v4: PreparedRecording, v5: PreparedRecording) -> None:
    if v4.recording.id != v5.recording.id:
        raise ModelError("v4 and v5 prepared recordings are misaligned")
    if v4.recording.content_sha256 != v5.recording.content_sha256:
        raise ModelError(f"v4 and v5 video snapshots differ for {v4.recording.id}")
    if v4.recording.rallies != v5.recording.rallies:
        raise ModelError(f"v4 and v5 truth differs for {v4.recording.id}")


def _prepare_predictions(
    rows: Sequence[Recording],
    v4_rally: LogisticModel,
    v4_serve: LogisticModel,
    v4_cache: str | Path,
    v5_rally: LogisticModel,
    v5_serve: LogisticModel,
    v5_cache: str | Path,
    *,
    progress: Callable[[str], None] | None,
) -> list[DualPrediction]:
    v4_decoder, v4_composition = _validate_model_pair(v4_rally, v4_serve)
    v5_decoder, v5_composition = _validate_model_pair(v5_rally, v5_serve)
    prepared_v4 = _prepare_many(
        rows, v4_rally.feature_config, v4_cache, progress=progress
    )
    prepared_v5 = _prepare_many(
        rows, v5_rally.feature_config, v5_cache, progress=progress
    )
    by_id_v5 = {item.recording.id: item for item in prepared_v5}
    if set(by_id_v5) != {item.recording.id for item in prepared_v4}:
        raise ModelError("v4 and v5 prepared recording sets differ")
    result: list[DualPrediction] = []
    for item_v4 in prepared_v4:
        item_v5 = by_id_v5[item_v4.recording.id]
        _validate_recording_pair(item_v4, item_v5)
        result.append(
            DualPrediction(
                recording=item_v4.recording,
                v4=_prediction_inputs(
                    item_v4, v4_rally, v4_serve, v4_decoder, v4_composition
                ),
                v5=_prediction_inputs(
                    item_v5, v5_rally, v5_serve, v5_decoder, v5_composition
                ),
                v4_live_scores=v4_rally.predict(item_v4.contextual_values),
                v4_serve_scores=v4_serve.predict(item_v4.contextual_values),
                v5_live_scores=v5_rally.predict(item_v5.contextual_values),
                v5_serve_scores=v5_serve.predict(item_v5.contextual_values),
            )
        )
    return result


def _model_bindings(
    v4_rally: LogisticModel,
    v4_serve: LogisticModel,
    v5_rally: LogisticModel,
    v5_serve: LogisticModel,
) -> dict[str, dict[str, str | None]]:
    return {
        "v4Rally": {"sha256": v4_rally.artifact_sha256},
        "v4Serve": {"sha256": v4_serve.artifact_sha256},
        "v5Rally": {"sha256": v5_rally.artifact_sha256},
        "v5Serve": {"sha256": v5_serve.artifact_sha256},
    }


def _check_evaluation_groups(
    split: str,
    rows: Sequence[Recording],
    models: Sequence[LogisticModel],
) -> None:
    evaluated = {row.source_group for row in rows}
    for model in models:
        protected = set(model.training_summary.get("trainingSourceGroups", []))
        if split != "validation":
            protected |= set(model.training_summary.get("validationSourceGroups", []))
        overlap = evaluated & protected
        if overlap:
            raise ModelError(
                f"evaluation split leaks trained/tuned source groups: {sorted(overlap)}"
            )


def _validate_decision(
    payload: dict[str, Any], manifest_sha256: str, models: dict[str, Any]
) -> BoundarySelectorConfig:
    if payload.get("experiment") != EXPERIMENT_ID or payload.get("split") != "validation":
        raise ModelError("fusion decision must be a validation report from this experiment")
    if payload.get("manifestSha256") != manifest_sha256:
        raise ModelError("fusion decision differs from the immutable manifest")
    if payload.get("models") != models:
        raise ModelError("fusion decision is bound to different model artifacts")
    selection = payload.get("selection", {})
    if not isinstance(selection, dict):
        raise ModelError("fusion decision selection metadata is invalid")
    if selection.get("selectedOn") != "validation":
        raise ModelError("fusion decision was not selected on validation")
    if selection.get("addOnly", {}).get("selectedPolicy") != "disabled":
        raise ModelError("unsupported add-only policy in fusion decision")
    try:
        config = BoundarySelectorConfig.from_dict(
            selection["boundarySelector"]["selectedConfig"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ModelError(f"fusion decision has an invalid boundary selector: {error}") from error
    if config not in _selector_grid():
        raise ModelError("fusion decision selector was not in the declared validation grid")
    return config


def evaluate_dual_serve_fusion_dataset(
    manifest_path: str | Path,
    v4_rally_model_path: str | Path,
    v4_serve_model_path: str | Path,
    v4_cache_dir: str | Path,
    v5_rally_model_path: str | Path,
    v5_serve_model_path: str | Path,
    v5_cache_dir: str | Path,
    *,
    split: str = "validation",
    decision_path: str | Path | None = None,
    retrospective: bool = False,
    output_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    role = assessment_role_for_split(split, retrospective=retrospective)
    if split != "validation" and decision_path is None:
        raise ModelError("non-validation fusion evaluation requires a frozen validation decision")
    if split == "validation" and decision_path is not None:
        raise ModelError("validation selects a new fusion decision and cannot replay one")
    destination = Path(output_path).expanduser().resolve() if output_path else None
    if destination is not None and destination.exists():
        raise ModelError(f"evaluation output already exists: {destination}")
    manifest = load_manifest(manifest_path)
    manifest_sha256 = _manifest_digest(manifest)
    rows = manifest.for_split(split)
    if not rows:
        raise ModelError(f"manifest has no recordings in split {split!r}")
    v4_rally = load_model(v4_rally_model_path)
    v4_serve = load_model(v4_serve_model_path)
    v5_rally = load_model(v5_rally_model_path)
    v5_serve = load_model(v5_serve_model_path)
    _validate_model_pair(v4_rally, v4_serve, manifest_sha256=manifest_sha256)
    _validate_model_pair(v5_rally, v5_serve, manifest_sha256=manifest_sha256)
    _check_evaluation_groups(
        split, rows, (v4_rally, v4_serve, v5_rally, v5_serve)
    )
    models = _model_bindings(v4_rally, v4_serve, v5_rally, v5_serve)
    frozen_config: BoundarySelectorConfig | None = None
    decision_sha256: str | None = None
    if decision_path is not None:
        decision_file = Path(decision_path).expanduser().resolve()
        decision_bytes = decision_file.read_bytes()
        decision_sha256 = hashlib.sha256(decision_bytes).hexdigest()
        frozen_config = _validate_decision(
            json.loads(decision_bytes), manifest_sha256, models
        )
    started = time.perf_counter()
    predictions = _prepare_predictions(
        rows,
        v4_rally,
        v4_serve,
        v4_cache_dir,
        v5_rally,
        v5_serve,
        v5_cache_dir,
        progress=progress,
    )
    v4_report = _report(
        predictions, {item.recording.id: item.v4.composed for item in predictions}
    )
    v5_report = _report(
        predictions, {item.recording.id: item.v5.composed for item in predictions}
    )
    selected, selected_config, grid = _boundary_variants(
        predictions, selected_config=frozen_config
    )
    add_only = _add_only_audit(predictions, select_policy=split == "validation")
    if split == "validation" and add_only["selectedPolicy"] != "disabled":
        raise ModelError(
            "a validation score gate passed, but add-only replay is not implemented"
        )
    report = {
        "schemaVersion": 1,
        "experiment": EXPERIMENT_ID,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "producer": f"volleycut-analysis/{__version__}",
        "dataset": manifest.name,
        "manifestSha256": manifest_sha256,
        "split": split,
        "assessmentRole": role,
        "models": models,
        "matching": {"minimumIntervalIoU": 0.5},
        "selection": {
            "selectedOn": "validation",
            "decisionReportSha256": decision_sha256,
            "addOnly": {
                "selectedPolicy": "disabled",
                "reason": add_only["selectionReason"],
            },
            "boundarySelector": {
                "selectedConfig": selected_config.to_dict(),
                "candidateCount": len(grid),
                "guardrails": (
                    "Prediction count, event precision, and ordinary-long strict matches "
                    "must not regress from v4; then maximize F1, short/fault matches, and time IoU."
                ),
                "candidates": grid if split == "validation" else None,
            },
        },
        "v4Composition": v4_report,
        "v5Composition": v5_report,
        "selectedCombined": selected,
        "fullV5OneToOneReplacementAblation": _full_boundary_ablation(
            predictions, "v5"
        ),
        "fullIntersectionAblation": _full_boundary_ablation(
            predictions, "intersection"
        ),
        "addOnlyConfidenceAudit": {
            "selectedPolicy": add_only["selectedPolicy"],
            "candidateEvidence": add_only["candidateEvidence"],
            "scoreGateSweep": add_only["scoreGateSweep"],
        },
        "processing": {
            "wallClockSeconds": time.perf_counter() - started,
            "includesFeatureCacheLoading": True,
        },
        "limitations": [
            (
                "Both source-group validation recordings are grass footage and were "
                "reused by earlier model selection."
            ),
            "The one-source indoor test has been inspected repeatedly and is retrospective only.",
            "Model sigmoid outputs are uncalibrated ranking scores, not comparable probabilities.",
            (
                "Composed interval confidence can mix rally and serve evidence; raw "
                "heads are reported separately."
            ),
            (
                "Outcome slices overlap, and slice precision/F1 are undefined because "
                "predictions are not outcome-typed."
            ),
        ],
    }
    if destination is not None:
        atomic_write_text(destination, json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
