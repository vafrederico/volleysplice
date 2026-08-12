"""Nested serve-anchored multistate experiment on a frozen feature substrate.

This module compares two decoders trained on exactly the same upstream feature
signature:

* the existing weighted binary logistic model plus the current hysteresis
  decoder; and
* four cross-fitted one-vs-rest heads for DEAD/SETUP/SERVE/LIVE followed by the
  constrained multistate decoder in :mod:`analysis.multistate`.

Rally boundaries create categorical training targets and fold-local duration
priors.  They are never feature columns.  Outcome tags are used only after
decoding to report slices; the optional immediate-result auxiliary head is
deliberately omitted from this first experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import FEATURE_VERSION, DecoderConfig, FeatureConfig, TrainingConfig
from .court_relative_experiment import derive_court_relative_features
from .feature_experiments import (
    DEFAULT_OBJECTIVE_MARGIN,
    DEFAULT_SIGN_CONSISTENCY,
    DEVELOPMENT_SPLITS,
    FeatureExperimentError,
    FeatureSet,
    OuterFold,
    _aggregate_artifacts,
    _fold_report,
    _inner_folds,
    _model_fingerprint,
    _prepared_for_groups,
    _run_candidate_fold,
    _seed,
    _select_decoder,
    _subset_prepared,
    build_fold_plan,
    classify_paired_deltas,
    objective,
    sha256_file,
)
from .metrics import (
    aggregate_evaluations,
    aggregate_outcome_slices,
    evaluate_intervals,
    outcome_slice_metrics,
)
from .model import LogisticModel, train_logistic_model
from .multistate import (
    ALLOWED_TRANSITIONS,
    STATE_ORDER,
    MultistateDecoderConfig,
    MultistateState,
    StateDurationPrior,
    build_state_targets,
    decode_multistate,
)
from .pipeline import PreparedRecording, _manifest_digest, _prepare_many
from .schema import DatasetManifest, Interval, load_manifest
from .transition_feature_experiment import (
    DerivedFeatureBlock,
    _rank_nonconstant_columns,
    append_feature_blocks,
    prepare_transition_candidates,
)


MULTISTATE_STUDY_SCHEMA_VERSION = 1
SETUP_RADIUS_SECONDS = 5.0
# Zero deliberately invokes build_state_targets' nearest-grid fallback, yielding
# exactly one SERVE sample per rally at the current 4-fps substrate.
SERVE_RADIUS_SECONDS = 0.0
DEFAULT_TRANSITION_BONUSES = (-2.0, -1.0, 0.0, 1.0, 2.0)
LIVE_RECALL_TOLERANCE = 0.01
ACCEPTED_UPSTREAM_KINDS = frozenset(
    {
        "volleycut-transition-feature-experiment-development",
        "volleycut-court-relative-feature-study-development",
    }
)


@dataclass(frozen=True)
class StateModelBundle:
    models: tuple[LogisticModel, ...]

    def __post_init__(self) -> None:
        if len(self.models) != len(STATE_ORDER):
            raise ValueError("one state model is required for every state")


@dataclass
class MultistateFoldArtifact:
    fold: OuterFold
    held_prepared: list[PreparedRecording]
    per_recording: list[dict[str, Any]]
    aggregate: dict[str, Any]
    inner_selection: dict[str, Any]
    epoch_caps: dict[str, int]
    transition_bonus: float
    model_fingerprints: dict[str, str]


def _signature_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(names).encode("utf-8")).hexdigest()


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureExperimentError(f"cannot read {label} {resolved}: {error}") from error
    if not isinstance(payload, dict):
        raise FeatureExperimentError(f"{label} must contain one JSON object")
    return resolved, payload


def _load_frozen_upstream_report(
    path: str | Path,
) -> tuple[Path, dict[str, Any], tuple[str, ...]]:
    resolved, report = _load_json(path, "upstream feature-selection report")
    finalization = report.get("finalizationPlan")
    names = finalization.get("featureNames") if isinstance(finalization, dict) else None
    if (
        report.get("kind") not in ACCEPTED_UPSTREAM_KINDS
        or report.get("freezeStatus") != "frozen-development-selection"
        or report.get("testLabelsUsed") is not False
        or not isinstance(report.get("featureConfig"), dict)
        or not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise FeatureExperimentError(
            "upstream report must be a frozen unopened-test development selection "
            "with finalizationPlan.featureNames"
        )
    frozen = tuple(names)
    if len(frozen) != len(set(frozen)):
        raise FeatureExperimentError("upstream frozen feature signature has duplicates")
    inner_limit = report.get("selectionProtocol", {}).get("innerFoldLimit")
    if inner_limit not in (None, 0):
        raise FeatureExperimentError(
            "multistate study requires a full nested upstream report, not an "
            "inner-fold smoke run"
        )
    expected_digest = finalization.get("featureSignatureSha256")
    if expected_digest is not None and expected_digest != _signature_sha256(frozen):
        raise FeatureExperimentError("upstream frozen feature signature hash is invalid")
    return resolved, report, frozen


def reconstruct_frozen_upstream_features(
    prepared: Sequence[PreparedRecording],
    feature_config: FeatureConfig,
    frozen_names: Sequence[str],
) -> list[PreparedRecording]:
    """Recreate the exact selected step-1/step-2 matrix from warm caches.

    The transition implementation provides a superset of every step-1
    candidate.  If the frozen signature contains step-2 court columns, the
    combined court bank is derived, within-recording ranked by the same rule as
    its experiment runner, and appended once.  The final matrix is then selected
    by name in the frozen order.
    """

    if not prepared:
        raise FeatureExperimentError("cannot reconstruct an empty upstream substrate")
    transition = prepare_transition_candidates(prepared, feature_config)
    augmented = list(transition.prepared)
    needs_court = any(name.startswith("derived/court_relative/") for name in frozen_names)
    if needs_court:
        court_augmented: list[PreparedRecording] = []
        for item in augmented:
            raw, names = derive_court_relative_features(item, "combined")
            court_augmented.append(
                append_feature_blocks(
                    item,
                    DerivedFeatureBlock(
                        values=_rank_nonconstant_columns(raw),
                        names=names,
                        groups={"court_relative:combined": tuple(range(len(names)))},
                        definitions={"normalization": "within-recording percentile rank"},
                    ),
                )
            )
        augmented = court_augmented

    expected = tuple(frozen_names)
    result: list[PreparedRecording] = []
    for item in augmented:
        indexes_by_name = {name: index for index, name in enumerate(item.contextual_names)}
        missing = [name for name in expected if name not in indexes_by_name]
        if missing:
            raise FeatureExperimentError(
                f"frozen upstream features are not implemented: {missing[:8]}"
            )
        indexes = tuple(indexes_by_name[name] for name in expected)
        selected = _subset_prepared((item,), indexes)[0]
        if selected.contextual_names != expected:
            raise FeatureExperimentError("upstream feature reconstruction changed order")
        result.append(selected)
    return result


def state_targets(item: PreparedRecording) -> np.ndarray:
    """Boundary-only four-state targets with exactly one SERVE row per rally."""

    targets = build_state_targets(
        item.sequence.times,
        item.recording.rallies,
        setup_radius_seconds=SETUP_RADIUS_SECONDS,
        serve_radius_seconds=SERVE_RADIUS_SECONDS,
    )
    expected_serves = sum(
        bool(
            np.any(
                item.sample_mask
                & (targets == int(MultistateState.SERVE))
                & (item.sequence.times >= rally.start - 0.5)
                & (item.sequence.times < rally.end)
            )
        )
        for rally in item.recording.rallies
    )
    actual_serves = int(
        np.sum(item.sample_mask & (targets == int(MultistateState.SERVE)))
    )
    if actual_serves != expected_serves:
        raise FeatureExperimentError(
            "serve targets must contain one unignored nearest-grid sample per eligible rally"
        )
    return targets


def _masked_target_runs(
    prepared: Sequence[PreparedRecording],
) -> tuple[dict[MultistateState, list[int]], dict[tuple[MultistateState, MultistateState], int]]:
    runs = {state: [] for state in STATE_ORDER}
    transitions: dict[tuple[MultistateState, MultistateState], int] = {}
    for item in prepared:
        targets = state_targets(item)
        indexes = np.flatnonzero(item.sample_mask)
        if not len(indexes):
            continue
        # Ignored spans split sequences; never learn a duration or transition
        # across an annotation-excluded gap.
        starts = np.r_[0, np.flatnonzero(np.diff(indexes) > 1) + 1]
        ends = np.r_[starts[1:], len(indexes)]
        for start, end in zip(starts, ends, strict=True):
            values = targets[indexes[start:end]]
            if not len(values):
                continue
            run_start = 0
            previous_state: MultistateState | None = None
            for offset in range(1, len(values) + 1):
                if offset < len(values) and values[offset] == values[run_start]:
                    continue
                state = MultistateState(int(values[run_start]))
                runs[state].append(offset - run_start)
                if previous_state is not None and state != previous_state:
                    transition = (previous_state, state)
                    if state in ALLOWED_TRANSITIONS[previous_state]:
                        transitions[transition] = transitions.get(transition, 0) + 1
                previous_state = state
                run_start = offset
    return runs, transitions


def estimate_fold_decoder(
    training: Sequence[PreparedRecording],
    *,
    transition_bonus: float,
) -> tuple[MultistateDecoderConfig, dict[str, Any]]:
    """Estimate duration and transition priors from one training fold only."""

    if not training:
        raise FeatureExperimentError("multistate priors require training recordings")
    runs, transition_counts = _masked_target_runs(training)
    priors: dict[MultistateState, StateDurationPrior] = {}
    duration_summary: dict[str, Any] = {}
    for state in STATE_ORDER:
        lengths = runs[state]
        if state == MultistateState.SERVE:
            if lengths and any(length != 1 for length in lengths):
                raise FeatureExperimentError("SERVE target runs must be one sample")
            prior = StateDurationPrior(
                log_scores=(0.0,),
                tail_log_score=0.0,
                minimum_samples=1,
                maximum_samples=1,
            )
            mean_samples = 1.0
            hazard = 1.0
        else:
            if not lengths:
                raise FeatureExperimentError(
                    f"training fold has no {state.name} duration examples"
                )
            mean_samples = float(np.mean(lengths))
            hazard = float(np.clip(1.0 / max(mean_samples, 1.0), 0.005, 0.95))
            prior = StateDurationPrior(
                log_scores=(math.log(hazard),),
                tail_log_score=math.log1p(-hazard),
                minimum_samples=1,
                maximum_samples=None,
            )
        priors[state] = prior
        duration_summary[state.name] = {
            "runs": len(lengths),
            "meanSamples": mean_samples,
            "geometricExitHazard": hazard,
            "minimumSamples": prior.minimum_samples,
            "maximumSamples": prior.maximum_samples,
        }

    transition_scores: dict[tuple[MultistateState, MultistateState], float] = {}
    transition_summary: dict[str, Any] = {}
    for source in STATE_ORDER:
        destinations = sorted(
            (state for state in ALLOWED_TRANSITIONS[source] if state != source),
            key=int,
        )
        if not destinations:
            continue
        denominator = sum(
            transition_counts.get((source, destination), 0) + 1
            for destination in destinations
        )
        for destination in destinations:
            count = transition_counts.get((source, destination), 0)
            probability = (count + 1) / denominator
            score = math.log(probability) + float(transition_bonus)
            transition_scores[(source, destination)] = score
            transition_summary[f"{source.name}->{destination.name}"] = {
                "count": count,
                "laplaceProbability": probability,
                "logScoreWithBonus": score,
            }

    config = MultistateDecoderConfig(
        duration_priors=priors,
        transition_log_scores=transition_scores,
    )
    config.validate()
    return config, {
        "trainingRecordingIds": [item.recording.id for item in training],
        "trainingSourceGroups": sorted(
            {item.recording.source_group for item in training}
        ),
        "transitionBonus": float(transition_bonus),
        "durations": duration_summary,
        "transitions": transition_summary,
    }


def _training_arrays_for_state(
    prepared: Sequence[PreparedRecording], state: MultistateState
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for item in prepared:
        targets = state_targets(item)
        values.append(item.contextual_values[item.sample_mask])
        labels.append((targets[item.sample_mask] == int(state)).astype(np.float32))
    return values, labels


def fit_state_models(
    training: Sequence[PreparedRecording],
    validation: Sequence[PreparedRecording],
    *,
    feature_config: FeatureConfig,
    training_config: TrainingConfig,
    seed_parts: Sequence[str],
    epoch_caps: Mapping[MultistateState, int] | None = None,
) -> StateModelBundle:
    if not training:
        raise FeatureExperimentError("state heads require training recordings")
    models: list[LogisticModel] = []
    for state in STATE_ORDER:
        train_values, train_labels = _training_arrays_for_state(training, state)
        validation_values, validation_labels = _training_arrays_for_state(
            validation, state
        )
        state_seed = _seed(training_config.seed, *seed_parts, state.name)
        state_config = replace(training_config, seed=state_seed)
        if epoch_caps is not None:
            cap = int(epoch_caps[state])
            state_config = replace(
                state_config,
                epochs=cap,
                patience=max(state_config.patience, cap + 1),
            )
        models.append(
            train_logistic_model(
                train_values,
                train_labels,
                validation_values,
                validation_labels,
                feature_config,
                training[0].contextual_names,
                DecoderConfig(),
                state_config,
            )
        )
    return StateModelBundle(tuple(models))


def state_log_scores(bundle: StateModelBundle, values: np.ndarray) -> np.ndarray:
    """Normalize the four independently balanced head scores into emissions."""

    probabilities = np.column_stack(
        [model.predict(values) for model in bundle.models]
    ).astype(np.float64)
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return np.log(probabilities)


def _clip_ignored_predictions(
    predictions: Sequence[Interval], ignored: Sequence[Any]
) -> list[Interval]:
    result = list(predictions)
    for interval in ignored:
        fragments: list[Interval] = []
        for prediction in result:
            if prediction.end <= interval.start or prediction.start >= interval.end:
                fragments.append(prediction)
                continue
            if prediction.start < interval.start:
                fragments.append(Interval(prediction.start, interval.start))
            if prediction.end > interval.end:
                fragments.append(Interval(interval.end, prediction.end))
        result = fragments
    return result


def evaluate_multistate_predictions(
    prepared: Sequence[PreparedRecording],
    emissions: Sequence[np.ndarray],
    configs: Sequence[MultistateDecoderConfig],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (len(prepared) == len(emissions) == len(configs)):
        raise ValueError("multistate evaluation inputs must be aligned")
    rows: list[dict[str, Any]] = []
    for item, scores, config in zip(prepared, emissions, configs, strict=True):
        decoded = decode_multistate(item.sequence.times, scores, config)
        predictions = _clip_ignored_predictions(
            [Interval(value.start, value.end) for value in decoded.intervals],
            item.recording.ignored_intervals,
        )
        metrics = evaluate_intervals(item.recording.rallies, predictions)
        metrics["outcomeSlices"] = outcome_slice_metrics(
            item.recording.rallies, predictions
        )
        metrics["id"] = item.recording.id
        metrics["sourceGroup"] = item.recording.source_group
        metrics["environment"] = item.recording.environment
        metrics["decodedStateSamples"] = {
            state.name: sum(value == state for value in decoded.states)
            for state in STATE_ORDER
        }
        rows.append(metrics)
    aggregate = aggregate_evaluations(rows)
    aggregate["outcomeSlices"] = aggregate_outcome_slices(
        [row["outcomeSlices"] for row in rows]
    )
    aggregate["objective"] = objective(aggregate)
    return rows, aggregate


def _metric_view(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "multiIoU": {
            "eventF1AtIou03": aggregate["eventF1AtIou03"],
            "eventF1AtIou05": aggregate["eventF1"],
            "eventF1AtIou07": aggregate["eventF1AtIou07"],
        },
        "liveDeadTime": {
            key: aggregate[key]
            for key in (
                "timeIoU",
                "liveTimeRecall",
                "liveTimePrecision",
                "trueLiveSeconds",
                "predictedLiveSeconds",
                "missedLiveSeconds",
                "deadSecondsRetained",
            )
        },
        "boundaries": {
            key: aggregate.get(key)
            for key in (
                "startBoundaryMaeSeconds",
                "endBoundaryMaeSeconds",
                "boundariesWithin025SecondRate",
                "boundariesWithin05SecondRate",
                "boundariesWithin1SecondRate",
            )
        },
        "exactCount": {
            "trueRallies": aggregate["trueRallies"],
            "predictedRallies": aggregate["predictedRallies"],
            "exactRallyCountRate": aggregate["exactRallyCountRate"],
        },
        "outcomeSlices": aggregate["outcomeSlices"],
    }


def _run_multistate_fold(
    prepared: Sequence[PreparedRecording],
    fold: OuterFold,
    *,
    feature_config: FeatureConfig,
    training_config: TrainingConfig,
    inner_fold_limit: int | None,
    transition_bonuses: Sequence[float],
) -> MultistateFoldArtifact:
    outer_training = _prepared_for_groups(prepared, fold.training_groups)
    held = _prepared_for_groups(prepared, (fold.held_out_group,))
    if not outer_training or not held:
        raise FeatureExperimentError("multistate outer fold is empty")

    inner_payloads: list[dict[str, Any]] = []
    epoch_history: dict[MultistateState, list[int]] = {
        state: [] for state in STATE_ORDER
    }
    for inner in _inner_folds(fold, inner_fold_limit):
        training = _prepared_for_groups(outer_training, inner.training_groups)
        validation = _prepared_for_groups(
            outer_training, (inner.validation_group,)
        )
        bundle = fit_state_models(
            training,
            validation,
            feature_config=feature_config,
            training_config=training_config,
            seed_parts=("multistate-inner", fold.held_out_group, inner.validation_group),
        )
        emissions = [
            state_log_scores(bundle, item.contextual_values) for item in validation
        ]
        for state, model in zip(STATE_ORDER, bundle.models, strict=True):
            epoch_history[state].append(int(model.training_summary["bestEpoch"]))
        inner_payloads.append(
            {
                "fold": inner,
                "training": training,
                "validation": validation,
                "emissions": emissions,
                "models": bundle,
            }
        )
    if not inner_payloads:
        raise FeatureExperimentError("multistate nested selection has no inner folds")

    candidates: list[dict[str, Any]] = []
    for bonus in transition_bonuses:
        all_validation: list[PreparedRecording] = []
        all_emissions: list[np.ndarray] = []
        all_configs: list[MultistateDecoderConfig] = []
        prior_summaries: list[dict[str, Any]] = []
        for payload in inner_payloads:
            config, summary = estimate_fold_decoder(
                payload["training"], transition_bonus=float(bonus)
            )
            all_validation.extend(payload["validation"])
            all_emissions.extend(payload["emissions"])
            all_configs.extend([config] * len(payload["validation"]))
            prior_summaries.append(summary)
        _, aggregate = evaluate_multistate_predictions(
            all_validation, all_emissions, all_configs
        )
        candidates.append(
            {
                "transitionBonus": float(bonus),
                "metrics": aggregate,
                "priorEstimates": prior_summaries,
            }
        )
    selected = max(
        candidates,
        key=lambda item: (
            float(item["metrics"]["objective"]),
            float(item["metrics"]["eventF1"]),
            float(item["metrics"]["liveTimePrecision"]),
            -abs(float(item["transitionBonus"])),
        ),
    )
    selected_bonus = float(selected["transitionBonus"])
    epoch_caps = {
        state: max(1, int(round(float(np.median(epoch_history[state])))))
        for state in STATE_ORDER
    }
    outer_config, outer_prior_summary = estimate_fold_decoder(
        outer_training, transition_bonus=selected_bonus
    )
    outer_bundle = fit_state_models(
        outer_training,
        (),
        feature_config=feature_config,
        training_config=training_config,
        seed_parts=("multistate-outer-refit", fold.held_out_group),
        epoch_caps=epoch_caps,
    )
    held_emissions = [
        state_log_scores(outer_bundle, item.contextual_values) for item in held
    ]
    rows, aggregate = evaluate_multistate_predictions(
        held, held_emissions, [outer_config] * len(held)
    )
    inner_folds = []
    for payload in inner_payloads:
        inner = payload["fold"]
        bundle = payload["models"]
        inner_folds.append(
            {
                **inner.to_dict(),
                "heads": {
                    state.name: {
                        "bestEpoch": int(model.training_summary["bestEpoch"]),
                        "epochsCompleted": int(
                            model.training_summary["epochsCompleted"]
                        ),
                        "bestValidationLoss": float(
                            model.training_summary["bestValidationLoss"]
                        ),
                        "modelFingerprint": _model_fingerprint(model),
                    }
                    for state, model in zip(STATE_ORDER, bundle.models, strict=True)
                },
            }
        )
    return MultistateFoldArtifact(
        fold=fold,
        held_prepared=held,
        per_recording=rows,
        aggregate=aggregate,
        inner_selection={
            "foldsUsed": inner_folds,
            "transitionBonusCandidates": [
                {
                    "transitionBonus": item["transitionBonus"],
                    "objective": item["metrics"]["objective"],
                    "eventF1": item["metrics"]["eventF1"],
                    "timeIoU": item["metrics"]["timeIoU"],
                    "liveTimeRecall": item["metrics"]["liveTimeRecall"],
                }
                for item in candidates
            ],
            "selectedTransitionBonus": selected_bonus,
            "selectedEpochCaps": {
                state.name: cap for state, cap in epoch_caps.items()
            },
            "selectedInnerPriorEstimates": selected["priorEstimates"],
            "outerTrainingPriorEstimate": outer_prior_summary,
        },
        epoch_caps={state.name: cap for state, cap in epoch_caps.items()},
        transition_bonus=selected_bonus,
        model_fingerprints={
            state.name: _model_fingerprint(model)
            for state, model in zip(STATE_ORDER, outer_bundle.models, strict=True)
        },
    )


def _aggregate_multistate_artifacts(
    artifacts: Sequence[MultistateFoldArtifact],
) -> dict[str, Any]:
    rows = [row for artifact in artifacts for row in artifact.per_recording]
    aggregate = aggregate_evaluations(rows)
    aggregate["outcomeSlices"] = aggregate_outcome_slices(
        [row["outcomeSlices"] for row in rows]
    )
    aggregate["objective"] = objective(aggregate)
    by_group = {
        artifact.fold.held_out_group: artifact.aggregate for artifact in artifacts
    }
    return {
        "aggregate": aggregate,
        "macroSourceGroup": {
            key: float(np.mean([float(row[key]) for row in by_group.values()]))
            for key in ("objective", "eventF1", "timeIoU", "liveTimeRecall")
        },
        "bySourceGroup": by_group,
        "recordings": rows,
    }


def _strict_count(slice_metrics: Mapping[str, Any]) -> int:
    return int(
        round(
            int(slice_metrics.get("rallies", 0))
            * float(slice_metrics.get("strictMatchRecall", 0.0))
        )
    )


def multistate_promotion_gate(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    paired_classification: Mapping[str, Any],
) -> dict[str, Any]:
    control_ordinary = control["outcomeSlices"]["ordinaryLong"]
    candidate_ordinary = candidate["outcomeSlices"]["ordinaryLong"]
    slice_counts = {
        name: {
            "control": _strict_count(control["outcomeSlices"][name]),
            "candidate": _strict_count(candidate["outcomeSlices"][name]),
        }
        for name in ("shortAtMost3Seconds", "serviceFault", "ordinaryLong")
    }
    for values in slice_counts.values():
        values["deltaCandidateMinusControl"] = (
            values["candidate"] - values["control"]
        )
    checks = {
        "objectiveEvidenceHelpful": (
            paired_classification.get("classification") == "helpful"
        ),
        "aggregateObjectiveHigher": (
            float(candidate["objective"]) > float(control["objective"])
        ),
        "shortStrictMatchesImproved": (
            slice_counts["shortAtMost3Seconds"]["candidate"]
            > slice_counts["shortAtMost3Seconds"]["control"]
        ),
        "serviceFaultStrictMatchesImproved": (
            slice_counts["serviceFault"]["candidate"]
            > slice_counts["serviceFault"]["control"]
        ),
        "ordinaryLongStrictMatchesPreserved": (
            slice_counts["ordinaryLong"]["candidate"]
            >= slice_counts["ordinaryLong"]["control"]
        ),
        "overallLiveRecallLossAtMostOnePoint": (
            float(candidate["liveTimeRecall"])
            >= float(control["liveTimeRecall"]) - LIVE_RECALL_TOLERANCE
        ),
    }
    return {
        "promoteMultistate": all(checks.values()),
        "checks": checks,
        "strictMatchCounts": slice_counts,
        "controlOrdinaryLongStrictMatches": _strict_count(control_ordinary),
        "candidateOrdinaryLongStrictMatches": _strict_count(candidate_ordinary),
        "liveRecallTolerance": LIVE_RECALL_TOLERANCE,
        "deltaObjectiveCandidateMinusControl": float(candidate["objective"])
        - float(control["objective"]),
        "deltaLiveTimeRecallCandidateMinusControl": float(
            candidate["liveTimeRecall"]
        )
        - float(control["liveTimeRecall"]),
    }


def _paired_comparison(
    control_artifacts: Sequence[Any],
    multistate_artifacts: Sequence[MultistateFoldArtifact],
    *,
    margin: float,
    sign_consistency: float,
) -> dict[str, Any]:
    controls = {item.fold.held_out_group: item.aggregate for item in control_artifacts}
    candidates = {
        item.fold.held_out_group: item.aggregate for item in multistate_artifacts
    }
    if set(controls) != set(candidates):
        raise FeatureExperimentError("paired multistate folds do not align")
    rows = []
    for group in sorted(controls):
        control = controls[group]
        candidate = candidates[group]
        rows.append(
            {
                "sourceGroup": group,
                "controlObjective": control["objective"],
                "multistateObjective": candidate["objective"],
                "deltaObjectiveMultistateMinusControl": candidate["objective"]
                - control["objective"],
                "deltaEventF1": candidate["eventF1"] - control["eventF1"],
                "deltaTimeIoU": candidate["timeIoU"] - control["timeIoU"],
                "deltaLiveTimeRecall": candidate["liveTimeRecall"]
                - control["liveTimeRecall"],
            }
        )
    classification = classify_paired_deltas(
        [row["deltaObjectiveMultistateMinusControl"] for row in rows],
        margin=margin,
        sign_consistency=sign_consistency,
    )
    return {
        "deltaDirection": "positive means multistate improves over the binary control",
        "classification": classification,
        "pairedSourceGroups": rows,
    }


def _multistate_fold_report(artifact: MultistateFoldArtifact) -> dict[str, Any]:
    return {
        **artifact.fold.to_dict(),
        "innerSelection": artifact.inner_selection,
        "outerRefit": {
            "epochCapsByState": artifact.epoch_caps,
            "transitionBonus": artifact.transition_bonus,
            "modelFingerprints": artifact.model_fingerprints,
        },
        "metrics": artifact.aggregate,
        "metricViews": _metric_view(artifact.aggregate),
        "recordings": artifact.per_recording,
    }


def _code_provenance() -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    tracked = (
        package / "multistate_feature_study.py",
        package / "multistate.py",
        package / "transition_feature_experiment.py",
        package / "court_relative_experiment.py",
        package / "feature_experiments.py",
        package / "model.py",
        package / "metrics.py",
    )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package.parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=package.parent,
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
            str(path.relative_to(package.parent)): sha256_file(path)
            for path in tracked
            if path.is_file()
        },
    }


def run_development_multistate_study(
    manifest: DatasetManifest,
    prepared: Sequence[PreparedRecording],
    *,
    upstream_report_path: str | Path,
    training_config: TrainingConfig,
    decoder_config: DecoderConfig,
    inner_fold_limit: int | None = None,
    transition_bonuses: Sequence[float] = DEFAULT_TRANSITION_BONUSES,
    objective_margin: float = DEFAULT_OBJECTIVE_MARGIN,
    sign_consistency: float = DEFAULT_SIGN_CONSISTENCY,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    upstream_path, upstream, frozen_names = _load_frozen_upstream_report(
        upstream_report_path
    )
    if upstream.get("manifestFileSha256") != sha256_file(manifest.path):
        raise FeatureExperimentError("upstream report does not match the manifest")
    if upstream.get("manifestSnapshotSha256") != _manifest_digest(manifest):
        raise FeatureExperimentError("recording snapshots changed after upstream selection")
    if upstream.get("recordingContentSha256") != {
        item.id: item.content_sha256 for item in manifest.recordings
    }:
        raise FeatureExperimentError(
            "recording content identities changed after upstream selection"
        )
    if upstream.get("featureVersion") != FEATURE_VERSION:
        raise FeatureExperimentError(
            "feature implementation version differs from the upstream selection"
        )
    if upstream.get("selectionProtocol", {}).get("folds") != [
        item.to_dict() for item in build_fold_plan(manifest.recordings)
    ]:
        raise FeatureExperimentError(
            "source-group folds differ from the frozen upstream selection"
        )
    feature_config = FeatureConfig.from_dict(upstream["featureConfig"])
    feature_config.validate()
    training_config.validate()
    decoder_config.validate()
    if not transition_bonuses or any(
        not math.isfinite(float(value)) for value in transition_bonuses
    ):
        raise ValueError("transition bonuses must be finite and non-empty")

    development_rows = [
        item for item in manifest.recordings if item.split in DEVELOPMENT_SPLITS
    ]
    expected_ids = {item.id for item in development_rows}
    if {item.recording.id for item in prepared} != expected_ids or any(
        item.recording.split not in DEVELOPMENT_SPLITS for item in prepared
    ):
        raise FeatureExperimentError(
            "prepared multistate data must contain exactly train+validation rows"
        )
    selected = reconstruct_frozen_upstream_features(
        prepared, feature_config, frozen_names
    )
    if _signature_sha256(selected[0].contextual_names) != _signature_sha256(
        frozen_names
    ):
        raise FeatureExperimentError("reconstructed upstream signature changed")
    folds = build_fold_plan(manifest.recordings)
    feature_set = FeatureSet(
        "frozen_upstream_same_features",
        tuple(range(len(frozen_names))),
        ("frozen_upstream",),
        "decoder architecture only",
    )
    control_artifacts = []
    multistate_artifacts = []
    for index, fold in enumerate(folds, start=1):
        if progress is not None:
            progress(
                f"Multistate study {index}/{len(folds)}: binary control, "
                f"hold out {fold.held_out_group}"
            )
        control_artifacts.append(
            _run_candidate_fold(
                selected,
                fold,
                feature_set,
                feature_config=feature_config,
                training_config=training_config,
                decoder_config=decoder_config,
                inner_fold_limit=inner_fold_limit,
            )
        )
        if progress is not None:
            progress(
                f"Multistate study {index}/{len(folds)}: four-state candidate, "
                f"hold out {fold.held_out_group}"
            )
        multistate_artifacts.append(
            _run_multistate_fold(
                selected,
                fold,
                feature_config=feature_config,
                training_config=training_config,
                inner_fold_limit=inner_fold_limit,
                transition_bonuses=transition_bonuses,
            )
        )

    control_oof = _aggregate_artifacts(control_artifacts)
    multistate_oof = _aggregate_multistate_artifacts(multistate_artifacts)
    paired = _paired_comparison(
        control_artifacts,
        multistate_artifacts,
        margin=objective_margin,
        sign_consistency=sign_consistency,
    )
    gate = multistate_promotion_gate(
        control_oof["aggregate"],
        multistate_oof["aggregate"],
        paired["classification"],
    )

    pooled_control_prepared = [
        item for artifact in control_artifacts for item in artifact.held_prepared
    ]
    pooled_control_probabilities = [
        values for artifact in control_artifacts for values in artifact.probabilities
    ]
    final_control_decoder, final_control_decoder_selection = _select_decoder(
        pooled_control_prepared, pooled_control_probabilities, decoder_config
    )
    control_epoch_cap = max(
        1,
        int(
            round(
                float(np.median([item.epoch_cap for item in control_artifacts]))
            )
        ),
    )
    final_state_epoch_caps = {
        state.name: max(
            1,
            int(
                round(
                    float(
                        np.median(
                            [item.epoch_caps[state.name] for item in multistate_artifacts]
                        )
                    )
                )
            ),
        )
        for state in STATE_ORDER
    }
    bonus_values = [item.transition_bonus for item in multistate_artifacts]
    final_bonus = min(
        (float(value) for value in transition_bonuses),
        key=lambda value: (abs(value - float(np.median(bonus_values))), abs(value)),
    )
    selected_architecture = (
        "serve_anchored_multistate" if gate["promoteMultistate"] else "binary_control"
    )
    return {
        "schemaVersion": MULTISTATE_STUDY_SCHEMA_VERSION,
        "kind": "volleycut-multistate-feature-study-development",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "freezeStatus": "frozen-development-selection",
        "assessmentRole": "development-only-source-group-out-of-fold-selection",
        "testLabelsUsed": False,
        "dataset": manifest.name,
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "manifestSnapshotSha256": _manifest_digest(manifest),
        "recordingContentSha256": {
            item.id: item.content_sha256 for item in manifest.recordings
        },
        "featureVersion": FEATURE_VERSION,
        "featureConfig": feature_config.to_dict(),
        "trainingConfig": training_config.to_dict(),
        "baseDecoderConfig": decoder_config.to_dict(),
        "upstream": {
            "report": str(upstream_path),
            "reportSha256": sha256_file(upstream_path),
            "kind": upstream.get("kind"),
            "selectedCandidate": upstream.get(
                "selectedCandidateForRetrospectiveTest"
            ),
            "featureCount": len(frozen_names),
            "featureNames": list(frozen_names),
            "featureSignatureSha256": _signature_sha256(frozen_names),
        },
        "selectionProtocol": {
            "outer": "leave one development sourceGroup out",
            "inner": "leave one remaining sourceGroup out",
            "innerFoldLimit": inner_fold_limit,
            "transitionBonusCandidates": [
                float(value) for value in transition_bonuses
            ],
            "durationPriors": (
                "geometric exit hazards estimated only from boundary-derived "
                "state runs in each training fold"
            ),
            "stateHeads": "four independently class-balanced one-vs-rest logistic heads",
            "serveTarget": (
                "exactly one nearest 4-fps sample per unignored rally; no +/- pulse"
            ),
            "objectiveMargin": objective_margin,
            "signConsistency": sign_consistency,
            "folds": [item.to_dict() for item in folds],
        },
        "sameFeatureBinaryControl": {
            "oof": control_oof,
            "metricViews": _metric_view(control_oof["aggregate"]),
            "outerFolds": [_fold_report(item) for item in control_artifacts],
        },
        "serveAnchoredMultistate": {
            "states": [state.name for state in STATE_ORDER],
            "allowedTransitions": {
                source.name: [state.name for state in sorted(destinations, key=int)]
                for source, destinations in ALLOWED_TRANSITIONS.items()
            },
            "oof": multistate_oof,
            "metricViews": _metric_view(multistate_oof["aggregate"]),
            "outerFolds": [
                _multistate_fold_report(item) for item in multistate_artifacts
            ],
            "optionalImmediateResultHead": {
                "implemented": False,
                "reason": (
                    "first version isolates state-sequence value; tags remain reporting-only"
                ),
            },
        },
        "pairedComparison": paired,
        "promotionDecision": {
            **gate,
            "selectedArchitecture": selected_architecture,
            "testMetricsConsulted": False,
        },
        "finalizationPlan": {
            "architecture": selected_architecture,
            "featureCount": len(frozen_names),
            "featureNames": list(frozen_names),
            "featureSignatureSha256": _signature_sha256(frozen_names),
            "binaryControl": {
                "epochCap": control_epoch_cap,
                "seed": _seed(
                    training_config.seed, "multistate-study-final-binary"
                ),
                "decoder": final_control_decoder.to_dict(),
                "decoderSelectionOnDevelopmentOof": final_control_decoder_selection,
            },
            "multistate": {
                "epochCapsByState": final_state_epoch_caps,
                "seedsByState": {
                    state.name: _seed(
                        training_config.seed,
                        "multistate-study-final",
                        state.name,
                    )
                    for state in STATE_ORDER
                },
                "transitionBonus": final_bonus,
                "durationPriors": "re-estimate from all development boundaries",
            },
        },
        "guardrails": [
            "Development prepares only train and validation recordings.",
            "Every learned score is source-group cross-fitted in outer evaluation.",
            "Each duration and transition prior uses only the corresponding training fold.",
            "Rally boundaries create targets but are never input columns.",
            "Outcome tags are reporting-only; the optional auxiliary head is omitted.",
            "The retrospective command refuses to open test unless every promotion gate passes.",
        ],
        "provenance": _code_provenance(),
        "processing": {
            "wallClockSeconds": round(time.perf_counter() - started, 3),
            "featureCacheNote": (
                "warm base caches are reused and the frozen upstream feature matrix "
                "is reconstructed deterministically"
            ),
        },
        "limitations": [
            "Only four independent development source groups are available.",
            "The four one-vs-rest probabilities are normalized but not jointly calibrated.",
            "Geometric state-duration priors are intentionally compact first-pass models.",
            "The categorical targets use a fixed five-second setup window.",
            "The immediate-result auxiliary tag head is not part of this version.",
        ],
    }


def _load_frozen_study(
    path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    resolved, report = _load_json(path, "multistate development report")
    if (
        report.get("schemaVersion") != MULTISTATE_STUDY_SCHEMA_VERSION
        or report.get("kind") != "volleycut-multistate-feature-study-development"
        or report.get("freezeStatus") != "frozen-development-selection"
        or report.get("testLabelsUsed") is not False
        or not isinstance(report.get("finalizationPlan"), dict)
    ):
        raise FeatureExperimentError(
            "study report is not a frozen unopened-test multistate development selection"
        )
    if report.get("promotionDecision", {}).get("promoteMultistate") is not True:
        raise FeatureExperimentError(
            "retrospective test is gated off because multistate did not pass development"
        )
    if report.get("selectionProtocol", {}).get("innerFoldLimit") not in (None, 0):
        raise FeatureExperimentError(
            "retrospective test requires a full nested development report, not an "
            "inner-fold smoke ablation"
        )
    if report["finalizationPlan"].get("architecture") != "serve_anchored_multistate":
        raise FeatureExperimentError("frozen multistate architecture is inconsistent")
    return resolved, report


def run_retrospective_multistate_test(
    manifest_path: str | Path,
    development_report_path: str | Path,
    cache_dir: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    report_path, development = _load_frozen_study(development_report_path)
    manifest = load_manifest(manifest_path)
    if development.get("manifestFileSha256") != sha256_file(manifest.path):
        raise FeatureExperimentError("study report does not match the manifest")
    if development.get("manifestSnapshotSha256") != _manifest_digest(manifest):
        raise FeatureExperimentError("recording snapshots changed after study freeze")
    if development.get("recordingContentSha256") != {
        item.id: item.content_sha256 for item in manifest.recordings
    }:
        raise FeatureExperimentError("recording identities changed after study freeze")
    if development.get("featureVersion") != FEATURE_VERSION:
        raise FeatureExperimentError("feature implementation version changed after study")
    current_provenance = _code_provenance()
    if development.get("provenance", {}).get("filesSha256") != current_provenance.get(
        "filesSha256"
    ):
        raise FeatureExperimentError("multistate experiment code changed after development")

    upstream_path = Path(development["upstream"]["report"])
    if sha256_file(upstream_path) != development["upstream"]["reportSha256"]:
        raise FeatureExperimentError("frozen upstream report changed after development")
    _, upstream, frozen_names = _load_frozen_upstream_report(upstream_path)
    feature_config = FeatureConfig.from_dict(upstream["featureConfig"])
    training_config = TrainingConfig(**development["trainingConfig"])
    finalization = development["finalizationPlan"]
    if tuple(finalization["featureNames"]) != frozen_names:
        raise FeatureExperimentError("study and upstream feature signatures disagree")

    development_rows = tuple(
        item for item in manifest.recordings if item.split in DEVELOPMENT_SPLITS
    )
    test_rows = manifest.for_split("test")
    if not development_rows or not test_rows:
        raise FeatureExperimentError("retrospective test requires development and test rows")
    if progress is not None:
        progress("Preparing frozen development features; protected test remains closed")
    development_raw = _prepare_many(
        development_rows, feature_config, cache_dir, progress=progress
    )
    selected_development = reconstruct_frozen_upstream_features(
        development_raw, feature_config, frozen_names
    )
    epoch_caps = {
        state: int(finalization["multistate"]["epochCapsByState"][state.name])
        for state in STATE_ORDER
    }
    bundle = fit_state_models(
        selected_development,
        (),
        feature_config=feature_config,
        training_config=training_config,
        seed_parts=("multistate-study-final",),
        epoch_caps=epoch_caps,
    )
    config, prior_summary = estimate_fold_decoder(
        selected_development,
        transition_bonus=float(finalization["multistate"]["transitionBonus"]),
    )

    if progress is not None:
        progress("Opening the protected retrospective test once with selection frozen")
    test_raw = _prepare_many(test_rows, feature_config, cache_dir, progress=progress)
    selected_test = reconstruct_frozen_upstream_features(
        test_raw, feature_config, frozen_names
    )
    emissions = [
        state_log_scores(bundle, item.contextual_values) for item in selected_test
    ]
    rows, aggregate = evaluate_multistate_predictions(
        selected_test, emissions, [config] * len(selected_test)
    )
    return {
        "schemaVersion": MULTISTATE_STUDY_SCHEMA_VERSION,
        "kind": "volleycut-multistate-feature-study-retrospective-test",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "assessmentRole": "explicit-single-source-retrospective-regression-test-access",
        "testLabelsOpened": True,
        "selectionLockedBeforeTest": True,
        "developmentReport": str(report_path),
        "developmentReportSha256": sha256_file(report_path),
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "selectedArchitecture": "serve_anchored_multistate",
        "featureCount": len(frozen_names),
        "featureSignatureSha256": _signature_sha256(frozen_names),
        "training": {
            "recordingIds": [item.recording.id for item in selected_development],
            "sourceGroups": sorted(
                {item.recording.source_group for item in selected_development}
            ),
            "testRowsUsedForFitting": False,
            "epochCapsByState": {
                state.name: epoch_caps[state] for state in STATE_ORDER
            },
            "modelFingerprints": {
                state.name: _model_fingerprint(model)
                for state, model in zip(STATE_ORDER, bundle.models, strict=True)
            },
            "durationAndTransitionPriorEstimate": prior_summary,
        },
        "test": {
            "aggregate": aggregate,
            "metricViews": _metric_view(aggregate),
            "recordings": rows,
        },
        "provenance": current_provenance,
        "guardrails": [
            "The development promotion gate passed before test was opened.",
            "Only train+validation recordings fit state heads and duration priors.",
            "Test metrics cannot revise the frozen architecture or hyperparameters.",
        ],
        "warning": (
            "This protected split is a retrospective regression check, not external "
            "generalization evidence."
        ),
    }
