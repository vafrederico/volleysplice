"""Bounded existing-label ablations for the serve-anchored multistate model.

This first runner isolates two diagnosed decoder defects without fitting a new
feature model or searching numeric hyperparameters:

* fold-prior correction of inverse-frequency-balanced one-vs-rest heads; and
* empirical SETUP plus deterministic two-geometric LIVE duration priors.

The Cartesian product is four candidates.  Every value is estimated from the
corresponding training fold; inner source-group OOF selects one candidate per
outer fold before the held source is decoded.
"""

from __future__ import annotations

import math
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import FEATURE_VERSION, FeatureConfig, TrainingConfig
from .feature_experiments import (
    DEFAULT_OBJECTIVE_MARGIN,
    DEFAULT_SIGN_CONSISTENCY,
    DEVELOPMENT_SPLITS,
    FeatureExperimentError,
    _model_fingerprint,
    _prepared_for_groups,
    build_fold_plan,
    classify_paired_deltas,
    sha256_file,
)
from .metrics import aggregate_evaluations, aggregate_outcome_slices
from .multistate import (
    STATE_ORDER,
    MultistateDecoderConfig,
    MultistateState,
    StateDurationPrior,
)
from .multistate_feature_study import (
    StateModelBundle,
    _masked_target_runs,
    _signature_sha256,
    estimate_fold_decoder,
    evaluate_multistate_predictions,
    fit_state_models,
    multistate_promotion_gate,
    reconstruct_frozen_upstream_features,
    state_log_scores,
)
from .multistate_followup import (
    _assert_fingerprint,
    _assert_metrics,
    _frozen_inner_by_group,
    _strict_slice_count,
    _validate_development_study,
)
from .pipeline import PreparedRecording, _manifest_digest
from .schema import DatasetManifest


EXISTING_LABELS_KIND = "volleycut-multistate-existing-label-ablation-development"
EMISSION_MODES = ("naive_balanced_ovr", "fold_prior_corrected_ovr")
DURATION_MODES = ("geometric_v1", "empirical_setup_live_mixture")
CANDIDATE_NAMES = tuple(
    f"{emission}__{duration}"
    for emission in EMISSION_MODES
    for duration in DURATION_MODES
)
REFERENCE_CANDIDATE = CANDIDATE_NAMES[0]
EMPIRICAL_SMOOTHING = 1.0
EMPIRICAL_MAX_SAMPLES = 32
LIVE_MIXTURE_MAX_SAMPLES = 160
MIXTURE_MAX_ITERATIONS = 100
MIXTURE_TOLERANCE = 1e-9
MIXTURE_COLLAPSE_HAZARD_GAP = 1e-4
INNER_MACRO_OBJECTIVE_GAIN = 0.01
INNER_TIE_MARGIN = 0.005


def _code_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    tracked = (
        root / "analysis" / "multistate_existing_labels.py",
        root / "scripts" / "evaluate-multistate-followup.py",
        root / "analysis" / "multistate_followup.py",
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


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    emission_mode: str
    duration_mode: str


def candidate_specs() -> tuple[CandidateSpec, ...]:
    return tuple(
        CandidateSpec(
            name=f"{emission}__{duration}",
            emission_mode=emission,
            duration_mode=duration,
        )
        for emission in EMISSION_MODES
        for duration in DURATION_MODES
    )


def raw_state_head_probabilities(
    bundle: StateModelBundle, values: np.ndarray
) -> np.ndarray:
    probabilities = np.column_stack(
        [model.predict(values) for model in bundle.models]
    ).astype(np.float64)
    return np.clip(probabilities, 1e-8, 1.0 - 1e-8)


def emission_log_scores(
    bundle: StateModelBundle,
    values: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    """Return normalized emissions, optionally undoing balanced-head priors."""

    if mode not in EMISSION_MODES:
        raise ValueError(f"unknown emission mode: {mode}")
    if mode == "naive_balanced_ovr":
        # Preserve the frozen v1 path byte-for-byte. In particular, v1 permits
        # a head probability of exactly 1.0 before row normalization.
        return state_log_scores(bundle, values)
    raw = raw_state_head_probabilities(bundle, values)
    corrected_columns = []
    for index, model in enumerate(bundle.models):
        summary = model.training_summary
        positive = float(summary["positiveSamples"])
        negative = float(summary["negativeSamples"])
        if positive <= 0 or negative <= 0:
            raise FeatureExperimentError("state head class counts must be positive")
        logit = np.log(raw[:, index]) - np.log1p(-raw[:, index])
        # Balanced training learns an artificial 50/50 prior. Restore the
        # observed fold prior odds without consulting validation/held rows.
        corrected_columns.append(
            1.0
            / (
                1.0
                + np.exp(
                    -np.clip(logit + math.log(positive / negative), -30, 30)
                )
            )
        )
    corrected = np.column_stack(corrected_columns)
    corrected = np.clip(corrected, 1e-12, 1.0)
    corrected /= np.sum(corrected, axis=1, keepdims=True)
    return np.log(corrected)


def emission_prior_summary(
    bundle: StateModelBundle, training: Sequence[PreparedRecording]
) -> dict[str, Any]:
    return {
        "trainingRecordingIds": [item.recording.id for item in training],
        "trainingSourceGroups": sorted(
            {item.recording.source_group for item in training}
        ),
        "states": {
            state.name: {
                "positiveSamples": int(model.training_summary["positiveSamples"]),
                "negativeSamples": int(model.training_summary["negativeSamples"]),
                "appliedLogPriorOdds": math.log(
                    float(model.training_summary["positiveSamples"])
                    / float(model.training_summary["negativeSamples"])
                ),
            }
            for state, model in zip(STATE_ORDER, bundle.models, strict=True)
        },
    }


def _geometric_probability(length: np.ndarray, hazard: float) -> np.ndarray:
    return hazard * np.power(1.0 - hazard, length - 1.0)


def _normalize_duration_table_with_tail(
    probabilities: np.ndarray, continuation_probability: float
) -> np.ndarray:
    if (
        probabilities.ndim != 1
        or not len(probabilities)
        or np.any(probabilities <= 0)
        or not 0.0 < continuation_probability < 1.0
    ):
        raise ValueError("duration table and continuation probability are invalid")
    total = float(
        np.sum(probabilities[:-1])
        + probabilities[-1] / (1.0 - continuation_probability)
    )
    return probabilities / total


def fit_two_geometric_mixture(
    lengths: Sequence[int],
) -> tuple[float, float, float, dict[str, Any]]:
    values = np.asarray(lengths, dtype=np.float64)
    if not len(values) or np.any(values < 1) or not np.isfinite(values).all():
        raise ValueError("mixture durations must be positive finite samples")
    short_hazard = float(np.clip(1.0 / np.percentile(values, 25), 0.005, 0.95))
    long_hazard = float(np.clip(1.0 / np.percentile(values, 75), 0.005, 0.95))
    if short_hazard < long_hazard:
        short_hazard, long_hazard = long_hazard, short_hazard
    weight = 0.5
    converged = False
    for iteration in range(1, MIXTURE_MAX_ITERATIONS + 1):
        short_probability = weight * _geometric_probability(values, short_hazard)
        long_probability = (1.0 - weight) * _geometric_probability(values, long_hazard)
        responsibilities = short_probability / np.maximum(
            short_probability + long_probability, 1e-300
        )
        next_weight = float(np.clip(np.mean(responsibilities), 0.05, 0.95))
        next_short = float(
            np.clip(
                np.sum(responsibilities)
                / np.sum(responsibilities * values),
                0.005,
                0.95,
            )
        )
        complement = 1.0 - responsibilities
        next_long = float(
            np.clip(
                np.sum(complement) / np.sum(complement * values),
                0.005,
                0.95,
            )
        )
        if next_short < next_long:
            next_short, next_long = next_long, next_short
            next_weight = 1.0 - next_weight
        delta = max(
            abs(next_weight - weight),
            abs(next_short - short_hazard),
            abs(next_long - long_hazard),
        )
        weight, short_hazard, long_hazard = next_weight, next_short, next_long
        if delta <= MIXTURE_TOLERANCE:
            converged = True
            break
    return weight, short_hazard, long_hazard, {
        "samples": len(values),
        "weightShort": weight,
        "shortExitHazard": short_hazard,
        "longExitHazard": long_hazard,
        "shortMeanSamples": 1.0 / short_hazard,
        "longMeanSamples": 1.0 / long_hazard,
        "iterations": iteration,
        "converged": converged,
        "componentHazardGap": short_hazard - long_hazard,
        "componentsCollapsed": (
            short_hazard - long_hazard < MIXTURE_COLLAPSE_HAZARD_GAP
        ),
        "collapseThreshold": MIXTURE_COLLAPSE_HAZARD_GAP,
    }


def _empirical_prior(lengths: Sequence[int]) -> tuple[StateDurationPrior, dict[str, Any]]:
    values = np.asarray(lengths, dtype=np.int64)
    if not len(values) or np.any(values < 1):
        raise ValueError("empirical durations must be positive")
    maximum = max(EMPIRICAL_MAX_SAMPLES, int(np.max(values)))
    counts = np.bincount(values, minlength=maximum + 1)[1 : maximum + 1].astype(
        np.float64
    )
    probabilities = (counts + EMPIRICAL_SMOOTHING) / (
        float(np.sum(counts)) + EMPIRICAL_SMOOTHING * len(counts)
    )
    continuation = 0.5
    probabilities = _normalize_duration_table_with_tail(
        probabilities, continuation
    )
    prior = StateDurationPrior(
        log_scores=tuple(float(math.log(value)) for value in probabilities),
        tail_log_score=math.log(continuation),
        minimum_samples=1,
        maximum_samples=None,
    )
    return prior, {
        "samples": len(values),
        "observedMinimumSamples": int(np.min(values)),
        "observedMaximumSamples": int(np.max(values)),
        "tableSamples": len(probabilities),
        "laplaceSmoothing": EMPIRICAL_SMOOTHING,
        "modeSamples": int(np.argmax(probabilities) + 1),
        "tailLogScore": prior.tail_log_score,
    }


def _mixture_prior(lengths: Sequence[int]) -> tuple[StateDurationPrior, dict[str, Any]]:
    weight, short_hazard, long_hazard, summary = fit_two_geometric_mixture(lengths)
    durations = np.arange(1, LIVE_MIXTURE_MAX_SAMPLES + 1, dtype=np.float64)
    probabilities = (
        weight * _geometric_probability(durations, short_hazard)
        + (1.0 - weight) * _geometric_probability(durations, long_hazard)
    )
    probabilities = np.maximum(probabilities, 1e-300)
    continuation = 1.0 - long_hazard
    probabilities = _normalize_duration_table_with_tail(
        probabilities, continuation
    )
    prior = StateDurationPrior(
        log_scores=tuple(float(math.log(value)) for value in probabilities),
        tail_log_score=math.log(continuation),
        minimum_samples=1,
        maximum_samples=None,
    )
    return prior, {
        **summary,
        "tableSamples": LIVE_MIXTURE_MAX_SAMPLES,
        "tailComponent": "long",
        "tailLogScore": prior.tail_log_score,
    }


def estimate_empirical_mixture_decoder(
    training: Sequence[PreparedRecording],
    *,
    transition_bonus: float,
) -> tuple[MultistateDecoderConfig, dict[str, Any]]:
    """Keep v1 transitions/DEAD, replace SETUP and LIVE duration priors."""

    base, base_summary = estimate_fold_decoder(
        training, transition_bonus=transition_bonus
    )
    runs, _ = _masked_target_runs(training)
    setup_prior, setup_summary = _empirical_prior(runs[MultistateState.SETUP])
    live_prior, live_summary = _mixture_prior(runs[MultistateState.LIVE])
    duration_priors = dict(base.duration_priors)
    duration_priors[MultistateState.SETUP] = setup_prior
    duration_priors[MultistateState.LIVE] = live_prior
    config = MultistateDecoderConfig(
        duration_priors=duration_priors,
        transition_log_scores=base.transition_log_scores,
    )
    config.validate()
    return config, {
        "trainingRecordingIds": [item.recording.id for item in training],
        "trainingSourceGroups": sorted(
            {item.recording.source_group for item in training}
        ),
        "transitionBonus": transition_bonus,
        "deadAndTransitionBaseline": base_summary,
        "setupEmpirical": setup_summary,
        "liveTwoGeometricMixture": live_summary,
    }


def _evaluation(
    prepared: Sequence[PreparedRecording],
    bundle: StateModelBundle,
    config: MultistateDecoderConfig,
    *,
    emission_mode: str,
) -> dict[str, Any]:
    emissions = [
        emission_log_scores(bundle, item.contextual_values, mode=emission_mode)
        for item in prepared
    ]
    rows, aggregate = evaluate_multistate_predictions(
        prepared, emissions, [config] * len(prepared)
    )
    return {"aggregate": aggregate, "recordings": rows}


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_evaluations(rows)
    aggregate["outcomeSlices"] = aggregate_outcome_slices(
        [row["outcomeSlices"] for row in rows]
    )
    from .feature_experiments import objective

    aggregate["objective"] = objective(aggregate)
    return aggregate


def _candidate_gate(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    groups = sorted(reference["bySourceGroup"])
    base = reference["aggregate"]
    improved = candidate["aggregate"]
    objective_deltas = [
        candidate["bySourceGroup"][group]["objective"]
        - reference["bySourceGroup"][group]["objective"]
        for group in groups
    ]
    checks = {
        "aggregateObjectiveHigher": improved["objective"] > base["objective"],
        "aggregateLiveRecallLossAtMostOnePoint": improved["liveTimeRecall"]
        >= base["liveTimeRecall"] - 0.01,
        "eachSourceLiveRecallLossAtMostThreePoints": all(
            candidate["bySourceGroup"][group]["liveTimeRecall"]
            >= reference["bySourceGroup"][group]["liveTimeRecall"] - 0.03
            for group in groups
        ),
        "ordinaryLongStrictMatchesPreserved": _strict_slice_count(
            improved, "ordinaryLong"
        )
        >= _strict_slice_count(base, "ordinaryLong"),
        "shortStrictMatchesPreserved": _strict_slice_count(
            improved, "shortAtMost3Seconds"
        )
        >= _strict_slice_count(base, "shortAtMost3Seconds"),
        "serviceFaultStrictMatchesPreserved": _strict_slice_count(
            improved, "serviceFault"
        )
        >= _strict_slice_count(base, "serviceFault"),
        "eventPrecisionLossAtMostOnePoint": improved["eventPrecision"]
        >= base["eventPrecision"] - 0.01,
        "deadSecondsRetainedAtMostTwoPercentHigher": improved[
            "deadSecondsRetained"
        ]
        <= base["deadSecondsRetained"] * 1.02 + 1e-9,
        "eachSourceEventF1LossAtMostThreePoints": all(
            candidate["bySourceGroup"][group]["eventF1"]
            >= reference["bySourceGroup"][group]["eventF1"] - 0.03
            for group in groups
        ),
        "macroObjectiveGainAtLeastOnePoint": candidate["macroSourceGroup"][
            "objective"
        ]
        >= reference["macroSourceGroup"]["objective"]
        + INNER_MACRO_OBJECTIVE_GAIN,
        "medianSourceObjectiveNonnegative": float(np.median(objective_deltas))
        >= 0.0,
        "atLeastTwoSourcesObjectiveNonnegative": sum(
            value >= 0.0 for value in objective_deltas
        )
        >= 2,
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "sourceObjectiveDeltas": dict(zip(groups, objective_deltas, strict=True)),
    }


def _selection_report(
    candidate_reports: Mapping[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if tuple(candidate_reports) != CANDIDATE_NAMES:
        raise ValueError("candidate reports do not follow the frozen order")
    reference = candidate_reports[REFERENCE_CANDIDATE]
    gates = {
        name: (
            {"eligible": True, "checks": {"reference": True}}
            if name == REFERENCE_CANDIDATE
            else _candidate_gate(reference, candidate_reports[name])
        )
        for name in CANDIDATE_NAMES
    }
    selected = REFERENCE_CANDIDATE
    for name in CANDIDATE_NAMES[1:]:
        if not gates[name]["eligible"]:
            continue
        if (
            candidate_reports[name]["aggregate"]["objective"]
            > candidate_reports[selected]["aggregate"]["objective"]
            + INNER_TIE_MARGIN
        ):
            selected = name
    return selected, {
        "candidateOrderSimplestFirst": list(CANDIDATE_NAMES),
        "tieMargin": INNER_TIE_MARGIN,
        "selectedCandidate": selected,
        "candidates": {
            name: {
                "gate": gates[name],
                "aggregate": candidate_reports[name]["aggregate"],
                "bySourceGroup": candidate_reports[name]["bySourceGroup"],
            }
            for name in CANDIDATE_NAMES
        },
    }


def _reports_by_group(
    rows_by_candidate: Mapping[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    result = {}
    for name in CANDIDATE_NAMES:
        rows = rows_by_candidate[name]
        groups = sorted({row["sourceGroup"] for row in rows})
        result[name] = {
            "aggregate": _aggregate_rows(rows),
            "bySourceGroup": {
                group: _aggregate_rows(
                    [row for row in rows if row["sourceGroup"] == group]
                )
                for group in groups
            },
            "recordings": rows,
        }
        result[name]["macroSourceGroup"] = {
            metric: float(
                np.mean(
                    [result[name]["bySourceGroup"][group][metric] for group in groups]
                )
            )
            for metric in ("objective", "eventF1", "timeIoU", "liveTimeRecall")
        }
    return result


def _outer_fold(
    selected: Sequence[PreparedRecording],
    fold: Any,
    *,
    feature_config: FeatureConfig,
    training_config: TrainingConfig,
    state_frozen: Mapping[str, Any],
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    outer_training = _prepared_for_groups(selected, fold.training_groups)
    held = _prepared_for_groups(selected, (fold.held_out_group,))
    frozen_inner = _frozen_inner_by_group(
        state_frozen["innerSelection"]["foldsUsed"]
    )
    transition_bonus = float(
        state_frozen["innerSelection"]["selectedTransitionBonus"]
    )
    rows_by_candidate = {name: [] for name in CANDIDATE_NAMES}
    inner_rows = []
    for index, inner in enumerate(fold.inner_folds, start=1):
        if progress is not None:
            progress(
                f"Existing-label {fold.held_out_group}: inner {index}/3 "
                f"validate {inner.validation_group}"
            )
        training = _prepared_for_groups(outer_training, inner.training_groups)
        validation = _prepared_for_groups(
            outer_training, (inner.validation_group,)
        )
        bundle = fit_state_models(
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
        fingerprints = {
            state.name: _model_fingerprint(model)
            for state, model in zip(STATE_ORDER, bundle.models, strict=True)
        }
        for state in STATE_ORDER:
            _assert_fingerprint(
                fingerprints[state.name],
                frozen_inner[inner.validation_group]["heads"][state.name][
                    "modelFingerprint"
                ],
                f"{fold.held_out_group}/{inner.validation_group}/{state.name}",
            )
        configs = {
            "geometric_v1": estimate_fold_decoder(
                training, transition_bonus=transition_bonus
            ),
            "empirical_setup_live_mixture": estimate_empirical_mixture_decoder(
                training, transition_bonus=transition_bonus
            ),
        }
        for spec in candidate_specs():
            evaluated = _evaluation(
                validation,
                bundle,
                configs[spec.duration_mode][0],
                emission_mode=spec.emission_mode,
            )
            rows_by_candidate[spec.name].extend(evaluated["recordings"])
        inner_rows.append(
            {
                **inner.to_dict(),
                "stateModelFingerprints": fingerprints,
                "emissionPriorEstimates": emission_prior_summary(bundle, training),
                "durationEstimates": {
                    mode: payload[1] for mode, payload in configs.items()
                },
            }
        )
    reports = _reports_by_group(rows_by_candidate)
    selected_name, selection = _selection_report(reports)
    selected_spec = next(spec for spec in candidate_specs() if spec.name == selected_name)

    if progress is not None:
        progress(
            f"Existing-label {fold.held_out_group}: refit outer; selected {selected_name}"
        )
    epoch_caps = {
        state: int(state_frozen["innerSelection"]["selectedEpochCaps"][state.name])
        for state in STATE_ORDER
    }
    bundle = fit_state_models(
        outer_training,
        (),
        feature_config=feature_config,
        training_config=training_config,
        seed_parts=("multistate-outer-refit", fold.held_out_group),
        epoch_caps=epoch_caps,
    )
    fingerprints = {
        state.name: _model_fingerprint(model)
        for state, model in zip(STATE_ORDER, bundle.models, strict=True)
    }
    if fingerprints != state_frozen["outerRefit"]["modelFingerprints"]:
        raise FeatureExperimentError("outer state models did not reproduce")
    configs = {
        "geometric_v1": estimate_fold_decoder(
            outer_training, transition_bonus=transition_bonus
        ),
        "empirical_setup_live_mixture": estimate_empirical_mixture_decoder(
            outer_training, transition_bonus=transition_bonus
        ),
    }
    fixed_candidates = {
        spec.name: _evaluation(
            held,
            bundle,
            configs[spec.duration_mode][0],
            emission_mode=spec.emission_mode,
        )
        for spec in candidate_specs()
    }
    reference = fixed_candidates[REFERENCE_CANDIDATE]
    _assert_metrics(
        reference["aggregate"],
        state_frozen["metrics"],
        f"{fold.held_out_group} v1 multistate",
    )
    candidate = fixed_candidates[selected_spec.name]
    return {
        "heldOutSourceGroup": fold.held_out_group,
        "trainingSourceGroups": list(fold.training_groups),
        "innerReproduction": inner_rows,
        "innerSelection": selection,
        "outerRefit": {
            "selectedCandidate": selected_name,
            "stateEpochCaps": {
                state.name: epoch_caps[state] for state in STATE_ORDER
            },
            "stateModelFingerprints": fingerprints,
            "emissionPriorEstimates": emission_prior_summary(bundle, outer_training),
            "transitionBonus": transition_bonus,
            "durationEstimates": {
                mode: payload[1] for mode, payload in configs.items()
            },
        },
        "v1Reference": reference,
        "selectedCandidate": candidate,
        "fixedCandidateDiagnostics": fixed_candidates,
    }


def _aggregate_outer(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Any]:
    recordings = [row for outer in rows for row in outer[key]["recordings"]]
    by_group = {
        str(outer["heldOutSourceGroup"]): outer[key]["aggregate"]
        for outer in rows
    }
    return {
        "aggregate": _aggregate_rows(recordings),
        "bySourceGroup": by_group,
        "macroSourceGroup": {
            metric: float(np.mean([row[metric] for row in by_group.values()]))
            for metric in ("objective", "eventF1", "timeIoU", "liveTimeRecall")
        },
        "recordings": recordings,
    }


def _aggregate_fixed_candidates(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        name: _aggregate_outer(
            [
                {
                    **row,
                    "fixed": row["fixedCandidateDiagnostics"][name],
                }
                for row in rows
            ],
            "fixed",
        )
        for name in CANDIDATE_NAMES
    }


def _paired_comparison(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    margin: float,
    consistency: float,
) -> dict[str, Any]:
    groups = sorted(reference["bySourceGroup"])
    rows = [
        {
            "sourceGroup": group,
            "referenceObjective": reference["bySourceGroup"][group]["objective"],
            "candidateObjective": candidate["bySourceGroup"][group]["objective"],
            "deltaObjectiveCandidateMinusReference": (
                candidate["bySourceGroup"][group]["objective"]
                - reference["bySourceGroup"][group]["objective"]
            ),
            "deltaLiveTimeRecall": (
                candidate["bySourceGroup"][group]["liveTimeRecall"]
                - reference["bySourceGroup"][group]["liveTimeRecall"]
            ),
        }
        for group in groups
    ]
    classification = classify_paired_deltas(
        [row["deltaObjectiveCandidateMinusReference"] for row in rows],
        margin=margin,
        sign_consistency=consistency,
    )
    return {
        "deltaDirection": "positive means the selected candidate improves",
        "classification": classification,
        "pairedSourceGroups": rows,
    }


def _promotion(
    v1_reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    operational_binary: Mapping[str, Any],
    *,
    margin: float,
    consistency: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    versus_v1 = _paired_comparison(
        v1_reference,
        candidate,
        margin=margin,
        consistency=consistency,
    )
    versus_binary = _paired_comparison(
        operational_binary,
        candidate,
        margin=margin,
        consistency=consistency,
    )
    v1 = v1_reference["aggregate"]
    improved = candidate["aggregate"]
    mechanism_checks = {
        "pairedObjectiveEvidenceHelpfulVsV1": (
            versus_v1["classification"]["classification"] == "helpful"
        ),
        "aggregateObjectiveHigherVsV1": improved["objective"] > v1["objective"],
        "liveRecallLossAtMostOnePointVsV1": improved["liveTimeRecall"]
        >= v1["liveTimeRecall"] - 0.01,
        "ordinaryLongStrictMatchesPreservedVsV1": _strict_slice_count(
            improved, "ordinaryLong"
        )
        >= _strict_slice_count(v1, "ordinaryLong"),
        "shortStrictMatchesPreservedVsV1": _strict_slice_count(
            improved, "shortAtMost3Seconds"
        )
        >= _strict_slice_count(v1, "shortAtMost3Seconds"),
        "serviceFaultStrictMatchesPreservedVsV1": _strict_slice_count(
            improved, "serviceFault"
        )
        >= _strict_slice_count(v1, "serviceFault"),
        "eventPrecisionLossAtMostOnePointVsV1": improved["eventPrecision"]
        >= v1["eventPrecision"] - 0.01,
        "deadSecondsRetainedAtMostTwoPercentHigherVsV1": improved[
            "deadSecondsRetained"
        ]
        <= v1["deadSecondsRetained"] * 1.02 + 1e-9,
        "eachSourceEventF1LossAtMostThreePointsVsV1": all(
            candidate["bySourceGroup"][group]["eventF1"]
            >= v1_reference["bySourceGroup"][group]["eventF1"] - 0.03
            for group in sorted(v1_reference["bySourceGroup"])
        ),
    }
    operational_gate = multistate_promotion_gate(
        operational_binary["aggregate"],
        improved,
        versus_binary["classification"],
    )
    operational_source_recall_guard = all(
        candidate["bySourceGroup"][group]["liveTimeRecall"]
        >= operational_binary["bySourceGroup"][group]["liveTimeRecall"] - 0.03
        for group in sorted(operational_binary["bySourceGroup"])
    )
    binary = operational_binary["aggregate"]
    operational_additional_checks = {
        "eachSourceLiveRecallLossAtMostThreePoints": operational_source_recall_guard,
        "eventPrecisionDoesNotWorsen": improved["eventPrecision"]
        >= binary["eventPrecision"],
        "deadSecondsRetainedDoesNotWorsen": improved["deadSecondsRetained"]
        <= binary["deadSecondsRetained"] + 1e-9,
        "eachSourceEventF1LossAtMostThreePoints": all(
            candidate["bySourceGroup"][group]["eventF1"]
            >= operational_binary["bySourceGroup"][group]["eventF1"] - 0.03
            for group in sorted(operational_binary["bySourceGroup"])
        ),
    }
    checks = {
        **mechanism_checks,
        **{
            f"operationalBinary_{name}": value
            for name, value in operational_gate["checks"].items()
        },
        **{
            f"operationalBinary_{name}": value
            for name, value in operational_additional_checks.items()
        },
    }
    return {
        "candidateMinusV1": versus_v1,
        "candidateMinusOperationalBinary": versus_binary,
    }, {
        "promoteSelectedCandidate": all(checks.values()),
        "checks": checks,
        "mechanismComparisonVsV1": {
            "deltaObjective": improved["objective"] - v1["objective"],
            "deltaLiveTimeRecall": improved["liveTimeRecall"]
            - v1["liveTimeRecall"],
        },
        "operationalBinaryGate": operational_gate,
    }


def run_existing_label_ablation(
    manifest: DatasetManifest,
    prepared: Sequence[PreparedRecording],
    *,
    multistate_report_path: str | Path,
    objective_margin: float = DEFAULT_OBJECTIVE_MARGIN,
    sign_consistency: float = DEFAULT_SIGN_CONSISTENCY,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    report_path, frozen, upstream_path, _upstream, names = (
        _validate_development_study(manifest, multistate_report_path)
    )
    if (
        frozen.get("promotionDecision", {}).get("selectedArchitecture")
        != "binary_control"
    ):
        raise FeatureExperimentError(
            "the frozen study does not identify its binary control as operational"
        )
    expected_ids = {
        item.id for item in manifest.recordings if item.split in DEVELOPMENT_SPLITS
    }
    if {item.recording.id for item in prepared} != expected_ids or any(
        item.recording.split not in DEVELOPMENT_SPLITS for item in prepared
    ):
        raise FeatureExperimentError("ablation requires exactly development recordings")
    feature_config = FeatureConfig.from_dict(frozen["featureConfig"])
    training_config = TrainingConfig(**frozen["trainingConfig"])
    selected = reconstruct_frozen_upstream_features(prepared, feature_config, names)
    state_frozen = {
        row["heldOutSourceGroup"]: row
        for row in frozen["serveAnchoredMultistate"]["outerFolds"]
    }
    outer_rows = []
    folds = build_fold_plan(manifest.recordings)
    for index, fold in enumerate(folds, start=1):
        if progress is not None:
            progress(
                f"Existing-label ablation outer {index}/{len(folds)}: "
                f"hold out {fold.held_out_group}"
            )
        outer_rows.append(
            _outer_fold(
                selected,
                fold,
                feature_config=feature_config,
                training_config=training_config,
                state_frozen=state_frozen[fold.held_out_group],
                progress=progress,
            )
        )
    reference = _aggregate_outer(outer_rows, "v1Reference")
    candidate = _aggregate_outer(outer_rows, "selectedCandidate")
    fixed_candidates = _aggregate_fixed_candidates(outer_rows)
    operational_binary = frozen["sameFeatureBinaryControl"]["oof"]
    _assert_metrics(
        reference["aggregate"],
        frozen["serveAnchoredMultistate"]["oof"]["aggregate"],
        "pooled v1 multistate",
    )
    if set(operational_binary["bySourceGroup"]) != set(
        candidate["bySourceGroup"]
    ):
        raise FeatureExperimentError(
            "operational binary and candidate source groups do not align"
        )
    paired, promotion = _promotion(
        reference,
        candidate,
        operational_binary,
        margin=objective_margin,
        consistency=sign_consistency,
    )
    selected_names = [row["innerSelection"]["selectedCandidate"] for row in outer_rows]
    final_candidate = min(
        CANDIDATE_NAMES,
        key=lambda name: (-selected_names.count(name), CANDIDATE_NAMES.index(name)),
    )
    unanimous_fixed_candidate = len(set(selected_names)) == 1
    if unanimous_fixed_candidate:
        for metric in ("eventF1", "timeIoU", "liveTimeRecall", "objective"):
            if not math.isclose(
                float(candidate["aggregate"][metric]),
                float(fixed_candidates[final_candidate]["aggregate"][metric]),
                abs_tol=1e-12,
            ):
                raise FeatureExperimentError(
                    "unanimous adaptive result differs from its fixed candidate"
                )
    promotion["checks"][
        "outerSelectionsUnanimouslyFreezeOneCandidate"
    ] = unanimous_fixed_candidate
    promotion["promoteSelectedCandidate"] = all(promotion["checks"].values())
    return {
        "schemaVersion": 1,
        "kind": EXISTING_LABELS_KIND,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "freezeStatus": "frozen-development-selection",
        "assessmentRole": "full-nested-development-source-group-oof-ablation",
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
        "featureNames": list(names),
        "featureSignatureSha256": _signature_sha256(names),
        "selectionProtocol": {
            "outer": "leave one development sourceGroup out",
            "inner": "leave one remaining sourceGroup out",
            "innerFoldLimit": None,
            "candidateNames": list(CANDIDATE_NAMES),
            "candidateScope": "fixed 2x2 emission-prior/duration-prior ablation",
            "priorCorrection": (
                "restore each balanced binary head's fold-observed prior odds, then renormalize"
            ),
            "durationCandidate": (
                "Laplace-smoothed empirical SETUP table and deterministic EM two-geometric LIVE mixture"
            ),
            "mixtureCollapsePolicy": (
                "report a collapsed-components flag when fitted hazard gap is below "
                f"{MIXTURE_COLLAPSE_HAZARD_GAP}; retain the candidate as a controlled "
                "SETUP-prior ablation rather than claiming bimodal LIVE behavior"
            ),
            "innerCandidateGuardrails": {
                "macroObjectiveGain": INNER_MACRO_OBJECTIVE_GAIN,
                "selectionTieMargin": INNER_TIE_MARGIN,
                "aggregateLiveRecallLoss": 0.01,
                "perSourceLiveRecallLoss": 0.03,
                "aggregateEventPrecisionLoss": 0.01,
                "maximumDeadSecondsRetainedRatio": 1.02,
                "perSourceEventF1Loss": 0.03,
                "ordinaryLongShortAndServiceFaultStrictMatches": "must not decline",
            },
            "objectiveMargin": objective_margin,
            "signConsistency": sign_consistency,
            "folds": [fold.to_dict() for fold in folds],
        },
        "operationalBinaryControl": {"oof": operational_binary},
        "v1Reference": {"oof": reference},
        "selectedCandidate": {
            "oof": candidate,
            "outerFolds": outer_rows,
        },
        "fixedCandidateDiagnostics": {
            "assessmentRole": (
                "outer-held diagnostics for interpretation only; not used for inner selection"
            ),
            "candidates": fixed_candidates,
        },
        "pairedComparison": paired,
        "promotionDecision": {
            **promotion,
            "selectedArchitecture": (
                "existing_label_multistate_candidate"
                if promotion["promoteSelectedCandidate"]
                else "binary_control"
            ),
            "testMetricsConsulted": False,
        },
        "finalizationPlan": {
            "architecture": (
                "existing_label_multistate_candidate"
                if promotion["promoteSelectedCandidate"]
                else "binary_control"
            ),
            "candidate": final_candidate,
            "sourceOuterSelections": dict(
                zip(
                    [fold.held_out_group for fold in folds],
                    selected_names,
                    strict=True,
                )
            ),
            "selectionRule": (
                "most frequent outer selection is descriptive only; promotion additionally "
                "requires every outer fold to select this same fixed candidate"
            ),
            "outerSelectionsUnanimous": unanimous_fixed_candidate,
            "stateEpochCaps": frozen["finalizationPlan"]["multistate"][
                "epochCapsByState"
            ],
            "stateSeeds": frozen["finalizationPlan"]["multistate"][
                "seedsByState"
            ],
            "transitionBonus": frozen["finalizationPlan"]["multistate"][
                "transitionBonus"
            ],
            "foldPriorRefit": (
                "re-estimate state prevalence and empirical/mixture duration priors "
                "from all development recordings after candidate freeze"
            ),
            "binaryFallback": frozen["finalizationPlan"]["binaryControl"],
            "featureNames": list(names),
            "featureSignatureSha256": _signature_sha256(names),
        },
        "guardrails": [
            "Only development recordings are prepared.",
            "Every state model reproduces the frozen nested-study fingerprint.",
            "Class priors and duration mixtures use training-fold labels only.",
            "Each outer candidate is selected from true inner OOF source groups.",
            "A heterogeneous nested policy cannot be converted into a modal deployable candidate.",
            "Final promotion must improve v1 and pass the original operational binary gate.",
            "No oracle boundaries or protected labels enter selection.",
        ],
        "provenance": _code_provenance(),
        "processing": {
            "wallClockSeconds": round(time.perf_counter() - started, 3),
            "preparedSplits": sorted(DEVELOPMENT_SPLITS),
            "protectedSplitsPrepared": False,
        },
    }
