"""Nested weak-tag immediate-result study for the multistate decoder.

This study asks whether existing ace/service-fault tags can select a short-LIVE
duration branch.  Tags are targets only: they are never inference inputs.  The
candidate graph always includes at least one LIVE sample and deliberately
removes the v1 direct SERVE-to-DEAD shortcut.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
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
    sha256_file,
)
from .multistate import (
    STATE_ORDER,
    MultistateDecoderConfig,
    MultistateState,
    StateDurationPrior,
    decode_multistate,
)
from .multistate_existing_labels import (
    EXISTING_LABELS_KIND,
    _aggregate_outer,
    _paired_comparison,
    _promotion as _base_promotion,
)
from .multistate_feature_study import (
    _signature_sha256,
    _clip_ignored_predictions,
    estimate_fold_decoder,
    fit_state_models,
    reconstruct_frozen_upstream_features,
    state_log_scores,
)
from .multistate_followup import (
    _aggregate_rows,
    _assert_fingerprint,
    _assert_metrics,
    _evaluated_row,
    _frozen_inner_by_group,
    _strict_slice_count,
    _validate_development_study,
)
from .multistate_result_decoder import ResultDecoderConfig, decode_multistate_result
from .multistate_tag_proxy import (
    TAG_PROXY_EPOCHS,
    TAG_PROXY_FEATURE_NAMES,
    TAG_PROXY_MAX_PROBABILITY,
    TAG_PROXY_MIN_PROBABILITY,
    extract_tag_proxy_anchors,
    fit_tag_proxy,
    predict_tag_proxy,
    predict_tag_proxy_sequence,
    tag_conditioned_live_runs,
    tag_proxy_fingerprint,
    tag_proxy_metrics,
)
from .pipeline import PreparedRecording, _manifest_digest
from .schema import DatasetManifest, Interval


IMMEDIATE_RESULT_KIND = "volleycut-multistate-immediate-result-proxy-development"
CANDIDATE_NAMES = (
    "v1_noop",
    "tag_duration_mixture",
    "bounded_proxy_tag_duration",
)
REFERENCE_CANDIDATE = CANDIDATE_NAMES[0]
INNER_MACRO_OBJECTIVE_GAIN = 0.01
INNER_TIE_MARGIN = 0.005
PROXY_LOG_LOSS_GAIN = 0.01


def _load_context_report(
    manifest: DatasetManifest,
    report_path: str | Path,
    multistate_report_path: Path,
) -> tuple[Path, dict[str, Any]]:
    path = Path(report_path).expanduser().resolve()
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureExperimentError(
            f"cannot read existing-label context report {path}: {error}"
        ) from error
    if (
        not isinstance(report, dict)
        or report.get("kind") != EXISTING_LABELS_KIND
        or report.get("freezeStatus") != "frozen-development-selection"
        or report.get("testLabelsUsed") is not False
        or report.get("promotionDecision", {}).get("selectedArchitecture")
        != "binary_control"
    ):
        raise FeatureExperimentError(
            "immediate-result study requires the frozen test-closed 2x2 context"
        )
    if report.get("manifestFileSha256") != sha256_file(manifest.path):
        raise FeatureExperimentError("2x2 context does not match the manifest")
    if report.get("manifestSnapshotSha256") != _manifest_digest(manifest):
        raise FeatureExperimentError("2x2 context recording snapshots changed")
    if report.get("multistateDevelopmentReportSha256") != sha256_file(
        multistate_report_path
    ):
        raise FeatureExperimentError("2x2 context does not match the multistate report")
    return path, report


def _code_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    tracked = (
        root / "analysis" / "multistate_immediate_result.py",
        root / "analysis" / "multistate_result_decoder.py",
        root / "analysis" / "multistate_tag_proxy.py",
        root / "scripts" / "evaluate-multistate-followup.py",
        root / "analysis" / "multistate_existing_labels.py",
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


def _geometric_prior(lengths: Sequence[int]) -> tuple[StateDurationPrior, dict[str, Any]]:
    values = np.asarray(lengths, dtype=np.float64)
    if not len(values) or np.any(values < 1) or not np.isfinite(values).all():
        raise FeatureExperimentError("tag-conditioned LIVE durations are invalid")
    mean = float(np.mean(values))
    hazard = float(np.clip(1.0 / mean, 0.005, 0.95))
    prior = StateDurationPrior(
        log_scores=(math.log(hazard),),
        tail_log_score=math.log1p(-hazard),
        minimum_samples=1,
        maximum_samples=None,
    )
    return prior, {
        "runs": len(values),
        "meanSamples": mean,
        "medianSamples": float(np.median(values)),
        "minimumSamples": int(np.min(values)),
        "maximumSamples": int(np.max(values)),
        "geometricExitHazard": hazard,
    }


def estimate_result_decoder(
    training: Sequence[PreparedRecording],
    *,
    transition_bonus: float,
) -> tuple[ResultDecoderConfig, float, dict[str, Any]]:
    base, base_summary = estimate_fold_decoder(
        training, transition_bonus=transition_bonus
    )
    runs = tag_conditioned_live_runs(training)
    ordinary, ordinary_summary = _geometric_prior(runs.ordinary)
    result, result_summary = _geometric_prior(runs.immediate_result)
    anchors = extract_tag_proxy_anchors(training)
    prevalence = float(np.mean(anchors.labels))
    if not 0.0 < prevalence < 1.0:
        raise FeatureExperimentError("tag proxy prevalence must be non-degenerate")
    config = ResultDecoderConfig(
        base=base,
        ordinary_live_prior=ordinary,
        result_live_prior=result,
    )
    config.validate()
    return config, prevalence, {
        "trainingRecordingIds": [item.recording.id for item in training],
        "trainingSourceGroups": sorted(
            {item.recording.source_group for item in training}
        ),
        "taggedImmediateResultPrevalence": prevalence,
        "ordinaryLive": ordinary_summary,
        "immediateResultLive": result_summary,
        "censoredRuns": list(runs.excluded),
        "baseDurationAndTransitionEstimate": base_summary,
    }


def _noop_evaluation(
    prepared: Sequence[PreparedRecording],
    emissions: Sequence[np.ndarray],
    configs: Sequence[MultistateDecoderConfig],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item, scores, config in zip(prepared, emissions, configs, strict=True):
        decoded = decode_multistate(item.sequence.times, scores, config)
        predictions = _clip_ignored_predictions(
            [Interval(value.start, value.end) for value in decoded.intervals],
            item.recording.ignored_intervals,
        )
        row = _evaluated_row(item, predictions)
        row["decoderDiagnostics"] = {
            "directServeToDeadTransitions": sum(
                left == MultistateState.SERVE and right == MultistateState.DEAD
                for left, right in zip(decoded.states, decoded.states[1:], strict=False)
            ),
            "ordinaryBranchCount": None,
            "resultBranchCount": None,
        }
        rows.append(row)
    return {"aggregate": _aggregate_rows(rows), "recordings": rows}


def _result_evaluation(
    prepared: Sequence[PreparedRecording],
    emissions: Sequence[np.ndarray],
    probabilities: Sequence[np.ndarray],
    configs: Sequence[ResultDecoderConfig],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item, scores, proxy, config in zip(
        prepared, emissions, probabilities, configs, strict=True
    ):
        decoded = decode_multistate_result(
            item.sequence.times, scores, proxy, config
        )
        predictions = _clip_ignored_predictions(
            [Interval(value.start, value.end) for value in decoded.intervals],
            item.recording.ignored_intervals,
        )
        row = _evaluated_row(item, predictions)
        row["decoderDiagnostics"] = {
            "directServeToDeadTransitions": 0,
            "ordinaryBranchCount": decoded.ordinary_branch_count,
            "resultBranchCount": decoded.result_branch_count,
            "ordinaryDurationsSamples": list(decoded.ordinary_durations_samples),
            "resultDurationsSamples": list(decoded.result_durations_samples),
        }
        if decoded.ordinary_branch_count + decoded.result_branch_count != len(
            decoded.intervals
        ):
            raise FeatureExperimentError("result branch and interval counts diverged")
        rows.append(row)
    return {"aggregate": _aggregate_rows(rows), "recordings": rows}


def _candidate_evaluations(
    prepared: Sequence[PreparedRecording],
    state_bundle: Any,
    base: MultistateDecoderConfig,
    result: ResultDecoderConfig,
    prevalence: float,
    proxy_model: Any,
) -> dict[str, dict[str, Any]]:
    emissions = [state_log_scores(state_bundle, item.contextual_values) for item in prepared]
    constant = [np.full(len(item.sequence.times), prevalence) for item in prepared]
    learned = [predict_tag_proxy_sequence(proxy_model, item) for item in prepared]
    return {
        "v1_noop": _noop_evaluation(
            prepared, emissions, [base] * len(prepared)
        ),
        "tag_duration_mixture": _result_evaluation(
            prepared, emissions, constant, [result] * len(prepared)
        ),
        "bounded_proxy_tag_duration": _result_evaluation(
            prepared, emissions, learned, [result] * len(prepared)
        ),
    }


def _proxy_rows(
    model: Any,
    validation: Sequence[PreparedRecording],
    *,
    baseline_prevalence: float,
) -> list[dict[str, Any]]:
    anchors = extract_tag_proxy_anchors(validation)
    probabilities = predict_tag_proxy(model, anchors)
    return [
        {
            "recordingId": recording,
            "sourceGroup": group,
            "outcome": outcome,
            "label": float(label),
            "probability": float(probability),
            "baselinePrevalence": baseline_prevalence,
        }
        for recording, group, outcome, label, probability in zip(
            anchors.recording_ids,
            anchors.source_groups,
            anchors.outcome_tags,
            anchors.labels,
            probabilities,
            strict=True,
        )
    ]


def _proxy_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise FeatureExperimentError("proxy report requires held anchor rows")

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        labels = np.asarray([row["label"] for row in selected], dtype=np.float64)
        scores = np.asarray([row["probability"] for row in selected], dtype=np.float64)
        baselines = np.asarray(
            [row["baselinePrevalence"] for row in selected], dtype=np.float64
        )
        model_loss = float(
            -np.mean(labels * np.log(scores) + (1 - labels) * np.log1p(-scores))
        )
        baseline_loss = float(
            -np.mean(
                labels * np.log(baselines)
                + (1 - labels) * np.log1p(-baselines)
            )
        )
        prevalence = float(np.mean(labels))
        if 0.0 < prevalence < 1.0:
            diagnostics = tag_proxy_metrics(
                labels,
                scores,
                baseline_prevalence=prevalence,
            )
        else:
            diagnostics = {
                "samples": len(labels),
                "positive": int(np.sum(labels)),
                "negative": int(len(labels) - np.sum(labels)),
                "prevalence": prevalence,
                "meanProbability": float(np.mean(scores)),
                "logLoss": model_loss,
                "brier": float(np.mean(np.square(scores - labels))),
                "baselinePrevalence": None,
                "baselineLogLoss": None,
                "baselineBrier": None,
                "deltaLogLossProxyMinusBaseline": None,
                "deltaBrierProxyMinusBaseline": None,
                "rocAuc": None,
                "averagePrecision": None,
            }
        diagnostics.update(
            {
                "foldSpecificBaselineLogLoss": baseline_loss,
                "logLossGainVsFoldPrevalence": baseline_loss - model_loss,
            }
        )
        return diagnostics

    groups = sorted({str(row["sourceGroup"]) for row in rows})
    outcomes = ("ace", "serviceFault", "ordinary")
    return {
        "aggregate": summarize(rows),
        "bySourceGroup": {
            group: summarize([row for row in rows if row["sourceGroup"] == group])
            for group in groups
        },
        "byOutcome": {
            outcome: summarize([row for row in rows if row["outcome"] == outcome])
            for outcome in outcomes
            if any(row["outcome"] == outcome for row in rows)
        },
        "rows": list(rows),
    }


def _reports_by_group(
    rows_by_candidate: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for name in CANDIDATE_NAMES:
        rows = rows_by_candidate[name]
        groups = sorted({row["sourceGroup"] for row in rows})
        by_group = {
            group: _aggregate_rows(
                [row for row in rows if row["sourceGroup"] == group]
            )
            for group in groups
        }
        reports[name] = {
            "aggregate": _aggregate_rows(rows),
            "bySourceGroup": by_group,
            "macroSourceGroup": {
                metric: float(np.mean([row[metric] for row in by_group.values()]))
                for metric in ("objective", "eventF1", "timeIoU", "liveTimeRecall")
            },
            "recordings": rows,
        }
    return reports


def _immediate_strict(metrics: Mapping[str, Any]) -> int:
    return _strict_slice_count(metrics, "ace") + _strict_slice_count(
        metrics, "serviceFault"
    )


def _inner_gate(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    proxy: Mapping[str, Any],
    needs_proxy: bool,
) -> dict[str, Any]:
    groups = sorted(reference["bySourceGroup"])
    baseline = reference["aggregate"]
    improved = candidate["aggregate"]
    deltas = [
        candidate["bySourceGroup"][group]["objective"]
        - reference["bySourceGroup"][group]["objective"]
        for group in groups
    ]
    checks = {
        "aggregateObjectiveHigher": improved["objective"] > baseline["objective"],
        "macroObjectiveGainAtLeastOnePoint": candidate["macroSourceGroup"][
            "objective"
        ]
        >= reference["macroSourceGroup"]["objective"] + INNER_MACRO_OBJECTIVE_GAIN,
        "medianSourceObjectiveNonnegative": float(np.median(deltas)) >= 0.0,
        "atLeastTwoSourcesObjectiveNonnegative": sum(value >= 0 for value in deltas)
        >= 2,
        "aggregateLiveRecallLossAtMostOnePoint": improved["liveTimeRecall"]
        >= baseline["liveTimeRecall"] - 0.01,
        "eachSourceLiveRecallLossAtMostThreePoints": all(
            candidate["bySourceGroup"][group]["liveTimeRecall"]
            >= reference["bySourceGroup"][group]["liveTimeRecall"] - 0.03
            for group in groups
        ),
        "eventPrecisionLossAtMostOnePoint": improved["eventPrecision"]
        >= baseline["eventPrecision"] - 0.01,
        "deadSecondsRetainedAtMostTwoPercentHigher": improved[
            "deadSecondsRetained"
        ]
        <= baseline["deadSecondsRetained"] * 1.02 + 1e-9,
        "eachSourceEventF1LossAtMostThreePoints": all(
            candidate["bySourceGroup"][group]["eventF1"]
            >= reference["bySourceGroup"][group]["eventF1"] - 0.03
            for group in groups
        ),
        "ordinaryLongStrictPreserved": _strict_slice_count(
            improved, "ordinaryLong"
        )
        >= _strict_slice_count(baseline, "ordinaryLong"),
        "shortStrictPreserved": _strict_slice_count(
            improved, "shortAtMost3Seconds"
        )
        >= _strict_slice_count(baseline, "shortAtMost3Seconds"),
        "aceStrictPreserved": _strict_slice_count(improved, "ace")
        >= _strict_slice_count(baseline, "ace"),
        "serviceFaultStrictPreserved": _strict_slice_count(
            improved, "serviceFault"
        )
        >= _strict_slice_count(baseline, "serviceFault"),
        "combinedImmediateStrictImproves": _immediate_strict(improved)
        >= _immediate_strict(baseline) + 1,
    }
    if needs_proxy:
        checks.update(
            {
                "proxyLogLossImprovesAtLeast001": proxy["aggregate"][
                    "logLossGainVsFoldPrevalence"
                ]
                >= PROXY_LOG_LOSS_GAIN,
                "proxyNonworseOnAtLeastTwoSources": sum(
                    row["logLossGainVsFoldPrevalence"] >= 0.0
                    for row in proxy["bySourceGroup"].values()
                )
                >= 2,
            }
        )
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "sourceObjectiveDeltas": dict(zip(groups, deltas, strict=True)),
    }


def _select_inner(
    reports: Mapping[str, dict[str, Any]], proxy: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    if tuple(reports) != CANDIDATE_NAMES:
        raise ValueError("immediate-result candidates do not follow frozen order")
    reference = reports[REFERENCE_CANDIDATE]
    gates = {
        name: (
            {"eligible": True, "checks": {"reference": True}}
            if name == REFERENCE_CANDIDATE
            else _inner_gate(
                reference,
                reports[name],
                proxy=proxy,
                needs_proxy=name == "bounded_proxy_tag_duration",
            )
        )
        for name in CANDIDATE_NAMES
    }
    selected = REFERENCE_CANDIDATE
    for name in CANDIDATE_NAMES[1:]:
        if gates[name]["eligible"] and (
            reports[name]["aggregate"]["objective"]
            > reports[selected]["aggregate"]["objective"] + INNER_TIE_MARGIN
        ):
            selected = name
    return selected, {
        "candidateOrderSimplestFirst": list(CANDIDATE_NAMES),
        "tieMargin": INNER_TIE_MARGIN,
        "selectedCandidate": selected,
        "proxyDiagnostics": proxy,
        "candidates": {
            name: {
                "gate": gates[name],
                "aggregate": reports[name]["aggregate"],
                "bySourceGroup": reports[name]["bySourceGroup"],
            }
            for name in CANDIDATE_NAMES
        },
    }


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
    proxy_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    for index, inner in enumerate(fold.inner_folds, start=1):
        if progress is not None:
            progress(
                f"Immediate-result {fold.held_out_group}: inner {index}/3 "
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
        result_config, prevalence, prior_summary = estimate_result_decoder(
            training, transition_bonus=transition_bonus
        )
        proxy_model = fit_tag_proxy(
            training,
            feature_config=feature_config,
            training_config=training_config,
            seed_parts=(
                "immediate-result-inner",
                fold.held_out_group,
                inner.validation_group,
            ),
        )
        evaluated = _candidate_evaluations(
            validation,
            bundle,
            result_config.base,
            result_config,
            prevalence,
            proxy_model,
        )
        for name in CANDIDATE_NAMES:
            rows_by_candidate[name].extend(evaluated[name]["recordings"])
        proxy_rows.extend(
            _proxy_rows(
                proxy_model, validation, baseline_prevalence=prevalence
            )
        )
        inner_rows.append(
            {
                **inner.to_dict(),
                "stateModelFingerprints": fingerprints,
                "tagProxyModelFingerprint": tag_proxy_fingerprint(proxy_model),
                "tagProxyTrainingSummary": proxy_model.training_summary,
                "durationAndBranchPriorEstimate": prior_summary,
            }
        )
    reports = _reports_by_group(rows_by_candidate)
    proxy_report = _proxy_report(proxy_rows)
    selected_name, selection = _select_inner(reports, proxy_report)

    if progress is not None:
        progress(
            f"Immediate-result {fold.held_out_group}: refit outer; "
            f"selected {selected_name}"
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
    result_config, prevalence, prior_summary = estimate_result_decoder(
        outer_training, transition_bonus=transition_bonus
    )
    proxy_model = fit_tag_proxy(
        outer_training,
        feature_config=feature_config,
        training_config=training_config,
        seed_parts=("immediate-result-outer", fold.held_out_group),
    )
    evaluated = _candidate_evaluations(
        held,
        bundle,
        result_config.base,
        result_config,
        prevalence,
        proxy_model,
    )
    _assert_metrics(
        evaluated[REFERENCE_CANDIDATE]["aggregate"],
        state_frozen["metrics"],
        f"{fold.held_out_group} v1 immediate-result control",
    )
    outer_proxy_rows = _proxy_rows(
        proxy_model, held, baseline_prevalence=prevalence
    )
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
            "tagProxyModelFingerprint": tag_proxy_fingerprint(proxy_model),
            "tagProxyTrainingSummary": proxy_model.training_summary,
            "durationAndBranchPriorEstimate": prior_summary,
        },
        "outerProxyDiagnostics": _proxy_report(outer_proxy_rows),
        "selectedCandidate": evaluated[selected_name],
        "fixedCandidateDiagnostics": evaluated,
    }


def _aggregate_fixed(
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


def _aggregate_outer_proxy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _proxy_report(
        [
            proxy_row
            for outer in rows
            for proxy_row in outer["outerProxyDiagnostics"]["rows"]
        ]
    )


def _promotion(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    binary: Mapping[str, Any],
    *,
    selected_name: str,
    outer_proxy: Mapping[str, Any],
    unanimous: bool,
    margin: float,
    consistency: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paired, base = _base_promotion(
        reference,
        candidate,
        binary,
        margin=margin,
        consistency=consistency,
    )
    improved = candidate["aggregate"]
    v1 = reference["aggregate"]
    operational = binary["aggregate"]
    checks = dict(base["checks"])
    checks.update(
        {
            "aceStrictPreservedVsV1": _strict_slice_count(improved, "ace")
            >= _strict_slice_count(v1, "ace"),
            "combinedImmediateStrictImprovesVsV1": _immediate_strict(improved)
            > _immediate_strict(v1),
            "aceStrictPreservedVsOperationalBinary": _strict_slice_count(
                improved, "ace"
            )
            >= _strict_slice_count(operational, "ace"),
            "combinedImmediateStrictImprovesVsOperationalBinary": _immediate_strict(
                improved
            )
            > _immediate_strict(operational),
            "outerSelectionsUnanimouslyFreezeOneCandidate": unanimous,
            "candidateRemovesDirectServeToDead": (
                selected_name != REFERENCE_CANDIDATE
                and all(
                    row["decoderDiagnostics"]["directServeToDeadTransitions"] == 0
                    for row in candidate["recordings"]
                )
            ),
        }
    )
    if selected_name == "bounded_proxy_tag_duration":
        checks.update(
            {
                "proxyLogLossImprovesAtLeast001": outer_proxy["aggregate"][
                    "logLossGainVsFoldPrevalence"
                ]
                >= PROXY_LOG_LOSS_GAIN,
                "proxyNonworseOnAtLeastThreeSources": sum(
                    row["logLossGainVsFoldPrevalence"] >= 0.0
                    for row in outer_proxy["bySourceGroup"].values()
                )
                >= 3,
            }
        )
    return paired, {
        **base,
        "promoteSelectedCandidate": all(checks.values()),
        "checks": checks,
    }


def run_immediate_result_study(
    manifest: DatasetManifest,
    prepared: Sequence[PreparedRecording],
    *,
    multistate_report_path: str | Path,
    existing_label_report_path: str | Path,
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
        raise FeatureExperimentError("frozen v1 binary control is not operational")
    context_path, context = _load_context_report(
        manifest, existing_label_report_path, report_path
    )
    expected_ids = {
        item.id for item in manifest.recordings if item.split in DEVELOPMENT_SPLITS
    }
    if {item.recording.id for item in prepared} != expected_ids or any(
        item.recording.split not in DEVELOPMENT_SPLITS for item in prepared
    ):
        raise FeatureExperimentError(
            "immediate-result study requires exactly development recordings"
        )
    feature_config = FeatureConfig.from_dict(frozen["featureConfig"])
    training_config = TrainingConfig(**frozen["trainingConfig"])
    selected = reconstruct_frozen_upstream_features(
        prepared, feature_config, names
    )
    target_summary = extract_tag_proxy_anchors(selected).summary()
    state_frozen = {
        row["heldOutSourceGroup"]: row
        for row in frozen["serveAnchoredMultistate"]["outerFolds"]
    }
    folds = build_fold_plan(manifest.recordings)
    outer_rows = []
    for index, fold in enumerate(folds, start=1):
        if progress is not None:
            progress(
                f"Immediate-result outer {index}/{len(folds)}: "
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
    candidate = _aggregate_outer(outer_rows, "selectedCandidate")
    fixed = _aggregate_fixed(outer_rows)
    reference = fixed[REFERENCE_CANDIDATE]
    _assert_metrics(
        reference["aggregate"],
        frozen["serveAnchoredMultistate"]["oof"]["aggregate"],
        "pooled immediate-result v1 control",
    )
    binary = frozen["sameFeatureBinaryControl"]["oof"]
    selected_names = [
        row["innerSelection"]["selectedCandidate"] for row in outer_rows
    ]
    final_name = min(
        CANDIDATE_NAMES,
        key=lambda name: (-selected_names.count(name), CANDIDATE_NAMES.index(name)),
    )
    unanimous = len(set(selected_names)) == 1
    if unanimous:
        for metric in ("eventF1", "timeIoU", "liveTimeRecall", "objective"):
            if not math.isclose(
                float(candidate["aggregate"][metric]),
                float(fixed[final_name]["aggregate"][metric]),
                abs_tol=1e-12,
            ):
                raise FeatureExperimentError(
                    "unanimous immediate-result result differs from fixed candidate"
                )
    outer_proxy = _aggregate_outer_proxy(outer_rows)
    paired, promotion = _promotion(
        reference,
        candidate,
        binary,
        selected_name=final_name,
        outer_proxy=outer_proxy,
        unanimous=unanimous,
        margin=objective_margin,
        consistency=sign_consistency,
    )
    eligible = promotion["promoteSelectedCandidate"]
    return {
        "schemaVersion": 1,
        "kind": IMMEDIATE_RESULT_KIND,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "freezeStatus": "frozen-development-selection",
        "assessmentRole": "full-nested-development-source-group-oof-weak-tag-proxy",
        "testLabelsUsed": False,
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "manifestSnapshotSha256": _manifest_digest(manifest),
        "recordingContentSha256": {
            item.id: item.content_sha256 for item in manifest.recordings
        },
        "multistateDevelopmentReport": str(report_path),
        "multistateDevelopmentReportSha256": sha256_file(report_path),
        "existingLabelContextReport": str(context_path),
        "existingLabelContextReportSha256": sha256_file(context_path),
        "existingLabelContextSelectedArchitecture": context["promotionDecision"][
            "selectedArchitecture"
        ],
        "upstreamDevelopmentReport": str(upstream_path),
        "upstreamDevelopmentReportSha256": sha256_file(upstream_path),
        "featureVersion": FEATURE_VERSION,
        "featureConfig": feature_config.to_dict(),
        "trainingConfig": training_config.to_dict(),
        "featureNames": list(names),
        "featureSignatureSha256": _signature_sha256(names),
        "targetProtocol": {
            "name": "taggedImmediateResultProxy",
            "positive": "ace OR service-fault tag at the unique gold serve sample",
            "negative": "every other rally, including untagged short rallies",
            "conflictingAceAndFault": "hard error",
            "durationCreatesTarget": False,
            "weakProxyNotVerifiedImmediateTruth": True,
            "developmentSupport": target_summary,
        },
        "proxyProtocol": {
            "featureNames": list(TAG_PROXY_FEATURE_NAMES),
            "featureCount": len(TAG_PROXY_FEATURE_NAMES),
            "fixedEpochs": TAG_PROXY_EPOCHS,
            "validationUsedForProxyFitting": False,
            "balancedOddsPriorCorrection": True,
            "probabilityClip": [
                TAG_PROXY_MIN_PROBABILITY,
                TAG_PROXY_MAX_PROBABILITY,
            ],
            "maximumLookaheadSeconds": 2.0,
        },
        "selectionProtocol": {
            "outer": "leave one development sourceGroup out",
            "inner": "leave one remaining sourceGroup out",
            "innerFoldLimit": None,
            "candidateNames": list(CANDIDATE_NAMES),
            "candidateScope": (
                "v1 no-op, constant-prevalence tag-conditioned LIVE durations, "
                "and one bounded learned proxy using the same durations"
            ),
            "latentGraph": (
                "DEAD->SETUP->SERVE->{LIVE_ORDINARY,LIVE_RESULT}->DEAD; "
                "no SERVE->DEAD"
            ),
            "objectiveMargin": objective_margin,
            "signConsistency": sign_consistency,
            "macroObjectiveGain": INNER_MACRO_OBJECTIVE_GAIN,
            "selectionTieMargin": INNER_TIE_MARGIN,
            "proxyLogLossGain": PROXY_LOG_LOSS_GAIN,
            "folds": [fold.to_dict() for fold in folds],
        },
        "operationalBinaryControl": {"oof": binary},
        "v1MultistateControl": {"oof": reference},
        "nestedSelectedCandidate": {
            "oof": candidate,
            "outerFolds": outer_rows,
        },
        "fixedCandidateDiagnostics": {
            "assessmentRole": (
                "outer-held interpretation only; fixed rows did not select inner candidates"
            ),
            "candidates": fixed,
        },
        "tagProxyOuterOofDiagnostics": outer_proxy,
        "pairedComparison": paired,
        "promotionDecision": {
            **promotion,
            "developmentEligibleForFreshSource": eligible,
            "selectedArchitecture": "binary_control",
            "nextAssessment": (
                "fresh independent source-group validation"
                if eligible
                else "retain binary control"
            ),
            "protectedRetrospectiveOpened": False,
            "testMetricsConsulted": False,
        },
        "finalizationPlan": {
            "candidate": final_name,
            "sourceOuterSelections": dict(
                zip(
                    [fold.held_out_group for fold in folds],
                    selected_names,
                    strict=True,
                )
            ),
            "outerSelectionsUnanimous": unanimous,
            "selectionRule": (
                "modal candidate is descriptive; a deployable candidate requires unanimous "
                "outer selection and every promotion gate"
            ),
            "operationalArchitecture": "binary_control",
            "binaryFallback": frozen["finalizationPlan"]["binaryControl"],
        },
        "guardrails": [
            "Only development recordings are prepared.",
            "Tags create anchor targets and duration strata but are never inference inputs.",
            "Every state model reproduces the frozen v1 nested fingerprint.",
            "Proxy fits, prevalences, and duration priors use training sources only.",
            "Every candidate event contains at least one LIVE sample.",
            "All fixed outer OOF rows are diagnostics and do not select inner candidates.",
            "Even a passing result advances only to a fresh source, not reused protected test.",
        ],
        "limitations": [
            "Ace/service-fault tags are a coarse weak proxy, not verified immediate-result labels.",
            "The proxy consumes up to two seconds of offline lookahead.",
            "Development sources have been reused across the ordered experiment program.",
        ],
        "provenance": _code_provenance(),
        "processing": {
            "wallClockSeconds": round(time.perf_counter() - started, 3),
            "preparedSplits": sorted(DEVELOPMENT_SPLITS),
            "protectedSplitsPrepared": False,
        },
    }


__all__ = [
    "CANDIDATE_NAMES",
    "IMMEDIATE_RESULT_KIND",
    "estimate_result_decoder",
    "run_immediate_result_study",
]
