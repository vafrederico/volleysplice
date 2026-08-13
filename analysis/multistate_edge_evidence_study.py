"""Nested fixed-strength serve and terminal transition-evidence study.

The four-arm family is deliberately small: no edge evidence, serve evidence,
terminal evidence, and both.  Specialists are independently cross-fitted and
their balanced probabilities are mapped to bounded likelihood-ratio proxies
``2*p - 1`` with coefficient one.  No threshold or evidence-strength search is
permitted.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import FEATURE_VERSION, DecoderConfig, FeatureConfig, TrainingConfig
from .dead_state import end_transition_labels_for_times
from .feature_experiments import (
    DEFAULT_OBJECTIVE_MARGIN,
    DEFAULT_SIGN_CONSISTENCY,
    DEVELOPMENT_SPLITS,
    FeatureExperimentError,
    _model_fingerprint,
    _prepared_for_groups,
    _seed,
    build_fold_plan,
    sha256_file,
)
from .metrics import (
    aggregate_evaluations,
    aggregate_outcome_slices,
    evaluate_intervals,
    outcome_slice_metrics,
)
from .model import (
    DEAD_STATE_TASK,
    SERVE_CONTACT_TASK,
    LogisticModel,
    train_logistic_model,
)
from .multistate import STATE_ORDER, MultistateDecoderConfig, MultistateState
from .multistate_existing_labels import (
    EXISTING_LABELS_KIND,
    _candidate_gate,
    _promotion as _base_promotion,
    _strict_slice_count,
    emission_log_scores,
    emission_prior_summary,
)
from .multistate_feature_study import (
    _clip_ignored_predictions,
    _signature_sha256,
    estimate_fold_decoder,
    fit_state_models,
    reconstruct_frozen_upstream_features,
)
from .multistate_followup import (
    _assert_fingerprint,
    _frozen_inner_by_group,
    _validate_development_study,
)
from .multistate_immediate_result import IMMEDIATE_RESULT_KIND
from .multistate_joint_emissions import JOINT_EMISSIONS_KIND
from .multistate_transition_evidence import (
    TransitionEvidence,
    decode_multistate_with_transition_evidence,
)
from .pipeline import PreparedRecording, _manifest_digest
from .schema import DatasetManifest, Interval
from .serve import ServeDetection, match_serve_contacts, serve_labels_for_times


EDGE_EVIDENCE_KIND = "volleycut-multistate-edge-evidence-development"
CONTROL_NAME = "prior_corrected_ovr_geometric_no_edge"
SERVE_NAME = "serve_edge"
TERMINAL_NAME = "terminal_edge"
BOTH_NAME = "serve_and_terminal_edges"
ARM_NAMES = (CONTROL_NAME, SERVE_NAME, TERMINAL_NAME, BOTH_NAME)
SERVE_RADIUS_SECONDS = 1.0
TERMINAL_BEFORE_SECONDS = 2.0
TERMINAL_AFTER_SECONDS = 2.0
TERMINAL_PRE_SERVE_NEGATIVE_SECONDS = 1.0
EDGE_EVIDENCE_COEFFICIENT = 1.0
INNER_TIE_MARGIN = 0.005


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureExperimentError(f"cannot read {label} {resolved}: {error}") from error
    if not isinstance(payload, dict):
        raise FeatureExperimentError(f"{label} must contain one JSON object")
    return resolved, payload


def _validate_context(
    manifest: DatasetManifest,
    path: str | Path,
    *,
    label: str,
    kind: str,
    multistate_sha256: str,
    required_hashes: Mapping[str, str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    resolved, report = _load_json(path, label)
    if (
        report.get("kind") != kind
        or report.get("freezeStatus") != "frozen-development-selection"
        or report.get("testLabelsUsed") is not False
        or report.get("selectionProtocol", {}).get("innerFoldLimit") not in (None, 0)
        or report.get("promotionDecision", {}).get("selectedArchitecture")
        != "binary_control"
    ):
        raise FeatureExperimentError(
            f"{label} must be full-nested, frozen, test-closed, and retain binary control"
        )
    if report.get("manifestFileSha256") != sha256_file(manifest.path):
        raise FeatureExperimentError(f"{label} manifest file changed")
    if report.get("manifestSnapshotSha256") != _manifest_digest(manifest):
        raise FeatureExperimentError(f"{label} manifest snapshots changed")
    if report.get("recordingContentSha256") != {
        item.id: item.content_sha256 for item in manifest.recordings
    }:
        raise FeatureExperimentError(f"{label} recording identities changed")
    if report.get("multistateDevelopmentReportSha256") != multistate_sha256:
        raise FeatureExperimentError(f"{label} does not match the multistate report")
    for field, expected in (required_hashes or {}).items():
        if report.get(field) != expected:
            raise FeatureExperimentError(f"{label} context hash changed: {field}")
    return resolved, report


def _code_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    files = (
        root / "analysis" / "multistate_edge_evidence_study.py",
        root / "analysis" / "multistate_transition_evidence.py",
        root / "analysis" / "multistate_existing_labels.py",
        root / "analysis" / "multistate_feature_study.py",
        root / "analysis" / "multistate_followup.py",
        root / "analysis" / "multistate.py",
        root / "analysis" / "serve.py",
        root / "analysis" / "dead_state.py",
        root / "analysis" / "model.py",
        root / "analysis" / "metrics.py",
        root / "scripts" / "evaluate-multistate-followup.py",
        root / "analysis" / "tests" / "test_multistate_edge_evidence_study.py",
        root / "analysis" / "tests" / "test_multistate_transition_evidence.py",
    )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True, timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=root,
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        head, dirty = None, None
    return {
        "gitHead": head,
        "gitDirty": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "filesSha256": {
            str(path.relative_to(root)): sha256_file(path)
            for path in files if path.is_file()
        },
    }


def _serve_target(item: PreparedRecording) -> tuple[np.ndarray, np.ndarray]:
    labels = serve_labels_for_times(
        item.sequence.times, item.recording.rallies, SERVE_RADIUS_SECONDS
    )
    return labels, np.asarray(item.sample_mask, dtype=bool)


def _terminal_target(item: PreparedRecording) -> tuple[np.ndarray, np.ndarray]:
    times = item.sequence.times
    labels, mask = end_transition_labels_for_times(
        times,
        item.recording.rallies,
        before_seconds=TERMINAL_BEFORE_SECONDS,
        after_seconds=TERMINAL_AFTER_SECONDS,
    )
    for rally in item.recording.rallies:
        setup = (
            (times >= max(0.0, rally.start - TERMINAL_PRE_SERVE_NEGATIVE_SECONDS))
            & (times < rally.start)
        )
        mask |= setup
        labels[setup] = 0.0
    return labels, mask & np.asarray(item.sample_mask, dtype=bool)


def _target_arrays(
    prepared: Sequence[PreparedRecording], kind: str
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    if kind not in ("serve", "terminal"):
        raise ValueError("specialist kind must be serve or terminal")
    values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    positives = 0
    negatives = 0
    for item in prepared:
        target, mask = _serve_target(item) if kind == "serve" else _terminal_target(item)
        if mask.shape != (len(item.sequence.times),) or target.shape != mask.shape:
            raise FeatureExperimentError(f"{kind} target is not sample-aligned")
        selected_values = np.ascontiguousarray(item.contextual_values[mask])
        selected_labels = np.ascontiguousarray(target[mask], dtype=np.float32)
        if not len(selected_values):
            raise FeatureExperimentError(f"{item.recording.id}: empty {kind} target")
        values.append(selected_values)
        labels.append(selected_labels)
        positives += int(np.sum(selected_labels > 0.5))
        negatives += int(np.sum(selected_labels <= 0.5))
    if positives == 0 or negatives == 0:
        raise FeatureExperimentError(f"{kind} specialist target needs both classes")
    return values, labels, {
        "recordingIds": [item.recording.id for item in prepared],
        "sourceGroups": sorted({item.recording.source_group for item in prepared}),
        "positiveSamples": positives,
        "negativeSamples": negatives,
    }


def _fit_specialist(
    training: Sequence[PreparedRecording],
    validation: Sequence[PreparedRecording],
    *,
    kind: str,
    feature_config: FeatureConfig,
    training_config: TrainingConfig,
    seed: int,
    epoch_cap: int | None = None,
) -> tuple[LogisticModel, dict[str, Any]]:
    train_values, train_labels, target_summary = _target_arrays(training, kind)
    if validation:
        validation_values, validation_labels, validation_summary = _target_arrays(
            validation, kind
        )
    else:
        validation_values, validation_labels = [], []
        validation_summary = None
    names = training[0].contextual_names
    if any(item.contextual_names != names for item in (*training, *validation)):
        raise FeatureExperimentError(f"{kind} specialist signatures diverged")
    config = replace(training_config, seed=seed)
    if epoch_cap is not None:
        if epoch_cap < 1:
            raise FeatureExperimentError("specialist epoch cap must be positive")
        config = replace(config, epochs=epoch_cap, patience=max(config.patience, epoch_cap + 1))
    model = train_logistic_model(
        train_values,
        train_labels,
        validation_values,
        validation_labels,
        feature_config,
        names,
        DecoderConfig(),
        config,
        prediction_task=(SERVE_CONTACT_TASK if kind == "serve" else DEAD_STATE_TASK),
    )
    return model, {
        "kind": kind,
        "target": target_summary,
        "validationTarget": validation_summary,
        "modelFingerprint": _model_fingerprint(model),
        "trainingSummary": model.training_summary,
    }


def _edge_values(model: LogisticModel, item: PreparedRecording) -> np.ndarray:
    probabilities = np.asarray(model.predict(item.contextual_values), dtype=np.float64)
    if probabilities.shape != (len(item.sequence.times),) or not np.isfinite(probabilities).all():
        raise FeatureExperimentError("specialist probabilities are invalid")
    return EDGE_EVIDENCE_COEFFICIENT * (2.0 * probabilities - 1.0)


def _arm_evidence(
    arm: str,
    item: PreparedRecording,
    serve_model: LogisticModel,
    terminal_model: LogisticModel,
) -> TransitionEvidence | None:
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown edge-evidence arm: {arm}")
    evidence: dict[tuple[MultistateState, MultistateState], np.ndarray] = {}
    if arm in (SERVE_NAME, BOTH_NAME):
        evidence[(MultistateState.SETUP, MultistateState.SERVE)] = _edge_values(
            serve_model, item
        )
    if arm in (TERMINAL_NAME, BOTH_NAME):
        evidence[(MultistateState.LIVE, MultistateState.DEAD)] = _edge_values(
            terminal_model, item
        )
    return evidence or None


def _serve_anchor_row(
    item: PreparedRecording, predictions: Sequence[Interval]
) -> dict[str, Any]:
    truth = [rally.start for rally in item.recording.rallies]
    detections = [ServeDetection(interval.start, 1.0) for interval in predictions]
    matches = match_serve_contacts(truth, detections, 0.5)
    return {
        "true": len(truth),
        "predicted": len(detections),
        "matched": len(matches),
        "errorsSeconds": [error for _, _, error in matches],
    }


def _aggregate_serve_anchor(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    true_count = sum(int(row["serveAnchorWithin05"]["true"]) for row in rows)
    predicted = sum(int(row["serveAnchorWithin05"]["predicted"]) for row in rows)
    matched = sum(int(row["serveAnchorWithin05"]["matched"]) for row in rows)
    errors = [
        abs(float(error))
        for row in rows for error in row["serveAnchorWithin05"]["errorsSeconds"]
    ]
    precision = matched / predicted if predicted else (1.0 if not true_count else 0.0)
    recall = matched / true_count if true_count else (1.0 if not predicted else 0.0)
    return {
        "toleranceSeconds": 0.5,
        "trueServes": true_count,
        "predictedServes": predicted,
        "matchedServes": matched,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "timingMaeSeconds": float(np.mean(errors)) if errors else None,
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_evaluations(rows)
    aggregate["outcomeSlices"] = aggregate_outcome_slices(
        [row["outcomeSlices"] for row in rows]
    )
    from .feature_experiments import objective

    aggregate["objective"] = objective(aggregate)
    aggregate["serveAnchorWithin05"] = _aggregate_serve_anchor(rows)
    return aggregate


def _evaluate_arms(
    prepared: Sequence[PreparedRecording],
    bundle: Any,
    decoder_config: MultistateDecoderConfig,
    serve_model: LogisticModel,
    terminal_model: LogisticModel,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        rows: list[dict[str, Any]] = []
        for item in prepared:
            emissions = emission_log_scores(
                bundle, item.contextual_values, mode="fold_prior_corrected_ovr"
            )
            decoded = decode_multistate_with_transition_evidence(
                item.sequence.times,
                emissions,
                decoder_config,
                _arm_evidence(arm, item, serve_model, terminal_model),
            )
            intervals = _clip_ignored_predictions(
                [Interval(value.start, value.end) for value in decoded.intervals],
                item.recording.ignored_intervals,
            )
            metrics = evaluate_intervals(item.recording.rallies, intervals)
            metrics["outcomeSlices"] = outcome_slice_metrics(
                item.recording.rallies, intervals
            )
            metrics.update(
                {
                    "id": item.recording.id,
                    "sourceGroup": item.recording.source_group,
                    "environment": item.recording.environment,
                    "decodedStateSamples": {
                        state.name: sum(value == state for value in decoded.states)
                        for state in STATE_ORDER
                    },
                    "serveAnchorWithin05": _serve_anchor_row(item, intervals),
                }
            )
            rows.append(metrics)
        result[arm] = {"aggregate": _aggregate_rows(rows), "recordings": rows}
    return result


def _reports_by_group(
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        rows = list(rows_by_arm[arm])
        groups = sorted({str(row["sourceGroup"]) for row in rows})
        by_group = {
            group: _aggregate_rows(
                [row for row in rows if row["sourceGroup"] == group]
            ) for group in groups
        }
        result[arm] = {
            "aggregate": _aggregate_rows(rows),
            "bySourceGroup": by_group,
            "macroSourceGroup": {
                metric: float(np.mean([row[metric] for row in by_group.values()]))
                for metric in ("objective", "eventF1", "timeIoU", "liveTimeRecall")
            },
            "recordings": rows,
        }
    return result


def _aggregate_outer_with_anchor(
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


def _finite_mae(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise FeatureExperimentError(f"{label} boundary MAE is unavailable")
    return float(value)


def _mechanism_gate(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    arm: str,
    independent_gates: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    base = _candidate_gate(control, candidate)
    base_metrics = control["aggregate"]
    improved = candidate["aggregate"]
    checks = dict(base["checks"])
    if arm in (SERVE_NAME, BOTH_NAME):
        checks.update(
            {
                "startBoundaryMaeNonworsening": _finite_mae(
                    improved["startBoundaryMaeSeconds"], "candidate start"
                ) <= _finite_mae(base_metrics["startBoundaryMaeSeconds"], "control start") + 1e-12,
                "serveAnchorWithin05RecallImproves": improved["serveAnchorWithin05"]["recall"]
                > base_metrics["serveAnchorWithin05"]["recall"] + 1e-12,
            }
        )
    if arm in (TERMINAL_NAME, BOTH_NAME):
        end_gain = _finite_mae(base_metrics["endBoundaryMaeSeconds"], "control end") - _finite_mae(
            improved["endBoundaryMaeSeconds"], "candidate end"
        )
        slice_gain = (
            _strict_slice_count(improved, "shortAtMost3Seconds")
            + _strict_slice_count(improved, "serviceFault")
            - _strict_slice_count(base_metrics, "shortAtMost3Seconds")
            - _strict_slice_count(base_metrics, "serviceFault")
        )
        checks.update(
            {
                "endBoundaryMaeNonworsening": end_gain >= -1e-12,
                "terminalMechanismGain": slice_gain >= 2 or end_gain >= 0.05 - 1e-12,
            }
        )
    if arm == BOTH_NAME:
        gates = independent_gates or {}
        checks["serveArmIndependentlyEligible"] = bool(
            gates.get(SERVE_NAME, {}).get("eligible")
        )
        checks["terminalArmIndependentlyEligible"] = bool(
            gates.get(TERMINAL_NAME, {}).get("eligible")
        )
    return {
        **base,
        "eligible": all(checks.values()),
        "checks": checks,
    }


def _select_inner(
    reports: Mapping[str, dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    if tuple(reports) != ARM_NAMES:
        raise ValueError("edge-evidence arms do not follow frozen order")
    control = reports[CONTROL_NAME]
    gates: dict[str, dict[str, Any]] = {
        CONTROL_NAME: {"eligible": True, "checks": {"control": True}}
    }
    gates[SERVE_NAME] = _mechanism_gate(control, reports[SERVE_NAME], arm=SERVE_NAME)
    gates[TERMINAL_NAME] = _mechanism_gate(
        control, reports[TERMINAL_NAME], arm=TERMINAL_NAME
    )
    gates[BOTH_NAME] = _mechanism_gate(
        control, reports[BOTH_NAME], arm=BOTH_NAME, independent_gates=gates
    )
    selected = CONTROL_NAME
    for arm in ARM_NAMES[1:]:
        if gates[arm]["eligible"] and (
            reports[arm]["aggregate"]["objective"]
            > reports[selected]["aggregate"]["objective"] + INNER_TIE_MARGIN
        ):
            selected = arm
    return selected, {
        "candidateOrderSimplestFirst": list(ARM_NAMES),
        "tieMargin": INNER_TIE_MARGIN,
        "selectedCandidate": selected,
        "candidates": {
            arm: {
                "gate": gates[arm],
                "aggregate": reports[arm]["aggregate"],
                "bySourceGroup": reports[arm]["bySourceGroup"],
            } for arm in ARM_NAMES
        },
    }


def _median_epoch(values: Sequence[int]) -> int:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all() or np.any(array < 1):
        raise FeatureExperimentError("specialist inner epochs are invalid")
    return max(1, int(round(float(np.median(array)))))


def _assert_control(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for metric in (
        "eventF1", "eventPrecision", "timeIoU", "liveTimeRecall",
        "liveTimePrecision", "deadSecondsRetained", "objective",
    ):
        if not math.isclose(float(actual[metric]), float(expected[metric]), abs_tol=1e-10):
            raise FeatureExperimentError(f"{label} control did not reproduce: {metric}")
    for slice_name in ("ordinaryLong", "shortAtMost3Seconds", "serviceFault"):
        if _strict_slice_count(actual, slice_name) != _strict_slice_count(expected, slice_name):
            raise FeatureExperimentError(f"{label} control slice did not reproduce: {slice_name}")


def _outer_fold(
    selected: Sequence[PreparedRecording],
    fold: Any,
    *,
    feature_config: FeatureConfig,
    training_config: TrainingConfig,
    state_frozen: Mapping[str, Any],
    expected_control: Mapping[str, Any],
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    outer_training = _prepared_for_groups(selected, fold.training_groups)
    held = _prepared_for_groups(selected, (fold.held_out_group,))
    frozen_inner = _frozen_inner_by_group(state_frozen["innerSelection"]["foldsUsed"])
    transition_bonus = float(state_frozen["innerSelection"]["selectedTransitionBonus"])
    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_NAMES}
    histories = {"serve": [], "terminal": []}
    inner_rows: list[dict[str, Any]] = []
    for index, inner in enumerate(fold.inner_folds, start=1):
        if progress:
            progress(
                f"Edge evidence {fold.held_out_group}: inner {index}/3 "
                f"validate {inner.validation_group}"
            )
        training = _prepared_for_groups(outer_training, inner.training_groups)
        validation = _prepared_for_groups(outer_training, (inner.validation_group,))
        bundle = fit_state_models(
            training, validation, feature_config=feature_config,
            training_config=training_config,
            seed_parts=("multistate-inner", fold.held_out_group, inner.validation_group),
        )
        fingerprints = {
            state.name: _model_fingerprint(model)
            for state, model in zip(STATE_ORDER, bundle.models, strict=True)
        }
        frozen_fold = frozen_inner[inner.validation_group]
        for state in STATE_ORDER:
            _assert_fingerprint(
                fingerprints[state.name], frozen_fold["heads"][state.name]["modelFingerprint"],
                f"{fold.held_out_group}/{inner.validation_group}/{state.name}",
            )
        specialists: dict[str, LogisticModel] = {}
        metadata: dict[str, Any] = {}
        for kind in ("serve", "terminal"):
            seed = _seed(
                training_config.seed, f"multistate-{kind}-edge-inner",
                fold.held_out_group, inner.validation_group,
            )
            specialists[kind], metadata[kind] = _fit_specialist(
                training, validation, kind=kind, feature_config=feature_config,
                training_config=training_config, seed=seed,
            )
            histories[kind].append(int(specialists[kind].training_summary["bestEpoch"]))
            metadata[kind]["seed"] = seed
        decoder, decoder_summary = estimate_fold_decoder(
            training, transition_bonus=transition_bonus
        )
        evaluated = _evaluate_arms(
            validation, bundle, decoder, specialists["serve"], specialists["terminal"]
        )
        for arm in ARM_NAMES:
            rows_by_arm[arm].extend(evaluated[arm]["recordings"])
        inner_rows.append(
            {
                **inner.to_dict(),
                "stateModelFingerprints": fingerprints,
                "emissionPriorEstimates": emission_prior_summary(bundle, training),
                "v1DecoderEstimate": decoder_summary,
                "specialists": metadata,
            }
        )
    reports = _reports_by_group(rows_by_arm)
    selected_name, selection = _select_inner(reports)
    epoch_caps = {kind: _median_epoch(histories[kind]) for kind in histories}

    if progress:
        progress(
            f"Edge evidence {fold.held_out_group}: refit outer; selected {selected_name}"
        )
    state_epoch_caps = {
        state: int(state_frozen["innerSelection"]["selectedEpochCaps"][state.name])
        for state in STATE_ORDER
    }
    bundle = fit_state_models(
        outer_training, (), feature_config=feature_config,
        training_config=training_config,
        seed_parts=("multistate-outer-refit", fold.held_out_group),
        epoch_caps=state_epoch_caps,
    )
    fingerprints = {
        state.name: _model_fingerprint(model)
        for state, model in zip(STATE_ORDER, bundle.models, strict=True)
    }
    if fingerprints != state_frozen["outerRefit"]["modelFingerprints"]:
        raise FeatureExperimentError("outer state models did not reproduce")
    specialists = {}
    metadata = {}
    for kind in ("serve", "terminal"):
        seed = _seed(
            training_config.seed, f"multistate-{kind}-edge-outer", fold.held_out_group
        )
        specialists[kind], metadata[kind] = _fit_specialist(
            outer_training, (), kind=kind, feature_config=feature_config,
            training_config=training_config, seed=seed, epoch_cap=epoch_caps[kind],
        )
        metadata[kind]["seed"] = seed
        metadata[kind]["frozenEpochCap"] = epoch_caps[kind]
    decoder, decoder_summary = estimate_fold_decoder(
        outer_training, transition_bonus=transition_bonus
    )
    evaluated = _evaluate_arms(
        held, bundle, decoder, specialists["serve"], specialists["terminal"]
    )
    _assert_control(
        evaluated[CONTROL_NAME]["aggregate"], expected_control,
        f"{fold.held_out_group} prior-corrected control",
    )
    return {
        "heldOutSourceGroup": fold.held_out_group,
        "trainingSourceGroups": list(fold.training_groups),
        "innerReproduction": inner_rows,
        "innerSelection": selection,
        "outerRefit": {
            "selectedCandidate": selected_name,
            "transitionBonus": transition_bonus,
            "v1DecoderEstimate": decoder_summary,
            "stateEpochCaps": {state.name: state_epoch_caps[state] for state in STATE_ORDER},
            "stateModelFingerprints": fingerprints,
            "emissionPriorEstimates": emission_prior_summary(bundle, outer_training),
            "specialistEpochCaps": epoch_caps,
            "specialists": metadata,
        },
        "control": evaluated[CONTROL_NAME],
        "selectedCandidate": evaluated[selected_name],
        "fixedCandidateDiagnostics": evaluated,
    }


def _aggregate_fixed(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        arm: _aggregate_outer_with_anchor(
            [{**row, "fixed": row["fixedCandidateDiagnostics"][arm]} for row in rows],
            "fixed",
        ) for arm in ARM_NAMES
    }


def _promotion(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    binary: Mapping[str, Any],
    fixed: Mapping[str, Mapping[str, Any]],
    *,
    selected_name: str,
    unanimous: bool,
    margin: float,
    consistency: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paired, base = _base_promotion(
        control, candidate, binary, margin=margin, consistency=consistency
    )
    serve_gate = _mechanism_gate(control, fixed[SERVE_NAME], arm=SERVE_NAME)
    terminal_gate = _mechanism_gate(control, fixed[TERMINAL_NAME], arm=TERMINAL_NAME)
    selected_gate = (
        {"eligible": False, "checks": {"selectedIsNotControl": False}}
        if selected_name == CONTROL_NAME
        else _mechanism_gate(
            control, candidate, arm=selected_name,
            independent_gates={SERVE_NAME: serve_gate, TERMINAL_NAME: terminal_gate},
        )
    )
    checks = dict(base["checks"])
    checks.update(
        {
            "selectedCandidateUsesEdgeEvidence": selected_name != CONTROL_NAME,
            "outerSelectionsUnanimouslyFreezeOneCandidate": unanimous,
            "selectedMechanismGatePasses": selected_gate["eligible"],
        }
    )
    if selected_name == BOTH_NAME:
        checks["serveArmIndependentlyPasses"] = serve_gate["eligible"]
        checks["terminalArmIndependentlyPasses"] = terminal_gate["eligible"]
    return paired, {
        **base,
        "checks": checks,
        "promoteSelectedCandidate": all(checks.values()),
        "selectedMechanismGate": selected_gate,
        "independentServeGate": serve_gate,
        "independentTerminalGate": terminal_gate,
    }


def run_edge_evidence_study(
    manifest: DatasetManifest,
    prepared: Sequence[PreparedRecording],
    *,
    multistate_report_path: str | Path,
    existing_label_report_path: str | Path,
    immediate_result_report_path: str | Path,
    joint_emissions_report_path: str | Path,
    objective_margin: float = DEFAULT_OBJECTIVE_MARGIN,
    sign_consistency: float = DEFAULT_SIGN_CONSISTENCY,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    multistate_path, frozen, upstream_path, _upstream, names = (
        _validate_development_study(manifest, multistate_report_path)
    )
    if frozen.get("promotionDecision", {}).get("selectedArchitecture") != "binary_control":
        raise FeatureExperimentError("frozen operational architecture is not binary control")
    multistate_sha = sha256_file(multistate_path)
    existing_path, existing = _validate_context(
        manifest, existing_label_report_path, label="existing-label report",
        kind=EXISTING_LABELS_KIND, multistate_sha256=multistate_sha,
    )
    existing_sha = sha256_file(existing_path)
    immediate_path, _immediate = _validate_context(
        manifest, immediate_result_report_path, label="immediate-result report",
        kind=IMMEDIATE_RESULT_KIND, multistate_sha256=multistate_sha,
        required_hashes={"existingLabelContextReportSha256": existing_sha},
    )
    immediate_sha = sha256_file(immediate_path)
    joint_path, _joint = _validate_context(
        manifest, joint_emissions_report_path, label="joint-emissions report",
        kind=JOINT_EMISSIONS_KIND, multistate_sha256=multistate_sha,
        required_hashes={
            "existingLabelContextReportSha256": existing_sha,
            "immediateResultContextReportSha256": immediate_sha,
        },
    )
    expected_ids = {
        item.id for item in manifest.recordings if item.split in DEVELOPMENT_SPLITS
    }
    if {item.recording.id for item in prepared} != expected_ids or any(
        item.recording.split not in DEVELOPMENT_SPLITS for item in prepared
    ):
        raise FeatureExperimentError("edge-evidence study requires exactly development rows")
    feature_config = FeatureConfig.from_dict(frozen["featureConfig"])
    training_config = TrainingConfig(**frozen["trainingConfig"])
    selected = reconstruct_frozen_upstream_features(prepared, feature_config, names)
    folds = build_fold_plan(manifest.recordings)
    state_frozen = {
        row["heldOutSourceGroup"]: row
        for row in frozen["serveAnchoredMultistate"]["outerFolds"]
    }
    control_context = existing["fixedCandidateDiagnostics"]["candidates"][
        "fold_prior_corrected_ovr__geometric_v1"
    ]
    if set(control_context["bySourceGroup"]) != {fold.held_out_group for fold in folds}:
        raise FeatureExperimentError("edge control context source groups do not align")
    outer_rows = []
    for index, fold in enumerate(folds, start=1):
        if progress:
            progress(
                f"Edge evidence outer {index}/{len(folds)}: hold out {fold.held_out_group}"
            )
        outer_rows.append(
            _outer_fold(
                selected, fold, feature_config=feature_config,
                training_config=training_config,
                state_frozen=state_frozen[fold.held_out_group],
                expected_control=control_context["bySourceGroup"][fold.held_out_group],
                progress=progress,
            )
        )
    candidate = _aggregate_outer_with_anchor(outer_rows, "selectedCandidate")
    fixed = _aggregate_fixed(outer_rows)
    control = fixed[CONTROL_NAME]
    _assert_control(control["aggregate"], control_context["aggregate"], "pooled")
    binary = frozen["sameFeatureBinaryControl"]["oof"]
    selected_names = [row["innerSelection"]["selectedCandidate"] for row in outer_rows]
    final_name = min(
        ARM_NAMES,
        key=lambda name: (-selected_names.count(name), ARM_NAMES.index(name)),
    )
    unanimous = len(set(selected_names)) == 1
    if unanimous:
        _assert_control(candidate["aggregate"], fixed[final_name]["aggregate"], "unanimous")
    paired, promotion = _promotion(
        control, candidate, binary, fixed, selected_name=final_name,
        unanimous=unanimous, margin=objective_margin, consistency=sign_consistency,
    )
    eligible = promotion["promoteSelectedCandidate"]
    return {
        "schemaVersion": 1,
        "kind": EDGE_EVIDENCE_KIND,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "freezeStatus": "frozen-development-selection",
        "assessmentRole": "full-nested-development-source-group-oof-fixed-edge-evidence-ablation",
        "testLabelsUsed": False,
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "manifestSnapshotSha256": _manifest_digest(manifest),
        "recordingContentSha256": {item.id: item.content_sha256 for item in manifest.recordings},
        "multistateDevelopmentReport": str(multistate_path),
        "multistateDevelopmentReportSha256": multistate_sha,
        "existingLabelContextReport": str(existing_path),
        "existingLabelContextReportSha256": existing_sha,
        "immediateResultContextReport": str(immediate_path),
        "immediateResultContextReportSha256": immediate_sha,
        "jointEmissionsContextReport": str(joint_path),
        "jointEmissionsContextReportSha256": sha256_file(joint_path),
        "upstreamDevelopmentReport": str(upstream_path),
        "upstreamDevelopmentReportSha256": sha256_file(upstream_path),
        "featureVersion": FEATURE_VERSION,
        "featureConfig": feature_config.to_dict(),
        "trainingConfig": training_config.to_dict(),
        "featureNames": list(names),
        "featureSignatureSha256": _signature_sha256(names),
        "evidenceProtocol": {
            "arms": list(ARM_NAMES),
            "control": "fold-prior-corrected OVR emissions with geometric v1 decoder",
            "serveTarget": {"helper": "serve_labels_for_times", "radiusSeconds": SERVE_RADIUS_SECONDS},
            "terminalTarget": {
                "helper": "end_transition_labels_for_times",
                "beforeSeconds": TERMINAL_BEFORE_SECONDS,
                "afterSeconds": TERMINAL_AFTER_SECONDS,
                "preServeAlreadyDeadNegativeSeconds": TERMINAL_PRE_SERVE_NEGATIVE_SECONDS,
            },
            "mapping": "balanced specialist probability p maps to e(t)=2*p-1",
            "coefficient": EDGE_EVIDENCE_COEFFICIENT,
            "serveEdge": "SETUP->SERVE at destination sample",
            "terminalEdge": "LIVE->DEAD at destination sample",
            "thresholdOrStrengthSearch": False,
            "bothArmRequiresIndependentServeAndTerminalEligibility": True,
        },
        "selectionProtocol": {
            "outer": "leave one development sourceGroup out",
            "inner": "leave one remaining sourceGroup out",
            "innerFoldLimit": None,
            "candidateNames": list(ARM_NAMES),
            "specialistOuterEpochRule": "rounded median of three inner bestEpoch values",
            "tieMargin": INNER_TIE_MARGIN,
            "objectiveMargin": objective_margin,
            "signConsistency": sign_consistency,
            "folds": [fold.to_dict() for fold in folds],
        },
        "operationalBinaryControl": {"oof": binary},
        "priorCorrectedMultistateControl": {"oof": control},
        "nestedSelectedCandidate": {"oof": candidate, "outerFolds": outer_rows},
        "fixedCandidateDiagnostics": {
            "assessmentRole": "outer-held interpretation only; arms did not tune strength or thresholds",
            "candidates": fixed,
        },
        "pairedComparison": paired,
        "promotionDecision": {
            **promotion,
            "developmentEligibleForFreshSource": eligible,
            "selectedArchitecture": "binary_control",
            "nextAssessment": "fresh independent source-group validation" if eligible else "retain binary control",
            "protectedRetrospectiveOpened": False,
            "testMetricsConsulted": False,
        },
        "finalizationPlan": {
            "candidate": final_name,
            "sourceOuterSelections": dict(zip(
                [fold.held_out_group for fold in folds], selected_names, strict=True
            )),
            "outerSelectionsUnanimous": unanimous,
            "specialistEpochCapsByHeldSource": {
                row["heldOutSourceGroup"]: row["outerRefit"]["specialistEpochCaps"]
                for row in outer_rows
            },
            "operationalArchitecture": "binary_control",
            "binaryFallback": frozen["finalizationPlan"]["binaryControl"],
        },
        "guardrails": [
            "Only train and validation recordings are prepared.",
            "State heads reproduce frozen v1 fingerprints; prior correction is training-fold-only.",
            "Serve and terminal specialists are fit independently inside every source fold.",
            "Evidence has fixed coefficient one and affects only its named legal transition.",
            "The both arm can advance only when serve and terminal arms independently pass.",
            "A passing result advances only to a fresh source; protected test remains closed.",
        ],
        "limitations": [
            "Targets use existing serve/end boundaries rather than the pending transition-cue pilot.",
            "Balanced probabilities are bounded evidence proxies, not calibrated posteriors.",
            "Development source groups have been reused across the ordered program.",
        ],
        "provenance": _code_provenance(),
        "processing": {
            "wallClockSeconds": round(time.perf_counter() - started, 3),
            "preparedSplits": sorted(DEVELOPMENT_SPLITS),
            "protectedSplitsPrepared": False,
        },
    }


__all__ = [
    "ARM_NAMES",
    "BOTH_NAME",
    "CONTROL_NAME",
    "EDGE_EVIDENCE_KIND",
    "SERVE_NAME",
    "TERMINAL_NAME",
    "run_edge_evidence_study",
]
