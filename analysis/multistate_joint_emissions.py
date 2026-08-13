"""Nested joint-multiclass emission study for the frozen multistate decoder.

This runner isolates the emission architecture.  The v1 geometric duration
priors, graph, and frozen transition bonus are held fixed while independent
balanced one-vs-rest heads are compared with one jointly normalized four-state
softmax model.  The primary joint model uses ordinary (empirical-prior)
categorical cross-entropy.  A separately named balanced-and-prior-corrected
joint model is retained as a secondary ablation.
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

from .config import FEATURE_VERSION, FeatureConfig, TrainingConfig
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
from .multistate import STATE_ORDER, MultistateDecoderConfig
from .multistate_existing_labels import (
    EXISTING_LABELS_KIND,
    _aggregate_outer,
    _aggregate_rows,
    _candidate_gate,
    _promotion as _base_promotion,
)
from .multistate_feature_study import (
    StateModelBundle,
    _signature_sha256,
    estimate_fold_decoder,
    evaluate_multistate_predictions,
    fit_state_models,
    reconstruct_frozen_upstream_features,
    state_log_scores,
    state_targets,
)
from .multistate_followup import (
    _assert_fingerprint,
    _assert_metrics,
    _frozen_inner_by_group,
    _validate_development_study,
    state_classification_report,
)
from .multistate_immediate_result import IMMEDIATE_RESULT_KIND
from .multistate_softmax import (
    MultistateSoftmaxModel,
    multistate_softmax_fingerprint,
    train_multistate_softmax,
)
from .pipeline import PreparedRecording, _manifest_digest
from .schema import DatasetManifest


JOINT_EMISSIONS_KIND = "volleycut-multistate-joint-emissions-development"
CANDIDATE_NAMES = (
    "v1_naive_ovr",
    "joint_empirical_softmax",
    "joint_balanced_prior_corrected_softmax",
)
REFERENCE_CANDIDATE = CANDIDATE_NAMES[0]
PRIMARY_JOINT_CANDIDATE = CANDIDATE_NAMES[1]
SECONDARY_JOINT_CANDIDATE = CANDIDATE_NAMES[2]
SOFTMAX_CANDIDATES = CANDIDATE_NAMES[1:]
SOFTMAX_PROTOCOLS: Mapping[str, tuple[str, bool]] = {
    PRIMARY_JOINT_CANDIDATE: ("empirical", False),
    SECONDARY_JOINT_CANDIDATE: ("balanced", True),
}
INNER_MACRO_OBJECTIVE_GAIN = 0.01
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


def _validate_context_report(
    manifest: DatasetManifest,
    report_path: str | Path,
    *,
    label: str,
    kind: str,
    multistate_report_path: Path,
    existing_label_report_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    path, report = _load_json(report_path, label)
    if (
        report.get("kind") != kind
        or report.get("freezeStatus") != "frozen-development-selection"
        or report.get("testLabelsUsed") is not False
        or report.get("selectionProtocol", {}).get("innerFoldLimit") not in (None, 0)
        or report.get("promotionDecision", {}).get("selectedArchitecture")
        != "binary_control"
    ):
        raise FeatureExperimentError(
            f"{label} must be frozen, full-nested, test-closed, and retain binary control"
        )
    if report.get("manifestFileSha256") != sha256_file(manifest.path):
        raise FeatureExperimentError(f"{label} does not match the manifest file")
    if report.get("manifestSnapshotSha256") != _manifest_digest(manifest):
        raise FeatureExperimentError(f"{label} recording snapshots changed")
    expected_recordings = {
        item.id: item.content_sha256 for item in manifest.recordings
    }
    if report.get("recordingContentSha256") != expected_recordings:
        raise FeatureExperimentError(f"{label} recording identities changed")
    if report.get("multistateDevelopmentReportSha256") != sha256_file(
        multistate_report_path
    ):
        raise FeatureExperimentError(f"{label} does not match the multistate report")
    if existing_label_report_path is not None:
        if report.get("existingLabelContextReportSha256") != sha256_file(
            existing_label_report_path
        ):
            raise FeatureExperimentError(
                f"{label} does not match the existing-label context"
            )
        if report.get("existingLabelContextSelectedArchitecture") != "binary_control":
            raise FeatureExperimentError(
                f"{label} did not retain the existing-label binary control"
            )
    return path, report


def _code_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    tracked = (
        root / "analysis" / "multistate_joint_emissions.py",
        root / "analysis" / "multistate_softmax.py",
        root / "analysis" / "multistate_immediate_result.py",
        root / "analysis" / "multistate_existing_labels.py",
        root / "analysis" / "multistate_followup.py",
        root / "analysis" / "multistate_feature_study.py",
        root / "analysis" / "multistate.py",
        root / "analysis" / "feature_experiments.py",
        root / "analysis" / "model.py",
        root / "analysis" / "metrics.py",
        root / "scripts" / "evaluate-multistate-followup.py",
        root / "analysis" / "tests" / "test_multistate_joint_emissions.py",
        root / "analysis" / "tests" / "test_multistate_softmax.py",
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


def _state_arrays(
    prepared: Sequence[PreparedRecording],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    values: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for item in prepared:
        mask = np.asarray(item.sample_mask, dtype=bool)
        if mask.shape != (len(item.sequence.times),):
            raise FeatureExperimentError(
                f"{item.recording.id}: state sample mask is misaligned"
            )
        selected_values = np.ascontiguousarray(item.contextual_values[mask])
        selected_targets = np.ascontiguousarray(state_targets(item)[mask])
        if not len(selected_values):
            raise FeatureExperimentError(
                f"{item.recording.id}: no usable state-training samples"
            )
        values.append(selected_values)
        targets.append(selected_targets)
    return values, targets


def _fit_softmax_models(
    training: Sequence[PreparedRecording],
    validation: Sequence[PreparedRecording],
    *,
    feature_config: FeatureConfig,
    training_config: TrainingConfig,
    seed: int,
    epoch_caps: Mapping[str, int] | None = None,
) -> dict[str, MultistateSoftmaxModel]:
    if not training:
        raise FeatureExperimentError("joint softmax requires training recordings")
    train_values, train_targets = _state_arrays(training)
    if validation:
        validation_values, validation_targets = _state_arrays(validation)
    else:
        validation_values, validation_targets = [], []
    names = training[0].contextual_names
    if any(item.contextual_names != names for item in (*training, *validation)):
        raise FeatureExperimentError("joint softmax feature signatures diverged")

    models: dict[str, MultistateSoftmaxModel] = {}
    for candidate in SOFTMAX_CANDIDATES:
        class_weighting, _prior_corrected = SOFTMAX_PROTOCOLS[candidate]
        config = replace(training_config, seed=seed)
        if epoch_caps is not None:
            cap = int(epoch_caps[candidate])
            if cap < 1:
                raise FeatureExperimentError("joint softmax epoch cap must be positive")
            config = replace(
                config,
                epochs=cap,
                patience=max(config.patience, cap + 1),
            )
        models[candidate] = train_multistate_softmax(
            train_values,
            train_targets,
            validation_values,
            validation_targets,
            feature_config,
            names,
            config,
            class_weighting=class_weighting,
        )
    return models


def _candidate_emissions(
    prepared: Sequence[PreparedRecording],
    state_bundle: StateModelBundle,
    softmax_models: Mapping[str, MultistateSoftmaxModel],
) -> dict[str, list[np.ndarray]]:
    emissions = {
        REFERENCE_CANDIDATE: [
            state_log_scores(state_bundle, item.contextual_values)
            for item in prepared
        ]
    }
    for candidate in SOFTMAX_CANDIDATES:
        _weighting, prior_corrected = SOFTMAX_PROTOCOLS[candidate]
        model = softmax_models[candidate]
        emissions[candidate] = [
            model.predict_log_scores(
                item.contextual_values,
                prior_corrected=prior_corrected,
            )
            for item in prepared
        ]
    return emissions


def _state_diagnostic_rows(
    prepared: Sequence[PreparedRecording],
    emissions: Sequence[np.ndarray],
) -> list[dict[str, Any]]:
    if len(prepared) != len(emissions):
        raise ValueError("state diagnostic inputs must be aligned")
    rows: list[dict[str, Any]] = []
    for item, log_scores in zip(prepared, emissions, strict=True):
        probabilities = np.exp(np.asarray(log_scores, dtype=np.float64))
        expected = (len(item.sequence.times), len(STATE_ORDER))
        if probabilities.shape != expected or not np.isfinite(probabilities).all():
            raise FeatureExperimentError("state diagnostic probabilities are invalid")
        if not np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-7):
            raise FeatureExperimentError("state diagnostic probabilities are not normalized")
        mask = np.asarray(item.sample_mask, dtype=bool)
        rows.append(
            {
                "recordingId": item.recording.id,
                "sourceGroup": item.recording.source_group,
                "targets": np.asarray(state_targets(item)[mask], dtype=np.int64),
                "probabilities": np.asarray(probabilities[mask], dtype=np.float64),
            }
        )
    return rows


def _state_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise FeatureExperimentError("state diagnostics require held rows")
    targets = np.concatenate(
        [np.asarray(row["targets"], dtype=np.int64) for row in rows]
    )
    probabilities = np.concatenate(
        [np.asarray(row["probabilities"], dtype=np.float64) for row in rows]
    )
    if probabilities.shape != (len(targets), len(STATE_ORDER)) or not len(targets):
        raise FeatureExperimentError("state diagnostic rows are misaligned")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise FeatureExperimentError("state diagnostic probabilities are invalid")
    if not np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-7):
        raise FeatureExperimentError("state diagnostic probabilities are not normalized")
    clipped = np.clip(probabilities, 1e-8, 1.0)
    one_hot = np.eye(len(STATE_ORDER), dtype=np.float64)[targets]
    report = state_classification_report(targets, probabilities)
    report.update(
        {
            "multiclassLogLoss": float(
                -np.mean(np.log(clipped[np.arange(len(targets)), targets]))
            ),
            "multiclassBrier": float(
                np.mean(np.sum(np.square(probabilities - one_hot), axis=1))
            ),
            "meanMaximumProbability": float(
                np.mean(np.max(probabilities, axis=1))
            ),
        }
    )
    return report


def _state_diagnostic_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups = sorted({str(row["sourceGroup"]) for row in rows})
    return {
        "aggregate": _state_diagnostics(rows),
        "bySourceGroup": {
            group: _state_diagnostics(
                [row for row in rows if row["sourceGroup"] == group]
            )
            for group in groups
        },
    }


def _candidate_evaluations(
    prepared: Sequence[PreparedRecording],
    state_bundle: StateModelBundle,
    softmax_models: Mapping[str, MultistateSoftmaxModel],
    decoder_config: MultistateDecoderConfig,
) -> dict[str, dict[str, Any]]:
    emissions = _candidate_emissions(prepared, state_bundle, softmax_models)
    result: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATE_NAMES:
        rows, aggregate = evaluate_multistate_predictions(
            prepared,
            emissions[candidate],
            [decoder_config] * len(prepared),
        )
        result[candidate] = {
            "aggregate": aggregate,
            "recordings": rows,
            "_stateDiagnosticRows": _state_diagnostic_rows(
                prepared, emissions[candidate]
            ),
        }
    return result


def _public_evaluation(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregate": evaluation["aggregate"],
        "recordings": evaluation["recordings"],
    }


def _reports_by_group(
    rows_by_candidate: Mapping[str, list[dict[str, Any]]],
    state_rows_by_candidate: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATE_NAMES:
        rows = rows_by_candidate[candidate]
        groups = sorted({str(row["sourceGroup"]) for row in rows})
        by_group = {
            group: _aggregate_rows(
                [row for row in rows if row["sourceGroup"] == group]
            )
            for group in groups
        }
        aggregate = _aggregate_rows(rows)
        reports[candidate] = {
            "aggregate": aggregate,
            "bySourceGroup": by_group,
            "macroSourceGroup": {
                metric: float(np.mean([row[metric] for row in by_group.values()]))
                for metric in ("objective", "eventF1", "timeIoU", "liveTimeRecall")
            },
            "stateDiagnostics": _state_diagnostic_report(
                state_rows_by_candidate[candidate]
            ),
            "recordings": rows,
        }
    return reports


def _select_inner(
    reports: Mapping[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if tuple(reports) != CANDIDATE_NAMES:
        raise ValueError("joint-emission candidates do not follow frozen order")
    reference = reports[REFERENCE_CANDIDATE]
    gates = {
        candidate: (
            {"eligible": True, "checks": {"reference": True}}
            if candidate == REFERENCE_CANDIDATE
            else _candidate_gate(reference, reports[candidate])
        )
        for candidate in CANDIDATE_NAMES
    }
    selected = REFERENCE_CANDIDATE
    for candidate in CANDIDATE_NAMES[1:]:
        if gates[candidate]["eligible"] and (
            reports[candidate]["aggregate"]["objective"]
            > reports[selected]["aggregate"]["objective"] + INNER_TIE_MARGIN
        ):
            selected = candidate
    return selected, {
        "candidateOrderSimplestFirst": list(CANDIDATE_NAMES),
        "primaryCandidate": PRIMARY_JOINT_CANDIDATE,
        "secondaryCandidate": SECONDARY_JOINT_CANDIDATE,
        "tieMargin": INNER_TIE_MARGIN,
        "selectedCandidate": selected,
        "calibrationUsedForSelection": False,
        "candidates": {
            candidate: {
                "gate": gates[candidate],
                "aggregate": reports[candidate]["aggregate"],
                "bySourceGroup": reports[candidate]["bySourceGroup"],
                "stateDiagnostics": reports[candidate]["stateDiagnostics"],
            }
            for candidate in CANDIDATE_NAMES
        },
    }


def _median_epoch_cap(values: Sequence[int]) -> int:
    epochs = np.asarray(values, dtype=np.float64)
    if not len(epochs) or not np.isfinite(epochs).all() or np.any(epochs < 1):
        raise FeatureExperimentError("softmax inner best epochs are invalid")
    return max(1, int(round(float(np.median(epochs)))))


def _softmax_metadata(
    models: Mapping[str, MultistateSoftmaxModel],
) -> dict[str, Any]:
    return {
        candidate: {
            "classWeighting": SOFTMAX_PROTOCOLS[candidate][0],
            "priorCorrectedAtInference": SOFTMAX_PROTOCOLS[candidate][1],
            "modelFingerprint": multistate_softmax_fingerprint(model),
            "trainingSummary": model.training_summary,
        }
        for candidate, model in models.items()
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
    if not outer_training or not held:
        raise FeatureExperimentError("joint-emission outer fold is empty")
    frozen_inner = _frozen_inner_by_group(
        state_frozen["innerSelection"]["foldsUsed"]
    )
    transition_bonus = float(
        state_frozen["innerSelection"]["selectedTransitionBonus"]
    )
    rows_by_candidate = {candidate: [] for candidate in CANDIDATE_NAMES}
    state_rows_by_candidate = {candidate: [] for candidate in CANDIDATE_NAMES}
    epoch_history = {candidate: [] for candidate in SOFTMAX_CANDIDATES}
    inner_rows: list[dict[str, Any]] = []

    for index, inner in enumerate(fold.inner_folds, start=1):
        if progress is not None:
            progress(
                f"Joint emissions {fold.held_out_group}: inner {index}/"
                f"{len(fold.inner_folds)} validate {inner.validation_group}"
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
        state_fingerprints = {
            state.name: _model_fingerprint(model)
            for state, model in zip(STATE_ORDER, bundle.models, strict=True)
        }
        frozen_fold = frozen_inner.get(inner.validation_group)
        if frozen_fold is None:
            raise FeatureExperimentError("frozen inner source group is missing")
        for state in STATE_ORDER:
            _assert_fingerprint(
                state_fingerprints[state.name],
                frozen_fold["heads"][state.name]["modelFingerprint"],
                f"{fold.held_out_group}/{inner.validation_group}/{state.name}",
            )
        decoder_config, decoder_summary = estimate_fold_decoder(
            training, transition_bonus=transition_bonus
        )
        joint_seed = _seed(
            training_config.seed,
            "joint-softmax-inner",
            fold.held_out_group,
            inner.validation_group,
        )
        softmax_models = _fit_softmax_models(
            training,
            validation,
            feature_config=feature_config,
            training_config=training_config,
            seed=joint_seed,
        )
        for candidate, model in softmax_models.items():
            epoch_history[candidate].append(
                int(model.training_summary["bestEpoch"])
            )
        evaluated = _candidate_evaluations(
            validation,
            bundle,
            softmax_models,
            decoder_config,
        )
        for candidate in CANDIDATE_NAMES:
            rows_by_candidate[candidate].extend(evaluated[candidate]["recordings"])
            state_rows_by_candidate[candidate].extend(
                evaluated[candidate]["_stateDiagnosticRows"]
            )
        inner_rows.append(
            {
                **inner.to_dict(),
                "stateModelFingerprints": state_fingerprints,
                "jointSoftmaxSeed": joint_seed,
                "jointSoftmaxModels": _softmax_metadata(softmax_models),
                "v1DecoderEstimate": decoder_summary,
            }
        )

    reports = _reports_by_group(rows_by_candidate, state_rows_by_candidate)
    selected_name, selection = _select_inner(reports)
    epoch_caps = {
        candidate: _median_epoch_cap(epoch_history[candidate])
        for candidate in SOFTMAX_CANDIDATES
    }

    if progress is not None:
        progress(
            f"Joint emissions {fold.held_out_group}: refit outer; "
            f"selected {selected_name}"
        )
    state_epoch_caps = {
        state: int(
            state_frozen["innerSelection"]["selectedEpochCaps"][state.name]
        )
        for state in STATE_ORDER
    }
    bundle = fit_state_models(
        outer_training,
        (),
        feature_config=feature_config,
        training_config=training_config,
        seed_parts=("multistate-outer-refit", fold.held_out_group),
        epoch_caps=state_epoch_caps,
    )
    state_fingerprints = {
        state.name: _model_fingerprint(model)
        for state, model in zip(STATE_ORDER, bundle.models, strict=True)
    }
    if state_fingerprints != state_frozen["outerRefit"]["modelFingerprints"]:
        raise FeatureExperimentError("outer state models did not reproduce")
    decoder_config, decoder_summary = estimate_fold_decoder(
        outer_training, transition_bonus=transition_bonus
    )
    outer_seed = _seed(
        training_config.seed,
        "joint-softmax-outer",
        fold.held_out_group,
    )
    softmax_models = _fit_softmax_models(
        outer_training,
        (),
        feature_config=feature_config,
        training_config=training_config,
        seed=outer_seed,
        epoch_caps=epoch_caps,
    )
    evaluated = _candidate_evaluations(
        held,
        bundle,
        softmax_models,
        decoder_config,
    )
    _assert_metrics(
        evaluated[REFERENCE_CANDIDATE]["aggregate"],
        state_frozen["metrics"],
        f"{fold.held_out_group} joint-emission v1 control",
    )
    fixed_public = {
        candidate: _public_evaluation(evaluated[candidate])
        for candidate in CANDIDATE_NAMES
    }
    outer_state_diagnostics = {
        candidate: _state_diagnostic_report(
            evaluated[candidate]["_stateDiagnosticRows"]
        )
        for candidate in CANDIDATE_NAMES
    }
    return {
        "heldOutSourceGroup": fold.held_out_group,
        "trainingSourceGroups": list(fold.training_groups),
        "innerReproduction": inner_rows,
        "innerSelection": selection,
        "outerRefit": {
            "selectedCandidate": selected_name,
            "transitionBonus": transition_bonus,
            "v1DecoderEstimate": decoder_summary,
            "stateEpochCaps": {
                state.name: state_epoch_caps[state] for state in STATE_ORDER
            },
            "stateModelFingerprints": state_fingerprints,
            "jointSoftmaxSeed": outer_seed,
            "jointSoftmaxEpochCaps": epoch_caps,
            "jointSoftmaxModels": _softmax_metadata(softmax_models),
        },
        "outerStateDiagnostics": outer_state_diagnostics,
        "v1Reference": fixed_public[REFERENCE_CANDIDATE],
        "selectedCandidate": fixed_public[selected_name],
        "fixedCandidateDiagnostics": fixed_public,
        "_stateDiagnosticRowsByCandidate": {
            candidate: evaluated[candidate]["_stateDiagnosticRows"]
            for candidate in CANDIDATE_NAMES
        },
        "_selectedStateDiagnosticRows": evaluated[selected_name][
            "_stateDiagnosticRows"
        ],
    }


def _aggregate_fixed(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        candidate: _aggregate_outer(
            [
                {
                    **row,
                    "fixed": row["fixedCandidateDiagnostics"][candidate],
                }
                for row in rows
            ],
            "fixed",
        )
        for candidate in CANDIDATE_NAMES
    }


def _promotion(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    binary: Mapping[str, Any],
    *,
    selected_name: str,
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
    checks = dict(base["checks"])
    checks.update(
        {
            "selectedCandidateIsJointMulticlass": selected_name
            != REFERENCE_CANDIDATE,
            "outerSelectionsUnanimouslyFreezeOneCandidate": unanimous,
        }
    )
    return paired, {
        **base,
        "checks": checks,
        "promoteSelectedCandidate": all(checks.values()),
    }


def run_joint_emissions_study(
    manifest: DatasetManifest,
    prepared: Sequence[PreparedRecording],
    *,
    multistate_report_path: str | Path,
    existing_label_report_path: str | Path,
    immediate_result_report_path: str | Path,
    objective_margin: float = DEFAULT_OBJECTIVE_MARGIN,
    sign_consistency: float = DEFAULT_SIGN_CONSISTENCY,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run full nested development OOF for joint versus independent emissions."""

    started = time.perf_counter()
    report_path, frozen, upstream_path, _upstream, names = (
        _validate_development_study(manifest, multistate_report_path)
    )
    if (
        frozen.get("promotionDecision", {}).get("selectedArchitecture")
        != "binary_control"
    ):
        raise FeatureExperimentError("frozen v1 binary control is not operational")
    existing_path, existing_context = _validate_context_report(
        manifest,
        existing_label_report_path,
        label="existing-label context report",
        kind=EXISTING_LABELS_KIND,
        multistate_report_path=report_path,
    )
    immediate_path, immediate_context = _validate_context_report(
        manifest,
        immediate_result_report_path,
        label="immediate-result context report",
        kind=IMMEDIATE_RESULT_KIND,
        multistate_report_path=report_path,
        existing_label_report_path=existing_path,
    )

    expected_ids = {
        item.id for item in manifest.recordings if item.split in DEVELOPMENT_SPLITS
    }
    if {item.recording.id for item in prepared} != expected_ids or any(
        item.recording.split not in DEVELOPMENT_SPLITS for item in prepared
    ):
        raise FeatureExperimentError(
            "joint-emission study requires exactly development recordings"
        )
    feature_config = FeatureConfig.from_dict(frozen["featureConfig"])
    training_config = TrainingConfig(**frozen["trainingConfig"])
    selected = reconstruct_frozen_upstream_features(
        prepared, feature_config, names
    )
    state_frozen = {
        row["heldOutSourceGroup"]: row
        for row in frozen["serveAnchoredMultistate"]["outerFolds"]
    }
    folds = build_fold_plan(manifest.recordings)
    if set(state_frozen) != {fold.held_out_group for fold in folds}:
        raise FeatureExperimentError("frozen state outer folds do not align")

    outer_rows: list[dict[str, Any]] = []
    for index, fold in enumerate(folds, start=1):
        if progress is not None:
            progress(
                f"Joint emissions outer {index}/{len(folds)}: "
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

    selected_state_rows = [
        diagnostic
        for outer in outer_rows
        for diagnostic in outer["_selectedStateDiagnosticRows"]
    ]
    fixed_state_rows = {
        candidate: [
            diagnostic
            for outer in outer_rows
            for diagnostic in outer["_stateDiagnosticRowsByCandidate"][candidate]
        ]
        for candidate in CANDIDATE_NAMES
    }
    selected_state_diagnostics = _state_diagnostic_report(selected_state_rows)
    fixed_state_diagnostics = {
        candidate: _state_diagnostic_report(fixed_state_rows[candidate])
        for candidate in CANDIDATE_NAMES
    }
    for outer in outer_rows:
        outer.pop("_selectedStateDiagnosticRows")
        outer.pop("_stateDiagnosticRowsByCandidate")

    candidate = _aggregate_outer(outer_rows, "selectedCandidate")
    fixed = _aggregate_fixed(outer_rows)
    reference = fixed[REFERENCE_CANDIDATE]
    _assert_metrics(
        reference["aggregate"],
        frozen["serveAnchoredMultistate"]["oof"]["aggregate"],
        "pooled joint-emission v1 control",
    )
    binary = frozen["sameFeatureBinaryControl"]["oof"]
    if set(binary["bySourceGroup"]) != set(candidate["bySourceGroup"]):
        raise FeatureExperimentError(
            "operational binary and joint candidate source groups do not align"
        )
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
                    "unanimous joint-emission result differs from fixed candidate"
                )
    paired, promotion = _promotion(
        reference,
        candidate,
        binary,
        selected_name=final_name,
        unanimous=unanimous,
        margin=objective_margin,
        consistency=sign_consistency,
    )
    eligible = promotion["promoteSelectedCandidate"]

    return {
        "schemaVersion": 1,
        "kind": JOINT_EMISSIONS_KIND,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "freezeStatus": "frozen-development-selection",
        "assessmentRole": "full-nested-development-source-group-oof-joint-emission-ablation",
        "testLabelsUsed": False,
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "manifestSnapshotSha256": _manifest_digest(manifest),
        "recordingContentSha256": {
            item.id: item.content_sha256 for item in manifest.recordings
        },
        "multistateDevelopmentReport": str(report_path),
        "multistateDevelopmentReportSha256": sha256_file(report_path),
        "existingLabelContextReport": str(existing_path),
        "existingLabelContextReportSha256": sha256_file(existing_path),
        "existingLabelContextSelectedArchitecture": existing_context[
            "promotionDecision"
        ]["selectedArchitecture"],
        "immediateResultContextReport": str(immediate_path),
        "immediateResultContextReportSha256": sha256_file(immediate_path),
        "immediateResultContextSelectedArchitecture": immediate_context[
            "promotionDecision"
        ]["selectedArchitecture"],
        "upstreamDevelopmentReport": str(upstream_path),
        "upstreamDevelopmentReportSha256": sha256_file(upstream_path),
        "featureVersion": FEATURE_VERSION,
        "featureConfig": feature_config.to_dict(),
        "trainingConfig": training_config.to_dict(),
        "featureNames": list(names),
        "featureSignatureSha256": _signature_sha256(names),
        "emissionProtocol": {
            "stateOrder": [state.name for state in STATE_ORDER],
            "v1Control": (
                "four independently inverse-frequency-balanced logistic heads, "
                "renormalized row-wise without prior correction"
            ),
            "primaryJointCandidate": {
                "name": PRIMARY_JOINT_CANDIDATE,
                "loss": "unweighted categorical cross-entropy",
                "classPrior": "learned natively from each training fold",
                "priorCorrection": False,
            },
            "secondaryJointCandidate": {
                "name": SECONDARY_JOINT_CANDIDATE,
                "role": "separately named secondary ablation",
                "loss": "inverse-frequency-balanced categorical cross-entropy",
                "classPrior": "training-fold empirical prior restored in logit space",
                "priorCorrection": True,
            },
            "decoderHeldFixed": (
                "v1 geometric fold-training durations, graph, and frozen transition bonus"
            ),
            "calibrationDiagnosticsUsedForSelection": False,
        },
        "selectionProtocol": {
            "outer": "leave one development sourceGroup out",
            "inner": "leave one remaining sourceGroup out",
            "innerFoldLimit": None,
            "candidateNames": list(CANDIDATE_NAMES),
            "candidateOrder": (
                "v1 control, primary empirical joint softmax, secondary "
                "balanced/corrected joint softmax"
            ),
            "candidateScope": "emission architecture only; decoder configuration unchanged",
            "softmaxOuterEpochRule": (
                "per mode, rounded median inner bestEpoch; outer refit has no validation"
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
        "operationalBinaryControl": {"oof": binary},
        "v1MultistateControl": {
            "oof": reference,
            "stateCrossFittedDiagnostics": fixed_state_diagnostics[
                REFERENCE_CANDIDATE
            ],
        },
        "nestedSelectedCandidate": {
            "oof": candidate,
            "stateCrossFittedDiagnostics": selected_state_diagnostics,
            "outerFolds": outer_rows,
        },
        "fixedCandidateDiagnostics": {
            "assessmentRole": (
                "outer-held diagnostics for interpretation only; fixed rows did "
                "not select inner candidates"
            ),
            "candidates": {
                name: {
                    **fixed[name],
                    "stateCrossFittedDiagnostics": fixed_state_diagnostics[name],
                }
                for name in CANDIDATE_NAMES
            },
        },
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
            "primaryCandidate": PRIMARY_JOINT_CANDIDATE,
            "secondaryCandidate": SECONDARY_JOINT_CANDIDATE,
            "sourceOuterSelections": dict(
                zip(
                    [fold.held_out_group for fold in folds],
                    selected_names,
                    strict=True,
                )
            ),
            "outerSelectionsUnanimous": unanimous,
            "selectionRule": (
                "modal candidate is descriptive; advancement requires unanimous "
                "outer selection and every v1/operational-binary promotion gate"
            ),
            "softmaxEpochCapsByHeldSource": {
                row["heldOutSourceGroup"]: row["outerRefit"][
                    "jointSoftmaxEpochCaps"
                ]
                for row in outer_rows
            },
            "operationalArchitecture": "binary_control",
            "binaryFallback": frozen["finalizationPlan"]["binaryControl"],
        },
        "guardrails": [
            "Only development recordings are prepared.",
            "Every independent OVR state model reproduces the frozen v1 fingerprint.",
            "Softmax normalization, class weighting, priors, and epoch selection are fold-local.",
            "The decoder graph, geometric priors, and transition bonus remain v1-exact.",
            "State calibration and confusion use only cross-fitted held-source predictions.",
            "All fixed outer OOF arms are diagnostics and do not select inner candidates.",
            "A passing result advances only to a fresh source; protected test stays closed.",
        ],
        "limitations": [
            "Joint softmax remains a linear model over the frozen feature signature.",
            "Cross-fitted calibration is descriptive and is not separately calibrated.",
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
    "JOINT_EMISSIONS_KIND",
    "PRIMARY_JOINT_CANDIDATE",
    "SECONDARY_JOINT_CANDIDATE",
    "run_joint_emissions_study",
]
