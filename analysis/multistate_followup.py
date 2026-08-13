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
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .artifacts import atomic_write_text
from .config import FEATURE_VERSION, DecoderConfig, FeatureConfig, TrainingConfig
from .decoder import decode_probabilities
from .feature_experiments import (
    DEVELOPMENT_SPLITS,
    FeatureExperimentError,
    _fit,
    _model_fingerprint,
    _prepared_for_groups,
    _seed,
    build_fold_plan,
    objective,
    sha256_file,
)
from .metrics import (
    aggregate_evaluations,
    aggregate_outcome_slices,
    evaluate_intervals,
    ordered_interval_matches,
    outcome_slice_metrics,
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
