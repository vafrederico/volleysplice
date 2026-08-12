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
from .feature_experiments import (
    DEFAULT_OBJECTIVE_MARGIN,
    DEFAULT_SIGN_CONSISTENCY,
    DEVELOPMENT_SPLITS,
    FeatureExperimentError,
    FeatureSet,
    _aggregate_artifacts,
    _fold_report,
    _model_fingerprint,
    _protected_summary,
    _run_candidate_fold,
    _seed,
    _select_decoder,
    _subset_prepared,
    _fit,
    build_fold_plan,
    classify_paired_deltas,
    objective,
    sha256_file,
)
from .features import ABSOLUTE_FEATURE_NAMES, percentile_rank_values
from .pipeline import (
    PreparedRecording,
    _evaluate_prepared_probabilities,
    _manifest_digest,
    _prepare_many,
)
from .schema import DatasetManifest, load_manifest


TRANSITION_EXPERIMENT_SCHEMA_VERSION = 1
DEFAULT_WINDOWS_SECONDS = (0.5, 1.0, 2.0, 4.0, 8.0)
EXPECTED_BASE_SIGNAL_COUNT = 90
EXPECTED_BASELINE_CONTEXT_COUNT = 450
HIGH_ACTIVITY_THRESHOLD = 0.65
SERVE_PROXY_THRESHOLD = 0.80
RECENT_SERVE_HORIZON_SECONDS = 30.0


@dataclass(frozen=True)
class DerivedFeatureBlock:
    """A once-per-timestamp feature block that can be appended without context expansion."""

    values: np.ndarray
    names: tuple[str, ...]
    groups: Mapping[str, tuple[int, ...]]
    definitions: Mapping[str, Any]


@dataclass(frozen=True)
class TransitionPrepared:
    prepared: tuple[PreparedRecording, ...]
    candidates: tuple[FeatureSet, ...]
    baseline_names: tuple[str, ...]
    derived_names: tuple[str, ...]
    multiscale_names: tuple[str, ...]
    interaction_names: tuple[str, ...]
    definitions: Mapping[str, Any]


SUMMARY_SOURCE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "player_motion": {
        "operator": "mean",
        "inputs": (
            "player_motion_mean",
            "player_motion_p90",
            "player_motion_active_fraction",
            "quality_gated_player_motion",
        ),
    },
    "receiving_motion_onset": {
        "operator": "maximum",
        "inputs": ("player_motion_onset", "receiving_formation_change_proxy"),
    },
    "motion_collapse": {
        "operator": "maximum",
        "inputs": ("player_motion_collapse", "synchronized_stand_down"),
    },
    "formation_contraction": {
        "operator": "one-minus-mean",
        "inputs": ("player_motion_spread_x", "player_motion_spread_y"),
    },
    "frame_difference": {
        "operator": "mean",
        "inputs": ("diff_mean", "diff_p90", "diff_active_fraction"),
    },
    "optical_flow": {
        "operator": "mean",
        "inputs": ("flow_mean", "flow_p90", "flow_active_fraction"),
    },
    "coherent_flow": {
        "operator": "geometric-mean",
        "inputs": ("flow_mean", "player_motion_coherence"),
    },
    "audio_transient": {
        "operator": "mean-times-audio-available",
        "inputs": (
            "audio_spectral_flux",
            "audio_rms_novelty",
            "audio_onset_strength",
            "audio_contact_like_transient",
            "audio_available",
        ),
    },
    "audio_cadence": {
        "operator": "value-times-audio-available",
        "inputs": ("audio_onset_cadence", "audio_available"),
    },
    "cadence_collapse": {
        "operator": "value-times-audio-available",
        "inputs": ("audio_cadence_collapse", "audio_available"),
    },
    "visibility_quality": {
        "operator": "identity",
        "inputs": ("visibility_quality",),
    },
    "camera_shift": {
        "operator": "identity",
        "inputs": ("camera_shift_magnitude",),
    },
}


INTERACTION_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "serve_peak_x_receiving_motion_onset",
        "formula": "serve_peak_proxy * receiving_motion_onset",
        "brainstormHypothesis": "serve peak x receiving-side motion onset",
    },
    {
        "id": "audio_transient_x_coherent_flow",
        "formula": "audio_transient * coherent_flow",
        "brainstormHypothesis": "audio transient x coherent court-directed optical flow",
    },
    {
        "id": "terminal_transient_x_motion_collapse",
        "formula": "audio_transient * motion_collapse",
        "brainstormHypothesis": "terminal transient x motion collapse",
    },
    {
        "id": "cadence_collapse_x_formation_contraction",
        "formula": "cadence_collapse * formation_contraction",
        "brainstormHypothesis": "cadence collapse x formation contraction",
    },
    {
        "id": "visibility_x_residual_player_motion",
        "formula": "visibility_quality * player_motion",
        "brainstormHypothesis": "visibility quality x residual player motion",
    },
    {
        "id": "camera_shift_x_frame_difference",
        "formula": "camera_shift * frame_difference",
        "brainstormHypothesis": "camera shift x frame difference",
    },
    {
        "id": "recent_serve_x_elapsed_x_dead_state",
        "formula": (
            "decayed_recent_serve_proxy * clipped_elapsed_since_serve_proxy "
            "* mean(motion_collapse, cadence_collapse, synchronized_stand_down)"
        ),
        "brainstormHypothesis": (
            "recent serve evidence x elapsed-live duration x dead-state evidence"
        ),
    },
)


def _duration_token(seconds: float) -> str:
    return f"{seconds:g}".replace(".", "p")


def _ranked_base_values(item: PreparedRecording) -> dict[str, np.ndarray]:
    if item.sequence.values.shape != (len(item.sequence.times), len(item.sequence.names)):
        raise FeatureExperimentError("raw feature sequence shape does not match its signature")
    ranked = percentile_rank_values(item.sequence.values)
    for index, name in enumerate(item.sequence.names):
        if name in ABSOLUTE_FEATURE_NAMES:
            ranked[:, index] = item.sequence.values[:, index]
    if not np.isfinite(ranked).all():
        raise FeatureExperimentError("base sequence contains non-finite values")
    return {name: ranked[:, index].astype(np.float64) for index, name in enumerate(item.sequence.names)}


def _require(base: Mapping[str, np.ndarray], *names: str) -> list[np.ndarray]:
    missing = [name for name in names if name not in base]
    if missing:
        raise FeatureExperimentError(
            f"transition features require missing base signals: {missing}"
        )
    return [base[name] for name in names]


def _semantic_sources(base: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, definition in SUMMARY_SOURCE_DEFINITIONS.items():
        inputs = _require(base, *definition["inputs"])
        operator = definition["operator"]
        if operator == "mean":
            values = np.mean(np.vstack(inputs), axis=0)
        elif operator == "maximum":
            values = np.max(np.vstack(inputs), axis=0)
        elif operator == "one-minus-mean":
            values = 1.0 - np.mean(np.vstack(inputs), axis=0)
        elif operator == "geometric-mean":
            values = np.sqrt(np.maximum(inputs[0], 0.0) * np.maximum(inputs[1], 0.0))
        elif operator == "mean-times-audio-available":
            values = np.mean(np.vstack(inputs[:-1]), axis=0) * inputs[-1]
        elif operator == "value-times-audio-available":
            values = inputs[0] * inputs[-1]
        elif operator == "identity":
            values = inputs[0]
        else:  # pragma: no cover - definitions are module constants
            raise FeatureExperimentError(f"unknown semantic-source operator {operator!r}")
        result[name] = np.clip(values, 0.0, 1.0)
    return result


def _sample_step_seconds(times: np.ndarray) -> float:
    if len(times) <= 1:
        return 1.0
    differences = np.diff(times)
    if not np.isfinite(differences).all() or np.any(differences <= 0):
        raise FeatureExperimentError("feature timestamps must be finite and strictly increasing")
    return float(np.median(differences))


def _rolling_mean(values: np.ndarray, span: int, *, trailing: bool = False) -> np.ndarray:
    count = len(values)
    if count == 0:
        return values.copy()
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    indexes = np.arange(count)
    if trailing:
        left = np.maximum(0, indexes - span + 1)
        right = indexes + 1
    else:
        radius = span
        left = np.maximum(0, indexes - radius)
        right = np.minimum(count, indexes + radius + 1)
    return (prefix[right] - prefix[left]) / np.maximum(right - left, 1)


def _centered_statistics(
    values: np.ndarray,
    *,
    radius: int,
    step_seconds: float,
) -> dict[str, np.ndarray]:
    if not len(values):
        empty = values.astype(np.float64, copy=True)
        return {name: empty for name in (
            "mean", "maximum", "variance", "q25", "q75", "slope",
            "acceleration", "post_minus_pre",
        )}
    width = 2 * radius + 1
    padded = np.pad(values.astype(np.float64), (radius, radius), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    valid = np.isfinite(windows)
    counts = np.sum(valid, axis=1)
    safe = np.where(valid, windows, 0.0)
    means = np.sum(safe, axis=1) / np.maximum(counts, 1)
    variances = np.sum(np.where(valid, np.square(windows - means[:, None]), 0.0), axis=1) / np.maximum(counts, 1)
    maximum = np.nanmax(windows, axis=1)
    q25 = _centered_nearest_rank_quantile(values, radius, 0.25)
    q75 = _centered_nearest_rank_quantile(values, radius, 0.75)

    coordinates = np.arange(-radius, radius + 1, dtype=np.float64) * step_seconds
    coordinate_rows = np.broadcast_to(coordinates, windows.shape)
    sum_t = np.sum(np.where(valid, coordinate_rows, 0.0), axis=1)
    sum_tt = np.sum(np.where(valid, coordinate_rows * coordinate_rows, 0.0), axis=1)
    sum_x = np.sum(safe, axis=1)
    sum_tx = np.sum(np.where(valid, coordinate_rows * windows, 0.0), axis=1)
    denominator = sum_tt - np.square(sum_t) / np.maximum(counts, 1)
    numerator = sum_tx - sum_t * sum_x / np.maximum(counts, 1)
    slope = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)

    if len(values) >= 3:
        first = np.gradient(values.astype(np.float64), step_seconds)
        second = np.gradient(first, step_seconds)
        acceleration = _rolling_mean(second, radius)
    else:
        acceleration = np.zeros(len(values), dtype=np.float64)

    before = windows[:, :radius]
    after = windows[:, radius + 1 :]

    def side_mean(side: np.ndarray) -> np.ndarray:
        side_valid = np.isfinite(side)
        side_count = np.sum(side_valid, axis=1)
        summed = np.sum(np.where(side_valid, side, 0.0), axis=1)
        return np.divide(
            summed,
            side_count,
            out=values.astype(np.float64, copy=True),
            where=side_count > 0,
        )

    post_minus_pre = side_mean(after) - side_mean(before)
    return {
        "mean": means,
        "maximum": maximum,
        "variance": variances,
        "q25": q25,
        "q75": q75,
        "slope": slope,
        "acceleration": acceleration,
        "post_minus_pre": post_minus_pre,
    }


def _centered_nearest_rank_quantile(
    values: np.ndarray,
    radius: int,
    quantile: float,
) -> np.ndarray:
    """Compute small centered-window quantiles without nanpercentile's row loop."""

    count = len(values)
    result = np.empty(count, dtype=np.float64)
    width = 2 * radius + 1
    if width <= count:
        windows = np.lib.stride_tricks.sliding_window_view(values, width)
        rank = int(round(quantile * (width - 1)))
        result[radius : count - radius] = np.partition(
            windows, rank, axis=1
        )[:, rank]
    for index in (*range(min(radius, count)), *range(max(radius, count - radius), count)):
        selected = values[max(0, index - radius) : min(count, index + radius + 1)]
        rank = int(round(quantile * (len(selected) - 1)))
        result[index] = np.partition(selected, rank)[rank]
    return result


def _high_run_fraction(
    values: np.ndarray,
    *,
    window_seconds: float,
    step_seconds: float,
) -> np.ndarray:
    high = values >= HIGH_ACTIVITY_THRESHOLD
    run = np.zeros(len(values), dtype=np.float64)
    current = 0
    for index, active in enumerate(high):
        current = current + 1 if active else 0
        run[index] = min(current * step_seconds, window_seconds) / window_seconds
    return run


def _seconds_since_crossing(
    values: np.ndarray,
    times: np.ndarray,
    *,
    threshold: float,
    cap_seconds: float,
) -> np.ndarray:
    result = np.full(len(values), cap_seconds, dtype=np.float64)
    previous = False
    most_recent: float | None = None
    for index, (timestamp, value) in enumerate(zip(times, values, strict=True)):
        active = bool(value >= threshold)
        if active and not previous:
            most_recent = float(timestamp)
        if most_recent is not None:
            result[index] = min(max(0.0, float(timestamp) - most_recent), cap_seconds)
        previous = active
    return result


def _recent_event_evidence(
    values: np.ndarray,
    times: np.ndarray,
    *,
    threshold: float,
    horizon_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    evidence = np.zeros(len(values), dtype=np.float64)
    elapsed = np.ones(len(values), dtype=np.float64)
    most_recent_time: float | None = None
    most_recent_value = 0.0
    previous = False
    for index, (timestamp, value) in enumerate(zip(times, values, strict=True)):
        active = bool(value >= threshold)
        if active and not previous:
            most_recent_time = float(timestamp)
            most_recent_value = float(value)
        elif active and value > most_recent_value:
            most_recent_time = float(timestamp)
            most_recent_value = float(value)
        if most_recent_time is not None:
            age = max(0.0, float(timestamp) - most_recent_time)
            if age <= horizon_seconds:
                evidence[index] = most_recent_value * math.exp(-age / horizon_seconds)
                elapsed[index] = age / horizon_seconds
        previous = active
    return evidence, elapsed


def _rank_nonconstant_columns(values: np.ndarray) -> np.ndarray:
    if not len(values):
        return values.astype(np.float32, copy=True)
    ranked = percentile_rank_values(values)
    for index in range(values.shape[1]):
        if float(np.ptp(values[:, index])) < 1e-12:
            ranked[:, index] = values[:, index]
    return ranked.astype(np.float32, copy=False)


def derive_transition_feature_block(
    item: PreparedRecording,
    feature_config: FeatureConfig,
    *,
    windows_seconds: Sequence[float] = DEFAULT_WINDOWS_SECONDS,
) -> DerivedFeatureBlock:
    """Derive the frozen transition bank from raw warm-cache signals.

    The block is aligned with the existing contextual matrix but is not itself
    sampled again at the feature config's context offsets.
    """

    if feature_config.sequence_normalization != "percentile-rank":
        raise FeatureExperimentError(
            "transition features require the frozen within-recording percentile-rank baseline"
        )
    windows = tuple(float(value) for value in windows_seconds)
    if not windows or any(not math.isfinite(value) or value <= 0 for value in windows):
        raise ValueError("transition windows must be finite and positive")
    if tuple(sorted(set(windows))) != windows:
        raise ValueError("transition windows must be strictly increasing")
    times = item.sequence.times.astype(np.float64, copy=False)
    step = _sample_step_seconds(times)
    base = _ranked_base_values(item)
    sources = _semantic_sources(base)

    multiscale_columns: list[np.ndarray] = []
    multiscale_names: list[str] = []
    means: dict[tuple[str, float], np.ndarray] = {}
    for source_name in SUMMARY_SOURCE_DEFINITIONS:
        values = sources[source_name]
        for window_seconds in windows:
            radius = max(1, int(round(window_seconds / (2.0 * step))))
            statistics = _centered_statistics(
                values,
                radius=radius,
                step_seconds=step,
            )
            span = max(1, int(round(window_seconds / step)))
            high = (values >= HIGH_ACTIVITY_THRESHOLD).astype(np.float64)
            statistics["high_persistence"] = _rolling_mean(high, span, trailing=True)
            statistics["high_run_fraction"] = _high_run_fraction(
                values,
                window_seconds=window_seconds,
                step_seconds=step,
            )
            means[(source_name, window_seconds)] = statistics["mean"]
            token = _duration_token(window_seconds)
            for statistic_name in (
                "mean",
                "maximum",
                "variance",
                "q25",
                "q75",
                "slope",
                "acceleration",
                "post_minus_pre",
                "high_persistence",
                "high_run_fraction",
            ):
                multiscale_columns.append(statistics[statistic_name])
                multiscale_names.append(
                    f"transition/multiscale/{source_name}/w{token}s/{statistic_name}"
                )

        short_long_pairs = ((windows[0], windows[-2]), (windows[1], windows[-1])) if len(windows) >= 4 else ((windows[0], windows[-1]),)
        for short, long in short_long_pairs:
            if short == long:
                continue
            multiscale_columns.append(means[(source_name, short)] - means[(source_name, long)])
            multiscale_names.append(
                "transition/multiscale/"
                f"{source_name}/short{_duration_token(short)}s_minus_long{_duration_token(long)}s"
            )

    cap = max(windows)
    for source_name in ("receiving_motion_onset", "motion_collapse", "audio_transient"):
        multiscale_columns.append(
            _seconds_since_crossing(
                sources[source_name],
                times,
                threshold=HIGH_ACTIVITY_THRESHOLD,
                cap_seconds=cap,
            )
            / cap
        )
        multiscale_names.append(
            f"transition/multiscale/{source_name}/time_since_threshold_crossing"
        )

    audio_available = _require(base, "audio_available")[0]
    serve_peak_proxy = np.maximum(
        _require(base, "audio_contact_like_transient")[0],
        _require(base, "audio_onset_strength")[0],
    ) * audio_available
    synchronized_stand_down = _require(base, "synchronized_stand_down")[0]
    dead_state = np.mean(
        np.vstack(
            (
                sources["motion_collapse"],
                sources["cadence_collapse"],
                synchronized_stand_down,
            )
        ),
        axis=0,
    )
    recent_serve, elapsed = _recent_event_evidence(
        serve_peak_proxy,
        times,
        threshold=SERVE_PROXY_THRESHOLD,
        horizon_seconds=RECENT_SERVE_HORIZON_SECONDS,
    )
    interactions = {
        "serve_peak_x_receiving_motion_onset": (
            serve_peak_proxy * sources["receiving_motion_onset"]
        ),
        "audio_transient_x_coherent_flow": (
            sources["audio_transient"] * sources["coherent_flow"]
        ),
        "terminal_transient_x_motion_collapse": (
            sources["audio_transient"] * sources["motion_collapse"]
        ),
        "cadence_collapse_x_formation_contraction": (
            sources["cadence_collapse"] * sources["formation_contraction"]
        ),
        "visibility_x_residual_player_motion": (
            sources["visibility_quality"] * sources["player_motion"]
        ),
        "camera_shift_x_frame_difference": (
            sources["camera_shift"] * sources["frame_difference"]
        ),
        "recent_serve_x_elapsed_x_dead_state": recent_serve * elapsed * dead_state,
    }
    expected_interactions = tuple(item["id"] for item in INTERACTION_DEFINITIONS)
    if tuple(interactions) != expected_interactions:
        raise FeatureExperimentError("interaction implementation and declaration differ")
    interaction_names = tuple(
        f"transition/interaction/{name}" for name in interactions
    )

    multiscale_raw = np.column_stack(multiscale_columns).astype(np.float64)
    interaction_raw = np.column_stack(tuple(interactions.values())).astype(np.float64)
    multiscale_values = _rank_nonconstant_columns(multiscale_raw)
    interaction_values = _rank_nonconstant_columns(interaction_raw)
    values = np.ascontiguousarray(
        np.concatenate((multiscale_values, interaction_values), axis=1),
        dtype=np.float32,
    )
    names = tuple(multiscale_names) + interaction_names
    if values.shape != (len(times), len(names)) or not np.isfinite(values).all():
        raise FeatureExperimentError("derived transition feature block is invalid")
    multiscale_indexes = tuple(range(len(multiscale_names)))
    groups: dict[str, tuple[int, ...]] = {"multiscale": multiscale_indexes}
    for offset, definition in enumerate(INTERACTION_DEFINITIONS, start=len(multiscale_names)):
        groups[f"interaction:{definition['id']}"] = (offset,)
    definitions = transition_feature_definitions(
        windows_seconds=windows,
        multiscale_names=tuple(multiscale_names),
        interaction_names=interaction_names,
    )
    return DerivedFeatureBlock(
        values=values,
        names=names,
        groups=groups,
        definitions=definitions,
    )


def append_feature_blocks(
    item: PreparedRecording,
    *blocks: DerivedFeatureBlock,
) -> PreparedRecording:
    """Append aligned derived blocks once, preserving the raw warm feature sequence."""

    values = [item.contextual_values]
    names = list(item.contextual_names)
    seen = set(names)
    for block in blocks:
        if block.values.ndim != 2 or block.values.shape[0] != len(item.sequence.times):
            raise FeatureExperimentError("derived block is not timestamp-aligned")
        if block.values.shape[1] != len(block.names):
            raise FeatureExperimentError("derived block names do not match its columns")
        duplicates = seen.intersection(block.names)
        if duplicates or len(set(block.names)) != len(block.names):
            raise FeatureExperimentError(
                f"derived feature names are not unique: {sorted(duplicates)}"
            )
        if not np.isfinite(block.values).all():
            raise FeatureExperimentError("derived feature block contains non-finite values")
        values.append(block.values.astype(np.float32, copy=False))
        names.extend(block.names)
        seen.update(block.names)
    return replace(
        item,
        contextual_values=np.ascontiguousarray(np.concatenate(values, axis=1)),
        contextual_names=tuple(names),
    )


def _validate_warm_substrate(
    prepared: Sequence[PreparedRecording],
    feature_config: FeatureConfig,
    *,
    require_warm_90: bool,
) -> None:
    if not prepared:
        raise FeatureExperimentError("transition experiment data is empty")
    raw_signature = prepared[0].sequence.names
    contextual_signature = prepared[0].contextual_names
    for item in prepared:
        if item.sequence.names != raw_signature:
            raise FeatureExperimentError("raw feature signatures differ across recordings")
        if item.contextual_names != contextual_signature:
            raise FeatureExperimentError("contextual signatures differ across recordings")
        if item.contextual_values.shape != (len(item.sequence.times), len(contextual_signature)):
            raise FeatureExperimentError("contextual matrix shape is invalid")
    if require_warm_90:
        if len(raw_signature) != EXPECTED_BASE_SIGNAL_COUNT:
            raise FeatureExperimentError(
                f"expected the warm 90-signal sequence, found {len(raw_signature)} signals"
            )
        if len(contextual_signature) != EXPECTED_BASELINE_CONTEXT_COUNT:
            raise FeatureExperimentError(
                "expected the frozen 450-column contextual baseline, found "
                f"{len(contextual_signature)} columns"
            )
        if tuple(feature_config.context_offsets_seconds) != (-2.0, -1.0, 0.0, 1.0, 2.0):
            raise FeatureExperimentError(
                "the 450-column control requires context offsets -2,-1,0,+1,+2 seconds"
            )


def prepare_transition_candidates(
    prepared: Sequence[PreparedRecording],
    feature_config: FeatureConfig,
    *,
    require_warm_90: bool = True,
) -> TransitionPrepared:
    _validate_warm_substrate(
        prepared,
        feature_config,
        require_warm_90=require_warm_90,
    )
    augmented: list[PreparedRecording] = []
    first_block: DerivedFeatureBlock | None = None
    for item in prepared:
        block = derive_transition_feature_block(item, feature_config)
        if first_block is None:
            first_block = block
        elif block.names != first_block.names or block.groups != first_block.groups:
            raise FeatureExperimentError("derived feature signatures differ across recordings")
        augmented.append(append_feature_blocks(item, block))
    assert first_block is not None
    baseline_count = len(prepared[0].contextual_names)
    baseline_indexes = tuple(range(baseline_count))
    multiscale_local = first_block.groups["multiscale"]
    multiscale_indexes = tuple(baseline_count + index for index in multiscale_local)
    interaction_ids = tuple(item["id"] for item in INTERACTION_DEFINITIONS)
    interaction_indexes = tuple(
        baseline_count + first_block.groups[f"interaction:{name}"][0]
        for name in interaction_ids
    )
    combined = baseline_indexes + multiscale_indexes + interaction_indexes
    candidates: list[FeatureSet] = [
        FeatureSet(
            "baseline_450",
            baseline_indexes,
            ("baseline_context",),
            "frozen 450-column baseline",
        ),
        FeatureSet(
            "baseline_plus_multiscale",
            baseline_indexes + multiscale_indexes,
            ("baseline_context", "multiscale_transition"),
            "multiscale transition bank",
        ),
        FeatureSet(
            "baseline_plus_interactions",
            baseline_indexes + interaction_indexes,
            ("baseline_context", "cross_modal_interactions"),
            "predeclared interaction bank",
        ),
        FeatureSet(
            "combined",
            combined,
            ("baseline_context", "multiscale_transition", "cross_modal_interactions"),
            "multiscale transition and interaction banks",
        ),
    ]
    for interaction_id, interaction_index in zip(
        interaction_ids, interaction_indexes, strict=True
    ):
        candidates.append(
            FeatureSet(
                f"combined_minus_{interaction_id}",
                tuple(index for index in combined if index != interaction_index),
                (
                    "baseline_context",
                    "multiscale_transition",
                    "cross_modal_interactions_ablation",
                ),
                interaction_id,
            )
        )
    return TransitionPrepared(
        prepared=tuple(augmented),
        candidates=tuple(candidates),
        baseline_names=prepared[0].contextual_names,
        derived_names=first_block.names,
        multiscale_names=tuple(first_block.names[index] for index in multiscale_local),
        interaction_names=tuple(
            first_block.names[first_block.groups[f"interaction:{name}"][0]]
            for name in interaction_ids
        ),
        definitions=first_block.definitions,
    )


def transition_feature_definitions(
    *,
    windows_seconds: Sequence[float] = DEFAULT_WINDOWS_SECONDS,
    multiscale_names: Sequence[str] = (),
    interaction_names: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "baseNormalization": (
            "The same within-recording percentile ranking as the frozen baseline is "
            "applied to raw base signals; absolute availability/quality signals retain "
            "their extractor scale."
        ),
        "derivedNormalization": (
            "Each nonconstant derived column is percentile-ranked within recording; "
            "constant structural-zero columns remain zero."
        ),
        "appendPolicy": (
            "Every derived value is appended exactly once at its aligned timestamp. "
            "Derived columns are not sampled again at -2,-1,0,+1,+2 seconds."
        ),
        "windowsSeconds": [float(value) for value in windows_seconds],
        "windowSemantics": (
            "Centered windows clipped at recording boundaries; radius is the nearest "
            "whole sample to half the declared duration. Means/maxima/variance/nearest-rank "
            "quartiles, "
            "linear slope, mean second derivative, post-minus-pre, trailing high-state "
            "persistence, and capped current high-run fraction are emitted."
        ),
        "highActivityThresholdAfterRanking": HIGH_ACTIVITY_THRESHOLD,
        "summarySources": SUMMARY_SOURCE_DEFINITIONS,
        "interactions": [dict(item) for item in INTERACTION_DEFINITIONS],
        "serveProxy": {
            "formula": (
                "max(audio_contact_like_transient, audio_onset_strength) * audio_available"
            ),
            "threshold": SERVE_PROXY_THRESHOLD,
            "recentHorizonSeconds": RECENT_SERVE_HORIZON_SECONDS,
            "warning": (
                "This is a contact proxy, not a source-group-cross-fitted serve model; "
                "non-serve contacts may trigger it."
            ),
        },
        "multiscaleFeatureCount": len(multiscale_names),
        "multiscaleFeatureNames": list(multiscale_names),
        "interactionFeatureCount": len(interaction_names),
        "interactionFeatureNames": list(interaction_names),
    }


def _multi_iou_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "eventF1AtIou03": metrics.get("eventF1AtIou03"),
        "eventF1AtIou05": metrics.get("eventF1"),
        "eventF1AtIou07": metrics.get("eventF1AtIou07"),
        "matchedRalliesAtIou03": metrics.get("matchedRalliesAtIou03"),
        "matchedRalliesAtIou05": metrics.get("matchedRallies"),
        "matchedRalliesAtIou07": metrics.get("matchedRalliesAtIou07"),
    }


def _paired_candidate_comparison(
    baseline: Sequence[Any],
    candidate: Sequence[Any],
    *,
    subject: str,
    margin: float,
    sign_consistency: float,
) -> dict[str, Any]:
    baseline_by_group = {item.fold.held_out_group: item for item in baseline}
    candidate_by_group = {item.fold.held_out_group: item for item in candidate}
    if set(baseline_by_group) != set(candidate_by_group):
        raise FeatureExperimentError("paired candidates do not share outer folds")
    rows: list[dict[str, Any]] = []
    for group in sorted(baseline_by_group):
        control = baseline_by_group[group].aggregate
        experiment = candidate_by_group[group].aggregate
        rows.append(
            {
                "sourceGroup": group,
                "baselineObjective": float(control["objective"]),
                "candidateObjective": float(experiment["objective"]),
                "deltaObjectiveCandidateMinusBaseline": float(
                    experiment["objective"] - control["objective"]
                ),
                "deltaEventF1AtIou03": float(
                    experiment["eventF1AtIou03"] - control["eventF1AtIou03"]
                ),
                "deltaEventF1AtIou05": float(
                    experiment["eventF1"] - control["eventF1"]
                ),
                "deltaEventF1AtIou07": float(
                    experiment["eventF1AtIou07"] - control["eventF1AtIou07"]
                ),
                "deltaTimeIoU": float(experiment["timeIoU"] - control["timeIoU"]),
                "deltaLiveTimeRecall": float(
                    experiment["liveTimeRecall"] - control["liveTimeRecall"]
                ),
            }
        )
    deltas = [float(item["deltaObjectiveCandidateMinusBaseline"]) for item in rows]
    return {
        "interpretationSubject": subject,
        "deltaDirection": "positive means the candidate improves over its paired baseline",
        "classification": classify_paired_deltas(
            deltas,
            margin=margin,
            sign_consistency=sign_consistency,
        ),
        "pairedSourceGroups": rows,
    }


def _transition_code_provenance() -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    tracked = (
        package / "transition_feature_experiment.py",
        package / "feature_experiments.py",
        package / "features.py",
        package / "model.py",
        package / "pipeline.py",
        package / "metrics.py",
        package / "decoder.py",
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


def _signature_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(names).encode("utf-8")).hexdigest()


def _candidate_dict(feature_set: FeatureSet, signature: Sequence[str]) -> dict[str, Any]:
    return {
        "name": feature_set.name,
        "featureCount": len(feature_set.indexes),
        "featureNames": [signature[index] for index in feature_set.indexes],
        "featureSignatureSha256": _signature_sha256(
            [signature[index] for index in feature_set.indexes]
        ),
        "families": list(feature_set.families),
        "interpretationSubject": feature_set.interpretation_subject,
    }


def run_development_transition_experiments(
    manifest: DatasetManifest,
    prepared: Sequence[PreparedRecording],
    *,
    feature_config: FeatureConfig,
    training_config: TrainingConfig,
    decoder_config: DecoderConfig,
    inner_fold_limit: int | None = None,
    objective_margin: float = DEFAULT_OBJECTIVE_MARGIN,
    sign_consistency: float = DEFAULT_SIGN_CONSISTENCY,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    development_rows = [
        item for item in manifest.recordings if item.split in DEVELOPMENT_SPLITS
    ]
    expected_ids = {item.id for item in development_rows}
    if {item.recording.id for item in prepared} != expected_ids:
        raise FeatureExperimentError(
            "prepared recordings must contain exactly train+validation development rows"
        )
    if any(item.recording.split not in DEVELOPMENT_SPLITS for item in prepared):
        raise FeatureExperimentError("protected test/challenge data entered development")
    feature_config.validate()
    training_config.validate()
    decoder_config.validate()
    transition = prepare_transition_candidates(prepared, feature_config)
    signature = transition.prepared[0].contextual_names
    folds = build_fold_plan(manifest.recordings)
    artifacts_by_candidate: dict[str, list[Any]] = {}
    candidate_reports: dict[str, Any] = {}
    total = len(transition.candidates) * len(folds)
    completed = 0
    for candidate in transition.candidates:
        artifacts: list[Any] = []
        for fold in folds:
            completed += 1
            if progress is not None:
                progress(
                    f"Transition experiment {completed}/{total}: {candidate.name}, "
                    f"hold out {fold.held_out_group}"
                )
            artifacts.append(
                _run_candidate_fold(
                    transition.prepared,
                    fold,
                    candidate,
                    feature_config=feature_config,
                    training_config=training_config,
                    decoder_config=decoder_config,
                    inner_fold_limit=inner_fold_limit,
                )
            )
        artifacts_by_candidate[candidate.name] = artifacts
        aggregated = _aggregate_artifacts(artifacts)
        candidate_reports[candidate.name] = {
            "featureSet": _candidate_dict(candidate, signature),
            "oof": aggregated,
            "multiIouMetrics": _multi_iou_metrics(aggregated["aggregate"]),
            "outcomeSlices": aggregated["aggregate"]["outcomeSlices"],
            "outerFolds": [_fold_report(item) for item in artifacts],
        }

    baseline_artifacts = artifacts_by_candidate["baseline_450"]
    comparisons = {
        candidate.name: _paired_candidate_comparison(
            baseline_artifacts,
            artifacts_by_candidate[candidate.name],
            subject=candidate.interpretation_subject,
            margin=objective_margin,
            sign_consistency=sign_consistency,
        )
        for candidate in transition.candidates
        if candidate.name != "baseline_450"
    }
    interaction_assessments: dict[str, Any] = {}
    combined_artifacts = artifacts_by_candidate["combined"]
    for definition in INTERACTION_DEFINITIONS:
        interaction_id = definition["id"]
        without_name = f"combined_minus_{interaction_id}"
        interaction_assessments[interaction_id] = _paired_candidate_comparison(
            artifacts_by_candidate[without_name],
            combined_artifacts,
            subject=interaction_id,
            margin=objective_margin,
            sign_consistency=sign_consistency,
        )

    primary_names = (
        "baseline_450",
        "baseline_plus_multiscale",
        "baseline_plus_interactions",
        "combined",
    )
    eligible = ["baseline_450"] + [
        name
        for name in primary_names[1:]
        if comparisons[name]["classification"]["classification"] == "helpful"
    ]
    selected_name = max(
        eligible,
        key=lambda name: (
            float(candidate_reports[name]["oof"]["aggregate"]["objective"]),
            -primary_names.index(name),
        ),
    )
    selected_artifacts = artifacts_by_candidate[selected_name]
    pooled_prepared = [item for artifact in selected_artifacts for item in artifact.held_prepared]
    pooled_probabilities = [
        values for artifact in selected_artifacts for values in artifact.probabilities
    ]
    frozen_decoder, decoder_selection = _select_decoder(
        pooled_prepared,
        pooled_probabilities,
        decoder_config,
    )
    epoch_cap = max(
        1,
        int(round(float(np.median([item.epoch_cap for item in selected_artifacts])))),
    )
    selected_feature_set = next(
        item for item in transition.candidates if item.name == selected_name
    )
    selected_names = tuple(signature[index] for index in selected_feature_set.indexes)
    final_seed = _seed(training_config.seed, "transition-final-refit", selected_name)
    provenance = _transition_code_provenance()
    return {
        "schemaVersion": TRANSITION_EXPERIMENT_SCHEMA_VERSION,
        "kind": "volleycut-transition-feature-experiment-development",
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
        "development": {
            "splits": sorted(DEVELOPMENT_SPLITS),
            "recordingIds": sorted(expected_ids),
            "sourceGroups": sorted({item.source_group for item in development_rows}),
        },
        "protected": _protected_summary(manifest.recordings),
        "featureVersion": FEATURE_VERSION,
        "featureConfig": feature_config.to_dict(),
        "trainingConfig": training_config.to_dict(),
        "baseDecoderConfig": decoder_config.to_dict(),
        "substrate": {
            "baseSignalCount": len(transition.prepared[0].sequence.names),
            "baselineContextCount": len(transition.baseline_names),
            "baselineFeatureNames": list(transition.baseline_names),
            "baselineSignatureSha256": _signature_sha256(transition.baseline_names),
            "derivedFeatureCount": len(transition.derived_names),
            "derivedSignatureSha256": _signature_sha256(transition.derived_names),
        },
        "featureDefinitions": transition.definitions,
        "selectionProtocol": {
            "outer": "leave one development sourceGroup out",
            "inner": "leave one remaining sourceGroup out",
            "innerFoldLimit": inner_fold_limit,
            "epochRefit": (
                "median inner best epoch caps each outer refit; no outer labels select "
                "the fitted checkpoint"
            ),
            "decoder": "selected from pooled inner out-of-fold probabilities",
            "objective": {
                "eventF1": 0.55,
                "timeIoU": 0.30,
                "liveTimeRecall": 0.15,
            },
            "objectiveMargin": objective_margin,
            "signConsistency": sign_consistency,
            "folds": [item.to_dict() for item in folds],
        },
        "metricProtocol": {
            "headlineIoU": 0.5,
            "reportedEventIoUThresholds": [0.3, 0.5, 0.7],
            "outcomeSlices": [
                "all",
                "shortAtMost3Seconds",
                "ace",
                "serviceFault",
                "ordinaryLong",
            ],
        },
        "candidates": candidate_reports,
        "pairedComparisonsAgainstBaseline": comparisons,
        "individualInteractionAblations": interaction_assessments,
        "selectedCandidateForRetrospectiveTest": selected_name,
        "candidateSelection": {
            "rule": (
                "Baseline is always eligible. An enhanced primary candidate is eligible "
                "only when paired source-group deltas classify it helpful; choose the "
                "eligible candidate with maximum development OOF objective, breaking ties "
                "toward the simpler declared order. Diagnostic ablations are ineligible."
            ),
            "eligibleCandidates": eligible,
            "selected": selected_name,
            "testMetricsConsulted": False,
        },
        "finalizationPlan": {
            "candidate": selected_name,
            "featureCount": len(selected_names),
            "featureNames": list(selected_names),
            "featureSignatureSha256": _signature_sha256(selected_names),
            "epochCap": epoch_cap,
            "seed": final_seed,
            "decoder": frozen_decoder.to_dict(),
            "decoderSelectionOnDevelopmentOof": decoder_selection,
            "fitRows": "all train+validation development recordings",
            "checkpointSelection": "minimum training loss up to frozen epoch cap",
        },
        "causalAndOfflineNotes": {
            "executionMode": "offline video cutter",
            "maximumDeclaredLookaheadSeconds": max(DEFAULT_WINDOWS_SECONDS) / 2.0,
            "centeredWindowLookahead": True,
            "causalFeatures": [
                "trailing high-state persistence",
                "current high-run fraction",
                "time since threshold crossing",
                "recent-serve proxy elapsed state",
            ],
            "offlineFeatures": [
                "centered means/maxima/variance/quantiles",
                "centered slopes and acceleration",
                "post-minus-pre transition differences",
                "the frozen baseline's positive temporal offsets",
            ],
            "warning": (
                "These metrics describe an offline cutter. Do not claim streaming latency "
                "or causal inference from this feature bank."
            ),
        },
        "guardrails": [
            "Only train and validation recordings are prepared during development.",
            "Source groups, not recordings, define every inner and outer split.",
            "No rally boundary, duration, outcome tag, or future annotation is a feature.",
            "Interactions are exactly the seven predeclared formulas; no product search occurs.",
            "Derived columns append once and are never multiplied across context offsets.",
            "The retrospective test command accepts only this frozen report and never selects on test.",
        ],
        "provenance": provenance,
        "processing": {
            "wallClockSeconds": round(time.perf_counter() - started, 3),
            "featureCacheNote": (
                "Raw 90-signal warm caches are reused; all transition columns are "
                "deterministically derived in memory."
            ),
        },
        "limitations": [
            "Only four independent development source groups are available.",
            "Paired classifications are practical evidence categories, not significance tests.",
            "The serve and terminal-transient inputs are generic contact proxies, not labeled events.",
            "Centered transition summaries use future frames and are suitable only for offline cutting.",
            "Within-recording ranking sacrifices absolute cross-recording magnitude information.",
        ],
    }


def _load_frozen_development_report(path: str | Path) -> tuple[Path, dict[str, Any]]:
    report_path = Path(path).expanduser().resolve()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureExperimentError(
            f"cannot read development report {report_path}: {error}"
        ) from error
    if (
        report.get("schemaVersion") != TRANSITION_EXPERIMENT_SCHEMA_VERSION
        or report.get("kind") != "volleycut-transition-feature-experiment-development"
        or report.get("freezeStatus") != "frozen-development-selection"
        or report.get("testLabelsUsed") is not False
        or not isinstance(report.get("selectedCandidateForRetrospectiveTest"), str)
    ):
        raise FeatureExperimentError(
            "development report is not a frozen unopened-test transition experiment"
        )
    return report_path, report


def run_retrospective_transition_test(
    manifest_path: str | Path,
    development_report_path: str | Path,
    cache_dir: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    report_path, development = _load_frozen_development_report(
        development_report_path
    )
    manifest = load_manifest(manifest_path)
    if development.get("manifestFileSha256") != sha256_file(manifest.path):
        raise FeatureExperimentError("development report does not match the supplied manifest")
    if development.get("manifestSnapshotSha256") != _manifest_digest(manifest):
        raise FeatureExperimentError(
            "recording snapshots changed after the development report was frozen"
        )
    if development.get("recordingContentSha256") != {
        item.id: item.content_sha256 for item in manifest.recordings
    }:
        raise FeatureExperimentError(
            "recording content identities changed after development"
        )
    if development.get("featureVersion") != FEATURE_VERSION:
        raise FeatureExperimentError("feature implementation version changed after development")
    current_provenance = _transition_code_provenance()
    if development.get("provenance", {}).get("filesSha256") != current_provenance.get(
        "filesSha256"
    ):
        raise FeatureExperimentError("transition experiment code changed after development")

    feature_config = FeatureConfig.from_dict(development["featureConfig"])
    training_config = TrainingConfig(**development["trainingConfig"])
    finalization = development["finalizationPlan"]
    selected_name = development["selectedCandidateForRetrospectiveTest"]
    if finalization.get("candidate") != selected_name:
        raise FeatureExperimentError("frozen candidate and finalization plan disagree")
    epoch_cap = int(finalization["epochCap"])
    if epoch_cap < 1:
        raise FeatureExperimentError("frozen epoch cap is invalid")
    decoder = DecoderConfig.from_dict(finalization["decoder"])
    seed = int(finalization["seed"])

    development_rows = tuple(
        item for item in manifest.recordings if item.split in DEVELOPMENT_SPLITS
    )
    test_rows = manifest.for_split("test")
    if not development_rows or not test_rows:
        raise FeatureExperimentError("retrospective test requires development and test rows")
    if progress is not None:
        progress("Preparing frozen train+validation development features")
    development_prepared = (
        _prepare_many(development_rows, feature_config, cache_dir)
        if progress is None
        else _prepare_many(
            development_rows, feature_config, cache_dir, progress=progress
        )
    )
    development_transition = prepare_transition_candidates(
        development_prepared, feature_config
    )
    selected = next(
        (item for item in development_transition.candidates if item.name == selected_name),
        None,
    )
    if selected is None:
        raise FeatureExperimentError("frozen selected candidate is not implemented")
    selected_development = _subset_prepared(
        development_transition.prepared, selected.indexes
    )
    selected_names = selected_development[0].contextual_names
    if (
        list(selected_names) != finalization.get("featureNames")
        or _signature_sha256(selected_names)
        != finalization.get("featureSignatureSha256")
    ):
        raise FeatureExperimentError("frozen selected feature signature changed")
    fit_config = replace(
        training_config,
        epochs=epoch_cap,
        patience=max(training_config.patience, epoch_cap + 1),
        seed=seed,
    )
    if progress is not None:
        progress("Fitting the frozen candidate on development rows only")
    model = _fit(
        selected_development,
        (),
        feature_config=feature_config,
        decoder=decoder,
        config=fit_config,
        seed=seed,
    )
    model.decoder = decoder

    if progress is not None:
        progress("Opening the retrospective test split once with selection frozen")
    test_prepared = (
        _prepare_many(test_rows, feature_config, cache_dir)
        if progress is None
        else _prepare_many(test_rows, feature_config, cache_dir, progress=progress)
    )
    test_transition = prepare_transition_candidates(test_prepared, feature_config)
    test_selected_spec = next(
        item for item in test_transition.candidates if item.name == selected_name
    )
    selected_test = _subset_prepared(test_transition.prepared, test_selected_spec.indexes)
    if any(item.contextual_names != selected_names for item in selected_test):
        raise FeatureExperimentError("test feature signature differs from frozen development")
    probabilities = [model.predict(item.contextual_values) for item in selected_test]
    per_recording, aggregate = _evaluate_prepared_probabilities(
        selected_test, probabilities, decoder
    )
    aggregate["objective"] = objective(aggregate)
    return {
        "schemaVersion": TRANSITION_EXPERIMENT_SCHEMA_VERSION,
        "kind": "volleycut-transition-feature-experiment-retrospective-test",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "assessmentRole": "explicit-single-source-retrospective-regression-test-access",
        "testLabelsOpened": True,
        "selectionLockedBeforeTest": True,
        "developmentReport": str(report_path),
        "developmentReportSha256": sha256_file(report_path),
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "selectedCandidate": selected_name,
        "featureCount": len(selected_names),
        "featureSignatureSha256": _signature_sha256(selected_names),
        "training": {
            "recordingIds": [item.recording.id for item in selected_development],
            "sourceGroups": sorted(
                {item.recording.source_group for item in selected_development}
            ),
            "splits": sorted(DEVELOPMENT_SPLITS),
            "testRowsUsedForFitting": False,
            "epochCap": epoch_cap,
            "seed": seed,
            "modelFingerprint": _model_fingerprint(model),
        },
        "decoder": decoder.to_dict(),
        "test": {
            "aggregate": aggregate,
            "multiIouMetrics": _multi_iou_metrics(aggregate),
            "outcomeSlices": aggregate["outcomeSlices"],
            "recordings": per_recording,
        },
        "provenance": current_provenance,
        "guardrails": [
            "Candidate, feature signature, epoch cap, seed, and decoder were frozen in development.",
            "Only train+validation rows fit the final model.",
            "Test metrics are reported once and cannot revise the selected candidate.",
        ],
        "warning": (
            "The protected split is a retrospective regression test, not evidence of "
            "external generalization. Do not revise this feature bank from its result."
        ),
    }
