"""Pure, predeclared court-relative features for the rectangle-ROI experiment.

The cached spatial grids are 3x3 and row-major: ``index = row * 3 + column``.
For a stationary camera centered behind an end line, row 0 is the far third of
the ROI, row 2 is the near third, column 0 is image-left, and column 2 is
image-right.  The center cell and the eight-cell outer ring are only proxies
for court interior and margins; a rectangle ROI cannot identify true court or
off-court polygons.

All variants require that fixed camera profile.  ``orientation_invariant``
removes the sign of near/far and left/right reflections.  ``fixed_endline``
retains signed, camera-relative directions.  ``combined`` is their union,
with shared core columns emitted once.  No variant reads rally labels, outcome
tags, environment, player counts, or side-switch markers.  Side switches are
reserved for reporting or a separately declared production-context ablation.

Reaction, contraction, and migration columns use future samples and are thus
offline/non-causal.  They are suitable for the existing offline cutter, but a
streaming product would need a causal replacement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .pipeline import PreparedRecording


EXPERIMENT_ID = "court-relative-directional-roi-v1"
GRID_SIZE = 3
REACTION_WINDOW_SECONDS = 0.5
REACTION_LAG_HORIZON_SECONDS = 1.0
MOTION_ACTIVITY_GATE_SCALE = 0.015
COURT_RELATIVE_VARIANTS = (
    "orientation_invariant",
    "fixed_endline",
    "combined",
)

_GRID_SOURCES = (
    ("player_motion", "player_motion_grid_"),
    ("frame_difference", "diff_grid_"),
    ("optical_flow", "flow_grid_"),
)

_SCALAR_SOURCE_NAMES = (
    "flow_median_x",
    "flow_median_y",
    "player_motion_centroid_x",
    "player_motion_centroid_y",
    "player_motion_spread_x",
    "player_motion_spread_y",
)

SOURCE_FEATURE_NAMES = (
    *(f"{prefix}{index}" for _, prefix in _GRID_SOURCES for index in range(9)),
    *_SCALAR_SOURCE_NAMES,
)

_CORE_FEATURE_NAMES = (
    *(
        f"derived/court_relative/{source}_center_minus_margin_contrast"
        for source, _ in _GRID_SOURCES
    ),
    "derived/court_relative/flow_along_axis_abs",
    "derived/court_relative/flow_across_axis_abs",
    "derived/court_relative/flow_along_axis_fraction",
    "derived/court_relative/player_motion_spread_total",
    "derived/court_relative/player_motion_spread_along_minus_across_contrast",
    "derived/court_relative/player_motion_contraction",
    "derived/court_relative/player_motion_migration_magnitude",
    "derived/court_relative/player_motion_migration_along_fraction",
)

_INVARIANT_FEATURE_NAMES = (
    *(
        name
        for source, _ in _GRID_SOURCES
        for name in (
            f"derived/court_relative/invariant/{source}_near_far_asymmetry_abs",
            f"derived/court_relative/invariant/{source}_left_right_asymmetry_abs",
        )
    ),
    "derived/court_relative/invariant/reaction_onset_stronger_side",
    "derived/court_relative/invariant/reaction_onset_weaker_side",
    "derived/court_relative/invariant/reaction_onset_asymmetry_abs",
    "derived/court_relative/invariant/reaction_lag_abs_seconds",
    "derived/court_relative/invariant/player_motion_centroid_across_offset_abs",
    "derived/court_relative/invariant/player_motion_centroid_along_offset_abs",
)

_FIXED_ENDLINE_FEATURE_NAMES = (
    *(
        name
        for source, _ in _GRID_SOURCES
        for name in (
            f"derived/court_relative/fixed/{source}_near_minus_far_contrast",
            f"derived/court_relative/fixed/{source}_right_minus_left_contrast",
        )
    ),
    "derived/court_relative/fixed/flow_toward_near_signed",
    "derived/court_relative/fixed/flow_toward_right_signed",
    "derived/court_relative/fixed/near_reaction_onset",
    "derived/court_relative/fixed/far_reaction_onset",
    "derived/court_relative/fixed/near_minus_far_reaction_onset",
    "derived/court_relative/fixed/far_minus_near_reaction_lag_seconds",
    "derived/court_relative/fixed/player_motion_centroid_toward_right_signed",
    "derived/court_relative/fixed/player_motion_centroid_toward_near_signed",
    "derived/court_relative/fixed/player_motion_migration_toward_right_signed",
    "derived/court_relative/fixed/player_motion_migration_toward_near_signed",
)

_VARIANT_FEATURE_NAMES = {
    "orientation_invariant": (*_CORE_FEATURE_NAMES, *_INVARIANT_FEATURE_NAMES),
    "fixed_endline": (*_CORE_FEATURE_NAMES, *_FIXED_ENDLINE_FEATURE_NAMES),
    "combined": (
        *_CORE_FEATURE_NAMES,
        *_INVARIANT_FEATURE_NAMES,
        *_FIXED_ENDLINE_FEATURE_NAMES,
    ),
}


@dataclass(frozen=True)
class CourtRelativeFeatureSpec:
    """Frozen feature signature and assumptions hashed into experiment reports."""

    variant: str
    feature_names: tuple[str, ...]

    @property
    def groups(self) -> dict[str, tuple[int, ...]]:
        """Local indexes suitable for ``DerivedFeatureBlock.groups``."""
        result: dict[str, tuple[int, ...]] = {}
        selectors = (
            ("court_relative:core", "/court_relative/invariant/", "/court_relative/fixed/"),
            ("court_relative:orientation_invariant", "/court_relative/invariant/", None),
            ("court_relative:fixed_endline", "/court_relative/fixed/", None),
        )
        for group, marker, second_marker in selectors:
            if second_marker is not None:
                indexes = tuple(
                    index
                    for index, name in enumerate(self.feature_names)
                    if marker not in name and second_marker not in name
                )
            else:
                indexes = tuple(
                    index
                    for index, name in enumerate(self.feature_names)
                    if marker in name
                )
            if indexes:
                result[group] = indexes
        return result

    def to_dict(self) -> dict[str, Any]:
        orientation_policy = {
            "orientation_invariant": (
                "absolute/reflection-invariant near-far and left-right summaries"
            ),
            "fixed_endline": (
                "signed image orientation: bottom=near, top=far, right=image-right"
            ),
            "combined": "union of invariant and signed fixed-endline summaries",
        }[self.variant]
        return {
            "experimentId": EXPERIMENT_ID,
            "family": "court_relative",
            "variant": self.variant,
            "featureNames": list(self.feature_names),
            "featureCount": len(self.feature_names),
            "featureGroups": {
                name: list(indexes) for name, indexes in self.groups.items()
            },
            "sourceFeatureNames": list(SOURCE_FEATURE_NAMES),
            "grid": {
                "size": GRID_SIZE,
                "order": "row-major",
                "indexFormula": "row * 3 + column",
                "rows": ["far-third", "middle-third", "near-third"],
                "columns": ["image-left", "center", "image-right"],
                "centerMarginProxy": (
                    "center cell versus eight-cell outer ring; not a court polygon"
                ),
            },
            "captureRequirements": {
                "roi": "required normalized rectangle used by cached features",
                "position": "centered-behind-endline",
                "stationary": True,
                "fullCourtVisible": True,
            },
            "orientationPolicy": orientation_policy,
            "geometryLimitations": (
                "rectangle thirds and image-axis flow are proxies only; no court "
                "polygon, homography, team identity, or player tracks are available"
            ),
            "temporal": {
                "reactionWindowSeconds": REACTION_WINDOW_SECONDS,
                "reactionLagHorizonSeconds": REACTION_LAG_HORIZON_SECONDS,
                "futureLooking": True,
                "causal": False,
            },
            "motionActivityGateScale": MOTION_ACTIVITY_GATE_SCALE,
            "sideSwitchPolicy": (
                "not used as a feature; report separately or evaluate only as an "
                "explicit production-context ablation"
            ),
            "excludedInputs": [
                "rally labels",
                "outcome tags",
                "environment",
                "playersPerTeam",
                "sideSwitches",
            ],
        }

    @property
    def sha256(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {**self.to_dict(), "specSha256": self.sha256}


def court_relative_feature_spec(variant: str) -> CourtRelativeFeatureSpec:
    """Return the immutable, predeclared spec for one candidate variant."""
    try:
        names = _VARIANT_FEATURE_NAMES[variant]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"unknown court-relative variant {variant!r}; "
            f"expected one of {COURT_RELATIVE_VARIANTS}"
        ) from error
    if len(names) != len(set(names)):
        raise RuntimeError(f"court-relative variant {variant!r} has duplicate names")
    return CourtRelativeFeatureSpec(variant=variant, feature_names=names)


COURT_RELATIVE_SPEC_HASHES = {
    variant: court_relative_feature_spec(variant).sha256
    for variant in COURT_RELATIVE_VARIANTS
}


def _validate_prepared(prepared: PreparedRecording) -> tuple[np.ndarray, np.ndarray]:
    sequence = prepared.sequence
    times = np.asarray(sequence.times, dtype=np.float64)
    values = np.asarray(sequence.values, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("court-relative feature times must be finite and one-dimensional")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("court-relative feature times must be strictly increasing")
    if values.ndim != 2 or values.shape != (len(times), len(sequence.names)):
        raise ValueError("cached feature values and names must be row-aligned")
    if len(sequence.names) != len(set(sequence.names)):
        raise ValueError("cached base feature names must be unique")
    if not np.isfinite(values).all():
        raise ValueError("cached court-relative source features must be finite")
    if prepared.contextual_values.ndim != 2 or len(prepared.contextual_values) != len(times):
        raise ValueError("contextual values must align with cached feature rows")
    if prepared.recording.roi is None:
        raise ValueError("court-relative features require a cached rectangle ROI")
    capture = prepared.recording.capture
    required_capture = {
        "position": "centered-behind-endline",
        "stationary": True,
        "fullCourtVisible": True,
    }
    mismatches = [
        key for key, expected in required_capture.items() if capture.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "court-relative features require the fixed end-line capture profile; "
            f"mismatched fields={mismatches}"
        )

    available = set(sequence.names)
    missing = sorted(set(SOURCE_FEATURE_NAMES) - available)
    grid_extras = sorted(
        name
        for name in available
        if any(name.startswith(prefix) for _, prefix in _GRID_SOURCES)
        and name not in SOURCE_FEATURE_NAMES
    )
    if missing or grid_extras:
        raise ValueError(
            "court-relative features require exact cached 3x3 grids; "
            f"missing={missing}, unexpectedGridColumns={grid_extras}"
        )
    return times, values


def _window_means(
    times: np.ndarray, values: np.ndarray, window_seconds: float
) -> tuple[np.ndarray, np.ndarray]:
    """Past excludes the current row; future includes it and is non-causal."""
    past = np.empty(len(values), dtype=np.float64)
    future = np.empty(len(values), dtype=np.float64)
    for index, time_value in enumerate(times):
        past_start = int(np.searchsorted(times, time_value - window_seconds, side="left"))
        if past_start < index:
            past[index] = float(np.mean(values[past_start:index]))
        else:
            past[index] = float(values[index])
        future_end = int(
            np.searchsorted(times, time_value + window_seconds, side="right")
        )
        future[index] = float(np.mean(values[index:future_end]))
    return past, future


def _reaction_lag_seconds(
    times: np.ndarray,
    near_onset: np.ndarray,
    far_onset: np.ndarray,
) -> np.ndarray:
    """Return far-peak time minus near-peak time in each forward horizon."""
    result = np.zeros(len(times), dtype=np.float64)
    for index, time_value in enumerate(times):
        end = int(
            np.searchsorted(
                times,
                time_value + REACTION_LAG_HORIZON_SECONDS,
                side="right",
            )
        )
        near_window = near_onset[index:end]
        far_window = far_onset[index:end]
        if (
            len(near_window) == 0
            or float(np.max(near_window)) <= 0.0
            or float(np.max(far_window)) <= 0.0
        ):
            continue
        near_index = index + int(np.argmax(near_window))
        far_index = index + int(np.argmax(far_window))
        result[index] = float(times[far_index] - times[near_index])
    return result


def _contrast(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    denominator = np.abs(positive) + np.abs(negative)
    return np.divide(
        positive - negative,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 1e-12,
    )


def derive_court_relative_features(
    prepared: PreparedRecording,
    variant: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Derive same-row ROI summaries without consulting any annotations.

    Returned columns are raw current-time derived values.  The nested candidate
    runner may rank/normalize them inside each recording before appending them;
    this function deliberately does not rebuild the base temporal context bank.
    """
    spec = court_relative_feature_spec(variant)
    times, source_values = _validate_prepared(prepared)
    indexes = {name: index for index, name in enumerate(prepared.sequence.names)}

    columns: dict[str, np.ndarray] = {}
    grids: dict[str, np.ndarray] = {}
    regions: dict[str, dict[str, np.ndarray]] = {}
    for source, prefix in _GRID_SOURCES:
        grid = source_values[
            :, [indexes[f"{prefix}{index}"] for index in range(GRID_SIZE**2)]
        ].reshape((-1, GRID_SIZE, GRID_SIZE))
        grids[source] = grid
        center = grid[:, 1, 1]
        margin = (np.sum(grid, axis=(1, 2)) - center) / 8.0
        far = np.mean(grid[:, 0, :], axis=1)
        near = np.mean(grid[:, 2, :], axis=1)
        left = np.mean(grid[:, :, 0], axis=1)
        right = np.mean(grid[:, :, 2], axis=1)
        regions[source] = {
            "near": near,
            "far": far,
            "left": left,
            "right": right,
        }
        columns[
            f"derived/court_relative/{source}_center_minus_margin_contrast"
        ] = _contrast(center, margin)
        columns[
            f"derived/court_relative/invariant/{source}_near_far_asymmetry_abs"
        ] = np.abs(_contrast(near, far))
        columns[
            f"derived/court_relative/invariant/{source}_left_right_asymmetry_abs"
        ] = np.abs(_contrast(right, left))
        columns[
            f"derived/court_relative/fixed/{source}_near_minus_far_contrast"
        ] = _contrast(near, far)
        columns[
            f"derived/court_relative/fixed/{source}_right_minus_left_contrast"
        ] = _contrast(right, left)

    flow_x = source_values[:, indexes["flow_median_x"]]
    flow_y = source_values[:, indexes["flow_median_y"]]
    flow_axis_total = np.abs(flow_x) + np.abs(flow_y)
    columns["derived/court_relative/flow_along_axis_abs"] = np.abs(flow_y)
    columns["derived/court_relative/flow_across_axis_abs"] = np.abs(flow_x)
    columns["derived/court_relative/flow_along_axis_fraction"] = np.divide(
        np.abs(flow_y),
        flow_axis_total,
        out=np.zeros_like(flow_axis_total),
        where=flow_axis_total > 1e-12,
    )
    columns["derived/court_relative/fixed/flow_toward_near_signed"] = flow_y
    columns["derived/court_relative/fixed/flow_toward_right_signed"] = flow_x

    centroid_x = source_values[:, indexes["player_motion_centroid_x"]]
    centroid_y = source_values[:, indexes["player_motion_centroid_y"]]
    spread_x = source_values[:, indexes["player_motion_spread_x"]]
    spread_y = source_values[:, indexes["player_motion_spread_y"]]
    spread_total = np.hypot(spread_x, spread_y)
    columns["derived/court_relative/player_motion_spread_total"] = spread_total
    columns[
        "derived/court_relative/player_motion_spread_along_minus_across_contrast"
    ] = _contrast(spread_y, spread_x)

    player_activity = np.mean(grids["player_motion"], axis=(1, 2))
    past_activity, future_activity = _window_means(
        times, player_activity, REACTION_WINDOW_SECONDS
    )
    activity_gate = np.clip(
        (past_activity + future_activity) / MOTION_ACTIVITY_GATE_SCALE,
        0.0,
        1.0,
    )
    past_spread, future_spread = _window_means(
        times, spread_total, REACTION_WINDOW_SECONDS
    )
    columns["derived/court_relative/player_motion_contraction"] = (
        np.maximum(past_spread - future_spread, 0.0) * activity_gate
    )

    past_centroid_x, future_centroid_x = _window_means(
        times, centroid_x, REACTION_WINDOW_SECONDS
    )
    past_centroid_y, future_centroid_y = _window_means(
        times, centroid_y, REACTION_WINDOW_SECONDS
    )
    migration_x = (future_centroid_x - past_centroid_x) * activity_gate
    migration_y = (future_centroid_y - past_centroid_y) * activity_gate
    migration_magnitude = np.hypot(migration_x, migration_y)
    migration_axis_total = np.abs(migration_x) + np.abs(migration_y)
    columns[
        "derived/court_relative/player_motion_migration_magnitude"
    ] = migration_magnitude
    columns[
        "derived/court_relative/player_motion_migration_along_fraction"
    ] = np.divide(
        np.abs(migration_y),
        migration_axis_total,
        out=np.zeros_like(migration_axis_total),
        where=migration_axis_total > 1e-12,
    )
    columns[
        "derived/court_relative/invariant/player_motion_centroid_across_offset_abs"
    ] = 2.0 * np.abs(centroid_x - 0.5)
    columns[
        "derived/court_relative/invariant/player_motion_centroid_along_offset_abs"
    ] = 2.0 * np.abs(centroid_y - 0.5)
    columns[
        "derived/court_relative/fixed/player_motion_centroid_toward_right_signed"
    ] = 2.0 * (centroid_x - 0.5)
    columns[
        "derived/court_relative/fixed/player_motion_centroid_toward_near_signed"
    ] = 2.0 * (centroid_y - 0.5)
    columns[
        "derived/court_relative/fixed/player_motion_migration_toward_right_signed"
    ] = migration_x
    columns[
        "derived/court_relative/fixed/player_motion_migration_toward_near_signed"
    ] = migration_y

    near_motion = regions["player_motion"]["near"]
    far_motion = regions["player_motion"]["far"]
    near_past, near_future = _window_means(
        times, near_motion, REACTION_WINDOW_SECONDS
    )
    far_past, far_future = _window_means(
        times, far_motion, REACTION_WINDOW_SECONDS
    )
    near_onset = np.maximum(near_future - near_past, 0.0)
    far_onset = np.maximum(far_future - far_past, 0.0)
    reaction_lag = _reaction_lag_seconds(times, near_onset, far_onset)
    columns[
        "derived/court_relative/invariant/reaction_onset_stronger_side"
    ] = np.maximum(near_onset, far_onset)
    columns[
        "derived/court_relative/invariant/reaction_onset_weaker_side"
    ] = np.minimum(near_onset, far_onset)
    columns[
        "derived/court_relative/invariant/reaction_onset_asymmetry_abs"
    ] = np.abs(_contrast(near_onset, far_onset))
    columns[
        "derived/court_relative/invariant/reaction_lag_abs_seconds"
    ] = np.abs(reaction_lag)
    columns["derived/court_relative/fixed/near_reaction_onset"] = near_onset
    columns["derived/court_relative/fixed/far_reaction_onset"] = far_onset
    columns[
        "derived/court_relative/fixed/near_minus_far_reaction_onset"
    ] = near_onset - far_onset
    columns[
        "derived/court_relative/fixed/far_minus_near_reaction_lag_seconds"
    ] = reaction_lag

    missing_columns = [name for name in spec.feature_names if name not in columns]
    if missing_columns:
        raise RuntimeError(f"court-relative implementation is missing {missing_columns}")
    result = np.column_stack([columns[name] for name in spec.feature_names])
    float32_limit = float(np.finfo(np.float32).max)
    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=float32_limit,
        neginf=-float32_limit,
    )
    result = np.clip(result, -float32_limit, float32_limit).astype(
        np.float32, copy=False
    )
    if result.shape != (len(times), len(spec.feature_names)) or not np.isfinite(
        result
    ).all():
        raise RuntimeError("court-relative feature bank is not finite and row-aligned")
    return np.ascontiguousarray(result), spec.feature_names


def append_court_relative_features(
    prepared: PreparedRecording,
    variant: str,
) -> PreparedRecording:
    """Append the compact derived bank once, without contextualizing it again."""
    derived_values, derived_names = derive_court_relative_features(prepared, variant)
    duplicates = sorted(set(prepared.contextual_names) & set(derived_names))
    if duplicates:
        raise ValueError(f"court-relative features are already present: {duplicates}")
    contextual = np.asarray(prepared.contextual_values)
    if not np.isfinite(contextual).all():
        raise ValueError("existing contextual values must be finite")
    combined = np.column_stack((contextual, derived_values)).astype(
        np.float32, copy=False
    )
    if not np.isfinite(combined).all():
        raise ValueError("appended court-relative feature matrix must be finite")
    return replace(
        prepared,
        contextual_values=np.ascontiguousarray(combined),
        contextual_names=(*prepared.contextual_names, *derived_names),
    )
