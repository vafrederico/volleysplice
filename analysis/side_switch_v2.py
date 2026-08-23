"""Side-conditioned features and temporal decoding for side-switch v2.

The v2 pipeline is deliberately small and auditable.  It uses the existing HOG
person proposals, separates central-court proposals into fixed near/far regions,
and compares the two possible color assignments across an inter-rally gap.  All
per-recording normalization and orientation binding are unsupervised.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from analysis.side_switch_appearance import (
    PersonDetection,
    detect_people,
    hellinger_distance,
    palette_vector,
)


MODEL_KIND = "volleycut-side-switch-specialist-v2"
MODEL_SCHEMA_VERSION = 2
FEATURE_ARTIFACT_KIND = "volleycut-side-switch-features-v2"
FEATURE_ARTIFACT_SCHEMA_VERSION = 2

FROZEN_RECORDING_SPLIT = {
    "train": (
        "raw-no-backup-PXL_20260816_164327879",
        "raw-no-backup-PXL_20260816_171720964",
        "raw-no-backup-PXL_20260816_190429172",
        "raw-no-backup-PXL_20260816_180646590",
        "raw-no-backup-PXL_20260816_183701800",
    ),
    "validation": (
        "raw-no-backup-PXL_20260816_210449857",
        "raw-no-backup-PXL_20260816_193307688",
    ),
    "evaluation": (
        "raw-no-backup-PXL_20260816_160023210",
        "raw-no-backup-PXL_20260816_161923155",
        "raw-no-backup-PXL_20260816_203801418",
        "raw-no-backup-PXL_20260816_212717581",
    ),
}
RECORDING_ROLE = {
    recording_id: role
    for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
    for recording_id in recording_ids
}

DERIVED_EXISTING_FEATURES = (
    "paletteDistanceMean",
    "paletteDistanceMinimum",
    "paletteDistanceDisagreement",
    "playerMinusBackgroundEvidence",
    "geometryStability",
    "paletteByGeometryStability",
    "areaPaletteByGeometryStability",
)
SIDE_CONDITIONED_FEATURES = (
    "sideSameAssignmentCostMedian",
    "sideSwappedAssignmentCostMedian",
    "sideSwapMarginMedian",
    "sideSwapMarginLowerQuartile",
    "sideSwapSupportFraction",
    "sideSwapMarginVariance",
    "sideUsablePairCount",
    "sideMinimumCoverage",
    "colorMomentFlipEvidenceMedian",
    "swapMarginByGeometryStability",
)
ALL_BASE_FEATURES = DERIVED_EXISTING_FEATURES + SIDE_CONDITIONED_FEATURES
FEATURE_SETS = {
    "derived-existing": DERIVED_EXISTING_FEATURES,
    "side-conditioned": SIDE_CONDITIONED_FEATURES,
    "combined": ALL_BASE_FEATURES,
    "combined-recording-normalized": tuple(
        f"normalized:{name}" for name in ALL_BASE_FEATURES
    ),
}


class SideSwitchV2Error(RuntimeError):
    pass


@dataclass(frozen=True)
class SideFrame:
    timestamp: float
    near_palette: np.ndarray | None
    far_palette: np.ndarray | None
    color_moment: np.ndarray | None
    central_detection_count: int


@dataclass(frozen=True)
class CourtPlayer:
    palette: np.ndarray
    area: float
    center_x: float
    foot_y: float


@dataclass(frozen=True)
class V2Event:
    event_id: str
    recording_id: str
    role: str
    rally_order: int
    decision: str
    label: int
    row: Mapping[str, Any]


@dataclass(frozen=True)
class V2Model:
    feature_set: str
    feature_names: tuple[str, ...]
    impute: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float
    l2: float

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise SideSwitchV2Error(
                f"model expects (*, {len(self.feature_names)}) features, got {matrix.shape}"
            )
        filled = np.where(np.isfinite(matrix), matrix, self.impute)
        normalized = (filled - self.mean) / self.scale
        logits = np.clip(normalized @ self.weights + self.bias, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "featureSet": self.feature_set,
            "featureNames": list(self.feature_names),
            "impute": self.impute.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "threshold": self.threshold,
            "l2": self.l2,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V2Model":
        feature_set = str(payload.get("featureSet", ""))
        expected_names = expanded_feature_names(feature_set)
        names = tuple(str(value) for value in payload.get("featureNames", []))
        if names != expected_names:
            raise SideSwitchV2Error("model feature names do not match its feature set")
        model = cls(
            feature_set=feature_set,
            feature_names=names,
            impute=np.asarray(payload.get("impute"), dtype=np.float64),
            mean=np.asarray(payload.get("mean"), dtype=np.float64),
            scale=np.asarray(payload.get("scale"), dtype=np.float64),
            weights=np.asarray(payload.get("weights"), dtype=np.float64),
            bias=float(payload.get("bias")),
            threshold=float(payload.get("threshold")),
            l2=float(payload.get("l2")),
        )
        expected_shape = (len(names),)
        arrays = (model.impute, model.mean, model.scale, model.weights)
        if any(value.shape != expected_shape for value in arrays):
            raise SideSwitchV2Error("model parameter shapes do not agree")
        if not all(np.isfinite(value).all() for value in arrays):
            raise SideSwitchV2Error("model parameters must be finite")
        if (
            np.any(model.scale <= 0)
            or not math.isfinite(model.bias)
            or not 0 <= model.threshold <= 1
            or model.l2 <= 0
        ):
            raise SideSwitchV2Error("model scalar parameters are invalid")
        return model


@dataclass(frozen=True)
class DecoderSettings:
    minimum_spacing_rallies: int
    close_switch_penalty: float
    orientation_weight: float
    extra_switch_penalty: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimumSpacingRallies": self.minimum_spacing_rallies,
            "closeSwitchPenalty": self.close_switch_penalty,
            "orientationWeight": self.orientation_weight,
            "extraSwitchPenalty": self.extra_switch_penalty,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecoderSettings":
        return cls(
            minimum_spacing_rallies=int(payload["minimumSpacingRallies"]),
            close_switch_penalty=float(payload["closeSwitchPenalty"]),
            orientation_weight=float(payload["orientationWeight"]),
            extra_switch_penalty=float(payload["extraSwitchPenalty"]),
        )


def _weighted_palette(
    palettes: Sequence[np.ndarray], weights: Sequence[float]
) -> np.ndarray | None:
    if not palettes:
        return None
    numeric = np.asarray(weights, dtype=np.float64)
    if numeric.shape != (len(palettes),) or not np.isfinite(numeric).all():
        raise ValueError("palette weights are invalid")
    numeric = np.maximum(numeric, 1e-9)
    result = np.average(np.stack(palettes), axis=0, weights=numeric)
    total = float(np.sum(result))
    return result / total if total > 0 else None


def _torso_crop(
    frame: np.ndarray, detection: PersonDetection
) -> np.ndarray | None:
    height, width = frame.shape[:2]
    left = max(0, round(detection.x + detection.width * 0.16))
    right = min(width, round(detection.x + detection.width * 0.84))
    top = max(0, round(detection.y + detection.height * 0.10))
    bottom = min(height, round(detection.y + detection.height * 0.66))
    if right - left < 6 or bottom - top < 8:
        return None
    return frame[top:bottom, left:right]


def extract_court_players(
    frame: np.ndarray,
    hog: cv2.HOGDescriptor,
    *,
    maximum_width: int = 1280,
    court_left: float = 0.12,
    court_right: float = 0.88,
    court_top: float = 0.38,
    court_bottom: float = 0.92,
) -> tuple[CourtPlayer, ...]:
    """Extract central-court proposals before choosing a near/far divider."""

    height, width = frame.shape[:2]
    working = frame
    if width > maximum_width:
        scale = maximum_width / width
        working = cv2.resize(
            frame,
            (maximum_width, max(2, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    height, width = working.shape[:2]
    players: list[CourtPlayer] = []
    for detection in detect_people(working, hog):
        center_x = (detection.x + detection.width / 2.0) / width
        foot_y = (detection.y + detection.height) / height
        if not (
            court_left <= center_x <= court_right
            and court_top <= foot_y <= court_bottom
        ):
            continue
        crop = _torso_crop(working, detection)
        if crop is None:
            continue
        players.append(
            CourtPlayer(
                palette=palette_vector(crop),
                area=detection.area,
                center_x=float(center_x),
                foot_y=float(foot_y),
            )
        )
    return tuple(players)


def calibrate_side_divider(
    player_frames: Sequence[Sequence[CourtPlayer]],
    *,
    fallback: float = 0.68,
    minimum_players: int = 12,
) -> tuple[float, dict[str, Any]]:
    """Estimate the near/far boundary with unlabeled weighted two-means."""

    players = [player for frame in player_frames for player in frame]
    if len(players) < minimum_players:
        return fallback, {
            "source": "fallback",
            "playerCount": len(players),
            "divider": fallback,
            "reason": "insufficient-player-proposals",
        }
    values = np.asarray([player.foot_y for player in players], dtype=np.float64)
    weights = np.sqrt(np.asarray([player.area for player in players], dtype=np.float64))
    centers = np.asarray(np.percentile(values, [30.0, 70.0]), dtype=np.float64)
    for _ in range(30):
        assignments = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        updated = centers.copy()
        for cluster in (0, 1):
            selected = assignments == cluster
            if int(np.sum(selected)) < 4:
                return fallback, {
                    "source": "fallback",
                    "playerCount": len(players),
                    "divider": fallback,
                    "reason": "unbalanced-position-clusters",
                }
            updated[cluster] = float(
                np.average(values[selected], weights=weights[selected])
            )
        updated.sort()
        if float(np.max(np.abs(updated - centers))) < 1e-7:
            centers = updated
            break
        centers = updated
    separation = float(centers[1] - centers[0])
    if separation < 0.055:
        return fallback, {
            "source": "fallback",
            "playerCount": len(players),
            "divider": fallback,
            "reason": "position-clusters-too-close",
            "clusterCenters": centers.tolist(),
        }
    divider = float(np.clip(np.mean(centers), 0.52, 0.82))
    return divider, {
        "source": "unlabeled-weighted-two-means",
        "playerCount": len(players),
        "divider": divider,
        "clusterCenters": centers.tolist(),
        "clusterSeparation": separation,
    }


def refine_side_divider(
    player_frames: Sequence[Sequence[CourtPlayer]],
    recording_divider: float,
) -> tuple[float, dict[str, Any]]:
    """Shrink a local calibration toward the recording-wide divider."""

    local, audit = calibrate_side_divider(
        player_frames,
        fallback=recording_divider,
        minimum_players=16,
    )
    if audit["source"] == "fallback":
        return recording_divider, {
            **audit,
            "source": "recording-divider-fallback",
            "recordingDivider": recording_divider,
        }
    count = int(audit["playerCount"])
    local_weight = min(0.75, count / 64.0)
    divider = (1.0 - local_weight) * recording_divider + local_weight * local
    return float(divider), {
        **audit,
        "source": "local-two-means-shrunk-to-recording",
        "localDivider": local,
        "recordingDivider": recording_divider,
        "localWeight": local_weight,
        "divider": float(divider),
    }


def side_frame_from_players(
    players: Sequence[CourtPlayer], timestamp: float, side_divider: float
) -> SideFrame:
    near_palettes: list[np.ndarray] = []
    near_weights: list[float] = []
    far_palettes: list[np.ndarray] = []
    far_weights: list[float] = []
    moment_terms: list[np.ndarray] = []
    moment_weights: list[float] = []
    for player in players:
        signed_position = float(player.foot_y - side_divider)
        position_weight = max(abs(signed_position), 0.015)
        weighted_area = player.area * position_weight
        moment_terms.append(
            player.palette * math.copysign(1.0, signed_position)
        )
        moment_weights.append(weighted_area)
        if signed_position >= 0:
            near_palettes.append(player.palette)
            near_weights.append(weighted_area)
        else:
            far_palettes.append(player.palette)
            far_weights.append(weighted_area)
    moment = None
    if moment_terms:
        moment = np.average(
            np.stack(moment_terms), axis=0, weights=np.asarray(moment_weights)
        )
    return SideFrame(
        timestamp=float(timestamp),
        near_palette=_weighted_palette(near_palettes, near_weights),
        far_palette=_weighted_palette(far_palettes, far_weights),
        color_moment=moment,
        central_detection_count=len(players),
    )


def summarize_side_frame(
    frame: np.ndarray,
    timestamp: float,
    hog: cv2.HOGDescriptor,
    *,
    side_divider: float = 0.68,
) -> SideFrame:
    """Convenience wrapper for a pre-calibrated near/far divider."""

    return side_frame_from_players(
        extract_court_players(frame, hog), timestamp, side_divider
    )


def _signed_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return None
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def side_pair_features(
    before_frames: Sequence[SideFrame], after_frames: Sequence[SideFrame]
) -> tuple[dict[str, float | None], np.ndarray | None, np.ndarray | None]:
    """Return frame-pair statistics and the two window color moments."""

    if not before_frames or len(before_frames) != len(after_frames):
        raise ValueError("before and after frame sequences must be non-empty and aligned")
    same_costs: list[float] = []
    swapped_costs: list[float] = []
    margins: list[float] = []
    flip_evidence: list[float] = []
    for before, after in zip(before_frames, after_frames, strict=True):
        if (
            before.near_palette is not None
            and before.far_palette is not None
            and after.near_palette is not None
            and after.far_palette is not None
        ):
            same = 0.5 * (
                hellinger_distance(before.near_palette, after.near_palette)
                + hellinger_distance(before.far_palette, after.far_palette)
            )
            swapped = 0.5 * (
                hellinger_distance(before.near_palette, after.far_palette)
                + hellinger_distance(before.far_palette, after.near_palette)
            )
            same_costs.append(same)
            swapped_costs.append(swapped)
            margins.append(same - swapped)
        if before.color_moment is not None and after.color_moment is not None:
            similarity = _signed_cosine(before.color_moment, after.color_moment)
            if similarity is not None:
                flip_evidence.append(-similarity)

    before_usable = sum(
        frame.near_palette is not None and frame.far_palette is not None
        for frame in before_frames
    )
    after_usable = sum(
        frame.near_palette is not None and frame.far_palette is not None
        for frame in after_frames
    )
    coverage = min(before_usable / len(before_frames), after_usable / len(after_frames))

    def median(values: Sequence[float]) -> float | None:
        return float(np.median(values)) if values else None

    before_moments = [
        frame.color_moment for frame in before_frames if frame.color_moment is not None
    ]
    after_moments = [
        frame.color_moment for frame in after_frames if frame.color_moment is not None
    ]
    before_moment = (
        np.mean(np.stack(before_moments), axis=0) if before_moments else None
    )
    after_moment = np.mean(np.stack(after_moments), axis=0) if after_moments else None
    features: dict[str, float | None] = {
        "sideSameAssignmentCostMedian": median(same_costs),
        "sideSwappedAssignmentCostMedian": median(swapped_costs),
        "sideSwapMarginMedian": median(margins),
        "sideSwapMarginLowerQuartile": (
            float(np.percentile(margins, 25.0)) if margins else None
        ),
        "sideSwapSupportFraction": (
            float(np.mean(np.asarray(margins) > 0.0)) if margins else None
        ),
        "sideSwapMarginVariance": float(np.var(margins)) if margins else None,
        "sideUsablePairCount": float(len(margins)),
        "sideMinimumCoverage": float(coverage),
        "colorMomentFlipEvidenceMedian": median(flip_evidence),
    }
    return features, before_moment, after_moment


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result):
            return result
    return None


def derived_existing_features(event: Mapping[str, Any]) -> dict[str, float | None]:
    existing = event.get("features")
    source = existing if isinstance(existing, Mapping) else {}
    equal = _finite_number(source.get("playerPaletteEqual"))
    area = _finite_number(source.get("playerPaletteArea"))
    full = _finite_number(source.get("fullFrameControl"))
    palette_values = [value for value in (equal, area) if value is not None]
    palette_mean = float(np.mean(palette_values)) if palette_values else None
    palette_minimum = min(palette_values) if palette_values else None
    disagreement = abs(equal - area) if equal is not None and area is not None else None
    changes = [
        _finite_number(source.get(name))
        for name in (
            "detectionCountChange",
            "boxAreaChange",
            "medianBoxHeightChange",
        )
    ]
    finite_changes = [value for value in changes if value is not None]
    stability = (
        1.0 - min(1.0, float(np.mean(finite_changes)))
        if finite_changes
        else None
    )
    return {
        "paletteDistanceMean": palette_mean,
        "paletteDistanceMinimum": palette_minimum,
        "paletteDistanceDisagreement": disagreement,
        "playerMinusBackgroundEvidence": (
            palette_mean - full
            if palette_mean is not None and full is not None
            else None
        ),
        "geometryStability": stability,
        "paletteByGeometryStability": (
            palette_mean * stability
            if palette_mean is not None and stability is not None
            else None
        ),
        "areaPaletteByGeometryStability": (
            area * stability if area is not None and stability is not None else None
        ),
    }


def add_interactions(features: Mapping[str, Any]) -> dict[str, float | None]:
    result = {name: _finite_number(features.get(name)) for name in ALL_BASE_FEATURES}
    margin = result.get("sideSwapMarginMedian")
    stability = result.get("geometryStability")
    result["swapMarginByGeometryStability"] = (
        margin * stability if margin is not None and stability is not None else None
    )
    return result


def bind_recording_normalization(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Attach unsupervised robust z-scores using every candidate per recording."""

    metadata: dict[str, Any] = {}
    for recording_id in sorted({str(row["recordingId"]) for row in rows}):
        recording_rows = [row for row in rows if row["recordingId"] == recording_id]
        feature_metadata: dict[str, Any] = {}
        for name in ALL_BASE_FEATURES:
            values = [
                _finite_number(row.get("features", {}).get(name))
                for row in recording_rows
            ]
            finite = np.asarray([value for value in values if value is not None])
            if len(finite):
                center = float(np.median(finite))
                mad = float(np.median(np.abs(finite - center)))
                scale = 1.4826 * mad
                fallback = "mad"
                if scale < 1e-8 and len(finite) >= 2:
                    scale = float(
                        (np.percentile(finite, 75.0) - np.percentile(finite, 25.0))
                        / 1.349
                    )
                    fallback = "iqr"
                if scale < 1e-8:
                    scale = 1.0
                    fallback = "unit"
            else:
                center, mad, scale, fallback = 0.0, 0.0, 1.0, "no-finite-values"
            for row, value in zip(recording_rows, values, strict=True):
                normalized = row.setdefault("normalizedFeatures", {})
                normalized[name] = (
                    None
                    if value is None
                    else float(np.clip((value - center) / scale, -10.0, 10.0))
                )
            feature_metadata[name] = {
                "finiteCount": int(len(finite)),
                "median": center,
                "mad": mad,
                "scale": scale,
                "scaleSource": fallback,
            }
        metadata[recording_id] = {
            "candidateCount": len(recording_rows),
            "features": feature_metadata,
        }
    return metadata


def bind_recording_orientation(
    rows: Sequence[dict[str, Any]],
    moments: Mapping[str, tuple[np.ndarray | None, np.ndarray | None]],
) -> dict[str, Any]:
    """Bind each recording's color moments to an unlabeled one-dimensional axis."""

    metadata: dict[str, Any] = {}
    for recording_id in sorted({str(row["recordingId"]) for row in rows}):
        recording_rows = [row for row in rows if row["recordingId"] == recording_id]
        vectors = [
            vector
            for row in recording_rows
            for vector in moments[row["eventId"]]
            if vector is not None
        ]
        if len(vectors) >= 2:
            matrix = np.stack(vectors).astype(np.float64)
            center = np.mean(matrix, axis=0)
            _, singular_values, right = np.linalg.svd(matrix - center, full_matrices=False)
            component = right[0]
            projections = (matrix - center) @ component
            projection_center = float(np.median(projections))
            projection_mad = float(np.median(np.abs(projections - projection_center)))
            projection_scale = max(1.4826 * projection_mad, 1e-8)
            explained = float(
                singular_values[0] ** 2 / max(float(np.sum(singular_values**2)), 1e-12)
            )
        else:
            dimension = len(vectors[0]) if vectors else 1
            center = np.zeros(dimension)
            component = np.zeros(dimension)
            component[0] = 1.0
            projection_center, projection_scale, explained = 0.0, 1.0, 0.0
        for row in recording_rows:
            before, after = moments[row["eventId"]]

            def project(value: np.ndarray | None) -> float | None:
                if value is None:
                    return None
                return float(
                    np.clip(
                        ((value - center) @ component - projection_center)
                        / projection_scale,
                        -6.0,
                        6.0,
                    )
                )

            row["beforeOrientation"] = project(before)
            row["afterOrientation"] = project(after)
        metadata[recording_id] = {
            "candidateCount": len(recording_rows),
            "usableWindowMoments": len(vectors),
            "componentDimension": int(len(component)),
            "explainedVarianceFraction": explained,
            "projectionMedian": projection_center,
            "projectionScale": projection_scale,
            "center": center.tolist(),
            "component": component.tolist(),
        }
    return metadata


def expanded_feature_names(feature_set: str) -> tuple[str, ...]:
    specifications = FEATURE_SETS.get(feature_set)
    if specifications is None:
        raise SideSwitchV2Error(f"unknown feature set: {feature_set}")
    return tuple(
        value
        for specification in specifications
        for value in (specification, f"{specification}:missing")
    )


def event_vector(row: Mapping[str, Any], feature_set: str) -> np.ndarray:
    specifications = FEATURE_SETS.get(feature_set)
    if specifications is None:
        raise SideSwitchV2Error(f"unknown feature set: {feature_set}")
    raw = row.get("features") if isinstance(row.get("features"), Mapping) else {}
    normalized = (
        row.get("normalizedFeatures")
        if isinstance(row.get("normalizedFeatures"), Mapping)
        else {}
    )
    values: list[float] = []
    for specification in specifications:
        source = normalized if specification.startswith("normalized:") else raw
        name = specification.split(":", 1)[-1]
        value = _finite_number(source.get(name))
        values.extend((math.nan if value is None else value, 1.0 if value is None else 0.0))
    return np.asarray(values, dtype=np.float64)


def matrix_for(events: Sequence[V2Event], feature_set: str) -> np.ndarray:
    if not events:
        return np.empty((0, len(expanded_feature_names(feature_set))))
    return np.stack([event_vector(event.row, feature_set) for event in events])


def labels_for(events: Sequence[V2Event]) -> np.ndarray:
    return np.asarray([event.label for event in events], dtype=np.float64)


def _fit_preprocessor(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    impute = np.zeros(values.shape[1], dtype=np.float64)
    for index in range(values.shape[1]):
        finite = values[np.isfinite(values[:, index]), index]
        impute[index] = float(np.median(finite)) if len(finite) else 0.0
    filled = np.where(np.isfinite(values), values, impute)
    mean = np.mean(filled, axis=0)
    scale = np.std(filled, axis=0)
    scale[scale < 1e-6] = 1.0
    return impute, mean, scale


def fit_model(events: Sequence[V2Event], feature_set: str, l2: float) -> V2Model:
    if l2 <= 0:
        raise SideSwitchV2Error("l2 must be positive")
    values = matrix_for(events, feature_set)
    labels = labels_for(events)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        raise SideSwitchV2Error("training events must contain both classes")
    impute, mean, scale = _fit_preprocessor(values)
    matrix = (np.where(np.isfinite(values), values, impute) - mean) / scale
    sample_weights = np.where(
        labels == 1,
        len(labels) / (2.0 * positives),
        len(labels) / (2.0 * negatives),
    )
    dimensions = matrix.shape[1]
    weights = np.zeros(dimensions)
    bias = 0.0
    design = np.column_stack((matrix, np.ones(len(matrix))))
    regularizer = np.diag(np.r_[np.full(dimensions, l2), 0.0])
    for _ in range(100):
        logits = np.clip(matrix @ weights + bias, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = (probabilities - labels) * sample_weights
        gradient = design.T @ error / np.sum(sample_weights)
        gradient[:-1] += l2 * weights
        curvature = probabilities * (1.0 - probabilities) * sample_weights
        hessian = (design.T * curvature) @ design / np.sum(sample_weights)
        hessian += regularizer
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        current_logits = matrix @ weights + bias
        current_loss = float(
            np.sum(
                (np.logaddexp(0.0, current_logits) - labels * current_logits)
                * sample_weights
            )
            / np.sum(sample_weights)
            + 0.5 * l2 * (weights @ weights)
        )
        step_scale = 1.0
        while step_scale > 1e-5:
            candidate_weights = weights - step_scale * step[:-1]
            candidate_bias = bias - step_scale * float(step[-1])
            candidate_logits = matrix @ candidate_weights + candidate_bias
            candidate_loss = float(
                np.sum(
                    (np.logaddexp(0.0, candidate_logits) - labels * candidate_logits)
                    * sample_weights
                )
                / np.sum(sample_weights)
                + 0.5 * l2 * (candidate_weights @ candidate_weights)
            )
            if candidate_loss <= current_loss + 1e-12:
                break
            step_scale *= 0.5
        weights = candidate_weights
        bias = candidate_bias
        if float(np.max(np.abs(step_scale * step))) < 1e-8:
            break
    return V2Model(
        feature_set,
        expanded_feature_names(feature_set),
        impute,
        mean,
        scale,
        weights,
        bias,
        0.5,
        l2,
    )


def binary_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    guesses = np.asarray(predicted, dtype=bool)
    if truth.shape != guesses.shape:
        raise SideSwitchV2Error("metric labels and predictions are not aligned")
    tp = int(np.sum((truth == 1) & guesses))
    fp = int(np.sum((truth == 0) & guesses))
    fn = int(np.sum((truth == 1) & ~guesses))
    tn = int(np.sum((truth == 0) & ~guesses))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "rows": len(truth),
        "positives": int(np.sum(truth == 1)),
        "negatives": int(np.sum(truth == 0)),
        "truePositives": tp,
        "falsePositives": fp,
        "falseNegatives": fn,
        "trueNegatives": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(truth) if len(truth) else None,
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.shape != scores.shape or not len(truth) or not np.isfinite(scores).all():
        raise SideSwitchV2Error("threshold inputs must be finite, non-empty, and aligned")
    thresholds = sorted({float(value) for value in scores}, reverse=True)
    thresholds.insert(0, math.nextafter(thresholds[0], math.inf))
    candidates = [
        {"threshold": threshold, **binary_metrics(truth, scores >= threshold)}
        for threshold in thresholds
    ]
    return max(
        candidates,
        key=lambda row: (
            float(row["f1"] or 0.0),
            float(row["precision"] or 0.0),
            float(row["recall"] or 0.0),
            float(row["threshold"]),
        ),
    )


def grouped_cross_fit(
    events: Sequence[V2Event], feature_set: str, l2: float
) -> tuple[np.ndarray, dict[str, Any]]:
    recording_ids = sorted({event.recording_id for event in events})
    if len(recording_ids) < 2:
        raise SideSwitchV2Error("grouped cross-fit needs at least two recordings")
    scores = np.full(len(events), np.nan)
    folds: list[dict[str, Any]] = []
    for held_id in recording_ids:
        train_indices = [i for i, event in enumerate(events) if event.recording_id != held_id]
        held_indices = [i for i, event in enumerate(events) if event.recording_id == held_id]
        train_events = [events[i] for i in train_indices]
        held_events = [events[i] for i in held_indices]
        model = fit_model(train_events, feature_set, l2)
        scores[held_indices] = model.predict_proba(matrix_for(held_events, feature_set))
        folds.append(
            {
                "heldRecordingId": held_id,
                "trainRows": len(train_events),
                "heldRows": len(held_events),
                "heldPositives": sum(event.label for event in held_events),
            }
        )
    if not np.isfinite(scores).all():
        raise SideSwitchV2Error("grouped cross-fit left events unscored")
    return scores, {"folds": folds, "selectedThresholdMetrics": select_threshold(labels_for(events), scores)}


def _orientation(value: Any) -> float:
    numeric = _finite_number(value)
    return float(np.clip(numeric, -3.0, 3.0)) if numeric is not None else 0.0


def _best_decode(
    events: Sequence[V2Event],
    probabilities: np.ndarray,
    threshold: float,
    settings: DecoderSettings,
    forced: tuple[int, int] | None = None,
) -> tuple[float, np.ndarray]:
    if len(events) != len(probabilities):
        raise SideSwitchV2Error("decoder events and probabilities are not aligned")
    if not events:
        return 0.0, np.empty(0, dtype=bool)
    threshold_logit = math.log(max(threshold, 1e-9) / max(1.0 - threshold, 1e-9))
    # State is (current orientation parity, rally order of last switch).
    states: dict[tuple[int, int | None], tuple[float, tuple[bool, ...]]] = {
        (0, None): (0.0, ()),
        (1, None): (0.0, ()),
    }
    for index, (event, probability) in enumerate(zip(events, probabilities, strict=True)):
        probability = float(np.clip(probability, 1e-9, 1.0 - 1e-9))
        next_states: dict[tuple[int, int | None], tuple[float, tuple[bool, ...]]] = {}
        allowed = (forced[1],) if forced is not None and forced[0] == index else (0, 1)
        for (parity, last_switch), (prior_score, path) in states.items():
            for decision in allowed:
                after_parity = parity ^ int(decision)
                classifier_score = math.log(probability if decision else 1.0 - probability)
                if decision:
                    classifier_score -= threshold_logit + settings.extra_switch_penalty
                spacing_penalty = 0.0
                if (
                    decision
                    and last_switch is not None
                    and settings.minimum_spacing_rallies > 0
                ):
                    distance = event.rally_order - last_switch
                    deficit = max(0, settings.minimum_spacing_rallies - distance)
                    spacing_penalty = settings.close_switch_penalty * (
                        deficit / settings.minimum_spacing_rallies
                    )
                before = _orientation(event.row.get("beforeOrientation"))
                after = _orientation(event.row.get("afterOrientation"))
                orientation_score = settings.orientation_weight * 0.5 * (
                    (1.0 if parity == 0 else -1.0) * before
                    + (1.0 if after_parity == 0 else -1.0) * after
                )
                score = prior_score + classifier_score + orientation_score - spacing_penalty
                key = (
                    after_parity,
                    event.rally_order if decision else last_switch,
                )
                existing = next_states.get(key)
                candidate_path = (*path, bool(decision))
                if existing is None or score > existing[0] + 1e-12:
                    next_states[key] = (score, candidate_path)
        states = next_states
    best_score, best_path = max(states.values(), key=lambda value: value[0])
    return best_score, np.asarray(best_path, dtype=bool)


def decode_sequence(
    events: Sequence[V2Event],
    probabilities: np.ndarray,
    threshold: float,
    settings: DecoderSettings,
    *,
    calculate_scores: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Decode one recording and optionally calculate per-event max-marginal scores."""

    score, predictions = _best_decode(events, probabilities, threshold, settings)
    if not calculate_scores:
        return predictions, probabilities.copy(), score
    marginal_scores = np.zeros(len(events))
    for index in range(len(events)):
        switch_score, _ = _best_decode(
            events, probabilities, threshold, settings, (index, 1)
        )
        no_switch_score, _ = _best_decode(
            events, probabilities, threshold, settings, (index, 0)
        )
        difference = float(np.clip(switch_score - no_switch_score, -30.0, 30.0))
        marginal_scores[index] = 1.0 / (1.0 + math.exp(-difference))
    return predictions, marginal_scores, score


def with_threshold(model: V2Model, threshold: float) -> V2Model:
    return replace(model, threshold=float(threshold))


def fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
