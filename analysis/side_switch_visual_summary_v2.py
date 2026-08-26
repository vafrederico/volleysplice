"""Reusable visual observations for directional, camera, and persistence research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from analysis.side_switch_v4 import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    CourtGeometry,
    SequenceSummary,
    _hellinger,
    _weighted_palette,
    normalize_court_frame,
    summarize_sequence,
    visual_features,
)
from analysis.side_switch_v5 import (
    PlayerSequenceSummary,
    _proposal_boxes,
    player_features,
    summarize_player_sequence,
)


Q1_REMOVED_FEATURE_NAMES = (
    "v4MaximumCameraShift",
    "v4MinimumAlignmentResponse",
    "minimumPlayerSideSeparation",
    "playerSideSeparationChange",
    "minimumProposalCoverage",
    "proposalCoverageChange",
    "minimumProposalCount",
    "proposalCountChange",
    "minimumNearSupport",
    "minimumFarSupport",
    "sideSupportImbalanceChange",
)
Q1_FEATURE_NAMES = (
    "beforeV4MaximumCameraShift",
    "afterV4MaximumCameraShift",
    "beforeV4MinimumAlignmentResponse",
    "afterV4MinimumAlignmentResponse",
    "beforePlayerSideSeparation",
    "afterPlayerSideSeparation",
    "beforeProposalCoverage",
    "afterProposalCoverage",
    "beforeProposalCount",
    "afterProposalCount",
    "beforeNearSupport",
    "afterNearSupport",
    "beforeFarSupport",
    "afterFarSupport",
)
C1_FEATURE_NAMES = (
    "cameraShiftDispersion",
    "alignmentResidualP90",
    "backgroundAppearanceChange",
    "sceneCutScore",
)
P1_FEATURE_NAMES = (
    "playerCrossSwapQ25",
    "playerWithinContinuityMinimum",
    "playerPersistentSwapMinimum",
    "courtCrossSwapQ25",
    "courtWithinContinuityMinimum",
    "courtPersistentSwapMinimum",
    "crossModalityPersistentMinimum",
    "crossModalityCrossSwapDisagreement",
    "minimumPersistentContextFraction",
)


@dataclass(frozen=True)
class CameraSummary:
    translation_vectors: tuple[tuple[float, float], ...]
    alignment_responses: tuple[float, ...]
    shift_dispersion: float
    residual_p90: float
    background_palette: np.ndarray
    frame_global_palettes: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class SideObservationV2:
    v4: SequenceSummary
    player: PlayerSequenceSummary
    camera: CameraSummary


def _camera_summary(
    frames: Sequence[np.ndarray], geometry: CourtGeometry
) -> CameraSummary:
    if len(frames) < 3:
        raise ValueError("camera summary needs at least three frames")
    normalized = [normalize_court_frame(frame, geometry) for frame in frames]
    reference_index = len(normalized) // 2
    reference_gray = cv2.cvtColor(normalized[reference_index], cv2.COLOR_BGR2GRAY)
    calibration_height = round(FRAME_HEIGHT * 0.42)
    reference = cv2.GaussianBlur(
        reference_gray[:calibration_height], (7, 7), 0
    ).astype(np.float32)
    window = cv2.createHanningWindow(
        (FRAME_WIDTH, calibration_height), cv2.CV_32F
    )
    aligned: list[np.ndarray] = []
    vectors: list[tuple[float, float]] = []
    responses: list[float] = []
    for frame in normalized:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current = cv2.GaussianBlur(
            gray[:calibration_height], (7, 7), 0
        ).astype(np.float32)
        (dx, dy), response = cv2.phaseCorrelate(reference, current, window)
        normalized_shift = math.hypot(dx / FRAME_WIDTH, dy / FRAME_HEIGHT)
        valid = (
            math.isfinite(normalized_shift)
            and math.isfinite(response)
            and response >= 0.02
            and normalized_shift <= 0.12
        )
        if valid:
            transform = np.asarray([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
            aligned_frame = cv2.warpAffine(
                frame,
                transform,
                (FRAME_WIDTH, FRAME_HEIGHT),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            vectors.append((float(dx / FRAME_WIDTH), float(dy / FRAME_HEIGHT)))
        else:
            aligned_frame = frame
            vectors.append((0.0, 0.0))
        responses.append(float(response) if math.isfinite(response) else 0.0)
        aligned.append(aligned_frame)

    vector_matrix = np.asarray(vectors, dtype=np.float64)
    vector_center = np.median(vector_matrix, axis=0)
    shift_dispersion = float(
        np.median(np.linalg.norm(vector_matrix - vector_center, axis=1))
    )
    gray_stack = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in aligned]
    ).astype(np.float64)
    median_gray = np.median(gray_stack, axis=0)
    residual_p90 = float(
        np.percentile(np.abs(gray_stack - median_gray) / 255.0, 90.0)
    )
    motion = (np.max(gray_stack, axis=0) - np.min(gray_stack, axis=0)) / 255.0
    hsv_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in aligned]
    background_palettes: list[np.ndarray] = []
    global_palettes: list[np.ndarray] = []
    for hsv, gray in zip(hsv_frames, gray_stack, strict=True):
        difference = np.abs(gray - median_gray) / 255.0
        proposal_mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=bool)
        for x, y, width, height in _proposal_boxes(difference):
            proposal_mask[y : y + height, x : x + width] = True
        background_weights = (
            (motion <= (10.0 / 255.0)) & ~proposal_mask
        ).astype(np.float64)
        background_palettes.append(_weighted_palette(hsv, background_weights))
        global_palettes.append(_weighted_palette(hsv, np.ones_like(motion)))
    background = np.mean(np.stack(background_palettes), axis=0)
    background /= max(float(np.sum(background)), 1e-12)
    return CameraSummary(
        translation_vectors=tuple(vectors),
        alignment_responses=tuple(responses),
        shift_dispersion=shift_dispersion,
        residual_p90=residual_p90,
        background_palette=background,
        frame_global_palettes=tuple(global_palettes),
    )


def summarize_observation(
    frames: Sequence[np.ndarray], geometry: CourtGeometry
) -> SideObservationV2:
    """Build all V2 summaries from one decoded sequence without using labels."""

    return SideObservationV2(
        v4=summarize_sequence(frames, geometry),
        player=summarize_player_sequence(frames, geometry),
        camera=_camera_summary(frames, geometry),
    )


def current_visual_features(
    before: SideObservationV2, after: SideObservationV2
) -> dict[str, float]:
    return player_features(
        before.player,
        after.player,
        visual_features(before.v4, after.v4),
    )


def q1_features(
    before: SideObservationV2, after: SideObservationV2
) -> dict[str, float]:
    values = {
        "beforeV4MaximumCameraShift": before.v4.maximum_camera_shift,
        "afterV4MaximumCameraShift": after.v4.maximum_camera_shift,
        "beforeV4MinimumAlignmentResponse": before.v4.minimum_alignment_response,
        "afterV4MinimumAlignmentResponse": after.v4.minimum_alignment_response,
        "beforePlayerSideSeparation": _hellinger(
            before.player.near_palette, before.player.far_palette
        ),
        "afterPlayerSideSeparation": _hellinger(
            after.player.near_palette, after.player.far_palette
        ),
        "beforeProposalCoverage": before.player.proposal_coverage,
        "afterProposalCoverage": after.player.proposal_coverage,
        "beforeProposalCount": before.player.proposal_count,
        "afterProposalCount": after.player.proposal_count,
        "beforeNearSupport": before.player.near_support,
        "afterNearSupport": after.player.near_support,
        "beforeFarSupport": before.player.far_support,
        "afterFarSupport": after.player.far_support,
    }
    if tuple(values) != Q1_FEATURE_NAMES:
        raise AssertionError("Q1 feature order changed")
    return values


def c1_features(
    before: SideObservationV2, after: SideObservationV2
) -> dict[str, float]:
    ordered_palettes = (
        *before.camera.frame_global_palettes,
        *after.camera.frame_global_palettes,
    )
    adjacent_distances = [
        _hellinger(left, right)
        for left, right in zip(ordered_palettes, ordered_palettes[1:], strict=False)
    ]
    values = {
        "cameraShiftDispersion": max(
            before.camera.shift_dispersion, after.camera.shift_dispersion
        ),
        "alignmentResidualP90": max(
            before.camera.residual_p90, after.camera.residual_p90
        ),
        "backgroundAppearanceChange": _hellinger(
            before.camera.background_palette, after.camera.background_palette
        ),
        "sceneCutScore": max(adjacent_distances, default=0.0),
    }
    if tuple(values) != C1_FEATURE_NAMES:
        raise AssertionError("C1 feature order changed")
    return values


def _player_margin(left: SideObservationV2, right: SideObservationV2) -> float:
    same = 0.5 * (
        _hellinger(left.player.near_palette, right.player.near_palette)
        + _hellinger(left.player.far_palette, right.player.far_palette)
    )
    swapped = 0.5 * (
        _hellinger(left.player.near_palette, right.player.far_palette)
        + _hellinger(left.player.far_palette, right.player.near_palette)
    )
    return same - swapped


def _court_margin(left: SideObservationV2, right: SideObservationV2) -> float:
    margins = []
    for left_palette, right_palette in (
        (left.v4.broad, right.v4.broad),
        (left.v4.tight, right.v4.tight),
    ):
        same = 0.5 * (
            _hellinger(left_palette.near, right_palette.near)
            + _hellinger(left_palette.far, right_palette.far)
        )
        swapped = 0.5 * (
            _hellinger(left_palette.near, right_palette.far)
            + _hellinger(left_palette.far, right_palette.near)
        )
        margins.append(same - swapped)
    return float(np.mean(margins))


def _modality_persistence(
    before: Sequence[SideObservationV2],
    after: Sequence[SideObservationV2],
    margin: Any,
) -> tuple[dict[str, float], dict[str, Any]]:
    cross = np.asarray(
        [margin(left, right) for left in before for right in after],
        dtype=np.float64,
    )
    before_within = np.asarray(
        [
            margin(before[left], before[right])
            for left in range(len(before))
            for right in range(left + 1, len(before))
        ],
        dtype=np.float64,
    )
    after_within = np.asarray(
        [
            margin(after[left], after[right])
            for left in range(len(after))
            for right in range(left + 1, len(after))
        ],
        dtype=np.float64,
    )
    cross_q25 = float(np.quantile(cross, 0.25))
    # Missing within-side pairs occur only near recording edges. Zero is neutral;
    # contextFraction exposes the missing evidence instead of fabricating continuity.
    before_continuity = (
        -float(np.median(before_within)) if len(before_within) else 0.0
    )
    after_continuity = (
        -float(np.median(after_within)) if len(after_within) else 0.0
    )
    within_minimum = min(before_continuity, after_continuity)
    persistent_minimum = min(cross_q25, within_minimum)
    values = {
        "crossSwapQ25": cross_q25,
        "withinContinuityMinimum": within_minimum,
        "persistentSwapMinimum": persistent_minimum,
    }
    diagnostics = {
        "crossPairCount": int(len(cross)),
        "crossSwapMedian": float(np.median(cross)),
        "crossSwapAgreementFraction": float(np.mean(cross > 0.0)),
        "beforeWithinPairCount": int(len(before_within)),
        "afterWithinPairCount": int(len(after_within)),
        "beforeContinuity": before_continuity,
        "afterContinuity": after_continuity,
    }
    return values, diagnostics


def p1_features(
    before: Sequence[SideObservationV2],
    after: Sequence[SideObservationV2],
    *,
    context_width: int = 3,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return the fixed K=3 boundary-persistence bundle and diagnostics."""

    if context_width != 3:
        raise ValueError("P1 context width is frozen at three")
    if not before or not after or len(before) > 3 or len(after) > 3:
        raise ValueError("P1 needs one to three observations on each side")
    player, player_diagnostics = _modality_persistence(
        before, after, _player_margin
    )
    court, court_diagnostics = _modality_persistence(before, after, _court_margin)
    context_fraction = min(len(before), len(after)) / context_width
    values = {
        "playerCrossSwapQ25": player["crossSwapQ25"],
        "playerWithinContinuityMinimum": player["withinContinuityMinimum"],
        "playerPersistentSwapMinimum": player["persistentSwapMinimum"],
        "courtCrossSwapQ25": court["crossSwapQ25"],
        "courtWithinContinuityMinimum": court["withinContinuityMinimum"],
        "courtPersistentSwapMinimum": court["persistentSwapMinimum"],
        "crossModalityPersistentMinimum": min(
            player["persistentSwapMinimum"], court["persistentSwapMinimum"]
        ),
        "crossModalityCrossSwapDisagreement": abs(
            player["crossSwapQ25"] - court["crossSwapQ25"]
        ),
        "minimumPersistentContextFraction": context_fraction,
    }
    if tuple(values) != P1_FEATURE_NAMES:
        raise AssertionError("P1 feature order changed")
    return values, {
        "contextWidth": context_width,
        "beforeObservationCount": len(before),
        "afterObservationCount": len(after),
        "contextFraction": context_fraction,
        "missingWithinPairPolicy": "neutral-zero-plus-context-fraction",
        "player": player_diagnostics,
        "court": court_diagnostics,
    }


def _palette_payload(value: np.ndarray) -> list[float]:
    return [round(float(item), 10) for item in value]


def observation_payload(
    observation: SideObservationV2,
    *,
    observation_id: str,
    start: float,
    end: float,
    sample_times: Sequence[float],
    observation_kind: str,
) -> dict[str, Any]:
    """Serialize a compact label-free observation for deterministic reuse."""

    return {
        "observationId": observation_id,
        "kind": observation_kind,
        "start": start,
        "end": end,
        "sampleTimes": [float(value) for value in sample_times],
        "court": {
            "broadNearPalette": _palette_payload(observation.v4.broad.near),
            "broadFarPalette": _palette_payload(observation.v4.broad.far),
            "broadInstability": observation.v4.broad.instability,
            "tightNearPalette": _palette_payload(observation.v4.tight.near),
            "tightFarPalette": _palette_payload(observation.v4.tight.far),
            "tightInstability": observation.v4.tight.instability,
            "globalPalette": _palette_payload(observation.v4.global_palette),
            "foregroundCoverage": observation.v4.foreground_coverage,
            "maximumCameraShift": observation.v4.maximum_camera_shift,
            "minimumAlignmentResponse": observation.v4.minimum_alignment_response,
        },
        "player": {
            "nearPalette": _palette_payload(observation.player.near_palette),
            "farPalette": _palette_payload(observation.player.far_palette),
            "globalPalette": _palette_payload(observation.player.global_palette),
            "paletteInstability": observation.player.palette_instability,
            "proposalCoverage": observation.player.proposal_coverage,
            "proposalCount": observation.player.proposal_count,
            "nearSupport": observation.player.near_support,
            "farSupport": observation.player.far_support,
        },
        "camera": {
            "translationVectors": [list(value) for value in observation.camera.translation_vectors],
            "alignmentResponses": list(observation.camera.alignment_responses),
            "shiftDispersion": observation.camera.shift_dispersion,
            "residualP90": observation.camera.residual_p90,
            "backgroundPalette": _palette_payload(
                observation.camera.background_palette
            ),
            "frameGlobalPalettes": [
                _palette_payload(value)
                for value in observation.camera.frame_global_palettes
            ],
        },
    }

