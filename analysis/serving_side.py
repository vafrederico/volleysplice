"""Interpretable serving-side evidence from existing rally labels.

The current label schema has rally start anchors but no structured serving-side
field.  This module therefore keeps two concerns separate:

* ``infer_note_serving_side`` extracts a deliberately conservative, weak target
  from existing rally notes for experiment reporting only.
* the frame helpers measure which vertical court half has the strongest visual
  change around the labeled rally start.

The near side is the lower/foreground half of the supplied ROI and the far side
is the upper/background half.  This is a camera-space proxy, not a court
geometry solution.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from analysis.side_switch_appearance import (
    PersonDetection,
    detect_people,
    hellinger_distance,
    palette_vector,
)


NEAR_SIDE = "near"
FAR_SIDE = "far"
SIDE_VALUES = (NEAR_SIDE, FAR_SIDE)

_NEAR_PATTERNS = (
    re.compile(r"\bnear[- ]side\s+(?:serve|server|contact|toss|baseline|action|sequence)\b"),
    re.compile(r"\bnear[- ]baseline\b"),
    re.compile(r"\bcamera[- ]side\s+baseline\b"),
    re.compile(r"\bforeground\s+(?:serve|server|toss|contact|action)\b"),
)
_FAR_PATTERNS = (
    re.compile(r"\bfar[- ]side\s+(?:serve|server|contact|toss|baseline|action|onset)\b"),
    re.compile(r"\bfar[- ]baseline\b"),
    re.compile(r"\bdistant\s+(?:serve|server|contact|onset)\b"),
    re.compile(r"\bserve\s+is\s+distant\b"),
    re.compile(r"\bserver\s+is\s+distant\b"),
    re.compile(r"\bserve\s+originates\s+at\s+the\s+far\s+side\b"),
    re.compile(r"\bserve\s+is\s+at\s+the\s+far\s+baseline\b"),
)


@dataclass(frozen=True)
class ServingSideCue:
    """A weak note-derived serving-side cue."""

    side: str | None
    strength: str
    reason: str
    matches: tuple[str, ...]


@dataclass(frozen=True)
class SideOccupancy:
    """HOG proposal occupancy assigned by detection footpoint."""

    near_area_fraction: float
    far_area_fraction: float
    near_count: int
    far_count: int
    near_score: float | None
    far_score: float | None


def _cue_strength(note: str) -> str:
    uncertain = re.search(
        r"\b(?:inferred|estimated|back[- ]timed|off[- ]screen|small|distant|uncertain|poorly resolved)\b",
        note,
    )
    if uncertain:
        return "weak"
    if re.search(r"\b(?:visible|clear|strong|confirmed)\b", note):
        return "strong"
    return "medium"


def infer_note_serving_side(note: str | None) -> ServingSideCue:
    """Return a conservative side cue from an existing rally note.

    Notes that mention both sides, or explicitly say ``far side or outside``,
    are intentionally treated as unknown.  Phrases such as ``far player`` and
    ``foreground ball`` do not match the serving-side patterns.
    """

    if not isinstance(note, str) or not note.strip():
        return ServingSideCue(None, "none", "no serving-side phrase", ())
    normalized = " ".join(note.casefold().split())
    near_matches = tuple(
        match.group(0) for pattern in _NEAR_PATTERNS if (match := pattern.search(normalized))
    )
    far_matches = tuple(
        match.group(0) for pattern in _FAR_PATTERNS if (match := pattern.search(normalized))
    )
    if "far side or outside" in normalized or (near_matches and far_matches):
        return ServingSideCue(
            None,
            "ambiguous",
            "serving-side phrases conflict or include an outside view",
            near_matches + far_matches,
        )
    if near_matches:
        return ServingSideCue(
            NEAR_SIDE,
            _cue_strength(normalized),
            "near/foreground serving-side phrase in rally note",
            near_matches,
        )
    if far_matches:
        return ServingSideCue(
            FAR_SIDE,
            _cue_strength(normalized),
            "far/distant serving-side phrase in rally note",
            far_matches,
        )
    return ServingSideCue(None, "none", "no unambiguous serving-side phrase", ())


def side_margin(near_value: float | None, far_value: float | None) -> float | None:
    """Return a signed near-minus-far margin in [-1, 1]."""

    if near_value is None or far_value is None:
        return None
    if not math.isfinite(near_value) or not math.isfinite(far_value):
        return None
    denominator = abs(near_value) + abs(far_value)
    if denominator <= 1e-9:
        return 0.0
    return float((near_value - far_value) / denominator)


def crop_roi(
    frame: np.ndarray,
    roi: tuple[float, float, float, float] | None,
) -> np.ndarray:
    """Crop a normalized ROI from a BGR frame, falling back to the full frame."""

    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR HxWx3 array")
    height, width = frame.shape[:2]
    if roi is None:
        return frame
    x, y, roi_width, roi_height = roi
    if not all(math.isfinite(value) for value in roi):
        raise ValueError("ROI must contain finite values")
    left = max(0, min(width - 1, round(x * width)))
    top = max(0, min(height - 1, round(y * height)))
    right = max(left + 1, min(width, round((x + roi_width) * width)))
    bottom = max(top + 1, min(height, round((y + roi_height) * height)))
    cropped = frame[top:bottom, left:right]
    if cropped.shape[0] < 2 or cropped.shape[1] < 2:
        raise ValueError("ROI crop is too small")
    return cropped


def _split_bounds(height: int, split_fraction: float) -> int:
    if not math.isfinite(split_fraction) or not 0.1 < split_fraction < 0.9:
        raise ValueError("split fraction must be between 0.1 and 0.9")
    return max(1, min(height - 1, round(height * split_fraction)))


def split_sides(
    frame: np.ndarray,
    split_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(far, near)`` vertical crops from an ROI frame."""

    split = _split_bounds(frame.shape[0], split_fraction)
    return frame[:split], frame[split:]


def _side_regions(
    frame: np.ndarray,
    split_fraction: float,
    baseline_fraction: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    split = _split_bounds(frame.shape[0], split_fraction)
    if baseline_fraction is None:
        return frame[:split], frame[split:]
    if not math.isfinite(baseline_fraction) or not 0.05 < baseline_fraction < 0.9:
        raise ValueError("baseline fraction must be between 0.05 and 0.9")
    far_end = max(1, round(split * baseline_fraction))
    near_start = min(frame.shape[0] - 1, split + round((frame.shape[0] - split) * (1 - baseline_fraction)))
    return frame[:far_end], frame[near_start:]


def pixel_motion(
    reference: np.ndarray,
    frames: Sequence[np.ndarray],
    split_fraction: float = 0.5,
    baseline_fraction: float | None = None,
) -> tuple[float, float, float | None]:
    """Measure mean grayscale change in far and near halves or edge bands."""

    if not frames:
        return 0.0, 0.0, None
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    far_reference, near_reference = _side_regions(
        reference,
        split_fraction,
        baseline_fraction,
    )
    near_values: list[float] = []
    far_values: list[float] = []
    for frame in frames:
        if frame.shape != reference.shape:
            raise ValueError("motion frames must have the same shape")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        difference = cv2.absdiff(reference_gray, gray).astype(np.float32) / 255.0
        far_values.append(float(np.mean(difference[: far_reference.shape[0]])))
        near_values.append(float(np.mean(difference[-near_reference.shape[0] :])))
    near = float(np.mean(near_values))
    far = float(np.mean(far_values))
    return near, far, side_margin(near, far)


def palette_change(
    reference: np.ndarray,
    frames: Sequence[np.ndarray],
    split_fraction: float = 0.5,
    baseline_fraction: float | None = None,
) -> tuple[float, float, float | None]:
    """Measure HSV palette change in far and near halves or edge bands."""

    if not frames:
        return 0.0, 0.0, None
    reference_far, reference_near = _side_regions(
        reference,
        split_fraction,
        baseline_fraction,
    )
    reference_vectors = (
        palette_vector(reference_far),
        palette_vector(reference_near),
    )
    far_values: list[float] = []
    near_values: list[float] = []
    for frame in frames:
        frame_far, frame_near = _side_regions(frame, split_fraction, baseline_fraction)
        far_values.append(hellinger_distance(reference_vectors[0], palette_vector(frame_far)))
        near_values.append(hellinger_distance(reference_vectors[1], palette_vector(frame_near)))
    near = float(np.mean(near_values))
    far = float(np.mean(far_values))
    return near, far, side_margin(near, far)


def _resize_for_hog(frame: np.ndarray, maximum_width: int = 960) -> np.ndarray:
    if frame.shape[1] <= maximum_width:
        return frame
    scale = maximum_width / frame.shape[1]
    return cv2.resize(
        frame,
        (maximum_width, max(128, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _side_detections(
    detections: Iterable[PersonDetection],
    frame_height: int,
    frame_width: int,
    split_fraction: float,
    baseline_fraction: float | None,
) -> SideOccupancy:
    split = _split_bounds(frame_height, split_fraction)
    far_limit = 0
    near_limit = frame_height
    if baseline_fraction is not None:
        if not math.isfinite(baseline_fraction) or not 0.05 < baseline_fraction < 0.9:
            raise ValueError("baseline fraction must be between 0.05 and 0.9")
        far_limit = max(1, round(split * baseline_fraction))
        near_limit = min(
            frame_height - 1,
            split + round((frame_height - split) * (1 - baseline_fraction)),
        )
    near_area = 0.0
    far_area = 0.0
    near_scores: list[float] = []
    far_scores: list[float] = []
    near_count = 0
    far_count = 0
    denominator = max(1, frame_height * frame_width)
    for detection in detections:
        footpoint = detection.y + detection.height
        area_fraction = detection.area / denominator
        if footpoint <= split and footpoint <= far_limit:
            far_area += area_fraction
            far_scores.append(detection.score)
            far_count += 1
        elif footpoint > split and footpoint >= near_limit:
            near_area += area_fraction
            near_scores.append(detection.score)
            near_count += 1
    return SideOccupancy(
        near_area_fraction=float(near_area),
        far_area_fraction=float(far_area),
        near_count=near_count,
        far_count=far_count,
        near_score=(float(np.mean(near_scores)) if near_scores else None),
        far_score=(float(np.mean(far_scores)) if far_scores else None),
    )


def hog_occupancy(
    frame: np.ndarray,
    hog: cv2.HOGDescriptor,
    split_fraction: float = 0.5,
    baseline_fraction: float | None = None,
) -> SideOccupancy:
    """Run the existing fixed HOG detector and split by footpoint."""

    resized = _resize_for_hog(frame)
    detections = detect_people(resized, hog)
    return _side_detections(
        detections,
        resized.shape[0],
        resized.shape[1],
        split_fraction,
        baseline_fraction,
    )


def occupancy_change_margin(
    before: SideOccupancy,
    after: SideOccupancy,
    field: str,
) -> float | None:
    """Compare absolute pre/action occupancy changes by side."""

    if field == "area":
        near_change = abs(after.near_area_fraction - before.near_area_fraction)
        far_change = abs(after.far_area_fraction - before.far_area_fraction)
    elif field == "count":
        near_change = abs(after.near_count - before.near_count)
        far_change = abs(after.far_count - before.far_count)
    else:
        raise ValueError(f"unsupported occupancy field: {field}")
    return side_margin(float(near_change), float(far_change))
