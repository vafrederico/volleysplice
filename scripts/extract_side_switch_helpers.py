"""Shared label-free helpers for side-switch feature extraction scripts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def crop_roi(frame: np.ndarray, roi: Mapping[str, Any]) -> np.ndarray:
    height, width = frame.shape[:2]
    left = max(0, min(width - 1, round(float(roi.get("x", 0.0)) * width)))
    top = max(0, min(height - 1, round(float(roi.get("y", 0.0)) * height)))
    right = max(
        left + 1,
        min(width, round((float(roi.get("x", 0.0)) + float(roi.get("width", 1.0))) * width)),
    )
    bottom = max(
        top + 1,
        min(height, round((float(roi.get("y", 0.0)) + float(roi.get("height", 1.0))) * height)),
    )
    return np.ascontiguousarray(frame[top:bottom, left:right])


def window_key(recording_id: str, window: Mapping[str, Any]) -> tuple[str, float, float]:
    return recording_id, round(float(window["start"]), 9), round(float(window["end"]), 9)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return result


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if float(np.ptp(left)) <= 1e-15 or float(np.ptp(right)) <= 1e-15:
        return None
    value = float(np.corrcoef(_rankdata(left), _rankdata(right))[0, 1])
    return value if math.isfinite(value) else None
