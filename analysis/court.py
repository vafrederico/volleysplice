from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CourtEstimate:
    confidence: float
    source: str
    roi: tuple[float, float, float, float]
    lines: tuple[tuple[float, float, float, float], ...]

    def as_dict(self) -> dict[str, object]:
        x, y, width, height = self.roi
        return {
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "roi": {
                "x": round(x, 4),
                "y": round(y, 4),
                "width": round(width, 4),
                "height": round(height, 4),
            },
            "lines": [
                {"x1": round(x1, 4), "y1": round(y1, 4), "x2": round(x2, 4), "y2": round(y2, 4)}
                for x1, y1, x2, y2 in self.lines
            ],
        }


def representative_frame(video_path: Path, duration: float, count: int = 21) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path.name}")
    frames: list[np.ndarray] = []
    for timestamp in np.linspace(duration * 0.05, duration * 0.95, count):
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp * 1000))
        ok, frame = capture.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        target_width = min(640, width)
        target_height = max(2, round(height * target_width / width))
        frames.append(cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA))
    capture.release()
    if not frames:
        raise RuntimeError("Could not sample representative video frames")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def estimate_court(frame: np.ndarray) -> CourtEstimate:
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 55, 150)
    edges[: int(height * 0.16), :] = 0
    detected = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=max(35, width // 12),
        minLineLength=max(35, int(width * 0.11)),
        maxLineGap=max(8, int(width * 0.035)),
    )

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    if detected is not None:
        for raw in detected[:, 0]:
            x1, y1, x2, y2 = (int(value) for value in raw)
            length = float(np.hypot(x2 - x1, y2 - y1))
            midpoint_y = (y1 + y2) / 2
            if midpoint_y < height * 0.18 or length < width * 0.11:
                continue
            candidates.append((length, (x1, y1, x2, y2)))

    candidates.sort(reverse=True, key=lambda item: item[0])
    selected = [line for _, line in candidates[:14]]
    fallback = (0.06, 0.2, 0.88, 0.75)
    if len(selected) < 3:
        return CourtEstimate(0.18, "fallback-region", fallback, tuple())

    points = np.array([(x, y) for line in selected for x, y in ((line[0], line[1]), (line[2], line[3]))])
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    coverage_x = (x_max - x_min) / width
    coverage_y = (y_max - y_min) / height
    if coverage_x < 0.42 or coverage_y < 0.18:
        return CourtEstimate(0.24, "fallback-region", fallback, tuple())

    margin_x = width * 0.04
    margin_y = height * 0.06
    left = max(0.02, (x_min - margin_x) / width)
    top = max(0.12, (y_min - margin_y) / height)
    right = min(0.98, (x_max + margin_x) / width)
    bottom = min(0.98, (y_max + margin_y) / height)
    confidence = min(0.9, 0.28 + len(selected) * 0.025 + coverage_x * 0.2 + coverage_y * 0.15)
    normalized_lines = tuple(
        (x1 / width, y1 / height, x2 / width, y2 / height)
        for x1, y1, x2, y2 in selected
    )
    return CourtEstimate(confidence, "detected-lines", (left, top, right - left, bottom - top), normalized_lines)


def save_preview(frame: np.ndarray, estimate: CourtEstimate, destination: Path) -> None:
    preview = frame.copy()
    height, width = preview.shape[:2]
    x, y, roi_width, roi_height = estimate.roi
    start = (round(x * width), round(y * height))
    end = (round((x + roi_width) * width), round((y + roi_height) * height))
    cv2.rectangle(preview, start, end, (52, 255, 223), 3)
    for x1, y1, x2, y2 in estimate.lines:
        cv2.line(
            preview,
            (round(x1 * width), round(y1 * height)),
            (round(x2 * width), round(y2 * height)),
            (53, 92, 255),
            2,
        )
    label = f"court estimate: {round(estimate.confidence * 100)}%"
    cv2.putText(preview, label, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (52, 255, 223), 2)
    if not cv2.imwrite(str(destination), preview):
        raise RuntimeError(f"Could not write {destination.name}")
