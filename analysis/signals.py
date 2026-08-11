from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .court import CourtEstimate


def robust_normalize(values: np.ndarray, minimum_spread: float = 1e-6) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    low = float(np.quantile(values, 0.2))
    high = float(np.quantile(values, 0.9))
    spread = high - low
    if spread < minimum_spread:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - low) / spread, 0, 1)


def motion_signal(video_path: Path, estimate: CourtEstimate, analysis_fps: float) -> tuple[np.ndarray, np.ndarray, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path.name}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30)
    stride = max(1, round(source_fps / analysis_fps))
    effective_fps = source_fps / stride
    x, y, roi_width, roi_height = estimate.roi
    times: list[float] = []
    values: list[float] = []
    shifts: list[float] = []
    previous: np.ndarray | None = None
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride:
            frame_index += 1
            continue
        frame_height, frame_width = frame.shape[:2]
        left = min(frame_width - 2, max(0, round(x * frame_width)))
        top = min(frame_height - 2, max(0, round(y * frame_height)))
        right = min(frame_width, max(left + 2, round((x + roi_width) * frame_width)))
        bottom = min(frame_height, max(top + 2, round((y + roi_height) * frame_height)))
        crop = frame[top:bottom, left:right]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        target_width = min(320, gray.shape[1])
        target_height = max(2, round(gray.shape[0] * target_width / gray.shape[1]))
        gray = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        motion = 0.0
        shift_size = 0.0
        if previous is not None:
            (shift_x, shift_y), response = cv2.phaseCorrelate(
                previous.astype(np.float32),
                gray.astype(np.float32),
            )
            max_shift = max(3.0, target_width * 0.04)
            if response > 0.08 and abs(shift_x) <= max_shift and abs(shift_y) <= max_shift:
                matrix = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
                aligned = cv2.warpAffine(gray, matrix, (target_width, target_height), borderMode=cv2.BORDER_REFLECT)
                shift_size = float(np.hypot(shift_x, shift_y) / target_width)
            else:
                aligned = gray
            difference = cv2.absdiff(previous, aligned)
            mean_change = float(np.mean(difference) / 255)
            changed_fraction = float(np.mean(difference > 14))
            motion = mean_change * 0.4 + changed_fraction * 0.6

        times.append(frame_index / source_fps)
        values.append(motion)
        shifts.append(shift_size)
        previous = gray
        frame_index += 1

    capture.release()
    stability = 1.0 - min(1.0, float(np.quantile(shifts, 0.9) * 16)) if shifts else 1.0
    return np.asarray(times), robust_normalize(np.asarray(values), minimum_spread=0.004), stability


def audio_signal(video_path: Path, duration: float, sample_period: float, has_audio: bool) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(0, duration + sample_period, sample_period)
    if not has_audio:
        return times, np.zeros_like(times)
    sample_rate = 8000
    samples_per_window = max(1, round(sample_rate * sample_period))
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "f32le", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("Could not read FFmpeg audio output")
    rms: list[float] = []
    byte_count = samples_per_window * 4
    while True:
        chunk = process.stdout.read(byte_count)
        if not chunk:
            break
        usable = len(chunk) - len(chunk) % 4
        samples = np.frombuffer(chunk[:usable], dtype="<f4")
        rms.append(float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0)
    _, stderr = process.communicate()
    if process.returncode:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or "FFmpeg audio extraction failed")
    audio_times = np.arange(len(rms), dtype=np.float64) * sample_period
    return audio_times, robust_normalize(np.asarray(rms))


def combine_signals(
    motion_times: np.ndarray,
    motion: np.ndarray,
    audio_times: np.ndarray,
    audio: np.ndarray,
) -> list[dict[str, float]]:
    if motion_times.size == 0:
        return []
    aligned_audio = np.interp(motion_times, audio_times, audio, left=0, right=0) if audio_times.size else np.zeros_like(motion)
    window = max(1, round(1 / max(0.01, float(np.median(np.diff(motion_times))) if motion_times.size > 1 else 1)))
    kernel = np.ones(window) / window
    smooth_motion = np.convolve(motion, kernel, mode="same")
    activity = np.clip(smooth_motion * 0.82 + aligned_audio * 0.18, 0, 1)
    return [
        {
            "time": round(float(timestamp), 3),
            "motion": round(float(motion_value), 4),
            "audio": round(float(audio_value), 4),
            "activity": round(float(activity_value), 4),
        }
        for timestamp, motion_value, audio_value, activity_value in zip(
            motion_times, smooth_motion, aligned_audio, activity, strict=True
        )
    ]
