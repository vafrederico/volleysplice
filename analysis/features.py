from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import FEATURE_VERSION, FeatureConfig


class VideoError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoMetadata:
    duration: float
    width: int
    height: int
    fps: float
    frame_count: int
    has_audio: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frameCount": self.frame_count,
            "hasAudio": self.has_audio,
        }


@dataclass(frozen=True)
class FeatureSequence:
    times: np.ndarray
    values: np.ndarray
    names: tuple[str, ...]
    metadata: VideoMetadata


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise VideoError(
            "OpenCV is not installed. Run `npm run analysis:setup` before video commands."
        ) from error
    return cv2


def _ffprobe_info(path: Path) -> dict[str, Any]:
    executable = shutil.which("ffprobe")
    if executable is None:
        return {}
    command = [
        executable,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,avg_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
        fps_value = video.get("avg_frame_rate")
        fps: float | None = None
        if isinstance(fps_value, str) and "/" in fps_value:
            numerator, denominator = fps_value.split("/", 1)
            if float(denominator) != 0:
                fps = float(numerator) / float(denominator)
        duration = float(payload.get("format", {}).get("duration", 0.0))
        return {
            "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
            "duration": duration if math.isfinite(duration) and duration > 0 else None,
            "width": video.get("width"),
            "height": video.get("height"),
            "fps": fps if fps is not None and math.isfinite(fps) and fps > 0 else None,
        }
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def probe_video(path: str | Path) -> VideoMetadata:
    cv2 = _cv2()
    video_path = Path(path).expanduser().resolve()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoError(f"cannot open video: {video_path}")
    try:
        raw_width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
        raw_frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        decodable = bool(capture.grab())
    finally:
        capture.release()
    if not decodable:
        raise VideoError(f"video contains no decodable frames: {video_path}")
    fallback = _ffprobe_info(video_path)

    def positive(raw: float, key: str) -> float:
        if math.isfinite(raw) and raw > 0:
            return raw
        candidate = fallback.get(key)
        return float(candidate) if isinstance(candidate, (int, float)) else 0.0

    width = int(round(positive(raw_width, "width")))
    height = int(round(positive(raw_height, "height")))
    fps = positive(raw_fps, "fps")
    frame_count = (
        int(round(raw_frame_count)) if math.isfinite(raw_frame_count) and raw_frame_count > 0 else 0
    )
    if width <= 0 or height <= 0 or fps <= 0:
        raise VideoError(f"video metadata is invalid for {video_path}")
    duration = frame_count / fps if frame_count > 0 else float(fallback.get("duration") or 0.0)
    if not math.isfinite(duration) or duration <= 0:
        raise VideoError(f"video duration is unavailable for {video_path}; normalize it with ffmpeg first")
    return VideoMetadata(
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        has_audio=fallback.get("has_audio"),
    )


def camera_warnings(metadata: VideoMetadata, capture: dict[str, Any] | None = None) -> list[str]:
    capture = capture or {}
    warnings: list[str] = []
    if metadata.width <= metadata.height:
        warnings.append("portrait/square video: use landscape capture")
    if metadata.width < 1280 or metadata.height < 720:
        warnings.append("resolution below the supported 720p floor; 1080p is preferred")
    if metadata.fps < 24:
        warnings.append("frame rate below 24 fps; 30 or 60 fps is preferred")
    if metadata.has_audio is False:
        warnings.append("video has no audio; visual v0 works, but future multimodal cues will be unavailable")
    if capture:
        if capture.get("stationary") is not True:
            warnings.append("camera is not confirmed stationary")
        if capture.get("fullCourtVisible") is not True:
            warnings.append("complete court is not confirmed visible")
        if capture.get("serviceAreasVisible") is not True:
            warnings.append("both service areas are not confirmed visible")
        if capture.get("position") != "centered-behind-endline":
            warnings.append("camera is not confirmed centered behind an end line")
    else:
        warnings.append("camera geometry needs manual confirmation; pixels alone cannot prove full-court visibility")
    return warnings


def _crop_roi(frame: np.ndarray, roi: tuple[float, float, float, float] | None) -> np.ndarray:
    if roi is None:
        return frame
    height, width = frame.shape[:2]
    x, y, roi_width, roi_height = roi
    left = min(width - 1, max(0, int(round(x * width))))
    top = min(height - 1, max(0, int(round(y * height))))
    right = min(width, max(left + 1, int(round((x + roi_width) * width))))
    bottom = min(height, max(top + 1, int(round((y + roi_height) * height))))
    return frame[top:bottom, left:right]


def _grid_means(image: np.ndarray, size: int) -> list[float]:
    result: list[float] = []
    for row in range(size):
        y0, y1 = round(row * image.shape[0] / size), round((row + 1) * image.shape[0] / size)
        for column in range(size):
            x0, x1 = round(column * image.shape[1] / size), round((column + 1) * image.shape[1] / size)
            cell = image[y0:y1, x0:x1]
            result.append(float(np.mean(cell)) if cell.size else 0.0)
    return result


def feature_names(config: FeatureConfig) -> tuple[str, ...]:
    grid_count = config.grid_size * config.grid_size
    static = [
        "luma_mean",
        "luma_std",
        "saturation_mean",
        "saturation_std",
        "edge_density",
        "sharpness",
    ]
    static.extend(f"luma_grid_{index}" for index in range(grid_count))
    dynamic = ["diff_mean", "diff_std", "diff_p90", "diff_active_fraction"]
    dynamic.extend(f"diff_grid_{index}" for index in range(grid_count))
    if config.use_optical_flow:
        dynamic.extend(
            ["flow_mean", "flow_p90", "flow_active_fraction", "flow_median_x", "flow_median_y"]
        )
        dynamic.extend(f"flow_grid_{index}" for index in range(grid_count))
    return tuple(static + dynamic)


def _frame_features(
    frame: np.ndarray,
    previous_gray: np.ndarray | None,
    config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _cv2()
    resized = cv2.resize(
        frame,
        (config.resize_width, config.resize_height),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 60, 140)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    values: list[float] = [
        float(np.mean(gray) / 255.0),
        float(np.std(gray) / 255.0),
        float(np.mean(hsv[:, :, 1]) / 255.0),
        float(np.std(hsv[:, :, 1]) / 255.0),
        float(np.mean(edges > 0)),
        float(min(np.var(laplacian) / 2000.0, 5.0)),
    ]
    values.extend(value / 255.0 for value in _grid_means(gray, config.grid_size))

    if previous_gray is None:
        difference = np.zeros_like(gray, dtype=np.float32)
    else:
        difference = cv2.absdiff(gray, previous_gray).astype(np.float32)
    values.extend(
        [
            float(np.mean(difference) / 255.0),
            float(np.std(difference) / 255.0),
            float(np.percentile(difference, 90) / 255.0),
            float(np.mean(difference >= 18.0)),
        ]
    )
    values.extend(value / 255.0 for value in _grid_means(difference, config.grid_size))

    if config.use_optical_flow:
        if previous_gray is None:
            flow = np.zeros((*gray.shape, 2), dtype=np.float32)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray,
                gray,
                None,
                0.5,
                2,
                13,
                2,
                5,
                1.1,
                0,
            )
        magnitude = np.linalg.norm(flow, axis=2)
        diagonal = float(np.hypot(config.resize_width, config.resize_height))
        values.extend(
            [
                float(np.mean(magnitude) / diagonal),
                float(np.percentile(magnitude, 90) / diagonal),
                float(np.mean(magnitude >= 1.0)),
                float(np.median(flow[:, :, 0]) / config.resize_width),
                float(np.median(flow[:, :, 1]) / config.resize_height),
            ]
        )
        values.extend(value / diagonal for value in _grid_means(magnitude, config.grid_size))

    return np.asarray(values, dtype=np.float32), gray


def extract_features(
    path: str | Path,
    config: FeatureConfig,
    roi: tuple[float, float, float, float] | None = None,
) -> FeatureSequence:
    config.validate()
    cv2 = _cv2()
    video_path = Path(path).expanduser().resolve()
    metadata = probe_video(video_path)
    if config.analysis_fps > metadata.fps + 1e-6:
        raise VideoError(
            f"analysis_fps {config.analysis_fps:g} exceeds source FPS {metadata.fps:g}; "
            "normalize at a higher frame rate or lower analysis_fps"
        )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoError(f"cannot open video: {video_path}")

    times: list[float] = []
    rows: list[np.ndarray] = []
    previous_gray: np.ndarray | None = None
    next_sample = 0.0
    frame_index = 0
    half_frame = 0.5 / metadata.fps
    try:
        while capture.grab():
            timestamp = frame_index / metadata.fps
            frame_index += 1
            if timestamp + half_frame < next_sample:
                continue
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                continue
            cropped = _crop_roi(frame, roi)
            values, previous_gray = _frame_features(cropped, previous_gray, config)
            times.append(timestamp)
            rows.append(values)
            next_sample += 1.0 / config.analysis_fps
    finally:
        capture.release()
    if not rows:
        raise VideoError(f"video yielded no decodable frames: {video_path}")
    names = feature_names(config)
    matrix = np.vstack(rows).astype(np.float32, copy=False)
    if matrix.shape[1] != len(names):
        raise VideoError("internal feature signature mismatch")
    if not np.isfinite(matrix).all() or not np.isfinite(times).all():
        raise VideoError("video feature extraction produced non-finite values")
    return FeatureSequence(
        times=np.asarray(times, dtype=np.float64),
        values=matrix,
        names=names,
        metadata=metadata,
    )


def contextualize(sequence: FeatureSequence, config: FeatureConfig) -> tuple[np.ndarray, tuple[str, ...]]:
    times = sequence.times
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for offset in config.context_offsets_seconds:
        targets = times + offset
        right = np.searchsorted(times, targets, side="left")
        right = np.clip(right, 0, len(times) - 1)
        left = np.clip(right - 1, 0, len(times) - 1)
        choose_left = np.abs(times[left] - targets) <= np.abs(times[right] - targets)
        nearest = np.where(choose_left, left, right)
        nearest = np.where(targets <= times[0], 0, nearest)
        nearest = np.where(targets >= times[-1], len(times) - 1, nearest)
        blocks.append(sequence.values[nearest])
        prefix = f"t{offset:+g}s/"
        names.extend(prefix + name for name in sequence.names)
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False), tuple(names)


def _valid_cached_sequence(sequence: FeatureSequence, config: FeatureConfig) -> bool:
    expected_names = feature_names(config)
    metadata = sequence.metadata
    return bool(
        sequence.names == expected_names
        and sequence.times.ndim == 1
        and len(sequence.times) > 0
        and sequence.values.shape == (len(sequence.times), len(expected_names))
        and np.isfinite(sequence.times).all()
        and np.isfinite(sequence.values).all()
        and (len(sequence.times) == 1 or np.all(np.diff(sequence.times) > 0))
        and math.isfinite(metadata.duration)
        and metadata.duration > 0
        and math.isfinite(metadata.fps)
        and metadata.fps > 0
        and metadata.width > 0
        and metadata.height > 0
        and metadata.frame_count >= 0
        and (metadata.has_audio is None or isinstance(metadata.has_audio, bool))
    )


def _cache_key(
    video: Path,
    config: FeatureConfig,
    roi: tuple[float, float, float, float] | None,
    content_sha256: str,
) -> str:
    payload = {
        "featureVersion": FEATURE_VERSION,
        "path": str(video),
        "contentSha256": content_sha256,
        "config": config.to_dict(),
        "roi": roi,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def cached_features(
    recording_id: str,
    video: str | Path,
    config: FeatureConfig,
    roi: tuple[float, float, float, float] | None,
    cache_dir: str | Path,
    *,
    content_sha256: str | None = None,
) -> FeatureSequence:
    video_path = Path(video).expanduser().resolve()
    destination = Path(cache_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in recording_id)
    digest = content_sha256 or _sha256_file(video_path)
    cache_path = destination / f"{safe_id}-{_cache_key(video_path, config, roi, digest)}.npz"
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                metadata = VideoMetadata(**json.loads(str(cached["metadata_json"].item())))
                cached_sequence = FeatureSequence(
                    times=cached["times"].astype(np.float64, copy=False),
                    values=cached["values"].astype(np.float32, copy=False),
                    names=tuple(str(item) for item in cached["names"]),
                    metadata=metadata,
                )
            if _valid_cached_sequence(cached_sequence, config):
                return cached_sequence
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    sequence = extract_features(video_path, config, roi)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{safe_id}-",
        suffix=".npz",
        dir=destination,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary_path,
            times=sequence.times,
            values=sequence.values,
            names=np.asarray(sequence.names),
            metadata_json=json.dumps(asdict(sequence.metadata), sort_keys=True, allow_nan=False),
        )
        temporary_path.replace(cache_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return sequence


def write_preview(
    video: str | Path,
    destination: str | Path,
    roi: tuple[float, float, float, float] | None = None,
) -> None:
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(Path(video).expanduser().resolve()))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise VideoError("could not decode a preview frame")
    if roi is not None:
        height, width = frame.shape[:2]
        x, y, roi_width, roi_height = roi
        left, top = int(round(x * width)), int(round(y * height))
        right, bottom = int(round((x + roi_width) * width)), int(round((y + roi_height) * height))
        cv2.rectangle(frame, (left, top), (right, bottom), (50, 220, 255), max(2, width // 500))
    if not cv2.imwrite(str(destination), frame):
        raise VideoError(f"could not write preview image: {destination}")
