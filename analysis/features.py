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

from .config import (
    NOISE_NORMALIZED_AUDIO_FEATURE_SET,
    FeatureConfig,
    feature_version_for_config,
)


OPENCV_VIDEO_DECODER = "opencv-ffmpeg-v1"
NVDEC_VIDEO_DECODER = "ffmpeg-nvdec-sampled-v1"
VIDEO_DECODERS = (OPENCV_VIDEO_DECODER, NVDEC_VIDEO_DECODER)


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
        "stream=codec_type,width,height,avg_frame_rate,nb_frames,duration:format=duration",
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
        frame_count_value = video.get("nb_frames")
        frame_count = (
            int(frame_count_value)
            if isinstance(frame_count_value, (int, str))
            and str(frame_count_value).isdigit()
            else None
        )
        return {
            "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
            "duration": duration if math.isfinite(duration) and duration > 0 else None,
            "width": video.get("width"),
            "height": video.get("height"),
            "fps": fps if fps is not None and math.isfinite(fps) and fps > 0 else None,
            "frame_count": frame_count if frame_count is not None and frame_count > 0 else None,
        }
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def probe_video(
    path: str | Path,
    *,
    video_decoder: str = OPENCV_VIDEO_DECODER,
) -> VideoMetadata:
    if video_decoder not in VIDEO_DECODERS:
        raise VideoError(f"unsupported video decoder: {video_decoder}")
    video_path = Path(path).expanduser().resolve()
    fallback = _ffprobe_info(video_path)
    if video_decoder == NVDEC_VIDEO_DECODER:
        width = int(fallback.get("width") or 0)
        height = int(fallback.get("height") or 0)
        fps = float(fallback.get("fps") or 0.0)
        frame_count = int(fallback.get("frame_count") or 0)
        duration = (
            frame_count / fps
            if frame_count > 0 and fps > 0
            else float(fallback.get("duration") or 0.0)
        )
        if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0 or duration <= 0:
            raise VideoError(f"ffprobe metadata is incomplete for NVDEC source: {video_path}")
        return VideoMetadata(
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            has_audio=fallback.get("has_audio"),
        )

    cv2 = _cv2()
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
        warnings.append(
            "video has no audio; audiovisual channels will be marked unavailable and zero-imputed"
        )
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


ABSOLUTE_FEATURE_NAMES = frozenset(
    {
        "audio_available",
        "focus_quality",
        "blur_probability",
        "occlusion_fraction",
        "visibility_quality",
        "camera_shift_response",
    }
)


def _frame_feature_names(config: FeatureConfig) -> tuple[str, ...]:
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
    if config.use_advanced_visual:
        dynamic.extend(
            [
                "focus_quality",
                "blur_probability",
                "dark_fraction",
                "bright_fraction",
                "low_texture_fraction",
                "occlusion_fraction",
                "visibility_quality",
                "camera_shift_x",
                "camera_shift_y",
                "camera_shift_magnitude",
                "camera_shift_response",
            ]
        )
    if config.use_optical_flow:
        dynamic.extend(
            ["flow_mean", "flow_p90", "flow_active_fraction", "flow_median_x", "flow_median_y"]
        )
        dynamic.extend(f"flow_grid_{index}" for index in range(grid_count))
        if config.use_advanced_visual:
            dynamic.extend(
                [
                    "player_motion_mean",
                    "player_motion_p90",
                    "player_motion_active_fraction",
                    "player_motion_active_zone_fraction",
                    "player_motion_spatial_entropy",
                    "player_motion_centroid_x",
                    "player_motion_centroid_y",
                    "player_motion_spread_x",
                    "player_motion_spread_y",
                    "player_motion_coherence",
                    "quality_gated_player_motion",
                ]
            )
            dynamic.extend(
                f"player_motion_grid_{index}" for index in range(grid_count)
            )
    return tuple(static + dynamic)


def _temporal_visual_feature_names(config: FeatureConfig) -> tuple[str, ...]:
    if not config.use_advanced_visual or not config.use_optical_flow:
        return ()
    return (
        "player_motion_onset",
        "player_motion_collapse",
        "synchronized_stand_down",
        "receiving_formation_change_proxy",
    )


def _audio_feature_names(config: FeatureConfig) -> tuple[str, ...]:
    if not config.use_audio:
        return ()
    legacy = (
        "audio_available",
        "audio_rms",
        "audio_peak",
        "audio_peak_to_rms",
        "audio_noise_floor",
        "audio_snr",
        "audio_spectral_flux",
        "audio_rms_novelty",
        "audio_onset_strength",
        "audio_contact_like_transient",
        "audio_onset_cadence",
        "audio_cadence_collapse",
        "audio_seconds_since_transient",
    )
    if config.audio_feature_set != NOISE_NORMALIZED_AUDIO_FEATURE_SET:
        return legacy
    normalized = (
        "audio_noise_removed_broadband",
        "audio_noise_normalized_flux",
    )
    bands = (
        "80_250",
        "250_500",
        "500_1000",
        "1000_2000",
        "2000_4000",
        "4000_7800",
    )
    frequency = tuple(
        name
        for band in bands
        for name in (
            f"audio_band_{band}_snr",
            f"audio_band_{band}_snr_flux",
        )
    )
    return (*legacy, *normalized, *frequency)


def feature_names(config: FeatureConfig) -> tuple[str, ...]:
    return (
        *_frame_feature_names(config),
        *_temporal_visual_feature_names(config),
        *_audio_feature_names(config),
    )


def _weighted_motion_geometry(magnitude: np.ndarray) -> tuple[float, float, float, float]:
    weights = np.where(magnitude >= 0.5, magnitude, 0.0).astype(np.float64, copy=False)
    total = float(np.sum(weights))
    if total <= 1e-9:
        return 0.5, 0.5, 0.0, 0.0
    height, width = magnitude.shape
    x_coordinates = (np.arange(width, dtype=np.float64) + 0.5) / width
    y_coordinates = (np.arange(height, dtype=np.float64) + 0.5) / height
    x_weights = np.sum(weights, axis=0)
    y_weights = np.sum(weights, axis=1)
    centroid_x = float(x_weights @ x_coordinates / total)
    centroid_y = float(y_weights @ y_coordinates / total)
    spread_x = float(np.sqrt(x_weights @ np.square(x_coordinates - centroid_x) / total))
    spread_y = float(np.sqrt(y_weights @ np.square(y_coordinates - centroid_y) / total))
    return centroid_x, centroid_y, spread_x, spread_y


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
    laplacian_variance = float(np.var(laplacian))
    luma_std = float(np.std(gray) / 255.0)
    edge_density = float(np.mean(edges > 0))
    values: list[float] = [
        float(np.mean(gray) / 255.0),
        luma_std,
        float(np.mean(hsv[:, :, 1]) / 255.0),
        float(np.std(hsv[:, :, 1]) / 255.0),
        edge_density,
        float(min(laplacian_variance / 2000.0, 5.0)),
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

    visibility_quality = 1.0
    camera_shift_magnitude = 0.0
    if config.use_advanced_visual:
        focus_quality = laplacian_variance / (laplacian_variance + 100.0)
        blur_probability = 1.0 - focus_quality
        dark_fraction = float(np.mean(gray <= 12))
        bright_fraction = float(np.mean(gray >= 243))
        gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.hypot(gradient_x, gradient_y)
        low_texture_fraction = float(np.mean(gradient < 8.0))
        occluded_cells = 0
        for row in range(config.grid_size):
            y0 = round(row * gray.shape[0] / config.grid_size)
            y1 = round((row + 1) * gray.shape[0] / config.grid_size)
            for column in range(config.grid_size):
                x0 = round(column * gray.shape[1] / config.grid_size)
                x1 = round((column + 1) * gray.shape[1] / config.grid_size)
                cell = gray[y0:y1, x0:x1]
                if cell.size and (
                    (float(np.mean(cell)) <= 16.0 or float(np.mean(cell)) >= 239.0)
                    and float(np.std(cell)) <= 8.0
                ):
                    occluded_cells += 1
        occlusion_fraction = occluded_cells / (config.grid_size * config.grid_size)
        exposure_quality = max(0.0, 1.0 - min(1.0, dark_fraction + bright_fraction))
        contrast_quality = min(1.0, luma_std / 0.12)
        visibility_quality = float(
            np.sqrt(max(0.0, focus_quality * exposure_quality * contrast_quality))
            * (1.0 - occlusion_fraction)
        )
        if previous_gray is None:
            shift_x = shift_y = response = 0.0
        else:
            try:
                shift, response = cv2.phaseCorrelate(
                    previous_gray.astype(np.float32), gray.astype(np.float32)
                )
                shift_x, shift_y = float(shift[0]), float(shift[1])
                response = float(np.clip(response, 0.0, 1.0))
            except cv2.error:
                shift_x = shift_y = response = 0.0
        diagonal = float(np.hypot(config.resize_width, config.resize_height))
        camera_shift_magnitude = float(np.hypot(shift_x, shift_y) / diagonal)
        values.extend(
            [
                focus_quality,
                blur_probability,
                dark_fraction,
                bright_fraction,
                low_texture_fraction,
                occlusion_fraction,
                visibility_quality,
                shift_x / config.resize_width,
                shift_y / config.resize_height,
                camera_shift_magnitude,
                response,
            ]
        )

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
        if config.use_advanced_visual:
            median_x = float(np.median(flow[:, :, 0]))
            median_y = float(np.median(flow[:, :, 1]))
            residual = flow - np.asarray([median_x, median_y], dtype=np.float32)
            residual_magnitude = np.linalg.norm(residual, axis=2)
            residual_grid_pixels = np.asarray(
                _grid_means(residual_magnitude, config.grid_size), dtype=np.float64
            )
            residual_grid = residual_grid_pixels / diagonal
            grid_total = float(np.sum(residual_grid_pixels))
            if grid_total > 1e-9:
                distribution = residual_grid_pixels / grid_total
                positive = distribution > 0
                entropy = float(
                    -np.sum(distribution[positive] * np.log(distribution[positive]))
                    / math.log(len(distribution))
                ) if len(distribution) > 1 else 0.0
            else:
                entropy = 0.0
            centroid_x, centroid_y, spread_x, spread_y = _weighted_motion_geometry(
                residual_magnitude
            )
            active = residual_magnitude >= 1.0
            if np.any(active):
                active_vectors = residual[active]
                coherence = float(
                    np.linalg.norm(np.mean(active_vectors, axis=0))
                    / max(float(np.mean(np.linalg.norm(active_vectors, axis=1))), 1e-6)
                )
            else:
                coherence = 0.0
            camera_gate = max(0.0, 1.0 - min(1.0, camera_shift_magnitude / 0.03))
            residual_mean = float(np.mean(residual_magnitude) / diagonal)
            values.extend(
                [
                    residual_mean,
                    float(np.percentile(residual_magnitude, 90) / diagonal),
                    float(np.mean(active)),
                    float(np.mean(residual_grid_pixels >= 0.75)),
                    entropy,
                    centroid_x,
                    centroid_y,
                    spread_x,
                    spread_y,
                    coherence,
                    residual_mean * visibility_quality * camera_gate,
                ]
            )
            values.extend(float(value) for value in residual_grid)

    return np.asarray(values, dtype=np.float32), gray


def _rolling_mean(values: np.ndarray, window_samples: int, *, future: bool) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("rolling input must be one-dimensional")
    if len(values) == 0:
        return values.astype(np.float32, copy=True)
    window_samples = max(1, int(window_samples))
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    indexes = np.arange(len(values))
    if future:
        starts = indexes
        ends = np.minimum(len(values), indexes + window_samples)
    else:
        starts = np.maximum(0, indexes - window_samples + 1)
        ends = indexes + 1
    return ((cumulative[ends] - cumulative[starts]) / (ends - starts)).astype(np.float32)


def _temporal_visual_features(
    matrix: np.ndarray,
    names: tuple[str, ...],
    config: FeatureConfig,
) -> np.ndarray:
    temporal_names = _temporal_visual_feature_names(config)
    if not temporal_names:
        return np.empty((len(matrix), 0), dtype=np.float32)
    indexes = {name: index for index, name in enumerate(names)}
    player_motion = matrix[:, indexes["player_motion_mean"]]
    active_zones = matrix[:, indexes["player_motion_active_zone_fraction"]]
    window = max(1, round(config.analysis_fps))
    short_window = max(1, round(0.5 * config.analysis_fps))
    past_motion = _rolling_mean(player_motion, window, future=False)
    future_motion = _rolling_mean(player_motion, short_window, future=True)
    past_zones = _rolling_mean(active_zones, window, future=False)
    future_zones = _rolling_mean(active_zones, short_window, future=True)
    onset = np.maximum(future_motion - past_motion, 0.0)
    collapse = np.maximum(past_motion - future_motion, 0.0)
    synchronized = collapse * np.maximum(past_zones - future_zones, 0.0)

    geometry = []
    for name in (
        "player_motion_centroid_x",
        "player_motion_centroid_y",
        "player_motion_spread_x",
        "player_motion_spread_y",
    ):
        values = matrix[:, indexes[name]]
        geometry.append(
            _rolling_mean(values, window, future=True)
            - _rolling_mean(values, window, future=False)
        )
    formation_change = np.sqrt(sum(np.square(value) for value in geometry))
    activity_gate = np.minimum(1.0, (past_motion + future_motion) / 0.015)
    formation_change *= activity_gate
    return np.column_stack((onset, collapse, synchronized, formation_change)).astype(
        np.float32, copy=False
    )


def _decode_audio_samples(
    video_path: Path,
    metadata: VideoMetadata,
    sample_rate: int,
) -> tuple[np.ndarray, bool]:
    if metadata.has_audio is False:
        return np.empty(0, dtype=np.float32), False
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise VideoError("FFmpeg is required when audio features are enabled")
    command = [
        executable,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    timeout = max(120.0, metadata.duration * 0.5)
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        raise VideoError(f"could not decode audio from {video_path}: {error}") from error
    if result.returncode != 0:
        if metadata.has_audio is None:
            return np.empty(0, dtype=np.float32), False
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise VideoError(f"could not decode audio from {video_path}: {detail or 'FFmpeg failed'}")
    if not result.stdout:
        return np.empty(0, dtype=np.float32), False
    samples = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32)
    samples *= np.float32(1.0 / 32768.0)
    return samples, True


def _sampled_frame_indexes(metadata: VideoMetadata, analysis_fps: float) -> list[int]:
    """Reproduce the historical OpenCV nearest-frame sampling contract."""

    indexes: list[int] = []
    next_sample = 0.0
    half_frame = 0.5 / metadata.fps
    for frame_index in range(metadata.frame_count):
        timestamp = frame_index / metadata.fps
        if timestamp + half_frame < next_sample:
            continue
        indexes.append(frame_index)
        next_sample += 1.0 / analysis_fps
    return indexes


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _nvdec_filter(metadata: VideoMetadata, analysis_fps: float) -> str:
    source_fps = format(metadata.fps, ".17g")
    target_fps = format(analysis_fps, ".17g")
    selection = (
        "gt(floor((n+0.5)*"
        f"{target_fps}/{source_fps})*{source_fps}/{target_fps},n-0.5)"
    )
    # OpenCV's historical FFmpeg capture path converts decoded YUV with the
    # default BT.601 matrix. Preserve that feature contract after NVDEC hands
    # the CUDA surface back to system FFmpeg.
    color = (
        "scale=in_color_matrix=bt601:out_color_matrix=bt601:"
        "in_range=full:out_range=full"
    )
    return f"select='{selection}',hwdownload,format=nv12,{color},format=bgr24"


def nvdec_available(path: str | Path) -> tuple[bool, str]:
    """Probe the system FFmpeg CUDA decoder using one real source frame."""

    executable = shutil.which("ffmpeg")
    if executable is None:
        return False, "ffmpeg is unavailable"
    video_path = Path(path).expanduser().resolve()
    command = [
        executable,
        "-nostdin",
        "-v",
        "error",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        "1",
        "-vf",
        "hwdownload,format=nv12",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "nv12",
        "-y",
        os.devnull,
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return False, str(error)
    if result.returncode == 0:
        return True, "FFmpeg CUDA/NVDEC probe succeeded"
    detail = result.stderr.decode("utf-8", errors="replace").strip()
    return False, detail or f"FFmpeg exited with status {result.returncode}"


def _nvdec_frames(
    video_path: Path,
    metadata: VideoMetadata,
    analysis_fps: float,
) -> Any:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise VideoError("FFmpeg is required for NVDEC video extraction")
    indexes = _sampled_frame_indexes(metadata, analysis_fps)
    frame_bytes = metadata.width * metadata.height * 3
    command = [
        executable,
        "-nostdin",
        "-v",
        "error",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        _nvdec_filter(metadata, analysis_fps),
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise VideoError("could not open FFmpeg NVDEC pipes")
    try:
        for frame_index in indexes:
            raw = _read_exact(process.stdout, frame_bytes)
            if len(raw) != frame_bytes:
                detail = process.stderr.read().decode("utf-8", errors="replace").strip()
                process.wait()
                raise VideoError(
                    "FFmpeg NVDEC ended before the expected sampled frame count"
                    + (f": {detail}" if detail else "")
                )
            yield frame_index, np.frombuffer(raw, dtype=np.uint8).reshape(
                metadata.height, metadata.width, 3
            )
        if process.stdout.read(1):
            raise VideoError("FFmpeg NVDEC produced more sampled frames than expected")
        detail = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if return_code != 0:
            raise VideoError(
                f"FFmpeg NVDEC failed with status {return_code}"
                + (f": {detail}" if detail else "")
            )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def _rank_vector(values: np.ndarray) -> np.ndarray:
    return percentile_rank_values(values.reshape(-1, 1))[:, 0]


def _audio_features_from_samples(
    samples: np.ndarray,
    times: np.ndarray,
    config: FeatureConfig,
    *,
    available: bool,
) -> np.ndarray:
    names = _audio_feature_names(config)
    if not names:
        return np.empty((len(times), 0), dtype=np.float32)
    if not available or len(samples) == 0:
        return np.zeros((len(times), len(names)), dtype=np.float32)

    frame_seconds = 0.05
    frame_samples = max(16, round(config.audio_sample_rate * frame_seconds))
    frame_count = max(1, math.ceil(len(samples) / frame_samples))
    padded = np.pad(samples, (0, frame_count * frame_samples - len(samples)))
    frames = padded.reshape(frame_count, frame_samples)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1)).astype(np.float32)
    peak = np.max(np.abs(frames), axis=1).astype(np.float32)
    window = np.hanning(frame_samples).astype(np.float32)
    fft_size = 1 << max(1, (frame_samples - 1).bit_length())
    use_noise_bands = (
        config.audio_feature_set == NOISE_NORMALIZED_AUDIO_FEATURE_SET
    )
    band_bounds = (
        (80.0, 250.0),
        (250.0, 500.0),
        (500.0, 1000.0),
        (1000.0, 2000.0),
        (2000.0, 4000.0),
        (4000.0, 7800.0),
    )
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / config.audio_sample_rate)
    band_masks = tuple(
        (frequencies >= lower) & (frequencies < upper)
        for lower, upper in band_bounds
    )
    band_power = (
        np.zeros((frame_count, len(band_masks)), dtype=np.float32)
        if use_noise_bands
        else None
    )
    previous_spectrum: np.ndarray | None = None
    spectral_flux = np.zeros(frame_count, dtype=np.float32)
    for index, frame in enumerate(frames):
        spectrum = np.abs(np.fft.rfft(frame * window, n=fft_size)).astype(np.float32)
        if band_power is not None:
            power = np.square(spectrum, dtype=np.float32)
            for band_index, mask in enumerate(band_masks):
                if np.any(mask):
                    band_power[index, band_index] = float(np.sum(power[mask]))
        spectrum /= max(float(np.sum(spectrum)), 1e-8)
        if previous_spectrum is not None:
            spectral_flux[index] = float(
                np.sqrt(np.sum(np.square(np.maximum(spectrum - previous_spectrum, 0.0))))
            )
        previous_spectrum = spectrum
    rms_novelty = np.maximum(rms - np.concatenate(([rms[0]], rms[:-1])), 0.0)
    peak_rank = _rank_vector(peak)
    novelty_rank = _rank_vector(rms_novelty)
    flux_rank = _rank_vector(spectral_flux)
    onset_strength = (0.45 * flux_rank + 0.35 * novelty_rank + 0.20 * peak_rank).astype(
        np.float32
    )
    contact_like = (onset_strength * np.sqrt(peak_rank)).astype(np.float32)
    strong_threshold = max(0.72, float(np.percentile(contact_like, 85)))
    strong = (contact_like >= strong_threshold) & (peak_rank >= 0.60)

    cadence_window = max(1, round(2.0 / frame_seconds))
    collapse_window = max(1, round(1.0 / frame_seconds))
    cadence = _rolling_mean(contact_like, cadence_window, future=False)
    future_cadence = _rolling_mean(contact_like, collapse_window, future=True)
    cadence_collapse = np.maximum(cadence - future_cadence, 0.0)
    elapsed = np.full(frame_count, 10.0, dtype=np.float32)
    last_time: float | None = None
    audio_times = (np.arange(frame_count, dtype=np.float64) + 0.5) * frame_seconds
    for index, timestamp in enumerate(audio_times):
        if strong[index]:
            last_time = float(timestamp)
            elapsed[index] = 0.0
        elif last_time is not None:
            elapsed[index] = min(10.0, float(timestamp) - last_time)

    noise_floor = np.empty(frame_count, dtype=np.float32)
    noise_window = max(1, round(10.0 / frame_seconds))
    band_noise_floor = (
        np.empty_like(band_power) if band_power is not None else None
    )
    for index in range(frame_count):
        start = max(0, index - noise_window + 1)
        noise_floor[index] = float(np.percentile(rms[start : index + 1], 20))
        if band_noise_floor is not None and band_power is not None:
            band_noise_floor[index] = np.percentile(
                band_power[start : index + 1], 20, axis=0
            )
    snr = np.log1p(np.maximum(rms - noise_floor, 0.0) / (noise_floor + 1e-4)).astype(
        np.float32
    )
    peak_to_rms = np.clip(peak / (rms + 1e-5), 0.0, 30.0).astype(np.float32)

    sources = {
        "audio_rms": rms,
        "audio_peak": peak,
        "audio_peak_to_rms": peak_to_rms,
        "audio_noise_floor": noise_floor,
        "audio_snr": snr,
        "audio_spectral_flux": spectral_flux,
        "audio_rms_novelty": rms_novelty,
        "audio_onset_strength": onset_strength,
        "audio_contact_like_transient": contact_like,
        "audio_onset_cadence": cadence,
        "audio_cadence_collapse": cadence_collapse,
        "audio_seconds_since_transient": elapsed,
    }
    if band_power is not None and band_noise_floor is not None:
        band_excess = np.maximum(band_power - band_noise_floor, 0.0)
        band_snr = np.log1p(
            band_excess / (band_noise_floor + np.float32(1e-8))
        ).astype(np.float32)
        previous_band_snr = np.vstack((band_snr[0], band_snr[:-1]))
        band_snr_flux = np.maximum(band_snr - previous_band_snr, 0.0)
        broadband = np.log1p(
            np.sum(band_excess, axis=1)
            / (np.sum(band_noise_floor, axis=1) + np.float32(1e-8))
        ).astype(np.float32)
        normalized_flux = np.sqrt(
            np.sum(np.square(band_snr_flux, dtype=np.float32), axis=1)
        ).astype(np.float32)
        sources["audio_noise_removed_broadband"] = broadband
        sources["audio_noise_normalized_flux"] = normalized_flux
        band_labels = (
            "80_250",
            "250_500",
            "500_1000",
            "1000_2000",
            "2000_4000",
            "4000_7800",
        )
        for band_index, label in enumerate(band_labels):
            sources[f"audio_band_{label}_snr"] = band_snr[:, band_index]
            sources[f"audio_band_{label}_snr_flux"] = band_snr_flux[:, band_index]

    mean_pooled = {
        "audio_rms",
        "audio_noise_floor",
        "audio_snr",
        "audio_onset_cadence",
        "audio_cadence_collapse",
        "audio_seconds_since_transient",
    }
    output = np.empty((len(times), len(names)), dtype=np.float32)
    output[:, 0] = 1.0
    half_width = 0.5 / config.analysis_fps
    for row, timestamp in enumerate(times):
        left = int(np.searchsorted(audio_times, timestamp - half_width, side="left"))
        right = int(np.searchsorted(audio_times, timestamp + half_width, side="right"))
        if right <= left:
            nearest = min(frame_count - 1, max(0, int(round(timestamp / frame_seconds - 0.5))))
            left, right = nearest, nearest + 1
        for column, name in enumerate(names[1:], start=1):
            source = sources[name]
            if name in mean_pooled:
                value = float(np.mean(source[left:right]))
            else:
                value = float(np.max(source[left:right]))
            output[row, column] = value
    return output


def extract_features(
    path: str | Path,
    config: FeatureConfig,
    roi: tuple[float, float, float, float] | None = None,
    *,
    video_decoder: str = OPENCV_VIDEO_DECODER,
) -> FeatureSequence:
    config.validate()
    if video_decoder not in VIDEO_DECODERS:
        raise VideoError(f"unsupported video decoder: {video_decoder}")
    cv2 = _cv2()
    video_path = Path(path).expanduser().resolve()
    metadata = probe_video(video_path, video_decoder=video_decoder)
    if config.analysis_fps > metadata.fps + 1e-6:
        raise VideoError(
            f"analysis_fps {config.analysis_fps:g} exceeds source FPS {metadata.fps:g}; "
            "normalize at a higher frame rate or lower analysis_fps"
        )
    times: list[float] = []
    rows: list[np.ndarray] = []
    previous_gray: np.ndarray | None = None

    def append(frame_index: int, frame: np.ndarray) -> None:
        nonlocal previous_gray
        timestamp = frame_index / metadata.fps
        cropped = _crop_roi(frame, roi)
        values, previous_gray = _frame_features(cropped, previous_gray, config)
        times.append(timestamp)
        rows.append(values)

    if video_decoder == NVDEC_VIDEO_DECODER:
        for frame_index, frame in _nvdec_frames(video_path, metadata, config.analysis_fps):
            append(frame_index, frame)
    else:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise VideoError(f"cannot open video: {video_path}")
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
                append(frame_index - 1, frame)
                next_sample += 1.0 / config.analysis_fps
        finally:
            capture.release()
    if not rows:
        raise VideoError(f"video yielded no decodable frames: {video_path}")
    frame_names = _frame_feature_names(config)
    matrix = np.vstack(rows).astype(np.float32, copy=False)
    if matrix.shape[1] != len(frame_names):
        raise VideoError("internal frame-feature signature mismatch")
    time_values = np.asarray(times, dtype=np.float64)
    temporal = _temporal_visual_features(matrix, frame_names, config)
    if config.use_audio:
        audio_samples, audio_available = _decode_audio_samples(
            video_path, metadata, config.audio_sample_rate
        )
        audio = _audio_features_from_samples(
            audio_samples,
            time_values,
            config,
            available=audio_available,
        )
    else:
        audio = np.empty((len(matrix), 0), dtype=np.float32)
    matrix = np.concatenate((matrix, temporal, audio), axis=1).astype(np.float32, copy=False)
    names = feature_names(config)
    if matrix.shape[1] != len(names):
        raise VideoError("internal feature signature mismatch")
    if not np.isfinite(matrix).all() or not np.isfinite(time_values).all():
        raise VideoError("video feature extraction produced non-finite values")
    return FeatureSequence(
        times=time_values,
        values=matrix,
        names=names,
        metadata=metadata,
    )


def contextualize(sequence: FeatureSequence, config: FeatureConfig) -> tuple[np.ndarray, tuple[str, ...]]:
    times = sequence.times
    values = sequence.values
    if config.sequence_normalization == "percentile-rank":
        ranked = percentile_rank_values(values)
        for index, name in enumerate(sequence.names):
            if name in ABSOLUTE_FEATURE_NAMES:
                ranked[:, index] = values[:, index]
        values = ranked
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
        blocks.append(values[nearest])
        prefix = f"t{offset:+g}s/"
        names.extend(prefix + name for name in sequence.names)
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False), tuple(names)


def percentile_rank_values(values: np.ndarray) -> np.ndarray:
    """Map every feature to tied within-recording percentile ranks."""
    if values.ndim != 2:
        raise ValueError("feature values must be a two-dimensional matrix")
    if len(values) == 0:
        return values.astype(np.float32, copy=True)
    if len(values) == 1:
        return np.full(values.shape, 0.5, dtype=np.float32)
    result = np.empty(values.shape, dtype=np.float32)
    denominator = float(len(values) - 1)
    for column in range(values.shape[1]):
        _, inverse, counts = np.unique(
            values[:, column],
            return_inverse=True,
            return_counts=True,
        )
        starts = np.cumsum(counts) - counts
        midranks = starts + (counts - 1) / 2.0
        result[:, column] = midranks[inverse] / denominator
    return result


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
    video_decoder: str = OPENCV_VIDEO_DECODER,
) -> str:
    payload = {
        "featureVersion": feature_version_for_config(config),
        "path": str(video),
        "contentSha256": content_sha256,
        "config": config.to_dict(),
        "roi": roi,
    }
    if video_decoder != OPENCV_VIDEO_DECODER:
        payload["videoDecoder"] = video_decoder
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def feature_cache_path(
    recording_id: str,
    video: str | Path,
    config: FeatureConfig,
    roi: tuple[float, float, float, float] | None,
    cache_dir: str | Path,
    *,
    content_sha256: str,
    video_decoder: str = OPENCV_VIDEO_DECODER,
) -> Path:
    """Return the immutable cache path used for one feature extraction."""

    video_path = Path(video).expanduser().resolve()
    destination = Path(cache_dir).expanduser().resolve()
    safe_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in recording_id
    )
    return destination / (
        f"{safe_id}-{_cache_key(video_path, config, roi, content_sha256, video_decoder)}.npz"
    )


def cached_features(
    recording_id: str,
    video: str | Path,
    config: FeatureConfig,
    roi: tuple[float, float, float, float] | None,
    cache_dir: str | Path,
    *,
    content_sha256: str | None = None,
    video_decoder: str = OPENCV_VIDEO_DECODER,
) -> FeatureSequence:
    if video_decoder not in VIDEO_DECODERS:
        raise VideoError(f"unsupported video decoder: {video_decoder}")
    video_path = Path(video).expanduser().resolve()
    destination = Path(cache_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    digest = content_sha256 or _sha256_file(video_path)
    cache_path = feature_cache_path(
        recording_id,
        video_path,
        config,
        roi,
        destination,
        content_sha256=digest,
        video_decoder=video_decoder,
    )
    safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in recording_id)
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_decoder = (
                    str(cached["video_decoder"].item())
                    if "video_decoder" in cached
                    else OPENCV_VIDEO_DECODER
                )
                metadata = VideoMetadata(**json.loads(str(cached["metadata_json"].item())))
                cached_sequence = FeatureSequence(
                    times=cached["times"].astype(np.float64, copy=False),
                    values=cached["values"].astype(np.float32, copy=False),
                    names=tuple(str(item) for item in cached["names"]),
                    metadata=metadata,
                )
            if cached_decoder == video_decoder and _valid_cached_sequence(cached_sequence, config):
                return cached_sequence
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    sequence = extract_features(video_path, config, roi, video_decoder=video_decoder)
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
            video_decoder=np.asarray(video_decoder),
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
