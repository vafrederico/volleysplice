"""Label-independent, continuous player/court motion research features.

The supplied ROI is only a court-membership proxy. Hip coordinates are image
coordinates, not ground-plane footpoints or identities. No labels, rally
predictions, outcome-based windows, or jersey descriptors enter this module.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

import cv2
import numpy as np

from analysis import side_switch_player_detector as pinned
from analysis.serving_side_flight import _affine_camera_flow


FEATURE_VERSION = "continuous-player-court-motion-v1"
SIDE_STATISTICS = (
    "count", "confidence", "center_x", "center_y", "spread_x", "spread_y",
    "pair_distance", "torso_area", "motion_coverage", "speed_mean", "speed_p90",
    "moving_fraction", "flow_x", "flow_y", "direction_coherence",
    "speed_rise_fraction", "speed_fall_fraction",
)
FEATURE_NAMES = (
    "player_detector_ran", "player_tracks_available", "player_observation_age_seconds",
    "player_detection_saturated", "player_rejected_roi_fraction",
    "player_net_geometry_supplied", "player_motion_available", "player_scene_jump",
    "player_camera_speed", "player_camera_fit_residual", "player_matched_fraction",
    "player_both_sides_visible", "player_both_sides_moving", "player_reaction_synchrony",
    "player_standdown_synchrony",
    *(f"player_{side}_{name}" for side in ("near", "far") for name in SIDE_STATISTICS),
)


@dataclass(frozen=True)
class PlayerMotionConfig:
    analysis_fps: float = 4.0
    detector_fps: float = 2.0
    motion_long_side: int = 320
    maximum_detections: int = 24
    net_y_ratio: float = 0.5
    net_geometry_supplied: bool = False
    court_left: float = 0.0
    court_right: float = 1.0
    court_top: float = 0.0
    court_bottom: float = 1.0
    moving_speed: float = 0.025
    maximum_track_age: float = 0.75
    maximum_transition_seconds: float = 0.5
    scene_jump_threshold: float = 0.45

    def __post_init__(self) -> None:
        numeric = [v for v in asdict(self).values() if isinstance(v, (float, int))]
        if not all(math.isfinite(float(v)) for v in numeric):
            raise ValueError("configuration must be finite")
        if self.analysis_fps != 4 or self.detector_fps != 2:
            raise ValueError("v1 fixes the analysis/detector grids at 4/2 Hz")
        if not 64 <= self.motion_long_side <= 640 or not 12 <= self.maximum_detections <= 100:
            raise ValueError("invalid motion resolution or detection cap")
        if not 0.1 <= self.net_y_ratio <= 0.9:
            raise ValueError("net ratio must lie in [0.1,0.9]")
        if not (0 <= self.court_left < self.court_right <= 1
                and 0 <= self.court_top < self.court_bottom <= 1):
            raise ValueError("court membership proxy must be inside ROI")
        if min(self.moving_speed, self.maximum_track_age, self.maximum_transition_seconds) <= 0:
            raise ValueError("motion thresholds must be positive")


def duplicate_detection(left: pinned.PlayerDetection, right: pinned.PlayerDetection) -> bool:
    """Torso-relative duplicate rule; never a fixed 10%-of-frame hip radius."""
    torso = min(math.hypot(left.hip_x-left.shoulder_x, left.hip_y-left.shoulder_y),
                math.hypot(right.hip_x-right.shoulder_x, right.hip_y-right.shoulder_y))
    hip_distance = math.hypot(left.hip_x-right.hip_x, left.hip_y-right.hip_y)
    return pinned._iou(left, right) >= 0.45 or hip_distance < 0.25 * torso


def select_detections(candidates: Sequence[pinned.PlayerDetection], limit: int) -> tuple[pinned.PlayerDetection, ...]:
    selected: list[pinned.PlayerDetection] = []
    for candidate in sorted(candidates, key=lambda d: (-d.score, d.hip_x, d.hip_y)):
        if all(not duplicate_detection(candidate, prior) for prior in selected):
            selected.append(candidate)
        if len(selected) >= limit:
            break
    return tuple(selected)


class RallyPersonDetector(pinned.QuantizedPersonDetector):
    """Reuse pinned weights/tiles/decoder; isolate the new 6v6 selection policy."""

    def __init__(self, model_dir: str, *, maximum_detections: int = 24, opencv_threads: int = 2):
        super().__init__(model_dir, opencv_threads=opencv_threads)
        if maximum_detections < 12:
            raise ValueError("rally detector must allow at least twelve people")
        self.maximum_detections = maximum_detections

    def detect(self, frame: np.ndarray) -> pinned.PersonDetectionResult:
        height, width = frame.shape[:2]
        candidates: list[pinned.PlayerDetection] = []
        raw_count, elapsed = 0, 0.0
        for tile in pinned.detector_tiles(width, height):
            crop = frame[tile.y:tile.y+tile.height, tile.x:tile.x+tile.width]
            blob, _, pad_x, pad_y = pinned._preprocess(crop)
            self.net.setInput(blob)
            start = time.perf_counter()
            geometry, scores = self.net.forward(self.output_names)
            elapsed += (time.perf_counter()-start)*1000
            decoded, count = pinned._decode_tile(geometry, scores, tile=tile,
                frame_width=width, frame_height=height, pad_x=pad_x, pad_y=pad_y)
            candidates.extend(decoded)
            raw_count += count
        return pinned.PersonDetectionResult(select_detections(candidates, self.maximum_detections), raw_count, elapsed)


@dataclass(frozen=True)
class Track:
    x: float
    y: float
    width: float
    height: float
    hip_x: float
    hip_y: float
    confidence: float
    observed_at: float
    track_id: int
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    previous_speed: float = 0.0
    motion_valid: bool = False

    @property
    def speed(self) -> float:
        return math.hypot(self.velocity_x, self.velocity_y)


def normalized_detection(detection: pinned.PlayerDetection, width: int, height: int,
                         timestamp: float, track_id: int) -> Track:
    return Track(detection.x/width, detection.y/height, detection.width/width,
        detection.height/height, detection.hip_x/width, detection.hip_y/height,
        detection.score, timestamp, track_id)


def court_y(value: float, net_ratio: float) -> float:
    return value/net_ratio*0.5 if value <= net_ratio else 0.5+(value-net_ratio)/(1-net_ratio)*0.5


def inside_court(track: Track, config: PlayerMotionConfig) -> bool:
    return (config.court_left <= track.hip_x <= config.court_right
            and config.court_top <= track.hip_y <= config.court_bottom)


def match_tracks(previous: Sequence[Track], current: Sequence[Track]) -> tuple[list[Track], int]:
    """Associate short anonymous tracks with spatial/scale gates, never identity claims."""
    result = list(current)
    if not previous or not current:
        return result, 0
    cost = np.full((len(previous), len(current)), 1e6, np.float64)
    for i, before in enumerate(previous):
        for j, after in enumerate(current):
            distance = math.hypot(before.hip_x-after.hip_x, before.hip_y-after.hip_y)
            ratio = (after.width*after.height)/max(before.width*before.height, 1e-9)
            if distance <= 0.12 and 0.3 <= ratio <= 3.0:
                cost[i, j] = distance + 0.025*abs(math.log(ratio))
    # Deterministic lowest-cost greedy links keep this bounded deployment path
    # independent of SciPy. Associations are deliberately short and uncertain.
    pairs = sorted((float(cost[i, j]), i, j) for i in range(len(previous)) for j in range(len(current)))
    used_rows, used_columns = set(), set()
    count = 0
    for value, i, j in pairs:
        if value >= 1e5 or i in used_rows or j in used_columns:
            continue
        used_rows.add(i)
        used_columns.add(j)
        prior = previous[i]
        result[j] = replace(current[j], track_id=prior.track_id,
            velocity_x=prior.velocity_x, velocity_y=prior.velocity_y,
            previous_speed=prior.previous_speed, motion_valid=prior.motion_valid)
        count += 1
    return result, count


def side_statistics(tracks: Sequence[Track], config: PlayerMotionConfig) -> dict[str, float]:
    output = {name: 0.0 for name in SIDE_STATISTICS}
    if not tracks:
        return output
    positions = np.asarray([(t.hip_x, court_y(t.hip_y, config.net_y_ratio)) for t in tracks])
    output.update(count=min(len(tracks)/6, 4.0), confidence=float(np.mean([t.confidence for t in tracks])),
        center_x=float(positions[:, 0].mean()), center_y=float(positions[:, 1].mean()),
        spread_x=float(positions[:, 0].std()), spread_y=float(positions[:, 1].std()),
        torso_area=float(np.mean([t.width*t.height for t in tracks])))
    distances = [float(np.linalg.norm(positions[i]-positions[j]))
                 for i in range(len(tracks)) for j in range(i+1, len(tracks))]
    output["pair_distance"] = float(np.mean(distances)) if distances else 0.0
    motion = [t for t in tracks if t.motion_valid]
    output["motion_coverage"] = len(motion)/len(tracks)
    if motion:
        speeds = np.asarray([t.speed for t in motion])
        vx, vy = float(np.mean([t.velocity_x for t in motion])), float(np.mean([t.velocity_y for t in motion]))
        output.update(speed_mean=float(speeds.mean()), speed_p90=float(np.quantile(speeds, .9)),
            moving_fraction=float(np.mean(speeds >= config.moving_speed)), flow_x=vx, flow_y=vy,
            direction_coherence=min(1., math.hypot(vx, vy)/max(float(speeds.mean()), 1e-9)),
            speed_rise_fraction=float(np.mean([t.speed-t.previous_speed >= config.moving_speed for t in motion])),
            speed_fall_fraction=float(np.mean([t.previous_speed-t.speed >= config.moving_speed for t in motion])))
    return output


def _patch_median(flow: np.ndarray, track: Track) -> np.ndarray:
    height, width = flow.shape[:2]
    x0, x1 = max(0, min(width-1, round(track.x*width))), max(1, min(width, round((track.x+track.width)*width)))
    y0, y1 = max(0, min(height-1, round(track.y*height))), max(1, min(height, round((track.y+track.height)*height)))
    return np.median(flow[y0:max(y0+1, y1), x0:max(x0+1, x1)].reshape(-1, 2), axis=0)


def propagate_tracks(tracks: Sequence[Track], flow: np.ndarray, residual: np.ndarray,
                     dt: float, timestamp: float, config: PlayerMotionConfig) -> list[Track]:
    height, width = flow.shape[:2]
    output = []
    for track in tracks:
        if timestamp-track.observed_at > config.maximum_track_age:
            continue
        raw = _patch_median(flow, track)/[width, height]
        velocity = _patch_median(residual, track)/[width*dt, height*dt]
        moved = replace(track, x=track.x+float(raw[0]), y=track.y+float(raw[1]),
            hip_x=track.hip_x+float(raw[0]), hip_y=track.hip_y+float(raw[1]),
            velocity_x=float(velocity[0]), velocity_y=float(velocity[1]),
            previous_speed=track.speed, motion_valid=True)
        if inside_court(moved, config):
            output.append(moved)
    return output


class PlayerMotionExtractor:
    def __init__(self, detector: Any, config: PlayerMotionConfig = PlayerMotionConfig()):
        self.detector, self.config = detector, config
        self.previous_gray: np.ndarray | None = None
        self.previous_time: float | None = None
        self.tracks: list[Track] = []
        self.tick, self.next_id = 0, 0
        self.last_detection_time: float | None = None
        self.total_detector_frames, self.total_tile_calls = 0, 0
        self.total_inference_ms = 0.0

    def push(self, frame: np.ndarray, timestamp: float, *, reset: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        if not math.isfinite(timestamp) or timestamp < 0 or (self.previous_time is not None and timestamp <= self.previous_time):
            raise ValueError("timestamps must be finite, nonnegative and strictly increasing")
        if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 16:
            raise ValueError("expected a nonempty BGR ROI")
        height, width = frame.shape[:2]
        scale = self.config.motion_long_side/max(height, width)
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (max(16, round(width*scale)), max(16, round(height*scale))))
        dt = 0 if self.previous_time is None else timestamp-self.previous_time
        transition = (not reset and self.previous_gray is not None
                      and self.previous_gray.shape == gray.shape
                      and 0 < dt <= self.config.maximum_transition_seconds+1e-8)
        jump, camera_speed, camera_residual = 0., 0., 0.
        if transition:
            jump = float(np.mean(np.abs(gray.astype(np.float32)-self.previous_gray))/255)
            transition = jump < self.config.scene_jump_threshold
        if transition:
            flow = cv2.calcOpticalFlowFarneback(self.previous_gray, gray, None, .5, 3, 15, 3, 5, 1.1, 0)
            camera = _affine_camera_flow(flow)
            residual = flow-camera
            normalizer = np.array([gray.shape[1]*dt, gray.shape[0]*dt])
            camera_speed = float(np.mean(np.linalg.norm(camera/normalizer, axis=2)))
            camera_residual = float(np.median(np.linalg.norm(residual/normalizer, axis=2)))
            predicted = propagate_tracks(self.tracks, flow, residual, dt, timestamp, self.config)
        else:
            predicted = []
        detector_ran = self.tick % 2 == 0 or not transition
        matched, rejected, saturated = 0, 0, 0
        raw_count, inference_ms = 0, 0.
        if detector_ran:
            detected = self.detector.detect(frame)  # Failure aborts: never silently encode a failed detector as absence.
            raw_count, inference_ms = detected.raw_candidates, detected.inference_milliseconds
            saturated = int(len(detected.detections) >= self.config.maximum_detections)
            current = []
            for detection in detected.detections:
                observation = normalized_detection(detection, width, height, timestamp, self.next_id)
                self.next_id += 1
                if inside_court(observation, self.config):
                    current.append(observation)
                else:
                    rejected += 1
            self.tracks, matched = match_tracks(predicted, current)
            self.last_detection_time = timestamp
            self.total_detector_frames += 1
            self.total_tile_calls += getattr(self.detector, "forward_passes_per_frame", 4)
            self.total_inference_ms += inference_ms
        else:
            self.tracks, matched = predicted, len(predicted)
        near = side_statistics([t for t in self.tracks if court_y(t.hip_y, self.config.net_y_ratio) >= .5], self.config)
        far = side_statistics([t for t in self.tracks if court_y(t.hip_y, self.config.net_y_ratio) < .5], self.config)
        output = dict(zip(FEATURE_NAMES[:15], [float(detector_ran), float(bool(self.tracks)),
            timestamp-self.last_detection_time if self.last_detection_time is not None else self.config.maximum_track_age,
            float(saturated), rejected/max(1, rejected+len(self.tracks)), float(self.config.net_geometry_supplied),
            float(transition), jump, camera_speed, camera_residual, matched/max(1, len(self.tracks)),
            float(bool(near["count"] and far["count"])), min(near["moving_fraction"], far["moving_fraction"]),
            min(near["speed_rise_fraction"], far["speed_rise_fraction"]),
            min(near["speed_fall_fraction"], far["speed_fall_fraction"])], strict=True))
        for side, values in (("near", near), ("far", far)):
            output.update({f"player_{side}_{key}": value for key, value in values.items()})
        vector = np.asarray([output[name] for name in FEATURE_NAMES], np.float32)
        if not np.isfinite(vector).all():
            raise ValueError("nonfinite player motion feature")
        audit = {"detectorRan": detector_ran, "visibleTracks": len(self.tracks),
            "nearTracks": round(near["count"]*6), "farTracks": round(far["count"]*6),
            "matchedTracks": matched, "rawDetectorCandidates": raw_count, "roiRejected": rejected,
            "detectorSaturated": bool(saturated), "inferenceMilliseconds": inference_ms,
            "motionAvailable": bool(transition), "sceneJump": jump,
            "trackIds": [t.track_id for t in self.tracks]}
        self.previous_gray, self.previous_time = gray, timestamp
        self.tick += 1
        return vector, audit


def nearest_frame_indexes(presentation_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    pts, targets = np.asarray(presentation_times, np.float64), np.asarray(target_times, np.float64)
    for name, values in (("presentation", pts), ("target", targets)):
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
            raise ValueError(f"{name} timestamps must be finite and strictly increasing")
    if abs(float(pts[0])) > 1e-6 or targets[0] < 0:
        raise ValueError("v1 requires zero media origin and nonnegative targets")
    right = np.minimum(np.searchsorted(pts, targets), len(pts)-1)
    left = np.maximum(right-1, 0)
    indexes = np.where(np.abs(targets-pts[left]) <= np.abs(pts[right]-targets), left, right)
    if np.any(np.diff(indexes) <= 0) or float(np.max(np.abs(pts[indexes]-targets))) > .125+1e-8:
        raise ValueError("invalid or repeated selected frames / >half-tick PTS error")
    return indexes.astype(np.int64)


def crop_roi(frame: np.ndarray, roi: Sequence[float] | None) -> np.ndarray:
    if roi is None:
        return frame
    if len(roi) != 4 or not all(math.isfinite(float(x)) for x in roi):
        raise ValueError("ROI must be finite x,y,width,height")
    x, y, w, h = (float(v) for v in roi)
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > 1+1e-8 or y+h > 1+1e-8:
        raise ValueError("ROI must lie inside source frame")
    height, width = frame.shape[:2]
    x0, y0 = min(width-1, round(x*width)), min(height-1, round(y*height))
    x1, y1 = max(x0+1, min(width, round((x+w)*width))), max(y0+1, min(height, round((y+h)*height)))
    return np.ascontiguousarray(frame[y0:y1, x0:x1])
