#!/usr/bin/env python3
"""Extract and evaluate frozen side-switch profiles on provisional held recordings."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.config import DecoderConfig, FeatureConfig
from analysis.model import (
    DEAD_STATE_TASK,
    RALLY_LIVE_TASK,
    SERVE_CONTACT_TASK,
    LogisticModel,
)
from analysis.pipeline import PreparedRecording, prepare_recording
from analysis.schema import Recording, load_manifest
from analysis.serving_side import crop_roi
from analysis.side_switch_appearance import read_frame
from analysis.side_switch_feature_development import (
    DECODER,
    evaluate_predictions,
)
from analysis.side_switch_full_union_ranker import (
    add_derived_features,
    decode_ranked_candidates,
    predict_classifier,
)
from analysis.side_switch_player_detector import (
    QuantizedPersonDetector,
    detector_identity,
)
from analysis.side_switch_production_replay import ProductionHeads, replay_trace
from analysis.side_switch_production_state import (
    ProductionComponent,
    ProductionTrace,
    TimeRange,
    gap_state_features,
    merge_production_components,
    rally_evidence,
)
from analysis.side_switch_t14_dominant_tracklet_medoid import (
    dominant_tracklet_transport_features,
    summarize_dominant_tracklet_endpoint,
)
from analysis.side_switch_t15_bilateral_consensus import bilateral_consensus_features
from analysis.side_switch_t16_source_resolved import source_resolved_features
from analysis.side_switch_t18_representativeness import representativeness_features
from analysis.side_switch_t19_cross_representation import (
    cross_representation_features,
)
from analysis.side_switch_t20_compact_candidate import compact_candidate_features
from analysis.side_switch_t1_transport import (
    endpoint_sample_times as t1_endpoint_sample_times,
    summarize_endpoint as summarize_t1_endpoint,
    transport_features as t1_transport_features,
)
from analysis.side_switch_t2_transport import t2_features
from analysis.side_switch_t3_jersey_transport import (
    endpoint_sample_times as t14_endpoint_sample_times,
    summarize_team,
)
from analysis.side_switch_t4_selective_far import (
    SelectiveFarEndpointSummary,
    selective_far_transport_features,
)
from analysis.side_switch_v4 import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    estimate_court_geometry,
    summarize_sequence,
    visual_features,
)
from analysis.side_switch_v5 import (
    VISUAL_FEATURE_NAMES,
    player_features,
    summarize_player_sequence,
)
from scripts.extract_side_switch_helpers import load_json, sha256_path


REPOSITORY = Path(__file__).resolve().parents[1]
LABELING_ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
WORKSPACE = Path("/mnt/freenas/volleycut/intake-2026-08-25-shoreline-kb")
REPORTS = LABELING_ROOT / "reports/side-switch"
RUNTIME_ROOT = REPOSITORY / "prod/public/runtime"
DEFAULT_FEATURES = REPORTS / "side-switch-provisional-recording-held-gold-r1-features.json"
DEFAULT_EVALUATION = REPORTS / "side-switch-provisional-recording-held-gold-r1-evaluation.json"
RECORDING_IDS = (
    "grass-source-11",
    "grass-source-07",
)
LABEL_SHA256 = {
    "grass-source-11": "ced51c22114541acec74b10d6992109e90425f6a53f0bc7e4581a6f4429fdaa5",
    "grass-source-07": "8bada37b00829755ef28e82fc2d549cba4c782b8e9fefbc58a31e6f087ac7a4e",
}
PREDICTION_SHA256 = {
    "grass-source-11": "b0e0361ee01c697edd5fa075b3e0ef586e30222c1e2f935fbc63a39293e496c1",
    "grass-source-07": "21eafa8bd7a9390a43cb27938294d9f200ac73ff242f3c9b4434c40799c22550",
}
NATIVE_VIDEO = {
    "grass-source-11": Path(
        "/mnt/freenas/volleycut-raw-no-backups/"
        "grass-source-11.mkv"
    ),
    "grass-source-07": Path(
        "/mnt/freenas/volleycut-raw-no-backups/"
        "20250614 - KB private event Grass Rev2s - Game 1 "
        "(w⧸ player-a vs player-b + player-c) (-15) [grass-source-07].mkv"
    ),
}
NATIVE_VIDEO_SHA256 = {
    "grass-source-11": "c4f4b7ade66c0318c79e46d078f6acab2a4bd4eae44fb7976e5bc8163ed8125f",
    "grass-source-07": "beac7f3ab7cc303e6e2acc613751eccefdc0ebdd50971c54f924dd1d16bd55b7",
}
NATIVE_VIDEO_SHAPE = {
    "grass-source-11": (2160, 3840),
    "grass-source-07": (1080, 1920),
}
FFMPEG_NATIVE_RECORDINGS = frozenset({"grass-source-11"})
PROFILE_ARTIFACTS = {
    "boundary-t0": (
        REPORTS / "side-switch-feature-development-t20-compact-medoid-disagreement-opened-v1.json",
        "b8f4bdb45d8f3c7fc679efbb372c6145a1e00e4b32c3783cecd5e901adaa0e9b",
        "boundary-union34-v1",
    ),
    "t2": (
        REPORTS / "side-switch-feature-development-t2-conditional-transport-opened-v1.json",
        "e21b56dfed42e4752996c0354ae2a3b80ed391d99ad0a1447165d44e38ad4a98",
        "boundary-union34-plus-conditional-identity-transport-t2",
    ),
    "t4": (
        REPORTS / "side-switch-feature-development-t4-selective-far-opened-v1.json",
        "16d617fb8427517b45c31196877d7482cb019f8902b989e5b2db19cfcef0020b",
        "boundary-union34-plus-selective-far-jersey-t4",
    ),
    "t14": (
        REPORTS / "side-switch-feature-development-t14-dominant-tracklet-medoid-opened-v1.json",
        "ba6b7493f4d2c98a5d6bd755947a1f917ce4a5d0b69a9cb07c7b48c903d54f06",
        "boundary-union34-plus-dominant-tracklet-medoid-t14",
    ),
    "t16": (
        REPORTS / "side-switch-feature-development-t16-source-resolved-medoid-opened-v1.json",
        "040d970e942819f0f7dd3bea0e9fc9ec46b4de0f15c7a015a777fa6508e69a05",
        "boundary-union34-plus-source-resolved-medoid-t16",
    ),
    "t18": (
        REPORTS / "side-switch-feature-development-t18-medoid-representativeness-opened-v1.json",
        "488b20863cca787d54e973e4f106ad98e6314418a151edffeaf768213492b7dd",
        "boundary-union34-plus-representative-medoid-t18",
    ),
    "t19": (
        REPORTS / "side-switch-feature-development-t19-cross-representation-consensus-opened-v1.json",
        "b67e2502f05cba27ba9a0b1fc19fd614506264b8319db0befd0bd17b33c615c7",
        "boundary-union34-plus-cross-representation-consensus-t19",
    ),
    "t20": (
        REPORTS / "side-switch-feature-development-t20-compact-medoid-disagreement-opened-v1.json",
        "b8f4bdb45d8f3c7fc679efbb372c6145a1e00e4b32c3783cecd5e901adaa0e9b",
        "boundary-union34-plus-compact-medoid-disagreement-t20",
    ),
}
RUNTIME_FILES = {
    "allLabels": ("model-1ca43e38eefc.json", "d2c2c11e8fed8b6c6ad77d244b613e81d5bab101939a8f57be5166b45ebca78f"),
    "previous": ("model-9c92b8e9333f.json", "d8cc42f70bc10576a5e03251b05981ceeee1a61a15c61cc5dfb68dd631e6f90d"),
    "sideSwitch": ("side-switch-c2570481c30d.json", "ab4197545fb916a37ee6ac1d69e74ddfc0123c09039cdfa88c4ef378dd3e27fc"),
}


@dataclass(frozen=True)
class SwitchCandidate:
    event_id: str
    kind: str
    transition_time: float
    gap_start: float
    gap_end: float
    generator_score: float
    source_ranges: tuple[ProductionComponent, ...]
    before_window: tuple[float, float]
    after_window: tuple[float, float]


def _runtime_head(
    runtime: Mapping[str, Any], key: str, prediction_task: str
) -> LogisticModel:
    feature_config = FeatureConfig.from_dict(runtime["featureConfig"])
    head = runtime[key]
    rally_decoder = runtime["rally"]["decoder"]
    training: dict[str, Any] = {}
    if key == "serve":
        training = {"serveDecoder": head["decoder"], "composition": head["composition"]}
    elif key == "deadState":
        training = {
            "selectedDeadStateDecoder": head["decoder"],
            "selectedRefinement": head["refinement"],
        }
    return LogisticModel(
        feature_config=feature_config,
        feature_names=tuple(str(name) for name in runtime["featureNames"]),
        mean=np.asarray(head["mean"], dtype=np.float32),
        scale=np.asarray(head["scale"], dtype=np.float32),
        weights=np.asarray(head["weights"], dtype=np.float32),
        bias=float(head["bias"]),
        decoder=DecoderConfig.from_dict(rally_decoder),
        training_summary=training,
        artifact_sha256=str(head.get("artifactSha256", "runtime")),
        feature_version=str(runtime["featureVersion"]),
        prediction_task=prediction_task,
    )


def _production_heads(runtime: Mapping[str, Any]) -> ProductionHeads:
    return ProductionHeads(
        rally=_runtime_head(runtime, "rally", RALLY_LIVE_TASK),
        serve=_runtime_head(runtime, "serve", SERVE_CONTACT_TASK),
        dead=_runtime_head(runtime, "deadState", DEAD_STATE_TASK),
    )


def _sample_times(start: float, end: float) -> list[float]:
    duration = end - start
    first = start + min(0.2, duration * 0.08)
    last = end - min(0.15, duration * 0.08)
    if last <= first:
        first = start + duration * 0.2
        last = start + duration * 0.8
    return [first + (last - first) * index / 6 for index in range(7)]


def _resized_frame(
    capture: cv2.VideoCapture,
    timestamp: float,
    duration: float,
    roi: tuple[float, float, float, float] | None,
) -> np.ndarray:
    bounded = min(max(0.0, timestamp), max(0.0, duration - 0.01))
    frame = crop_roi(read_frame(capture, bounded), roi)
    return cv2.resize(
        frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA
    )


def _native_frame(
    path: Path,
    capture: cv2.VideoCapture | None,
    timestamp: float,
    shape: tuple[int, int],
) -> np.ndarray:
    if capture is not None:
        return read_frame(capture, timestamp)
    command = (
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.9f}",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    )
    completed = subprocess.run(command, check=True, capture_output=True)
    height, width = shape
    expected = height * width * 3
    if len(completed.stdout) != expected:
        raise RuntimeError(
            f"FFmpeg returned {len(completed.stdout)} bytes instead of {expected} "
            f"for {path} at {timestamp:.3f}s"
        )
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(height, width, 3).copy()


def _switch_candidates(
    trace: ProductionTrace, components: Sequence[ProductionComponent], runtime: Mapping[str, Any]
) -> list[SwitchCandidate]:
    candidates: list[SwitchCandidate] = []
    for before, after in zip(components, components[1:], strict=False):
        if after.start <= before.end:
            continue
        candidates.append(
            SwitchCandidate(
                event_id=f"switch:boundary:{before.start:.3f}:{after.start:.3f}",
                kind="adjacent-rally-boundary",
                transition_time=(before.end + after.start) / 2,
                gap_start=before.end,
                gap_end=after.start,
                generator_score=0.0,
                source_ranges=(before, after),
                before_window=(before.start, before.end),
                after_window=(after.start, after.end),
            )
        )
    generator = runtime["candidateGenerator"]
    threshold = float(generator["internalPeakThreshold"])
    separation = float(generator["internalPeakMinimumSeparationSeconds"])
    edge = float(generator["internalPeakRangeEdgeExclusionSeconds"])
    half_width = float(generator["internalPeakProposalHalfWidthSeconds"])
    dead = np.asarray(trace.dead_state_scores["all-labels-v2"])
    for component in components:
        eligible = [
            (float(trace.times[index]), index)
            for index in range(len(trace.times))
            if component.start + edge <= trace.times[index] <= component.end - edge
            and dead[index] >= threshold
        ]
        eligible.sort(key=lambda item: (-dead[item[1]], item[0], item[1]))
        selected: list[tuple[float, int]] = []
        for candidate in eligible:
            if all(abs(candidate[0] - other[0]) >= separation for other in selected):
                selected.append(candidate)
        for timestamp, index in sorted(selected):
            candidates.append(
                SwitchCandidate(
                    event_id=(
                        f"switch:internal-dead-peak:{component.start:.3f}:"
                        f"{round(timestamp * 1000)}"
                    ),
                    kind="internal-dead-state-peak",
                    transition_time=timestamp,
                    gap_start=max(component.start, timestamp - half_width),
                    gap_end=min(component.end, timestamp + half_width),
                    generator_score=float(dead[index]),
                    source_ranges=(component,),
                    before_window=(timestamp - 4, timestamp - 1),
                    after_window=(timestamp + 1, timestamp + 4),
                )
            )
    return sorted(
        candidates, key=lambda item: (item.transition_time, item.kind, item.event_id)
    )


def _component_range(component: ProductionComponent) -> TimeRange:
    return TimeRange(component.start, component.end)


def _t4_from_t14(summary: Any) -> SelectiveFarEndpointSummary:
    base = summary.base
    return SelectiveFarEndpointSummary(
        near=summarize_team(base.near_stable),
        far=summarize_team(base.far_stable),
        selected_counts=base.selected_counts,
        full_raw_candidate_counts=base.full_raw_candidate_counts,
        far_raw_candidate_counts=base.far_raw_candidate_counts,
        background_fallbacks=base.background_fallbacks,
        full_detector_inference_milliseconds=base.full_detector_inference_milliseconds,
        far_detector_inference_milliseconds=base.far_detector_inference_milliseconds,
        far_crop_top=base.far_crop_top,
        far_crop_bottom=base.far_crop_bottom,
        frame_height=base.frame_height,
    )


def _extract_recording(
    recording: Recording,
    prepared: PreparedRecording,
    old_heads: ProductionHeads,
    new_heads: ProductionHeads,
    switch_runtime: Mapping[str, Any],
    detector_dir: Path,
    opencv_threads: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    trace = replay_trace(prepared, old_heads, new_heads)
    components = merge_production_components(trace.ranges)
    candidates = _switch_candidates(trace, components, switch_runtime)
    prediction_path = WORKSPACE / "predictions/score-markers" / f"{recording.id}.json"
    if sha256_path(prediction_path) != PREDICTION_SHA256[recording.id]:
        raise ValueError(f"prediction snapshot changed for {recording.id}")
    prediction = load_json(prediction_path)
    expected_probabilities = np.asarray(
        prediction["sideSwitchModel"]["candidateProbabilities"], dtype=np.float64
    )
    if len(expected_probabilities) != len(candidates):
        raise ValueError(f"candidate count changed for {recording.id}")

    duration = prepared.sequence.metadata.duration
    capture = cv2.VideoCapture(str(recording.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open proxy video: {recording.video}")
    try:
        calibration = [
            _resized_frame(capture, timestamp, duration, recording.roi)
            for component in components[:7]
            for timestamp in _sample_times(component.start, component.end)
        ]
        geometry = estimate_court_geometry(calibration)
        rows: list[dict[str, Any]] = []
        for number, candidate in enumerate(candidates, 1):
            before_frames = [
                _resized_frame(capture, timestamp, duration, recording.roi)
                for timestamp in _sample_times(*candidate.before_window)
            ]
            after_frames = [
                _resized_frame(capture, timestamp, duration, recording.roi)
                for timestamp in _sample_times(*candidate.after_window)
            ]
            broad_before = summarize_sequence(before_frames, geometry)
            broad_after = summarize_sequence(after_frames, geometry)
            broad = visual_features(broad_before, broad_after)
            players = player_features(
                summarize_player_sequence(before_frames, geometry),
                summarize_player_sequence(after_frames, geometry),
                broad,
            )
            before_evidence = rally_evidence(
                _component_range(candidate.source_ranges[0]), trace, components
            )
            after_evidence = rally_evidence(
                _component_range(candidate.source_ranges[-1]), trace, components
            )
            state, suppression = gap_state_features(
                before_evidence,
                after_evidence,
                candidate.gap_start,
                candidate.gap_end,
                trace,
            )
            if suppression:
                raise ValueError("suppression unexpectedly entered gold extraction")
            features = {
                **{name: round(float(players[name]), 8) for name in VISUAL_FEATURE_NAMES},
                **{name: round(float(value), 8) for name, value in list(state.items())[:10]},
            }
            row = add_derived_features(
                {
                    "eventId": candidate.event_id,
                    "recordingId": recording.id,
                    "kind": candidate.kind,
                    "gapStart": candidate.gap_start,
                    "gapEnd": candidate.gap_end,
                    "transitionTime": candidate.transition_time,
                    "score": candidate.generator_score if candidate.generator_score else None,
                    "sourceRangeId": None,
                    "features": features,
                    "comparisonWindows": {
                        "before": {"start": candidate.before_window[0], "end": candidate.before_window[1]},
                        "after": {"start": candidate.after_window[0], "end": candidate.after_window[1]},
                        "framesPerWindow": 7,
                    },
                }
            )
            rows.append(row)
            if number % 20 == 0 or number == len(candidates):
                print(
                    f"{recording.id}: base {number}/{len(candidates)}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        capture.release()

    runtime_scores = predict_classifier(switch_runtime["classifier"], rows)
    parity_error = float(np.max(np.abs(runtime_scores - expected_probabilities)))
    if parity_error > 1e-7:
        raise ValueError(
            f"production side-switch parity failed for {recording.id}: {parity_error}"
        )

    boundaries = [row for row in rows if row["kind"] == "adjacent-rally-boundary"]
    windows: dict[tuple[float, float], Mapping[str, Any]] = {}
    for row in boundaries:
        for window in row["comparisonWindows"].values():
            if not isinstance(window, Mapping):
                continue
            key = (
                round(float(window["start"]), 9),
                round(float(window["end"]), 9),
            )
            windows[key] = window
    native_path = NATIVE_VIDEO[recording.id]
    if sha256_path(native_path) != NATIVE_VIDEO_SHA256[recording.id]:
        raise ValueError(f"native video changed for {recording.id}")
    detector = QuantizedPersonDetector(detector_dir, opencv_threads=opencv_threads)
    native = (
        None
        if recording.id in FFMPEG_NATIVE_RECORDINGS
        else cv2.VideoCapture(str(native_path))
    )
    if native is not None and not native.isOpened():
        raise RuntimeError(f"cannot open native video: {native_path}")
    t1_summaries: dict[tuple[float, float], Any] = {}
    t14_summaries: dict[tuple[float, float], Any] = {}
    try:
        for number, (key, window) in enumerate(sorted(windows.items()), 1):
            t1_frames = [
                crop_roi(
                    _native_frame(
                        native_path,
                        native,
                        timestamp,
                        NATIVE_VIDEO_SHAPE[recording.id],
                    ),
                    recording.roi,
                )
                for timestamp in t1_endpoint_sample_times(
                    float(window["start"]), float(window["end"])
                )
            ]
            t1_summaries[key] = summarize_t1_endpoint(
                t1_frames, geometry.net_y_ratio, detector
            )
            t14_frames = [
                crop_roi(
                    _native_frame(
                        native_path,
                        native,
                        timestamp,
                        NATIVE_VIDEO_SHAPE[recording.id],
                    ),
                    recording.roi,
                )
                for timestamp in t14_endpoint_sample_times(
                    float(window["start"]), float(window["end"])
                )
            ]
            t14_summaries[key] = summarize_dominant_tracklet_endpoint(
                t14_frames, geometry.net_y_ratio, detector
            )
            if number % 20 == 0 or number == len(windows):
                print(
                    f"{recording.id}: endpoints {number}/{len(windows)}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if native is not None:
            native.release()

    for row in boundaries:
        before_window = row["comparisonWindows"]["before"]
        after_window = row["comparisonWindows"]["after"]
        before_key = (
            round(float(before_window["start"]), 9),
            round(float(before_window["end"]), 9),
        )
        after_key = (
            round(float(after_window["start"]), 9),
            round(float(after_window["end"]), 9),
        )
        before_t1 = t1_summaries[before_key]
        after_t1 = t1_summaries[after_key]
        t1_values, t1_diagnostics = t1_transport_features(before_t1, after_t1)
        row["features"].update(t1_values)
        row["t1Transport"] = {
            "status": "ok",
            "before": before_t1.to_diagnostic(),
            "after": after_t1.to_diagnostic(),
            **t1_diagnostics,
        }
        t2_values, _ = t2_features(row)
        row["features"].update(t2_values)

        before_t14 = t14_summaries[before_key]
        after_t14 = t14_summaries[after_key]
        t4, _ = selective_far_transport_features(
            _t4_from_t14(before_t14), _t4_from_t14(after_t14)
        )
        t14, t14_diagnostics = dominant_tracklet_transport_features(
            before_t14, after_t14
        )
        row["features"].update(t4)
        row["features"].update(t14)
        row["t14DominantTrackletMedoid"] = {
            "status": "ok",
            "before": before_t14.to_diagnostic(),
            "after": after_t14.to_diagnostic(),
            **t14_diagnostics,
        }
        t15_values, t15_diagnostics = bilateral_consensus_features(row)
        row["features"].update(t15_values)
        row["t15BilateralMedoidConsensus"] = {
            "status": "ok",
            **t15_diagnostics,
        }
        row["features"].update(source_resolved_features(row))
        row["features"].update(representativeness_features(row))
        row["features"].update(cross_representation_features(row))
        row["features"].update(compact_candidate_features(row))

    audit = {
        "recordingId": recording.id,
        "sourceGroup": recording.source_group,
        "proxyVideoPath": str(recording.video),
        "proxyVideoSizeBytes": recording.video.stat().st_size,
        "proxyVideoSha256": sha256_path(recording.video),
        "nativeVideoPath": str(native_path),
        "nativeVideoSizeBytes": native_path.stat().st_size,
        "nativeVideoSha256": NATIVE_VIDEO_SHA256[recording.id],
        "nativeVideoDecoder": (
            "ffmpeg-cli-lossless-bgr24"
            if recording.id in FFMPEG_NATIVE_RECORDINGS
            else "opencv-video-capture"
        ),
        "productionComponents": len(components),
        "candidates": len(rows),
        "boundaryCandidates": len(boundaries),
        "internalCandidates": len(rows) - len(boundaries),
        "endpointWindows": len(windows),
        "productionProbabilityMaximumAbsoluteDifference": parity_error,
        "courtGeometry": geometry.to_dict(),
        "elapsedSeconds": time.perf_counter() - started,
    }
    return rows, audit


def _verified_runtime(name: str) -> Mapping[str, Any]:
    filename, expected = RUNTIME_FILES[name]
    path = RUNTIME_ROOT / filename
    actual = sha256_path(path)
    if actual != expected:
        raise ValueError(f"runtime changed for {name}: {actual}")
    return load_json(path)


def extract_features(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.features.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feature artifact: {output}")
    manifest_path = (WORKSPACE / "manifests/inference-only.json").resolve()
    manifest = load_manifest(manifest_path, require_videos=True)
    selected = [value for value in manifest.recordings if value.id in RECORDING_IDS]
    if tuple(value.id for value in selected) != RECORDING_IDS:
        by_id = {value.id: value for value in selected}
        selected = [by_id[value] for value in RECORDING_IDS]
    if tuple(value.id for value in selected) != RECORDING_IDS:
        raise ValueError("manifest does not contain the frozen gold recording IDs")
    if any(value.consent.get("train") for value in selected):
        raise ValueError("gold recordings must remain train=false")

    all_labels = _verified_runtime("allLabels")
    previous = _verified_runtime("previous")
    switch_runtime = _verified_runtime("sideSwitch")
    new_heads = _production_heads(all_labels)
    old_heads = _production_heads(previous)
    if new_heads.rally.feature_config.to_dict() != old_heads.rally.feature_config.to_dict():
        raise ValueError("production feature configurations differ")
    cache_dir = (WORKSPACE / "features/audiovisual-audio-normalized-v3").resolve()
    detector_dir = (
        LABELING_ROOT
        / "models/third-party/opencv-zoo-mediapipe-person-int8bq-2023mar"
    ).resolve()

    def run(recording: Recording) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prepared = prepare_recording(
            recording, new_heads.rally.feature_config, cache_dir
        )
        return _extract_recording(
            recording,
            prepared,
            old_heads,
            new_heads,
            switch_runtime,
            detector_dir,
            args.opencv_threads,
        )

    results: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=min(args.recording_workers, len(selected))) as pool:
        futures = {recording.id: pool.submit(run, recording) for recording in selected}
        for recording_id, future in futures.items():
            results[recording_id] = future.result()
    rows = [row for recording_id in RECORDING_IDS for row in results[recording_id][0]]
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-provisional-recording-held-gold-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "label-independent-provisional-gold-features",
        "scope": {
            "recordingIds": list(RECORDING_IDS),
            "recordings": len(RECORDING_IDS),
            "candidates": len(rows),
            "boundaryCandidates": sum(row["kind"] == "adjacent-rally-boundary" for row in rows),
            "internalCandidates": sum(row["kind"] == "internal-dead-state-peak" for row in rows),
        },
        "rows": rows,
        "extractionAudit": {key: results[key][1] for key in RECORDING_IDS},
        "sources": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_path(manifest_path)},
            "predictions": {
                key: {
                    "path": str(WORKSPACE / "predictions/score-markers" / f"{key}.json"),
                    "sha256": PREDICTION_SHA256[key],
                }
                for key in RECORDING_IDS
            },
            "runtime": {
                key: {
                    "path": str(RUNTIME_ROOT / filename),
                    "sha256": expected,
                }
                for key, (filename, expected) in RUNTIME_FILES.items()
            },
            "extractor": {"path": str(Path(__file__).resolve()), "sha256": sha256_path(Path(__file__).resolve())},
        },
        "personDetector": detector_identity(detector_dir),
        "limitations": [
            "No label file or side-switch marker was loaded during extraction.",
            "Base candidates come from frozen production inference, not corrected human rallies.",
            "T2/T4/T14/T16/T18/T19/T20 native-resolution features use the immutable source MKV files.",
            "T15 diagnostics are materialized only as an input to passing T16; the failed-gate T15 profile is not evaluated.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _profile_classifier(
    artifact_path: Path, expected_hash: str, profile_name: str
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    actual = sha256_path(artifact_path)
    if actual != expected_hash:
        raise ValueError(f"profile artifact changed: {artifact_path}: {actual}")
    artifact = load_json(artifact_path)
    profile = artifact["profiles"][profile_name]
    classifier = profile["fullDevelopment"]["classifier"]
    return classifier, {
        "path": str(artifact_path),
        "sha256": actual,
        "profile": profile_name,
        "threshold": float(classifier["threshold"]),
    }


def _inside_ignored(timestamp: float, intervals: Sequence[Mapping[str, Any]]) -> bool:
    return any(float(value["start"]) <= timestamp < float(value["end"]) for value in intervals)


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    features_path = args.features.resolve()
    output = args.evaluation.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation artifact: {output}")
    feature_payload = load_json(features_path)
    if tuple(feature_payload["scope"]["recordingIds"]) != RECORDING_IDS:
        raise ValueError("feature artifact does not match the frozen gold scope")
    rows = [dict(row) for row in feature_payload["rows"]]

    markers: dict[str, list[dict[str, Any]]] = {}
    ignored: dict[str, list[Mapping[str, Any]]] = {}
    label_audit: dict[str, Any] = {}
    for recording_id in RECORDING_IDS:
        label_path = WORKSPACE / "labels/full" / f"{recording_id}.labels.json"
        actual = sha256_path(label_path)
        if actual != LABEL_SHA256[recording_id]:
            raise ValueError(f"provisional gold label changed for {recording_id}: {actual}")
        label = load_json(label_path)
        annotation = label["annotation"]
        local_markers = [
            {"id": f"{recording_id}:gold:{index:02d}", "recordingId": recording_id, "time": float(value["time"])}
            for index, value in enumerate(label["sideSwitches"], 1)
        ]
        local_ignored = list(label.get("ignoredIntervals", []))
        if any(_inside_ignored(value["time"], local_ignored) for value in local_markers):
            raise ValueError(f"gold marker lies inside ignored time for {recording_id}")
        markers[recording_id] = local_markers
        ignored[recording_id] = local_ignored
        label_audit[recording_id] = {
            "path": str(label_path),
            "sha256": actual,
            "annotationStatus": annotation.get("status"),
            "continuousVideoReviewed": annotation.get("continuousVideoReviewed"),
            "reviewedAt": annotation.get("reviewedAt"),
            "humanEvents": len(local_markers),
            "modelOriginMarkersRetained": sum(
                value.get("origin") == "model" for value in label["sideSwitches"]
            ),
            "ignoredIntervals": local_ignored,
        }

    eligible_rows = [
        row
        for row in rows
        if not _inside_ignored(float(row["transitionTime"]), ignored[str(row["recordingId"])])
    ]
    boundary_rows = [
        row for row in eligible_rows if row["kind"] == "adjacent-rally-boundary"
    ]
    classifier_sources: dict[str, Any] = {}
    classifiers: dict[str, Mapping[str, Any]] = {}
    for identifier, (path, expected_hash, profile_name) in PROFILE_ARTIFACTS.items():
        classifiers[identifier], classifier_sources[identifier] = _profile_classifier(
            path, expected_hash, profile_name
        )
    runtime = _verified_runtime("sideSwitch")
    classifiers = {"deployed-union34": runtime["classifier"], **classifiers}
    classifier_sources = {
        "deployed-union34": {
            "path": str(RUNTIME_ROOT / RUNTIME_FILES["sideSwitch"][0]),
            "sha256": RUNTIME_FILES["sideSwitch"][1],
            "profile": "deployed full-union union34",
            "threshold": float(runtime["classifier"]["threshold"]),
        },
        **classifier_sources,
    }

    profile_results: dict[str, Any] = {}
    for identifier, classifier in classifiers.items():
        local_rows = eligible_rows if identifier == "deployed-union34" else boundary_rows
        probabilities = predict_classifier(classifier, local_rows)
        selected = decode_ranked_candidates(
            local_rows, probabilities, float(classifier["threshold"]), DECODER
        )
        coverage = evaluate_predictions(
            local_rows,
            np.ones(len(local_rows), dtype=bool),
            markers,
            4.0,
            inventory=False,
        )
        profile_results[identifier] = {
            "candidateRows": len(local_rows),
            "candidateCoverage": {
                "coveredHumanEvents": coverage["truePositives"],
                "humanEvents": sum(len(value) for value in markers.values()),
                "recall": coverage["recall"],
            },
            "strict": evaluate_predictions(
                local_rows, selected, markers, 0.0, inventory=True
            ),
            "primary": evaluate_predictions(
                local_rows, selected, markers, 4.0, inventory=True
            ),
            "candidateScores": [
                {
                    "eventId": row["eventId"],
                    "recordingId": row["recordingId"],
                    "kind": row["kind"],
                    "transitionTime": row["transitionTime"],
                    "gapStart": row["gapStart"],
                    "gapEnd": row["gapEnd"],
                    "probability": float(probability),
                    "selected": bool(keep),
                }
                for row, probability, keep in zip(
                    local_rows, probabilities, selected, strict=True
                )
            ],
        }

    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-provisional-recording-held-gold-evaluation-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "provisional-recording-held-gold",
        "scope": {
            "recordingIds": list(RECORDING_IDS),
            "recordings": len(RECORDING_IDS),
            "humanEvents": sum(len(value) for value in markers.values()),
            "candidateRowsBeforeIgnored": len(rows),
            "candidateRowsAfterIgnored": len(eligible_rows),
            "boundaryRowsAfterIgnored": len(boundary_rows),
        },
        "goldMarkers": markers,
        "labelAudit": label_audit,
        "profiles": profile_results,
        "sources": {
            "features": {"path": str(features_path), "sha256": sha256_path(features_path)},
            "classifiers": classifier_sources,
            "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256_path(Path(__file__).resolve())},
        },
        "contract": {
            "trainingUse": "none",
            "thresholdOrDecoderSelectionUse": "none",
            "strictPaddingSeconds": 0.0,
            "primaryPaddingSeconds": 4.0,
            "ignoredIntervalPolicy": "exclude candidate when transition anchor is inside ignored interval",
            "decoder": DECODER.to_dict(),
        },
        "limitations": [
            "Both labels are in-progress snapshots.",
            "grass-source-11 is not marked continuously reviewed.",
            "Both source groups appeared in historical side-switch development.",
            "The labeling tasks were seeded with the deployed model's proposals.",
            "Seven human events across two recordings give wide sampling uncertainty.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "profiles" not in payload:
        return {"scope": payload["scope"], "extractionAudit": payload["extractionAudit"]}
    return {
        "scope": payload["scope"],
        "profiles": {
            key: {
                "coverage": value["candidateCoverage"],
                "strict": {name: value["strict"][name] for name in ("proposals", "truePositives", "falsePositives", "falseNegatives", "precision", "recall", "f1")},
                "primary": {name: value["primary"][name] for name in ("proposals", "truePositives", "falsePositives", "falseNegatives", "precision", "recall", "f1")},
            }
            for key, value in payload["profiles"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("extract", "evaluate"))
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--recording-workers", type=int, default=2)
    parser.add_argument("--opencv-threads", type=int, default=10)
    args = parser.parse_args()
    if args.recording_workers < 1 or args.opencv_threads < 1:
        parser.error("worker and OpenCV thread counts must be positive")
    payload = extract_features(args) if args.phase == "extract" else evaluate(args)
    print(json.dumps(_summary(payload), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
