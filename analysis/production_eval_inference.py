"""Label-independent replay of every checked-in production inference head.

The functions in this module consume only model outputs, source video frames, and
frozen runtime artifacts.  Human-edited rallies and score markers are deliberately
not accepted as inputs so that generated artifacts remain valid evaluation
predictions rather than accidentally echoing their targets.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from analysis.config import DecoderConfig
from analysis.decoder import decode_probabilities
from analysis.features import (
    NVDEC_VIDEO_DECODER,
    OPENCV_VIDEO_DECODER,
    FeatureSequence,
    VideoError,
    contextualize,
)
from analysis.serving_side_flight import (
    OFFSETS_SECONDS as FLIGHT_OFFSETS,
    extract_flight_features,
    feature_names as flight_feature_names,
)
from analysis.serving_side_v2 import (
    FEATURE_NAMES as COURT_FEATURE_NAMES,
    OFFSETS_SECONDS as COURT_OFFSETS,
    extract_window_features,
    service_zone_masks,
)
from analysis.side_switch_full_union_features import whole_rally_sample_times
from analysis.side_switch_full_union_ranker import (
    UnionDecoderSettings,
    add_derived_features,
    decode_ranked_candidates,
    predict_classifier,
)
from analysis.side_switch_v4 import (
    FRAME_HEIGHT as SIDE_SWITCH_HEIGHT,
    FRAME_WIDTH as SIDE_SWITCH_WIDTH,
    estimate_court_geometry,
    summarize_sequence,
    visual_features,
)
from analysis.side_switch_v5 import (
    VISUAL_FEATURE_NAMES,
    player_features,
    summarize_player_sequence,
)


ALL_LABELS_SOURCE = "all-labels-v2"
PREVIOUS_SOURCE = "previous-production"
PRODUCTION_SOURCES = (ALL_LABELS_SOURCE, PREVIOUS_SOURCE)
SERVING_SIDE_WIDTH = 192
SERVING_SIDE_HEIGHT = 108
SERVING_SIDE_ANCHOR_CONTRACT = "merged-production-interval-start-v1"
SIDE_SWITCH_CANDIDATE_CONTRACT = "range-boundaries-dead-peaks-v1"
SIDE_SWITCH_STATE_FEATURE_NAMES = (
    "productionBeforeSupportCount",
    "productionAfterSupportCount",
    "productionMinimumAdjacentSupportCount",
    "productionMinimumAdjacentRallyPeak",
    "productionGapLiveFraction",
    "productionGapMeanRallyScore",
    "productionGapPeakRallyScore",
    "productionGapMeanDeadStateScore",
    "productionGapPeakDeadStateScore",
    "productionGapDurationSeconds",
)


def time_key(value: float) -> int:
    """Return the production nine-decimal timestamp identity."""

    return round(float(value) * 1_000_000_000)


def merge_production_ranges(
    ranges: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Mirror the browser's overlap-connected production ensemble union."""

    if set(ranges) != set(PRODUCTION_SOURCES):
        raise ValueError("production range merge requires both frozen model sources")
    tagged: list[dict[str, Any]] = []
    for source in PRODUCTION_SOURCES:
        for item in ranges[source]:
            start, end = float(item["start"]), float(item["end"])
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise ValueError("production model range is malformed")
            tagged.append(
                {
                    "start": start,
                    "end": end,
                    "confidence": float(item.get("confidence", 1.0)),
                    "source": source,
                }
            )
    tagged.sort(key=lambda item: (item["start"], item["end"]))
    clusters: list[list[dict[str, Any]]] = []
    for item in tagged:
        current_end = (
            max((value["end"] for value in clusters[-1]), default=-math.inf)
            if clusters
            else -math.inf
        )
        if not clusters or item["start"] >= current_end:
            clusters.append([item])
        else:
            clusters[-1].append(item)

    output: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters, start=1):
        sources = {str(item["source"]) for item in cluster}
        agreement = (
            "both-models"
            if len(sources) == 2
            else "all-labels-v2-only"
            if ALL_LABELS_SOURCE in sources
            else "previous-production-only"
        )
        source_confidences = [
            max(float(item["confidence"]) for item in cluster if item["source"] == source)
            for source in sources
        ]
        raw_confidence = sum(source_confidences) / len(source_confidences)
        confidence = (
            raw_confidence
            if agreement == "both-models"
            else min(0.49, raw_confidence * 0.6)
        )
        output.append(
            {
                "id": f"R{index:03d}",
                "start": min(float(item["start"]) for item in cluster),
                "end": max(float(item["end"]) for item in cluster),
                "confidence": confidence,
                "included": True,
                "agreement": agreement,
            }
        )
    return output


def suppression_inference(
    runtime: Mapping[str, Any],
    sequence: FeatureSequence,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Run the frozen production suppression head and held decoder."""

    contextual_values, contextual_names = contextualize(sequence, runtime_feature_config(runtime))
    names = tuple(str(value) for value in runtime["featureNames"])
    if contextual_names != names:
        raise ValueError("suppression feature signature differs from regenerated features")
    head = runtime["head"]
    values = contextual_values.astype(np.float32, copy=False).astype(np.float64)
    mean = np.asarray(head["mean"], dtype=np.float32).astype(np.float64)
    scale = np.asarray(head["scale"], dtype=np.float32).astype(np.float64)
    weights = np.asarray(head["weights"], dtype=np.float32).astype(np.float64)
    if any(vector.shape != (len(names),) for vector in (mean, scale, weights)):
        raise ValueError("suppression runtime parameter shape changed")
    logits = np.clip(((values - mean) / scale) @ weights + float(head["bias"]), -30.0, 30.0)
    probabilities = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    decoder = DecoderConfig.from_dict(dict(head["decoder"]))
    intervals, _ = decode_probabilities(
        sequence.times,
        probabilities,
        sequence.metadata.duration,
        decoder,
        float(runtime["analysisFps"]),
    )
    return probabilities, [value.to_dict() for value in intervals]


def runtime_feature_config(runtime: Mapping[str, Any]) -> Any:
    """Build the shared core/suppression feature configuration lazily."""

    from analysis.config import FeatureConfig

    raw = runtime.get("featureConfig")
    if isinstance(raw, Mapping):
        return FeatureConfig.from_dict(dict(raw))
    # The suppression browser artifact omits the duplicated configuration.  Its
    # feature version/signature is shared with both core production bundles, so
    # callers may attach the verified core configuration before invoking us.
    raise ValueError("runtime does not carry a featureConfig")


def _js_round(value: float) -> int:
    return math.floor(value + 0.5)


def _crop_bounds(
    width: int,
    height: int,
    roi: tuple[float, float, float, float] | None,
) -> tuple[int, int, int, int]:
    x, y, roi_width, roi_height = roi or (0.0, 0.0, 1.0, 1.0)
    left = max(0, min(width - 1, _js_round(x * width)))
    top = max(0, min(height - 1, _js_round(y * height)))
    right = max(left + 1, min(width, _js_round((x + roi_width) * width)))
    bottom = max(top + 1, min(height, _js_round((y + roi_height) * height)))
    return left, top, right, bottom


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


def _requested_frame_indexes(
    requested_times: Sequence[float],
    *,
    duration: float,
    fps: float,
    frame_count: int,
) -> dict[int, tuple[int, ...]]:
    by_frame: dict[int, list[int]] = {}
    for requested in requested_times:
        clamped = min(max(0.0, float(requested)), max(0.0, duration - 0.01))
        # The browser's sequential sampler retains the latest frame at or before
        # each target timestamp.
        index = min(frame_count - 1, max(0, math.floor(clamped * fps + 1e-10)))
        by_frame.setdefault(index, []).append(time_key(clamped))
    return {index: tuple(keys) for index, keys in sorted(by_frame.items())}


def sampled_roi_frames(
    video_path: Path,
    *,
    width: int,
    height: int,
    duration: float,
    fps: float,
    frame_count: int,
    roi: tuple[float, float, float, float] | None,
    requested_times: Sequence[float],
    video_decoder: str,
) -> Iterator[tuple[tuple[int, ...], np.ndarray]]:
    """Yield ROI crops for arbitrary timestamps from one recording-owned decoder."""

    by_frame = _requested_frame_indexes(
        requested_times,
        duration=duration,
        fps=fps,
        frame_count=frame_count,
    )
    if not by_frame:
        return
    left, top, right, bottom = _crop_bounds(width, height, roi)
    crop_width, crop_height = right - left, bottom - top
    selected = tuple(by_frame)
    if video_decoder == OPENCV_VIDEO_DECODER:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise VideoError(f"cannot open specialist video: {video_path}")
        selected_set = set(selected)
        try:
            for index in range(selected[-1] + 1):
                if not capture.grab():
                    raise VideoError(f"CPU specialist decoder ended at frame {index}")
                if index not in selected_set:
                    continue
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    raise VideoError(f"could not retrieve specialist frame {index}")
                yield by_frame[index], frame[top:bottom, left:right]
        finally:
            capture.release()
        return
    if video_decoder != NVDEC_VIDEO_DECODER:
        raise VideoError(f"unsupported specialist decoder: {video_decoder}")
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise VideoError("FFmpeg is required for NVDEC specialist inference")
    selection = "+".join(f"eq(n\\,{index})" for index in selected)
    color = (
        "scale=in_color_matrix=bt601:out_color_matrix=bt601:"
        "in_range=full:out_range=full"
    )
    filters = (
        f"select='{selection}',hwdownload,format=nv12,{color},format=bgr24,"
        f"crop={crop_width}:{crop_height}:{left}:{top}"
    )
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
        filters,
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
        raise VideoError("could not open specialist FFmpeg pipes")
    frame_bytes = crop_width * crop_height * 3
    try:
        for index in selected:
            raw = _read_exact(process.stdout, frame_bytes)
            if len(raw) != frame_bytes:
                detail = process.stderr.read().decode("utf-8", errors="replace").strip()
                process.wait()
                raise VideoError(
                    "NVDEC specialist pass ended before all requested frames"
                    + (f": {detail}" if detail else "")
                )
            yield by_frame[index], np.frombuffer(raw, dtype=np.uint8).reshape(
                crop_height, crop_width, 3
            )
        if process.stdout.read(1):
            raise VideoError("NVDEC specialist pass emitted unexpected extra frames")
        detail = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if return_code != 0:
            raise VideoError(
                f"NVDEC specialist pass failed with status {return_code}"
                + (f": {detail}" if detail else "")
            )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def specialist_frame_maps(
    video_path: Path,
    sequence: FeatureSequence,
    roi: tuple[float, float, float, float] | None,
    serving_times: Sequence[float],
    switch_times: Sequence[float],
    video_decoder: str,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Decode the union schedule once and materialize both specialist sizes."""

    import cv2

    serving_keys = {time_key(value) for value in serving_times}
    switch_keys = {time_key(value) for value in switch_times}
    requested = sorted({*serving_times, *switch_times})
    serving: dict[int, np.ndarray] = {}
    switches: dict[int, np.ndarray] = {}
    metadata = sequence.metadata
    for keys, crop in sampled_roi_frames(
        video_path,
        width=metadata.width,
        height=metadata.height,
        duration=metadata.duration,
        fps=metadata.fps,
        frame_count=metadata.frame_count,
        roi=roi,
        requested_times=requested,
        video_decoder=video_decoder,
    ):
        serve_frame: np.ndarray | None = None
        switch_frame: np.ndarray | None = None
        for key in keys:
            if key in serving_keys:
                if serve_frame is None:
                    resized = cv2.resize(
                        crop,
                        (SERVING_SIDE_WIDTH, SERVING_SIDE_HEIGHT),
                        interpolation=cv2.INTER_AREA,
                    )
                    serve_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                serving[key] = serve_frame
            if key in switch_keys:
                if switch_frame is None:
                    switch_frame = cv2.resize(
                        crop,
                        (SIDE_SWITCH_WIDTH, SIDE_SWITCH_HEIGHT),
                        interpolation=cv2.INTER_AREA,
                    )
                switches[key] = switch_frame
    if serving_keys != set(serving) or switch_keys != set(switches):
        raise VideoError("specialist decoder did not satisfy the complete frame plan")
    return serving, switches


def _clamped_time(anchor: float, offset: float, duration: float) -> float:
    return min(max(0.0, anchor + offset), max(0.0, duration - 0.01))


def serving_side_frame_times(
    intervals: Sequence[Mapping[str, Any]], duration: float
) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                _clamped_time(float(interval["start"]), float(offset), duration)
                for interval in intervals
                for offset in (*COURT_OFFSETS, *FLIGHT_OFFSETS)
            }
        )
    )


def _frames_for_offsets(
    frames: Mapping[int, np.ndarray],
    anchor: float,
    offsets: Sequence[float],
    duration: float,
) -> list[np.ndarray]:
    return [frames[time_key(_clamped_time(anchor, float(offset), duration))] for offset in offsets]


def _tied_percentile_ranks(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    rows, columns = matrix.shape
    if not np.isfinite(matrix).all():
        raise ValueError("serving-side raw features must be finite")
    result = np.zeros_like(matrix)
    if rows == 1:
        result.fill(0.5)
        return result
    for column in range(columns):
        order = np.argsort(matrix[:, column], kind="stable")
        sorted_values = matrix[order, column]
        start = 0
        while start < rows:
            end = start + 1
            while end < rows and sorted_values[end] == sorted_values[start]:
                end += 1
            result[order[start:end], column] = ((start + end - 1) / 2.0) / (rows - 1)
            start = end
    return result


def _serve_head_evidence(
    model_id: str,
    times: np.ndarray,
    probabilities: np.ndarray,
    detections: Sequence[Mapping[str, Any]],
    anchor: float,
    threshold: float,
    window_seconds: float,
) -> dict[str, Any]:
    distances = np.abs(times - anchor)
    selected = np.flatnonzero(distances <= window_seconds + 1e-9)
    if not len(selected):
        selected = np.asarray([int(np.argmin(distances))])
    peak = int(selected[0])
    for index in selected[1:]:
        if probabilities[int(index)] > probabilities[peak]:
            peak = int(index)
    nearest = (
        min(detections, key=lambda item: abs(float(item["time"]) - anchor))
        if detections
        else None
    )
    probability = float(probabilities[peak])
    return {
        "modelId": model_id,
        "threshold": threshold,
        "peakProbability": probability,
        "peakTime": float(times[peak]),
        "crossesThreshold": probability >= threshold,
        "nearestDetection": dict(nearest) if nearest is not None else None,
    }


def serving_side_inference(
    runtime: Mapping[str, Any],
    intervals: Sequence[Mapping[str, Any]],
    duration: float,
    frames: Mapping[int, np.ndarray],
    times: np.ndarray,
    serve_probabilities: Mapping[str, np.ndarray],
    serve_detections: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], np.ndarray]:
    """Run current serving-side features, within-recording ranks, head, and gate."""

    candidates = [dict(value) for value in intervals if value.get("included") is True]
    masks, _ = service_zone_masks(
        SERVING_SIDE_HEIGHT,
        SERVING_SIDE_WIDTH,
        roi=(0.0, 0.0, 1.0, 1.0),
        court_geometry=None,
    )
    names = tuple(str(value) for value in runtime["featureNames"])
    expected_names = tuple(
        [
            *(f"v2:{name}" for name in COURT_FEATURE_NAMES),
            *(f"flight:{name}" for name in flight_feature_names(4, 6)),
        ]
    )
    if names != expected_names:
        raise ValueError("serving-side production feature signature changed")
    raw_rows: list[np.ndarray] = []
    for candidate in candidates:
        anchor = float(candidate["start"])
        court = extract_window_features(
            _frames_for_offsets(frames, anchor, COURT_OFFSETS, duration), masks
        )
        flight = extract_flight_features(
            _frames_for_offsets(frames, anchor, FLIGHT_OFFSETS, duration), 4, 6
        )
        raw_rows.append(
            np.asarray(
                [
                    *(court[name] for name in COURT_FEATURE_NAMES),
                    *(flight[name] for name in flight_feature_names(4, 6)),
                ],
                dtype=np.float64,
            )
        )
    raw = np.vstack(raw_rows) if raw_rows else np.empty((0, len(names)), dtype=np.float64)
    ranked = _tied_percentile_ranks(raw) if len(raw) else raw.copy()
    model = runtime["model"]
    impute = np.asarray(model["impute"], dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    probabilities = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                ((np.where(np.isfinite(ranked), ranked, impute) - mean) / scale) @ weights
                + float(model["bias"]),
                -30.0,
                30.0,
            )
        )
    )
    gate = runtime["gate"]
    output_rows: list[dict[str, Any]] = []
    model_ids = {
        ALL_LABELS_SOURCE: "model-1ca43e38eefc",
        PREVIOUS_SOURCE: "model-9c92b8e9333f",
    }
    for index, candidate in enumerate(candidates):
        anchor = float(candidate["start"])
        evidence = {
            source: _serve_head_evidence(
                model_ids[source],
                times,
                serve_probabilities[source],
                serve_detections[source],
                anchor,
                float(gate["serveHeadThreshold"]),
                float(gate["serveHeadWindowSeconds"]),
            )
            for source in PRODUCTION_SOURCES
        }
        head_crosses = any(value["crossesThreshold"] for value in evidence.values())
        if head_crosses:
            source, reasons, is_serve = "serve-head", [], True
        elif candidate["agreement"] == "both-models":
            source, reasons, is_serve = (
                "production-rally-recovery",
                ["production-rally-recovery"],
                True,
            )
        else:
            source, reasons, is_serve = "none", [], False
        probability = float(probabilities[index])
        side = "near" if probability >= float(runtime["sideThreshold"]) else "far"
        review = runtime["reviewBand"]
        if float(review["farUpperExclusive"]) <= probability < float(
            review["nearLowerInclusive"]
        ):
            reasons.insert(0, "side-score")
        verdict = "not-serve" if not is_serve else "review" if reasons else side
        output_rows.append(
            {
                "id": candidate["id"],
                "anchor": anchor,
                "interval": {
                    "start": anchor,
                    "end": float(candidate["end"]),
                    "agreement": candidate["agreement"],
                },
                "nearProbability": probability,
                "side": side,
                "verdict": verdict,
                "serveDecisionSource": source,
                "reviewReasons": reasons,
                "serveEvidence": {
                    "allLabelsV2": evidence[ALL_LABELS_SOURCE],
                    "previousProduction": evidence[PREVIOUS_SOURCE],
                },
            }
        )
    return (
        {
            "modelId": runtime["modelId"],
            "modelFingerprint": runtime["fingerprint"],
            "featureVersion": runtime["featureVersion"],
            "anchorContract": SERVING_SIDE_ANCHOR_CONTRACT,
            "features": {"rows": len(raw), "columns": len(names)},
            "candidates": output_rows,
        },
        raw,
    )


def generate_side_switch_candidates(
    intervals: Sequence[Mapping[str, Any]],
    times: np.ndarray,
    dead_state_probabilities: np.ndarray,
    runtime: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ranges = [dict(value) for value in intervals if value.get("included") is True]
    result: list[dict[str, Any]] = []
    for before, after in zip(ranges, ranges[1:], strict=False):
        result.append(
            {
                "id": f"switch:boundary:{before['id']}:{after['id']}",
                "kind": "adjacent-rally-boundary",
                "gapStart": float(before["end"]),
                "gapEnd": float(after["start"]),
                "transitionTime": (float(before["end"]) + float(after["start"])) / 2.0,
                "generatorScore": 0.0,
                "sourceRangeIds": [before["id"], after["id"]],
                "beforeWindow": {
                    "start": float(before["start"]),
                    "end": float(before["end"]),
                },
                "afterWindow": {
                    "start": float(after["start"]),
                    "end": float(after["end"]),
                },
            }
        )
    config = runtime["candidateGenerator"]
    for source_range in ranges:
        start, end = float(source_range["start"]), float(source_range["end"])
        eligible = [
            index
            for index, timestamp in enumerate(times)
            if timestamp >= start + float(config["internalPeakRangeEdgeExclusionSeconds"])
            and timestamp <= end - float(config["internalPeakRangeEdgeExclusionSeconds"])
            and dead_state_probabilities[index] >= float(config["internalPeakThreshold"])
        ]
        eligible.sort(
            key=lambda index: (
                -float(dead_state_probabilities[index]),
                float(times[index]),
                index,
            )
        )
        selected: list[int] = []
        for index in eligible:
            if all(
                abs(float(times[index] - times[other]))
                >= float(config["internalPeakMinimumSeparationSeconds"])
                for other in selected
            ):
                selected.append(index)
        for index in sorted(selected, key=lambda value: float(times[value])):
            timestamp = float(times[index])
            result.append(
                {
                    "id": f"switch:internal-dead-peak:{source_range['id']}:{round(timestamp * 1000)}",
                    "kind": "internal-dead-state-peak",
                    "gapStart": max(
                        start,
                        timestamp
                        - float(config["internalPeakProposalHalfWidthSeconds"]),
                    ),
                    "gapEnd": min(
                        end,
                        timestamp
                        + float(config["internalPeakProposalHalfWidthSeconds"]),
                    ),
                    "transitionTime": timestamp,
                    "generatorScore": float(dead_state_probabilities[index]),
                    "sourceRangeIds": [source_range["id"]],
                    "beforeWindow": {"start": timestamp - 4.0, "end": timestamp - 1.0},
                    "afterWindow": {"start": timestamp + 1.0, "end": timestamp + 4.0},
                }
            )
    return sorted(result, key=lambda item: (item["transitionTime"], item["kind"], item["id"]))


def side_switch_frame_times(
    intervals: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    duration: float,
) -> tuple[float, ...]:
    requested: set[float] = set()
    for interval in list(intervals)[:7]:
        requested.update(whole_rally_sample_times(float(interval["start"]), float(interval["end"])))
    for candidate in candidates:
        for window_name in ("beforeWindow", "afterWindow"):
            window = candidate[window_name]
            requested.update(whole_rally_sample_times(float(window["start"]), float(window["end"])))
    return tuple(
        sorted(
            min(max(0.0, value), max(0.0, duration - 0.01))
            for value in requested
        )
    )


def _values_in_window(
    times: np.ndarray, values: np.ndarray, start: float, end: float
) -> np.ndarray:
    selected = values[(times >= start) & (times < end)]
    if len(selected):
        return selected
    center = (start + end) / 2.0
    return values[np.asarray([int(np.argmin(np.abs(times - center)))])]


def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _union_duration(
    start: float, end: float, ranges: Sequence[Mapping[str, Any]]
) -> float:
    clipped = sorted(
        (
            max(start, float(value["start"])),
            min(end, float(value["end"])),
        )
        for value in ranges
        if min(end, float(value["end"])) > max(start, float(value["start"]))
    )
    if not clipped:
        return 0.0
    total, active_start, active_end = 0.0, clipped[0][0], clipped[0][1]
    for left, right in clipped[1:]:
        if left <= active_end:
            active_end = max(active_end, right)
        else:
            total += active_end - active_start
            active_start, active_end = left, right
    return total + active_end - active_start


def _side_switch_state_features(
    candidate: Mapping[str, Any],
    intervals: Sequence[Mapping[str, Any]],
    production_ranges: Mapping[str, Sequence[Mapping[str, Any]]],
    times: np.ndarray,
    rally_scores: Mapping[str, np.ndarray],
    dead_scores: Mapping[str, np.ndarray],
) -> dict[str, float]:
    by_id = {str(value["id"]): value for value in intervals}

    def evidence(source_range: Mapping[str, Any]) -> tuple[float, float]:
        sources = [
            source
            for source in PRODUCTION_SOURCES
            if any(
                _overlap(
                    float(source_range["start"]),
                    float(source_range["end"]),
                    float(value["start"]),
                    float(value["end"]),
                )
                > 0.0
                for value in production_ranges[source]
            )
        ]
        peaks = [
            float(
                np.max(
                    _values_in_window(
                        times,
                        rally_scores[source],
                        float(source_range["start"]),
                        float(source_range["end"]),
                    )
                )
            )
            for source in sources
        ]
        return float(len(sources)), min(peaks) if peaks else 0.0

    before = by_id[str(candidate["sourceRangeIds"][0])]
    after = by_id[str(candidate["sourceRangeIds"][-1])]
    before_support, before_peak = evidence(before)
    after_support, after_peak = evidence(after)
    rally = np.maximum(rally_scores[ALL_LABELS_SOURCE], rally_scores[PREVIOUS_SOURCE])
    dead = np.maximum(dead_scores[ALL_LABELS_SOURCE], dead_scores[PREVIOUS_SOURCE])
    start, end = float(candidate["gapStart"]), float(candidate["gapEnd"])
    rally_window = _values_in_window(times, rally, start, end)
    dead_window = _values_in_window(times, dead, start, end)
    duration = max(end - start, 1e-6)
    source_ranges = [
        *production_ranges[ALL_LABELS_SOURCE],
        *production_ranges[PREVIOUS_SOURCE],
    ]
    values = (
        before_support,
        after_support,
        min(before_support, after_support),
        min(before_peak, after_peak),
        _union_duration(start, end, source_ranges) / duration,
        float(np.mean(rally_window)),
        float(np.max(rally_window)),
        float(np.mean(dead_window)),
        float(np.max(dead_window)),
        end - start,
    )
    return dict(zip(SIDE_SWITCH_STATE_FEATURE_NAMES, values, strict=True))


def side_switch_inference(
    recording_id: str,
    runtime: Mapping[str, Any],
    intervals: Sequence[Mapping[str, Any]],
    production_ranges: Mapping[str, Sequence[Mapping[str, Any]]],
    times: np.ndarray,
    rally_scores: Mapping[str, np.ndarray],
    dead_scores: Mapping[str, np.ndarray],
    candidates: Sequence[Mapping[str, Any]],
    frames: Mapping[int, np.ndarray],
    duration: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    calibration = [
        frames[time_key(min(max(0.0, value), max(0.0, duration - 0.01)))]
        for interval in list(intervals)[:7]
        for value in whole_rally_sample_times(float(interval["start"]), float(interval["end"]))
    ]
    if not candidates:
        return (
            {
                "modelId": runtime["modelId"],
                "modelFingerprint": runtime["fingerprint"],
                "featureVersion": runtime["featureVersion"],
                "candidateContract": SIDE_SWITCH_CANDIDATE_CONTRACT,
                "features": {"rows": 0, "columns": len(runtime["classifier"]["featureNames"])},
                "proposals": [],
                "candidates": [],
            },
            np.empty((0, len(runtime["classifier"]["featureNames"])), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    geometry = estimate_court_geometry(calibration)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        sequences: list[list[np.ndarray]] = []
        for window_name in ("beforeWindow", "afterWindow"):
            window = candidate[window_name]
            sequences.append(
                [
                    frames[time_key(min(max(0.0, value), max(0.0, duration - 0.01)))]
                    for value in whole_rally_sample_times(float(window["start"]), float(window["end"]))
                ]
            )
        before, after = sequences
        v4_values = visual_features(
            summarize_sequence(before, geometry), summarize_sequence(after, geometry)
        )
        visual = player_features(
            summarize_player_sequence(before, geometry),
            summarize_player_sequence(after, geometry),
            v4_values,
        )
        if tuple(visual) != VISUAL_FEATURE_NAMES:
            raise ValueError("side-switch visual feature contract changed")
        state = _side_switch_state_features(
            candidate,
            intervals,
            production_ranges,
            times,
            rally_scores,
            dead_scores,
        )
        row = add_derived_features(
            {
                "eventId": candidate["id"],
                "recordingId": recording_id,
                "kind": candidate["kind"],
                "gapStart": candidate["gapStart"],
                "gapEnd": candidate["gapEnd"],
                "transitionTime": candidate["transitionTime"],
                "score": candidate["generatorScore"],
                "features": {**visual, **state},
            }
        )
        rows.append(row)
    probabilities = predict_classifier(runtime["classifier"], rows)
    decoder = runtime["decoder"]
    selected = decode_ranked_candidates(
        rows,
        probabilities,
        float(runtime["classifier"]["threshold"]),
        UnionDecoderSettings(
            minimum_index_separation=int(decoder["minimumCandidateIndexSeparation"]),
            minimum_time_separation_seconds=float(decoder["minimumTimeSeparationSeconds"]),
            free_predictions_per_recording=int(decoder["freePredictionsPerRecording"]),
            count_penalty_logit=float(decoder["countPenaltyLogitPerExcessPrediction"]),
        ),
    )
    names = tuple(str(value) for value in runtime["classifier"]["featureNames"])
    matrix = np.asarray(
        [[float(row["features"][name]) for name in names] for row in rows],
        dtype=np.float64,
    )
    proposals = [
        {
            **{key: value for key, value in candidate.items() if key not in {"beforeWindow", "afterWindow"}},
            "probability": float(probabilities[index]),
            "selected": bool(selected[index]),
        }
        for index, candidate in enumerate(candidates)
    ]
    selected_rows = [
        {
            "id": candidate["id"],
            "timestamp": float(candidate["transitionTime"]),
            "probability": float(probabilities[index]),
            "kind": candidate["kind"],
            "sourceRangeIds": list(candidate["sourceRangeIds"]),
        }
        for index, candidate in enumerate(candidates)
        if selected[index]
    ]
    return (
        {
            "modelId": runtime["modelId"],
            "modelFingerprint": runtime["fingerprint"],
            "featureVersion": runtime["featureVersion"],
            "candidateContract": SIDE_SWITCH_CANDIDATE_CONTRACT,
            "features": {"rows": len(matrix), "columns": len(names)},
            "courtGeometry": geometry.to_dict(),
            "proposals": proposals,
            "candidates": selected_rows,
        },
        matrix,
        probabilities,
    )


__all__ = [
    "ALL_LABELS_SOURCE",
    "PREVIOUS_SOURCE",
    "PRODUCTION_SOURCES",
    "generate_side_switch_candidates",
    "merge_production_ranges",
    "sampled_roi_frames",
    "serving_side_frame_times",
    "serving_side_inference",
    "side_switch_frame_times",
    "side_switch_inference",
    "specialist_frame_maps",
    "suppression_inference",
    "time_key",
]
