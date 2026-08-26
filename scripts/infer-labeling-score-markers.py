#!/usr/bin/env python3
"""Run the frozen production score-marker models for prepared labeling tasks."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from analysis.annotations import load_label_document
from analysis.artifacts import atomic_write_text
from analysis.config import DecoderConfig, FeatureConfig
from analysis.model import (
    DEAD_STATE_TASK,
    RALLY_LIVE_TASK,
    SERVE_CONTACT_TASK,
    LogisticModel,
)
from analysis.pipeline import PreparedRecording, prepare_recording
from analysis.schema import load_manifest
from analysis.serving_side import crop_roi
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
from analysis.side_switch_appearance import read_frame
from analysis.side_switch_production_replay import ProductionHeads, replay_trace
from analysis.side_switch_production_state import (
    ProductionComponent,
    ProductionTrace,
    TimeRange,
    gap_state_features,
    merge_production_components,
    rally_evidence,
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


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = Path("/mnt/freenas/volleycut/intake-2026-08-25-shoreline-kb")
DEFAULT_RUNTIME = REPOSITORY / "prod/public/runtime"
SERVING_MODEL_FILENAME = "serving-side-85bc3325fbd4.json"
SWITCH_MODEL_FILENAME = "side-switch-c2570481c30d.json"
ALL_LABELS_FILENAME = "model-1ca43e38eefc.json"
PREVIOUS_FILENAME = "model-9c92b8e9333f.json"
SERVING_RESIZE = (192, 108)


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


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_replace_text(path: Path, value: str) -> None:
    """Atomically update a mutable labeling task or regenerated prediction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-",
        suffix=path.suffix or ".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def runtime_head(
    runtime: Mapping[str, Any],
    key: str,
    prediction_task: str,
) -> LogisticModel:
    feature_config = FeatureConfig.from_dict(runtime["featureConfig"])
    head = runtime[key]
    if not isinstance(head, Mapping):
        raise ValueError(f"runtime head is malformed: {key}")
    rally_decoder = runtime["rally"]["decoder"]
    training: dict[str, Any] = {}
    if key == "serve":
        training = {
            "serveDecoder": head["decoder"],
            "composition": head["composition"],
        }
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


def production_heads(runtime: Mapping[str, Any]) -> ProductionHeads:
    return ProductionHeads(
        rally=runtime_head(runtime, "rally", RALLY_LIVE_TASK),
        serve=runtime_head(runtime, "serve", SERVE_CONTACT_TASK),
        dead=runtime_head(runtime, "deadState", DEAD_STATE_TASK),
    )


def tied_ranks(values: np.ndarray) -> np.ndarray:
    rows, columns = values.shape
    result = np.zeros_like(values, dtype=np.float64)
    if rows == 1:
        result.fill(0.5)
        return result
    for column in range(columns):
        order = np.argsort(values[:, column], kind="stable")
        sorted_values = values[order, column]
        start = 0
        while start < rows:
            end = start + 1
            while end < rows and sorted_values[end] == sorted_values[start]:
                end += 1
            rank = ((start + end - 1) / 2) / (rows - 1)
            result[order[start:end], column] = rank
            start = end
    return result


def logistic_probability(features: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    impute = np.asarray(model["impute"], dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    filled = np.where(np.isfinite(features), features, impute)
    logits = np.clip(((filled - mean) / scale) @ weights + float(model["bias"]), -30, 30)
    return 1 / (1 + np.exp(-logits))


def clamped_time(value: float, duration: float) -> float:
    return min(max(0.0, value), max(0.0, duration - 0.01))


def resized_frame(
    capture: cv2.VideoCapture,
    time: float,
    duration: float,
    roi: tuple[float, float, float, float] | None,
    size: tuple[int, int],
) -> np.ndarray:
    frame = crop_roi(read_frame(capture, clamped_time(time, duration)), roi)
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def serving_markers(
    capture: cv2.VideoCapture,
    prepared: PreparedRecording,
    trace: ProductionTrace,
    components: Sequence[ProductionComponent],
    runtime: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    duration = prepared.sequence.metadata.duration
    roi = prepared.recording.roi
    raw_rows: list[np.ndarray] = []
    for index, component in enumerate(components, start=1):
        court_frames = [
            resized_frame(capture, component.start + offset, duration, roi, SERVING_RESIZE)
            for offset in COURT_OFFSETS
        ]
        masks, _ = service_zone_masks(
            SERVING_RESIZE[1],
            SERVING_RESIZE[0],
            roi=roi or (0.0, 0.0, 1.0, 1.0),
            court_geometry=None,
        )
        court = extract_window_features(court_frames, masks)
        flight_frames = [
            resized_frame(capture, component.start + offset, duration, roi, SERVING_RESIZE)
            for offset in FLIGHT_OFFSETS
        ]
        flight = extract_flight_features(flight_frames, 4, 6)
        raw_rows.append(
            np.asarray(
                [*[court[name] for name in COURT_FEATURE_NAMES],
                 *[flight[name] for name in flight_feature_names(4, 6)]],
                dtype=np.float64,
            )
        )
        if index % 20 == 0 or index == len(components):
            print(f"  serving side {index}/{len(components)}", flush=True)
    raw = np.stack(raw_rows) if raw_rows else np.empty((0, 237), dtype=np.float64)
    ranked = tied_ranks(raw) if len(raw) else raw
    probabilities = logistic_probability(ranked, runtime["model"]) if len(ranked) else np.empty(0)
    far_review = float(runtime["reviewBand"]["farUpperExclusive"])
    near_review = float(runtime["reviewBand"]["nearLowerInclusive"])
    side_threshold = float(runtime["sideThreshold"])
    serve_threshold = float(runtime["gate"]["serveHeadThreshold"])
    markers: list[dict[str, Any]] = []
    for index, (component, probability) in enumerate(
        zip(components, probabilities, strict=True), start=1
    ):
        anchor = component.start
        selected = np.abs(trace.times - anchor) <= 1.0 + 1e-9
        if not np.any(selected):
            selected[int(np.argmin(np.abs(trace.times - anchor)))] = True
        head_peaks = [
            float(np.max(np.asarray(trace.serve_scores[source])[selected]))
            for source in ("all-labels-v2", "previous-production")
        ]
        recovered = len(component.sources) == 2
        if max(head_peaks) < serve_threshold and not recovered:
            continue
        model_side = "near" if probability >= side_threshold else "far"
        needs_review = far_review <= probability < near_review or max(head_peaks) < serve_threshold
        markers.append(
            {
                "time": round(anchor, 3),
                "side": "review" if needs_review else model_side,
                "origin": "model",
                "modelSide": model_side,
                "modelConfidence": round(float(probability), 8),
                "modelId": str(runtime["modelId"]),
                "rallyId": f"R{index:03d}",
            }
        )
    return markers, raw


def sample_times(start: float, end: float) -> list[float]:
    duration = end - start
    first = start + min(0.2, duration * 0.08)
    last = end - min(0.15, duration * 0.08)
    if last <= first:
        first = start + duration * 0.2
        last = start + duration * 0.8
    return [first + ((last - first) * index) / 6 for index in range(7)]


def switch_candidates(
    trace: ProductionTrace,
    components: Sequence[ProductionComponent],
    runtime: Mapping[str, Any],
) -> list[SwitchCandidate]:
    candidates: list[SwitchCandidate] = []
    for before, after in zip(components, components[1:], strict=False):
        # Production components can touch after the replay decoder merges its
        # overlapping model ranges. A boundary without positive dead time is
        # not a valid gap-state candidate.
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
        for time, index in sorted(selected):
            candidates.append(
                SwitchCandidate(
                    event_id=f"switch:internal-dead-peak:{component.start:.3f}:{round(time * 1000)}",
                    kind="internal-dead-state-peak",
                    transition_time=time,
                    gap_start=max(component.start, time - half_width),
                    gap_end=min(component.end, time + half_width),
                    generator_score=float(dead[index]),
                    source_ranges=(component,),
                    before_window=(time - 4, time - 1),
                    after_window=(time + 1, time + 4),
                )
            )
    return sorted(candidates, key=lambda item: (item.transition_time, item.kind, item.event_id))


def component_range(component: ProductionComponent) -> TimeRange:
    return TimeRange(component.start, component.end)


def switch_markers(
    capture: cv2.VideoCapture,
    prepared: PreparedRecording,
    trace: ProductionTrace,
    components: Sequence[ProductionComponent],
    runtime: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    duration = prepared.sequence.metadata.duration
    roi = prepared.recording.roi
    candidates = switch_candidates(trace, components, runtime)
    if not candidates:
        return [], np.empty((0, 34)), np.empty(0)
    calibration = [
        resized_frame(capture, time, duration, roi, (FRAME_WIDTH, FRAME_HEIGHT))
        for component in components[:7]
        for time in sample_times(component.start, component.end)
    ]
    geometry = estimate_court_geometry(calibration)
    rows: list[np.ndarray] = []
    for index, candidate in enumerate(candidates, start=1):
        before_frames = [
            resized_frame(capture, time, duration, roi, (FRAME_WIDTH, FRAME_HEIGHT))
            for time in sample_times(*candidate.before_window)
        ]
        after_frames = [
            resized_frame(capture, time, duration, roi, (FRAME_WIDTH, FRAME_HEIGHT))
            for time in sample_times(*candidate.after_window)
        ]
        before_broad = summarize_sequence(before_frames, geometry)
        after_broad = summarize_sequence(after_frames, geometry)
        before_players = summarize_player_sequence(before_frames, geometry)
        after_players = summarize_player_sequence(after_frames, geometry)
        visual = player_features(
            before_players,
            after_players,
            visual_features(before_broad, after_broad),
        )
        before_evidence = rally_evidence(component_range(candidate.source_ranges[0]), trace)
        after_evidence = rally_evidence(component_range(candidate.source_ranges[-1]), trace)
        state, _ = gap_state_features(
            before_evidence,
            after_evidence,
            candidate.gap_start,
            candidate.gap_end,
            trace,
        )
        rows.append(
            np.asarray(
                [
                    *[visual[name] for name in VISUAL_FEATURE_NAMES],
                    *list(state.values())[:10],
                    1.0 if candidate.kind == "internal-dead-state-peak" else 0.0,
                    candidate.generator_score,
                ],
                dtype=np.float64,
            )
        )
        if index % 20 == 0 or index == len(candidates):
            print(f"  side switches {index}/{len(candidates)}", flush=True)
    features = np.stack(rows)
    probabilities = logistic_probability(features, runtime["classifier"])
    ranked = sorted(
        range(len(candidates)),
        key=lambda index: (
            -probabilities[index],
            candidates[index].transition_time,
            candidates[index].event_id,
        ),
    )
    chronological = sorted(
        range(len(candidates)),
        key=lambda index: (candidates[index].transition_time, candidates[index].event_id),
    )
    ordinal = {candidate_index: order for order, candidate_index in enumerate(chronological)}
    decoder = runtime["decoder"]
    threshold = float(runtime["classifier"]["threshold"])
    threshold_logit = math.log(threshold / (1 - threshold))
    selected: list[int] = []
    for index in ranked:
        if any(
            (
                float(decoder["minimumCandidateIndexSeparation"]) > 0
                and abs(ordinal[index] - ordinal[other])
                < float(decoder["minimumCandidateIndexSeparation"])
            )
            or (
                float(decoder["minimumTimeSeparationSeconds"]) > 0
                and abs(candidates[index].transition_time - candidates[other].transition_time)
                < float(decoder["minimumTimeSeparationSeconds"])
            )
            for other in selected
        ):
            continue
        score = float(np.clip(probabilities[index], 1e-9, 1 - 1e-9))
        excess = max(0.0, len(selected) + 1 - float(decoder["freePredictionsPerRecording"]))
        margin = (
            math.log(score / (1 - score))
            - threshold_logit
            - float(decoder["countPenaltyLogitPerExcessPrediction"]) * excess
        )
        if margin >= -1e-12:
            selected.append(index)
    selected.sort(key=lambda index: (candidates[index].transition_time, candidates[index].event_id))
    markers = [
        {
            "time": round(candidates[index].transition_time, 3),
            "origin": "model",
            "modelConfidence": round(float(probabilities[index]), 8),
            "modelId": str(runtime["modelId"]),
            "modelEventId": candidates[index].event_id,
        }
        for index in selected
    ]
    return markers, features, probabilities


def task_path(workspace: Path, recording_id: str) -> Path:
    return workspace / "tasks/full" / f"{recording_id}.labels.json"


def infer_recording(
    prepared: PreparedRecording,
    old_heads: ProductionHeads,
    new_heads: ProductionHeads,
    serving_runtime: Mapping[str, Any],
    switch_runtime: Mapping[str, Any],
    workspace: Path,
    force: bool,
) -> dict[str, Any]:
    output = workspace / "predictions/score-markers" / f"{prepared.recording.id}.json"
    if output.exists() and not force:
        print(f"{prepared.recording.id}: score markers already exist; skipping", flush=True)
        return dict(load_json(output))
    expected_names = tuple(str(name) for name in new_heads.rally.feature_names)
    if prepared.contextual_names != expected_names:
        raise ValueError(f"{prepared.recording.id}: frozen production feature signature differs")
    trace = replay_trace(prepared, old_heads, new_heads)
    components = merge_production_components(trace.ranges)
    capture = cv2.VideoCapture(str(prepared.recording.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {prepared.recording.video}")
    try:
        serves, serving_features = serving_markers(
            capture, prepared, trace, components, serving_runtime
        )
        switches, switch_features, switch_probabilities = switch_markers(
            capture, prepared, trace, components, switch_runtime
        )
    finally:
        capture.release()
    created_at = datetime.now(UTC).isoformat()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-labeling-score-marker-inference-v1",
        "createdAt": created_at,
        "recordingId": prepared.recording.id,
        "rallyModel": {
            "modelId": "ensemble-overlap-union-model-1ca43e38eefc-model-9c92b8e9333f",
            "components": len(components),
        },
        "servingSideModel": {
            "modelId": serving_runtime["modelId"],
            "fingerprint": serving_runtime["fingerprint"],
            "featureVersion": serving_runtime["featureVersion"],
            "featureRows": len(serving_features),
            "featureColumns": serving_features.shape[1] if serving_features.ndim == 2 else 0,
        },
        "sideSwitchModel": {
            "modelId": switch_runtime["modelId"],
            "fingerprint": switch_runtime["fingerprint"],
            "featureVersion": switch_runtime["featureVersion"],
            "featureRows": len(switch_features),
            "featureColumns": switch_features.shape[1] if switch_features.ndim == 2 else 0,
            "candidateProbabilities": switch_probabilities.tolist(),
        },
        "serveMarkers": serves,
        "sideSwitches": switches,
        "rallies": [
            {
                "start": round(component.start, 3),
                "end": round(component.end, 3),
                "tags": [
                    "ai-prelabel",
                    f"model-agreement:{'both-models' if len(component.sources) == 2 else 'all-labels-v2-only' if component.sources == ('all-labels-v2',) else 'previous-production-only'}",
                ],
            }
            for component in components
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized_payload = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if force:
        atomic_replace_text(output, serialized_payload)
    else:
        atomic_write_text(output, serialized_payload)

    labels_path = task_path(workspace, prepared.recording.id)
    labels = dict(load_json(labels_path))
    labels["annotation"] = {
        **labels["annotation"],
        "status": "in-progress",
        "reviewedAt": None,
        "notes": labels["annotation"].get("notes")
        or "Initialized from frozen production rally, serving-side, and side-switch models. Validate every marker and the full recording.",
    }
    labels["prelabel"] = {
        "analysisMethod": "frozen-production-rally-and-score-specialists-v1",
        "candidateFile": str(output),
        "analyzedAt": created_at,
        "ambiguities": [
            "All model rallies, serving sides, and side switches are unvalidated starting labels.",
            "Review markers labeled 'review', correct sides and timestamps, add misses, and delete false positives.",
        ],
    }
    labels["rallies"] = payload["rallies"]
    labels["serveMarkers"] = serves
    labels["sideSwitches"] = switches
    atomic_replace_text(
        labels_path,
        json.dumps(labels, indent=2, allow_nan=False) + "\n",
    )
    load_label_document(labels_path, require_complete=False, require_video=True)
    print(
        f"{prepared.recording.id}: {len(components)} rallies, "
        f"{len(serves)} serve markers, {len(switches)} side switches",
        flush=True,
    )
    return payload


def run(args: argparse.Namespace) -> None:
    workspace = args.workspace.resolve()
    runtime_root = args.runtime_root.resolve()
    manifest = load_manifest(args.manifest.resolve(), require_videos=True)
    all_labels_runtime = load_json(runtime_root / ALL_LABELS_FILENAME)
    previous_runtime = load_json(runtime_root / PREVIOUS_FILENAME)
    serving_runtime = load_json(runtime_root / SERVING_MODEL_FILENAME)
    switch_runtime = load_json(runtime_root / SWITCH_MODEL_FILENAME)
    new_heads = production_heads(all_labels_runtime)
    old_heads = production_heads(previous_runtime)
    feature_config = new_heads.rally.feature_config
    if feature_config.to_dict() != old_heads.rally.feature_config.to_dict():
        raise ValueError("production runtimes do not share a feature configuration")
    requested_ids = set(args.recording_id or ())
    recordings = [
        recording
        for recording in manifest.recordings
        if not requested_ids or recording.id in requested_ids
    ]
    missing_ids = requested_ids - {recording.id for recording in recordings}
    if missing_ids:
        raise ValueError(f"recording ids are absent from the manifest: {sorted(missing_ids)}")
    results = []
    for recording in recordings:
        print(f"{recording.id}: loading cached production features", flush=True)
        prepared = prepare_recording(recording, feature_config, args.cache_dir.resolve())
        results.append(
            infer_recording(
                prepared,
                old_heads,
                new_heads,
                serving_runtime,
                switch_runtime,
                workspace,
                args.force,
            )
        )
    summary = {
        "recordings": len(results),
        "rallies": sum(len(result["rallies"]) for result in results),
        "serveMarkers": sum(len(result["serveMarkers"]) for result in results),
        "sideSwitches": sum(len(result["sideSwitches"]) for result in results),
    }
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    value.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_WORKSPACE / "manifests/inference-only.json",
    )
    value.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_WORKSPACE / "features/audiovisual-audio-normalized-v3",
    )
    value.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    value.add_argument(
        "--recording-id",
        action="append",
        help="only infer the named recording; repeat to select multiple recordings",
    )
    value.add_argument("--force", action="store_true")
    return value


if __name__ == "__main__":
    run(parser().parse_args())
