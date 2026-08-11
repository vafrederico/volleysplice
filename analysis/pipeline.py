from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .artifacts import atomic_write_text
from .config import DecoderConfig, FeatureConfig, TrainingConfig
from .decoder import DecodedInterval, decode_probabilities
from .features import (
    FeatureSequence,
    cached_features,
    camera_warnings,
    contextualize,
    extract_features,
    probe_video,
    write_preview,
)
from .metrics import (
    aggregate_evaluations,
    aggregate_interval_selection,
    aggregate_outcome_slices,
    evaluate_interval_selection,
    evaluate_intervals,
    outcome_slice_metrics,
)
from .model import LogisticModel, ModelError, load_model, train_logistic_model
from .schema import (
    DatasetManifest,
    Interval,
    ManifestError,
    Recording,
    labels_for_times,
    load_manifest,
    mask_for_times,
)
from .version import __version__


@dataclass(frozen=True)
class PreparedRecording:
    recording: Recording
    sequence: FeatureSequence
    contextual_values: np.ndarray
    contextual_names: tuple[str, ...]
    labels: np.ndarray
    sample_mask: np.ndarray


def _manifest_digest(manifest: DatasetManifest) -> str:
    snapshots: list[dict[str, Any]] = []
    for recording in manifest.recordings:
        stat = recording.video.stat()
        sidecar = recording.video.with_suffix(recording.video.suffix + ".provenance.json")
        snapshots.append(
            {
                "id": recording.id,
                "size": stat.st_size,
                "mtimeNs": stat.st_mtime_ns,
                "contentSha256": recording.content_sha256,
                "provenanceSha256": (
                    hashlib.sha256(sidecar.read_bytes()).hexdigest() if sidecar.is_file() else None
                ),
            }
        )
    canonical = json.dumps(
        {"manifest": manifest.raw, "videoSnapshots": snapshots},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prepare_recording(
    recording: Recording,
    feature_config: FeatureConfig,
    cache_dir: str | Path,
) -> PreparedRecording:
    metadata = probe_video(recording.video)
    if feature_config.analysis_fps > metadata.fps + 1e-6:
        raise ManifestError(
            f"{recording.id}: analysis FPS {feature_config.analysis_fps:g} exceeds "
            f"source FPS {metadata.fps:g}"
        )
    if recording.rallies and recording.rallies[-1].end > metadata.duration + 1e-6:
        raise ManifestError(
            f"{recording.id}: final annotation ends at {recording.rallies[-1].end:.3f}s, "
            f"after video duration {metadata.duration:.3f}s"
        )
    sequence = cached_features(
        recording.id,
        recording.video,
        feature_config,
        recording.roi,
        cache_dir,
        content_sha256=recording.content_sha256,
    )
    values, names = contextualize(sequence, feature_config)
    return PreparedRecording(
        recording=recording,
        sequence=sequence,
        contextual_values=values,
        contextual_names=names,
        labels=labels_for_times(sequence.times, recording.rallies),
        sample_mask=mask_for_times(sequence.times, recording.ignored_intervals),
    )


def _prepare_many(
    recordings: Sequence[Recording],
    feature_config: FeatureConfig,
    cache_dir: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> list[PreparedRecording]:
    prepared: list[PreparedRecording] = []
    for index, recording in enumerate(recordings, start=1):
        if progress is not None:
            progress(f"Preparing features {index}/{len(recordings)}: {recording.id}")
        prepared.append(prepare_recording(recording, feature_config, cache_dir))
    return prepared


def _evaluate_prepared(
    prepared: Sequence[PreparedRecording],
    model: LogisticModel,
    decoder: DecoderConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probabilities = [model.predict(item.contextual_values) for item in prepared]
    return _evaluate_prepared_probabilities(prepared, probabilities, decoder)


def _evaluate_prepared_probabilities(
    prepared: Sequence[PreparedRecording],
    probabilities: Sequence[np.ndarray],
    decoder: DecoderConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(prepared) != len(probabilities):
        raise ModelError("prepared recordings and probability sequences must be aligned")
    per_recording: list[dict[str, Any]] = []
    for item, item_probabilities in zip(prepared, probabilities, strict=True):
        predictions, _ = decode_probabilities(
            item.sequence.times,
            item_probabilities,
            item.sequence.metadata.duration,
            decoder,
            1.0 / float(np.median(np.diff(item.sequence.times)))
            if len(item.sequence.times) > 1
            else 1.0,
        )
        scored_predictions = [
            Interval(start=prediction.start, end=prediction.end) for prediction in predictions
        ]
        for ignored in item.recording.ignored_intervals:
            fragments: list[Interval] = []
            for prediction in scored_predictions:
                if prediction.end <= ignored.start or prediction.start >= ignored.end:
                    fragments.append(prediction)
                    continue
                if prediction.start < ignored.start:
                    fragments.append(Interval(prediction.start, ignored.start))
                if prediction.end > ignored.end:
                    fragments.append(Interval(ignored.end, prediction.end))
            scored_predictions = fragments
        metrics = evaluate_intervals(item.recording.rallies, scored_predictions)
        metrics["outcomeSlices"] = outcome_slice_metrics(
            item.recording.rallies, scored_predictions
        )
        metrics["id"] = item.recording.id
        metrics["sourceGroup"] = item.recording.source_group
        metrics["environment"] = item.recording.environment
        metrics["playersPerTeam"] = item.recording.game.get("playersPerTeam")
        metrics["targetPoints"] = item.recording.game.get("targetPoints")
        capture = item.recording.capture
        metrics["captureProfile"] = (
            "supported"
            if item.recording.roi is not None
            and capture.get("stationary") is True
            and capture.get("fullCourtVisible") is True
            and capture.get("serviceAreasVisible") is True
            and capture.get("position") == "centered-behind-endline"
            else "unknown-or-unsupported"
        )
        per_recording.append(metrics)
    aggregate = aggregate_evaluations(per_recording)
    aggregate["outcomeSlices"] = aggregate_outcome_slices(
        [item["outcomeSlices"] for item in per_recording]
    )
    return per_recording, aggregate


def _evaluate_prepared_probabilities_for_selection(
    prepared: Sequence[PreparedRecording],
    probabilities: Sequence[np.ndarray],
    decoder: DecoderConfig,
) -> dict[str, float | int]:
    """Evaluate exactly the fields used by decoder selection, and nothing else."""
    if len(prepared) != len(probabilities):
        raise ModelError("prepared recordings and probability sequences must be aligned")
    per_recording: list[dict[str, float | int]] = []
    for item, item_probabilities in zip(prepared, probabilities, strict=True):
        predictions, _ = decode_probabilities(
            item.sequence.times,
            item_probabilities,
            item.sequence.metadata.duration,
            decoder,
            1.0 / float(np.median(np.diff(item.sequence.times)))
            if len(item.sequence.times) > 1
            else 1.0,
        )
        scored_predictions = [
            Interval(start=prediction.start, end=prediction.end)
            for prediction in predictions
        ]
        for ignored in item.recording.ignored_intervals:
            fragments: list[Interval] = []
            for prediction in scored_predictions:
                if prediction.end <= ignored.start or prediction.start >= ignored.end:
                    fragments.append(prediction)
                    continue
                if prediction.start < ignored.start:
                    fragments.append(Interval(prediction.start, ignored.start))
                if prediction.end > ignored.end:
                    fragments.append(Interval(ignored.end, prediction.end))
            scored_predictions = fragments
        per_recording.append(
            evaluate_interval_selection(item.recording.rallies, scored_predictions)
        )
    return aggregate_interval_selection(per_recording)


def _tune_decoder(
    prepared: Sequence[PreparedRecording],
    model: LogisticModel,
    base: DecoderConfig,
) -> tuple[DecoderConfig, dict[str, Any]]:
    if not prepared:
        return base, {"status": "skipped", "reason": "no validation recordings"}
    best = base
    best_metrics: dict[str, Any] | None = None
    best_objective = -1.0
    best_tie_break: tuple[float, ...] = (-1.0,)
    candidate_count = 0
    probabilities = [model.predict(item.contextual_values) for item in prepared]

    def consider(candidate: DecoderConfig) -> None:
        nonlocal best, best_metrics, best_objective, best_tie_break, candidate_count
        candidate_count += 1
        aggregate = _evaluate_prepared_probabilities_for_selection(
            prepared, probabilities, candidate
        )
        objective = (
            0.55 * aggregate["eventF1"]
            + 0.30 * aggregate["timeIoU"]
            + 0.15 * aggregate["liveTimeRecall"]
        )
        tie_break = (
            aggregate["eventF1"],
            aggregate["liveTimePrecision"],
            -abs(aggregate["predictedRallies"] - aggregate["trueRallies"]),
            aggregate["timeIoU"],
            -abs(candidate.enter_threshold - base.enter_threshold),
        )
        if objective > best_objective + 1e-9 or (
            abs(objective - best_objective) <= 1e-9 and tie_break > best_tie_break
        ):
            best_objective = objective
            best_tie_break = tie_break
            best = candidate
            best_metrics = aggregate

    for smoothing_seconds in (0.5, 1.0, 1.5, 2.0):
        for threshold in np.arange(0.35, 0.851, 0.05):
            for exit_delta in (0.05, 0.10, 0.15):
                for min_live_seconds in (1.0, 2.0, 3.0):
                    for bridge_gap_seconds in (0.0, 0.5, 1.0, 1.5):
                        enter_threshold = float(round(threshold, 2))
                        candidate = replace(
                            base,
                            smoothing_seconds=smoothing_seconds,
                            enter_threshold=enter_threshold,
                            exit_threshold=float(round(enter_threshold - exit_delta, 2)),
                            min_live_seconds=min_live_seconds,
                            bridge_gap_seconds=bridge_gap_seconds,
                            short_event_min_seconds=min_live_seconds,
                            short_event_threshold=1.0,
                        )
                        consider(candidate)

    base_selected = best
    short_candidates = {
        (base_selected.min_live_seconds, 1.0),
        *{
            (minimum, threshold)
            for minimum in (0.25, 0.5, 0.75, 1.0)
            for threshold in (0.7, 0.8, 0.9, 1.0)
            if minimum <= base_selected.min_live_seconds
            and threshold >= base_selected.enter_threshold
        },
    }
    for minimum, threshold in sorted(short_candidates):
        consider(
            replace(
                base_selected,
                short_event_min_seconds=minimum,
                short_event_threshold=threshold,
            )
        )
    _, best_metrics = _evaluate_prepared_probabilities(prepared, probabilities, best)
    return best, {
        "status": "selected-on-validation",
        "objective": "0.55*eventF1 + 0.30*timeIoU + 0.15*liveTimeRecall",
        "tieBreak": (
            "higher event F1, live-time precision, rally-count proximity, time IoU, "
            f"then nearest base enter threshold ({base.enter_threshold:g})"
        ),
        "objectiveValue": best_objective,
        "candidateCount": candidate_count,
        "selected": best.to_dict(),
        "validationMetrics": best_metrics,
    }


def train_dataset(
    manifest_path: str | Path,
    model_destination: str | Path,
    cache_dir: str | Path,
    *,
    feature_config: FeatureConfig | None = None,
    training_config: TrainingConfig | None = None,
    decoder_config: DecoderConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    model_target = Path(model_destination).expanduser().resolve()
    if model_target.exists() and (not model_target.is_dir() or any(model_target.iterdir())):
        raise ModelError(f"model destination is not an empty directory: {model_target}")
    feature_config = feature_config or FeatureConfig()
    training_config = training_config or TrainingConfig()
    decoder_config = decoder_config or DecoderConfig()
    feature_config.validate()
    training_config.validate()
    decoder_config.validate()
    manifest = load_manifest(manifest_path)
    train_rows = manifest.for_split("train")
    validation_rows = manifest.for_split("validation")
    if not train_rows:
        raise ManifestError("manifest has no training recordings")
    if not validation_rows:
        raise ManifestError("manifest has no validation recordings; decoder selection must not use test data")
    training_started = time.perf_counter()
    if progress is None:
        training = _prepare_many(train_rows, feature_config, cache_dir)
        validation = _prepare_many(validation_rows, feature_config, cache_dir)
    else:
        training = _prepare_many(train_rows, feature_config, cache_dir, progress=progress)
        validation = _prepare_many(validation_rows, feature_config, cache_dir, progress=progress)
    validation_live = sum(
        float(np.sum(item.labels[item.sample_mask] > 0.5)) for item in validation
    )
    validation_samples = sum(int(np.sum(item.sample_mask)) for item in validation)
    if validation_live == 0 or validation_live == validation_samples:
        raise ManifestError("validation data must contain both live and dead samples")
    signature = training[0].contextual_names
    for item in (*training, *validation):
        if item.contextual_names != signature:
            raise ModelError("extracted feature signatures differ between recordings")
    if progress is not None:
        progress("Fitting the class-weighted temporal logistic model")
    model = train_logistic_model(
        [item.contextual_values[item.sample_mask] for item in training],
        [item.labels[item.sample_mask] for item in training],
        [item.contextual_values[item.sample_mask] for item in validation],
        [item.labels[item.sample_mask] for item in validation],
        feature_config,
        signature,
        decoder_config,
        training_config,
    )
    if progress is not None:
        progress("Selecting interval decoder parameters on validation data")
    decoder, selection = _tune_decoder(validation, model, decoder_config)
    model.decoder = decoder
    model.training_summary.update(
        {
            "dataset": manifest.name,
            "manifestSha256": _manifest_digest(manifest),
            "trainingRecordingIds": [item.recording.id for item in training],
            "validationRecordingIds": [item.recording.id for item in validation],
            "trainingSourceGroups": sorted({item.recording.source_group for item in training}),
            "validationSourceGroups": sorted({item.recording.source_group for item in validation}),
            "recordingContentSha256": {
                item.recording.id: item.recording.content_sha256
                for item in (*training, *validation)
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": __import__("cv2").__version__,
                "wallClockSeconds": round(time.perf_counter() - training_started, 3),
            },
            "decoderSelection": selection,
        }
    )
    _, validation_metrics = _evaluate_prepared(validation, model, decoder)
    model_dir = model.save(model_target)
    return {
        "model": str(model_dir),
        "trainingRecordings": len(training),
        "validationRecordings": len(validation),
        "bestEpoch": model.training_summary["bestEpoch"],
        "decoder": decoder.to_dict(),
        "validation": validation_metrics,
    }


def infer_video(
    video_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    *,
    roi: tuple[float, float, float, float] | None = None,
    title: str | None = None,
    capture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inference_started = time.perf_counter()
    model = load_model(model_path)
    video = Path(video_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    analysis_path = destination / "analysis.json"
    preview_path = destination / "court-preview.jpg"
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ModelError(f"analysis output is not an empty directory: {destination}")
    sequence = extract_features(video, model.feature_config, roi)
    values, names = contextualize(sequence, model.feature_config)
    if names != model.feature_names:
        raise ModelError("inference extractor signature does not match the model")
    probabilities = model.predict(values)
    intervals, smoothed = decode_probabilities(
        sequence.times,
        probabilities,
        sequence.metadata.duration,
        model.decoder,
        model.feature_config.analysis_fps,
    )
    dynamic_name = "diff_active_fraction"
    dynamic_index = sequence.names.index(dynamic_name) if dynamic_name in sequence.names else None
    warnings = camera_warnings(sequence.metadata, capture)
    if roi is None:
        warnings.append("no court ROI supplied; full-frame motion may include spectators or adjacent courts")
    roi_payload = None
    if roi is not None:
        roi_payload = {"x": roi[0], "y": roi[1], "width": roi[2], "height": roi[3]}
    created_at = datetime.now(timezone.utc).isoformat()
    processing_seconds = time.perf_counter() - inference_started
    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "id": destination.name,
        "title": title or video.stem,
        "createdAt": created_at,
        "source": {
            "filename": video.name,
            **sequence.metadata.to_dict(),
        },
        "assets": {"courtPreviewPath": "court-preview.jpg"},
        "analysis": {
            "method": "court-motion-temporal-logistic-v0",
            "modelVersion": Path(model_path).expanduser().resolve().name,
            "producer": f"volleycut-analysis/{__version__}",
            "analysisFps": model.feature_config.analysis_fps,
            "featureConfig": model.feature_config.to_dict(),
            "decoder": model.decoder.to_dict(),
            "modelSha256": model.artifact_sha256,
            "processingSeconds": round(processing_seconds, 3),
            "processingToVideoRatio": (
                processing_seconds / sequence.metadata.duration
                if sequence.metadata.duration > 0
                else None
            ),
            "warnings": warnings,
            "court": {
                "source": "manual-roi" if roi is not None else "full-frame-fallback",
                "roi": roi_payload,
            },
        },
        "rallies": [
            {
                "id": f"R{index:03d}",
                "start": max(0.0, round(interval.start, 3)),
                "end": min(sequence.metadata.duration, round(interval.end, 3)),
                "confidence": round(interval.confidence, 5),
                "included": True,
                "evidence": {"meanLiveProbability": round(interval.confidence, 5)},
            }
            for index, interval in enumerate(intervals, start=1)
        ],
        "signals": [
            {
                "time": round(float(time), 3),
                "liveProbability": round(float(probability), 5),
                **(
                    {"motion": round(float(sequence.values[index, dynamic_index]), 5)}
                    if dynamic_index is not None
                    else {}
                ),
            }
            for index, (time, probability) in enumerate(zip(sequence.times, smoothed, strict=True))
        ],
    }
    payload_text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    created_directory = not destination.exists()
    destination.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_preview_name = tempfile.mkstemp(
        prefix=".court-preview-", suffix=".jpg", dir=destination
    )
    os.close(descriptor)
    temporary_preview = Path(temporary_preview_name)
    temporary_preview.unlink()
    try:
        write_preview(video, temporary_preview, roi)
        temporary_preview.replace(preview_path)
        try:
            atomic_write_text(analysis_path, payload_text)
        except Exception:
            preview_path.unlink(missing_ok=True)
            raise
    except Exception:
        temporary_preview.unlink(missing_ok=True)
        analysis_path.unlink(missing_ok=True)
        if created_directory:
            try:
                destination.rmdir()
            except OSError:
                pass
        raise
    return payload


def evaluate_dataset(
    manifest_path: str | Path,
    model_path: str | Path,
    cache_dir: str | Path,
    *,
    split: str = "test",
    output_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    evaluation_started = time.perf_counter()
    if output_path is not None and Path(output_path).expanduser().resolve().exists():
        raise ModelError(f"evaluation output already exists: {Path(output_path).expanduser().resolve()}")
    model = load_model(model_path)
    manifest = load_manifest(manifest_path)
    current_digest = _manifest_digest(manifest)
    training_digest = model.training_summary.get("manifestSha256")
    if training_digest != current_digest:
        raise ModelError(
            "evaluation manifest differs from the immutable manifest recorded during training"
        )
    recordings = manifest.for_split(split)
    if not recordings:
        raise ManifestError(f"manifest has no recordings in split {split!r}")
    protected_groups = set(model.training_summary.get("trainingSourceGroups", []))
    if split != "validation":
        protected_groups |= set(model.training_summary.get("validationSourceGroups", []))
    overlap = protected_groups & {recording.source_group for recording in recordings}
    if overlap:
        raise ModelError(f"evaluation split leaks trained/tuned source groups: {sorted(overlap)}")
    prepared = (
        _prepare_many(recordings, model.feature_config, cache_dir)
        if progress is None
        else _prepare_many(
            recordings,
            model.feature_config,
            cache_dir,
            progress=progress,
        )
    )
    for item in prepared:
        if item.contextual_names != model.feature_names:
            raise ModelError(f"feature signature mismatch for {item.recording.id}")
    per_recording, aggregate = _evaluate_prepared(prepared, model, model.decoder)
    environments = {
        environment: aggregate_evaluations(
            [item for item in per_recording if item["environment"] == environment]
        )
        for environment in sorted({item["environment"] for item in per_recording})
    }
    capture_profiles = {
        profile: aggregate_evaluations(
            [item for item in per_recording if item["captureProfile"] == profile]
        )
        for profile in sorted({item["captureProfile"] for item in per_recording})
    }
    source_groups = {
        source_group: aggregate_evaluations(
            [item for item in per_recording if item["sourceGroup"] == source_group]
        )
        for source_group in sorted({item["sourceGroup"] for item in per_recording})
    }

    def grouped_metrics(key: str) -> dict[str, Any]:
        values = {item[key] for item in per_recording}
        return {
            "unknown" if value is None else str(value): aggregate_evaluations(
                [item for item in per_recording if item[key] == value]
            )
            for value in sorted(values, key=lambda item: (item is None, item if item is not None else 0))
        }

    processing_seconds = time.perf_counter() - evaluation_started
    video_seconds = sum(item.sequence.metadata.duration for item in prepared)
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "dataset": manifest.name,
        "manifestSha256": current_digest,
        "modelSha256": model.artifact_sha256,
        "split": split,
        "assessmentRole": "tuning-only" if split == "validation" else "held-out-evaluation",
        "decoder": model.decoder.to_dict(),
        "matching": {"minimumIntervalIoU": 0.5},
        "processing": {
            "wallClockSeconds": processing_seconds,
            "videoSeconds": video_seconds,
            "processingToVideoRatio": processing_seconds / video_seconds if video_seconds else None,
            "note": "Feature-cache state affects this end-to-end runtime.",
        },
        "aggregate": aggregate,
        "byEnvironment": environments,
        "bySourceGroup": source_groups,
        "byCaptureProfile": capture_profiles,
        "byPlayersPerTeam": grouped_metrics("playersPerTeam"),
        "byTargetPoints": grouped_metrics("targetPoints"),
        "recordings": per_recording,
    }
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        if destination.exists():
            raise ModelError(f"evaluation output already exists: {destination}")
        atomic_write_text(
            destination,
            json.dumps(report, indent=2, allow_nan=False) + "\n",
        )
    return report


def smoke_test() -> dict[str, Any]:
    """Exercise model fitting, persistence and decoding without real videos or OpenCV."""
    fps = 4.0
    times = np.arange(0, 48, 1 / fps, dtype=np.float64)
    intervals = [(6.0, 12.0), (19.0, 28.0), (35.0, 40.0)]

    def make_sequence(seed_offset: int) -> tuple[np.ndarray, np.ndarray]:
        local = np.random.default_rng(11 + seed_offset)
        labels = np.zeros(len(times), dtype=np.float32)
        for start, end in intervals:
            labels[(times >= start) & (times < end)] = 1.0
        motion = labels + local.normal(0, 0.16, len(times))
        posture = 0.65 * labels + local.normal(0, 0.22, len(times))
        distraction = local.normal(0, 0.35, len(times))
        return np.column_stack((motion, posture, distraction)).astype(np.float32), labels

    train_pairs = [make_sequence(index) for index in range(3)]
    validation_pairs = [make_sequence(8)]
    feature_config = FeatureConfig(use_optical_flow=False, context_offsets_seconds=(0.0,))
    decoder = DecoderConfig(
        smoothing_seconds=0.5,
        enter_threshold=0.5,
        exit_threshold=0.4,
        min_live_seconds=0.5,
        bridge_gap_seconds=0.5,
    )
    model = train_logistic_model(
        [pair[0] for pair in train_pairs],
        [pair[1] for pair in train_pairs],
        [pair[0] for pair in validation_pairs],
        [pair[1] for pair in validation_pairs],
        feature_config,
        ("motion", "posture", "distraction"),
        decoder,
        TrainingConfig(epochs=80, batch_size=128, patience=12, seed=13),
    )
    probabilities = model.predict(validation_pairs[0][0])
    predictions, _ = decode_probabilities(times, probabilities, float(times[-1] + 1 / fps), decoder, fps)

    @dataclass(frozen=True)
    class SyntheticInterval:
        start: float
        end: float

    truth = [SyntheticInterval(start, end) for start, end in intervals]
    metrics = evaluate_intervals(truth, predictions)
    with tempfile.TemporaryDirectory(prefix="volleycut-smoke-") as directory:
        model.save(Path(directory) / "model")
        restored = load_model(Path(directory) / "model")
        parity = bool(np.allclose(probabilities, restored.predict(validation_pairs[0][0]), atol=1e-7))
    passed = metrics["eventRecall"] >= 0.95 and metrics["timeIoU"] >= 0.8 and parity
    return {
        "passed": bool(passed),
        "modelSaveLoadParity": parity,
        "eventRecall": metrics["eventRecall"],
        "timeIoU": metrics["timeIoU"],
        "predictedRallies": metrics["predictedRallies"],
    }
