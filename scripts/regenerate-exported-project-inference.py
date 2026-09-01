#!/usr/bin/env python3
"""Regenerate missing deterministic production features and score traces."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from analysis.exported_project_dataset import (
    load_exported_project_dataset,
    sampled_fingerprint,
    sha256_file,
)
from analysis.features import (
    NVDEC_VIDEO_DECODER,
    OPENCV_VIDEO_DECODER,
    feature_cache_path,
    nvdec_available,
)
from analysis.pipeline import prepare_recording
from analysis.runtime_models import load_production_runtime
from analysis.schema import load_manifest
from analysis.side_switch_production_replay import replay_trace


ALL_LABELS_FILENAME = "model-1ca43e38eefc.json"
PREVIOUS_FILENAME = "model-9c92b8e9333f.json"


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_replace_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=path.suffix or ".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _range_rows(values: Any) -> list[dict[str, float]]:
    return [
        {"start": float(row.start), "end": float(row.end)}
        for row in values
    ]


def _serve_rows(values: Any) -> list[dict[str, float]]:
    return [
        {"time": float(row.time), "confidence": float(row.confidence)}
        for row in values
    ]


def _valid_existing(
    metadata_path: Path,
    trace_path: Path,
    dataset_sha256: str,
    *,
    feature_version: str,
    feature_config: dict[str, Any],
    runtime_sha256: dict[str, str],
    video_decoder: str,
) -> dict[str, Any] | None:
    if not metadata_path.is_file() or not trace_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict) or metadata.get("datasetSha256") != dataset_sha256:
        return None
    if metadata.get("featureVersion") != feature_version:
        return None
    if metadata.get("featureConfig") != feature_config:
        return None
    if metadata.get("videoDecoder") != video_decoder:
        return None
    models = metadata.get("models")
    if not isinstance(models, dict) or any(
        not isinstance(models.get(key), dict)
        or models[key].get("runtimeSha256") != digest
        for key, digest in runtime_sha256.items()
    ):
        return None
    cache_value = metadata.get("featureCachePath")
    cache_digest = metadata.get("featureCacheSha256")
    if not isinstance(cache_value, str) or not isinstance(cache_digest, str):
        return None
    cache_path = Path(cache_value)
    if not cache_path.is_file() or sha256_file(cache_path) != cache_digest:
        return None
    if metadata.get("traceSha256") != sha256_file(trace_path):
        return None
    return metadata


def _resolve_video_decoder(requested: str, source_video: Path) -> tuple[str, dict[str, Any]]:
    if requested == "opencv":
        return OPENCV_VIDEO_DECODER, {
            "requested": requested,
            "available": None,
            "detail": "OpenCV decoder explicitly requested",
        }
    available, detail = nvdec_available(source_video)
    probe = {"requested": requested, "available": available, "detail": detail}
    if available:
        return NVDEC_VIDEO_DECODER, probe
    if requested == "nvdec":
        raise RuntimeError(f"NVDEC was requested but is unavailable: {detail}")
    return OPENCV_VIDEO_DECODER, probe


def _maximum_feature_workers(recordings: int) -> int:
    logical_cpus = os.cpu_count() or 1
    if logical_cpus <= 8:
        return 1
    return min(recordings, logical_cpus - 2)


def _feature_worker_layout(recordings: int, requested_workers: int) -> tuple[int, int, int]:
    logical_cpus = os.cpu_count() or 1
    maximum = _maximum_feature_workers(recordings)
    worker_count = maximum if requested_workers == 0 else min(requested_workers, maximum)
    reserved_cpus = 2 if logical_cpus > 8 and worker_count > 1 else 0
    opencv_threads = max(1, (logical_cpus - reserved_cpus) // worker_count)
    return worker_count, opencv_threads, reserved_cpus


def _decoder_worker_layout(
    recordings: int,
    requested_workers: int,
    video_decoder: str,
) -> tuple[int, int, int]:
    if video_decoder == OPENCV_VIDEO_DECODER:
        return 1, os.cpu_count() or 1, 0
    return _feature_worker_layout(recordings, requested_workers)


def _configure_feature_worker(opencv_threads: int) -> None:
    try:
        import cv2

        cv2.setNumThreads(opencv_threads)
    except ImportError:
        pass


def _regenerate_recording_worker(
    recording: Any,
    feature_config_payload: dict[str, Any],
    feature_version: str,
    cache_dir_value: str,
    output_dir_value: str,
    dataset_path_value: str,
    dataset_sha256: str,
    video_decoder: str,
    new_runtime_path_value: str,
    old_runtime_path_value: str,
    new_runtime_sha256: str,
    old_runtime_sha256: str,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir_value)
    output_dir = Path(output_dir_value)
    dataset_path = Path(dataset_path_value)
    new_runtime_path = Path(new_runtime_path_value)
    old_runtime_path = Path(old_runtime_path_value)
    new_runtime, new_heads = load_production_runtime(new_runtime_path)
    old_runtime, old_heads = load_production_runtime(old_runtime_path)
    feature_config = new_heads.rally.feature_config
    if (
        feature_config.to_dict() != feature_config_payload
        or old_heads.rally.feature_config.to_dict() != feature_config_payload
        or new_heads.rally.feature_version != feature_version
    ):
        raise ValueError("worker production runtime feature contract changed")

    trace_path = output_dir / "traces" / f"{recording.id}.npz"
    metadata_path = output_dir / "records" / f"{recording.id}.json"
    prepared = prepare_recording(
        recording,
        feature_config,
        cache_dir,
        video_decoder=video_decoder,
    )
    trace = replay_trace(prepared, old_heads, new_heads)
    cache_path = feature_cache_path(
        recording.id,
        recording.video,
        feature_config,
        recording.roi,
        cache_dir,
        content_sha256=str(recording.content_sha256),
        video_decoder=video_decoder,
    )
    if not cache_path.is_file():
        raise RuntimeError(f"feature cache was not materialized: {cache_path}")
    _atomic_npz(
        trace_path,
        times=np.asarray(trace.times, dtype=np.float64),
        all_labels_v2_rally=np.asarray(
            trace.rally_scores["all-labels-v2"], dtype=np.float32
        ),
        all_labels_v2_serve=np.asarray(
            trace.serve_scores["all-labels-v2"], dtype=np.float32
        ),
        all_labels_v2_dead_state=np.asarray(
            trace.dead_state_scores["all-labels-v2"], dtype=np.float32
        ),
        previous_production_rally=np.asarray(
            trace.rally_scores["previous-production"], dtype=np.float32
        ),
        previous_production_serve=np.asarray(
            trace.serve_scores["previous-production"], dtype=np.float32
        ),
        previous_production_dead_state=np.asarray(
            trace.dead_state_scores["previous-production"], dtype=np.float32
        ),
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schemaVersion": 1,
        "kind": "volleycut-regenerated-production-inference-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "recordingId": recording.id,
        "datasetPath": str(dataset_path),
        "datasetSha256": dataset_sha256,
        "sourceVideoPath": str(recording.video),
        "sourceVideoSha256": recording.content_sha256,
        "featureVersion": feature_version,
        "featureConfig": feature_config_payload,
        "videoDecoder": video_decoder,
        "featureRows": len(prepared.sequence.times),
        "featureColumns": len(prepared.sequence.names),
        "featureCachePath": str(cache_path),
        "featureCacheSha256": sha256_file(cache_path),
        "tracePath": str(trace_path),
        "traceSha256": sha256_file(trace_path),
        "models": {
            "allLabelsV2": {
                "modelId": new_runtime.get("modelId"),
                "runtimePath": str(new_runtime_path),
                "runtimeSha256": new_runtime_sha256,
            },
            "previousProduction": {
                "modelId": old_runtime.get("modelId"),
                "runtimePath": str(old_runtime_path),
                "runtimeSha256": old_runtime_sha256,
            },
        },
        "decodedRanges": {
            source: _range_rows(trace.ranges[source])
            for source in ("all-labels-v2", "previous-production")
        },
        "serveDetections": {
            source: _serve_rows(trace.serves[source])
            for source in ("all-labels-v2", "previous-production")
        },
        "llmLabelingUsed": False,
    }
    _atomic_replace_text(
        metadata_path,
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
    )
    return metadata


def regenerate(
    dataset_path: Path,
    runtime_root: Path,
    cache_dir: Path,
    output_dir: Path,
    *,
    requested_video_decoder: str = "auto",
    workers: int = 0,
    source_verification: str = "sampled",
) -> dict[str, Any]:
    dataset_path = dataset_path.expanduser().resolve()
    runtime_root = runtime_root.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    dataset = load_exported_project_dataset(dataset_path)
    dataset_sha256 = sha256_file(dataset_path)
    manifest_path = dataset_path.parent / str(dataset["inferenceManifestPath"])
    if sha256_file(manifest_path) != dataset["inferenceManifestSha256"]:
        raise ValueError("inference manifest has changed since dataset import")
    manifest = load_manifest(manifest_path, require_videos=False)
    records_by_id = {str(row["recordingId"]): row for row in dataset["records"]}
    if set(records_by_id) != {recording.id for recording in manifest.recordings}:
        raise ValueError("dataset and inference manifest recording ids differ")
    if workers < 0:
        raise ValueError("workers must be zero (automatic) or positive")
    if source_verification not in {"sampled", "full"}:
        raise ValueError("source verification must be sampled or full")
    worker_count, opencv_threads, reserved_cpus = _feature_worker_layout(
        len(manifest.recordings), workers
    )
    verification_workers = worker_count
    references_by_id = {
        recording_id: json.loads(
            (dataset_path.parent / str(record["referencePath"])).read_text(encoding="utf-8")
        )
        for recording_id, record in records_by_id.items()
    }

    def verify_recording(recording: Any) -> Any:
        if not recording.video.is_file():
            raise FileNotFoundError(f"source video is missing: {recording.video}")
        reference = references_by_id[recording.id]
        expected = str(records_by_id[recording.id]["videoSha256"])
        if recording.video.stat().st_size != int(reference["videoSizeBytes"]):
            raise ValueError(f"{recording.id}: source video size differs from imported dataset")
        if source_verification == "full":
            if sha256_file(recording.video) != expected:
                raise ValueError(
                    f"{recording.id}: source video hash differs from imported dataset"
                )
        elif sampled_fingerprint(recording.video) != reference["sampledFingerprint"]:
            raise ValueError(
                f"{recording.id}: sampled source fingerprint differs from imported dataset"
            )
        return replace(recording, content_sha256=expected)

    print(
        f"verifying {len(manifest.recordings)} source identities via "
        f"{source_verification} checks with {verification_workers} workers",
        flush=True,
    )
    verified_by_id: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=verification_workers) as executor:
        futures = {
            executor.submit(verify_recording, recording): recording.id
            for recording in manifest.recordings
        }
        for future in as_completed(futures):
            recording_id = futures[future]
            verified_by_id[recording_id] = future.result()
            print(f"verified source: {recording_id}", flush=True)
    manifest = replace(
        manifest,
        recordings=tuple(verified_by_id[row.id] for row in manifest.recordings),
    )
    video_decoder, decoder_probe = _resolve_video_decoder(
        requested_video_decoder, manifest.recordings[0].video
    )
    worker_count, opencv_threads, reserved_cpus = _decoder_worker_layout(
        len(manifest.recordings), workers, video_decoder
    )
    try:
        import cv2

        cv2.setNumThreads(opencv_threads)
    except ImportError:
        pass
    print(
        f"decoder={video_decoder} workers={worker_count} "
        f"probe={decoder_probe['detail']}",
        flush=True,
    )

    new_runtime_path = runtime_root / ALL_LABELS_FILENAME
    old_runtime_path = runtime_root / PREVIOUS_FILENAME
    new_runtime_sha256 = sha256_file(new_runtime_path)
    old_runtime_sha256 = sha256_file(old_runtime_path)
    runtime_sha256 = {
        "allLabelsV2": new_runtime_sha256,
        "previousProduction": old_runtime_sha256,
    }
    new_runtime, new_heads = load_production_runtime(new_runtime_path)
    old_runtime, old_heads = load_production_runtime(old_runtime_path)
    if new_heads.rally.feature_config.to_dict() != old_heads.rally.feature_config.to_dict():
        raise ValueError("production runtimes have different feature configurations")
    feature_config = new_heads.rally.feature_config
    feature_config_payload = feature_config.to_dict()
    feature_version = new_heads.rally.feature_version
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_id: dict[str, dict[str, Any]] = {}
    pending = []
    for index, recording in enumerate(manifest.recordings, start=1):
        trace_path = output_dir / "traces" / f"{recording.id}.npz"
        metadata_path = output_dir / "records" / f"{recording.id}.json"
        existing = _valid_existing(
            metadata_path,
            trace_path,
            dataset_sha256,
            feature_version=feature_version,
            feature_config=feature_config_payload,
            runtime_sha256=runtime_sha256,
            video_decoder=video_decoder,
        )
        if existing is not None:
            results_by_id[recording.id] = existing
            continue
        pending.append(recording)
        print(
            f"[{index}/{len(manifest.recordings)}] queued production features: {recording.id}",
            flush=True,
        )

    def submit_arguments(recording: Any) -> tuple[Any, ...]:
        return (
            recording,
            feature_config_payload,
            feature_version,
            str(cache_dir),
            str(output_dir),
            str(dataset_path),
            dataset_sha256,
            video_decoder,
            str(new_runtime_path.resolve()),
            str(old_runtime_path.resolve()),
            new_runtime_sha256,
            old_runtime_sha256,
        )

    if pending and worker_count == 1:
        _configure_feature_worker(opencv_threads)
        for completed, recording in enumerate(pending, start=1):
            results_by_id[recording.id] = _regenerate_recording_worker(
                *submit_arguments(recording)
            )
            print(
                f"[{completed}/{len(pending)}] completed production features: {recording.id}",
                flush=True,
            )
    elif pending:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_configure_feature_worker,
            initargs=(opencv_threads,),
        ) as executor:
            futures = {
                executor.submit(_regenerate_recording_worker, *submit_arguments(recording)): recording.id
                for recording in pending
            }
            completed = 0
            for future in as_completed(futures):
                recording_id = futures[future]
                results_by_id[recording_id] = future.result()
                completed += 1
                print(
                    f"[{completed}/{len(pending)}] completed production features: {recording_id}",
                    flush=True,
                )
    results = [results_by_id[recording.id] for recording in manifest.recordings]
    created = len(pending)
    reused = len(results) - created

    index = {
        "schemaVersion": 1,
        "kind": "volleycut-regenerated-production-inference-index-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "datasetPath": str(dataset_path),
        "datasetSha256": dataset_sha256,
        "featureCacheDirectory": str(cache_dir),
        "featureVersion": feature_version,
        "featureConfig": feature_config_payload,
        "videoDecoder": video_decoder,
        "sourceVerification": {
            "method": (
                "full-sha256" if source_verification == "full" else "sampled-sha256-v1+size"
            ),
            "storedFullSha256Trusted": source_verification == "sampled",
            "workers": verification_workers,
        },
        "decoderProbe": decoder_probe,
        "workers": worker_count,
        "logicalCpus": os.cpu_count() or 1,
        "reservedCpus": reserved_cpus,
        "opencvThreadsPerWorker": opencv_threads,
        "runtimeSha256": runtime_sha256,
        "recordings": [
            {
                "recordingId": row["recordingId"],
                "metadataPath": str(output_dir / "records" / f"{row['recordingId']}.json"),
                "metadataSha256": sha256_file(
                    output_dir / "records" / f"{row['recordingId']}.json"
                ),
                "featureCachePath": row["featureCachePath"],
                "featureCacheSha256": row["featureCacheSha256"],
                "tracePath": row["tracePath"],
                "traceSha256": row["traceSha256"],
            }
            for row in results
        ],
        "summary": {"created": created, "reused": reused},
    }
    index_path = output_dir / "index.json"
    _atomic_replace_text(index_path, json.dumps(index, indent=2, allow_nan=False) + "\n")
    ledger_rows = [
        (sha256_file(index_path), index_path),
        *[
            (sha256_file(output_dir / "records" / f"{row['recordingId']}.json"), output_dir / "records" / f"{row['recordingId']}.json")
            for row in results
        ],
        *[(str(row["traceSha256"]), Path(str(row["tracePath"]))) for row in results],
        *[
            (str(row["featureCacheSha256"]), Path(str(row["featureCachePath"])))
            for row in results
        ],
    ]
    _atomic_replace_text(
        output_dir / "checksums.sha256",
        "".join(f"{digest}  {path}\n" for digest, path in ledger_rows),
    )
    return index


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset", required=True, type=Path)
    value.add_argument("--runtime-root", required=True, type=Path)
    value.add_argument("--cache-dir", required=True, type=Path)
    value.add_argument("--output-dir", required=True, type=Path)
    value.add_argument(
        "--video-decoder",
        choices=("auto", "opencv", "nvdec"),
        default="auto",
        help="Use NVDEC when available by default; explicit nvdec fails instead of falling back.",
    )
    value.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Concurrent recordings; zero uses one worker at <=8 logical CPUs, otherwise "
            "up to logical CPUs minus two, capped by recording count."
        ),
    )
    value.add_argument(
        "--source-verification",
        choices=("sampled", "full"),
        default="sampled",
        help=(
            "Use the imported size plus 2 MiB sampled fingerprint by default; "
            "full re-reads every source video."
        ),
    )
    return value


def main() -> int:
    arguments = parser().parse_args()
    result = regenerate(
        arguments.dataset,
        arguments.runtime_root,
        arguments.cache_dir,
        arguments.output_dir,
        requested_video_decoder=arguments.video_decoder,
        workers=arguments.workers,
        source_verification=arguments.source_verification,
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
