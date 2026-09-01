#!/usr/bin/env python3
"""Run every current production model without consuming corrected labels."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from analysis.exported_project_dataset import (
    load_exported_project_dataset,
    sha256_file,
)
from analysis.features import (
    NVDEC_VIDEO_DECODER,
    OPENCV_VIDEO_DECODER,
    FeatureSequence,
    VideoMetadata,
)
from analysis.production_eval_inference import (
    ALL_LABELS_SOURCE,
    PREVIOUS_SOURCE,
    generate_side_switch_candidates,
    merge_production_ranges,
    serving_side_frame_times,
    serving_side_inference,
    side_switch_frame_times,
    side_switch_inference,
    specialist_frame_maps,
    suppression_inference,
)
from analysis.schema import load_manifest


ALL_LABELS_FILENAME = "model-1ca43e38eefc.json"
PREVIOUS_FILENAME = "model-9c92b8e9333f.json"
SUPPRESSION_FILENAME = "suppression-39eddf581639.json"
SERVING_SIDE_FILENAME = "serving-side-85bc3325fbd4.json"
SIDE_SWITCH_FILENAME = "side-switch-c2570481c30d.json"
RUNTIME_FILENAMES = {
    "allLabelsV2": ALL_LABELS_FILENAME,
    "previousProduction": PREVIOUS_FILENAME,
    "suppression": SUPPRESSION_FILENAME,
    "servingSide": SERVING_SIDE_FILENAME,
    "sideSwitch": SIDE_SWITCH_FILENAME,
}


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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_sequence(path: Path) -> FeatureSequence:
    with np.load(path, allow_pickle=False) as payload:
        metadata = VideoMetadata(**json.loads(str(payload["metadata_json"].item())))
        return FeatureSequence(
            times=payload["times"].astype(np.float64, copy=True),
            values=payload["values"].astype(np.float32, copy=True),
            names=tuple(str(value) for value in payload["names"]),
            metadata=metadata,
        )


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _worker_layout(recordings: int, requested: int, video_decoder: str) -> tuple[int, int, int]:
    logical_cpus = os.cpu_count() or 1
    if video_decoder == OPENCV_VIDEO_DECODER or logical_cpus <= 8:
        return 1, logical_cpus, 0
    maximum = min(recordings, logical_cpus - 2)
    workers = maximum if requested == 0 else min(requested, maximum)
    reserved = 2 if workers > 1 else 0
    threads = max(1, (logical_cpus - reserved) // workers)
    return workers, threads, reserved


def _configure_worker(opencv_threads: int) -> None:
    import cv2

    cv2.setNumThreads(opencv_threads)


def _runtime_paths(runtime_root: Path) -> dict[str, Path]:
    result = {name: runtime_root / filename for name, filename in RUNTIME_FILENAMES.items()}
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing production runtime assets: {missing}")
    return result


def _runtime_identities(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        runtime = _load_json(path)
        result[name] = {
            "modelId": str(runtime["modelId"]),
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return result


def _core_ranges(metadata: Mapping[str, Any]) -> dict[str, list[dict[str, float]]]:
    raw = metadata["decodedRanges"]
    return {
        ALL_LABELS_SOURCE: [dict(value) for value in raw[ALL_LABELS_SOURCE]],
        PREVIOUS_SOURCE: [dict(value) for value in raw[PREVIOUS_SOURCE]],
    }


def _valid_existing(
    metadata_path: Path,
    arrays_path: Path,
    *,
    core_metadata_sha256: str,
    core_trace_sha256: str,
    runtime_sha256: Mapping[str, str],
    video_decoder: str,
) -> dict[str, Any] | None:
    if not metadata_path.is_file() or not arrays_path.is_file():
        return None
    try:
        metadata = _load_json(metadata_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        metadata.get("kind") != "volleycut-exported-project-model-eval-inference-v1"
        or metadata.get("coreInput", {}).get("metadataSha256") != core_metadata_sha256
        or metadata.get("coreInput", {}).get("traceSha256") != core_trace_sha256
        or metadata.get("runtimeSha256") != dict(runtime_sha256)
        or metadata.get("videoDecoder") != video_decoder
        or metadata.get("arraysSha256") != sha256_file(arrays_path)
        or metadata.get("labelsUsedAsInferenceInputs") is not False
    ):
        return None
    return metadata


def _worker(
    record: dict[str, Any],
    core_row: dict[str, Any],
    runtime_path_values: dict[str, str],
    runtime_sha256: dict[str, str],
    output_dir_value: str,
    dataset_path_value: str,
    dataset_sha256: str,
    video_decoder: str,
    opencv_threads: int,
) -> dict[str, Any]:
    _configure_worker(opencv_threads)
    output_dir = Path(output_dir_value)
    dataset_path = Path(dataset_path_value)
    runtime_paths = {name: Path(value) for name, value in runtime_path_values.items()}
    metadata_path = Path(str(core_row["metadataPath"]))
    trace_path = Path(str(core_row["tracePath"]))
    core_metadata_sha256 = sha256_file(metadata_path)
    core_trace_sha256 = sha256_file(trace_path)
    if (
        core_metadata_sha256 != core_row["metadataSha256"]
        or core_trace_sha256 != core_row["traceSha256"]
    ):
        raise ValueError(f"core inference checkpoint changed for {record['recordingId']}")
    arrays_path = output_dir / "arrays" / f"{record['recordingId']}.npz"
    result_path = output_dir / "records" / f"{record['recordingId']}.json"
    existing = _valid_existing(
        result_path,
        arrays_path,
        core_metadata_sha256=core_metadata_sha256,
        core_trace_sha256=core_trace_sha256,
        runtime_sha256=runtime_sha256,
        video_decoder=video_decoder,
    )
    if existing is not None:
        return {**existing, "checkpointReused": True}

    core_metadata = _load_json(metadata_path)
    trace = _load_trace(trace_path)
    cache_path = Path(str(core_metadata["featureCachePath"]))
    if sha256_file(cache_path) != core_metadata["featureCacheSha256"]:
        raise ValueError(f"feature cache changed for {record['recordingId']}")
    sequence = _load_sequence(cache_path)
    if not np.array_equal(sequence.times, trace["times"]):
        raise ValueError(f"feature/trace timestamps differ for {record['recordingId']}")

    production_ranges = _core_ranges(core_metadata)
    intervals = merge_production_ranges(production_ranges)
    rally_scores = {
        ALL_LABELS_SOURCE: trace["all_labels_v2_rally"],
        PREVIOUS_SOURCE: trace["previous_production_rally"],
    }
    serve_scores = {
        ALL_LABELS_SOURCE: trace["all_labels_v2_serve"],
        PREVIOUS_SOURCE: trace["previous_production_serve"],
    }
    dead_scores = {
        ALL_LABELS_SOURCE: trace["all_labels_v2_dead_state"],
        PREVIOUS_SOURCE: trace["previous_production_dead_state"],
    }
    serve_detections = {
        ALL_LABELS_SOURCE: core_metadata["serveDetections"][ALL_LABELS_SOURCE],
        PREVIOUS_SOURCE: core_metadata["serveDetections"][PREVIOUS_SOURCE],
    }

    suppression_runtime = _load_json(runtime_paths["suppression"])
    suppression_runtime["featureConfig"] = core_metadata["featureConfig"]
    suppression_probabilities, suppression_intervals = suppression_inference(
        suppression_runtime, sequence
    )
    serving_runtime = _load_json(runtime_paths["servingSide"])
    switch_runtime = _load_json(runtime_paths["sideSwitch"])
    switch_candidates = generate_side_switch_candidates(
        intervals,
        sequence.times,
        dead_scores[ALL_LABELS_SOURCE],
        switch_runtime,
    )
    serving_times = serving_side_frame_times(intervals, sequence.metadata.duration)
    switch_times = side_switch_frame_times(
        intervals, switch_candidates, sequence.metadata.duration
    )
    serving_frames, switch_frames = specialist_frame_maps(
        Path(str(core_metadata["sourceVideoPath"])),
        sequence,
        tuple(float(value) for value in record["roi"]) if record.get("roi") else None,
        serving_times,
        switch_times,
        video_decoder,
    )
    serving_output, serving_features = serving_side_inference(
        serving_runtime,
        intervals,
        sequence.metadata.duration,
        serving_frames,
        sequence.times,
        serve_scores,
        serve_detections,
    )
    switch_output, switch_features, switch_probabilities = side_switch_inference(
        str(record["recordingId"]),
        switch_runtime,
        intervals,
        production_ranges,
        sequence.times,
        rally_scores,
        dead_scores,
        switch_candidates,
        switch_frames,
        sequence.metadata.duration,
    )
    _atomic_npz(
        arrays_path,
        times=sequence.times.astype(np.float64),
        suppression_probabilities=suppression_probabilities.astype(np.float32),
        serving_side_features=serving_features.astype(np.float64),
        side_switch_features=switch_features.astype(np.float64),
        side_switch_candidate_probabilities=switch_probabilities.astype(np.float64),
    )
    result = {
        "schemaVersion": 1,
        "kind": "volleycut-exported-project-model-eval-inference-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "recordingId": record["recordingId"],
        "datasetPath": str(dataset_path),
        "datasetSha256": dataset_sha256,
        "videoDecoder": video_decoder,
        "sourceVideoPath": core_metadata["sourceVideoPath"],
        "sourceVideoSha256": core_metadata["sourceVideoSha256"],
        "labelsUsedAsInferenceInputs": False,
        "llmLabelingUsed": False,
        "evaluationTarget": {
            "referencePath": str(dataset_path.parent / str(record["referencePath"])),
            "referenceSha256": record["referenceSha256"],
            "feedbackPath": record["feedbackPath"],
            "feedbackSha256": record["feedbackSha256"],
            "policy": "immutable comparison target; never consumed by inference",
        },
        "coreInput": {
            "metadataPath": str(metadata_path),
            "metadataSha256": core_metadata_sha256,
            "tracePath": str(trace_path),
            "traceSha256": core_trace_sha256,
            "featureCachePath": str(cache_path),
            "featureCacheSha256": core_metadata["featureCacheSha256"],
        },
        "runtimeSha256": runtime_sha256,
        "predictedEnsembleRanges": intervals,
        "suppression": {
            "modelId": suppression_runtime["modelId"],
            "artifactSha256": suppression_runtime["artifactSha256"],
            "weightsSha256": suppression_runtime["weightsSha256"],
            "decoderVersion": suppression_runtime["decoderVersion"],
            "probabilityRows": len(suppression_probabilities),
            "decodedIntervals": suppression_intervals,
        },
        "servingSide": serving_output,
        "sideSwitch": switch_output,
        "framePlan": {
            "sharedSequentialDecode": True,
            "servingSideTimestamps": len(serving_times),
            "sideSwitchTimestamps": len(switch_times),
            "unionTimestamps": len({*serving_times, *switch_times}),
        },
        "arraysPath": str(arrays_path),
        "arraysSha256": sha256_file(arrays_path),
    }
    _atomic_replace_text(result_path, json.dumps(result, indent=2, allow_nan=False) + "\n")
    return {**result, "checkpointReused": False}


def regenerate(
    dataset_path: Path,
    core_index_path: Path,
    runtime_root: Path,
    output_dir: Path,
    *,
    workers: int = 0,
) -> dict[str, Any]:
    dataset_path = dataset_path.expanduser().resolve()
    core_index_path = core_index_path.expanduser().resolve()
    runtime_root = runtime_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if workers < 0:
        raise ValueError("workers must be zero (automatic) or positive")
    dataset = load_exported_project_dataset(dataset_path)
    dataset_sha256 = sha256_file(dataset_path)
    manifest_path = dataset_path.parent / str(dataset["inferenceManifestPath"])
    if sha256_file(manifest_path) != dataset["inferenceManifestSha256"]:
        raise ValueError("exported-project inference manifest changed")
    manifest = load_manifest(manifest_path, require_videos=False)
    manifest_by_id = {recording.id: recording for recording in manifest.recordings}
    core_index = _load_json(core_index_path)
    if core_index.get("datasetSha256") != dataset_sha256:
        raise ValueError("core inference index belongs to a different dataset revision")
    video_decoder = str(core_index["videoDecoder"])
    if video_decoder not in {NVDEC_VIDEO_DECODER, OPENCV_VIDEO_DECODER}:
        raise ValueError("core inference index uses an unsupported decoder")
    paths = _runtime_paths(runtime_root)
    identities = _runtime_identities(paths)
    runtime_sha256 = {name: value["sha256"] for name, value in identities.items()}
    core_rows = {str(value["recordingId"]): dict(value) for value in core_index["recordings"]}
    records = [
        {
            **dict(value),
            "roi": list(manifest_by_id[str(value["recordingId"])].roi)
            if manifest_by_id[str(value["recordingId"])].roi is not None
            else None,
        }
        for value in dataset["records"]
    ]
    if set(core_rows) != {str(value["recordingId"]) for value in records}:
        raise ValueError("core inference index and exported-project dataset differ")
    worker_count, opencv_threads, reserved = _worker_layout(
        len(records), workers, video_decoder
    )
    runtime_values = {name: str(path) for name, path in paths.items()}
    arguments = [
        (
            record,
            core_rows[str(record["recordingId"])],
            runtime_values,
            runtime_sha256,
            str(output_dir),
            str(dataset_path),
            dataset_sha256,
            video_decoder,
            opencv_threads,
        )
        for record in records
    ]
    results_by_id: dict[str, dict[str, Any]] = {}
    if worker_count == 1:
        for index, values in enumerate(arguments, start=1):
            result = _worker(*values)
            results_by_id[str(result["recordingId"])] = result
            print(f"[{index}/{len(arguments)}] completed all-model eval inference: {result['recordingId']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_worker, *values): str(values[0]["recordingId"]) for values in arguments}
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results_by_id[str(result["recordingId"])] = result
                completed += 1
                print(f"[{completed}/{len(arguments)}] completed all-model eval inference: {result['recordingId']}", flush=True)
    results = [results_by_id[str(record["recordingId"])] for record in records]
    index = {
        "schemaVersion": 1,
        "kind": "volleycut-exported-project-model-eval-inference-index-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "datasetPath": str(dataset_path),
        "datasetSha256": dataset_sha256,
        "coreInferenceIndexPath": str(core_index_path),
        "coreInferenceIndexSha256": sha256_file(core_index_path),
        "labelsUsedAsInferenceInputs": False,
        "llmLabelingUsed": False,
        "runtimeModels": identities,
        "videoDecoder": video_decoder,
        "workers": worker_count,
        "logicalCpus": os.cpu_count() or 1,
        "reservedCpus": reserved,
        "opencvThreadsPerWorker": opencv_threads,
        "recordings": [
            {
                "recordingId": result["recordingId"],
                "metadataPath": str(output_dir / "records" / f"{result['recordingId']}.json"),
                "metadataSha256": sha256_file(output_dir / "records" / f"{result['recordingId']}.json"),
                "arraysPath": result["arraysPath"],
                "arraysSha256": result["arraysSha256"],
            }
            for result in results
        ],
        "summary": {
            "recordings": len(results),
            "created": sum(not result["checkpointReused"] for result in results),
            "reused": sum(result["checkpointReused"] for result in results),
            "predictedRallies": sum(len(result["predictedEnsembleRanges"]) for result in results),
            "suppressionIntervals": sum(len(result["suppression"]["decodedIntervals"]) for result in results),
            "servingSideCandidates": sum(len(result["servingSide"]["candidates"]) for result in results),
            "sideSwitchProposals": sum(len(result["sideSwitch"]["proposals"]) for result in results),
            "sideSwitchPredictions": sum(len(result["sideSwitch"]["candidates"]) for result in results),
        },
    }
    index_path = output_dir / "index.json"
    _atomic_replace_text(index_path, json.dumps(index, indent=2, allow_nan=False) + "\n")
    checksum_rows = [
        (sha256_file(index_path), index_path),
        *[
            (sha256_file(Path(str(row["metadataPath"]))), Path(str(row["metadataPath"])))
            for row in index["recordings"]
        ],
        *[
            (sha256_file(Path(str(row["arraysPath"]))), Path(str(row["arraysPath"])))
            for row in index["recordings"]
        ],
    ]
    _atomic_replace_text(
        output_dir / "checksums.sha256",
        "".join(f"{digest}  {path}\n" for digest, path in checksum_rows),
    )
    return index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--core-index", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=0)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    result = regenerate(
        arguments.dataset,
        arguments.core_index,
        arguments.runtime_root,
        arguments.output,
        workers=arguments.workers,
    )
    print(json.dumps({"output": str(arguments.output / "index.json"), **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
