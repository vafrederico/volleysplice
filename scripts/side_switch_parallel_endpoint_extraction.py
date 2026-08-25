"""Shared deterministic recording-parallel endpoint extraction for side-switch research."""

from __future__ import annotations

import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2

from analysis.side_switch_appearance import read_frame
from analysis.side_switch_player_detector import QuantizedPersonDetector
from analysis.side_switch_t3_jersey_transport import endpoint_sample_times
from scripts.extract_side_switch_helpers import crop_roi, window_key


@dataclass(frozen=True)
class ParallelEndpointExtraction:
    endpoints: dict[tuple[str, float, float], Any]
    audit: dict[str, dict[str, Any]]
    recording_workers: int
    opencv_threads_per_worker: int


def extract_parallel_endpoints(
    *,
    rows: list[dict[str, Any]],
    scope: Mapping[str, Any],
    video_audit: Mapping[str, Mapping[str, Any]],
    detector_dir: Path,
    summarize: Callable[[list[Any], float, QuantizedPersonDetector], Any],
    iteration: str,
    recording_workers: int,
    opencv_threads: int,
) -> ParallelEndpointExtraction:
    if recording_workers < 1 or opencv_threads < 1:
        raise ValueError(f"{iteration} worker/thread counts must be positive")
    recording_ids = tuple(str(value) for value in scope["recordingIds"])
    rows_by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_recording[str(row["recordingId"])].append(row)
    thread_state = threading.local()

    def process_recording(
        number: int, recording_id: str
    ) -> tuple[str, dict[tuple[str, float, float], Any], dict[str, Any]]:
        detector = getattr(thread_state, "detector", None)
        if detector is None:
            detector = QuantizedPersonDetector(
                detector_dir, opencv_threads=opencv_threads
            )
            thread_state.detector = detector
        boundaries = [
            row
            for row in rows_by_recording[recording_id]
            if str(row["kind"]) == "adjacent-rally-boundary"
        ]
        windows: dict[tuple[str, float, float], Mapping[str, Any]] = {}
        for row in boundaries:
            for side in ("before", "after"):
                window = row["comparisonWindows"][side]
                windows[window_key(recording_id, window)] = window
        prior = video_audit[recording_id]
        video_path = Path(str(prior["videoPath"])).resolve()
        if not video_path.is_file() or video_path.stat().st_size != int(
            prior["videoSizeBytes"]
        ):
            raise ValueError(f"{iteration} video provenance changed for {recording_id}")
        net_y_ratio = float(prior["courtGeometry"]["netYRatio"])
        roi = prior["roi"]
        print(
            f"[{number}/{len(recording_ids)}] {recording_id}: "
            f"{len(boundaries)} boundaries, {len(windows)} endpoint windows",
            file=sys.stderr,
            flush=True,
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open {iteration} video: {video_path}")
        local_started = time.perf_counter()
        local: dict[tuple[str, float, float], Any] = {}
        try:
            for key, window in sorted(windows.items(), key=lambda value: value[0][1:]):
                frames = [
                    crop_roi(read_frame(capture, timestamp), roi)
                    for timestamp in endpoint_sample_times(
                        float(window["start"]), float(window["end"])
                    )
                ]
                local[key] = summarize(frames, net_y_ratio, detector)
        finally:
            capture.release()
        audit = {
            "videoPath": str(video_path),
            "videoSizeBytes": int(prior["videoSizeBytes"]),
            "videoSha256": str(prior["videoSha256"]),
            "roi": roi,
            "courtGeometry": prior["courtGeometry"],
            "boundaryRows": len(boundaries),
            "endpointWindows": len(windows),
            "frameRequests": len(windows) * 5,
            "detectorTileCalls": len(windows) * 40,
            "frameErrors": 0,
            "elapsedSeconds": time.perf_counter() - local_started,
        }
        print(
            f"[{number}/{len(recording_ids)}] {recording_id}: complete",
            file=sys.stderr,
            flush=True,
        )
        return recording_id, local, audit

    endpoints: dict[tuple[str, float, float], Any] = {}
    extraction_audit: dict[str, dict[str, Any]] = {}
    maximum_workers = min(recording_workers, len(recording_ids))
    with ThreadPoolExecutor(max_workers=maximum_workers) as executor:
        futures = [
            executor.submit(process_recording, number, recording_id)
            for number, recording_id in enumerate(recording_ids, 1)
        ]
        for future in futures:
            recording_id, local, audit = future.result()
            endpoints.update(local)
            extraction_audit[recording_id] = audit
    return ParallelEndpointExtraction(
        endpoints=endpoints,
        audit=extraction_audit,
        recording_workers=maximum_workers,
        opencv_threads_per_worker=opencv_threads,
    )
