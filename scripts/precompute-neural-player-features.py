#!/usr/bin/env python3
"""Write label-independent player-motion NPZ caches on the exact AV time grid.

Input is a sanitized {"records": [...]} manifest, never a label manifest. Each
row needs id/video/contentSha256/roi/sourceGroup/split/environment/avCache.
The command uses CPU OpenCV and streams frames without staging source video.
--max-seconds creates explicitly partial engineering caches, not trainable ones.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value

from analysis.player_motion_features import (FEATURE_NAMES, FEATURE_VERSION, PlayerMotionConfig,
    PlayerMotionExtractor, RallyPersonDetector, crop_roi, nearest_frame_indexes)
from analysis.side_switch_player_detector import detector_identity, sha256_file


PROTECTED_GROUPS = {private_value('source-group-008')}
ALLOWED_FIELDS = {"id", "video", "contentSha256", "roi", "sourceGroup", "split",
                  "environment", "avCache", "netYRatio", "durationSeconds"}


def _write_json_new(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def read_records(path: Path, requested: list[str]) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("expected sanitized records-only extraction manifest")
    ids = [row.get("id") for row in rows]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("record IDs must be nonempty and unique")
    if set(requested)-set(ids):
        raise ValueError("unknown requested recording")
    selected = []
    for row in rows:
        if requested and row["id"] not in requested:
            continue
        if set(row)-ALLOWED_FIELDS:
            raise ValueError(f"not a sanitized extraction row: {row['id']}: {sorted(set(row)-ALLOWED_FIELDS)}")
        if row.get("split") not in {"train", "validation"} or row.get("sourceGroup") in PROTECTED_GROUPS:
            raise ValueError("protected or nondevelopment recording refused")
        if row.get("environment") not in {"indoor", "grass"}:
            raise ValueError("only current non-beach indoor/grass development scope is allowed")
        if not row.get("sourceGroup") or len(row.get("contentSha256", "")) != 64:
            raise ValueError("missing source identity")
        if not isinstance(row.get("avCache"), dict) or set(row["avCache"]) != {"path", "sha256"}:
            raise ValueError("avCache must contain only path and sha256")
        if any(character in row["id"] for character in ("/", "\\")) or row["id"] in {".", ".."}:
            raise ValueError("record ID must be a safe filename")
        selected.append(row)
    return selected


def read_av_times(entry: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(entry["path"])
    if sha256_file(path) != entry["sha256"]:
        raise ValueError("AV cache hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(data["times"], np.float64)
        if "values" not in data or data["values"].shape != (len(times), 104):
            raise ValueError("expected current AV104 cache")
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or times[0] < 0 or np.any(np.diff(times) <= 0)
            or np.any(np.abs(np.diff(times)-.25) > .1)):
        raise ValueError("invalid or irregular AV timeline")
    return times, {"path": str(path), "sha256": entry["sha256"], "sizeBytes": path.stat().st_size}


def packet_timeline(video: Path) -> tuple[np.ndarray, dict[str, Any]]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts:stream=time_base,duration,nb_frames,start_time,codec_name,width,height",
        "-of", "json", str(video)]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
    payload = json.loads(result.stdout)
    if len(payload.get("streams", [])) != 1:
        raise ValueError("expected one selected video stream")
    stream = payload["streams"][0]
    ticks = np.sort(np.asarray([int(p["pts"]) for p in payload["packets"]], np.int64))
    numerator, denominator = (int(x) for x in stream["time_base"].split("/"))
    pts = ticks.astype(np.float64)*numerator/denominator
    if (not len(pts) or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0)
            or len(pts) != int(stream["nb_frames"]) or abs(float(pts[0])) > 1e-6):
        raise ValueError("v1 requires unique zero-origin frame PTS, one packet per display frame")
    return pts, {"stream": stream, "command": command,
        "presentationTimesSha256": hashlib.sha256(pts.tobytes()).hexdigest(),
        "frameCount": len(pts)}


def iter_selected_frames(capture: Any, indexes: np.ndarray, pts: np.ndarray,
                         *, full_recording: bool):
    ordinal, cursor = -1, 0
    while capture.grab():
        ordinal += 1
        if cursor >= len(indexes):
            if not full_recording:
                break
            continue
        if ordinal != indexes[cursor]:
            continue
        ok, frame = capture.retrieve()
        if not ok or frame is None or not frame.size:
            raise ValueError("selected presentation frame cannot decode")
        actual = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
        if abs(actual-float(pts[ordinal])) > 2e-6:
            raise ValueError(f"decoder PTS mismatch at frame {ordinal}: {actual} vs {pts[ordinal]}")
        cursor += 1
        yield ordinal, frame, actual
    if cursor != len(indexes) or (full_recording and ordinal+1 != len(pts)):
        raise ValueError("incomplete sequential decode/selection")


def _save_cache(path: Path, *, times: np.ndarray, values: np.ndarray, metadata: dict[str, Any],
                source_pts: np.ndarray, source_indexes: np.ndarray, frame_hashes: list[str], audit: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing cache overwrite: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".player-", suffix=".npz", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, times=np.asarray(times, np.float64), values=np.asarray(values, np.float32),
                names=np.asarray(FEATURE_NAMES), metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)),
                selected_source_pts=np.asarray(source_pts, np.float64), selected_source_indexes=source_indexes,
                selected_frame_sha256=np.asarray(frame_hashes),
                audit_json=np.asarray(json.dumps(audit, separators=(",", ":"), allow_nan=False)))
            handle.flush()
            os.fsync(handle.fileno())
        # Hard link publishes an atomic complete artifact without replacing a concurrent writer.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def extract_record(row: dict[str, Any], args: Any, detector: RallyPersonDetector,
                   manifest_identity: dict[str, Any], detector_metadata: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    sources = [Path(__file__), REPO/"analysis/player_motion_features.py",
               REPO/"analysis/side_switch_player_detector.py", REPO/"analysis/serving_side_flight.py"]
    detector_family = getattr(args, "detector_family", "mediapipe")
    if detector_family == "nanodet":
        sources.append(REPO/"analysis/nanodet_person_detector.py")
    source_code = {str(path.relative_to(REPO)): sha256_file(path) for path in sources}
    if sha256_file(args.manifest) != manifest_identity["sha256"]:
        raise ValueError("extraction manifest changed")
    target = args.output_dir/f"{row['id']}.player-motion.npz"
    receipt = target.with_suffix(".json")
    if target.exists() or receipt.exists():
        raise FileExistsError(f"refusing output overwrite: {target}")
    video = Path(row["video"])
    initial_stat = video.stat()
    print(f"{row['id']}: verifying source and AV identities", flush=True)
    if sha256_file(video) != row["contentSha256"]:
        raise ValueError("source video content hash mismatch")
    all_times, av_identity = read_av_times(row["avCache"])
    times = all_times if args.max_seconds is None else all_times[all_times < args.max_seconds]
    if not len(times):
        raise ValueError("partial pilot has no samples")
    full_recording = args.max_seconds is None
    pts, media = packet_timeline(video)
    duration = float(media["stream"]["duration"])
    if (all_times[-1] >= duration or all_times[0] > .125
            or duration-float(all_times[-1]) > .36):
        raise ValueError("AV timeline does not cover the complete source duration")
    if row.get("durationSeconds") is not None and abs(float(row["durationSeconds"])-duration) > .01:
        raise ValueError("manifest duration differs from source media")
    indexes = nearest_frame_indexes(pts, times)
    raw_roi = row.get("roi")
    roi = tuple(raw_roi[k] for k in ("x", "y", "width", "height")) if isinstance(raw_roi, dict) else raw_roi
    net = row.get("netYRatio")
    config = PlayerMotionConfig(net_y_ratio=0.5 if net is None else float(net),
        net_geometry_supplied=net is not None, maximum_detections=args.maximum_detections,
        motion_long_side=args.motion_long_side)
    extractor = PlayerMotionExtractor(detector, config)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError("source video cannot open")
    vectors, audit, hashes, observed_pts = [], [], [], []
    try:
        for ordinal, frame, source_pts in iter_selected_frames(capture, indexes, pts, full_recording=full_recording):
            tick = len(vectors)
            roi_frame = crop_roi(frame, roi)
            vector, info = extractor.push(roi_frame, float(times[tick]))
            vectors.append(vector)
            info.update(time=float(times[tick]), sourceFrameOrdinal=int(ordinal), sourcePtsSeconds=float(source_pts))
            audit.append(info)
            hashes.append(hashlib.sha256(roi_frame.tobytes()).hexdigest())
            observed_pts.append(source_pts)
            if len(vectors) % 120 == 0 or len(vectors) == len(times):
                print(f"{row['id']}: {len(vectors)}/{len(times)} ticks; {time.perf_counter()-started:.1f}s", flush=True)
    finally:
        capture.release()
    final_stat = video.stat()
    if initial_stat.st_size != final_stat.st_size or initial_stat.st_mtime_ns != final_stat.st_mtime_ns:
        raise ValueError("source changed during extraction")
    values = np.stack(vectors)
    if any(sha256_file(REPO/name) != digest for name, digest in source_code.items()):
        raise ValueError("extractor source changed during extraction")
    if (sha256_file(args.manifest) != manifest_identity["sha256"]
            or sha256_file(Path(row["avCache"]["path"])) != av_identity["sha256"]):
        raise ValueError("manifest or AV cache changed during extraction")
    if detector_family == "nanodet":
        from analysis.nanodet_person_detector import nanodet_identity
        if nanodet_identity(args.detector_dir) != detector_metadata:
            raise ValueError("NanoDet artifacts changed during extraction")
    selection_metadata = getattr(detector, "selection_metadata", {
        "maximumDetections": config.maximum_detections, "perSideCap": None, "nmsIou": .45,
        "duplicateHipRadiusMinimumTorsoFraction": .25,
        "tracking": "greedy-spatial-scale-gated-short-association; not persistent identity"})
    metadata = {"schemaVersion": 1, "featureVersion": FEATURE_VERSION,
        "createdAt": datetime.now(UTC).isoformat(), "recordingId": row["id"], "sourceGroup": row["sourceGroup"],
        "environment": row["environment"], "labelIndependent": True, "labelsUsed": False,
        "fullRecording": full_recording, "trainingEligible": full_recording,
        "pilotMaximumSeconds": args.max_seconds, "manifest": manifest_identity,
        "sourceVideo": {"path": str(video), "sha256": row["contentSha256"], "sizeBytes": initial_stat.st_size},
        "avCache": av_identity, "roi": roi, "config": asdict(config), "detector": detector_metadata,
        "detectorSelectionOverride": selection_metadata,
        "decoder": "sequential-opencv-packet-pts-verified-v1", "media": media,
        "normalization": "physical image-coordinate units/seconds; no percentile transform or learned normalization",
        "geometry": "ROI membership proxy; piecewise net-normalized hip y; no ground-plane or identity claim",
        "missingEvidence": "empty detections are not dead-play labels; flow/context gap resets tracks; detection failure aborts",
        "alignment": {"avTimesExact": True, "timesSha256": hashlib.sha256(times.tobytes()).hexdigest(),
            "maximumSourcePtsErrorSeconds": float(np.max(np.abs(np.asarray(observed_pts)-times)))},
        "sourceCode": source_code,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
            "opencvThreads": cv2.getNumThreads(), "opencvBuildSha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()},
        "statistics": {"ticks": len(times), "features": len(FEATURE_NAMES),
            "detectorFrames": extractor.total_detector_frames, "detectorTileCalls": extractor.total_tile_calls,
            "detectorInferenceSeconds": extractor.total_inference_ms/1000,
            "anyTrackFraction": float(np.mean([r["visibleTracks"] > 0 for r in audit])),
            "bothSidesFraction": float(np.mean([r["nearTracks"] > 0 and r["farTracks"] > 0 for r in audit])),
            "saturatedDetectorFrames": sum(r["detectorSaturated"] for r in audit),
            "wallSeconds": time.perf_counter()-started},
        "limitations": ["Unqualified detector/track availability is not person recall or precision.",
            "Torso-patch flow can contain background; camera compensation can absorb coordinated player motion.",
            "Fallback net at image midline and rectangular ROI do not establish exact court ownership.",
            "2Hz detection plus 4Hz flow does not resolve every contact or distant player."]}
    _save_cache(target, times=times, values=values, metadata=metadata, source_pts=np.asarray(observed_pts),
        source_indexes=indexes, frame_hashes=hashes, audit=audit)
    result = {"path": str(target), "sha256": sha256_file(target), "sizeBytes": target.stat().st_size,
              "metadata": metadata}
    _write_json_new(receipt, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--recording-id", action="append", default=[])
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--detector-family", choices=("mediapipe", "nanodet"), default="mediapipe")
    parser.add_argument("--detector-mode", choices=("single-roi", "four-tiles"),
                        help="NanoDet defaults to official single-ROI inference; MediaPipe uses four tiles")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opencv-threads", type=int, default=2)
    parser.add_argument("--maximum-detections", type=int, default=24)
    parser.add_argument("--motion-long-side", type=int, default=320)
    parser.add_argument("--max-seconds", type=float, help="prefix-only engineering pilot; published trainingEligible=false")
    parser.add_argument("--index", type=Path, help="optional new consolidated cache index JSON")
    args = parser.parse_args()
    if args.max_seconds is not None and (not np.isfinite(args.max_seconds) or args.max_seconds <= 0):
        raise ValueError("max seconds must be finite and positive")
    PlayerMotionConfig(maximum_detections=args.maximum_detections, motion_long_side=args.motion_long_side)
    rows = read_records(args.manifest, args.recording_id)
    if args.index is not None and args.index.exists():
        raise FileExistsError(f"refusing index overwrite: {args.index}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_identity = {"path": str(args.manifest), "sha256": sha256_file(args.manifest)}
    if args.detector_family == "nanodet":
        from analysis.nanodet_person_detector import NanoDetPersonDetector, nanodet_identity
        detector_metadata = nanodet_identity(args.detector_dir)
        detector = NanoDetPersonDetector(args.detector_dir, maximum_detections=args.maximum_detections,
                                       opencv_threads=args.opencv_threads, tile_mode=args.detector_mode or "single-roi")
    else:
        if args.detector_mode not in {None, "four-tiles"}:
            raise ValueError("MediaPipe adapter supports only its pinned four-tile mode")
        detector_metadata = detector_identity(args.detector_dir)
        detector = RallyPersonDetector(str(args.detector_dir), maximum_detections=args.maximum_detections,
                                      opencv_threads=args.opencv_threads)
    outputs = []
    for row in rows:
        result = extract_record(row, args, detector, manifest_identity, detector_metadata)
        outputs.append({"id": row["id"], "path": result["path"], "sha256": result["sha256"],
            "trainingEligible": result["metadata"]["trainingEligible"],
            "statistics": result["metadata"]["statistics"]})
        print(json.dumps({"recordingId": row["id"], "cache": result["path"], "sha256": result["sha256"],
            "statistics": result["metadata"]["statistics"]}), flush=True)
    if args.index is not None:
        args.index.parent.mkdir(parents=True, exist_ok=True)
        _write_json_new(args.index, {"schemaVersion": 1, "featureVersion": FEATURE_VERSION,
            "manifest": manifest_identity, "featureNames": list(FEATURE_NAMES),
            "featureDimension": len(FEATURE_NAMES), "records": outputs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
