#!/usr/bin/env python3
"""Label-blind media-PTS repair of seven Pixel AV/DINO feature pairs.

The old sources, caches and extraction plan are immutable. Both new modalities
consume the same sequentially decoded presentation frame for each 4 Hz tick.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("frozen_dino_prep", REPO/"scripts/prepare-neural-dino-transfer.py")
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)
ROOT = old.OUTPUT
DECODER = "sequential-opencv-nearest-media-pts-v1"
PLAN_NAME = "pts-repair-plan-v1.json"


def array_sha(value, dtype):
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def grid_and_selection(pts, duration):
    pts = np.asarray(pts, dtype=np.float64)
    old.require(pts.ndim == 1 and len(pts) and np.isfinite(pts).all()
                and np.all(np.diff(pts) > 0), "presentation timestamps must be strictly increasing")
    old.require(abs(float(pts[0])) <= 1e-6, "nonzero video origin needs an explicit new contract")
    times = np.arange(int(np.ceil(float(duration)*4-1e-9)), dtype=np.float64)/4
    times = times[times < duration]
    right = np.minimum(np.searchsorted(pts, times), len(pts)-1)
    left = np.maximum(right-1, 0)
    indexes = np.where(np.abs(times-pts[left]) <= np.abs(pts[right]-times), left, right).astype(np.int64)
    errors = np.abs(pts[indexes]-times)
    old.require(len(times) and float(errors.max()) <= .125+1e-9, "media PTS nearest-frame error exceeds half tick")
    old.require(np.all(np.diff(indexes) > 0), "4Hz selection unexpectedly repeats frames")
    return times, indexes


def align_audio(samples, sample_rate, start_seconds):
    old.require(math.isfinite(start_seconds) and abs(start_seconds) <= .25, "unexpected audio origin")
    shift = int(round(start_seconds*sample_rate))
    if shift > 0:
        samples = np.concatenate((np.zeros(shift, np.float32), samples))
    elif shift < 0:
        samples = samples[-shift:]
    return np.ascontiguousarray(samples, np.float32), shift


def selected_frames(capture, indexes, presentation_times, position_msec):
    """Decode display order without seeking; verify PTS and complete frame count."""
    ordinal, cursor = -1, 0
    while capture.grab():
        ordinal += 1
        if cursor >= len(indexes) or ordinal != indexes[cursor]:
            continue
        ok, frame = capture.retrieve()
        old.require(ok and frame is not None and frame.size, "selected media frame cannot decode")
        actual_pts = float(capture.get(position_msec))/1000
        old.require(abs(actual_pts-presentation_times[ordinal]) <= 2e-6,
                    f"decoded PTS differs at ordinal{ordinal}: {actual_pts} vs {presentation_times[ordinal]}")
        cursor += 1
        yield ordinal, frame, actual_pts
    old.require(ordinal+1 == len(presentation_times) and cursor == len(indexes), "decoded display frame count/selection incomplete")


def inventory(path):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
               "-show_entries", "packet=pts:stream=time_base,duration,nb_frames,start_time,codec_name", "-of", "json", str(path)]
    video = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    stream = video["streams"][0]
    ticks = np.sort(np.asarray([int(p["pts"]) for p in video["packets"]], np.int64))
    num, den = (int(x) for x in stream["time_base"].split("/"))
    pts = ticks.astype(np.float64)*num/den
    old.require(len(ticks) == int(stream["nb_frames"]) and np.all(np.diff(ticks) > 0), "packet/frame inventory is not one unique PTS per frame")
    audio_command = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                     "stream=start_time,duration,sample_rate", "-of", "json", str(path)]
    audio = json.loads(subprocess.run(audio_command, check=True, capture_output=True, text=True).stdout)["streams"][0]
    frame_command = ["ffprobe", "-v", "error", "-read_intervals", "%+0.2", "-select_streams", "a:0", "-show_frames",
                     "-show_entries", "frame=best_effort_timestamp_time,pts_time,nb_samples", "-of", "json", str(path)]
    frames = json.loads(subprocess.run(frame_command, check=True, capture_output=True, text=True).stdout)["frames"]
    old.require(bool(frames), "no decoded audio frame to verify stream origin")
    first_audio_pts = float(frames[0]["best_effort_timestamp_time"])
    old.require(abs(first_audio_pts-float(audio["start_time"])) <= 1/float(audio["sample_rate"]),
                "decoded audio frame origin differs from stream start; explicit revised alignment required")
    audio.update({"firstDecodedFramePtsSeconds": first_audio_pts, "firstDecodedFrameSamples": int(frames[0]["nb_samples"]),
                  "originVerificationCommand": frame_command})
    return ticks, pts, stream, audio, command


def npz_new(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def decoder_runtime():
    import cv2
    return {"opencv": cv2.__version__, "opencvBuildSha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
            "ffprobe": subprocess.run(["ffprobe","-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0],
            "ffmpeg": subprocess.run(["ffmpeg","-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0],
            "python": platform.python_version(), "numpy": np.__version__}


def verify_plan(root):
    plan = old.read(root/PLAN_NAME)
    base = old.read(root/"extraction-plan.json")
    old.check_sources(base)
    old.require(old.sha256(root/"extraction-plan.json") == plan["originalExtractionPlan"]["sha256"], "original extraction plan changed")
    old.require(old.sha256(plan["protocolAmendment"]["path"]) == plan["protocolAmendment"]["sha256"], "protocol amendment changed")
    for item in plan["sourceCode"].values():
        old.require(old.sha256(item["path"]) == item["sha256"], "PTS repair source changed after freeze")
    old.require(decoder_runtime() == plan["decoderRuntime"], "PTS decoder runtime changed")
    return plan, base


def prepare(root):
    path = root/PLAN_NAME
    if path.exists():
        return verify_plan(root)[0]
    base = old.read(root/"extraction-plan.json")
    old.check_sources(base)
    source = old.identity(Path(__file__).resolve())
    record_ids = [r["recordingId"] for r in base["records"] if r["tier"] == "coverage"]
    old.require(len(record_ids) == 7, "repair scope must be exactly seven reviewed Pixel recordings")
    amendment = old.read(root/"protocol-amendment-pts-v1.json")
    revision = amendment["featureRevision"]
    old.require(revision["recordingIds"] == record_ids and revision["gridFps"] == 4
                and revision["labelsChanged"] is False and revision["exactAndDraftInputsChanged"] is False,
                "amendment scope differs from declared label-blind repair")
    plan = {"schemaVersion": 1, "kind": "label-blind-media-pts-repair", "createdAt": datetime.now(timezone.utc).isoformat(),
            "originalManifest": old.identity(old.PARENT), "originalExtractionPlan": old.identity(root/"extraction-plan.json"),
            "protocolAmendment": old.identity(root/"protocol-amendment-pts-v1.json"),
            "recordingIds": record_ids, "decoder": DECODER, "maximumWorkers": 2,
            "decoderRuntime": decoder_runtime(),
            "sourceCode": {**base["sourceCode"], Path(__file__).name: source},
            "selection": "For exact media grid0,.25,...<video duration select nearest presentation PTS; earlier frame wins ties; maximum absolute error .125s. Sorted unique packet PTS must match sequential display ordinal and every selected OpenCV PTS; no random seek.",
            "sharedFrames": "A single decoded BGR frame feeds both existing104 AV visual functions and frozen DINO image preprocessing.",
            "audio": "Existing mono16k PCM decoding and104-channel audio formula; align decoded PCM origin to reported audio stream start, then evaluate at exact media grid ticks.",
            "unchanged": "All18 labels/groups/ignored/review/consent; exact8 and draft3 feature bytes; frozen DINO weights/ROI/336/batch8/image normalization/token pooling; AV104 formulas/config.",
            "supersedes": "Seven random-seek DINO caches and ordinal-clock AV cache associations only. Historical files retained; no candidate outcomes inspected.",
            "extractorConfigSha256": old.CONFIG_SHA, "semanticBackbone": base["semanticBackbone"]}
    snapshots = root/"pts-extraction-sources"
    snapshots.mkdir(exist_ok=False)
    for name, entry in plan["sourceCode"].items():
        with (snapshots/name).open("xb") as f:
            f.write(Path(entry["path"]).read_bytes())
    old.write_new(path, plan)
    print(json.dumps({"prepared": old.identity(path)}), flush=True)
    return plan


def worker(root, recording_id):
    import cv2
    from analysis import features as feat
    from analysis.config import FeatureConfig, feature_version_for_config
    from analysis.dinov2_embeddings import DinoExtractorConfig, _write_npz_no_replace
    started = time.perf_counter()
    plan, base = verify_plan(root)
    old.require(recording_id in plan["recordingIds"], "recording outside repair scope")
    row = next(r for r in old.read(old.PARENT)["coverageRows"] if r["id"] == recording_id)
    item = next(r for r in base["records"] if r["recordingId"] == recording_id)
    leaf = root/"pts-records-v1"/recording_id
    leaf.mkdir(parents=True, exist_ok=False)
    source = old.identity(item["sourceVideoPath"])
    source["mtimeNs"] = Path(item["sourceVideoPath"]).stat().st_mtime_ns
    old.require(source["sha256"] == item["recordingContentSha256"], "source bytes changed")
    ticks, pts, stream, audio_info, command = inventory(item["sourceVideoPath"])
    metadata = feat.probe_video(item["sourceVideoPath"])
    duration = float(stream["duration"])
    old.require(abs(metadata.duration-duration) < .001 and metadata.frame_count == len(pts), "source duration/frame inventory disagrees")
    metadata = feat.VideoMetadata(duration, metadata.width, metadata.height, metadata.fps, len(pts), metadata.has_audio)
    times, indexes = grid_and_selection(pts, duration)
    avconfig = FeatureConfig.from_dict(row["featureCaches"]["audiovisual"]["config"])
    old.require(len(feat.feature_names(avconfig)) == 104 and avconfig.analysis_fps == 4, "AV signature differs")
    dinoconfig = DinoExtractorConfig()
    backbone = old.load_backbone()
    roi = tuple(item["roi"]) if item["roi"] is not None else None
    capture = cv2.VideoCapture(item["sourceVideoPath"])
    old.require(capture.isOpened(), "video cannot open")
    previous = None
    visual, embeddings, batch, selected_hashes, observed_pts = [], [], [], [], []
    cursor = 0
    try:
        for ordinal, frame, actual_pts in selected_frames(capture, indexes, pts, cv2.CAP_PROP_POS_MSEC):
            values, previous = feat._frame_features(feat._crop_roi(frame, roi), previous, avconfig)
            visual.append(values)
            selected_hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
            observed_pts.append(actual_pts)
            batch.append(frame)
            cursor += 1
            if len(batch) == dinoconfig.batch_size or cursor == len(indexes):
                embeddings.append(backbone.embed_frames(batch, (roi,)*len(batch), input_size=dinoconfig.input_size))
                batch = []
            if cursor % 256 == 0 or cursor == len(indexes):
                print(f"{datetime.now(timezone.utc).isoformat()} {recording_id} {cursor}/{len(indexes)} mediaPTS ticks", flush=True)
    finally:
        capture.release()
    old.require(cursor == len(times), "decoded sample count incomplete")
    matrix = np.vstack(visual).astype(np.float32, copy=False)
    temporal = feat._temporal_visual_features(matrix, feat._frame_feature_names(avconfig), avconfig)
    pcm, available = feat._decode_audio_samples(Path(item["sourceVideoPath"]), metadata, avconfig.audio_sample_rate)
    audio_start = float(audio_info["start_time"])
    pcm, shift = align_audio(pcm, avconfig.audio_sample_rate, audio_start)
    audio_values = feat._audio_features_from_samples(pcm, times, avconfig, available=available)
    values = np.concatenate((matrix, temporal, audio_values), axis=1).astype(np.float32)
    names = feat.feature_names(avconfig)
    tokens = np.ascontiguousarray(np.concatenate(embeddings), np.float32)
    old.require(values.shape == (len(times), 104) and tokens.shape == (len(times), 10, 384)
                and np.isfinite(values).all() and np.isfinite(tokens).all(), "invalid extracted arrays")
    selection_path = leaf/"selection.npz"
    npz_new(selection_path, times=times, packetPtsTicks=ticks, presentationTimes=pts, selectedOrdinals=indexes,
            selectedPresentationTimes=pts[indexes], observedPresentationTimes=np.asarray(observed_pts, np.float64),
            selectedFrameSha256=np.asarray(selected_hashes))
    selection = {"schemaVersion": 1, "decoder": DECODER, "recordingId": recording_id, "source": source,
                 "repairPlan": old.identity(root/PLAN_NAME), "arrays": old.identity(selection_path),
                 "timeBase": stream["time_base"], "videoDurationSeconds": duration, "videoFrameCount": len(pts),
                 "sampleCount": len(times), "selectionRule": plan["selection"], "inventoryCommand": command,
                 "sharedFrameIdentity": "Each selected full BGR frame SHA identifies the same frame passed to AV and DINO; both cache times are the exact grid.",
                 "gridSha256": array_sha(times, "<f8"), "presentationTimesSha256": array_sha(pts, "<f8"),
                 "selectedOrdinalsSha256": array_sha(indexes, "<i8"), "selectedPresentationTimesSha256": array_sha(pts[indexes], "<f8"),
                 "frameHashesSha256": hashlib.sha256("\n".join(selected_hashes).encode()).hexdigest(),
                 "maximumFrameSelectionErrorSeconds": float(np.abs(pts[indexes]-times).max()),
                 "maximumDecodedPtsErrorSeconds": float(np.abs(np.asarray(observed_pts)-pts[indexes]).max()),
                 "audio": {"stream": audio_info, "startOffsetSeconds": audio_start, "alignmentSamples": shift,
                           "sampleRate": avconfig.audio_sample_rate, "formulaUnchanged": True, "evaluationTimes": "exact same media4Hz grid"}}
    selection_json = leaf/"decoder-selection.json"
    old.write_new(selection_json, selection)
    decoder_binding = {"decoder": DECODER, "selection": old.identity(selection_json), "arrays": old.identity(selection_path)}
    av_path = leaf/"audiovisual.npz"
    npz_new(av_path, times=times, values=values, names=np.asarray(names),
            metadata_json=np.asarray(json.dumps(asdict(metadata), sort_keys=True)), video_decoder=np.asarray(DECODER),
            decoder_selection_json=np.asarray(json.dumps(decoder_binding, sort_keys=True)))
    dino_path = leaf/"dino.npz"
    md = {"schemaVersion": 1, "kind": "volleycut-frozen-dinov2-vits14-cache", "createdAt": datetime.now(timezone.utc).isoformat(),
          "recordingId": recording_id, "sourceVideoPath": str(Path(item["sourceVideoPath"]).resolve()),
          "recordingContentSha256": source["sha256"], "video": metadata.to_dict(),
          "analysisTimestamps": {"sampleFps": 4., "count": len(times), "sha256": array_sha(times, "<f8"), "cadence": "exact4Hz media presentation grid"},
          "roi": list(roi) if roi is not None else None, "roiRule": "declared court ROI otherwise full frame",
          "preprocessing": dinoconfig.to_dict(), "extractorConfigSha256": dinoconfig.config_sha256,
          "backbone": backbone.identity(), "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": backbone.torch.__version__, "device": backbone.device},
          "array": {"shape": list(tokens.shape), "layout": "timestamp,token,dimension", "dtype": "float16", "classTokenIndex": 0, "regionTokenIndexes": list(range(1,10))},
          "labelsUsed": False, "completed": True, "decoderSelection": decoder_binding}
    _write_npz_no_replace(dino_path, timestamps=times, tokens=tokens, metadata=md)
    av = copy.deepcopy(row["featureCaches"]["audiovisual"])
    av.update(old.identity(av_path))
    av.update({"metadata": asdict(metadata), "shape": list(values.shape), "videoDecoder": DECODER,
               "timestamps": {"count": len(times), "first": float(times[0]), "last": float(times[-1]), "sha256": array_sha(times,"<f8")},
               "timestampRule": "exact4Hz media presentation grid; same selected truePTS frame as DINO", "decoderSelection": decoder_binding,
               "identityVerification": {"sourceFullyHashed": True, "recordingId": recording_id, "contentSha256": source["sha256"], "repairPlanSha256": old.sha256(root/PLAN_NAME)}})
    _, join = old.alignment(times, times)
    sidecar_path = leaf/"dino-metadata.json"
    old.write_new(sidecar_path, {"recordingId": recording_id, "extractionPlan": old.identity(root/"extraction-plan.json"),
                               "repairPlan": old.identity(root/PLAN_NAME), "sourceVideoVerified": source,
                               "cache": old.identity(dino_path), "status": "created-media-pts", "cacheMetadata": md, "nearestAlignment": join})
    record = {"recordingId": recording_id, "tier": "coverage", "sourceGroup": row["sourceGroup"],
              "dinoPath": str(dino_path), "dinoSha256": old.sha256(dino_path), "metadataPath": str(sidecar_path), "metadataSha256": old.sha256(sidecar_path),
              "audiovisualPath": str(av_path), "audiovisualSha256": av["sha256"], "nearestAlignment": join, "shape": list(tokens.shape), "storedDtype": "float16",
              "sourceVideoVerified": source, "rawSourceContentSha256": item["rawSourceContentSha256"], "status": "created-media-pts",
              "decoderSelectionPath": str(selection_json), "decoderSelectionSha256": old.sha256(selection_json), "wallSeconds": time.perf_counter()-started, "passed": True}
    stat = Path(item["sourceVideoPath"]).stat()
    old.require(stat.st_size == source["sizeBytes"] and stat.st_mtime_ns == source["mtimeNs"], "source changed during extraction")
    verify_plan(root)
    old.write_new(leaf/"record-audit.json", {"record": record, "audiovisual": av, "decoderSelection": old.identity(selection_json), "passed": True})
    print(json.dumps({"completed": recording_id, "wallSeconds": record["wallSeconds"], "recordAudit": old.identity(leaf/"record-audit.json")}), flush=True)


def validate_retained_audit(retained, base):
    expected = {r["recordingId"]: r for r in base["records"] if r["tier"] != "coverage"}
    old.require(len(expected) == 11 and retained.get("passed") is True and retained.get("unchangedProxies") == 11,
                "retained11 mediaPTS audit must pass for correct population")
    records = retained["records"]
    old.require(len({r["recordingId"] for r in records}) == len(records), "duplicate retained-audit source")
    by_id = {r["recordingId"]: r for r in records}
    for rid, item in expected.items():
        row = by_id[rid]
        old.require(row["AV"]["path"] == item["audiovisual"]["path"]
                    and row["AV"]["sha256"] == item["audiovisual"]["sha256"], "retained audit AV identity differs")
        error = row["maximumAbsoluteClockErrorSeconds"]
        old.require(math.isfinite(error) and 0 <= error <= .125, "retained input exceeds allowed mediaPTS drift")


def finalize(root, retained_audit, amendment):
    plan, base = verify_plan(root)
    retained = old.read(retained_audit)
    validate_retained_audit(retained, base)
    old.require(amendment.exists(), "protocol amendment is required before manifest publication")
    original = old.read(old.PARENT)
    manifest = copy.deepcopy(original)
    repaired = {rid: old.read(root/"pts-records-v1"/rid/"record-audit.json") for rid in plan["recordingIds"]}
    for row in manifest["coverageRows"]:
        report = repaired[row["id"]]
        old.require(report["passed"], "incomplete repaired source")
        row["featureCaches"]["audiovisual"] = report["audiovisual"]
    manifest["originalManifest"] = {"path": str(old.PARENT), "sha256": old.sha256(old.PARENT)}
    manifest["protocolAmendment"] = {"path": str(amendment), "sha256": old.sha256(amendment)}
    manifest["featureRevision"] = copy.deepcopy(old.read(amendment)["featureRevision"])
    old.require(old.sha256(amendment) == plan["protocolAmendment"]["sha256"], "final amendment differs from extraction contract")
    records = []
    for item in base["records"]:
        rid = item["recordingId"]
        if rid in repaired:
            record = repaired[rid]["record"]
        else:
            p = root/"record-audits"/f"{rid}.json"
            if not p.exists():
                old.worker(root, rid)
            record = old.read(p)
        old.require(record["passed"], "record incomplete")
        for field, digest in (("dinoPath","dinoSha256"),("metadataPath","metadataSha256"),("audiovisualPath","audiovisualSha256")):
            old.require(old.sha256(record[field]) == record[digest], "cache/sidecar changed before finalization")
        records.append(record)
    output = root/"manifest-pts-v1.json"
    old.write_new(output, manifest)
    wrapper = {"schemaVersion": 1, "kind": "same18-media-pts-repaired-dino-transfer-inputs", "createdAt": datetime.now(timezone.utc).isoformat(),
               "expandedManifestPath": str(output), "expandedManifestSha256": old.sha256(output), "originalManifest": manifest["originalManifest"],
               "protocolAmendment": manifest["protocolAmendment"], "featureRevision": manifest["featureRevision"],
               "extractionPlan": old.identity(root/"extraction-plan.json"), "repairPlan": old.identity(root/PLAN_NAME),
               "extractorConfigSha256": old.CONFIG_SHA, "semanticBackbone": base["semanticBackbone"], "sourceCode": plan["sourceCode"],
               "records": records, "labelsAndIgnoredIntervals": "inherit unchanged from original expanded manifest",
               "protectedTestOpened": False, "beachIncluded": False, "embeddingLabelsUsed": False}
    old.write_new(root/"dino-manifest.json", wrapper)
    old.write_new(root/"dataset-audit-pts-v1.json", {"passed": True, "manifest": old.identity(output), "dinoManifest": old.identity(root/"dino-manifest.json"),
                  "originalManifest": old.identity(old.PARENT), "repairPlan": old.identity(root/PLAN_NAME), "retainedInputAudit": old.identity(retained_audit),
                  "records": 18, "repairedCoveragePairs": 7, "unchangedExactDraftPairs": 11,
                  "recordAudits": [old.identity(root/"pts-records-v1"/rid/"record-audit.json") for rid in plan["recordingIds"]]})
    print(json.dumps({"manifest": old.identity(output), "dinoManifest": old.identity(root/"dino-manifest.json"), "audit": old.identity(root/"dataset-audit-pts-v1.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker-id")
    parser.add_argument("--workers", type=int, choices=(1,2), default=2)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--retained-audit", type=Path)
    parser.add_argument("--amendment", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.worker_id:
        worker(args.root, args.worker_id)
        return
    if args.finalize_only:
        old.require(args.retained_audit is not None and args.amendment is not None, "finalization requires retained audit and protocol amendment")
        finalize(args.root, retained_audit=args.retained_audit, amendment=args.amendment)
        return
    plan = prepare(args.root)
    if args.prepare_only:
        return
    logs = args.root/"pts-extraction-logs"
    logs.mkdir(exist_ok=True)
    pending = [rid for rid in plan["recordingIds"] if not (args.root/"pts-records-v1"/rid/"record-audit.json").exists()]
    def launch(rid):
        path = logs/f"{rid}-{time.time_ns()}.log"
        with path.open("x") as f:
            subprocess.run([sys.executable,"-u",str(Path(__file__).resolve()),"--root",str(args.root),"--worker-id",rid], stdout=f, stderr=subprocess.STDOUT, check=True)
        print(f"DONE {rid} {path}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(launch, pending))
    print("All seven repaired pairs completed. Explicit finalization awaits independent audits and protocol amendment.", flush=True)


if __name__ == "__main__":
    main()
