#!/usr/bin/env python3
"""Freeze, then run exactly two person-detector policies on label-free NAS inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
IDS = (private_value('grass-source-03'), private_value('grass-source-01'),
       private_value('indoor-source-01'), private_value('indoor-source-08'))
SEQUENCE_IDS = {IDS[0], IDS[2]}
COMMIT = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"
ASSET_URLS = {
    "yolox-source.zip": f"https://codeload.github.com/Megvii-BaseDetection/YOLOX/zip/{COMMIT}",
    "yolox_nano.pth": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth",
    "ssdlite320_mobilenet_v3_large_coco-a79551df.pth": "https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth",
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "sizeBytes": path.stat().st_size}


def write_new(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def fixed_times(duration):
    if duration <= 40:
        raise ValueError("qualification expects full-length recordings")
    return [0., 5., 10.5, 20.5, 29.5, duration * .25, duration * .5, duration * .75]


def load_helpers():
    spec = importlib.util.spec_from_file_location("person_qualification_existing_helpers",
                                                REPO / "scripts/precompute-neural-player-features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze(args):
    import cv2
    import torch
    import torchvision
    helpers = load_helpers()
    rows = helpers.read_records(args.manifest, list(IDS))
    if {row["id"] for row in rows} != set(IDS):
        raise ValueError("fixed recording scope missing")
    if (args.output / "protocol.json").exists():
        raise FileExistsError("protocol already frozen")
    assets = args.output / "assets"
    files = sorted(p for p in assets.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if not set(ASSET_URLS).issubset({p.name for p in files}):
        raise ValueError("download the exact official assets before freezing")
    receipt = {"kind": "person-alternative-assets-v1", "upstreamYoloXCommit": COMMIT,
               "officialDownloads": ASSET_URLS, "files": [identity(p) for p in files]}
    # Archive actual imported Torchvision implementation and license, including
    # architecture, anchors, NMS, normalization, resize and box restoration.
    tvroot = Path(torchvision.__file__).parent
    tvnames = ["models/detection/ssdlite.py", "models/detection/ssd.py", "models/detection/transform.py",
               "models/detection/anchor_utils.py", "models/detection/_utils.py", "models/detection/backbone_utils.py",
               "models/mobilenetv3.py", "ops/boxes.py", "ops/misc.py"]
    tvarchive = assets / "torchvision-runtime-sources"
    tvarchive.mkdir(exist_ok=True)
    for name in tvnames:
        target = tvarchive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tvroot / name, target)
    receipt["torchvisionSources"] = [identity(tvarchive / name) for name in tvnames]
    receipt["runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
                          "torchvision": torchvision.__version__, "opencv": cv2.__version__}
    dependencies = args.output / "python-dependencies"
    receipt["nasDependencyFiles"] = [identity(p) for p in sorted(dependencies.rglob("*"))
                                      if p.is_file() and "__pycache__" not in p.parts]
    write_new(args.output / "asset-receipt.json", receipt)
    source_paths = [Path(__file__), REPO / "analysis/person_alternative_detectors.py",
                    REPO / "scripts/precompute-neural-player-features.py", REPO / "analysis/player_motion_features.py"]
    archive = args.output / "registered-sources"
    archive.mkdir()
    source_refs = []
    for path in source_paths:
        target = archive / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_refs.append({"working": identity(path), "archived": identity(target)})
    protocol = {"kind": "person-alternatives-qualification-v1", "frozenAtUtc": datetime.now(UTC).isoformat(),
                "combinedProtocol": identity(args.output.parent / "protocol.md"),
                "manifest": identity(args.manifest), "assetReceipt": identity(args.output / "asset-receipt.json"),
                "sources": source_refs, "records": rows, "fixedTimesRule": "0,5,10.5,20.5,29.5 seconds plus 25/50/75 percent duration",
                "fixedFramesPerRecord": 8, "sequenceRecords": sorted(SEQUENCE_IDS),
                "sequenceTimes": "0 through 29.5 seconds, step 0.5", "sourceSelection": "nearest actual presentation timestamp; verify seek output",
                "view": "one existing manifest ROI; no tiles", "cpuThreads": 2, "gpuAllowed": False,
                "scoreThreshold": .25, "capPerFrame": 24, "perSideCap": None,
                "models": {"yolox-nano-416": {"inputSize": 416, "nmsIou": .45, "personClass": 0,
                    "preprocess": "official ValTransform legacy=False: BGR 0..255 float32, aspect resize, top-left 114 letterbox",
                    "postprocess": "official class-aware postprocess: best class, objectness times class score, NMS then person filter; inverse scale and clip",
                    "license": "Apache-2.0 (official YOLOX repository)"},
                    "ssdlite320-mobilenetv3-large": {"inputSize": 320, "nmsIou": .55, "personClass": 1,
                    "preprocess": "RGB float32 0..1; official SSD transform normalization and fixed resize",
                    "postprocess": "official anchors, decode, score threshold, per-class NMS, topk=300, detectionsPerImage=300; person filter; inverse mapping and clip",
                    "license": "BSD-3-Clause (Torchvision source; pretrained weights from official PyTorch download)"}},
                "qualificationGate": "Reject a feature-training arm if fixed overlays show repeated obvious duplicates, non-person locations, or missing distant players in either environment; otherwise permit bounded feature proof-of-concept only.",
                "thresholdSweep": False, "labelsUsed": False, "protectedTestOpened": False,
                "detectorPrecisionRecallClaim": False, "trainingEligible": False,
                "latency": "Exclude load and one synthetic warmup; forward-only and pre/forward/post per frame separately; full study wall includes media IO, hashes and overlays"}
    write_new(args.output / "protocol.json", protocol)
    print(json.dumps({"protocol": identity(args.output / "protocol.json"), "assets": identity(args.output / "asset-receipt.json")}))


def verify_protocol(args):
    path = args.output / "protocol.json"
    protocol = json.loads(path.read_text())
    for ref in [protocol["manifest"], protocol["assetReceipt"], protocol["combinedProtocol"]]:
        if digest(ref["path"]) != ref["sha256"]:
            raise ValueError("registered artifact changed")
    for source in protocol["sources"]:
        for ref in source.values():
            if digest(ref["path"]) != ref["sha256"]:
                raise ValueError("registered source changed")
    receipt = json.loads(Path(protocol["assetReceipt"]["path"]).read_text())
    for ref in receipt["files"] + receipt["torchvisionSources"] + receipt["nasDependencyFiles"]:
        if digest(ref["path"]) != ref["sha256"]:
            raise ValueError(f"asset changed: {ref['path']}")
    return protocol


def draw_overlay(frame, result, label):
    import cv2
    image = frame.copy()
    for i, row in enumerate(result["detections"]):
        x1, y1, x2, y2 = (int(round(x)) for x in row["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 240, 255), max(1, image.shape[1] // 800))
        cv2.putText(image, f"{i}:{row['score']:.2f}", (x1, max(15, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(.4, image.shape[1] / 1800), (0, 0, 0), 3)
        cv2.putText(image, f"{i}:{row['score']:.2f}", (x1, max(15, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(.4, image.shape[1] / 1800), (0, 240, 255), 1)
    resized = cv2.resize(image, (960, int(round(image.shape[0] * 960 / image.shape[1]))))
    cv2.rectangle(resized, (0, 0), (960, 28), (0, 0, 0), -1)
    cv2.putText(resized, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return resized


def run(args):
    import cv2
    import numpy as np
    import torch
    from analysis.person_alternative_detectors import YoloXNanoDetector, SSDliteDetector, box_geometry_diagnostics
    from analysis.player_motion_features import crop_roi, nearest_frame_indexes
    started = time.perf_counter()
    protocol = verify_protocol(args)
    output = args.output / "results"
    output.mkdir()  # Never overwrite an outcome after looking at it.
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    cv2.setNumThreads(2)
    helpers = load_helpers()
    models = [YoloXNanoDetector(args.output / "assets"), SSDliteDetector(args.output / "assets")]
    for model in models:
        model.detect(np.zeros((480, 640, 3), np.uint8))
    timings, records = {model.name: [] for model in models}, []
    for row in protocol["records"]:
        print(f"Verifying and sampling {row['id']}", flush=True)
        video = Path(row["video"])
        if digest(video) != row["contentSha256"]:
            raise ValueError("video source changed")
        pts, media = helpers.packet_timeline(video)
        duration = float(media["stream"]["duration"])
        if abs(duration - row["durationSeconds"]) > .01:
            raise ValueError("source duration changed")
        fixed = fixed_times(duration)
        sequence = list(np.arange(0., 30., .5)) if row["id"] in SEQUENCE_IDS else []
        requested = sorted(set(fixed + sequence))
        indexes = nearest_frame_indexes(pts, np.asarray(requested))
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError("cannot decode source")
        roi = tuple(row["roi"][k] for k in ("x", "y", "width", "height"))
        frame_rows, sheets = [], {model.name: [] for model in models}
        record_dir = output / row["id"]
        record_dir.mkdir()
        try:
            for target, ordinal in zip(requested, indexes):
                frame_started = time.perf_counter()
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(ordinal))
                success, full = capture.read()
                actual = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                if not success or abs(actual - pts[ordinal]) > 2e-6:
                    raise ValueError(f"seek PTS mismatch: {actual} != {pts[ordinal]}")
                frame = crop_roi(full, roi)
                selection_ms = (time.perf_counter() - frame_started) * 1000
                is_fixed = target in fixed
                frame_row = {"requestedSeconds": target, "ptsSeconds": actual, "frameOrdinal": int(ordinal),
                             "fixedReviewFrame": is_fixed, "sequenceFrame": target in sequence,
                             "roiPixelsShape": list(frame.shape), "roiPixelsSha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                             "seekDecodeCropMs": selection_ms, "models": {}}
                if is_fixed:
                    rawpath = record_dir / f"{target:.3f}-source.png"
                    if not cv2.imwrite(str(rawpath), frame):
                        raise IOError("cannot save review frame")
                    frame_row["sourceFrame"] = identity(rawpath)
                for model in models:
                    result = model.detect(frame)
                    result["geometry"] = box_geometry_diagnostics(result["detections"], frame.shape[1], frame.shape[0])
                    label = f"{model.name} | t={actual:.3f}s | n={len(result['detections'])}"
                    overlay = draw_overlay(frame, result, label)
                    path = record_dir / f"{target:.3f}-{model.name}.jpg"
                    if not cv2.imwrite(str(path), overlay):
                        raise IOError("cannot save overlay")
                    result["overlay"] = identity(path)
                    frame_row["models"][model.name] = result
                    timings[model.name].append({"record": row["id"], "targetSeconds": target,
                                               "forwardMs": result["forwardMs"], "preForwardPostMs": result["preForwardPostMs"]})
                    if is_fixed:
                        sheets[model.name].append(overlay)
                frame_rows.append(frame_row)
        finally:
            capture.release()
        for model in models:
            thumbs = [cv2.resize(im, (640, 380)) for im in sheets[model.name]]
            sheet = np.vstack([np.hstack(thumbs[i:i + 2]) for i in range(0, 8, 2)])
            path = output / f"{row['id']}-{model.name}-sheet.jpg"
            if not cv2.imwrite(str(path), sheet):
                raise IOError("cannot save sheet")
        record = {"id": row["id"], "sourceGroup": row["sourceGroup"], "videoIdentity": identity(video),
                  "media": media, "frames": frame_rows}
        write_new(record_dir / "record.json", record)
        records.append(identity(record_dir / "record.json"))
        print(f"Completed {row['id']}: {len(frame_rows)} distinct selected frames", flush=True)
    summaries = {}
    for model in models:
        forward = np.asarray([r["forwardMs"] for r in timings[model.name]])
        total = np.asarray([r["preForwardPostMs"] for r in timings[model.name]])
        summaries[model.name] = {"parameters": model.parameters, "uniqueFrames": len(forward),
            "forwardMedianMs": float(np.median(forward)), "forwardP95Ms": float(np.percentile(forward, 95)),
            "preForwardPostMedianMs": float(np.median(total)), "preForwardPostP95Ms": float(np.percentile(total, 95)),
            "timings": timings[model.name]}
    write_new(output / "report.json", {"kind": "person-alternatives-qualification-results-v1",
        "protocol": identity(args.output / "protocol.json"), "records": records, "models": summaries,
        "fullStudyWallSeconds": time.perf_counter() - started, "cpuThreads": 2, "gpuUsed": False,
        "labelsUsed": False, "protectedTestOpened": False, "trainingEligible": False,
        "interpretation": "Engineering observability check; no manual box ground truth or detector recall estimate. Visual decision separate."})
    print(json.dumps({name: {k: v for k, v in summary.items() if k != "timings"} for name, summary in summaries.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not str(args.output).startswith(private_value('private-reference-0060')):
        raise ValueError("this study writes only to the direct NAS mount")
    for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME"):
        path = args.output / "runtime" / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.output / "python-dependencies"))
    if args.action == "freeze":
        if args.manifest is None:
            parser.error("--manifest required for freeze")
        freeze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
