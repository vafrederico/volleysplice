"""Pinned OpenCV Zoo NanoDet person adapter using its official inference code.

The exact 2022nov graph is identified by bytes rather than the inconsistent
upstream README architecture name. NanoDet returns person boxes, not pose
landmarks: anatomical-looking fields below are explicitly box-derived proxies.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from analysis.side_switch_player_detector import (PlayerDetection, PersonDetectionResult,
    _Tile, detector_tiles, sha256_file)


MODEL_ID = "opencv-zoo-nanodet-416-fp32-2022nov"
UPSTREAM_COMMIT = "47534e27c9851bb1128ccc0102f1145e27f23f98"
PINS = {
    "model.onnx": "4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186",
    "nanodet.py": "5ba84922eaefdc48d0066cffdfb721d0d434c5f3b92213563161e00da26cea1b",
    "demo.py": "43aee38581c4b9c22e5aa16317082f998b4a5164b80f4440f3d692f8e30854e1",
    "README.md": "718e15301612de9735d478c573bf876aaa3551912b2c693bd9275a9d4a4d85d0",
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}
SCORE_THRESHOLD, NMS_THRESHOLD = .35, .6


def nanodet_identity(model_dir: str | Path) -> dict[str, Any]:
    directory = Path(model_dir).expanduser().resolve()
    metadata = json.loads((directory/"model.json").read_text(encoding="utf-8"))
    if metadata.get("modelId") != MODEL_ID or metadata.get("upstreamCommit") != UPSTREAM_COMMIT:
        raise ValueError("unexpected NanoDet model identity")
    artifacts = {}
    for name, digest in PINS.items():
        path = directory/name
        if sha256_file(path) != digest:
            raise ValueError(f"NanoDet pinned artifact mismatch: {name}")
        artifacts[name] = {"path": str(path), "sha256": digest, "sizeBytes": path.stat().st_size}
    return {"modelId": MODEL_ID, "directory": str(directory), "upstreamCommit": UPSTREAM_COMMIT,
        "metadataPath": str(directory/"model.json"), "metadataSha256": sha256_file(directory/"model.json"),
        "modelPath": str(directory/"model.onnx"), "modelSha256": PINS["model.onnx"],
        "modelSizeBytes": artifacts["model.onnx"]["sizeBytes"], "artifacts": artifacts,
        "architectureQualification": metadata["architectureQualification"]}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import pinned source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def box_to_player(box: np.ndarray, score: float) -> PlayerDetection:
    """Return torso/hip *proxies* for unchanged motion reducers, not measured pose."""
    left, top, right, bottom = (float(v) for v in box)
    width, height = right-left, bottom-top
    if not np.isfinite(box).all() or width <= 0 or height <= 0:
        raise ValueError("person box must be finite and positive")
    return PlayerDetection(left+.20*width, top+.20*height, .60*width, .45*height,
        left+.5*width, top+.60*height, left+.5*width, top+.25*height, float(score))


class NanoDetPersonDetector:
    def __init__(self, model_dir: str | Path, *, maximum_detections: int = 24, opencv_threads: int = 2,
                 tile_mode: str = "single-roi"):
        if maximum_detections < 12 or opencv_threads < 1:
            raise ValueError("invalid detector cap/thread count")
        if tile_mode not in {"single-roi", "four-tiles"}:
            raise ValueError("unknown NanoDet view policy")
        self.tile_mode = tile_mode
        self.forward_passes_per_frame = 1 if tile_mode == "single-roi" else 4
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.metadata = nanodet_identity(self.model_dir)
        self.maximum_detections = maximum_detections
        official = _load_module("volleycut_official_nanodet", self.model_dir/"nanodet.py")
        # The pinned demo imports `nanodet`; bind it only during import so we can
        # reuse its exact letterbox/unletterbox functions without global changes.
        previous = sys.modules.get("nanodet")
        try:
            sys.modules["nanodet"] = official
            self.demo = _load_module("volleycut_official_nanodet_demo", self.model_dir/"demo.py")
        finally:
            if previous is None:
                sys.modules.pop("nanodet", None)
            else:
                sys.modules["nanodet"] = previous
        cv2.setNumThreads(opencv_threads)
        self.model = official.NanoDet(str(self.model_dir/"model.onnx"),
            prob_threshold=SCORE_THRESHOLD, iou_threshold=NMS_THRESHOLD,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV, target_id=cv2.dnn.DNN_TARGET_CPU)
        self.selection_metadata = {"maximumDetections": maximum_detections, "perSideCap": None,
            "classId": 0, "className": "person", "scoreThreshold": SCORE_THRESHOLD,
            "nmsIou": NMS_THRESHOLD, "viewPolicy": tile_mode,
            "tiles": "single full native ROI" if tile_mode == "single-roi" else "four 0.62-coverage ownership tiles",
            "forwardPassesPerFrame": self.forward_passes_per_frame,
            "prePostProcessing": "Pinned official NanoDet.infer plus demo letterbox/unletterbox",
            "proxyGeometry": "torso=x20%-80%,y20%-65%; hip=centerX,y60%; shoulder=centerX,y25% of person box; not pose landmarks",
            "rawCandidateCount": "post-NMS person boxes before tile ownership and global NMS",
            "tracking": "greedy-spatial-scale-gated-short-association; not persistent identity"}
        self.last_body_boxes: list[dict[str, Any]] = []

    def detect(self, frame: np.ndarray) -> PersonDetectionResult:
        height, width = frame.shape[:2]
        boxes, scores = [], []
        raw_count, elapsed = 0, 0.
        tiles = (_Tile(0,0,width,height,0,0),) if self.tile_mode == "single-roi" else detector_tiles(width, height)
        for tile in tiles:
            crop = frame[tile.y:tile.y+tile.height, tile.x:tile.x+tile.width]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image, scale = self.demo.letterbox(rgb)
            started = time.perf_counter()
            predictions = self.model.infer(image)
            elapsed += (time.perf_counter()-started)*1000
            for pred in predictions:
                if int(pred[-1]) != 0:
                    continue
                raw_count += 1
                box = np.asarray(self.demo.unletterbox(pred[:4], crop.shape[:2], scale), np.float64)
                box += [tile.x, tile.y, tile.x, tile.y]
                box[[0,2]] = np.clip(box[[0,2]], 0, width)
                box[[1,3]] = np.clip(box[[1,3]], 0, height)
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                proxy = box_to_player(box, float(pred[-2]))
                if (self.tile_mode == "four-tiles" and
                        (int(proxy.hip_x >= width*.5), int(proxy.hip_y >= height*.5)) != (tile.owner_column, tile.owner_row)):
                    continue
                boxes.append(box)
                scores.append(float(pred[-2]))
        if boxes:
            xywh = [[float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1])] for b in boxes]
            kept = np.asarray(cv2.dnn.NMSBoxes(xywh, scores, SCORE_THRESHOLD, NMS_THRESHOLD)).reshape(-1).tolist()
            kept = sorted(kept, key=lambda i: (-scores[i], float(boxes[i][0]), float(boxes[i][1])))[:self.maximum_detections]
        else:
            kept = []
        self.last_body_boxes = [{"box": boxes[i].tolist(), "score": scores[i]} for i in kept]
        return PersonDetectionResult(tuple(box_to_player(boxes[i], scores[i]) for i in kept), raw_count, elapsed)
