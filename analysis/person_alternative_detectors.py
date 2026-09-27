"""CPU-only, fixed-policy adapters for a bounded person-observability study.

These adapters are research-only. A person box does not establish a volleyball
player, a court side, a persistent identity, or a rally label.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

YOLOX_COMMIT = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"
SCORE_THRESHOLD = 0.25
MAXIMUM_DETECTIONS = 24


def sanitize_boxes(boxes, scores, width: int, height: int):
    """Clip inverse-mapped boxes; preserve confidences and deterministic order."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4).copy()
    scores = np.asarray(scores, np.float32).reshape(-1)
    if len(boxes) != len(scores) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("nonfinite or mismatched detections")
    if width < 1 or height < 1:
        raise ValueError("invalid image dimensions")
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    valid = (scores >= SCORE_THRESHOLD) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    indexes = np.flatnonzero(valid).tolist()
    indexes.sort(key=lambda i: (-float(scores[i]), float(boxes[i, 0]), float(boxes[i, 1])))
    all_boxes = [{"box": boxes[i].tolist(), "score": float(scores[i])} for i in indexes]
    return all_boxes[:MAXIMUM_DETECTIONS], len(all_boxes)


def box_geometry_diagnostics(detections, width: int, height: int):
    """Geometric warnings only: overlap is not ground-truth duplicate identity."""
    boxes = np.asarray([row["box"] for row in detections], np.float64).reshape(-1, 4)
    contained_pairs = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            intersect = float(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod())
            aa, ab = float((a[2:] - a[:2]).prod()), float((b[2:] - b[:2]).prod())
            containment = intersect / min(aa, ab)
            if containment >= .8:
                contained_pairs.append({"a": i, "b": j, "containment": containment,
                                        "iou": intersect / (aa + ab - intersect)})
    feet = boxes[:, 3] / height if len(boxes) else np.empty(0)
    return {"count": len(boxes), "highContainmentPairs": contained_pairs,
            "upperImageFeetCount": int((feet < .5).sum()),
            "lowerImageFeetCount": int((feet >= .5).sum()),
            "imageHalfIsCourtGeometry": False,
            "minimumBoxHeightPixels": float(np.min(boxes[:, 3] - boxes[:, 1])) if len(boxes) else None}


class YoloXNanoDetector:
    name = "yolox-nano-416"

    def __init__(self, assets: Path):
        source = assets / f"YOLOX-{YOLOX_COMMIT}"
        sys.path.insert(0, str(source))
        from yolox.exp import get_exp
        from yolox.utils import postprocess
        spec = importlib.util.spec_from_file_location("qualification_yolox_augment", source / "yolox/data/data_augment.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.preprocess = module.ValTransform(legacy=False)
        self.postprocess = postprocess
        exp = get_exp(str(source / "exps/default/yolox_nano.py"), None)
        self.model = exp.get_model().eval().cpu()
        checkpoint = torch.load(assets / "yolox_nano.pth", map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.parameters = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def detect(self, frame):
        started = time.perf_counter()
        height, width = frame.shape[:2]
        ratio = min(416 / height, 416 / width)
        tensor, _ = self.preprocess(frame, None, (416, 416))
        tensor = torch.from_numpy(tensor).unsqueeze(0)
        forward_started = time.perf_counter()
        output = self.model(tensor)
        forward_ms = (time.perf_counter() - forward_started) * 1000
        output = self.postprocess(output, 80, SCORE_THRESHOLD, .45, class_agnostic=False)[0]
        if output is None:
            boxes, scores = np.empty((0, 4)), np.empty(0)
        else:
            output = output[output[:, 6] == 0].numpy()
            boxes, scores = output[:, :4] / ratio, output[:, 4] * output[:, 5]
        result, count = sanitize_boxes(boxes, scores, width, height)
        return {"detections": result, "uncappedPersonCount": count, "forwardMs": forward_ms,
                "preForwardPostMs": (time.perf_counter() - started) * 1000}


class SSDliteDetector:
    name = "ssdlite320-mobilenetv3-large"

    def __init__(self, assets: Path):
        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=91,
            score_thresh=SCORE_THRESHOLD, nms_thresh=.55).eval().cpu()
        checkpoint = torch.load(assets / "ssdlite320_mobilenet_v3_large_coco-a79551df.pth",
                                map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint, strict=True)
        self.parameters = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def detect(self, frame):
        started = time.perf_counter()
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
        # Separate official transform/backbone/head/postprocess for comparable
        # forward-only timing, preserving the official SSD inference operations.
        images, _ = self.model.transform([tensor], None)
        forward_started = time.perf_counter()
        features = self.model.backbone(images.tensors)
        features = list(features.values())
        outputs = self.model.head(features)
        forward_ms = (time.perf_counter() - forward_started) * 1000
        anchors = self.model.anchor_generator(images, features)
        predictions = self.model.postprocess_detections(outputs, anchors, images.image_sizes)
        predictions = self.model.transform.postprocess(predictions, images.image_sizes, [(height, width)])[0]
        mask = predictions["labels"] == 1  # torchvision COCO includes background ID 0.
        result, count = sanitize_boxes(predictions["boxes"][mask].numpy(),
                                       predictions["scores"][mask].numpy(), width, height)
        return {"detections": result, "uncappedPersonCount": count, "forwardMs": forward_ms,
                "preForwardPostMs": (time.perf_counter() - started) * 1000}
