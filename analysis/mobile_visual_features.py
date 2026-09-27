"""Label-blind, frozen MobileNetV3 regional features on media presentation time.

This is an experiment extractor, not a production inference implementation.
It reads source video frames rather than the 192x108 AV feature representation.
The four spatial pools are image-relative hypotheses, not detected court zones.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 1
MODEL_NAME = "mobilenet_v3_small"
WEIGHTS_NAME = "MobileNet_V3_Small_Weights.IMAGENET1K_V1"
WEIGHTS_URL = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
TOKEN_COUNT = 4
EMBEDDING_DIMENSION = 576
REGION_NAMES = ("global", "near_image_lower_half", "far_image_upper_half", "net_image_middle_band")
QUALITY_NAMES = ("content_fraction", "luminance_mean", "luminance_std", "laplacian_variance",
                 "clipped_pixel_fraction", "selected_pts_offset_seconds")
RGB_MEAN = np.asarray((.485, .456, .406), np.float32)
RGB_STD = np.asarray((.229, .224, .225), np.float32)
DECODER = "sequential-opencv-nearest-media-pts-v1"


class MobileVisualError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MobileVisualError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class MobileVisualConfig:
    sample_fps: float = 2.
    input_size: int = 224
    batch_size: int = 16

    def validate(self) -> None:
        require(self.sample_fps == 2., "initial mobile visual contract uses exactly 2Hz")
        require(self.input_size == 224, "initial mobile visual contract uses 224x224")
        require(isinstance(self.batch_size, int) and self.batch_size > 0, "batch_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"sampleFps": self.sample_fps, "inputSize": self.input_size,
                "resize": "single native-source ROI, aspect-preserving bilinear letterbox; black padding",
                "normalization": {"mean": RGB_MEAN.tolist(), "std": RGB_STD.tolist()},
                "spatialMap": "torchvision mobilenet_v3_small.features final 576-channel map",
                "pooling": "fractional-area weighted averages in content coordinates; excludes padding",
                "regionNames": list(REGION_NAMES), "regionVerticalFractions": [[0., 1.], [.5, 1.], [0., .5], [.4, .6]],
                "regionInterpretation": "Image-relative bands, not calibrated court/near-far/net detections",
                "sampling": "nearest media PTS to exact 2Hz grid; earlier frame wins ties",
                "maximumPtsErrorSeconds": .25, "decoder": DECODER}

    @property
    def config_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def normalize_roi(roi: Any) -> tuple[float, float, float, float]:
    if roi is None:
        values = (0., 0., 1., 1.)
    elif isinstance(roi, Mapping):
        values = tuple(float(roi[k]) for k in ("x", "y", "width", "height"))
    else:
        values = tuple(float(v) for v in roi)
    require(len(values) == 4 and all(math.isfinite(v) for v in values), "ROI must contain four finite values")
    x, y, width, height = values
    require(x >= 0 and y >= 0 and width > 0 and height > 0
            and x + width <= 1. + 1e-9 and y + height <= 1. + 1e-9, "ROI must lie inside normalized image bounds")
    return values


def preprocess_frame(frame: np.ndarray, roi: Any = None, input_size: int = 224) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BGR uint8 -> normalized CHW, content box (left,top,right,bottom), quality.

    Coordinates are normalized to the letterboxed input. Quality excludes padding.
    The last quality field (PTS offset) is populated by the decoder later.
    """
    import cv2
    require(frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8 and frame.size > 0,
            "expected nonempty BGR uint8 frame")
    require(input_size > 0, "input_size must be positive")
    x, y, width, height = normalize_roi(roi)
    source_h, source_w = frame.shape[:2]
    x0, y0 = int(round(x * source_w)), int(round(y * source_h))
    x1, y1 = min(source_w, int(round((x + width) * source_w))), min(source_h, int(round((y + height) * source_h)))
    require(x1 > x0 and y1 > y0, "ROI is empty at source resolution")
    crop = frame[y0:y1, x0:x1]
    scale = min(input_size / crop.shape[1], input_size / crop.shape[0])
    rw, rh = max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))
    rgb = cv2.resize(crop[:, :, ::-1], (rw, rh), interpolation=cv2.INTER_LINEAR)
    left, top = (input_size - rw) // 2, (input_size - rh) // 2
    canvas = np.zeros((input_size, input_size, 3), np.uint8)
    canvas[top:top + rh, left:left + rw] = rgb
    normalized = (canvas.astype(np.float32) / 255. - RGB_MEAN) / RGB_STD
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    quality = np.asarray([rw * rh / input_size ** 2, gray.mean(), gray.std(),
                          cv2.Laplacian(gray, cv2.CV_32F).var(),
                          np.mean((gray <= 2 / 255.) | (gray >= 253 / 255.)), 0.], np.float32)
    box = np.asarray([left, top, left + rw, top + rh], np.float64) / input_size
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)), box, quality


def regional_pool_weights(content_boxes: np.ndarray, map_height: int, map_width: int) -> np.ndarray:
    """Fractional overlap of feature-map cells with each content-relative band."""
    boxes = np.asarray(content_boxes, np.float64)
    require(boxes.ndim == 2 and boxes.shape[1] == 4 and len(boxes) > 0 and np.isfinite(boxes).all(), "invalid content boxes")
    require(map_height > 0 and map_width > 0, "invalid feature-map dimensions")
    require(np.all(boxes[:, :2] >= 0) and np.all(boxes[:, 2:] <= 1)
            and np.all(boxes[:, 2:] > boxes[:, :2]), "content boxes must have positive area in unit image")
    weights = np.empty((len(boxes), TOKEN_COUNT, map_height, map_width), np.float64)
    ys, xs = np.arange(map_height + 1) / map_height, np.arange(map_width + 1) / map_width
    for i, (left, top, right, bottom) in enumerate(boxes):
        for j, (low, high) in enumerate(((0., 1.), (.5, 1.), (0., .5), (.4, .6))):
            region_top, region_bottom = top + low * (bottom - top), top + high * (bottom - top)
            overlap_x = np.maximum(0., np.minimum(xs[1:], right) - np.maximum(xs[:-1], left))
            overlap_y = np.maximum(0., np.minimum(ys[1:], region_bottom) - np.maximum(ys[:-1], region_top))
            area = overlap_y[:, None] * overlap_x[None, :]
            require(area.sum() > 0, "regional pool has no content")
            weights[i, j] = area / area.sum()
    return np.ascontiguousarray(weights, np.float32)


class MobileVisualBackbone:
    def __init__(self, model: Any, *, checkpoint: Path, checkpoint_sha256: str, device: str = "cpu") -> None:
        import torch
        import torchvision
        self.torch = torch
        self.model = model.to(device).eval()
        self.device = device
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._identity = {"modelName": MODEL_NAME, "weightsName": WEIGHTS_NAME, "weightsUrl": WEIGHTS_URL,
                          "checkpointSha256": checkpoint_sha256, "torch": torch.__version__,
                          "torchvision": torchvision.__version__, "tokenCount": TOKEN_COUNT,
                          "embeddingDimension": EMBEDDING_DIMENSION, "frozen": True}
        self.checkpoint = Path(checkpoint)

    def identity(self) -> dict[str, Any]:
        return dict(self._identity)

    def embed_frames(self, frames: Sequence[np.ndarray], roi: Any = None, *, input_size: int = 224) -> tuple[np.ndarray, np.ndarray]:
        require(bool(frames), "at least one frame is required")
        inputs, boxes, quality = zip(*(preprocess_frame(frame, roi, input_size) for frame in frames))
        tensor = self.torch.from_numpy(np.stack(inputs)).to(self.device)
        with self.torch.inference_mode():
            spatial = self.model.features(tensor)
            require(spatial.ndim == 4 and spatial.shape[1] == EMBEDDING_DIMENSION, "unexpected MobileNet feature map")
            weights = self.torch.from_numpy(regional_pool_weights(np.stack(boxes), spatial.shape[2], spatial.shape[3])).to(spatial.device)
            pooled = self.torch.einsum("bchw,brhw->brc", spatial.float(), weights)
        tokens = pooled.detach().cpu().numpy().astype(np.float32)
        require(tokens.shape == (len(frames), TOKEN_COUNT, EMBEDDING_DIMENSION) and np.isfinite(tokens).all(), "invalid embedding output")
        return np.ascontiguousarray(tokens), np.stack(quality)


def load_mobile_backbone(checkpoint: str | Path, *, checkpoint_sha256: str | None = None,
                         allow_download: bool = False, device: str = "cpu") -> MobileVisualBackbone:
    """Load the explicit ImageNet V1 artifact; downloads require an opt-in flag."""
    import torch
    from torchvision.models import mobilenet_v3_small
    path = Path(checkpoint).expanduser().resolve()
    if not path.exists():
        require(allow_download, "checkpoint missing; use --allow-download to fetch the pinned official artifact")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix="mobilenet-", suffix=".pth", dir=path.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            torch.hub.download_url_to_file(WEIGHTS_URL, str(temporary), hash_prefix="047dcff4", progress=True)
            require(not path.exists(), "checkpoint appeared during download")
            temporary.rename(path)
        finally:
            temporary.unlink(missing_ok=True)
    actual_sha = sha256_file(path)
    require(actual_sha.startswith("047dcff4"), "checkpoint does not match the official ImageNet V1 hash prefix")
    if checkpoint_sha256 is not None:
        require(actual_sha == checkpoint_sha256.lower(), "checkpoint full SHA-256 mismatch")
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return MobileVisualBackbone(model, checkpoint=path, checkpoint_sha256=actual_sha, device=device)


def sample_selection(presentation_times: np.ndarray, duration: float, sample_fps: float = 2.) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(presentation_times, np.float64)
    require(pts.ndim == 1 and len(pts) > 0 and np.isfinite(pts).all() and np.all(np.diff(pts) > 0), "invalid presentation timeline")
    require(abs(float(pts[0])) <= 1e-6, "nonzero video origin requires an explicit alignment contract")
    require(math.isfinite(duration) and duration > 0 and math.isfinite(sample_fps) and sample_fps > 0, "invalid duration/cadence")
    times = np.arange(int(math.ceil(duration * sample_fps - 1e-9)), dtype=np.float64) / sample_fps
    times = times[times < duration]
    right = np.minimum(np.searchsorted(pts, times), len(pts) - 1)
    left = np.maximum(0, right - 1)
    indexes = np.where(abs(times - pts[left]) <= abs(pts[right] - times), left, right).astype(np.int64)
    require(len(times) > 0 and np.max(abs(pts[indexes] - times)) <= .5 / sample_fps + 1e-9, "PTS coverage exceeds half-tick tolerance")
    require(np.all(np.diff(indexes) > 0), "sampling unexpectedly repeats display frames")
    return times, indexes


def selected_frames(capture: Any, indexes: np.ndarray, pts: np.ndarray, position_msec: int, *, partial: bool = False):
    """Sequential display-order decode, with actual timestamps checked at samples."""
    ordinal, cursor = -1, 0
    while (not partial or ordinal + 1 < len(pts)) and capture.grab():
        ordinal += 1
        if cursor >= len(indexes) or ordinal != indexes[cursor]:
            continue
        ok, frame = capture.retrieve()
        require(ok and frame is not None and frame.size > 0, "selected frame could not decode")
        actual = float(capture.get(position_msec)) / 1000.
        require(math.isfinite(actual) and abs(actual - pts[ordinal]) <= 2e-6, "decoded PTS differs from inventory")
        cursor += 1
        yield ordinal, frame, actual
    require(ordinal + 1 == len(pts) and cursor == len(indexes), "truncated or inconsistent sequential decode")


def presentation_inventory(video: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets", "-show_entries",
               "packet=pts:stream=time_base,duration,nb_frames,start_time,codec_name,width,height", "-of", "json", str(video)]
    payload = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    stream = payload["streams"][0]
    ticks = np.sort(np.asarray([int(packet["pts"]) for packet in payload["packets"]], np.int64))
    numerator, denominator = (int(v) for v in stream["time_base"].split("/"))
    require(len(ticks) == int(stream["nb_frames"]) and np.all(np.diff(ticks) > 0), "packet inventory is not one PTS per display frame")
    pts = ticks.astype(np.float64) * numerator / denominator
    return pts, {"stream": stream, "inventoryCommand": command,
                 "presentationTimesSha256": hashlib.sha256(pts.astype("<f8").tobytes()).hexdigest()}


@dataclass(frozen=True)
class MobileVisualCache:
    path: Path
    timestamps: np.ndarray
    tokens: np.ndarray
    quality: np.ndarray
    selected_presentation_times: np.ndarray
    metadata: Mapping[str, Any]


def load_mobile_visual_cache(path: str | Path, *, expected_identity: Mapping[str, Any] | None = None) -> MobileVisualCache:
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as payload:
            times = np.asarray(payload["timestamps"], np.float64)
            tokens = np.asarray(payload["tokens"], np.float32)
            quality = np.asarray(payload["quality"], np.float32)
            selected = np.asarray(payload["selected_presentation_times"], np.float64)
            metadata = json.loads(str(payload["metadata_json"].item()))
    except (OSError, KeyError, ValueError) as error:
        raise MobileVisualError(f"invalid mobile visual cache {path}: {error}") from error
    require(isinstance(metadata, dict) and metadata.get("schemaVersion") == SCHEMA_VERSION
            and metadata.get("completed") is True and metadata.get("labelsUsed") is False, "invalid cache completion/schema")
    require(times.ndim == 1 and len(times) > 0 and np.all(np.diff(times) > 0) and times[0] == 0., "invalid cache time grid")
    require(tokens.shape == (len(times), TOKEN_COUNT, EMBEDDING_DIMENSION)
            and quality.shape == (len(times), len(QUALITY_NAMES)) and selected.shape == times.shape,
            "invalid cache array dimensions")
    require(all(np.isfinite(v).all() for v in (times, tokens, quality, selected)), "nonfinite cache values")
    require(np.allclose(times, np.arange(len(times)) / 2., atol=1e-10, rtol=0)
            and np.max(abs(selected - times)) <= .25 + 1e-9, "cache sample cadence/PTS contract differs")
    require(np.allclose(quality[:, -1], selected - times, atol=1e-7, rtol=0), "cache PTS quality differs")
    if expected_identity is not None:
        require(metadata.get("identity") == dict(expected_identity), "cache identity mismatch")
    return MobileVisualCache(path, times, tokens, quality, selected, metadata)


def align_mobile_features(cache: MobileVisualCache, target_times: np.ndarray, *, maximum_age: float = .5) -> dict[str, np.ndarray]:
    """Causal hold on the declared feature grid, with explicit age/missing flags.

    Selection itself is nearest-frame offline sampling: a frame can be up to .25s
    later than its grid tick. This is recorded, not advertised as online causality.
    """
    target = np.asarray(target_times, np.float64)
    require(target.ndim == 1 and np.isfinite(target).all() and np.all(np.diff(target) >= 0), "invalid target times")
    require(math.isfinite(maximum_age) and maximum_age > 0, "maximum_age must be positive")
    indexes = np.searchsorted(cache.timestamps, target, side="right") - 1
    clipped = np.maximum(indexes, 0)
    ages = target - cache.timestamps[clipped]
    valid = (indexes >= 0) & (ages >= 0) & (ages < maximum_age + 1e-9)
    tokens, quality = cache.tokens[clipped].copy(), cache.quality[clipped].copy()
    tokens[~valid], quality[~valid] = 0., 0.
    return {"tokens": tokens, "quality": quality, "feature_age_seconds": np.where(valid, ages, 0.).astype(np.float32),
            "available": valid.astype(np.float32), "source_indexes": np.where(valid, indexes, -1)}


def extract_recording_cache(recording: Mapping[str, Any], backbone: MobileVisualBackbone,
                            config: MobileVisualConfig, cache_dir: str | Path, *, progress: Any = None,
                            max_seconds: float | None = None) -> tuple[MobileVisualCache, str]:
    """Extract one content-addressed cache without reading labels or AV arrays."""
    import cv2
    config.validate()
    require(max_seconds is None or math.isfinite(max_seconds) and max_seconds > 0, "max_seconds must be positive")
    video = Path(recording["video"]).expanduser().resolve()
    recording_id = str(recording["id"])
    source_sha = str(recording["contentSha256"]).lower()
    require(bool(re.fullmatch(r"[0-9a-f]{64}", source_sha)), "manifest source SHA-256 is required")
    roi = normalize_roi(recording.get("roi"))
    identity = {"recordingId": recording_id, "recordingContentSha256": source_sha, "roi": list(roi),
                "config": config.to_dict(), "backbone": backbone.identity(),
                "extractorSourceSha256": sha256_file(__file__), "opencv": cv2.__version__,
                "opencvBuildSha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
                "maximumExtractionSeconds": max_seconds}
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", recording_id)
    destination = Path(cache_dir) / f"{safe_id}-{canonical_sha256(identity)[:20]}.npz"
    if destination.exists():
        return load_mobile_visual_cache(destination, expected_identity=identity), "reused"
    before = video.stat()
    require(sha256_file(video) == source_sha, "source video SHA-256 differs from manifest")
    pts, inventory = presentation_inventory(video)
    video_duration = float(inventory["stream"]["duration"])
    duration = min(video_duration, max_seconds) if max_seconds is not None else video_duration
    partial = duration < video_duration
    if partial:
        pts = pts[pts < duration]
    times, indexes = sample_selection(pts, duration, config.sample_fps)
    capture = cv2.VideoCapture(str(video))
    require(capture.isOpened(), "cannot open source video")
    batches, qualities, batch_frames, hashes, observed = [], [], [], [], []
    try:
        for _, frame, actual in selected_frames(capture, indexes, pts, cv2.CAP_PROP_POS_MSEC, partial=partial):
            hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
            observed.append(actual)
            batch_frames.append(frame)
            if len(batch_frames) == config.batch_size or len(observed) == len(times):
                tokens, quality = backbone.embed_frames(batch_frames, roi, input_size=config.input_size)
                batches.append(tokens)
                qualities.append(quality)
                batch_frames = []
                if progress is not None:
                    progress(f"{recording_id}: {len(observed)}/{len(times)} mobile visual frames")
    finally:
        capture.release()
    after = video.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "source changed during extraction")
    tokens = np.concatenate(batches)
    quality = np.concatenate(qualities)
    quality[:, -1] = np.asarray(observed) - times
    require(np.isfinite(tokens).all() and np.max(abs(tokens)) <= np.finfo(np.float16).max, "embedding not representable as finite float16")
    metadata = {"schemaVersion": SCHEMA_VERSION, "kind": "volleycut-frozen-mobile-visual-cache",
                "createdAt": datetime.now(timezone.utc).isoformat(), "identity": identity,
                "recordingId": recording_id, "recordingContentSha256": source_sha,
                "sourceVideoPath": str(video), "sourceSizeBytes": before.st_size, "sourceMtimeNs": before.st_mtime_ns,
                "video": inventory["stream"], "decoderInventory": inventory, "labelsUsed": False, "completed": True,
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "device": backbone.device,
                            "batchSize": config.batch_size}, "qualityNames": list(QUALITY_NAMES),
                "array": {"shape": list(tokens.shape), "dtype": "float16", "regionNames": list(REGION_NAMES)},
                "timestamps": {"count": len(times), "first": float(times[0]), "last": float(times[-1]),
                               "sampleFps": 2., "sha256": hashlib.sha256(times.astype("<f8").tobytes()).hexdigest()},
                "extractedDurationSeconds": duration, "partialVideo": partial,
                "alignment": "hold most recent nominal2Hz feature at AV ticks; pass age, availability and signed sourcePTS offset",
                "maximumDecodedPtsErrorSeconds": float(np.max(abs(np.asarray(observed) - pts[indexes])))}
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".mobile-", suffix=".npz", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, timestamps=times, tokens=tokens.astype(np.float16), quality=quality,
                                selected_presentation_times=np.asarray(observed, np.float64), selected_ordinals=indexes,
                                selected_frame_sha256=np.asarray(hashes), metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
            handle.flush()
            os.fsync(handle.fileno())
        load_mobile_visual_cache(name, expected_identity=identity)
        # Atomic publication without replacing an existing immutable artifact.
        os.link(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)
    return load_mobile_visual_cache(destination, expected_identity=identity), "created"
