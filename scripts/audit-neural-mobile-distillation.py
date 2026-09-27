#!/usr/bin/env python3
"""Independent, read-only student/temporal audit; CPU replay, never fitting.

Reuses the pinned prior independent scaler/exposure and endpoint-sweep auditors.
Student membership, valid frame selection, BN buffers, image/teacher lineage,
strict 99% selection and partial-scope handling are reconstructed separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import importlib.util
from itertools import combinations
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
LEGACY_AUDITOR_SHA = "d8892b6335569e9622f3f3fa659c4e2f93861084c92cf104535084355683bdff"
TIERS = ("exact", "draft", "coverage")
TEACHER_RTOL, TEACHER_ATOL = 1e-4, 3e-4
FEATURE_RTOL, FEATURE_ATOL = 1e-4, 2e-4


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity(path):
    return {"path": str(path), "sha256": digest(path)}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Evidence:
    def __init__(self):
        self.files = {}

    def bind(self, reference):
        require(isinstance(reference, dict) and set(reference) >= {"path", "sha256"}, "Missing artifact binding")
        path = Path(reference["path"])
        stat = path.stat()
        state = (reference["sha256"], stat.st_size, stat.st_mtime_ns)
        if str(path) in self.files:
            require(self.files[str(path)]["state"] == state, "Artifact changed during audit: " + str(path))
        else:
            require(digest(path) == reference["sha256"], "Artifact hash mismatch: " + str(path))
            self.files[str(path)] = {"reference": reference, "state": state}
        return path

    def capture(self, path):
        ref = identity(path)
        self.bind(ref)
        return ref

    def finish(self):
        for path, record in self.files.items():
            stat = Path(path).stat()
            require((stat.st_size, stat.st_mtime_ns) == record["state"][1:], "Evidence changed while auditing")
        return [r["reference"] for r in self.files.values()]


def legacy_helpers(evidence):
    path = REPO / "scripts/audit-neural-recognition.py"
    evidence.bind({"path": str(path), "sha256": LEGACY_AUDITOR_SHA})
    spec = importlib.util.spec_from_file_location("independent_recognition_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tensor, interval = module.helpers()
    for name, sha in module.HELPERS.items():
        evidence.bind({"path": str(REPO / "scripts" / name), "sha256": sha})
    return module, tensor, interval


def independent_members(data, excluded):
    """Exclude every held source in every tier, without using trainer helpers."""
    rows = {tier: [r for r in data[tier] if r.example.group not in excluded] for tier in TIERS}
    permitted = [r for tier in TIERS for r in rows[tier]]
    held = [r for r in data["exact"] if r.example.group in excluded]
    require(rows["exact"] and held, "Invalid source partition")
    require(not set(excluded) & {r.example.group for r in permitted}, "Student source leakage")
    require(len({r.example.id for r in permitted}) == len(permitted), "Duplicate training identity")
    return rows, permitted, held


def check_student_membership(meta, data, excluded, seed, contract_sha):
    rows, permitted, held = independent_members(data, excluded)
    expected = {"contractSha256": contract_sha, "seed": seed, "excludedGroups": sorted(excluded),
                "trainIds": [r.example.id for r in permitted],
                "trainGroups": sorted({r.example.group for r in permitted})}
    for key, value in expected.items():
        require(meta.get(key) == value, "Student membership differs: " + key)
    return rows, permitted, held


def independent_teaching_indexes(times, valid_times, valid):
    # Brute-force nearest valid-grid association makes tie behavior explicit.
    nearest = np.asarray([int(np.argmin(np.abs(valid_times - t))) for t in times])
    require(np.max(np.abs(valid_times[nearest] - times)) <= .251, "Teaching/valid grids differ")
    allowed = np.flatnonzero(valid[nearest])
    require(len(allowed), "No valid teaching image")
    return allowed[np.linspace(0, len(allowed) - 1, min(128, len(allowed)), dtype=np.int64)]


def exact_source_frame(capture, ordinal, presentation_time, expected_sha256, state=None):
    """Recover an exact saved frame despite VFR ordinal-seek rounding.

    OpenCV's ordinal seek can be inaccurate on Pixel VFR footage. Try the fixed
    nine-frame neighborhood first; if absent, decode sequentially from the start
    and reuse that cursor for later samples. Both paths require the original
    decoded SHA and PTS at the unchanged 2 microsecond tolerance.
    """
    import cv2
    state = {} if state is None else state
    def matches(frame):
        actual = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
        return abs(actual - presentation_time) < 2e-6 and hashlib.sha256(frame.tobytes()).hexdigest() == expected_sha256
    if "sequentialOrdinal" not in state:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(ordinal) - 4))
        for _ in range(9):
            success, frame = capture.read()
            if not success:
                break
            if matches(frame):
                return frame
        print("AUDIT VFR seek fallback: sequential source replay", flush=True)
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        state["sequentialOrdinal"] = -1
    require(state["sequentialOrdinal"] < ordinal, "Sequential audit samples must increase")
    while state["sequentialOrdinal"] < ordinal:
        require(capture.grab(), "Sequential source replay truncated")
        state["sequentialOrdinal"] += 1
    success, frame = capture.retrieve()
    require(success and matches(frame), "Exact sampled source SHA and PTS differ in sequential replay")
    return frame


def select99(candidates):
    require(len(candidates) == 192, "Expected four checkpoints times 48 decoders")
    eligible = [r for r in candidates if r["innerR_core"] >= .99]
    return {"recallEligibilityFloor": .99, "feasible": bool(eligible), "candidateCount": len(candidates),
            "eligibleCandidateCount": len(eligible), "maximumInnerRecall": max(r["innerR_core"] for r in candidates),
            "selected": max(eligible, key=lambda r: r["innerF1_padP_coreR"]) if eligible else None}


def verify_registration(study, evidence, old):
    ref = evidence.capture(study / "preregistration.json")
    registration = read(ref["path"])
    c = registration["contract"]
    require(canonical(c) == registration["sha256"], "Student registration hash differs")
    expected = {"kind": "fold-isolated-dino-distilled-mobile-v1", "seeds": [3407, 1729, 20260918],
                "checkpointEpochs": [5, 15, 30, 60], "studentEpochs": 8, "studentBatchSize": 16,
                "studentLearningRate": 1e-4, "studentWeightDecay": 1e-4, "batchNormStatisticsFrozen": True,
                "selectionFloor": .99, "targetPaddingSeconds": 2, "paddingSeconds": [0, 1, 2, 3],
                "joinGapSeconds": 3, "protectedTestOpened": False, "productionPromotionAllowed": False}
    for key, value in expected.items():
        require(c.get(key) == value, "Registered study recipe differs: " + key)
    for name in ("protocol", "source", "images", "teacherTargets", "mobileCheckpoint", "controlRegistration", "engineering"):
        evidence.bind(c[name])
    require(len(c["groups"]) == 4 and c["groups"] == sorted(set(c["groups"])), "Source inventory differs")
    for name, reference in c["code"].items():
        require(Path(name).name == name, "Unsafe archived source name")
        evidence.bind(reference)
        evidence.bind({"path": str(study / "registered-sources" / name), "sha256": reference["sha256"]})
    require({p.name for p in (study / "registered-sources").iterdir()} == set(c["code"]), "Archived source inventory differs")
    control = read(c["controlRegistration"]["path"])
    require(canonical(control["contract"]) == control["sha256"], "Control registration changed")
    for name, sha in control["contract"]["code"].items():
        require(c["code"][name]["sha256"] == sha, "Temporal recipe dependency changed")
    engineering = read(c["engineering"]["path"])
    require(engineering.get("passed") is True and engineering["studentSource"] == c["code"]["neural_mobile_distillation.py"]
            and engineering["protocol"] == c["protocol"], "Engineering gate absent or stale")
    evidence.bind(engineering["source"])
    require(engineering["protectedTestOpened"] is False and engineering["noHeldSourceRallyOutcomesUsed"] is True
            and len(engineering["records"]) == 4 and len(engineering["repeatRuns"]) == 2,
            "Engineering scope/repeat inventory differs")
    require(engineering["repeatRuns"][0] == engineering["repeatRuns"][1], "Engineering determinism failed")
    repeat = engineering["repeatRuns"][0]
    require(repeat["initialSha256"] != repeat["finalSha256"] and repeat["batchNormStatisticsUnchanged"] is True
            and len(repeat["losses"]) == 2 and np.isfinite(repeat["losses"]).all(), "Engineering learning/BN check absent")
    for row in engineering["records"]:
        evidence.bind(row["inputReceipt"])
        require(row["historicalHalfUlpTolerancePassed"] is True and len(row["frameIndexes"]) == 8
                and np.isfinite(row["maximumSpatialDifference"]) and np.isfinite(row["maximumHistoricalTokenDifference"]),
                "Engineering real-frame parity failed")
    return registration, engineering


def load_inputs(contract, evidence):
    from analysis import neural_short_boost_transfer as source
    from analysis.recognition_temporal_model import RecognitionConfig, model_metadata
    data = source.load_data(Path(contract["source"]["path"]), None, with_dino=False)
    for tier in TIERS:
        require([r.example.id for r in data[tier]] == contract["population"][tier + "Rows"], "Input population differs")
    require(sorted({r.example.group for r in data["exact"]}) == contract["groups"], "Exact groups differ")
    config = RecognitionConfig(family="mobile", head="tcn", scalar_dimension=8)
    require(model_metadata(config) == contract["model"], "Temporal architecture changed")
    raw = read(contract["source"]["path"])
    # The historical loader verifies labels/caches and raw/proxy duplicate scope.
    # Bind those references explicitly in this audit's inventory as well.
    evidence.bind(raw["exactManifest"])
    for tier in TIERS:
        for row in raw[tier + "Rows"]:
            for reference in row.get("featureCaches", {}).values():
                if isinstance(reference, dict) and {"path", "sha256"} <= set(reference):
                    evidence.bind(reference)
    return data, config


def audit_images_teacher(contract, data, evidence):
    import cv2
    import torch
    from analysis import mobile_visual_features as mobile, dinov2_embeddings as dino
    from analysis import distillation_image_inputs as inputs_module
    inputs, targets = read(contract["images"]["path"]), read(contract["teacherTargets"]["path"])
    evidence.bind(inputs["registration"])
    evidence.bind(targets["registration"])
    image_registration = read(inputs["registration"]["path"])
    target_registration = read(targets["registration"]["path"])
    require(target_registration["inputIndex"] == contract["images"] and target_registration["precision"] == "float32"
            and target_registration["labelsUsed"] is False, "Teacher input/precision registration differs")
    require(image_registration["protocol"] == contract["protocol"] and image_registration["source"] == contract["source"]
            and image_registration["labelsUsed"] is False and image_registration["protectedTestOpened"] is False,
            "Image registration source/scope differs")
    for name in ("sourceCode", "manifest", "originalMobileIndex"):
        evidence.bind(image_registration[name])
    evidence.bind(target_registration["sourceCode"])
    require(target_registration["checkpointSha256"] == contract["teacherCheckpointSha256"], "Teacher checkpoint differs")
    checkpoint = inputs_module.ASSETS / "dinov2_vits14_pretrain.pth"
    evidence.bind({"path": str(checkpoint), "sha256": contract["teacherCheckpointSha256"]})
    backbone = dino.load_pinned_dinov2(inputs_module.ASSETS / ("dinov2-" + target_registration["repositoryCommit"]),
        repository_commit=target_registration["repositoryCommit"], checkpoint=checkpoint,
        checkpoint_sha256=contract["teacherCheckpointSha256"], device="cpu")
    images, teachers = {r["id"]: r for r in inputs["records"]}, {r["id"]: r for r in targets["records"]}
    examples = {r.example.id: r.example for tier in TIERS for r in data[tier]}
    sanitized = read(image_registration["manifest"]["path"])
    manifest_rows = {r["id"]: r for r in sanitized["records"]}
    require(len(images) == len(inputs["records"]) == len(teachers) == len(targets["records"]) == 18
            and set(images) == set(teachers) == set(examples) == set(manifest_rows), "Image/teacher population differs")
    checks = []
    for identifier, image in images.items():
        print("AUDIT image/teacher " + identifier, flush=True)
        teacher, e = teachers[identifier], examples[identifier]
        require(image["sourceGroup"] == teacher["sourceGroup"] == e.group
                and image["contract"]["source"] == manifest_rows[identifier], "Image source association differs")
        require(teacher["input"] == image["arrays"]["teaching336"] and teacher["registration"] == targets["registration"]
                and teacher["labelsUsed"] is False and image["labelsUsed"] is False, "Teacher association/scope differs")
        require(image["contract"]["validMaskSha256"] == hashlib.sha256(e.valid.tobytes()).hexdigest(), "Teaching validity changed")
        for ref in image["contract"]["code"].values():
            evidence.bind(ref)
        original_path = evidence.bind(image["contract"]["originalMobileCache"])
        image_path = evidence.bind(image["arrays"]["images224"])
        teacher_image_path = evidence.bind(image["arrays"]["teaching336"])
        timing_path = evidence.bind(image["arrays"]["timing"])
        target_path = evidence.bind(teacher["output"])
        with np.load(timing_path, allow_pickle=False) as timecache, np.load(original_path, allow_pickle=False) as original:
            times, pts, quality, boxes, teaching = (timecache[k] for k in ("times", "selected_pts", "quality", "boxes", "teaching_indexes"))
            expected = independent_teaching_indexes(times, e.times, e.valid)
            require(np.array_equal(expected, teaching) and expected.tolist() == image["contract"]["teachingIndexes"], "Valid teaching membership differs")
            require(np.array_equal(times, original["timestamps"]) and np.array_equal(pts, original["selected_presentation_times"])
                    and np.array_equal(quality, original["quality"]), "Image timing/quality changed")
            pixels, teaching_pixels = np.load(image_path, mmap_mode="r"), np.load(teacher_image_path, mmap_mode="r")
            values = np.load(target_path, allow_pickle=False)
            require(pixels.shape == (len(times), 3, 224, 224) and pixels.dtype == np.uint8
                    and teaching_pixels.shape == (len(teaching), 3, 336, 336) and teaching_pixels.dtype == np.uint8
                    and values.shape == (len(teaching), 10, 384) and values.dtype == np.float32 and np.isfinite(values).all(),
                    "Image or teacher tensor schema differs")
            selected = sorted({0, len(teaching) // 2, len(teaching) - 1})
            video = Path(manifest_rows[identifier]["video"])
            cap = cv2.VideoCapture(str(video))
            require(cap.isOpened(), "Source video unavailable for sampled frame replay")
            seek_state = {}
            try:
                for index in selected:
                    ordinal = int(original["selected_ordinals"][teaching[index]])
                    frame = exact_source_frame(cap, ordinal, pts[teaching[index]], str(original["selected_frame_sha256"][teaching[index]]), seek_state)
                    normalized, box, q = mobile.preprocess_frame(frame, manifest_rows[identifier]["roi"])
                    normalized_saved = (pixels[teaching[index]].astype(np.float32) / 255. - mobile.RGB_MEAN[:, None, None]) / mobile.RGB_STD[:, None, None]
                    require(np.array_equal(normalized, normalized_saved) and np.array_equal(box, boxes[teaching[index]])
                            and np.array_equal(q[:5], quality[teaching[index], :5]), "Sampled mobile preprocessing differs")
                    expected_teacher = dino.preprocess_frames([frame], [mobile.normalize_roi(manifest_rows[identifier]["roi"])], input_size=336)[0]
                    actual_teacher = (teaching_pixels[index].astype(np.float32) / 255. - dino.IMAGENET_RGB_MEAN[:, None, None]) / dino.IMAGENET_RGB_STD[:, None, None]
                    require(np.array_equal(expected_teacher, actual_teacher), "Teacher pixels are not from the same sampled frame/ROI")
            finally:
                cap.release()
            teacher_batch = (teaching_pixels[selected].astype(np.float32) / 255. - dino.IMAGENET_RGB_MEAN[None, :, None, None]) / dino.IMAGENET_RGB_STD[None, :, None, None]
            with torch.inference_mode():
                replay = dino._extract_feature_tokens(backbone.model, torch, torch.from_numpy(teacher_batch), 336).numpy()
            require(np.allclose(replay, values[selected], rtol=TEACHER_RTOL, atol=TEACHER_ATOL), "CPU teacher replay differs")
            checks.append({"id": identifier, "frames": len(times), "teachingFrames": len(teaching),
                "sampledTeachingIndexes": selected, "sampledImageIndexes": teaching[selected].tolist(),
                "teacherMaximumAbsoluteDifference": float(np.max(np.abs(replay - values[selected]))),
                "sampledPixelsAndSourceHashesExact": True})
    del backbone
    return images, teachers, checks


def audit_prepared_images(index_path):
    """Early image-only gate: no AV loader, model, teacher, or training needed.

    Valid-mask selection is deliberately left to the final artifact audit, which
    binds the label/coverage loader. This gate checks stored selection membership,
    input hashes, original PTS/quality, and first/middle/last actual teaching frames.
    """
    import cv2
    from analysis import mobile_visual_features as mobile, dinov2_embeddings as dino
    cv2.setNumThreads(1)
    evidence = Evidence()
    index_ref = evidence.capture(index_path)
    index = read(index_ref["path"])
    registration = read(evidence.bind(index["registration"]))
    require(registration["kind"] == "distillation-image-inputs-v1"
            and registration["labelsUsed"] is False and registration["protectedTestOpened"] is False,
            "Image registration scope differs")
    for name in ("manifest", "originalMobileIndex", "source", "protocol", "sourceCode"):
        evidence.bind(registration[name])
    manifest = read(registration["manifest"]["path"])
    sources = {r["id"]: r for r in manifest["records"]}
    require(len(index["records"]) == 18 and len({r["id"] for r in index["records"]}) == 18
            and {r["id"] for r in index["records"]} == set(sources), "Image population differs")
    checks = []
    for record in index["records"]:
        identifier = record["id"]
        print("AUDIT prepared image " + identifier, flush=True)
        source = sources[identifier]
        require(record["contract"]["source"] == source and record["sourceGroup"] == source["sourceGroup"]
                and record["labelsUsed"] is False and record["allDecodedFrameHashesMatch"] is True,
                "Prepared image source association differs")
        for ref in record["contract"]["code"].values():
            evidence.bind(ref)
        original_path = evidence.bind(record["contract"]["originalMobileCache"])
        arrays = {k: evidence.bind(v) for k, v in record["arrays"].items()}
        with np.load(arrays["timing"], allow_pickle=False) as timing, np.load(original_path, allow_pickle=False) as original:
            times, pts, quality, boxes, teaching = (timing[k] for k in ("times", "selected_pts", "quality", "boxes", "teaching_indexes"))
            require(teaching.tolist() == record["contract"]["teachingIndexes"] and len(teaching) == record["teachingFrames"] == 128
                    and np.all(np.diff(teaching) > 0) and teaching[0] >= 0 and teaching[-1] < len(times),
                    "Stored teaching membership differs")
            require(np.array_equal(times, original["timestamps"]) and np.array_equal(pts, original["selected_presentation_times"])
                    and np.array_equal(quality, original["quality"]) and record["frames"] == len(times),
                    "Original image timing/quality differs")
            pixels = np.load(arrays["images224"], mmap_mode="r", allow_pickle=False)
            teaching_pixels = np.load(arrays["teaching336"], mmap_mode="r", allow_pickle=False)
            require(pixels.shape == (len(times), 3, 224, 224) and pixels.dtype == np.uint8
                    and teaching_pixels.shape == (128, 3, 336, 336) and teaching_pixels.dtype == np.uint8,
                    "Prepared image tensor schema differs")
            selected = [0, len(teaching) // 2, len(teaching) - 1]
            cap = cv2.VideoCapture(source["video"])
            require(cap.isOpened(), "Source unavailable for image replay")
            seek_state = {}
            try:
                for index_in_teaching in selected:
                    image_index = teaching[index_in_teaching]
                    frame = exact_source_frame(cap, int(original["selected_ordinals"][image_index]), pts[image_index],
                                               str(original["selected_frame_sha256"][image_index]), seek_state)
                    normalized, box, q = mobile.preprocess_frame(frame, source["roi"])
                    saved = (pixels[image_index].astype(np.float32) / 255. - mobile.RGB_MEAN[:, None, None]) / mobile.RGB_STD[:, None, None]
                    require(np.array_equal(normalized, saved) and np.array_equal(box, boxes[image_index])
                            and np.array_equal(q[:5], quality[image_index, :5]), "Sampled student pixel/quality mismatch")
                    expected = dino.preprocess_frames([frame], [mobile.normalize_roi(source["roi"])], input_size=336)[0]
                    teacher = (teaching_pixels[index_in_teaching].astype(np.float32) / 255. - dino.IMAGENET_RGB_MEAN[:, None, None]) / dino.IMAGENET_RGB_STD[:, None, None]
                    require(np.array_equal(expected, teacher), "Teacher image differs from same source frame/ROI")
            finally:
                cap.release()
            checks.append({"id": identifier, "frames": len(times), "teachingFrames": len(teaching),
                "sampledTeachingIndexes": selected, "sampledImageIndexes": teaching[selected].tolist(),
                "sampledPixelsAndSourceHashesExact": True, "originalTimingAndQualityExact": True})
    return {"kind": "independent-distillation-prepared-image-audit-v1", "passed": True,
        "index": index_ref, "registration": index["registration"], "auditor": identity(Path(__file__)),
        "records": checks, "recordings": len(checks), "frames": sum(r["frames"] for r in checks),
        "teachingFrames": sum(r["teachingFrames"] for r in checks), "sampledFrames": 3 * len(checks),
        "scope": "All prepared arrays hashed; all saved PTS and quality identical to historical Mobile cache; three raw decoded source frames and both input sizes per recording replayed exactly. Raw videos not fully rehashed. Valid-mask membership and teacher values deferred to final study audit.",
        "samplerRepair": "OpenCV ordinal seek can misidentify Pixel VFR frames. Try a fixed nine-frame neighborhood, then sequential display-order decode when needed. Both require the original decoded SHA and PTS at unchanged tolerance. Preparation and model sources untouched.",
        "evidence": evidence.finish(), "gpuUsed": False, "encoderExecuted": False,
        "trainingPerformed": False, "protectedTestOpened": False, "finishedAtUtc": datetime.now(UTC).isoformat()}


def audit_student(folder, data, excluded, seed, registration, images, teachers, evidence):
    import torch
    from analysis import mobile_visual_features as mobile
    meta_path = folder / "student/completed.json"
    meta_ref = evidence.capture(meta_path)
    meta = read(meta_path)
    rows, permitted, held = check_student_membership(meta, data, excluded, seed, registration["sha256"])
    membership = []
    weights = []
    group_records = Counter(r.example.group for r in permitted)
    for row in permitted:
        e, record = row.example, images[row.example.id]
        indexes = record["contract"]["teachingIndexes"]
        membership.append({"id": e.id, "group": e.group, "frameIndexes": indexes,
                           "teacher": teachers[e.id]["output"], "image": record["arrays"]["images224"]})
        weights.extend([1. / (group_records[e.group] * len(indexes))] * len(indexes))
    require(meta["membership"] == membership and meta["trainingFrames"] == len(weights), "Student frame/target membership differs")
    require(meta["batchNormStatisticsFrozen"] is True and meta["encoderParameters"] == 927008
            and meta["trainingOnlyProjectorParameters"] == 221568, "Student architecture/BN contract differs")
    require(len(meta["history"]) == 8, "Student epoch count differs")
    rng, exposure = np.random.default_rng(seed), hashlib.sha256()
    for epoch, history in enumerate(meta["history"], start=1):
        exposure.update(rng.permutation(len(weights)).astype("<i8").tobytes())
        require(history["epoch"] == epoch and history["exposureSha256"] == exposure.hexdigest()
                and np.isfinite(history["weightedMeanLoss"]) and -.00001 <= history["weightedMeanLoss"] <= 2.00001,
                "Student history/exposure differs")
    checkpoint = evidence.bind(meta["weights"])
    require(checkpoint == folder / "student/weights.npz", "Student checkpoint path differs")
    initial = mobile.load_mobile_backbone(registration["contract"]["mobileCheckpoint"]["path"],
        checkpoint_sha256=registration["contract"]["mobileCheckpoint"]["sha256"], device="cpu").model.features
    initial.eval()
    template = initial.state_dict()
    with np.load(checkpoint, allow_pickle=False) as state:
        require(set(state.files) == {*('encoder::' + k for k in template), "projector::weight", "projector::bias"}, "Student state inventory differs")
        bn_keys = [k for k in template if k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
        encoder_state, changed_tensors = {}, 0
        for key, original in template.items():
            value = state["encoder::" + key]
            require(value.shape == tuple(original.shape) and np.isfinite(value).all(), "Invalid student tensor")
            require(value.dtype == original.numpy().dtype, "Student tensor dtype differs")
            if key in bn_keys:
                require(np.array_equal(value, original.numpy()), "Frozen BatchNorm buffer changed: " + key)
            elif not np.array_equal(value, original.numpy()):
                changed_tensors += 1
            encoder_state[key] = torch.from_numpy(value.copy())
        require(state["projector::weight"].shape == (384, 576) and state["projector::bias"].shape == (384,)
                and np.isfinite(state["projector::weight"]).all() and np.isfinite(state["projector::bias"]).all(), "Invalid projection tensors")
        require(changed_tensors > 0, "No student encoder weights learned")
    initial.load_state_dict(encoder_state, strict=True)
    initial.eval()
    return initial, rows, permitted, held, {"completed": meta_ref, "weights": meta["weights"],
        "excludedGroups": sorted(excluded), "trainIds": meta["trainIds"], "trainGroups": meta["trainGroups"],
        "trainingFrames": len(weights), "epochs": 8, "exposureSha256": exposure.hexdigest(),
        "initialToFinalBNBuffersExact": True, "checkedBNBuffers": len(bn_keys), "changedEncoderTensors": changed_tensors,
        "lossStart": meta["history"][0]["weightedMeanLoss"], "lossEnd": meta["history"][-1]["weightedMeanLoss"]}


def audit_features(folder, encoder, student_check, permitted, held, images, registration, evidence):
    import torch
    from analysis import mobile_visual_features as mobile
    index_ref = evidence.capture(folder / "features.json")
    index = read(index_ref["path"])
    require(index["student"] == student_check["completed"], "Feature encoder receipt differs")
    needed = permitted + held
    expected_ids = [r.example.id for r in needed]
    require([r["id"] for r in index["records"]] == expected_ids, "Fold feature scope/order differs")
    by_id, checks = {}, []
    for record, row in zip(index["records"], needed):
        e, image = row.example, images[row.example.id]
        require(record["sourceGroup"] == e.group and record["encoder"] == student_check["weights"]
                and record["imageInput"] == image["arrays"] and record["contractSha256"] == registration["sha256"]
                and record["labelsUsed"] is False, "Feature lineage differs")
        path = evidence.bind(record["output"])
        require(path == folder / "features" / (e.id + ".npz"), "Feature path differs")
        evidence.capture(folder / "features" / (e.id + ".json"))
        require(read(folder / "features" / (e.id + ".json")) == record, "Feature index/receipt differs")
        with np.load(path, allow_pickle=False) as feature, np.load(image["arrays"]["timing"]["path"], allow_pickle=False) as timing:
            require(set(feature.files) == {"timestamps", "tokens", "quality", "selected_presentation_times"}, "Feature array inventory differs")
            times, tokens, quality = (feature[k] for k in ("timestamps", "tokens", "quality"))
            require(np.array_equal(times, timing["times"]) and np.array_equal(quality, timing["quality"])
                    and np.array_equal(feature["selected_presentation_times"], timing["selected_pts"]), "Feature timing/scalars differ")
            require(tokens.dtype == np.float16 and tokens.shape == (len(times), 4, 576) and np.isfinite(tokens).all()
                    and record["shape"] == list(tokens.shape), "Student feature schema differs")
            selected = sorted({0, len(times) // 2, len(times) - 1})
            pixels = np.load(image["arrays"]["images224"]["path"], mmap_mode="r")
            x = (pixels[selected].astype(np.float32) / 255. - mobile.RGB_MEAN[None, :, None, None]) / mobile.RGB_STD[None, :, None, None]
            with torch.inference_mode():
                spatial = encoder(torch.from_numpy(x))
                pool = torch.from_numpy(mobile.regional_pool_weights(timing["boxes"][selected], *spatial.shape[-2:]))
                replay = torch.einsum("bchw,brhw->brc", spatial, pool).numpy()
            expected = tokens[selected].astype(np.float32)
            # Account explicitly for the recorded FP16 token rounding.
            ulp = np.spacing(np.abs(tokens[selected])).astype(np.float32)
            bound = .5 * ulp + FEATURE_ATOL + FEATURE_RTOL * np.abs(expected)
            error = np.abs(replay - expected)
            require(np.all(error <= bound), "CPU encoder/FP16 feature replay differs: " + e.id)
            indexes = np.searchsorted(times, e.times, side="right") - 1
            require(np.all(indexes >= 0), "Missing aligned feature")
            ages = e.times - times[indexes]
            require(np.all((ages >= 0) & (ages <= .5 + 1e-9)), "Feature age outside original contract")
            values = np.concatenate((e.values, tokens[indexes].astype(np.float32).reshape(len(e.times), -1),
                quality[indexes], ages.astype(np.float32)[:, None], np.ones((len(e.times), 1), np.float32)), axis=1).astype(np.float32)
            require(np.array_equal(values[:, :104], e.values) and values.shape[1] == 2416 and np.isfinite(values).all(), "Aligned AV changed")
            by_id[e.id] = replace(row, example=replace(e, values=values))
            checks.append({"id": e.id, "output": record["output"], "sampledFrames": selected,
                "encoderMaximumAbsoluteDifference": float(error.max()), "maximumToleranceRatio": float((error / bound).max()),
                "timestampsQualityAndAVExact": True})
    return by_id, {"index": index_ref, "records": checks}


def audit_fit(folder, data, excluded, epochs, seed, registration, images, teachers, config, evidence, old, tensor):
    encoder, rows, permitted, held, student = audit_student(folder, data, excluded, seed, registration, images, teachers, evidence)
    attached, features = audit_features(folder, encoder, student, permitted, held, images, registration, evidence)
    del encoder
    # The legacy independent fit audit needs all exact IDs, including excluded
    # rows; excluded auxiliary rows never enter scaler/exposure or head replay.
    fold_data = {tier: [attached.get(r.example.id, r) for r in data[tier]] for tier in TIERS}
    adapted_registration = {**registration, "contract": {**registration["contract"], "lossArm": "short_boost"}}
    predictions, temporal = old.audit_fit(folder / "temporal", fold_data, excluded, epochs, seed,
        adapted_registration, config, tensor)
    evidence.capture(folder / "temporal/completed.json")
    for name, sha in temporal["checkpointArtifacts"].items():
        evidence.bind({"path": str(folder / "temporal" / name), "sha256": sha})
    return predictions, {"folder": str(folder), "student": student, "features": features, "temporal": temporal}


def reselect(examples, predictions, contract, interval, old):
    from analysis import neural_expanded_development as expanded
    from analysis.crop_evaluation import RecordingIntervals, evaluate_f1_pad_p_core_r
    require(contract["decoderCandidates"] == expanded.decoder_candidates(), "Decoder policy changed")
    candidates = []
    for epoch in sorted(predictions):
        require(set(predictions[epoch]) == {e.id for e in examples}, "Inner source probability scope differs")
        for decoder in contract["decoderCandidates"]:
            rows = [e.row(expanded.base.decode(e, predictions[epoch][e.id], decoder)) for e in examples]
            independent = interval.pooled_metric([interval.parse_record(r) for r in old.serialized_rows(rows)], 2)
            metric = evaluate_f1_pad_p_core_r([RecordingIntervals(r["id"], "development", r["durationSeconds"],
                tuple(r["rallies"]), tuple(r["predictions"]), tuple(r["ignoredIntervals"])) for r in rows], [2.], 3.)[0]
            for key in ("P_pad", "R_core", "F1_padP_coreR"):
                old.compare(metric[key], independent[key], "independent inner " + key)
            candidates.append({"epoch": epoch, "decoder": decoder, "innerF1_padP_coreR": metric["F1_padP_coreR"],
                               "innerR_core": metric["R_core"]})
    return select99(candidates), candidates


def audit_seed(study, seed, registration, data, images, teachers, config, evidence, old, tensor, interval):
    from analysis import neural_expanded_development as expanded
    from analysis.neural_evaluation import evaluate_predictions
    c, owners, checks = registration["contract"], {}, []
    groups = c["groups"]
    folder = study / "fits" / str(seed)
    expected_folders = {f"inner-{a}-{b}" for a, b in combinations(range(4), 2)}
    for a, b in combinations(range(4), 2):
        print(f"AUDIT seed={seed} inner={a}-{b}", flush=True)
        owners[(a, b)], checked = audit_fit(folder / f"inner-{a}-{b}", data, {groups[a], groups[b]},
            tuple(c["checkpointEpochs"]), seed, registration, images, teachers, config, evidence, old, tensor)
        checks.append(checked)
    rows, selections, grids, views = [], [], [], []
    for oi, outer in enumerate(groups):
        examples = [r.example for r in data["exact"] if r.example.group != outer]
        predictions = {epoch: {} for epoch in c["checkpointEpochs"]}
        for ii, inner in enumerate(groups):
            if oi == ii:
                continue
            owner = tuple(sorted((oi, ii)))
            allowed = {r.example.id for r in data["exact"] if r.example.group == inner}
            for epoch in predictions:
                require(not set(predictions[epoch]) & allowed, "Duplicated logical validation recording")
                predictions[epoch].update({key: owners[owner][epoch][key] for key in allowed})
            views.append({"outerGroup": outer, "innerGroup": inner, "owner": f"inner-{owner[0]}-{owner[1]}", "ids": sorted(allowed)})
        selected, candidates = reselect(examples, predictions, c, interval, old)
        selected["heldSourceGroup"] = outer
        stored_selection = read(evidence.capture(study / "selections" / f"{seed}-{oi}.json")["path"])
        old.compare(stored_selection, {"contractSha256": registration["sha256"], "candidates": candidates, **selected}, "strict 99% selection")
        selections.append(selected)
        grids.append({"heldSourceGroup": outer, "candidates": candidates})
        if not selected["feasible"]:
            require(not (folder / f"outer-{oi}").exists(), "Infeasible fold has unauthorized fallback fit")
            continue
        expected_folders.add(f"outer-{oi}")
        print(f"AUDIT seed={seed} outer={oi}", flush=True)
        chosen = selected["selected"]
        scores, checked = audit_fit(folder / f"outer-{oi}", data, {outer}, (chosen["epoch"],), seed,
            registration, images, teachers, config, evidence, old, tensor)
        checks.append(checked)
        rows.extend(r.example.row(expanded.base.decode(r.example, scores[chosen["epoch"]][r.example.id], chosen["decoder"]))
                    for r in data["exact"] if r.example.group == outer)
    require({p.name for p in folder.iterdir() if p.is_dir()} == expected_folders, "Unregistered physical fit exists")
    complete = all(s["feasible"] for s in selections)
    evaluated = evaluate_predictions(rows) if complete else None
    result_ref = evidence.capture(study / f"result-{seed}.json")
    stored = read(result_ref["path"])
    old.compare(stored, {"contractSha256": registration["sha256"], "seed": seed, "selections": selections,
        "completeEvaluation": complete, "evaluation": evaluated, "predictions": old.serialized_rows(rows),
        "partialScopeNotRankable": not complete, "protectedTestOpened": False}, "saved student result")
    metrics = old.independent_metrics(rows, evaluated, interval) if complete else None
    return {"seed": seed, "result": result_ref, "fits": checks, "logicalViews": views, "selections": selections,
            "selectionGrids": grids, "completeEvaluation": complete, "partialScopeNotRankable": not complete,
            "independent": metrics, "evaluation": evaluated}


def audit(study):
    import cv2
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    evidence = Evidence()
    old, tensor, interval = legacy_helpers(evidence)
    registration, engineering = verify_registration(study, evidence, old)
    c = registration["contract"]
    report_ref = evidence.capture(study / "report.json")
    report = read(report_ref["path"])
    require(report["contractSha256"] == registration["sha256"] and report["protectedTestOpened"] is False
            and [r["seed"] for r in report["results"]] == c["seeds"], "Final report identity differs")
    require({p.name for p in (study / "fits").iterdir() if p.is_dir()} == {str(s) for s in c["seeds"]}, "Unexpected seed fits")
    data, config = load_inputs(c, evidence)
    images, teachers, image_checks = audit_images_teacher(c, data, evidence)
    results = []
    for seed in c["seeds"]:
        checked = audit_seed(study, seed, registration, data, images, teachers, config, evidence, old, tensor, interval)
        old.compare(next(r for r in report["results"] if r["seed"] == seed), read(checked["result"]["path"]), "report/seed binding")
        results.append(checked)
    return {"kind": "independent-mobile-distillation-audit-v1", "passed": True, "study": str(study),
        "registration": identity(study / "preregistration.json"), "report": report_ref,
        "contractSha256": registration["sha256"], "auditor": identity(Path(__file__)),
        "legacyAuditorSha256": LEGACY_AUDITOR_SHA, "engineering": c["engineering"], "imagesAndTeacher": image_checks,
        "counts": {"seeds": len(results), "physicalStudentFits": sum(len(r["fits"]) for r in results),
                   "physicalTemporalFits": sum(len(r["fits"]) for r in results), "logicalInnerViews": 12 * len(results)},
        "replayScope": "3 same-frame teacher samples per recording; 3 encoder feature frames per fold/record; first/middle/last real-context head chunks per held recording/checkpoint. Not full training or full-tick replay.",
        "tolerance": {"teacher": {"rtol": TEACHER_RTOL, "atol": TEACHER_ATOL},
            "encoder": {"rtol": FEATURE_RTOL, "atol": FEATURE_ATOL, "storedFloat16Rounding": "plus half local FP16 ULP"},
            "head": {"rtol": old.REPLAY_RTOL, "atol": old.REPLAY_ATOL}},
        "inputScope": "All prepared image/teacher/feature bytes hashed; sampled raw frames match prior decoded hashes; entire raw videos not rehashed.",
        "evidence": evidence.finish(), "results": results, "protectedTestOpened": False,
        "gpuUsed": False, "fittingPerformed": False, "finishedAtUtc": datetime.now(UTC).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--study", type=Path)
    scope.add_argument("--image-index", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(str(args.output.resolve()).startswith(private_value('private-reference-0060')), "Audit writes must use direct NAS")
    require(not args.output.exists(), "Audit output already exists")
    for name in ("TMPDIR", "TMP", "TEMP", "TORCH_HOME", "HF_HOME", "XDG_CACHE_HOME", "CUDA_CACHE_PATH"):
        path = (args.study.parent if args.study else args.image_index.parent.parent) / "audit-runtime" / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    sys.dont_write_bytecode = True
    result = audit(args.study.resolve()) if args.study else audit_prepared_images(args.image_index.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    counts = result["counts"] if "counts" in result else {"recordings": result["recordings"]}
    print(json.dumps({"passed": True, "output": str(args.output), "counts": counts}))


if __name__ == "__main__":
    main()
