#!/usr/bin/env python3
"""Fixed first-inner student/head CPU ONNX qualification; no training or GPU."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
ROOT = Path(private_value('private-reference-0100'))
RUNTIME_HELPER_SHA = "f408d94c3cb21295aacc4257ea03ccf3e28a9d19c707b01d998ab7387d0fcf34"
BROWSER_HELPER_SHA = "918978de78d58355d59f1ccd43c7256fd659c89a071feb5ddbefbd9d37606c6a"
IDS = (private_value('grass-source-03'), private_value('grass-source-01'), private_value('indoor-source-01'), private_value('indoor-source-08'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(path):
    return {"path": str(path), "sha256": digest(path)}


def verify(reference):
    require(digest(reference["path"]) == reference["sha256"], "Artifact changed: " + reference["path"])
    return Path(reference["path"])


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def freeze(args):
    helper = REPO / "scripts/qualify-neural-recognition-runtime.py"
    browser_helper = REPO / "scripts/qualify-neural-recognition-browser.mjs"
    require(digest(helper) == RUNTIME_HELPER_SHA and digest(browser_helper) == BROWSER_HELPER_SHA, "Frozen runtime helper changed")
    payload = {"kind": "distilled-mobile-runtime-protocol-v1", "combinedProtocol": identity(ROOT / "protocol.md"),
        "createdAtUtc": datetime.now(UTC).isoformat(), "studentSeed": 3407, "fit": "inner-0-1", "temporalEpoch": 60,
        "checkpointSelection": "Fixed before study outcomes; first registered inner source pair, fixed epoch60",
        "encoderFrames": {"recordings": list(IDS), "rule": "8 linspace indexes 0..teachingFrames-1 per record, same as engineering qualification", "total": 32},
        "encoderInput": "prepared float32 normalized224 RGB and original regional pool weights; actual trained encoder; projection omitted",
        "temporalInput": "actual trained mobile TCN with scaler; 7 true-length fixtures (1/7/190/252/411 ticks), 4 chunk-equivalence checks",
        "cpu": {"threads": 1, "provider": "CPUExecutionProvider", "repeats": 10},
        "browser": {"runtime": "onnxruntime-web1.22.0", "provider": "wasm", "threads": 1, "repeats": 10},
        "source": identity(Path(__file__)), "browserSource": identity(REPO / "scripts/qualify-distilled-mobile-browser.mjs"),
        "helpers": [identity(helper), identity(browser_helper)],
        "gpuUsed": False, "physicalPhoneMeasured": False,
        "limits": ["Separate prepared-input graphs; not pixel-to-rally application qualification.",
            "Float16 token cache roundtrip, 2Hz/4Hz alignment, quality extraction, AV computation and video decode are not in these graphs.",
            "Desktop CPU/browser timing does not establish phone memory, thermal behavior or latency."]}
    path = ROOT / "runtime-distilled-mobile-protocol-v1.json"
    write_new(path, payload)
    print(json.dumps({"protocol": identity(path)}))


def load_helper(protocol):
    for ref in protocol["helpers"] + [protocol["source"], protocol["browserSource"], protocol["combinedProtocol"]]:
        verify(ref)
    path = REPO / "scripts/qualify-neural-recognition-runtime.py"
    spec = importlib.util.spec_from_file_location("frozen_recognition_runtime", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def qualify(args):
    import numpy as np
    import torch
    from analysis import mobile_visual_features as mobile
    from analysis.recognition_temporal_model import RecognitionConfig, model_for, model_metadata
    from analysis.neural_recognition_fit import standardized
    from types import SimpleNamespace
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol_path = ROOT / "runtime-distilled-mobile-protocol-v1.json"
    protocol = read(protocol_path)
    helper = load_helper(protocol)
    registration = read(args.study / "preregistration.json")
    c = registration["contract"]
    require(helper.canonical_hash(c) == registration["sha256"], "Student study registration changed")
    require(c["protocol"] == protocol["combinedProtocol"], "Study/runtime protocol differs")
    for ref in c["code"].values():
        verify(ref)
    folder = args.study / "fits/3407/inner-0-1"
    student = read(folder / "student/completed.json")
    completed = read(folder / "temporal/completed.json")
    excluded = c["groups"][:2]
    require(student["contractSha256"] == completed["contractSha256"] == registration["sha256"]
        and student["seed"] == completed["seed"] == 3407 and student["excludedGroups"] == excluded
        and completed["validationGroups"] == excluded and len(student["history"]) == 8
        and completed["epochs"] == [5, 15, 30, 60]
        and completed["scalerTrainIds"] == completed["trainIds"], "Fixed trained fit identity differs")
    require(not set(excluded) & set(student["trainGroups"])
        and not set(excluded) & set(completed["trainGroups"])
        and not any(set(excluded) & set(groups) for groups in completed["auxiliaryGroups"].values()), "Trained checkpoint source leakage")
    verify(student["weights"])
    backbone = mobile.load_mobile_backbone(verify(c["mobileCheckpoint"]), checkpoint_sha256=c["mobileCheckpoint"]["sha256"], device="cpu")
    with np.load(student["weights"]["path"], allow_pickle=False) as saved:
        state = {key.removeprefix("encoder::"): torch.from_numpy(saved[key].copy()) for key in saved.files if key.startswith("encoder::")}
    backbone.model.features.load_state_dict(state, strict=True)
    encoder = helper.ImageEncoder(backbone.model.features).eval()
    config = RecognitionConfig(family="mobile", head="tcn", scalar_dimension=8)
    require(model_metadata(config) == c["model"], "Unexpected temporal architecture")
    temporal_path = folder / "temporal/weights-60.npz"
    require(digest(temporal_path) == completed["artifacts"][temporal_path.name], "Temporal checkpoint changed")
    with np.load(temporal_path, allow_pickle=False) as saved:
        mean, scale = saved["mean"], saved["scale"]
        state = {key.removeprefix("model::"): torch.from_numpy(saved[key].copy()) for key in saved.files if key.startswith("model::")}
    require(mean.shape == scale.shape == (112,) and np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale > 0), "Invalid trained scaler")
    temporal = model_for(config).cpu().eval()
    temporal.load_state_dict(state, strict=True)
    wrapper = helper.ScaledTemporal(temporal, config, mean, scale).eval()
    provenance = {"study": str(args.study), "registration": identity(args.study / "preregistration.json"),
        "contractSha256": registration["sha256"], "seed": 3407, "fit": "inner-0-1", "excludedGroups": excluded,
        "studentCompleted": identity(folder / "student/completed.json"), "studentWeights": student["weights"],
        "temporalCompleted": identity(folder / "temporal/completed.json"), "temporalWeights": identity(temporal_path),
        "temporalEpoch": 60, "checkpointSelection": protocol["checkpointSelection"], "scalerIncludedInGraph": True}
    input_index = read(verify(c["images"]))
    images = {r["id"]: r for r in input_index["records"]}
    engineering = read(verify(c["engineering"]))
    engine_records = {r["id"]: r for r in engineering["records"]}
    encoder_cases, input_refs = [], []
    for number, identifier in enumerate(IDS):
        record = images[identifier]
        selected = np.linspace(0, record["teachingFrames"] - 1, 8, dtype=np.int64)
        with np.load(verify(record["arrays"]["timing"]), allow_pickle=False) as timing:
            indexes = timing["teaching_indexes"][selected]
            boxes = timing["boxes"][indexes]
        require(indexes.tolist() == engine_records[identifier]["frameIndexes"], "Runtime frames differ from fixed engineering samples")
        pixels = np.load(verify(record["arrays"]["images224"]), mmap_mode="r")
        normalized = (pixels[indexes].astype(np.float32) / 255. - mobile.RGB_MEAN[None, :, None, None]) / mobile.RGB_STD[None, :, None, None]
        pool = mobile.regional_pool_weights(boxes, 7, 7)
        for i, index in enumerate(indexes):
            image, weights = normalized[i:i + 1], pool[i:i + 1]
            with torch.inference_mode():
                spatial = backbone.model.features(torch.from_numpy(image))
                expected = torch.einsum("bchw,brhw->brc", spatial, torch.from_numpy(weights)).numpy()
            encoder_cases.append({"name": f"source{number}_frame{int(index)}", "inputs": {"image": image, "pool_weights": weights},
                "independent_expected": expected, "description": f"{identifier}, prepared image index {int(index)}; original ROI pools, actual trained encoder"})
        input_refs.append({"id": identifier, "arrays": record["arrays"], "frameIndexes": indexes.tolist(),
                           "engineeringInputReceipt": engine_records[identifier]["inputReceipt"]})
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"schemaVersion": 1, "createdAt": datetime.now(UTC).isoformat(), "status": "running",
        "scope": "Actual distilled encoder and fixed trained dynamic mobile head, desktop CPU engineering parity",
        "protocol": identity(protocol_path), "provenance": provenance, "imageFixtures": input_refs,
        "gpuUsed": False, "trainingPerformed": False, "physicalPhoneMeasured": False, "productionAppStarted": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "onnx": helper.onnx.__version__,
                        "onnxruntime": helper.ort.__version__, "threads": 1, "provider": "CPUExecutionProvider"},
        "limitations": protocol["limits"]}
    try:
        models = [helper.qualify_graph(args.output, "mobile_image_encoder", encoder, encoder_cases, "tokens", repeats=10,
            extra={"role": "image-encoder", "state": "trained distilled encoder with regional pooling",
                   "checkpoint": {**provenance, "scalerIncludedInGraph": False}, "classifierAndDistillationProjectionOmitted": True})]
        cases, halo, rules = helper.dynamic_mobile_cases(wrapper, config, np.random.default_rng(helper.SEED))
        scaler_checks = []
        for case in cases:
            normalized = standardized(SimpleNamespace(times=np.arange(case["inputs"]["features"].shape[1]),
                values=case["inputs"]["features"][0]), mean, scale, config)
            expected = helper.torch_output(temporal, {"features": normalized[None]})
            scaler_checks.append(helper.compare(helper.torch_output(wrapper, case["inputs"]), expected))
            case["independent_expected"] = expected
        models.append(helper.qualify_graph(args.output, "mobile_tcn_dynamic_trained", wrapper, cases, "logits", repeats=10,
            dynamic_axes={"features": {1: "time"}, "logits": {1: "time"}},
            extra={"role": "temporal-head", "state": "trained completed checkpoint with fitted scaler", "checkpoint": provenance,
                "config": asdict(config), "scalerChecks": scaler_checks, "wholeSequenceHaloChecks": halo,
                "chunkEquivalence": rules, "centralTicks": 128, "realHaloTicks": 62}))
        report.update(status="pass", models=models)
    except Exception as error:
        report.update(status="fail", failure={"type": type(error).__name__, "message": str(error)})
    report["wallSeconds"] = time.perf_counter() - started
    path = args.output / "cpu-report.json"
    write_new(path, report)
    if report["status"] == "pass":
        write_new(args.output / "fixture-manifest.json", {"schemaVersion": 1, "kind": "distilled-mobile-portability-fixtures-v1",
            "cpuReport": helper.artifact(path), "models": models, "gpuUsed": False, "temporalModelsTrained": True,
            "encoderTrained": True, "physicalPhoneMeasured": False, "runtimeProtocol": identity(protocol_path)})
    print(json.dumps({"status": report["status"], "failure": report.get("failure"), "report": str(path),
        "models": [{"name": m["name"], "bytes": m["graph"]["sizeBytes"], "medianMs": m["preparedInputCpuLatency"]["medianMilliseconds"]}
                   for m in report.get("models", [])]}), flush=True)
    return 0 if report["status"] == "pass" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--study", type=Path, default=ROOT / "distilled-mobile-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime-distilled-mobile-cpu-v1")
    args = parser.parse_args()
    require(str(args.output.resolve()).startswith(private_value('private-reference-0060')), "Runtime artifacts must use direct NAS")
    for name in ("TMPDIR", "TMP", "TEMP", "TORCH_HOME", "HF_HOME", "XDG_CACHE_HOME", "CUDA_CACHE_PATH"):
        path = ROOT / "runtime-distilled-mobile-temp" / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    sys.dont_write_bytecode = True
    if args.action == "freeze":
        freeze(args)
        return 0
    return qualify(args)


if __name__ == "__main__":
    raise SystemExit(main())
