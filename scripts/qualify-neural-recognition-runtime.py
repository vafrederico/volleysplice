#!/usr/bin/env python3
"""Desktop CPU export/parity fixtures for recognition graphs; no training/GPU.

Default temporal heads are explicitly seeded, untrained engineering models.
With --trained-study, qualify the fixed seed3407/outer-0 completed checkpoint
and its scaler instead. No checkpoint is chosen by its measured quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import onnx
import onnxruntime as ort
import torch

from analysis.mobile_visual_features import load_mobile_backbone, preprocess_frame, regional_pool_weights
from analysis.recognition_temporal_model import RecognitionConfig, model_for
from analysis.neural_recognition_fit import scalar_indexes, standardized
from analysis.neural_event_balanced_development import canonical_hash

TICKS = 252
HALO = 62
CORE = 128
SEED = 20260922


def artifact(path):
    content = Path(path).read_bytes()
    return {"file": Path(path).name, "sha256": hashlib.sha256(content).hexdigest(), "sizeBytes": len(content)}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def compare(actual, expected, *, atol=1e-5, rtol=1e-4):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite or wrong-shaped parity output")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return {"passed": True, "maximumAbsoluteError": float(np.max(np.abs(actual - expected))),
            "meanAbsoluteError": float(np.mean(np.abs(actual - expected))), "atol": atol, "rtol": rtol}


def tensor_file(output, stem, name, values):
    array = np.ascontiguousarray(values)
    dtype = "bool" if array.dtype == np.bool_ else "float32"
    extension = "u8" if dtype == "bool" else "f32"
    path = output / f"{stem}-{name}.{extension}"
    (array.astype(np.uint8) if dtype == "bool" else array.astype("<f4")).tofile(path)
    return {**artifact(path), "name": name, "dtype": dtype, "shape": list(array.shape)}


def torch_output(model, inputs):
    with torch.inference_mode():
        return model(*(torch.from_numpy(np.ascontiguousarray(value)) for value in inputs.values())).cpu().numpy()


def benchmark(call, repeats):
    for _ in range(2):
        call()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        values.append((time.perf_counter() - started) * 1000)
    return {"warmupCalls": 2, "repeats": repeats, "medianMilliseconds": float(np.median(values)),
            "p95Milliseconds": float(np.percentile(values, 95)), "samplesMilliseconds": values}


class ImageEncoder(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, image, pool_weights):
        spatial = self.features(image).flatten(2).transpose(1, 2)
        return torch.matmul(pool_weights.flatten(2), spatial)


class ScaledTemporal(torch.nn.Module):
    """Include the fitted scalar preprocessing; frozen visual tokens stay raw."""
    def __init__(self, model, config, mean, scale):
        super().__init__()
        self.model, self.attention = model, config.head == "transformer"
        indexes = np.concatenate((np.arange(104), scalar_indexes(config)))
        means = np.zeros(config.input_dimension, np.float32)
        scales = np.ones(config.input_dimension, np.float32)
        selected = np.zeros(config.input_dimension, np.bool_)
        means[indexes], scales[indexes], selected[indexes] = mean, scale, True
        self.register_buffer("mean", torch.from_numpy(means))
        self.register_buffer("scale", torch.from_numpy(scales))
        self.register_buffer("scaled_features", torch.from_numpy(selected))

    def forward(self, features, valid_mask=None):
        normalized = torch.clamp((features - self.mean) / self.scale, -10, 10)
        values = torch.where(self.scaled_features, normalized, features)
        return self.model(values, valid_mask) if self.attention else self.model(values)


def trained_checkpoint(study):
    """Bind one predeclared fold and completed weights; never inspect F1/results."""
    registration_path = study / "preregistration.json"
    registration = json.loads(registration_path.read_text())
    contract = registration["contract"]
    if canonical_hash(contract) != registration["sha256"]:
        raise ValueError("trained study contract hash differs")
    for name, digest in contract["code"].items():
        if artifact(REPO / "analysis" / name)["sha256"] != digest:
            raise ValueError(f"trained model source changed: {name}")
    folder = study / "fits" / "3407" / "outer-0"
    completed_path = folder / "completed.json"
    completed = json.loads(completed_path.read_text())
    if (completed["contractSha256"] != registration["sha256"] or completed["seed"] != 3407
            or len(completed["epochs"]) != 1 or completed["validationGroups"] != [contract["groups"][0]]
            or completed["scalerTrainIds"] != completed["trainIds"]):
        raise ValueError("expected completed seed3407/outer-0 fit and training-only scaler")
    config = RecognitionConfig(**contract["config"])
    epoch = completed["epochs"][0]
    weights_path = folder / f"weights-{epoch}.npz"
    weights_identity = artifact(weights_path)
    if completed["artifacts"][weights_path.name] != weights_identity["sha256"]:
        raise ValueError("trained weights hash differs")
    with np.load(weights_path, allow_pickle=False) as saved:
        mean, scale = saved["mean"], saved["scale"]
        state = {key.removeprefix("model::"): torch.from_numpy(saved[key].copy())
                 for key in saved.files if key.startswith("model::")}
    expected_scalars = 104 + config.scalar_dimension
    if (mean.shape != (expected_scalars,) or scale.shape != mean.shape or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not all(torch.isfinite(value).all() for value in state.values())):
        raise ValueError("trained weights/scalers are nonfinite or wrongly shaped")
    model = model_for(config).cpu().eval()
    model.load_state_dict(state, strict=True)
    wrapper = ScaledTemporal(model, config, mean, scale).eval()
    provenance = {"study": str(study.resolve()), "registration": artifact(registration_path),
                  "contractSha256": registration["sha256"], "completedFit": artifact(completed_path),
                  "weights": {**weights_identity, "path": str(weights_path.resolve())},
                  "seed": 3407, "outerIndex": 0, "epoch": epoch,
                  "foldSelection": "fixed first registered source group; not chosen by observed quality",
                  "scalerIncludedInGraph": True, "featureExtractionIncluded": False}
    return wrapper, model, config, mean, scale, provenance


def qualify_graph(output, name, model, cases, output_name, *, repeats, extra=None, dynamic_axes=None):
    path = output / f"{name}.onnx"
    sample_inputs = cases[0]["inputs"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.onnx.export(model, tuple(torch.from_numpy(v) for v in sample_inputs.values()), str(path),
                          input_names=list(sample_inputs), output_names=[output_name], opset_version=17,
                          dynamo=False, external_data=False, do_constant_folding=True, dynamic_axes=dynamic_axes)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph, full_check=True)
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer):
        raise ValueError("graph must have no external weights")
    if any(node.domain not in ("", "ai.onnx") for node in graph.graph.node):
        raise ValueError("only standard decomposed ONNX operators qualify")
    if dynamic_axes:
        shapes = {item.name: item.type.tensor_type.shape.dim for item in (*graph.graph.input, *graph.graph.output)}
        for tensor_name, axes in dynamic_axes.items():
            if any(shapes[tensor_name][axis].dim_param != label for axis, label in axes.items()):
                raise ValueError("requested dynamic axis was frozen during export")
    if name == "mobile_image_encoder" and any("classifier" in tensor.name for tensor in graph.graph.initializer):
        raise ValueError("classification head leaked into feature encoder")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    started = time.perf_counter()
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    creation_ms = (time.perf_counter() - started) * 1000
    session.disable_fallback()
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("CPU provider required")
    results, actual_by_case = [], {}
    for case in cases:
        expected = torch_output(model, case["inputs"])
        actual = session.run([output_name], case["inputs"])[0]
        actual_by_case[case["name"]] = actual
        tolerance = {"atol": 3e-5, "rtol": 2e-4} if name == "mobile_image_encoder" else {"atol": 1e-5, "rtol": 1e-4}
        check = compare(actual, expected, **tolerance)
        if "independent_expected" in case:
            check["independentReference"] = compare(expected, case["independent_expected"], **tolerance)
        stem = name + "-" + case["name"]
        results.append({"name": case["name"], "inputs": [tensor_file(output, stem, key, value) for key, value in case["inputs"].items()],
                        "expected": tensor_file(output, stem, output_name, expected), "cpuParity": check,
                        "description": case["description"], "tolerance": tolerance})
    equivalence = []
    for rule in (extra or {}).get("chunkEquivalence", []):
        chunk, reference = actual_by_case[rule["chunkCase"]], actual_by_case[rule["referenceCase"]]
        a, b, length = rule["chunkStart"], rule["referenceStart"], rule["length"]
        equivalence.append({**rule, **compare(chunk[:, a:a + length], reference[:, b:b + length])})
    return {"name": name, "graph": {**artifact(path), "gzipBytes": len(gzip.compress(path.read_bytes(), mtime=0)),
                                    "inputShapes": {key: list(value.shape) for key, value in sample_inputs.items()},
                                    "dynamicAxes": dynamic_axes or {},
                                    "outputName": output_name, "outputShape": list(torch_output(model, sample_inputs).shape),
                                    "operatorCounts": dict(sorted(Counter(node.op_type for node in graph.graph.node).items()))},
            "parameters": sum(parameter.numel() for parameter in model.parameters()), "cases": results,
            "sessionCreationMilliseconds": creation_ms,
            "preparedInputCpuLatency": benchmark(lambda: session.run([output_name], sample_inputs), repeats),
            "cpuChunkEquivalence": equivalence,
            "exportWarnings": sorted({str(item.message) for item in caught}), **(extra or {})}


def temporal_cases(model, config, rng):
    dimension = config.input_dimension
    full = rng.normal(0, .7, (1, 411, dimension)).astype(np.float32)
    values = np.ascontiguousarray(full[:, :TICKS])
    attention = config.head == "transformer"
    base = {"features": values}
    if attention:
        base["valid_mask"] = np.ones((1, TICKS), np.bool_)
    cases = [{"name": "full_chunk", "inputs": base, "description": "252 real prepared-feature ticks; all four raw logits"}]
    with torch.inference_mode():
        whole = model(torch.from_numpy(full)).numpy()
    halo_checks = []
    for left in (0, 128):
        inputs = {"features": np.ascontiguousarray(full[:, left:left + TICKS])}
        if attention:
            inputs["valid_mask"] = np.ones((1, TICKS), np.bool_)
        output = torch_output(model, inputs)
        halo_checks.append(compare(output[:, HALO:HALO + CORE], whole[:, left + HALO:left + HALO + CORE]))
        if left:
            cases.append({"name": "shifted_chunk", "inputs": inputs,
                          "description": "Second interior chunk; 62 real halo ticks match whole-sequence central128 logits"})
    if attention:
        for name, spans in (("masked_segments", ((11, 73), (75, 231))), ("short_segment", ((13, 20),)), ("all_masked", ())):
            mask = np.zeros((1, TICKS), np.bool_)
            expected = np.zeros((1, TICKS, 4), np.float32)
            for left, right in spans:
                mask[:, left:right] = True
                with torch.inference_mode():
                    expected[:, left:right] = model(torch.from_numpy(values[:, left:right])).numpy()
            masked = values.copy()
            masked[~mask] = np.nan
            cases.append({"name": name, "inputs": {"features": masked, "valid_mask": mask},
                          "independent_expected": expected,
                          "description": "Masked NaN padding; expected output comes from independent true-length segments; invalid logits zero"})
    return cases, halo_checks


def dynamic_mobile_cases(model, config, rng):
    """True-length calls cover both recording edges and independently reset gaps."""
    if (config.family, config.head, config.scalar_dimension) != ("mobile", "tcn", 8):
        raise ValueError("dynamic mobile qualification requires the registered mobile TCN/Q8 contract")
    full = rng.normal(0, .7, (1, 411, config.input_dimension)).astype(np.float32)
    cases = []
    for name, left, right, description in (
        ("full_chunk", 0, 252, "252 real ticks; central128 has 62 real halo ticks on each side"),
        ("shifted_chunk", 128, 380, "Second interior chunk; real context only"),
        ("recording_start", 0, 190, "True recording start; first128 core ticks and62 right halo ticks"),
        ("recording_end", 221, 411, "True recording end;62 left halo ticks and final128 core ticks"),
        ("length1_segment", 207, 208, "Independent one-tick segment; no synthetic padding or state from previous calls"),
        ("length7_segment", 159, 166, "Independent seven-tick segment; reset at both true edges"),
        ("whole_sequence", 0, 411, "Whole-sequence reference for CPU/browser chunk-boundary checks")):
        cases.append({"name": name, "inputs": {"features": np.ascontiguousarray(full[:, left:right])},
                      "description": description})
    rules = [
        {"chunkCase": "full_chunk", "referenceCase": "whole_sequence", "chunkStart": 62, "referenceStart": 62, "length": 128},
        {"chunkCase": "shifted_chunk", "referenceCase": "whole_sequence", "chunkStart": 62, "referenceStart": 190, "length": 128},
        {"chunkCase": "recording_start", "referenceCase": "whole_sequence", "chunkStart": 0, "referenceStart": 0, "length": 128},
        {"chunkCase": "recording_end", "referenceCase": "whole_sequence", "chunkStart": 62, "referenceStart": 283, "length": 128}]
    outputs = {case["name"]: torch_output(model, case["inputs"]) for case in cases}
    checks = []
    for rule in rules:
        a, b, n = rule["chunkStart"], rule["referenceStart"], rule["length"]
        checks.append({**rule, **compare(outputs[rule["chunkCase"]][:, a:a + n],
                                        outputs[rule["referenceCase"]][:, b:b + n])})
    return cases, checks, rules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--trained-study", type=Path, action="append", default=[],
                        help="qualify only these completed studies' fixed seed3407/outer-0 weights and scalers")
    parser.add_argument("--dynamic-mobile-tcn", action="store_true",
                        help="qualify only mobile TCN with true dynamic time, including short/recording-edge chunks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if not args.trained_study and not args.checkpoint and not args.dynamic_mobile_tcn:
        parser.error("default architecture qualification requires --checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()
    report = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(), "status": "running",
              "scope": "Desktop single-thread CPU engineering qualification; not phone performance or head accuracy",
              "seed": SEED, "gpuUsed": False, "trainingPerformed": False, "datasetVideosDecoded": False,
              "physicalPhoneMeasured": False, "productionAppStarted": False,
              "script": artifact(Path(__file__)),
              "sourceCode": {name: artifact(REPO / "analysis" / name)["sha256"] for name in (
                  "mobile_visual_features.py", "recognition_temporal_model.py", "local_attention_temporal_model.py",
                  "expanded_temporal_model.py", "transfer_temporal_model.py", "neural_recognition_fit.py")},
              "environment": {"python": platform.python_version(),
                  "platform": platform.platform(), "torch": torch.__version__, "onnx": onnx.__version__,
                  "onnxruntime": ort.__version__, "threads": 1, "provider": "CPUExecutionProvider"},
              "limitations": ["Only explicitly identified trained checkpoint/scaler bytes qualify; seeded graphs prove architecture compatibility only.",
                  "Image encoder is pretrained FP32; mobile INT8 and native delegates are unqualified.",
                  ("Dynamic mobile graph requires true-length segment calls and 62 real halo ticks; never pad across ignored gaps."
                   if args.dynamic_mobile_tcn else "TCN252 static graphs qualify interior chunks only; short/true-edge graphs require separate deployment handling."),
                  "Latency excludes decode, image preprocessing, pool-weight construction and UI; identified trained graphs include fitted scalar scaling.",
                  "Other experiments are running on this desktop; timings are diagnostic, not isolated performance estimates."]}
    try:
        models = []
        dynamic_axes = {"features": {1: "time"}, "logits": {1: "time"}} if args.dynamic_mobile_tcn else None
        if args.trained_study:
            seen = set()
            for study in args.trained_study:
                wrapper, model, config, mean, scale, provenance = trained_checkpoint(study)
                name = config.family + "_" + config.head + "_trained_outer0" + ("_dynamic" if dynamic_axes else "")
                if name in seen:
                    raise ValueError("duplicate trained architecture requested")
                seen.add(name)
                rules = []
                if dynamic_axes:
                    cases, halo_checks, rules = dynamic_mobile_cases(wrapper, config, rng)
                else:
                    cases, halo_checks = temporal_cases(wrapper, config, rng)
                scaler_checks = []
                for case in cases:
                    normalized = standardized(SimpleNamespace(values=case["inputs"]["features"][0],
                        times=np.arange(case["inputs"]["features"].shape[1])), mean, scale, config)
                    reference_inputs = {**case["inputs"], "features": normalized[None]}
                    expected = torch_output(model, reference_inputs)
                    scaler_checks.append(compare(torch_output(wrapper, case["inputs"]), expected))
                    if "independent_expected" not in case:
                        case["independent_expected"] = expected
                models.append(qualify_graph(args.output, name, wrapper, cases, "logits", repeats=args.repeats, dynamic_axes=dynamic_axes,
                    extra={"state": "trained completed checkpoint with fitted scaler", "config": asdict(config),
                           "checkpoint": provenance, "scalerChecks": scaler_checks,
                           "wholeSequenceHaloChecks": halo_checks, "chunkEquivalence": rules,
                           "centralTicks": CORE, "realHaloTicks": HALO}))
        elif dynamic_axes:
            config = RecognitionConfig(family="mobile", head="tcn", scalar_dimension=8)
            torch.manual_seed(SEED)
            model = model_for(config).cpu().eval()
            cases, halo_checks, rules = dynamic_mobile_cases(model, config, rng)
            models.append(qualify_graph(args.output, "mobile_tcn_dynamic", model, cases, "logits",
                repeats=args.repeats, dynamic_axes=dynamic_axes,
                extra={"state": "seeded untrained engineering head", "config": asdict(config),
                       "wholeSequenceHaloChecks": halo_checks, "chunkEquivalence": rules,
                       "centralTicks": CORE, "realHaloTicks": HALO}))
        else:
            backbone = load_mobile_backbone(args.checkpoint, checkpoint_sha256=args.checkpoint_sha256, device="cpu")
            encoder = ImageEncoder(backbone.model.features).eval()
            encoder_cases = []
            for name, height, width in (("landscape", 180, 320), ("portrait", 320, 180), ("square", 224, 224)):
                frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                values, box, _ = preprocess_frame(frame)
                tokens, _ = backbone.embed_frames([frame])
                encoder_cases.append({"name": name, "inputs": {"image": values[None], "pool_weights": regional_pool_weights(box[None], 7, 7)},
                                      "independent_expected": tokens, "description": "Same preprocessing/pools as frozen extractor; MatMul output vs extractor einsum"})
            models.append(qualify_graph(args.output, "mobile_image_encoder", encoder, encoder_cases, "tokens", repeats=args.repeats,
                                       extra={"state": "ImageNet1K_V1 pretrained image encoder; classifier omitted", "backbone": backbone.identity()}))
            for family, head, scalars in (("av", "transformer", 0), ("dino", "transformer", 0), ("mobile", "tcn", 8), ("player", "tcn", 49)):
                config = RecognitionConfig(family=family, head=head, scalar_dimension=scalars)
                torch.manual_seed(SEED)
                model = model_for(config).cpu().eval()
                cases, halo_checks = temporal_cases(model, config, rng)
                models.append(qualify_graph(args.output, family + "_" + head, model, cases, "logits", repeats=args.repeats,
                                           extra={"state": "seeded untrained engineering head", "config": asdict(config),
                                                  "wholeSequenceHaloChecks": halo_checks, "centralTicks": CORE, "realHaloTicks": HALO}))
        report.update(status="pass", models=models)
    except Exception as error:
        report.update(status="fail", failure={"type": type(error).__name__, "message": str(error)})
    report["wallSeconds"] = time.perf_counter() - started
    report_path = args.output / "cpu-report.json"
    write_json(report_path, report)
    if report["status"] == "pass":
        write_json(args.output / "fixture-manifest.json", {"schemaVersion": 1, "kind": "recognition-portability-fixtures-v1",
                   "cpuReport": artifact(report_path), "models": report["models"], "gpuUsed": False,
                   "temporalModelsTrained": bool(args.trained_study), "physicalPhoneMeasured": False})
    print(json.dumps({"status": report["status"], "report": str(report_path), "failure": report.get("failure"),
                      "models": [{"name": row["name"], "bytes": row["graph"]["sizeBytes"],
                                  "medianMs": row["preparedInputCpuLatency"]["medianMilliseconds"]} for row in report.get("models", [])]}), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
