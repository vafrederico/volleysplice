#!/usr/bin/env python3
"""Export one predetermined trained mobile research checkpoint and qualify CPU parity.

This never trains, ranks models, extracts features, opens protected footage, or
modifies production assets. Inputs are the frozen development manifest and one
inner-selected outer-refit checkpoint. Dynamic time preserves real segment edges;
the graph also supports 252-tick chunks with real 62-tick halos and 128-tick cores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.compact_temporal_model import HEAD_NAMES
from analysis.config import FeatureConfig, NOISE_NORMALIZED_AUDIO_FEATURE_SET
from analysis.features import ABSOLUTE_FEATURE_NAMES, feature_names
from analysis.neural_development import (
    Example, decode, file_sha256, load_examples, model_for, predict, segments,
)


INPUT_FEATURES = 104
CORE_TICKS = 128
HALO_TICKS = 62
EXPORT_TICKS = CORE_TICKS + 2 * HALO_TICKS
ATOL = 1e-5
RTOL = 1e-4
MOBILE_KINDS = ("linear", "mlp", "tcn")


def identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": file_sha256(path), "sizeBytes": path.stat().st_size}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_checkpoint(
    directory: Path, epoch: int, kind: str, contract_hash: str,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray, dict[str, Any]]:
    """Verify all recorded artifacts before opening the non-pickle tensor archive."""
    if kind not in MOBILE_KINDS:
        raise ValueError("Only trained mobile linear/mlp/tcn checkpoints are exportable here")
    completed = read_json(directory / "completed.json")
    if completed["kind"] != kind or completed["contractSha256"] != contract_hash:
        raise ValueError("Checkpoint kind or study contract changed")
    if epoch not in completed["epochs"]:
        raise ValueError("Requested epoch was not completed")
    artifacts = completed["artifacts"]
    weights_name = f"weights-{epoch}.npz"
    if weights_name not in artifacts or f"predictions-{epoch}.npz" not in artifacts:
        raise ValueError("Completed checkpoint lacks weights/prediction artifact identities")
    for name, digest in artifacts.items():
        if Path(name).name != name or "\\" in name or name in {".", ".."}:
            raise ValueError("Checkpoint artifact must be a local filename")
        if file_sha256(directory / name) != digest:
            raise ValueError(f"Checkpoint artifact hash mismatch: {name}")
    model = model_for(kind).cpu().eval()
    expected_state = model.state_dict()
    with np.load(directory / weights_name, allow_pickle=False) as archive:
        expected_keys = {"mean", "scale", *(f"model::{name}" for name in expected_state)}
        if set(archive.files) != expected_keys:
            raise ValueError("Checkpoint tensor names do not match the declared model")
        mean, scale = archive["mean"].copy(), archive["scale"].copy()
        for name, values in (("mean", mean), ("scale", scale)):
            if values.shape != (INPUT_FEATURES,) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError(f"Invalid checkpoint {name}")
        if not np.all(scale > 0):
            raise ValueError("Checkpoint scales must be positive")
        state = {}
        for name, expected in expected_state.items():
            values = archive[f"model::{name}"]
            if tuple(values.shape) != tuple(expected.shape) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError(f"Invalid checkpoint tensor: {name}")
            state[name] = torch.from_numpy(values.copy())
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != completed["parameters"]:
        raise ValueError("Checkpoint parameter count changed")
    return model, mean, scale, completed


class ScaledProbabilityModel(torch.nn.Module):
    """Input: recording-ranked AV with absolute columns restored. Output: probabilities."""
    def __init__(self, model: torch.nn.Module, mean: np.ndarray, scale: np.ndarray) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.from_numpy(mean.copy()))
        self.register_buffer("scale", torch.from_numpy(scale.copy()))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp((values - self.mean) / self.scale, -10.0, 10.0)
        return torch.sigmoid(self.model(normalized))


def export_graph(model: torch.nn.Module, destination: Path) -> dict[str, Any]:
    dummy = torch.zeros(1, EXPORT_TICKS, INPUT_FEATURES, dtype=torch.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.onnx.export(
            model, (dummy,), str(destination), export_params=True, opset_version=17,
            input_names=["features"], output_names=["probabilities"], dynamo=False,
            dynamic_axes={"features": {1: "ticks"}, "probabilities": {1: "ticks"}},
            do_constant_folding=True, external_data=False,
        )
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph, full_check=True)
    dimensions = graph.graph.input[0].type.tensor_type.shape.dim
    if dimensions[0].dim_value != 1 or dimensions[1].dim_param != "ticks" or dimensions[2].dim_value != INPUT_FEATURES:
        raise ValueError("ONNX input must be batch1, dynamic time, 104 features")
    if any(value.data_location == onnx.TensorProto.EXTERNAL for value in graph.graph.initializer):
        raise ValueError("Research export must be self-contained")
    if any(node.domain not in ("", "ai.onnx") for node in graph.graph.node):
        raise ValueError("Research export contains nonstandard ONNX operators")
    return {
        **identity(destination), "opset": 17, "inputShape": [1, "ticks", INPUT_FEATURES],
        "outputShape": [1, "ticks", 3], "onnxCheckerPassed": True,
        "operatorCounts": dict(sorted(Counter(node.op_type for node in graph.graph.node).items())),
        "warnings": sorted({str(item.message) for item in caught}),
    }


def compare(actual: np.ndarray, expected: np.ndarray, description: str) -> dict[str, Any]:
    if actual.shape != expected.shape or actual.dtype != np.float32 or not np.isfinite(actual).all():
        raise ValueError(f"Invalid probability output: {description}")
    if not np.isfinite(expected).all() or np.any((actual < 0) | (actual > 1)):
        raise ValueError(f"Nonfinite/reference or out-of-range output: {description}")
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL, err_msg=description)
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {"description": description, "pass": True, "shape": list(actual.shape),
            "maximumAbsoluteProbabilityError": float(delta.max()),
            "meanAbsoluteProbabilityError": float(delta.mean())}


def sequence_trace(
    forward: Callable[[np.ndarray], np.ndarray], example: Example, *, chunked: bool,
) -> np.ndarray:
    """Preserve true/ignored segment edges without synthetic input padding."""
    result = np.zeros((len(example.times), 3), dtype=np.float32)
    for start, finish in segments(example.valid):
        if not chunked:
            result[start:finish] = forward(example.values[None, start:finish, :INPUT_FEATURES])[0]
            continue
        for left in range(start, finish, CORE_TICKS):
            right = min(finish, left + CORE_TICKS)
            input_left, input_right = max(start, left - HALO_TICKS), min(finish, right + HALO_TICKS)
            probabilities = forward(example.values[None, input_left:input_right, :INPUT_FEATURES])[0]
            result[left:right] = probabilities[left - input_left:right - input_left]
    return result


def benchmark(call: Callable[[], Any], repeats: int) -> dict[str, Any]:
    for _ in range(3):
        call()
    measurements = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        measurements.append((time.perf_counter() - start) * 1000)
    return {"warmupCalls": 3, "repeats": repeats, "medianMilliseconds": float(np.median(measurements)),
            "p95Milliseconds": float(np.percentile(measurements, 95))}


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    prereg_path = args.study / "preregistration.json"
    prereg = read_json(prereg_path)
    contract = prereg["contract"]
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != prereg["sha256"]:
        raise ValueError("Preregistration digest changed")
    if file_sha256(args.manifest) != contract["manifestSha256"]:
        raise ValueError("Development manifest differs from training")
    root = Path(__file__).resolve().parents[1]
    for name, digest in contract["code"].items():
        if Path(name).name != name or file_sha256(root / "analysis" / name) != digest:
            raise ValueError(f"Training implementation differs from frozen checkpoint: {name}")
    groups = contract["groups"]
    if args.kind not in contract["kinds"] or args.seed not in contract["seeds"] or not 0 <= args.outer_index < len(groups):
        raise ValueError("Requested kind, seed or outer fold is outside the study contract")
    result_path = args.study / f"result-{args.kind}-{args.seed}.json"
    result = read_json(result_path)
    if result["kind"] != args.kind or result["seed"] != args.seed:
        raise ValueError("Study result identity changed")
    selections = [row for row in result["selections"] if row["heldSourceGroup"] == groups[args.outer_index]]
    if len(selections) != 1:
        raise ValueError("Expected one inner-selected checkpoint for the requested outer group")
    selection = selections[0]
    epoch = selection["epoch"]
    checkpoint = args.study / "fits" / args.kind / str(args.seed) / f"outer-{args.outer_index}" / "refit"
    model, mean, scale, completed = load_checkpoint(checkpoint, epoch, args.kind, prereg["sha256"])
    if completed["seed"] != args.seed or completed["epochs"] != [epoch]:
        raise ValueError("Expected the inner-selected refit checkpoint, not a later checkpoint choice")
    examples = load_examples(args.manifest, with_dino=False)
    held = [row for row in examples if row.group == groups[args.outer_index]]
    fitting = [row for row in examples if row.group != groups[args.outer_index]]
    if completed["validationIds"] != [row.id for row in held] or completed["trainIds"] != [row.id for row in fitting]:
        raise ValueError("Manifest fold identities differ from the checkpoint")
    wrapper = ScaledProbabilityModel(model, mean, scale).cpu().eval()
    graph_path = args.output / f"{args.kind}-seed{args.seed}-outer{args.outer_index}-epoch{epoch}.onnx"
    graph = export_graph(wrapper, graph_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    started = time.perf_counter()
    session = ort.InferenceSession(str(graph_path), sess_options=options, providers=["CPUExecutionProvider"])
    session.disable_fallback()
    creation_ms = (time.perf_counter() - started) * 1000
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("CPU-only qualification provider changed")

    def pytorch_forward(values: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return wrapper(torch.from_numpy(np.ascontiguousarray(values))).numpy()

    def onnx_forward(values: np.ndarray) -> np.ndarray:
        return session.run(["probabilities"], {"features": np.ascontiguousarray(values)})[0]

    # Explicitly test both very short segments and the 252-tick deployment chunk.
    checks = []
    source_values = held[0].values
    for length in (1, 7, 17, 63, 127, 251, 252, 411):
        if length <= len(source_values):
            values = source_values[None, :length, :INPUT_FEATURES]
            checks.append(compare(onnx_forward(values), pytorch_forward(values), f"real feature slice length {length}"))
    record_results = []
    trace_arrays = {}
    with np.load(checkpoint / f"predictions-{epoch}.npz", allow_pickle=False) as frozen:
        for row in held:
            root_probability = predict(model, row, mean, scale, args.kind, "cpu")
            torch_full = sequence_trace(pytorch_forward, row, chunked=False)
            torch_chunked = sequence_trace(pytorch_forward, row, chunked=True)
            ort_full = sequence_trace(onnx_forward, row, chunked=False)
            ort_chunked = sequence_trace(onnx_forward, row, chunked=True)
            for label, actual, expected in (
                ("wrapper vs original CPU predictor", torch_chunked, root_probability),
                ("CPU vs saved training-device trace", torch_chunked, frozen[row.id]),
                ("PyTorch chunks vs whole segments", torch_chunked, torch_full),
                ("ONNX whole vs PyTorch whole", ort_full, torch_full),
                ("ONNX chunks vs whole segments", ort_chunked, ort_full),
                ("ONNX chunks vs PyTorch chunks", ort_chunked, torch_chunked),
            ):
                checks.append(compare(actual, expected, f"{row.id}: {label}"))
            decoded = {label: [interval.to_dict() for interval in decode(row, probabilities, selection["decoder"])]
                       for label, probabilities in (("saved", frozen[row.id]), ("pytorch", torch_chunked), ("onnx", ort_chunked))}
            if decoded["saved"] != decoded["pytorch"] or decoded["pytorch"] != decoded["onnx"]:
                raise ValueError(f"Decoded interval parity failed: {row.id}")
            trace_arrays.update({f"pytorch::{row.id}": torch_chunked, f"onnx::{row.id}": ort_chunked,
                                 f"times::{row.id}": row.times, f"valid::{row.id}": row.valid})
            record_results.append({"id": row.id, "sourceGroup": row.group, "ticks": len(row.times),
                                   "durationSeconds": row.duration, "validSegments": segments(row.valid),
                                   "decodedIntervalCount": len(decoded["onnx"]), "decodedIntervalsEqual": True,
                                   "decodedIntervals": decoded["onnx"],
                                   "onnxPreparedTraceLatency": benchmark(lambda: sequence_trace(onnx_forward, row, chunked=True), args.repeats)})
    traces_path = args.output / "parity-traces.npz"
    np.savez_compressed(traces_path, **trace_arrays)
    scaler_path = args.output / "scaler.npz"
    np.savez_compressed(scaler_path, mean=mean, scale=scale)
    schema = feature_names(FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET))
    metadata = {
        "schemaVersion": 1, "kind": args.kind, "seed": args.seed, "outerIndex": args.outer_index,
        "epoch": epoch, "config": model.config.to_dict(), "graph": graph,
        "input": {"name": "features", "dtype": "float32", "shape": [1, "ticks", INPUT_FEATURES],
                  "analysisFps": 4, "featureNames": list(schema),
                  "externalNormalization": "within-recording percentile ranks with absolute columns restored",
                  "absoluteFeatureNames": sorted(ABSOLUTE_FEATURE_NAMES),
                  "graphNormalization": "saved fit-only (x-mean)/scale, clipped to [-10,10]",
                  "videoDecodeAndFeatureExtractionIncluded": False},
        "output": {"name": "probabilities", "shape": [1, "ticks", 3], "heads": list(HEAD_NAMES),
                   "endSemantics": "end-event pulse, not sustained dead-state probability"},
        "chunking": {"coreTicks": CORE_TICKS, "realHaloTicks": HALO_TICKS,
                     "usualInteriorInputTicks": EXPORT_TICKS, "minimumInputTicks": 1,
                     "boundaryRule": "clip each input to real valid-segment boundaries; never synthesize padded input ticks",
                     "ignoredRule": "reset graph context and decoder per valid segment; ignored rows carry zero output"},
        "decoder": selection["decoder"], "scaler": identity(scaler_path),
        "researchOnly": True, "productionPromotionAllowed": False,
        "checkpoint": identity(checkpoint / f"weights-{epoch}.npz"),
        "completed": identity(checkpoint / "completed.json"), "preregistration": identity(prereg_path),
        "manifest": identity(args.manifest), "selectionResult": identity(result_path),
        "selectionPolicy": "CLI-predetermined kind/seed/fold; epoch and decoder fixed by training inner folds",
    }
    metadata_path = args.output / "model-metadata.json"
    write_json(metadata_path, metadata)
    return {
        "status": "pass", "graph": graph, "metadata": identity(metadata_path), "traces": identity(traces_path),
        "kind": args.kind, "seed": args.seed, "outerIndex": args.outer_index, "epoch": epoch,
        "numericalChecks": checks, "maximumAbsoluteProbabilityError": max(row["maximumAbsoluteProbabilityError"] for row in checks),
        "records": record_results, "decodedIntervalsEqual": True,
        "sessionCreationMilliseconds": creation_ms,
        "onnx252TickLatency": benchmark(lambda: onnx_forward(source_values[None, :EXPORT_TICKS, :INPUT_FEATURES]), args.repeats),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New immutable qualification directory")
    parser.add_argument("--kind", choices=MOBILE_KINDS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--outer-index", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1 or args.outer_index < 0:
        parser.error("threads/repeats must be positive and outer index nonnegative")
    args.study, args.manifest, args.output = args.study.resolve(), args.manifest.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    report = {
        "schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(),
        "scope": "trained research checkpoint export and desktop CPU probability/interval parity",
        "trainingPerformed": False, "featuresExtracted": False, "protectedTestOpened": False,
        "gpuUsed": False, "phoneBrowserPerformanceMeasured": False, "productionPromotionAllowed": False,
        "acceptance": {"atol": ATOL, "rtol": RTOL, "decodedIntervalsExactlyEqual": True},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                        "provider": "CPUExecutionProvider", "intraOpThreads": args.threads, "interOpThreads": 1},
        "script": identity(Path(__file__).resolve()),
        "limitations": ["Desktop CPU diagnostics are not phone or browser measurements.",
                        "Latency uses prepared features and excludes media decode, extraction, I/O and UI.",
                        "Full-recording percentile features remain an external preprocessing requirement.",
                        "FP32 only; quantization, browser and physical-device portability remain unqualified.",
                        "This single outer-refit model is an engineering artifact, not a final all-development refit."],
    }
    try:
        report.update(qualify(args))
    except Exception as error:
        report.update(status="fail", failure={"type": type(error).__name__, "message": str(error)})
    report["wallSeconds"] = time.perf_counter() - started
    destination = args.output / "trained-runtime-qualification.json"
    write_json(destination, report)
    print(json.dumps({"status": report["status"], "report": identity(destination),
                      "maximumAbsoluteProbabilityError": report.get("maximumAbsoluteProbabilityError"),
                      "failure": report.get("failure")}), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
