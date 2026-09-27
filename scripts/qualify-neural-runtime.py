#!/usr/bin/env python3
"""Qualify synthetic compact-model ONNX graphs on desktop CPU only.

No video, labels, trained model, phone, browser, or production asset is accessed.
The immutable output directory contains seeded synthetic-weight static ONNX graphs
and numerical/latency diagnostics. A CPU pass is not mobile runtime qualification.
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

from analysis.compact_temporal_model import (
    MODEL_KINDS,
    CompactTemporalConfig,
    CompactTemporalNetwork,
    trainable_parameter_count,
)


CHUNK_TICKS = 252
CENTRAL_TICKS = 128
INPUT_FEATURES = 104
EXPORT_OPSET = 17
ABSOLUTE_TOLERANCE = 1e-5
RELATIVE_TOLERANCE = 1e-4


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sizeBytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def cpu_name() -> str:
    info = Path("/proc/cpuinfo")
    if info.is_file():
        for line in info.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def tensor_output(model: CompactTemporalNetwork, values: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        return model(torch.from_numpy(np.ascontiguousarray(values))).numpy()


def compare(actual: np.ndarray, expected: np.ndarray, description: str) -> dict[str, Any]:
    if actual.shape != expected.shape or actual.dtype != np.float32:
        raise ValueError(f"{description}: invalid shape/dtype: {actual.shape}/{actual.dtype}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"{description}: nonfinite output")
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    np.testing.assert_allclose(actual, expected, atol=ABSOLUTE_TOLERANCE,
                               rtol=RELATIVE_TOLERANCE, err_msg=description)
    return {
        "comparison": description,
        "pass": True,
        "shape": list(actual.shape),
        "maximumAbsoluteLogitError": float(difference.max()),
        "meanAbsoluteLogitError": float(difference.mean()),
        "atol": ABSOLUTE_TOLERANCE,
        "rtol": RELATIVE_TOLERANCE,
    }


def export_model(model: CompactTemporalNetwork, path: Path, ticks: int) -> dict[str, Any]:
    values = torch.zeros((1, ticks, INPUT_FEATURES), dtype=torch.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.onnx.export(
            model,
            (values,),
            str(path),
            input_names=["features"],
            output_names=["logits"],
            opset_version=EXPORT_OPSET,
            dynamo=False,
            export_params=True,
            do_constant_folding=True,
            dynamic_axes=None,
            external_data=False,
        )
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph, full_check=True)
    shape = [dimension.dim_value for dimension in graph.graph.input[0].type.tensor_type.shape.dim]
    if shape != [1, ticks, INPUT_FEATURES]:
        raise ValueError(f"Export did not preserve static input shape: {shape}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer):
        raise ValueError("Expected one self-contained ONNX artifact")
    custom_domains = sorted({node.domain for node in graph.graph.node if node.domain not in ("", "ai.onnx")})
    if custom_domains:
        raise ValueError(f"Unexpected custom ONNX operators: {custom_domains}")
    return {
        **file_identity(path),
        "inputShape": shape,
        "outputShape": [1, ticks, 3],
        "opset": EXPORT_OPSET,
        "onnxCheckerPassed": True,
        "operatorCounts": dict(sorted(Counter(node.op_type for node in graph.graph.node).items())),
        "exportWarnings": sorted({str(item.message) for item in caught}),
    }


def cpu_session(path: Path, threads: int) -> tuple[ort.InferenceSession, float]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    started = time.perf_counter()
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    session.disable_fallback()
    milliseconds = 1000 * (time.perf_counter() - started)
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError(f"Unexpected active providers: {session.get_providers()}")
    return session, milliseconds


def ort_output(session: ort.InferenceSession, values: np.ndarray) -> np.ndarray:
    return session.run(["logits"], {"features": np.ascontiguousarray(values)})[0]


def chunk_windows(total_ticks: int, halo_ticks: int) -> list[tuple[int, int, int, int]]:
    """Fixed-width real windows, with recording edges preserved by shifting windows.

    Values are (output_left, output_right, input_left, input_right). Never append
    synthetic ticks at recording boundaries: per-layer TCN padding defines them.
    """
    if total_ticks < CHUNK_TICKS or CHUNK_TICKS < CENTRAL_TICKS + 2 * halo_ticks:
        raise ValueError("Fixed chunk is too short for the requested halo/sequence")
    windows = []
    for left in range(0, total_ticks, CENTRAL_TICKS):
        right = min(total_ticks, left + CENTRAL_TICKS)
        input_left = min(max(0, left - halo_ticks), total_ticks - CHUNK_TICKS)
        input_right = input_left + CHUNK_TICKS
        if input_left > 0 and left - input_left < halo_ticks:
            raise ValueError("Insufficient real left halo")
        if input_right < total_ticks and input_right - right < halo_ticks:
            raise ValueError("Insufficient real right halo")
        windows.append((left, right, input_left, input_right))
    return windows


def stitched_output(
    forward: Callable[[np.ndarray], np.ndarray],
    values: np.ndarray,
    windows: list[tuple[int, int, int, int]],
) -> np.ndarray:
    output = np.empty((1, values.shape[1], 3), dtype=np.float32)
    for left, right, input_left, input_right in windows:
        prediction = forward(values[:, input_left:input_right])
        output[:, left:right] = prediction[:, left - input_left:right - input_left]
    return output


def inputs(ticks: int, seed: int) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    shape = (1, ticks, INPUT_FEATURES)
    structured = np.zeros(shape, dtype=np.float32)
    structured[0, :, 0] = np.linspace(-5, 5, ticks, dtype=np.float32)
    structured[0, ::2, 1::2] = 3
    structured[0, 1::2, 1::2] = -3
    for index in {0, min(61, ticks - 1), ticks // 2, max(0, ticks - 63), ticks - 1}:
        structured[0, index, 2::2] = 5
    return {
        "normal": generator.normal(size=shape).astype(np.float32),
        "zero": np.zeros(shape, dtype=np.float32),
        "boundary-impulses-ramp-alternation": structured,
    }


def latency(call: Callable[[], Any], repeats: int, warmup: int) -> dict[str, Any]:
    started = time.perf_counter()
    call()
    first_call = 1000 * (time.perf_counter() - started)
    for _ in range(warmup):
        call()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        timings.append(1000 * (time.perf_counter() - started))
    return {
        "firstMeasuredCallMilliseconds": first_call,
        "warmupCalls": warmup,
        "measuredCalls": repeats,
        "medianMilliseconds": float(np.median(timings)),
        "p95Milliseconds": float(np.percentile(timings, 95)),
        "minimumMilliseconds": min(timings),
        "maximumMilliseconds": max(timings),
    }


def qualify(kind: str, args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    model = CompactTemporalNetwork(CompactTemporalConfig(kind=kind)).cpu().eval()
    windows = chunk_windows(args.full_ticks, model.config.halo_ticks)
    chunk_path = args.output / f"{kind}-synthetic-static-{CHUNK_TICKS}.onnx"
    full_path = args.output / f"{kind}-synthetic-static-{args.full_ticks}.onnx"
    artifacts = {
        "portableChunk": export_model(model, chunk_path, CHUNK_TICKS),
        "fullSequenceDiagnostic": export_model(model, full_path, args.full_ticks),
    }
    chunk_session, chunk_load = cpu_session(chunk_path, args.threads)
    full_session, full_load = cpu_session(full_path, args.threads)
    full_inputs = inputs(args.full_ticks, args.seed + 1)
    chunk_inputs = inputs(CHUNK_TICKS, args.seed + 2)
    checks = []
    for name, values in chunk_inputs.items():
        checks.append(compare(ort_output(chunk_session, values), tensor_output(model, values),
                              f"chunk ONNX vs PyTorch: {name}"))
    for name, values in full_inputs.items():
        expected = tensor_output(model, values)
        whole_onnx = ort_output(full_session, values)
        chunk_onnx = stitched_output(lambda x: ort_output(chunk_session, x), values, windows)
        chunk_torch = stitched_output(lambda x: tensor_output(model, x), values, windows)
        checks.append(compare(whole_onnx, expected, f"full ONNX vs PyTorch: {name}"))
        checks.append(compare(chunk_onnx, whole_onnx, f"ONNX fixed chunks vs full: {name}"))
        checks.append(compare(chunk_torch, expected, f"PyTorch fixed chunks vs full: {name}"))
        checks.append(compare(chunk_onnx, expected, f"ONNX stitched vs PyTorch full: {name}"))
    full_values = full_inputs["normal"]
    chunk_values = chunk_inputs["normal"]
    timings = {
        "onnxChunkSessionCreationMilliseconds": chunk_load,
        "onnxFullSessionCreationMilliseconds": full_load,
        "onnxSingle252TickChunk": latency(lambda: ort_output(chunk_session, chunk_values), args.repeats, args.warmup),
        "onnxFullSequence": latency(lambda: ort_output(full_session, full_values), args.repeats, args.warmup),
        "onnxStitchedFullSequence": latency(
            lambda: stitched_output(lambda x: ort_output(chunk_session, x), full_values, windows),
            args.repeats, args.warmup,
        ),
        "pytorchSingle252TickChunk": latency(lambda: tensor_output(model, chunk_values), args.repeats, args.warmup),
        "pytorchFullSequence": latency(lambda: tensor_output(model, full_values), args.repeats, args.warmup),
    }
    return {
        "kind": kind,
        "status": "pass",
        "weights": "seeded synthetic random initialization; not trained and not production",
        "config": model.config.to_dict(),
        "trainableParameters": trainable_parameter_count(model),
        "rawFloat32ParameterBytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "artifacts": artifacts,
        "numericalChecks": checks,
        "maximumAbsoluteLogitError": max(row["maximumAbsoluteLogitError"] for row in checks),
        "chunkContract": {
            "fixedInputTicks": CHUNK_TICKS,
            "centralOutputTicks": CENTRAL_TICKS,
            "requiredRealHaloTicks": model.config.halo_ticks,
            "fullSequenceTicks": args.full_ticks,
            "chunksPerFullSequence": len(windows),
            "boundaryRule": "shift fixed window to true recording edge; never append synthetic input ticks",
            "shortRecordings": "below252ticks not qualified; require a separate valid-shape graph or dynamic export",
        },
        "latency": timings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New immutable output directory")
    parser.add_argument("--full-ticks", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.full_ticks < CHUNK_TICKS or args.threads < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("Invalid ticks, thread count, repeats, or warmup")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic static-shape ONNX export and desktop CPU inference qualification only",
        "trainingPerformed": False,
        "labelsOrVideosRead": False,
        "gpuUsed": False,
        "phoneBrowserPerformanceMeasured": False,
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu": cpu_name(), "torch": torch.__version__, "numpy": np.__version__,
            "onnx": onnx.__version__, "onnxruntime": ort.__version__,
            "provider": "CPUExecutionProvider", "intraOpThreads": args.threads,
            "interOpThreads": 1,
        },
        "seed": args.seed,
        "sourceFiles": [file_identity(Path(__file__).resolve()),
                        file_identity(Path(__file__).resolve().parents[1] / "analysis/compact_temporal_model.py")],
        "acceptance": {"rtol": RELATIVE_TOLERANCE, "atol": ABSOLUTE_TOLERANCE,
                       "allFinite": True, "staticShapes": True, "standardOnnxOperatorsOnly": True},
        "limitations": [
            "Synthetic untrained weights establish operator/numerical portability, not volleyball accuracy.",
            "CPU timings are desktop diagnostics, not measurements on a phone or browser WASM/WebGPU.",
            "Inference timing excludes video decode, feature extraction, transfers, UI, download and thermal effects.",
            "Latency first measured call follows numerical warmup; it is not process-cold inference.",
            "Float32 only; quantized/FP16 conversion and trained-model qualification remain required.",
            "No physical mobile device or browser was accessed by this script.",
        ],
        "models": [],
    }
    try:
        for kind in MODEL_KINDS:
            result = qualify(kind, args)
            report["models"].append(result)
            print(json.dumps({"kind": kind, "status": result["status"],
                              "maxLogitError": result["maximumAbsoluteLogitError"],
                              "chunkOnnxBytes": result["artifacts"]["portableChunk"]["sizeBytes"]}), flush=True)
        report["status"] = "pass"
    except Exception as error:
        report["status"] = "fail"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    report["wallSeconds"] = time.perf_counter() - started
    report_path = args.output / "runtime-qualification.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "report": file_identity(report_path)}), flush=True)
    if report["status"] != "pass":
        print(json.dumps(report["failure"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
