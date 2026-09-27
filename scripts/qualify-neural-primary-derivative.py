#!/usr/bin/env python3
"""ONNX CPU qualification of a predetermined trimmed engineering checkpoint.

Consumes immutable conversion metadata and existing exporter helpers. No fitting,
decoder/model selection, browser/phone execution or production modification.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO/"scripts"/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trim = script_module("primary_derivative_trim", "trim-neural-auxiliary-head.py")
export = script_module("primary_derivative_export", "export-neural-checkpoint.py")
from analysis.neural_development import load_examples, segments


def checked_input(item: dict) -> Path:
    path = Path(item["path"])
    trim.require(trim.sha256(path) == item["sha256"], f"Input identity changed: {path}")
    return path


def qualify(derivative_directory: Path, output: Path, *, threads: int, repeats: int) -> dict:
    trim.require(not output.exists(), "Refusing to overwrite qualification output")
    metadata_path = derivative_directory/"conversion-metadata.json"
    conversion = trim.read_json(metadata_path)
    trim.require(conversion["status"] == "passed-engineering-conversion" and not conversion["productionPromotionAllowed"], "Invalid conversion status")
    kind, epoch = conversion["modelKind"], conversion["epoch"]
    for name, item in conversion["inputs"].items():
        if name == "frozenModules":
            for module_item in item.values():
                checked_input(module_item)
        else:
            checked_input(item)
    checked_input(conversion["converter"])
    derivative_path = checked_input(conversion["derivativeCheckpoint"])
    source_path = checked_input(conversion["inputs"]["originalCheckpoint"])
    completed_path = checked_input(conversion["inputs"]["completed"])
    completed = trim.read_json(completed_path)
    saved_path = source_path.parent/f"predictions-{epoch}.npz"
    trim.require(trim.sha256(saved_path) == completed["artifacts"][saved_path.name], "Saved validation trace changed")
    prereg = trim.read_json(checked_input(conversion["inputs"]["preregistration"]))
    trim.require(prereg["sha256"] == conversion["contractSha256"] == completed["contractSha256"], "Study contract identity mismatch")
    manifest_path = checked_input(conversion["inputs"]["expandedManifest"])
    trim.require(trim.sha256(manifest_path) == prereg["contract"]["manifestSha256"], "Expanded manifest changed")
    manifest = trim.read_json(manifest_path)
    exact_path = checked_input(manifest["exactManifest"])
    examples = load_examples(exact_path, with_dino=False)
    held = [row for row in examples if row.id in completed["validationIds"]]
    trim.require([row.id for row in held] == completed["validationIds"], "Validation recording scope mismatch")
    derivative = trim.CompactTemporalNetwork(trim.CompactTemporalConfig(kind=kind)).cpu().eval()
    original = trim.ExpandedTemporalNetwork(trim.ExpandedTemporalConfig(kind=kind)).cpu().eval()
    original_arrays = trim.strict_arrays(source_path, original)
    derivative_arrays = trim.strict_arrays(derivative_path, derivative)
    trim.load_model(original, original_arrays)
    trim.load_model(derivative, derivative_arrays)
    expected_trimmed = trim.trim_arrays(original_arrays, kind)
    trim.require(all(np.array_equal(value, derivative_arrays[name]) for name, value in expected_trimmed.items()), "Derivative no longer exactly matches head-only trimming")
    original_wrapper = export.ScaledProbabilityModel(original, original_arrays["mean"], original_arrays["scale"]).cpu().eval()
    derivative_wrapper = export.ScaledProbabilityModel(derivative, derivative_arrays["mean"], derivative_arrays["scale"]).cpu().eval()
    output.mkdir(parents=True, exist_ok=False)
    graph_path = output/"model.onnx"
    graph = export.export_graph(derivative_wrapper, graph_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    started = time.perf_counter()
    session = ort.InferenceSession(str(graph_path), sess_options=options, providers=["CPUExecutionProvider"])
    session.disable_fallback()
    creation_ms = (time.perf_counter()-started)*1000
    trim.require(session.get_providers() == ["CPUExecutionProvider"], "Unexpected execution provider")

    def torch_original(values):
        with torch.inference_mode():
            return original_wrapper(torch.from_numpy(np.ascontiguousarray(values))).numpy()[:, :, :3]

    def torch_derivative(values):
        with torch.inference_mode():
            return derivative_wrapper(torch.from_numpy(np.ascontiguousarray(values))).numpy()

    def ort_forward(values):
        return session.run(["probabilities"], {"features": np.ascontiguousarray(values)})[0]

    checks, record_reports, trace_arrays = [], [], {}
    for length in (1, 7, 63, 127, 251, 252, 411):
        values = held[0].values[None, :length, :104]
        checks.append(export.compare(torch_derivative(values), torch_original(values), f"trimmed/original actual feature slice {length}"))
        checks.append(export.compare(ort_forward(values), torch_original(values), f"ONNX/original actual feature slice {length}"))
    with np.load(saved_path, allow_pickle=False) as frozen:
        trim.require(set(frozen.files) == set(completed["validationIds"]), "Saved validation trace recording IDs differ")
        for row in held:
            original_full = export.sequence_trace(torch_original, row, chunked=False)
            original_chunked = export.sequence_trace(torch_original, row, chunked=True)
            derivative_full = export.sequence_trace(torch_derivative, row, chunked=False)
            derivative_chunked = export.sequence_trace(torch_derivative, row, chunked=True)
            ort_full = export.sequence_trace(ort_forward, row, chunked=False)
            ort_chunked = export.sequence_trace(ort_forward, row, chunked=True)
            for label, actual, expected in (
                ("original CPU chunks / saved GPU first3", original_chunked, frozen[row.id][:, :3]),
                ("original chunks / original full", original_chunked, original_full),
                ("trimmed full / original full", derivative_full, original_full),
                ("trimmed chunks / original chunks", derivative_chunked, original_chunked),
                ("trimmed chunks / trimmed full", derivative_chunked, derivative_full),
                ("ONNX full / original full", ort_full, original_full),
                ("ONNX chunks / original chunks", ort_chunked, original_chunked),
                ("ONNX chunks / ONNX full", ort_chunked, ort_full),
            ):
                checks.append(export.compare(actual, expected, f"{row.id}: {label}"))
            trace_arrays.update({f"features::{row.id}": row.values[:, :104], f"original::{row.id}": original_chunked,
                                 f"pytorch::{row.id}": derivative_chunked, f"onnx::{row.id}": ort_chunked,
                                 f"times::{row.id}": row.times, f"valid::{row.id}": row.valid})
            record_reports.append({"id": row.id, "sourceGroup": row.group, "ticks": len(row.times),
                                   "validSegments": segments(row.valid), "durationSeconds": row.duration,
                                   "onnxFullPreparedFeatureLatency": export.benchmark(lambda: export.sequence_trace(ort_forward, row, chunked=False), repeats),
                                   "onnxChunkedPreparedFeatureLatency": export.benchmark(lambda: export.sequence_trace(ort_forward, row, chunked=True), repeats)})
    traces_path = output/"parity-traces.npz"
    with traces_path.open("xb") as handle:
        np.savez_compressed(handle, **trace_arrays)
    values = held[0].values[None, :252, :104]
    cpu_model = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    metadata = {"schemaVersion": 1, "status": "passed-engineering-onnx-cpu-qualification",
                "createdAt": datetime.now(timezone.utc).isoformat(), "modelKind": kind,
                "engineeringSample": {key: conversion[key] for key in ("cohort", "seed", "outerIndex", "fold", "epoch", "selectionPolicy")},
                "researchOnly": True, "productionPromotionAllowed": False, "modelSelectionPerformed": False,
                "contractSha256": conversion["contractSha256"], "graph": graph, "config": derivative.config.to_dict(),
                "parameters": sum(p.numel() for p in derivative.parameters()),
                "input": {"shape": [1, "ticks", 104], "dtype": "float32", "analysisFps": 4,
                          "externalNormalization": "recording percentile ranks with absolute feature columns restored",
                          "graphNormalization": "saved exact-fit (x-mean)/scale clipped to [-10,10]",
                          "videoDecodeAndFeatureExtractionIncluded": False},
                "output": {"shape": [1, "ticks", 3], "heads": ["live", "serve", "end"],
                           "endSemantics": "end event, not sustained dead-state probability", "keepHeadRemoved": True},
                "chunking": {"coreTicks": 128, "haloTicks": 62, "usualInputTicks": 252,
                             "rule": "real segment edges; reset at ignored spans; no synthetic input padding"},
                "numericalChecks": checks, "maximumAbsoluteProbabilityError": max(row["maximumAbsoluteProbabilityError"] for row in checks),
                "traces": trim.identity(traces_path), "records": record_reports,
                "runtime": {"scope": "Desktop CPU prepared-feature inference only; not phone or browser validation. Concurrent GPU study can affect host timings.",
                            "sessionCreationMilliseconds": creation_ms, "onnx252TickLatency": export.benchmark(lambda: ort_forward(values), repeats),
                            "providers": session.get_providers(), "threads": threads, "cpu": cpu_model,
                            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                            "onnx": onnx.__version__, "onnxruntime": ort.__version__},
                "inputs": {"conversionMetadata": trim.identity(metadata_path), "originalCheckpoint": trim.identity(source_path),
                           "derivativeCheckpoint": trim.identity(derivative_path), "completed": trim.identity(completed_path),
                           "savedValidationPredictions": trim.identity(saved_path), "expandedManifest": trim.identity(manifest_path),
                           "exactManifest": trim.identity(exact_path), "preregistration": conversion["inputs"]["preregistration"],
                           "exportHelpers": trim.identity(REPO/"scripts/export-neural-checkpoint.py"),
                           "converter": trim.identity(REPO/"scripts/trim-neural-auxiliary-head.py"),
                           "script": trim.identity(Path(__file__).resolve())},
                "limitations": ["Predetermined epoch5 inner-fit engineering sample; not an inner-selected outer-refit model or production candidate.",
                                "No decoder or interval selection is performed; qualification establishes probability-trace parity only.",
                                "Phones, browsers, thermal behavior, video decoding and feature extraction have not been measured here."]}
    export.write_json(output/"qualification.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derivative", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    trim.require(args.threads > 0 and args.repeats > 0, "Threads and repeats must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    result = qualify(args.derivative.resolve(), args.output.resolve(), threads=args.threads, repeats=args.repeats)
    print(json.dumps({"status": result["status"], "graphBytes": result["graph"]["sizeBytes"],
                      "maximumAbsoluteProbabilityError": result["maximumAbsoluteProbabilityError"],
                      "chunkLatency": result["runtime"]["onnx252TickLatency"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
