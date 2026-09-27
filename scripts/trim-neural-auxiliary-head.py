#!/usr/bin/env python3
"""Remove only the fourth keep output from a frozen expanded-study checkpoint.

This CPU-only engineering conversion does not train, choose a candidate, change
the three primary outputs, or authorize production promotion. It creates a new
immutable derivative directory, never training-style completed.json metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.compact_temporal_model import CompactTemporalConfig, CompactTemporalNetwork
from analysis.expanded_temporal_model import ExpandedTemporalConfig, ExpandedTemporalNetwork

KINDS = ("linear", "tcn")
RANDOM_SEED = 20260919
PARITY_LENGTHS = (1, 63, 252)
ATOL = 1e-5
RTOL = 1e-5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path), "sizeBytes": path.stat().st_size}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def strict_arrays(path: Path, model: torch.nn.Module) -> dict[str, np.ndarray]:
    expected = {"mean": (104,), "scale": (104,),
                **{f"model::{name}": tuple(tensor.shape) for name, tensor in model.state_dict().items()}}
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(expected) and set(archive.files) == set(expected),
                "Checkpoint tensor names differ from the declared architecture")
        arrays = {}
        for name, shape in expected.items():
            value = archive[name]
            require(value.dtype == np.float32 and value.shape == shape and np.isfinite(value).all(),
                    f"Invalid tensor dtype/shape/values: {name}")
            arrays[name] = value.copy()
    require(np.all(arrays["scale"] > 0), "Scaler values must be positive")
    return arrays


def load_model(model: torch.nn.Module, arrays: dict) -> torch.nn.Module:
    model.load_state_dict({name: torch.from_numpy(arrays[f"model::{name}"].copy()) for name in model.state_dict()}, strict=True)
    return model.cpu().eval()


def trim_arrays(arrays: dict, kind: str) -> dict:
    require(kind in KINDS, "Only linear and TCN expanded checkpoints are supported")
    head = "context_head" if kind == "linear" else "head"
    names = {f"model::{head}.weight", f"model::{head}.bias"}
    result = {}
    for name, value in arrays.items():
        if name in names:
            require(value.shape[0] == 4, "Expected exactly four source output heads")
            result[name] = value[:3].copy()
        else:
            result[name] = value.copy()
    require(names <= set(result), "Final head tensors are missing")
    return result


def parity(original: torch.nn.Module, derivative: torch.nn.Module, values: np.ndarray, label: str) -> dict:
    require(values.dtype == np.float32 and values.ndim == 3 and values.shape[2] == 104
            and np.isfinite(values).all(), "Invalid parity input")
    tensor = torch.from_numpy(np.ascontiguousarray(values))
    with torch.inference_mode():
        expected = original(tensor).numpy()[:, :, :3]
        actual = derivative(tensor).numpy()
    require(expected.shape == actual.shape and np.isfinite(actual).all(), "Invalid derivative logits")
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL, err_msg=label)
    differences = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    return {"label": label, "shape": list(actual.shape), "passed": True,
            "maximumAbsoluteLogitError": float(differences.max()),
            "meanAbsoluteLogitError": float(differences.mean()), "atol": ATOL, "rtol": RTOL}


def convert(study: Path, checkpoint: Path, epoch: int, kind: str, output: Path,
            *, verify_validation: bool = False) -> dict:
    require(not output.exists(), "Refusing to overwrite an existing derivative directory")
    require(kind in KINDS and epoch > 0, "Unsupported kind or epoch")
    study, checkpoint, output = study.resolve(), checkpoint.resolve(), output.resolve()
    relative = checkpoint.relative_to(study / "fits")
    parts = relative.parts
    require(len(parts) == 5, "Expected cohort/kind/seed/outer-N/inner-N-or-refit checkpoint directory")
    cohort, path_kind, seed_text, outer_text, fold = parts
    prereg_path, completed_path = study / "preregistration.json", checkpoint / "completed.json"
    prereg, completed = read_json(prereg_path), read_json(completed_path)
    contract = prereg["contract"]
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    require(hashlib.sha256(canonical.encode()).hexdigest() == prereg["sha256"], "Invalid preregistration digest")
    require(path_kind == kind and cohort in contract["cohorts"] and kind in contract["kinds"]
            and int(seed_text) in contract["seeds"], "Checkpoint path is outside declared study scope")
    require(outer_text.startswith("outer-") and 0 <= int(outer_text[6:]) < len(contract["groups"]), "Invalid outer fold")
    require(fold == "refit" or (fold.startswith("inner-") and 0 <= int(fold[6:]) < len(contract["groups"])-1), "Invalid inner fold")
    require(completed["kind"] == kind and completed["seed"] == int(seed_text)
            and completed["contractSha256"] == prereg["sha256"] and epoch in completed["epochs"], "Completed checkpoint identity mismatch")
    require(epoch in contract["checkpointEpochs"], "Checkpoint epoch is outside predeclared schedule")
    inputs = {"preregistration": identity(prereg_path), "completed": identity(completed_path), "frozenModules": {}}
    for name, digest in contract["code"].items():
        require(Path(name).name == name and "\\" not in name, "Invalid frozen module path")
        path = REPO / "analysis" / name
        require(sha256(path) == digest, f"Frozen module changed: {name}")
        inputs["frozenModules"][name] = identity(path)
    name = f"weights-{epoch}.npz"
    require(name in completed["artifacts"], "Requested weights are not hash-bound by completed metadata")
    for artifact, digest in completed["artifacts"].items():
        require(Path(artifact).name == artifact and "\\" not in artifact, "Checkpoint artifact must be a local filename")
        require(sha256(checkpoint/artifact) == digest, f"Changed checkpoint artifact: {artifact}")
    source_path = checkpoint / name
    inputs["originalCheckpoint"] = identity(source_path)
    original = ExpandedTemporalNetwork(ExpandedTemporalConfig(kind=kind)).cpu().eval()
    derivative = CompactTemporalNetwork(CompactTemporalConfig(kind=kind)).cpu().eval()
    require(sum(parameter.numel() for parameter in original.parameters()) == completed["parameters"], "Source parameter count differs")
    original_arrays = strict_arrays(source_path, original)
    derivative_arrays = trim_arrays(original_arrays, kind)
    load_model(original, original_arrays)
    load_model(derivative, derivative_arrays)
    generator = np.random.default_rng(RANDOM_SEED)
    checks = [parity(original, derivative, generator.standard_normal((1, ticks, 104)).astype(np.float32),
                     f"fixed random normalized sequence: {ticks} ticks") for ticks in PARITY_LENGTHS]
    if verify_validation:
        from analysis.neural_development import load_examples, segments, standardized
        manifest_path = study.parent / "manifest.json"
        require(sha256(manifest_path) == contract["manifestSha256"], "Expanded manifest hash mismatch")
        manifest = read_json(manifest_path)
        reference = manifest["exactManifest"]
        exact_path = Path(reference["path"])
        require(sha256(exact_path) == reference["sha256"], "Exact manifest hash mismatch")
        inputs["expandedManifest"], inputs["exactManifest"] = identity(manifest_path), identity(exact_path)
        examples = load_examples(exact_path, with_dino=False)
        held = [row for row in examples if row.id in completed["validationIds"]]
        require([row.id for row in held] == completed["validationIds"], "Validation feature scope mismatch")
        for row in held:
            values = standardized(row, original_arrays["mean"], original_arrays["scale"], kind)
            for segment_index, (start, finish) in enumerate(segments(row.valid)):
                checks.append(parity(original, derivative, values[None, start:finish],
                                     f"actual validation {row.id} segment {segment_index}"))
    # Preserve immutable input identity even if a separate worker is active nearby.
    require(identity(source_path) == inputs["originalCheckpoint"] and identity(completed_path) == inputs["completed"], "Input changed during conversion")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / f"weights-{epoch}-primary-only.npz"
    with destination.open("xb") as handle:
        np.savez_compressed(handle, **derivative_arrays)
    reloaded = strict_arrays(destination, derivative)
    require(all(np.array_equal(value, reloaded[key]) for key, value in derivative_arrays.items()), "Serialized derivative tensors differ")
    head = "context_head" if kind == "linear" else "head"
    changed = {f"model::{head}.weight", f"model::{head}.bias"}
    require(all(np.array_equal(value, derivative_arrays[key]) for key, value in original_arrays.items() if key not in changed), "A trunk/scaler tensor changed")
    metadata = {
        "schemaVersion": 1, "kind": "expanded-neural-primary-head-derivative",
        "createdAt": datetime.now(timezone.utc).isoformat(), "status": "passed-engineering-conversion",
        "modelKind": kind, "cohort": cohort, "seed": int(seed_text), "outerIndex": int(outer_text[6:]),
        "fold": fold, "epoch": epoch, "contractSha256": prereg["sha256"],
        "selectionPolicy": "Checkpoint supplied as a predeclared engineering sample; no evaluation scores or candidate ranking used.",
        "researchOnly": True, "productionPromotionAllowed": False, "trainingPerformed": False,
        "headsRetained": ["live", "serve", "end"], "headRemoved": "keep",
        "operation": "Copy every trunk/scaler tensor unchanged; truncate only final weight and bias rows from four to first three.",
        "changedTensorNames": sorted(changed), "unchangedTensorCount": len(original_arrays)-len(changed),
        "originalParameters": sum(p.numel() for p in original.parameters()),
        "derivativeParameters": sum(p.numel() for p in derivative.parameters()),
        "config": derivative.config.to_dict(), "inputs": inputs, "derivativeCheckpoint": identity(destination),
        "parity": {"randomSeed": RANDOM_SEED, "randomLengths": list(PARITY_LENGTHS),
                   "actualValidationFeaturesChecked": verify_validation, "checks": checks,
                   "maximumAbsoluteLogitError": max(row["maximumAbsoluteLogitError"] for row in checks)},
        "scalerPreservedExactly": True, "allTrunkTensorsPreservedExactly": True,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": "cpu"},
        "converter": identity(Path(__file__).resolve()),
        "exportCompatibility": "Tensor schema matches the frozen compact three-head model. Existing export_graph/ScaledProbabilityModel helpers can consume it after strict loading; the old export CLI assumes the earlier study layout and cannot directly select this expanded-study inner-fit sample.",
    }
    with (output/"conversion-metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-validation", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    result = convert(args.study, args.checkpoint, args.epoch, args.kind, args.output,
                     verify_validation=args.verify_validation)
    print(json.dumps({"status": result["status"], "derivative": result["derivativeCheckpoint"],
                      "parameters": result["derivativeParameters"],
                      "maximumAbsoluteLogitError": result["parity"]["maximumAbsoluteLogitError"]}, indent=2))


if __name__ == "__main__":
    main()
