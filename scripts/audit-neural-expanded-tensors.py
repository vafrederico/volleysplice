#!/usr/bin/env python3
"""Independently audit frozen expanded-study scalers, model tensors and predictions.

No training, GPU work, source-video reads, or bound-code edits. --progress checks
the currently completed fit snapshot without writing. Final output is exclusive
create and requires the completed report and all preregistered fits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value
from analysis.config import FeatureConfig, NOISE_NORMALIZED_AUDIO_FEATURE_SET
from analysis.features import ABSOLUTE_FEATURE_NAMES, feature_names


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    return {"path": str(path), "sha256": sha256(path), "sizeBytes": path.stat().st_size}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def independent_percentile_normalization(raw, names):
    """Average tied ranks via sorted insertion positions, not training's unique()."""
    normalized = np.empty(raw.shape, dtype=np.float32)
    for j, name in enumerate(names):
        if name in ABSOLUTE_FEATURE_NAMES:
            normalized[:, j] = raw[:, j]
        elif len(raw) == 1:
            normalized[:, j] = .5
        else:
            ordered = np.sort(raw[:, j])
            first = np.searchsorted(ordered, raw[:, j], side="left")
            after = np.searchsorted(ordered, raw[:, j], side="right")
            normalized[:, j] = (first + after - 1.) / (2. * (len(raw) - 1))
    return normalized


def load_cache_inputs(manifest):
    names_expected = feature_names(FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET))
    inputs = {}
    for tier, key in (("exact", "exactRows"), ("draft", "draftRows"), ("coverage", "coverageRows")):
        for row in manifest[key]:
            require(row["environment"] in {"grass", "indoor"} and row["sourceGroup"] != private_value('source-group-008'), "Excluded source")
            cache = row["featureCaches"]["audiovisual"]
            path = Path(cache["path"])
            require(sha256(path) == cache["sha256"], f"Changed input cache: {row['id']}")
            with np.load(path, allow_pickle=False) as n:
                times = n["times"].astype(np.float64)
                raw = n["values"].astype(np.float32)
                names = tuple(str(x) for x in n["names"])
            require(names == names_expected and raw.shape == (len(times), 104), "Unexpected input shape/schema")
            require(len(times) > 0 and np.isfinite(times).all() and np.isfinite(raw).all(), "Nonfinite input")
            require(np.all(np.diff(times) > 0) and np.all(np.abs(np.diff(times) - .25) <= .10), "Bad 4Hz timeline")
            valid = np.ones(len(times), dtype=bool)
            for ignored in row.get("ignoredIntervals", []):
                valid &= ~((times >= ignored["start"]) & (times < ignored["end"]))
            if tier == "coverage":
                window = row["gameWindow"]
                valid &= (times >= window["start"]) & (times < window["end"])
            inputs[row["id"]] = {"tier": tier, "times": times, "valid": valid,
                                  "normalized": independent_percentile_normalization(raw, names),
                                  "cache": cache, "group": row["sourceGroup"]}
    return inputs


def independent_scaler(ids, inputs):
    require(bool(ids) and len(set(ids)) == len(ids), "Empty/duplicate scaler IDs")
    require(all(inputs[rid]["tier"] == "exact" for rid in ids), "Auxiliary data entered scaler scope")
    matrix = np.concatenate([inputs[rid]["normalized"][inputs[rid]["valid"]] for rid in ids])
    require(len(matrix) > 0, "No valid scaler ticks")
    mean64 = np.sum(matrix, axis=0, dtype=np.float64) / len(matrix)
    # Explicit centered sum of squares is independent of the runner's np.std().
    centered = matrix.astype(np.float64) - mean64
    std64 = np.sqrt(np.sum(centered * centered, axis=0) / len(matrix))
    return mean64.astype(np.float32), np.maximum(std64, 1e-4).astype(np.float32), len(matrix)


def parameter_contract(kinds):
    import torch
    from analysis.expanded_temporal_model import ExpandedTemporalConfig, ExpandedTemporalNetwork
    torch.set_num_threads(1)
    result = {}
    for kind in kinds:
        config = ExpandedTemporalConfig(kind=kind)
        model = ExpandedTemporalNetwork(config).cpu()
        shapes = {f"model::{name}": list(value.shape) for name, value in model.state_dict().items()}
        result[kind] = {"config": config.to_dict(), "stateShapes": shapes,
                        "parameterCount": sum(p.numel() for p in model.parameters())}
        del model
    return result


def audit(manifest_path, study, progress):
    prereg_path = study / "preregistration.json"
    prereg = read(prereg_path)
    contract = prereg["contract"]
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    require(hashlib.sha256(canonical.encode()).hexdigest() == prereg["sha256"], "Changed preregistration contract")
    require(sha256(manifest_path) == contract["manifestSha256"], "Changed expanded manifest")
    repo = Path(__file__).resolve().parents[1]
    for name in ("expanded_temporal_model.py", "compact_temporal_model.py", "features.py", "config.py"):
        require(sha256(repo / "analysis" / name) == contract["code"][name], f"Changed tensor/feature contract code: {name}")
    manifest = read(manifest_path)
    completed_paths = sorted((study / "fits").rglob("completed.json"))
    expected_fits = len(contract["cohorts"]) * len(contract["kinds"]) * len(contract["seeds"]) * len(contract["groups"]) ** 2
    complete = (study / "report.json").is_file() and len(completed_paths) == expected_fits
    require(progress or complete, f"Study incomplete: {len(completed_paths)}/{expected_fits} fits; use --progress")
    inputs = load_cache_inputs(manifest)
    architectures = parameter_contract(contract["kinds"])
    scaler_cache = {}
    fits = []
    failures = []
    checkpoint_count = prediction_array_count = 0
    max_mean_error = max_scale_error = 0.
    exact_scaler_equality_count = 0
    for path in completed_paths:
        try:
            meta_hash = sha256(path)
            meta = read(path)
            require(meta["contractSha256"] == prereg["sha256"], "Fit has wrong contract")
            ids = meta["trainIds"]
            require(meta["scalerTrainIds"] == ids, "Scaler IDs differ from exact training IDs")
            key = tuple(ids)
            if key not in scaler_cache:
                scaler_cache[key] = independent_scaler(ids, inputs)
            mean, scale, valid_ticks = scaler_cache[key]
            architecture = architectures[meta["kind"]]
            require(meta["parameters"] == architecture["parameterCount"], "Parameter count mismatch")
            expected_artifacts = {f"{stem}-{epoch}.npz" for epoch in meta["epochs"] for stem in ("weights", "predictions")}
            require(set(meta["artifacts"]) == expected_artifacts, "Unexpected resume artifact set")
            epochs = []
            for epoch in meta["epochs"]:
                wp, pp = (path.parent / f"{stem}-{epoch}.npz" for stem in ("weights", "predictions"))
                wh, ph = sha256(wp), sha256(pp)
                require(wh == meta["artifacts"][wp.name] and ph == meta["artifacts"][pp.name], "Changed resume artifact")
                with np.load(wp, allow_pickle=False) as n:
                    expected_keys = set(architecture["stateShapes"]) | {"mean", "scale"}
                    require(set(n.files) == expected_keys, "Wrong model/scaler tensor keys")
                    for tensor_name in n.files:
                        value = n[tensor_name]
                        expected_shape = [104] if tensor_name in {"mean", "scale"} else architecture["stateShapes"][tensor_name]
                        require(list(value.shape) == expected_shape, f"Wrong tensor shape: {tensor_name}")
                        require(value.dtype == np.float32 and np.isfinite(value).all(), f"Invalid tensor values/dtype: {tensor_name}")
                    require(np.all(n["scale"] >= np.float32(1e-4)), "Scale below minimum")
                    mean_error = float(np.max(np.abs(n["mean"].astype(np.float64) - mean)))
                    scale_error = float(np.max(np.abs(n["scale"].astype(np.float64) - scale)))
                    require(np.allclose(n["mean"], mean, rtol=0., atol=1e-6), f"Scaler mean mismatch {mean_error}")
                    require(np.allclose(n["scale"], scale, rtol=0., atol=1e-6), f"Scaler scale mismatch {scale_error}")
                    exact_equal = np.array_equal(n["mean"], mean) and np.array_equal(n["scale"], scale)
                predictions = []
                with np.load(pp, allow_pickle=False) as n:
                    require(set(n.files) == set(meta["validationIds"]), "Prediction keys do not match held recordings")
                    for rid in meta["validationIds"]:
                        source = inputs[rid]
                        require(source["tier"] == "exact", "Auxiliary recording prediction entered exact evaluation")
                        p = n[rid]
                        require(p.shape == (len(source["times"]), 4), f"Wrong T x 4 prediction shape: {rid}")
                        require(p.dtype == np.float32 and np.isfinite(p).all(), f"Nonfinite/wrong dtype predictions: {rid}")
                        require(np.all((p >= 0.) & (p <= 1.)), f"Predictions outside probability bounds: {rid}")
                        require(np.all(p[~source["valid"]] == 0.), f"Ignored-tick prediction mismatch: {rid}")
                        predictions.append({"id": rid, "shape": list(p.shape), "minimum": float(p.min()), "maximum": float(p.max()),
                                            "ignoredTicks": int((~source["valid"]).sum()),
                                            "inputCacheSha256": source["cache"]["sha256"],
                                            "firstTimestamp": float(source["times"][0]), "lastTimestamp": float(source["times"][-1])})
                        prediction_array_count += 1
                epochs.append({"epoch": epoch, "weightsSha256": wh, "predictionsSha256": ph,
                               "meanMaximumAbsoluteError": mean_error, "scaleMaximumAbsoluteError": scale_error,
                               "scalerExactlyEqual": exact_equal, "predictions": predictions})
                checkpoint_count += 1
                exact_scaler_equality_count += int(exact_equal)
                max_mean_error, max_scale_error = max(max_mean_error, mean_error), max(max_scale_error, scale_error)
            require(sha256(path) == meta_hash, "Completed fit metadata changed during audit")
            fits.append({"path": str(path.parent), "completedSha256": meta_hash, "kind": meta["kind"],
                         "seed": meta["seed"], "scalerTrainIds": ids, "scalerValidTickCount": valid_ticks,
                         "validationIds": meta["validationIds"], "checkpoints": epochs, "passed": True})
        except Exception as error:
            failures.append({"path": str(path), "error": str(error)})
    return {"kind": "neural-expanded-independent-tensor-audit-v1", "createdAt": datetime.now(timezone.utc).isoformat(),
            "passed": not failures, "studyComplete": complete, "partialSnapshot": not complete,
            "manifest": identity(manifest_path), "preregistration": identity(prereg_path),
            "report": identity(study / "report.json") if complete else None,
            "completedFitCountAtStart": len(completed_paths), "expectedFitCount": expected_fits,
            "passedFitCount": len(fits), "failedFitCount": len(failures),
            "checkedCheckpointCount": checkpoint_count, "checkedPredictionArrayCount": prediction_array_count,
            "uniqueExactScalerPopulations": len(scaler_cache), "scalerExactlyEqualCheckpointCount": exact_scaler_equality_count,
            "maximumMeanAbsoluteError": max_mean_error, "maximumScaleAbsoluteError": max_scale_error,
            "tolerance": {"absolute": 1e-6, "relative": 0.}, "architectures": architectures,
            "scalerMethod": "Independently compute tied percentile ranks using sorted insertion positions across each full recording, restore absolute channels, exclude ignored ticks, concatenate exact training only, float64 sum/centered sum-of-squares, float32 mean and max(std,1e-4). Does not call training normalization/scaler functions.",
            "predictionTimelineContract": "Prediction NPZ has no timestamps. Verify exact validation IDs and T x 4 against verified source-cache times; finite monotonic approx4Hz cache times and ignored-tick zeros. This checks index/length binding, not independent frame-level temporal replay.",
            "inputCaches": [{"id": rid, "tier": item["tier"], "path": item["cache"]["path"], "sha256": item["cache"]["sha256"],
                             "ticks": len(item["times"]), "validTicks": int(item["valid"].sum())} for rid, item in inputs.items()],
            "fits": fits, "failures": failures, "auditScript": identity(Path(__file__).resolve()),
            "limits": ["No training, GPU use, full video reads, or protected-label/outcome reads.",
                       "Separate study-summary audit owns population membership, exposure equivalence, metrics and selection reconstruction.",
                       "This verifies finite compatible stored tensors and independent exact-only scalers, not complete prediction replay from every checkpoint."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(private_value('private-reference-0070')))
    parser.add_argument("--study", type=Path)
    parser.add_argument("--progress", action="store_true", help="Validate completed snapshot and print summary without writing")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(not (args.progress and args.output), "--progress must not write an output")
    output = args.output or args.root / "tensor-scaler-audit-v1.json"
    if not args.progress:
        require(not output.exists(), "Refusing to overwrite an audit artifact")
    result = audit(args.root / "manifest.json", args.study or args.root / "study", args.progress)
    if not args.progress:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")
    summary = {k: result[k] for k in ("passed", "studyComplete", "completedFitCountAtStart", "expectedFitCount", "passedFitCount", "failedFitCount", "checkedCheckpointCount", "checkedPredictionArrayCount", "uniqueExactScalerPopulations", "scalerExactlyEqualCheckpointCount", "maximumMeanAbsoluteError", "maximumScaleAbsoluteError", "failures")}
    if not args.progress:
        summary["artifact"] = identity(output)
    print(json.dumps(summary, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
