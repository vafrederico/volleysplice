#!/usr/bin/env python3
"""Preserve all-development temporal heads regardless of the control-arm winner.

These delivery refits use the registered recipe and full-development source-held
checkpoint selection. They add no outer-test performance estimate or new search.
"""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from analysis import serving_side_temporal as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    study.configure_torch()
    reg, rows, raw = study.load_registration(args.root)
    summary = study.read(args.root/"summary.json")
    values = torch.from_numpy(raw).to(args.device)
    heads = []
    for seed in study.SEEDS:
        scores = {e: np.full(len(rows), np.nan) for e in study.EPOCHS}
        for group in reg["contract"]["sourceGroups"]:
            indexes, held = study.saved_scores(args.root, "temporal", seed, (group,))
            for epoch in study.EPOCHS:
                scores[epoch][indexes] = held[epoch]
        choice, candidates = study.select_operating_point(rows, scores)
        if summary["selectedArm"] == "temporal":
            folder = args.root/"final"/str(seed)/"fit"
        else:
            base = args.root/"requested-temporal-heads"/str(seed)
            study.immutable(base/"selection.json", {"selected": choice, "candidates": candidates})
            folder = base/"fit"
            study.fit(rows, values, (), seed, folder, reg["sha256"], epochs=(choice["epoch"],))
        weights = folder/f"weights-{choice['epoch']}.npz"
        gpu_model = study.load_weights(weights, args.device)
        cpu_model = study.load_weights(weights, "cpu")
        gpu = study.predict(gpu_model, values[:16])
        cpu = study.predict(cpu_model, torch.from_numpy(raw[:16]))
        error = float(np.max(abs(cpu-gpu)))
        study.require(error <= 1e-5, "Delivered temporal head CPU/GPU parity differs")
        heads.append({"seed": seed, "epoch": choice["epoch"], "threshold": choice["threshold"],
                      "weightsSha256": study.digest(weights), "parameters": sum(p.numel() for p in cpu_model.parameters()),
                      "float32WeightBytes": 4*sum(p.numel() for p in cpu_model.parameters()),
                      "cpuGpuMaximumError": error})
    result = {"schemaVersion": 1, "kind": "requested-temporal-serving-side-heads-v1", "studyIndex": "private-reference-0225",
              "contractSha256": reg["sha256"], "arm": "temporal", "rowsFitted": len(rows),
              "deliveryCodeSha256": study.digest(Path(__file__)), "heads": heads,
              "newCandidateSearchOrOuterTestEstimate": False}
    study.immutable(args.root/"requested-temporal-heads.json", result)
    study.immutable(args.report, result)
    print(result)


if __name__ == "__main__":
    main()
