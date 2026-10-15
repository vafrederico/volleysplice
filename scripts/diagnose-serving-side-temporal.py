#!/usr/bin/env python3
"""Post-hoc diagnostics of the immutable embedding-only temporal study.

Source-specific oracle thresholds are deliberately labeled as label-exposed
diagnostics. They are never fed to the hybrid study or used for model selection.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from analysis import serving_side_temporal as study
from analysis.private_ledger import private_value


def auc(truth, p):
    near, far = p[truth == 1], p[truth == 0]
    if not len(near) or not len(far):
        return None
    return float((np.sum(near[:,None] > far)+.5*np.sum(near[:,None] == far))/(len(near)*len(far)))


def calibration(rows, p, thresholds):
    y = np.asarray([r["label"] for r in rows], np.int64)
    def one(ix):
        truth, scores = y[ix], p[ix]
        near, far = scores[truth == 1], scores[truth == 0]
        candidates = np.unique(np.r_[0., scores, np.nextafter(scores.max(), np.inf)])
        choices = [(study.metrics(truth, scores >= t)["balancedAccuracy"], -abs(t-.5), -t) for t in candidates]
        best = max(choices)
        return {"rows": len(ix), "auc": auc(truth, scores),
                "nearMeanScore": float(near.mean()), "farMeanScore": float(far.mean()),
                "nearMedianScore": float(np.median(near)), "farMedianScore": float(np.median(far)),
                "brierScore": float(np.mean((truth-scores)**2)),
                "logLoss": float(np.mean(-truth*np.log(np.clip(scores,1e-7,1))-(1-truth)*np.log(np.clip(1-scores,1e-7,1)))),
                "nestedBalancedAccuracy": study.metrics(truth,scores >= thresholds[ix])["balancedAccuracy"],
                "fixedHalfBalancedAccuracy": study.metrics(truth,scores >= .5)["balancedAccuracy"],
                "labelExposedOracleBalancedAccuracy": best[0], "labelExposedOracleThreshold": float(-best[2]),
                "meanNestedThreshold": float(thresholds[ix].mean())}
    groups = {g: one(np.asarray([i for i,r in enumerate(rows) if r["sourceGroup"] == g]))
              for g in sorted({r["sourceGroup"] for r in rows})}
    return {"pooled": one(np.arange(len(rows))), "bySourceGroup": groups,
            "macroAuc": float(np.mean([r["auc"] for r in groups.values()])),
            "macroLabelExposedOracleBalancedAccuracy": float(np.mean([r["labelExposedOracleBalancedAccuracy"] for r in groups.values()])),
            "fixedHalfEvaluation": study.evaluate(rows, p >= .5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    study.configure_torch()
    root = Path(private_value("private-reference-0225"))
    output = Path(private_value("private-reference-0231"))
    reg, rows, windows = study.load_registration(root)
    summary = study.read(root/"summary.json")
    groups = reg["contract"]["sourceGroups"]
    raw = torch.from_numpy(windows)
    y = np.asarray([r["label"] for r in rows])
    results, predicted, cuts = [], [], []
    with np.load(root/"nested-predictions.npz", allow_pickle=False) as f:
        saved = {k:f[k].copy() for k in f.files}
    for seed in study.SEEDS:
        p, t = saved[f"temporal-{seed}-probabilities"], saved[f"temporal-{seed}-thresholds"]
        predicted.append(p); cuts.append(t)
        folds = []
        perturbed = {k: np.full(len(rows), np.nan) for k in ("anchor-only", "reverse-visual-time", "no-past", "no-future")}
        for group in groups:
            train, held = study.fold_indices(rows,(group,))
            choice = study.read(root/"selection"/"temporal"/str(seed)/(group+".json"))["selected"]
            model = study.load_weights(root/"fits"/"temporal"/str(seed)/group/f"weights-{choice['epoch']}.npz", "cpu")
            training = study.predict(model,raw[train]); held_replay = study.predict(model,raw[held])
            study.require(np.max(abs(held_replay-p[held])) < 2e-5, "Diagnostic CPU replay differs")
            folds.append({"sourceGroup": group, "epoch": choice["epoch"], "threshold": choice["threshold"],
                          "trainEvaluation": study.evaluate([rows[i] for i in train],training >= choice["threshold"]),
                          "heldEvaluation": study.evaluate([rows[i] for i in held],p[held] >= t[held]),
                          "trainLogLoss": float(np.mean(-y[train]*np.log(np.clip(training,1e-7,1))-(1-y[train])*np.log(np.clip(1-training,1e-7,1)))),
                          "trainAuc": auc(y[train],training)})
            local = windows[held]
            for kind in perturbed:
                changed = local.copy()
                if kind == "anchor-only":
                    changed = study.arm_values(changed, "anchor-only")
                elif kind == "reverse-visual-time":
                    changed[..., :3840] = local[:,::-1,:3840]
                    changed[..., 3840] = local[:,::-1,3840]
                    changed[..., 3842] = -local[:,::-1,3842]
                else:
                    ix = slice(0,6) if kind == "no-past" else slice(7,13)
                    changed[:,ix,:3841] = 0.; changed[:,ix,3842] = 0.
                perturbed[kind][held] = study.predict(model,torch.from_numpy(changed))
        results.append({"seed": seed, "calibration": calibration(rows,p,t), "folds": folds,
                        "meanFoldTrainingMacroBalancedAccuracy": float(np.mean([v["trainEvaluation"]["sourceGroupMacroBalancedAccuracy"] for v in folds])),
                        "meanFoldTrainingAuc": float(np.mean([v["trainAuc"] for v in folds])),
                        "perturbations": {k:{"evaluation":study.evaluate(rows,v>=t),
                                               "meanAbsoluteProbabilityChange":float(np.mean(abs(v-p)))} for k,v in perturbed.items()}})
        print({"diagnosedSeed":seed},flush=True)
    ensemble = calibration(rows,np.mean(predicted,axis=0),np.mean(cuts,axis=0))
    result = {"schemaVersion":1, "kind":"serving-side-temporal-post-hoc-diagnostics-v1",
              "studyIndex":"private-reference-0231", "priorStudyIndex":"private-reference-0225",
              "priorContractSha256":reg["sha256"], "diagnosticCodeSha256":study.digest(__file__),
              "seeds":results, "ensembleCalibration":ensemble,
              "meanSeedTrainingMacroBalancedAccuracy":float(np.mean([r["meanFoldTrainingMacroBalancedAccuracy"] for r in results])),
              "meanSeedHeldMacroBalancedAccuracy":summary["arms"]["temporal"]["meanSeedMacroBalancedAccuracy"],
              "meanSeedMacroAuc":float(np.mean([r["calibration"]["macroAuc"] for r in results])),
              "meanSeedLabelExposedOracleMacroBalancedAccuracy":float(np.mean([r["calibration"]["macroLabelExposedOracleBalancedAccuracy"] for r in results])),
              "oracleThresholdsAreLabelExposedAndNotDeployable":True,
              "perturbationsAreInferenceOnlyAndNotTrainedAblations":True,
              "usedForHybridModelSelection":False}
    study.immutable(output/"earlier-head-diagnostics.json",result)
    study.immutable(args.report,result)
    print({k:v for k,v in result.items() if k.startswith('meanSeed')},flush=True)


if __name__ == '__main__':
    main()
