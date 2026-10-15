#!/usr/bin/env python3
"""Post-hoc feature-reliance and paired source-bootstrap diagnostics.

All original operating points remain fixed. Permutations are label-blind and
local to recordings; these are inference perturbations, not retrained models.
"""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from analysis import serving_side_hybrid as h
from analysis.private_ledger import private_value


def recording_permutation(rows, rng):
    result=np.arange(len(rows))
    for recording in sorted({r["recordingIndex"] for r in rows}):
        ix=np.asarray([i for i,r in enumerate(rows) if r["recordingIndex"]==recording])
        result[ix]=rng.permutation(ix)
    return result


def source_variance(rows, values):
    """Descriptive one-way variance fraction per nonconstant feature; no labels."""
    x=np.asarray(values,np.float64);center=x.mean(axis=0)
    total=((x-center)**2).sum(axis=0)
    groups=np.asarray([r["sourceGroup"] for r in rows])
    between=sum(np.sum(groups==g)*(x[groups==g].mean(axis=0)-center)**2 for g in np.unique(groups))
    valid=total>1e-10;ratios=between[valid]/total[valid]
    return {"meanBetweenSourceVarianceFraction":float(ratios.mean()),
            "medianBetweenSourceVarianceFraction":float(np.median(ratios)),"nonconstantFeatures":int(valid.sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args();h.prior.configure_torch()
    root=Path(private_value(h.STUDY_INDEX));reg,rows,windows,features=h.load(root)
    summary=h.read(root/"summary.json");groups=reg["contract"]["sourceGroups"]
    raw=torch.from_numpy(windows).to(args.device);fg=torch.from_numpy(features.astype(np.float32)).to(args.device)
    kinds=("court-flow-permuted","flight-permuted","all-specialist-permuted","embeddings-permuted","anchor-only-context")
    results=[]
    for seed in h.prior.SEEDS:
        probabilities={k:np.full(len(rows),np.nan) for k in kinds};thresholds=np.full(len(rows),np.nan)
        original=np.full(len(rows),np.nan)
        for group in groups:
            ix=np.asarray([i for i,r in enumerate(rows) if r["sourceGroup"]==group]);local=[rows[i] for i in ix]
            choice=h.read(root/"selection"/"hybrid-full"/str(seed)/(group+".json"))["selected"]
            head=h.load_head(root/"fits"/"hybrid"/str(seed)/group/f"weights-{choice['epoch']}.npz","hybrid",args.device)
            b,c=h.components(head,raw[ix],fg[ix]);original[ix]=h.probability(b,c,1.);thresholds[ix]=choice["threshold"]
            # The same permutation feeds all feature-family comparisons.
            order=recording_permutation(local,np.random.default_rng(seed))
            for kind in kinds:
                x=raw[ix];s=fg[ix].clone()
                if kind=="court-flow-permuted":s[:,:82]=s[order,:82]
                elif kind=="flight-permuted":s[:,82:]=s[order,82:]
                elif kind=="all-specialist-permuted":s=s[order]
                elif kind=="embeddings-permuted":x=x[order]
                else:x=torch.from_numpy(h.prior.arm_values(windows[ix],"anchor-only")).to(args.device)
                b,c=h.components(head,x,s);probabilities[kind][ix]=h.probability(b,c,1.)
        baseline=h.prior.evaluate(rows,original>=thresholds)
        results.append({"seed":seed,"original":baseline,"perturbations":{k:{"evaluation":h.prior.evaluate(rows,p>=thresholds),
                         "meanAbsoluteProbabilityChange":float(np.mean(abs(p-original)))} for k,p in probabilities.items()}})
        print({"hybridDiagnosedSeed":seed},flush=True)
    comparisons={}
    rng=np.random.default_rng(20261013);resamples=rng.integers(0,len(groups),size=(20000,len(groups)))
    def group_values(variant):
        return np.mean([[v["evaluation"]["bySourceGroup"][g]["balancedAccuracy"] for g in groups]
                        for v in summary["variants"][variant]["seeds"]],axis=0)
    for candidate,control in (("hybrid-full","logistic"),("hybrid-selected","logistic"),
                              ("hybrid-full","specialist-only-full"),("hybrid-selected","specialist-only-selected")):
        delta=group_values(candidate)-group_values(control);sampled=delta[resamples].mean(axis=1)
        comparisons[candidate+"-minus-"+control]={"macroBalancedAccuracyDifference":float(delta.mean()),
              "descriptiveSourceBootstrap95PercentInterval":np.quantile(sampled,[.025,.975]).tolist(),
              "sourceGroups":len(groups),"resamples":len(resamples),"groupDifferences":dict(zip(groups,delta.tolist()))}
    curves={}
    for arm in h.ARMS:
        points=[]
        for epoch in h.prior.EPOCHS:
            train_metrics=[];held_metrics=[];losses=[]
            for seed in h.prior.SEEDS:
                p=np.full(len(rows),np.nan)
                for group in groups:
                    train,held=h.prior.fold_indices(rows,(group,))
                    folder=root/"fits"/arm/str(seed)/group
                    with np.load(folder/"scores.npz") as scores:
                        fitting=h.probability(scores[f"{epoch}-train-base"],scores[f"{epoch}-train-residual"],1.)
                        p[held]=h.probability(scores[f"{epoch}-held-base"],scores[f"{epoch}-held-residual"],1.)
                    train_metrics.append(h.prior.evaluate([rows[i] for i in train],fitting>=.5)["sourceGroupMacroBalancedAccuracy"])
                    losses.append(h.read(folder/"completed.json")["history"][epoch-1]["loss"])
                held_metrics.append(h.prior.evaluate(rows,p>=.5)["sourceGroupMacroBalancedAccuracy"])
            points.append({"epoch":epoch,"meanTrainingMacroBA":float(np.mean(train_metrics)),
                           "meanHeldMacroBA":float(np.mean(held_metrics)),"meanTrainingObjective":float(np.mean(losses))})
        curves[arm]=points
    tokens=windows[...,:3840].reshape(len(rows),13,4,960).astype(np.float64)
    anchor=tokens[:,6]
    normalized=(anchor-anchor.mean(axis=-1,keepdims=True))/np.sqrt(anchor.var(axis=-1,keepdims=True)+1e-5)
    changes=tokens[:,8:10].mean(axis=1)-tokens[:,4:6].mean(axis=1)
    valid=np.all(windows[:,4:10,3840]==1,axis=1)
    shifts={"rawAnchorEmbeddings":source_variance(rows,anchor.reshape(len(rows),-1)),
            "perTokenLayerNormalizedAnchorEmbeddings":source_variance(rows,normalized.reshape(len(rows),-1)),
            "afterMinusBeforeEmbeddings":source_variance([r for r,ok in zip(rows,valid,strict=True) if ok],changes[valid].reshape(int(valid.sum()),-1)),
            "fullyAvailableDifferenceRows":int(valid.sum()),
            "recordingRankSpecialistFeatures":source_variance(rows,features),
            "interpretation":"Label-free descriptive association, not a causal ablation. Recording ranks remove location shifts by construction. Layer normalization here excludes learned affine parameters."}
    result={"schemaVersion":1,"kind":"serving-side-hybrid-post-hoc-diagnostics-v1","studyIndex":h.STUDY_INDEX,
            "contractSha256":reg["sha256"],"codeSha256":h.digest(__file__),"seeds":results,
            "meanMacroBalancedAccuracyByPerturbation":{k:float(np.mean([r["perturbations"][k]["evaluation"]["sourceGroupMacroBalancedAccuracy"] for r in results])) for k in kinds},
            "pairedSourceBootstrap":comparisons,"fixedHalfFullCorrectionEpochCurves":curves,
            "featureSourceVariance":shifts,
            "epochCurvesArePostHocAndNotUsedForSelection":True,"permutationsAreLabelBlindWithinRecording":True,
            "perturbationsAreInferenceOnlyAndNotTrainedAblations":True,"usedForSelection":False,
            "bootstrapScope":"Descriptive development uncertainty; eight source groups, fixed fitted predictions, seed means paired within source. Not a protected-test confidence claim."}
    h.immutable(root/"hybrid-diagnostics.json",result);h.immutable(args.report,result)
    print(result["meanMacroBalancedAccuracyByPerturbation"],flush=True)


if __name__=='__main__':main()
