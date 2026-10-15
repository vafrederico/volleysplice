#!/usr/bin/env python3
"""Replay hybrid features, partitions, operating points, and reported decisions."""
from __future__ import annotations
import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from analysis import serving_side_hybrid as model_code
from analysis.private_ledger import private_value


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


REPO = Path(__file__).resolve().parents[1]
replay = module("temporal_replay", REPO/"scripts/audit-serving-side-temporal.py")
read, check, sha, same = replay.read, replay.check, replay.sha, replay.same


def prob(base, residual, multiplier):
    z = np.clip(base.astype(np.float64)+multiplier*residual.astype(np.float64), -35., 35.)
    return np.exp(-np.logaddexp(0., -z))


def choose(rows, scores, multipliers, thresholds):
    winner = None
    # Independent confusion counting: no call to the study selector or metrics.
    for multiplier in multipliers:
        for epoch in sorted(scores)[:1] if multiplier == 0 else sorted(scores):
            probabilities = prob(*scores[epoch], multiplier)
            check(np.isfinite(probabilities).all(), "Incomplete inner predictions")
            for threshold in thresholds:
                ev = replay.replay_metrics(rows, probabilities >= threshold)
                e = 0 if multiplier == 0 else epoch
                rank = (ev["sourceGroupMacroBalancedAccuracy"], ev["pooled"]["balancedAccuracy"],
                        ev["pooled"]["macroF1"], ev["worstSourceGroupBalancedAccuracy"],
                        -multiplier, -e, -abs(threshold-.5), -threshold)
                if winner is None or rank > winner[0]:
                    winner = rank, e, multiplier, threshold
    return winner[1:]


def reconstruct_features(c, rows, ledger):
    old = read(c["priorStudy"]["path"])["contract"]
    dataset = read(old["inputs"]["dataset"]["path"])
    original = [r for r in dataset["rows"] if r["environment"] != "beach"]
    baseline = read(old["inputs"]["baseline"]["path"])
    names = read(c["data"]["names"]["path"])
    check(names == baseline["finalModel"]["featureNames"], "Specialist signature differs")
    production = read(REPO/"prod/public/runtime/serving-side-85bc3325fbd4.json")
    check(names == production["featureNames"]
          and production["fingerprint"] == baseline["finalModel"]["fingerprint"],
          "Specialist signature differs from the packaged production model")
    aliases = ledger["originalToAlias"]
    values = np.empty((len(rows),len(names)),np.float64)
    for i,(source,row) in enumerate(zip(original,rows,strict=True)):
        check(aliases[source["recordingId"]] == row["recordingIndex"]
              and aliases[source["sourceGroup"]] == row["sourceGroup"]
              and source["label"] == row["label"] and source["sourceSplit"] != "test", "Feature row alignment differs")
        cohort = [r for r in original if r["recordingId"] == source["recordingId"]]
        for j,name in enumerate(names):
            family,key = name.split(":",1)
            if family == "v2":
                values[i,j] = source["v2RecordingRankFeatures"][key]
            else:
                population = np.asarray([r["configurations"]["192x108-r4c6"][key] for r in cohort])
                current = source["configurations"]["192x108-r4c6"][key]
                # Average zero-based tie rank, via counts rather than sorted runs.
                values[i,j] = .5 if len(cohort) == 1 else (np.sum(population < current)+.5*(np.sum(population == current)-1))/(len(cohort)-1)
    with np.load(c["data"]["features"]["path"]) as data:
        np.testing.assert_array_equal(values,data["values"])
    return values


def verify_base(path, rows, features, contract):
    saved = read(path); model = saved["model"]
    train = np.asarray([i for i,r in enumerate(rows) if r["sourceGroup"] not in saved["excludedGroups"]])
    held = np.asarray([i for i,r in enumerate(rows) if r["sourceGroup"] in saved["excludedGroups"]])
    check(saved["contractSha256"] == contract and saved["trainingRows"] == train.tolist()
          and saved["heldRows"] == held.tolist(), "Base partition differs")
    check(model_code.canonical(model) == saved["modelSha256"], "Base receipt hash differs")
    x = features[train]; y = np.asarray([rows[i]["label"] for i in train])
    mean = x.mean(axis=0); scale=x.std(axis=0); scale[scale<1e-6]=1.
    np.testing.assert_array_equal(mean,model["mean"]);np.testing.assert_array_equal(scale,model["scale"])
    np.testing.assert_array_equal(np.median(x,axis=0),model["impute"])
    z=(x-mean)/scale; w=np.asarray(model["weights"])
    p=1/(1+np.exp(-np.clip(z@w+model["bias"],-30.,30.)))
    weights=np.where(y==1,len(y)/(2*y.sum()),len(y)/(2*(len(y)-y.sum())))
    residual=(p-y)*weights/weights.sum()
    gradient=np.r_[z.T@residual+.1*w,residual.sum()]
    check(np.max(abs(gradient))<1e-7 and model["l2"]==.1,"Base optimum or L2 differs")
    return float(np.max(abs(gradient)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    model_code.prior.configure_torch()
    root=Path(private_value(model_code.STUDY_INDEX)); reg=read(root/"registration.json");c=reg["contract"]
    summary=read(args.report)
    check(model_code.canonical(c)==reg["sha256"]==summary["contractSha256"],"Registration differs")
    for ref in [*c["sourceCode"].values(),*c["data"].values(),c["protocol"],c["priorStudy"]]:
        check(sha(ref["path"])==ref["sha256"],"Registered artifact changed")
    ledger_path=os.environ.get("VOLLEYCUT_PRIVATE_LEDGER_POSIX") or os.environ["VOLLEYCUT_PRIVATE_LEDGER"]
    ledger=read(ledger_path)
    rows=read(c["data"]["rows"]["path"]);groups=c["sourceGroups"];epochs=c["epochs"];seeds=c["seeds"]
    check(len(rows)==968 and len(groups)==8,"Audit population differs")
    features=reconstruct_features(c,rows,ledger)
    with np.load(c["data"]["windows"]["path"]) as data:
        windows=data["values"].copy()
    old=read(c["priorStudy"]["path"])
    for ref in old["contract"]["inputs"].values():
        check(sha(ref["path"])==ref["sha256"],"Prior reviewed input changed")
    window_count=replay.replay_windows(old["contract"],rows,windows,ledger)
    bases=list((root/"bases").glob("*.json"))
    check(len(bases)==37,"Missing fold or final logistic fit")
    gradient=max(verify_base(p,rows,features,reg["sha256"]) for p in bases)
    bank={};fits=0
    for arm in c["arms"]:
        for seed in seeds:
            for excluded in [*itertools.combinations(groups,2),*((g,) for g in groups)]:
                key="+".join(sorted(excluded));folder=root/"fits"/arm/str(seed)/key
                meta=read(folder/"completed.json")
                train=[i for i,r in enumerate(rows) if r["sourceGroup"] not in excluded]
                held=[i for i,r in enumerate(rows) if r["sourceGroup"] in excluded]
                check(meta["contractSha256"]==reg["sha256"] and meta["seed"]==seed and meta["arm"]==arm
                      and meta["trainingRows"]==train and meta["heldRows"]==held
                      and meta["excludedGroups"]==sorted(excluded) and meta["epochs"]==epochs,"Neural fold isolation failed")
                for name,digest in meta["artifacts"].items():
                    check(sha(folder/name)==digest,"Neural checkpoint bytes changed")
                with np.load(folder/"scores.npz") as f:
                    bank[(arm,seed,tuple(sorted(excluded)))]=(np.asarray(held),{e:(f[f"{e}-held-base"].copy(),f[f"{e}-held-residual"].copy()) for e in epochs})
                fits+=1
    print({"featureRowsReplayed":len(rows),"fitsVerified":fits},flush=True)
    arrays=np.load(root/"nested-predictions.npz")
    replay_error=0.;reloaded=set();selection_count=0;gap_reports={}
    fg=torch.from_numpy(features.astype(np.float32)).to(args.device)
    raw=torch.from_numpy(windows).to(args.device)
    for variant,record in summary["variants"].items():
        arm="hybrid" if variant=="logistic" or variant.startswith("hybrid") else "specialist-only"
        multipliers=(0.,) if variant=="logistic" else ((1.,) if variant.endswith("-full") else c["multipliers"])
        allp,allt,evs,gaps=[],[],[],[]
        for item in record["seeds"]:
            seed=item["seed"];p=np.full(len(rows),np.nan);t=p.copy()
            for group in groups:
                train=np.asarray([i for i,r in enumerate(rows) if r["sourceGroup"]!=group]);local={v:i for i,v in enumerate(train)}
                inner={e:(np.full(len(train),np.nan),np.full(len(train),np.nan)) for e in epochs}
                for other in groups:
                    if other==group:continue
                    ix,pair=bank[(arm,seed,tuple(sorted((group,other))))]
                    for e in epochs:
                        for index,b,d in zip(ix,*pair[e],strict=True):
                            if rows[index]["sourceGroup"]==other:
                                inner[e][0][local[index]]=b;inner[e][1][local[index]]=d
                choice=choose([rows[i] for i in train],inner,multipliers,c["thresholds"])
                selection=read(root/"selection"/variant/str(seed)/(group+".json"))
                selected=selection["selected"]
                check(selection["innerRows"]==train.tolist()
                      and choice==(selected["epoch"],selected["multiplier"],selected["threshold"]),"Nested choice replay failed")
                epoch,multiplier,threshold=choice;used=epoch or epochs[0]
                ix,scores=bank[(arm,seed,(group,))]
                p[ix]=prob(*scores[used],multiplier);t[ix]=threshold
                same(replay.replay_metrics([rows[i] for i in train],prob(*inner[used],multiplier)>=threshold),selected["evaluation"])
                if (arm,seed,group,used) not in reloaded:
                    path=root/"fits"/arm/str(seed)/group/f"weights-{used}.npz"
                    head=model_code.load_head(path,arm,args.device)
                    b,d=model_code.components(head,raw[ix],fg[ix])
                    difference=max(float(np.max(abs(b-scores[used][0]))),float(np.max(abs(d-scores[used][1]))))
                    replay_error=max(replay_error,difference)
                    check(difference<=1e-5,"Held checkpoint replay differs")
                    reloaded.add((arm,seed,group,used))
                with np.load(root/"fits"/arm/str(seed)/group/"scores.npz") as archive:
                    tp=prob(archive[f"{used}-train-base"],archive[f"{used}-train-residual"],multiplier)
                gaps.append({"seed":seed,"sourceGroup":group,
                             "trainMacroBA":replay.replay_metrics([rows[i] for i in train],tp>=threshold)["sourceGroupMacroBalancedAccuracy"],
                             "heldBA":replay.replay_metrics([rows[i] for i in ix],p[ix]>=threshold)["pooled"]["balancedAccuracy"]})
                selection_count+=1
            np.testing.assert_allclose(p,arrays[f"{variant}-{seed}-probabilities"],atol=3e-16,rtol=1e-14)
            np.testing.assert_array_equal(t,arrays[f"{variant}-{seed}-thresholds"])
            ev=replay.replay_metrics(rows,p>=t);same(ev,item["evaluation"])
            evs.append(ev);allp.append(p);allt.append(t)
        pred=np.mean(allp,axis=0)>=np.mean(allt,axis=0)
        same(replay.replay_metrics(rows,pred),record["ensemble"])
        same(float(np.mean([e["sourceGroupMacroBalancedAccuracy"] for e in evs])),record["meanSeedMacroBalancedAccuracy"])
        same(float(np.mean([e["pooled"]["balancedAccuracy"] for e in evs])),record["meanSeedPooledBalancedAccuracy"])
        same(min(e["worstSourceGroupBalancedAccuracy"] for e in evs),record["minimumSeedWorstGroupBalancedAccuracy"])
        truth=np.asarray([r["label"] for r in rows]);baseline=np.asarray([r["baselineNear"] for r in rows])
        table=np.bincount(2*(baseline==truth).astype(int)+(pred==truth).astype(int),minlength=4)
        same(dict(zip(("bothWrong","headOnlyCorrect","baselineOnlyCorrect","bothCorrect"),map(int,table))),record["pairedErrorsAgainstHistoricalSpecialist"])
        gap_reports[variant]={"meanTrainingMacroBA":float(np.mean([g["trainMacroBA"] for g in gaps])),"folds":gaps}
        print({"variantReplayed":variant},flush=True)
    final_error=0.
    for entry in summary["finalHeads"]:
        seed=entry["seed"];scores={e:(np.full(len(rows),np.nan),np.full(len(rows),np.nan)) for e in epochs}
        for group in groups:
            ix,sc=bank[("hybrid",seed,(group,))]
            for e in epochs:scores[e][0][ix],scores[e][1][ix]=sc[e]
        multipliers=(1.,) if entry["kind"]=="full" else c["multipliers"]
        check(choose(rows,scores,multipliers,c["thresholds"])==(entry["epoch"],entry["multiplier"],entry["threshold"]),"Final full-development selection differs")
        if entry["epoch"]:
            folder=root/"final"/str(seed)/"fit";meta=read(folder/"completed.json")
            check(meta["trainingRows"]==list(range(len(rows))) and meta["heldRows"]==[]
                  and meta["excludedGroups"]==[] and meta["contractSha256"]==reg["sha256"],"Final fitting population differs")
            path=folder/f"weights-{entry['epoch']}.npz"
            check(sha(path)==entry["weightsSha256"]==meta["artifacts"][path.name],"Final checkpoint changed")
            gpu=model_code.load_head(path,"hybrid",args.device);cpu=model_code.load_head(path,"hybrid","cpu")
            gb,gc=model_code.components(gpu,raw[:16],fg[:16]);cb,cc=model_code.components(cpu,raw[:16].cpu(),fg[:16].cpu())
            error=float(np.max(abs(prob(gb,gc,entry["multiplier"])-prob(cb,cc,entry["multiplier"]))))
            check(error<2e-5,"Final device parity differs");final_error=max(final_error,error)
        else:
            check(entry["baseModelSha256"]==read(root/"bases"/"all-development.json")["modelSha256"],"Final base differs")
    privacy=module("privacy_replay",REPO/"scripts/audit-branch-privacy.py")
    identities={k for k in ledger.get("originalToAlias",{}) if len(k)>8};identities.update(ledger.get("denyTokens",[]))
    identities.update(v for v in ledger["privateValues"].values() if len(v)>12 and ("/" in v or "\\" in v))
    paths=[REPO/name for name in c["sourceCode"]]
    paths.extend([Path(__file__),args.report,REPO/"analysis/tests/test_serving_side_hybrid.py",
                  REPO/"scripts/diagnose-serving-side-temporal.py",REPO/"docs/research/serving-side-hybrid-head-2026-10-13.md",
                  REPO/"docs/research/serving-side-temporal-diagnostics-2026-10-13.json"])
    for path in paths:
        check(not privacy.categories(path.read_text(),identities,str(path.relative_to(REPO))),"Public research file privacy check failed")
    result={"status":"pass","contractSha256":reg["sha256"],"auditCodeSha256":sha(__file__),
            "inputWindowsIndependentlyReplayed":window_count,"specialistRowsIndependentlyReplayed":len(rows),
            "specialistFeaturesPerRow":features.shape[1],"logisticFitsVerified":len(bases),
            "maximumLogisticGradient":gradient,"nestedFitsAudited":fits,"outerSelectionsReplayed":selection_count,
            "heldCheckpointsReplayed":len(reloaded),"heldPredictionMaximumError":replay_error,
            "finalCpuGpuMaximumError":final_error,"trainingHeldGaps":gap_reports,
            "publicFilesPrivacyChecked":len(paths),"protectedLabelsOpened":False,"productionChanged":False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='trainingHeldGaps'},flush=True)


if __name__=='__main__':main()
