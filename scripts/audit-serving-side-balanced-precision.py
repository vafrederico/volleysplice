#!/usr/bin/env python3
"""Independently replay the matched Balanced serving-side encoder comparison."""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import serving_side_balanced_precision as study

spec = importlib.util.spec_from_file_location("independent", REPO/"scripts/audit-serving-side-selective.py")
independent = importlib.util.module_from_spec(spec); spec.loader.exec_module(independent)
check, same, read, sha = independent.check, independent.same, independent.read, independent.sha


def independent_window(times, tokens, anchor, duration, excluded):
    ranges = []
    for span in sorted(excluded, key=lambda r: float(r["start"])):
        start, end = max(0., float(span["start"])), min(duration, float(span["end"]))
        if end <= start:
            continue
        if ranges and start <= ranges[-1][1]:
            ranges[-1][1] = max(end, ranges[-1][1])
        else:
            ranges.append([start, end])
    left, right = 0., duration
    for start, end in ranges:
        check(not start <= anchor < end, "Anchor in ignored time")
        if end <= anchor:
            left = max(left, end)
        elif start > anchor:
            right = min(right, start)
    eligible = np.flatnonzero((times >= left)&(times < right))
    result = np.zeros((13, 3843), np.float32)
    for j, offset in enumerate(np.arange(-3., 3.01, .5)):
        result[j, 3841] = offset/3.
        requested = anchor+offset
        if not left <= requested < right or not len(eligible):
            continue
        index = eligible[np.argmin(abs(times[eligible]-requested))]
        if abs(times[index]-requested) <= .250001:
            result[j, :3840] = tokens[index].reshape(-1)
            result[j, 3840] = 1.
            result[j, 3842] = (times[index]-requested)/.25
    check(result[6, 3840] == 1, "Missing anchor embedding")
    return result


def verify_inputs(reg, rows, windows, features):
    c = reg["contract"]; previous = read(c["priorRegistration"]["path"])
    for ref in previous["contract"]["data"].values():
        check(sha(ref["path"]) == ref["sha256"], "Maximum coverage frozen data changed")
    same(rows, read(previous["contract"]["data"]["rows"]["path"]), "matched-rows")
    with np.load(previous["contract"]["data"]["features"]["path"], allow_pickle=False) as data:
        for k in ("ranks", "quality"):
            np.testing.assert_array_equal(features[k], data[k])
    old_windows = np.load(previous["contract"]["data"]["windows"]["path"], allow_pickle=False)
    np.testing.assert_array_equal(windows[..., 3840:], old_windows[..., 3840:])
    manifest = read(c["manifest"]["path"])
    check(c["encoderSha256"] == manifest["variants"]["high-f1"]["encoderWeightsSha256"]
          == sha(c["encoder"]["path"]), "Balanced weights differ")
    check(c["encoderSha256"] != manifest["variants"]["high-recall"]["encoderWeightsSha256"], "Wrong encoder variant")
    original = read(c["originalRegistration"]["path"])["contract"]
    for ref in original["inputs"].values():
        check(sha(ref["path"]) == ref["sha256"], "Original frozen input changed")
    raw_rows = [r for r in read(original["inputs"]["dataset"]["path"])["rows"] if r["environment"] != "beach"]
    catalog = {r["id"]: r for r in read(original["inputs"]["catalog"]["path"])["records"]}
    ignored = {r["recordingId"]: r["intervals"] for r in read(original["inputs"]["quality"]["path"])["records"]}
    for recording in dict.fromkeys(r["recordingIndex"] for r in rows):
        ref = c["receipts"][recording]; receipt = read(ref["receipt"]["path"])
        old_ref = original["receipts"][recording]; old_receipt = read(old_ref["receipt"]["path"])
        for old_identity in old_ref.values():
            check(sha(old_identity["path"]) == old_identity["sha256"], "Original frozen cache changed")
        same(receipt["imageInput"], old_receipt["imageInput"], "image-input-lineage")
        same(receipt["recipe"], old_receipt["recipe"], "encoder-recipe")
        check(receipt["labelsUsed"] is False and receipt["encoder"]["sha256"] == c["encoderSha256"], "Cache provenance differs")
        with np.load(ref["features"]["path"], allow_pickle=False) as data, np.load(old_ref["features"]["path"], allow_pickle=False) as old:
            check(data["tokens"].dtype == np.float16 and data["tokens"].shape == old["tokens"].shape, "Embedding precision/shape differs")
            for k in ("timestamps", "quality", "selected_presentation_times"):
                np.testing.assert_array_equal(data[k], old[k])
            times, tokens = data["timestamps"].copy(), data["tokens"].copy()
        for i, (row, raw) in enumerate(zip(rows, raw_rows, strict=True)):
            if row["recordingIndex"] != recording:
                continue
            record = raw["recordingId"]
            check(raw["sourceSplit"] != "test" and not catalog[record].get("protected")
                  and raw["label"] == row["label"] and receipt["id"] == record
                  and receipt["sourceGroup"] == catalog[record]["sourceGroup"], "Source identity differs")
            excluded = catalog[record].get("ignoredIntervals", [])+ignored.get(record, [])
            expected = independent_window(times, tokens, float(raw["serveAnchor"]),
                                          float(catalog[record]["durationSeconds"]), excluded)
            np.testing.assert_array_equal(windows[i], expected)
    transformed = study.base.temporal_transform(windows, "centered")
    for i, window in enumerate(windows):
        valid = window[:, 3840] > .5
        total = np.zeros(3840, np.float32)
        for frame in window[valid, :3840]:
            total += frame
        expected = (window[:, :3840]-total/max(int(valid.sum()), 1))*valid[:, None]
        np.testing.assert_allclose(transformed[i, :, :3840], expected, atol=4e-6, rtol=0)
        np.testing.assert_array_equal(transformed[i, :, 3840:], window[:, 3840:])
    return transformed


def audit(report, device):
    study.base.prior.h.prior.configure_torch()
    root, fast, reg, rows, windows, features = study.load()
    summary = read(report);same(summary, read(root/"summary.json"), "public-private-summary")
    check(summary["scope"] == {"serves": 968, "recordings": 26, "sourceGroups": 8}
          and summary["contractSha256"] == reg["sha256"], "Scope differs")
    check(summary["encoderVariant"] == reg["contract"]["encoderVariant"] == "high-f1"
          and summary["encoderSha256"] == reg["contract"]["encoderSha256"]
          and summary["embeddingPrecision"] == "fp16-rounded"
          and summary["newNestedNeuralFits"] == 108 and summary["finalNeuralFits"] == 3
          and summary["teacherPartitions"] == 37 and summary["protectedLabelsOpened"] is False
          and summary["productionChanged"] is False, "Public model metadata differs")
    prior = read(reg["contract"]["priorSummary"]["path"])
    same(summary["maximumCoverageControl"], prior["experiments"][study.ARM], "frozen-control")
    transformed = verify_inputs(reg, rows, windows, features)
    groups = reg["contract"]["sourceGroups"]; truth = np.asarray([r["label"] for r in rows])
    exclusions = [*itertools.combinations(groups, 2), *((g,) for g in groups), ()]
    bank, raw_bank, bases = {}, {}, {}
    maximum_gradient = 0.; artifacts = 0
    for excluded in exclusions:
        train, held = independent.partition(rows, excluded)
        folder = fast/"teachers"/independent.fold_name(excluded); meta = read(folder/"completed.json")
        check(meta["contractSha256"] == reg["sha256"] and meta["excludedGroups"] == list(excluded)
              and meta["trainingRows"] == train.tolist() and meta["heldRows"] == held.tolist(), "Teacher exclusions differ")
        same(meta, read(root/folder.relative_to(fast)/"completed.json"), "durable-teacher")
        probabilities = []
        for model, x in zip(meta["models"], (features["ranks"], features["ranks"][:, :82], features["ranks"][:, 82:]), strict=True):
            maximum_gradient = max(maximum_gradient, independent.verify_logistic(model, x[train], truth[train]))
            probabilities.append(independent.logistic_probability(model, x))
        check(sha(folder/"scores.npy") == meta["scoresSha256"] == sha(root/folder.relative_to(fast)/"scores.npy"), "Teacher hash differs")
        np.testing.assert_array_equal(np.column_stack(probabilities), np.load(folder/"scores.npy", allow_pickle=False))
        bases[excluded] = meta["models"][0]
    for seed in study.base.SEEDS:
        for excluded in exclusions:
            train, held = independent.partition(rows, excluded)
            folder = fast/"fits"/study.ARM/str(seed)/independent.fold_name(excluded)
            meta = read(folder/"completed.json")
            check(meta["seed"] == seed and meta["contractSha256"] == reg["sha256"] and meta["arm"] == study.ARM
                  and meta["excludedGroups"] == list(excluded) and meta["trainingRows"] == train.tolist()
                  and meta["heldRows"] == held.tolist() and len(meta["history"]) == 60
                  and all(np.isfinite(x["loss"]) for x in meta["history"]), "Neural fit provenance differs")
            same(meta, read(root/folder.relative_to(fast)/"completed.json"), "durable-fit")
            for name, expected in meta["artifacts"].items():
                check(sha(folder/name) == expected == sha(root/folder.relative_to(fast)/name), "Checkpoint hash differs")
                artifacts += 1
            with np.load(folder/"scores.npz", allow_pickle=False) as data:
                raw_bank[(seed, excluded)] = {int(k): data[k].copy() for k in data.files}
            bank[(seed, excluded)] = {k: independent.persisted_scores(v) for k,v in raw_bank[(seed, excluded)].items()}
            mean, scale = features["quality"][train].mean(axis=0), features["quality"][train].std(axis=0)
            scale[scale < 1e-6] = 1.
            for epoch in study.base.EPOCHS:
                check(raw_bank[(seed, excluded)][epoch].shape == (len(held), 4), "Score shape differs")
                with np.load(folder/f"weights-{epoch}.npz", allow_pickle=False) as data:
                    check(all(np.isfinite(data[k]).all() for k in data.files), "Nonfinite weights")
                    for field in ("mean", "scale", "weights", "bias"):
                        np.testing.assert_array_equal(data['base_'+field], np.asarray(bases[excluded][field], np.float32))
                    np.testing.assert_array_equal(data["quality_mean"], mean.astype(np.float32))
                    np.testing.assert_array_equal(data["quality_scale"], scale.astype(np.float32))
    print(json.dumps({"phase":"fit-audit", "neuralFits":111, "teachers":37}), flush=True)
    saved = summary["result"]
    same(saved, read(fast/"results"/study.ARM/"summary.json"), "per-arm-summary")
    prediction_path = fast/"results"/study.ARM/"predictions.npz"
    check(sha(prediction_path) == saved["predictionsSha256"] == sha(root/"results"/study.ARM/"predictions.npz"), "Predictions hash differs")
    with np.load(prediction_path, allow_pickle=False) as data:
        published = {k:data[k].copy() for k in data.files}
    entries = {str(t):[] for t in study.base.TARGETS}; joint_inputs = {t:[] for t in study.base.TARGETS}
    final_choices = []; selected = set(); outer_count = final_count = 0
    for seed in study.base.SEEDS:
        outputs = {t:np.full(len(rows), -2, np.int8) for t in study.base.TARGETS}
        chosen = {t:np.full((len(rows),2),np.nan) for t in study.base.TARGETS}
        folds = {t:[] for t in study.base.TARGETS}; final = None
        for outer in groups:
            train, held = independent.partition(rows, (outer,));local = {i:j for j,i in enumerate(train)}
            out = bank[(seed,(outer,))]; candidates = {k:np.full((len(train),2),np.nan) for k in out}
            if final is None:
                final = {k:np.full((len(rows),2),np.nan) for k in out}
            for k in out:
                final[k][held] = out[k]
            for inner in groups:
                if inner == outer:continue
                excluded = tuple(sorted((outer,inner))); _,indexes = independent.partition(rows,excluded)
                keep = np.asarray([rows[i]["sourceGroup"] == inner for i in indexes])
                positions = [local[i] for i in indexes[keep]]
                for k,value in bank[(seed,excluded)].items():candidates[k][positions] = value[keep]
            for target in study.base.TARGETS:
                policy = independent.select([rows[i] for i in train],candidates,target)
                relative = Path("selection")/study.ARM/str(seed)/str(target)/(outer+".json")
                expected = {"innerRows":train.tolist(),"outerGroup":outer,"selected":policy}
                same(read(fast/relative),expected,"inner-policy");same(read(root/relative),expected,"durable-policy")
                outputs[target][held] = independent.decisions(out[policy["epoch"]],policy["farCutoff"],policy["nearCutoff"])
                chosen[target][held] = out[policy["epoch"]]
                folds[target].append({"sourceGroup":outer,**{k:policy[k] for k in ("epoch","farCutoff","nearCutoff","feasible")}})
                selected.add((seed,outer,policy["epoch"]));outer_count += 1
        for target in study.base.TARGETS:
            np.testing.assert_array_equal(outputs[target],published[f"{seed}-{target}-decisions"])
            np.testing.assert_array_equal(chosen[target],published[f"{seed}-{target}-scores"])
            entries[str(target)].append({"seed":seed,"evaluation":independent.metrics(rows,outputs[target],chosen[target][:,0]),"folds":folds[target]})
            joint_inputs[target].append(outputs[target]); policy = independent.select(rows,final,target)
            relative = Path("final-selection")/study.ARM/str(seed)/f"{target}.json"
            same(read(fast/relative),policy,"final-policy");same(read(root/relative),policy,"durable-final-policy")
            final_choices.append({"seed":seed,"target":target,**{k:policy[k] for k in ("epoch","farCutoff","nearCutoff","feasible")}})
            final_count += 1
    targets = {}; paired = {}
    previous_root = Path(reg["contract"]["priorSummary"]["path"]).parent
    previous_path = previous_root/"results"/study.ARM/"predictions.npz"
    check(sha(previous_path) == summary["maximumCoverageControl"]["predictionsSha256"], "Control predictions differ")
    with np.load(previous_path,allow_pickle=False) as data:
        for target in study.base.TARGETS:
            stacked = np.stack(joint_inputs[target]);joint = np.full(len(rows),-1,np.int8)
            for side in (0,1):joint[np.all(stacked == side,axis=0)] = side
            np.testing.assert_array_equal(joint,published[f"consensus-{target}-decisions"])
            item = entries[str(target)]
            targets[str(target)] = {"seeds":item,"meanSeeds":independent.means([x['evaluation'] for x in item]),
                "consensus":independent.metrics(rows,joint),"infeasibleOuterPolicies":sum(not f["feasible"] for e in item for f in e["folds"])}
            control = data[f"consensus-{target}-decisions"]
            status = lambda x:np.where(x < 0,0,np.where(x == truth,1,2))
            old,new = status(control),status(joint)
            names = ["review","correct","wrong"]
            transitions = {a:{b:int(np.sum((old==i)&(new==j))) for j,b in enumerate(names)} for i,a in enumerate(names)}
            old_metrics = summary["maximumCoverageControl"]["targets"][str(target)]["consensus"]["pooled"]
            new_metrics = targets[str(target)]["consensus"]["pooled"]
            check(sum(sum(row.values()) for row in transitions.values()) == len(rows), "Paired denominator differs")
            for status, field in (("review", "reviewed"), ("correct", "correct"), ("wrong", "errors")):
                check(sum(transitions[status].values()) == old_metrics[field]
                      and sum(row[status] for row in transitions.values()) == new_metrics[field], "Paired marginal differs")
            paired[str(target)] = {"maximumCoverageToBalanced":transitions,
                "bothAcceptedSideDisagreements":int(np.sum((control>=0)&(joint>=0)&(control!=joint)))}
            check(paired[str(target)]["bothAcceptedSideDisagreements"] == transitions["wrong"]["correct"]+transitions["correct"]["wrong"],
                  "Paired side disagreement differs")
    same(saved["targets"],targets,"aggregate-results");same(saved["finalChoices"],final_choices,"final-choices")
    print(json.dumps({"phase":"policy-audit","outerPolicies":outer_count,"finalPolicies":final_count}),flush=True)
    raw = torch.from_numpy(transformed).to(device)
    fg = torch.from_numpy(np.column_stack((features['ranks'],features['quality'])).astype(np.float32)).to(device)
    maximum_reload = maximum_parity = 0.
    for seed,outer,epoch in sorted(selected):
        _,held = independent.partition(rows,(outer,))
        path = fast/"fits"/study.ARM/str(seed)/outer/f"weights-{epoch}.npz"
        output = study.base.prior.predict(study.base.load_head(path,device),raw[held],fg[held])
        error = float(np.max(abs(output-raw_bank[(seed,(outer,))][epoch])))
        check(error <= (0. if device.startswith('cuda') else 3e-5), "Checkpoint replay differs")
        maximum_reload = max(maximum_reload,error)
    for item in summary["finalNeuralHeads"]:
        path = fast/"fits"/study.ARM/str(item['seed'])/"all-development"/f"weights-{item['epoch']}.npz"
        check(sha(path) == item['weightsSha256'] and path.stat().st_size == item['bytes'], "Delivery identity differs")
        policy = read(fast/"final-selection"/study.ARM/str(item['seed'])/f"{item['target']}.json")
        for key in ("epoch","farCutoff","nearCutoff","feasible"):same(item[key],policy[key],"delivery-policy")
        cpu,gpu = study.base.load_head(path,"cpu"),study.base.load_head(path,device)
        a = independent.persisted_scores(study.base.prior.predict(cpu,raw[:16].cpu(),fg[:16].cpu()))
        b = independent.persisted_scores(study.base.prior.predict(gpu,raw[:16],fg[:16]))
        error = float(np.max(abs(a-b)));check(error < 2e-5,"CPU/GPU parity differs")
        same(item['cpuGpuMaximumError'],error,"published-parity");maximum_parity = max(maximum_parity,error)
    samples = [json.loads(line) for line in (root/'resources.jsonl').read_text().splitlines()]
    expected = {"sampleCount":len(samples),"sampleIntervalSeconds":10,"peakResidentMiB":max(x['residentKiB'] for x in samples)/1024,
        "maximumProcessSwapKiB":max(x['processSwapKiB'] for x in samples),"minimumWslAvailableGiB":min(x['availableKiB'] for x in samples)/1024**2,
        "wslSwapCounterDelta":{k:samples[-1]['swapCounters'][k]-samples[0]['swapCounters'][k] for k in samples[0]['swapCounters']},
        "singleFittingProcess":True,"cpuThreads":2}
    same(summary['resources'],expected,"resource-summary")
    check(sha(root/'resources.jsonl') == sha(fast/'resources.jsonl'),"Resource logs differ")
    check(outer_count == 72 and final_count == 9 and len(summary['finalNeuralHeads']) == 9, "Completion counts differ")
    return {"kind":"serving-side-balanced-precision-independent-audit-v1","studyIndex":study.STUDY,
        "contractSha256":reg['sha256'],"summarySha256":sha(report),"auditorSha256":sha(__file__),
        "independentSelectorSha256":sha(REPO/"scripts/audit-serving-side-selective.py"),
        "teacherPartitionsVerified":37,"nestedNeuralFitsVerified":108,"finalNeuralFitsVerified":3,
        "artifactHashesVerified":artifacts,"outerPoliciesReplayed":outer_count,"finalPoliciesReplayed":final_count,
        "selectedCheckpointsReloaded":len(selected),"maximumReloadLogitError":maximum_reload,
        "finalNeuralConfigurationsVerified":9,"maximumCpuGpuScoreError":maximum_parity,
        "maximumLogisticGradient":maximum_gradient,"matchedInputsIndependentlyRebuilt":True,
        "allAggregateAndConsensusMetricsReplayed":True,"pairedConsensusOutcomes":paired,
        "protectedLabelsOpened":False,"productionChanged":False,"passed":True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    args = parser.parse_args();result = audit(args.report,args.device)
    study.immutable(Path(study.base.private_value(study.STUDY))/"independent-audit.json",result)
    study.immutable(args.output,result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":main()
