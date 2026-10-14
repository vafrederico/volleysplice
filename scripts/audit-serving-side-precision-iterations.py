#!/usr/bin/env python3
"""Independent provenance, fitting-optimum, policy, and checkpoint replay."""
import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import serving_side_precision_iterations as study

spec = importlib.util.spec_from_file_location("independent_selector", REPO/"scripts/audit-serving-side-selective.py")
independent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(independent)
check, same, read, sha = independent.check, independent.same, independent.read, independent.sha


def verify_logistic(model, x, truth):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-6] = 1.
    np.testing.assert_array_equal(model["impute"], np.median(x, axis=0))
    np.testing.assert_allclose(model["mean"], mean, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(model["scale"], scale, atol=1e-14, rtol=1e-14)
    z = (x-mean)/scale; w = np.asarray(model["weights"])
    p = independent.logistic_probability(model, x)
    weights = np.where(truth == 1, len(truth)/(2*truth.sum()), len(truth)/(2*(len(truth)-truth.sum())))
    residual = (p-truth)*weights/weights.sum()
    gradient = np.r_[z.T@residual+model["l2"]*w, residual.sum()]
    maximum = float(np.max(abs(gradient)))
    check(maximum < 1e-7, "Logistic optimum does not satisfy tolerance")
    return maximum


def gate_features(probabilities, quality, nonlinear):
    p = probabilities
    chosen = np.where((p[:, 0] >= .5)[:, None], p, 1-p)
    values = np.column_stack((p[:, 0], abs(2*p[:, 0]-1), chosen[:, 1], chosen[:, 2],
                              chosen.min(axis=1), np.ptp(p, axis=1),
                              np.logical_or((p >= .5).all(axis=1), (p < .5).all(axis=1))))
    if nonlinear:
        products = np.column_stack([values[:, i]*quality[:, j] for i in range(7) for j in range(12)])
        return np.column_stack((values, quality, values**2, products))
    return np.column_stack((values, quality))


def direct_features(data, arm):
    rank, quality = data["ranks"], data["quality"]
    average = (quality[:, :4]+quality[:, 4:8]+quality[:, 8:])/3
    if arm == study.CONTROL:
        return rank
    extra = {"01-quality-mean": [average], "02-quality-phases": [quality],
             "03-raw-court": [quality, data["court"]],
             "04-embedding-changes": [quality, data["changes"]]}
    if arm == "05-quality-interactions":
        product = np.column_stack([(rank[:, i]-.5)*average[:, j]
                                  for i in data["interactionColumns"] for j in range(4)])
        return np.column_stack((rank, quality, product))
    return np.column_stack((rank, *extra[arm]))


def verify_features(reg, rows, windows, data):
    c = reg["contract"]
    previous = read(c["priorRegistration"]["path"])
    for key in ("rows", "windows", "features"):
        path = independent.verify_reference(previous["contract"]["data"][key])
        if key == "rows":
            same(read(path), rows, "unchanged-rows")
        else:
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), windows if key == "windows" else data["ranks"])
    raw = read(c["inputs"]["rawCourt"]["path"])
    raw_rows = {(r["recordingId"], r["rallyId"], float(r["serveAnchor"])): r for r in raw["rows"]}
    dataset = read(c["inputs"]["dataset"]["path"])
    q, court = [], []
    ledger = read(os.environ.get("VOLLEYCUT_PRIVATE_LEDGER_POSIX") or os.environ["VOLLEYCUT_PRIVATE_LEDGER"])
    aliases = ledger["originalToAlias"]
    for row in dataset["rows"]:
        check(row["sourceSplit"] != "test", "Protected input row")
        if row["environment"] == "beach":
            continue
        current = rows[len(q)]
        check(current["label"] == row["label"] and current["sourceGroup"] == aliases[row["sourceGroup"]]
              and current["recordingIndex"] == aliases[row["recordingId"]], "Feature row identity differs")
        flight = row["configurations"]["192x108-r4c6"]
        q.append([value for phase in ("launch", "early", "late") for value in
            (flight[phase+":energyMean"], flight[phase+":activeFraction"],
             flight[phase+":smallComponentEnergyFraction"], abs(flight[phase+":flowY"]))])
        source = raw_rows[(row["recordingId"], row["rallyId"], float(row["serveAnchor"]))]
        court.append([source["courtFlowFeatures"][n] for n in raw["featureNames"]])
    np.testing.assert_array_equal(q, data["quality"])
    np.testing.assert_array_equal(court, data["court"])
    projection = np.random.default_rng(271828).normal(size=(3840, 32)).astype(np.float32)/np.sqrt(32.)
    valid = windows[..., 3840:3841]
    means, present = [], []
    for start, stop in ((0, 6), (7, 13), (7, 10), (10, 13)):
        count = valid[:, start:stop].sum(axis=1)
        means.append((windows[:, start:stop, :3840]*valid[:, start:stop]).sum(axis=1)/np.maximum(count, 1))
        present.append(count[:, 0] > 0)
    contrasts = np.column_stack(((means[1]-means[0])*(present[0]&present[1])[:, None] @ projection,
                                  (means[3]-means[2])*(present[2]&present[3])[:, None] @ projection))
    np.testing.assert_array_equal(contrasts, data["changes"])
    for mode in ("centered", "differences"):
        actual = study.temporal_transform(windows[:32], mode)
        for i in range(32):
            available = windows[i, :, 3840] > .5
            if mode == "centered":
                mean = windows[i, available, :3840].sum(axis=0)/available.sum()
                np.testing.assert_allclose(actual[i, available, :3840], windows[i, available, :3840]-mean, atol=2e-6)
                np.testing.assert_array_equal(actual[i, ~available, :3840], 0.)
            else:
                check(actual[i, 0, 3840] == 0, "First difference must be unavailable")
                for t in range(1, 13):
                    pair = available[t] and available[t-1]
                    expected = windows[i, t, :3840]-windows[i, t-1, :3840] if pair else np.zeros(3840)
                    np.testing.assert_array_equal(actual[i, t, :3840], expected)
                    check(actual[i, t, 3840] == pair, "Difference mask differs")
            np.testing.assert_array_equal(actual[i, :, 3841], windows[i, :, 3841])


def audit(report, device):
    study.prior.h.prior.configure_torch()
    root, fast, reg, rows, windows, data = study.load()
    summary = read(report); same(read(root/"summary.json"), summary, "public-private-summary")
    check(summary["contractSha256"] == reg["sha256"] and summary["scope"] ==
          {"serves": 968, "recordings": 26, "sourceGroups": 8}, "Study scope differs")
    previous = read(reg['contract']['priorRegistration']['path'])
    previous_summary = read(reg['contract']['priorSummary']['path'])
    check(previous_summary['contractSha256'] == previous['sha256'], "Historical control registration differs")
    same(summary['historicalControls'], previous_summary['variants'], "historical-controls")
    samples = [json.loads(line) for line in (root/'resources.jsonl').read_text().splitlines()]
    expected_resource = {"sampleCount": len(samples), "sampleIntervalSeconds": 10,
        "peakResidentMiB": max(x['residentKiB'] for x in samples)/1024,
        "maximumProcessSwapKiB": max(x['processSwapKiB'] for x in samples),
        "minimumWslAvailableGiB": min(x['availableKiB'] for x in samples)/1024**2,
        "wslSwapCounterDelta": {k: samples[-1]['swapCounters'][k]-samples[0]['swapCounters'][k]
                                for k in samples[0]['swapCounters']},
        "singleFittingProcess": True, "cpuThreads": 2}
    same(summary['resources'], expected_resource, "resource-summary")
    check(sha(fast/'resources.jsonl') == sha(root/'resources.jsonl'), "Resource logs differ")
    verify_features(reg, rows, windows, data)
    groups = reg["contract"]["sourceGroups"]
    truth = np.asarray([r["label"] for r in rows])
    exclusions = [*itertools.combinations(groups, 2), *((g,) for g in groups)]
    teachers, teacher_models, maximum_gradient = {}, {}, 0.
    artifact_count, cpu_fits, neural_fits = 0, 0, 0
    for count in (3, 2, 1, 0):
        for excluded in itertools.combinations(groups, count):
            folder = fast/"teachers"/independent.fold_name(excluded)
            meta = read(folder/"completed.json"); train, held = independent.partition(rows, excluded)
            check(meta["trainingRows"] == train.tolist() and meta["heldRows"] == held.tolist()
                  and meta["excludedGroups"] == sorted(excluded) and meta["contractSha256"] == reg["sha256"], "Teacher provenance differs")
            same(read(root/folder.relative_to(fast)/"completed.json"), meta, "durable-teacher")
            p = []
            for model, x in zip(meta["models"], (data["ranks"], data["ranks"][:, :82], data["ranks"][:, 82:]), strict=True):
                maximum_gradient = max(maximum_gradient, verify_logistic(model, x[train], truth[train]))
                check(model["l2"] == .1, "Teacher regularization differs")
                p.append(independent.logistic_probability(model, x))
            p = np.column_stack(p)
            check(sha(folder/"scores.npy") == meta["scoresSha256"]
                  == sha(root/folder.relative_to(fast)/"scores.npy"), "Teacher score hash differs")
            np.testing.assert_array_equal(p, np.load(folder/"scores.npy", allow_pickle=False))
            teachers[excluded], teacher_models[excluded] = p, meta["models"]
    bank, raw_bank, selected = {}, {}, set()
    for arm in (*study.CPU_ARMS, *study.NEURAL_ARMS):
        seeds = study.SEEDS if arm in study.NEURAL_ARMS else ("deterministic",)
        for seed in seeds:
            for excluded in [*exclusions, ()]:
                folder = fast/"fits"/arm/str(seed)/independent.fold_name(excluded)
                meta = read(folder/"completed.json"); train, held = independent.partition(rows, excluded)
                check(meta["contractSha256"] == reg["sha256"] and meta["arm"] == arm
                      and meta["trainingRows"] == train.tolist() and meta["heldRows"] == held.tolist()
                      and meta["excludedGroups"] == sorted(excluded), "Fit provenance differs")
                same(read(root/folder.relative_to(fast)/"completed.json"), meta, "durable-fit")
                for name, expected_sha in meta["artifacts"].items():
                    check(sha(folder/name) == expected_sha == sha(root/folder.relative_to(fast)/name), "Fit hash differs")
                    artifact_count += 1
                with np.load(folder/"scores.npz", allow_pickle=False) as saved:
                    raw = {int(k): saved[k].copy() for k in saved.files}
                if arm in study.CPU_ARMS:
                    cpu_fits += 1
                    if arm.startswith(("06-", "07-")):
                        p = np.full((len(rows), 3), np.nan); source_receipts = []
                        for group in groups:
                            if group in excluded:
                                continue
                            teacher_excluded = tuple(sorted((*excluded, group)))
                            indexes = np.asarray([i for i in train if rows[i]["sourceGroup"] == group])
                            p[indexes] = teachers[teacher_excluded][indexes]
                            source_receipts.append({"group": group, "teacherExcludedGroups": list(teacher_excluded), "rows": indexes.tolist()})
                        same(meta["gateSupervision"], source_receipts, "gate-cross-fitting")
                        nonlinear = arm.startswith("07-")
                        xtrain = gate_features(p[train], data["quality"][train], nonlinear)
                        xheld = gate_features(teachers[excluded][held], data["quality"][held], nonlinear)
                        y = ((p[train, 0] >= .5) == truth[train]).astype(int)
                    else:
                        x = direct_features(data, arm); xtrain, xheld, y = x[train], x[held], truth[train]
                    check(meta["featureCount"] == xtrain.shape[1] and meta["trainingTargetPositives"] == int(y.sum()), "CPU target differs")
                    models = read(folder/"models.json")
                    for candidate, l2 in study.L2.items():
                        model = models[str(candidate)]; check(model["l2"] == l2, "Candidate L2 differs")
                        maximum_gradient = max(maximum_gradient, verify_logistic(model, xtrain, y))
                        p = independent.logistic_probability(model, xheld)
                        expected = (np.column_stack(((teachers[excluded][held, 0] >= .5).astype(float), p))
                                    if arm.startswith(("06-", "07-")) else np.column_stack((p, np.ones(len(p)))))
                        np.testing.assert_allclose(raw[candidate], expected, atol=2e-14, rtol=0)
                    values = raw
                else:
                    neural_fits += 1
                    check(meta["seed"] == seed and len(meta["history"]) == 60
                          and all(np.isfinite(x["loss"]) for x in meta["history"]), "Neural training history differs")
                    mean, scale = data["quality"][train].mean(axis=0), data["quality"][train].std(axis=0)
                    scale[scale < 1e-6] = 1.
                    for epoch in study.EPOCHS:
                        check(raw[epoch].shape == (len(held), 4), "Neural output shape differs")
                        with np.load(folder/f"weights-{epoch}.npz", allow_pickle=False) as weights:
                            check(all(np.isfinite(weights[k]).all() for k in weights.files), "Nonfinite checkpoint")
                            for field in ("mean", "scale", "weights", "bias"):
                                np.testing.assert_array_equal(weights['base_'+field], np.asarray(teacher_models[excluded][0][field], np.float32))
                            np.testing.assert_array_equal(weights["quality_mean"], mean.astype(np.float32))
                            np.testing.assert_array_equal(weights["quality_scale"], scale.astype(np.float32))
                    values = {k: independent.persisted_scores(v) for k, v in raw.items()}
                bank[(arm, seed, excluded)] = values
                raw_bank[(arm, seed, excluded)] = raw
        print(json.dumps({"phase": "fit-audit", "arm": arm}), flush=True)
    return finish_audit(root, fast, reg, rows, windows, data, summary, report, device, bank, raw_bank,
                        maximum_gradient, artifact_count, cpu_fits, neural_fits)


def finish_audit(root, fast, reg, rows, windows, data, summary, report, device, bank, raw_bank,
                 maximum_gradient, artifact_count, cpu_fits, neural_fits):
    groups = reg["contract"]["sourceGroups"]
    selected = set(); outer_count = final_count = 0
    for arm in (*study.CPU_ARMS, *study.NEURAL_ARMS):
        seeds = study.SEEDS if arm in study.NEURAL_ARMS else ("deterministic",)
        saved = summary["experiments"][arm]
        same(read(fast/"results"/arm/"summary.json"), saved, "per-arm-summary")
        prediction_path = fast/"results"/arm/"predictions.npz"
        check(sha(prediction_path) == saved["predictionsSha256"] == sha(root/"results"/arm/"predictions.npz"), "Prediction hash differs")
        with np.load(prediction_path, allow_pickle=False) as source:
            published = {k: source[k].copy() for k in source.files}
        entries = {str(t): [] for t in study.TARGETS}; joint_inputs = {t: [] for t in study.TARGETS}; final_choices = []
        for seed in seeds:
            outputs = {t: np.full(len(rows), -2, np.int8) for t in study.TARGETS}
            chosen = {t: np.full((len(rows), 2), np.nan) for t in study.TARGETS}
            folds = {t: [] for t in study.TARGETS}; final = None
            for outer in groups:
                train, held = independent.partition(rows, (outer,)); local = {i: j for j, i in enumerate(train)}
                out = bank[(arm, seed, (outer,))]
                candidates = {k: np.full((len(train), 2), np.nan) for k in out}
                if final is None:
                    final = {k: np.full((len(rows), 2), np.nan) for k in out}
                for k in out:
                    final[k][held] = out[k]
                for inner in groups:
                    if inner == outer:
                        continue
                    excluded = tuple(sorted((outer, inner)))
                    _, indexes = independent.partition(rows, excluded)
                    keep = np.asarray([rows[i]["sourceGroup"] == inner for i in indexes])
                    positions = [local[i] for i in indexes[keep]]
                    for k, value in bank[(arm, seed, excluded)].items():
                        candidates[k][positions] = value[keep]
                for target in study.TARGETS:
                    policy = independent.select([rows[i] for i in train], candidates, target)
                    relative = Path("selection")/arm/str(seed)/str(target)/(outer+".json")
                    expected = {"innerRows": train.tolist(), "outerGroup": outer, "selected": policy}
                    same(read(fast/relative), expected, "inner-policy"); same(read(root/relative), expected, "durable-policy")
                    outer_count += 1
                    outputs[target][held] = independent.decisions(out[policy["epoch"]], policy["farCutoff"], policy["nearCutoff"])
                    chosen[target][held] = out[policy["epoch"]]
                    folds[target].append({"sourceGroup": outer, **{k: policy[k] for k in ("epoch", "farCutoff", "nearCutoff", "feasible")}})
                    if arm in study.NEURAL_ARMS:
                        selected.add((arm, seed, outer, policy["epoch"]))
            for target in study.TARGETS:
                np.testing.assert_array_equal(outputs[target], published[f"{seed}-{target}-decisions"])
                np.testing.assert_array_equal(chosen[target], published[f"{seed}-{target}-scores"])
                evaluation = independent.metrics(rows, outputs[target], chosen[target][:, 0])
                entries[str(target)].append({"seed": seed, "evaluation": evaluation, "folds": folds[target]})
                joint_inputs[target].append(outputs[target])
                policy = independent.select(rows, final, target)
                relative = Path("final-selection")/arm/str(seed)/f"{target}.json"
                same(read(fast/relative), policy, "final-policy"); same(read(root/relative), policy, "durable-final-policy")
                final_count += 1
                final_choices.append({"seed": seed, "target": target, **{k: policy[k] for k in ("epoch", "farCutoff", "nearCutoff", "feasible")}})
        targets = {}
        for target in study.TARGETS:
            stacked = np.stack(joint_inputs[target]); joint = np.full(len(rows), -1, np.int8)
            for side in (0, 1):
                joint[np.all(stacked == side, axis=0)] = side
            np.testing.assert_array_equal(joint, published[f"consensus-{target}-decisions"])
            item = entries[str(target)]
            targets[str(target)] = {"seeds": item, "meanSeeds": independent.means([x['evaluation'] for x in item]),
                "consensus": independent.metrics(rows, joint),
                "infeasibleOuterPolicies": sum(not f["feasible"] for e in item for f in e["folds"])}
        same(saved["targets"], targets, "aggregate-results")
        same(saved["finalChoices"], final_choices, "final-choices")
        print(json.dumps({"phase": "policy-audit", "arm": arm}), flush=True)
    fg = torch.from_numpy(np.column_stack((data['ranks'], data['quality'])).astype(np.float32)).to(device)
    maximum_reload = maximum_parity = 0.; deliveries = 0
    for arm in study.NEURAL_ARMS:
        transformed = study.temporal_transform(windows, study.mode_for(arm))
        raw = torch.from_numpy(transformed).to(device); del transformed
        for a, seed, outer, epoch in sorted(selected):
            if a != arm:
                continue
            _, held = independent.partition(rows, (outer,))
            path = fast/"fits"/arm/str(seed)/outer/f"weights-{epoch}.npz"
            model = study.load_head(path, device)
            outputs = study.prior.predict(model, raw[held], fg[held])
            expected = raw_bank[(arm, seed, (outer,))][epoch]
            error = float(np.max(abs(outputs-expected)))
            check(error <= (0. if device.startswith('cuda') else 3e-5), "Checkpoint replay differs")
            maximum_reload = max(maximum_reload, error)
        for item in summary["finalNeuralHeads"]:
            if item["arm"] != arm:
                continue
            relative = Path("fits")/arm/str(item["seed"])/"all-development"/f"weights-{item['epoch']}.npz"
            path = fast/relative
            check(sha(path) == item["weightsSha256"] and path.stat().st_size == item["bytes"], "Delivery artifact differs")
            policy = read(fast/"final-selection"/arm/str(item['seed'])/f"{item['target']}.json")
            for key in ("epoch", "farCutoff", "nearCutoff", "feasible"):
                same(item[key], policy[key], "delivery-policy")
            cpu, gpu = study.load_head(path, "cpu"), study.load_head(path, device)
            a = independent.persisted_scores(study.prior.predict(cpu, raw[:16].cpu(), fg[:16].cpu()))
            b = independent.persisted_scores(study.prior.predict(gpu, raw[:16], fg[:16]))
            error = float(np.max(abs(a-b))); check(error < 2e-5, "CPU/GPU parity differs")
            same(item['cpuGpuMaximumError'], error, "published-parity")
            maximum_parity = max(maximum_parity, error); deliveries += 1
        del raw; torch.cuda.empty_cache()
    check(cpu_fits == 296 and neural_fits == 333 and outer_count == 408 and final_count == 51 and deliveries == 27,
          "Audit completion counts differ")
    return {"kind": "serving-side-ten-precision-iterations-independent-audit-v1", "studyIndex": study.STUDY,
        "contractSha256": reg['sha256'], "summarySha256": sha(report), "auditorSha256": sha(__file__),
        "independentSelectorSha256": sha(REPO/"scripts/audit-serving-side-selective.py"),
        "teacherPartitionsVerified": 93, "cpuFittingPartitionsVerified": cpu_fits,
        "nestedNeuralFitsVerified": 324, "finalNeuralFitsVerified": 9, "artifactHashesVerified": artifact_count,
        "outerPoliciesReplayed": outer_count, "finalPoliciesReplayed": final_count,
        "selectedCheckpointsReloaded": len(selected), "maximumReloadLogitError": maximum_reload,
        "finalNeuralConfigurationsVerified": deliveries, "maximumCpuGpuScoreError": maximum_parity,
        "maximumLogisticGradient": maximum_gradient, "allAggregateAndConsensusMetricsReplayed": True,
        "allTeacherTargetGroupsExcluded": True, "frozenFeaturesIndependentlyRebuilt": True,
        "publicPrivateSummaryEqual": True, "protectedLabelsOpened": False, "productionChanged": False, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = audit(args.report, args.device)
    root = Path(study.private_value(study.STUDY))
    study.immutable(root/"independent-audit.json", result); study.immutable(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
