#!/usr/bin/env python3
"""Independently replay selective fitting, inner policies, and delivered decisions."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from analysis import serving_side_selective as model_code
from analysis.private_ledger import private_value


GRID = tuple([n / 100 for n in range(100)] + [.995, .999, .9999, 1., 2.])
VARIANTS = ("logistic", "bce-hybrid", "evidence-agreement", "selective")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def same(actual, expected, where="receipt"):
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and actual.keys() == expected.keys(), f"Fields differ: {where}")
        for key in expected:
            same(actual[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), f"List differs: {where}")
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            same(a, e, f"{where}[{index}]")
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        check(actual is not None and math.isfinite(actual) and abs(actual - expected) <= 2e-12,
              f"Numeric replay differs: {where}")
    else:
        check(actual == expected, f"Value differs: {where}")


def verify_reference(ref):
    check(sha(ref["path"]) == ref["sha256"], "Registered artifact changed")
    return Path(ref["path"])


def fold_name(excluded):
    return "+".join(sorted(excluded)) or "all-development"


def partition(rows, excluded):
    train = np.asarray([i for i, row in enumerate(rows) if row["sourceGroup"] not in excluded], np.int64)
    held = np.asarray([i for i, row in enumerate(rows) if row["sourceGroup"] in excluded], np.int64)
    check(not set(train).intersection(held) and len(train) + len(held) == len(rows), "Partition overlap")
    return train, held


def sigmoid(logits):
    return 1. / (1. + np.exp(-np.clip(np.asarray(logits, np.float64), -35., 35.)))


def persisted_scores(outputs):
    values = np.asarray(outputs)
    check(values.ndim == 2 and values.shape[1] == 4 and np.isfinite(values).all(), "Invalid cost-head outputs")
    evidence = sigmoid(values[:, :2])
    acceptance = evidence.sum(axis=1)
    check(np.all(acceptance > 0.) and np.all(acceptance <= 1. + 1e-12), "Invalid cost-head mass")
    check(np.all(np.abs(values[:, 2:]) <= 3.), "Cost-head residual bound differs")
    return np.column_stack((evidence[:, 1]/acceptance, np.minimum(acceptance, 1.)))


def decisions(values, far_cutoff, near_cutoff):
    values = np.asarray(values)
    check(values.ndim == 2 and values.shape[1] == 2 and np.isfinite(values).all(), "Invalid decision scores")
    result = np.full(len(values), -1, np.int8)
    for side, cutoff in ((0, far_cutoff), (1, near_cutoff)):
        routed = values[:, 0] >= .5 if side else values[:, 0] < .5
        confidence = values[:, 1] * (values[:, 0] if side else 1. - values[:, 0])
        result[routed & (confidence > 0.) & (confidence >= cutoff)] = side
    return result


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def counts(labels, predicted):
    # Each true-side row contains [review, far, near] output counts.
    confusion = np.bincount(3 * labels + predicted + 1, minlength=6).reshape(2, 3)
    far, near = int(confusion[:, 1].sum()), int(confusion[:, 2].sum())
    correct = int(confusion[0, 1] + confusion[1, 2])
    accepted, total = far + near, len(labels)
    lower = None
    if accepted:
        p, z = correct / accepted, 1.959963984540054
        lower = max(0., (p + z*z/(2*accepted) - z*math.sqrt(p*(1-p)/accepted + z*z/(4*accepted*accepted)))
                    / (1 + z*z/accepted))
    return {"rows": total, "accepted": accepted, "reviewed": total-accepted,
            "correct": correct, "errors": accepted-correct, "precision": ratio(correct, accepted),
            "nearPrecision": ratio(int(confusion[1, 2]), near), "farPrecision": ratio(int(confusion[0, 1]), far),
            "nearAccepted": near, "farAccepted": far, "coverage": ratio(accepted, total),
            "reviewFraction": ratio(total-accepted, total), "silentErrorFraction": ratio(accepted-correct, total),
            "correctAutoFraction": ratio(correct, total), "precisionLowerWilson95DESCRIPTIVE": lower}


def metrics(rows, prediction, underlying=None):
    truth = np.asarray([row["label"] for row in rows], np.int64)
    prediction = np.asarray(prediction, np.int64)
    check(prediction.shape == truth.shape and np.isin(prediction, (-1, 0, 1)).all(), "Invalid verdicts")
    groups = np.asarray([row["sourceGroup"] for row in rows])
    pooled = counts(truth, prediction)
    by_group = {group: counts(truth[groups == group], prediction[groups == group]) for group in sorted(set(groups))}
    by_label = {name: counts(truth[truth == side], prediction[truth == side]) for side, name in ((0, "far"), (1, "near"))}
    supported = [item["precision"] for item in by_group.values() if item["precision"] is not None]
    if underlying is not None:
        wrong = (np.asarray(underlying) >= .5) != truth
        captured = int(np.count_nonzero(wrong & (prediction == -1)))
        pooled.update(underlyingErrors=int(wrong.sum()), reviewedUnderlyingErrors=captured,
                      underlyingErrorReviewCapture=ratio(captured, int(wrong.sum())))
    return {"pooled": pooled,
            "sourceGroupMacroCoverage": float(np.mean([item["coverage"] for item in by_group.values()])),
            "sourceGroupMacroPrecision": float(np.mean(supported)) if supported else None,
            "worstSupportedGroupPrecision": min(supported) if supported else None,
            "supportedSourceGroups": len(supported), "bySourceGroup": by_group, "byLabel": by_label,
            "perLabelCoverage": {name: item["coverage"] for name, item in by_label.items()}}


def select(rows, candidates, target):
    """Independent threshold enumeration using per-side bincounts and row-of-grid scans."""
    truth = np.asarray([row["label"] for row in rows], np.int64)
    _, codes, group_sizes = np.unique([row["sourceGroup"] for row in rows], return_inverse=True, return_counts=True)
    winner, feasible_count = None, 0
    for epoch in sorted(candidates):
        values = np.asarray(candidates[epoch])
        check(values.shape == (len(rows), 2) and np.isfinite(values).all(), "Incomplete inner predictions")
        side = (values[:, 0] >= .5).astype(int)
        confidence = values[:, 1] * np.where(side, values[:, 0], 1. - values[:, 0])
        options = [[], []]
        for label in (0, 1):
            for cutoff in GRID:
                mask = (side == label) & (confidence > 0) & (confidence >= cutoff)
                accepted = int(mask.sum())
                if accepted < 20:
                    continue
                gc = np.bincount(codes[mask], minlength=len(group_sizes))
                right = mask & (truth == label)
                correct = int(right.sum())
                if np.count_nonzero(gc) < 4 or correct / accepted < target:
                    continue
                gr = np.bincount(codes[right], minlength=len(group_sizes))
                options[label].append((cutoff, gc, gr, correct / accepted))
        if not options[0] or not options[1]:
            continue
        near_count = np.stack([item[1] for item in options[1]])
        near_right = np.stack([item[2] for item in options[1]])
        for far_cutoff, far_count, far_right, far_precision in options[0]:
            accepted = near_count + far_count
            correct = near_right + far_right
            support = accepted > 0
            by_group = np.divide(correct, accepted, out=np.zeros_like(correct, dtype=float), where=support)
            feasible = by_group.sum(axis=1) / support.sum(axis=1) >= target
            feasible_count += int(feasible.sum())
            for index in np.flatnonzero(feasible):
                near_cutoff, _, _, near_precision = options[1][index]
                errors = int((accepted[index] - correct[index]).sum())
                rank = (float((accepted[index] / group_sizes).mean()), float(accepted[index].sum() / len(rows)),
                        min(far_precision, near_precision), -errors, -int(epoch), far_cutoff+near_cutoff,
                        min(far_cutoff, near_cutoff), far_cutoff, near_cutoff)
                if winner is None or rank > winner[0]:
                    winner = rank, epoch, far_cutoff, near_cutoff
    feasible = winner is not None
    epoch, far, near = winner[1:] if feasible else (min(candidates), 2., 2.)
    values = candidates[epoch]
    return {"epoch": epoch, "farCutoff": far, "nearCutoff": near, "feasible": feasible,
            "targetPrecision": target, "innerEvaluation": metrics(rows, decisions(values, far, near), values[:, 0]),
            "candidateCount": len(candidates)*len(GRID)**2, "feasibleCandidateCount": feasible_count}


def verify_logistic(model, features, truth):
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale < 1e-6] = 1.
    np.testing.assert_array_equal(model["mean"], mean)
    np.testing.assert_array_equal(model["scale"], scale)
    np.testing.assert_array_equal(model["impute"], np.median(features, axis=0))
    z, weights = (features-mean)/scale, np.asarray(model["weights"])
    p = 1 / (1 + np.exp(-np.clip(z@weights + model["bias"], -30., 30.)))
    sample_weights = np.where(truth == 1, len(truth)/(2*truth.sum()), len(truth)/(2*(len(truth)-truth.sum())))
    residual = (p-truth)*sample_weights/sample_weights.sum()
    gradient = np.r_[z.T@residual + .1*weights, residual.sum()]
    maximum = float(np.max(abs(gradient)))
    check(model["l2"] == .1 and maximum < 1e-7, "Logistic optimum differs")
    return maximum


def logistic_probability(model, features):
    z = (features-np.asarray(model["mean"]))/np.asarray(model["scale"])
    return 1/(1+np.exp(-np.clip(z@np.asarray(model["weights"])+model["bias"], -30., 30.)))


def means(evaluations):
    keys = ("precision", "nearPrecision", "farPrecision", "coverage", "reviewFraction", "accepted",
            "reviewed", "correct", "errors", "silentErrorFraction", "correctAutoFraction")
    result = {}
    for key in keys:
        values = [item["pooled"][key] for item in evaluations if item["pooled"][key] is not None]
        result[key] = float(np.mean(values)) if values else None
    result["sourceGroupMacroCoverage"] = float(np.mean([item["sourceGroupMacroCoverage"] for item in evaluations]))
    supported = [item["sourceGroupMacroPrecision"] for item in evaluations if item["sourceGroupMacroPrecision"] is not None]
    result["sourceGroupMacroPrecision"] = float(np.mean(supported)) if supported else None
    result["seedsWithAcceptedPredictions"] = sum(item["pooled"]["accepted"] > 0 for item in evaluations)
    return result


def self_check():
    # This is the only mode importing study metrics; the real audit never does.
    from analysis import serving_side_selective_metrics as production
    rng = np.random.default_rng(29)
    rows = [{"label": i % 2, "sourceGroup": f"group-{i//20}"} for i in range(140)]
    truth = np.asarray([row["label"] for row in rows])
    for iteration in range(8):
        candidates = {}
        for epoch in (5, 15, 30):
            if iteration == 0:
                q, a = np.full(len(rows), .6), np.ones(len(rows))
            else:
                q = np.where(truth, .5 + .5*rng.random(len(rows)), .5*rng.random(len(rows)))
                if iteration % 2:
                    flipped = rng.choice(len(rows), 8, replace=False)
                    q[flipped] = 1-q[flipped]
                a = rng.random(len(rows)) if iteration % 3 else np.ones(len(rows))
            candidates[epoch] = np.column_stack((q, a))
        for target in (.95, .98, .99):
            same(production.select(rows, candidates, target), select(rows, candidates, target), "synthetic-selector")
    print(json.dumps({"phase": "selective-audit-self-check", "cases": 24, "passed": True}), flush=True)


def audit(report, device):
    model_code.h.prior.configure_torch()
    root, fast = Path(private_value("private-reference-0236")), Path(private_value("private-reference-0237"))
    reg, summary = read(root/"registration.json"), read(report)
    c = reg["contract"]
    check(canonical(c) == reg["sha256"] == summary["contractSha256"], "Registration differs")
    same(read(root/"summary.json"), summary, "public-private-summary")
    refs = [*c["sourceCode"].values(), *c["data"].values(), c["priorRegistration"],
            *c["priorFits"].values(), *c["priorBases"].values()]
    for ref in refs:
        verify_reference(ref)
    for name, ref in c["sourceCode"].items():
        if name.endswith(".py"):
            check(sha(REPO/name) == ref["sha256"], "Imported source differs")
    rows = read(c["data"]["rows"]["path"])
    windows = np.load(c["data"]["windows"]["path"], allow_pickle=False)
    features = np.load(c["data"]["features"]["path"], allow_pickle=False)
    groups, seeds, epochs, targets = c["sourceGroups"], c["seeds"], c["epochs"], c["targets"]
    check(c["warmupEpochs"] == 0 and c["falsePositiveWeight"] == 49.
          and c["positiveWeight"] == 1. and c["residualPenalty"] == .001, "Changed objective differs")
    check(len(rows) == 968 and len(groups) == 8 and features.shape == (968, 237)
          and windows.shape == (968, 13, 3843) and len({r["recordingIndex"] for r in rows}) == 26,
          "Audit scope differs")
    check(all(row["rowIndex"] == i and row["label"] in (0, 1) for i, row in enumerate(rows)), "Row identity differs")
    previous = read(c["priorRegistration"]["path"])
    check(previous["sha256"] == c["priorContractSha256"], "Previous contract differs")
    for key in ("rows", "features", "windows"):
        previous_path = verify_reference(previous["contract"]["data"][key])
        if key == "rows":
            same(read(previous_path), rows, "frozen-rows")
        else:
            with np.load(previous_path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["values"], features if key == "features" else windows)
    truth = np.asarray([row["label"] for row in rows])
    exclusions = [*itertools.combinations(groups, 2), *((g,) for g in groups)]
    bases, max_gradient = {}, 0.
    for excluded in [*exclusions, ()]:
        key = fold_name(excluded)
        base = read(verify_reference(c["priorBases"][key]))
        train, held = partition(rows, excluded)
        check(base["contractSha256"] == c["priorContractSha256"] and base["excludedGroups"] == sorted(excluded)
              and base["trainingRows"] == train.tolist() and base["heldRows"] == held.tolist()
              and canonical(base["model"]) == base["modelSha256"], "Cached base provenance differs")
        max_gradient = max(max_gradient, verify_logistic(base["model"], features[train], truth[train]))
        bases[key] = base["model"]
    bank, fit_meta, final_meta = {}, {}, {}
    artifact_count, fit_count, final_count = 0, 0, 0
    for variant in ("selective", "bce-hybrid"):
        for seed in seeds:
            for excluded in [*exclusions, *([()] if variant == "selective" else [])]:
                key = fold_name(excluded)
                if not excluded:
                    folder = fast/"final"/str(seed)/"fit"
                    durable = root/"final"/str(seed)/"fit"
                else:
                    relative = Path("fits" if variant == "selective" else "previous/fits/hybrid")/str(seed)/key
                    folder = fast/relative
                    durable = root/relative if variant == "selective" else folder
                meta = read(folder/"completed.json")
                train, held = partition(rows, excluded)
                expected_contract = reg["sha256"] if variant == "selective" else c["priorContractSha256"]
                check(meta["trainingRows"] == train.tolist() and meta["heldRows"] == held.tolist()
                      and meta["excludedGroups"] == sorted(excluded) and meta["seed"] == seed
                      and meta["contractSha256"] == expected_contract and meta["epochs"] == epochs,
                      "Neural fit provenance differs")
                if variant == "bce-hybrid":
                    check(sha(folder/"completed.json") == c["priorFits"][f"{seed}/{key}"]["sha256"]
                          and meta["arm"] == "hybrid", "Old neural receipt differs")
                same(read(durable/"completed.json"), meta, "durable-fit")
                for filename, digest in meta["artifacts"].items():
                    check(sha(folder/filename) == digest and sha(durable/filename) == digest, "Neural artifact hash differs")
                    artifact_count += 1
                with np.load(folder/"scores.npz", allow_pickle=False) as saved:
                    outputs = {}
                    for epoch in epochs:
                        if variant == "selective":
                            held_raw = saved[f"{epoch}-held"]
                            train_raw = saved[f"{epoch}-train"]
                            check(held_raw.shape == (len(held), 4) and train_raw.shape == (len(train), 4)
                                  and np.isfinite(held_raw).all() and np.isfinite(train_raw).all(), "Saved score shape differs")
                            outputs[epoch] = persisted_scores(held_raw)
                            history = meta["history"][epoch-1]
                            check(history["warmup"] is False, "Unexpected cost-head warmup")
                            for label, indexes, raw in (("held", held, held_raw), ("train", train, train_raw)):
                                if len(raw):
                                    probability = persisted_scores(raw)
                                    same(history[label], {"meanAcceptance": float(probability[:, 1].mean()),
                                        "rawSideAccuracy": float(((probability[:, 0] >= .5) == truth[indexes]).mean())}, "training-diagnostic")
                        else:
                            base_logits, residual = saved[f"{epoch}-held-base"], saved[f"{epoch}-held-residual"]
                            outputs[epoch] = np.column_stack((sigmoid(base_logits.astype(float)+residual.astype(float)), np.ones(len(held))))
                        with np.load(folder/f"weights-{epoch}.npz", allow_pickle=False) as weights:
                            for field in ("mean", "scale", "weights", "bias"):
                                np.testing.assert_array_equal(weights[f"base_{field}"], np.asarray(bases[key][field], np.float32))
                            check(all(np.isfinite(weights[field]).all() for field in weights.files), "Nonfinite checkpoint")
                            if variant == "selective":
                                check(not any(field.startswith("acceptance.") for field in weights.files)
                                      and weights["correction.3.weight"].shape == (2, 64)
                                      and weights["correction.3.bias"].shape == (2,), "Unexpected cost-head state")
                if excluded:
                    bank[(variant, seed, key)] = (held, outputs)
                    fit_meta[(variant, seed, key)] = (folder, meta)
                    fit_count += variant == "selective"
                else:
                    final_meta[seed] = (folder, meta)
                    final_count += 1
    print(json.dumps({"phase": "selective-audit", "neuralFits": fit_count, "finalFits": final_count}), flush=True)
    control_count = 0
    for excluded in [*exclusions, ()]:
        key = fold_name(excluded)
        folder, durable = fast/"agreement"/key, root/"agreement"/key
        meta = read(folder/"models.json")
        same(read(durable/"models.json"), meta, "durable-control")
        train, held = partition(rows, excluded)
        if not excluded:
            held = np.arange(len(rows))
        check(meta["contractSha256"] == reg["sha256"] and meta["excludedGroups"] == sorted(excluded)
              and meta["trainingRows"] == train.tolist() and meta["heldRows"] == held.tolist(), "Control partition differs")
        check(sha(folder/"scores.npz") == meta["scoresSha256"] == sha(durable/"scores.npz"), "Control hash differs")
        family = []
        for name, columns in (("full", slice(None)), ("court", slice(0, 82)), ("flight", slice(82, None))):
            model = meta["models"][name]
            max_gradient = max(max_gradient, verify_logistic(model, features[train, columns], truth[train]))
            family.append(logistic_probability(model, features[held, columns]))
        same(meta["models"]["full"], bases[key], "common-logistic-control")
        family = np.column_stack(family)
        route = family[:, 0] >= .5
        agrees = ((family >= .5) == route[:, None]).all(axis=1)
        confidence = np.where(route[:, None], family, 1-family).min(axis=1)
        acceptance = np.where(agrees, confidence / np.where(route, family[:, 0], 1-family[:, 0]), 0.)
        scores = np.column_stack((family[:, 0], acceptance))
        with np.load(folder/"scores.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["familyProbabilities"], family)
            np.testing.assert_array_equal(saved["scores"], scores)
        for seed in seeds:
            bank[("evidence-agreement", seed, key)] = (held, {0: scores})
            bank[("logistic", seed, key)] = (held, {0: np.column_stack((family[:, 0], np.ones(len(held))))})
        control_count += 1
    with np.load(fast/"nested-predictions.npz", allow_pickle=False) as data:
        published_arrays = {key: data[key].copy() for key in data.files}
    check(sha(fast/"nested-predictions.npz") == summary["predictionsSha256"] == sha(root/"nested-predictions.npz"), "Nested score hash differs")
    selected_checkpoints, policy_count, final_policy_count = set(), 0, 0
    reconstructed, final_choices = {}, {}
    selection_cache = {}
    for variant in VARIANTS:
        entries = {target: [] for target in targets}
        verdicts = {target: [] for target in targets}
        for seed in seeds:
            predicted = {target: np.full(len(rows), -2, np.int8) for target in targets}
            out_scores = {target: np.full((len(rows), 2), np.nan) for target in targets}
            folds = {target: [] for target in targets}
            final_candidates = None
            for outer in groups:
                train, held = partition(rows, (outer,))
                outer_ix, outer_scores = bank[(variant, seed, outer)]
                np.testing.assert_array_equal(outer_ix, held)
                inner = {epoch: np.full((len(train), 2), np.nan) for epoch in outer_scores}
                positions = {int(index): local for local, index in enumerate(train)}
                for group in groups:
                    if group == outer:
                        continue
                    indexes, candidate = bank[(variant, seed, fold_name((outer, group)))]
                    source_positions = [j for j, index in enumerate(indexes) if rows[index]["sourceGroup"] == group]
                    local_positions = [positions[int(indexes[j])] for j in source_positions]
                    for epoch in inner:
                        inner[epoch][local_positions] = candidate[epoch][source_positions]
                if final_candidates is None:
                    final_candidates = {epoch: np.full((len(rows), 2), np.nan) for epoch in outer_scores}
                for epoch in final_candidates:
                    final_candidates[epoch][held] = outer_scores[epoch]
                for target in targets:
                    cache_key = (variant, seed if variant in ("selective", "bce-hybrid") else 0, outer, target)
                    if cache_key not in selection_cache:
                        selection_cache[cache_key] = select([rows[index] for index in train], inner, target)
                    choice = selection_cache[cache_key]
                    saved_path = Path("selection")/variant/str(seed)/str(target)/(outer+".json")
                    saved = read(fast/saved_path)
                    same(read(root/saved_path), saved, "durable-selection")
                    check(saved["contractSha256"] == reg["sha256"] and saved["outerExcludedGroup"] == outer
                          and saved["innerRows"] == train.tolist(), "Selection inspected outer rows")
                    same(saved["selected"], choice, "inner-policy")
                    values = outer_scores[choice["epoch"]]
                    predicted[target][held] = decisions(values, choice["farCutoff"], choice["nearCutoff"])
                    out_scores[target][held] = values
                    folds[target].append({"sourceGroup": outer, "epoch": choice["epoch"], "farCutoff": choice["farCutoff"],
                        "nearCutoff": choice["nearCutoff"], "feasible": choice["feasible"],
                        "warmupSelected": False})
                    if variant in ("selective", "bce-hybrid"):
                        selected_checkpoints.add((variant, seed, outer, choice["epoch"]))
                    policy_count += 1
            for target in targets:
                name = f"{variant}-{seed}-{target}"
                np.testing.assert_array_equal(published_arrays[name+"-decisions"], predicted[target])
                np.testing.assert_array_equal(published_arrays[name+"-scores"], out_scores[target])
                reconstructed[name+"-decisions"] = predicted[target]
                reconstructed[name+"-scores"] = out_scores[target]
                entries[target].append({"seed": seed, "evaluation": metrics(rows, predicted[target], out_scores[target][:, 0]), "folds": folds[target]})
                verdicts[target].append(predicted[target])
                choice = select(rows, final_candidates, target)
                path = Path("final-selection")/variant/str(seed)/f"{target}.json"
                same(read(fast/path), choice, "final-policy")
                same(read(root/path), choice, "durable-final-policy")
                final_choices[name] = choice
                final_policy_count += 1
            print(json.dumps({"phase": "selective-audit-policies", "variant": variant, "seed": seed}), flush=True)
        for target in targets:
            stacked = np.stack(verdicts[target])
            joint = np.full(len(rows), -1, np.int8)
            for side in (0, 1):
                joint[(stacked == side).all(axis=0)] = side
            name = f"{variant}-consensus-{target}-decisions"
            np.testing.assert_array_equal(published_arrays[name], joint)
            reconstructed[name] = joint
            expected = {"seeds": entries[target], "meanSeeds": means([item["evaluation"] for item in entries[target]]),
                "consensus": metrics(rows, joint),
                "infeasibleOuterPolicies": sum(not fold["feasible"] for item in entries[target] for fold in item["folds"]),
                "warmupOuterPolicies": sum(fold["warmupSelected"] for item in entries[target] for fold in item["folds"])}
            same(summary["variants"][variant][str(target)], expected, "aggregate-results")
    check(reconstructed.keys() == published_arrays.keys(), "Unexpected nested prediction arrays")
    raw = torch.from_numpy(windows).to(device)
    fg = torch.from_numpy(features.astype(np.float32)).to(device)
    max_reload = 0.
    for variant, seed, outer, epoch in sorted(selected_checkpoints):
        folder, meta = fit_meta[(variant, seed, outer)]
        indexes = np.asarray(meta["heldRows"])
        with np.load(folder/"scores.npz", allow_pickle=False) as saved:
            if variant == "selective":
                head = model_code.load_head(folder/f"weights-{epoch}.npz", device)
                actual = model_code.predict(head, raw[indexes], fg[indexes])
                expected = saved[f"{epoch}-held"]
            else:
                head = model_code.h.load_head(folder/f"weights-{epoch}.npz", "hybrid", device)
                base, residual = model_code.h.components(head, raw[indexes], fg[indexes])
                actual = np.column_stack((base, residual))
                expected = np.column_stack((saved[f"{epoch}-held-base"], saved[f"{epoch}-held-residual"]))
        error = float(np.max(np.abs(actual-expected)))
        check(error <= (0. if device.startswith("cuda") else 3e-5), "Selected checkpoint prediction differs")
        max_reload = max(max_reload, error)
    max_parity, delivery_count = 0., 0
    check(len(summary["finalSelectiveHeads"]) == len(seeds)*len(targets), "Final delivery count differs")
    for delivery in summary["finalSelectiveHeads"]:
        seed, target = delivery["seed"], delivery["target"]
        choice = final_choices[f"selective-{seed}-{target}"]
        for key in ("epoch", "farCutoff", "nearCutoff", "feasible"):
            same(delivery[key], choice[key], "delivery-policy")
        check(delivery["warmupOnly"] is False, "Warmup metadata differs")
        folder, meta = final_meta[seed]
        path = folder/f"weights-{choice['epoch']}.npz"
        check(sha(path) == delivery["weightsSha256"] and path.stat().st_size == delivery["bytes"], "Final weight metadata differs")
        cpu = model_code.load_head(path, "cpu")
        accelerator = model_code.load_head(path, device)
        cpu_out = model_code.predict(cpu, torch.from_numpy(windows[:16]), torch.from_numpy(features[:16].astype(np.float32)))
        accelerator_out = model_code.predict(accelerator, raw[:16], fg[:16])
        error = float(np.max(abs(persisted_scores(cpu_out)-persisted_scores(accelerator_out))))
        check(error < 2e-5, "Final CPU/accelerator parity failed")
        if device.startswith("cuda"):
            same(delivery["cpuGpuMaximumError"], error, "delivery-parity")
        max_parity = max(max_parity, error)
        delivery_count += 1
    check(fit_count == 108 and final_count == 3 and control_count == 37 and policy_count == 288
          and final_policy_count == 36, "Completed audit counts differ")
    return {"kind": "serving-side-selective-independent-audit-v1", "studyIndex": "private-reference-0236",
        "fastArtifactIndex": "private-reference-0237", "contractSha256": reg["sha256"],
        "auditorSha256": sha(__file__), "summarySha256": sha(report), "rows": len(rows),
        "nestedNeuralFitsVerified": fit_count, "previousBceFitsVerified": 108, "finalNeuralFitsVerified": final_count,
        "cachedLogisticBasesVerified": len(bases), "agreementPartitionsVerified": control_count,
        "fitArtifactHashesVerified": artifact_count, "maximumLogisticGradient": max_gradient,
        "outerPoliciesIndependentlySelected": policy_count, "finalPoliciesIndependentlySelected": final_policy_count,
        "selectedCheckpointsReloaded": len(selected_checkpoints), "maximumReloadLogitError": max_reload,
        "finalHeadsVerified": delivery_count, "maximumCpuAcceleratorProbabilityError": max_parity,
        "allAggregateMetricsReplayed": True, "allConsensusVerdictsReplayed": True,
        "publicPrivateSummaryEqual": True, "frozenFeaturesAndWindowsEqual": True,
        "outerLabelsExcludedFromSelection": True, "protectedLabelsOpened": False,
        "productionChanged": False, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    check(args.report is not None, "Supply the aggregate report")
    result = audit(args.report, args.device)
    root = Path(private_value("private-reference-0236"))
    model_code.immutable(root/"independent-audit.json", result)
    if args.output is not None:
        model_code.immutable(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
