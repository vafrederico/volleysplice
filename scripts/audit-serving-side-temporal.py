#!/usr/bin/env python3
"""Independently replay temporal serving-side selection and reported metrics."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from analysis.serving_side_temporal import arm_values, configure_torch, load_weights, predict


def read(path):
    return json.loads(Path(path).read_text())


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def replay_metrics(rows, predicted):
    """Calculate confusion matrices directly, without importing study metrics."""
    truth = np.asarray([r["label"] for r in rows], np.int64)
    prediction = np.asarray(predicted, np.int64)
    def one(indexes):
        confusion = np.bincount(2*truth[indexes]+prediction[indexes], minlength=4)
        tn, fp, fn, tp = map(int, confusion)
        def divide(a, b):
            return a/b if b else None
        nr, fr = divide(tp, tp+fn), divide(tn, tn+fp)
        return {"rows": len(indexes), "truePositive": tp, "trueNegative": tn,
                "falsePositive": fp, "falseNegative": fn, "accuracy": divide(tp+tn, len(indexes)),
                "balancedAccuracy": None if nr is None or fr is None else .5*(nr+fr),
                "macroF1": .5*((divide(2*tp, 2*tp+fp+fn) or 0.)+(divide(2*tn, 2*tn+fp+fn) or 0.)),
                "nearPrecision": divide(tp, tp+fp), "nearRecall": nr,
                "farPrecision": divide(tn, tn+fn), "farRecall": fr}
    groups, environments = {}, {}
    for key, output in (("sourceGroup", groups), ("environment", environments)):
        for group in sorted({r[key] for r in rows}):
            indexes = np.asarray([i for i, row in enumerate(rows) if row[key] == group])
            output[group] = one(indexes)
    ba = [v["balancedAccuracy"] for v in groups.values()]
    check(None not in ba, "Incomplete class support")
    return {"pooled": one(np.arange(len(rows))), "bySourceGroup": groups, "byEnvironment": environments,
            "sourceGroupMacroBalancedAccuracy": float(np.mean(ba)), "worstSourceGroupBalancedAccuracy": min(ba)}


def same(actual, expected):
    if isinstance(expected, dict):
        check(actual.keys() == expected.keys(), "Metric fields differ")
        for key in expected:
            same(actual[key], expected[key])
    elif expected is None:
        check(actual is None, "Unexpected metric value")
    elif isinstance(expected, (float, int)):
        check(abs(actual-expected) <= 1e-12, "Metric replay differs")
    else:
        check(actual == expected, "Value differs")


def choose(rows, scores, epochs, thresholds):
    winner = None
    for epoch in epochs:
        check(np.isfinite(scores[epoch]).all(), "Missing selection score")
        for threshold in thresholds:
            evaluation = replay_metrics(rows, scores[epoch] >= threshold)
            rank = (evaluation["sourceGroupMacroBalancedAccuracy"], evaluation["pooled"]["balancedAccuracy"],
                    evaluation["pooled"]["macroF1"], evaluation["worstSourceGroupBalancedAccuracy"],
                    -epoch, -abs(threshold-.5), -threshold)
            if winner is None or rank > winner[0]:
                winner = (rank, epoch, threshold)
    return winner[1:]


def replay_windows(contract, rows, raw, ledger):
    """Brute-force nearest frames independently of the prepared-window sampler."""
    source = read(contract["inputs"]["dataset"]["path"])
    original = [r for r in source["rows"] if r["environment"] != "beach"]
    catalog = {r["id"]: r for r in read(contract["inputs"]["catalog"]["path"])["records"]}
    quality = {r["recordingId"]: r["intervals"] for r in read(contract["inputs"]["quality"]["path"])["records"]}
    offsets = np.asarray(contract["offsetSeconds"])
    aliases = ledger["originalToAlias"]
    checked = 0
    for recording in sorted({r["recordingId"] for r in original}):
        alias = aliases[recording]
        with np.load(contract["receipts"][alias]["features"]["path"], allow_pickle=False) as archive:
            times, tokens = archive["timestamps"].copy(), archive["tokens"].astype(np.float32)
        record = catalog[recording]
        spans = record.get("ignoredIntervals", []) + quality.get(recording, [])
        for index, row in enumerate(original):
            if row["recordingId"] != recording:
                continue
            check(rows[index]["recordingIndex"] == alias and rows[index]["label"] == row["label"]
                  and rows[index]["sourceGroup"] == aliases[row["sourceGroup"]], "Reviewed row order/identity differs")
            anchor = row["serveAnchor"]
            left, right = 0., record["durationSeconds"]
            for span in spans:
                start, end = span["start"], span["end"]
                check(not start <= anchor < end, "Anchor in ignored footage")
                if end <= anchor:
                    left = max(left, end)
                if start > anchor:
                    right = min(right, start)
            available_indexes = np.flatnonzero((times >= left)&(times < right))
            expected = np.zeros((13, 3843), np.float32)
            expected[:, 3841] = offsets/3.
            for tick, offset in enumerate(offsets):
                requested = anchor+offset
                if not left <= requested < right or not len(available_indexes):
                    continue
                chosen = available_indexes[np.argmin(abs(times[available_indexes]-requested))]
                if abs(times[chosen]-requested) > .250001:
                    continue
                expected[tick, :3840] = tokens[chosen].reshape(-1)
                expected[tick, 3840] = 1.
                expected[tick, 3842] = (times[chosen]-requested)/.25
            np.testing.assert_array_equal(expected, raw[index])
            checked += 1
    check(checked == len(rows), "Incomplete input-window replay")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.root
    reg, summary = read(root/"registration.json"), read(args.report)
    contract = reg["contract"]
    canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    check(canonical == reg["sha256"] == summary["contractSha256"], "Contract hash differs")
    rows = read(contract["data"]["rows"]["path"])
    groups, epochs, seeds, arms = (contract[k] for k in ("sourceGroups", "epochs", "seeds", "arms"))
    check(len(rows) == 968 and len(groups) == 8 and len(seeds) == 3, "Scope differs")
    for ref in list(contract["data"].values()) + list(contract["inputs"].values()) + list(contract["sourceCode"].values()):
        check(sha(ref["path"]) == ref["sha256"], "Input/source hash differs")
    for receipt in contract["receipts"].values():
        for ref in receipt.values():
            check(sha(ref["path"]) == ref["sha256"], "Embedding receipt or bytes differ")
    with np.load(contract["data"]["windows"]["path"], allow_pickle=False) as archive:
        raw = archive["values"].copy()
    window_count = replay_windows(contract, rows, raw, read(args.ledger))
    configure_torch()
    fit_count = 0
    for meta_path in (root/"fits").glob("*/*/*/completed.json"):
        meta = read(meta_path)
        train, held = meta["trainingRows"], meta["heldRows"]
        excluded = set(meta["excludedGroups"])
        check(set(train).isdisjoint(held) and set(train+held) == set(range(len(rows))), "Fold rows overlap or are missing")
        check({rows[i]["sourceGroup"] for i in train}.isdisjoint(excluded)
              and {rows[i]["sourceGroup"] for i in held} == excluded, "Source isolation fails")
        check(meta["contractSha256"] == reg["sha256"] and meta["epochs"] == epochs, "Fit identity differs")
        for name, digest in meta["artifacts"].items():
            check(sha(meta_path.parent/name) == digest, "Weights or scores changed")
        fit_count += 1
    check(fit_count == 216, "Incomplete nested fits")
    def scores(arm, seed, excluded):
        folder = root/"fits"/arm/str(seed)/"+".join(sorted(excluded))
        metadata = read(folder/"completed.json")
        with np.load(folder/"scores.npz", allow_pickle=False) as archive:
            return metadata["heldRows"], {e: archive[str(e)].copy() for e in epochs}
    with np.load(root/"nested-predictions.npz", allow_pickle=False) as archive:
        nested = {k: archive[k].copy() for k in archive.files}
    parity_max_error, selected_folds = 0., 0
    same(replay_metrics(rows, [r["baselineNear"] for r in rows]), summary["baseline"])
    for arm in arms:
        values = torch.from_numpy(arm_values(raw, arm)).to(args.device)
        all_p, all_t = [], []
        for seed in seeds:
            probabilities = np.full(len(rows), np.nan)
            thresholds = np.full(len(rows), np.nan)
            for outer in groups:
                inner_indexes = [i for i, row in enumerate(rows) if row["sourceGroup"] != outer]
                inner_rows = [rows[i] for i in inner_indexes]
                local = {i: j for j, i in enumerate(inner_indexes)}
                inner_scores = {e: np.full(len(inner_rows), np.nan) for e in epochs}
                for inner in groups:
                    if inner == outer:
                        continue
                    indexes, pair = scores(arm, seed, (outer, inner))
                    for e in epochs:
                        for i, probability in zip(indexes, pair[e], strict=True):
                            if rows[i]["sourceGroup"] == inner:
                                inner_scores[e][local[i]] = probability
                epoch, threshold = choose(inner_rows, inner_scores, epochs, contract["thresholds"])
                selection = read(root/"selection"/arm/str(seed)/(outer+".json"))
                check(selection["innerRows"] == inner_indexes and outer not in {r["sourceGroup"] for r in inner_rows}, "Selection includes outer labels")
                check((epoch, threshold) == (selection["selected"]["epoch"], selection["selected"]["threshold"]), "Independent checkpoint/threshold selection differs")
                indexes, held_scores = scores(arm, seed, (outer,))
                probabilities[indexes] = held_scores[epoch]
                thresholds[indexes] = threshold
                weights = root/"fits"/arm/str(seed)/outer/f"weights-{epoch}.npz"
                replayed = predict(load_weights(weights, args.device), values[indexes])
                error = float(np.max(np.abs(replayed-held_scores[epoch])))
                parity_max_error = max(parity_max_error, error)
                check(error == 0., "Restored checkpoint prediction differs from training-time scores")
                selected_folds += 1
            np.testing.assert_array_equal(probabilities, nested[f"{arm}-{seed}-probabilities"])
            np.testing.assert_array_equal(thresholds, nested[f"{arm}-{seed}-thresholds"])
            expected = next(s for s in summary["arms"][arm]["seeds"] if s["seed"] == seed)
            same(replay_metrics(rows, probabilities >= thresholds), expected["evaluation"])
            all_p.append(probabilities)
            all_t.append(thresholds)
        same(replay_metrics(rows, np.mean(all_p, axis=0) >= np.mean(all_t, axis=0)), summary["arms"][arm]["ensemble"])
        del values
    final_parity_error = 0.
    winner = summary["selectedArm"]
    historical = read(Path(__file__).resolve().parents[1]/summary["historicalLinearControl"]["source"])
    check(abs(historical["serving"]["head"]["balancedAccuracy"]
              - summary["historicalLinearControl"]["pooledBalancedAccuracy"]) <= 1e-12
          and abs(historical["serving"]["groupMacroBalancedAccuracy"]
                  - summary["historicalLinearControl"]["sourceGroupMacroBalancedAccuracy"]) <= 1e-12,
          "Historical control differs from its saved report")
    for final in summary["finalHeads"]:
        seed = final["seed"]
        full_oof = {e: np.full(len(rows), np.nan) for e in epochs}
        for group in groups:
            indexes, s = scores(winner, seed, (group,))
            for e in epochs:
                full_oof[e][indexes] = s[e]
        check(choose(rows, full_oof, epochs, contract["thresholds"]) == (final["epoch"], final["threshold"]), "Final selection differs")
        weights = root/"final"/str(seed)/"fit"/f"weights-{final['epoch']}.npz"
        check(sha(weights) == final["weightsSha256"], "Final weights changed")
        probe = torch.from_numpy(arm_values(raw[:16], winner))
        cpu = predict(load_weights(weights, "cpu"), probe)
        gpu = predict(load_weights(weights, args.device), probe.to(args.device))
        error = float(np.max(np.abs(cpu-gpu)))
        final_parity_error = max(final_parity_error, error)
        check(error <= 1e-5, "CPU/GPU final-head parity differs")
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("privacy", repo/"scripts/audit-branch-privacy.py")
    privacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(privacy)
    ledger = read(args.ledger)
    identities = {k for k in ledger.get("originalToAlias", {}) if len(k) > 8}
    identities.update(ledger.get("denyTokens", []))
    identities.update(v for v in ledger.get("privateValues", {}).values() if len(v) > 12 and ("/" in v or "\\" in v))
    paths = [repo/name for name in contract["sourceCode"]]
    paths.extend([Path(__file__), args.report, repo/"analysis/tests/test_serving_side_temporal.py",
                  repo/"docs/research/serving-side-temporal-head-2026-10-13.md"])
    findings = []
    for path in paths:
        kinds = privacy.categories(path.read_text(), identities, str(path.relative_to(repo)))
        if kinds:
            findings.append({"file": str(path.relative_to(repo)), "categories": kinds})
    check(not findings, "Publication privacy check failed")
    result = {"status": "pass", "contractSha256": reg["sha256"], "nestedFitsAudited": fit_count,
              "inputWindowsIndependentlyReplayed": window_count,
              "auditCodeSha256": sha(Path(__file__)),
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__},
              "outerSelectionsReplayed": selected_folds, "heldPredictionReloadMaximumError": parity_max_error,
              "finalHeadCpuGpuMaximumError": final_parity_error, "publicFilesPrivacyChecked": len(paths),
              "protectedLabelsOpened": False, "sourceIsolation": "pass", "metricReplay": "pass"}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
