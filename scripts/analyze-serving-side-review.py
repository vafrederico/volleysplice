#!/usr/bin/env python3
"""Describe production review rules and frozen hybrid predictions on known serves.

This post-hoc analysis never fits a model or changes an operating point. All
candidate-level inputs and output stay in the external private artifact store.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    if path.exists():
        require(path.read_text() == content, "Preserve previous analysis output")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def lookup_selected(items, originals):
    wanted = {r["rallyId"] for r in originals}
    selected = [r for r in items if r["rallyId"] in wanted]
    result = {r["rallyId"]: r for r in selected}
    require(len(selected) == len(result) == len(originals), "Incomplete or duplicate identity join")
    for row in originals:
        require(result[row["rallyId"]]["recordingId"] == row["recordingId"], "Recording join differs")
    return [result[row["rallyId"]] for row in originals]


def metrics(y, prediction, mask):
    truth, predicted = y[mask], prediction[mask]
    n = len(truth)
    confusion = {
        str(label): {str(guess): int(((truth == label) & (predicted == guess)).sum())
                     for guess in (0, 1)} for label in (0, 1)
    }
    recalls = [float((predicted[truth == label] == label).mean())
               if np.any(truth == label) else None for label in (0, 1)]
    return {"correct": int((truth == predicted).sum()),
            "accuracy": float((truth == predicted).mean()) if n else None,
            "balancedAccuracy": float(np.mean(recalls)) if all(r is not None for r in recalls) else None,
            "confusionFar0Near1": confusion}


def paired(y, specialist, hybrid, mask):
    old, new = specialist == y, hybrid == y
    return {"bothCorrect": int((mask & old & new).sum()),
            "hybridFixes": int((mask & ~old & new).sum()),
            "hybridRegressions": int((mask & old & ~new).sum()),
            "bothWrong": int((mask & ~old & ~new).sum())}


NODE_REPLAY = r"""
import fs from 'node:fs';
import { parseServingSideRuntime, servingSideNearProbability, composeServingSideVerdict }
  from './prod/src/lib/on-device/serving-side-model.ts';
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const runtime = parseServingSideRuntime(input.runtime);
const zero = Array(runtime.featureNames.length).fill(0);
const call = (model, features, heads, agreement) => composeServingSideVerdict(
  {id: 'audit', start: 0, end: 1, confidence: 1, included: true, agreement}, features, [0],
  {probabilities: [heads[0]], detections: []},
  {probabilities: [heads[1]], detections: []}, model);
const constantModel = p => ({...runtime, model: {...runtime.model, weights: zero,
  bias: Math.log(p / (1-p))}});
const kinds = Object.fromEntries(Object.keys(input.probabilities).map(k => [k, []]));
const packaged = [];
for (let i = 0; i < input.features.length; i++) {
  packaged.push(servingSideNearProbability(input.features[i], runtime));
  for (const [kind, probabilities] of Object.entries(input.probabilities)) {
    const model = constantModel(probabilities[i]);
    const verdicts = {};
    for (const agreement of ['all-labels-v2-only', 'both-models', 'neural']) {
      const v = call(model, zero, input.heads[i], agreement);
      verdicts[agreement] = {verdict: v.verdict, side: v.side, source: v.serveDecisionSource,
        sideReview: v.reviewReasons.includes('side-score')};
    }
    kinds[kind].push(verdicts);
  }
}
// Exercise exact inclusion/exclusion at an actually representable score.
const p = servingSideNearProbability(zero, constantModel(.4));
const boundaryModel = constantModel(.4);
const lower = call({...boundaryModel, reviewBand: {farUpperExclusive:p, nearLowerInclusive:p+.1}},
  zero, [.9, .1], 'both-models');
const upper = call({...boundaryModel, reviewBand: {farUpperExclusive:p-.1, nearLowerInclusive:p}},
  zero, [.9, .1], 'both-models');
if (lower.verdict !== 'review' || upper.verdict === 'review') throw Error('Band boundary contract differs');
const at = call(runtime, input.features[0], [.85, .1], 'neural');
const below = call(runtime, input.features[0], [.85-1e-10, .1], 'neural');
if (at.serveDecisionSource !== 'serve-head' || below.serveDecisionSource !== 'neural-rally-recovery')
  throw Error('Serve gate boundary contract differs');
process.stdout.write(JSON.stringify({packaged, kinds, boundaryChecks:4}));
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    study = Path(private_value("private-reference-0231"))
    reg = read(study / "registration.json")
    contract = reg["contract"]
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    require(hashlib.sha256(canonical.encode()).hexdigest() == reg["sha256"], "Hybrid contract changed")
    audit = read(study / "independent-audit.json")
    require(audit["status"] == "pass" and audit["contractSha256"] == reg["sha256"], "Hybrid audit missing")
    for name in ("rows", "features", "names"):
        ref = contract["data"][name]
        require(digest(ref["path"]) == ref["sha256"], "Frozen hybrid input changed")
    rows = read(contract["data"]["rows"]["path"])
    with np.load(contract["data"]["features"]["path"], allow_pickle=False) as data:
        features = data["values"]
    paths = {"dataset": Path(private_value("private-reference-0226")),
             "specialistHeld": Path(private_value("private-reference-0227")),
             "gate": Path(private_value("private-reference-0232")),
             "productionReplay": Path(private_value("private-reference-0233")),
             "hybridPredictions": study / "nested-predictions.npz",
             "runtime": REPO / "prod/public/runtime/serving-side-85bc3325fbd4.json"}
    originals = [r for r in read(paths["dataset"])["rows"] if r["environment"] != "beach"]
    ledger_path = os.environ.get("VOLLEYCUT_PRIVATE_LEDGER_" + ("WINDOWS" if os.name == "nt" else "POSIX")) \
        or os.environ.get("VOLLEYCUT_PRIVATE_LEDGER")
    aliases = read(ledger_path)["originalToAlias"]
    require(len(originals) == len(rows) == 968 and features.shape == (968, 237), "Study population changed")
    for i, (raw, row) in enumerate(zip(originals, rows, strict=True)):
        require(raw["decision"] in ("near", "far") and raw["sourceSplit"] != "test", "Unqualified study row")
        require(row["rowIndex"] == i and raw["label"] == row["label"]
                and aliases[raw["recordingId"]] == row["recordingIndex"]
                and aliases[raw["sourceGroup"]] == row["sourceGroup"], "Hybrid identity differs")
    baseline = read(paths["specialistHeld"])
    held = lookup_selected(baseline["selectedPredictions"], originals)
    gate_data = read(paths["gate"])
    gates = lookup_selected(gate_data["rows"], originals)
    replay = read(paths["productionReplay"])
    saved = lookup_selected(replay["predictions"], originals)
    runtime = read(paths["runtime"])
    require(runtime["fingerprint"] == replay["modelFingerprint"], "Production fingerprint differs")
    require(runtime["featureNames"] == read(contract["data"]["names"]["path"]), "Feature order differs")
    require(replay["sources"]["serveGateEvidence"]["sha256"] == digest(paths["gate"]), "Gate evidence changed")
    for original, old, gate, deployed in zip(originals, held, gates, saved, strict=True):
        require(old["decision"] == deployed["decision"] == original["decision"], "Reviewed label differs")
        require(abs(original["serveAnchor"] - gate["serveAnchor"]) < 1e-9, "Anchor join differs")
        require(deployed["evaluationRole"] == "development-in-sample", "Unexpected production replay scope")
    for name, model in gate_data["models"].items():
        require(model["threshold"] == runtime["gate"]["serveHeadThreshold"], "Serve threshold differs")
        path = REPO / "prod/public/runtime" / (model["modelId"] + ".json")
        require(digest(path) == model["bundleSha256"], "Production serve head differs")
    require(gate_data["gate"]["anchorWindowSeconds"] == runtime["gate"]["serveHeadWindowSeconds"], "Gate window differs")
    head_names = ("allLabelsV2", "previousProduction")
    head_scores = np.array([[g["heads"][name]["peakProbability"] for name in head_names] for g in gates])
    head_pass = np.any(head_scores >= runtime["gate"]["serveHeadThreshold"], axis=1)
    recovery = np.array([g["decisionSource"] == "production-rally-recovery" for g in gates])
    for i, gate in enumerate(gates):
        expected = "serve-head" if head_pass[i] else "production-rally-recovery" \
            if gate["productionRally"]["anchorContained"] and gate["productionRally"]["bothModels"] else "none"
        require(gate["decisionSource"] == expected == saved[i]["serveDecisionSource"], "Historical gate replay differs")
    model = runtime["model"]
    logits = ((features - np.array(model["mean"])) / np.array(model["scale"])) @ np.array(model["weights"]) + model["bias"]
    probabilities = {"historical-source-held": np.array([r["probabilityNear"] for r in held]),
                     "deployed-historical-replay": np.array([r["nearProbability"] for r in saved]),
                     "deployed-on-study-features": 1 / (1 + np.exp(-np.clip(logits, -30, 30)))}
    lo, hi = runtime["reviewBand"]["farUpperExclusive"], runtime["reviewBand"]["nearLowerInclusive"]
    y = np.array([r["label"] for r in rows])
    seeds = contract["seeds"]
    hybrid = {}
    with np.load(paths["hybridPredictions"], allow_pickle=False) as data:
        for variant in ("hybrid-selected", "hybrid-full"):
            p = np.stack([data[f"{variant}-{seed}-probabilities"] for seed in seeds])
            t = np.stack([data[f"{variant}-{seed}-thresholds"] for seed in seeds])
            require(p.shape == t.shape == (3, len(rows)) and np.isfinite(p).all()
                    and np.isfinite(t).all(), "Invalid hybrid prediction array")
            hybrid[variant] = {str(seed): p[i] >= t[i] for i, seed in enumerate(seeds)}
            hybrid[variant]["ensemble"] = p.mean(axis=0) >= t.mean(axis=0)
    js = subprocess.run([args.node, "--input-type=module", "-e", NODE_REPLAY], cwd=REPO,
                        input=json.dumps({"runtime": runtime, "features": features.tolist(),
                            "heads": head_scores.tolist(),
                            "probabilities": {k: v.tolist() for k, v in probabilities.items()}}),
                        text=True, capture_output=True, check=True)
    replayed = json.loads(js.stdout)
    error = float(np.max(abs(np.array(replayed["packaged"]) - probabilities["deployed-on-study-features"])))
    require(error < 1e-12, "Production TypeScript probability parity failed")
    scenarios, details = {}, []
    for kind, p in probabilities.items():
        band = (p >= lo) & (p < hi)
        specialist = p >= runtime["sideThreshold"]
        for i, item in enumerate(replayed["kinds"][kind]):
            for agreement, actual in item.items():
                rescued = not head_pass[i] and agreement in ("both-models", "neural")
                expected = "not-serve" if not head_pass[i] and not rescued else "review" \
                    if band[i] or rescued else "near" if specialist[i] else "far"
                require(actual["verdict"] == expected and actual["sideReview"] == bool(band[i])
                        and actual["side"] == ("near" if specialist[i] else "far"), "Production verdict parity failed")
        masks = {"side-score": band, "outside-side-band": ~band,
                 "historical-gate-review": recovery | (head_pass & band),
                 "historical-gate-not-serve": ~(head_pass | recovery),
                 "neural-candidate-review-counterfactual": ~head_pass | band,
                 "weak-serve-heads": ~head_pass,
                 "side-score-and-weak-heads": band & ~head_pass}
        def describe(mask):
            result = {"rows": int(mask.sum()), "fractionOfAllServes": float(mask.mean()),
                      "near": int(y[mask].sum()), "far": int(mask.sum() - y[mask].sum()),
                      "specialistSideGuess": metrics(y, specialist, mask), "hybrid": {}}
            for variant, predictions in hybrid.items():
                evaluated = {key: metrics(y, pred, mask) for key, pred in predictions.items()}
                result["hybrid"][variant] = {"predictions": evaluated,
                    "meanSeedAccuracy": float(np.mean([evaluated[str(s)]["accuracy"] for s in seeds])) if mask.any() else None,
                    "pairedEnsemble": paired(y, specialist, predictions["ensemble"], mask)}
            return result
        scenarios[kind] = {name: describe(mask) for name, mask in masks.items()}
        scenarios[kind]["side-score-by-source"] = {
            group: {"allServes": sum(r["sourceGroup"] == group for r in rows),
                    **describe(band & np.array([r["sourceGroup"] == group for r in rows]))}
            for group in sorted({r["sourceGroup"] for r in rows})}
        for i, row in enumerate(rows):
            details.append({**row, "scenario": kind, "specialistNearProbability": float(p[i]),
                            "masks": {name: bool(mask[i]) for name, mask in masks.items()},
                            "hybridPredictions": {v: {s: int(pred[i]) for s, pred in ps.items()} for v, ps in hybrid.items()}})
    band_change = ((probabilities["deployed-historical-replay"] >= lo) & (probabilities["deployed-historical-replay"] < hi)) \
        != ((probabilities["deployed-on-study-features"] >= lo) & (probabilities["deployed-on-study-features"] < hi))
    result = {"kind": "serving-side-review-posthoc-v1", "studyIndex": "private-reference-0231",
              "gateInputIndex": "private-reference-0232", "productionReplayIndex": "private-reference-0233",
              "hybridContractSha256": reg["sha256"], "codeSha256": digest(__file__),
              "inputSha256": {k: digest(v) for k, v in paths.items()},
              "productionPolicySha256": digest(REPO / "prod/src/lib/on-device/serving-side-model.ts"),
              "scope": {"serves": len(rows), "recordings": len({r["recordingIndex"] for r in rows}),
                        "sourceGroups": len({r["sourceGroup"] for r in rows}), "near": int(y.sum()),
                        "far": int(len(y) - y.sum()), "allAreReviewedTrueServes": True},
              "policy": {"nearThreshold": runtime["sideThreshold"], "reviewLowerInclusive": lo,
                         "reviewUpperExclusive": hi, "serveHeadThreshold": runtime["gate"]["serveHeadThreshold"],
                         "serveHeadWindowSeconds": runtime["gate"]["serveHeadWindowSeconds"]},
              "gates": {"serveHeadPasses": int(head_pass.sum()), "historicalRallyRecoveries": int(recovery.sum()),
                        "historicalNotServe": int((~(head_pass | recovery)).sum())},
              "scenarios": scenarios,
              "productionFeaturePopulationSensitivity": {
                  "historicalVersusStudyMaximumProbabilityDifference": float(abs(probabilities["deployed-historical-replay"] - probabilities["deployed-on-study-features"]).max()),
                  "reviewBandMembershipChanges": int(band_change.sum())},
              "verification": {"identityJoins": len(rows), "productionTypeScriptProbabilityMaximumError": error,
                               "productionVerdictsReplayed": 3 * 3 * len(rows), "boundaryChecks": replayed["boundaryChecks"]},
              "interpretation": [
                  "Primary comparative evidence uses source-held predictions; the historical specialist threshold was development-selected.",
                  "Deployed weights were fitted on this development corpus; their review masks are descriptive and label-exposed.",
                  "Hybrid predictions always hold out the entire source group; all thresholds, epochs, and mixtures remain frozen.",
                  "Historical gate review uses recorded both-model rally evidence at human-reviewed serve anchors.",
                  "Neural-candidate review is conditional on every anchor being emitted by a neural rally model; detection was not rerun.",
                  "No population here measures actual production traffic, false serve detections, missed rallies, or human-unclear labels.",
                  "Side correctness cannot validate whether a weak-head candidate is a real serve.",
                  "Historical deployed replay used its historical candidate ranks; the hybrid retains its registered study feature population.",
                  "This is a post-hoc descriptive slice, not a model-selection or promotion result."],
              "protectedRowsScored": 0, "modelsRefitted": False, "productionChanged": False}
    out = study / "review-analysis"
    write_new(out / "rows.json", details)
    write_new(out / "summary.json", result)
    write_new(args.report, result)
    print(json.dumps({"status": "pass", "rows": len(rows), "verification": result["verification"]}))


if __name__ == "__main__":
    main()
