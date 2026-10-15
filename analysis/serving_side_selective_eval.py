"""Frozen-policy nested evaluation, refitting, and delivery of selective heads."""
from __future__ import annotations

import itertools
from pathlib import Path
import shutil

import numpy as np
import torch

from . import serving_side_selective as s
from . import serving_side_selective_metrics as m
from .serving_side_evidence_agreement import fit_predict

VARIANTS = ("logistic", "bce-hybrid", "evidence-agreement", "selective")


def controls(fast, rows, features, exclusions, contract):
    cache = {}
    for excluded in exclusions:
        folder = fast / "agreement" / s.h.fold_name(excluded)
        if (folder / "models.json").exists():
            meta = s.read(folder / "models.json")
            s.require(meta["contractSha256"] == contract and meta["excludedGroups"] == sorted(excluded), "Control provenance differs")
            s.require(s.digest(folder / "scores.npz") == meta["scoresSha256"], "Control scores changed")
            with np.load(folder / "scores.npz", allow_pickle=False) as data:
                scores = data["scores"]; family = data["familyProbabilities"]
        else:
            result = fit_predict(rows, features, excluded)
            folder.mkdir(parents=True, exist_ok=True)
            scores, family = result.pop("scores"), result.pop("familyProbabilities")
            np.savez_compressed(folder / "scores.npz", scores=scores, familyProbabilities=family)
            meta = {**result, "excludedGroups": sorted(excluded), "contractSha256": contract,
                    "scoresSha256": s.digest(folder / "scores.npz")}
            s.immutable(folder / "models.json", meta)
        cache[("evidence-agreement", s.h.fold_name(excluded))] = (np.array(meta["heldRows"]), {0: scores})
        cache[("logistic", s.h.fold_name(excluded))] = (np.array(meta["heldRows"]), {0: np.column_stack((family[:, 0], np.ones(len(family))))})
    return cache


def consensus(predictions):
    values = np.stack(predictions)
    return np.where(np.all(values == values[0], axis=0) & (values[0] >= 0), values[0], -1).astype(np.int8)


def seed_summary(evaluations):
    keys = ("precision", "nearPrecision", "farPrecision", "coverage", "reviewFraction", "accepted",
            "reviewed", "correct", "errors", "silentErrorFraction", "correctAutoFraction")
    result = {}
    for key in keys:
        values = [e["pooled"][key] for e in evaluations if e["pooled"][key] is not None]
        result[key] = float(np.mean(values)) if values else None
    result["sourceGroupMacroCoverage"] = float(np.mean([e["sourceGroupMacroCoverage"] for e in evaluations]))
    supported = [e["sourceGroupMacroPrecision"] for e in evaluations if e["sourceGroupMacroPrecision"] is not None]
    result["sourceGroupMacroPrecision"] = float(np.mean(supported)) if supported else None
    result["seedsWithAcceptedPredictions"] = sum(e["pooled"]["accepted"] > 0 for e in evaluations)
    return result


def evaluate_and_refit(report, device="cuda"):
    s.h.prior.configure_torch()
    root, fast, reg, rows, windows, features = s.load()
    groups, seeds = reg["contract"]["sourceGroups"], reg["contract"]["seeds"]
    exclusions = list(itertools.combinations(groups, 2)) + [(g,) for g in groups]
    control = controls(fast, rows, features, exclusions, reg["sha256"])
    cached = {}
    def get(variant, seed, excluded):
        key = variant, seed, s.h.fold_name(excluded)
        if key not in cached:
            if variant in ("logistic", "evidence-agreement"):
                cached[key] = control[(variant, key[-1])]
            else:
                cached[key] = s.frozen_scores(fast, variant, seed, excluded, reg)
            ix, values = cached[key]
            expected = np.array([i for i, row in enumerate(rows) if row["sourceGroup"] in excluded])
            s.require(np.array_equal(ix, expected) and all(v.shape == (len(ix), 2) for v in values.values()), "Held score alignment differs")
        return cached[key]
    arrays, results, final_choices = {}, {}, {}
    for variant in VARIANTS:
        per_target = {str(target): [] for target in s.TARGETS}
        decisions_by_target = {str(target): [] for target in s.TARGETS}
        for seed in seeds:
            predictions = {target: np.full(len(rows), -2, np.int8) for target in s.TARGETS}
            chosen_scores = {target: np.full((len(rows), 2), np.nan) for target in s.TARGETS}
            folds = {target: [] for target in s.TARGETS}
            final_candidates = None
            for outer in groups:
                train, held = s.h.prior.fold_indices(rows, (outer,))
                local = {v: i for i, v in enumerate(train)}
                _, first = get(variant, seed, (outer,))
                candidates = {e: np.full((len(train), 2), np.nan) for e in first}
                for inner in groups:
                    if inner == outer:
                        continue
                    ix, values = get(variant, seed, tuple(sorted((outer, inner))))
                    keep = np.array([rows[i]["sourceGroup"] == inner for i in ix])
                    positions = np.array([local[i] for i in ix[keep]])
                    for epoch in candidates:
                        candidates[epoch][positions] = values[epoch][keep]
                s.require(all(np.isfinite(v).all() for v in candidates.values()), "Missing inner held predictions")
                indexes, outer_scores = get(variant, seed, (outer,))
                s.require(np.array_equal(indexes, held), "Outer score order differs")
                if final_candidates is None:
                    final_candidates = {e: np.full((len(rows), 2), np.nan) for e in outer_scores}
                for epoch in final_candidates:
                    final_candidates[epoch][held] = outer_scores[epoch]
                for target in s.TARGETS:
                    choice = m.select([rows[i] for i in train], candidates, target)
                    outputs = outer_scores[choice["epoch"]]
                    predictions[target][held] = m.decisions(outputs, choice["farCutoff"], choice["nearCutoff"])
                    chosen_scores[target][held] = outputs
                    s.immutable(fast / "selection" / variant / str(seed) / str(target) / (outer + ".json"),
                        {"contractSha256": reg["sha256"], "outerExcludedGroup": outer,
                         "innerRows": train.tolist(), "selected": choice})
                    folds[target].append({"sourceGroup": outer, "epoch": choice["epoch"],
                        "farCutoff": choice["farCutoff"], "nearCutoff": choice["nearCutoff"],
                        "feasible": choice["feasible"], "warmupSelected": variant == "selective" and choice["epoch"] <= s.ACTIVE_WARMUP})
            for target in s.TARGETS:
                s.require(np.isfinite(chosen_scores[target]).all() and np.all(predictions[target] >= -1), "Missing outer decision")
                ev = m.evaluate(rows, predictions[target], chosen_scores[target][:, 0])
                per_target[str(target)].append({"seed": seed, "evaluation": ev, "folds": folds[target]})
                decisions_by_target[str(target)].append(predictions[target])
                arrays[f"{variant}-{seed}-{target}-decisions"] = predictions[target]
                arrays[f"{variant}-{seed}-{target}-scores"] = chosen_scores[target]
                final_choice = m.select(rows, final_candidates, target)
                final_choices[f"{variant}-{seed}-{target}"] = final_choice
                s.immutable(fast / "final-selection" / variant / str(seed) / f"{target}.json", final_choice)
            print({"phase": "selective-evaluation", "variant": variant, "seed": seed}, flush=True)
        results[variant] = {}
        for target in s.TARGETS:
            entries = per_target[str(target)]
            joint = consensus(decisions_by_target[str(target)])
            arrays[f"{variant}-consensus-{target}-decisions"] = joint
            results[variant][str(target)] = {"seeds": entries,
                "meanSeeds": seed_summary([entry["evaluation"] for entry in entries]),
                "consensus": m.evaluate(rows, joint),
                "infeasibleOuterPolicies": sum(not f["feasible"] for entry in entries for f in entry["folds"]),
                "warmupOuterPolicies": sum(f["warmupSelected"] for entry in entries for f in entry["folds"])}
    np.savez_compressed(fast / "nested-predictions.npz", **arrays)
    # Refit only after all outer predictions and final policies have been fixed.
    raw = torch.from_numpy(windows).to(device); fg = torch.from_numpy(features.astype(np.float32)).to(device)
    delivery = []
    for seed in seeds:
        folder = fast / "final" / str(seed) / "fit"
        s.fit(fast, rows, raw, features, fg, (), seed, reg["sha256"],
              reg["contract"]["priorContractSha256"], folder=folder)
        for target in s.TARGETS:
            choice = final_choices[f"selective-{seed}-{target}"]
            path = folder / f"weights-{choice['epoch']}.npz"
            gpu = s.load_head(path, device); cpu = s.load_head(path, "cpu")
            cpu_out = s.predict(cpu, torch.from_numpy(windows[:16]), torch.from_numpy(features[:16].astype(np.float32)))
            gpu_out = s.predict(gpu, raw[:16], fg[:16])
            error = float(np.max(abs(s.scores(cpu_out) - s.scores(gpu_out))))
            s.require(error < 2e-5, "Final CPU/GPU parity failed")
            delivery.append({"seed": seed, "target": target, "epoch": choice["epoch"],
                "farCutoff": choice["farCutoff"], "nearCutoff": choice["nearCutoff"],
                "feasible": choice["feasible"], "warmupOnly": choice["epoch"] <= s.ACTIVE_WARMUP,
                "weightsSha256": s.digest(path), "bytes": path.stat().st_size, "cpuGpuMaximumError": error})
    controls(fast, rows, features, [()], reg["sha256"])
    for name in ("fits", "agreement", "selection", "final-selection", "final"):
        shutil.copytree(fast / name, root / name, dirs_exist_ok=True)
    shutil.copyfile(fast / "nested-predictions.npz", root / "nested-predictions.npz")
    result = {"kind": "serving-side-selective-results-v1", "studyIndex": s.STUDY_INDEX,
        "fastArtifactIndex": s.FAST_INDEX, "contractSha256": reg["sha256"],
        "scope": {"serves": len(rows), "recordings": len({r["recordingIndex"] for r in rows}),
                  "sourceGroups": len(groups)}, "primaryTarget": .98, "targets": list(s.TARGETS),
        "variants": results, "finalSelectiveHeads": delivery,
        "predictionsSha256": s.digest(root / "nested-predictions.npz"),
        "newNestedNeuralFits": 108, "finalNeuralFits": 3, "newAgreementFittingPartitions": 37,
        "protectedLabelsOpened": False, "productionChanged": False,
        "precisionIsEmpiricalTargetNotGuarantee": True,
        "allRowsAreReviewedTrueServes": True,
        "consensusPolicy": "all three seed policies accept and agree; otherwise review"}
    s.immutable(root / "summary.json", result)
    s.immutable(Path(report), result)
    print({"phase": "selective-complete", "studyIndex": s.STUDY_INDEX}, flush=True)
    return result
