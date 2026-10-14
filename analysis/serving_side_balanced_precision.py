"""Matched serving-side precision evaluation with the Balanced frozen encoder."""
from __future__ import annotations

import itertools
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from . import serving_side_precision_iterations as base
from . import serving_side_temporal as temporal

STUDY = "private-reference-0242"
FAST = "private-reference-0243"
ENCODER = "private-reference-0244"
BANK = "private-reference-0245"
ARM = "09-neural-centered"
PROTOCOL = "docs/research/serving-side-balanced-precision-2026-10-14.md"
MANIFEST = "prod/public/runtime/rally-models/manifest.json"
CODE = ("analysis/serving_side_balanced_precision.py",
        "scripts/train-serving-side-balanced-precision.py", *base.CODE)
read, require, verified = base.read, base.require, base.verified
reference, immutable, canonical, digest = base.reference, base.immutable, base.canonical, base.digest


def prepare(repo):
    repo = Path(repo)
    root, fast = (Path(base.private_value(k)) for k in (STUDY, FAST))
    require(not root.exists() and not fast.exists(), "Preserve existing Balanced artifacts")
    prior_root, _, previous, rows, old_windows, features = base.load()
    original_path = Path(base.private_value("private-reference-0225"))/"registration.json"
    original = read(original_path)
    require(canonical(original["contract"]) == original["sha256"], "Original registration changed")
    inputs = original["contract"]["inputs"]
    for ref in inputs.values():
        verified(ref)
    dataset = read(inputs["dataset"]["path"])
    raw_rows = [r for r in dataset["rows"] if r["environment"] != "beach"]
    catalog = {r["id"]: r for r in read(inputs["catalog"]["path"])["records"]}
    ignored = {r["recordingId"]: r["intervals"] for r in read(inputs["quality"]["path"])["records"]}
    manifest = read(repo/MANIFEST)
    encoder, bank = (Path(base.private_value(k)) for k in (ENCODER, BANK))
    encoder_hash = manifest["variants"]["high-f1"]["encoderWeightsSha256"]
    require(digest(encoder) == encoder_hash and encoder_hash != original["contract"]["encoderSha256"],
            "Balanced encoder identity differs")
    receipts, sequences, aliases = {}, [], {}
    ledger = read(os.environ.get("VOLLEYCUT_PRIVATE_LEDGER_POSIX") or os.environ["VOLLEYCUT_PRIVATE_LEDGER"])
    aliases = ledger["originalToAlias"]
    require(len(raw_rows) == len(rows) == 968, "Frozen population differs")
    for r, prepared in zip(raw_rows, rows, strict=True):
        record = r["recordingId"]
        require(r["sourceSplit"] != "test" and not catalog[record].get("protected")
                and prepared["recordingIndex"] == aliases[record]
                and prepared["sourceGroup"] == aliases[r["sourceGroup"]]
                and prepared["label"] == r["label"], "Frozen row identity differs")
        if prepared["recordingIndex"] not in receipts:
            old = original["contract"]["receipts"][prepared["recordingIndex"]]
            old_receipt = read(verified(old["receipt"]))
            receipt_path = bank/(record+".json")
            receipt = read(receipt_path)
            require(receipt["id"] == record and receipt["sourceGroup"] == catalog[record]["sourceGroup"]
                    and receipt["labelsUsed"] is False and receipt["encoder"]["sha256"] == encoder_hash
                    and receipt["imageInput"] == old_receipt["imageInput"]
                    and receipt["recipe"] == old_receipt["recipe"], "Balanced bank lineage differs")
            feature_path = verified(receipt["output"])
            with np.load(feature_path, allow_pickle=False) as new, np.load(verified(old["features"]), allow_pickle=False) as old_data:
                require(new["tokens"].dtype == np.float16, "Expected FP16-rounded embeddings")
                for field in ("timestamps", "quality", "selected_presentation_times"):
                    np.testing.assert_array_equal(new[field], old_data[field])
            receipts[prepared["recordingIndex"]] = {"receipt": reference(receipt_path), "features": reference(feature_path)}
        # Load one recording at a time; never retain the whole embedding bank.
    for record in dict.fromkeys(r["recordingId"] for r in raw_rows):
        ref = receipts[aliases[record]]
        with np.load(ref["features"]["path"], allow_pickle=False) as data:
            times, tokens = data["timestamps"].copy(), data["tokens"].astype(np.float32)
        for i, r in enumerate(raw_rows):
            if r["recordingId"] != record:
                continue
            exclusions = catalog[record].get("ignoredIntervals", [])+ignored.get(record, [])
            sequences.append((i, temporal.sample_window(times, tokens, float(r["serveAnchor"]),
                              float(catalog[record]["durationSeconds"]), exclusions)))
    sequences.sort(key=lambda pair: pair[0])
    windows = np.stack([x for _, x in sequences])
    require(windows.shape == (968, 13, 3843) and np.isfinite(windows).all(), "Balanced windows differ")
    np.testing.assert_array_equal(windows[..., 3840:], old_windows[..., 3840:])
    require(not np.array_equal(windows[..., :3840], old_windows[..., :3840]), "Encoder replacement has no effect")
    root.mkdir(parents=True); fast.mkdir(parents=True)
    immutable(root/"rows.json", rows)
    np.save(fast/"windows.npy", windows, allow_pickle=False)
    np.savez_compressed(fast/"features.npz", ranks=features["ranks"], quality=features["quality"])
    for name in ("windows.npy", "features.npz"):
        shutil.copyfile(fast/name, root/name)
    sources = {}
    for name in dict.fromkeys((*CODE, PROTOCOL)):
        destination = root/"source-snapshot"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo/name, destination); sources[name] = reference(destination)
    contract = {"kind": "serving-side-balanced-centered-precision-v1", "studyIndex": STUDY, "fastIndex": FAST,
        "arm": ARM, "encoderVariant": "high-f1", "encoderSha256": encoder_hash,
        "embeddingPrecision": "fp16-rounded", "seeds": list(base.SEEDS), "epochs": list(base.EPOCHS),
        "targets": list(base.TARGETS), "primaryTarget": .98, "sourceGroups": previous["contract"]["sourceGroups"],
        "sourceCode": sources, "priorRegistration": reference(prior_root/"registration.json"),
        "priorSummary": reference(prior_root/"summary.json"), "originalRegistration": reference(original_path),
        "encoder": reference(encoder), "manifest": reference(repo/MANIFEST), "receipts": receipts,
        "data": {"rows": reference(root/"rows.json"), "windows": reference(fast/"windows.npy"),
                 "features": reference(fast/"features.npz")},
        "resourcePolicy": "one fitting process; two CPU threads; sequential seeds and source exclusions",
        "encoderIsFrozenAndMayHaveSeenHeldGroupPixels": True, "protectedLabelsOpened": False}
    immutable(root/"registration.json", {"contract": contract, "sha256": canonical(contract)})
    return {"contractSha256": canonical(contract), "rows": len(rows), "recordings": len(receipts),
            "encoderSha256": encoder_hash, "frameTimingAndMasksMatch": True}


def load():
    root, fast = (Path(base.private_value(k)) for k in (STUDY, FAST))
    registration = read(root/"registration.json"); contract = registration["contract"]
    require(canonical(contract) == registration["sha256"], "Balanced registration changed")
    refs = [*contract["sourceCode"].values(), *contract["data"].values(), contract["encoder"], contract["manifest"],
            contract["priorRegistration"], contract["priorSummary"], contract["originalRegistration"]]
    refs.extend(ref for pair in contract["receipts"].values() for ref in pair.values())
    for ref in refs:
        verified(ref)
    for name, ref in contract["sourceCode"].items():
        if name.endswith(".py"):
            require(digest(Path(__file__).resolve().parents[1]/name) == ref["sha256"], "Imported source changed")
    rows = read(contract["data"]["rows"]["path"])
    windows = np.load(contract["data"]["windows"]["path"], allow_pickle=False)
    with np.load(contract["data"]["features"]["path"], allow_pickle=False) as data:
        features = {k: data[k].copy() for k in data.files}
    return root, fast, registration, rows, windows, features


def run(report, device="cuda"):
    base.prior.h.prior.configure_torch()
    root, fast, reg, rows, windows, features = load()
    groups = reg["contract"]["sourceGroups"]
    exclusions = list(itertools.combinations(groups, 2))+[(g,) for g in groups]
    with base.Monitor(fast) as monitor:
        fg = torch.from_numpy(np.column_stack((features["ranks"], features["quality"])).astype(np.float32)).to(device)
        transformed = base.temporal_transform(windows, "centered")
        raw = torch.from_numpy(transformed).to(device); del transformed
        completed = 0
        for seed in base.SEEDS:
            for excluded in exclusions:
                base.memory_guard(); base.fit_neural(fast, reg, rows, features, raw, fg, ARM, seed, excluded)
                completed += 1
                base.progress(fast, phase="neural-fitting", seed=seed, completed=completed, total=108)
            shutil.copytree(fast/"fits"/ARM/str(seed), root/"fits"/ARM/str(seed), dirs_exist_ok=True)
        results = base.nested(fast, reg, rows, ARM)
        base.progress(fast, phase="nested-evaluation")
        # Freeze held-source evaluation and final policies before all-development refits.
        for seed in base.SEEDS:
            base.memory_guard(); base.fit_neural(fast, reg, rows, features, raw, fg, ARM, seed, ())
        deliveries = []
        for choice in results["finalChoices"]:
            path = fast/"fits"/ARM/str(choice["seed"])/base.fold_name(())/f"weights-{choice['epoch']}.npz"
            gpu, cpu = base.load_head(path, device), base.load_head(path, "cpu")
            a = base.cost_scores(base.prior.predict(gpu, raw[:16], fg[:16])[:, :2])
            b = base.cost_scores(base.prior.predict(cpu, raw[:16].cpu(), fg[:16].cpu())[:, :2])
            error = float(np.max(abs(a-b))); require(error < 2e-5, "Final parity failed")
            deliveries.append({"arm": ARM, **choice, "weightsSha256": digest(path),
                               "bytes": path.stat().st_size, "cpuGpuMaximumError": error})
        del raw, fg; torch.cuda.empty_cache()
        for name in ("fits", "teachers", "selection", "final-selection", "results"):
            shutil.copytree(fast/name, root/name, dirs_exist_ok=True)
    samples = monitor.samples
    resources = {"sampleCount": len(samples), "sampleIntervalSeconds": 10,
        "peakResidentMiB": max(x["residentKiB"] for x in samples)/1024,
        "maximumProcessSwapKiB": max(x["processSwapKiB"] for x in samples),
        "minimumWslAvailableGiB": min(x["availableKiB"] for x in samples)/1024**2,
        "wslSwapCounterDelta": {k: samples[-1]["swapCounters"][k]-samples[0]["swapCounters"][k]
                                for k in samples[0]["swapCounters"]},
        "singleFittingProcess": True, "cpuThreads": 2}
    shutil.copyfile(fast/"resources.jsonl", root/"resources.jsonl")
    previous = read(verified(reg["contract"]["priorSummary"]))
    result = {"kind": "serving-side-balanced-centered-precision-results-v1", "studyIndex": STUDY,
        "fastArtifactIndex": FAST, "contractSha256": reg["sha256"], "encoderVariant": "high-f1",
        "encoderSha256": reg["contract"]["encoderSha256"], "embeddingPrecision": "fp16-rounded",
        "scope": {"serves": len(rows), "recordings": len({r['recordingIndex'] for r in rows}), "sourceGroups": len(groups)},
        "primaryTarget": .98, "targets": list(base.TARGETS), "result": results,
        "maximumCoverageControl": previous["experiments"][ARM], "finalNeuralHeads": deliveries,
        "resources": resources, "newNestedNeuralFits": 108, "finalNeuralFits": 3,
        "teacherPartitions": 37, "protectedLabelsOpened": False, "productionChanged": False}
    immutable(root/"summary.json", result); immutable(Path(report), result)
    base.progress(fast, phase="complete", encoderVariant="high-f1")
    return result
