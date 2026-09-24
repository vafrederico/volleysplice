"""Isolate per-original-rally positive live-loss weighting on frozen inputs.

Historical sources and artifacts remain immutable. Each worker owns one
cohort/seed partition; fitting and inner selection otherwise match that study.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import neural_expanded_development as expanded
from . import neural_event_weighting as weighting_module

base = expanded.base
COHORTS = expanded.COHORTS
KINDS = ("tcn",)
EPOCHS = expanded.EPOCHS
ROOT = Path(private_value('private-reference-0057'))
REFERENCE_STUDY = ROOT / "2026-09-19-expanded/study"
MANIFEST = ROOT / "2026-09-19-expanded/manifest.json"
OUTPUT = ROOT / "2026-09-19-event-balanced/study"
RECOVERY_SCREEN = {
    "meanF1MinimumDelta": -.005, "meanRCoreMinimumDelta": -.005,
    "meanLongRCoreMinimumDelta": -.005, "meanEventF1MinimumDelta": -.01,
    "maximumCompleteLossRatio": .8, "maximumShortCompleteLossRatio": .8,
    "maximumIncompleteLossDelta": 0, "maximumShortIncompleteLossDelta": 0,
    "completeLossReductionSeedCount": 2, "shortCompleteLossReductionSeedCount": 2,
    "shortDurationSeconds": 3, "coverageScope": "primaryExportCoverage",
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def weighting_metadata(rows, mode):
    return {"mode": mode, "rowDiagnosticsApply": mode == "per_rally",
            "rows": {tier: [weighting_module.live_event_weights(row)[1] for row in records]
                     for tier, records in rows.items()}}


def fit_model(train, auxiliary, validation, kind, seed, epochs, destination, device,
              contract_hash, weighting="per_rally"):
    """Frozen fitter with separate live multipliers; uniform is preflight only."""
    import torch
    from .expanded_temporal_model import ExpandedTemporalConfig, ExpandedTemporalNetwork

    if kind != "tcn" or weighting not in ("per_rally", "uniform"):
        raise ValueError("unsupported architecture or live weighting mode")
    valid_groups = {e.group for e in validation}
    all_train = train + [r for rows in auxiliary.values() for r in rows]
    if not train or not validation or valid_groups & {r.example.group for r in all_train}:
        raise ValueError("overlapping or empty source fold")
    identity = {"trainIds": [r.example.id for r in train],
                "auxiliaryIds": {tier: [r.example.id for r in rows] for tier, rows in auxiliary.items()},
                "validationIds": [e.id for e in validation]}
    row_groups = {"exact": train, **auxiliary}
    weighting_audit = weighting_metadata(row_groups, weighting)
    completed = destination / "completed.json"
    if completed.exists():
        meta = read(completed)
        inventory = {f"{stem}-{epoch}.npz" for epoch in epochs for stem in ("weights", "predictions")}
        if (meta["contractSha256"] != contract_hash or any(meta[k] != v for k, v in identity.items())
                or meta["epochs"] != list(epochs) or meta["kind"] != kind or meta["seed"] != seed
                or meta.get("liveLossWeighting") != weighting_audit or set(meta["artifacts"]) != inventory):
            raise ValueError("resume contract, identity, weighting or artifact inventory changed")
        for name, digest in meta["artifacts"].items():
            if base.file_sha256(destination/name) != digest:
                raise ValueError("resume artifact changed")
        result = {}
        for epoch in epochs:
            with np.load(destination/f"predictions-{epoch}.npz", allow_pickle=False) as data:
                result[epoch] = {e.id: data[e.id] for e in validation}
        return result
    if destination.exists():
        raise FileExistsError(f"refusing incomplete fit destination: {destination}")
    destination.mkdir(parents=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    mean, scale = base.fit_scaler([r.example for r in train])
    counts = expanded.supervision_counts(train)
    positive = np.asarray(counts["positiveMass"])
    positive_weight = np.minimum(20., np.sqrt((np.asarray(counts["valid"])-positive)/np.maximum(positive, 1.)))
    pos_weight = torch.tensor(positive_weight, dtype=torch.float32, device=device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model = ExpandedTemporalNetwork(ExpandedTemporalConfig(kind=kind)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    pools = {tier: weighting_module.make_weighted_chunks(rows, mean, scale, kind, weighting)
             for tier, rows in row_groups.items()}
    pools = {k: v for k, v in pools.items() if v}
    rngs = {tier: np.random.default_rng(np.random.SeedSequence([seed, i]))
            for i, tier in enumerate(expanded.HEAD_WEIGHTS)}
    exposure = {tier: hashlib.sha256() for tier in pools}
    outputs, history = {}, []
    steps = 0
    started = time.perf_counter()
    for epoch in range(1, max(epochs)+1):
        model.train()
        loss_sum = 0.
        batches = expanded.epoch_batches(pools["exact"], rngs["exact"])
        for exact_indexes in batches:
            optimizer.zero_grad(set_to_none=True)
            combined = None
            for stream, tier in enumerate(expanded.HEAD_WEIGHTS):
                if tier not in pools:
                    continue
                indexes = exact_indexes if tier == "exact" else rngs[tier].choice(
                    len(pools[tier]), len(exact_indexes), replace=True,
                    p=expanded.sampling_weights(pools[tier])).tolist()
                exposure[tier].update(json.dumps([epoch, steps, indexes], separators=(",", ":")).encode())
                torch.manual_seed(seed + steps*11 + stream*100000000)
                loss = weighting_module.batch_loss(model, pools[tier], indexes, pos_weight,
                                                    expanded.HEAD_WEIGHTS[tier], device)
                combined = loss if combined is None else combined + loss
            if not torch.isfinite(combined):
                raise ValueError("non-finite training loss")
            combined.backward()
            expanded.clip_gradients(model, kind)
            optimizer.step()
            loss_sum += float(combined.detach().cpu())
            steps += 1
        history.append({"epoch": epoch, "loss": loss_sum/len(batches), "optimizerSteps": steps,
                        "exposureSha256": {tier: digest.hexdigest() for tier, digest in exposure.items()}})
        if epoch in epochs:
            outputs[epoch] = {e.id: expanded.predict(model, e, mean, scale, kind, device) for e in validation}
            np.savez_compressed(destination/f"predictions-{epoch}.npz", **outputs[epoch])
            state = {f"model::{k}": v.detach().cpu().numpy() for k, v in model.state_dict().items()}
            np.savez_compressed(destination/f"weights-{epoch}.npz", mean=mean, scale=scale, **state)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    base.write_json(completed, {"contractSha256": contract_hash, **identity, "kind": kind, "seed": seed,
        "epochs": epochs, "trainGroups": sorted({r.example.group for r in train}),
        "auxiliaryGroups": {tier: sorted({r.example.group for r in rows}) for tier, rows in auxiliary.items()},
        "validationGroups": sorted(valid_groups), "scalerTrainIds": identity["trainIds"],
        "positiveWeight": positive_weight.tolist(),
        "supervisedCounts": {tier: expanded.supervision_counts(rows) for tier, rows in row_groups.items()},
        "liveLossWeighting": weighting_audit,
        "parameters": sum(p.numel() for p in model.parameters()), "history": history, "optimizerSteps": steps,
        "exposureSha256": {tier: digest.hexdigest() for tier, digest in exposure.items()},
        "wallSeconds": time.perf_counter()-started,
        "peakAllocatedCudaBytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
        "artifacts": {f"{stem}-{epoch}.npz": base.file_sha256(destination/f"{stem}-{epoch}.npz")
                      for epoch in epochs for stem in ("weights", "predictions")}})
    return outputs


def initialize(manifest_path, output, reference_study, device):
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    reference = read(reference_study/"preregistration.json")
    old_contract = reference["contract"]
    if canonical_hash(old_contract) != reference["sha256"]:
        raise ValueError("invalid historical registration")
    for name, digest in old_contract["code"].items():
        if base.file_sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"historical source changed: {name}")
    if base.file_sha256(manifest_path) != old_contract["manifestSha256"]:
        raise ValueError("historical manifest changed")
    reference_report = read(reference_study/"report.json")
    reference_summary = read(reference_study/"summary.json")
    if (reference_report["contractSha256"] != reference["sha256"]
            or reference_report["protectedTestOpened"] or reference_report["productionPromotionAllowed"]
            or len(reference_report["results"]) != 18
            or reference_summary["status"] != "completed-expanded-development-audit"):
        raise ValueError("historical baseline is not the audited completed study")
    data = expanded.load_data(manifest_path)
    groups = sorted({r.example.group for r in data["exact"]})
    environment = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                   "device": device, "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None}
    if environment != old_contract["environment"] or groups != old_contract["groups"]:
        raise ValueError("new environment or groups differ from frozen reference")
    contract = copy.deepcopy(old_contract)
    contract.update({"schemaVersion": 1, "experiment": "per-original-rally-live-loss-v1", "kinds": list(KINDS),
        "referenceStudy": {"path": str(reference_study), "contractSha256": reference["sha256"],
            "preregistrationFileSha256": base.file_sha256(reference_study/"preregistration.json"),
            "reportSha256": base.file_sha256(reference_study/"report.json"),
            "summarySha256": base.file_sha256(reference_study/"summary.json")},
        "retentionRecoveryScreen": RECOVERY_SCREEN,
        "liveLossWeighting": {"mode": "per_rally", "method": weighting_module.METHOD,
            "formula": "Within recording: P/(N*n_i) on supervised live-positive ticks of original event i; P total positive ticks, N nonempty events, n_i event ticks",
            "scope": ["exact", "draft"], "cap": None,
            "otherTicks": "multiplier1; original masks unchanged, unknown/ignored still zero loss",
            "zeroSupervisedEvents": "excluded from P and N; remain unknown, never negative",
            "ignoredSplits": "same original event identity across valid pieces",
            "normalization": "original unweighted valid-mask denominator per head; exact-only positive class weights unchanged",
            "otherHeads": "serve/end/keep losses unchanged", "sampling": "unchanged frozen chunks, batches and RNG streams",
            "evaluation": "original unweighted labels and interval metrics; no event multipliers in scoring"},
        "code": {**old_contract["code"], **{name: base.file_sha256(Path(__file__).with_name(name))
                 for name in ("neural_event_balanced_development.py", "neural_event_weighting.py")}},
        "environment": environment})
    contract["training"]["loss"] = "per-head mean over original valid ticks; per-rally multipliers only on positive live BCE; fixed tier weights unchanged"
    contract_hash = canonical_hash(contract)
    prereg = output/"preregistration.json"
    if prereg.exists():
        if read(prereg) != {"sha256": contract_hash, "contract": contract}:
            raise ValueError("preregistration changed; refuse overwrite")
    else:
        base.write_json(prereg, {"sha256": contract_hash, "contract": contract})
    diagnostics = {"manifestSha256": contract["manifestSha256"], "contractSha256": contract_hash,
                   "rows": {tier: [weighting_module.live_event_weights(row)[1] for row in rows]
                            for tier, rows in data.items()}}
    diagnostics_path = output/"weight-diagnostics.json"
    if diagnostics_path.exists():
        if read(diagnostics_path) != diagnostics:
            raise ValueError("registered weight diagnostics changed")
    else:
        base.write_json(diagnostics_path, diagnostics)
    return data, contract, contract_hash


def run_job(data, contract, contract_hash, cohort, seed, output, device):
    from .neural_evaluation import evaluate_predictions

    if cohort not in COHORTS or seed not in base.SEEDS:
        raise ValueError("partition outside frozen cohort/seed grid")
    groups = contract["groups"]
    outer_rows, selections = [], []
    for outer_index, outer in enumerate(groups):
        fitting = [r for r in data["exact"] if r.example.group != outer]
        held = [r.example for r in data["exact"] if r.example.group == outer]
        predictions = {epoch: {} for epoch in EPOCHS}
        for inner_index, inner in enumerate(g for g in groups if g != outer):
            training = [r for r in fitting if r.example.group != inner]
            validating = [r.example for r in fitting if r.example.group == inner]
            auxiliary = expanded.auxiliary_for_fold(data, cohort, {outer, inner})
            location = output/"fits"/cohort/"tcn"/str(seed)/f"outer-{outer_index}"/f"inner-{inner_index}"
            print(f"FIT {cohort} seed={seed} outer={outer} inner={inner}", flush=True)
            outputs = fit_model(training, auxiliary, validating, "tcn", seed, EPOCHS, location, device, contract_hash)
            for epoch in EPOCHS:
                predictions[epoch].update(outputs[epoch])
        selected = expanded.choose_settings([r.example for r in fitting], predictions)
        epoch = selected["epoch"]
        location = output/"fits"/cohort/"tcn"/str(seed)/f"outer-{outer_index}"/"refit"
        print(f"REFIT {cohort} seed={seed} outer={outer} epoch={epoch} feasible={selected['recallEligibilityPassed']}", flush=True)
        probabilities = fit_model(fitting, expanded.auxiliary_for_fold(data, cohort, {outer}), held,
                                  "tcn", seed, (epoch,), location, device, contract_hash)[epoch]
        outer_rows.extend(e.row(base.decode(e, probabilities[e.id], selected["decoder"])) for e in held)
        selections.append({"heldSourceGroup": outer, **selected})
    evaluation = evaluate_predictions(outer_rows)
    result = {"cohort": cohort, "kind": "tcn", "seed": seed, "contractSha256": contract_hash,
              "selections": selections, "evaluation": evaluation,
              "predictions": [{**r, **{k: [i.to_dict() for i in r[k]]
                                       for k in ("rallies", "ignoredIntervals", "predictions")}} for r in outer_rows]}
    destination = output/f"result-{cohort}-tcn-{seed}.json"
    if destination.exists():
        if read(destination) != result:
            raise ValueError("completed partition result changed")
    else:
        base.write_json(destination, result)
    print(f"RESULT {cohort} seed={seed} F1_padP_coreR={evaluation['objective']:.6f}", flush=True)


def run_study(manifest_path, output, reference_study, device, workers):
    if (output/"report.json").exists():
        raise FileExistsError("completed study exists")
    _, contract, contract_hash = initialize(manifest_path, output, reference_study, device)
    jobs = [(cohort, seed) for cohort in COHORTS for seed in base.SEEDS]

    def launch(job):
        cohort, seed = job
        command = [sys.executable, "-u", "-m", "analysis.neural_event_balanced_development",
                   "--manifest", str(manifest_path), "--output", str(output),
                   "--reference-study", str(reference_study), "--device", device,
                   "--job", f"{cohort}:{seed}"]
        subprocess.run(command, check=True)

    # Separate spawned interpreters own disjoint destinations and Torch RNGs.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(launch, jobs))
    _, final_contract, final_hash = initialize(manifest_path, output, reference_study, device)
    if final_hash != contract_hash or final_contract != contract:
        raise ValueError("registered contract changed during training")
    results = [read(output/f"result-{cohort}-tcn-{seed}.json") for cohort, seed in jobs]
    if any(r["contractSha256"] != contract_hash for r in results):
        raise ValueError("partition contract mismatch")
    report = {"schemaVersion": 1, "contractSha256": contract_hash,
              "manifestSha256": contract["manifestSha256"], "status": "completed-event-balanced-development",
              "records": contract["evaluationPopulation"]["records"], "sourceGroups": contract["groups"],
              "protectedTestOpened": False, "productionPromotionAllowed": False, "results": results}
    base.write_json(output/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--reference-study", type=Path, default=REFERENCE_STUDY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--job", help="Internal disjoint cohort:seed partition")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    paths = (args.manifest.resolve(), args.output.resolve(), args.reference_study.resolve())
    if args.job or args.register_only:
        data, contract, contract_hash = initialize(*paths, args.device)
        if args.job:
            cohort, seed = args.job.split(":")
            run_job(data, contract, contract_hash, cohort, int(seed), paths[1], args.device)
        else:
            print(json.dumps({"contractSha256": contract_hash, "status": "registered-no-training"}))
    else:
        run_study(*paths, args.device, args.workers)


if __name__ == "__main__":
    main()
