"""Matched bounded short-rally loss study in AV and frozen-DINO temporal models."""
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import neural_expanded_development as expanded
from . import neural_short_boost_weighting as weighting_module
from .neural_event_balanced_development import RECOVERY_SCREEN, canonical_hash, read
from .transfer_temporal_model import MODEL_KINDS, model_for, model_metadata

base = expanded.base
COHORTS, EPOCHS = expanded.COHORTS, expanded.EPOCHS
KINDS = MODEL_KINDS
ARMS = ("baseline", "global_control", "short_boost")
ROOT = Path(private_value('private-reference-0057'))
REFERENCE_STUDY = ROOT / "2026-09-19-expanded/study"
ORIGINAL_MANIFEST = ROOT / "2026-09-19-expanded/manifest.json"
EXPERIMENT = ROOT / "2026-09-19-short-boost-transfer"
MANIFEST = EXPERIMENT / "manifest-pts-v1.json"
PROTOCOL_SHA256 = "f8a75a3d6a6d61fd12efca6790e942906bac3a9cb7c9e00ec2c79fd65d7a0826"
AMENDMENT_SHA256 = "6e6d9733abe7c3b020c64afe3b0df1653a2d62dd565162bdd1912beef6b06d1a"
BASELINE_REUSE_COHORTS = ("exact", "draft")


def reuses_reference(cohort, kind, arm):
    return kind == "tcn" and arm == "baseline" and cohort in BASELINE_REUSE_COHORTS


def verify_input_revision(manifest_path, historical_manifest_sha):
    """Only the seven coverage AV cache objects may change before this study."""
    manifest = read(manifest_path)
    original_identity = manifest["originalManifest"]
    original_path = Path(original_identity["path"])
    if (original_identity["sha256"] != historical_manifest_sha
            or base.file_sha256(original_path) != historical_manifest_sha):
        raise ValueError("original expanded manifest changed")
    original = read(original_path)
    amendment_identity = manifest["protocolAmendment"]
    amendment_path = Path(amendment_identity["path"])
    if (amendment_identity["sha256"] != AMENDMENT_SHA256
            or base.file_sha256(amendment_path) != AMENDMENT_SHA256):
        raise ValueError("pretraining PTS amendment changed")
    amendment = read(amendment_path)
    document = amendment["document"]
    if (base.file_sha256(Path(document["path"])) != document["sha256"]
            or amendment["originalManifest"] != original_identity
            or manifest["featureRevision"] != amendment["featureRevision"]):
        raise ValueError("PTS amendment provenance differs")
    restored = copy.deepcopy(manifest)
    for field in ("originalManifest", "protocolAmendment", "featureRevision"):
        restored.pop(field)
    if len(restored["coverageRows"]) != 7 or len(original["coverageRows"]) != 7:
        raise ValueError("coverage repair population changed")
    changed_ids = []
    for current, previous in zip(restored["coverageRows"], original["coverageRows"]):
        if (current["id"] != previous["id"] or
                current["featureCaches"]["audiovisual"] == previous["featureCaches"]["audiovisual"]):
            raise ValueError("missing or reordered PTS coverage repair")
        changed_ids.append(current["id"])
        current["featureCaches"]["audiovisual"] = previous["featureCaches"]["audiovisual"]
    if restored != original or changed_ids != manifest["featureRevision"]["recordingIds"] or len(changed_ids) != 7:
        raise ValueError("PTS revision changes labels, groups or noncoverage inputs")
    return manifest


def weighting_metadata(rows, mode):
    return {"mode": mode, "rows": {tier: [weighting_module.live_event_weights(r, mode)[1] for r in records]
                                   for tier, records in rows.items()}}


def load_data(manifest_path, dino_manifest_path, with_dino=True):
    """Load unchanged supervision and optionally append nearest frozen DINO tokens."""
    data = expanded.load_data(manifest_path)
    if not with_dino:
        return data
    manifest = read(dino_manifest_path)
    if (manifest["expandedManifestPath"] != str(manifest_path)
            or manifest["expandedManifestSha256"] != base.file_sha256(manifest_path)
            or manifest.get("protectedTestOpened") is not False or manifest.get("beachIncluded") is not False
            or manifest.get("embeddingLabelsUsed") is not False):
        raise ValueError("DINO provenance does not match expanded dataset")
    expanded_manifest = read(manifest_path)
    source_rows = {r["id"]: r for key in ("exactRows", "draftRows", "coverageRows") for r in expanded_manifest[key]}
    entries = manifest["records"]
    by_id = {r["recordingId"]: r for r in entries}
    expected = {r.example.id for rows in data.values() for r in rows}
    if len(by_id) != len(entries) or set(by_id) != expected or len(expected) != 18:
        raise ValueError("DINO population differs from the fixed18")
    for tier, rows in data.items():
        for index, row in enumerate(rows):
            entry, e = by_id[row.example.id], row.example
            source = source_rows[e.id]
            if (entry["tier"] != tier or entry["sourceGroup"] != e.group or entry.get("passed") is not True
                    or entry["audiovisualSha256"] != source["featureCaches"]["audiovisual"]["sha256"]
                    or entry["sourceVideoVerified"]["sha256"] != source["contentSha256"]):
                raise ValueError(f"DINO source association changed: {e.id}")
            for path_key, hash_key in (("dinoPath", "dinoSha256"), ("metadataPath", "metadataSha256")):
                if base.file_sha256(Path(entry[path_key])) != entry[hash_key]:
                    raise ValueError(f"changed DINO input: {e.id}/{path_key}")
            with np.load(entry["dinoPath"], allow_pickle=False) as cache:
                timestamps = cache["timestamps"].astype(np.float64)
                tokens = cache["tokens"]
                metadata = json.loads(str(cache["metadata_json"].item()))
            wrapper = read(Path(entry["metadataPath"]))
            roi = [float(source["roi"][k]) for k in ("x", "y", "width", "height")] if source.get("roi") else None
            if (wrapper["recordingId"] != e.id or wrapper["cacheMetadata"] != metadata
                    or wrapper["cache"]["sha256"] != entry["dinoSha256"]
                    or metadata["recordingId"] != e.id or metadata["recordingContentSha256"] != source["contentSha256"]
                    or metadata["extractorConfigSha256"] != manifest["extractorConfigSha256"]
                    or metadata["roi"] != roi or metadata.get("labelsUsed") is not False
                    or metadata.get("completed") is not True
                    or any(metadata["backbone"][k] != v for k, v in manifest["semanticBackbone"].items())):
                raise ValueError(f"DINO metadata association/recipe changed: {e.id}")
            if (timestamps.ndim != 1 or not len(timestamps) or tokens.shape != (len(timestamps), 10, 384)
                    or tokens.dtype != np.float16 or not np.isfinite(tokens).all()
                    or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0)):
                raise ValueError(f"invalid frozen DINO values: {e.id}")
            right = np.clip(np.searchsorted(timestamps, e.times), 0, len(timestamps)-1)
            left = np.maximum(0, right-1)
            nearest = np.where(np.abs(timestamps[left]-e.times) <= np.abs(timestamps[right]-e.times), left, right)
            if np.max(np.abs(timestamps[nearest]-e.times)) > .125+1e-8:
                raise ValueError(f"unaligned DINO timestamps: {e.id}")
            alignment = entry["nearestAlignment"]
            for key, value in (("avTimesSha256", e.times.astype("<f8")),
                               ("dinoTimesSha256", timestamps.astype("<f8")),
                               ("nearestIndexesSha256", nearest.astype("<i8"))):
                if hashlib.sha256(value.tobytes()).hexdigest() != alignment[key]:
                    raise ValueError(f"DINO alignment identity changed: {e.id}/{key}")
            values = np.concatenate((e.values, tokens[nearest].reshape(len(e.times), -1)), axis=1).astype(np.float32)
            rows[index] = replace(row, example=replace(e, values=values))
    return data


def fit_model(train, auxiliary, validation, kind, seed, epochs, destination, device,
              contract_hash, weighting="short_boost"):
    """Frozen sampling/optimization with one separate live-loss multiplier."""
    import torch

    if kind not in KINDS or weighting not in ARMS:
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
                or meta.get("lossArm") != weighting or meta.get("liveLossWeighting") != weighting_audit
                or set(meta["artifacts"]) != inventory or meta.get("model") != model_metadata(kind)):
            raise ValueError("resume contract, identity, weighting or artifact inventory changed")
        for name, digest in meta["artifacts"].items():
            if base.file_sha256(destination/name) != digest:
                raise ValueError("resume artifact changed")
        result = {}
        for epoch in epochs:
            with np.load(destination/f"predictions-{epoch}.npz", allow_pickle=False) as cache:
                result[epoch] = {e.id: cache[e.id] for e in validation}
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
    model = model_for(kind).to(device)
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
    base.write_json(completed, {"contractSha256": contract_hash, **identity, "kind": kind,
        "architecture": kind, "lossArm": weighting, "seed": seed, "epochs": epochs,
        "trainGroups": sorted({r.example.group for r in train}),
        "auxiliaryGroups": {tier: sorted({r.example.group for r in rows}) for tier, rows in auxiliary.items()},
        "validationGroups": sorted(valid_groups), "scalerTrainIds": identity["trainIds"],
        "positiveWeight": positive_weight.tolist(),
        "supervisedCounts": {tier: expanded.supervision_counts(rows) for tier, rows in row_groups.items()},
        "liveLossWeighting": weighting_audit, "model": model_metadata(kind),
        "parameters": sum(p.numel() for p in model.parameters()), "history": history, "optimizerSteps": steps,
        "exposureSha256": {tier: digest.hexdigest() for tier, digest in exposure.items()},
        "wallSeconds": time.perf_counter()-started,
        "peakAllocatedCudaBytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
        "artifacts": {f"{stem}-{epoch}.npz": base.file_sha256(destination/f"{stem}-{epoch}.npz")
                      for epoch in epochs for stem in ("weights", "predictions")}})
    return outputs


def initialize(manifest_path, dino_manifest_path, output, reference_study, preflight_path, device, kind=None):
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    reference = read(reference_study/"preregistration.json")
    old = reference["contract"]
    if canonical_hash(old) != reference["sha256"]:
        raise ValueError("invalid historical registration")
    for name, digest in old["code"].items():
        if base.file_sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"historical source changed: {name}")
    manifest = verify_input_revision(manifest_path, old["manifestSha256"])
    ref_report, ref_summary = read(reference_study/"report.json"), read(reference_study/"summary.json")
    if (ref_report["contractSha256"] != reference["sha256"] or ref_report["protectedTestOpened"]
            or ref_report["productionPromotionAllowed"] or len(ref_report["results"]) != 18
            or ref_summary["status"] != "completed-expanded-development-audit"):
        raise ValueError("reference baseline not completed and audited")
    code = {**old["code"], **{name: base.file_sha256(Path(__file__).with_name(name)) for name in (
        "neural_event_balanced_development.py", "neural_event_weighting.py", "neural_short_boost_transfer.py",
        "neural_short_boost_weighting.py", "transfer_temporal_model.py")}}
    preflight = read(preflight_path)
    if not preflight["passed"] or preflight["code"] != code:
        raise ValueError("passed preflight does not bind current sources")
    if (preflight["manifest"]["sha256"] != base.file_sha256(manifest_path)
            or preflight["dinoManifest"]["sha256"] != base.file_sha256(dino_manifest_path)):
        raise ValueError("preflight input manifest changed")
    data = load_data(manifest_path, dino_manifest_path, with_dino=kind != "tcn")
    groups = sorted({r.example.group for r in data["exact"]})
    environment = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                   "device": device, "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None}
    if environment != old["environment"] or groups != old["groups"]:
        raise ValueError("environment or source groups differ from reference")
    protocol_path = dino_manifest_path.parent/"protocol-initial.md"
    if base.file_sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("prospective protocol snapshot changed")
    contract = copy.deepcopy(old)
    contract.update({"schemaVersion": 1, "experiment": "bounded-short-boost-dino-transfer-v1",
        "manifestSha256": base.file_sha256(manifest_path), "originalManifest": manifest["originalManifest"],
        "protocolAmendment": manifest["protocolAmendment"], "featureRevision": manifest["featureRevision"],
        "kinds": list(KINDS), "lossArms": list(ARMS), "models": {k: model_metadata(k) for k in KINDS},
        "protocolSnapshot": {"path": str(protocol_path), "sha256": PROTOCOL_SHA256},
        "dinoManifest": {"path": str(dino_manifest_path), "sha256": base.file_sha256(dino_manifest_path)},
        "preflight": {"path": str(preflight_path), "sha256": base.file_sha256(preflight_path)},
        "referenceStudy": {"path": str(reference_study), "contractSha256": reference["sha256"],
            "preregistrationFileSha256": base.file_sha256(reference_study/"preregistration.json"),
            "reportSha256": base.file_sha256(reference_study/"report.json"),
            "summarySha256": base.file_sha256(reference_study/"summary.json"),
            "baselineReuseCohorts": list(BASELINE_REUSE_COHORTS),
            "reuse": "six immutable compact exact/draft baseline cells;96 original fits; corrected reviewed-export baseline retrained"},
        "retentionRecoveryScreen": RECOVERY_SCREEN,
        "transferScreen": {"requireRecoveryPass": ["short_boost-minus-baseline", "short_boost-minus-global_control"],
            "architectures": list(KINDS), "sameCohortRequired": True, "innerSelectionsMustBeFeasible": True,
            "differenceInDifferences": "descriptive paired interaction; zero interaction can still replicate benefit",
            "promotionAllowed": False},
        "liveLossWeighting": {"method": weighting_module.METHOD, "shortDurationSeconds": 3,
            "shortPositiveMultiplier": 2, "scope": ["exact", "draft"],
            "baseline": "all multipliers1",
            "short_boost": "multiplier2 only on existing supervised positive live ticks of original duration<=3s; all others1",
            "global_control": "float32(1+S/P) on all supervised live positives within recording;P0=>1; negatives1",
            "P": "existing supervised positive live tick count", "S": "subset belonging to original<=3s events",
            "massMatching": "full-record positive loss mass within float32 rounding; not per sampled batch",
            "ignoredSplits": "original event identity retained", "zeroSupervisedEvents": "remain unknown, never negative",
            "denominator": "original unweighted valid-mask count per head",
            "otherLossesAndSampling": "unchanged exact-only class weights, tier/head weights, chunks and RNG streams",
            "evaluation": "unweighted exact labels and canonical interval metrics"},
        "code": code, "environment": environment})
    contract["training"]["loss"] = "original per-head valid mean; fixed positive-live multipliers per arm; unchanged tier/head weights"
    contract_hash = canonical_hash(contract)
    registration = {"sha256": contract_hash, "contract": contract}
    path = output/"preregistration.json"
    if path.exists():
        if read(path) != registration:
            raise ValueError("preregistration changed; refuse overwrite")
    else:
        base.write_json(path, registration)
    diagnostics = {"contractSha256": contract_hash, "arms": {
        arm: weighting_metadata(data, arm) for arm in ARMS}}
    path = output/"weight-diagnostics.json"
    if path.exists():
        if read(path) != diagnostics:
            raise ValueError("registered weight diagnostics changed")
    else:
        base.write_json(path, diagnostics)
    return data, contract, contract_hash


def result_path(output, cohort, kind, arm, seed):
    return output/f"result-{cohort}-{kind}-{arm}-{seed}.json"


def run_job(data, contract, contract_hash, cohort, kind, arm, seed, output, device):
    from .neural_evaluation import evaluate_predictions

    if cohort not in COHORTS or kind not in KINDS or arm not in ARMS or seed not in base.SEEDS:
        raise ValueError("partition outside frozen grid")
    if reuses_reference(cohort, kind, arm):
        reference = contract["referenceStudy"]
        old_result = next(r for r in read(Path(reference["path"])/"report.json")["results"]
                          if (r["cohort"], r["kind"], r["seed"]) == (cohort, kind, seed))
        result = {**old_result, "architecture": kind, "lossArm": arm, "contractSha256": contract_hash,
            "origin": {"type": "reused-reference", "studyPath": reference["path"],
                "reportSha256": reference["reportSha256"], "referenceContractSha256": reference["contractSha256"],
                "fitRoot": str(Path(reference["path"])/"fits"/cohort/kind/str(seed))}}
    else:
        fit_root = output/"fits"/cohort/kind/arm/str(seed)
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
                location = fit_root/f"outer-{outer_index}"/f"inner-{inner_index}"
                print(f"FIT {cohort} {kind} {arm} seed={seed} outer={outer} inner={inner}", flush=True)
                outputs = fit_model(training, auxiliary, validating, kind, seed, EPOCHS, location,
                                    device, contract_hash, arm)
                for epoch in EPOCHS:
                    predictions[epoch].update(outputs[epoch])
            selected = expanded.choose_settings([r.example for r in fitting], predictions)
            epoch = selected["epoch"]
            location = fit_root/f"outer-{outer_index}"/"refit"
            print(f"REFIT {cohort} {kind} {arm} seed={seed} outer={outer} epoch={epoch} feasible={selected['recallEligibilityPassed']}", flush=True)
            probabilities = fit_model(fitting, expanded.auxiliary_for_fold(data, cohort, {outer}), held,
                                      kind, seed, (epoch,), location, device, contract_hash, arm)[epoch]
            outer_rows.extend(e.row(base.decode(e, probabilities[e.id], selected["decoder"])) for e in held)
            selections.append({"heldSourceGroup": outer, **selected})
        result = {"cohort": cohort, "kind": kind, "architecture": kind, "lossArm": arm, "seed": seed,
            "contractSha256": contract_hash, "origin": {"type": "trained", "fitRoot": str(fit_root)},
            "selections": selections, "evaluation": evaluate_predictions(outer_rows),
            "predictions": [{**r, **{k: [i.to_dict() for i in r[k]]
                                     for k in ("rallies", "ignoredIntervals", "predictions")}} for r in outer_rows]}
    destination = result_path(output, cohort, kind, arm, seed)
    if destination.exists():
        if read(destination) != result:
            raise ValueError("completed partition result changed")
    else:
        base.write_json(destination, result)
    print(f"RESULT {cohort} {kind} {arm} seed={seed} F1_padP_coreR={result['evaluation']['objective']:.6f}", flush=True)


def run_study(manifest, dino_manifest, output, reference, preflight, device, workers):
    if (output/"report.json").exists():
        raise FileExistsError("completed study exists")
    data, contract, digest = initialize(manifest, dino_manifest, output, reference, preflight, device)
    jobs = [(c, k, a, s) for c in COHORTS for k in KINDS for a in ARMS for s in base.SEEDS]
    for cohort, kind, arm, seed in jobs:
        if reuses_reference(cohort, kind, arm):
            run_job(data, contract, digest, cohort, kind, arm, seed, output, device)
    del data
    fresh = [j for j in jobs if not reuses_reference(*j[:3])]
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()
    execution = {"contractSha256": digest, "startedAt": timestamp, "workers": workers,
                 "freshResultCells": len(fresh), "reusedResultCells": len(jobs)-len(fresh),
                 "jobOrder": [list(j) for j in fresh], "status": "running"}
    execution_path = output/f"execution-{time.time_ns()}.json"
    base.write_json(execution_path, execution)
    log_root = output/"job-logs"
    log_root.mkdir(exist_ok=True)

    def launch(job):
        command = [sys.executable, "-u", "-m", "analysis.neural_short_boost_transfer",
            "--manifest", str(manifest), "--dino-manifest", str(dino_manifest), "--output", str(output),
            "--reference-study", str(reference), "--preflight", str(preflight), "--device", device,
            "--job", ":".join(map(str, job))]
        label = "-".join(map(str, job))
        print(f"START {label} {datetime.now(timezone.utc).isoformat()}", flush=True)
        with (log_root/f"{label}-{time.time_ns()}.log").open("x") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        print(f"DONE {label} {datetime.now(timezone.utc).isoformat()}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(launch, job) for job in fresh]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        _, final_contract, final_digest = initialize(manifest, dino_manifest, output, reference, preflight, device)
        if final_digest != digest or final_contract != contract:
            raise ValueError("contract changed during training")
        results = [read(result_path(output, *job)) for job in jobs]
        if any(r["contractSha256"] != digest for r in results):
            raise ValueError("partition contract mismatch")
        report = {"schemaVersion": 1, "contractSha256": digest, "manifestSha256": contract["manifestSha256"],
            "status": "completed-short-boost-transfer-development", "records": contract["evaluationPopulation"]["records"],
            "sourceGroups": contract["groups"], "protectedTestOpened": False,
            "productionPromotionAllowed": False, "results": results}
        base.write_json(output/"report.json", report)
    except BaseException as error:
        execution.update({"status": "failed", "errorType": type(error).__name__, "error": str(error),
                          "endedAt": datetime.now(timezone.utc).isoformat(), "wallSeconds": time.perf_counter()-started})
        base.write_json(execution_path, execution)
        raise
    execution.update({"status": "completed", "endedAt": datetime.now(timezone.utc).isoformat(),
                      "wallSeconds": time.perf_counter()-started, "reportSha256": base.file_sha256(output/"report.json")})
    base.write_json(execution_path, execution)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--dino-manifest", type=Path, default=EXPERIMENT/"dino-manifest.json")
    parser.add_argument("--output", type=Path, default=EXPERIMENT/"study")
    parser.add_argument("--reference-study", type=Path, default=REFERENCE_STUDY)
    parser.add_argument("--preflight", type=Path, default=EXPERIMENT/"preflight/preflight-report.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--job", help="Internal cohort:kind:lossArm:seed partition")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    paths = tuple(p.resolve() for p in (args.manifest, args.dino_manifest, args.output, args.reference_study, args.preflight))
    if args.job or args.register_only:
        parts = args.job.split(":") if args.job else None
        data, contract, digest = initialize(*paths, args.device, kind=parts[1] if parts else None)
        if parts:
            run_job(data, contract, digest, *parts[:3], int(parts[3]), paths[2], args.device)
        else:
            print(json.dumps({"contractSha256": digest, "status": "registered-no-training"}))
    else:
        run_study(*paths, args.device, args.workers)


if __name__ == "__main__":
    main()
