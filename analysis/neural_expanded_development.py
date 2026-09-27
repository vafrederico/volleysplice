"""Controlled exact/draft/reviewed-export augmentation on held source groups.

The frozen first study remains unchanged. Every optimizer step contains the same
exact batch across cohorts; auxiliary losses add supervision, never gold labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import neural_development as base
from .crop_evaluation import RecordingIntervals, evaluate_f1_pad_p_core_r, pad_and_merge_intervals, subtract_intervals
from .features import ABSOLUTE_FEATURE_NAMES, feature_names, percentile_rank_values
from .schema import Interval, labels_for_times, mask_for_times

COHORTS = ("exact", "draft", "reviewed_export")
KINDS = ("linear", "tcn")
EPOCHS = (5, 15, 30, 60)
HEAD_WEIGHTS = {"exact": (1., .5, .5, .25), "draft": (.5, 0., 0., 0.), "coverage": (0., 0., 0., .25)}


@dataclass
class Supervised:
    example: base.Example
    mask: np.ndarray
    tier: str


def intervals(rows):
    return tuple(Interval(float(r["start"]), float(r["end"]), tuple(r.get("tags", ()))) for r in rows)


def exact_supervision(example):
    keep = subtract_intervals(pad_and_merge_intervals(example.truth, example.duration, 2., 3.), example.ignored)
    targets = np.column_stack((example.targets, labels_for_times(example.times, keep))).astype(np.float32)
    masks = np.column_stack((base.supervision_mask(example), example.valid)).astype(np.float32)
    return Supervised(replace(example, targets=targets), masks, "exact")


def auxiliary_supervision(example, tier, keep=()):
    targets = np.zeros((len(example.times), 4), dtype=np.float32)
    masks = np.zeros_like(targets)
    if tier == "draft":
        targets[:, 0] = labels_for_times(example.times, example.truth)
        masks[:, 0] = example.valid
        # Approximate endpoints are unknown, including short fully-censored rallies.
        for row in (*example.truth, *example.ignored):
            for edge in (row.start, row.end):
                masks[np.abs(example.times - edge) <= 1., 0] = 0.
    elif tier == "coverage":
        targets[:, 3] = labels_for_times(example.times, keep)
        masks[:, 3] = example.valid
    else:
        raise ValueError("unknown auxiliary tier")
    return Supervised(replace(example, targets=targets), masks, tier)


def load_data(path):
    manifest = json.loads(path.read_text())
    reference = manifest["exactManifest"]
    if base.file_sha256(Path(reference["path"])) != reference["sha256"]:
        raise ValueError("changed exact manifest")
    all_rows = manifest["exactRows"] + manifest["draftRows"] + manifest["coverageRows"]
    seen_ids, seen_hashes = set(), set()
    for row in all_rows:
        if row["environment"] == "beach" or row["sourceGroup"] in base.PROTECTED_GROUPS or row.get("split") == "test":
            raise ValueError("excluded development source")
        if row["consent"].get("train") is not True:
            raise ValueError("training authorization absent")
        identities = {row[k] for k in ("contentSha256", "sourceContentSha256") if row.get(k)}
        if not identities or row["id"] in seen_ids or identities & seen_hashes:
            raise ValueError("duplicate raw/proxy lineage")
        seen_ids.add(row["id"])
        seen_hashes.update(identities)
    exact = [exact_supervision(e) for e in base.load_examples(Path(reference["path"]), False)]
    if [r.example.id for r in exact] != [r["id"] for r in manifest["exactRows"]]:
        raise ValueError("exact manifest identity mismatch")
    result = {"exact": exact, "draft": [], "coverage": []}
    expected_names = feature_names(base.FeatureConfig(audio_feature_set=base.NOISE_NORMALIZED_AUDIO_FEATURE_SET))
    for tier, key in (("draft", "draftRows"), ("coverage", "coverageRows")):
        for row in manifest[key]:
            if row.get("trainingOnly") is not True or row["annotation"].get("continuousVideoReviewed") is not True:
                raise ValueError("auxiliary annotation review scope absent")
            cache = row["featureCaches"]["audiovisual"]
            if base.file_sha256(Path(cache["path"])) != cache["sha256"]:
                raise ValueError("changed auxiliary cache")
            with np.load(cache["path"], allow_pickle=False) as data:
                times = data["times"].astype(np.float64)
                raw = data["values"].astype(np.float32)
                names = tuple(str(n) for n in data["names"])
                metadata = json.loads(str(data["metadata_json"].item()))
            if (names != expected_names or raw.shape != (len(times), 104) or not np.isfinite(raw).all()
                    or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
                    or np.any(np.abs(np.diff(times) - .25) > .10)):
                raise ValueError("invalid auxiliary feature schema/timeline")
            values = percentile_rank_values(raw)
            for index, name in enumerate(names):
                if name in ABSOLUTE_FEATURE_NAMES:
                    values[:, index] = raw[:, index]
            ignored = intervals(row.get("ignoredIntervals", []))
            valid = mask_for_times(times, ignored)
            if tier == "coverage":
                if row["targetContract"].get("negativesOutsideKeepAuthorizedByFullManualReview") is not True:
                    raise ValueError("coverage negative authorization absent")
                window = row["gameWindow"]
                valid &= (times >= window["start"]) & (times < window["end"])
            example = base.Example(row["id"], row["sourceGroup"], float(metadata["duration"]), times, values,
                                   np.zeros((len(times), 3), np.float32), valid,
                                   intervals(row.get("rallies", [])), ignored, row["environment"])
            result[tier].append(auxiliary_supervision(example, tier, intervals(row.get("keepTargets", []))))
    return result


def auxiliary_for_fold(data, cohort, excluded_groups):
    tiers = () if cohort == "exact" else ("draft",) if cohort == "draft" else ("draft", "coverage")
    return {tier: [r for r in data[tier] if r.example.group not in excluded_groups] for tier in tiers}


def make_chunks(rows, mean, scale, kind):
    chunks = []
    counts = {r.example.group: sum(s.example.group == r.example.group for s in rows) for r in rows}
    for row in rows:
        e = row.example
        values = base.standardized(e, mean, scale, kind)
        one = []
        for start, end in base.segments(e.valid):
            for core_start in range(start, end, 128):
                core_end = min(end, core_start + 128)
                left, right = max(start, core_start-62), min(end, core_end+62)
                mask = np.zeros((right-left, 4), dtype=np.float32)
                mask[core_start-left:core_end-left] = row.mask[core_start:core_end]
                if mask.any():
                    one.append((values[left:right], e.targets[left:right], mask))
        for x, y, mask in one:
            chunks.append((x, y, mask, 1. / (counts[e.group] * len(one))))
    return chunks


def sampling_weights(chunks):
    weights = np.asarray([r[3] for r in chunks], dtype=np.float64)
    return weights / weights.sum()


def epoch_batches(chunks, rng):
    chosen = rng.choice(len(chunks), size=len(chunks), replace=True, p=sampling_weights(chunks))
    buckets = {}
    for index in chosen:
        buckets.setdefault(len(chunks[index][0]), []).append(int(index))
    return [indexes[start:start+16] for indexes in buckets.values() for start in range(0, len(indexes), 16)]


def masked_head_loss(logits, targets, mask, pos_weight, head_weight):
    import torch
    elements = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    sums = (elements * mask).sum(dim=(0, 1))
    counts = mask.sum(dim=(0, 1))
    return ((sums / counts.clamp_min(1)) * head_weight).sum()


def batch_loss(model, chunks, indexes, pos_weight, head_weights, device):
    """No synthetic input padding. Reduce by valid ticks per head across buckets."""
    import torch
    buckets = {}
    for index in indexes:
        buckets.setdefault(len(chunks[index][0]), []).append(index)
    counts = np.sum([chunks[i][2].sum(axis=0) for i in indexes], axis=0)
    denominator = torch.tensor(np.maximum(counts, 1), device=device)
    weights = torch.tensor(head_weights, device=device)
    total = None
    for batch_indices in buckets.values():
        x, y, mask = [torch.from_numpy(np.stack([chunks[i][j] for i in batch_indices])).to(device) for j in range(3)]
        elements = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_weight, reduction="none")
        loss = ((elements * mask).sum(dim=(0, 1)) / denominator * weights).sum()
        total = loss if total is None else total + loss
    return total


def clip_gradients(model, kind):
    import torch
    if kind != "linear":
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        return
    # Keep-only gradients must not rescale independent primary linear heads.
    weight, bias = model.context_head.weight.grad, model.context_head.bias.grad
    norms = torch.sqrt(weight.square().sum(dim=1) + bias.square())
    factors = (1. / (norms + 1e-6)).clamp(max=1.)
    weight.mul_(factors[:, None])
    bias.mul_(factors)


def predict(model, example, mean, scale, kind, device):
    import torch
    model.eval()
    result = np.zeros((len(example.times), 4), np.float32)
    values = base.standardized(example, mean, scale, kind)
    with torch.inference_mode():
        for start, finish in base.segments(example.valid):
            for left_core in range(start, finish, 128):
                right_core = min(finish, left_core+128)
                left, right = max(start, left_core-62), min(finish, right_core+62)
                scores = torch.sigmoid(model(torch.from_numpy(values[left:right][None]).to(device)))[0].cpu().numpy()
                result[left_core:right_core] = scores[left_core-left:right_core-left]
    return result


def supervision_counts(rows):
    if not rows:
        return {"valid": [0]*4, "positiveMass": [0]*4}
    return {"valid": np.sum([r.mask.sum(axis=0) for r in rows], axis=0).astype(int).tolist(),
            "positiveMass": np.sum([(r.example.targets*r.mask).sum(axis=0) for r in rows], axis=0).tolist()}


def fit_model(train, auxiliary, validation, kind, seed, epochs, destination, device, contract_hash):
    import torch
    from .expanded_temporal_model import ExpandedTemporalConfig, ExpandedTemporalNetwork
    valid_groups = {e.group for e in validation}
    all_train = train + [r for rows in auxiliary.values() for r in rows]
    if not train or not validation or valid_groups & {r.example.group for r in all_train}:
        raise ValueError("overlapping or empty source fold")
    identity = {"trainIds": [r.example.id for r in train],
                "auxiliaryIds": {tier: [r.example.id for r in rows] for tier, rows in auxiliary.items()},
                "validationIds": [e.id for e in validation]}
    completed = destination / "completed.json"
    if completed.exists():
        meta = json.loads(completed.read_text())
        if (meta["contractSha256"] != contract_hash or any(meta[k] != v for k, v in identity.items())
                or meta["epochs"] != list(epochs) or meta["kind"] != kind or meta["seed"] != seed):
            raise ValueError("resume contract or identity changed")
        for name, digest in meta["artifacts"].items():
            if base.file_sha256(destination/name) != digest:
                raise ValueError("resume artifact changed")
        result = {}
        for epoch in epochs:
            with np.load(destination/f"predictions-{epoch}.npz", allow_pickle=False) as data:
                result[epoch] = {e.id: data[e.id] for e in validation}
        return result
    destination.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    mean, scale = base.fit_scaler([r.example for r in train])
    counts = supervision_counts(train)
    positive = np.asarray(counts["positiveMass"])
    positive_weight = np.minimum(20., np.sqrt((np.asarray(counts["valid"])-positive)/np.maximum(positive, 1.)))
    pos_weight = torch.tensor(positive_weight, dtype=torch.float32, device=device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model = ExpandedTemporalNetwork(ExpandedTemporalConfig(kind=kind)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    pools = {tier: make_chunks(rows, mean, scale, kind) for tier, rows in {"exact": train, **auxiliary}.items()}
    pools = {k: v for k, v in pools.items() if v}
    rngs = {tier: np.random.default_rng(np.random.SeedSequence([seed, i])) for i, tier in enumerate(HEAD_WEIGHTS)}
    exposure = {tier: hashlib.sha256() for tier in pools}
    outputs, history = {}, []
    steps = 0
    started = time.perf_counter()
    for epoch in range(1, max(epochs)+1):
        model.train()
        loss_sum = 0.
        batches = epoch_batches(pools["exact"], rngs["exact"])
        for exact_indexes in batches:
            optimizer.zero_grad(set_to_none=True)
            combined = None
            for stream, tier in enumerate(HEAD_WEIGHTS):
                if tier not in pools:
                    continue
                indexes = exact_indexes if tier == "exact" else rngs[tier].choice(len(pools[tier]), len(exact_indexes), replace=True, p=sampling_weights(pools[tier])).tolist()
                exposure[tier].update(json.dumps([epoch, steps, indexes], separators=(",", ":")).encode())
                # Domain-separated masks keep exact and draft stochastic exposure paired.
                torch.manual_seed(seed + steps*11 + stream*100000000)
                loss = batch_loss(model, pools[tier], indexes, pos_weight, HEAD_WEIGHTS[tier], device)
                combined = loss if combined is None else combined + loss
            if not torch.isfinite(combined):
                raise ValueError("non-finite training loss")
            combined.backward()
            clip_gradients(model, kind)
            optimizer.step()
            loss_sum += float(combined.detach().cpu())
            steps += 1
        history.append({"epoch": epoch, "loss": loss_sum/len(batches), "optimizerSteps": steps,
                        "exposureSha256": {tier: digest.hexdigest() for tier, digest in exposure.items()}})
        if epoch in epochs:
            outputs[epoch] = {e.id: predict(model, e, mean, scale, kind, device) for e in validation}
            np.savez_compressed(destination/f"predictions-{epoch}.npz", **outputs[epoch])
            state = {f"model::{k}": v.detach().cpu().numpy() for k, v in model.state_dict().items()}
            np.savez_compressed(destination/f"weights-{epoch}.npz", mean=mean, scale=scale, **state)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    base.write_json(completed, {"contractSha256": contract_hash, **identity, "kind": kind, "seed": seed,
        "epochs": epochs, "trainGroups": sorted({r.example.group for r in train}),
        "auxiliaryGroups": {tier: sorted({r.example.group for r in rows}) for tier, rows in auxiliary.items()},
        "validationGroups": sorted(valid_groups), "scalerTrainIds": identity["trainIds"],
        "positiveWeight": positive_weight.tolist(), "supervisedCounts": {tier: supervision_counts(rows) for tier, rows in {"exact": train, **auxiliary}.items()},
        "parameters": sum(p.numel() for p in model.parameters()), "history": history, "optimizerSteps": steps,
        "exposureSha256": {tier: digest.hexdigest() for tier, digest in exposure.items()},
        "wallSeconds": time.perf_counter()-started,
        "peakAllocatedCudaBytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
        "artifacts": {f"{stem}-{epoch}.npz": base.file_sha256(destination/f"{stem}-{epoch}.npz") for epoch in epochs for stem in ("weights", "predictions")}})
    return outputs


def decoder_candidates():
    return [{"smoothing": s, "enter": e, "minimum": m, "boundary": b}
            for s in (.5, 1.) for e in (.2, .35, .5, .65, .8, .9) for m in (.25, 1.) for b in (False, True)]


def select_candidate(candidates):
    eligible = [c for c in candidates if c["innerR_core"] >= .95]
    pool = eligible or candidates
    best = max(pool, key=lambda c: c["innerF1_padP_coreR"])
    return {**best, "recallEligibilityPassed": bool(eligible), "recallEligibilityFloor": .95}


def choose_settings(examples, predictions):
    candidates = []
    for epoch, probabilities in sorted(predictions.items()):
        for settings in decoder_candidates():
            rows = [RecordingIntervals(e.id, "development", e.duration, e.truth,
                        tuple(base.decode(e, probabilities[e.id], settings)), e.ignored) for e in examples]
            score = evaluate_f1_pad_p_core_r(rows, [2.], 3.)[0]
            candidates.append({"epoch": epoch, "decoder": settings, "innerF1_padP_coreR": score["F1_padP_coreR"], "innerR_core": score["R_core"]})
    return select_candidate(candidates)


def run_study(manifest_path, output, device):
    import torch
    from .neural_evaluation import evaluate_predictions
    if (output/"report.json").exists():
        raise FileExistsError("completed study exists")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    data = load_data(manifest_path)
    groups = sorted({r.example.group for r in data["exact"]})
    files = ("neural_expanded_development.py", "expanded_temporal_model.py", "neural_development.py", "compact_temporal_model.py", "neural_evaluation.py", "crop_evaluation.py", "decoder.py", "features.py", "config.py", "schema.py", "metrics.py")
    contract = {"schemaVersion": 1, "manifestSha256": base.file_sha256(manifest_path), "cohorts": COHORTS, "kinds": KINDS,
        "seeds": base.SEEDS, "checkpointEpochs": EPOCHS, "groups": groups, "primaryMetric": "F1_padP_coreR",
        "targetPaddingSeconds": 2, "joinGapSeconds": 3, "paddingSweep": [0, 1, 2, 3], "decoderCandidates": decoder_candidates(),
        "selection": {"recallEligibilityFloor": .95, "ranking": "F1_padP_coreR descending after eligibility; highest F1 flagged infeasible if no eligible candidate", "scope": "pooled exact-only inner source-held predictions"},
        "training": {"headNames": ["live", "serve", "end", "keep"], "headWeights": HEAD_WEIGHTS, "loss": "each head mean over its own valid tick count; fixed tier multiplier not canceled",
            "optimizer": "AdamW", "learningRate": .001, "weightDecay": .0001, "batchSize": 16, "coreTicks": 128, "haloTicks": 62,
            "scalerAndPositiveWeightFit": "exact training only", "positiveWeight": "sqrt negative/positive capped20",
            "sampling": "group/recording-balanced replacement; same exact pool, RNG and batches across cohorts; auxiliary same batch count and size per step with separate RNG streams",
            "dropout": "domain-separated unique per-step exact/draft/coverage torch seeds", "gradientClip": "norm1 globally for TCN; independently per output row for linear",
            "draft": "live only, mask within1second of approximate rally/ignored boundaries; .5 loss weight", "coverage": "actual keepTargets, no extra padding; gameWindow minus ignored; keep-only .25 weight",
            "exactKeep": "canonical2s padding and strict<3s join, then subtract ignored", "tf32": False,
            "decoder": "first three logits only; auxiliary keep does not directly set primary cuts"},
        "evaluationPopulation": {"records": len(data["exact"]), "rallies": sum(len(r.example.truth) for r in data["exact"]), "groups": groups},
        "screen": {"meanF1Gain": .02, "meanRCoreMinimumDelta": -.005, "positiveSeedCount": 2, "positiveGroupMajority": True},
        "protectedTestOpened": False, "productionPromotionAllowed": False,
        "code": {name: base.file_sha256(Path(__file__).with_name(name)) for name in files},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": device, "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None}}
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    prereg = output/"preregistration.json"
    if prereg.exists():
        if json.loads(prereg.read_text())["sha256"] != contract_hash:
            raise ValueError("preregistration changed; refuse overwrite")
    else:
        base.write_json(prereg, {"sha256": contract_hash, "contract": contract})
    results = []
    for cohort in COHORTS:
        for kind in KINDS:
            for seed in base.SEEDS:
                outer_rows, selections = [], []
                for outer_index, outer in enumerate(groups):
                    fitting = [r for r in data["exact"] if r.example.group != outer]
                    held = [r.example for r in data["exact"] if r.example.group == outer]
                    inner_predictions = {epoch: {} for epoch in EPOCHS}
                    for inner_index, inner in enumerate(g for g in groups if g != outer):
                        training = [r for r in fitting if r.example.group != inner]
                        validating = [r.example for r in fitting if r.example.group == inner]
                        auxiliary = auxiliary_for_fold(data, cohort, {outer, inner})
                        location = output/"fits"/cohort/kind/str(seed)/f"outer-{outer_index}"/f"inner-{inner_index}"
                        print(f"FIT {cohort} {kind} seed={seed} outer={outer} inner={inner}", flush=True)
                        predictions = fit_model(training, auxiliary, validating, kind, seed, EPOCHS, location, device, contract_hash)
                        for epoch in EPOCHS:
                            inner_predictions[epoch].update(predictions[epoch])
                    selected = choose_settings([r.example for r in fitting], inner_predictions)
                    epoch = selected["epoch"]
                    location = output/"fits"/cohort/kind/str(seed)/f"outer-{outer_index}"/"refit"
                    print(f"REFIT {cohort} {kind} seed={seed} outer={outer} epoch={epoch} feasible={selected['recallEligibilityPassed']}", flush=True)
                    probabilities = fit_model(fitting, auxiliary_for_fold(data, cohort, {outer}), held, kind, seed, (epoch,), location, device, contract_hash)[epoch]
                    outer_rows.extend(e.row(base.decode(e, probabilities[e.id], selected["decoder"])) for e in held)
                    selections.append({"heldSourceGroup": outer, **selected})
                evaluation = evaluate_predictions(outer_rows)
                result = {"cohort": cohort, "kind": kind, "seed": seed, "selections": selections, "evaluation": evaluation,
                    "predictions": [{**r, **{k: [i.to_dict() for i in r[k]] for k in ("rallies", "ignoredIntervals", "predictions")}} for r in outer_rows]}
                base.write_json(output/f"result-{cohort}-{kind}-{seed}.json", result)
                results.append(result)
                print(f"RESULT {cohort} {kind} seed={seed} F1_padP_coreR={evaluation['objective']:.6f}", flush=True)
    report = {"schemaVersion": 1, "contractSha256": contract_hash, "manifestSha256": contract["manifestSha256"],
              "status": "completed-expanded-development-screen", "records": len(data["exact"]), "sourceGroups": groups,
              "protectedTestOpened": False, "productionPromotionAllowed": False, "results": results}
    base.write_json(output/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_study(args.manifest.resolve(), args.output.resolve(), args.device)


if __name__ == "__main__":
    main()
