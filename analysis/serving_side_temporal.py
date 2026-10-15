"""Source-held serving-side research over an immutable frozen embedding bank.

No recording names or private locations are defaults. Only the temporal head is
trained; the supplied image encoder remains frozen. Public summaries use indexes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .compact_temporal_model import CompactTemporalConfig, CompactTemporalNetwork


SEEDS = (3407, 1729, 20260918)
EPOCHS = (5, 15, 30, 60)
OFFSETS = np.arange(-3., 3.01, .5)
THRESHOLDS = tuple(float(v) for v in np.round(np.arange(.05, .951, .025), 6))
ARMS = ("temporal", "anchor-only")
CODE = ("analysis/serving_side_temporal.py", "analysis/compact_temporal_model.py",
        "scripts/train-serving-side-temporal.py")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def immutable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(canonical(read(path)) == canonical(value), "Immutable artifact differs")
        return
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reference(path):
    return {"path": str(Path(path)), "sha256": digest(path)}


def verified(ref):
    require(digest(ref["path"]) == ref["sha256"], "Registered input changed")
    return Path(ref["path"])


def merge_intervals(spans, duration):
    result = []
    for span in sorted(spans, key=lambda s: float(s["start"])):
        left, right = max(0., float(span["start"])), min(duration, float(span["end"]))
        require(math.isfinite(left) and math.isfinite(right), "Nonfinite exclusion")
        if right <= left:
            continue
        if result and left <= result[-1][1]:
            result[-1] = (result[-1][0], max(right, result[-1][1]))
        else:
            result.append((left, right))
    return result


def sample_window(times, tokens, anchor, duration, excluded):
    """Only sample within the anchor's contiguous valid half-open segment."""
    require(tokens.shape == (len(times), 4, 960) and len(times) > 0
            and np.isfinite(times).all() and np.isfinite(tokens).all()
            and np.all(np.diff(times) > 0), "Invalid embedding sequence")
    require(0 <= anchor < duration, "Anchor outside video")
    left, right = 0., duration
    for start, end in merge_intervals(excluded, duration):
        require(not start <= anchor < end, "Anchor is in excluded footage")
        if end <= anchor:
            left = max(left, end)
        if start > anchor:
            right = min(right, start)
    valid_indexes = np.flatnonzero((times >= left) & (times < right))
    requested = anchor + OFFSETS
    values = np.zeros((len(OFFSETS), 4, 960), np.float32)
    scalars = np.zeros((len(OFFSETS), 3), np.float32)
    scalars[:, 1] = OFFSETS / 3.
    if len(valid_indexes):
        valid_times = times[valid_indexes]
        high = np.searchsorted(valid_times, requested).clip(0, len(valid_times)-1)
        low = (high-1).clip(0, len(valid_times)-1)
        chosen = np.where(abs(valid_times[low]-requested) <= abs(valid_times[high]-requested), low, high)
        actual = valid_times[chosen]
        available = ((requested >= left) & (requested < right)
                     & (abs(actual-requested) <= .250001))
        values[available] = tokens[valid_indexes[chosen[available]]]
        scalars[available, 0] = 1.
        scalars[available, 2] = (actual[available]-requested[available]) / .25
    require(scalars[6, 0] == 1, "Anchor has no qualified embedding")
    return np.concatenate((values.reshape(len(OFFSETS), -1), scalars), axis=1)


def arm_values(values, arm):
    require(arm in ARMS, "Unregistered arm")
    if arm == "temporal":
        return values
    result = values.copy()
    result[:, :6, :3840] = 0.
    result[:, 7:, :3840] = 0.
    result[:, :6, 3840] = 0.
    result[:, 7:, 3840] = 0.
    result[:, :6, 3842] = 0.
    result[:, 7:, 3842] = 0.
    return result


@dataclass(frozen=True)
class HeadConfig:
    token_dimension: int = 960
    token_count: int = 4
    projection_dimension: int = 16
    hidden_dimension: int = 64
    dropout: float = .1


class ServingSideTemporalNetwork(nn.Module):
    def __init__(self, config=HeadConfig()):
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(config.token_dimension)
        self.projection = nn.Linear(config.token_dimension, config.projection_dimension)
        self.temporal = CompactTemporalNetwork(CompactTemporalConfig(
            input_dimension=config.token_count*config.projection_dimension+3,
            hidden_dimension=config.hidden_dimension, kernel_size=3,
            dilations=(1, 2, 3), dropout=config.dropout))
        self.temporal.head = nn.Linear(config.hidden_dimension, 1)
        self.float()

    def forward(self, values):
        require(values.ndim == 3 and values.shape[1:] == (13, 3843)
                and values.dtype == torch.float32, "Head requires FP32 [B,13,3843]")
        tokens = values[..., :3840].reshape(-1, 13, 4, 960)
        projected = F.relu(self.projection(self.norm(tokens)))
        projected = projected * values[..., 3840:3841, None]
        combined = torch.cat((projected.flatten(-2), values[..., 3840:]), dim=-1)
        return self.temporal(combined)[:, 6, 0]


def configure_torch():
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def metrics(truth, predicted):
    truth, predicted = np.asarray(truth, bool), np.asarray(predicted, bool)
    tp = int(np.sum(truth & predicted))
    tn = int(np.sum(~truth & ~predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    def ratio(a, b):
        return a/b if b else None
    near_recall, far_recall = ratio(tp, tp+fn), ratio(tn, tn+fp)
    return {"rows": len(truth), "truePositive": tp, "trueNegative": tn,
            "falsePositive": fp, "falseNegative": fn,
            "accuracy": ratio(tp+tn, len(truth)),
            "balancedAccuracy": (near_recall+far_recall)/2 if near_recall is not None and far_recall is not None else None,
            "macroF1": (ratio(2*tp, 2*tp+fp+fn) or 0.)/2 + (ratio(2*tn, 2*tn+fp+fn) or 0.)/2,
            "nearPrecision": ratio(tp, tp+fp), "nearRecall": near_recall,
            "farPrecision": ratio(tn, tn+fn), "farRecall": far_recall}


def evaluate(rows, predicted):
    truth = np.asarray([r["label"] for r in rows])
    groups, environments = {}, {}
    for key, result in (("sourceGroup", groups), ("environment", environments)):
        for group in sorted({r[key] for r in rows}):
            ix = [i for i, row in enumerate(rows) if row[key] == group]
            result[group] = metrics(truth[ix], np.asarray(predicted)[ix])
    balanced = [m["balancedAccuracy"] for m in groups.values() if m["balancedAccuracy"] is not None]
    require(len(balanced) == len(groups), "Source fold lacks a near/far class")
    return {"pooled": metrics(truth, predicted), "sourceGroupMacroBalancedAccuracy": float(np.mean(balanced)),
            "worstSourceGroupBalancedAccuracy": float(min(balanced)),
            "bySourceGroup": groups, "byEnvironment": environments}


def select_operating_point(rows, scores):
    require(set(scores) == set(EPOCHS), "Incomplete checkpoint scores")
    candidates = []
    for epoch in EPOCHS:
        require(len(scores[epoch]) == len(rows) and np.isfinite(scores[epoch]).all(), "Invalid inner scores")
        for threshold in THRESHOLDS:
            ev = evaluate(rows, scores[epoch] >= threshold)
            rank = (ev["sourceGroupMacroBalancedAccuracy"], ev["pooled"]["balancedAccuracy"],
                    ev["pooled"]["macroF1"], ev["worstSourceGroupBalancedAccuracy"],
                    -epoch, -abs(threshold-.5), -threshold)
            candidates.append({"epoch": epoch, "threshold": threshold, "evaluation": ev, "rank": rank})
    selected = max(candidates, key=lambda c: c["rank"])
    return {"epoch": selected["epoch"], "threshold": selected["threshold"],
            "evaluation": selected["evaluation"]}, candidates


def fold_indices(rows, excluded):
    require(len(set(excluded)) == len(excluded), "Duplicate excluded group")
    train = np.asarray([i for i, r in enumerate(rows) if r["sourceGroup"] not in excluded], np.int64)
    held = np.asarray([i for i, r in enumerate(rows) if r["sourceGroup"] in excluded], np.int64)
    require(len(train) and (not excluded or len(held)), "Empty fold")
    require({rows[i]["sourceGroup"] for i in train}.isdisjoint(excluded), "Fold leakage")
    require({rows[i]["label"] for i in train} == {0, 1}, "Training fold lacks a class")
    return train, held


def training_weights(rows):
    counts = {g: sum(r["sourceGroup"] == g for r in rows) for g in {r["sourceGroup"] for r in rows}}
    labels = np.asarray([r["label"] for r in rows])
    positives = int(labels.sum())
    require(0 < positives < len(rows), "Both classes required")
    classes = np.sqrt(len(rows)/(2*np.asarray([len(rows)-positives, positives], np.float64)))
    weights = np.asarray([classes[r["label"]]/counts[r["sourceGroup"]] for r in rows], np.float32)
    return weights/weights.mean()


def load_weights(path, device):
    model = ServingSideTemporalNetwork().to(device)
    with np.load(path, allow_pickle=False) as archive:
        state = {k.removeprefix("model::"): torch.from_numpy(archive[k].copy()).to(device)
                 for k in archive.files if k.startswith("model::")}
    model.load_state_dict(state, strict=True)
    return model.eval()


def predict(model, values):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(values), 64):
            result.append(torch.sigmoid(model(values[start:start+64])).cpu().numpy())
    return np.concatenate(result) if result else np.empty(0, np.float32)


def fit(rows, values, excluded, seed, folder, contract_hash, *, epochs=EPOCHS):
    train, held = fold_indices(rows, excluded)
    folder = Path(folder)
    identity = {"contractSha256": contract_hash, "seed": seed, "excludedGroups": sorted(excluded),
                "trainingRows": train.tolist(), "heldRows": held.tolist(), "epochs": list(epochs)}
    completed = folder/"completed.json"
    if completed.exists():
        meta = read(completed)
        require(all(meta[k] == v for k, v in identity.items()), "Fit resume identity differs")
        for name, sha in meta["artifacts"].items():
            require(digest(folder/name) == sha, "Fit checkpoint or scores changed")
        with np.load(folder/"scores.npz", allow_pickle=False) as archive:
            return {epoch: archive[str(epoch)].copy() for epoch in epochs}
    require(not folder.exists(), "Refusing incomplete fit; preserve it before a deliberate retry")
    folder.mkdir(parents=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = values.device
    model = ServingSideTemporalNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    train_ix = torch.from_numpy(train).to(device)
    held_values = values[torch.from_numpy(held).to(device)]
    labels = torch.tensor([rows[i]["label"] for i in train], dtype=torch.float32, device=device)
    weights = torch.from_numpy(training_weights([rows[i] for i in train])).to(device)
    rng = np.random.default_rng(seed)
    history, outputs = [], {}
    started = time.monotonic()
    for epoch in range(1, max(epochs)+1):
        model.train()
        losses = []
        for start_indexes in np.array_split(rng.permutation(len(train)), math.ceil(len(train)/32)):
            ix = torch.from_numpy(start_indexes).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values[train_ix[ix]])
            loss = (F.binary_cross_entropy_with_logits(logits, labels[ix], reduction="none")*weights[ix]).mean()
            require(bool(torch.isfinite(loss)), "Nonfinite temporal loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            require(bool(torch.isfinite(norm)), "Nonfinite temporal gradients")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        if epoch in epochs:
            outputs[epoch] = predict(model, held_values)
            state = {"model::"+k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
            np.savez_compressed(folder/f"weights-{epoch}.npz", **state)
    np.savez_compressed(folder/"scores.npz", **{str(k): v for k, v in outputs.items()})
    artifacts = {p.name: digest(p) for p in folder.glob("*.npz")}
    immutable(completed, {**identity, "history": history, "wallSeconds": time.monotonic()-started,
                          "parameters": sum(p.numel() for p in model.parameters()), "artifacts": artifacts})
    return outputs


def prepare(inputs_path, root, repo):
    """Freeze reviewed labels and valid context before any held-label selection."""
    root, repo = Path(root), Path(repo)
    require(not (root/"registration.json").exists(), "Registration already exists")
    inputs = read(inputs_path)
    ledger = read(inputs["ledger"])
    aliases = ledger["originalToAlias"]
    source = read(inputs["dataset"])
    require(source["scope"] == "development" and len(source["rows"]) == 1027
            and source["dataPolicy"]["protectedTestIncluded"] is False, "Unexpected serving population")
    rows = [r for r in source["rows"] if r["environment"] != "beach"]
    require(len(rows) == 968 and len({r["recordingId"] for r in rows}) == 26
            and len({r["sourceGroup"] for r in rows}) == 8, "Non-beach population changed")
    reviews = read(inputs["reviews"])
    corrections = read(inputs["corrections"])
    effective = {**reviews["decisions"], **corrections["corrections"]}
    require(all(effective.get(r["rallyId"]) == r["decision"] and r["decision"] in ("near", "far")
                and r["label"] == int(r["decision"] == "near") for r in rows), "Effective serving labels changed")
    catalog = {r["id"]: r for r in read(inputs["catalog"])["records"]}
    quality = {r["recordingId"]: r["intervals"] for r in read(inputs["quality"])["records"]}
    image_group_to_side_groups = {}
    for row in rows:
        image_group_to_side_groups.setdefault(catalog[row["recordingId"]]["sourceGroup"], set()).add(row["sourceGroup"])
    require(all(len(groups) == 1 for groups in image_group_to_side_groups.values()),
            "An image source group crosses serving-side folds")
    for name in ("humanLabelCorrections", "sourceQualityExclusions"):
        verified(source["sources"][name])
    baseline = {r["rallyId"]: r for r in read(inputs["baseline"])["selectedPredictions"]}
    bank, receipts, prepared_rows, sequences, masked = {}, {}, [], [], 0
    for row in rows:
        recording = row["recordingId"]
        require(row["sourceSplit"] != "test" and not catalog[recording].get("protected")
                and row["sourceGroup"] not in source["dataPolicy"]["protectedSourceGroupsExcluded"], "Protected source")
        require(recording in aliases and row["sourceGroup"] in aliases, "Missing source ledger mapping")
        require(baseline[row["rallyId"]]["recordingId"] == recording
                and baseline[row["rallyId"]]["decision"] == row["decision"], "Baseline identity mismatch")
        if recording not in bank:
            receipt_path = Path(inputs["features"])/(recording+".json")
            receipt = read(receipt_path)
            require(receipt["id"] == recording and receipt["sourceGroup"] == catalog[recording]["sourceGroup"]
                    and receipt["labelsUsed"] is False and receipt["encoder"]["sha256"] == inputs["encoderSha256"],
                    "Embedding encoder or provenance changed")
            feature_path = verified(receipt["output"])
            with np.load(feature_path, allow_pickle=False) as archive:
                bank[recording] = archive["timestamps"].copy(), archive["tokens"].astype(np.float32)
            receipts[aliases[recording]] = {"receipt": reference(receipt_path), "features": reference(feature_path)}
        excluded = catalog[recording].get("ignoredIntervals", []) + quality.get(recording, [])
        sequence = sample_window(*bank[recording], float(row["serveAnchor"]), float(catalog[recording]["durationSeconds"]), excluded)
        masked += int(np.sum(sequence[:, 3840] == 0))
        sequences.append(sequence)
        prepared_rows.append({"rowIndex": len(prepared_rows), "recordingIndex": aliases[recording],
                              "sourceGroup": aliases[row["sourceGroup"]], "environment": row["environment"],
                              "label": row["label"], "baselineNear": baseline[row["rallyId"]]["prediction"] == "near"})
    require(digest(inputs["encoder"]) == inputs["encoderSha256"], "Frozen encoder bytes changed")
    root.mkdir(parents=True, exist_ok=True)
    array = np.stack(sequences)
    np.savez_compressed(root/"windows.npz", values=array)
    immutable(root/"rows.json", prepared_rows)
    data = {"windows": reference(root/"windows.npz"), "rows": reference(root/"rows.json")}
    contract = {"kind": "serving-side-temporal-frozen-large-v1", "seeds": list(SEEDS), "epochs": list(EPOCHS),
                "offsetSeconds": OFFSETS.tolist(), "thresholds": list(THRESHOLDS), "arms": list(ARMS),
                "head": asdict(HeadConfig()), "encoderSha256": inputs["encoderSha256"],
                "inputs": {k: reference(inputs[k]) for k in ("dataset", "baseline", "catalog", "reviews", "corrections", "quality", "encoder")},
                "sourceCode": {name: reference(repo/name) for name in CODE}, "data": data, "receipts": receipts,
                "rows": len(prepared_rows), "maskedContextFrames": masked,
                "sourceGroups": sorted({r["sourceGroup"] for r in prepared_rows}),
                "selection": "nested source-group macro BA; train-only weights; fixed threshold grid",
                "encoderIsFrozenAndMayHaveSeenHeldGroupPixels": True}
    immutable(root/"registration.json", {"contract": contract, "sha256": canonical(contract)})
    return {"rows": len(prepared_rows), "sourceGroups": len(contract["sourceGroups"]), "maskedContextFrames": masked,
            "contractSha256": canonical(contract), "windowsShape": list(array.shape)}


def load_registration(root):
    reg = read(Path(root)/"registration.json")
    require(canonical(reg["contract"]) == reg["sha256"], "Registration hash changed")
    for ref in reg["contract"]["sourceCode"].values():
        verified(ref)
    for ref in reg["contract"]["inputs"].values():
        verified(ref)
    rows = read(verified(reg["contract"]["data"]["rows"]))
    with np.load(verified(reg["contract"]["data"]["windows"]), allow_pickle=False) as archive:
        values = archive["values"].copy()
    require(values.shape == (len(rows), 13, 3843) and np.isfinite(values).all(), "Prepared windows changed")
    return reg, rows, values


def progress(root, **values):
    path = Path(root)/"progress.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)
    print(json.dumps(values, allow_nan=False), flush=True)


def run(root, device="cuda"):
    configure_torch()
    root = Path(root)
    reg, rows, raw = load_registration(root)
    groups = reg["contract"]["sourceGroups"]
    folds = list(itertools.combinations(groups, 2)) + [(g,) for g in groups]
    completed, total = 0, len(folds)*len(SEEDS)*len(ARMS)
    for arm in ARMS:
        values = torch.from_numpy(arm_values(raw, arm)).to(device)
        for seed in SEEDS:
            for excluded in folds:
                key = "+".join(excluded)
                folder = root/"fits"/arm/str(seed)/key
                progress(root, phase="fitting", arm=arm, seed=seed, excludedGroups=list(excluded),
                         completedFits=completed, totalFits=total)
                fit(rows, values, excluded, seed, folder, reg["sha256"])
                completed += 1
        del values
    progress(root, phase="nested-fits-complete", completedFits=completed, totalFits=total)


def saved_scores(root, arm, seed, excluded):
    folder = Path(root)/"fits"/arm/str(seed)/"+".join(sorted(excluded))
    meta = read(folder/"completed.json")
    require(meta["contractSha256"] == read(Path(root)/"registration.json")["sha256"]
            and meta["seed"] == seed and meta["excludedGroups"] == sorted(excluded)
            and meta["epochs"] == list(EPOCHS), "Saved fit identity differs")
    for name, sha in meta["artifacts"].items():
        require(digest(folder/name) == sha, "Saved fit artifact changed")
    with np.load(folder/"scores.npz", allow_pickle=False) as archive:
        return np.asarray(meta["heldRows"], np.int64), {e: archive[str(e)].copy() for e in EPOCHS}


def nested_predictions(root, rows, groups, arm, seed):
    outer_probabilities = np.full(len(rows), np.nan)
    thresholds = np.full(len(rows), np.nan)
    folds = []
    for outer in groups:
        train, held = fold_indices(rows, [outer])
        local = {i: j for j, i in enumerate(train)}
        scores = {epoch: np.full(len(train), np.nan) for epoch in EPOCHS}
        for inner in groups:
            if inner == outer:
                continue
            indexes, pair = saved_scores(root, arm, seed, (outer, inner))
            for epoch in EPOCHS:
                for index, value in zip(indexes, pair[epoch], strict=True):
                    if rows[index]["sourceGroup"] == inner:
                        scores[epoch][local[index]] = value
        choice, candidates = select_operating_point([rows[i] for i in train], scores)
        immutable(Path(root)/"selection"/arm/str(seed)/(outer+".json"),
                  {"outerExcludedGroup": outer, "innerRows": train.tolist(), "selected": choice,
                   "candidates": candidates})
        indexes, outer_scores = saved_scores(root, arm, seed, (outer,))
        require(np.array_equal(indexes, held), "Outer prediction order differs")
        outer_probabilities[held] = outer_scores[choice["epoch"]]
        thresholds[held] = choice["threshold"]
        folds.append({"sourceGroup": outer, "epoch": choice["epoch"], "threshold": choice["threshold"],
                      "heldRows": len(held), "innerSourceGroups": [g for g in groups if g != outer]})
    require(np.isfinite(outer_probabilities).all() and np.isfinite(thresholds).all(), "Missing outer predictions")
    return outer_probabilities, thresholds, folds


def summarize(root, report_path, device="cuda"):
    configure_torch()
    root = Path(root)
    reg, rows, raw = load_registration(root)
    groups = reg["contract"]["sourceGroups"]
    truth = np.asarray([r["label"] for r in rows], bool)
    baseline = np.asarray([r["baselineNear"] for r in rows])
    baseline_ev = evaluate(rows, baseline)
    arms, private_scores = {}, {}
    for arm in ARMS:
        arm_seeds, probabilities, thresholds = [], [], []
        for seed in SEEDS:
            p, t, folds = nested_predictions(root, rows, groups, arm, seed)
            probabilities.append(p)
            thresholds.append(t)
            ev = evaluate(rows, p >= t)
            arm_seeds.append({"seed": seed, "evaluation": ev, "folds": folds})
            private_scores[f"{arm}-{seed}-probabilities"] = p
            private_scores[f"{arm}-{seed}-thresholds"] = t
        ensemble_p, ensemble_t = np.mean(probabilities, axis=0), np.mean(thresholds, axis=0)
        predicted = ensemble_p >= ensemble_t
        arms[arm] = {"seeds": arm_seeds, "ensemble": evaluate(rows, predicted),
                     "meanSeedMacroBalancedAccuracy": float(np.mean([s["evaluation"]["sourceGroupMacroBalancedAccuracy"] for s in arm_seeds])),
                     "meanSeedPooledBalancedAccuracy": float(np.mean([s["evaluation"]["pooled"]["balancedAccuracy"] for s in arm_seeds])),
                     "minimumSeedWorstGroupBalancedAccuracy": float(min(s["evaluation"]["worstSourceGroupBalancedAccuracy"] for s in arm_seeds)),
                     "pairedErrors": {"baselineOnlyCorrect": int(np.sum((baseline == truth)&(predicted != truth))),
                                      "headOnlyCorrect": int(np.sum((baseline != truth)&(predicted == truth))),
                                      "bothCorrect": int(np.sum((baseline == truth)&(predicted == truth))),
                                      "bothWrong": int(np.sum((baseline != truth)&(predicted != truth)))}}
    winner = max(ARMS, key=lambda a: (arms[a]["meanSeedMacroBalancedAccuracy"],
                                    arms[a]["meanSeedPooledBalancedAccuracy"],
                                    arms[a]["minimumSeedWorstGroupBalancedAccuracy"], a == "anchor-only"))
    values = torch.from_numpy(arm_values(raw, winner)).to(device)
    final = []
    for seed in SEEDS:
        oof = {epoch: np.full(len(rows), np.nan) for epoch in EPOCHS}
        for group in groups:
            indexes, scores = saved_scores(root, winner, seed, (group,))
            for epoch in EPOCHS:
                oof[epoch][indexes] = scores[epoch]
        selected, candidates = select_operating_point(rows, oof)
        immutable(root/"final"/str(seed)/"selection.json", {"selected": selected, "candidates": candidates})
        folder = root/"final"/str(seed)/"fit"
        fit(rows, values, (), seed, folder, reg["sha256"], epochs=(selected["epoch"],))
        weights = folder/f"weights-{selected['epoch']}.npz"
        model = load_weights(weights, device)
        probe = predict(model, values[:8])
        restored = predict(load_weights(weights, device), values[:8])
        require(np.array_equal(probe, restored), "Checkpoint reload inference differs")
        final.append({"seed": seed, "epoch": selected["epoch"], "threshold": selected["threshold"],
                      "weightsSha256": digest(weights), "weightsFileBytes": weights.stat().st_size,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "float32WeightBytes": 4*sum(p.numel() for p in model.parameters()),
                      "reloadPredictionParity": True})
    np.savez_compressed(root/"nested-predictions.npz", **private_scores)
    qualifies = (arms[winner]["meanSeedMacroBalancedAccuracy"] > baseline_ev["sourceGroupMacroBalancedAccuracy"]
                 and arms[winner]["minimumSeedWorstGroupBalancedAccuracy"] >= baseline_ev["worstSourceGroupBalancedAccuracy"])
    result = {"schemaVersion": 1, "kind": "serving-side-temporal-head-results-v1", "studyIndex": "private-reference-0225",
              "contractSha256": reg["sha256"], "encoderSha256": reg["contract"]["encoderSha256"],
              "rows": len(rows), "recordings": len({r["recordingIndex"] for r in rows}), "sourceGroups": len(groups),
              "maskedContextFrames": reg["contract"]["maskedContextFrames"], "baseline": baseline_ev, "arms": arms,
              "selectedArm": winner, "meetsSpecialistFollowupGuardrails": qualifies, "finalHeads": final,
              "historicalLinearControl": {"source": "docs/research/distilled-large-specialist-head-screen-direct-l2-1.json",
                                          "pooledBalancedAccuracy": .8283469945355191,
                                          "sourceGroupMacroBalancedAccuracy": .8435765617655291,
                                          "developmentSelectedAndNotNested": True},
              "limitations": ["Development-selected arm, not a fresh protected-test estimate.",
                              "Frozen encoder may have seen held groups' unlabeled imagery and had historical rally-based selection.",
                              "Human-reviewed anchors; inferred-start, serve-gate and native-device accuracy not measured."]}
    immutable(root/"summary.json", result)
    immutable(report_path, result)
    progress(root, phase="complete", selectedArm=winner, meanMacroBA=arms[winner]["meanSeedMacroBalancedAccuracy"],
             baselineMacroBA=baseline_ev["sourceGroupMacroBalancedAccuracy"], qualifies=qualifies)
    return result
