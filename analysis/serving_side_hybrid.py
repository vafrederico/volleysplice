"""Nested development research combining specialist features and frozen video tokens."""
from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import serving_side_temporal as prior
from .private_ledger import private_value
from .serving_side_development_eval import tied_recording_ranks
from .serving_side_v2 import fit_logistic, LogisticModel


ARMS = ("hybrid", "specialist-only")
MULTIPLIERS = (0., .25, .5, 1.)
STUDY_INDEX = "private-reference-0231"
CODE = ("analysis/serving_side_hybrid.py", "analysis/serving_side_temporal.py",
        "analysis/compact_temporal_model.py", "analysis/serving_side_v2.py",
        "analysis/serving_side_specialist.py", "analysis/serving_side_development_eval.py",
        "analysis/private_ledger.py", "scripts/train-serving-side-hybrid.py")
require, read, digest, canonical = prior.require, prior.read, prior.digest, prior.canonical
immutable, reference, verified = prior.immutable, prior.reference, prior.verified


def specialist_features(dataset, baseline, rows, aliases):
    original = [r for r in dataset["rows"] if r["environment"] != "beach"]
    require(len(original) == len(rows), "Specialist population differs")
    for i, (a, b) in enumerate(zip(original, rows, strict=True)):
        require(b["rowIndex"] == i and aliases[a["recordingId"]] == b["recordingIndex"]
                and aliases[a["sourceGroup"]] == b["sourceGroup"] and a["label"] == b["label"],
                "Specialist row identity differs")
        require(a["decision"] in ("near", "far") and a["sourceSplit"] != "test",
                "Unqualified specialist row")
    model = baseline["finalModel"]
    require(model["configuration"] == "192x108-r4c6"
            and model["featureFamily"] == "v2-plus-flight-recording-rank"
            and model["parameters"]["l2"] == .1, "Incumbent feature contract differs")
    v2 = dataset["v2FeatureNames"]
    flight = next(c["featureNames"] for c in dataset["configurations"] if c["name"] == model["configuration"])
    names = ["v2:"+n for n in v2] + ["flight:"+n for n in flight]
    require(len(v2) == 82 and len(flight) == 155 and names == model["featureNames"],
            "Incumbent ordered feature signature differs")
    court = np.asarray([[r["v2RecordingRankFeatures"][n] for n in v2] for r in original], np.float64)
    raw = np.asarray([[r["configurations"][model["configuration"]][n] for n in flight] for r in original], np.float64)
    require(np.isfinite(court).all() and np.isfinite(raw).all(), "Nonfinite specialist inputs")
    values = np.column_stack((court, tied_recording_ranks(original, raw)))
    require(values.shape == (len(rows), 237) and np.all((values >= 0)&(values <= 1)),
            "Invalid recording-rank matrix")
    return values, names


class HybridHead(nn.Module):
    """The linear skip and its scaler are buffers fitted outside the held groups."""
    def __init__(self, base: LogisticModel, arm="hybrid"):
        super().__init__()
        require(arm in ARMS, "Unregistered hybrid arm")
        self.arm = arm
        for name in ("mean", "scale", "weights"):
            self.register_buffer("base_"+name, torch.tensor(getattr(base, name), dtype=torch.float32))
        self.register_buffer("base_bias", torch.tensor(base.bias, dtype=torch.float32))
        self.visual = prior.ServingSideTemporalNetwork()
        self.visual.temporal.head = nn.Identity()
        self.specialist = nn.Sequential(nn.Linear(237, 32), nn.ReLU())
        self.correction = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Dropout(.1), nn.Linear(64, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.float()

    def components(self, windows, features):
        require(features.ndim == 2 and features.shape == (len(windows), 237)
                and features.dtype == torch.float32, "Hybrid requires FP32 specialist matrix")
        z = (features-self.base_mean)/self.base_scale
        base = (z @ self.base_weights + self.base_bias).clamp(-30., 30.)
        if self.arm == "hybrid":
            tokens = windows[..., :3840].reshape(-1, 13, 4, 960)
            projected = F.relu(self.visual.projection(self.visual.norm(tokens)))
            projected = projected * windows[..., 3840:3841, None]
            inputs = torch.cat((projected.flatten(-2), windows[..., 3840:]), dim=-1)
            temporal = self.visual.temporal(inputs)[:, 6]
        else:
            temporal = z.new_zeros((len(z), 64))
        fused = torch.cat((temporal, self.specialist(z)), dim=1)
        residual = 3.*torch.tanh(self.correction(fused)[:, 0])
        return base, residual

    def forward(self, windows, features, multiplier=1.):
        base, residual = self.components(windows, features)
        return base + multiplier*residual


def components(model, windows, features):
    model.eval()
    base, correction = [], []
    with torch.inference_mode():
        for start in range(0, len(windows), 64):
            b, c = model.components(windows[start:start+64], features[start:start+64])
            base.append(b.cpu().numpy()); correction.append(c.cpu().numpy())
    empty = np.empty(0, np.float32)
    return (np.concatenate(base), np.concatenate(correction)) if base else (empty, empty)


def probability(base, correction, multiplier):
    logits = np.asarray(base, np.float64) + multiplier*np.asarray(correction, np.float64)
    return 1/(1+np.exp(-np.clip(logits, -35., 35.)))


def load_head(path, arm, device):
    blank = LogisticModel(np.zeros(237), np.zeros(237), np.ones(237), np.zeros(237), 0., .1)
    model = HybridHead(blank, arm).to(device)
    with np.load(path, allow_pickle=False) as data:
        model.load_state_dict({k: torch.from_numpy(data[k].copy()).to(device) for k in data.files}, strict=True)
    return model.eval()


def fold_name(excluded):
    return "+".join(sorted(excluded)) or "all-development"


def fit_base(root, rows, features, excluded, contract):
    train, held = prior.fold_indices(rows, excluded)
    path = Path(root)/"bases"/(fold_name(excluded)+".json")
    identity = {"contractSha256": contract, "excludedGroups": sorted(excluded),
                "trainingRows": train.tolist(), "heldRows": held.tolist()}
    if path.exists():
        saved = read(path)
        require(all(saved[k] == v for k, v in identity.items()), "Base fit identity differs")
        require(saved["modelSha256"] == canonical(saved["model"]), "Base parameters changed")
        return LogisticModel.from_dict(saved["model"])
    model = fit_logistic(features[train], np.asarray([rows[i]["label"] for i in train]), l2=.1)
    immutable(path, {**identity, "model": model.to_dict(), "modelSha256": canonical(model.to_dict())})
    return model


def fit(root, rows, raw, features, features_gpu, excluded, seed, arm, contract, *, epochs=prior.EPOCHS, folder=None):
    train, held = prior.fold_indices(rows, excluded)
    folder = Path(folder) if folder is not None else Path(root)/"fits"/arm/str(seed)/fold_name(excluded)
    identity = {"contractSha256": contract, "seed": seed, "arm": arm, "excludedGroups": sorted(excluded),
                "trainingRows": train.tolist(), "heldRows": held.tolist(), "epochs": list(epochs)}
    done = folder/"completed.json"
    if done.exists():
        saved = read(done)
        require(all(saved[k] == v for k, v in identity.items()), "Hybrid fit resume differs")
        for name, sha in saved["artifacts"].items():
            require(digest(folder/name) == sha, "Hybrid fit artifact changed")
        return
    require(not folder.exists(), "Preserve incomplete fit before retry")
    base = fit_base(root, rows, features, excluded, contract)
    folder.mkdir(parents=True)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = raw.device
    model = HybridHead(base, arm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    train_ix = torch.tensor(train, device=device)
    held_ix = torch.tensor(held, device=device)
    truth = torch.tensor([rows[i]["label"] for i in train], device=device, dtype=torch.float32)
    weights = torch.from_numpy(prior.training_weights([rows[i] for i in train])).to(device)
    rng = np.random.default_rng(seed)
    history, scores = [], {}
    started = time.monotonic()
    for epoch in range(1, max(epochs)+1):
        model.train(); losses = []
        for batch in np.array_split(rng.permutation(len(train)), math.ceil(len(train)/32)):
            ix = torch.tensor(batch, device=device)
            optimizer.zero_grad(set_to_none=True)
            b, c = model.components(raw[train_ix[ix]], features_gpu[train_ix[ix]])
            loss = ((F.binary_cross_entropy_with_logits(b+c, truth[ix], reduction="none")+.01*c.square())*weights[ix]).mean()
            require(bool(torch.isfinite(loss)), "Nonfinite hybrid loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        if epoch in epochs:
            for name, ix in (("held", held_ix), ("train", train_ix)):
                b, c = components(model, raw[ix], features_gpu[ix])
                scores[f"{epoch}-{name}-base"] = b
                scores[f"{epoch}-{name}-residual"] = c
            np.savez_compressed(folder/f"weights-{epoch}.npz",
                                **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    np.savez_compressed(folder/"scores.npz", **scores)
    immutable(done, {**identity, "history": history, "wallSeconds": time.monotonic()-started,
                     "parameters": sum(p.numel() for p in model.parameters()),
                     "artifacts": {p.name: digest(p) for p in folder.glob("*.npz")}})


def prepare(repo):
    root, old = Path(private_value(STUDY_INDEX)), Path(private_value("private-reference-0225"))
    require(not (root/"registration.json").exists(), "Hybrid registration already exists")
    reg, rows, raw = prior.load_registration(old)
    dataset = read(verified(reg["contract"]["inputs"]["dataset"]))
    baseline = read(verified(reg["contract"]["inputs"]["baseline"]))
    import os
    ledger_path = os.environ.get("VOLLEYCUT_PRIVATE_LEDGER_POSIX") or os.environ["VOLLEYCUT_PRIVATE_LEDGER"]
    ledger = read(ledger_path)
    features, names = specialist_features(dataset, baseline, rows, ledger["originalToAlias"])
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root/"specialist-features.npz", values=features)
    immutable(root/"feature-names.json", names)
    contract = {"kind": "serving-side-hybrid-v1", "studyIndex": STUDY_INDEX,
                "priorStudy": reference(old/"registration.json"), "priorContractSha256": reg["sha256"],
                "seeds": list(prior.SEEDS), "epochs": list(prior.EPOCHS), "multipliers": list(MULTIPLIERS),
                "thresholds": list(prior.THRESHOLDS), "arms": list(ARMS),
                "sourceGroups": reg["contract"]["sourceGroups"], "encoderSha256": reg["contract"]["encoderSha256"],
                "data": {"features": reference(root/"specialist-features.npz"), "names": reference(root/"feature-names.json"),
                         **reg["contract"]["data"]},
                "sourceCode": {name: reference(Path(repo)/name) for name in CODE},
                "protocol": reference(Path(repo)/"docs/research/serving-side-hybrid-head-2026-10-13.md"),
                "featureCount": 237, "baseL2": .1, "residualBound": 3., "residualPenalty": .01,
                "selection": "nested source-group macro BA; less correction then earlier epoch on metric ties",
                "protectedLabelsOpened": False}
    # Archive source and protocol bytes before fitting. Runtime checks use the archive.
    import shutil
    for name in (*CODE, "docs/research/serving-side-hybrid-head-2026-10-13.md"):
        dest = root/"source-snapshot"/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(repo)/name, dest)
        require(digest(dest) == digest(Path(repo)/name), "Source archive differs")
        if name in contract["sourceCode"]:
            contract["sourceCode"][name] = reference(dest)
        else:
            contract["protocol"] = reference(dest)
    (root/"source-snapshot"/"analysis"/"__init__.py").touch()
    immutable(root/"registration.json", {"contract": contract, "sha256": canonical(contract)})
    return {"rows": len(rows), "features": len(names), "contractSha256": canonical(contract)}


def load(root):
    root = Path(root)
    reg = read(root/"registration.json"); c = reg["contract"]
    require(canonical(c) == reg["sha256"], "Hybrid registration changed")
    for ref in [*c["sourceCode"].values(), *c["data"].values(), c["protocol"], c["priorStudy"]]:
        verified(ref)
    oldreg = read(c["priorStudy"]["path"])
    require(oldreg["sha256"] == c["priorContractSha256"], "Prior contract differs")
    for ref in oldreg["contract"]["inputs"].values():
        verified(ref)
    rows = read(c["data"]["rows"]["path"])
    with np.load(c["data"]["windows"]["path"], allow_pickle=False) as f:
        windows = f["values"].copy()
    with np.load(c["data"]["features"]["path"], allow_pickle=False) as f:
        features = f["values"].copy()
    return reg, rows, windows, features


def run(root, device="cuda"):
    prior.configure_torch()
    reg, rows, windows, features = load(root)
    raw = torch.from_numpy(windows).to(device)
    fg = torch.from_numpy(features.astype(np.float32)).to(device)
    groups = reg["contract"]["sourceGroups"]
    folds = list(itertools.combinations(groups, 2)) + [(g,) for g in groups]
    completed = 0
    for arm in ARMS:
        for seed in prior.SEEDS:
            for excluded in folds:
                prior.progress(root, phase="hybrid-fitting", arm=arm, seed=seed, excludedGroups=excluded,
                               completedFits=completed, totalFits=len(folds)*len(ARMS)*len(prior.SEEDS))
                fit(root, rows, raw, features, fg, excluded, seed, arm, reg["sha256"])
                completed += 1
    prior.progress(root, phase="hybrid-fits-complete", completedFits=completed)


def saved_scores(root, arm, seed, excluded):
    folder = Path(root)/"fits"/arm/str(seed)/fold_name(excluded)
    meta = read(folder/"completed.json")
    require(meta["contractSha256"] == read(Path(root)/"registration.json")["sha256"]
            and meta["arm"] == arm and meta["seed"] == seed and meta["excludedGroups"] == sorted(excluded)
            and meta["epochs"] == list(prior.EPOCHS), "Hybrid saved fit differs")
    for name, sha in meta["artifacts"].items():
        require(digest(folder/name) == sha, "Hybrid checkpoint changed")
    with np.load(folder/"scores.npz", allow_pickle=False) as f:
        return np.asarray(meta["heldRows"]), {e: (f[f"{e}-held-base"].copy(), f[f"{e}-held-residual"].copy()) for e in prior.EPOCHS}


def select(rows, scores, multipliers=MULTIPLIERS):
    """Vectorized threshold metrics; labels from outer groups are never supplied."""
    require(set(scores) == set(prior.EPOCHS), "Hybrid checkpoint set differs")
    truth = np.asarray([r["label"] for r in rows], bool)
    groups = [np.asarray([r["sourceGroup"] == g for r in rows]) for g in sorted({r["sourceGroup"] for r in rows})]
    thresholds = np.asarray(prior.THRESHOLDS)
    best = None
    for multiplier in multipliers:
        for epoch in (prior.EPOCHS[:1] if multiplier == 0 else prior.EPOCHS):
            base, residual = scores[epoch]
            require(base.shape == residual.shape == (len(rows),) and np.isfinite(base).all()
                    and np.isfinite(residual).all(), "Invalid hybrid selection scores")
            p = probability(base, residual, multiplier)
            pred = p[:, None] >= thresholds
            def ba(ix):
                require(np.any(ix&truth) and np.any(ix&~truth), "Selection source lacks a class")
                return .5*(pred[ix&truth].mean(axis=0)+(~pred[ix&~truth]).mean(axis=0))
            group_ba = np.stack([ba(ix) for ix in groups])
            pooled = ba(np.ones(len(rows), bool))
            tp, tn = pred[truth].sum(axis=0), (~pred[~truth]).sum(axis=0)
            fp, fn = (~truth).sum()-tn, truth.sum()-tp
            f1 = tp/(2*tp+fp+fn) + tn/(2*tn+fp+fn)
            choice_epoch = 0 if multiplier == 0 else epoch
            for i, threshold in enumerate(thresholds):
                rank = (float(group_ba[:, i].mean()), float(pooled[i]), float(f1[i]), float(group_ba[:, i].min()),
                        -multiplier, -choice_epoch, -abs(float(threshold)-.5), -float(threshold))
                if best is None or rank > best[0]:
                    best = (rank, choice_epoch, multiplier, float(threshold), p >= threshold)
    require(best is not None, "Empty hybrid selection")
    return {"epoch": best[1], "multiplier": best[2], "threshold": best[3], "evaluation": prior.evaluate(rows, best[4])}


def nested(root, rows, groups, arm, seed, multipliers, variant):
    probabilities = np.full(len(rows), np.nan); thresholds = probabilities.copy(); folds = []
    for outer in groups:
        train, held = prior.fold_indices(rows, (outer,))
        local = {v: i for i, v in enumerate(train)}
        scores = {e: (np.full(len(train), np.nan), np.full(len(train), np.nan)) for e in prior.EPOCHS}
        for inner in groups:
            if inner == outer:
                continue
            indexes, pair = saved_scores(root, arm, seed, (outer, inner))
            for e in prior.EPOCHS:
                for index, b, c in zip(indexes, *pair[e], strict=True):
                    if rows[index]["sourceGroup"] == inner:
                        scores[e][0][local[index]], scores[e][1][local[index]] = b, c
        chosen = select([rows[i] for i in train], scores, multipliers)
        immutable(Path(root)/"selection"/variant/str(seed)/(outer+".json"),
                  {"outerExcludedGroup": outer, "innerRows": train.tolist(), "selected": chosen})
        indexes, outer_scores = saved_scores(root, arm, seed, (outer,))
        require(np.array_equal(indexes, held), "Hybrid held row order differs")
        e = chosen["epoch"] or prior.EPOCHS[0]
        probabilities[held] = probability(*outer_scores[e], chosen["multiplier"])
        thresholds[held] = chosen["threshold"]
        folds.append({"sourceGroup": outer, "epoch": chosen["epoch"], "multiplier": chosen["multiplier"],
                      "threshold": chosen["threshold"], "innerMacroBA": chosen["evaluation"]["sourceGroupMacroBalancedAccuracy"]})
    require(np.isfinite(probabilities).all(), "Missing hybrid outer scores")
    return probabilities, thresholds, folds


def paired(rows, predicted):
    truth = np.asarray([r["label"] for r in rows]); baseline = np.asarray([r["baselineNear"] for r in rows])
    return {"baselineOnlyCorrect": int(((baseline == truth)&(predicted != truth)).sum()),
            "headOnlyCorrect": int(((baseline != truth)&(predicted == truth)).sum()),
            "bothCorrect": int(((baseline == truth)&(predicted == truth)).sum()),
            "bothWrong": int(((baseline != truth)&(predicted != truth)).sum())}


def summarize(root, report, device="cuda"):
    prior.configure_torch(); root = Path(root)
    reg, rows, windows, features = load(root); groups = reg["contract"]["sourceGroups"]
    variants, arrays = {}, {}
    configs = [(arm+"-"+kind, arm, mult) for arm in ARMS for kind, mult in (("full", (1.,)), ("selected", MULTIPLIERS))]
    configs += [("logistic", "hybrid", (0.,))]
    for variant, arm, mult in configs:
        evaluations, probs, cuts = [], [], []
        for seed in prior.SEEDS:
            p, t, folds = nested(root, rows, groups, arm, seed, mult, variant)
            evaluations.append({"seed": seed, "evaluation": prior.evaluate(rows, p >= t), "folds": folds})
            probs.append(p); cuts.append(t)
            arrays[f"{variant}-{seed}-probabilities"] = p; arrays[f"{variant}-{seed}-thresholds"] = t
        ensemble = np.mean(probs, axis=0) >= np.mean(cuts, axis=0)
        variants[variant] = {"seeds": evaluations, "ensemble": prior.evaluate(rows, ensemble),
                             "meanSeedMacroBalancedAccuracy": float(np.mean([v["evaluation"]["sourceGroupMacroBalancedAccuracy"] for v in evaluations])),
                             "meanSeedPooledBalancedAccuracy": float(np.mean([v["evaluation"]["pooled"]["balancedAccuracy"] for v in evaluations])),
                             "minimumSeedWorstGroupBalancedAccuracy": min(v["evaluation"]["worstSourceGroupBalancedAccuracy"] for v in evaluations),
                             "pairedErrorsAgainstHistoricalSpecialist": paired(rows, ensemble)}
        prior.progress(root, phase="hybrid-selection", variant=variant, meanMacroBA=variants[variant]["meanSeedMacroBalancedAccuracy"])
    raw = torch.from_numpy(windows).to(device); fg = torch.from_numpy(features.astype(np.float32)).to(device)
    final = []
    for seed in prior.SEEDS:
        scores = {e: (np.full(len(rows), np.nan), np.full(len(rows), np.nan)) for e in prior.EPOCHS}
        for group in groups:
            indexes, held = saved_scores(root, "hybrid", seed, (group,))
            for e in prior.EPOCHS:
                scores[e][0][indexes], scores[e][1][indexes] = held[e]
        choices = {kind: select(rows, scores, mult) for kind, mult in (("full", (1.,)), ("selected", MULTIPLIERS))}
        epochs = tuple(sorted({v["epoch"] for v in choices.values() if v["epoch"]}))
        folder = root/"final"/str(seed)/"fit"
        fit(root, rows, raw, features, fg, (), seed, "hybrid", reg["sha256"], epochs=epochs, folder=folder)
        for kind, choice in choices.items():
            immutable(root/"final"/str(seed)/(kind+"-selection.json"), choice)
            entry = {"seed": seed, "kind": kind, **{k: choice[k] for k in ("epoch", "multiplier", "threshold")}}
            if choice["epoch"]:
                path = folder/f"weights-{choice['epoch']}.npz"
                model = load_head(path, "hybrid", device)
                b, c = components(model, raw[:16], fg[:16])
                cb, cc = components(load_head(path, "hybrid", "cpu"), raw[:16].cpu(), fg[:16].cpu())
                error = float(np.max(abs(probability(b,c,choice["multiplier"])-probability(cb,cc,choice["multiplier"]))))
                require(error < 2e-5, "Final hybrid CPU/GPU parity failed")
                entry.update({"weightsSha256": digest(path), "fileBytes": path.stat().st_size,
                              "parameters": sum(p.numel() for p in model.parameters()), "cpuGpuMaximumError": error})
            else:
                entry["baseModelSha256"] = read(root/"bases"/"all-development.json")["modelSha256"]
            final.append(entry)
    np.savez_compressed(root/"nested-predictions.npz", **arrays)
    baseline = prior.evaluate(rows, [r["baselineNear"] for r in rows])
    h = variants["hybrid-selected"]
    result = {"schemaVersion": 1, "kind": "serving-side-hybrid-results-v1", "studyIndex": STUDY_INDEX,
              "contractSha256": reg["sha256"], "encoderSha256": reg["contract"]["encoderSha256"],
              "rows": len(rows), "recordings": len({r["recordingIndex"] for r in rows}), "sourceGroups": len(groups),
              "specialistFeatures": 237, "variants": variants, "historicalSpecialist": baseline, "finalHeads": final,
              "meetsHistoricalSpecialistGuardrails": h["meanSeedMacroBalancedAccuracy"] > baseline["sourceGroupMacroBalancedAccuracy"]
                   and h["minimumSeedWorstGroupBalancedAccuracy"] >= baseline["worstSourceGroupBalancedAccuracy"],
              "protectedLabelsOpened": False, "productionChanged": False}
    immutable(root/"summary.json", result); immutable(report, result)
    prior.progress(root, phase="hybrid-complete", meanMacroBA=h["meanSeedMacroBalancedAccuracy"])
    return result
