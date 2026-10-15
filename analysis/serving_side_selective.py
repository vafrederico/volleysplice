"""Precision-first neural serving-side research with an explicit review output."""
from __future__ import annotations

import itertools
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import serving_side_hybrid as h
from .private_ledger import private_value
from .serving_side_v2 import LogisticModel
from .serving_side_cost_sensitive import CostSensitiveHead, cost_sensitive_loss, scores as cost_scores

STUDY_INDEX = "private-reference-0236"
FAST_INDEX = "private-reference-0237"
TARGETS = (.95, .98, .99)
REVIEW_COST = .02
WARMUP = 5
ACTIVE_WARMUP = 0
PROTOCOL = "docs/research/serving-side-precision-head-2026-10-14.md"
CODE = ("analysis/serving_side_selective.py", "analysis/serving_side_selective_metrics.py",
        "analysis/serving_side_cost_sensitive.py",
        "analysis/serving_side_evidence_agreement.py", "analysis/serving_side_selective_eval.py",
        "scripts/evaluate-serving-side-selective.py", "scripts/train-serving-side-selective.py",
        *h.CODE)
read, require, digest, canonical, immutable, reference, verified = (
    h.read, h.require, h.digest, h.canonical, h.immutable, h.reference, h.verified)


class SelectiveHead(h.HybridHead):
    def __init__(self, base):
        super().__init__(base)
        self.acceptance = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Dropout(.1), nn.Linear(32, 1))
        nn.init.zeros_(self.acceptance[-1].weight)
        nn.init.constant_(self.acceptance[-1].bias, math.log(9.))
        self.float()

    def outputs(self, windows, features):
        z = (features - self.base_mean) / self.base_scale
        base = (z @ self.base_weights + self.base_bias).clamp(-30., 30.)
        tokens = windows[..., :3840].reshape(-1, 13, 4, 960)
        projected = F.relu(self.visual.projection(self.visual.norm(tokens)))
        projected = projected * windows[..., 3840:3841, None]
        inputs = torch.cat((projected.flatten(-2), windows[..., 3840:]), dim=-1)
        temporal = self.visual.temporal(inputs)[:, 6]
        fused = torch.cat((temporal, self.specialist(z)), dim=1)
        residual = 3. * torch.tanh(self.correction(fused)[:, 0])
        return base + residual, self.acceptance(fused)[:, 0], residual


def selective_loss(logit, acceptance_logit, truth, residual, epoch):
    bce = F.binary_cross_entropy_with_logits(logit, truth, reduction="none")
    if epoch <= WARMUP:
        return bce + .01 * residual.square()
    p = torch.sigmoid(logit)
    true_probability = truth * p + (1-truth) * (1-p)
    acceptance = torch.sigmoid(acceptance_logit)
    reward = acceptance * true_probability + (1-REVIEW_COST) * (1-acceptance)
    return -torch.log(reward.clamp_min(1e-7)) + .1 * bce + .01 * residual.square()


def predict(model, windows, features):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(windows), 64):
            result = model.outputs(windows[start:start+64], features[start:start+64])
            if isinstance(model, CostSensitiveHead):
                outputs.append(torch.cat(result, dim=1).cpu().numpy())
            else:
                outputs.append(torch.stack(result, dim=1).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, 4 if isinstance(model, CostSensitiveHead) else 3), np.float32)


def scores(logits):
    if np.asarray(logits).shape[1] == 4:
        return cost_scores(np.asarray(logits)[:, :2])
    return 1 / (1 + np.exp(-np.clip(np.asarray(logits, np.float64)[:, :2], -35., 35.)))


def load_head(path, device="cpu"):
    blank = LogisticModel(np.zeros(237), np.zeros(237), np.ones(237), np.zeros(237), 0., .1)
    with np.load(path, allow_pickle=False) as data:
        model = (SelectiveHead(blank) if any(k.startswith("acceptance.") for k in data.files)
                 else CostSensitiveHead(blank)).to(device)
        model.load_state_dict({k: torch.from_numpy(data[k].copy()).to(device) for k in data.files}, strict=True)
    return model.eval()


def prepare(repo):
    repo = Path(repo)
    root, fast, previous = (Path(private_value(k)) for k in (STUDY_INDEX, FAST_INDEX, h.STUDY_INDEX))
    require(not (root / "registration.json").exists(), "Keep completed registration immutable")
    reg, rows, windows, features = h.load(previous)
    require(len(rows) == 968 and len({r["sourceGroup"] for r in rows}) == 8, "Scope differs")
    root.mkdir(parents=True, exist_ok=True); fast.mkdir(parents=True, exist_ok=True)
    np.save(fast / "windows.npy", windows, allow_pickle=False)
    np.save(fast / "features.npy", features, allow_pickle=False)
    immutable(root / "rows.json", rows)
    # Cache old fitting receipts once; subsequent work reads from the fast store.
    for name in ("fits/hybrid", "bases"):
        dest = fast / "previous" / name
        require(not dest.exists(), "Prior cache already exists")
        shutil.copytree(previous / name, dest)
    old_fits = {}
    for seed in h.prior.SEEDS:
        for excluded in itertools.chain(itertools.combinations(reg["contract"]["sourceGroups"], 2),
                                        ((g,) for g in reg["contract"]["sourceGroups"])):
            folder = fast / "previous/fits/hybrid" / str(seed) / h.fold_name(excluded)
            meta = read(folder / "completed.json")
            require(meta["contractSha256"] == reg["sha256"], "Cached previous fit differs")
            require(digest(folder / "scores.npz") == meta["artifacts"]["scores.npz"], "Cached scores differ")
            old_fits[f"{seed}/{h.fold_name(excluded)}"] = reference(folder / "completed.json")
    source_refs = {}
    for name in dict.fromkeys((*CODE, PROTOCOL)):
        dest = root / "source-snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, dest)
        source_refs[name] = reference(dest)
    (root / "source-snapshot/analysis/__init__.py").touch()
    contract = {"kind": "serving-side-cost-sensitive-v1", "studyIndex": STUDY_INDEX, "fastIndex": FAST_INDEX,
        "priorRegistration": reference(previous / "registration.json"), "priorContractSha256": reg["sha256"],
        "sourceGroups": reg["contract"]["sourceGroups"], "seeds": list(h.prior.SEEDS), "epochs": list(h.prior.EPOCHS),
        "targets": list(TARGETS), "primaryTarget": .98, "falsePositiveWeight": 49.,
        "positiveWeight": 1., "residualPenalty": .001, "warmupEpochs": ACTIVE_WARMUP,
        "encoderSha256": reg["contract"]["encoderSha256"], "sourceCode": source_refs,
        "data": {"rows": reference(root / "rows.json"), "windows": reference(fast / "windows.npy"),
                 "features": reference(fast / "features.npy")}, "priorFits": old_fits,
        "priorBases": {p.stem: reference(p) for p in (fast / "previous/bases").glob("*.json")},
        "selection": "inner-held per-side and source-macro precision constraints then maximum source-macro coverage",
        "protectedLabelsOpened": False}
    immutable(root / "registration.json", {"contract": contract, "sha256": canonical(contract)})
    return {"contractSha256": canonical(contract), "rows": len(rows), "newFits": 108}


def load():
    root, fast = (Path(private_value(k)) for k in (STUDY_INDEX, FAST_INDEX))
    reg = read(root / "registration.json"); c = reg["contract"]
    require(canonical(c) == reg["sha256"], "Registration changed")
    for ref in [*c["sourceCode"].values(), *c["data"].values(), c["priorRegistration"],
                *c["priorFits"].values(), *c["priorBases"].values()]:
        verified(ref)
    for name, ref in c["sourceCode"].items():
        if name.endswith(".py"):
            require(digest(Path(__file__).resolve().parents[1] / name) == ref["sha256"], "Imported research source differs from registration")
    rows = read(c["data"]["rows"]["path"])
    windows = np.load(c["data"]["windows"]["path"], allow_pickle=False)
    features = np.load(c["data"]["features"]["path"], allow_pickle=False)
    return root, fast, reg, rows, windows, features


def fit(fast, rows, windows, features, feature_gpu, excluded, seed, contract, prior_contract, *, folder=None):
    train, held = h.prior.fold_indices(rows, excluded)
    folder = Path(folder) if folder else fast / "fits" / str(seed) / h.fold_name(excluded)
    identity = {"contractSha256": contract, "seed": seed, "excludedGroups": sorted(excluded),
                "trainingRows": train.tolist(), "heldRows": held.tolist(), "epochs": list(h.prior.EPOCHS)}
    if (folder / "completed.json").exists():
        done = read(folder / "completed.json")
        require(all(done[k] == v for k, v in identity.items()), "Resume identity differs")
        for name, sha in done["artifacts"].items():
            require(digest(folder / name) == sha, "Fit artifact changed")
        return
    require(not folder.exists(), "Preserve incomplete fitting directory")
    base_path = fast / "previous/bases" / (h.fold_name(excluded) + ".json")
    if base_path.exists():
        saved = read(base_path)
        require(saved["contractSha256"] == prior_contract and saved["excludedGroups"] == sorted(excluded), "Base provenance differs")
        require(saved["trainingRows"] == train.tolist() and saved["heldRows"] == held.tolist(), "Base scope differs")
        require(canonical(saved["model"]) == saved["modelSha256"], "Base changed")
        base = LogisticModel.from_dict(saved["model"])
    else:
        base = h.fit_logistic(features[train], np.array([rows[i]["label"] for i in train]), l2=.1)
    folder.mkdir(parents=True)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = CostSensitiveHead(base).to(windows.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    train_ix, held_ix = (torch.tensor(ix, device=windows.device) for ix in (train, held))
    truth = torch.tensor([rows[i]["label"] for i in train], device=windows.device, dtype=torch.float32)
    weights = torch.from_numpy(h.prior.training_weights([rows[i] for i in train])).to(windows.device)
    rng = np.random.default_rng(seed)
    history, arrays, started = [], {}, time.monotonic()
    for epoch in range(1, max(h.prior.EPOCHS)+1):
        model.train(); losses = []
        for batch in np.array_split(rng.permutation(len(train)), math.ceil(len(train)/32)):
            ix = torch.tensor(batch, device=windows.device)
            optimizer.zero_grad(set_to_none=True)
            logits, residual = model.outputs(windows[train_ix[ix]], feature_gpu[train_ix[ix]])
            loss = (cost_sensitive_loss(logits, truth[ix], residual) * weights[ix]).mean()
            require(bool(torch.isfinite(loss)), "Nonfinite selective loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "warmup": False}
        if epoch in h.prior.EPOCHS:
            for name, ix in (("held", held_ix), ("train", train_ix)):
                outputs = predict(model, windows[ix], feature_gpu[ix])
                arrays[f"{epoch}-{name}"] = outputs
                if len(outputs):
                    s = scores(outputs); t = np.array([rows[i]["label"] for i in (held if name == "held" else train)])
                    entry[name] = {"meanAcceptance": float(s[:, 1].mean()),
                                   "rawSideAccuracy": float(((s[:, 0] >= .5) == t).mean())}
            np.savez_compressed(folder / f"weights-{epoch}.npz", **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
        history.append(entry)
    np.savez_compressed(folder / "scores.npz", **arrays)
    immutable(folder / "completed.json", {**identity, "history": history,
        "wallSeconds": time.monotonic()-started, "parameters": sum(p.numel() for p in model.parameters()),
        "artifacts": {p.name: digest(p) for p in folder.glob("*.npz")}})


def run(device="cuda"):
    h.prior.configure_torch()
    root, fast, reg, rows, raw, features = load()
    windows = torch.from_numpy(raw).to(device); fg = torch.from_numpy(features.astype(np.float32)).to(device)
    groups = reg["contract"]["sourceGroups"]
    exclusions = list(itertools.combinations(groups, 2)) + [(g,) for g in groups]
    done = 0
    for seed in h.prior.SEEDS:
        for excluded in exclusions:
            fit(fast, rows, windows, features, fg, excluded, seed, reg["sha256"], reg["contract"]["priorContractSha256"])
            done += 1
            progress = {"phase": "selective-fitting", "completedFits": done, "totalFits": 108,
                        "seed": seed, "excludedGroups": list(excluded)}
            (fast / "progress.json").write_text(__import__("json").dumps(progress))
            print(progress, flush=True)
    shutil.copytree(fast / "fits", root / "fits", dirs_exist_ok=True)


def frozen_scores(fast, variant, seed, excluded, reg):
    folder = fast / ("fits" if variant == "selective" else "previous/fits/hybrid") / str(seed) / h.fold_name(excluded)
    meta = read(folder / "completed.json")
    expected = reg["sha256"] if variant == "selective" else reg["contract"]["priorContractSha256"]
    if variant != "selective":
        require(digest(folder / "completed.json") == reg["contract"]["priorFits"][f"{seed}/{h.fold_name(excluded)}"]["sha256"], "Prior receipt changed")
        require(meta["arm"] == "hybrid", "Prior arm differs")
    require(meta["excludedGroups"] == sorted(excluded) and meta["seed"] == seed
            and meta["epochs"] == list(h.prior.EPOCHS) and meta["contractSha256"] == expected, "Saved score scope differs")
    require(digest(folder / "scores.npz") == meta["artifacts"]["scores.npz"], "Scores changed")
    result = {}
    with np.load(folder / "scores.npz", allow_pickle=False) as data:
        for epoch in h.prior.EPOCHS:
            if variant == "selective":
                result[epoch] = scores(data[f"{epoch}-held"])
            else:
                p = h.probability(data[f"{epoch}-held-base"], data[f"{epoch}-held-residual"], 1.)
                result[epoch] = np.column_stack((p, np.ones(len(p))))
    return np.array(meta["heldRows"]), result
