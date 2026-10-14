"""Ten frozen precision experiments; all private inputs resolve through the ledger."""
from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import serving_side_selective as prior
from . import serving_side_selective_metrics as metrics
from .serving_side_cost_sensitive import CostSensitiveHead, cost_sensitive_loss, scores as cost_scores
from .serving_side_selective_eval import consensus, seed_summary
from .serving_side_v2 import LogisticModel, fit_logistic
from .private_ledger import private_value

STUDY = "private-reference-0240"
FAST = "private-reference-0241"
PROTOCOL = "docs/research/serving-side-precision-iterations-2026-10-14.md"
ARMS = {
    "01-quality-mean": "Four absolute quality averages",
    "02-quality-phases": "Twelve phase-specific quality measurements",
    "03-raw-court": "Raw court flow plus phase quality",
    "04-embedding-changes": "Projected embedding changes plus phase quality",
    "05-quality-interactions": "Motion quality by near/far evidence interactions",
    "06-correctness-linear": "Cross-fitted correctness head",
    "07-correctness-interactions": "Cross-fitted nonlinear correctness features",
    "08-neural-quality": "Temporal cost head plus phase quality",
    "09-neural-centered": "Centered temporal cost head plus quality",
    "10-neural-differences": "Temporal difference cost head plus quality",
}
CONTROL = "control-ranks-tuned"
CPU_ARMS = (CONTROL, *list(ARMS)[:7])
NEURAL_ARMS = tuple(list(ARMS)[7:])
L2 = {1: 1., 2: .1, 3: .01}
SEEDS = prior.h.prior.SEEDS
EPOCHS = prior.h.prior.EPOCHS
TARGETS = prior.TARGETS
PHASES = ("launch", "early", "late")
QUALITY = ("energyMean", "activeFraction", "smallComponentEnergyFraction", "absoluteFlowY")
require, read, digest, canonical = prior.require, prior.read, prior.digest, prior.canonical
immutable, reference, verified = prior.immutable, prior.reference, prior.verified
fold_name = prior.h.fold_name
CODE = ("analysis/serving_side_precision_iterations.py", "scripts/train-serving-side-precision-iterations.py",
        "analysis/serving_side_selective_eval.py", "scripts/diagnose-serving-side-quality.py",
        *prior.CODE)


def phase_quality(original):
    raw = original["configurations"]["192x108-r4c6"]
    return np.asarray([[abs(float(raw[f"{p}:flowY"])) if k == "absoluteFlowY"
                        else float(raw[f"{p}:{k}"]) for k in QUALITY] for p in PHASES], np.float64).reshape(-1)


def temporal_transform(windows, mode):
    """Window-local transformations cannot use another row or a label."""
    require(mode in ("raw", "centered", "differences"), "Unknown temporal transform")
    x = np.array(windows, dtype=np.float32, copy=True)
    valid = x[..., 3840] > .5
    if mode == "centered":
        mean = (x[..., :3840] * valid[..., None]).sum(axis=1, keepdims=True)
        mean /= np.maximum(valid.sum(axis=1, keepdims=True)[..., None], 1)
        x[..., :3840] = (x[..., :3840] - mean) * valid[..., None]
    elif mode == "differences":
        pair = valid[:, 1:] & valid[:, :-1]
        x[:, 1:, :3840] = (x[:, 1:, :3840] - x[:, :-1, :3840]) * pair[..., None]
        x[:, 0, :3840] = 0.
        x[:, 0, 3840] = 0.
        x[:, 1:, 3840] = pair
        x[..., 3842] *= x[..., 3840]
    return x


def projected_changes(windows):
    rng = np.random.default_rng(271828)
    projection = rng.normal(size=(3840, 32)).astype(np.float32) / np.sqrt(32.)
    valid = windows[..., 3840:3841]
    def average(start, stop):
        count = valid[:, start:stop].sum(axis=1)
        values = (windows[:, start:stop, :3840]*valid[:, start:stop]).sum(axis=1)
        return values/np.maximum(count, 1), count[:, 0] > 0
    pre, p = average(0, 6); post, q = average(7, 13)
    early, a = average(7, 10); late, b = average(10, 13)
    return np.column_stack(((post-pre)*((p&q)[:, None]) @ projection,
                            (late-early)*((a&b)[:, None]) @ projection)).astype(np.float64)


def identity(row):
    return row["recordingId"], row["rallyId"], float(row["serveAnchor"])


def prepare(repo):
    repo = Path(repo)
    root, fast = Path(private_value(STUDY)), Path(private_value(FAST))
    require(not (root/"registration.json").exists(), "Preserve existing iteration registration")
    _, old_fast, old, rows, windows, ranks = prior.load()
    dataset_path = Path(private_value("private-reference-0226"))
    original_reg = read(verified(read(verified(old["contract"]["priorRegistration"]))["contract"]["priorStudy"]))
    registered_dataset = original_reg["contract"]["inputs"]["dataset"]
    require(dataset_path.resolve() == verified(registered_dataset).resolve(), "Dataset ledger mapping differs")
    dataset = read(dataset_path)
    raw_ref = dataset["sources"]["v2DevelopmentDataset"]
    raw = read(verified(raw_ref))
    raw_rows = {identity(r): r for r in raw["rows"]}
    require(len(raw_rows) == len(raw["rows"]), "Raw court identities are not unique")
    ledger_path = os.environ.get("VOLLEYCUT_PRIVATE_LEDGER_POSIX") or os.environ["VOLLEYCUT_PRIVATE_LEDGER"]
    aliases = read(ledger_path)["originalToAlias"]
    quality, court = [], []
    for r in dataset["rows"]:
        require(r["sourceSplit"] != "test", "Protected rows are outside this study")
        if r["environment"] == "beach":
            continue
        current = rows[len(quality)]
        require(current["label"] == r["label"] and r["decision"] in ("near", "far")
                and current["recordingIndex"] == aliases[r["recordingId"]]
                and current["sourceGroup"] == aliases[r["sourceGroup"]], "Current feature identity differs")
        original = raw_rows[identity(r)]
        require(original["sourceSplit"] != "test" and original["sourceGroup"] == r["sourceGroup"],
                "Raw court scope differs")
        # Historical labels may have been corrected; only frozen current labels are used.
        quality.append(phase_quality(r))
        court.append([original["courtFlowFeatures"][n] for n in raw["featureNames"]])
    quality, court = np.asarray(quality), np.asarray(court, np.float64)
    require(quality.shape == (968, 12) and court.shape == (968, 82), "Feature dimensions differ")
    require(np.isfinite(quality).all() and np.isfinite(court).all(), "Features must be finite")
    names = read(Path(private_value("private-reference-0231"))/"feature-names.json")
    interaction_columns = [names.index(f"v2:{p}:nearMinusFar:{k}")
                           for p in ("pre", "contact", "post") for k in ("flowMean", "activeFraction")]
    root.mkdir(parents=True, exist_ok=True); fast.mkdir(parents=True, exist_ok=True)
    immutable(root/"rows.json", rows)
    np.save(fast/"windows.npy", windows, allow_pickle=False)
    np.savez_compressed(fast/"features.npz", ranks=ranks, quality=quality, court=court,
                        changes=projected_changes(windows), interactionColumns=interaction_columns)
    for name in ("windows.npy", "features.npz"):
        shutil.copyfile(fast/name, root/name)
    source = {}
    for name in dict.fromkeys((*CODE, PROTOCOL)):
        dest = root/"source-snapshot"/name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(repo/name, dest)
        source[name] = reference(dest)
    (root/"source-snapshot/analysis/__init__.py").touch()
    contract = {"kind": "serving-side-ten-precision-iterations-v1", "studyIndex": STUDY, "fastIndex": FAST,
        "arms": ARMS, "control": CONTROL, "seeds": list(SEEDS), "epochs": list(EPOCHS),
        "l2Candidates": L2, "targets": list(TARGETS), "primaryTarget": .98,
        "sourceGroups": old["contract"]["sourceGroups"], "sourceCode": source,
        "initialImplementationArchive": reference(Path(private_value("private-reference-0238"))/"registration.json"),
        "priorRegistration": reference(Path(private_value(prior.STUDY_INDEX))/"registration.json"),
        "priorSummary": reference(Path(private_value(prior.STUDY_INDEX))/"summary.json"),
        "inputs": {"dataset": reference(dataset_path), "rawCourt": raw_ref},
        "data": {"rows": reference(root/"rows.json"), "windows": reference(fast/"windows.npy"),
                 "features": reference(fast/"features.npz")},
        "resourcePolicy": "one fitting process; two CPU threads; sequential neural arms and seeds",
        "protectedLabelsOpened": False}
    immutable(root/"registration.json", {"contract": contract, "sha256": canonical(contract)})
    return {"contractSha256": canonical(contract), "rows": len(rows), "arms": len(ARMS)}


def load():
    root, fast = Path(private_value(STUDY)), Path(private_value(FAST))
    reg = read(root/"registration.json"); c = reg["contract"]
    require(canonical(c) == reg["sha256"], "Iteration registration changed")
    for ref in [*c["data"].values(), *c["inputs"].values(), *c["sourceCode"].values(),
                c["priorRegistration"], c["priorSummary"], c["initialImplementationArchive"]]:
        verified(ref)
    for name, ref in c["sourceCode"].items():
        if name.endswith(".py"):
            require(digest(Path(__file__).resolve().parents[1]/name) == ref["sha256"], "Imported source changed")
    rows = read(c["data"]["rows"]["path"])
    windows = np.load(c["data"]["windows"]["path"], allow_pickle=False)
    with np.load(c["data"]["features"]["path"], allow_pickle=False) as source:
        features = {k: source[k].copy() for k in source.files}
    return root, fast, reg, rows, windows, features


def design(features, arm):
    ranks, quality = features["ranks"], features["quality"]
    mean = quality.reshape(-1, 3, 4).mean(axis=1)
    if arm == CONTROL:
        return ranks
    if arm == "01-quality-mean":
        return np.column_stack((ranks, mean))
    if arm == "02-quality-phases":
        return np.column_stack((ranks, quality))
    if arm == "03-raw-court":
        return np.column_stack((ranks, quality, features["court"]))
    if arm == "04-embedding-changes":
        return np.column_stack((ranks, quality, features["changes"]))
    if arm == "05-quality-interactions":
        products = (ranks[:, features["interactionColumns"], None]-.5)*mean[:, None, :]
        return np.column_stack((ranks, quality, products.reshape(len(ranks), -1)))
    raise ValueError("Unknown direct feature arm")


def evidence(probabilities, quality, nonlinear=False):
    p = np.asarray(probabilities, np.float64)
    require(p.shape == (len(quality), 3), "Teacher probabilities are not aligned")
    side = p[:, 0] >= .5
    chosen = np.where(side[:, None], p, 1-p)
    e = np.column_stack((p[:, 0], 2*np.abs(p[:, 0]-.5), chosen[:, 1], chosen[:, 2],
                         chosen.min(axis=1), p.max(axis=1)-p.min(axis=1),
                         ((p >= .5) == side[:, None]).all(axis=1)))
    if nonlinear:
        return np.column_stack((e, quality, e*e, (e[:, :, None]*quality[:, None, :]).reshape(len(e), 84)))
    return np.column_stack((e, quality))


def partition(rows, excluded):
    return prior.h.prior.fold_indices(rows, excluded)


def teacher_partitions(groups):
    return [tuple(x) for count in (3, 2, 1, 0) for x in itertools.combinations(groups, count)]


def teacher(fast, reg, rows, ranks, excluded):
    folder = fast/"teachers"/fold_name(excluded)
    train, held = partition(rows, excluded)
    identity_meta = {"contractSha256": reg["sha256"], "excludedGroups": sorted(excluded),
                     "trainingRows": train.tolist(), "heldRows": held.tolist()}
    if (folder/"completed.json").exists():
        saved = read(folder/"completed.json")
        require(all(saved[k] == v for k, v in identity_meta.items()), "Teacher identity changed")
        require(digest(folder/"scores.npy") == saved["scoresSha256"], "Teacher scores changed")
        return np.load(folder/"scores.npy", allow_pickle=False)
    require(not folder.exists(), "Preserve incomplete teacher fit")
    folder.mkdir(parents=True)
    truth = np.array([rows[i]["label"] for i in train])
    values = (ranks, ranks[:, :82], ranks[:, 82:])
    models = [fit_logistic(x[train], truth, l2=.1) for x in values]
    p = np.column_stack([model.predict_proba(x) for model, x in zip(models, values, strict=True)])
    np.save(folder/"scores.npy", p, allow_pickle=False)
    immutable(folder/"completed.json", {**identity_meta, "models": [m.to_dict() for m in models],
                                        "scoresSha256": digest(folder/"scores.npy")})
    return p


def gate_data(fast, reg, rows, features, excluded):
    train, held = partition(rows, excluded)
    groups = reg["contract"]["sourceGroups"]
    out = np.full((len(rows), 3), np.nan)
    sources = []
    for group in groups:
        if group in excluded:
            continue
        teacher_excluded = tuple(sorted((*excluded, group)))
        indexes = np.array([i for i in train if rows[i]["sourceGroup"] == group])
        require(all(rows[i]["sourceGroup"] in teacher_excluded for i in indexes), "Teacher target leakage")
        out[indexes] = teacher(fast, reg, rows, features["ranks"], teacher_excluded)[indexes]
        sources.append({"group": group, "teacherExcludedGroups": list(teacher_excluded),
                        "rows": indexes.tolist()})
    require(np.isfinite(out[train]).all(), "Missing cross-fitted gate supervision")
    prediction = teacher(fast, reg, rows, features["ranks"], excluded)
    if len(excluded):
        require(all(rows[i]["sourceGroup"] in excluded for i in held), "Gate held teacher leakage")
    return train, held, out[train], prediction[held], sources


def fit_cpu(fast, reg, rows, features, arm, excluded):
    folder = fast/"fits"/arm/"deterministic"/fold_name(excluded)
    train, held = partition(rows, excluded)
    identity_meta = {"contractSha256": reg["sha256"], "arm": arm, "excludedGroups": sorted(excluded),
                     "trainingRows": train.tolist(), "heldRows": held.tolist(), "candidates": list(L2)}
    if (folder/"completed.json").exists():
        verify_fit(folder, identity_meta)
        return
    require(not folder.exists(), "Preserve incomplete CPU fit")
    folder.mkdir(parents=True)
    started = time.monotonic()
    gate = arm.startswith(("06-", "07-"))
    sources = []
    if gate:
        _, _, p_train, p_held, sources = gate_data(fast, reg, rows, features, excluded)
        nonlinear = arm.startswith("07-")
        x_train = evidence(p_train, features["quality"][train], nonlinear)
        x_held = evidence(p_held, features["quality"][held], nonlinear)
        y = ((p_train[:, 0] >= .5) == np.array([rows[i]["label"] for i in train])).astype(int)
    else:
        x = design(features, arm); x_train, x_held = x[train], x[held]
        y = np.array([rows[i]["label"] for i in train])
    scores, models = {}, {}
    for candidate, l2 in L2.items():
        model = fit_logistic(x_train, y, l2=l2)
        p = model.predict_proba(x_held)
        scores[str(candidate)] = (np.column_stack(((p_held[:, 0] >= .5).astype(float), p)) if gate
                                   else np.column_stack((p, np.ones(len(p)))))
        models[str(candidate)] = model.to_dict()
    np.savez_compressed(folder/"scores.npz", **scores)
    immutable(folder/"models.json", models)
    immutable(folder/"completed.json", {**identity_meta, "gateSupervision": sources,
        "featureCount": x_train.shape[1], "trainingTargetPositives": int(y.sum()),
        "wallSeconds": time.monotonic()-started,
        "artifacts": {p.name: digest(p) for p in folder.iterdir() if p.is_file()}})


def verify_fit(folder, identity_meta):
    meta = read(folder/"completed.json")
    require(all(meta[k] == v for k, v in identity_meta.items()), "Fit identity changed")
    for name, sha in meta["artifacts"].items():
        require(digest(folder/name) == sha, "Fit artifact changed")
    return meta


class QualityHead(CostSensitiveHead):
    def __init__(self, base, mean, scale):
        super().__init__(base)
        self.register_buffer("quality_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("quality_scale", torch.tensor(scale, dtype=torch.float32))
        self.specialist[0] = nn.Linear(249, 32)

    def components(self, windows, features):
        require(features.shape == (len(windows), 249), "Quality head expects 249 features")
        z = (features[:, :237]-self.base_mean)/self.base_scale
        quality = (features[:, 237:]-self.quality_mean)/self.quality_scale
        base = (z @ self.base_weights+self.base_bias).clamp(-30., 30.)
        tokens = windows[..., :3840].reshape(-1, 13, 4, 960)
        projected = F.relu(self.visual.projection(self.visual.norm(tokens)))
        projected = projected*windows[..., 3840:3841, None]
        inputs = torch.cat((projected.flatten(-2), windows[..., 3840:]), dim=-1)
        temporal = self.visual.temporal(inputs)[:, 6]
        fused = torch.cat((temporal, self.specialist(torch.cat((z, quality), dim=1))), dim=1)
        return base, 3.*torch.tanh(self.correction(fused))


def load_head(path, device="cpu"):
    base = LogisticModel(np.zeros(237), np.zeros(237), np.ones(237), np.zeros(237), 0., .1)
    model = QualityHead(base, np.zeros(12), np.ones(12)).to(device)
    with np.load(path, allow_pickle=False) as data:
        model.load_state_dict({k: torch.from_numpy(data[k].copy()).to(device) for k in data.files}, strict=True)
    return model.eval()


def fit_neural(fast, reg, rows, features, raw, fg, arm, seed, excluded):
    folder = fast/"fits"/arm/str(seed)/fold_name(excluded)
    train, held = partition(rows, excluded)
    identity_meta = {"contractSha256": reg["sha256"], "arm": arm, "seed": seed,
        "excludedGroups": sorted(excluded), "trainingRows": train.tolist(), "heldRows": held.tolist(),
        "candidates": list(EPOCHS)}
    if (folder/"completed.json").exists():
        verify_fit(folder, identity_meta)
        return
    require(not folder.exists(), "Preserve incomplete neural fit")
    folder.mkdir(parents=True)
    teacher(fast, reg, rows, features["ranks"], excluded)
    base = LogisticModel.from_dict(read(fast/"teachers"/fold_name(excluded)/"completed.json")["models"][0])
    quality = features["quality"][train]
    mean, scale = quality.mean(axis=0), quality.std(axis=0)
    scale[scale < 1e-6] = 1.
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = QualityHead(base, mean, scale).to(raw.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    tx = torch.tensor(train, device=raw.device); hx = torch.tensor(held, device=raw.device)
    truth = torch.tensor([rows[i]["label"] for i in train], device=raw.device, dtype=torch.float32)
    weights = torch.from_numpy(prior.h.prior.training_weights([rows[i] for i in train])).to(raw.device)
    rng = np.random.default_rng(seed)
    arrays, history = {}, []
    started = time.monotonic()
    for epoch in range(1, max(EPOCHS)+1):
        model.train(); losses = []
        for batch in np.array_split(rng.permutation(len(train)), math.ceil(len(train)/32)):
            indexes = torch.tensor(batch, device=raw.device)
            optimizer.zero_grad(set_to_none=True)
            logits, residual = model.outputs(raw[tx[indexes]], fg[tx[indexes]])
            loss = (cost_sensitive_loss(logits, truth[indexes], residual)*weights[indexes]).mean()
            require(bool(torch.isfinite(loss)), "Nonfinite quality-head loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        if epoch in EPOCHS:
            arrays[str(epoch)] = prior.predict(model, raw[hx], fg[hx])
            np.savez_compressed(folder/f"weights-{epoch}.npz",
                                **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    np.savez_compressed(folder/"scores.npz", **arrays)
    immutable(folder/"completed.json", {**identity_meta, "history": history,
        "parameters": sum(p.numel() for p in model.parameters()), "wallSeconds": time.monotonic()-started,
        "artifacts": {p.name: digest(p) for p in folder.glob("*.npz")}})


def frozen_scores(fast, arm, seed, excluded):
    folder = fast/"fits"/arm/str(seed)/fold_name(excluded)
    meta = verify_fit(folder, {"arm": arm, "excludedGroups": sorted(excluded)})
    with np.load(folder/"scores.npz", allow_pickle=False) as saved:
        values = {int(k): cost_scores(saved[k][:, :2]) if arm in NEURAL_ARMS else saved[k].copy()
                  for k in saved.files}
    return np.asarray(meta["heldRows"]), values


def nested(fast, reg, rows, arm):
    groups = reg["contract"]["sourceGroups"]
    seeds = SEEDS if arm in NEURAL_ARMS else ("deterministic",)
    results = {str(t): [] for t in TARGETS}; arrays = {}; final_choices = []
    for seed in seeds:
        predictions = {t: np.full(len(rows), -2, np.int8) for t in TARGETS}
        chosen = {t: np.full((len(rows), 2), np.nan) for t in TARGETS}
        folds = {t: [] for t in TARGETS}; final = None
        for outer in groups:
            train, held = partition(rows, (outer,)); local = {i: j for j, i in enumerate(train)}
            outer_indexes, out = frozen_scores(fast, arm, seed, (outer,))
            require(np.array_equal(held, outer_indexes), "Outer row order differs")
            candidates = {k: np.full((len(train), 2), np.nan) for k in out}
            if final is None:
                final = {k: np.full((len(rows), 2), np.nan) for k in out}
            for k in out:
                final[k][held] = out[k]
            for inner in groups:
                if inner == outer:
                    continue
                indexes, values = frozen_scores(fast, arm, seed, tuple(sorted((outer, inner))))
                keep = np.array([rows[i]["sourceGroup"] == inner for i in indexes])
                positions = [local[i] for i in indexes[keep]]
                for k in candidates:
                    candidates[k][positions] = values[k][keep]
            require(all(np.isfinite(v).all() for v in candidates.values()), "Missing inner prediction")
            for target in TARGETS:
                choice = metrics.select([rows[i] for i in train], candidates, target)
                prediction = metrics.decisions(out[choice["epoch"]], choice["farCutoff"], choice["nearCutoff"])
                predictions[target][held] = prediction; chosen[target][held] = out[choice["epoch"]]
                immutable(fast/"selection"/arm/str(seed)/str(target)/(outer+".json"),
                          {"innerRows": train.tolist(), "outerGroup": outer, "selected": choice})
                folds[target].append({"sourceGroup": outer, **{k: choice[k] for k in
                    ("epoch", "farCutoff", "nearCutoff", "feasible")}})
        for target in TARGETS:
            require(np.all(predictions[target] >= -1), "Outer predictions incomplete")
            ev = metrics.evaluate(rows, predictions[target], chosen[target][:, 0])
            results[str(target)].append({"seed": seed, "evaluation": ev, "folds": folds[target]})
            arrays[f"{seed}-{target}-decisions"] = predictions[target]
            arrays[f"{seed}-{target}-scores"] = chosen[target]
            choice = metrics.select(rows, final, target)
            immutable(fast/"final-selection"/arm/str(seed)/f"{target}.json", choice)
            final_choices.append({"seed": seed, "target": target, **{k: choice[k] for k in
                ("epoch", "farCutoff", "nearCutoff", "feasible")}})
    summary = {}
    for target in TARGETS:
        entries = results[str(target)]
        joint = consensus([arrays[f"{seed}-{target}-decisions"] for seed in seeds])
        arrays[f"consensus-{target}-decisions"] = joint
        summary[str(target)] = {"seeds": entries, "meanSeeds": seed_summary([e["evaluation"] for e in entries]),
            "consensus": metrics.evaluate(rows, joint),
            "infeasibleOuterPolicies": sum(not f["feasible"] for e in entries for f in e["folds"])}
    folder = fast/"results"/arm; folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/"predictions.npz", **arrays)
    immutable(folder/"summary.json", {"arm": arm, "targets": summary, "finalChoices": final_choices,
                                       "predictionsSha256": digest(folder/"predictions.npz")})
    return read(folder/"summary.json")


def resource_sample():
    def fields(path):
        return {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                for line in Path(path).read_text().splitlines() if ":" in line}
    memory, process = fields("/proc/meminfo"), fields("/proc/self/status")
    swap = {line.split()[0]: int(line.split()[1]) for line in Path("/proc/vmstat").read_text().splitlines()
            if line.split()[0] in ("pswpin", "pswpout")}
    return {"unixTime": time.time(), "residentKiB": int(process["VmRSS"].split()[0]),
            "processSwapKiB": int(process["VmSwap"].split()[0]),
            "availableKiB": int(memory["MemAvailable"].split()[0]), "swapCounters": swap}


def memory_guard():
    sample = resource_sample()
    require(sample["availableKiB"] >= 2*1024**2 and sample["processSwapKiB"] == 0,
            "Memory guard stopped between fits; preserve completed work and relieve host pressure")


class Monitor:
    def __init__(self, fast):
        self.fast = fast; self.stop = threading.Event(); self.samples = []
        self.thread = threading.Thread(target=self.loop, daemon=True)
    def sample(self):
        item = resource_sample(); self.samples.append(item)
        with (self.fast/"resources.jsonl").open("a") as stream:
            stream.write(json.dumps(item)+"\n")
    def loop(self):
        while not self.stop.wait(10):
            self.sample()
    def __enter__(self):
        self.sample(); self.thread.start(); return self
    def __exit__(self, *args):
        self.stop.set(); self.thread.join(); self.sample()


def progress(fast, **values):
    (fast/"progress.json").write_text(json.dumps(values))
    print(json.dumps(values), flush=True)


def mode_for(arm):
    return {"08-neural-quality": "raw", "09-neural-centered": "centered",
            "10-neural-differences": "differences"}[arm]


def run(report, device="cuda"):
    prior.h.prior.configure_torch()
    root, fast, reg, rows, windows, features = load()
    groups = reg["contract"]["sourceGroups"]
    exclusions = list(itertools.combinations(groups, 2))+[(g,) for g in groups]
    with Monitor(fast) as monitor:
        for number, excluded in enumerate(teacher_partitions(groups), 1):
            memory_guard(); teacher(fast, reg, rows, features["ranks"], excluded)
            progress(fast, phase="teachers", completed=number, total=93)
        for arm in CPU_ARMS:
            for number, excluded in enumerate(exclusions, 1):
                memory_guard(); fit_cpu(fast, reg, rows, features, arm, excluded)
                progress(fast, phase="cpu-fitting", arm=arm, completed=number, total=36)
        fg = torch.from_numpy(np.column_stack((features["ranks"], features["quality"])).astype(np.float32)).to(device)
        for arm in NEURAL_ARMS:
            transformed = temporal_transform(windows, mode_for(arm))
            raw = torch.from_numpy(transformed).to(device); del transformed
            completed = 0
            for seed in SEEDS:
                for excluded in exclusions:
                    memory_guard(); fit_neural(fast, reg, rows, features, raw, fg, arm, seed, excluded)
                    completed += 1
                    progress(fast, phase="neural-fitting", arm=arm, seed=seed, completed=completed, total=108)
            del raw; torch.cuda.empty_cache()
            # Keep durable checkpoints after each expensive arm without parallel jobs.
            shutil.copytree(fast/"fits"/arm, root/"fits"/arm, dirs_exist_ok=True)
        results = {}
        for arm in (*CPU_ARMS, *NEURAL_ARMS):
            results[arm] = nested(fast, reg, rows, arm)
            progress(fast, phase="nested-evaluation", arm=arm)
        # All held-source results and final policies are fixed before refitting.
        for arm in CPU_ARMS:
            memory_guard(); fit_cpu(fast, reg, rows, features, arm, ())
        deliveries = []
        for arm in NEURAL_ARMS:
            transformed = temporal_transform(windows, mode_for(arm))
            raw = torch.from_numpy(transformed).to(device); del transformed
            for seed in SEEDS:
                memory_guard(); fit_neural(fast, reg, rows, features, raw, fg, arm, seed, ())
            for choice in results[arm]["finalChoices"]:
                path = fast/"fits"/arm/str(choice["seed"])/fold_name(())/f"weights-{choice['epoch']}.npz"
                gpu, cpu = load_head(path, device), load_head(path, "cpu")
                a = cost_scores(prior.predict(gpu, raw[:16], fg[:16])[:, :2])
                b = cost_scores(prior.predict(cpu, raw[:16].cpu(), fg[:16].cpu())[:, :2])
                error = float(np.max(abs(a-b))); require(error < 2e-5, "Final CPU/GPU parity failed")
                deliveries.append({"arm": arm, **choice, "weightsSha256": digest(path),
                                   "bytes": path.stat().st_size, "cpuGpuMaximumError": error})
            del raw; torch.cuda.empty_cache()
        for name in ("fits", "teachers", "selection", "final-selection", "results"):
            shutil.copytree(fast/name, root/name, dirs_exist_ok=True)
    samples = monitor.samples
    resource = {"sampleCount": len(samples), "sampleIntervalSeconds": 10,
        "peakResidentMiB": max(x["residentKiB"] for x in samples)/1024,
        "maximumProcessSwapKiB": max(x["processSwapKiB"] for x in samples),
        "minimumWslAvailableGiB": min(x["availableKiB"] for x in samples)/1024**2,
        "wslSwapCounterDelta": {k: samples[-1]["swapCounters"][k]-samples[0]["swapCounters"][k]
                                for k in samples[0]["swapCounters"]},
        "singleFittingProcess": True, "cpuThreads": 2}
    shutil.copyfile(fast/"resources.jsonl", root/"resources.jsonl")
    result = {"kind": "serving-side-ten-precision-iterations-results-v1", "studyIndex": STUDY,
        "fastArtifactIndex": FAST, "contractSha256": reg["sha256"],
        "scope": {"serves": len(rows), "recordings": len({r['recordingIndex'] for r in rows}), "sourceGroups": len(groups)},
        "primaryTarget": .98, "targets": list(TARGETS), "experiments": results,
        "finalNeuralHeads": deliveries, "resources": resource,
        "newNestedNeuralFits": 324, "finalNeuralFits": 9, "teacherPartitions": 93,
        "cpuFittingPartitions": len(CPU_ARMS)*37, "protectedLabelsOpened": False,
        "productionChanged": False, "historicalControls": read(verified(reg['contract']['priorSummary']))['variants']}
    immutable(root/"summary.json", result); immutable(Path(report), result)
    progress(fast, phase="complete", experiments=10)
    return result
