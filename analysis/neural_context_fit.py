"""Optional, unregistered context-comparison fitter; no data-loading entrypoint.

Only the explicitly selected model factory/profile differs from the frozen
short-boost fitter. Original scaler, masks, 62-tick real halos, sampling, dropout
streams, optimizer, checkpoint schedule and prediction function are retained.
No imported module globals are patched. The caller owns registration and data
identity; this module does not select or launch an experiment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import neural_expanded_development as expanded
from . import neural_short_boost_weighting as weighting_module
from . import transfer_temporal_model as frozen_models
from .neural_event_balanced_development import canonical_hash, read
from .short_context_temporal_model import MODEL_KINDS, model_for, model_metadata

base = expanded.base
ARMS = weighting_module.WEIGHTING_MODES


def weighting_metadata(rows, mode):
    return {"mode": mode, "rows": {tier: [weighting_module.live_event_weights(r, mode)[1] for r in records]
                                   for tier, records in rows.items()}}


def training_identity(train, auxiliary, kind, seed, contract_hash, *, context="short", weighting="baseline"):
    """Role-independent identity; validation membership and path are excluded.

    Ordered records are retained because their order affects sampling. The
    external contract must bind feature/label/source bytes. Checkpoint epochs
    remain separately resume-bound; evaluation at an epoch does not alter RNG.
    """
    if kind not in MODEL_KINDS or weighting not in ARMS:
        raise ValueError("unsupported architecture or live weighting mode")
    return {"contractSha256": contract_hash, "kind": kind, "seed": seed, "lossArm": weighting,
            "contextProfile": context, "model": model_metadata(kind, context=context),
            "trainIds": [r.example.id for r in train],
            "auxiliaryIds": {tier: [r.example.id for r in rows] for tier, rows in auxiliary.items()},
            "trainGroups": sorted({r.example.group for r in train}),
            "auxiliaryGroups": {tier: sorted({r.example.group for r in rows}) for tier, rows in auxiliary.items()},
            "liveLossWeighting": weighting_metadata({"exact": train, **auxiliary}, weighting)}


def _validate_fold(train, auxiliary, validation):
    if set(auxiliary) - {"draft", "coverage"}:
        raise ValueError("unsupported auxiliary tier")
    all_train = list(train) + [r for rows in auxiliary.values() for r in rows]
    train_ids = [r.example.id for r in all_train]
    valid_ids = [e.id for e in validation]
    if (not train or not validation or len(set(train_ids)) != len(train_ids)
            or len(set(valid_ids)) != len(valid_ids) or set(train_ids) & set(valid_ids)
            or {e.group for e in validation} & {r.example.group for r in all_train}):
        raise ValueError("overlapping, duplicate or empty source fold")


def _verify_artifacts(destination, meta):
    inventory = {f"{stem}-{epoch}.npz" for epoch in meta["epochs"] for stem in ("weights", "predictions")}
    if set(meta["artifacts"]) != inventory:
        raise ValueError("resume artifact inventory changed")
    for name, digest in meta["artifacts"].items():
        if base.file_sha256(destination/name) != digest:
            raise ValueError("resume artifact changed")


def _validated_predictions(path, examples):
    with np.load(path, allow_pickle=False) as cache:
        if set(cache.files) != {e.id for e in examples}:
            raise ValueError("prediction recording inventory differs")
        result = {}
        for e in examples:
            values = cache[e.id]
            if (values.shape != (len(e.times), 4) or values.dtype != np.float32
                    or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))
                    or np.any(values[~e.valid] != 0)):
                raise ValueError("invalid checkpoint predictions")
            result[e.id] = values.copy()
    return result


def fit_model(train, auxiliary, validation, kind, seed, epochs, destination, device,
              contract_hash, weighting="baseline", *, context="short"):
    """Preserve the frozen training recipe with explicit context injection."""
    import torch

    destination = Path(destination)
    _validate_fold(train, auxiliary, validation)
    epochs = tuple(epochs)
    if (not epochs or any(isinstance(e, bool) or not isinstance(e, int) or e < 1 for e in epochs)
            or epochs != tuple(sorted(set(epochs)))):
        raise ValueError("checkpoint epochs must be positive, sorted and unique")
    train_identity = training_identity(train, auxiliary, kind, seed, contract_hash, context=context, weighting=weighting)
    train_identity_sha = canonical_hash(train_identity)
    valid_groups = {e.group for e in validation}
    identity = {"trainIds": train_identity["trainIds"], "auxiliaryIds": train_identity["auxiliaryIds"],
                "validationIds": [e.id for e in validation]}
    row_groups = {"exact": train, **auxiliary}
    weighting_audit = train_identity["liveLossWeighting"]
    completed = destination / "completed.json"
    if completed.exists():
        meta = read(completed)
        if (meta["contractSha256"] != contract_hash or any(meta[k] != v for k, v in identity.items())
                or meta["epochs"] != list(epochs) or meta["kind"] != kind or meta["seed"] != seed
                or meta.get("lossArm") != weighting or meta.get("liveLossWeighting") != weighting_audit
                or meta.get("model") != model_metadata(kind, context=context)
                or meta.get("contextProfile") != context or meta.get("trainingIdentity") != train_identity
                or meta.get("trainingIdentitySha256") != train_identity_sha
                or meta.get("validationGroups") != sorted(valid_groups)):
            raise ValueError("resume contract, identity, context or weighting changed")
        _verify_artifacts(destination, meta)
        return {epoch: _validated_predictions(destination/f"predictions-{epoch}.npz", validation) for epoch in epochs}
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
    model = model_for(kind, context=context).to(device)
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
        "contextProfile": context, "trainingIdentity": train_identity, "trainingIdentitySha256": train_identity_sha,
        "trainGroups": train_identity["trainGroups"], "auxiliaryGroups": train_identity["auxiliaryGroups"],
        "validationGroups": sorted(valid_groups), "scalerTrainIds": identity["trainIds"],
        "positiveWeight": positive_weight.tolist(),
        "supervisedCounts": {tier: expanded.supervision_counts(rows) for tier, rows in row_groups.items()},
        "liveLossWeighting": weighting_audit, "model": model_metadata(kind, context=context),
        "parameters": sum(p.numel() for p in model.parameters()), "history": history, "optimizerSteps": steps,
        "exposureSha256": {tier: digest.hexdigest() for tier, digest in exposure.items()},
        "wallSeconds": time.perf_counter()-started,
        "peakAllocatedCudaBytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
        "artifacts": {f"{stem}-{epoch}.npz": base.file_sha256(destination/f"{stem}-{epoch}.npz")
                      for epoch in epochs for stem in ("weights", "predictions")}})
    return outputs


def predict_checkpoint(destination, epoch, examples, *, kind, context, expected_identity, device="cpu"):
    """Verify and replay a tensor-only checkpoint on excluded-group examples.

    Required expected_identity fields: contractSha256, trainIds, auxiliaryIds,
    kind, seed, lossArm. Any additional supplied field must also match. Current
    context checkpoints bind full metadata/profile; historical transfer files
    are accepted only with explicit original context and their original model
    metadata. Their external contract/file hashes remain caller responsibilities.
    This helper writes nothing and preserves the caller's CPU initialization RNG.
    """
    import torch

    destination = Path(destination)
    meta = read(destination/"completed.json")
    required = {"contractSha256", "trainIds", "auxiliaryIds", "kind", "seed", "lossArm"}
    if (not required <= set(expected_identity) or any(meta.get(k) != v for k, v in expected_identity.items())
            or expected_identity["kind"] != kind or epoch not in meta["epochs"]):
        raise ValueError("checkpoint expected identity differs or is incomplete")
    if "contextProfile" in meta:
        if (meta["contextProfile"] != context or meta["model"] != model_metadata(kind, context=context)
                or canonical_hash(meta["trainingIdentity"]) != meta["trainingIdentitySha256"]
                or any(meta["trainingIdentity"].get(k) != meta[k] for k in meta["trainingIdentity"])):
            raise ValueError("checkpoint context/model identity differs")
    elif context != "original" or meta["model"] != frozen_models.model_metadata(kind):
        raise ValueError("historical checkpoint requires original context/model")
    training_groups = set(meta["trainGroups"]) | {g for groups in meta["auxiliaryGroups"].values() for g in groups}
    training_ids = set(meta["trainIds"]) | {rid for ids in meta["auxiliaryIds"].values() for rid in ids}
    if (not examples or len({e.id for e in examples}) != len(examples)
            or training_groups & {e.group for e in examples} or training_ids & {e.id for e in examples}):
        raise ValueError("checkpoint prediction overlaps fitting source groups")
    _verify_artifacts(destination, meta)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        model = model_for(kind, context=context)
    with np.load(destination/f"weights-{epoch}.npz", allow_pickle=False) as cache:
        template = model.state_dict()
        if set(cache.files) != {"mean", "scale", *(f"model::{name}" for name in template)}:
            raise ValueError("checkpoint tensor inventory differs")
        mean, scale = cache["mean"].copy(), cache["scale"].copy()
        if (mean.shape != (104,) or scale.shape != (104,) or mean.dtype != np.float32 or scale.dtype != np.float32
                or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError("invalid checkpoint scaler")
        state = {}
        for name, parameter in template.items():
            value = cache[f"model::{name}"]
            if value.shape != tuple(parameter.shape) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("invalid checkpoint model tensor")
            state[name] = torch.from_numpy(value.copy())
    model.load_state_dict(state, strict=True)
    model.to(device)
    return {e.id: expanded.predict(model, e, mean, scale, kind, device) for e in examples}
