"""Fold-isolated four-head fitting with explicit new-feature/model injection.

The historical sample order, real halos, short-boost loss, optimizer, epoch
schedule and dropout seeding are preserved; no legacy module globals are patched.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import neural_expanded_development as expanded
from . import neural_short_boost_weighting as weighting
from .neural_context_fit import _validate_fold, _validated_predictions, _verify_artifacts
from .neural_event_balanced_development import canonical_hash, read
from .recognition_temporal_model import model_for, model_metadata

base = expanded.base


def scalar_indexes(config):
    if not config.scalar_dimension:
        return np.empty(0, dtype=np.int64)
    start = 104 + (config.token_count * config.token_dimension if config.family == 'mobile' else 0)
    return np.arange(start, start + config.scalar_dimension)


def fit_scaler(rows, config):
    mean, scale = base.fit_scaler([r.example for r in rows])
    indexes = scalar_indexes(config)
    if len(indexes):
        values = np.concatenate([r.example.values[r.example.valid][:, indexes] for r in rows])
        mean = np.concatenate((mean, values.mean(0, dtype=np.float64).astype(np.float32)))
        scale = np.concatenate((scale, np.maximum(values.std(0, dtype=np.float64), 1e-4).astype(np.float32)))
    return mean, scale


def standardized(example, mean, scale, config):
    if example.values.shape != (len(example.times), config.input_dimension):
        raise ValueError('Feature dimensions differ from registered model')
    values = example.values.copy()
    values[:, :104] = np.clip((values[:, :104] - mean[:104]) / scale[:104], -10, 10)
    indexes = scalar_indexes(config)
    if len(indexes):
        values[:, indexes] = np.clip((values[:, indexes] - mean[104:]) / scale[104:], -10, 10)
    return values


def make_chunks(rows, mean, scale, config, mode):
    chunks = []
    counts = {r.example.group: sum(s.example.group == r.example.group for s in rows) for r in rows}
    for row in rows:
        e = row.example
        values = standardized(e, mean, scale, config)
        multipliers, _ = weighting.live_event_weights(row, mode)
        one = []
        for start, end in base.segments(e.valid):
            for core_start in range(start, end, 128):
                core_end = min(end, core_start + 128)
                left, right = max(start, core_start - 62), min(end, core_end + 62)
                mask = np.zeros((right-left, 4), dtype=np.float32)
                mask[core_start-left:core_end-left] = row.mask[core_start:core_end]
                if mask.any():
                    one.append((values[left:right], e.targets[left:right], mask, multipliers[left:right]))
        for x, y, mask, live in one:
            chunks.append((x, y, mask, 1. / (counts[e.group] * len(one)), live))
    return chunks


def predict(model, example, mean, scale, config, device):
    import torch
    model.eval()
    result = np.zeros((len(example.times), 4), np.float32)
    values = standardized(example, mean, scale, config)
    with torch.inference_mode():
        for start, end in base.segments(example.valid):
            for core_start in range(start, end, 128):
                core_end = min(end, core_start + 128)
                left, right = max(start, core_start-62), min(end, core_end+62)
                probabilities = torch.sigmoid(model(torch.from_numpy(values[left:right][None]).to(device)))[0].cpu().numpy()
                result[core_start:core_end] = probabilities[core_start-left:core_end-left]
    return result


def fit_model(train, auxiliary, validation, config, seed, epochs, destination, device,
              contract_hash, mode='short_boost'):
    import torch
    _validate_fold(train, auxiliary, validation)
    epochs = tuple(epochs)
    if (not epochs or any(isinstance(e, bool) or not isinstance(e, int) or e < 1 for e in epochs)
            or epochs != tuple(sorted(set(epochs)))):
        raise ValueError('Invalid epochs')
    if mode not in weighting.WEIGHTING_MODES:
        raise ValueError('Unknown loss mode')
    destination = Path(destination)
    rows = {'exact': train, **auxiliary}
    identity = {'contractSha256': contract_hash, 'kind': f'{config.family}_{config.head}',
                'seed': seed, 'lossArm': mode, 'epochs': list(epochs), 'model': model_metadata(config),
                'trainIds': [r.example.id for r in train],
                'auxiliaryIds': {tier: [r.example.id for r in records] for tier, records in auxiliary.items()},
                'validationIds': [e.id for e in validation],
                'trainGroups': sorted({r.example.group for r in train}),
                'auxiliaryGroups': {tier: sorted({r.example.group for r in records}) for tier, records in auxiliary.items()},
                'validationGroups': sorted({e.group for e in validation}),
                'liveLossWeighting': {'mode': mode, 'rows': {
                    tier: [weighting.live_event_weights(r, mode)[1] for r in records] for tier, records in rows.items()}}}
    done = destination / 'completed.json'
    if done.exists():
        meta = read(done)
        if any(meta.get(k) != v for k, v in identity.items()):
            raise ValueError('Resume identity changed')
        _verify_artifacts(destination, meta)
        return {epoch: _validated_predictions(destination/f'predictions-{epoch}.npz', validation) for epoch in epochs}
    if destination.exists():
        raise FileExistsError(f'Refusing incomplete fit directory: {destination}')
    destination.mkdir(parents=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    mean, scale = fit_scaler(train, config)
    counts = expanded.supervision_counts(train)
    positive = np.asarray(counts['positiveMass'])
    positive_weight = np.minimum(20., np.sqrt((np.asarray(counts['valid'])-positive) / np.maximum(positive, 1.)))
    pos_weight = torch.tensor(positive_weight, dtype=torch.float32, device=device)
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    model = model_for(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    pools = {tier: make_chunks(records, mean, scale, config, mode) for tier, records in rows.items()}
    pools = {tier: chunks for tier, chunks in pools.items() if chunks}
    rngs = {tier: np.random.default_rng(np.random.SeedSequence([seed, i])) for i, tier in enumerate(expanded.HEAD_WEIGHTS)}
    exposure = {tier: hashlib.sha256() for tier in pools}
    history, outputs, steps = [], {}, 0
    started = time.perf_counter()
    for epoch in range(1, max(epochs)+1):
        model.train()
        loss_sum = 0.
        batches = expanded.epoch_batches(pools['exact'], rngs['exact'])
        for exact_indexes in batches:
            optimizer.zero_grad(set_to_none=True)
            combined = None
            for stream, tier in enumerate(expanded.HEAD_WEIGHTS):
                if tier not in pools:
                    continue
                indexes = exact_indexes if tier == 'exact' else rngs[tier].choice(
                    len(pools[tier]), len(exact_indexes), replace=True, p=expanded.sampling_weights(pools[tier])).tolist()
                exposure[tier].update(json.dumps([epoch, steps, indexes], separators=(',', ':')).encode())
                torch.manual_seed(seed + steps*11 + stream*100000000)
                loss = weighting.batch_loss(model, pools[tier], indexes, pos_weight, expanded.HEAD_WEIGHTS[tier], device)
                combined = loss if combined is None else combined + loss
            if not torch.isfinite(combined):
                raise ValueError('Non-finite loss')
            combined.backward()
            expanded.clip_gradients(model, 'tcn')
            optimizer.step()
            loss_sum += float(combined.detach().cpu())
            steps += 1
        history.append({'epoch': epoch, 'loss': loss_sum/len(batches), 'optimizerSteps': steps,
                        'exposureSha256': {tier: value.hexdigest() for tier, value in exposure.items()}})
        if epoch in epochs:
            outputs[epoch] = {e.id: predict(model, e, mean, scale, config, device) for e in validation}
            np.savez_compressed(destination/f'predictions-{epoch}.npz', **outputs[epoch])
            state = {f'model::{key}': value.detach().cpu().numpy() for key, value in model.state_dict().items()}
            np.savez_compressed(destination/f'weights-{epoch}.npz', mean=mean, scale=scale, **state)
        base.write_json(destination/'progress.json', {'epoch': epoch, 'maximumEpoch': max(epochs),
                        'wallSeconds': time.perf_counter()-started, 'optimizerSteps': steps})
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    base.write_json(done, {**identity, 'scalerTrainIds': identity['trainIds'], 'positiveWeight': positive_weight.tolist(),
        'supervisedCounts': {tier: expanded.supervision_counts(records) for tier, records in rows.items()},
        'history': history, 'optimizerSteps': steps,
        'exposureSha256': {tier: value.hexdigest() for tier, value in exposure.items()},
        'wallSeconds': time.perf_counter()-started,
        'peakAllocatedCudaBytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
        'artifacts': {f'{stem}-{epoch}.npz': base.file_sha256(destination/f'{stem}-{epoch}.npz')
                      for epoch in epochs for stem in ('weights', 'predictions')}})
    return outputs
