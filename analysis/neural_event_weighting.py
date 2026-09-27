"""Equalize positive live-loss mass per original rally within each recording.

This module leaves source labels, supervision masks, positive-class weights and
sampling untouched. A chunk retains the frozen expanded-study fields at indexes
0..3 and appends an aligned live-loss multiplier at index 4.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import neural_expanded_development as expanded


METHOD = "per-record-original-event-positive-mass-v1"


def live_event_weights(row):
    """Return float32[T] loss multipliers and label-only event diagnostics.

    Eligible ticks are live positives with the existing live mask and validity.
    An original interval remains one event if ignored time divides its observed
    pieces. With P eligible ticks, N nonempty events and n_i ticks in event i,
    its multiplier is P/(N*n_i). All other ticks have multiplier one; their
    original masks still decide whether they contribute to loss.
    """
    e = row.example
    times = np.asarray(e.times)
    valid = np.asarray(e.valid, dtype=bool)
    targets, mask = np.asarray(e.targets), np.asarray(row.mask)
    if (times.ndim != 1 or valid.shape != times.shape
            or targets.shape != (len(times), 4) or mask.shape != targets.shape
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isfinite(targets).all() or not np.isfinite(mask).all()
            or not np.isin(targets[:, 0], (0., 1.)).all()
            or not np.isin(mask, (0., 1.)).all()):
        raise ValueError("invalid aligned binary live supervision")
    if np.any(mask[~valid, 0] != 0):
        raise ValueError("ignored/invalid tick has live supervision")
    if row.tier not in ("exact", "draft", "coverage"):
        raise ValueError("unknown supervision tier")
    ordered = sorted(e.truth, key=lambda interval: (interval.start, interval.end))
    for index, interval in enumerate(ordered):
        if (not np.isfinite((interval.start, interval.end)).all()
                or interval.end <= interval.start):
            raise ValueError("invalid original event interval")
        if index and interval.start < ordered[index-1].end:
            raise ValueError("overlapping original event identities")
    eligible = valid & (mask[:, 0] > 0) & (targets[:, 0] > 0)
    event_ticks, records = [], []
    ownership = np.zeros(len(times), dtype=np.int32)
    for index, interval in enumerate(e.truth):
        inside = (times >= interval.start) & (times < interval.end)
        ticks = np.flatnonzero(inside & eligible)
        ownership[ticks] += 1
        event_ticks.append(ticks)
        records.append({"eventIndex": index, "start": float(interval.start),
                        "end": float(interval.end), "tags": list(interval.tags),
                        "positiveSupervisedTicks": len(ticks),
                        "invalidTicksInsideEvent": int((inside & ~valid).sum()),
                        "maskedValidTicksInsideEvent": int((inside & valid & (mask[:, 0] == 0)).sum())})
    if np.any(ownership[eligible] != 1):
        raise ValueError("positive live tick lacks a unique original event")
    count = int(eligible.sum())
    nonempty = sum(bool(len(ticks)) for ticks in event_ticks)
    weights = np.ones(len(times), dtype=np.float32)
    for record, ticks in zip(records, event_ticks):
        multiplier = count / (nonempty * len(ticks)) if len(ticks) else None
        if multiplier is not None:
            weights[ticks] = multiplier
        record["multiplier"] = multiplier
        record["weightedPositiveMass"] = float(weights[ticks].sum(dtype=np.float64))
    event_multipliers = [record["multiplier"] for record in records if record["multiplier"] is not None]
    percentile_levels = (0, 25, 50, 75, 90, 95, 99, 100)

    def percentiles(values):
        return {str(level): float(value) for level, value in
                zip(percentile_levels, np.percentile(values, percentile_levels))} if len(values) else {}

    diagnostics = {
        "method": METHOD, "recordingId": e.id, "sourceGroup": e.group, "tier": row.tier,
        "originalEventCount": len(records), "eligibleEventCount": nonempty,
        "zeroSupervisedEventCount": len(records)-nonempty,
        "positiveSupervisedTicks": count,
        "weightedPositiveMass": float(weights[eligible].sum(dtype=np.float64)),
        "minimumPositiveMultiplier": float(weights[eligible].min()) if count else None,
        "maximumPositiveMultiplier": float(weights[eligible].max()) if count else None,
        "liveMultiplierSha256": hashlib.sha256(weights.astype("<f4", copy=False).tobytes(order="C")).hexdigest(),
        "liveMultiplierHashEncoding": "little-endian float32, C-order, one value per original cache tick",
        "negativeSupervisedTicks": int((valid & (mask[:, 0] > 0) & (targets[:, 0] == 0)).sum()),
        "eventMultiplierPercentiles": percentiles(event_multipliers),
        "positiveTickMultiplierPercentiles": percentiles(weights[eligible]),
        "events": records,
    }
    return weights, diagnostics


def make_weighted_chunks(rows, mean, scale, kind, weighting="per_rally"):
    """Append live weights to the unchanged frozen chunk/sampling contract."""
    if weighting not in ("per_rally", "uniform"):
        raise ValueError("unknown live weighting mode")
    rows = list(rows)
    chunks = expanded.make_chunks(rows, mean, scale, kind)
    aligned = []
    for row in rows:
        weights, _ = live_event_weights(row)
        if weighting == "uniform":
            weights = np.ones_like(weights)
        for start, end in expanded.base.segments(row.example.valid):
            for core_start in range(start, end, 128):
                core_end = min(end, core_start+128)
                left, right = max(start, core_start-62), min(end, core_end+62)
                # Exactly the frozen chunk's inclusion predicate and context.
                if row.mask[core_start:core_end].any():
                    aligned.append(weights[left:right])
    if len(aligned) != len(chunks) or any(len(w) != len(c[0]) for c, w in zip(chunks, aligned)):
        raise ValueError("weighted chunk alignment differs from frozen chunking")
    return [(*chunk, weights) for chunk, weights in zip(chunks, aligned)]


def batch_loss(model, chunks, indexes, pos_weight, head_weights, device):
    """Apply multipliers only to live BCE; divide by original mask counts."""
    import torch

    if not indexes:
        raise ValueError("empty loss batch")
    buckets = {}
    for index in indexes:
        chunk = chunks[index]
        if (len(chunk) != 5 or chunk[4].shape != (len(chunk[0]),)
                or not np.isfinite(chunk[4]).all() or np.any(chunk[4] <= 0)):
            raise ValueError("invalid live multiplier chunk")
        buckets.setdefault(len(chunk[0]), []).append(index)
    counts = np.sum([chunks[i][2].sum(axis=0) for i in indexes], axis=0)
    denominator = torch.tensor(np.maximum(counts, 1), device=device)
    weights = torch.tensor(head_weights, device=device)
    total = None
    for batch_indices in buckets.values():
        x, y, mask = [torch.from_numpy(np.stack([chunks[i][j] for i in batch_indices])).to(device) for j in range(3)]
        live = torch.from_numpy(np.stack([chunks[i][4] for i in batch_indices])).to(device)
        elements = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_weight, reduction="none")
        elements = torch.cat((elements[:, :, :1]*live[:, :, None], elements[:, :, 1:]), dim=2)
        loss = ((elements * mask).sum(dim=(0, 1)) / denominator * weights).sum()
        total = loss if total is None else total + loss
    return total
