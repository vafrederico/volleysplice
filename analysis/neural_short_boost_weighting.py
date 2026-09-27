"""Fixed short-rally live loss and its recording-matched positive-mass control.

Only existing supervised live-positive ticks receive multipliers. Short means
the ORIGINAL interval duration is at most three seconds. The short arm doubles
those ticks; long rallies retain their original live objective. The global
control applies float32(1+S/P) to all supervised positives in that recording.
This matches full-recording positive mass within float32 rounding, not the mass
of each randomly sampled minibatch. Masks, other heads and denominators stay
unchanged in every arm.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import neural_event_weighting as original_weighting
from . import neural_expanded_development as expanded
from .neural_event_weighting import batch_loss

METHOD = "fixed-short-rally-live-boost-with-record-matched-control-v1"
WEIGHTING_MODES = ("baseline", "global_control", "short_boost")
MODEL_KINDS = ("tcn", "dino_tcn")
SHORT_SECONDS = 3.0
SHORT_MULTIPLIER = 2.0


def live_event_weights(row, mode="short_boost"):
    """Return float32[T] multipliers and original-event diagnostic records.

    Reuse the frozen validator for binary alignment, unique original-event
    ownership and exclusion of ignored time. Its old inverse-duration weights
    are discarded; no source targets, masks or metadata are mutated.
    """
    if mode not in WEIGHTING_MODES:
        raise ValueError("unknown short-boost weighting mode")
    _, validated = original_weighting.live_event_weights(row)
    e = row.example
    eligible = e.valid & (row.mask[:, 0] > 0) & (e.targets[:, 0] > 0)
    if row.tier == "coverage" and eligible.any():
        raise ValueError("coverage tier must not supervise live positives")
    short = np.zeros(len(e.times), dtype=bool)
    for interval in e.truth:
        if interval.end - interval.start <= SHORT_SECONDS:
            short |= (e.times >= interval.start) & (e.times < interval.end)
    short &= eligible
    positive_ticks, short_ticks = int(eligible.sum()), int(short.sum())
    ideal_control = 1. + short_ticks / positive_ticks if positive_ticks else 1.
    control = np.float32(ideal_control)
    weights = np.ones(len(e.times), dtype=np.float32)
    if mode == "short_boost":
        weights[short] = SHORT_MULTIPLIER
    elif mode == "global_control":
        weights[eligible] = control
    records = []
    for source in validated["events"]:
        inside = (e.times >= source["start"]) & (e.times < source["end"])
        selected = inside & eligible
        count = source["positiveSupervisedTicks"]
        duration = source["end"] - source["start"]
        records.append({
            **{key: value for key, value in source.items() if key not in ("multiplier", "weightedPositiveMass")},
            "originalDurationSeconds": duration, "isShortOriginalEvent": duration <= SHORT_SECONDS,
            "multiplier": float(weights[selected][0]) if count else None,
            "weightedPositiveMass": float(weights[selected].sum(dtype=np.float64)),
        })
    expected_mass = positive_ticks + (short_ticks if mode != "baseline" else 0)
    actual_mass = float(weights[eligible].sum(dtype=np.float64))
    # At multipliers in[1,2], the float32 rounding error per tick is at most
    # epsilon. Sum in float64 so this bound measures storage, not reduction error.
    tolerance = positive_ticks * float(np.finfo(np.float32).eps)
    if abs(actual_mass - expected_mass) > tolerance:
        raise ValueError("positive-mass control exceeds float32 rounding tolerance")
    diagnostics = {
        **{key: validated[key] for key in ("recordingId", "sourceGroup", "tier", "originalEventCount",
                                          "eligibleEventCount", "zeroSupervisedEventCount", "negativeSupervisedTicks")},
        "method": METHOD, "mode": mode, "shortDurationSeconds": SHORT_SECONDS,
        "shortPositiveMultiplier": SHORT_MULTIPLIER,
        "positiveSupervisedTicks": positive_ticks, "shortPositiveSupervisedTicks": short_ticks,
        "longPositiveSupervisedTicks": positive_ticks - short_ticks,
        "shortOriginalEventCount": sum(record["isShortOriginalEvent"] for record in records),
        "eligibleShortEventCount": sum(record["isShortOriginalEvent"] and record["positiveSupervisedTicks"] > 0 for record in records),
        "zeroSupervisedShortEventCount": sum(record["isShortOriginalEvent"] and record["positiveSupervisedTicks"] == 0 for record in records),
        "idealGlobalPositiveMultiplier": ideal_control, "globalPositiveMultiplierFloat32": float(control),
        "expectedPositiveMass": expected_mass, "weightedPositiveMass": actual_mass,
        "positiveMassRoundingError": actual_mass - expected_mass, "positiveMassAbsoluteTolerance": tolerance,
        "minimumPositiveMultiplier": float(weights[eligible].min()) if positive_ticks else None,
        "maximumPositiveMultiplier": float(weights[eligible].max()) if positive_ticks else None,
        "liveMultiplierSha256": hashlib.sha256(weights.astype("<f4", copy=False).tobytes(order="C")).hexdigest(),
        "liveMultiplierHashEncoding": "little-endian float32, C-order, one value per original cache tick",
        "events": records,
    }
    return weights, diagnostics


def make_weighted_chunks(rows, mean, scale, kind, weighting="short_boost"):
    """Keep frozen real-halo chunks and append one aligned multiplier vector.

    The frozen expanded chunker already uses62 ticks of context for BOTH TCN
    representations. Passing the external kind unchanged to that chunker keeps
    AV scaling and the raw3944-dimensional DINO input contract distinct.
    """
    if kind not in MODEL_KINDS:
        raise ValueError("unsupported short-boost temporal architecture")
    if weighting not in WEIGHTING_MODES:
        raise ValueError("unknown short-boost weighting mode")
    rows = list(rows)
    chunks = expanded.make_chunks(rows, mean, scale, kind)
    aligned = []
    for row in rows:
        weights, _ = live_event_weights(row, weighting)
        for start, end in expanded.base.segments(row.example.valid):
            for core_start in range(start, end, 128):
                core_end = min(end, core_start+128)
                left, right = max(start, core_start-62), min(end, core_end+62)
                if row.mask[core_start:core_end].any():
                    aligned.append(weights[left:right])
    dimension = 3944 if kind == "dino_tcn" else 104
    if len(aligned) != len(chunks) or any(
            len(weights) != len(chunk[0]) or chunk[0].shape != (len(weights), dimension)
            for chunk, weights in zip(chunks, aligned)):
        raise ValueError("weighted chunk alignment or representation differs from fixed TCN contract")
    return [(*chunk, weights) for chunk, weights in zip(chunks, aligned)]
