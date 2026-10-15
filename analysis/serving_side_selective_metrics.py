"""Precision-first serving-side evaluation and inner-fold policy selection.

Scores contain the near probability and learned acceptance probability. Routing
is fixed at near probability >= 0.5. Thresholds act on acceptance probability
times the probability of the routed side. Selection must receive inner-fold
predictions only; this module never fits a model or accesses external data.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np


CUTOFFS = np.asarray([value / 100 for value in range(100)] + [.995, .999, .9999, 1., 2.])
MIN_ACCEPTED_PER_SIDE = 20
MIN_GROUPS_PER_SIDE = 4


def _rows(rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([row["label"] for row in rows])
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must be binary 0/1")
    groups = np.asarray([str(row["sourceGroup"]) for row in rows], dtype=str)
    if any(not value for value in groups):
        raise ValueError("Each row requires a nonempty source group")
    return labels.astype(np.int8), groups


def _scores(scores: np.ndarray, count: int | None = None) -> np.ndarray:
    result = np.asarray(scores, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or (count is not None and len(result) != count):
        raise ValueError("Scores must have shape (rows, 2)")
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Scores must be finite probabilities in [0, 1]")
    return result


def decisions(scores: np.ndarray, far_cutoff: float, near_cutoff: float) -> np.ndarray:
    """Return -1 for review, 0 for far, 1 for near; zero acceptance always reviews.

    Positive score equality with the relevant cutoff accepts. An acceptance score
    of zero can encode an evidence disagreement veto, including at cutoff zero.
    """
    values = _scores(scores)
    cutoffs = np.asarray([far_cutoff, near_cutoff], dtype=np.float64)
    if not np.isfinite(cutoffs).all() or np.any((cutoffs < 0) | (cutoffs > 2)):
        raise ValueError("Cutoffs must be finite values in [0, 2]")
    side = (values[:, 0] >= .5).astype(np.int8)
    confidence = values[:, 1] * np.where(side == 1, values[:, 0], 1 - values[:, 0])
    return np.where((confidence > 0) & (confidence >= cutoffs[side]), side, -1).astype(np.int8)


def _divide(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _wilson_lower(correct: int, accepted: int) -> float | None:
    if not accepted:
        return None
    z = 1.959963984540054
    estimate = correct / accepted
    numerator = estimate + z * z / (2 * accepted)
    numerator -= z * math.sqrt(estimate * (1 - estimate) / accepted + z * z / (4 * accepted * accepted))
    return float(max(0., numerator / (1 + z * z / accepted)))


def _counts(labels: np.ndarray, prediction: np.ndarray) -> dict:
    accepted_mask = prediction >= 0
    total = len(labels)
    accepted = int(accepted_mask.sum())
    correct = int(((prediction == labels) & accepted_mask).sum())
    errors = accepted - correct
    near = prediction == 1
    far = prediction == 0
    return {
        "rows": total,
        "accepted": accepted,
        "reviewed": total - accepted,
        "correct": correct,
        "errors": errors,
        "precision": _divide(correct, accepted),
        "nearPrecision": _divide(int((near & (labels == 1)).sum()), int(near.sum())),
        "farPrecision": _divide(int((far & (labels == 0)).sum()), int(far.sum())),
        "nearAccepted": int(near.sum()),
        "farAccepted": int(far.sum()),
        "coverage": _divide(accepted, total),
        "reviewFraction": _divide(total - accepted, total),
        "silentErrorFraction": _divide(errors, total),
        "correctAutoFraction": _divide(correct, total),
        # This interval is descriptive, not a source-shift or selection guarantee.
        "precisionLowerWilson95DESCRIPTIVE": _wilson_lower(correct, accepted),
    }


def evaluate(rows: Sequence[dict], prediction: np.ndarray, underlying: np.ndarray | None = None) -> dict:
    """Report automatic precision and review burden, with zero acceptance explicit.

    ``underlying`` may contain binary guesses or near probabilities. Its guesses
    are routed at >= 0.5 when calculating capture of otherwise wrong predictions.
    Source-group macro coverage includes every group. Macro precision and worst
    supported group precision include only groups with any accepted predictions.
    """
    labels, groups = _rows(rows)
    prediction = np.asarray(prediction)
    if prediction.shape != labels.shape or not np.all(np.isin(prediction, [-1, 0, 1])):
        raise ValueError("Predictions must contain one of -1, 0, 1 for each row")
    prediction = prediction.astype(np.int8)
    pooled = _counts(labels, prediction)
    by_group = {group: _counts(labels[groups == group], prediction[groups == group])
                for group in sorted(set(groups))}
    supported = [item["precision"] for item in by_group.values() if item["precision"] is not None]
    by_label = {name: _counts(labels[labels == side], prediction[labels == side])
                for side, name in ((0, "far"), (1, "near"))}
    result = {
        "pooled": pooled,
        "sourceGroupMacroCoverage": float(np.mean([item["coverage"] for item in by_group.values()])) if by_group else None,
        "sourceGroupMacroPrecision": float(np.mean(supported)) if supported else None,
        "worstSupportedGroupPrecision": float(min(supported)) if supported else None,
        "supportedSourceGroups": len(supported),
        "bySourceGroup": by_group,
        "byLabel": by_label,
        "perLabelCoverage": {name: item["coverage"] for name, item in by_label.items()},
    }
    if underlying is not None:
        underlying = np.asarray(underlying, dtype=np.float64)
        if underlying.shape != labels.shape or not np.isfinite(underlying).all() or np.any((underlying < 0) | (underlying > 1)):
            raise ValueError("Underlying guesses must be binary or near probabilities, one per row")
        wrong = (underlying >= .5) != labels
        error_count = int(wrong.sum())
        captured = int((wrong & (prediction == -1)).sum())
        pooled.update({
            "underlyingErrors": error_count,
            "reviewedUnderlyingErrors": captured,
            "underlyingErrorReviewCapture": _divide(captured, error_count),
        })
    return result


def _side_statistics(labels: np.ndarray, group_codes: np.ndarray, group_count: int,
                     routed: np.ndarray, confidence: np.ndarray, side: int) -> dict:
    """One threshold-by-row table per side, never a threshold-pair-by-row table."""
    accepted = ((confidence[None, :] > 0) & (confidence[None, :] >= CUTOFFS[:, None])
                & (routed[None, :] == side))
    correct = accepted & (labels[None, :] == side)
    count = accepted.sum(axis=1)
    correct_count = correct.sum(axis=1)
    group_accepted = np.stack([accepted[:, group_codes == group].sum(axis=1) for group in range(group_count)], axis=1)
    group_correct = np.stack([correct[:, group_codes == group].sum(axis=1) for group in range(group_count)], axis=1)
    precision = np.divide(correct_count, count, out=np.zeros(len(CUTOFFS), dtype=float), where=count > 0)
    return {"accepted": count, "correct": correct_count, "precision": precision,
            "groupAccepted": group_accepted, "groupCorrect": group_correct,
            "groups": (group_accepted > 0).sum(axis=1)}


def select(rows: Sequence[dict], candidates: Mapping[int, np.ndarray], target: float) -> dict:
    """Select epoch and per-side cutoffs using only the supplied inner rows.

    Each accepted side must attain ``target`` precision, have >= 20 accepted rows,
    and cover >= 4 source groups. Accepted source-group macro precision must also
    attain ``target``. Feasible policies rank by macro coverage, pooled coverage,
    minimum side precision, fewer errors, earlier epoch, then larger cutoff sum,
    larger minimum cutoff, larger far cutoff, larger near cutoff. A failed search
    returns an explicit review-all policy. Candidate counts include the full grid.
    """
    if not isinstance(target, (int, float, np.integer, np.floating)) or not np.isfinite(target) or not 0 < target <= 1:
        raise ValueError("Target precision must be in (0, 1]")
    labels, groups = _rows(rows)
    if not len(rows):
        raise ValueError("Selection requires at least one inner row")
    if not candidates:
        raise ValueError("Selection requires at least one candidate epoch")
    if any(isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)) or epoch < 0 for epoch in candidates):
        raise ValueError("Candidate epochs must be nonnegative integers")
    epochs = sorted(candidates)
    _, group_codes, group_sizes = np.unique(groups, return_inverse=True, return_counts=True)
    group_count = len(group_sizes)
    best_rank = None
    best_policy = None
    feasible_count = 0
    for epoch in epochs:
        values = _scores(candidates[epoch], len(rows))
        routed = (values[:, 0] >= .5).astype(np.int8)
        confidence = values[:, 1] * np.where(routed == 1, values[:, 0], 1 - values[:, 0])
        far = _side_statistics(labels, group_codes, group_count, routed, confidence, 0)
        near = _side_statistics(labels, group_codes, group_count, routed, confidence, 1)
        side_ok = lambda stats: ((stats["accepted"] >= MIN_ACCEPTED_PER_SIDE)
                                 & (stats["groups"] >= MIN_GROUPS_PER_SIDE)
                                 & (stats["precision"] >= target))
        far_indices = np.flatnonzero(side_ok(far))
        near_indices = np.flatnonzero(side_ok(near))
        if not len(far_indices) or not len(near_indices):
            continue
        # Combining aggregate counts keeps memory proportional to grid^2*groups,
        # independent of the number of rows in the inner validation population.
        accepted = far["groupAccepted"][far_indices, None, :] + near["groupAccepted"][None, near_indices, :]
        correct = far["groupCorrect"][far_indices, None, :] + near["groupCorrect"][None, near_indices, :]
        supported = accepted > 0
        precision = np.divide(correct, accepted, out=np.zeros_like(correct, dtype=float), where=supported)
        macro_precision = precision.sum(axis=2) / supported.sum(axis=2)
        feasible = macro_precision >= target
        feasible_count += int(feasible.sum())
        if not feasible.any():
            continue
        macro_coverage = (accepted / group_sizes).mean(axis=2)
        pooled_coverage = accepted.sum(axis=2) / len(rows)
        minimum_precision = np.minimum(far["precision"][far_indices, None], near["precision"][None, near_indices])
        errors = (accepted - correct).sum(axis=2)
        fi, ni = np.nonzero(feasible)
        far_cutoffs = CUTOFFS[far_indices[fi]]
        near_cutoffs = CUTOFFS[near_indices[ni]]
        # lexsort's last key is primary. Epoch is compared between epoch winners.
        order = np.lexsort((near_cutoffs, far_cutoffs, np.minimum(far_cutoffs, near_cutoffs),
                            far_cutoffs + near_cutoffs, -errors[fi, ni], minimum_precision[fi, ni],
                            pooled_coverage[fi, ni], macro_coverage[fi, ni]))
        winner = int(order[-1])
        f, n = int(fi[winner]), int(ni[winner])
        far_cutoff, near_cutoff = float(far_cutoffs[winner]), float(near_cutoffs[winner])
        rank = (float(macro_coverage[f, n]), float(pooled_coverage[f, n]), float(minimum_precision[f, n]),
                -int(errors[f, n]), -int(epoch), far_cutoff + near_cutoff,
                min(far_cutoff, near_cutoff), far_cutoff, near_cutoff)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_policy = (int(epoch), far_cutoff, near_cutoff)
    feasible = best_policy is not None
    epoch, far_cutoff, near_cutoff = best_policy if feasible else (int(epochs[0]), 2., 2.)
    values = _scores(candidates[epoch], len(rows))
    return {
        "epoch": epoch, "farCutoff": far_cutoff, "nearCutoff": near_cutoff,
        "feasible": feasible, "targetPrecision": float(target),
        "innerEvaluation": evaluate(rows, decisions(values, far_cutoff, near_cutoff), values[:, 0]),
        "candidateCount": len(epochs) * len(CUTOFFS) ** 2,
        "feasibleCandidateCount": feasible_count,
    }
