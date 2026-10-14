"""Cost-sensitive near/far heads for selective serving-side research.

The independent sigmoid outputs are cost-transformed evidence. Their remaining
mass provides an abstention score; none of these values is asserted to be a
calibrated probability of correctness or of needing human review.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import serving_side_hybrid as h


FALSE_ACCEPT_COST = 49.
LOGIT_SHIFT = math.log(FALSE_ACCEPT_COST)
RESIDUAL_PENALTY = .001


class CostSensitiveHead(h.HybridHead):
    """Shared hybrid representation with bounded, independent far/near outputs."""

    def __init__(self, base, arm="hybrid"):
        super().__init__(base, arm)
        self.correction[-1] = nn.Linear(64, 2)
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.float()

    def components(self, windows, features):
        h.require(features.ndim == 2 and features.shape == (len(windows), 237)
                  and features.dtype == torch.float32, "Cost-sensitive head requires FP32 specialist matrix")
        z = (features - self.base_mean) / self.base_scale
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
        return base, 3. * torch.tanh(self.correction(fused))

    def outputs(self, windows, features):
        """Return logits and bounded residuals, both shaped (rows, 2): far, near."""
        base, residual = self.components(windows, features)
        logits = torch.stack((-base, base), dim=1) + residual - LOGIT_SHIFT
        return logits, residual

    def forward(self, windows, features):
        return self.outputs(windows, features)[0]


def cost_sensitive_loss(logits, truth, residual):
    """Per-row BCE with a 49:1 false-accept cost and bounded-residual penalty.

    A positive target retains derivative (sigmoid(logit)-1)/50; sending both
    outputs to zero therefore cannot minimize the unregularized objective.
    Source weighting and batch averaging belong to the caller.
    """
    if logits.ndim != 2 or logits.shape[1] != 2 or residual.shape != logits.shape or truth.shape != (len(logits),):
        raise ValueError("Expected logits/residual (rows, 2) and truth (rows,)")
    truth = truth.to(dtype=logits.dtype)
    targets = torch.stack((1. - truth, truth), dim=1)
    weights = 1. + (FALSE_ACCEPT_COST - 1.) * (1. - targets)
    binary = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (weights * binary).sum(dim=1) / (FALSE_ACCEPT_COST + 1.) + RESIDUAL_PENALTY * residual.square().mean(dim=1)


def scores(logits: np.ndarray) -> np.ndarray:
    """Convert [far, near] logits to [near route probability, acceptance score].

    Float64 sigmoid with clipping to +/-35 is the persisted replay contract. The
    acceptance score is relative evidence, not a calibrated correctness estimate.
    """
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Cost-sensitive logits must be finite with shape (rows, 2)")
    raw = 1. / (1. + np.exp(-np.clip(values, -35., 35.)))
    acceptance = raw.sum(axis=1)
    if np.any(acceptance > 1. + 1e-12) or np.any(acceptance <= 0):
        raise ValueError("Independent near/far evidence exceeds the available probability mass")
    route_near = raw[:, 1] / acceptance
    return np.column_stack((route_near, np.minimum(acceptance, 1.)))
