"""Bounded local-attention controls for the four-head temporal study.

Only feature fusion lives here: no fitting, extraction, scaling or decoding.
The default two radius-31 blocks have the same 62-tick real-input halo as the
current TCN. Attention is decomposed into ordinary tensor operations, without
MultiheadAttention or scaled_dot_product_attention export dependencies.

``valid_mask`` describes REAL CONTEXT, not label availability. An unlabeled but
observable tick must remain valid. False ticks are excluded from attention and
form barriers between contiguous segments. Camera cuts without an intervening
invalid tick must still be split by the caller. Invalid output logits are zero;
the caller must also mask their targets/probabilities (sigmoid(0) is 0.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

from .compact_temporal_model import (
    CompactTemporalError, Tensor, _validate_values, nn, require_torch,
    trainable_parameter_count,
)
from .expanded_temporal_model import HEAD_NAMES
from .transfer_temporal_model import (
    AUDIOVISUAL_DIMENSION, DINO_INPUT_DIMENSION, FUSED_DIMENSION,
    TOKEN_COUNT, TOKEN_DIMENSION, TOKEN_PROJECTION_DIMENSION,
)


MODEL_KINDS = ("transformer", "dino_transformer")
PARAMETER_COUNTS = {"transformer": 31176, "dino_transformer": 44504}


@dataclass(frozen=True)
class LocalAttentionTemporalConfig:
    kind: Literal["transformer", "dino_transformer"] = "transformer"
    input_dimension: int = AUDIOVISUAL_DIMENSION
    hidden_dimension: int = 40
    attention_heads: int = 2
    feedforward_dimension: int = 80
    block_count: int = 2
    attention_radius_ticks: int = 31
    dropout: float = 0.1

    def validate(self) -> None:
        if self.kind not in MODEL_KINDS:
            raise CompactTemporalError(f"unsupported local-attention kind: {self.kind!r}")
        for name in (
            "input_dimension", "hidden_dimension", "attention_heads",
            "feedforward_dimension", "block_count", "attention_radius_ticks",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CompactTemporalError(f"{name} must be a positive integer")
        if self.hidden_dimension % self.attention_heads:
            raise CompactTemporalError("hidden_dimension must divide evenly into attention_heads")
        if isinstance(self.dropout, bool) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise CompactTemporalError("dropout must be finite in [0, 1)")

    @property
    def halo_ticks(self) -> int:
        self.validate()
        return self.block_count * self.attention_radius_ticks

    @property
    def receptive_field_ticks(self) -> int:
        return 2 * self.halo_ticks + 1

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "kind": self.kind, "inputDimension": self.input_dimension,
            "hiddenDimension": self.hidden_dimension,
            "attentionHeads": self.attention_heads,
            "feedforwardDimension": self.feedforward_dimension,
            "blockCount": self.block_count,
            "attentionRadiusTicks": self.attention_radius_ticks,
            "dropout": self.dropout, "headNames": list(HEAD_NAMES),
            "primaryHeadCount": 3, "auxiliaryHeadNames": [HEAD_NAMES[3]],
            "auxiliaryHeadIndices": [3], "haloTicks": self.halo_ticks,
            "receptiveFieldTicks": self.receptive_field_ticks,
            "positionEncoding": "learned per-head relative-offset bias in each block",
            "normalization": "external-fold-local-scaler and per-tick LayerNorm",
            "attentionImplementation": "decomposed MatMul/Add/Where/Softmax; dense bounded chunks",
            "validMask": "optional bool[B,T] real-context mask; false ticks separate segments",
            "invalidOutputs": "zero logits; caller must mask loss and probabilities",
        }


def _context_mask(values: Tensor, valid_mask: Tensor | None) -> Tensor:
    torch = require_torch()
    if valid_mask is None:
        return torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
    if (not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool
            or valid_mask.ndim != 2 or valid_mask.shape != values.shape[:2]
            or valid_mask.device != values.device):
        raise CompactTemporalError("valid_mask must be bool[B,T] on the input device")
    return valid_mask


def _zero_invalid(values: Tensor, valid_mask: Tensor) -> Tensor:
    torch = require_torch()
    # Where, rather than multiplication, also removes NaN/Inf padding before
    # learned projections; masked NaN activations can otherwise poison gradients.
    return torch.where(valid_mask.unsqueeze(-1), values, torch.zeros_like(values))


if nn is not None:
    class _LocalAttentionBlock(nn.Module):
        def __init__(self, config: LocalAttentionTemporalConfig) -> None:
            super().__init__()
            width = config.hidden_dimension
            self.head_count = config.attention_heads
            self.head_dimension = width // self.head_count
            self.radius = config.attention_radius_ticks
            self.attention_norm = nn.LayerNorm(width)
            self.qkv = nn.Linear(width, 3 * width)
            self.output_projection = nn.Linear(width, width)
            self.relative_bias = nn.Parameter(require_torch().zeros(self.head_count, 2 * self.radius + 1))
            self.feedforward_norm = nn.LayerNorm(width)
            self.feedforward = nn.Sequential(
                nn.Linear(width, config.feedforward_dimension), nn.ReLU(),
                nn.Linear(config.feedforward_dimension, width),
            )
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, values: Tensor, allowed: Tensor, offsets: Tensor,
                    valid_mask: Tensor) -> Tensor:
            torch = require_torch()
            batch, ticks, width = values.shape
            projected = self.qkv(self.attention_norm(values))
            projected = projected.reshape(batch, ticks, 3, self.head_count, self.head_dimension)
            query, key, value = projected.unbind(dim=2)
            query, key, value = (item.transpose(1, 2) for item in (query, key, value))
            scores = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dimension ** -0.5)
            bias_index = (offsets + self.radius).clamp(0, 2 * self.radius)
            bias = self.relative_bias.index_select(1, bias_index.reshape(-1))
            scores = scores + bias.reshape(self.head_count, ticks, ticks).unsqueeze(0)
            scores = torch.where(allowed.unsqueeze(1), scores, torch.full_like(scores, float("-inf")))
            probabilities = torch.softmax(scores, dim=-1)
            attended = torch.matmul(probabilities, value).transpose(1, 2).reshape(batch, ticks, width)
            hidden = values + self.dropout(self.output_projection(attended))
            hidden = hidden + self.dropout(self.feedforward(self.feedforward_norm(hidden)))
            return _zero_invalid(hidden, valid_mask)


    class LocalAttentionTemporalNetwork(nn.Module):
        """Return [B,T,4] logits with bounded, segment-local context.

        For evaluation, overlapping chunks agree on their interiors if each
        artificial edge includes ``config.halo_ticks`` real ticks. Learned
        positions depend only on relative offsets, never on chunk placement.
        Full-sequence calls are supported for checks, but allocate quadratic
        attention buffers; deploy with bounded chunks, not full recordings.
        """

        def __init__(self, config: LocalAttentionTemporalConfig | None = None) -> None:
            super().__init__()
            self.config = config or LocalAttentionTemporalConfig()
            self.config.validate()
            if self.config.kind != "transformer":
                raise CompactTemporalError("use DinoLocalAttentionNetwork for dino_transformer")
            width = self.config.hidden_dimension
            self.input_projection = nn.Linear(self.config.input_dimension, width)
            self.blocks = nn.ModuleList(_LocalAttentionBlock(self.config) for _ in range(self.config.block_count))
            self.output_norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, len(HEAD_NAMES))
            self.float()

        def forward(self, values: Tensor, valid_mask: Tensor | None = None) -> Tensor:
            _validate_values(values, self.config.input_dimension)
            valid_mask = _context_mask(values, valid_mask)
            torch = require_torch()
            ticks = values.shape[1]
            positions = torch.arange(ticks, device=values.device)
            offsets = positions.unsqueeze(0) - positions.unsqueeze(1)
            local = offsets.abs() <= self.config.attention_radius_ticks
            # Distinct IDs prevent even one layer from jumping over an unknown
            # gap. Equality is unchanged by the chunk's arbitrary ID offset.
            segment_ids = torch.cumsum((~valid_mask).to(torch.int64), dim=1)
            same_segment = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)
            allowed = local.unsqueeze(0) & same_segment & valid_mask.unsqueeze(1)
            # Every invalid query gets a self-only row, avoiding all -inf -> NaN
            # softmax. Its values/logits remain zero and cannot reach valid keys.
            allowed = allowed & valid_mask.unsqueeze(2)
            allowed = allowed | ((~valid_mask).unsqueeze(2) & (offsets == 0).unsqueeze(0))
            hidden = self.input_projection(_zero_invalid(values, valid_mask))
            hidden = _zero_invalid(hidden, valid_mask)
            for block in self.blocks:
                hidden = block(hidden, allowed, offsets, valid_mask)
            return _zero_invalid(self.head(self.output_norm(hidden)), valid_mask)


    class DinoLocalAttentionNetwork(nn.Module):
        """The existing DINO 10x384 -> 10x16 fusion with a local-attention head."""

        def __init__(self, config: LocalAttentionTemporalConfig | None = None) -> None:
            super().__init__()
            self.config = config or LocalAttentionTemporalConfig(kind="dino_transformer", input_dimension=DINO_INPUT_DIMENSION)
            self.config.validate()
            if self.config.kind != "dino_transformer" or self.config.input_dimension != DINO_INPUT_DIMENSION:
                raise CompactTemporalError(f"DINO attention requires dino_transformer with {DINO_INPUT_DIMENSION} inputs")
            self.audiovisual_dimension = AUDIOVISUAL_DIMENSION
            self.token_count = TOKEN_COUNT
            self.token_dimension = TOKEN_DIMENSION
            self.projection_dimension = TOKEN_PROJECTION_DIMENSION
            self.input_dimension = DINO_INPUT_DIMENSION
            self.token_norm = nn.LayerNorm(TOKEN_DIMENSION)
            self.token_projection = nn.Linear(TOKEN_DIMENSION, TOKEN_PROJECTION_DIMENSION)
            self.activation = nn.ReLU()
            self.temporal = LocalAttentionTemporalNetwork(replace(
                self.config, kind="transformer", input_dimension=FUSED_DIMENSION,
            ))
            self.float()

        def forward(self, values: Tensor, valid_mask: Tensor | None = None) -> Tensor:
            _validate_values(values, self.input_dimension)
            valid_mask = _context_mask(values, valid_mask)
            torch = require_torch()
            values = _zero_invalid(values, valid_mask)
            audiovisual = values[:, :, :self.audiovisual_dimension]
            tokens = values[:, :, self.audiovisual_dimension:].reshape(
                values.shape[0], values.shape[1], self.token_count, self.token_dimension,
            )
            projected = self.activation(self.token_projection(self.token_norm(tokens)))
            return self.temporal(torch.cat((audiovisual, projected.flatten(start_dim=2)), dim=-1), valid_mask)


else:
    class LocalAttentionTemporalNetwork:  # type: ignore[no-redef]
        def __init__(self, config: LocalAttentionTemporalConfig | None = None) -> None:
            del config
            require_torch()

    class DinoLocalAttentionNetwork(LocalAttentionTemporalNetwork):  # type: ignore[no-redef]
        pass


def model_for(kind: str):
    if kind == "transformer":
        return LocalAttentionTemporalNetwork()
    if kind == "dino_transformer":
        return DinoLocalAttentionNetwork()
    raise CompactTemporalError(f"unsupported local-attention kind: {kind!r}")


def model_metadata(kind: str) -> dict[str, Any]:
    """Static recipe metadata, without allocating models or consuming RNG."""
    if kind not in MODEL_KINDS:
        raise CompactTemporalError(f"unsupported local-attention kind: {kind!r}")
    config = LocalAttentionTemporalConfig(kind=kind, input_dimension=(
        DINO_INPUT_DIMENSION if kind == "dino_transformer" else AUDIOVISUAL_DIMENSION
    ))
    metadata = {
        **config.to_dict(), "parameters": PARAMETER_COUNTS[kind],
        "temporalKind": "transformer", "featureRateHz": 4,
        "audiovisualDimension": AUDIOVISUAL_DIMENSION,
        "standardization": "AV104 only: exact-training-fold mean/scale, then clip[-10,10]",
        "inputPreprocessing": "Same per-record AV percentiles/absolute-channel exceptions as frozen study; caller supplies them before fold scaling",
        "segmentBoundaries": "Real context only; reset at ignored gaps and camera cuts; invalid ticks mask context, not supervision",
        "featureExtractionIncluded": False,
        "recommendedChunkTicks": 256, "recommendedCentralTicks": 128,
        "recommendedChunkHaloTicks": 64,
        "mobileQualification": "Not benchmarked; static decomposed graph still requires backend qualification",
    }
    if kind == "dino_transformer":
        metadata.update({
            "inputOrder": "104 scaled AV values, then token-major flattened10x384 raw frozen DINO values",
            "tokenCount": TOKEN_COUNT, "tokenDimension": TOKEN_DIMENSION,
            "tokenProjectionDimension": TOKEN_PROJECTION_DIMENSION, "fusedDimension": FUSED_DIMENSION,
            "tokenNormalization": "LayerNorm over each token's384 channels only; learned affine; eps1e-5",
            "tokenProjection": "Shared Linear384->16 then ReLU; preserve token order",
            "dinoStandardization": "No percentile transform, fold scaler, clipping or across-time normalization",
        })
    return metadata
