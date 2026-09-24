"""Matched four-head AV and frozen-DINO temporal models for transfer research.

The compact branch is the unchanged expanded-study implementation. The DINO
branch reuses the original token fusion implementation and only widens its last
linear layer to add keep supervision. Feature extraction and scaler fitting are
external; no DINO encoder or feature cache is created by this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from .compact_temporal_model import (
    CompactTemporalConfig,
    CompactTemporalError,
    DinoFusionNetwork,
    nn,
    require_torch,
    trainable_parameter_count,
)
from .expanded_temporal_model import (
    HEAD_NAMES,
    KEEP_HEAD_INDEX,
    PRIMARY_HEAD_COUNT,
    ExpandedTemporalConfig,
    ExpandedTemporalNetwork,
)

MODEL_KINDS = ("tcn", "dino_tcn")
AUDIOVISUAL_DIMENSION = 104
TOKEN_COUNT = 10
TOKEN_DIMENSION = 384
TOKEN_PROJECTION_DIMENSION = 16
DINO_INPUT_DIMENSION = AUDIOVISUAL_DIMENSION + TOKEN_COUNT * TOKEN_DIMENSION
FUSED_DIMENSION = AUDIOVISUAL_DIMENSION + TOKEN_COUNT * TOKEN_PROJECTION_DIMENSION
PARAMETER_COUNTS = {"tcn": 29700, "dino_tcn": 46868}
PRIMARY_PARAMETER_COUNTS = {"tcn": 29635, "dino_tcn": 46803}


def model_metadata(kind: str) -> dict[str, Any]:
    """Describe the fixed experiment architecture without allocating a model/RNG."""
    if kind not in MODEL_KINDS:
        raise CompactTemporalError(f"unsupported transfer model kind: {kind!r}")
    config = ExpandedTemporalConfig(input_dimension=DINO_INPUT_DIMENSION if kind == "dino_tcn" else 104)
    metadata = {
        **config.to_dict(), "kind": kind, "temporalKind": "tcn",
        "parameters": PARAMETER_COUNTS[kind], "primaryOnlyParameters": PRIMARY_PARAMETER_COUNTS[kind],
        "audiovisualDimension": AUDIOVISUAL_DIMENSION, "featureRateHz": 4,
        "standardization": "AV104 only: exact-training-fold mean/scale, then clip[-10,10]",
        "inputPreprocessing": "Same per-record AV percentiles/absolute-channel exceptions as frozen study; caller supplies them before fold scaling",
        "segmentBoundaries": "Real context only; reset at ignored gaps, no synthetic ticks beyond true segment edges",
        "featureExtractionIncluded": False,
    }
    if kind == "dino_tcn":
        metadata.update({
            "normalization": "external-fold-local-AV-scaler-and-per-token-LayerNorm",
            "inputOrder": "104 scaled AV values, then token-major flattened10x384 raw frozen DINO values",
            "tokenCount": TOKEN_COUNT, "tokenDimension": TOKEN_DIMENSION,
            "tokenProjectionDimension": TOKEN_PROJECTION_DIMENSION, "fusedDimension": FUSED_DIMENSION,
            "tokenNormalization": "LayerNorm over each token's384 channels only; learned affine; eps1e-5",
            "tokenProjection": "Shared Linear384->16 then ReLU; preserve token order",
            "dinoStandardization": "No percentile transform, fold scaler, clipping or across-time normalization",
        })
    return metadata


if nn is not None:
    class DinoTransferNetwork(DinoFusionNetwork):
        """Return float32[B,T,4] logits from AV104 + ten384-channel DINO tokens.

        Token operations and the temporal backbone are inherited unchanged. The
        temporal model retains a125-tick inclusive receptive field and62-tick
        real-context halo. LayerNorm never couples distinct ticks or tokens.
        """

        def __init__(self) -> None:
            super().__init__(CompactTemporalConfig(kind="tcn", input_dimension=DINO_INPUT_DIMENSION))
            self.config = ExpandedTemporalConfig(**asdict(self.config))
            self.temporal.config = ExpandedTemporalConfig(**asdict(self.temporal.config))
            self.temporal.head = nn.Linear(self.temporal.head.in_features, len(HEAD_NAMES))
            self.float()

else:
    class DinoTransferNetwork:  # type: ignore[no-redef]
        def __init__(self) -> None:
            require_torch()


def model_for(kind: str):
    """Build the fixed model; no additional initialization or random draws."""
    if kind == "tcn":
        return ExpandedTemporalNetwork(ExpandedTemporalConfig(kind="tcn"))
    if kind == "dino_tcn":
        return DinoTransferNetwork()
    raise CompactTemporalError(f"unsupported transfer model kind: {kind!r}")


def primary_only_model(model):
    """Copy trained parameters into a three-head inference derivative.

    Removes only the unused keep output row. The supplied training model and
    process RNG stay unchanged. This is an in-memory research utility, not an
    export, production promotion, or qualification of DINO extraction on phones.
    """
    require_torch()
    if isinstance(model, DinoTransferNetwork):
        kind = "dino_tcn"
    elif isinstance(model, ExpandedTemporalNetwork) and model.config.kind == "tcn":
        kind = "tcn"
    else:
        raise CompactTemporalError("expected a four-head transfer TCN")
    temporal = model.temporal if kind == "dino_tcn" else model
    if temporal.head.out_features != len(HEAD_NAMES):
        raise CompactTemporalError("expected four output heads before trimming")
    derivative = deepcopy(model)
    temporal = derivative.temporal if kind == "dino_tcn" else derivative
    previous = temporal.head
    # Slice an existing module instead of constructing Linear, which draws RNG.
    previous.weight = nn.Parameter(previous.weight[:PRIMARY_HEAD_COUNT].detach().clone(), requires_grad=False)
    previous.bias = nn.Parameter(previous.bias[:PRIMARY_HEAD_COUNT].detach().clone(), requires_grad=False)
    previous.out_features = PRIMARY_HEAD_COUNT
    derivative.config = CompactTemporalConfig(**asdict(derivative.config))
    if kind == "dino_tcn":
        derivative.temporal.config = CompactTemporalConfig(**asdict(derivative.temporal.config))
    derivative.requires_grad_(False)
    derivative.eval()
    return derivative
