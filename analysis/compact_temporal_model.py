"""Small float32 live/serve/end networks over the current audiovisual schema.

Inputs are already normalized using fitting-partition statistics. No layer fits
statistics or normalizes across time. Callers must split recordings at unknown
gaps/camera cuts and mask padded targets; this module never infers label validity.
The contextual controls repeat the nearest endpoint at a recording boundary,
matching the production feature context. TCN convolutions use zero padding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

try:  # Other analysis utilities must remain importable without PyTorch.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - depends on the selected environment
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


HEAD_NAMES = ("live", "serve", "end")
CONTEXT_OFFSETS_TICKS = (-8, -4, 0, 4, 8)
MODEL_KINDS = ("linear", "mlp", "tcn")


class CompactTemporalError(ValueError):
    """Invalid compact-model configuration or tensor contract."""


def require_torch() -> Any:
    if torch is None:
        raise RuntimeError("Compact temporal models require PyTorch")
    return torch


@dataclass(frozen=True)
class CompactTemporalConfig:
    kind: Literal["linear", "mlp", "tcn"] = "tcn"
    input_dimension: int = 104
    hidden_dimension: int = 64
    dropout: float = 0.1
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)

    def validate(self) -> None:
        if self.kind not in MODEL_KINDS:
            raise CompactTemporalError(f"unsupported model kind: {self.kind!r}")
        for name, value in (
            ("input_dimension", self.input_dimension),
            ("hidden_dimension", self.hidden_dimension),
            ("kernel_size", self.kernel_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CompactTemporalError(f"{name} must be a positive integer")
        if self.kernel_size % 2 != 1:
            raise CompactTemporalError("kernel_size must be odd")
        if not self.dilations or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.dilations
        ):
            raise CompactTemporalError("dilations must contain positive integers")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise CompactTemporalError("dropout must be finite and in [0, 1)")

    @property
    def halo_ticks(self) -> int:
        self.validate()
        if self.kind != "tcn":
            return max(abs(offset) for offset in CONTEXT_OFFSETS_TICKS)
        # Exactly ONE temporal convolution per residual block.
        return (self.kernel_size // 2) * sum(self.dilations)

    @property
    def receptive_field_ticks(self) -> int:
        """Inclusive temporal span; the contextual heads sample five positions."""
        return 2 * self.halo_ticks + 1

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "kind": self.kind,
            "inputDimension": self.input_dimension,
            "hiddenDimension": self.hidden_dimension,
            "dropout": self.dropout,
            "kernelSize": self.kernel_size,
            "dilations": list(self.dilations),
            "contextOffsetsTicks": list(CONTEXT_OFFSETS_TICKS),
            "headNames": list(HEAD_NAMES),
            "haloTicks": self.halo_ticks,
            "receptiveFieldTicks": self.receptive_field_ticks,
            "normalization": "external-fold-local-scaler",
        }


def _validate_values(values: Tensor, input_dimension: int | None = None) -> None:
    torch_module = require_torch()
    if not isinstance(values, torch_module.Tensor) or values.ndim != 3:
        raise CompactTemporalError("values must have shape [batch,time,features]")
    if any(size < 1 for size in values.shape):
        raise CompactTemporalError("batch, time and feature dimensions must be nonempty")
    if input_dimension is not None and values.shape[-1] != input_dimension:
        raise CompactTemporalError(f"expected {input_dimension} input features")
    if values.dtype != torch_module.float32:
        raise CompactTemporalError("values must be float32")


def contextualize_ticks(values: Tensor) -> Tensor:
    """Concatenate -2,-1,0,+1,+2-second features on an aligned 4 Hz grid.

    Edge indices are clamped, including sequences shorter than the eight-tick
    halo. The resulting order is offset-major, as in production contextualize.
    Uneven source timestamps must be aligned by the caller before using this
    tick-based control; this function does not resample them.
    """
    _validate_values(values)
    torch_module = require_torch()
    indices = torch_module.arange(values.shape[1], device=values.device)
    return torch_module.cat(
        [
            values.index_select(1, (indices + offset).clamp(0, values.shape[1] - 1))
            for offset in CONTEXT_OFFSETS_TICKS
        ],
        dim=-1,
    )


if nn is not None:

    class _DepthwiseResidualBlock(nn.Module):
        def __init__(self, width: int, kernel_size: int, dilation: int, dropout: float) -> None:
            super().__init__()
            self.depthwise = nn.Conv1d(
                width,
                width,
                kernel_size,
                padding=(kernel_size // 2) * dilation,
                dilation=dilation,
                groups=width,
            )
            self.pointwise = nn.Conv1d(width, width, 1)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, values: Tensor) -> Tensor:
            residual = self.activation(self.depthwise(values))
            residual = self.pointwise(residual)
            return self.activation(values + self.dropout(residual))


    class CompactTemporalNetwork(nn.Module):
        """Return float32 logits [B,T,3] in live, serve, end order.

        The caller controls seeds and deterministic backend settings. Dropout is
        active only during training. In eval mode, an overlapping chunk has the
        same valid interior as a full sequence (within float32 roundoff) when it
        includes ``config.halo_ticks`` real input ticks on each artificial edge.
        Do not append synthetic input ticks beyond actual recording boundaries:
        the model's own per-layer padding defines those edges.
        """

        def __init__(self, config: CompactTemporalConfig | None = None) -> None:
            super().__init__()
            self.config = config or CompactTemporalConfig()
            self.config.validate()
            width = self.config.hidden_dimension
            if self.config.kind == "linear":
                self.context_head = nn.Linear(
                    self.config.input_dimension * len(CONTEXT_OFFSETS_TICKS), len(HEAD_NAMES)
                )
            elif self.config.kind == "mlp":
                self.context_head = nn.Sequential(
                    nn.Linear(self.config.input_dimension * len(CONTEXT_OFFSETS_TICKS), width),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(width, len(HEAD_NAMES)),
                )
            else:
                self.input_projection = nn.Linear(self.config.input_dimension, width)
                self.activation = nn.ReLU()
                self.blocks = nn.ModuleList(
                    _DepthwiseResidualBlock(
                        width, self.config.kernel_size, dilation, self.config.dropout
                    )
                    for dilation in self.config.dilations
                )
                self.head = nn.Linear(width, len(HEAD_NAMES))
            # Do not inherit a process-global float64 default in research tools.
            self.float()

        def forward(self, values: Tensor) -> Tensor:
            _validate_values(values, self.config.input_dimension)
            if self.config.kind != "tcn":
                return self.context_head(contextualize_ticks(values))
            hidden = self.activation(self.input_projection(values)).transpose(1, 2)
            for block in self.blocks:
                hidden = block(hidden)
            return self.head(hidden.transpose(1, 2))

    class DinoFusionNetwork(nn.Module):
        """Match the compact temporal control while adding frozen DINO tokens.

        Input order is ``[104 scaled AV values, 10 flattened 384-wide tokens]``
        by default. Only AV is scaled outside the model. LayerNorm acts within
        each individual token, never across batch/time/token positions; a shared
        projection reduces each token to 16 values before temporal fusion.
        ``config.input_dimension`` describes the flat input, while
        ``temporal.config.input_dimension`` describes the projected input.
        """

        def __init__(
            self,
            config: CompactTemporalConfig | None = None,
            *,
            audiovisual_dimension: int = 104,
            token_count: int = 10,
            token_dimension: int = 384,
            projection_dimension: int = 16,
        ) -> None:
            super().__init__()
            for name, value in (
                ("audiovisual_dimension", audiovisual_dimension),
                ("token_count", token_count),
                ("token_dimension", token_dimension),
                ("projection_dimension", projection_dimension),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise CompactTemporalError(f"{name} must be a positive integer")
            self.audiovisual_dimension = audiovisual_dimension
            self.token_count = token_count
            self.token_dimension = token_dimension
            self.projection_dimension = projection_dimension
            self.input_dimension = audiovisual_dimension + token_count * token_dimension
            self.config = config or CompactTemporalConfig(input_dimension=self.input_dimension)
            self.config.validate()
            if self.config.input_dimension != self.input_dimension:
                raise CompactTemporalError(
                    f"DINO fusion config must declare {self.input_dimension} flat input features"
                )
            self.token_norm = nn.LayerNorm(token_dimension)
            self.token_projection = nn.Linear(token_dimension, projection_dimension)
            self.activation = nn.ReLU()
            self.temporal = CompactTemporalNetwork(replace(
                self.config,
                input_dimension=audiovisual_dimension + token_count * projection_dimension,
            ))
            self.float()

        def forward(self, values: Tensor) -> Tensor:
            _validate_values(values, self.input_dimension)
            audiovisual = values[:, :, :self.audiovisual_dimension]
            tokens = values[:, :, self.audiovisual_dimension:].reshape(
                values.shape[0], values.shape[1], self.token_count, self.token_dimension
            )
            projected = self.activation(self.token_projection(self.token_norm(tokens)))
            projected = projected.flatten(start_dim=2)
            return self.temporal(torch.cat((audiovisual, projected), dim=-1))


else:

    class CompactTemporalNetwork:  # type: ignore[no-redef]
        def __init__(self, config: CompactTemporalConfig | None = None) -> None:
            del config
            require_torch()


    class DinoFusionNetwork:  # type: ignore[no-redef]
        def __init__(self, config: CompactTemporalConfig | None = None, **kwargs: Any) -> None:
            del config, kwargs
            require_torch()


def trainable_parameter_count(model: Any) -> int:
    if not hasattr(model, "parameters"):
        raise TypeError("model must expose parameters()")
    return sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
