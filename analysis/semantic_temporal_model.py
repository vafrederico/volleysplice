"""Small tensor-only temporal head for frozen DINOv2 plus audiovisual signals."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


try:  # Keep the repository's CPU-first utilities importable without PyTorch.
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - depends on the selected model environment
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


HEAD_NAMES = ("live", "serve", "end")


class SemanticTemporalError(RuntimeError):
    """Raised for invalid temporal model inputs or training state."""


def require_torch() -> Any:
    if torch is None:
        raise SemanticTemporalError(
            "Track T requires PyTorch; install the current CUDA wheel in the temporal environment"
        )
    return torch


@dataclass(frozen=True)
class TemporalModelConfig:
    dino_token_count: int = 10
    dino_dimension: int = 384
    dino_projection_dimension: int = 32
    audiovisual_dimension: int = 90
    hidden_dimension: int = 128
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    groups: int = 8
    kernel_size: int = 3
    dropout: float = 0.15
    use_audiovisual: bool = True

    def validate(self) -> None:
        if self.dino_token_count < 0 or self.dino_dimension < 1:
            raise ValueError("DINO token count must be nonnegative and dimension positive")
        if self.dino_token_count and self.dino_projection_dimension < 1:
            raise ValueError("DINO projection dimension must be positive when DINO is used")
        if self.use_audiovisual and self.audiovisual_dimension < 1:
            raise ValueError("audiovisual dimension must be positive when AV is used")
        if self.hidden_dimension < 1 or self.hidden_dimension % self.groups:
            raise ValueError("hidden dimension must be positive and divisible by groups")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if not self.dilations or any(item < 1 for item in self.dilations):
            raise ValueError("dilations must contain positive values")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not self.dino_token_count and not self.use_audiovisual:
            raise ValueError("at least one input representation is required")

    @property
    def input_dimension(self) -> int:
        return (
            self.dino_token_count * self.dino_projection_dimension
            + (self.audiovisual_dimension if self.use_audiovisual else 0)
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "dinoTokenCount": self.dino_token_count,
            "dinoDimension": self.dino_dimension,
            "dinoProjectionDimension": self.dino_projection_dimension,
            "audiovisualDimension": self.audiovisual_dimension,
            "hiddenDimension": self.hidden_dimension,
            "dilations": list(self.dilations),
            "groups": self.groups,
            "kernelSize": self.kernel_size,
            "dropout": self.dropout,
            "useAudiovisual": self.use_audiovisual,
            "receptiveFieldTicks": 1 + sum(
                2 * (self.kernel_size - 1) * dilation for dilation in self.dilations
            ),
        }


if nn is not None:

    class ResidualConvBlock(nn.Module):
        def __init__(self, hidden_dimension: int, dilation: int, groups: int, kernel_size: int, dropout: float) -> None:
            super().__init__()
            padding = dilation * (kernel_size // 2)
            self.conv1 = nn.Conv1d(
                hidden_dimension,
                hidden_dimension,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
            self.norm1 = nn.GroupNorm(groups, hidden_dimension)
            self.conv2 = nn.Conv1d(
                hidden_dimension,
                hidden_dimension,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
            self.norm2 = nn.GroupNorm(groups, hidden_dimension)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, values: Tensor) -> Tensor:
            residual = values
            values = self.conv1(values)
            values = self.activation(self.norm1(values))
            values = self.dropout(values)
            values = self.conv2(values)
            values = self.norm2(values)
            return self.activation(residual + self.dropout(values))


    class SemanticTemporalNetwork(nn.Module):
        """The frozen-DINO plus 90-signal five-block temporal architecture."""

        def __init__(self, config: TemporalModelConfig | None = None) -> None:
            super().__init__()
            self.config = config or TemporalModelConfig()
            self.config.validate()
            if self.config.dino_token_count:
                self.dino_norm = nn.LayerNorm(self.config.dino_dimension)
                self.dino_projection = nn.Linear(
                    self.config.dino_dimension,
                    self.config.dino_projection_dimension,
                )
            else:
                self.dino_norm = None
                self.dino_projection = None
            self.input_projection = nn.Linear(
                self.config.input_dimension,
                self.config.hidden_dimension,
            )
            self.blocks = nn.ModuleList(
                ResidualConvBlock(
                    self.config.hidden_dimension,
                    dilation,
                    self.config.groups,
                    self.config.kernel_size,
                    self.config.dropout,
                )
                for dilation in self.config.dilations
            )
            self.heads = nn.ModuleDict(
                {name: nn.Linear(self.config.hidden_dimension, 1) for name in HEAD_NAMES}
            )
            parameters = trainable_parameter_count(self)
            if parameters >= 2_000_000:
                raise SemanticTemporalError(
                    f"semantic temporal head has {parameters} trainable parameters; limit is below 2 million"
                )

        def forward(
            self,
            dino_tokens: Tensor | None,
            audiovisual: Tensor | None,
        ) -> dict[str, Tensor]:
            torch_module = require_torch()
            representations: list[Tensor] = []
            if self.config.dino_token_count:
                if dino_tokens is None or dino_tokens.ndim != 4:
                    raise SemanticTemporalError("DINO inputs must have shape [batch,time,tokens,dimension]")
                if dino_tokens.shape[2] != self.config.dino_token_count or dino_tokens.shape[3] != self.config.dino_dimension:
                    raise SemanticTemporalError("DINO input shape differs from the model configuration")
                batch, time_steps = dino_tokens.shape[:2]
                projected = self.dino_projection(self.dino_norm(dino_tokens))
                representations.append(projected.reshape(batch, time_steps, -1))
            elif dino_tokens is not None and dino_tokens.numel():
                raise SemanticTemporalError("this model configuration does not accept DINO inputs")
            if self.config.use_audiovisual:
                if audiovisual is None or audiovisual.ndim != 3:
                    raise SemanticTemporalError("audiovisual inputs must have shape [batch,time,signals]")
                if audiovisual.shape[2] != self.config.audiovisual_dimension:
                    raise SemanticTemporalError("audiovisual input shape differs from the model configuration")
                representations.append(audiovisual)
            elif audiovisual is not None and audiovisual.numel():
                raise SemanticTemporalError("this model configuration does not accept audiovisual inputs")
            if not representations:
                raise SemanticTemporalError("model received no representations")
            values = torch_module.cat(representations, dim=-1)
            values = self.input_projection(values).transpose(1, 2)
            for block in self.blocks:
                values = block(values)
            values = values.transpose(1, 2)
            return {
                name: self.heads[name](values).squeeze(-1)
                for name in HEAD_NAMES
            }

else:

    class SemanticTemporalNetwork:  # type: ignore[no-redef]
        def __init__(self, config: TemporalModelConfig | None = None) -> None:
            del config
            raise SemanticTemporalError("PyTorch is not installed")


def trainable_parameter_count(model: Any) -> int:
    if not hasattr(model, "parameters"):
        raise TypeError("model must expose parameters()")
    return sum(int(item.numel()) for item in model.parameters() if bool(item.requires_grad))


def seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def fit_audiovisual_scaler(
    sequences: list[np.ndarray],
    masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit standardization from training-fold ticks only."""

    if len(sequences) != len(masks) or not sequences:
        raise ValueError("sequences and masks must be nonempty and aligned")
    selected: list[np.ndarray] = []
    for values, mask in zip(sequences, masks, strict=True):
        if values.ndim != 2 or mask.ndim != 1 or len(values) != len(mask):
            raise ValueError("audiovisual values and masks have incompatible shapes")
        if np.any(mask):
            selected.append(np.asarray(values[mask], dtype=np.float64))
    if not selected:
        raise ValueError("training-fold masks contain no usable ticks")
    matrix = np.concatenate(selected, axis=0)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return mean.astype(np.float32), scale.astype(np.float32)


def standardize_audiovisual(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or mean.ndim != 1 or scale.ndim != 1 or values.shape[1] != len(mean) or len(mean) != len(scale):
        raise ValueError("audiovisual scaler shapes are incompatible")
    result = (values.astype(np.float32) - mean.astype(np.float32)) / scale.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("standardized audiovisual values are non-finite")
    return np.ascontiguousarray(result, dtype=np.float32)


def positive_weights(
    targets: Mapping[str, Sequence[np.ndarray] | np.ndarray],
    masks: list[np.ndarray] | np.ndarray,
) -> dict[str, float]:
    """Compute fold-local BCE positive weights and clip them to [1, 20]."""

    if isinstance(masks, np.ndarray):
        mask_rows = [masks]
    else:
        mask_rows = masks
    result: dict[str, float] = {}
    for name in HEAD_NAMES:
        raw_values = targets[name]
        if isinstance(raw_values, np.ndarray):
            values = [raw_values] if raw_values.ndim == 1 else list(raw_values)
        else:
            values = list(raw_values)
        if len(mask_rows) != len(values):
            raise ValueError("target rows and masks are not aligned")
        positive_count = negative_count = 0.0
        for row, mask in zip(values, mask_rows, strict=True):
            selected = np.asarray(row)[np.asarray(mask, dtype=bool)]
            positive_count += float(np.sum(selected > 0.5))
            negative_count += float(np.sum(selected <= 0.5))
        if positive_count <= 0:
            weight = 1.0
        else:
            weight = negative_count / positive_count
        result[name] = float(np.clip(weight, 1.0, 20.0))
    return result


def weighted_temporal_loss(
    logits: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    valid_mask: Tensor,
    positive_weight: Mapping[str, float],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Calculate the declared live + .5 serve + .5 end objective."""

    torch_module = require_torch()
    if valid_mask.ndim != 2:
        raise SemanticTemporalError("valid_mask must have shape [batch,time]")
    losses: dict[str, Tensor] = {}
    selected = valid_mask.bool()
    if not bool(torch_module.any(selected)):
        raise SemanticTemporalError("weighted loss received no valid ticks")
    for name in HEAD_NAMES:
        if name not in logits or name not in targets:
            raise SemanticTemporalError(f"missing {name!r} head/target")
        if logits[name].shape != targets[name].shape or logits[name].shape != valid_mask.shape:
            raise SemanticTemporalError(f"{name} logits, targets, and mask are misaligned")
        weight = torch_module.tensor(
            float(np.clip(positive_weight.get(name, 1.0), 1.0, 20.0)),
            device=logits[name].device,
            dtype=logits[name].dtype,
        )
        per_tick = F.binary_cross_entropy_with_logits(
            logits[name], targets[name], pos_weight=weight, reduction="none"
        )
        losses[name] = per_tick[selected].mean()
    total = losses["live"] + 0.5 * losses["serve"] + 0.5 * losses["end"]
    if not bool(torch_module.isfinite(total)):
        raise SemanticTemporalError("temporal loss is non-finite")
    return total, losses
