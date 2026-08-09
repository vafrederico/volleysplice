from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


FEATURE_VERSION = "court-motion-flow-v1"
MODEL_TYPE = "weighted-logistic-v1"


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class FeatureConfig:
    analysis_fps: float = 4.0
    resize_width: int = 192
    resize_height: int = 108
    grid_size: int = 3
    use_optical_flow: bool = True
    context_offsets_seconds: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)

    def validate(self) -> None:
        if not _finite_number(self.analysis_fps) or not 0.25 <= self.analysis_fps <= 30:
            raise ValueError("analysis_fps must be between 0.25 and 30")
        if (
            not isinstance(self.resize_width, int)
            or isinstance(self.resize_width, bool)
            or not isinstance(self.resize_height, int)
            or isinstance(self.resize_height, bool)
            or self.resize_width < 32
            or self.resize_height < 32
        ):
            raise ValueError("resize dimensions must both be at least 32 pixels")
        if not isinstance(self.grid_size, int) or isinstance(self.grid_size, bool) or not 1 <= self.grid_size <= 6:
            raise ValueError("grid_size must be between 1 and 6")
        if not isinstance(self.use_optical_flow, bool):
            raise ValueError("use_optical_flow must be boolean")
        if not self.context_offsets_seconds:
            raise ValueError("at least one context offset is required")
        if 0.0 not in self.context_offsets_seconds:
            raise ValueError("context offsets must include 0 seconds")
        if not all(_finite_number(offset) for offset in self.context_offsets_seconds):
            raise ValueError("context offsets must be finite")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["context_offsets_seconds"] = list(self.context_offsets_seconds)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureConfig":
        known = {
            "analysis_fps",
            "resize_width",
            "resize_height",
            "grid_size",
            "use_optical_flow",
            "context_offsets_seconds",
        }
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown feature configuration keys: {sorted(unknown)}")
        kwargs = dict(value)
        if "context_offsets_seconds" in kwargs:
            if not isinstance(kwargs["context_offsets_seconds"], (list, tuple)) or not all(
                _finite_number(item) for item in kwargs["context_offsets_seconds"]
            ):
                raise ValueError("context offsets must be a finite numeric array")
            kwargs["context_offsets_seconds"] = tuple(
                float(item) for item in kwargs["context_offsets_seconds"]
            )
        result = cls(**kwargs)
        result.validate()
        return result


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 180
    batch_size: int = 2048
    learning_rate: float = 0.02
    l2: float = 1e-4
    patience: int = 20
    seed: int = 7

    def validate(self) -> None:
        if not isinstance(self.epochs, int) or isinstance(self.epochs, bool) or self.epochs < 1:
            raise ValueError("epochs must be positive")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not _finite_number(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not _finite_number(self.l2) or self.l2 < 0:
            raise ValueError("l2 must be non-negative")
        if not isinstance(self.patience, int) or isinstance(self.patience, bool) or self.patience < 1:
            raise ValueError("patience must be positive")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecoderConfig:
    smoothing_seconds: float = 1.0
    enter_threshold: float = 0.5
    exit_threshold: float = 0.4
    min_live_seconds: float = 1.0
    bridge_gap_seconds: float = 2.0

    def validate(self) -> None:
        if not _finite_number(self.smoothing_seconds) or self.smoothing_seconds < 0:
            raise ValueError("smoothing_seconds must be non-negative")
        if (
            not _finite_number(self.enter_threshold)
            or not _finite_number(self.exit_threshold)
            or not 0 < self.exit_threshold <= self.enter_threshold < 1
        ):
            raise ValueError("thresholds must satisfy 0 < exit <= enter < 1")
        if (
            not _finite_number(self.min_live_seconds)
            or not _finite_number(self.bridge_gap_seconds)
            or self.min_live_seconds < 0
            or self.bridge_gap_seconds < 0
        ):
            raise ValueError("decoder durations must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DecoderConfig":
        result = cls(**value)
        result.validate()
        return result


@dataclass(frozen=True)
class PipelineConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
