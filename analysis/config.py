from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


FEATURE_VERSION = "audiovisual-motion-quality-v2"
AUDIO_NORMALIZED_FEATURE_VERSION = "audiovisual-noise-normalized-audio-v3"
LEGACY_FEATURE_VERSIONS = {"court-motion-flow-v1"}
MODEL_TYPE = "weighted-logistic-v1"
SEQUENCE_NORMALIZATIONS = {"none", "percentile-rank"}
LEGACY_AUDIO_FEATURE_SET = "legacy-v2"
NOISE_NORMALIZED_AUDIO_FEATURE_SET = "noise-normalized-bands-v3"
AUDIO_FEATURE_SETS = {
    LEGACY_AUDIO_FEATURE_SET,
    NOISE_NORMALIZED_AUDIO_FEATURE_SET,
}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class FeatureConfig:
    analysis_fps: float = 4.0
    resize_width: int = 192
    resize_height: int = 108
    grid_size: int = 3
    use_optical_flow: bool = True
    use_advanced_visual: bool = True
    use_audio: bool = True
    audio_sample_rate: int = 16000
    audio_feature_set: str = LEGACY_AUDIO_FEATURE_SET
    context_offsets_seconds: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    sequence_normalization: str = "percentile-rank"

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
        if not isinstance(self.use_advanced_visual, bool):
            raise ValueError("use_advanced_visual must be boolean")
        if not isinstance(self.use_audio, bool):
            raise ValueError("use_audio must be boolean")
        if (
            not isinstance(self.audio_sample_rate, int)
            or isinstance(self.audio_sample_rate, bool)
            or not 2000 <= self.audio_sample_rate <= 48000
        ):
            raise ValueError("audio_sample_rate must be between 2000 and 48000")
        if self.audio_feature_set not in AUDIO_FEATURE_SETS:
            raise ValueError(
                f"audio_feature_set must be one of {sorted(AUDIO_FEATURE_SETS)}"
            )
        if self.audio_feature_set == NOISE_NORMALIZED_AUDIO_FEATURE_SET:
            if not self.use_audio:
                raise ValueError(
                    "noise-normalized audio features require use_audio to be enabled"
                )
            if self.audio_sample_rate < 16000:
                raise ValueError(
                    "noise-normalized audio features require at least a 16000 Hz sample rate"
                )
        if not self.context_offsets_seconds:
            raise ValueError("at least one context offset is required")
        if 0.0 not in self.context_offsets_seconds:
            raise ValueError("context offsets must include 0 seconds")
        if not all(_finite_number(offset) for offset in self.context_offsets_seconds):
            raise ValueError("context offsets must be finite")
        if self.sequence_normalization not in SEQUENCE_NORMALIZATIONS:
            raise ValueError(
                "sequence_normalization must be one of "
                f"{sorted(SEQUENCE_NORMALIZATIONS)}"
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["context_offsets_seconds"] = list(self.context_offsets_seconds)
        # Keep old artifact/cache payloads byte-for-byte compatible. The explicit
        # field is only needed for the opt-in v3 feature signature.
        if self.audio_feature_set == LEGACY_AUDIO_FEATURE_SET:
            value.pop("audio_feature_set")
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureConfig":
        known = {
            "analysis_fps",
            "resize_width",
            "resize_height",
            "grid_size",
            "use_optical_flow",
            "use_advanced_visual",
            "use_audio",
            "audio_sample_rate",
            "audio_feature_set",
            "context_offsets_seconds",
            "sequence_normalization",
        }
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown feature configuration keys: {sorted(unknown)}")
        kwargs = dict(value)
        # Models saved before sequence-level normalization existed used raw values.
        kwargs.setdefault("sequence_normalization", "none")
        # The original visual-only artifact predates these switches. Missing keys must
        # remain disabled when loading it; direct FeatureConfig() calls use the v2 defaults.
        kwargs.setdefault("use_advanced_visual", False)
        kwargs.setdefault("use_audio", False)
        kwargs.setdefault("audio_sample_rate", 16000)
        kwargs.setdefault("audio_feature_set", LEGACY_AUDIO_FEATURE_SET)
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


def feature_version_for_config(
    config: FeatureConfig, *, legacy_visual: bool = False
) -> str:
    """Return the immutable extractor version implied by a feature config."""
    if config.audio_feature_set == NOISE_NORMALIZED_AUDIO_FEATURE_SET:
        return AUDIO_NORMALIZED_FEATURE_VERSION
    if legacy_visual:
        return "court-motion-flow-v1"
    return FEATURE_VERSION


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
    short_event_min_seconds: float = 0.5
    short_event_threshold: float = 0.8

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
            or not _finite_number(self.short_event_min_seconds)
            or self.min_live_seconds < 0
            or self.bridge_gap_seconds < 0
            or self.short_event_min_seconds < 0
        ):
            raise ValueError("decoder durations must be non-negative")
        if (
            not _finite_number(self.short_event_threshold)
            or not self.enter_threshold <= self.short_event_threshold <= 1
        ):
            raise ValueError(
                "short_event_threshold must be between enter_threshold and 1"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DecoderConfig":
        kwargs = dict(value)
        # Preserve the old decoder exactly: artifacts without a short-event path
        # continue to remove every run shorter than min_live_seconds.
        if "short_event_min_seconds" not in kwargs:
            kwargs["short_event_min_seconds"] = float(kwargs.get("min_live_seconds", 1.0))
        kwargs.setdefault("short_event_threshold", 1.0)
        result = cls(**kwargs)
        result.validate()
        return result


@dataclass(frozen=True)
class PipelineConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
