"""Four-head AV research controls for the expanded-data development study.

The frozen compact implementation supplies every feature and temporal operation.
Only its final linear layer is widened to add an auxiliary keep-coverage logit.
Callers define supervision and decoding: the first three logits retain the
live/serve/end order, and keep must not enter the existing primary decoder.
All cohorts use the same four-head architecture and objective definition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .compact_temporal_model import (
    MODEL_KINDS,
    CompactTemporalConfig,
    CompactTemporalNetwork,
    nn,
    require_torch,
    trainable_parameter_count,
)


HEAD_NAMES = ("live", "serve", "end", "keep")
PRIMARY_HEAD_COUNT = 3
KEEP_HEAD_INDEX = 3


@dataclass(frozen=True)
class ExpandedTemporalConfig(CompactTemporalConfig):
    """Compact temporal configuration with an explicit four-head contract."""

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "headNames": list(HEAD_NAMES),
            "primaryHeadCount": PRIMARY_HEAD_COUNT,
            "auxiliaryHeadNames": [HEAD_NAMES[KEEP_HEAD_INDEX]],
            "auxiliaryHeadIndices": [KEEP_HEAD_INDEX],
        }


if nn is not None:

    class ExpandedTemporalNetwork(CompactTemporalNetwork):
        """Return float32 [B,T,4] logits with an auxiliary shared keep head.

        This changes only the output layer of the frozen compact network. The
        TCN still has five depthwise residual blocks, a 125-tick receptive field,
        and a 62-tick halo. Contextual controls still use their eight-tick halo.
        Eval chunks must contain real halos and end at true segment boundaries;
        the caller resets both supervision and inference at ignored spans.
        """

        def __init__(self, config: CompactTemporalConfig | None = None) -> None:
            expanded = ExpandedTemporalConfig(**asdict(config)) if config is not None else ExpandedTemporalConfig()
            super().__init__(expanded)
            if self.config.kind == "linear":
                previous = self.context_head
                self.context_head = nn.Linear(previous.in_features, len(HEAD_NAMES))
            elif self.config.kind == "mlp":
                previous = self.context_head[-1]
                self.context_head[-1] = nn.Linear(previous.in_features, len(HEAD_NAMES))
            else:
                previous = self.head
                self.head = nn.Linear(previous.in_features, len(HEAD_NAMES))
            self.float()

else:

    class ExpandedTemporalNetwork:  # type: ignore[no-redef]
        def __init__(self, config: CompactTemporalConfig | None = None) -> None:
            del config
            require_torch()
