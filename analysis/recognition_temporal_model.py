"""Explicit input-family/temporal-head adapters for the recognition experiment."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn

from .expanded_temporal_model import ExpandedTemporalConfig, ExpandedTemporalNetwork
from .transfer_temporal_model import model_for as transfer_model


@dataclass(frozen=True)
class RecognitionConfig:
    family: str = 'av'
    head: str = 'transformer'
    scalar_dimension: int = 0
    token_count: int = 4
    token_dimension: int = 576
    projection_dimension: int = 16

    def validate(self):
        for name in ('scalar_dimension', 'token_count', 'token_dimension', 'projection_dimension'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{name} must be an integer')
        if self.family not in ('av', 'dino', 'player', 'mobile') or self.head not in ('tcn', 'transformer'):
            raise ValueError('Unsupported recognition model')
        if self.scalar_dimension < 0 or (self.family in ('av', 'dino') and self.scalar_dimension):
            raise ValueError('Invalid scalar dimension')
        if self.family == 'player' and not self.scalar_dimension:
            raise ValueError('Player family requires features')
        if min(self.token_count, self.token_dimension, self.projection_dimension) < 1:
            raise ValueError('Invalid visual token geometry')

    @property
    def input_dimension(self):
        self.validate()
        return 3944 if self.family == 'dino' else 104 + self.scalar_dimension + (
            self.token_count * self.token_dimension if self.family == 'mobile' else 0)


def temporal_head(head, dimension):
    if head == 'tcn':
        return ExpandedTemporalNetwork(ExpandedTemporalConfig(kind='tcn', input_dimension=dimension))
    from .local_attention_temporal_model import LocalAttentionTemporalConfig, LocalAttentionTemporalNetwork
    return LocalAttentionTemporalNetwork(LocalAttentionTemporalConfig(input_dimension=dimension))


class MobileRecognitionNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_norm = nn.LayerNorm(config.token_dimension)
        self.token_projection = nn.Linear(config.token_dimension, config.projection_dimension)
        self.temporal = temporal_head(config.head, 104 + config.scalar_dimension + config.token_count * config.projection_dimension)

    def forward(self, values):
        c = self.config
        tokens = values[..., 104:104+c.token_count*c.token_dimension].reshape(*values.shape[:2], c.token_count, c.token_dimension)
        projected = torch.relu(self.token_projection(self.token_norm(tokens))).flatten(-2)
        scalars = values[..., 104+c.token_count*c.token_dimension:]
        return self.temporal(torch.cat((values[..., :104], projected, scalars), dim=-1))


def model_for(config):
    config.validate()
    if config.family == 'av' and config.head == 'tcn':
        return transfer_model('tcn')
    if config.family == 'dino':
        if config.head == 'tcn':
            return transfer_model('dino_tcn')
        from .local_attention_temporal_model import model_for as attention_model
        return attention_model('dino_transformer')
    if config.family == 'mobile':
        return MobileRecognitionNetwork(config).float()
    return temporal_head(config.head, config.input_dimension).float()


def model_metadata(config):
    with torch.random.fork_rng(devices=[]):
        model = model_for(config)
    return {**asdict(config), 'inputDimension': config.input_dimension,
            'parameters': sum(p.numel() for p in model.parameters()),
            'heads': ['live', 'serve', 'end', 'keep'], 'receptiveFieldTicks': 125,
            'contextRadiusTicks': 62, 'featureRateHz': 4,
            'featureExtractionIncluded': False}
