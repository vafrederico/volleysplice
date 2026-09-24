"""Unregistered shorter-context feasibility adapter; no training or data access.

Build the frozen transfer network with its original initialization, then change
only the five depthwise temporal convolutions' dilation/padding attributes.
This preserves parameter names, shapes, values and initialization RNG draws.
It is not part of the registered short-boost experiment or a selected follow-up.

The effective receptive field describes these temporal networks only. Their AV
inputs retain the existing feature windows and per-record preprocessing; DINO
image extraction is unchanged. Training can retain the original 252-tick chunks
and 62-tick real halos to pair sampling, tensor shapes and dropout RNG streams.
Do not use a smaller training chunk merely because the model needs less context.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from .compact_temporal_model import CompactTemporalError
from . import transfer_temporal_model as frozen


Context = Literal["original", "short"]
CONTEXT_DILATIONS = {"original": (1, 2, 4, 8, 16), "short": (1, 1, 2, 2, 2)}
KERNEL_SIZE = 5
ORIGINAL_TRAINING_HALO_TICKS = 62
ORIGINAL_TRAINING_CHUNK_TICKS = 252
MODEL_KINDS = frozen.MODEL_KINDS
PARAMETER_COUNTS = frozen.PARAMETER_COUNTS


def _dilations(context: Context) -> tuple[int, ...]:
    if context not in CONTEXT_DILATIONS:
        raise CompactTemporalError(f"unsupported context profile: {context!r}")
    return CONTEXT_DILATIONS[context]


def model_metadata(kind: str, *, context: Context = "short") -> dict[str, Any]:
    """Metadata only; does not allocate tensors or consume RNG."""
    dilations = _dilations(context)
    halo = (KERNEL_SIZE // 2) * sum(dilations)
    return {
        **frozen.model_metadata(kind),
        "contextProfile": context,
        "dilations": list(dilations),
        "kernelSize": KERNEL_SIZE,
        "haloTicks": halo,
        "receptiveFieldTicks": 2 * halo + 1,
        "temporalCenterSpanSeconds": 2 * halo / 4,
        "contextScope": "Temporal network only; existing AV windows/per-record transforms and DINO image extraction unchanged",
        "originalPairedTrainingHaloTicks": ORIGINAL_TRAINING_HALO_TICKS,
        "originalPairedTrainingChunkTicks": ORIGINAL_TRAINING_CHUNK_TICKS,
        "initialization": "Frozen four-head transfer factory; no additional parameter allocation or random draws",
        "status": "research-only-requires-registered-caller",
    }


def model_for(kind: str, *, context: Context = "short"):
    """Return the same four-head class/state schema with selected temporal span.

    Dilation/padding are not serialized by state_dict. Callers must persist and
    validate the context profile beside any checkpoint; loading the same tensor
    dictionary with another profile silently changes the computation.
    """
    dilations = _dilations(context)
    model = frozen.model_for(kind)
    temporal = model.temporal if kind == "dino_tcn" else model
    if (temporal.config.kernel_size != KERNEL_SIZE
            or tuple(temporal.config.dilations) != CONTEXT_DILATIONS["original"]
            or len(temporal.blocks) != len(dilations)):
        raise CompactTemporalError("frozen temporal architecture differs")
    for block, dilation in zip(temporal.blocks, dilations):
        convolution = block.depthwise
        if convolution.kernel_size != (KERNEL_SIZE,) or convolution.padding_mode != "zeros":
            raise CompactTemporalError("frozen depthwise convolution differs")
        padding = (KERNEL_SIZE // 2) * dilation
        convolution.dilation = (dilation,)
        convolution.padding = (padding,)
        # Keep the built-in Conv1d representation internally consistent as well
        # as its zero-padding forward path; this allocates no tensors or RNG.
        convolution._reversed_padding_repeated_twice = (padding, padding)
    model.config = replace(model.config, dilations=dilations)
    if kind == "dino_tcn":
        temporal.config = replace(temporal.config, dilations=dilations)
    for network in (model, temporal):
        network.context_profile = context
        network.halo_ticks = network.config.halo_ticks
        network.receptive_field_ticks = network.config.receptive_field_ticks
    return model


def primary_only_model(model):
    """Use the frozen three-head derivative; preserves the selected context."""
    return frozen.primary_only_model(model)
