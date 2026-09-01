"""Load frozen browser runtime heads into the analysis replay model contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import DecoderConfig, FeatureConfig
from .model import DEAD_STATE_TASK, RALLY_LIVE_TASK, SERVE_CONTACT_TASK, LogisticModel
from .side_switch_production_replay import ProductionHeads


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {source}")
    return value


def runtime_head(
    runtime: Mapping[str, Any],
    key: str,
    prediction_task: str,
) -> LogisticModel:
    feature_config = FeatureConfig.from_dict(runtime["featureConfig"])
    head = runtime[key]
    if not isinstance(head, Mapping):
        raise ValueError(f"runtime head is malformed: {key}")
    training: dict[str, Any] = {}
    if key == "serve":
        training = {
            "serveDecoder": head["decoder"],
            "composition": head["composition"],
        }
    elif key == "deadState":
        training = {
            "selectedDeadStateDecoder": head["decoder"],
            "selectedRefinement": head["refinement"],
        }
    return LogisticModel(
        feature_config=feature_config,
        feature_names=tuple(str(name) for name in runtime["featureNames"]),
        mean=np.asarray(head["mean"], dtype=np.float32),
        scale=np.asarray(head["scale"], dtype=np.float32),
        weights=np.asarray(head["weights"], dtype=np.float32),
        bias=float(head["bias"]),
        decoder=DecoderConfig.from_dict(runtime["rally"]["decoder"]),
        training_summary=training,
        artifact_sha256=str(head.get("artifactSha256", "runtime")),
        feature_version=str(runtime["featureVersion"]),
        prediction_task=prediction_task,
    )


def production_heads(runtime: Mapping[str, Any]) -> ProductionHeads:
    return ProductionHeads(
        rally=runtime_head(runtime, "rally", RALLY_LIVE_TASK),
        serve=runtime_head(runtime, "serve", SERVE_CONTACT_TASK),
        dead=runtime_head(runtime, "deadState", DEAD_STATE_TASK),
    )


def load_production_runtime(path: str | Path) -> tuple[dict[str, Any], ProductionHeads]:
    runtime = load_json_object(path)
    return runtime, production_heads(runtime)
