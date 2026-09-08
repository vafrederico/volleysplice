#!/usr/bin/env python3
"""Export the three-head VolleySplice browser inference bundle as plain JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODEL_NAMES = {
    "rally": "rally",
    "serve": "serve",
    "deadState": "dead-state",
}

MODEL_ID = "model-1ca43e38eefc"

MODEL_VERSIONS = {
    "rally": "environment-specialists-v2-all-labels-rally",
    "serve": "environment-specialists-v2-all-labels-serve",
    "deadState": "environment-specialists-v2-all-labels-dead-state",
}

ARTIFACT_SHA256 = {
    "rally": "d084247aa09fd60b10a45458d150c4c1e1132b79b262009d2295e6c6b5e75693",
    "serve": "0fc2f32e25d1784ec131ced2bb874ba90ba8e625726da0c728205e37d8ff11f6",
    "deadState": "1ca43e38eefc0a3b0554b8818fe5c77492dc8bd697301fc55d229bf4092b2329",
}

SERVE_DECODER = {
    "threshold": 0.85,
    "minSeparationSeconds": 10.0,
    "timeOffsetSeconds": 0.25,
}

SERVE_COMPOSITION = {
    "method": "serve-anchor-permissive-live-fallback-v1",
    "associationSeconds": 0.5,
    "fallbackSeconds": 2.5,
    "maxRescueSeconds": 5.0,
    "permissiveDecoder": {
        "smoothing_seconds": 2.0,
        "enter_threshold": 0.5,
        "exit_threshold": 0.4,
        "min_live_seconds": 0.25,
        "bridge_gap_seconds": 0.0,
        "short_event_min_seconds": 0.25,
        "short_event_threshold": 0.9,
    },
}

DEAD_STATE_DECODER = {
    "deadThreshold": 0.9,
    "liveResetThreshold": 0.4,
    "minimumLiveSamples": 1,
    "minimumDeadSamples": 1,
    "minAfterServeSeconds": 0.0,
    "maxAfterServeSeconds": 1.5,
    "timeOffsetSeconds": 0.0,
}

DEAD_STATE_REFINEMENT = {
    "method": "refine-v5-end",
    "endWindowSeconds": 0.75,
    "missingTransition": "exact-noop",
    "rescueMissingIntervals": False,
}


def _head(
    models_dir: Path, role: str, name: str
) -> tuple[dict[str, object], dict[str, object]]:
    model_dir = models_dir / name
    metadata = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    with np.load(model_dir / "weights.npz", allow_pickle=False) as weights:
        payload: dict[str, object] = {
            "version": MODEL_VERSIONS[role],
            "artifactSha256": ARTIFACT_SHA256[role],
            "predictionTask": metadata["predictionTask"],
            "mean": weights["mean"].astype(np.float32).tolist(),
            "scale": weights["scale"].astype(np.float32).tolist(),
            "weights": weights["weights"].astype(np.float32).tolist(),
            "bias": float(weights["bias"].reshape(-1)[0]),
        }
    return metadata, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models_dir", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()

    metadata: dict[str, dict[str, object]] = {}
    heads: dict[str, dict[str, object]] = {}
    for role, name in MODEL_NAMES.items():
        model_metadata, head = _head(arguments.models_dir, role, name)
        metadata[role] = model_metadata
        heads[role] = head

    rally = metadata["rally"]
    feature_names = rally["featureNames"]
    for role, model_metadata in metadata.items():
        if model_metadata["featureNames"] != feature_names:
            raise SystemExit(f"{role} feature signature differs from rally model")
        if model_metadata["featureConfig"] != rally["featureConfig"]:
            raise SystemExit(f"{role} feature configuration differs from rally model")

    heads["rally"]["decoder"] = rally["decoder"]
    heads["serve"]["decoder"] = SERVE_DECODER
    heads["serve"]["composition"] = SERVE_COMPOSITION
    heads["deadState"]["decoder"] = DEAD_STATE_DECODER
    heads["deadState"]["refinement"] = DEAD_STATE_REFINEMENT

    payload = {
        "schemaVersion": 1,
        "modelId": MODEL_ID,
        "analysisFps": rally["featureConfig"]["analysis_fps"],
        "featureVersion": rally["featureVersion"],
        "featureConfig": rally["featureConfig"],
        "featureNames": feature_names,
        **heads,
    }
    arguments.destination.parent.mkdir(parents=True, exist_ok=True)
    arguments.destination.write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
