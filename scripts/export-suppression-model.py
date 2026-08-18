#!/usr/bin/env python3
"""Export a VolleyCut suppression logistic head as deterministic plain JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODEL_ID = "suppression-overlap-exclusion-retrained"
ARTIFACT_SHA256 = "39eddf58163901930434ea422a802686ae921ae1e8fe23c20a3c12e5f453da93"
WEIGHTS_SHA256 = "a943749b69c98a1bc926f8efc9fe60c67c1226534fe09a892c632519217aa3bb"
HELD_DECODER = {
    "smoothing_seconds": 1.0,
    "enter_threshold": 0.75,
    "exit_threshold": 0.65,
    "min_live_seconds": 0.5,
    "bridge_gap_seconds": 0.5,
    "short_event_min_seconds": 0.25,
    "short_event_threshold": 0.9,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()

    metadata = json.loads(
        (arguments.model_dir / "model.json").read_text(encoding="utf-8")
    )
    if metadata["weightsSha256"] != WEIGHTS_SHA256:
        raise SystemExit("suppression weights hash does not match the frozen decision")
    with np.load(arguments.model_dir / "weights.npz", allow_pickle=False) as values:
        head = {
            "mean": values["mean"].astype(np.float32).tolist(),
            "scale": values["scale"].astype(np.float32).tolist(),
            "weights": values["weights"].astype(np.float32).tolist(),
            "bias": float(values["bias"].reshape(-1)[0]),
        }

    payload = {
        "schemaVersion": 1,
        "modelId": MODEL_ID,
        "artifactSha256": ARTIFACT_SHA256,
        "weightsSha256": WEIGHTS_SHA256,
        "analysisFps": metadata["featureConfig"]["analysis_fps"],
        "featureVersion": metadata["featureVersion"],
        "featureNames": metadata["featureNames"],
        "decoderVersion": "held-production-suppression-decoder-v1",
        "decoder": HELD_DECODER,
        "head": head,
    }
    arguments.destination.parent.mkdir(parents=True, exist_ok=True)
    arguments.destination.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
