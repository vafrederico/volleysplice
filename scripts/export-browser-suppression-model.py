#!/usr/bin/env python3
"""Export the held VolleyCut suppression head as deterministic browser JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.model import load_model


EXPECTED_ARTIFACT_SHA256 = (
    "39eddf58163901930434ea422a802686ae921ae1e8fe23c20a3c12e5f453da93"
)
EXPECTED_WEIGHTS_SHA256 = (
    "a943749b69c98a1bc926f8efc9fe60c67c1226534fe09a892c632519217aa3bb"
)
HELD_DECODER = {
    "smoothing_seconds": 1.0,
    "enter_threshold": 0.75,
    "exit_threshold": 0.65,
    "min_live_seconds": 0.5,
    "bridge_gap_seconds": 0.5,
    "short_event_min_seconds": 0.25,
    "short_event_threshold": 0.9,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args()

    model_dir = arguments.model_dir.expanduser().resolve()
    model = load_model(model_dir)
    if model.artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
        raise SystemExit(
            "suppression artifact differs from the held product decision: "
            f"{model.artifact_sha256}"
        )
    weights_path = model_dir / "weights.npz"
    if sha256(weights_path) != EXPECTED_WEIGHTS_SHA256:
        raise SystemExit("suppression weights differ from the held product decision")

    with np.load(weights_path, allow_pickle=False) as weights:
        payload = {
            "schemaVersion": 1,
            "modelId": "suppression-overlap-exclusion-retrained",
            "artifactSha256": EXPECTED_ARTIFACT_SHA256,
            "weightsSha256": EXPECTED_WEIGHTS_SHA256,
            "decoderVersion": "held-production-suppression-decoder-v1",
            "analysisFps": model.feature_config.analysis_fps,
            "featureVersion": model.feature_version,
            "featureNames": list(model.feature_names),
            "head": {
                "mean": weights["mean"].astype(np.float32).tolist(),
                "scale": weights["scale"].astype(np.float32).tolist(),
                "weights": weights["weights"].astype(np.float32).tolist(),
                "bias": float(weights["bias"].reshape(-1)[0]),
                "decoder": HELD_DECODER,
            },
        }

    arguments.destination.parent.mkdir(parents=True, exist_ok=True)
    arguments.destination.write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )
    emitted_sha256 = sha256(arguments.destination)
    manifest = {
        "schemaVersion": 1,
        "source": {
            "modelDirectory": str(model_dir),
            "artifactSha256": EXPECTED_ARTIFACT_SHA256,
            "metadataSha256": sha256(model_dir / "model.json"),
            "weightsSha256": EXPECTED_WEIGHTS_SHA256,
        },
        "emitted": {
            "filename": arguments.destination.name,
            "sha256": emitted_sha256,
        },
        "decoderVersion": "held-production-suppression-decoder-v1",
        "decoder": HELD_DECODER,
    }
    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
