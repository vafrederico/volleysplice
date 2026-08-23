#!/usr/bin/env python3
"""Bind production-state side-switch inputs and outputs to one revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from analysis.artifacts import atomic_write_text
from analysis.side_switch_v3 import FROZEN_RECORDING_SPLIT


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
INTAKE = Path("/mnt/freenas/volleycut/intake-2026-08-13")
DEFAULT_V6_PROVENANCE = ROOT / "reports/side-switch/side-switch-v6-provenance.json"
DEFAULT_OUTPUT = (
    ROOT / "reports/side-switch/side-switch-production-state-v1-provenance.json"
)
EXPECTED_V6_PROVENANCE_SHA256 = (
    "28541061320879877cb348355f92c6a1a80980b3fda754513667b1574c0e2fc5"
)
FEATURE_CACHE = (
    INTAKE
    / "experiments/environment-specialists-v2/features/"
    "audiovisual-audio-normalized-v3"
)
FEEDBACK_ROOT = Path("/mnt/freenas/volleycut/model-feedback")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _feature_sources() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    feedback_by_recording: dict[str, Path] = {}
    for path in sorted(FEEDBACK_ROOT.glob("*/bundle.json")):
        payload = _load(path)
        filename = str(payload.get("source", {}).get("file", {}).get("name", ""))
        if filename:
            feedback_by_recording[f"raw-no-backup-{Path(filename).stem}"] = path
    for role, recording_ids in FROZEN_RECORDING_SPLIT.items():
        for recording_id in recording_ids:
            if recording_id in feedback_by_recording:
                path = feedback_by_recording[recording_id]
                source_kind = "model-feedback-embedded-f104"
            else:
                matches = sorted(FEATURE_CACHE.glob(f"{recording_id}-*.npz"))
                if len(matches) != 1:
                    raise ValueError(
                        f"expected one cached F104 input for {recording_id}: {matches}"
                    )
                path = matches[0]
                source_kind = "cached-f104-npz"
            result.append(
                {
                    "recordingId": recording_id,
                    "role": role,
                    "kind": source_kind,
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
            )
    return result


def _all_labels_v2_inputs() -> dict[str, Path]:
    bundle_path = (
        INTAKE / "experiments/environment-specialists-v2/models/all-labels/bundle.json"
    )
    bundle = _load(bundle_path)
    heads = bundle.get("heads")
    if not isinstance(heads, Mapping):
        raise ValueError("all-labels-v2 bundle has no head mapping")
    result = {"allLabelsV2Bundle": bundle_path}
    for bundle_key, output_key in (
        ("rally", "Rally"),
        ("serve", "Serve"),
        ("deadState", "DeadState"),
    ):
        head = heads.get(bundle_key)
        if not isinstance(head, Mapping) or not head.get("path"):
            raise ValueError(f"all-labels-v2 bundle is missing {bundle_key}")
        model_dir = Path(str(head["path"])).expanduser().resolve()
        result[f"allLabelsV2{output_key}Metadata"] = model_dir / "model.json"
        result[f"allLabelsV2{output_key}Weights"] = model_dir / "weights.npz"
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite provenance: {output}")
    revision = subprocess.run(
        ["git", "rev-parse", args.implementation_revision],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    inherited_path = args.v6_provenance.expanduser().resolve()
    inherited_sha = _sha256(inherited_path)
    if inherited_sha != EXPECTED_V6_PROVENANCE_SHA256:
        raise ValueError("v6 provenance identity changed")
    inherited = _load(inherited_path)
    frozen_split = {
        role: list(recording_ids)
        for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
    }
    if inherited.get("frozenSplit") != frozen_split:
        raise ValueError("v6 provenance does not cover the frozen split")

    artifacts = {
        "productionState": ROOT
        / "reports/side-switch/side-switch-production-state-v1.json",
        "groundedManifest": ROOT
        / "reports/side-switch/side-switch-production-grounded-corpus-v1.json",
        "v5OriginalStateFeatures": ROOT
        / "reports/side-switch/side-switch-v5-production-state-v1-features.json",
        "v5GroundedBaseFeatures": ROOT
        / "reports/side-switch/side-switch-v5-serve-grounded-base-features.json",
        "v5GroundedStateFeatures": ROOT
        / "reports/side-switch/side-switch-v5-serve-grounded-production-state-v1-features.json",
        "v5Model": ROOT / "models/side-switch-v5-production-state-v1/model.json",
        "v5DevelopmentDataset": ROOT
        / "models/side-switch-v5-production-state-v1/dataset-development.json",
        "v5Evaluation": ROOT
        / "reports/side-switch/side-switch-v5-production-state-v1-evaluation.json",
        "v5Attribution": ROOT
        / "reports/side-switch/side-switch-v5-production-state-v1-attribution.json",
        "v6OriginalStateFeatures": ROOT
        / "reports/side-switch/side-switch-v6-production-state-v1-features.json",
        "v6GroundedBaseFeatures": ROOT
        / "reports/side-switch/side-switch-v6-serve-grounded-base-features.json",
        "v6GroundedStateFeatures": ROOT
        / "reports/side-switch/side-switch-v6-serve-grounded-production-state-v1-features.json",
        "v6Model": ROOT / "models/side-switch-v6-production-state-v1/model.json",
        "v6DevelopmentDataset": ROOT
        / "models/side-switch-v6-production-state-v1/dataset-development.json",
        "v6Evaluation": ROOT
        / "reports/side-switch/side-switch-v6-production-state-v1-evaluation.json",
        "v6Attribution": ROOT
        / "reports/side-switch/side-switch-v6-production-state-v1-attribution.json",
    }
    production_inputs = {
        **_all_labels_v2_inputs(),
        "previousRallyMetadata": Path("/mnt/freenas/volleycut")
        / "labeling-v1-2026-08-09-no-beach-2026-08-12/models/"
        "full-audiovisual-audio-normalized-v3/model.json",
        "previousRallyWeights": Path("/mnt/freenas/volleycut")
        / "labeling-v1-2026-08-09-no-beach-2026-08-12/models/"
        "full-audiovisual-audio-normalized-v3/weights.npz",
        "previousServeMetadata": Path("/mnt/freenas/volleycut")
        / "labeling-v1-2026-08-09-no-beach-2026-08-12/models/"
        "serve-specialist-audio-normalized-v5/model.json",
        "previousServeWeights": Path("/mnt/freenas/volleycut")
        / "labeling-v1-2026-08-09-no-beach-2026-08-12/models/"
        "serve-specialist-audio-normalized-v5/weights.npz",
        "previousDeadStateMetadata": Path("/mnt/freenas/volleycut")
        / "labeling-v1-2026-08-09-no-beach-2026-08-12/models/"
        "dead-state-transition-audio-normalized-v5-no-legacy-final/model.json",
        "previousDeadStateWeights": Path("/mnt/freenas/volleycut")
        / "labeling-v1-2026-08-09-no-beach-2026-08-12/models/"
        "dead-state-transition-audio-normalized-v5-no-legacy-final/weights.npz",
        "suppressionMetadata": INTAKE
        / "experiments/feedback-suppression-v3-2026-08-16/models/"
        "suppression-overlap-exclusion-retrained/model.json",
        "suppressionWeights": INTAKE
        / "experiments/feedback-suppression-v3-2026-08-16/models/"
        "suppression-overlap-exclusion-retrained/weights.npz",
    }
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-production-state-provenance-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "implementationRevision": revision,
        "frozenSplit": frozen_split,
        "sources": inherited["sources"],
        "sourceManifests": {
            "v6Provenance": {
                "path": str(inherited_path),
                "sha256": inherited_sha,
                "inheritance": (
                    "reuses the exact 21-video hashes, label identities, split, "
                    "V4 geometry, and V5/V6 feature lineage"
                ),
            },
            **dict(inherited.get("sourceManifests", {})),
        },
        "productionFeatureSources": _feature_sources(),
        "productionModelInputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in production_inputs.items()
        },
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in artifacts.items()
        },
        "notes": {
            "labelIndependence": (
                "production-state extraction and serve grounding do not inspect "
                "side-switch labels; labels enter only fitting and evaluation"
            ),
            "suppression": (
                "quarantined diagnostic only because its positive target contains "
                "side switches and its fitting recordings overlap all splits"
            ),
            "v5Decision": (
                "positive exploratory review-ranking result; no runtime promotion"
            ),
            "v6Decision": "rejected; frozen v6 remains unchanged",
            "evaluationStatus": (
                "candidate-conditioned retrospective confirmation, not a pristine test"
            ),
        },
    }
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-provenance", type=Path, default=DEFAULT_V6_PROVENANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--implementation-revision", default="HEAD")
    return parser


def main() -> None:
    payload = build(_parser().parse_args())
    print(
        json.dumps(
            {
                "implementationRevision": payload["implementationRevision"],
                "sources": len(payload["sources"]),
                "featureSources": len(payload["productionFeatureSources"]),
                "artifacts": len(payload["artifacts"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
