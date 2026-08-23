#!/usr/bin/env python3
"""Verify the checked-in side-switch production-port contract against frozen sources."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    REPOSITORY / "data/side-switch-current-research-winner-production-port-v1.json"
)


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 changed: {actual}")


def verify(contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = _load(contract_path)
    if contract.get("status") != "implemented-production-beta":
        raise ValueError("port status must identify the shipped production beta")
    if contract.get("automaticProductionUse") is not True:
        raise ValueError("the shipped production beta must enable automatic marker use")

    pointer_spec = contract["winnerPointer"]
    pointer_path = (REPOSITORY / str(pointer_spec["path"])).resolve()
    _require_hash(pointer_path, str(pointer_spec["sha256"]), "winner pointer")
    pointer = _load(pointer_path)
    if pointer["winner"]["id"] != pointer_spec["winnerId"]:
        raise ValueError("winner ID diverged from the selected pointer")

    model_spec = contract["sourceModel"]
    model_path = Path(str(model_spec["path"])).resolve()
    _require_hash(model_path, str(model_spec["sha256"]), "source model")
    model = _load(model_path)
    selection = model["selection"]
    classifier = model["classifier"]
    names = list(contract["orderedFeatureNames"])
    grouped = [
        *contract["featureGroups"]["v5Visual22"],
        *contract["featureGroups"]["productionState10"],
        *contract["featureGroups"]["candidateMetadata2"],
    ]
    if names != grouped or len(names) != 34 or len(set(names)) != 34:
        raise ValueError("checked-in 34-feature grouping is malformed")
    if (
        names != selection["variant"]["featureNames"]
        or names != classifier["featureNames"]
    ):
        raise ValueError("port feature order diverged from the source model")
    if selection["variant"]["id"] != "union34-top2-x2":
        raise ValueError("source model is no longer the promoted fixed variant")
    if pointer["winner"]["featureProfile"] != model_spec["featureProfile"]:
        raise ValueError("winner feature profile diverged")
    threshold = float(model_spec["threshold"])
    if not math.isclose(threshold, float(selection["threshold"]), abs_tol=1e-15):
        raise ValueError("model selection threshold diverged")
    if not math.isclose(
        threshold, float(pointer["winner"]["threshold"]), abs_tol=1e-15
    ):
        raise ValueError("winner pointer threshold diverged")

    runtime_spec = contract["runtimeAsset"]
    runtime_path = (REPOSITORY / str(runtime_spec["path"])).resolve()
    _require_hash(runtime_path, str(runtime_spec["sha256"]), "browser runtime")
    runtime = _load(runtime_path)
    runtime_classifier = runtime["classifier"]
    if (
        runtime["modelId"] != pointer_spec["winnerId"]
        or runtime["fingerprint"] != f"sha256:{model_spec['sha256']}"
        or runtime_classifier["featureNames"] != names
    ):
        raise ValueError("browser runtime identity or feature signature diverged")
    for parameter in ("impute", "mean", "scale", "weights", "bias"):
        if runtime_classifier[parameter] != classifier[parameter]:
            raise ValueError(f"browser classifier diverged: {parameter}")
    if not math.isclose(
        float(runtime_classifier["threshold"]), threshold, abs_tol=1e-15
    ):
        raise ValueError("browser classifier threshold diverged")

    decoder_contract = contract["decoder"]
    decoder_model = model["decoder"]
    decoder_pointer = pointer["winner"]["decoder"]
    decoder_pairs = {
        "minimumCandidateIndexSeparation": "minimumIndexSeparation",
        "minimumTimeSeparationSeconds": "minimumTimeSeparationSeconds",
        "freePredictionsPerRecording": "freePredictionsPerRecording",
        "countPenaltyLogitPerExcessPrediction": "countPenaltyLogit",
        "usesCadence": "usesCadence",
        "reanchorOnSelection": "reanchorOnSelection",
        "hardMaximumPredictions": "hardMaximumPredictions",
    }
    for contract_name, artifact_name in decoder_pairs.items():
        value = decoder_contract[contract_name]
        if (
            value != decoder_model[artifact_name]
            or value != decoder_pointer[artifact_name]
        ):
            raise ValueError(f"decoder field diverged: {contract_name}")

    features_spec = contract["sourceFeatures"]
    features_path = Path(str(features_spec["path"])).resolve()
    _require_hash(features_path, str(features_spec["sha256"]), "source features")
    features = _load(features_path)
    profile = features["profile"]
    if len(profile["featureNames"]) != int(features_spec["storedFeatureCount"]):
        raise ValueError("stored feature count diverged")
    if profile["primaryFrozenHeadFeatureNames"] != names[:32]:
        raise ValueError("selected visual/state feature prefix diverged")
    production_ids = {
        features["productionModels"]["allLabelsV2"]["modelId"],
        features["productionModels"]["previousProduction"]["modelId"],
    }
    if production_ids != set(contract["runtimeDependencies"]["productionModelSources"]):
        raise ValueError("production model dependencies diverged")

    candidate_source = features["sources"]["candidates"]
    candidate_path = Path(str(candidate_source["path"])).resolve()
    _require_hash(candidate_path, str(candidate_source["sha256"]), "candidate union")
    candidate = _load(candidate_path)["selected"]["config"]
    generator = contract["candidateGenerator"]
    expected_candidate = {
        "signal": "deadState",
        "threshold": generator["internalPeakThreshold"],
        "minimumPeakSeparationSeconds": generator[
            "internalPeakMinimumSeparationSeconds"
        ],
        "rangeEdgeExclusionSeconds": generator[
            "internalPeakRangeEdgeExclusionSeconds"
        ],
        "proposalHalfWidthSeconds": generator[
            "internalPeakProposalHalfWidthSeconds"
        ],
    }
    if candidate != expected_candidate:
        raise ValueError("candidate-generator contract diverged")

    return {
        "winnerId": pointer_spec["winnerId"],
        "features": len(names),
        "threshold": threshold,
        "candidateConfig": candidate,
        "status": contract["status"],
        "runtimeAsset": str(runtime_path),
    }


def main() -> None:
    print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
