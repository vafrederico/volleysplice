#!/usr/bin/env python3
"""Export the selected side-switch winner as a minimal browser runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from analysis.artifacts import atomic_write_text


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    REPOSITORY / "data/side-switch-current-research-winner-production-port-v1.json"
)
DEFAULT_OUTPUT = REPOSITORY / "prod/public/runtime/side-switch-c2570481c30d.json"


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


def export(contract_path: Path, output: Path) -> Mapping[str, Any]:
    contract = _load(contract_path)
    model_spec = contract["sourceModel"]
    source_path = Path(str(model_spec["path"])).resolve()
    source_hash = _sha256(source_path)
    if source_hash != model_spec["sha256"]:
        raise ValueError(f"side-switch source model changed: {source_hash}")
    source = _load(source_path)
    classifier = dict(source["classifier"])
    classifier["threshold"] = float(model_spec["threshold"])
    if list(classifier["featureNames"]) != list(contract["orderedFeatureNames"]):
        raise ValueError("side-switch feature order diverged from the port contract")
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-browser-runtime-v1",
        "modelId": contract["winnerPointer"]["winnerId"],
        "fingerprint": f"sha256:{source_hash}",
        "featureVersion": "SIDE-SWITCH-UNION34-V1",
        "classifier": classifier,
        "candidateGenerator": contract["candidateGenerator"],
        "visualExtraction": contract["visualExtraction"],
        "decoder": contract["decoder"],
    }
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = export(args.contract.resolve(), args.output.resolve())
    print(json.dumps({"output": str(args.output), "modelId": payload["modelId"]}, indent=2))


if __name__ == "__main__":
    main()
