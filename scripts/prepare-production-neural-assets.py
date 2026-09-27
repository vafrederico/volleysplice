"""Materialize the pinned production neural bundles outside the source tree.

Only frozen weights and allowlisted inference configuration are exported. Source
locations resolve through the private ledger; no media, labels, provenance paths,
or research qualification receipts enter either application bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from analysis.private_ledger import private_value

CONFIG_KEYS = (
    "mean", "scale", "decoder", "weightsSha256", "encoderWeightsSha256", "epoch",
    "family", "modelIdentity", "selectionMode", "draw", "recallTargetPercent",
    "tokenDimension", "config", "trainingProjectorIncluded", "dinoRequiredForInference",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def inference_config(source: Path, selection: str) -> bytes:
    value = json.loads(source.read_text(encoding="utf-8"))
    result = {key: value[key] for key in CONFIG_KEYS}
    if (result["selectionMode"] != selection or result["tokenDimension"] != 3840
            or result["dinoRequiredForInference"] is not False
            or result["trainingProjectorIncluded"] is not False):
        raise ValueError("Selected inference configuration identity differs")
    for key in ("mean", "scale"):
        if len(result[key]) != 112 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in result[key]
        ):
            raise ValueError("Invalid frozen normalization parameters")
    if any(v <= 0 for v in result["scale"]):
        raise ValueError("Normalization scale must be positive")
    return canonical(result)


def prepare(source: Path, output: Path) -> dict:
    # Resolve symlinks too: generated release payloads should not fill the checkout.
    output = output.resolve()
    if output == REPOSITORY or REPOSITORY in output.parents:
        raise ValueError("Set an external output directory for the generated bundles")
    summary = {}
    for platform in ("android", "web"):
        manifest_path = REPOSITORY / "models" / "distilled-large" / f"{platform}-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["schemaVersion"] != 1 or manifest["defaultVariant"] != "high-recall":
            raise ValueError("Unexpected release manifest contract")
        root = output / platform / "rally-models"
        count = total = 0
        for variant in manifest["variants"].values():
            selection = variant["directory"]
            if selection not in ("recall", "f1"):
                raise ValueError("Unexpected selected variant")
            original = source / f"graphs-{selection}"
            graphs = original if platform == "android" else (
                source / "browser-distilled-large-runs" / "portable-pooling-v1" / f"graphs-{selection}"
            )
            payloads = {
                "encoder": (graphs / "mobile-large-encoder-fp32.onnx").read_bytes(),
                "temporal": (original / "mobile-large-tcn-dynamic-fp32.onnx").read_bytes(),
                "pipeline": inference_config(original / "mobile-large-pipeline.json", selection),
            }
            for component, entry in variant["files"].items():
                name = entry["name"]
                if Path(name).name != name or name in ("", ".", ".."):
                    raise ValueError("Manifest contains a nonlocal asset name")
                data = payloads[component]
                if len(data) != entry["sizeBytes"] or digest(data) != entry["sha256"]:
                    raise ValueError(f"Frozen {platform}/{selection}/{component} identity differs")
            # Validate a whole matched set before making it available to builds.
            destination = root / selection
            destination.mkdir(parents=True, exist_ok=True)
            for component, entry in variant["files"].items():
                data = payloads[component]
                target = destination / entry["name"]
                if not target.is_file() or digest(target.read_bytes()) != entry["sha256"]:
                    temporary = target.with_suffix(target.suffix + ".partial")
                    temporary.write_bytes(data)
                    temporary.replace(target)
                count += 1
                total += len(data)
        (root / "manifest.json").write_bytes(canonical(manifest))
        summary[platform] = {"variants": 2, "files": count, "payloadBytes": total}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=os.environ.get("VOLLEYCUT_NEURAL_ASSETS_DIR"))
    args = parser.parse_args()
    if args.output is None:
        parser.error("Provide --output or VOLLEYCUT_NEURAL_ASSETS_DIR")
    indexed = private_value("private-reference-0223")
    source = Path(os.environ.get("VOLLEYCUT_DEVICE_ARTIFACT_ROOT", indexed))
    print(json.dumps({"passed": True, "bundles": prepare(source, args.output)}))


if __name__ == "__main__":
    main()
