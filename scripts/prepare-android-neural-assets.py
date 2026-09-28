"""Rebuild the pinned Android bundle from the checked-in public web bundle.

The web encoder differs from the native encoder only in its regional pooling
operation. Reverse that qualified rewrite, then verify every resulting byte
against the Android release manifest before making a bundle available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import onnx
from onnx import helper


REPOSITORY = Path(__file__).resolve().parents[1]
MANIFESTS = REPOSITORY / "models" / "distilled-large"
WEB_ASSETS = REPOSITORY / "prod" / "public" / "runtime" / "rally-models"
POOL_PREFIX = "portable_region_pool_"


def checked_bytes(path: Path, entry: dict) -> bytes:
    data = path.read_bytes()
    if len(data) != entry["sizeBytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Pinned neural asset differs: {path}")
    return data


def native_encoder(web_graph: bytes) -> bytes:
    model = onnx.load_model_from_string(web_graph)
    nodes = model.graph.node
    matches = [index for index, node in enumerate(nodes)
               if node.op_type == "MatMul" and list(node.output) == ["tokens"]]
    if len(matches) != 1 or matches[0] < 3:
        raise ValueError("Unexpected portable regional pooling graph")
    index = matches[0]
    reshape_spatial, transpose, reshape_weights, matmul = nodes[index - 3:index + 1]
    if (tuple(node.op_type for node in (reshape_spatial, transpose, reshape_weights, matmul))
            != ("Reshape", "Transpose", "Reshape", "MatMul")
            or list(reshape_spatial.output) != [POOL_PREFIX + "spatial_flat"]
            or list(transpose.input) != [POOL_PREFIX + "spatial_flat"]
            or list(transpose.output) != [POOL_PREFIX + "spatial_transposed"]
            or list(reshape_weights.output) != [POOL_PREFIX + "weights_flat"]
            or list(matmul.input) != [POOL_PREFIX + "weights_flat", POOL_PREFIX + "spatial_transposed"]
            or list(reshape_spatial.input)[1:] != [POOL_PREFIX + "spatial_shape"]
            or list(reshape_weights.input)[1:] != [POOL_PREFIX + "weights_shape"]):
        raise ValueError("Unexpected portable regional pooling graph")
    inputs = [reshape_spatial.input[0], reshape_weights.input[0]]
    del nodes[index - 3:index + 1]
    nodes.insert(index - 3, helper.make_node(
        "Einsum", inputs, ["tokens"], name="/Einsum", equation="bchw,brhw->brc"))

    shapes = {POOL_PREFIX + "spatial_shape", POOL_PREFIX + "weights_shape"}
    if {item.name for item in model.graph.initializer if item.name in shapes} != shapes:
        raise ValueError("Portable regional pooling shapes are missing")
    for position in range(len(model.graph.initializer) - 1, -1, -1):
        if model.graph.initializer[position].name in shapes:
            del model.graph.initializer[position]

    metadata = [index for index, item in enumerate(model.metadata_props)
                if item.key == "regionalPoolingImplementation"
                and item.value == "einsum-to-matmul-regional-pool-v1"]
    if len(metadata) != 1:
        raise ValueError("Unexpected regional pooling metadata")
    del model.metadata_props[metadata[0]]
    onnx.checker.check_model(model)
    return model.SerializeToString()


def prepare(output: Path) -> None:
    output = output.resolve()
    if output == REPOSITORY or REPOSITORY in output.parents:
        raise ValueError("Use an output directory outside the checkout")
    web_manifest_path = MANIFESTS / "web-manifest.json"
    android_manifest_path = MANIFESTS / "android-manifest.json"
    web_manifest = json.loads(web_manifest_path.read_text(encoding="utf-8"))
    android_manifest = json.loads(android_manifest_path.read_text(encoding="utf-8"))
    if (json.loads((WEB_ASSETS / "manifest.json").read_text(encoding="utf-8"))
            != web_manifest or web_manifest["variants"].keys() != android_manifest["variants"].keys()):
        raise ValueError("Checked-in web and Android model manifests differ")

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="android-neural-", dir=output) as temporary:
        staged = Path(temporary) / "rally-models"
        for key, android_variant in android_manifest["variants"].items():
            web_variant = web_manifest["variants"][key]
            if (android_variant["directory"] != web_variant["directory"]
                    or android_variant["id"] != web_variant["id"]):
                raise ValueError("Web and Android model selections differ")
            directory = android_variant["directory"]
            if directory not in ("recall", "f1"):
                raise ValueError("Unexpected model directory")
            for component, android_entry in android_variant["files"].items():
                web_entry = web_variant["files"][component]
                if android_entry["name"] != web_entry["name"]:
                    raise ValueError("Web and Android asset names differ")
                name = android_entry["name"]
                if Path(name).name != name:
                    raise ValueError("Unsafe asset name")
                web_data = checked_bytes(WEB_ASSETS / directory / name, web_entry)
                native_data = native_encoder(web_data) if component == "encoder" else web_data
                if (len(native_data) != android_entry["sizeBytes"]
                        or hashlib.sha256(native_data).hexdigest() != android_entry["sha256"]):
                    raise ValueError(f"Pinned Android neural asset differs: {directory}/{name}")
                target = staged / directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(native_data)
        shutil.copyfile(android_manifest_path, staged / "manifest.json")
        destination = output / "android" / "rally-models"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if output not in destination.resolve().parents:
            raise ValueError("Android bundle destination escapes the output directory")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(staged), str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output)
    print("Prepared both pinned Android neural model variants")


if __name__ == "__main__":
    main()
