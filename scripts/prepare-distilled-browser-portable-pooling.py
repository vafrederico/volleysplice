"""Replace only regional Einsum pooling with an equivalent browser MatMul graph.

The original selected graphs remain immutable. Every learned initializer is
preserved byte-for-byte, and both derived encoders are independently qualified
against the original CPU golden outputs before a completion receipt is written.
No fitting, calibration, precision conversion or model selection occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value


KIND = "einsum-to-matmul-regional-pool-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def initializers(model):
    return {item.name: hashlib.sha256(item.SerializeToString()).hexdigest() for item in model.graph.initializer}


def rewrite(model):
    selected = [node for node in model.graph.node if node.op_type == "Einsum"]
    if len(selected) != 1 or len(selected[0].input) != 2 or list(selected[0].output) != ["tokens"]:
        raise ValueError("Expected exactly one frozen regional pooling operation")
    old = selected[0]
    attributes = {item.name: helper.get_attribute_value(item) for item in old.attribute}
    if attributes != {"equation": b"bchw,brhw->brc"}:
        raise ValueError("Unexpected Einsum contract")
    existing = {name for node in model.graph.node for name in (*node.input, *node.output)}
    prefix = "portable_region_pool_"
    if any(name.startswith(prefix) for name in existing):
        raise ValueError("Portable pooling names already exist")
    before = initializers(model)
    shapes = [numpy_helper.from_array(np.array([1, 960, 49], np.int64), prefix + "spatial_shape"),
              numpy_helper.from_array(np.array([1, 4, 49], np.int64), prefix + "weights_shape")]
    replacement = [
        helper.make_node("Reshape", [old.input[0], shapes[0].name], [prefix + "spatial_flat"]),
        helper.make_node("Transpose", [prefix + "spatial_flat"], [prefix + "spatial_transposed"], perm=[0, 2, 1]),
        helper.make_node("Reshape", [old.input[1], shapes[1].name], [prefix + "weights_flat"]),
        helper.make_node("MatMul", [prefix + "weights_flat", prefix + "spatial_transposed"], ["tokens"]),
    ]
    nodes = []
    for node in model.graph.node:
        nodes.extend(replacement if node is old else [node])
    # Protobuf repeated access may not preserve Python wrapper identity.
    if len(nodes) != len(model.graph.node) + 3:
        nodes = []
        for node in model.graph.node:
            nodes.extend(replacement if node.SerializeToString() == old.SerializeToString() else [node])
    if len(nodes) != len(model.graph.node) + 3:
        raise ValueError("Unexpected regional replacement count")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(shapes)
    after = initializers(model)
    if {key: after[key] for key in before} != before or len(after) != len(before) + 2:
        raise ValueError("Rewrite altered a trained initializer")
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata["regionalPoolingImplementation"] = KIND
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    return dict(preservedInitializerCount=len(before), addedShapeInitializers=2,
                preservedInitializerManifestSha256=hashlib.sha256(
                    json.dumps(before, sort_keys=True, separators=(",", ":")).encode()).hexdigest())


def session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def checked_array(root, entry):
    name = entry["file"]
    if Path(name).name != name:
        raise ValueError("Golden filename is not local")
    path = root / name
    if path.stat().st_size != entry["sizeBytes"] or digest(path) != entry["sha256"]:
        raise ValueError("Golden fixture identity differs")
    return np.fromfile(path, dtype=np.dtype(entry["dtype"]).newbyteorder("<")).reshape(entry["shape"])


def qualify(graph, fixtures, mode):
    manifest = read(fixtures / "manifest.json")
    if manifest.get("kind") != "distilled-large-browser-image-fixtures-v1" or len(manifest["cases"]) != 8:
        raise ValueError("Expected complete independent original-encoder goldens")
    runtime, checks = session(graph), []
    for case in manifest["cases"]:
        inputs = dict(image=checked_array(fixtures, case["expected"]["pixels"]),
                      pool_weights=checked_array(fixtures, case["expected"]["poolWeights"]))
        actual = runtime.run(None, inputs)[0]
        expected = checked_array(fixtures, case["encoderExpected"][mode]["rawTokens"])
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
        rounded = actual.astype(np.float16).astype(np.float32)
        expected_rounded = checked_array(fixtures, case["encoderExpected"][mode]["roundedTokens"])
        checks.append(dict(case=case["id"], passed=True,
                           maxAbsoluteError=float(np.max(abs(actual - expected))),
                           meanAbsoluteError=float(np.mean(abs(actual - expected))),
                           roundedExact=bool(np.array_equal(rounded, expected_rounded)),
                           roundedMaximumDifference=float(np.max(abs(rounded - expected_rounded))),
                           roundedDifferentValues=int(np.count_nonzero(rounded != expected_rounded))))
    return checks


def tiny_probes(output):
    folder = output / "operator-probes"
    folder.mkdir()
    features = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 1, 2, 2])
    weights = helper.make_tensor_value_info("weights", TensorProto.FLOAT, [1, 1, 2, 2])
    result = helper.make_tensor_value_info("tokens", TensorProto.FLOAT, [1, 1, 1])
    graph = helper.make_graph([helper.make_node("Einsum", ["features", "weights"], ["tokens"],
                                               equation="bchw,brhw->brc")], "regional-pool-control", [features, weights], [result])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    path = folder / "regional-pool-einsum.onnx"
    onnx.save(model, path)
    reference = session(path)
    cases = []
    for name, values, pool in (("uniform", [2.] * 4, [.25] * 4),
                               ("signed", [-1., 2., -3., 4.], [.1, .2, .3, .4])):
        x = np.array(values, np.float32).reshape(1, 1, 2, 2)
        w = np.array(pool, np.float32).reshape(1, 1, 2, 2)
        expected = np.einsum("bchw,brhw->brc", x, w)
        np.testing.assert_allclose(reference.run(None, dict(features=x, weights=w))[0], expected, atol=1e-7)
        cases.append(dict(id=name, graph=dict(file=path.name, sha256=digest(path), sizeBytes=path.stat().st_size),
                          inputs=[dict(name="features", dtype="float32", shape=list(x.shape), values=x.reshape(-1).tolist()),
                                  dict(name="weights", dtype="float32", shape=list(w.shape), values=w.reshape(-1).tolist())],
                          expected=dict(outputName="tokens", dtype="float32", shape=list(expected.shape), values=expected.reshape(-1).tolist())))
    write(folder / "manifest.json", dict(schemaVersion=1, kind="distilled-large-webgpu-operator-probes-v1", cases=cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--source-index", default="private-reference-0223")
    args = parser.parse_args()
    output = args.output.resolve()
    repository = Path(__file__).resolve().parents[1]
    if output == repository or repository in output.parents:
        raise ValueError("Derived graph artifacts must remain outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    source = Path(private_value(args.source_index))
    tiny_probes(output)
    results = []
    for mode in ("recall", "f1"):
        original = source / ("graphs-" + mode)
        contract, config = read(original / "input-contract.json"), read(original / "mobile-large-pipeline.json")
        for name, expected in contract["hashes"].items():
            if Path(name).name != name or digest(original / name) != expected:
                raise ValueError("Original frozen graph identity differs")
        if contract["selectionMode"] != mode or config["selectionMode"] != mode:
            raise ValueError("Original selected model association differs")
        destination = output / ("graphs-" + mode)
        destination.mkdir()
        encoder_name = "mobile-large-encoder-fp32.onnx"
        model = onnx.load(original / encoder_name)
        proof = rewrite(model)
        onnx.save(model, destination / encoder_name)
        for name in contract["hashes"]:
            if name != encoder_name:
                shutil.copyfile(original / name, destination / name)
        checks = qualify(destination / encoder_name, args.fixtures, mode)
        provenance = dict(kind=KIND, sourceArtifactIndex=args.source_index,
            originalEncoderSha256=digest(original / encoder_name),
            originalInputContractSha256=digest(original / "input-contract.json"),
            originalQualificationSha256=digest(original / "qualification.json"),
            derivedEncoderSha256=digest(destination / encoder_name),
            originalGoldenManifestSha256=digest(args.fixtures / "manifest.json"),
            scriptSha256=digest(__file__), trainingPerformed=False, precision="fp32", **proof)
        derived_contract = {**contract, "hashes": {name: digest(destination / name) for name in contract["hashes"]},
                            "graphRewrite": provenance,
                            "scope": "Selected trained weights unchanged; mathematically equivalent reshape/transpose/MatMul regional pooling."}
        write(destination / "input-contract.json", derived_contract)
        qualification = dict(kind="distilled-large-portable-pooling-cpu-qualification-v1", passed=True,
            selectionMode=mode, studentWeightsSha256=config["encoderWeightsSha256"],
            temporalWeightsSha256=config["weightsSha256"], encoderSha256=provenance["derivedEncoderSha256"],
            graphRewrite=provenance, encoderChecks=checks,
            scope="New CPU qualification against original frozen encoder goldens; browser GPU qualification remains separate.")
        write(destination / "qualification.json", qualification)
        results.append(dict(selection=mode, **qualification))
    write(output / "preparation.json", dict(schemaVersion=1, passed=True, kind=KIND, selections=results))
    print(json.dumps(dict(passed=True, kind=KIND,
        selections=[dict(selection=row["selection"], maxAbsoluteError=max(c["maxAbsoluteError"] for c in row["encoderChecks"]),
                         roundedDifferentValues=sum(c["roundedDifferentValues"] for c in row["encoderChecks"])) for row in results])))


if __name__ == "__main__":
    main()
