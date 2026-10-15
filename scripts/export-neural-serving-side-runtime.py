"""Export the frozen encoder-specific centered serving heads for browser testing.

Private checkpoints and validation inputs resolve only through the external ledger.
Only model weights, public identities, frozen policies and aggregate parity travel
with the browser assets. Run with the optional PyTorch/ONNX research dependencies.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from analysis.private_ledger import private_value
from analysis.serving_side_precision_iterations import load_head, temporal_transform
from analysis.serving_side_cost_sensitive import scores
from analysis.serving_side_selective_metrics import decisions


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "high-recall": ("private-reference-0240", "serving-side-precision-iterations-2026-10-14.json"),
    "high-f1": ("private-reference-0242", "serving-side-balanced-precision-2026-10-14.json"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


class ServingEnsemble(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = nn.ModuleList(heads)

    def forward(self, windows, features):
        valid = windows[..., 3840:3841]
        tokens = windows[..., :3840]
        mean = (tokens * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        centered = torch.cat(((tokens - mean) * valid, windows[..., 3840:]), dim=-1)
        return torch.stack([head(centered, features) for head in self.heads], dim=1)


def main():
    torch.set_num_threads(2)
    rally = json.loads((ROOT / "prod/public/runtime/rally-models/manifest.json").read_text())
    manifest = {"schemaVersion": 1, "primaryTarget": .98, "policy": "all-three-seeds-agree",
                "inputContract": "centered-13x3843-plus-249-v1", "variants": {}}
    validation = {"kind": "neural-serving-browser-export-parity-v1", "syntheticInput": "all-valid-ramp-midrank-v1", "variants": {}}
    for selection, (index, report_name) in SOURCES.items():
        private_root = Path(private_value(index))
        report = json.loads((ROOT / "docs/research" / report_name).read_text())
        registration = json.loads((private_root / "registration.json").read_text())
        if registration["sha256"] != report["contractSha256"]:
            raise ValueError("Serving-head registration does not match the audited report")
        bundle = rally["variants"][selection]
        if selection == "high-f1":
            encoder_hash = report["encoderSha256"]
        else:
            original = json.loads((Path(private_value("private-reference-0225")) / "registration.json").read_text())
            encoder_hash = original["contract"]["encoderSha256"]
        if encoder_hash != bundle["encoderWeightsSha256"]:
            raise ValueError("Serving-head training encoder does not match the rally bundle")
        choices = [v for v in report["finalNeuralHeads"]
                   if v["arm"] == "09-neural-centered" and v["target"] == .98]
        if [v["seed"] for v in choices] != [3407, 1729, 20260918] or not all(v["feasible"] for v in choices):
            raise ValueError("The frozen three-seed policy is incomplete")
        heads = []
        for choice in choices:
            path = private_root / "fits/09-neural-centered" / str(choice["seed"]) / "all-development" / f"weights-{choice['epoch']}.npz"
            if digest(path.read_bytes()) != choice["weightsSha256"]:
                raise ValueError("Serving checkpoint integrity failed")
            heads.append(load_head(path))
        model = ServingEnsemble(heads).eval()
        target = ROOT / "prod/public/runtime/serving-models" / bundle["directory"]
        target.mkdir(parents=True, exist_ok=True)
        path = target / "head.onnx"
        torch.onnx.export(model, (torch.zeros(1, 13, 3843), torch.zeros(1, 249)), path,
                          input_names=["windows", "features"], output_names=["logits"],
                          dynamic_axes={"windows": {0: "batch"}, "features": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=17, dynamo=False)
        graph = onnx.load(path)
        graph.producer_name, graph.producer_version, graph.doc_string = "VolleySplice", "1", ""
        graph.graph.name, graph.graph.doc_string = "centered-serving-side", ""
        del graph.metadata_props[:]
        for node in graph.graph.node:
            node.doc_string = ""
        onnx.checker.check_model(graph)
        onnx.save(graph, path)
        windows = np.load(private_root / "windows.npy", allow_pickle=False)
        centered = temporal_transform(windows, "centered")
        with np.load(private_root / "features.npz", allow_pickle=False) as data:
            features = np.column_stack((data["ranks"], data["quality"])).astype(np.float32)
        session = ort.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
        maximum_error = 0.
        expected_all = np.empty((len(windows), 3, 2), np.float32)
        for start in range(0, len(windows), 16):
            inputs = {"windows": windows[start:start+16], "features": features[start:start+16]}
            actual = session.run(None, inputs)[0]
            with torch.inference_mode():
                expected = np.stack([head(torch.from_numpy(centered[start:start+16]),
                                          torch.from_numpy(inputs["features"])).numpy() for head in heads], axis=1)
            expected_all[start:start+16] = expected
            maximum_error = max(maximum_error, float(np.max(abs(actual-expected))))
            for seed_index, choice in enumerate(choices):
                a, b = scores(actual[:, seed_index]), scores(expected[:, seed_index])
                if not np.array_equal(decisions(a, choice["farCutoff"], choice["nearCutoff"]),
                                      decisions(b, choice["farCutoff"], choice["nearCutoff"])):
                    raise ValueError("ONNX changed a frozen serving-side acceptance decision")
        if maximum_error > 2e-5:
            raise ValueError("ONNX serving-side numerical parity failed")
        public = {"id": f"serving-side-centered-{bundle['directory']}-v1", "rallyModelId": bundle["id"],
                  "directory": bundle["directory"], "encoderSha256": bundle["files"]["encoder"]["sha256"],
                  "encoderWeightsSha256": encoder_hash,
                  "file": {"name": "head.onnx", "sha256": digest(path.read_bytes()), "sizeBytes": path.stat().st_size},
                  "seeds": [{k: v[k] for k in ("seed", "epoch", "weightsSha256", "farCutoff", "nearCutoff")} for v in choices]}
        public["fingerprint"] = digest(json.dumps(public, sort_keys=True, separators=(",", ":")).encode())
        manifest["variants"][selection] = public
        synthetic = np.zeros((1, 13, 3843), np.float32)
        for frame in range(13):
            synthetic[0, frame, :3840] = frame * .05 + (np.arange(3840) % 7) * .01
            synthetic[0, frame, 3840:3842] = (1, (frame - 6) / 6)
        synthetic_features = np.full((1, 249), .01, np.float32)
        synthetic_features[:, :237] = .5
        with torch.inference_mode():
            synthetic_logits = model(torch.from_numpy(synthetic), torch.from_numpy(synthetic_features)).numpy()[0]
        validation["variants"][selection] = {"validationRows": len(windows), "maximumLogitError": maximum_error,
            "acceptanceDecisionsMatch": True, "sha256": public["file"]["sha256"], "syntheticLogits": synthetic_logits.tolist()}
        with tempfile.TemporaryDirectory(prefix="serving-parity-") as temporary:
            directory = Path(temporary)
            for name, values in (("windows", windows), ("features", features), ("logits", expected_all)):
                values.astype("<f4").tofile(directory / name)
            descriptor = {"selection": selection, "rows": len(windows),
                          **{name: str(directory / name) for name in ("windows", "features", "logits")}}
            (directory / "inputs.json").write_text(json.dumps(descriptor))
            result = subprocess.run(["node", "--experimental-strip-types", "scripts/validate-neural-serving-browser.mjs",
                                     str(directory / "inputs.json")], cwd=ROOT, capture_output=True, text=True, check=True)
            validation["variants"][selection]["browserWasm"] = json.loads(result.stdout)
        print(json.dumps({"selection": selection, **{k: v for k, v in validation["variants"][selection].items() if k != "syntheticLogits"}}))
    text = json.dumps(manifest, indent=2) + "\n"
    pinned = ROOT / "models/serving-side/web-manifest.json"
    pinned.parent.mkdir(parents=True, exist_ok=True)
    pinned.write_text(text)
    (pinned.parent / "export-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (ROOT / "prod/public/runtime/serving-models/manifest.json").write_text(text)


if __name__ == "__main__":
    main()
