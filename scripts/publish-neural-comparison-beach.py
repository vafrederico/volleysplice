#!/usr/bin/env python3
"""Run the two frozen high-recall TCNs on beach footage and publish UI references.

The temporal checkpoints and decoders are already selected. Feature extraction and
model inference are label blind; human labels are read only after prediction to
verify that the existing labeling catalog remains available and unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from analysis import neural_generalization_inputs as inputs
from analysis.mobile_visual_features import (MobileVisualConfig, extract_recording_cache,
                                             load_mobile_backbone)
from analysis.neural_development import decode
from analysis.neural_generalization_experiment import load_checkpoint, MODELS
from analysis.neural_recall_sweep import SweepExample
from analysis.neural_recognition_fit import predict


MODEL_IDS = (
    "neural-dino-tcn-fp32-high-recall",
    "neural-mobile-tcn-fp32-high-recall",
)
LEDGER_COMPARISON_MODELS = "neural-comparison-index.models"
LEDGER_INVENTORY = "recall-sweep-inventory.records"
LEDGER_PHASE1_FEATURES = "phase1-manifest.recordings"
LEDGER_BEACH_ARTIFACTS = "beach-publication-artifacts"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encoded(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = encoded(value)
    if path.exists() and path.read_bytes() == data:
        return
    path.write_bytes(data)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ledger_reference(ledger, index, digest, **fields):
    """Return a portable artifact identity without serializing a filesystem path."""
    require(isinstance(index, int) and index >= 0, "Ledger reference index is invalid")
    require(isinstance(digest, str) and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            "Ledger reference digest is invalid")
    return {"ledger": ledger, "index": index, **fields, "sha256": digest}


def public_model(model):
    return {key: value for key, value in model.items() if not key.startswith("_")}


def require_portable(value, label):
    """Reject host paths before a generated JSON artifact is published."""
    def visit(node):
        if isinstance(node, dict):
            require("path" not in node, f"{label} contains a path field")
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, str):
            normalized = node.replace("\\", "/")
            require(not normalized.startswith(("/", "file://", "//"))
                    and not (len(normalized) >= 3 and normalized[1:3] == ":/"),
                    f"{label} contains an absolute path")
    visit(value)
    return value


def selected_models(index):
    by_id = {row["modelId"]: {**row, "_ledgerIndex": position}
             for position, row in enumerate(index["models"])}
    require(set(MODEL_IDS) <= set(by_id), "High-recall model selections are absent from the comparison index")
    result = [by_id[key] for key in MODEL_IDS]
    expected = {
        MODEL_IDS[0]: ("dino-tcn", "original-medium", 3407, 30),
        MODEL_IDS[1]: ("mobile-tcn", "expanded-large", 3407, 15),
    }
    for row in result:
        require((row["model"], row["variant"], row["seed"], row["epoch"]) == expected[row["modelId"]],
                "Frozen high-recall selection changed: " + row["modelId"])
        require(row["selectionMode"] == "recall" and row["precision"] == "fp32",
                "Unexpected beach publication selection")
    return result


def feature_sources(inventory_rows, phase1):
    prior = {row["id"]: {"index": position, "row": row}
             for position, row in enumerate(phase1["recordings"])}
    result = {}
    for row in inventory_rows:
        indexed = prior.get(row["id"])
        require(indexed is not None, "Beach source is absent from the frozen feature inventory")
        source = indexed["row"]
        require(source["contentSha256"] == row["contentSha256"]
                and source["sourceGroup"] == row["sourceGroup"]
                and source["environment"] == "beach", "Beach feature identity changed")
        features = source["featureCaches"]
        for name in ("audiovisual", "dino"):
            require(sha(features[name]["path"]) == features[name]["sha256"], f"Changed {name} feature cache")
        result[row["id"]] = indexed
    return result


def mobile_features(rows, checkpoint, cache_dir, device):
    require(sha(checkpoint) == "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f",
            "MobileNet checkpoint changed")
    backbone = load_mobile_backbone(checkpoint, checkpoint_sha256=sha(checkpoint), device=device)
    config = MobileVisualConfig(batch_size=16)
    result = {}
    for row in rows:
        sanitized = {key: row[key] for key in ("id", "video", "contentSha256", "roi", "sourceGroup", "environment")}
        cache, status = extract_recording_cache(sanitized, backbone, config, cache_dir,
            progress=lambda message: print(message, flush=True))
        result[row["id"]] = {"path": str(cache.path.resolve()), "sha256": sha(cache.path), "status": status,
                             "shape": list(cache.tokens.shape)}
    return result, backbone.identity(), asdict(config)


def inference_example(row, source, mobile_reference, family):
    features = {
        "audiovisual": {key: source["featureCaches"]["audiovisual"][key] for key in ("path", "sha256")},
        "dino": {"fp32": {key: source["featureCaches"]["dino"][key] for key in ("path", "sha256")}},
        "mobile": {key: mobile_reference[key] for key in ("path", "sha256")},
    }
    sanitized = {key: row[key] for key in ("id", "sourceGroup", "durationSeconds", "environment")}
    sanitized["featureOrigin"] = "opencv-av104-v3"
    base = inputs.example_from_row(sanitized, features, inference=True)
    example = inputs.attach_visual(base, features, family, "fp32")
    require(not example.truth and not example.ignored and example.valid.all(), "Labels reached beach inference")
    return example


def infer(model_row, fit, example, mobile_reference, source, source_index, recording_index,
          recording_sha256, destination, device):
    family = MODELS[model_row["model"]].family
    output = destination / model_row["modelId"] / (example.id + ".npz")
    receipt_path = output.with_suffix(".json")
    weights = fit / "temporal" / f"weights-{model_row['epoch']}.npz"
    lineage = {
        "model": ledger_reference(LEDGER_COMPARISON_MODELS, model_row["_ledgerIndex"],
                                  hashlib.sha256(encoded(public_model(model_row))).hexdigest(),
                                  modelId=model_row["modelId"]),
        "fit": ledger_reference(LEDGER_COMPARISON_MODELS, model_row["_ledgerIndex"],
                                sha(fit / "fit-result.json"), field="fit-result"),
        "weights": ledger_reference(LEDGER_COMPARISON_MODELS, model_row["_ledgerIndex"],
                                    sha(weights), field=f"weights-epoch-{model_row['epoch']}"),
        "recording": ledger_reference(LEDGER_INVENTORY, recording_index, recording_sha256,
                                      recordingId=example.id),
        "audiovisual": ledger_reference(LEDGER_PHASE1_FEATURES, source_index,
                                        source["featureCaches"]["audiovisual"]["sha256"],
                                        field="featureCaches.audiovisual"),
        "visual": (ledger_reference(LEDGER_PHASE1_FEATURES, source_index,
                                    source["featureCaches"]["dino"]["sha256"],
                                    field="featureCaches.dino")
                   if family == "dino" else ledger_reference(LEDGER_BEACH_ARTIFACTS, recording_index,
                                                               mobile_reference["sha256"], field="mobileFeatures")),
        "labelsUsed": False, "ignoredIntervalsUsed": False,
    }
    if receipt_path.exists() and output.exists():
        receipt = read(receipt_path)
        require(sha(output) == receipt["output"]["sha256"], "Beach inference output changed")
        prior = receipt["lineage"]
        if prior != lineage:
            # One-time migration from the original path-bearing receipt. Hashes and IDs
            # must agree before it is replaced by ledger indexes.
            require(prior.get("model", {}).get("modelId") == model_row["modelId"]
                    and prior.get("fit", {}).get("sha256") == lineage["fit"]["sha256"]
                    and prior.get("weights", {}).get("sha256") == lineage["weights"]["sha256"]
                    and prior.get("audiovisual", {}).get("sha256") == lineage["audiovisual"]["sha256"]
                    and prior.get("visual", {}).get("sha256") == lineage["visual"]["sha256"]
                    and prior.get("labelsUsed") is False and prior.get("ignoredIntervalsUsed") is False,
                    "Beach inference resume differs")
        with np.load(output, allow_pickle=False) as data:
            require(np.array_equal(data["times"], example.times), "Beach inference timeline changed")
            scores = data["scores"].copy()
        receipt = {"lineage": lineage,
                   "output": ledger_reference(LEDGER_BEACH_ARTIFACTS, recording_index, sha(output),
                                              field="inference", modelIndex=model_row["_ledgerIndex"]),
                   "samples": len(example.times)}
        write(receipt_path, require_portable(receipt, "beach inference receipt"))
        return scores, receipt
    model, mean, scale = load_checkpoint(weights, MODELS[model_row["model"]], device)
    scores = predict(model, example, mean, scale, MODELS[model_row["model"]], device)
    require(scores.shape == (len(example.times), 4) and np.isfinite(scores).all(), "Invalid beach inference")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, times=example.times, scores=scores)
    receipt = {"lineage": lineage,
               "output": ledger_reference(LEDGER_BEACH_ARTIFACTS, recording_index, sha(output),
                                          field="inference", modelIndex=model_row["_ledgerIndex"]),
               "samples": len(example.times)}
    write(receipt_path, require_portable(receipt, "beach inference receipt"))
    return scores, receipt


def publish_recording(row, models, predictions, output):
    references = []
    for model in models:
        times, scores, receipt = predictions[model["modelId"]]
        shell = SweepExample(row["id"], row["sourceGroup"], row["durationSeconds"], times,
                             np.ones(len(times), dtype=bool), (), ())
        rallies = [{"start": event.start, "end": event.end} for event in decode(shell, scores, model["decoder"])]
        description = (f"Frozen {model['model'].upper()} high-recall checkpoint, {model['variant']} draw {model['seed']}, "
                       f"epoch {model['epoch']}. Calibration target {model['recallTargetPercent']}% retained-play recall at 2s padding "
                       "on the non-beach selection panel; this beach run is label-blind and the target is not a guarantee for this video. "
                       "Core boundaries remain separate; export padding joins positive gaps strictly under 3s.")
        references.append({"modelId": model["modelId"], "modelLabel": model["modelLabel"], "description": description,
            "rallies": rallies, "exportRallies": rallies, "exportPolicy": "model-predictions", "research": {
                "recommendation": description,
                "signals": {"times": times.tolist(), **{head: scores[:, index].tolist()
                    for index, head in enumerate(("live", "serve", "end", "keep"))}},
                "boundaryFlags": [], "reviewRegions": [],
                "queue": {"budgetFraction": 0, "reviewSeconds": 0, "selectedParentCount": 0},
                "provenance": {**public_model(model), "labelsUsed": False, "ignoredIntervalsUsed": False,
                               "source": receipt["output"], "sourceSha256": receipt["output"]["sha256"],
                               "productionExposure": row.get("productionExposure"), "labelTier": row["tier"]}}})
    recording = {"recordingId": row["id"], "videoFilename": Path(row["video"]).name,
                 "durationSeconds": row["durationSeconds"], "contentSha256": row["contentSha256"],
                 "references": references}
    relative = f"recordings/{row['id']}.json"
    manifest = {"schemaVersion": 1, "kind": "volleycut-labeling-research-references",
                "recordings": [recording]}
    write(output / relative, require_portable(manifest, "beach research reference"))
    return {"id": row["id"], "name": recording["videoFilename"], "file": relative, "tier": row["tier"],
            "modelIds": list(MODEL_IDS)}, {model["modelId"]: len(reference_row["rallies"])
                                           for model, reference_row in zip(models, references, strict=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase1-manifest", type=Path, required=True)
    parser.add_argument("--mobile-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    index_path = args.output / "index.json"
    catalog_path = args.output / "labeling-catalog.json"
    index = read(index_path)
    catalog_before = catalog_path.read_bytes()
    models = selected_models(index)
    inventory_path = args.study / "inventory-v1/inventory-v2.json"
    inventory = read(inventory_path)
    rows = [{**row, "_ledgerIndex": position} for position, row in enumerate(inventory["records"])
            if row["environment"] == "beach"]
    require(len(rows) == 2 and all(row["tier"] == "completed-exact" for row in rows), "Unexpected beach inventory")
    catalog = read(catalog_path)
    catalog_rows = {row["recordingId"]: row for row in catalog["records"]}
    require(all(row["id"] in catalog_rows and catalog_rows[row["id"]]["rallies"] == row["rallies"]
                and catalog_rows[row["id"]]["ignoredIntervals"] == row["ignoredIntervals"] for row in rows),
            "Published human beach labels differ from the frozen inventory")
    phase1 = read(args.phase1_manifest)
    sources = feature_sources(rows, phase1)
    mobile, backbone, mobile_config = mobile_features(rows, args.mobile_checkpoint,
        args.output / "beach-features/mobile", args.device)

    entries, counts, inference_receipts = [], {}, []
    for row in rows:
        predictions = {}
        for model in models:
            fit = (args.study / "randomized-variants-v1/fits" / model["variant"] / model["model"]
                   / f"split-{model['seed']}")
            source = sources[row["id"]]
            example = inference_example(row, source["row"], mobile[row["id"]], MODELS[model["model"]].family)
            scores, receipt = infer(model, fit, example, mobile[row["id"]], source["row"], source["index"],
                                    row["_ledgerIndex"], row["contentSha256"],
                                    args.output / "beach-inference", args.device)
            predictions[model["modelId"]] = (example.times, scores, receipt)
            receipt_path = args.output / "beach-inference" / model["modelId"] / (row["id"] + ".json")
            inference_receipts.append(ledger_reference(LEDGER_BEACH_ARTIFACTS, row["_ledgerIndex"],
                                                       sha(receipt_path), field="inferenceReceipt",
                                                       modelIndex=model["_ledgerIndex"]))
        entry, counts[row["id"]] = publish_recording(row, models, predictions, args.output)
        entries.append(entry)

    by_id = {row["id"]: row for row in index["recordings"]}
    by_id.update({row["id"]: row for row in entries})
    index["recordings"] = list(by_id.values())
    write(index_path, index)
    require(catalog_path.read_bytes() == catalog_before, "Human labeling catalog changed during beach publication")
    audit = {"schemaVersion": 2, "kind": "volleycut-neural-comparison-beach-publication-v2",
             "recordings": [{"recordingId": row["id"], "inventoryIndex": row["_ledgerIndex"],
                              "phase1FeatureIndex": sources[row["id"]]["index"],
                              "comparisonIndex": next(position for position, entry in enumerate(index["recordings"])
                                                      if entry["id"] == row["id"])} for row in rows],
             "models": [{"modelId": model["modelId"], "comparisonModelIndex": model["_ledgerIndex"]}
                        for model in models],
             "predictionCounts": counts, "labelsUsedForInference": False,
             "ignoredIntervalsUsedForInference": False, "humanCatalogUnchangedSha256": sha(catalog_path),
             "sourceLedgers": {"inventory": {"ledger": LEDGER_INVENTORY, "sha256": sha(inventory_path)},
                               "phase1Features": {"ledger": LEDGER_PHASE1_FEATURES,
                                                  "sha256": sha(args.phase1_manifest)}},
             "mobileBackbone": backbone, "mobileConfig": mobile_config,
             "inferenceReceipts": inference_receipts,
             "selectionScope": "Frozen non-beach common-unseen choices; no beach fitting, calibration, or selection"}
    audit["mobileFeatures"] = [{"recordingId": row["id"],
                                **ledger_reference(LEDGER_BEACH_ARTIFACTS, row["_ledgerIndex"],
                                                   mobile[row["id"]]["sha256"], field="mobileFeatures"),
                                "shape": mobile[row["id"]]["shape"]} for row in rows]
    write(args.output / "beach-audit.json", require_portable(audit, "beach publication audit"))
    print(json.dumps({"recordings": counts, "models": list(MODEL_IDS), "audit": "beach-audit.json"}))


if __name__ == "__main__":
    main()
