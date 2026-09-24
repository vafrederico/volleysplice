#!/usr/bin/env python3
"""Audit the frozen detector study and exact SSD adapter vs official inference."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    study = args.study.resolve()
    if not str(study).startswith(private_value('private-reference-0060')):
        raise ValueError("NAS study required")
    sys.dont_write_bytecode = True
    for name in ("TMPDIR", "TMP", "TEMP", "TORCH_HOME", "XDG_CACHE_HOME"):
        path = study / "runtime" / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    sys.path.insert(0, str(study / "python-dependencies"))
    import cv2
    import numpy as np
    import torch
    import torchvision
    from analysis.person_alternative_detectors import SSDliteDetector, sanitize_boxes
    spec = importlib.util.spec_from_file_location("qualifier", REPO / "scripts/qualify-person-alternatives.py")
    qualifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualifier)
    protocol = qualifier.verify_protocol(argparse.Namespace(output=study))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    cv2.setNumThreads(2)
    report = json.loads((study / "results/report.json").read_text())
    assert qualifier.digest(protocol["combinedProtocol"]["path"]) == "1b2c764fde6553e0f19ac3bf6379aa02b9a12f1a19dec86fe61ab934a888eea4"
    receipt = json.loads((study / "asset-receipt.json").read_text())
    tvroot = Path(torchvision.__file__).parent
    for ref in receipt["torchvisionSources"]:
        relative = Path(ref["path"]).relative_to(study / "assets/torchvision-runtime-sources")
        assert qualifier.digest(tvroot / relative) == ref["sha256"], str(relative)
    records, summaries, fixed_total, frame_total, references = [], {}, 0, 0, []
    for reference in report["records"]:
        assert qualifier.digest(reference["path"]) == reference["sha256"]
        record = json.loads(Path(reference["path"]).read_text())
        rows = record["frames"]
        fixed = [f for f in rows if f["fixedReviewFrame"]]
        sequence = [f for f in rows if f["sequenceFrame"]]
        assert len(fixed) == 8
        assert len(sequence) == (60 if record["id"] in qualifier.SEQUENCE_IDS else 0)
        assert len({f["frameOrdinal"] for f in rows}) == len(rows)
        fixed_total += len(fixed)
        frame_total += len(rows)
        summaries[record["id"]] = {}
        for frame in rows:
            if frame.get("sourceFrame"):
                ref = frame["sourceFrame"]
                assert qualifier.digest(ref["path"]) == ref["sha256"]
                pixels = cv2.imread(ref["path"])
                assert qualifier.hashlib.sha256(pixels.tobytes()).hexdigest() == frame["roiPixelsSha256"]
            for model, result in frame["models"].items():
                ref = result["overlay"]
                assert qualifier.digest(ref["path"]) == ref["sha256"]
                assert result["uncappedPersonCount"] >= len(result["detections"])
        for model in report["models"]:
            counts = [f["models"][model]["geometry"]["count"] for f in rows]
            seqcounts = [f["models"][model]["geometry"]["count"] for f in sequence]
            summaries[record["id"]][model] = {
                "fixedCounts": [f["models"][model]["geometry"]["count"] for f in fixed],
                "fixedFramesWithContainmentWarning": sum(bool(f["models"][model]["geometry"]["highContainmentPairs"]) for f in fixed),
                "allSelectedCountRange": [min(counts), max(counts)],
                "allSelectedZeroCountFrames": counts.count(0),
                "allSelectedCappedFrames": sum(f["models"][model]["uncappedPersonCount"] > 24 for f in rows),
                "sequenceAdjacentCountChanges": sum(a != b for a, b in zip(seqcounts, seqcounts[1:])),
                "sequenceCountRange": [min(seqcounts), max(seqcounts)] if seqcounts else None,
                "sequenceFramesWithContainmentWarning": sum(bool(f["models"][model]["geometry"]["highContainmentPairs"]) for f in sequence),
                "warningInterpretation": "Geometric overlap/count changes are not labeled duplicates, identity switches, or detector error rates."}
            if sequence:
                thumbs = []
                for f in sequence:
                    image = cv2.imread(f["models"][model]["overlay"]["path"])
                    image = cv2.resize(image, (320, 180))
                    cv2.rectangle(image, (0, 0), (320, 18), (0, 0, 0), -1)
                    cv2.putText(image, f"t={f['ptsSeconds']:.1f} n={f['models'][model]['geometry']['count']}",
                                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                    thumbs.append(image)
                sheet = np.vstack([np.hstack(thumbs[i:i + 5]) for i in range(0, 60, 5)])
                path = study / "results" / f"{record['id']}-{model}-sequence-sheet.jpg"
                assert not path.exists()
                assert cv2.imwrite(str(path), sheet)
                references.append(qualifier.identity(path))
        records.append(record)
    assert fixed_total == 32 and frame_total == 142
    # Comparing the separated timing adapter with the untouched public SSD
    # forward path catches transform/decode/restore discrepancies.
    adapter = SSDliteDetector(study / "assets")
    parity = []
    with torch.inference_mode():
        for record in records:
            frame = [f for f in record["frames"] if f["fixedReviewFrame"] and f["requestedSeconds"] == 20.5][0]
            pixels = cv2.imread(frame["sourceFrame"]["path"])
            actual = adapter.detect(pixels)
            tensor = torch.from_numpy(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)
            expected = adapter.model([tensor])[0]
            mask = expected["labels"] == 1
            boxes, uncapped = sanitize_boxes(expected["boxes"][mask].numpy(), expected["scores"][mask].numpy(),
                                              pixels.shape[1], pixels.shape[0])
            assert actual["detections"] == boxes and actual["uncappedPersonCount"] == uncapped
            assert actual["detections"] == frame["models"][adapter.name]["detections"]
            parity.append({"record": record["id"], "time": frame["ptsSeconds"], "publicForwardExactMatch": True,
                           "savedRunExactMatch": True, "detections": len(boxes)})
    payload = {"kind": "person-alternatives-qualification-audit-v1", "passed": True,
        "protocol": qualifier.identity(study / "protocol.json"), "report": qualifier.identity(study / "results/report.json"),
        "auditSource": qualifier.identity(Path(__file__)), "fixedFrames": fixed_total, "distinctFrames": frame_total,
        "runtimeTorchvisionSourcesMatchArchive": True, "sourcePngPixelsMatch": True,
        "ssdPublicForwardParity": parity, "summaries": summaries, "sequenceSheets": references,
        "detectorPrecisionRecallMeasured": False, "rallyTrainingExecuted": False}
    qualifier.write_new(study / "audit.json", payload)
    print(json.dumps({"passed": True, "audit": qualifier.identity(study / "audit.json"), "summaries": summaries}))


if __name__ == "__main__":
    main()
