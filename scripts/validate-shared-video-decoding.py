"""Compare independently decoded and shared-decoder native artifacts, failing closed.

Exact artifact locations are supplied at runtime; no private input defaults.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compare(reference, candidate, require_pixels=False):
    documents = [read(folder / "result.json") for folder in (reference, candidate)]
    assert all(doc["status"] == "complete" and doc["cacheMode"] == "bypass" for doc in documents)
    a, b = [doc["results"][0] for doc in documents]
    checks = {}
    for key in ("visualExtractorVersion", "audioExtractorVersion", "cacheMode"):
        checks[key] = documents[0][key] == documents[1][key]
    contracts = [read(folder / "input-contract.json") for folder in (reference, candidate)]
    checks["frozenGraphHashes"] = contracts[0]["hashes"] == contracts[1]["hashes"]
    for key in ("roi", "sampleRows", "rallyCount", "rallies", "productionRallies",
                "servingSideReady", "sideSwitchReady"):
        checks[key] = a[key] == b[key]
    for key in ("times", "decoder", "featureRows", "featureDimension", "precision"):
        checks["neural/" + key] = a["neural"][key] == b["neural"][key]
    va, vb = a["neural"]["video"], b["neural"]["video"]
    for key in ("sampleCount", "sampleTimestamps", "quality", "decodedColorProfiles", "frameSelection",
                "seconds", "startSeconds", "sampleFps", "decoder", "hardwareDecoder"):
        checks["video/" + key] = va[key] == vb[key]
    if require_pixels:
        checks["video/pixelHashes"] = (len(va.get("pixelHashes", [])) == va["sampleCount"]
            and va["pixelHashes"] == vb.get("pixelHashes"))
    # Specialist reports include performance fields; compare their actual inputs
    # and decisions recursively while excluding only known timing fields.
    def without_timings(value):
        if isinstance(value, dict):
            return {k: without_timings(v) for k, v in value.items()
                    if not k.endswith("Ms") and k not in ("profileMs", "performanceMilliseconds")}
        if isinstance(value, list):
            return [without_timings(v) for v in value]
        return value
    for key in ("servingSide", "sideSwitch"):
        checks[key] = without_timings(a[key]) == without_timings(b[key])
    tensors = {}
    for suffix in ("tokens", "features", "probabilities"):
        paths = [folder / (row["id"] + "-" + suffix + ".f32")
                 for folder, row in zip((reference, candidate), (a, b))]
        raw = [path.read_bytes() for path in paths]
        values = [np.frombuffer(data, dtype="<f4") for data in raw]
        same_shape = values[0].shape == values[1].shape
        tensors[suffix] = dict(bytes=[len(data) for data in raw],
            sha256=[hashlib.sha256(data).hexdigest() for data in raw],
            exact=raw[0] == raw[1],
            maxAbsoluteError=float(np.max(np.abs(values[0] - values[1]))) if same_shape else None,
            finite=all(bool(np.isfinite(v).all()) for v in values))
        checks["tensor/" + suffix] = tensors[suffix]["exact"] and tensors[suffix]["finite"]
    return dict(schema="shared-video-parity-v1", exact=all(checks.values()), checks=checks,
                tensors=tensors, pixelsRequired=require_pixels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pixels", action="store_true")
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.require_pixels)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exact": result["exact"], "failed": [k for k,v in result["checks"].items() if not v]}))
    if not result["exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
