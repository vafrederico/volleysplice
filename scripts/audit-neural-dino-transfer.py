#!/usr/bin/env python3
"""Independently verify DINO source association, recipe and AV time alignment."""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(private_value('private-reference-0084'))
PARENT_SHA = "48fe4fb5bcb74b2ee998562f61c3a7e8df43e921f8882e1b44fe1c12db5d451f"
PLAN_SHA = "fd36fa7d4652660d6bac1a09c3b516be9938a7f30f5ad55588707597ea1b2f97"
CONFIG_SHA = "91698926e4560f13e9ef7e991662060b096a13a4097856cc5083fd1259f257a9"
BACKBONE = {"modelName": "dinov2_vits14", "repositoryCommit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
            "checkpointSha256": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
            "embeddingDimension": 384, "tokenCount": 10, "patchSize": 14}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical_path(path):
    path = str(Path(path))
    return private_value('private-reference-0060')+path[len(private_value('private-reference-0087')):] if path.startswith(private_value('private-reference-0087')) else path


def array_sha(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def validate_record(record, row, plan):
    require(record["recordingId"] == row["id"] and record["sourceGroup"] == row["sourceGroup"], "record identity/group mismatch")
    require(sha(record["dinoPath"]) == record["dinoSha256"], "DINO cache hash mismatch")
    require(sha(record["metadataPath"]) == record["metadataSha256"], "metadata sidecar hash mismatch")
    sidecar = read(record["metadataPath"])
    require(sidecar["recordingId"] == row["id"] and sidecar["cache"]["path"] == record["dinoPath"]
            and sidecar["cache"]["sha256"] == record["dinoSha256"], "sidecar association differs")
    require(sidecar["extractionPlan"]["sha256"] == PLAN_SHA, "sidecar references another extraction plan")
    source = sidecar["sourceVideoVerified"]
    require(source == record["sourceVideoVerified"], "source hash audit differs between sidecar and wrapper")
    require(canonical_path(source["path"]) == canonical_path(row["video"])
            and source["sha256"] == row["contentSha256"], "fully hashed video identity mismatch")
    stat = Path(row["video"]).stat()
    require(stat.st_size == source["sizeBytes"] and stat.st_mtime_ns == source["mtimeNs"], "video changed after full hash/extraction")
    require(record["rawSourceContentSha256"] == row.get("sourceContentSha256"), "raw lineage binding differs")
    with np.load(record["dinoPath"], allow_pickle=False) as data:
        timestamps = data["timestamps"]
        tokens = data["tokens"]
        md = json.loads(str(data["metadata_json"].item()))
    require(md == sidecar["cacheMetadata"], "embedded metadata differs from sidecar")
    require(md["recordingId"] == row["id"] and md["recordingContentSha256"] == row["contentSha256"]
            and canonical_path(md["sourceVideoPath"]) == canonical_path(row["video"]), "embedded source binding differs")
    roi = [float(row["roi"][key]) for key in ("x", "y", "width", "height")] if row.get("roi") else None
    require(md["roi"] == roi, "declared ROI differs")
    require({key: md["backbone"][key] for key in BACKBONE} == BACKBONE, "semantic backbone differs")
    require(md["extractorConfigSha256"] == CONFIG_SHA and md["preprocessing"] == plan["extractorConfig"], "extractor recipe differs")
    config_hash = hashlib.sha256(json.dumps(md["preprocessing"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(config_hash == CONFIG_SHA, "preprocessing dictionary does not hash to pinned recipe")
    require(md["runtime"] == plan["historicalRuntime"] and md["completed"] is True and md["labelsUsed"] is False,
            "runtime or completion/label policy differs")
    require(timestamps.dtype == np.float64 and timestamps.ndim == 1 and len(timestamps)
            and np.isfinite(timestamps).all() and np.all(np.diff(timestamps) > 0), "invalid DINO times")
    require(tokens.dtype == np.float16 and tokens.shape == (len(timestamps), 10, 384) and np.isfinite(tokens).all(), "invalid DINO tokens")
    grid = np.arange(max(1, int(np.ceil(md["video"]["duration"]*4-1e-9))), dtype=np.float64)/4
    grid = grid[grid < md["video"]["duration"]+1e-9]
    require(np.array_equal(timestamps, grid), "DINO timestamps differ from exact 4Hz contract")
    require(md["analysisTimestamps"]["count"] == len(timestamps)
            and md["analysisTimestamps"]["sha256"] == array_sha(timestamps, "<f8"), "embedded timestamp digest differs")
    av = row["featureCaches"]["audiovisual"]
    require(record["audiovisualPath"] == av["path"] and record["audiovisualSha256"] == av["sha256"]
            and sha(av["path"]) == av["sha256"], "AV cache association changed")
    with np.load(av["path"], allow_pickle=False) as data:
        times = data["times"].astype(np.float64)
        require(data["values"].shape == (len(times), 104) and np.isfinite(data["values"]).all(), "invalid AV values")
        require(tuple(str(x) for x in data["names"]) == tuple(av["names"]), "AV feature schema changed")
    require(np.isfinite(times).all() and np.all(np.diff(times) > 0), "invalid AV timestamps")
    right = np.minimum(np.searchsorted(timestamps, times), len(timestamps)-1)
    left = np.maximum(right-1, 0)
    left_errors, right_errors = np.abs(times-timestamps[left]), np.abs(times-timestamps[right])
    indexes = np.where(left_errors <= right_errors, left, right)
    errors = np.abs(times-timestamps[indexes])
    require(float(errors.max()) <= .125+1e-8, "time alignment exceeds tolerance")
    join = record["nearestAlignment"]
    require(join == sidecar["nearestAlignment"], "sidecar alignment differs")
    expected = {"avTimesSha256": array_sha(times, "<f8"), "dinoTimesSha256": array_sha(timestamps, "<f8"),
                "nearestIndexesSha256": array_sha(indexes, "<i8"), "avTicks": len(times), "dinoTicks": len(timestamps),
                "maximumErrorSeconds": float(errors.max()), "meanErrorSeconds": float(errors.mean()),
                "tieCount": int(((left != right) & (left_errors == right_errors)).sum()), "toleranceSeconds": .125}
    require(all(join[key] == value for key, value in expected.items()), "alignment evidence does not recompute")
    return {"recordingId": row["id"], "sourceGroup": row["sourceGroup"], "dinoSha256": record["dinoSha256"],
            "metadataSha256": record["metadataSha256"], "sourceContentSha256": source["sha256"],
            "roi": roi, "storedSourcePath": md["sourceVideoPath"], "canonicalSourcePath": canonical_path(row["video"]),
            "shape": list(tokens.shape), "alignment": expected, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    require(sha(args.root/"extraction-plan.json") == PLAN_SHA, "extraction plan changed")
    plan = read(args.root/"extraction-plan.json")
    require(sha(plan["expandedManifestPath"]) == PARENT_SHA == plan["expandedManifestSha256"], "parent dataset changed")
    parent = read(plan["expandedManifestPath"])
    rows = {row["id"]: (tier, row) for tier, key in (("exact", "exactRows"), ("draft", "draftRows"), ("coverage", "coverageRows")) for row in parent[key]}
    wrapper_path = args.root/"dino-manifest.json"
    if args.progress:
        records = [read(p) for p in sorted((args.root/"record-audits").glob("*.json"))]
    else:
        wrapper = read(wrapper_path)
        require(wrapper["expandedManifestPath"] == plan["expandedManifestPath"]
                and wrapper["expandedManifestSha256"] == PARENT_SHA, "wrapper parent differs")
        records = wrapper["records"]
        require(len(records) == 18 and {r["recordingId"] for r in records} == set(rows), "final source population differs")
    audits = []
    for record in records:
        tier, row = rows[record["recordingId"]]
        require(record["tier"] == tier and row["sourceGroup"] != private_value('source-group-008')
                and row["environment"] in ("grass", "indoor"), "excluded/mismatched tier source")
        audits.append(validate_record(record, row, plan))
    result = {"kind": "independent-dino-association-audit-v1", "createdAt": datetime.now(timezone.utc).isoformat(),
              "passed": True, "complete": not args.progress, "recordings": len(audits), "audits": audits,
              "manifestSha256": sha(wrapper_path) if not args.progress else None, "extractionPlanSha256": PLAN_SHA,
              "expandedManifestSha256": PARENT_SHA, "scriptSha256": sha(Path(__file__)),
              "sourceBytes": "Extraction workers fully hashed each source. This independent pass checks those hashes against immutable source identities and verifies unchanged current file size/mtime; it does not read every video byte a second time.",
              "sourcePathAlias": private_value('private-reference-0086')}
    if not args.progress:
        output = args.root/"dino-association-audit-v1.json"
        with output.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")
        result["artifact"] = {"path": str(output), "sha256": sha(output)}
    print(json.dumps({key: value for key, value in result.items() if key != "audits"}, indent=2))


if __name__ == "__main__":
    main()
