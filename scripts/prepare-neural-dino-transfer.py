#!/usr/bin/env python3
"""Bind and complete frozen DINOv2 caches for the existing 18-row development set.

Old manifests, videos and caches are read-only. New extraction calls the existing
pinned extractor unchanged. At most two independent processes own disjoint new
cache paths. A final wrapper manifest is published only after all 18 pass audit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
NAS = Path(private_value('private-reference-0057'))
PARENT = NAS/"2026-09-19-expanded/manifest.json"
PARENT_SHA = "48fe4fb5bcb74b2ee998562f61c3a7e8df43e921f8882e1b44fe1c12db5d451f"
OUTPUT = NAS/"2026-09-19-short-boost-transfer"
COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
WEIGHTS_SHA = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
ASSETS = Path(private_value('private-reference-0058'))
MODEL_REPO = ASSETS/f"dinov2-{COMMIT}"
CHECKPOINT = ASSETS/"dinov2_vits14_pretrain.pth"
CONFIG_SHA = "91698926e4560f13e9ef7e991662060b096a13a4097856cc5083fd1259f257a9"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha256(path), "sizeBytes": path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def semantic_backbone(value):
    return {key: value[key] for key in ("modelName", "repositoryCommit", "checkpointSha256",
                                       "embeddingDimension", "tokenCount", "patchSize")}


def alignment(av_times, dino_times):
    import numpy as np
    require(av_times.ndim == dino_times.ndim == 1 and len(av_times) and len(dino_times)
            and np.isfinite(av_times).all() and np.isfinite(dino_times).all()
            and np.all(np.diff(av_times) > 0) and np.all(np.diff(dino_times) > 0), "invalid timelines")
    right = np.clip(np.searchsorted(dino_times, av_times), 0, len(dino_times)-1)
    left = np.maximum(0, right-1)
    ld, rd = np.abs(dino_times[left]-av_times), np.abs(dino_times[right]-av_times)
    nearest = np.where(ld <= rd, left, right).astype(np.int64)
    errors = np.abs(dino_times[nearest]-av_times)
    require(float(errors.max()) <= .125+1e-8, "DINO/AV nearest alignment exceeds half tick")
    return nearest, {"rule": "nearest DINO timestamp to each actual AV tick; earlier timestamp wins ties",
                     "toleranceSeconds": .125, "maximumErrorSeconds": float(errors.max()),
                     "meanErrorSeconds": float(errors.mean()), "avTicks": len(av_times),
                     "dinoTicks": len(dino_times), "tieCount": int(((left != right) & (ld == rd)).sum()),
                     "avTimesSha256": hashlib.sha256(av_times.astype("<f8").tobytes()).hexdigest(),
                     "dinoTimesSha256": hashlib.sha256(dino_times.astype("<f8").tobytes()).hexdigest(),
                     "nearestIndexesSha256": hashlib.sha256(nearest.astype("<i8").tobytes()).hexdigest()}


def frozen_sources():
    return {"prepare-neural-dino-transfer.py": identity(Path(__file__).resolve()),
            "dinov2_embeddings.py": identity(REPO/"analysis/dinov2_embeddings.py"),
            "features.py": identity(REPO/"analysis/features.py"),
            "config.py": identity(REPO/"analysis/config.py")}


def check_sources(plan):
    require(sha256(PARENT) == plan["expandedManifestSha256"] == PARENT_SHA, "expanded manifest changed")
    for entry in plan["sourceCode"].values():
        require(sha256(entry["path"]) == entry["sha256"], "pinned extraction/preparation code changed")
    require(sha256(CHECKPOINT) == WEIGHTS_SHA, "pinned checkpoint changed")
    actual_commit = subprocess.run(["git", "-C", str(MODEL_REPO), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    require(actual_commit == COMMIT, "DINO repository commit changed")
    dirty = subprocess.run(["git", "-C", str(MODEL_REPO), "status", "--porcelain", "--untracked-files=no"], check=True, capture_output=True, text=True).stdout
    require(not dirty.strip(), "DINO tracked source changed")


def load_backbone():
    import torch
    from analysis.dinov2_embeddings import load_pinned_dinov2
    torch.set_num_threads(2)
    require(importlib.util.find_spec("xformers") is None, "historical fallback backend requires xformers absent")
    return load_pinned_dinov2(MODEL_REPO, repository_commit=COMMIT, checkpoint=CHECKPOINT,
                            checkpoint_sha256=WEIGHTS_SHA, device="cuda")


def recording(item):
    return SimpleNamespace(id=item["recordingId"], video=Path(item["sourceVideoPath"]),
                           content_sha256=item["recordingContentSha256"],
                           roi=tuple(item["roi"]) if item["roi"] is not None else None)


def validate_cache(item, path, plan):
    import numpy as np
    from analysis.dinov2_embeddings import load_dino_cache, sample_timestamps
    cache = load_dino_cache(path, recording_id=item["recordingId"],
                            recording_content_sha256=item["recordingContentSha256"], expected_config_sha256=CONFIG_SHA)
    md = cache.metadata
    require(semantic_backbone(md["backbone"]) == plan["semanticBackbone"], "DINO semantic backbone differs")
    require(md["preprocessing"] == plan["extractorConfig"] and md["roi"] == item["roi"]
            and md.get("labelsUsed") is False and md.get("completed") is True, "cache recipe/ROI/label contract differs")
    require(md["runtime"] == plan["historicalRuntime"], "DINO runtime differs from inherited caches")
    require(np.array_equal(cache.timestamps, sample_timestamps(md["video"]["duration"], 4.)), "DINO timeline is not exact 4Hz grid")
    av = item["audiovisual"]
    require(sha256(av["path"]) == av["sha256"], "current audiovisual cache changed")
    with np.load(av["path"], allow_pickle=False) as n:
        av_times = n["times"].astype(np.float64)
        require(n["values"].shape == (len(av_times), 104) and np.isfinite(n["values"]).all(), "invalid AV values")
        require(tuple(str(x) for x in n["names"]) == tuple(av["names"]), "AV feature names changed")
    nearest, join = alignment(av_times, cache.timestamps)
    with np.load(path, allow_pickle=False) as n:
        require(n["tokens"].dtype == np.float16 and n["timestamps"].dtype == np.float64, "DINO stored dtypes differ")
    return cache, nearest, join


def probe(backbone, item, count=32, reference=None):
    import cv2
    import numpy as np
    capture = cv2.VideoCapture(item["sourceVideoPath"])
    require(capture.isOpened(), "cannot open probe video")
    fps, frame_count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, started = [], time.perf_counter()
    try:
        for index in range(count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_count-1, round(index/4*fps)))
            ok, frame = capture.read()
            require(ok and frame is not None and frame.size, "probe frame could not decode")
            frames.append(frame)
    finally:
        capture.release()
    decode_seconds = time.perf_counter()-started
    started = time.perf_counter()
    roi = tuple(item["roi"]) if item["roi"] is not None else None
    tokens = np.concatenate([backbone.embed_frames(frames[i:i+8], (roi,)*len(frames[i:i+8]), input_size=336)
                             for i in range(0, len(frames), 8)])
    backbone.torch.cuda.synchronize()
    embed_seconds = time.perf_counter()-started
    result = {"recordingId": item["recordingId"], "samples": count, "frameShape": list(frames[0].shape),
              "decodeSeconds": decode_seconds, "embeddingSeconds": embed_seconds,
              "samplesPerSecond": count/(decode_seconds+embed_seconds),
              "peakAllocatedCudaBytes": backbone.torch.cuda.max_memory_allocated()}
    if reference:
        with np.load(reference, allow_pickle=False) as n:
            old = n["tokens"][:count]
        current = tokens.astype(np.float16)
        result["reproductionBitExact"] = current.tobytes() == old.tobytes()
        result["maximumStoredTokenDifference"] = float(np.abs(current.astype(np.float32)-old.astype(np.float32)).max())
        require(result["reproductionBitExact"], f"fresh pinned embedding differs from inherited cache: {item['recordingId']}")
    return result


def prepare(output):
    import cv2
    import numpy as np
    import torch
    from analysis.dinov2_embeddings import DinoExtractorConfig
    path = output/"extraction-plan.json"
    if path.exists():
        plan = read(path)
        check_sources(plan)
        return plan
    require(sha256(PARENT) == PARENT_SHA, "wrong expanded dataset revision")
    manifest = read(PARENT)
    rows = [(tier, row) for tier, key in (("exact", "exactRows"), ("draft", "draftRows"), ("coverage", "coverageRows"))
            for row in manifest[key]]
    require([sum(t == tier for t, _ in rows) for tier in ("exact", "draft", "coverage")] == [8, 3, 7], "wrong cohort population")
    require(all(row["environment"] in ("grass", "indoor") and row["sourceGroup"] != private_value('source-group-008')
                and row["split"] != "test" and row["consent"].get("train") is True for _, row in rows), "excluded source")
    config = DinoExtractorConfig()
    require(config.config_sha256 == CONFIG_SHA, "pinned extractor config changed")
    first_dino = next(row["featureCaches"]["dino"] for _, row in rows if row["featureCaches"].get("dino"))
    first_md = first_dino["metadata"]
    historical_runtime = first_md["runtime"]
    runtime = {"device": "cuda", "numpy": np.__version__, "python": platform.python_version(), "torch": torch.__version__}
    require(runtime == historical_runtime, "current runtime differs from inherited caches")
    plan = {"schemaVersion": 1, "kind": "same18-pinned-dino-extraction-plan", "createdAt": datetime.now(timezone.utc).isoformat(),
            "expandedManifestPath": str(PARENT), "expandedManifestSha256": PARENT_SHA,
            "sourceCode": frozen_sources(), "modelRepositoryCommit": COMMIT, "checkpoint": identity(CHECKPOINT),
            "modelRepositoryPath": str(MODEL_REPO), "semanticBackbone": semantic_backbone(first_md["backbone"]),
            "extractorConfig": config.to_dict(), "extractorConfigSha256": CONFIG_SHA, "historicalRuntime": historical_runtime,
            "environment": {**runtime, "opencv": cv2.__version__, "gpu": torch.cuda.get_device_name()},
            "labelsUsedForExtraction": False, "maximumWorkers": 2,
            "backbonePathRelocation": "Historical /tmp checkout and checkpoint no longer exist. Identical official commit and checkpoint bytes restored at durable local paths; filesystem locations are provenance, not a changed feature recipe.",
            "records": []}
    check_sources(plan)
    seen_hashes = set()
    for tier, row in rows:
        hashes = {row[key] for key in ("contentSha256", "sourceContentSha256") if row.get(key)}
        require(not seen_hashes.intersection(hashes), "duplicate source lineage")
        seen_hashes.update(hashes)
        for field in ("labelSource", "normalizationProvenance", "referenceSource", "feedbackSource", "consentProvenance"):
            reference = row.get(field, {})
            if reference.get("path") and reference.get("sha256"):
                require(sha256(reference["path"]) == reference["sha256"], f"source provenance changed: {row['id']} {field}")
        av = row["featureCaches"]["audiovisual"]
        require(sha256(av["path"]) == av["sha256"], "AV input changed")
        item = {"recordingId": row["id"], "tier": tier, "sourceGroup": row["sourceGroup"],
                "sourceVideoPath": row["video"], "recordingContentSha256": row["contentSha256"],
                "rawSourceContentSha256": row.get("sourceContentSha256"), "sourceSizeBytes": Path(row["video"]).stat().st_size,
                "durationSeconds": row["durationSeconds"], "audiovisual": av,
                "roi": [float(row["roi"][k]) for k in ("x", "y", "width", "height")] if row.get("roi") else None,
                "existingDino": row["featureCaches"].get("dino")}
        if item["existingDino"]:
            require(sha256(item["existingDino"]["path"]) == item["existingDino"]["sha256"], "inherited DINO cache changed")
            _, _, join = validate_cache(item, Path(item["existingDino"]["path"]), plan)
            item["existingNearestAlignment"] = join
        plan["records"].append(item)
    require(sum(bool(item["existingDino"]) for item in plan["records"]) == 6, "expected six reusable caches")
    backbone = load_backbone()
    plan["freshBackboneIdentity"] = backbone.identity()
    plan["reproductionProbes"] = [probe(backbone, item, count=8, reference=item["existingDino"]["path"])
                                  for item in plan["records"] if item["existingDino"]]
    missing = [item for item in plan["records"] if not item["existingDino"]]
    proxy = next(item for item in missing if item["tier"] != "coverage")
    raw = next(item for item in missing if item["tier"] == "coverage")
    plan["throughputProbes"] = [probe(backbone, proxy), probe(backbone, raw)]
    rates = {"proxy": plan["throughputProbes"][0]["samplesPerSecond"], "raw": plan["throughputProbes"][1]["samplesPerSecond"]}
    plan["missingCacheCount"] = len(missing)
    plan["missingApproximateTicks"] = sum(int(np.ceil(item["durationSeconds"]*4)) for item in missing)
    plan["estimatedSerialSecondsExcludingHashAndCompression"] = sum(item["durationSeconds"]*4/rates["raw" if item["tier"] == "coverage" else "proxy"] for item in missing)
    output.mkdir(parents=True, exist_ok=True)
    snapshots = output/"extraction-sources"
    snapshots.mkdir(exist_ok=False)
    for name, entry in plan["sourceCode"].items():
        with (snapshots/name).open("xb") as f:
            f.write(Path(entry["path"]).read_bytes())
    # Preserve the public pinned implementation and weights as immutable artifacts.
    assets = output/"extractor-assets"
    assets.mkdir(exist_ok=False)
    archive = assets/f"dinov2-{COMMIT}.tar"
    with archive.open("xb") as f:
        subprocess.run(["git", "-C", str(MODEL_REPO), "archive", COMMIT], stdout=f, check=True)
    checkpoint_copy = assets/CHECKPOINT.name
    with checkpoint_copy.open("xb") as dest, CHECKPOINT.open("rb") as src:
        while block := src.read(1024*1024):
            dest.write(block)
    require(sha256(checkpoint_copy) == WEIGHTS_SHA, "checkpoint snapshot differs")
    plan["archivedAssets"] = [identity(archive), identity(checkpoint_copy)]
    write_new(path, plan)
    print(json.dumps({"prepared": str(path), "sha256": sha256(path), "reused": 6, "missing": 12,
                      "missingTicks": plan["missingApproximateTicks"],
                      "estimatedSerialMinutes": plan["estimatedSerialSecondsExcludingHashAndCompression"]/60}), flush=True)
    return plan


def worker(output, identifier):
    from analysis.dinov2_embeddings import DinoExtractorConfig, extract_recording_cache
    plan = read(output/"extraction-plan.json")
    check_sources(plan)
    item = next(item for item in plan["records"] if item["recordingId"] == identifier)
    completion = output/"record-audits"/f"{identifier}.json"
    require(not completion.exists(), "record already audited")
    started = time.perf_counter()
    print(f"VERIFY SOURCE {identifier}", flush=True)
    source = identity(item["sourceVideoPath"])
    source["mtimeNs"] = Path(item["sourceVideoPath"]).stat().st_mtime_ns
    require(source["sha256"] == item["recordingContentSha256"] and source["sizeBytes"] == item["sourceSizeBytes"], "source video bytes changed")
    if item["existingDino"]:
        path = Path(item["existingDino"]["path"])
        require(sha256(path) == item["existingDino"]["sha256"], "inherited cache bytes changed")
        status = "reused-inherited"
    else:
        backbone = load_backbone()
        cache, status = extract_recording_cache(recording(item), backbone, DinoExtractorConfig(), output/"dino-caches",
                                               progress=lambda message: print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True))
        path = cache.path
    cache, nearest, join = validate_cache(item, path, plan)
    final_stat = Path(item["sourceVideoPath"]).stat()
    require(final_stat.st_size == source["sizeBytes"] and final_stat.st_mtime_ns == source["mtimeNs"],
            "source size or modification time changed during extraction")
    check_sources(plan)
    metadata_path = output/"dino-metadata"/f"{identifier}.json"
    write_new(metadata_path, {"recordingId": identifier, "extractionPlan": identity(output/"extraction-plan.json"),
                              "sourceVideoVerified": source, "cache": identity(path), "status": status,
                              "cacheMetadata": cache.metadata, "nearestAlignment": join})
    report = {"recordingId": identifier, "tier": item["tier"], "sourceGroup": item["sourceGroup"],
              "dinoPath": str(path), "dinoSha256": sha256(path), "metadataPath": str(metadata_path),
              "metadataSha256": sha256(metadata_path), "audiovisualPath": item["audiovisual"]["path"],
              "audiovisualSha256": item["audiovisual"]["sha256"], "nearestAlignment": join,
              "shape": list(cache.tokens.shape), "storedDtype": "float16", "sourceVideoVerified": source,
              "rawSourceContentSha256": item["rawSourceContentSha256"], "status": status,
              "wallSeconds": time.perf_counter()-started, "passed": True}
    write_new(completion, report)
    print(f"COMPLETE {identifier} {status} {report['wallSeconds']:.1f}s", flush=True)


def finalize(output):
    plan = read(output/"extraction-plan.json")
    check_sources(plan)
    records = []
    for item in plan["records"]:
        report = read(output/"record-audits"/f"{item['recordingId']}.json")
        require(report["passed"] and report["recordingId"] == item["recordingId"], "incomplete record audit")
        require(sha256(report["dinoPath"]) == report["dinoSha256"] and sha256(report["metadataPath"]) == report["metadataSha256"], "published cache or metadata changed")
        _, _, join = validate_cache(item, Path(report["dinoPath"]), plan)
        require(join == report["nearestAlignment"], "final timeline audit differs")
        records.append(report)
    wrapper = {"schemaVersion": 1, "kind": "same18-frozen-dino-transfer-inputs", "createdAt": datetime.now(timezone.utc).isoformat(),
               "expandedManifestPath": str(PARENT), "expandedManifestSha256": PARENT_SHA,
               "extractionPlan": identity(output/"extraction-plan.json"), "extractorConfigSha256": CONFIG_SHA,
               "semanticBackbone": plan["semanticBackbone"], "sourceCode": plan["sourceCode"],
               "records": records, "labelsAndIgnoredIntervals": "inherit unchanged from hash-bound expanded manifest",
               "protectedTestOpened": False, "beachIncluded": False, "embeddingLabelsUsed": False}
    write_new(output/"dino-manifest.json", wrapper)
    audit = {"schemaVersion": 1, "passed": True, "manifest": identity(output/"dino-manifest.json"),
             "expandedManifest": identity(PARENT), "records": len(records), "groups": len({r["sourceGroup"] for r in records}),
             "tierCounts": {tier: sum(r["tier"] == tier for r in records) for tier in ("exact", "draft", "coverage")},
             "reused": sum(r["status"] == "reused-inherited" for r in records),
             "newlyGenerated": sum(r["status"] != "reused-inherited" for r in records),
             "sourceVideosFullyHashed": len(records), "dinoCachesFullyHashed": len(records), "audiovisualCachesFullyHashed": len(records),
             "maximumAlignmentErrorSeconds": max(r["nearestAlignment"]["maximumErrorSeconds"] for r in records),
             "dinoTicks": sum(r["shape"][0] for r in records), "avTicks": sum(r["nearestAlignment"]["avTicks"] for r in records),
             "reproductionProbes": plan["reproductionProbes"], "recordAudits": [identity(output/"record-audits"/f"{r['recordingId']}.json") for r in records],
             "noHistoricalArtifactModified": True, "extractorCodeUnmodified": True}
    write_new(output/"dataset-audit.json", audit)
    print(json.dumps({"completed": True, "manifest": identity(output/"dino-manifest.json"), "audit": identity(output/"dataset-audit.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker-id")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.worker_id:
        worker(args.output, args.worker_id)
        return
    require(not (args.output/"dino-manifest.json").exists(), "final transfer manifest already exists")
    if args.finalize_only:
        finalize(args.output)
        return
    plan = prepare(args.output)
    if args.prepare_only:
        return
    logs = args.output/"extraction-logs"
    logs.mkdir(exist_ok=True)
    pending = [item for item in plan["records"] if not (args.output/"record-audits"/f"{item['recordingId']}.json").exists()]
    # Missing proxies first, then raw videos, then read-only inherited audits.
    pending.sort(key=lambda item: (bool(item["existingDino"]), item["tier"] == "coverage", item["recordingId"]))

    def launch(item):
        identifier = item["recordingId"]
        log_path = logs/f"{identifier}-{int(time.time_ns())}.log"
        with log_path.open("x") as handle:
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--output", str(args.output),
                            "--worker-id", identifier], stdout=handle, stderr=subprocess.STDOUT, check=True)
        print(f"DONE {identifier}; log={log_path}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(launch, pending))
    finalize(args.output)


if __name__ == "__main__":
    main()
