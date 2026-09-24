#!/usr/bin/env python3
"""Precompute one far-future DINO cache alongside the frozen two-worker queue.

The original queue later verifies/reuses this cache and publishes its usual row
audit. This helper never writes row audits, wrapper manifests or dataset reports.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--record-output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    spec = importlib.util.spec_from_file_location("frozen_dino_preparation", REPO/"scripts/prepare-neural-dino-transfer.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    from analysis.dinov2_embeddings import DinoExtractorConfig, cache_path_for, extract_recording_cache
    plan_path = args.output/"extraction-plan.json"
    plan = prep.read(plan_path)
    prep.check_sources(plan)
    item = next(r for r in plan["records"] if r["recordingId"] == args.recording_id)
    prep.require(item["existingDino"] is None and item["tier"] == "coverage", "precompute is for a missing far-future raw cache only")

    def assert_unclaimed():
        prep.require(not list((args.output/"extraction-logs").glob(f"{args.recording_id}-*.log")),
                     "main queue already claimed this recording")
        prep.require(not (args.output/"record-audits"/f"{args.recording_id}.json").exists(), "record already audited")
    assert_unclaimed()
    args.record_output.mkdir(parents=True, exist_ok=False)
    with (args.record_output/"worker-source.py").open("xb") as f:
        f.write(Path(__file__).read_bytes())
    report = {"purpose": "Scheduling-only third process; unchanged pinned extractor; main later audits/reuses cache",
              "createdAt": datetime.now(timezone.utc).isoformat(), "worker": prep.identity(Path(__file__)),
              "workerSnapshot": prep.identity(args.record_output/"worker-source.py"),
              "originalExtractionPlan": prep.identity(plan_path), "recordingId": args.recording_id,
              "concurrencySupplement": {"originalCoordinatorWorkers": 2, "additionalBoundedCacheWorker": 1,
                                         "maximumTotalWorkers": 3,
                                         "reason": "Measured two-worker RSS about3GB, WSL available10.7GB, CPU about10/24 cores; authorized third worker"},
              "passed": False}
    started = time.perf_counter()
    try:
        print(f"HASH SOURCE {args.recording_id}", flush=True)
        source = prep.identity(item["sourceVideoPath"])
        source["mtimeNs"] = Path(item["sourceVideoPath"]).stat().st_mtime_ns
        prep.require(source["sha256"] == item["recordingContentSha256"], "source content differs")
        assert_unclaimed()
        backbone = prep.load_backbone()
        config = DinoExtractorConfig()
        destination = cache_path_for(args.output/"dino-caches", item["recordingId"], item["recordingContentSha256"], config, backbone)
        prep.require(not destination.parent.exists(), "precompute destination already exists")
        cache, status = extract_recording_cache(prep.recording(item), backbone, config, args.output/"dino-caches",
                                               progress=lambda text: print(f"{datetime.now(timezone.utc).isoformat()} {text}", flush=True))
        _, _, join = prep.validate_cache(item, cache.path, plan)
        stat = Path(item["sourceVideoPath"]).stat()
        prep.require(stat.st_size == source["sizeBytes"] and stat.st_mtime_ns == source["mtimeNs"], "source changed during precompute")
        prep.check_sources(plan)
        report.update({"passed": True, "source": source, "cache": prep.identity(cache.path),
                       "status": status, "nearestAlignment": join})
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["wallSeconds"] = time.perf_counter()-started
        report["completedAt"] = datetime.now(timezone.utc).isoformat()
        prep.write_new(args.record_output/"outcome.json", report)
        print(json.dumps({"passed": report["passed"], "recordingId": args.recording_id,
                          "wallSeconds": report["wallSeconds"], "artifact": prep.identity(args.record_output/"outcome.json")}), flush=True)


if __name__ == "__main__":
    main()
