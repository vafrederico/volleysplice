#!/usr/bin/env python3
"""Extract label-blind regional MobileNetV3 features for a declared video manifest.

Accepts a records-only manifest or the existing expanded AV manifest. Only video
identity, ROI and split fields are copied; label files and AV arrays are unopened.
Defaults to CPU and one recording at a time. No training is performed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.mobile_visual_features import (MobileVisualConfig, extract_recording_cache,
                                             load_mobile_backbone, require, sha256_file)


def manifest_records(document):
    raw = document.get("records", document.get("recordings"))
    if raw is None:
        raw = [row for key in ("exactRows", "draftRows", "coverageRows") for row in document.get(key, [])]
    require(isinstance(raw, list) and bool(raw), "manifest contains no video records")
    records = []
    for row in raw:
        records.append({"id": row.get("id", row.get("recordingId")),
                        "video": row.get("video", row.get("sourceVideoPath")),
                        "contentSha256": row.get("contentSha256", row.get("recordingContentSha256")),
                        "roi": row.get("roi"), "sourceGroup": row.get("sourceGroup"),
                        "split": row.get("split"), "environment": row.get("environment")})
    require(all(row["id"] and row["video"] and row["contentSha256"] for row in records), "video identity fields missing")
    require(len({row["id"] for row in records}) == len(records), "duplicate recording IDs")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--recording-id", action="append", default=[])
    parser.add_argument("--max-seconds", type=float, help="bounded engine pilot from video start; produces an explicitly partial cache")
    parser.add_argument("--list-only", action="store_true", help="validate selected video fields; no model load or media reads")
    args = parser.parse_args()
    config = MobileVisualConfig(batch_size=args.batch_size)
    config.validate()
    digest = sha256_file(args.manifest)
    if args.manifest_sha256:
        require(digest == args.manifest_sha256.lower(), "manifest SHA-256 mismatch")
    records = manifest_records(json.loads(args.manifest.read_text(encoding="utf-8")))
    requested = set(args.recording_id)
    require(not requested - {row["id"] for row in records}, "unknown requested recording IDs")
    selected = [row for row in records if not requested or row["id"] in requested]
    require(all(row["split"] in ("train", "validation") for row in selected), "only declared development splits allowed")
    require(all(row["environment"] != "beach" and not row["id"].lower().startswith("beach-") for row in selected), "beach is outside current scope")
    require(not args.output.exists(), "refusing to overwrite extraction report")
    if args.list_only:
        print(json.dumps({"manifestSha256": digest, "records": selected, "config": config.to_dict(), "labelsUsed": False}, indent=2))
        return
    import torch
    require(args.threads > 0, "threads must be positive")
    torch.set_num_threads(args.threads)
    backbone = load_mobile_backbone(args.checkpoint, checkpoint_sha256=args.checkpoint_sha256,
                                    allow_download=args.allow_download, device=args.device)
    rows = []
    for recording in selected:
        cache, status = extract_recording_cache(recording, backbone, config, args.cache_dir,
                                                max_seconds=args.max_seconds,
                                                progress=lambda message: print(message, file=sys.stderr, flush=True))
        rows.append({"id": recording["id"], "recordingId": recording["id"], "sourceGroup": recording["sourceGroup"],
                     "split": recording["split"], "recordingContentSha256": recording["contentSha256"],
                     "cachePath": str(cache.path.resolve()), "cacheSha256": sha256_file(cache.path),
                     "shape": list(cache.tokens.shape), "qualityShape": list(cache.quality.shape), "status": status})
    require(sha256_file(args.manifest) == digest, "manifest changed during extraction")
    report = {"schemaVersion": 1, "kind": "volleycut-mobile-visual-extraction-index",
              "createdAt": datetime.now(timezone.utc).isoformat(), "manifest": str(args.manifest.resolve()),
              "manifestSha256": digest, "sourceSha256": sha256_file(__file__),
              "labelsUsed": False, "protectedTestOpened": False, "beachIncluded": False,
              "backbone": backbone.identity(), "config": config.to_dict(), "records": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "recordings": len(rows), "completed": True}))


if __name__ == "__main__":
    main()
