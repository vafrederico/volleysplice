#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from analysis.artifacts import atomic_write_text
from analysis.feature_experiments import sha256_file
from analysis.highres_embedding_experiment import (
    DEFAULT_BACKBONE_LAYER,
    DEFAULT_INPUT_SIZE,
    DEFAULT_PROJECTION_DIMENSION,
    DEFAULT_PROJECTION_SEED,
    DEFAULT_SAMPLE_FPS,
    DEFAULT_SOURCE_DIMENSION,
    OpenCvFrozenBackbone,
    build_extraction_spec,
    extract_recording_highres_cache,
)
from analysis.schema import load_manifest


def _workspace_defaults() -> tuple[Path, Path]:
    data_root = Path(os.environ.get("VOLLEYCUT_DATA_ROOT", "data")).expanduser()
    workspace = Path(
        os.environ.get(
            "VOLLEYCUT_LABELING_WORKSPACE",
            data_root / "labeling-v1-2026-08-09",
        )
    ).expanduser()
    return workspace / "manifests" / "full-gold-v1.json", workspace


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    default_manifest, workspace = _workspace_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Generate content-addressed CPU/OpenCV MobileNetV2 crop caches. "
            "The backbone is never downloaded and existing cache files are never overwritten."
        )
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument(
        "--backbone-sha256",
        required=True,
        help="required expected SHA-256 for the explicitly supplied ONNX backbone",
    )
    parser.add_argument("--layer", default=DEFAULT_BACKBONE_LAYER)
    parser.add_argument("--source-dimension", type=_positive_int, default=DEFAULT_SOURCE_DIMENSION)
    parser.add_argument("--input-size", type=_positive_int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--sample-fps", type=_positive_float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument(
        "--projection-dimension",
        type=_positive_int,
        default=DEFAULT_PROJECTION_DIMENSION,
    )
    parser.add_argument("--projection-seed", type=int, default=DEFAULT_PROJECTION_SEED)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=workspace / "features" / "feature-order-2026-08-12" / "highres-mobilenetv2",
    )
    parser.add_argument("--output", type=Path, required=True, help="new extraction index JSON")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test", "challenge"),
        default=("train", "validation"),
    )
    parser.add_argument(
        "--recording-id",
        action="append",
        default=[],
        help="optionally restrict extraction to one or more recording IDs",
    )
    parser.add_argument(
        "--allow-protected-splits",
        action="store_true",
        help="required acknowledgement before extracting test/challenge video",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing extraction index: {output}")
    requested_splits = set(arguments.splits)
    if requested_splits.intersection({"test", "challenge"}) and not arguments.allow_protected_splits:
        raise ValueError(
            "test/challenge extraction requires --allow-protected-splits; development defaults "
            "to train+validation only"
        )
    expected_sha256 = arguments.backbone_sha256.strip().lower()
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("--backbone-sha256 must be 64 hexadecimal characters")
    backbone = OpenCvFrozenBackbone(
        arguments.backbone,
        layer=arguments.layer,
        expected_source_dimension=arguments.source_dimension,
        input_size=arguments.input_size,
    )
    if backbone.sha256 != expected_sha256:
        raise ValueError(
            f"backbone SHA-256 mismatch: expected {expected_sha256}, got {backbone.sha256}"
        )
    spec, projection = build_extraction_spec(
        backbone,
        sample_fps=arguments.sample_fps,
        projection_dimension=arguments.projection_dimension,
        projection_seed=arguments.projection_seed,
    )
    manifest = load_manifest(arguments.manifest)
    requested_ids = set(arguments.recording_id)
    known_ids = {item.id for item in manifest.recordings}
    missing = requested_ids - known_ids
    if missing:
        raise ValueError(f"unknown --recording-id values: {sorted(missing)}")
    recordings = tuple(
        item
        for item in manifest.recordings
        if item.split in requested_splits and (not requested_ids or item.id in requested_ids)
    )
    if not recordings:
        raise ValueError("no recordings match the requested split/ID filters")
    rows: list[dict[str, object]] = []
    progress = lambda message: print(message, file=sys.stderr, flush=True)
    for index, recording in enumerate(recordings, start=1):
        progress(f"High-res extraction {index}/{len(recordings)}: {recording.id}")
        cache, status = extract_recording_highres_cache(
            recording,
            backbone,
            spec,
            projection,
            arguments.cache_dir,
            progress=progress,
        )
        rows.append(
            {
                "recordingId": recording.id,
                "split": recording.split,
                "sourceGroup": recording.source_group,
                "recordingContentSha256": recording.content_sha256,
                "cache": str(cache.path),
                "cacheFileSha256": sha256_file(cache.path),
                "status": status,
                "samples": len(cache.times),
            }
        )
    report = {
        "schemaVersion": 1,
        "kind": "volleycut-frozen-highres-cache-extraction-index",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "selectedSplits": sorted(requested_splits),
        "protectedSplitAcknowledged": bool(arguments.allow_protected_splits),
        "labelsUsed": False,
        "extractorConfigSha256": spec.config_sha256,
        "extractor": spec.to_dict(),
        "recordings": rows,
    }
    written = atomic_write_text(
        output, json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "index": str(written),
                "recordings": len(rows),
                "created": sum(item["status"] == "created" for item in rows),
                "reused": sum(item["status"] == "reused" for item in rows),
                "backboneSha256": backbone.sha256,
                "extractorConfigSha256": spec.config_sha256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
