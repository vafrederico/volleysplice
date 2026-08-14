#!/usr/bin/env python3
"""Extract validated frozen DINOv2-S/14 caches for Track T."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis.artifacts import atomic_write_text
from analysis.dinov2_embeddings import (
    DEFAULT_DINO_BATCH_SIZE,
    DEFAULT_DINO_CHUNK_SIZE,
    DEFAULT_DINO_INPUT_SIZE,
    DinoEmbeddingError,
    DinoExtractorConfig,
    _cv2,
    extract_recording_cache,
    load_pinned_dinov2,
    qualify_backbone,
    sha256_file,
)
from analysis.schema import load_manifest


def _workspace_defaults() -> tuple[Path, Path]:
    data_root = Path(os.environ.get("VOLLEYCUT_DATA_ROOT", "/mnt/z/volleycut")).expanduser()
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


def _git_head(repository: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip().lower()
    except (OSError, subprocess.SubprocessError):
        return None


def build_parser() -> argparse.ArgumentParser:
    default_manifest, workspace = _workspace_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Extract content-addressed frozen DINOv2 ViT-S/14 pooled tokens. "
            "The local repository and checkpoint must be explicitly pinned; no downloads occur."
        )
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--dinov2-repository", type=Path, required=True)
    parser.add_argument("--dinov2-repository-commit", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=_positive_int, default=DEFAULT_DINO_INPUT_SIZE)
    parser.add_argument("--sample-fps", type=_positive_float, default=4.0)
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_DINO_BATCH_SIZE)
    parser.add_argument("--chunk-size", type=_positive_int, default=DEFAULT_DINO_CHUNK_SIZE)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=workspace / "features" / "dinov2-vits14-v1",
    )
    parser.add_argument("--output", type=Path, required=True, help="new extraction index/report JSON")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test", "challenge"),
        default=("train", "validation"),
    )
    parser.add_argument("--recording-id", action="append", default=[])
    parser.add_argument(
        "--allow-protected-splits",
        action="store_true",
        help="required acknowledgement before reading test/challenge video",
    )
    parser.add_argument(
        "--resource-sweep",
        action="store_true",
        help="qualify one first-frame input-size sweep without writing a feature cache",
    )
    parser.add_argument(
        "--resource-video",
        type=Path,
        help=(
            "optional proxy path for --resource-sweep; bypasses the manifest and is useful "
            "when the canonical label manifest is not mounted"
        ),
    )
    parser.add_argument(
        "--benchmark-input-sizes",
        nargs="+",
        type=_positive_int,
        default=(224, 336),
    )
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    requested_splits = set(arguments.splits)
    if requested_splits.intersection({"test", "challenge"}) and not arguments.allow_protected_splits:
        raise ValueError(
            "test/challenge extraction requires --allow-protected-splits; development defaults to train+validation"
        )
    repository = arguments.dinov2_repository.expanduser().resolve()
    head = _git_head(repository)
    if head is not None and head != arguments.dinov2_repository_commit.lower():
        raise ValueError(
            f"DINO repository commit mismatch: requested {arguments.dinov2_repository_commit}, local HEAD is {head}"
        )
    if len(arguments.dinov2_repository_commit) != 40:
        raise ValueError("--dinov2-repository-commit must be a full 40-character commit")
    checkpoint_sha = arguments.checkpoint_sha256.strip().lower()
    if len(checkpoint_sha) != 64 or any(item not in "0123456789abcdef" for item in checkpoint_sha):
        raise ValueError("--checkpoint-sha256 must be a 64-character hexadecimal digest")
    if arguments.resource_sweep and len(arguments.recording_id) > 1:
        raise ValueError("--resource-sweep accepts at most one --recording-id")


def _read_first_frame_path(video: Path) -> Any:
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(video.expanduser().resolve()))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None or not frame.size:
        raise RuntimeError(f"cannot decode first frame: {video}")
    return frame


def main() -> int:
    arguments = build_parser().parse_args()
    _validate_arguments(arguments)
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite extraction report: {output}")
    manifest = None
    recordings: tuple[Any, ...] = ()
    if not (arguments.resource_sweep and arguments.resource_video is not None):
        manifest = load_manifest(arguments.manifest)
        requested_ids = set(arguments.recording_id)
        known_ids = {item.id for item in manifest.recordings}
        missing = requested_ids - known_ids
        if missing:
            raise ValueError(f"unknown --recording-id values: {sorted(missing)}")
        recordings = tuple(
            item
            for item in manifest.recordings
            if item.split in set(arguments.splits) and (not requested_ids or item.id in requested_ids)
        )
        if not recordings:
            raise ValueError("no recordings match the requested split/ID filters")
    elif not arguments.resource_video.expanduser().resolve().is_file():
        raise ValueError(f"resource video does not exist: {arguments.resource_video}")
    backbone = load_pinned_dinov2(
        arguments.dinov2_repository,
        repository_commit=arguments.dinov2_repository_commit,
        checkpoint=arguments.checkpoint,
        checkpoint_sha256=arguments.checkpoint_sha256.strip().lower(),
        device=arguments.device,
    )
    if arguments.resource_sweep:
        recording = recordings[0] if recordings else None
        resource_video = (
            recording.video
            if recording is not None
            else arguments.resource_video.expanduser().resolve()
        )
        report = {
            "schemaVersion": 1,
            "kind": "volleycut-dinov2-resource-qualification",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "labelsUsed": False,
            "manifest": str(manifest.path) if manifest is not None else None,
            "recordingId": recording.id if recording is not None else resource_video.stem,
            "sourceVideoPath": str(resource_video),
            "recordingContentSha256": recording.content_sha256 if recording is not None else sha256_file(resource_video),
            "inputSizes": list(arguments.benchmark_input_sizes),
            "qualification": qualify_backbone(
                backbone,
                _read_first_frame_path(resource_video),
                input_sizes=arguments.benchmark_input_sizes,
                rois=(recording.roi,) if recording is not None else (None,),
            ),
        }
        atomic_write_text(output, json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "output": str(output),
                    "recordingId": recording.id if recording is not None else resource_video.stem,
                },
                indent=2,
            )
        )
        return 0

    config = DinoExtractorConfig(
        sample_fps=arguments.sample_fps,
        input_size=arguments.input_size,
        batch_size=arguments.batch_size,
        chunk_size=arguments.chunk_size,
    )
    config.validate()
    rows: list[dict[str, Any]] = []
    for index, recording in enumerate(recordings, start=1):
        print(f"DINO extraction {index}/{len(recordings)}: {recording.id}", file=sys.stderr, flush=True)
        cache, status = extract_recording_cache(
            recording,
            backbone,
            config,
            arguments.cache_dir,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
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
                "samples": len(cache.timestamps),
                "shape": list(cache.tokens.shape),
            }
        )
    report = {
        "schemaVersion": 1,
        "kind": "volleycut-dinov2-cache-extraction-index",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "selectedSplits": sorted(set(arguments.splits)),
        "protectedSplitAcknowledged": bool(arguments.allow_protected_splits),
        "labelsUsed": False,
        "extractorConfig": config.to_dict(),
        "extractorConfigSha256": config.config_sha256,
        "backbone": backbone.identity(),
        "recordings": rows,
    }
    atomic_write_text(output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "recordings": len(rows),
                "created": sum(item["status"] == "created" for item in rows),
                "reused": sum(item["status"] == "reused" for item in rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DinoEmbeddingError, RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"Track T extraction failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
