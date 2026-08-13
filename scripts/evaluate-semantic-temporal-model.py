#!/usr/bin/env python3
"""Run the leakage-safe Track T DINOv2 temporal study."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.config import FeatureConfig
from analysis.dinov2_embeddings import load_dino_cache, sha256_file
from analysis.features import cached_features
from analysis.schema import load_manifest
from analysis.semantic_temporal_study import (
    DEVELOPMENT_SPLITS,
    SEED_VALUES,
    SemanticTemporalExample,
    SemanticTemporalStudyError,
    SemanticTrainingConfig,
    build_sequence_example,
    candidate_specs,
    run_nested_development_study,
)


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


def build_parser() -> argparse.ArgumentParser:
    default_manifest, workspace = _workspace_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate the frozen DINOv2-S plus audiovisual temporal head using "
            "nested source-group development folds. Test is never loaded by default."
        )
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument(
        "--dino-index",
        type=Path,
        required=True,
        help="validated extraction index from extract-dinov2-embeddings.py",
    )
    parser.add_argument(
        "--audiovisual-cache-dir",
        type=Path,
        default=workspace / "features" / "audiovisual-v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new immutable development report JSON",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--max-epochs", type=_positive_int, default=40)
    parser.add_argument("--patience", type=_positive_int, default=6)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=_positive_int, default=1)
    parser.add_argument(
        "--exclude-environment",
        choices=("beach",),
        help="run the exact frozen no-beach sensitivity study",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        choices=tuple(item.name for item in candidate_specs()),
        help="optional development smoke restriction; omit for all preregistered candidates",
    )
    parser.add_argument(
        "--open-retrospective-test",
        action="store_true",
        help="requires a frozen passing development report; protected test execution is separate",
    )
    parser.add_argument(
        "--development-report",
        type=Path,
        help="required with --open-retrospective-test and must already pass the promotion gate",
    )
    return parser


def _load_index(path: Path) -> dict[str, Path]:
    try:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read DINO extraction index {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("labelsUsed") is not False:
        raise ValueError("DINO extraction index must explicitly declare labelsUsed=false")
    rows = payload.get("recordings")
    if not isinstance(rows, list):
        raise ValueError("DINO extraction index recordings must be an array")
    result: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("recordingId"), str) or not isinstance(row.get("cache"), str):
            raise ValueError("DINO extraction index contains an invalid recording row")
        if row["recordingId"] in result:
            raise ValueError(f"DINO extraction index repeats {row['recordingId']!r}")
        cache = Path(row["cache"]).expanduser().resolve()
        if not cache.is_file():
            raise ValueError(f"DINO cache does not exist: {cache}")
        expected_sha = row.get("cacheFileSha256")
        if isinstance(expected_sha, str) and sha256_file(cache) != expected_sha:
            raise ValueError(f"DINO cache checksum mismatch: {cache}")
        result[row["recordingId"]] = cache
    return result


def _load_examples(
    manifest_path: Path,
    dino_index_path: Path,
    audiovisual_cache_dir: Path,
    *,
    exclude_environment: str | None,
) -> tuple[SemanticTemporalExample, ...]:
    manifest = load_manifest(manifest_path)
    index = _load_index(dino_index_path)
    feature_config = FeatureConfig(analysis_fps=4.0)
    feature_config.validate()
    selected = tuple(
        item
        for item in manifest.recordings
        if item.split in DEVELOPMENT_SPLITS
        and (exclude_environment is None or item.environment != exclude_environment)
    )
    if not selected:
        raise ValueError("no development recordings remain after filtering")
    examples: list[SemanticTemporalExample] = []
    for recording in selected:
        if recording.id not in index:
            raise ValueError(f"DINO index has no development cache for {recording.id}")
        cache = load_dino_cache(
            index[recording.id],
            recording_id=recording.id,
            recording_content_sha256=recording.content_sha256,
        )
        sequence = cached_features(
            recording.id,
            recording.video,
            feature_config,
            recording.roi,
            audiovisual_cache_dir,
            content_sha256=recording.content_sha256,
        )
        examples.append(build_sequence_example(recording, cache, sequence))
    return tuple(examples)


def _verify_protected_gate(report_path: Path) -> None:
    try:
        payload = json.loads(report_path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read development report {report_path}: {error}") from error
    if payload.get("freezeStatus") != "frozen-development-selection" or payload.get("testLabelsUsed") is not False:
        raise ValueError("protected test requires a frozen unopened-test development report")
    decision = payload.get("promotionDecision", {})
    if decision.get("promotionGatePassed") is not True:
        raise ValueError("protected test is closed because the development promotion gate did not pass")
    raise ValueError(
        "protected-test materialization requires a separately frozen model/checkpoint artifact; "
        "this command only implements the development study"
    )


def main() -> int:
    arguments = build_parser().parse_args()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite development report: {output}")
    if arguments.open_retrospective_test:
        if arguments.development_report is None:
            raise ValueError("--open-retrospective-test requires --development-report")
        _verify_protected_gate(arguments.development_report)
    examples = _load_examples(
        arguments.manifest,
        arguments.dino_index,
        arguments.audiovisual_cache_dir,
        exclude_environment=arguments.exclude_environment,
    )
    seeds = tuple(arguments.seeds) if arguments.seeds else SEED_VALUES
    training_config = SemanticTrainingConfig(
        max_epochs=arguments.max_epochs,
        patience=arguments.patience,
        batch_size=arguments.batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
    )
    report = run_nested_development_study(
        examples,
        seeds=seeds,
        device=arguments.device,
        training_config=training_config,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    report["manifest"] = str(arguments.manifest.expanduser().resolve())
    report["manifestFileSha256"] = sha256_file(arguments.manifest)
    report["dinoIndex"] = str(arguments.dino_index.expanduser().resolve())
    report["dinoIndexSha256"] = sha256_file(arguments.dino_index)
    report["audiovisualCacheDir"] = str(arguments.audiovisual_cache_dir.expanduser().resolve())
    report["environmentFilter"] = arguments.exclude_environment
    report["candidateRestriction"] = list(arguments.candidate) if arguments.candidate else None
    if arguments.candidate:
        report["warning"] = "candidate restriction is a smoke run and is not an official frozen study"
    atomic_write_text(output, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "recordings": len(examples),
                "sourceGroups": len({item.source_group for item in examples}),
                "selectedCandidate": report.get("selectedCandidate"),
                "promotionGatePassed": report.get("promotionDecision", {}).get("promotionGatePassed"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SemanticTemporalStudyError, RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"Track T study failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
