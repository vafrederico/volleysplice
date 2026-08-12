#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from analysis.artifacts import atomic_write_text
from analysis.config import FeatureConfig
from analysis.highres_embedding_experiment import (
    DEFAULT_BACKBONE_LAYER,
    DEFAULT_INPUT_SIZE,
    DEFAULT_LONG_WINDOW_SECONDS,
    DEFAULT_PROJECTION_DIMENSION,
    DEFAULT_PROJECTION_SEED,
    DEFAULT_SAMPLE_FPS,
    DEFAULT_SHORT_WINDOW_SECONDS,
    DEFAULT_SOURCE_DIMENSION,
    OpenCvFrozenBackbone,
    _load_frozen_highres_study,
    build_extraction_spec,
    run_development_highres_study,
    run_retrospective_highres_test,
)
from analysis.pipeline import _prepare_many
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


def _add_cache_arguments(
    parser: argparse.ArgumentParser,
    default_manifest: Path,
    workspace: Path,
) -> None:
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument(
        "--warm-cache-dir",
        type=Path,
        default=workspace / "features" / "audiovisual-v2",
    )
    parser.add_argument(
        "--highres-cache-dir",
        type=Path,
        default=(
            workspace
            / "features"
            / "feature-order-2026-08-12"
            / "highres-mobilenetv2"
        ),
    )


def _add_frozen_extractor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument(
        "--backbone-sha256",
        required=True,
        help="required expected SHA-256 for the explicitly supplied ONNX backbone",
    )
    parser.add_argument("--layer", default=DEFAULT_BACKBONE_LAYER)
    parser.add_argument(
        "--source-dimension", type=_positive_int, default=DEFAULT_SOURCE_DIMENSION
    )
    parser.add_argument("--input-size", type=_positive_int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--sample-fps", type=_positive_float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument(
        "--projection-dimension",
        type=_positive_int,
        default=DEFAULT_PROJECTION_DIMENSION,
    )
    parser.add_argument("--projection-seed", type=int, default=DEFAULT_PROJECTION_SEED)
    parser.add_argument(
        "--short-window-seconds",
        type=_positive_float,
        default=DEFAULT_SHORT_WINDOW_SECONDS,
    )
    parser.add_argument(
        "--long-window-seconds",
        type=_positive_float,
        default=DEFAULT_LONG_WINDOW_SECONDS,
    )


def build_parser() -> argparse.ArgumentParser:
    default_manifest, workspace = _workspace_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Run a nested development ablation for frozen high-resolution crop "
            "summaries or, only after promotion, one gated retrospective test."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    development = commands.add_parser(
        "development",
        help="run train+validation-only nested source-group OOF comparison",
    )
    _add_cache_arguments(development, default_manifest, workspace)
    _add_frozen_extractor_arguments(development)
    development.add_argument("--upstream-report", type=Path, required=True)
    development.add_argument("--output", type=Path, required=True)

    retrospective = commands.add_parser(
        "retrospective-test",
        help=(
            "fit the promoted frozen development winner and evaluate pre-extracted "
            "protected caches once"
        ),
    )
    _add_cache_arguments(retrospective, default_manifest, workspace)
    _add_frozen_extractor_arguments(retrospective)
    retrospective.add_argument("--development-report", type=Path, required=True)
    retrospective.add_argument(
        "--test-cache-index",
        type=Path,
        required=True,
        help=(
            "protected extraction index created separately with "
            "extract-highres-embeddings.py --splits test --allow-protected-splits"
        ),
    )
    retrospective.add_argument("--output", type=Path, required=True)
    retrospective.add_argument(
        "--open-test",
        action="store_true",
        help="required acknowledgement that this mode accesses protected test labels",
    )
    return parser


def _frozen_spec(arguments: argparse.Namespace):
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
    spec, _ = build_extraction_spec(
        backbone,
        sample_fps=arguments.sample_fps,
        projection_dimension=arguments.projection_dimension,
        projection_seed=arguments.projection_seed,
    )
    return spec


def _write_report(path: Path, report: dict[str, object]) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {destination}")
    return atomic_write_text(
        destination,
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )


def main() -> int:
    arguments = build_parser().parse_args()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output}")
    if arguments.command == "retrospective-test" and not arguments.open_test:
        raise ValueError(
            "retrospective-test requires the explicit --open-test acknowledgement"
        )
    progress = lambda message: print(message, file=sys.stderr, flush=True)

    if arguments.command == "development":
        spec = _frozen_spec(arguments)
        upstream = json.loads(
            arguments.upstream_report.expanduser().resolve().read_text(encoding="utf-8")
        )
        feature_config = FeatureConfig.from_dict(dict(upstream["featureConfig"]))
        manifest = load_manifest(arguments.manifest)
        development_rows = tuple(
            item
            for item in manifest.recordings
            if item.split in {"train", "validation"}
        )
        progress(
            f"Loading warm features for {len(development_rows)} development recordings; "
            "protected test is not prepared"
        )
        prepared = _prepare_many(
            development_rows,
            feature_config,
            arguments.warm_cache_dir,
            progress=progress,
        )
        report = run_development_highres_study(
            manifest,
            prepared,
            upstream_report_path=arguments.upstream_report,
            spec=spec,
            cache_dir=arguments.highres_cache_dir,
            short_window_seconds=arguments.short_window_seconds,
            long_window_seconds=arguments.long_window_seconds,
            progress=progress,
        )
    else:
        # Refuse a failed/smoke development result before loading even the
        # backbone, let alone a manifest or protected cache.
        _load_frozen_highres_study(arguments.development_report)
        spec = _frozen_spec(arguments)
        report = run_retrospective_highres_test(
            arguments.manifest,
            arguments.development_report,
            arguments.warm_cache_dir,
            arguments.highres_cache_dir,
            spec=spec,
            test_cache_index_path=arguments.test_cache_index,
            short_window_seconds=arguments.short_window_seconds,
            long_window_seconds=arguments.long_window_seconds,
            progress=progress,
        )

    written = _write_report(output, report)
    print(
        json.dumps(
            {
                "report": str(written),
                "kind": report["kind"],
                "selectedCandidate": report.get(
                    "selectedCandidateForRetrospectiveTest",
                    report.get("selectedCandidate"),
                ),
                "highresPromoted": bool(
                    report.get("candidateSelection", {}).get(
                        "highresPromoted",
                        report.get("highresPromotionGatePassed", False),
                    )
                ),
                "testLabelsOpened": bool(report.get("testLabelsOpened", False)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
