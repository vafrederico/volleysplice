#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from analysis.artifacts import atomic_write_text
from analysis.config import DecoderConfig, FeatureConfig, TrainingConfig
from analysis.feature_experiments import (
    DEFAULT_PADDING_SECONDS,
    run_development_experiments,
    run_fixed_split_final_test,
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


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    default_manifest, workspace = _workspace_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe source-group feature experiments, or explicitly open the "
            "fixed test split after a development report is frozen."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    development = subparsers.add_parser(
        "development",
        help="run train+validation-only nested source-group experiments",
    )
    development.add_argument("--manifest", type=Path, default=default_manifest)
    development.add_argument(
        "--cache-dir",
        type=Path,
        default=workspace / "features" / "audiovisual-v2-superset",
    )
    development.add_argument("--output", type=Path, required=True)
    development.add_argument("--analysis-fps", type=float, default=4.0)
    development.add_argument("--resize-width", type=int, default=192)
    development.add_argument("--resize-height", type=int, default=108)
    development.add_argument("--grid-size", type=int, default=3)
    development.add_argument("--no-optical-flow", action="store_true")
    development.add_argument("--no-advanced-visual", action="store_true")
    development.add_argument("--no-audio", action="store_true")
    development.add_argument("--audio-sample-rate", type=int, default=16000)
    development.add_argument(
        "--sequence-normalization",
        choices=("none", "percentile-rank"),
        default="percentile-rank",
    )
    development.add_argument("--epochs", type=_positive, default=60)
    development.add_argument("--batch-size", type=_positive, default=2048)
    development.add_argument("--learning-rate", type=float, default=0.02)
    development.add_argument("--l2", type=float, default=1e-4)
    development.add_argument("--patience", type=_positive, default=15)
    development.add_argument("--seed", type=int, default=7)
    development.add_argument("--permutation-repeats", type=_positive, default=10)
    development.add_argument("--minimum-shift-seconds", type=float, default=5.0)
    development.add_argument(
        "--inner-fold-limit",
        type=int,
        default=0,
        help="0 uses every inner LOGO fold; a positive value is a declared runtime ablation",
    )
    development.add_argument(
        "--padding-seconds",
        type=float,
        nargs="+",
        default=list(DEFAULT_PADDING_SECONDS),
    )

    final = subparsers.add_parser(
        "final-test",
        help="train the fixed-split full model and explicitly open test once",
    )
    final.add_argument("--manifest", type=Path, default=default_manifest)
    final.add_argument(
        "--cache-dir",
        type=Path,
        default=workspace / "features" / "audiovisual-v2-superset",
    )
    final.add_argument("--development-report", type=Path, required=True)
    final.add_argument("--model", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument(
        "--open-test",
        action="store_true",
        help="required explicit acknowledgement that this command accesses test labels",
    )
    return parser


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

    progress = lambda message: print(message, file=sys.stderr, flush=True)
    if arguments.command == "development":
        feature_config = FeatureConfig(
            analysis_fps=arguments.analysis_fps,
            resize_width=arguments.resize_width,
            resize_height=arguments.resize_height,
            grid_size=arguments.grid_size,
            use_optical_flow=not arguments.no_optical_flow,
            use_advanced_visual=not arguments.no_advanced_visual,
            use_audio=not arguments.no_audio,
            audio_sample_rate=arguments.audio_sample_rate,
            sequence_normalization=arguments.sequence_normalization,
        )
        training_config = TrainingConfig(
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            l2=arguments.l2,
            patience=arguments.patience,
            seed=arguments.seed,
        )
        feature_config.validate()
        training_config.validate()
        if arguments.inner_fold_limit < 0:
            raise ValueError("--inner-fold-limit cannot be negative")
        if arguments.minimum_shift_seconds < 0:
            raise ValueError("--minimum-shift-seconds cannot be negative")
        manifest = load_manifest(arguments.manifest)
        development_rows = tuple(
            item for item in manifest.recordings if item.split in {"train", "validation"}
        )
        progress(
            f"Preparing one audiovisual feature superset for {len(development_rows)} "
            "development recordings; test is not prepared"
        )
        prepared = _prepare_many(
            development_rows,
            feature_config,
            arguments.cache_dir,
            progress=progress,
        )
        report = run_development_experiments(
            manifest,
            prepared,
            feature_config=feature_config,
            training_config=training_config,
            decoder_config=DecoderConfig(),
            permutation_repeats=arguments.permutation_repeats,
            padding_seconds=arguments.padding_seconds,
            minimum_shift_seconds=arguments.minimum_shift_seconds,
            inner_fold_limit=(
                arguments.inner_fold_limit if arguments.inner_fold_limit else None
            ),
            progress=progress,
        )
    else:
        if not arguments.open_test:
            raise ValueError("final-test requires the explicit --open-test acknowledgement")
        report = run_fixed_split_final_test(
            arguments.manifest,
            arguments.development_report,
            arguments.cache_dir,
            arguments.model,
            progress=progress,
        )

    written = _write_report(output, report)
    print(
        json.dumps(
            {
                "report": str(written),
                "kind": report["kind"],
                "testLabelsOpened": bool(report.get("testLabelsOpened", False)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
