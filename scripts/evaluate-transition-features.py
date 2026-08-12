#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from analysis.artifacts import atomic_write_text
from analysis.config import DecoderConfig, FeatureConfig, TrainingConfig
from analysis.pipeline import _prepare_many
from analysis.schema import load_manifest
from analysis.transition_feature_experiment import (
    DEFAULT_OBJECTIVE_MARGIN,
    DEFAULT_SIGN_CONSISTENCY,
    run_development_transition_experiments,
    run_retrospective_transition_test,
)


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


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.5 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in (0.5, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    default_manifest, workspace = _workspace_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic multiscale transition and predeclared "
            "cross-modal interaction features from warm 90-signal caches."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    development = commands.add_parser(
        "development",
        help="run train+validation-only nested leave-one-sourceGroup-out experiments",
    )
    development.add_argument("--manifest", type=Path, default=default_manifest)
    development.add_argument(
        "--cache-dir",
        type=Path,
        default=workspace / "features" / "audiovisual-v2",
    )
    development.add_argument("--output", type=Path, required=True)
    development.add_argument("--analysis-fps", type=float, default=4.0)
    development.add_argument("--resize-width", type=int, default=192)
    development.add_argument("--resize-height", type=int, default=108)
    development.add_argument("--grid-size", type=int, default=3)
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
    development.add_argument(
        "--inner-fold-limit",
        type=int,
        default=0,
        help="0 uses every inner LOGO fold; positive values are runtime smoke ablations",
    )
    development.add_argument(
        "--objective-margin",
        type=float,
        default=DEFAULT_OBJECTIVE_MARGIN,
    )
    development.add_argument(
        "--sign-consistency",
        type=_probability,
        default=DEFAULT_SIGN_CONSISTENCY,
    )

    retrospective = commands.add_parser(
        "retrospective-test",
        help="fit the frozen development winner on all development rows and open test once",
    )
    retrospective.add_argument("--manifest", type=Path, default=default_manifest)
    retrospective.add_argument(
        "--cache-dir",
        type=Path,
        default=workspace / "features" / "audiovisual-v2",
    )
    retrospective.add_argument("--development-report", type=Path, required=True)
    retrospective.add_argument("--output", type=Path, required=True)
    retrospective.add_argument(
        "--open-test",
        action="store_true",
        help="required acknowledgement that this mode accesses protected test labels",
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
        if arguments.inner_fold_limit < 0:
            raise ValueError("--inner-fold-limit cannot be negative")
        if arguments.objective_margin < 0:
            raise ValueError("--objective-margin cannot be negative")
        feature_config = FeatureConfig(
            analysis_fps=arguments.analysis_fps,
            resize_width=arguments.resize_width,
            resize_height=arguments.resize_height,
            grid_size=arguments.grid_size,
            use_optical_flow=True,
            use_advanced_visual=True,
            use_audio=True,
            audio_sample_rate=arguments.audio_sample_rate,
            context_offsets_seconds=(-2.0, -1.0, 0.0, 1.0, 2.0),
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
        manifest = load_manifest(arguments.manifest)
        development_rows = tuple(
            item
            for item in manifest.recordings
            if item.split in {"train", "validation"}
        )
        progress(
            f"Loading warm 90-signal features for {len(development_rows)} "
            "development recordings; protected test is not prepared"
        )
        prepared = _prepare_many(
            development_rows,
            feature_config,
            arguments.cache_dir,
            progress=progress,
        )
        report = run_development_transition_experiments(
            manifest,
            prepared,
            feature_config=feature_config,
            training_config=training_config,
            decoder_config=DecoderConfig(),
            inner_fold_limit=(
                arguments.inner_fold_limit if arguments.inner_fold_limit else None
            ),
            objective_margin=arguments.objective_margin,
            sign_consistency=arguments.sign_consistency,
            progress=progress,
        )
    else:
        if not arguments.open_test:
            raise ValueError(
                "retrospective-test requires the explicit --open-test acknowledgement"
            )
        report = run_retrospective_transition_test(
            arguments.manifest,
            arguments.development_report,
            arguments.cache_dir,
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
                "testLabelsOpened": bool(report.get("testLabelsOpened", False)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
