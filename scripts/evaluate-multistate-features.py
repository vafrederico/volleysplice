#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from analysis.artifacts import atomic_write_text
from analysis.config import DecoderConfig, FeatureConfig, TrainingConfig
from analysis.multistate_feature_study import (
    DEFAULT_TRANSITION_BONUSES,
    run_development_multistate_study,
    run_retrospective_multistate_test,
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
            "Compare a serve-anchored four-state model with the same-feature "
            "binary logistic control under nested source-group OOF evaluation."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    development = commands.add_parser(
        "development",
        help="run nested development-only comparison without preparing test",
    )
    development.add_argument("--manifest", type=Path, default=default_manifest)
    development.add_argument(
        "--cache-dir",
        type=Path,
        default=workspace / "features" / "audiovisual-v2",
    )
    development.add_argument("--upstream-report", type=Path, required=True)
    development.add_argument("--output", type=Path, required=True)
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
        help="0 uses every inner fold; positive values are runtime smoke ablations",
    )

    retrospective = commands.add_parser(
        "retrospective-test",
        help="open test only if the frozen development report promoted multistate",
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
        help="required acknowledgement that this command accesses test labels",
    )
    return parser


def _write_report(path: Path, report: dict[str, object]) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {destination}")
    return atomic_write_text(
        destination, json.dumps(report, indent=2, allow_nan=False) + "\n"
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
        upstream = json.loads(
            arguments.upstream_report.expanduser().resolve().read_text(encoding="utf-8")
        )
        feature_config = FeatureConfig.from_dict(upstream["featureConfig"])
        manifest = load_manifest(arguments.manifest)
        development_rows = tuple(
            item for item in manifest.recordings if item.split in {"train", "validation"}
        )
        progress(
            f"Loading warm features for {len(development_rows)} development recordings; "
            "protected test is not prepared"
        )
        prepared = _prepare_many(
            development_rows, feature_config, arguments.cache_dir, progress=progress
        )
        report = run_development_multistate_study(
            manifest,
            prepared,
            upstream_report_path=arguments.upstream_report,
            training_config=TrainingConfig(
                epochs=arguments.epochs,
                batch_size=arguments.batch_size,
                learning_rate=arguments.learning_rate,
                l2=arguments.l2,
                patience=arguments.patience,
                seed=arguments.seed,
            ),
            decoder_config=DecoderConfig(),
            inner_fold_limit=(
                arguments.inner_fold_limit if arguments.inner_fold_limit else None
            ),
            transition_bonuses=DEFAULT_TRANSITION_BONUSES,
            progress=progress,
        )
    else:
        if not arguments.open_test:
            raise ValueError(
                "retrospective-test requires the explicit --open-test acknowledgement"
            )
        report = run_retrospective_multistate_test(
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
                "selectedArchitecture": report.get(
                    "promotionDecision", {}
                ).get("selectedArchitecture", report.get("selectedArchitecture")),
                "testLabelsOpened": bool(report.get("testLabelsOpened", False)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
