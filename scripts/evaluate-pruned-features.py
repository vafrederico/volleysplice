#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from analysis.artifacts import atomic_write_text
from analysis.config import FeatureConfig
from analysis.feature_experiments import FeatureExperimentError
from analysis.pipeline import _prepare_many
from analysis.pruned_feature_experiments import (
    run_fixed_split_pruned_test,
    run_targeted_pruning_experiments,
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


def build_parser() -> argparse.ArgumentParser:
    default_manifest, workspace = _workspace_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen targeted-pruning development experiments, then optionally "
            "perform one explicitly acknowledged retrospective test evaluation."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    development = commands.add_parser(
        "development",
        help="run nested source-group pruning without preparing or opening test",
    )
    development.add_argument("--manifest", type=Path, default=default_manifest)
    development.add_argument("--baseline-report", type=Path, required=True)
    development.add_argument(
        "--cache-dir",
        type=Path,
        default=workspace / "features" / "audiovisual-v2",
    )
    development.add_argument("--output", type=Path, required=True)

    final = commands.add_parser(
        "final-test",
        help="fit a frozen pruned candidate and open the single protected test source",
    )
    final.add_argument("--manifest", type=Path, default=default_manifest)
    final.add_argument("--pruning-report", type=Path, required=True)
    final.add_argument(
        "--cache-dir",
        type=Path,
        default=workspace / "features" / "audiovisual-v2",
    )
    final.add_argument("--model", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument(
        "--candidate",
        help="frozen pruned candidate; defaults to the development-selected candidate",
    )
    final.add_argument(
        "--allow-unpromoted-diagnostic",
        action="store_true",
        help=(
            "allow a failed/non-selected candidate to be evaluated retrospectively; "
            "this never makes it promotion-eligible"
        ),
    )
    final.add_argument(
        "--open-test",
        action="store_true",
        help="required acknowledgement that this command reads protected test labels",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    destination = arguments.output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {destination}")
    progress = lambda message: print(message, file=sys.stderr, flush=True)
    if arguments.command == "development":
        baseline_path = arguments.baseline_report.expanduser().resolve()
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            feature_config = FeatureConfig.from_dict(baseline["featureConfig"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise FeatureExperimentError(
                f"cannot load feature configuration from {baseline_path}: {error}"
            ) from error
        manifest = load_manifest(arguments.manifest)
        development = tuple(
            item
            for item in manifest.recordings
            if item.split in {"train", "validation"}
        )
        progress(
            f"Preparing the frozen audiovisual superset for {len(development)} "
            "development recordings; test is not prepared"
        )
        prepared = _prepare_many(
            development,
            feature_config,
            arguments.cache_dir,
            progress=progress,
        )
        report = run_targeted_pruning_experiments(
            manifest,
            prepared,
            baseline_path,
            progress=progress,
        )
    else:
        if not arguments.open_test:
            raise ValueError("final-test requires the explicit --open-test acknowledgement")
        report = run_fixed_split_pruned_test(
            arguments.manifest,
            arguments.pruning_report,
            arguments.cache_dir,
            arguments.model,
            candidate=arguments.candidate,
            allow_unpromoted_diagnostic=arguments.allow_unpromoted_diagnostic,
            progress=progress,
        )
    written = atomic_write_text(
        destination,
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )
    print(
        json.dumps(
            {
                "report": str(written),
                "kind": report["kind"],
                **(
                    {
                        "selectedCandidateForFinalTest": report[
                            "selectedCandidateForFinalTest"
                        ],
                        "testLabelsUsed": report["testLabelsUsed"],
                    }
                    if arguments.command == "development"
                    else {
                        "candidate": report["candidate"],
                        "promotionEligibleFromDevelopment": report[
                            "promotionEligibleFromDevelopment"
                        ],
                        "testLabelsOpened": report["testLabelsOpened"],
                    }
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
