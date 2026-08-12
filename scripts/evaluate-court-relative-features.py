#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from analysis.artifacts import atomic_write_text
from analysis.config import FeatureConfig
from analysis.court_relative_feature_study import (
    load_frozen_step1_report,
    run_development_court_experiments,
    run_retrospective_court_test,
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


def build_parser() -> argparse.ArgumentParser:
    default_manifest, workspace = _workspace_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Compare rectangle-ROI court-relative features against the exact frozen "
            "step-1 transition-feature winner."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    development = commands.add_parser(
        "development",
        help="run full train+validation-only nested source-group experiments",
    )
    development.add_argument("--manifest", type=Path, default=default_manifest)
    development.add_argument(
        "--cache-dir",
        type=Path,
        default=workspace / "features" / "audiovisual-v2",
    )
    development.add_argument("--step1-development-report", type=Path, required=True)
    development.add_argument("--output", type=Path, required=True)

    retrospective = commands.add_parser(
        "retrospective-test",
        help="fit the frozen court winner on development and open test once",
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
        _, upstream = load_frozen_step1_report(arguments.step1_development_report)
        feature_config = FeatureConfig.from_dict(dict(upstream["featureConfig"]))
        manifest = load_manifest(arguments.manifest)
        development_rows = tuple(
            item
            for item in manifest.recordings
            if item.split in {"train", "validation"}
        )
        progress(
            f"Loading warm features for {len(development_rows)} development "
            "recordings; protected test/challenge rows are not prepared"
        )
        prepared = _prepare_many(
            development_rows,
            feature_config,
            arguments.cache_dir,
            progress=progress,
        )
        report = run_development_court_experiments(
            manifest,
            prepared,
            step1_report_path=arguments.step1_development_report,
            progress=progress,
        )
    else:
        if not arguments.open_test:
            raise ValueError(
                "retrospective-test requires the explicit --open-test acknowledgement"
            )
        report = run_retrospective_court_test(
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
