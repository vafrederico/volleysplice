#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from analysis.artifacts import atomic_write_text
from analysis.config import FeatureConfig
from analysis.multistate_followup import (
    run_conservative_hybrid_study,
    run_multistate_oof_diagnostics,
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
            "Reproduce frozen multistate outer-fold scores and report development-only "
            "state calibration, serve rejection, duration, and boundary-oracle diagnostics."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    diagnostic = commands.add_parser(
        "diagnostics",
        help="write a reusable development OOF cache and diagnostic report",
    )
    diagnostic.add_argument("--manifest", type=Path, default=default_manifest)
    diagnostic.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=workspace / "features" / "audiovisual-v2",
    )
    diagnostic.add_argument("--multistate-report", type=Path, required=True)
    diagnostic.add_argument("--oof-cache-output", type=Path, required=True)
    diagnostic.add_argument("--output", type=Path, required=True)
    hybrid = commands.add_parser(
        "hybrid-development",
        help="run the fixed conservative hybrid under full nested source-group OOF",
    )
    hybrid.add_argument("--manifest", type=Path, default=default_manifest)
    hybrid.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=workspace / "features" / "audiovisual-v2",
    )
    hybrid.add_argument("--multistate-report", type=Path, required=True)
    hybrid.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output}")
    multistate = json.loads(
        arguments.multistate_report.expanduser().resolve().read_text(encoding="utf-8")
    )
    feature_config = FeatureConfig.from_dict(multistate["featureConfig"])
    manifest = load_manifest(arguments.manifest)
    development = tuple(
        recording
        for recording in manifest.recordings
        if recording.split in {"train", "validation"}
    )
    progress = lambda message: print(message, file=sys.stderr, flush=True)
    progress(
        f"Loading warm features for {len(development)} development recordings; "
        "protected test is not prepared"
    )
    prepared = _prepare_many(
        development,
        feature_config,
        arguments.feature_cache_dir,
        progress=progress,
    )
    if arguments.command == "diagnostics":
        report = run_multistate_oof_diagnostics(
            manifest,
            prepared,
            multistate_report_path=arguments.multistate_report,
            cache_output=arguments.oof_cache_output,
            progress=progress,
        )
    else:
        report = run_conservative_hybrid_study(
            manifest,
            prepared,
            multistate_report_path=arguments.multistate_report,
            progress=progress,
        )
    written = atomic_write_text(
        output,
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )
    print(
        json.dumps(
            {
                "report": str(written),
                "kind": report["kind"],
                "oofCacheIndex": report.get("oofCacheIndex"),
                "selectedArchitecture": report.get("promotionDecision", {}).get(
                    "selectedArchitecture"
                ),
                "testLabelsUsed": report["testLabelsUsed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
