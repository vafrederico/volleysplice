#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from analysis.component_selector_study import (
    DEFAULT_SELECTOR_L2,
    prepare_component_selector_oof,
    run_component_selector_development,
    write_development_report,
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


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    default_manifest, workspace = _workspace_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Prepare fresh source-group-OOF v4/v5 candidates, then train and "
            "assess the leakage-audited overlap-component selector."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare-oof",
        help=(
            "fit fold-specific rally/serve generators from warm caches; protected "
            "test is never prepared"
        ),
    )
    prepare.add_argument("--manifest", type=Path, default=default_manifest)
    prepare.add_argument(
        "--v4-rally-template",
        type=Path,
        default=workspace / "models" / "full-audiovisual-v2-final",
    )
    prepare.add_argument(
        "--v4-serve-template",
        type=Path,
        default=workspace / "models" / "serve-specialist-audiovisual-v4",
    )
    prepare.add_argument(
        "--v4-cache-dir",
        type=Path,
        default=workspace / "features" / "audiovisual-v2",
    )
    prepare.add_argument(
        "--v5-rally-template",
        type=Path,
        default=workspace / "models" / "full-audiovisual-audio-normalized-v3",
    )
    prepare.add_argument(
        "--v5-serve-template",
        type=Path,
        default=workspace / "models" / "serve-specialist-audio-normalized-v5",
    )
    prepare.add_argument(
        "--v5-cache-dir",
        type=Path,
        default=workspace / "features" / "audiovisual-audio-normalized-v3",
    )
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument(
        "--epoch-cap",
        type=_nonnegative,
        default=60,
        help="maximum epochs per fold-specific head; 0 reuses each template's full limit",
    )
    prepare.add_argument("--seed", type=int, default=7)

    development = commands.add_parser(
        "development",
        help=(
            "fit on training OOF component rows and assess on the generator-held "
            "validation source group"
        ),
    )
    development.add_argument("--manifest", type=Path, default=default_manifest)
    development.add_argument("--oof-cache", type=Path, required=True)
    development.add_argument("--output", type=Path, required=True)
    development.add_argument(
        "--selector-l2", type=_positive_float, default=DEFAULT_SELECTOR_L2
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    progress = lambda message: print(message, file=sys.stderr, flush=True)
    if arguments.command == "prepare-oof":
        output_dir = arguments.output_dir.expanduser().resolve()
        if output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite existing OOF cache: {output_dir}"
            )
        payload = prepare_component_selector_oof(
            arguments.manifest,
            arguments.v4_rally_template,
            arguments.v4_serve_template,
            arguments.v4_cache_dir,
            arguments.v5_rally_template,
            arguments.v5_serve_template,
            arguments.v5_cache_dir,
            output_dir,
            epoch_cap=arguments.epoch_cap,
            seed=arguments.seed,
            progress=progress,
        )
        summary = {
            "oofCache": str(output_dir),
            "recordings": len(payload["recordings"]),
            "folds": len(payload["folds"]),
            "testLabelsUsed": payload["testLabelsUsed"],
            "testRecordingsPrepared": payload["testRecordingsPrepared"],
        }
    else:
        output = arguments.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing report: {output}")
        report = run_component_selector_development(
            arguments.manifest,
            arguments.oof_cache,
            selector_l2=arguments.selector_l2,
        )
        written = write_development_report(output, report)
        assessment = report["assessment"]["variants"]
        summary = {
            "report": str(written),
            "assessmentRole": report["assessmentRole"],
            "testLabelsUsed": report["testLabelsUsed"],
            "v4EventF1": assessment["v4"]["aggregate"]["eventF1"],
            "frozenIntersectionEventF1": assessment["frozenIntersection"][
                "aggregate"
            ]["eventF1"],
            "trainedSelectorEventF1": assessment["trainedSelector"]["aggregate"][
                "eventF1"
            ],
        }
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
