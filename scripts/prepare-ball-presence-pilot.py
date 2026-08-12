#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.ball_annotation import prepare_ball_presence_pilot


def _defaults() -> tuple[Path, Path]:
    data_root = Path(os.environ.get("VOLLEYCUT_DATA_ROOT", "data")).expanduser()
    workspace = Path(
        os.environ.get(
            "VOLLEYCUT_LABELING_WORKSPACE",
            data_root / "labeling-v1-2026-08-09",
        )
    ).expanduser()
    return (
        workspace / "manifests" / "full-gold-v1.json",
        workspace / "ball-presence-pilot" / "round-01",
    )


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    default_manifest, default_output = _defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic, development-only ball-presence annotation tasks: "
            "six 3-second windows per recording and exact 15 fps proxy frames."
        )
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--round", dest="round_index", type=_positive, default=1)
    parser.add_argument(
        "--expected-recordings",
        type=_positive,
        default=8,
        help="fail closed unless this many train/validation recordings are present",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    report = prepare_ball_presence_pilot(
        arguments.manifest,
        arguments.output,
        round_index=arguments.round_index,
        expected_recordings=arguments.expected_recordings,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
