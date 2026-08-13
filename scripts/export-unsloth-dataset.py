#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.schema import load_manifest
from analysis.unsloth_dataset import (
    DEFAULT_SAMPLE_FPS,
    DEFAULT_STRIDE_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    export_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a validated VolleyCut manifest as an Unsloth video-chat JSONL dataset."
    )
    parser.add_argument("manifest", type=Path, help="Frozen VolleyCut manifest")
    parser.add_argument("output", type=Path, help="New output directory (must not exist)")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--stride-seconds", type=float, default=DEFAULT_STRIDE_SECONDS)
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument(
        "--exclude-environment",
        action="append",
        default=[],
        metavar="ENVIRONMENT",
        help="Exclude an environment (repeat for multiple environments)",
    )
    parser.add_argument(
        "--source-data-root",
        type=Path,
        help="export-machine root containing every source video",
    )
    parser.add_argument(
        "--consumer-data-root",
        help="corresponding absolute data root on the training machine",
    )
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    metadata = export_dataset(
        manifest,
        arguments.output,
        window_seconds=arguments.window_seconds,
        stride_seconds=arguments.stride_seconds,
        sample_fps=arguments.sample_fps,
        excluded_environments=arguments.exclude_environment,
        source_data_root=arguments.source_data_root,
        consumer_data_root=arguments.consumer_data_root,
    )
    print(json.dumps({"output": str(arguments.output.resolve()), **metadata["splits"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
