#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.transition_label_gate import (
    build_transition_label_gate,
    require_transition_development_ready,
    write_transition_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read transition-cue label drafts without mutation, separate development "
            "from protected debt, and freeze a future experiment gate."
        )
    )
    parser.add_argument(
        "--labels-dir",
        action="append",
        default=[],
        type=Path,
        help="directory to scan recursively for *.labels.json (repeatable)",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        type=Path,
        help="individual label document (repeatable)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional new JSON report path; existing files are never overwritten",
    )
    parser.add_argument(
        "--require-development-ready",
        action="store_true",
        help="return a nonzero status unless every development recording has five cued rallies",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = list(args.label)
    for directory in args.labels_dir:
        if not directory.is_dir():
            raise SystemExit(f"label directory does not exist: {directory}")
        paths.extend(directory.rglob("*.labels.json"))
    if not paths:
        raise SystemExit("provide at least one --labels-dir or --label")

    report = build_transition_label_gate(paths)
    if args.output is None:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        destination = write_transition_gate(args.output, report)
        print(destination)
    if args.require_development_ready:
        try:
            require_transition_development_ready(report)
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
