#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.artifacts import atomic_write_text
from analysis.label_readiness import build_label_readiness_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read label JSON without mutation and report remaining feature-label debt."
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

    report = build_label_readiness_report(paths)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
        return 0

    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, rendered)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
