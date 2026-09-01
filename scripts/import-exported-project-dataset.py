#!/usr/bin/env python3
"""Import an explicitly reviewed production-project export batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.exported_project_dataset import build_exported_project_dataset


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-root", required=True, type=Path)
    value.add_argument("--output-root", required=True, type=Path)
    value.add_argument("--feedback-glob", required=True)
    value.add_argument("--source-group", required=True)
    value.add_argument(
        "--split",
        choices=("train", "validation", "test", "challenge"),
        default="challenge",
    )
    value.add_argument("--environment", default="grass")
    return value


def main() -> int:
    arguments = parser().parse_args()
    result = build_exported_project_dataset(
        arguments.source_root,
        arguments.output_root,
        feedback_glob=arguments.feedback_glob,
        source_group=arguments.source_group,
        split=arguments.split,
        environment=arguments.environment,
    )
    print(
        json.dumps(
            {
                "dataset": str((arguments.output_root / "dataset.json").resolve()),
                "recordings": len(result["records"]),
                "retainedCoreRanges": sum(
                    int(record["retainedCoreRangeCount"])
                    for record in result["records"]
                ),
                "serveEvents": sum(
                    int(record["serveEventCount"]) for record in result["records"]
                ),
                "sideSwitches": sum(
                    int(record["sideSwitchCount"]) for record in result["records"]
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
