#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.artifacts import atomic_write_text
from analysis.ball_detector_evaluation import (
    ProtocolRequirements,
    evaluate_ball_detector_tasks,
)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _threshold(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("threshold must be greater than zero and at most one")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    defaults = ProtocolRequirements()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cryptographically verified blind-merge development ball-presence tasks. "
            "Test/challenge tasks and unbound label/proposal combinations are rejected."
        )
    )
    parser.add_argument(
        "--task",
        type=Path,
        action="append",
        required=True,
        help=(
            "blind-merged task JSON or directory to search recursively; repeat for multiple "
            "inputs"
        ),
    )
    parser.add_argument(
        "--pilot-index",
        type=Path,
        help=(
            "immutable pilot index required for the coverage portion of threshold freezing"
        ),
    )
    parser.add_argument(
        "--sol-task",
        type=Path,
        action="append",
        help=(
            "completed detector-blind Sol review JSON or directory; repeat as needed. "
            "When supplied, exact coverage of every human task is required."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=_threshold,
        action="append",
        dest="thresholds",
        help=(
            "explicit confidence threshold to sweep; repeat as needed. By default every "
            "unique positive frame probability is evaluated."
        ),
    )
    parser.add_argument(
        "--selection-target",
        choices=("primary", "any"),
        default="primary",
        help="frame-presence F1 target used only for development threshold selection",
    )
    parser.add_argument("--min-recordings", type=_positive_integer, default=defaults.min_recordings)
    parser.add_argument("--min-source-groups", type=_positive_integer, default=defaults.min_source_groups)
    parser.add_argument("--min-windows", type=_positive_integer, default=defaults.min_windows)
    parser.add_argument("--min-reviewed-frames", type=_positive_integer, default=defaults.min_reviewed_frames)
    parser.add_argument(
        "--min-primary-positive-frames",
        type=_positive_integer,
        default=defaults.min_primary_positive_frames,
    )
    parser.add_argument(
        "--min-any-ball-positive-frames",
        type=_positive_integer,
        default=defaults.min_any_ball_positive_frames,
    )
    parser.add_argument(
        "--min-ball-free-frames",
        type=_positive_integer,
        default=defaults.min_ball_free_frames,
    )
    parser.add_argument(
        "--min-primary-positive-windows",
        type=_positive_integer,
        default=defaults.min_primary_positive_windows,
    )
    parser.add_argument(
        "--min-any-ball-positive-windows",
        type=_positive_integer,
        default=defaults.min_any_ball_positive_windows,
    )
    parser.add_argument(
        "--min-ball-free-windows",
        type=_positive_integer,
        default=defaults.min_ball_free_windows,
    )
    parser.add_argument(
        "--min-primary-positive-source-groups",
        type=_positive_integer,
        default=defaults.min_primary_positive_source_groups,
    )
    parser.add_argument(
        "--min-ball-free-source-groups",
        type=_positive_integer,
        default=defaults.min_ball_free_source_groups,
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    requirements = ProtocolRequirements(
        min_recordings=arguments.min_recordings,
        min_source_groups=arguments.min_source_groups,
        min_windows=arguments.min_windows,
        min_reviewed_frames=arguments.min_reviewed_frames,
        min_primary_positive_frames=arguments.min_primary_positive_frames,
        min_any_ball_positive_frames=arguments.min_any_ball_positive_frames,
        min_ball_free_frames=arguments.min_ball_free_frames,
        min_primary_positive_windows=arguments.min_primary_positive_windows,
        min_any_ball_positive_windows=arguments.min_any_ball_positive_windows,
        min_ball_free_windows=arguments.min_ball_free_windows,
        min_primary_positive_source_groups=arguments.min_primary_positive_source_groups,
        min_ball_free_source_groups=arguments.min_ball_free_source_groups,
    )
    report = evaluate_ball_detector_tasks(
        arguments.task,
        sol_task_inputs=arguments.sol_task,
        thresholds=arguments.thresholds,
        selection_target=arguments.selection_target,
        requirements=requirements,
        pilot_index_path=arguments.pilot_index,
    )
    destination = arguments.output.expanduser().resolve()
    written = atomic_write_text(
        destination,
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    print(
        json.dumps(
            {
                "report": str(written),
                "coverageSufficient": report["protocolGate"]["coverageSufficient"],
                "outOfFoldQualityPassed": report["outOfFoldQualityGate"]["passes"],
                "thresholdFreezeGatePassed": report["thresholdFreezeGate"]["passes"],
                "frozenAllDevelopmentThreshold": report["thresholdSelection"][
                    "frozenAllDevelopmentThreshold"
                ],
                "promotedThreshold": report["thresholdSelection"]["promotedThreshold"],
                "downstreamEligible": report["downstreamEligibility"]["eligible"],
                "diagnosticBestThreshold": report["thresholdSelection"][
                    "diagnosticBestThreshold"
                ],
                "frames": report["coverage"]["frames"],
                "solComparisonAvailable": report["solComparison"]["available"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
