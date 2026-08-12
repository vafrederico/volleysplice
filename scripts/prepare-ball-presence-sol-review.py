#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.ball_detector_evaluation import prepare_detector_blind_sol_review


def _nonempty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("value must not be empty")
    return value.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a detector-blind, unreviewed Sol copy from one development sampling "
            "task. The source must contain unreviewed annotations and empty suggestions."
        )
    )
    parser.add_argument("--source-task", type=Path, required=True)
    parser.add_argument(
        "--pilot-index",
        type=Path,
        required=True,
        help="immutable pilot index whose initialTaskSha256 must match the source task",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new file in a sibling-depth directory; existing files are never overwritten",
    )
    parser.add_argument(
        "--agent-id",
        type=_nonempty,
        required=True,
        help="stable identity of the Sol labeling agent",
    )
    parser.add_argument(
        "--model-id",
        type=_nonempty,
        required=True,
        help="exact model identifier used by the agent",
    )
    parser.add_argument(
        "--run-id",
        type=_nonempty,
        required=True,
        help="unique labeling invocation/batch identifier",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    written = prepare_detector_blind_sol_review(
        arguments.source_task,
        arguments.output,
        pilot_index_path=arguments.pilot_index,
        agent_id=arguments.agent_id,
        model_id=arguments.model_id,
        run_id=arguments.run_id,
    )
    payload = json.loads(written.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "solReviewTask": str(written),
                "taskId": payload["immutable"]["taskId"],
                "reviewStatus": payload["annotations"]["review"]["status"],
                "suggestionStatus": payload["suggestions"]["status"],
                "detectorSuggestionsAbsent": payload["solReviewProvenance"][
                    "detectorSuggestionsAbsent"
                ],
                "sourceTaskSha256": payload["solReviewProvenance"]["sourceTask"][
                    "sha256"
                ],
                "preparationReceipt": payload["solReviewProvenance"][
                    "preparationReceipt"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
