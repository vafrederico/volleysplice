#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.ball_detector_evaluation import (
    merge_blind_review_with_detector_suggestions,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Blind-safe post-review merge of complete human labels and separately generated "
            "detector suggestions for the exact same immutable ball-presence task."
        )
    )
    parser.add_argument("--reviewed-labels", type=Path, required=True)
    parser.add_argument("--detector-proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    output = merge_blind_review_with_detector_suggestions(
        arguments.reviewed_labels,
        arguments.detector_proposals,
        arguments.output,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "mergedTask": str(output),
                "sha256": _sha256(output),
                "taskId": payload["immutable"]["taskId"],
                "reviewStatus": payload["annotations"]["review"]["status"],
                "suggestionStatus": payload["suggestions"]["status"],
                "blindMergeProvenance": payload["blindMergeProvenance"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
