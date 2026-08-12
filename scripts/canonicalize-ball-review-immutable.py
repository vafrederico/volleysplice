#!/usr/bin/env python3
"""Safely rebuild review artifacts with pristine immutable JSON number lexemes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.ball_review_repair import canonicalize_ball_review_immutables


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild ball-review JSON into a new sibling directory, replacing only each "
            "immutable value with its SHA-pinned pristine-task JSON."
        )
    )
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument(
        "--reviews",
        type=Path,
        help="Source review directory (default: <pilot-root>/reviews)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New sibling directory; it must not already exist",
    )
    arguments = parser.parse_args()
    receipt = canonicalize_ball_review_immutables(
        arguments.pilot_root,
        arguments.output,
        reviews_directory=arguments.reviews,
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
