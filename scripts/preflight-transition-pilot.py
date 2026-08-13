#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from analysis.transition_pilot_preflight import (
    REGISTERED_EXPERIMENTS,
    build_transition_pilot_preflight,
    write_transition_pilot_preflight,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail closed before feature preparation unless transition labels, "
            "hard negatives, immutable snapshots, and a development-only manifest are ready."
        )
    )
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--snapshot-ledger", required=True, type=Path)
    parser.add_argument("--transition-gate", required=True, type=Path)
    parser.add_argument("--candidate", required=True, choices=REGISTERED_EXPERIMENTS)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_transition_pilot_preflight(
        baseline_manifest_path=args.baseline_manifest,
        manifest_path=args.manifest,
        snapshot_ledger_path=args.snapshot_ledger,
        gate_path=args.transition_gate,
        candidate=args.candidate,
    )
    print(write_transition_pilot_preflight(args.output, report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
