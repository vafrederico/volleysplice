#!/usr/bin/env python3
"""Install the pinned quantized person detector used by side-switch v6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.side_switch_player_detector import (
    detector_identity,
    install_pinned_model,
)


DEFAULT_DESTINATION = Path(
    "/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/third-party/"
    "opencv-zoo-mediapipe-person-int8bq-2023mar"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    model_dir = install_pinned_model(args.destination)
    print(json.dumps(detector_identity(model_dir), indent=2))


if __name__ == "__main__":
    main()
