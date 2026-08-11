#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.ball_detector import (
    build_suggestion_index,
    infer_annotation_task,
    infer_video_sidecar,
    install_pinned_model,
)
from analysis.schema import load_manifest


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def _positive_probability(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("value must be greater than 0 and at most 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install the pinned CPU ball detector or generate unreviewed annotation "
            "suggestions for a ball-presence pilot task."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    install = subparsers.add_parser(
        "install-model",
        help="download and verify the pinned OpenCV Zoo YOLOX-s artifact",
    )
    install.add_argument("--destination", type=Path, required=True)

    infer = subparsers.add_parser(
        "infer-task",
        help="run the pinned detector on task images; output is not ground truth",
    )
    infer.add_argument("--task", type=Path, required=True)
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--score-floor", type=_positive_probability, default=0.01)
    infer.add_argument("--nms-threshold", type=_probability, default=0.5)
    infer.add_argument("--maximum-detections", type=_positive, default=20)
    infer.add_argument("--opencv-threads", type=_positive, default=6)
    infer.add_argument(
        "--limit",
        type=_positive,
        help="smoke-test only: process the first N task frames",
    )

    sidecar = subparsers.add_parser(
        "infer-recording",
        help="create a complete high-rate score sidecar for one manifest recording",
    )
    sidecar.add_argument("--manifest", type=Path, required=True)
    sidecar.add_argument("--recording-id", required=True)
    sidecar.add_argument("--model", type=Path, required=True)
    sidecar.add_argument("--output", type=Path, required=True)
    sidecar.add_argument("--detector-fps", type=float, default=15.0)
    sidecar.add_argument("--score-floor", type=_positive_probability, default=0.01)
    sidecar.add_argument("--nms-threshold", type=_probability, default=0.5)
    sidecar.add_argument("--maximum-detections", type=_positive, default=20)
    sidecar.add_argument("--opencv-threads", type=_positive, default=6)
    sidecar.add_argument(
        "--open-protected-video",
        action="store_true",
        help="required to access a test/challenge video after the pilot is frozen",
    )

    index = subparsers.add_parser(
        "index-suggestions",
        help="validate and hash every complete proposal task for a Stage-A pilot",
    )
    index.add_argument("--pilot-index", type=Path, required=True)
    index.add_argument("--suggestions", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.command == "install-model":
        installed = install_pinned_model(arguments.destination)
        print(json.dumps({"model": str(installed), "verified": True}, indent=2))
        return 0

    if arguments.command == "index-suggestions":
        output = build_suggestion_index(
            arguments.pilot_index,
            arguments.suggestions,
            arguments.output,
        )
        print(
            json.dumps(
                {
                    "artifact": str(output),
                    "status": "unreviewed-proposals-not-ground-truth",
                },
                indent=2,
            )
        )
        return 0

    if arguments.command == "infer-task":
        output = infer_annotation_task(
            arguments.task,
            arguments.model,
            arguments.output,
            score_floor=arguments.score_floor,
            nms_threshold=arguments.nms_threshold,
            maximum_detections=arguments.maximum_detections,
            opencv_threads=arguments.opencv_threads,
            limit=arguments.limit,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        status = "unreviewed-proposals-not-ground-truth"
    else:
        manifest = load_manifest(arguments.manifest)
        matches = [
            item for item in manifest.recordings if item.id == arguments.recording_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"recording id must match exactly one manifest row: {arguments.recording_id!r}"
            )
        recording = matches[0]
        if (
            recording.split in {"test", "challenge"}
            and not arguments.open_protected_video
        ):
            raise ValueError(
                "test/challenge video access requires --open-protected-video after freeze"
            )
        output = infer_video_sidecar(
            recording.video,
            arguments.model,
            arguments.output,
            recording_id=recording.id,
            source_group=recording.source_group,
            split=recording.split,
            expected_video_sha256=recording.content_sha256,
            detector_fps=arguments.detector_fps,
            score_floor=arguments.score_floor,
            nms_threshold=arguments.nms_threshold,
            maximum_detections=arguments.maximum_detections,
            opencv_threads=arguments.opencv_threads,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        status = "unvalidated-generic-detector-sidecar"
    print(
        json.dumps(
            {
                "artifact": str(output),
                "status": status,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
