#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def main() -> int:
    data_root = Path(os.environ.get("VOLLEYCUT_DATA_ROOT", "data")).expanduser().resolve()
    workspace = Path(
        os.environ.get(
            "VOLLEYCUT_LABELING_WORKSPACE",
            data_root / "labeling-v1-2026-08-09",
        )
    ).expanduser().resolve()
    parser = argparse.ArgumentParser(
        description="Run one immutable trained model over every recording in a manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=workspace / "manifests" / "full-gold-v1.json",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=data_root / "analyses")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    model_version = model_path.name
    if not SAFE_ID.fullmatch(model_version) or not (model_path / "model.json").is_file():
        raise ValueError(f"Model directory is invalid: {model_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    recordings = payload.get("recordings")
    if not isinstance(recordings, list):
        raise ValueError(f"Manifest has no recordings array: {manifest_path}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        recordings = recordings[: args.limit]

    completed = skipped = 0
    for row in recordings:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("Manifest contains an invalid recording")
        recording_id = row["id"]
        relative_video = row.get("video")
        roi = row.get("roi")
        if (
            not SAFE_ID.fullmatch(recording_id)
            or not isinstance(relative_video, str)
            or not isinstance(roi, dict)
            or any(not isinstance(roi.get(key), (int, float)) for key in ("x", "y", "width", "height"))
        ):
            raise ValueError(f"Recording is missing a safe id, video, or ROI: {recording_id}")
        video_path = (manifest_path.parent / relative_video).resolve()
        destination = output_root / f"model-{model_version}--{recording_id}"
        if destination.is_dir() and (destination / "analysis.json").is_file():
            print(f"Skipping existing {model_version} inference for {recording_id}")
            skipped += 1
            continue
        if destination.exists():
            raise ValueError(f"Incomplete inference destination already exists: {destination}")
        roi_value = ",".join(str(roi[key]) for key in ("x", "y", "width", "height"))
        print(f"Inferring {model_version} on {recording_id}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "analysis",
                "infer",
                "--model",
                str(model_path),
                "--video",
                str(video_path),
                "--roi",
                roi_value,
                "--title",
                recording_id,
                "--output",
                str(destination),
            ],
            check=True,
        )
        completed += 1
    print(f"Inference complete: {completed} created, {skipped} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
