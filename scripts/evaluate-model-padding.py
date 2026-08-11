#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analysis.crop_evaluation import RecordingIntervals, evaluate_crop_padding
from analysis.schema import Interval, load_manifest


SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def analysis_predictions(
    path: Path,
    expected_id: str,
    expected_model_version: str,
) -> tuple[float, tuple[Interval, ...], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read inference output {path}: {error}") from error
    source = payload.get("source")
    analysis = payload.get("analysis")
    rows = payload.get("rallies")
    if payload.get("schemaVersion") != 1 or payload.get("id") != expected_id:
        raise ValueError(f"unexpected analysis identity in {path}")
    if not isinstance(source, dict) or not isinstance(source.get("duration"), (int, float)):
        raise ValueError(f"analysis source duration is missing in {path}")
    if not isinstance(rows, list):
        raise ValueError(f"analysis rallies are missing in {path}")
    if not isinstance(analysis, dict):
        raise ValueError(f"analysis metadata is missing in {path}")
    recorded_version = analysis.get("modelVersion")
    if recorded_version is not None and recorded_version != expected_model_version:
        raise ValueError(f"analysis model version does not match its directory in {path}")
    model_sha256 = analysis.get("modelSha256")
    if (
        not isinstance(model_sha256, str)
        or len(model_sha256) != 64
        or any(character not in "0123456789abcdef" for character in model_sha256.lower())
    ):
        raise ValueError(f"analysis model SHA-256 is missing or invalid in {path}")
    duration = float(source["duration"])
    predictions: list[Interval] = []
    previous_end = -1.0
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("start"), (int, float))
            or not isinstance(row.get("end"), (int, float))
        ):
            raise ValueError(f"invalid rally {index} in {path}")
        start, end = float(row["start"]), float(row["end"])
        if start < previous_end or start < 0 or end <= start or end > duration + 1e-6:
            raise ValueError(f"out-of-range or unordered rally {index} in {path}")
        predictions.append(Interval(start, min(end, duration)))
        previous_end = end
    return duration, tuple(predictions), model_sha256.lower()


def main() -> int:
    data_root = Path(os.environ.get("VOLLEYCUT_DATA_ROOT", "data")).expanduser().resolve()
    workspace = Path(
        os.environ.get(
            "VOLLEYCUT_LABELING_WORKSPACE",
            data_root / "labeling-v1-2026-08-09",
        )
    ).expanduser().resolve()
    parser = argparse.ArgumentParser(
        description="Measure the crop coverage/cost tradeoff from symmetric model padding."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=workspace / "manifests" / "full-gold-v1.json",
    )
    parser.add_argument("--analyses-root", type=Path, default=data_root / "analyses")
    parser.add_argument("--model-version", action="append", required=True)
    parser.add_argument("--padding-seconds", type=float, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    destination = args.output.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing report: {destination}")
    model_versions = list(dict.fromkeys(args.model_version))
    if any(not SAFE_ID.fullmatch(version) for version in model_versions):
        raise ValueError("model versions must contain only letters, digits, underscores, or hyphens")
    paddings = sorted(set(args.padding_seconds))
    if not paddings or paddings[0] < 0:
        raise ValueError("padding values must be non-negative")

    manifest_path = args.manifest.expanduser().resolve()
    analyses_root = args.analyses_root.expanduser().resolve()
    manifest = load_manifest(manifest_path, require_videos=False)
    reports: dict[str, Any] = {}
    for model_version in model_versions:
        recordings: list[RecordingIntervals] = []
        output_hashes: dict[str, str] = {}
        model_hashes: set[str] = set()
        for recording in manifest.recordings:
            analysis_id = f"model-{model_version}--{recording.id}"
            analysis_path = analyses_root / analysis_id / "analysis.json"
            duration, predictions, model_sha256 = analysis_predictions(
                analysis_path,
                analysis_id,
                model_version,
            )
            model_hashes.add(model_sha256)
            if recording.rallies and recording.rallies[-1].end > duration + 1e-6:
                raise ValueError(f"gold labels exceed inference duration for {recording.id}")
            recordings.append(
                RecordingIntervals(
                    id=recording.id,
                    split=recording.split,
                    duration=duration,
                    truth=recording.rallies,
                    predictions=predictions,
                )
            )
            output_hashes[recording.id] = sha256_file(analysis_path)

        if len(model_hashes) != 1:
            raise ValueError(
                f"inference outputs for {model_version} reference multiple model artifacts: "
                f"{sorted(model_hashes)}"
            )

        splits = sorted({recording.split for recording in recordings})
        reports[model_version] = {
            "allRecordings": evaluate_crop_padding(recordings, paddings),
            "bySplit": {
                split: evaluate_crop_padding(
                    [recording for recording in recordings if recording.split == split],
                    paddings,
                )
                for split in splits
            },
            "analysisSha256": output_hashes,
            "modelArtifactSha256": next(iter(model_hashes)),
        }

    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_path),
        "manifestFileSha256": sha256_file(manifest_path),
        "analysesRoot": str(analyses_root),
        "modelVersions": model_versions,
        "paddingPolicy": {
            "symmetricSecondsBeforeAndAfter": paddings,
            "mergeTouchingOrOverlappingCrops": True,
            "clipToVideoBounds": True,
            "note": "Padding is applied only for export evaluation; model core predictions are unchanged.",
        },
        "matching": {"minimumIntervalIoU": 0.5},
        "models": reports,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"report": str(destination), "models": model_versions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
