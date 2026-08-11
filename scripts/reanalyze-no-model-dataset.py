#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from analysis import METHOD
from analysis.court import estimate_court, representative_frame, save_preview
from analysis.rallies import DetectionSettings, detect_rallies
from analysis.signals import combine_signals, motion_signal


ANALYSIS_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
SUFFIX = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,19}")


def _roi(payload: dict[str, Any]) -> tuple[float, float, float, float]:
    return tuple(float(payload[key]) for key in ("x", "y", "width", "height"))  # type: ignore[return-value]


def _same_roi(first: tuple[float, ...], second: tuple[float, ...]) -> bool:
    return all(abs(left - right) <= 0.0002 for left, right in zip(first, second, strict=True))


def _warnings(
    *,
    court_source: str,
    has_audio: bool,
    camera_stability: float,
    rallies: list[dict[str, object]],
) -> list[str]:
    warnings = [
        "Heuristic suggestions require human review and are not trained volleyball classifications."
    ]
    if court_source == "fallback-region":
        warnings.append(
            "Court lines were inconclusive or implausibly narrow; motion used a broad fallback region."
        )
    if not has_audio:
        warnings.append("No audio stream was available; suggestions use court motion only.")
    if not rallies:
        warnings.append("No rally candidates passed the conservative activity thresholds.")
    if camera_stability < 0.55:
        warnings.append("Significant camera movement was detected; rally boundaries may be unreliable.")
    return warnings


def reanalyze_no_model(source_directory: Path, destination: Path) -> Path:
    source_json = source_directory / "analysis.json"
    source = json.loads(source_json.read_text(encoding="utf-8"))
    proxy_path = source_directory / "proxy.mp4"
    if not proxy_path.is_file():
        raise ValueError(f"analysis proxy is missing: {proxy_path}")
    if destination.exists():
        raise ValueError(f"analysis already exists: {destination}")
    staging = destination.parent / f".{destination.name}.incomplete"
    if staging.exists():
        raise ValueError(f"incomplete analysis already exists: {staging}")
    staging.mkdir(parents=True)

    try:
        duration = float(source["proxy"]["duration"])
        analysis_fps = float(source["analysis"]["analysisFps"])
        frame = representative_frame(proxy_path, duration)
        court = estimate_court(frame)
        save_preview(frame, court, staging / "court-preview.jpg")

        old_signals = source.get("signals", [])
        old_roi = _roi(source["analysis"]["court"]["roi"])
        if _same_roi(old_roi, court.roi):
            signals = old_signals
            camera_stability = float(source["analysis"]["cameraStability"])
            signal_source = "reused"
        else:
            print(
                f"RECOMPUTE {source['id']}: court ROI {old_roi} -> {court.roi}",
                flush=True,
            )
            motion_times, motion, camera_stability = motion_signal(
                proxy_path,
                court,
                analysis_fps,
            )
            audio_times = np.asarray([sample["time"] for sample in old_signals], dtype=np.float64)
            audio = np.asarray([sample["audio"] for sample in old_signals], dtype=np.float64)
            signals = combine_signals(motion_times, motion, audio_times, audio)
            signal_source = "recomputed-court-roi"

        settings = DetectionSettings()
        rallies = detect_rallies(signals, duration, camera_stability, settings)
        destination_id = destination.name
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        payload = {
            **source,
            "id": destination_id,
            "title": f"{source['title']} — analyzer v2",
            "createdAt": now,
            "assets": {
                "proxyUrl": f"/api/media/{destination_id}/proxy.mp4",
                "courtPreviewUrl": f"/api/media/{destination_id}/court-preview.jpg",
            },
            "analysis": {
                **source["analysis"],
                "method": METHOD,
                "derivedFrom": str(source["id"]),
                "signalSource": signal_source,
                "rallyDetector": settings.as_dict(),
                "cameraStability": round(camera_stability, 3),
                "warnings": _warnings(
                    court_source=court.source,
                    has_audio=bool(source["proxy"].get("hasAudio")),
                    camera_stability=camera_stability,
                    rallies=rallies,
                ),
                "court": court.as_dict(),
            },
            "rallies": rallies,
            "signals": signals,
        }
        try:
            os.link(proxy_path, staging / "proxy.mp4")
        except OSError:
            shutil.copy2(proxy_path, staging / "proxy.mp4")
        temporary_json = staging / "analysis.json.tmp"
        temporary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary_json.replace(staging / "analysis.json")
        staging.replace(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-run court validation and rally decoding from immutable analysis proxies."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("VOLLEYCUT_DATA_ROOT", "data")),
    )
    parser.add_argument("--suffix", default="v2")
    parser.add_argument("analysis_ids", nargs="*")
    arguments = parser.parse_args()
    if not SUFFIX.fullmatch(arguments.suffix):
        parser.error("--suffix must contain only letters, numbers, underscores, and hyphens")
    invalid_ids = [item for item in arguments.analysis_ids if not ANALYSIS_ID.fullmatch(item)]
    if invalid_ids:
        parser.error(f"invalid analysis ID: {invalid_ids[0]}")
    data_root = arguments.data_root.expanduser().resolve()
    analyses_root = data_root / "analyses"
    if arguments.analysis_ids:
        source_directories = [analyses_root / item for item in arguments.analysis_ids]
    else:
        source_directories = sorted(
            path.parent
            for path in analyses_root.glob("*/analysis.json")
            if not path.parent.name.endswith(f"-{arguments.suffix}")
        )

    completed = 0
    skipped = 0
    for source_directory in source_directories:
        destination = analyses_root / f"{source_directory.name}-{arguments.suffix}"
        if not ANALYSIS_ID.fullmatch(destination.name):
            raise ValueError(f"derived analysis ID is too long or invalid: {destination.name}")
        if destination.exists():
            print(f"SKIP {destination.name}: analysis already exists", flush=True)
            skipped += 1
            continue
        print(f"RUN  {source_directory.name} -> {destination.name}", flush=True)
        result = reanalyze_no_model(source_directory, destination)
        print(f"DONE {result / 'analysis.json'}", flush=True)
        completed += 1
    print(f"Dataset re-analysis summary: completed={completed} skipped={skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
