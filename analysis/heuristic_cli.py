from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from . import HEURISTIC_METHOD, ANALYSIS_SCHEMA_VERSION
from .court import estimate_court, representative_frame, save_preview
from .ffmpeg import MediaToolError, create_proxy, probe, require_media_tools
from .rallies import DetectionSettings, detect_rallies
from .signals import audio_signal, combine_signals, motion_signal


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48] or "analysis"


def build_parser() -> argparse.ArgumentParser:
    configured_root = os.environ.get("VOLLEYCUT_DATA_ROOT", "").strip()
    default_output_root = (
        Path(configured_root).expanduser() / "analyses"
        if configured_root
        else Path("data/analyses")
    )
    parser = argparse.ArgumentParser(
        prog="analyze-no-model",
        description="Create a local review proxy and heuristic rally suggestions.",
    )
    parser.add_argument("video", type=Path, help="Source volleyball recording")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help=f"Generated analysis root (default: {default_output_root})",
    )
    parser.add_argument("--id", dest="analysis_id", help="Optional URL-safe analysis identifier")
    parser.add_argument("--title", help="Display title (defaults to the source filename)")
    parser.add_argument("--analysis-fps", type=float, default=4.0, help="Motion samples per second")
    return parser


def analyze_no_model(args: argparse.Namespace) -> Path:
    source = args.video.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Video does not exist: {source}")
    if not 1 <= args.analysis_fps <= 12:
        raise ValueError("--analysis-fps must be between 1 and 12")
    require_media_tools()

    now = datetime.now(UTC)
    requested_id = args.analysis_id or f"{slugify(source.stem)}-{now.strftime('%Y%m%d-%H%M%S')}"
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", requested_id):
        raise ValueError("--id must contain only letters, numbers, underscores, and hyphens (max 80)")
    output_root = args.output_root.resolve()
    destination = output_root / requested_id
    if destination.exists():
        raise ValueError(f"Analysis already exists: {destination}")
    staging = output_root / f".{requested_id}.incomplete"
    if staging.exists():
        raise ValueError(f"Incomplete analysis already exists: {staging}")
    staging.mkdir(parents=True)

    try:
        print(f"[1/5] Probing {source.name}")
        source_info = probe(source)
        proxy_path = staging / "proxy.mp4"
        print("[2/5] Creating normalized review proxy")
        proxy_backend = create_proxy(source, proxy_path)
        proxy_info = probe(proxy_path)

        print("[3/5] Estimating stable court lines and region")
        frame = representative_frame(proxy_path, proxy_info["duration"])
        court = estimate_court(frame)
        save_preview(frame, court, staging / "court-preview.jpg")

        print("[4/5] Measuring court motion and audio activity")
        motion_times, motion, camera_stability = motion_signal(
            proxy_path, court, args.analysis_fps
        )
        period = 1 / args.analysis_fps
        audio_times, audio = audio_signal(
            proxy_path, proxy_info["duration"], period, proxy_info["hasAudio"]
        )
        signals = combine_signals(motion_times, motion, audio_times, audio)

        print("[5/5] Suggesting conservative rally intervals")
        detection_settings = DetectionSettings()
        rallies = detect_rallies(
            signals,
            proxy_info["duration"],
            camera_stability,
            detection_settings,
        )
        warnings: list[str] = [
            "Heuristic suggestions require human review and are not trained volleyball classifications."
        ]
        if court.source == "fallback-region":
            warnings.append(
                "Court lines were inconclusive; motion used a conservative fallback region."
            )
        if not proxy_info["hasAudio"]:
            warnings.append("No audio stream was available; suggestions use court motion only.")
        if not rallies:
            warnings.append("No rally candidates passed the conservative activity thresholds.")
        if camera_stability < 0.55:
            warnings.append(
                "Significant camera movement was detected; rally boundaries may be unreliable."
            )

        payload = {
            "schemaVersion": ANALYSIS_SCHEMA_VERSION,
            "id": requested_id,
            "title": args.title or source.stem.replace("_", " ").replace("-", " ").strip().title(),
            "createdAt": now.isoformat().replace("+00:00", "Z"),
            "source": {"filename": source.name, **source_info},
            "proxy": proxy_info,
            "assets": {
                "proxyUrl": f"/api/media/{requested_id}/proxy.mp4",
                "courtPreviewUrl": f"/api/media/{requested_id}/court-preview.jpg",
            },
            "analysis": {
                "method": HEURISTIC_METHOD,
                "proxyBackend": proxy_backend,
                "analysisFps": args.analysis_fps,
                "rallyDetector": detection_settings.as_dict(),
                "cameraStability": round(camera_stability, 3),
                "warnings": warnings,
                "court": court.as_dict(),
            },
            "rallies": rallies,
            "signals": signals,
        }
        temporary_json = staging / "analysis.json.tmp"
        temporary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary_json.replace(staging / "analysis.json")
        staging.replace(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        destination = analyze_no_model(args)
    except (ValueError, MediaToolError, RuntimeError) as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1
    print(f"Analysis ready: {destination / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
