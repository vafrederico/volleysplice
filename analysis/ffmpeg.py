from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


class MediaToolError(RuntimeError):
    pass


def require_media_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise MediaToolError(f"Missing required media tools: {', '.join(missing)}")


def _run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise MediaToolError(detail or f"Command failed: {command[0]}") from exc


def _fraction(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    numerator, separator, denominator = value.partition("/")
    try:
        if separator:
            return float(numerator) / float(denominator)
        return float(value)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: Path) -> dict[str, Any]:
    result = _run([
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        str(path),
    ])
    payload = json.loads(result.stdout)
    video = next((stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"), None)
    if video is None:
        raise MediaToolError(f"No video stream found in {path.name}")
    audio = next((stream for stream in payload.get("streams", []) if stream.get("codec_type") == "audio"), None)
    raw_duration = video.get("duration") or payload.get("format", {}).get("duration") or 0
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        raise MediaToolError(f"Could not determine a positive duration for {path.name}")
    return {
        "duration": duration,
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": _fraction(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        "hasAudio": audio is not None,
    }


def create_proxy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(source),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-vf", "scale=w='min(960,iw)':h=-2:flags=lanczos,fps=30,setsar=1,format=yuv420p",
        "-fps_mode", "cfr",
        "-avoid_negative_ts", "make_zero",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "24",
        "-c:a", "aac",
        "-b:a", "96k",
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        "-movflags", "+faststart",
        str(destination),
    ])
