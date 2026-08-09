from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


class MediaToolError(RuntimeError):
    pass


def proxy_backend() -> str:
    return os.environ.get("VOLLEYCUT_PROXY_BACKEND", "software").strip() or "software"


def require_media_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    backend = proxy_backend()
    if backend == "jellyfin-vaapi" and shutil.which("docker") is None:
        missing.append("docker")
    if backend not in {"software", "jellyfin-vaapi"}:
        raise MediaToolError(f"Unsupported VOLLEYCUT_PROXY_BACKEND: {backend}")
    if backend == "jellyfin-vaapi" and not Path("/dev/dri/renderD128").exists():
        raise MediaToolError("The jellyfin-vaapi proxy backend requires /dev/dri/renderD128")
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


def create_proxy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backend = proxy_backend()
    if backend == "jellyfin-vaapi":
        _create_jellyfin_vaapi_proxy(source, destination)
    else:
        _create_software_proxy(source, destination)
    return backend


def _create_software_proxy(source: Path, destination: Path) -> None:
    _run([
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(source),
        "-map", "0:v:0",
        "-map", "0:a:0?",
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


def _create_jellyfin_vaapi_proxy(source: Path, destination: Path) -> None:
    device = Path("/dev/dri/renderD128")
    image = os.environ.get(
        "VOLLEYCUT_VAAPI_IMAGE",
        "jellyfin/jellyfin@sha256:17285f9cce63b3519ccad82b84497fc22482c20f8b8bbe0580d51fec5deaa6fd",
    )
    _run([
        "docker", "run", "--rm",
        "--network", "none",
        "--entrypoint", "/usr/lib/jellyfin-ffmpeg/ffmpeg",
        "--device", str(device),
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--group-add", str(device.stat().st_gid),
        "-v", f"{source.parent}:/input:ro",
        "-v", f"{destination.parent}:/output",
        image,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-hwaccel", "vaapi",
        "-hwaccel_device", str(device),
        "-hwaccel_output_format", "vaapi",
        "-i", f"/input/{source.name}",
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-vf", "scale_vaapi=w=960:h=-2:format=nv12",
        "-r", "30",
        "-fps_mode", "cfr",
        "-avoid_negative_ts", "make_zero",
        "-c:v", "h264_vaapi",
        "-qp", "24",
        "-c:a", "aac",
        "-b:a", "96k",
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        "-movflags", "+faststart",
        f"/output/{destination.name}",
    ])
