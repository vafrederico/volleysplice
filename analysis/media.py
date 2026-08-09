from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class NormalizationError(RuntimeError):
    pass


X264_PRESETS = {
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_video(
    source: str | Path,
    destination: str | Path,
    *,
    fps: float = 30.0,
    max_width: int = 1920,
    crf: int = 20,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    preset: str = "medium",
    threads: int = 2,
) -> dict[str, Any]:
    """Create a constant-frame-rate yuv420p master or excerpt without altering the source."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise NormalizationError("ffmpeg is required for video normalization")
    source_path = Path(source).expanduser().resolve()
    output_path = Path(destination).expanduser().resolve()
    provenance_path = output_path.with_suffix(output_path.suffix + ".provenance.json")
    if not source_path.is_file():
        raise NormalizationError(f"source video does not exist: {source_path}")
    if output_path.suffix.lower() != ".mp4":
        raise NormalizationError("normalized output must use the .mp4 extension")
    if output_path.exists() or provenance_path.exists():
        raise NormalizationError(f"refusing to overwrite normalization output: {output_path}")
    if (
        not math.isfinite(fps)
        or not 1 <= fps <= 120
        or not isinstance(max_width, int)
        or isinstance(max_width, bool)
        or max_width < 320
        or not isinstance(crf, int)
        or isinstance(crf, bool)
        or not 0 <= crf <= 51
        or preset not in X264_PRESETS
        or not isinstance(threads, int)
        or isinstance(threads, bool)
        or not 1 <= threads <= 32
        or not isinstance(start_seconds, (int, float))
        or isinstance(start_seconds, bool)
        or not math.isfinite(start_seconds)
        or start_seconds < 0
        or (
            duration_seconds is not None
            and (
                not isinstance(duration_seconds, (int, float))
                or isinstance(duration_seconds, bool)
                or not math.isfinite(duration_seconds)
                or duration_seconds <= 0
            )
        )
    ):
        raise NormalizationError(
            "invalid fps, max-width, CRF, start, duration, preset, or thread count"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_stat = source_path.stat()
    source_digest = _sha256(source_path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}-",
        suffix=".mp4",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    video_filter = f"fps={fps:g},scale=w='trunc(min({max_width},iw)/2)*2':h=-2"
    command = [
        executable,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-filter_threads",
        str(threads),
    ]
    if start_seconds > 0:
        command.extend(["-ss", f"{start_seconds:g}"])
    command.extend(
        [
            "-threads",
            str(threads),
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
        ]
    )
    if duration_seconds is not None:
        command.extend(["-t", f"{duration_seconds:g}"])
    command.extend(
        [
            "-vf",
            video_filter,
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-threads",
            str(threads),
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(temporary_path),
        ]
    )
    sidecar_descriptor, sidecar_temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}-provenance-",
        suffix=".json",
        dir=output_path.parent,
    )
    os.close(sidecar_descriptor)
    temporary_provenance = Path(sidecar_temporary_name)
    try:
        subprocess.run(command, check=True)
        after_stat = source_path.stat()
        if (
            after_stat.st_size != source_stat.st_size
            or after_stat.st_mtime_ns != source_stat.st_mtime_ns
        ):
            raise NormalizationError("source video changed while it was being normalized")
        provenance = {
            "schemaVersion": 1,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "source": {
                "filename": source_path.name,
                "sizeBytes": source_stat.st_size,
                "sha256": source_digest,
            },
            "segment": {
                "sourceStartSeconds": float(start_seconds),
                "requestedDurationSeconds": (
                    float(duration_seconds) if duration_seconds is not None else None
                ),
            },
            "normalized": {
                "filename": output_path.name,
                "sizeBytes": temporary_path.stat().st_size,
                "sha256": _sha256(temporary_path),
                "fps": fps,
                "maxWidth": max_width,
                "crf": crf,
                "preset": preset,
                "threads": threads,
            },
            "note": "Annotation timestamps refer to this normalized constant-frame-rate file.",
        }
        temporary_provenance.write_text(
            json.dumps(provenance, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
        try:
            temporary_provenance.replace(provenance_path)
        except OSError:
            output_path.unlink(missing_ok=True)
            raise
    except (OSError, subprocess.CalledProcessError, NormalizationError) as error:
        temporary_path.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)
        if isinstance(error, NormalizationError):
            raise
        raise NormalizationError(f"ffmpeg normalization failed: {error}") from error
    return {"video": str(output_path), "provenance": str(provenance_path), **provenance}
