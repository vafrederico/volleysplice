#!/usr/bin/env python3
"""Create a resumable visual/audio evidence pack without modifying the source video."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import wave
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def probe(video: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError(f"no video stream found in {video}")
    duration = float(payload.get("format", {}).get("duration", 0.0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid duration reported for {video}")
    return {
        "durationSeconds": duration,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "averageFrameRate": video_stream.get("avg_frame_rate"),
        "hasAudio": any(stream.get("codec_type") == "audio" for stream in streams),
        "ffprobe": payload,
    }


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def mark_complete(directory: Path) -> None:
    (directory / ".complete").write_text("complete\n", encoding="utf-8")


def extract_frames(
    video: Path,
    directory: Path,
    *,
    fps: float,
    max_width: int,
    start: float | None = None,
    duration: float | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / ".complete").is_file():
        return
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        command += ["-ss", f"{start:.3f}"]
    command += ["-i", str(video)]
    if duration is not None:
        command += ["-t", f"{duration:.3f}"]
    command += [
        "-an",
        "-vf",
        f"fps={fps:g},scale=min(iw\\,{max_width}):-2",
        "-q:v",
        "3",
        str(directory / "frame-%06d.jpg"),
    ]
    run(command)
    mark_complete(directory)


def extract_sheets(
    video: Path,
    directory: Path,
    *,
    sample_seconds: float,
    tile_columns: int,
    tile_rows: int,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / ".complete").is_file():
        return
    tile_count = tile_columns * tile_rows
    run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-an",
            "-vf",
            (
                f"fps=1/{sample_seconds:g},scale=320:-2,"
                f"tile={tile_columns}x{tile_rows}:nb_frames={tile_count}:padding=2:margin=2"
            ),
            "-fps_mode",
            "vfr",
            "-q:v",
            "3",
            str(directory / "sheet-%04d.jpg"),
        ]
    )
    mark_complete(directory)


def extract_audio(video: Path, audio_path: Path) -> None:
    marker = audio_path.with_suffix(".complete")
    if marker.is_file() and audio_path.is_file():
        return
    run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ]
    )
    marker.write_text("complete\n", encoding="utf-8")


def audio_features(audio_path: Path, output_path: Path, window_ms: int) -> int:
    if output_path.is_file():
        with output_path.open("r", encoding="utf-8", newline="") as source:
            return max(0, sum(1 for _ in source) - 1)
    rows: list[dict[str, float]] = []
    with wave.open(str(audio_path), "rb") as source:
        if source.getsampwidth() != 2 or source.getnchannels() != 1:
            raise ValueError("prepared audio must be mono signed 16-bit PCM")
        rate = source.getframerate()
        chunk_frames = max(1, round(rate * window_ms / 1000))
        previous_rms = 0.0
        index = 0
        while True:
            raw = source.readframes(chunk_frames)
            if not raw:
                break
            samples = array("h")
            samples.frombytes(raw)
            if sys.byteorder != "little":
                samples.byteswap()
            if not samples:
                break
            scale = 32768.0
            rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples)) / scale
            peak = max(abs(sample) for sample in samples) / scale
            crossings = sum(
                1
                for left, right in zip(samples, samples[1:])
                if (left < 0 <= right) or (right < 0 <= left)
            )
            zcr = crossings / max(1, len(samples) - 1)
            delta = max(0.0, rms - previous_rms)
            rows.append(
                {
                    "time": (index * chunk_frames + len(samples) / 2) / rate,
                    "rms": rms,
                    "peak": peak,
                    "zcr": zcr,
                    "positiveRmsDelta": delta,
                }
            )
            previous_rms = rms
            index += 1
    # Estimate ordinary codec/noise variation from absolute consecutive changes.
    # Using positive deltas alone often has a zero median and zero MAD, which
    # makes harmless fluctuations in a steady tone look like repeated onsets.
    changes = [
        abs(rows[index]["rms"] - rows[index - 1]["rms"])
        for index in range(1, len(rows))
    ]
    median = statistics.median(changes) if changes else 0.0
    deviations = [abs(value - median) for value in changes]
    mad = statistics.median(deviations) if deviations else 0.0
    # A nearly constant track has MAD≈0. Keep the diagnostic bounded instead of
    # presenting floating-point noise as an enormous transient score.
    denominator = max(1e-6, 1.4826 * mad)
    with output_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "timeSeconds",
                "rms",
                "peak",
                "zeroCrossingRate",
                "positiveRmsDelta",
                "transientRobustZ",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timeSeconds": f"{row['time']:.3f}",
                    "rms": f"{row['rms']:.8f}",
                    "peak": f"{row['peak']:.8f}",
                    "zeroCrossingRate": f"{row['zcr']:.8f}",
                    "positiveRmsDelta": f"{row['positiveRmsDelta']:.8f}",
                    "transientRobustZ": f"{max(-20.0, min(20.0, (row['positiveRmsDelta'] - median) / denominator)):.5f}",
                }
            )
    return len(rows)


def write_frame_index(directory: Path, fps: float, start: float = 0.0) -> int:
    frames = sorted(directory.glob("frame-*.jpg"))
    with (directory / "index.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["filename", "approximateTimeSeconds"])
        for index, frame in enumerate(frames):
            writer.writerow([frame.name, f"{start + index / fps:.3f}"])
    return len(frames)


def parse_window(value: str, duration: float) -> tuple[float, float]:
    try:
        start_text, end_text = value.split(":", 1)
        start = float(start_text)
        end = float(end_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("windows must use START:END seconds") from error
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise argparse.ArgumentTypeError("window must satisfy 0 <= START < END")
    if end > duration + 1e-6:
        raise argparse.ArgumentTypeError(f"window end {end} exceeds duration {duration:.3f}")
    return start, end


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--coarse-fps", type=float, default=1.0)
    parser.add_argument("--dense-fps", type=float, default=4.0)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--sheet-step-seconds", type=float, default=3.0)
    parser.add_argument("--audio-window-ms", type=int, default=50)
    parser.add_argument("--window", action="append", default=[], metavar="START:END")
    args = parser.parse_args()
    if args.coarse_fps <= 0 or args.dense_fps <= 0 or args.max_width < 64:
        parser.error("frame rates must be positive and max width must be at least 64")
    if args.sheet_step_seconds <= 0 or args.audio_window_ms < 10:
        parser.error("sheet step must be positive and audio window must be at least 10 ms")

    video = args.video.expanduser().resolve()
    if not video.is_file():
        parser.error(f"video does not exist: {video}")
    output = args.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = probe(video)
    source_stat = video.stat()
    identity = {
        "path": str(video),
        "sizeBytes": source_stat.st_size,
        "mtimeNs": source_stat.st_mtime_ns,
    }
    config = {
        "coarseFps": args.coarse_fps,
        "denseFps": args.dense_fps,
        "maxWidth": args.max_width,
        "sheetStepSeconds": args.sheet_step_seconds,
        "audioWindowMs": args.audio_window_ms,
    }
    manifest_path = output / "evidence.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("source") != identity or existing.get("config") != config:
            parser.error("evidence directory belongs to a different source or configuration")
        manifest = existing
    else:
        if any(output.iterdir()):
            parser.error("non-empty evidence directory has no evidence.json identity file")
        manifest = {
            "schemaVersion": 1,
            "source": identity,
            "config": config,
            "probe": {key: value for key, value in metadata.items() if key != "ffprobe"},
            "status": "preparing",
            "windows": [],
        }
        atomic_json(manifest_path, manifest)

    coarse_directory = output / "coarse-frames"
    extract_frames(
        video,
        coarse_directory,
        fps=args.coarse_fps,
        max_width=args.max_width,
    )
    coarse_count = write_frame_index(coarse_directory, args.coarse_fps)
    sheets_directory = output / "contact-sheets"
    extract_sheets(
        video,
        sheets_directory,
        sample_seconds=args.sheet_step_seconds,
        tile_columns=5,
        tile_rows=4,
    )

    feature_rows = 0
    if metadata["hasAudio"]:
        audio_path = output / "audio-mono-16khz.wav"
        extract_audio(video, audio_path)
        feature_rows = audio_features(
            audio_path,
            output / "audio-features.csv",
            args.audio_window_ms,
        )

    recorded_windows = {
        (float(row["start"]), float(row["end"]))
        for row in manifest.get("windows", [])
        if isinstance(row, dict) and "start" in row and "end" in row
    }
    for value in args.window:
        start, end = parse_window(value, metadata["durationSeconds"])
        directory = output / "dense" / f"{start:010.3f}-{end:010.3f}"
        extract_frames(
            video,
            directory,
            fps=args.dense_fps,
            max_width=args.max_width,
            start=start,
            duration=end - start,
        )
        count = write_frame_index(directory, args.dense_fps, start=start)
        if (start, end) not in recorded_windows:
            manifest.setdefault("windows", []).append(
                {"start": start, "end": end, "frames": count}
            )
            recorded_windows.add((start, end))

    manifest.update(
        {
            "status": "ready",
            "preparedAt": datetime.now(timezone.utc).isoformat(),
            "coarseFrames": coarse_count,
            "contactSheets": len(list(sheets_directory.glob("sheet-*.jpg"))),
            "audioFeatureRows": feature_rows,
        }
    )
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
