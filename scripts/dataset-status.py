#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


VIDEO_ID = re.compile(r"\[([A-Za-z0-9_-]{11})\]")
MEDIA_SUFFIXES = {".mkv", ".mp4", ".mov", ".webm", ".m4v"}


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return "0 B"


def format_duration(seconds: Any) -> str:
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "—"
    minutes, remainder = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{remainder:02d}" if hours else f"{minutes}:{remainder:02d}"


def video_id_from_name(name: str) -> str | None:
    match = VIDEO_ID.search(name)
    return match.group(1) if match else None


def selected_resolution(info: dict[str, Any]) -> str:
    requested = info.get("requested_formats")
    formats = requested if isinstance(requested, list) else [info]
    dimensions = [
        (item.get("width"), item.get("height"))
        for item in formats
        if isinstance(item, dict) and item.get("width") and item.get("height")
    ]
    if not dimensions:
        return "—"
    width, height = max(dimensions, key=lambda pair: int(pair[0]) * int(pair[1]))
    return f"{width}×{height}"


def collect_analyses(root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return results
    for path in root.glob("*/analysis.json"):
        payload = load_json(path)
        if not payload:
            continue
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        video_id = video_id_from_name(str(source.get("filename", "")))
        if not video_id:
            continue
        created = str(payload.get("createdAt", ""))
        if video_id not in results or created > str(results[video_id].get("createdAt", "")):
            results[video_id] = payload
    return results


def build_status(data_root: Path) -> str:
    source_manifest = load_json(data_root / "manifests" / "sources.json") or {}
    sources = source_manifest.get("sources") if isinstance(source_manifest.get("sources"), list) else []
    analyses = collect_analyses(data_root / "analyses")
    rows: list[list[str]] = []
    complete_count = 0
    analyzed_count = 0
    total_bytes = 0

    for source in sources:
        if not isinstance(source, dict):
            continue
        video_id = str(source.get("id", ""))
        surface = str(source.get("surface", "unknown"))
        raw_dir = data_root / "raw" / surface
        info_paths = [path for path in raw_dir.glob("*.info.json") if video_id_from_name(path.name) == video_id]
        info = load_json(info_paths[0]) if info_paths else None
        media = [
            path for path in raw_dir.iterdir()
            if path.is_file() and video_id_from_name(path.name) == video_id and path.suffix.lower() in MEDIA_SUFFIXES
        ] if raw_dir.is_dir() else []
        partials = [path for path in raw_dir.glob("*.part") if video_id_from_name(path.name) == video_id]
        if media:
            raw_status = "complete"
            size = sum(path.stat().st_size for path in media)
            complete_count += 1
        elif partials:
            raw_status = "partial"
            size = sum(path.stat().st_size for path in partials)
        else:
            raw_status = "missing"
            size = 0
        total_bytes += size

        analysis = analyses.get(video_id)
        if analysis:
            analyzed_count += 1
            detail = analysis.get("analysis") if isinstance(analysis.get("analysis"), dict) else {}
            court = detail.get("court") if isinstance(detail.get("court"), dict) else {}
            rallies = analysis.get("rallies") if isinstance(analysis.get("rallies"), list) else []
            source_detail = analysis.get("source") if isinstance(analysis.get("source"), dict) else {}
            duration = float(source_detail.get("duration", 0) or 0)
            core_seconds = sum(
                max(0, float(rally.get("end", 0)) - float(rally.get("start", 0)))
                for rally in rallies
                if isinstance(rally, dict)
            )
            coverage = core_seconds / duration if duration > 0 else 0
            backend = str(detail.get("proxyBackend") or "software")
            analysis_status = (
                f"{len(rallies)} candidates; {coverage:.0%} core; "
                f"court {float(court.get('confidence', 0)):.0%}; {backend}"
            )
        else:
            analysis_status = "pending"

        rows.append([
            surface,
            video_id,
            str(info.get("title", "—")) if info else "—",
            format_duration(info.get("duration")) if info else "—",
            selected_resolution(info) if info else "—",
            f"{raw_status} ({human_size(size)})",
            analysis_status,
        ])

    generated = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    lines = [
        "# VolleyCut dataset status",
        "",
        f"Generated: `{generated}`",
        "",
        f"- Raw downloads complete: **{complete_count}/{len(rows)}**",
        f"- Analyses complete: **{analyzed_count}/{len(rows)}**",
        f"- Current raw bytes (including partials): **{human_size(total_bytes)}**",
        "",
        "| Surface | YouTube ID | Title | Duration | Resolution | Raw | Analysis |",
        "|---|---|---|---:|---:|---|---|",
    ]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    lines.extend([
        "",
        "Confidence is heuristic and uncalibrated. Candidate counts are not verified rally counts.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a Markdown status report for the external VolleyCut dataset.")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("VOLLEYCUT_DATA_ROOT", "data")))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    output = args.output or data_root / "manifests" / "status.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_status(data_root), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
