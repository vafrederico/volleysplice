#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


MEDIA_SUFFIXES = {".mkv", ".mp4", ".mov", ".webm", ".m4v"}


def load_sources(data_root: Path) -> list[dict[str, str]]:
    manifest = data_root / "manifests" / "sources.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError(f"Invalid sources list in {manifest}")
    return [item for item in sources if isinstance(item, dict)]


def find_media(data_root: Path, source: dict[str, str]) -> Path | None:
    raw_dir = data_root / "raw" / source["surface"]
    matches = [
        path for path in raw_dir.iterdir()
        if path.is_file()
        and f"[{source['id']}]" in path.name
        and path.suffix.lower() in MEDIA_SUFFIXES
    ] if raw_dir.is_dir() else []
    return sorted(matches)[0] if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze complete videos in the external VolleyCut source manifest.")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("VOLLEYCUT_DATA_ROOT", "data")))
    parser.add_argument("--surface", choices=("grass", "indoor", "beach"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    sources = load_sources(data_root)
    if args.surface:
        sources = [source for source in sources if source.get("surface") == args.surface]
    if args.limit is not None:
        sources = sources[:max(0, args.limit)]

    processed = 0
    skipped = 0
    unavailable = 0
    failed = 0
    for source in sources:
        surface = source.get("surface", "unknown")
        video_id = source.get("id", "unknown")
        analysis_id = f"{surface}-{video_id}"
        if (data_root / "analyses" / analysis_id / "analysis.json").is_file():
            print(f"SKIP {analysis_id}: analysis already exists", flush=True)
            skipped += 1
            continue
        media = find_media(data_root, source)
        if media is None:
            print(f"WAIT {analysis_id}: raw media is not complete", flush=True)
            unavailable += 1
            continue
        print(f"RUN  {analysis_id}: {media.name}", flush=True)
        result = subprocess.run([
            sys.executable,
            "-m", "analysis",
            str(media),
            "--id", analysis_id,
        ])
        if result.returncode:
            failed += 1
            print(f"FAIL {analysis_id}: analyzer exited {result.returncode}", flush=True)
        else:
            processed += 1

    print(
        f"Dataset analysis summary: processed={processed} skipped={skipped} "
        f"waiting={unavailable} failed={failed}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
