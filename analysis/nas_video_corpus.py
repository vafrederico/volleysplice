"""Small helpers for reproducible NAS-wide diagnostic manifests.

The manifest is deliberately independent of the annotation schema.  It can
contain completed labels, unfinished human drafts, blind prelabels, and raw
videos whose rally windows came from a model-feedback artifact.  Consumers
must use ``targetStatus`` before treating a row as an evaluation target.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MANIFEST_KIND = "volleycut-full-nas-video-corpus-v1"


def load_manifest(path: Path) -> dict[str, Any]:
    """Read and minimally validate a full-NAS corpus manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != MANIFEST_KIND:
        raise ValueError(f"{path} is not a {MANIFEST_KIND} manifest")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} contains no corpus records")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}.records[{index}] must be an object")
        for field in ("recordingId", "environment", "videoPath", "rallies"):
            if field not in record:
                raise ValueError(f"{path}.records[{index}] is missing {field}")
        if not isinstance(record["rallies"], list):
            raise ValueError(f"{path}.records[{index}].rallies must be a list")
    return payload


def manifest_path_record(record: dict[str, Any]) -> Path:
    video = record.get("videoPath")
    if not isinstance(video, str) or not video:
        raise ValueError(f"{record.get('recordingId', 'record')} has no videoPath")
    return Path(video).expanduser().resolve()


def manifest_target_status(record: dict[str, Any]) -> str:
    value = record.get("targetStatus", "gold")
    return value if isinstance(value, str) and value else "gold"

