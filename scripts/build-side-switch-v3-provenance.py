#!/usr/bin/env python3
"""Publish immutable source and artifact provenance for side-switch v3."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from analysis.artifacts import atomic_write_text
from analysis.side_switch_v3 import FROZEN_RECORDING_SPLIT, RECORDING_ROLE


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_MANIFEST = ROOT / "reports/full-nas-video-corpus-v1.json"
DEFAULT_DECISIONS = (
    ROOT / "reports/side-switch/appearance-review-decisions-full-nas-v1.json"
)
DEFAULT_FEATURES = ROOT / "reports/side-switch/side-switch-v3-features.json"
DEFAULT_MODEL = (
    ROOT / "models/side-switch-specialist-v3-reanchored-capped6/model.json"
)
DEFAULT_DATASET = (
    ROOT
    / "models/side-switch-specialist-v3-reanchored-capped6/dataset-development.json"
)
DEFAULT_EVALUATION = (
    ROOT
    / "reports/side-switch/side-switch-specialist-v3-reanchored-capped6-evaluation.json"
)
DEFAULT_OUTPUT = ROOT / "reports/side-switch/side-switch-v3-provenance.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _decision_map_sha256(payload: Mapping[str, Any]) -> str:
    decisions = payload.get("decisions")
    if not isinstance(decisions, Mapping):
        raise ValueError("review payload has no decisions map")
    encoded = json.dumps(
        decisions, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v3 provenance: {output}")
    revision = subprocess.run(
        ["git", "rev-parse", args.implementation_revision],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = _load(manifest_path)
    raw_records = manifest.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("corpus manifest has no records")
    records = {
        str(record["recordingId"]): record
        for record in raw_records
        if isinstance(record, Mapping)
        and str(record.get("recordingId", "")) in RECORDING_ROLE
    }
    if set(records) != set(RECORDING_ROLE):
        raise ValueError("corpus manifest does not cover the frozen v3 split")
    ordered_ids = [
        recording_id
        for role in ("train", "validation", "evaluation")
        for recording_id in FROZEN_RECORDING_SPLIT[role]
    ]
    video_paths = {
        recording_id: Path(str(records[recording_id]["videoPath"])).resolve()
        for recording_id in ordered_ids
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            recording_id: executor.submit(_sha256, video_paths[recording_id])
            for recording_id in ordered_ids
        }
        video_hashes = {
            recording_id: futures[recording_id].result()
            for recording_id in ordered_ids
        }

    sources: list[dict[str, Any]] = []
    for role in ("train", "validation", "evaluation"):
        for recording_id in FROZEN_RECORDING_SPLIT[role]:
            record = records[recording_id]
            video_path = video_paths[recording_id]
            label_value = record.get("labelPath")
            label_path = (
                Path(str(label_value)).resolve()
                if isinstance(label_value, str) and label_value
                else None
            )
            sources.append(
                {
                    "recordingId": recording_id,
                    "role": role,
                    "sourceGroup": record.get("sourceGroup"),
                    "targetStatus": record.get("targetStatus"),
                    "videoPath": str(video_path),
                    "videoFilename": record.get("videoFilename"),
                    "videoSha256": video_hashes[recording_id],
                    "sizeBytes": video_path.stat().st_size,
                    "durationSeconds": record.get("durationSeconds"),
                    "labelPath": str(label_path) if label_path is not None else None,
                    "labelSha256": (
                        _sha256(label_path) if label_path is not None else None
                    ),
                }
            )

    decisions_path = args.decisions.expanduser().resolve()
    decisions = _load(decisions_path)
    artifacts = {
        "featureArtifact": args.features.expanduser().resolve(),
        "model": args.model.expanduser().resolve(),
        "developmentDataset": args.dataset.expanduser().resolve(),
        "evaluation": args.evaluation.expanduser().resolve(),
    }
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-v3-provenance",
        "createdAt": datetime.now(UTC).isoformat(),
        "implementationRevision": revision,
        "frozenSplit": {
            role: list(recording_ids)
            for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
        },
        "sources": sources,
        "sourceManifests": {
            "corpus": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "reviewDecisions": {
                "path": str(decisions_path),
                "sha256": _sha256(decisions_path),
                "decisionMapSha256": _decision_map_sha256(decisions),
                "savedAt": decisions.get("savedAt"),
            },
        },
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in artifacts.items()
        },
        "notes": {
            "videoHash": "full-file SHA-256 computed while publishing this manifest",
            "evaluationStatus": "retrospective, not a pristine promotion test",
        },
    }
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--implementation-revision", default="HEAD")
    return parser


def main() -> None:
    payload = build(_parser().parse_args())
    print(
        json.dumps(
            {
                "implementationRevision": payload["implementationRevision"],
                "sources": len(payload["sources"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
