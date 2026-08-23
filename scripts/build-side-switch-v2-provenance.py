#!/usr/bin/env python3
"""Publish immutable source and artifact provenance for side-switch v2."""

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
from analysis.side_switch_v2 import FROZEN_RECORDING_SPLIT, RECORDING_ROLE


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_CORPUS_MANIFEST = ROOT / "reports/full-nas-video-corpus-v1.json"
DEFAULT_FEATURES = ROOT / "reports/side-switch/side-switch-v2-features.json"
DEFAULT_DECISIONS = (
    ROOT / "reports/side-switch/appearance-review-decisions-full-nas-v1.json"
)
DEFAULT_MODEL = ROOT / "models/side-switch-specialist-v2/model.json"
DEFAULT_DATASET = (
    ROOT / "models/side-switch-specialist-v2/dataset-development.json"
)
DEFAULT_EVALUATION = (
    ROOT / "reports/side-switch/side-switch-specialist-v2-evaluation.json"
)
DEFAULT_OUTPUT = ROOT / "reports/side-switch/side-switch-v2-provenance.json"
IMPLEMENTATION_REVISION = "9382fb9"

EXPECTED_VIDEO_SHA256 = {
    "raw-no-backup-PXL_20260816_160023210": "416a678aec9d2a4f9df4f2a9eb716aad2e002e8116660946af430700cbd6faa9",
    "raw-no-backup-PXL_20260816_161923155": "4101dd5845d9f8089826c89005395d6aceedcf364b774f0361d84a1df018b4ed",
    "raw-no-backup-PXL_20260816_164327879": "5f4bc33abe4f837537ddccc441ad044550b5bd66e8983fc60ad249c809bd278e",
    "raw-no-backup-PXL_20260816_171720964": "9710033cf1e8fa503b4c92e15a17c3789549d67c31099289153be27ff19089ba",
    "raw-no-backup-PXL_20260816_180646590": "e563a2532f7a80ae8a92fc2b007587216d16fdc0da280d77fa152e6079933530",
    "raw-no-backup-PXL_20260816_183701800": "45fcdbf30e30c84220fa4fbc009777af640092dcb77f5ad690161215e5e92833",
    "raw-no-backup-PXL_20260816_190429172": "8a13bbd6ca7559e17e50e5e02b2e5452e5609a572d3c6f9100cd7c6c9ba63b5c",
    "raw-no-backup-PXL_20260816_193307688": "9c3ca0db5e99f43f3c58d81b1fb4e255666dc256a1613382c129c14caae52674",
    "raw-no-backup-PXL_20260816_203801418": "ea59bc2312a68afad66a997bc9c35deec72964afda474a4a181ad062415879f9",
    "raw-no-backup-PXL_20260816_210449857": "e1373cb4f0f7149ebd8957ac2c12d0505ce3fa576a9336b012dae6cc7affe4b3",
    "raw-no-backup-PXL_20260816_212717581": "7b9c6961519304b9300aa88d328409c9185643f67e754fb42ec26a58095773c5",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _decision_map_sha256(payload: Mapping[str, Any]) -> str:
    decisions = payload.get("decisions")
    if not isinstance(decisions, Mapping):
        raise ValueError("review payload has no decisions map")
    encoded = json.dumps(
        decisions, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sampled_fingerprint(feedback_path: Path) -> dict[str, Any]:
    payload = _load_json(feedback_path)
    source = payload.get("source")
    file_value = source.get("file") if isinstance(source, Mapping) else None
    if not isinstance(file_value, Mapping):
        raise ValueError(f"model feedback has no source file identity: {feedback_path}")
    return {
        "sizeBytes": int(file_value["sizeBytes"]),
        "lastModifiedMs": int(file_value["lastModifiedMs"]),
        "sampledFingerprint": str(file_value["sampledFingerprint"]),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v2 provenance: {output}")
    revision = subprocess.run(
        ["git", "rev-parse", args.implementation_revision],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest_path = args.corpus_manifest.expanduser().resolve()
    manifest = _load_json(manifest_path)
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
        raise ValueError("corpus manifest does not cover the frozen v2 split")

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
            feedback_path = Path(str(record["labelPath"])).resolve()
            actual_sha = video_hashes[recording_id]
            expected_sha = EXPECTED_VIDEO_SHA256[recording_id]
            if actual_sha != expected_sha:
                raise ValueError(f"video bytes changed for {recording_id}")
            sources.append(
                {
                    "recordingId": recording_id,
                    "role": role,
                    "sourceGroup": record.get("sourceGroup"),
                    "videoPath": str(video_path),
                    "videoFilename": record.get("videoFilename"),
                    "videoSha256": actual_sha,
                    **_sampled_fingerprint(feedback_path),
                    "durationSeconds": record.get("durationSeconds"),
                    "feedbackPath": str(feedback_path),
                    "feedbackSha256": _sha256(feedback_path),
                }
            )

    decisions_path = args.decisions.expanduser().resolve()
    decisions = _load_json(decisions_path)
    artifacts = {
        "featureArtifact": args.features.expanduser().resolve(),
        "model": args.model.expanduser().resolve(),
        "developmentDataset": args.dataset.expanduser().resolve(),
        "evaluation": args.evaluation.expanduser().resolve(),
    }
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-v2-provenance",
        "createdAt": datetime.now(UTC).isoformat(),
        "implementationRevision": revision,
        "frozenSplit": {
            role: list(recording_ids)
            for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
        },
        "sources": sources,
        "sourceManifests": {
            "corpus": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            },
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
            "videoHash": "full-file SHA-256, verified while publishing this manifest",
            "decisionIdentity": (
                "decisionMapSha256 excludes mutable savedAt metadata; sha256 binds the "
                "exact file used for the frozen run"
            ),
        },
    }
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--implementation-revision", default=IMPLEMENTATION_REVISION
    )
    return parser


def main() -> None:
    payload = build(_parser().parse_args())
    print(json.dumps({"sources": len(payload["sources"])}, indent=2))


if __name__ == "__main__":
    main()
