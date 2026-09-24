#!/usr/bin/env python3
"""Freeze the user-directed grass/indoor subset of the phase1 development data.

All kept rows are inherited byte-value-identically from the pinned parent JSON.
No labels, feature values, consent, splits, or source groups are changed. Every
kept proxy, label, normalization sidecar and AV/DINO cache is hash-verified.
Only new manifest.json and dataset-audit.json output files are written.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value

from analysis.schema import load_manifest


ROOT = Path(private_value('private-reference-0057'))
PARENT_SHA256 = "d2723dd95c884a07bfe48df974975607b86afa0e540966be070db151a4ebff26"
PROTECTED_GROUP = private_value('source-group-008')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_identity(identity: dict) -> None:
    path = Path(identity["path"])
    require(path.is_file(), f"Missing frozen input: {path}")
    if "sizeBytes" in identity:
        require(path.stat().st_size == identity["sizeBytes"], f"Changed input size: {path}")
    require(sha256(path) == identity["sha256"], f"Changed frozen input bytes: {path}")


def prepare(parent_path: Path, destination: Path) -> tuple[dict, dict]:
    outputs = [destination / "manifest.json", destination / "dataset-audit.json"]
    require(not any(path.exists() for path in outputs), "Refusing to overwrite immutable subset artifacts")
    require(sha256(parent_path) == PARENT_SHA256, "Parent manifest does not match the pinned phase1 SHA-256")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_audit_path = parent_path.parent / "dataset-audit.json"
    parent_audit = json.loads(parent_audit_path.read_text(encoding="utf-8"))
    require(parent_audit["manifestSha256"] == PARENT_SHA256, "Parent audit and manifest disagree")
    # Reuse the standard structural/split/consent validator without opening any
    # excluded source video. Kept proxy bytes are fully verified below.
    load_manifest(parent_path, require_videos=False)
    records = [copy.deepcopy(row) for row in parent["recordings"] if row["environment"] in {"grass", "indoor"}]
    require(len(records) == 6, "Expected exactly six inherited grass/indoor recordings")
    groups = sorted({row["sourceGroup"] for row in records})
    require(len(groups) == 3 and PROTECTED_GROUP not in groups, "Unexpected or protected source groups")
    protected = set(parent["protectedSourceGroups"]) | {PROTECTED_GROUP}
    seen = set()
    for row in records:
        require(row["sourceGroup"] not in protected and row["split"] in {"train", "validation"},
                f"Protected or nondevelopment recording: {row['id']}")
        require(row["consent"].get("train") is True and row["consent"].get("analyze") is True,
                f"Missing explicit consent: {row['id']}")
        require(row["targetStatus"] == "exact-human-reviewed", f"Unexpected label status: {row['id']}")
        for identity in (row["labelSource"], row["normalizationProvenance"],
                         row["featureCaches"]["audiovisual"], row["featureCaches"]["dino"]):
            verify_identity(identity)
        provenance = json.loads(Path(row["normalizationProvenance"]["path"]).read_text(encoding="utf-8"))
        video = Path(row["video"])
        require(video.stat().st_size == provenance["normalized"]["sizeBytes"], f"Changed proxy size: {video}")
        require(row["contentSha256"] == provenance["normalized"]["sha256"] == sha256(video),
                f"Changed proxy bytes: {video}")
        require(row["sourceContentSha256"] == provenance["source"]["sha256"],
                f"Changed raw-source lineage: {row['id']}")
        row_digests = {row["contentSha256"], row["sourceContentSha256"]}
        require(not (row_digests & seen), f"Duplicate source derivative: {row['id']}")
        seen.update(row_digests)
        print(f"Verified inherited label, provenance, proxy, AV and DINO: {row['id']}", flush=True)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        **{key: copy.deepcopy(value) for key, value in parent.items()
           if key not in {"name", "createdAt", "recordings", "evaluationProtocol"}},
        "name": "neural-nonbeach-original-development-6-v1", "createdAt": now,
        "parentManifest": {"path": str(parent_path), "sha256": PARENT_SHA256},
        "developmentScope": "user-directed adaptive development follow-up: grass and indoor only",
        "excludedEnvironments": ["beach"],
        "evaluationProtocol": "nested source-group evaluation; beach excluded from fitting, selection, and evaluation; no random frame/rally/recording split",
        "recordings": records,
    }
    for row in records:
        original = next(value for value in parent["recordings"] if value["id"] == row["id"])
        require(row == original, "Inherited row was altered")
    manifest_text = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    audit = {
        "schemaVersion": 1, "kind": "volleycut-neural-development-audit", "createdAt": now,
        "manifestSha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "parentManifest": {"path": str(parent_path), "sha256": PARENT_SHA256},
        "parentAudit": {"path": str(parent_audit_path), "sha256": sha256(parent_audit_path)},
        "sourceManifest": parent["sourceManifest"], "recordingCount": len(records),
        "sourceGroupCount": len(groups), "rallyCount": sum(len(row["rallies"]) for row in records),
        "durationSeconds": sum(row["durationSeconds"] for row in records),
        "groups": {group: [row["id"] for row in records if row["sourceGroup"] == group] for group in groups},
        "cacheCounts": {"current104ChannelAV": len(records), "frozenDino": len(records)},
        "excluded": [{"id": row["id"], "sourceGroup": row["sourceGroup"], "environment": row["environment"],
                      "reason": "user-directed beach exclusion from fitting, selection, and evaluation"}
                     for row in parent["recordings"] if row["environment"] not in {"grass", "indoor"}],
        "inheritedExclusions": parent_audit["excluded"],
        "scopeNotes": [
            "This scope follows inspection of the earlier development study and user direction; it is an adaptive development follow-up, not a newly untouched test.",
            "Every retained recording row is exactly equal to its pinned parent row, including labels, consent, split, ignored ranges, hashes, and cache paths.",
            "No beach recordings enter training, inner selection, outer evaluation, or aggregate metrics.",
            "Three source groups imply two outer-fitting groups (four videos); each inner fit has only one training group (two videos) and one validation group.",
            "Keep the existing four model kinds, three seeds, epoch grid, and decoder grid unchanged for scope-comparable follow-up.",
            "Source hashes are inherited from immutable normalization lineage; raw masters are not rehashed.",
            "Protected-source relatives, weak export coverage, incomplete/unvalidated labels, and new intake remain excluded.",
        ],
        "inputVerification": "pinned parent manifest; all kept label/provenance/AV/DINO file hashes; full kept proxy-byte SHA-256; exact inherited rows",
        "preparationScript": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    require(sha256(parent_path) == PARENT_SHA256, "Parent changed during verification")
    destination.mkdir(parents=True, exist_ok=True)
    for path, payload in zip(outputs, (manifest_text, json.dumps(audit, indent=2, allow_nan=False) + "\n")):
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    load_manifest(outputs[0], require_videos=False)
    print(json.dumps({"manifest": str(outputs[0]), "manifestSha256": audit["manifestSha256"],
                      "recordings": audit["recordingCount"], "sourceGroups": audit["sourceGroupCount"],
                      "rallies": audit["rallyCount"], "durationSeconds": audit["durationSeconds"]}), flush=True)
    return manifest, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, default=ROOT / "2026-09-18-phase1/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "2026-09-19-nonbeach")
    args = parser.parse_args()
    prepare(args.parent_manifest, args.output_dir)


if __name__ == "__main__":
    main()
