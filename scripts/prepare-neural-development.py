#!/usr/bin/env python3
"""Freeze exact development labels and validate existing AV/DINO caches.

This is a read-only audit of inputs. It never extracts features or promotes
draft/export labels. Outputs are new, exclusive-create JSON artifacts.
Run under WSL, where the original NAS paths and cache identities are valid.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value

from analysis.annotations import load_label_document
from analysis.config import FeatureConfig, NOISE_NORMALIZED_AUDIO_FEATURE_SET, feature_version_for_config
from analysis.dinov2_embeddings import load_dino_cache
from analysis.features import feature_cache_path, feature_names


PROTECTED_GROUP = private_value('source-group-008')
DEFAULT_ROOT = Path(private_value('private-reference-0059'))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_that(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_identity(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path), "sizeBytes": path.stat().st_size}


def lineage(video: Path) -> tuple[dict, dict]:
    sidecar = video.with_suffix(video.suffix + ".provenance.json")
    provenance = read_json(sidecar)
    assert_that(video.is_file(), f"Missing video: {video}")
    assert_that(provenance["normalized"]["sizeBytes"] == video.stat().st_size,
                f"Proxy size does not match provenance: {video}")
    return provenance, file_identity(sidecar)


def audiovisual_cache(row: dict, root: Path, config: FeatureConfig) -> tuple[dict, np.ndarray]:
    roi = tuple(row["roi"][key] for key in ("x", "y", "width", "height")) if row.get("roi") else None
    cache = feature_cache_path(
        row["id"], row["video"], config, roi,
        root / "features/audiovisual-audio-normalized-v3",
        content_sha256=row["contentSha256"],
    )
    assert_that(cache.is_file(), f"Missing exact current-schema AV cache: {cache}")
    with np.load(cache, allow_pickle=False) as data:
        times = np.asarray(data["times"], dtype=np.float64)
        values = np.asarray(data["values"], dtype=np.float32)
        names = tuple(str(value) for value in data["names"])
        metadata = json.loads(str(data["metadata_json"].item()))
        decoder = str(data["video_decoder"].item()) if "video_decoder" in data else "opencv-ffmpeg-v1"
    assert_that(names == feature_names(config) and len(names) == 104, f"Wrong AV schema: {cache}")
    assert_that(times.ndim == 1 and len(times) > 0 and values.shape == (len(times), 104),
                f"Wrong AV array shapes: {cache}")
    assert_that(np.isfinite(times).all() and np.isfinite(values).all() and np.all(np.diff(times) > 0),
                f"Invalid AV arrays: {cache}")
    assert_that(decoder == "opencv-ffmpeg-v1", f"Unexpected AV decoder: {cache}")
    assert_that(abs(metadata["duration"] - row["durationSeconds"]) < 0.05,
                f"AV duration mismatch: {cache}")
    assert_that(times[0] >= 0 and times[-1] < metadata["duration"], f"AV time outside video: {cache}")
    return {
        **file_identity(cache), "featureVersion": feature_version_for_config(config),
        "config": config.to_dict(), "featureNames": list(names), "shape": list(values.shape),
        "timestampsKey": "times", "valuesKey": "values", "metadataKey": "metadata_json",
        "timestampRule": "actual nearest CFR frame timestamps; do not replace with arange/4",
        "metadata": metadata, "videoDecoder": decoder,
        "identityVerification": "exact feature_cache_path(config, ROI, proxy SHA-256, absolute proxy path)",
    }, times


def dino_cache(row: dict, root: Path, times: np.ndarray) -> dict:
    candidates = list((root / "features/dinov2-vits14-v1").glob(f"{row['id']}-*/cache.npz"))
    assert_that(len(candidates) == 1, f"Expected one DINO cache for {row['id']}, got {len(candidates)}")
    cache = load_dino_cache(candidates[0], recording_id=row["id"],
                            recording_content_sha256=row["contentSha256"])
    metadata = cache.metadata
    roi = [row["roi"][key] for key in ("x", "y", "width", "height")] if row.get("roi") else None
    assert_that(metadata.get("labelsUsed") is False and metadata.get("roi") == roi,
                f"DINO provenance mismatch: {cache.path}")
    assert_that(metadata.get("preprocessing", {}).get("sampleFps") == 4.0 and
                metadata.get("preprocessing", {}).get("inputSize") == 336,
                f"DINO preprocessing mismatch: {cache.path}")
    assert_that(metadata.get("backbone", {}).get("modelName") == "dinov2_vits14",
                f"DINO backbone mismatch: {cache.path}")
    indexes = np.clip(np.searchsorted(cache.timestamps, times), 0, len(cache.timestamps) - 1)
    previous = np.clip(indexes - 1, 0, len(cache.timestamps) - 1)
    indexes = np.where(np.abs(cache.timestamps[previous] - times) <
                       np.abs(cache.timestamps[indexes] - times), previous, indexes)
    maximum_error = float(np.max(np.abs(cache.timestamps[indexes] - times)))
    assert_that(maximum_error <= 0.125 + 1e-9, f"DINO/AV timestamp mismatch: {cache.path}")
    return {
        **file_identity(cache.path), "shape": list(cache.tokens.shape),
        "timestampsKey": "timestamps", "valuesKey": "tokens", "metadataKey": "metadata_json",
        "storedDtype": "float16", "maximumAlignmentErrorSeconds": maximum_error,
        "alignment": "nearest DINO timestamp to each actual AV timestamp; tolerance 0.125 seconds",
        "metadata": metadata,
    }


def prepare(data_root: Path, verify_video_bytes: bool) -> tuple[dict, dict]:
    workspace = data_root / "labeling-v1-2026-08-09"
    source_path = workspace / "manifests/full-gold-v1.json"
    source = read_json(source_path)
    source_identity = file_identity(source_path)
    config = FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET)
    protected = {PROTECTED_GROUP} | {r["sourceGroup"] for r in source["recordings"] if r["split"] == "test"}
    protected_digests: set[str] = set()
    excluded = []
    # Only metadata and normalization lineage are read for protected recordings.
    for item in source["recordings"]:
        if item["sourceGroup"] not in protected:
            continue
        video = (source_path.parent / item["video"]).resolve()
        provenance, _ = lineage(video)
        protected_digests.update((provenance["source"]["sha256"], provenance["normalized"]["sha256"]))
        excluded.append({"id": item["id"], "sourceGroup": item["sourceGroup"], "reason": "protected source group"})
    records = []
    seen_digests: set[str] = set()
    for item in source["recordings"]:
        if item["sourceGroup"] in protected:
            continue
        assert_that(item["split"] in {"train", "validation"}, f"Unexpected development split: {item['id']}")
        assert_that(item.get("consent", {}).get("analyze") is True and
                    item.get("consent", {}).get("train") is True, f"Missing consent: {item['id']}")
        label_path = workspace / "completed/full-v1" / f"{item['id']}.labels.json"
        label = load_label_document(label_path, require_complete=True, require_video=False)
        assert_that(label.source_group == item["sourceGroup"] and label.split == item["split"],
                    f"Label identity mismatch: {item['id']}")
        for field in ("rallies", "ignoredIntervals", "hardNegatives", "sideSwitches", "annotation"):
            assert_that(label.payload.get(field, []) == item.get(field, []),
                        f"Frozen label/manifest mismatch in {field}: {item['id']}")
        video = (source_path.parent / item["video"]).resolve()
        assert_that(video == label.video, f"Label video mismatch: {item['id']}")
        provenance, provenance_identity = lineage(video)
        digest = label.payload["recording"]["contentSha256"]
        assert_that(digest == provenance["normalized"]["sha256"], f"Source hash mismatch: {item['id']}")
        row_digests = {digest, provenance["source"]["sha256"]}
        assert_that(not (row_digests & protected_digests), f"Protected source derivative: {item['id']}")
        assert_that(not (row_digests & seen_digests), f"Duplicate source derivative: {item['id']}")
        seen_digests.update(row_digests)
        if verify_video_bytes:
            assert_that(sha256(video) == digest, f"Proxy content hash mismatch: {video}")
        row = copy.deepcopy(item)
        row.update({"video": str(video), "contentSha256": digest, "sourceContentSha256": provenance["source"]["sha256"],
                    "durationSeconds": label.duration, "targetStatus": "exact-human-reviewed",
                    "labelSource": file_identity(label_path), "normalizationProvenance": provenance_identity,
                    "videoHashVerification": "full-proxy-bytes" if verify_video_bytes else "label+sidecar+size"})
        av, times = audiovisual_cache(row, workspace, config)
        dino = dino_cache(row, workspace, times)
        row["featureCaches"] = {"audiovisual": av, "dino": dino}
        records.append(row)
        print(f"Validated {row['id']}: {len(times)} ticks, 104 AV channels, DINO 10x384", flush=True)
    assert_that(len(records) == 8 and len({r['sourceGroup'] for r in records}) == 4,
                "Original development scope must contain eight recordings in four source groups")
    # Explicitly audit, but do not silently promote, later completed intake labels.
    intake = data_root / "intake-2026-08-13"
    intake_plan = read_json(intake / "manifests/intake-plan.json")
    plan_by_id = {r["id"]: r for r in intake_plan["recordings"]}
    for label_path in sorted((intake / "completed/full-v1").glob("*.labels.json")):
        document = read_json(label_path)
        metadata = document["recording"]
        plan_row = plan_by_id.get(metadata["id"], {})
        reason = ("protected source group" if metadata["sourceGroup"] in protected else
                  "outside frozen original development scope; intake plan has no explicit train consent"
                  if plan_row.get("consent", {}).get("train") is not True else
                  "outside frozen original development scope; requires separately versioned expansion")
        excluded.append({"id": metadata["id"], "sourceGroup": metadata["sourceGroup"],
                         "labelSource": file_identity(label_path), "reason": reason})
    corpus = read_json(workspace / "reports/full-nas-video-corpus-v3.json")
    for item in corpus["records"]:
        if item["sourceGroup"] in protected and item["recordingId"] not in {x["id"] for x in excluded}:
            excluded.append({"id": item["recordingId"], "sourceGroup": item["sourceGroup"],
                             "reason": "protected source-group relative despite challenge split"})
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schemaVersion": 1, "name": "neural-phase1-original-development-8-v1", "createdAt": now,
        "annotationPolicy": source["annotationPolicy"], "sourceManifest": source_identity,
        "developmentOnly": True, "protectedSourceGroups": sorted(protected),
        "evaluationProtocol": "nested source-group evaluation; no frame/rally/random-recording splits",
        "recordings": records,
    }
    audit = {
        "schemaVersion": 1, "kind": "volleycut-neural-development-audit", "createdAt": now,
        "sourceManifest": source_identity, "recordingCount": len(records),
        "sourceGroupCount": len({r['sourceGroup'] for r in records}),
        "rallyCount": sum(len(r["rallies"]) for r in records),
        "durationSeconds": sum(r["durationSeconds"] for r in records),
        "groups": {group: [r["id"] for r in records if r["sourceGroup"] == group]
                   for group in sorted({r["sourceGroup"] for r in records})},
        "cacheCounts": {"current104ChannelAV": len(records), "frozenDino": len(records)},
        "excluded": excluded,
        "exclusionPolicy": ["all protected source-group and hash-lineage relatives", "weak export coverage",
                            "unvalidated AI candidates", "incomplete/draft labels", "non-training footage"],
        "scopeNotes": ["Original split values retained as provenance; outer folds operate on whole source groups.",
                       "Ignored intervals are copied verbatim and must mask all training losses and evaluation unions.",
                       "Pilot subsets and old revisions are not additional recordings.",
                       "No video extraction, label editing, or model fitting was performed.",
                       "Source hashes come from immutable normalization provenance; raw masters were not rehashed."],
        "inputVerification": "full proxy bytes + label/provenance/cache hashes" if verify_video_bytes else
                             "label/provenance/cache hashes + proxy file size; full proxy bytes not rehashed",
        "preparationScript": file_identity(Path(__file__).resolve()),
    }
    assert_that(sha256(source_path) == source_identity["sha256"], "Source manifest changed during audit")
    for row in records:
        for identity in (row["labelSource"], row["normalizationProvenance"]):
            assert_that(sha256(Path(identity["path"])) == identity["sha256"], "Input changed during audit")
    return manifest, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_ROOT / "neural-experiments/2026-09-18-phase1")
    parser.add_argument("--skip-video-byte-hash", action="store_true",
                        help="Use already-frozen label/provenance digest plus size; record weaker verification")
    args = parser.parse_args()
    destinations = [args.output_dir / name for name in ("manifest.json", "dataset-audit.json")]
    assert_that(not any(path.exists() for path in destinations), "Refusing to overwrite immutable output artifacts")
    manifest, audit = prepare(args.data_root.resolve(), not args.skip_video_byte_hash)
    manifest_text = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    audit["manifestSha256"] = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for destination, payload in zip(destinations, (manifest_text, json.dumps(audit, indent=2, allow_nan=False) + "\n")):
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    print(json.dumps({"manifest": str(destinations[0]), "audit": str(destinations[1]),
                      "recordings": audit["recordingCount"], "groups": audit["sourceGroupCount"],
                      "rallies": audit["rallyCount"], "manifestSha256": audit["manifestSha256"]}))


if __name__ == "__main__":
    main()
