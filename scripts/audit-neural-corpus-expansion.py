#!/usr/bin/env python3
"""Inventory current label revisions and raw/project lineage; never train or relabel.

Writes only a new audit JSON/Markdown pair. Original labels, caches, and frozen
experiments are read-only. Run with the WSL analysis Python environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value
from analysis.annotations import load_label_document
from analysis.config import FeatureConfig, NOISE_NORMALIZED_AUDIO_FEATURE_SET
from analysis.features import feature_cache_path, feature_names

PROTECTED = private_value('source-group-008')


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sizeBytes": path.stat().st_size}


def fingerprint(path):
    size = path.stat().st_size
    h = hashlib.sha256(struct.pack("<Q", size))
    with path.open("rb") as f:
        if size <= 2 * 1024 * 1024:
            h.update(f.read())
        else:
            h.update(f.read(1024 * 1024))
            f.seek(size - 1024 * 1024)
            h.update(f.read(1024 * 1024))
    return "sampled-sha256-v1:" + h.hexdigest()


def summarize(rows):
    return {"recordings": len(rows), "intervals": sum(r["intervalCount"] for r in rows),
            "sourceGroups": sorted({r["sourceGroup"] for r in rows}),
            "sourceGroupCount": len({r["sourceGroup"] for r in rows}),
            "durationSeconds": sum(r.get("durationSeconds", 0) for r in rows),
            "ids": [r["id"] for r in rows]}


def audit(root):
    main = root / "labeling-v1-2026-08-09"
    catalog_path = main / "reports/full-nas-video-corpus-v3.json"
    catalog = read(catalog_path)
    rows = {}
    for r in catalog["records"]:
        rows[r["recordingId"]] = {
            "id": r["recordingId"], "sourceGroup": r["sourceGroup"],
            "environment": r["environment"], "durationSeconds": r["durationSeconds"],
            "intervalCount": len(r.get("rallies", [])), "tier": r["targetStatus"],
            "catalogClassification": r["targetStatus"], "catalogSourceType": r["sourceType"],
            "videoPath": r.get("videoPath"), "consentEvidence": [], "labelRevisions": [],
        }
    manifest_paths = [main / "manifests/full-gold-v1.json"]
    for p in (root / "intake-2026-08-13/experiments/environment-specialists-v2/manifests").glob("*.json"):
        manifest_paths.append(p)
    for intake in ("intake-2026-08-13", "intake-2026-08-25-shoreline-kb"):
        manifest_paths.extend((root / intake / "manifests").glob("*.json"))
    manifests = [(p, read(p)) for p in sorted(set(manifest_paths))]
    selected = {}
    paths = []
    # Priority expresses workflow authority, not modification time: completed
    # current documents override draft copies; task templates are the last resort.
    for base in (main, root / "intake-2026-08-13", root / "intake-2026-08-25-shoreline-kb"):
        for relative, priority in (("tasks/full", 1), ("labels/full", 2),
                                   ("completed/full-v1", 3), ("labels/full-v3", 4),
                                   ("completed/full-v3", 5)):
            paths.extend((p, priority) for p in (base / relative).glob("*.labels.json"))
    for p, priority in sorted(paths):
        d = read(p)
        rec = d["recording"]
        rid = rec["id"]
        row = rows.setdefault(rid, {"id": rid, "consentEvidence": [], "labelRevisions": []})
        row["labelRevisions"].append({**identity(p), "annotation": d.get("annotation", {}),
                                      "intervalCount": len(d.get("rallies", []))})
        if priority >= selected.get(rid, (0, None))[0]:
            selected[rid] = (priority, p)
    config = FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET)
    cache_roots = [main, root / "intake-2026-08-13/experiments/environment-specialists-v2",
                   root / "intake-2026-08-25-shoreline-kb",
                   root / private_value('private-reference-0063')]
    for rid, (_, p) in selected.items():
        d = read(p)
        rec = d["recording"]
        a = d.get("annotation", {})
        row = rows[rid]
        row.update({"sourceGroup": rec["sourceGroup"], "environment": rec["environment"],
                    "durationSeconds": rec["durationSeconds"], "annotation": a,
                    "intervalCount": len(d.get("rallies", [])), "labelSource": identity(p),
                    "labelConsent": rec.get("consent"), "ignoredIntervalCount": len(d.get("ignoredIntervals", [])),
                    "hardNegativeCount": len(d.get("hardNegatives", [])),
                    "serveMarkerCount": len(d.get("serveMarkers", [])),
                    "sideSwitchCount": len(d.get("sideSwitches", [])),
                    "contentSha256": rec.get("contentSha256")})
        weak = rec.get("capture", {}).get("targetStatus") == "reviewed-export-coverage" or "weak coverage" in a.get("notes", "").lower()
        if weak:
            row["tier"] = "reviewed-export-coverage"
        elif a.get("status") == "complete" and a.get("continuousVideoReviewed"):
            row["tier"] = "completed-exact"
        elif a.get("continuousVideoReviewed") and a.get("annotator", "").lower() in {"v", "vini"}:
            row["tier"] = "human-continuously-reviewed-draft"
        elif a.get("annotator", "").lower() in {"v", "vini"}:
            row["tier"] = "human-partially-reviewed-draft"
        else:
            row["tier"] = "unvalidated-candidate"
        if rec["sourceGroup"] == PROTECTED:
            row["excludedReason"] = "protected group and all source derivatives"
            continue
        if rec["environment"] == "beach":
            row["excludedReason"] = "user-directed no-beach scope"
            continue
        try:
            label = load_label_document(p, require_complete=False, require_video=False)
            if not label.video.is_file():
                raise ValueError(f"Missing source video: {label.video}")
            row["videoPath"] = str(label.video)
            row["structuralValidation"] = "passed existing label loader (not an independent boundary review)"
        except Exception as e:
            row["structuralValidation"] = str(e)
            continue
        provenance = label.video.with_suffix(label.video.suffix + ".provenance.json")
        if provenance.exists():
            prov = read(provenance)
            row["normalizationProvenance"] = identity(provenance)
            row["rawSource"] = prov.get("source")
            row["proxyProvenanceMatchesLabel"] = prov.get("normalized", {}).get("sha256") == rec.get("contentSha256")
        # Manifest loading canonicalizes ROI coordinates to float. JSON ints have
        # a different cache-key spelling even for numerically identical ROIs.
        roi = tuple(float(rec["roi"][k]) for k in ("x", "y", "width", "height")) if rec.get("roi") else None
        caches = []
        for cache_root in cache_roots:
            cp = feature_cache_path(rid, str(label.video), config, roi,
                                    cache_root / "features/audiovisual-audio-normalized-v3",
                                    content_sha256=rec.get("contentSha256"))
            if not cp.exists():
                continue
            with np.load(cp, allow_pickle=False) as n:
                valid = tuple(n["names"].tolist()) == feature_names(config) and n["values"].shape == (len(n["times"]), 104)
                valid = valid and np.isfinite(n["values"]).all() and np.all(np.diff(n["times"]) > 0)
                metadata = json.loads(str(n["metadata_json"].item()))
                caches.append({**identity(cp), "validCurrent104Schema": bool(valid),
                               "shape": list(n["values"].shape), "durationSeconds": metadata["duration"],
                               "identity": "exact current config+ROI+stored source hash+absolute video path cache key"})
        row["currentAvCaches"] = caches
        row["dinoCachePaths"] = [str(cp) for cp in (main / "features/dinov2-vits14-v1").glob(f"{rid}-*/cache.npz")]
    for p, d in manifests:
        for rec in d.get("recordings", []):
            if rec.get("id") in rows:
                rows[rec["id"]]["consentEvidence"].append({"path": str(p), "manifestName": d.get("name"),
                    "role": d.get("experiment", {}).get("role"), "consent": rec.get("consent"),
                    "split": rec.get("split")})
    for row in rows.values():
        row["hasAffirmativeTrainingManifest"] = any(e.get("consent", {}).get("train") is True for e in row["consentEvidence"] if e.get("consent"))
        if row["sourceGroup"] == PROTECTED:
            row["excludedReason"] = "protected group and all source derivatives"
        if row["environment"] == "beach":
            row["excludedReason"] = "user-directed no-beach scope"
    # Raw/project duplicates are collapsed by source name and sampled identity.
    raw_root = root.parent / "volleycut-raw-no-backup"
    feedback_paths = list(raw_root.glob("*.model-feedback*.json")) + list((root / "model-feedback").glob("*/bundle.json"))
    by_source = defaultdict(list)
    for p in feedback_paths:
        d = read(p)
        by_source[d["source"]["file"]["name"]].append((p, d))
    projects = []
    for name, versions in sorted(by_source.items()):
        p, d = max(versions, key=lambda v: (v[1].get("corrections", {}).get("updatedAt", ""), v[1].get("generatedAt", ""), str(v[0])))
        source, correction = d["source"], d["corrections"]
        raw = raw_root / name
        actual_fingerprint = fingerprint(raw)
        cuts = correction.get("correctedRanges", [])
        labels = correction.get("labels", {})
        key = "raw-no-backup-" + raw.stem
        editlists = [ep for ep in raw_root.glob(raw.stem + "*edit-list*.json")]
        ranges = d.get("finalExportIntervals", [])
        export_map = []
        offset = 0.
        for interval in ranges:
            duration = interval["end"] - interval["start"]
            export_map.append({"outputStart": offset, "outputEnd": offset + duration,
                               "sourceStart": interval["start"], "sourceEnd": interval["end"]})
            offset += duration
        entry = {"id": key, "sourceName": name, "projectId": source.get("projectId"),
                 "rawPath": str(raw), "rawSizeBytes": raw.stat().st_size,
                 "durationSeconds": source["media"]["duration"], "feedback": identity(p),
                 "feedbackRevisions": [identity(vp) for vp, _ in versions],
                 "sampledFingerprint": actual_fingerprint,
                 "sourceIdentityVerified": actual_fingerprint == source["file"]["sampledFingerprint"] and raw.stat().st_size == source["file"]["sizeBytes"],
                 "projectSchemaVersion": d["schemaVersion"], "coreCutCount": len(cuts),
                 "includedCoreCutCount": sum(bool(c.get("included")) for c in cuts),
                 "manualIncludedCutCount": sum(c.get("origin") == "manual" and bool(c.get("included")) for c in cuts),
                 "correctionLabelCounts": {k: len(v) for k, v in labels.items() if isinstance(v, list)},
                 "userTouchedCutCount": len(correction.get("userTouchedCutIds", [])),
                 "ignoredIntervals": correction.get("ignoredIntervals", []),
                 "padding": {k: correction.get(k) for k in ("beforePaddingSeconds", "afterPaddingSeconds", "joinGapSeconds")},
                 "featuresEmbedded": bool(d.get("features")),
                 "featureRuntime": source.get("runtimeVariant"), "editLists": [identity(ep) for ep in editlists],
                 "outputToRawMapping": export_map, "exportDurationSeconds": offset,
                 "exactBoundaryCertification": False,
                 "sourceGroup": private_value('source-group-004') if "20260829" in name else private_value('source-group-006'),
                 "groupNote": "Keep entire capture date/session together; derivatives do not create independent groups.",
                 "environment": "grass" if "20260829" in name else "unknown-in-current-metadata"}
        projects.append(entry)
        rows[key]["projectPair"] = {k: entry[k] for k in ("feedback", "sourceIdentityVerified", "sourceGroup", "environment")}
        if "20260816" in name:
            rows[key]["tier"] = "corrected-project-coverage-boundary-unverified"
            rows[key]["intervalCount"] = entry["includedCoreCutCount"]
            rows[key]["sourceGroup"] = entry["sourceGroup"]
    exports_path = root / private_value('private-reference-0085')
    exports = read(exports_path)
    export_details = []
    for item in exports["records"]:
        p = exports_path.parent / item["referencePath"]
        d = read(p)
        a = d["annotations"]
        export_details.append({"id": item["recordingId"], "reference": identity(p),
                              "annotationScope": d["annotationScope"],
                              "annotationCounts": {k: len(v) for k, v in a.items() if isinstance(v, list)},
                              "sourceHash": d["videoSha256"], "feedbackHash": d["feedbackSha256"]})
    snapshots = []
    a09root = root / "a09-v3-labeled-20260909"
    for p in sorted(a09root.glob("*.json")):
        if p.stem not in rows or rows[p.stem]["sourceGroup"] == PROTECTED:
            continue
        d = read(p)
        source = Path(d["labelSource"])
        snapshots.append({"id": p.stem, "path": str(p), "labelQuality": d.get("labelQuality"),
                          "labelSource": str(source), "labelSourceSha256": d.get("labelSourceSha256"),
                          "sourceHashStillMatches": identity(source)["sha256"] == d.get("labelSourceSha256"),
                          "serveTargetType": d.get("serveTargetType"),
                          "note": "Derived label snapshot plus model prediction, not a new independently labeled recording."})
    safe = [r for r in rows.values() if not r.get("excludedReason") and r["environment"] in {"grass", "indoor"}]
    exact = [r for r in safe if r["tier"] == "completed-exact"]
    historic = [r for r in safe if r["tier"] == "human-continuously-reviewed-draft" and r["hasAffirmativeTrainingManifest"]]
    all_reviewed = [r for r in safe if r["tier"] in {"completed-exact", "human-continuously-reviewed-draft"}]
    summary = {"completedExact": summarize(exact), "additionalHistoricallyTrainedDrafts": summarize(historic),
               "completedPlusHistoricallyTrainedDrafts": summarize(exact + historic),
               "allCompletedAndContinuouslyReviewedDrafts": summarize(all_reviewed),
               "categoryCounts": dict(Counter(r["tier"] for r in rows.values())),
               "inventoryUniqueRecordings": len(rows), "rawProjectPairs": len(projects),
               "rawProjectIdentityVerified": sum(p["sourceIdentityVerified"] for p in projects),
               "completedExactCurrentAvAvailable": sum(bool(r.get("currentAvCaches")) for r in exact),
               "historicDraftCurrentAvAvailable": sum(bool(r.get("currentAvCaches")) for r in historic)}
    return {"kind": "neural-corpus-expansion-audit-v1", "createdAt": datetime.now(timezone.utc).isoformat(),
            "summary": summary, "records": sorted(rows.values(), key=lambda x: x["id"]),
            "projectRawPairs": projects, "exportDataset": identity(exports_path), "exportReferences": export_details,
            "a09LabelSnapshots": snapshots, "priorCatalog": identity(catalog_path),
            "trainingManifestEvidence": [identity(p) for p, _ in manifests],
            "auditScript": identity(Path(__file__).resolve()),
            "limits": ["No training, label edits, feature extraction, or changes to frozen experiments.",
                       "Label-loader validity and continuous-review flags do not independently certify boundary accuracy.",
                       "Consent false in inference-only manifests is retained as historical scope evidence; it is not automatically a rights refusal. Affirmative train=true historical manifests are listed separately.",
                       "All protected source lineage remains excluded; protected outcome metrics are not used by this audit.",
                       "Raw/project identity checks read source size plus first/last 1 MiB. Full raw bytes were not rehashed.",
                       "Feature cache identity uses recorded video hash, current schema/config/ROI/path; proxy bytes were not rehashed in this inventory.",
                       "Aug16 environment remains unknown in metadata; verify no-beach eligibility before adding. Capture-date grouping is conservative and provisional.",
                       "A09 all-video files are UI/inference snapshots and audit results. A09 v3 files copy existing labels and add predictions; neither directory supplies new independent sources.",
                       "confirmedModelRanges/included/userTouchedCutIds do not certify exact starts and ends. Padded merged finalExportIntervals are coverage targets."]}


def markdown(a):
    s = a["summary"]
    lines = ["# Current corpus expansion audit", "", f"Audited {a['createdAt']}; no training or source edits.", "",
             "| Proposed tier | Recordings | Intervals | Independent source groups | Hours |",
             "|---|---:|---:|---:|---:|"]
    for key in ("completedExact", "completedPlusHistoricallyTrainedDrafts", "allCompletedAndContinuouslyReviewedDrafts"):
        t = s[key]
        lines.append(f"| {key} | {t['recordings']} | {t['intervals']} | {t['sourceGroupCount']} | {t['durationSeconds']/3600:.3f} |")
    lines += ["", "The next strict boundary cohort can add completed Bauos (31 intervals, existing Shoreline group) and zxtl (56, new YMCA group). Both have affirmative historical training-manifest evidence. The three additional historically trained human-reviewed drafts are jA3 (43), ugSoh (39), and PKK (41); retain their in-progress quality tier.",
              "", f"Current matching 104-channel AV caches exist for {s['completedExactCurrentAvAvailable']}/8 completed exact records and {s['historicDraftCurrentAvAvailable']}/3 historical drafts. These allow an expanded AV TCN/linear study without new feature extraction. DINO cache paths are listed separately; absent paths require extraction before a DINO study.",
              "", "Project exports retain source-clock core cuts separately from padded keep ranges and merged final exports. Use raw video as input and source-clock targets; do not train on an exported concatenation as if it preserved original dead time. The deterministic output-to-raw map is included for all 18 paired sources. Keep each raw/export/proxy family in one fold and conservatively group capture sessions.",
              "", "Eleven Aug16 feedback projects preserve inclusion/exclusion decisions and a few manual ranges but do not certify exact endpoints. Seven Aug29 projects already have a weak-coverage import, serve markers and switches. Train explicit coverage/serve/suppression auxiliary objectives with reduced or masked endpoint supervision; do not pool weak coverage into the exact-boundary evaluation. Two full-v3 files marked complete explicitly retain weak coverage provenance.",
              "", "A09 September directories are derived snapshots/predictions over existing labels, not additional fully reviewed recordings. The current inventory also includes the six Aug25 intake sources omitted from the 37-record v3 catalog.",
              "", "| Recording | Tier | Group | Intervals | Train=true manifest | Current AV cache |",
              "|---|---|---|---:|---|---|"]
    for r in a["records"]:
        lines.append(f"| {r['id']} | {r.get('excludedReason',r['tier'])} | {r['sourceGroup']} | {r['intervalCount']} | {r['hasAffirmativeTrainingManifest']} | {bool(r.get('currentAvCaches'))} |")
    lines += ["", "## Limits", ""] + ["- " + x for x in a["limits"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(private_value('private-reference-0059')))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="corpus-expansion-audit")
    args = parser.parse_args()
    if Path(args.output_stem).name != args.output_stem:
        raise SystemExit("Output stem must be a filename, not a path")
    paths = [args.output_dir / f"{args.output_stem}.{ext}" for ext in ("json", "md")]
    if any(p.exists() for p in paths):
        raise SystemExit("Refusing to overwrite an existing audit artifact")
    a = audit(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, content in zip(paths, (json.dumps(a, indent=2) + "\n", markdown(a))):
        with path.open("x", encoding="utf-8") as f:
            f.write(content)
    print(json.dumps(a["summary"], indent=2))
    print(json.dumps([identity(p) for p in paths], indent=2))


if __name__ == "__main__":
    main()
