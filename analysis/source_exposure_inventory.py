"""Read-only full-source inventory and deployed-model exposure audit.

Labels describe authority, not inferred accuracy. Project membership is coverage
supervision, never independently certified serve-contact/dead-ball boundaries.
No prediction, fitting, calibration, or metric code is called here.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import hashlib
import json
import struct
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROTECTED = frozenset({private_value('source-group-008')})
LABEL_PRIORITIES = {"tasks/full": 1, "labels/full": 2, "completed/full-v1": 3,
                    "labels/full-v3": 4, "completed/full-v3": 5}
EXACT = "completed-exact"
DRAFT = "human-continuously-reviewed-draft"
COVERAGE = "reviewed-export-coverage"
OLD_HEADS = {"rally": "full-audiovisual-audio-normalized-v3",
             "serve": "serve-specialist-audio-normalized-v5",
             "deadState": "dead-state-transition-audio-normalized-v5-no-legacy-final"}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path), "sizeBytes": path.stat().st_size}


def sampled_fingerprint(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256(struct.pack("<Q", size))
    with path.open("rb") as handle:
        digest.update(handle.read(min(size, 1024 * 1024)))
        if size > 1024 * 1024:
            handle.seek(max(1024 * 1024, size - 1024 * 1024))
            digest.update(handle.read())
    return "sampled-sha256-v1:" + digest.hexdigest()


def label_tier(document: dict, catalog_status: str | None = None) -> str:
    annotation = document.get("annotation", {})
    notes = annotation.get("notes", "").lower()
    weak = (catalog_status == COVERAGE or annotation.get("humanReviewImported") or
            document.get("recording", {}).get("capture", {}).get("targetStatus") == COVERAGE or
            "weak coverage" in notes or "independent serve-contact/dead-ball endpoint labeling is not asserted" in notes)
    if weak:
        return COVERAGE
    human = annotation.get("annotator", "").strip().lower() in {"v", "vini"}
    if annotation.get("status") == "complete" and annotation.get("continuousVideoReviewed") and human:
        return EXACT
    if human and annotation.get("continuousVideoReviewed"):
        return DRAFT
    if human:
        return "human-partially-reviewed-draft"
    return "unvalidated-candidate" if document.get("rallies") else "unlabeled"


def choose_feedback(versions: list[tuple[Path, dict]]) -> tuple[Path, dict]:
    """Latest corrected revision, then generation time; equal-time conflicts fail.

    Duplicated bundle/raw copies with identical corrections are one revision.
    A deterministic path tie-break affects provenance path only, never targets.
    """
    def key(item):
        return (item[1].get("corrections", {}).get("updatedAt") or "", item[1].get("generatedAt") or "")
    best_key = max(map(key, versions))
    finalists = [item for item in versions if key(item) == best_key]
    def targets(item):
        d = item[1]
        return json.dumps({k: d.get(k) for k in ("source", "corrections", "finalExportIntervals", "finalExportProvenance")}, sort_keys=True)
    if len({targets(item) for item in finalists}) != 1:
        raise ValueError("conflicting feedback revisions at identical correction/generation timestamps")
    return min(finalists, key=lambda item: str(item[0]))


def subtract_ranges(start: float, end: float, masks: list[dict]) -> list[tuple[float, float]]:
    parts = [(start, end)]
    for mask in masks:
        updated = []
        for left, right in parts:
            if mask["end"] <= left or mask["start"] >= right:
                updated.append((left, right))
            else:
                if left < mask["start"]:
                    updated.append((left, mask["start"]))
                if mask["end"] < right:
                    updated.append((mask["end"], right))
        parts = updated
    return parts


def feedback_targets(document: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Source-coordinate coverage + literal final exports, without re-padding."""
    corrections = document["corrections"]
    duration = float(document["source"]["media"]["duration"])
    window = document["source"].get("gameWindow") or {"start": 0, "end": duration}
    ignored = list(corrections.get("ignoredIntervals", []))
    for left, right in ((0., window["start"]), (window["end"], duration)):
        if right > left:
            ignored.append({"start": left, "end": right, "reason": "outside-game-window"})
    exports = document["finalExportIntervals"]
    active = {rid for row in exports for rid in row.get("cutIds", [])}
    cores = []
    for row in corrections["correctedRanges"]:
        if not row.get("included") or (active and row["id"] not in active):
            continue
        masks = ignored + [r for r in document.get("finalExportProvenance", [])
                           if str(r.get("kind", "")).startswith("suppression-") and row["id"] in r.get("cutIds", [])]
        for i, (left, right) in enumerate(subtract_ranges(row["coreStart"], row["coreEnd"], masks)):
            if right > left:
                cores.append({"id": f"{row['id']}::part:{i+1}", "sourceCutId": row["id"], "start": left, "end": right,
                              "tags": ["human-reviewed-project-coverage", "weak-rally-coverage"],
                              "sourceOrigin": row.get("origin")})
    keep, mapping, offset = [], [], 0.
    for row in exports:
        left, right = float(row["start"]), float(row["end"])
        if not 0 <= left < right <= duration + .01:
            raise ValueError("invalid final source export interval")
        keep.append({"start": left, "end": right})
        mapping.append({"outputStart": offset, "outputEnd": offset + right-left,
                        "sourceStart": left, "sourceEnd": right, "cutIds": row.get("cutIds", [])})
        offset += right-left
    return sorted(cores, key=lambda r: (r["start"], r["end"])), keep, ignored, mapping


def exposure_status(row: dict, heads: list[dict]) -> dict:
    aliases = set(row["aliases"]) | {row["id"]}
    direct, same_source, related, calibration_direct, calibration_related, unknown = [], [], [], [], [], []
    for head in heads:
        fit = set(head.get("trainingRecordingIds", []))
        calibrate = set(head.get("calibrationRecordingIds", []))
        if row["id"] in fit:
            direct.append(head["id"])
        elif aliases & fit:
            same_source.append(head["id"])
        elif row["sourceGroup"] in head.get("trainingSourceGroups", []):
            related.append(head["id"])
        if aliases & calibrate:
            calibration_direct.append(head["id"])
        elif row["sourceGroup"] in head.get("calibrationSourceGroups", []):
            calibration_related.append(head["id"])
        if not head.get("fitScopeComplete", False) or not head.get("calibrationScopeComplete", False):
            unknown.append(head["id"])
    fit_exposed = bool(direct or same_source or related)
    status = ("direct-training" if direct else "same-source-training" if same_source else
              "same-group-training" if related else "unknown" if unknown else
              "calibration-only" if calibration_direct or calibration_related else "known-no-fit-or-calibration")
    return {"status": status, "directTrainingHeads": direct, "sameSourceTrainingHeads": same_source,
            "sameGroupTrainingHeads": related, "directCalibrationHeads": calibration_direct,
            "sameGroupCalibrationHeads": calibration_related, "unknownScopeHeads": unknown,
            "trainingOrRelated": fit_exposed,
            "primaryTrainingClean": not fit_exposed and not unknown,
            "strictNoFitOrCalibration": not fit_exposed and not calibration_direct and not calibration_related and not unknown,
            "note": "Known clean means no recorded fit/calibration exposure within this pinned lineage; historical diagnostic viewing is separate."}


def eligible_for_training(row: dict) -> bool:
    return row["environment"] in {"grass", "indoor"} and row["sourceGroup"] not in PROTECTED and row["tier"] in {EXACT, DRAFT, COVERAGE}


def model_lineage(root: Path, repo: Path, rows: list[dict]) -> list[dict]:
    alias_groups = {}
    for row in rows:
        for alias in row["aliases"] + [row["id"]]:
            if alias in alias_groups and alias_groups[alias] != row["sourceGroup"]:
                raise ValueError(f"same-source alias assigned to inconsistent groups: {alias}")
            alias_groups[alias] = row["sourceGroup"]
    def groups(ids):
        missing = sorted(set(ids) - set(alias_groups))
        if missing:
            raise ValueError(f"unresolved production training/calibration aliases: {missing}")
        return sorted({alias_groups[rid] for rid in ids})
    result = []
    variants = [("previous-production", "model-9c92b8e9333f.json", root / "labeling-v1-2026-08-09-no-beach-2026-08-12/models", OLD_HEADS),
                ("all-labels-v2", "model-1ca43e38eefc.json", root / "intake-2026-08-13/experiments/environment-specialists-v2/models/all-labels",
                 {"rally": "rally", "serve": "serve", "deadState": "dead-state"})]
    previous_calibration = []
    for variant, runtime_name, model_root, roles in variants:
        runtime_path = repo / "prod/public/runtime" / runtime_name
        runtime = read(runtime_path)
        for role, name in roles.items():
            p = model_root / name / "model.json"
            metadata = read(p)
            weights = p.parent / metadata["weightsFile"]
            if sha256(weights) != metadata["weightsSha256"]:
                raise ValueError("model weight hash mismatch")
            artifact = hashlib.sha256(p.read_bytes() + weights.read_bytes()).hexdigest()
            if artifact != runtime[role]["artifactSha256"]:
                raise ValueError("deployed runtime does not bind inspected model")
            t = metadata["training"]
            fit = t["trainingRecordingIds"]
            calibration = t.get("validationRecordingIds", [])
            if variant == "previous-production":
                previous_calibration.extend(calibration)
            else:
                # v2 explicitly reuses production epoch caps/decoders/refinement.
                calibration = sorted(set(calibration + previous_calibration))
            result.append({"id": f"{variant}/{role}", "affectsRallyExport": True,
                           "runtime": identity(runtime_path), "metadata": identity(p), "weights": identity(weights),
                           "artifactSha256": artifact, "trainingRecordingIds": fit,
                           "trainingSourceGroups": groups(fit), "calibrationRecordingIds": calibration,
                           "calibrationSourceGroups": groups(calibration), "fitScopeComplete": True,
                           "calibrationScopeComplete": True,
                           "calibrationProvenance": "training.validationRecordingIds; v2 inherits prior production recipe selection",
                           "metadataDeclaredFitGroups": t.get("trainingSourceGroups", []),
                           "selectionPolicy": t.get("selectionPolicy")})
    p = root / "intake-2026-08-13/experiments/feedback-suppression-v3-2026-08-16/models/suppression-overlap-exclusion-retrained/model.json"
    d = read(p); t = d["training"]; weights = p.parent / d["weightsFile"]
    runtime_path = repo / "prod/public/runtime/suppression-39eddf581639.json"
    manifest_path = repo / "prod/public/runtime/suppression-39eddf581639.manifest.json"
    manifest = read(manifest_path)
    artifact = hashlib.sha256(p.read_bytes() + weights.read_bytes()).hexdigest()
    if artifact != manifest["source"]["artifactSha256"] or sha256(p) != manifest["source"]["metadataSha256"] or sha256(weights) != d["weightsSha256"]:
        raise ValueError("suppression runtime source lineage mismatch")
    calibration = sorted(set(t.get("validationRecordingIds", []) + t["productionEnsembleDecoderSelection"]["scope"] + previous_calibration))
    result.append({"id": "production-suppression", "affectsRallyExport": True,
                   "runtime": identity(runtime_path), "runtimeManifest": identity(manifest_path),
                   "metadata": identity(p), "weights": identity(weights), "artifactSha256": artifact,
                   "trainingRecordingIds": t["trainingRecordingIds"], "trainingSourceGroups": groups(t["trainingRecordingIds"]),
                   "calibrationRecordingIds": calibration, "calibrationSourceGroups": groups(calibration),
                   "fitScopeComplete": True, "calibrationScopeComplete": True,
                   "appliesTo": ["guarded one-model-supported suppression", "aggressive whole-rally suppression"],
                   "calibrationProvenance": "training.productionEnsembleDecoderSelection.scope plus inherited production recipe"})
    # Serving-side and side-switch do not set rally export membership. Record
    # them independently so the stricter whole-product panel remains available.
    p = root / "labeling-v1-2026-08-09/reports/serving-side/serving-side-flight-v3-development.json"
    d = read(p); runtime_path = repo / "prod/public/runtime/serving-side-85bc3325fbd4.json"; runtime = read(runtime_path)
    if sha256(p) != runtime["sourceReportSha256"] or d["finalModel"]["fingerprint"] != runtime["fingerprint"]:
        raise ValueError("serving-side runtime source mismatch")
    fit = sorted({r["recordingId"] for r in d["selectedPredictions"]})
    cp = root / "labeling-v1-2026-08-09/reports/serving-side/serving-side-flight-v3-calibration-abstention-development.json"
    c = read(cp)
    if c["modelFingerprint"] != runtime["fingerprint"] or c["sources"]["fixedFlightEvaluation"]["sha256"] != sha256(p):
        raise ValueError("serving-side calibration identity mismatch")
    result.append({"id": "serving-side", "affectsRallyExport": False, "runtime": identity(runtime_path),
                   "metadata": identity(p), "calibrationArtifact": identity(cp), "trainingRecordingIds": fit,
                   "trainingSourceGroups": groups(fit), "calibrationRecordingIds": fit,
                   "calibrationSourceGroups": groups(fit), "fitScopeComplete": True, "calibrationScopeComplete": True,
                   "evidence": "finalModel fit to all 1027 selected development rows; cross-fit development threshold and review band"})
    p = root / "labeling-v1-2026-08-09/models/side-switch-hard-negative-mining-v1/model.json"
    d = read(p); runtime_path = repo / "prod/public/runtime/side-switch-c2570481c30d.json"
    if "sha256:" + sha256(p) != read(runtime_path)["fingerprint"]:
        raise ValueError("side-switch runtime source mismatch")
    fit = sorted(d["hardNegativeMining"]["finalMiningAudit"]["byRecording"])
    result.append({"id": "side-switch", "affectsRallyExport": False, "runtime": identity(runtime_path),
                   "metadata": identity(p), "trainingRecordingIds": fit, "trainingSourceGroups": groups(fit),
                   "calibrationRecordingIds": fit, "calibrationSourceGroups": groups(fit),
                   "fitScopeComplete": True, "calibrationScopeComplete": True,
                   "evidence": "all-opened-development final mining fit and selected threshold/variant"})
    return result


def summary(rows: list[dict]) -> dict:
    return {"recordings": len(rows), "intervals": sum(len(r["rallies"]) for r in rows),
            "durationSeconds": sum(r["durationSeconds"] for r in rows),
            "sourceGroups": sorted({r["sourceGroup"] for r in rows}), "ids": [r["id"] for r in rows]}


def build_inventory(root: Path, repo: Path, output: Path) -> dict:
    from .annotations import load_label_document
    from .config import FeatureConfig, NOISE_NORMALIZED_AUDIO_FEATURE_SET
    from .features import feature_cache_path

    catalog_paths = [root / "labeling-v1-2026-08-09/reports/full-nas-video-corpus-v3.json",
                     root / private_value('private-reference-0062')]
    rows, catalogs = {}, {}
    for path in catalog_paths:
        for rec in read(path)["records"]:
            catalogs[rec["recordingId"]] = rec
            rows[rec["recordingId"]] = {"id": rec["recordingId"], "sourceGroup": rec["sourceGroup"],
                "environment": rec["environment"], "durationSeconds": rec["durationSeconds"],
                "video": rec["videoPath"], "contentSha256": rec.get("videoSha256"), "roi": rec.get("roi"),
                "rallies": rec.get("rallies", []), "ignoredIntervals": rec.get("ignoredIntervals", []),
                "tier": COVERAGE if "coverage" in rec.get("targetStatus", "") else "unvalidated-candidate",
                "annotation": {}, "aliases": [], "labelRevisions": [], "featureCaches": {"audiovisual": []},
                "catalogLabel": {"path": rec.get("labelPath"), "sha256": rec.get("labelSha256")}}
    candidates = defaultdict(list)
    for subroot in ("labeling-v1-2026-08-09", "intake-2026-08-13", "intake-2026-08-25-shoreline-kb"):
        for subdir, priority in LABEL_PRIORITIES.items():
            for path in sorted((root / subroot / subdir).glob("*.labels.json")):
                doc = read(path); rid = doc["recording"]["id"]
                candidates[rid].append((priority, path, doc))
    for rid, versions in candidates.items():
        priority, path, doc = max(versions, key=lambda item: (item[0], str(item[1])))
        rec = doc["recording"]
        row = rows.setdefault(rid, {"id": rid, "aliases": [], "featureCaches": {"audiovisual": []}})
        label = load_label_document(path, require_complete=False, require_video=False)
        row.update({"sourceGroup": rec["sourceGroup"], "environment": rec["environment"],
                    "durationSeconds": rec["durationSeconds"], "video": str(label.video),
                    "contentSha256": rec.get("contentSha256"), "roi": rec.get("roi"),
                    "rallies": doc.get("rallies", []), "ignoredIntervals": doc.get("ignoredIntervals", []),
                    "hardNegatives": doc.get("hardNegatives", []), "serveMarkers": doc.get("serveMarkers", []),
                    "sideSwitches": doc.get("sideSwitches", []), "annotation": doc.get("annotation", {}),
                    "annotationPolicy": doc.get("annotationPolicy"), "labelSource": identity(path),
                    "tier": label_tier(doc, catalogs.get(rid, {}).get("targetStatus")),
                    "labelRevisions": [{**identity(p), "priority": rank, "annotation": d.get("annotation"),
                                        "intervalCount": len(d.get("rallies", []))} for rank, p, d in versions]})
    # Current main roots + feedback repositories, not an old 18-row training subset.
    raw_root = root.parent / "volleycut-raw-no-backup"
    feedback_versions = defaultdict(list)
    for path in sorted(list(raw_root.glob("*.model-feedback*.json")) + list((root / "model-feedback").glob("*/bundle.json"))):
        doc = read(path); feedback_versions[doc["source"]["file"]["name"]].append((path, doc))
    for name, versions in feedback_versions.items():
        path, doc = choose_feedback(versions); source = doc["source"]; raw = raw_root / name
        rid = "raw-no-backup-" + raw.stem
        if rid not in rows:
            raise ValueError(f"uncatalogued reviewed feedback needs explicit group/environment: {rid}")
        row = rows[rid]
        fingerprint = sampled_fingerprint(raw)
        if fingerprint != source["file"]["sampledFingerprint"] or raw.stat().st_size != source["file"]["sizeBytes"]:
            raise ValueError(f"feedback source identity mismatch: {rid}")
        cores, keep, ignored, mapping = feedback_targets(doc)
        if "20260816" in rid:
            frame = output / "environment-frames" / (raw.stem + ".jpg")
            if not frame.exists():
                raise ValueError("Aug16 grass classification requires retained 60s frame evidence")
            row.update(environment="grass", sourceGroup=private_value('source-group-006'),
                       groupNormalization={"catalogGroup": row["sourceGroup"],
                                           "policy": "all camera orientations/courts from same capture date remain one source group"},
                       environmentEvidence={**identity(frame), "atSeconds": 60,
                       "decision": "visual inspection confirms grass court, no rally labels inferred"})
        if any(version["source"]["file"]["sampledFingerprint"] != fingerprint or
               version["source"]["file"]["sizeBytes"] != raw.stat().st_size for _, version in versions):
            raise ValueError(f"feedback revision source identity mismatch: {rid}")
        row["aliases"] = sorted({version["source"]["projectId"] for _, version in versions} | {name, raw.stem})
        row["feedback"] = {**identity(path), "revisions": [identity(p) for p, _ in versions],
                           "source": source, "sourceIdentityVerified": True, "actualSampledFingerprint": fingerprint,
                           "authority": "latest corrections.updatedAt, then generatedAt; equal-time target conflicts reject",
                           "reviewEvidence": "user confirmed all kept/discarded project exports manually reviewed",
                           "userTouchedCutCount": len(doc["corrections"].get("userTouchedCutIds", [])),
                           "correctedRangeCount": len(doc["corrections"]["correctedRanges"]),
                           "correctionLabelCounts": {k: len(v) for k, v in doc["corrections"].get("labels", {}).items() if isinstance(v, list)}}
        row["keepTargets"] = keep
        row["exportToSourceMapping"] = mapping
        row["feedbackCoreRanges"] = cores
        row["gameWindow"] = source.get("gameWindow")
        row["feedbackPadding"] = {k: doc["corrections"].get(k) for k in ("beforePaddingSeconds", "afterPaddingSeconds", "joinGapSeconds")}
        row["editLists"] = [identity(p) for p in sorted(raw_root.glob(raw.stem + "*edit-list*.json"))]
        row["exportVideos"] = [{"path": str(p), "sizeBytes": p.stat().st_size,
                                "sameSourceDerivative": True, "mappingVerifiedAgainstBytes": False}
                               for p in sorted(raw_root.glob(raw.stem + "-cut*.mp4"))]
        row["tier"] = COVERAGE
        row["coverageReview"] = {"exhaustiveKeptDiscardedReviewConfirmed": True,
                                 "authority": "user: All manually reviewed",
                                 "reviewScope": "kept/discarded export coverage across the project game window",
                                 "independentEndpointGold": False,
                                 "sourceAnnotationPreserved": True}
        if "labelSource" not in row:
            row.update(rallies=cores, ignoredIntervals=ignored, labelSource=identity(path),
                       annotation={"humanReviewImported": True, "continuousVideoReviewed": False,
                                   "independentEndpointGold": False}, roi=source.get("featureRoi"))
        # Metadata/current labels can refine coverage, but no revision silently
        # overwrites the separate literal final-export target.
        if doc.get("features"):
            features = doc["features"]
            row["featureCaches"]["audiovisual"].append({"container": identity(path),
                "field": "features", "featureOrigin": source.get("runtimeVariant", "unknown"),
                "rows": features.get("rows"), "columns": features.get("columns"),
                "analysisFps": features.get("analysisFps"), "featureRoi": source.get("featureRoi"),
                "interchangeableWithOpenCv": False})
    config = FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET)
    cache_roots = [root / "labeling-v1-2026-08-09", root / "intake-2026-08-13/experiments/environment-specialists-v2",
                   root / "intake-2026-08-25-shoreline-kb", root / private_value('private-reference-0063')]
    for row in rows.values():
        video = Path(row["video"])
        row["mediaIdentity"] = {"path": str(video), "sizeBytes": video.stat().st_size,
                                "declaredContentSha256": row.get("contentSha256"), "fullBytesRehashedByInventory": False}
        prov = video.with_suffix(video.suffix + ".provenance.json")
        if prov.exists():
            data = read(prov)
            row["normalizationProvenance"] = {**identity(prov), "source": data.get("source"), "segment": data.get("segment"), "normalized": data.get("normalized")}
            if data["normalized"]["sha256"] != row.get("contentSha256"):
                raise ValueError(f"proxy/label content hash mismatch: {row['id']}")
            row["aliases"].append(data["source"]["filename"])
            row["aliases"].append(data["source"]["sha256"])
        row["aliases"] = sorted(set(row["aliases"] + [video.name, video.stem] + ([row["contentSha256"]] if row.get("contentSha256") else [])))
        roi = tuple(float(row["roi"][k]) for k in ("x", "y", "width", "height")) if row.get("roi") else None
        for cache_root in cache_roots:
            for dirname, origin in (("audiovisual-audio-normalized-v3", "opencv-av104-v3"),
                                    ("audiovisual-audio-normalized-v3-nvdec-v1", "nvdec-av104-v3")):
                path = feature_cache_path(row["id"], str(video), config, roi, cache_root / "features" / dirname, content_sha256=row.get("contentSha256"))
                if path.exists():
                    row["featureCaches"]["audiovisual"].append({**identity(path), "featureOrigin": origin,
                        "identity": "exact current config/source-hash/path/ROI cache-key", "featureRoi": row.get("roi")})
        row["protected"] = row["sourceGroup"] in PROTECTED
        row["inferenceOnly"] = row["protected"] or row["tier"] not in {EXACT, DRAFT, COVERAGE}
        row["eligibleForVariantTrainingBeforePanelReservation"] = eligible_for_training(row)
        row["evaluationTier"] = ("excluded-beach" if row["environment"] == "beach" else
                                 "exact-semantic-rallies" if row["tier"] == EXACT else
                                 "reviewed-draft-boundaries" if row["tier"] == DRAFT else
                                 "reviewed-export-coverage" if row["tier"] == COVERAGE else "inference-only-no-accuracy")
        row["targetContract"] = {"independentEndpointGold": row["tier"] == EXACT,
            "eventMetricsAreSemanticGold": row["tier"] == EXACT,
            "literalKeepTargetsAlreadyPaddedAndJoined": row["tier"] == COVERAGE,
            "doNotTrainNegativesOutsideReviewedScope": True}
        row["consent"] = {"analyze": row["environment"] != "beach",
                          "train": eligible_for_training(row), "authority": "current explicit user scope; protected always excluded"}
        row["exclusionReasons"] = (["user-directed beach exclusion"] if row["environment"] == "beach" else
                                   ["partial review cannot support exhaustive negatives or accuracy"] if row["tier"] == "human-partially-reviewed-draft" else
                                   ["unvalidated model/AI candidates are not human gold"] if row["tier"] == "unvalidated-candidate" else
                                   ["no rally labels"] if row["tier"] == "unlabeled" else [])
    ordered = sorted(rows.values(), key=lambda r: r["id"])
    heads = model_lineage(root, repo, ordered)
    for row in ordered:
        row["productionExposure"] = {"rallyPipeline": exposure_status(row, [h for h in heads if h["affectsRallyExport"]]),
                                     "wholeProduct": exposure_status(row, heads)}
    nonbeach = [r for r in ordered if r["environment"] != "beach"]
    scored = [r for r in nonbeach if r["tier"] in {EXACT, DRAFT, COVERAGE}]
    return {"schemaVersion": 1, "kind": "source-exposure-inventory-v1", "createdAt": datetime.now(timezone.utc).isoformat(),
            "records": ordered, "productionHeads": heads, "sources": [identity(p) for p in catalog_paths],
            "implementation": identity(Path(__file__).resolve()),
            "authorityPolicy": {"labelPriority": LABEL_PRIORITIES, "tieBreak": "lexicographic path within same workflow priority",
                                "projectReview": "explicit user confirmation all kept/discarded exports manually reviewed",
                                "protected": "metadata permitted; evaluation only after frozen selections; never fit/calibrate/select",
                                "noBoundaryPromotion": True},
            "summary": {"all": summary(ordered), "nonBeach": summary(nonbeach), "scorableBySeparateTier": summary(scored),
                        "byTier": {tier: summary([r for r in nonbeach if r["tier"] == tier]) for tier in sorted({r["tier"] for r in nonbeach})},
                        "protected": summary([r for r in nonbeach if r["protected"]]),
                        "rallyPipelineTrainingCleanScorable": summary([r for r in scored if r["productionExposure"]["rallyPipeline"]["primaryTrainingClean"]]),
                        "rallyPipelineStrictCleanScorable": summary([r for r in scored if r["productionExposure"]["rallyPipeline"]["strictNoFitOrCalibration"]]),
                        "featureOrigins": dict(Counter(c["featureOrigin"] for r in nonbeach for c in r["featureCaches"]["audiovisual"]))},
            "limitations": ["No new predictions, scores, training or calibration generated.",
                "Label authority/review metadata is not an independent reannotation or accuracy guarantee.",
                "Full raw/proxy bytes are not rehashed; source feedback size + first/last1MiB sampled identity verified.",
                "Aug16 remains one conservative capture-session group despite camera/court changes; all11 fixed60s frames visually confirm grass.",
                "Native-Android, OpenCV and NVDEC AV104 caches have distinct origins and cannot be silently interchanged.",
                "Historical diagnostic/research exposure is not a clean untouched-test claim; known no-fit/calibration is pinned lineage only.",
                "The protected SPU source group was used in historical production suppression decoder selection via f7T727Rezdc; no current fitting is allowed.",
                "Pilot, proxy, A09 label snapshots, feedback projects and cut exports are aliases/derivatives, not independent sources."]}
