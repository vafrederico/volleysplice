#!/usr/bin/env python3
"""Freeze exact, continuously reviewed draft, and manually reviewed export tiers.

Only new immutable preparation artifacts are written. No feature extraction,
training, label changes, or protected outcome reads occur.
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
from analysis.exported_project_dataset import sampled_fingerprint
from analysis.features import feature_cache_path, feature_names
from analysis.schema import load_manifest

PARENT_HASH = "04365c5fe26a3186acad90cba5b4f595588d17d24240251c26a2f9b2aab60532"
AUDIT_HASH = "02bc294d5ececa9e39e0045548ac7c680a7b25e2a9e382fc278d13ba0cccacde"
SUPPLEMENT_HASH = "5aa69b50c3022cf807491e3c744988f6eca188b30f39f3dfd452baa8050a4713"
PROTECTED = private_value('source-group-008')
EXACT_ADDITIONS = (private_value('grass-source-03'), private_value('indoor-source-08'))
DRAFT_ADDITIONS = (private_value('grass-source-05'), private_value('indoor-source-06'), private_value('indoor-source-04'))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    return {"path": str(path), "sha256": sha256(path), "sizeBytes": path.stat().st_size}


def verify(stored):
    actual = identity(Path(stored["path"]))
    require(actual["sha256"] == stored["sha256"], f"Changed frozen input: {stored['path']}")
    return actual


def av_cache(row, path, config):
    roi = tuple(float(row["roi"][k]) for k in ("x", "y", "width", "height")) if row.get("roi") else None
    expected = feature_cache_path(row["id"], row["video"], config, roi, path.parent,
                                  content_sha256=row["contentSha256"])
    require(expected == path, f"Cache identity mismatch: {path}")
    with np.load(path, allow_pickle=False) as n:
        times, values = n["times"], n["values"]
        names = tuple(str(x) for x in n["names"])
        metadata = json.loads(str(n["metadata_json"].item()))
        decoder = str(n["video_decoder"].item()) if "video_decoder" in n else "opencv-ffmpeg-v1"
        require(names == feature_names(config) and values.shape == (len(times), 104), "Wrong AV schema")
        require(np.isfinite(values).all() and np.isfinite(times).all() and np.all(np.diff(times) > 0), "Invalid AV values/times")
        require(abs(metadata["duration"] - row["durationSeconds"]) < .1 and decoder == "opencv-ffmpeg-v1", "AV media mismatch")
        return {**identity(path), "featureVersion": feature_version_for_config(config),
                "config": config.to_dict(), "featureNames": list(names), "names": list(names),
                "shape": list(values.shape), "storedDtype": str(values.dtype), "metadata": metadata,
                "timestampsKey": "times", "valuesKey": "values", "metadataKey": "metadata_json",
                "videoDecoder": decoder, "nominalSampleFps": config.analysis_fps,
                "timestamps": {"count": len(times), "first": float(times[0]), "last": float(times[-1]),
                               "medianStepSeconds": float(np.median(np.diff(times)))},
                "timestampRule": "actual nearest source-frame timestamps; use stored times, not arange/4",
                "identityVerification": "current feature_cache_path with float ROI, current config, content SHA-256, absolute video path"}


def prepare(root, output):
    names = ("manifest.json", "exact-manifest.json", "dataset-audit.json")
    require(not any((output / name).exists() for name in names), "Refusing to overwrite frozen artifacts")
    prior = root / "neural-experiments/2026-09-19-nonbeach"
    parent_path, audit_path, supplement_path = (prior / name for name in
        ("manifest.json", "corpus-expansion-audit-v2.json", "corpus-expansion-cache-supplement.json"))
    for path, expected in ((parent_path, PARENT_HASH), (audit_path, AUDIT_HASH), (supplement_path, SUPPLEMENT_HASH)):
        require(sha256(path) == expected, f"Unexpected parent: {path}")
    parent, corpus, supplement = read(parent_path), read(audit_path), read(supplement_path)
    corpus_rows = {r["id"]: r for r in corpus["records"]}
    historical = supplement["historicalTrainingConsent"]["manifest"]
    verify(historical)
    historical_rows = {r["id"]: r for r in read(Path(historical["path"]))["recordings"]}
    protected_hashes = set()
    for item in supplement["lineageVerification"]["protectedMetadataOnly"]:
        verify(item["labelSource"])
        verify(item["normalizationProvenance"])
        protected_hashes.update((item["sourceSha256"], item["proxySha256"]))
    config = FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET)
    now = datetime.now(timezone.utc).isoformat()
    authorization = {
        "source": "current conversation; exact quotes relayed by coordinating root agent",
        "reviewQuestion": "For the project exports you mean, were the kept and discarded sections manually reviewed, or are some exports still based on automatic cuts?",
        "userReviewConfirmation": "All manually reviewed",
        "userExecutionAuthorization": "use more videos if you want to, continue executing",
        "recordedAt": now,
        "scope": "train expanded non-beach development study using existing labels and manually reviewed exports; no protected data or production changes",
    }
    exact = copy.deepcopy(parent["recordings"])
    drafts = []
    for rid in EXACT_ADDITIONS + DRAFT_ADDITIONS:
        candidate = corpus_rows[rid]
        verify(candidate["labelSource"])
        label = load_label_document(Path(candidate["labelSource"]["path"]),
                                    require_complete=rid in EXACT_ADDITIONS, require_video=False)
        payload = label.payload
        annotation = payload["annotation"]
        require(annotation.get("continuousVideoReviewed") is True, f"Full review not supported: {rid}")
        require(annotation.get("annotator", "").lower() in {"v", "vini"}, f"Human review not supported: {rid}")
        consent = historical_rows[rid]["consent"]
        require(consent.get("analyze") is True and consent.get("train") is True, "Missing historical consent")
        row = copy.deepcopy(payload["recording"])
        row.update({"video": str(label.video), "sourceLabelSplit": label.split,
                    "split": "validation" if label.source_group == private_value('source-group-007') else "train",
                    "consent": copy.deepcopy(consent), "consentProvenance": historical,
                    "labelSource": candidate["labelSource"], "normalizationProvenance": candidate["normalizationProvenance"],
                    "sourceContentSha256": candidate["rawSource"]["sha256"],
                    "durationSeconds": label.duration,
                    "featureCaches": {"audiovisual": candidate["currentAvCaches"][0]}})
        for key in ("rallies", "ignoredIntervals", "hardNegatives", "sideSwitches", "serveMarkers", "annotation"):
            row[key] = copy.deepcopy(payload.get(key, []))
        if rid in EXACT_ADDITIONS:
            row["targetStatus"] = "exact-human-reviewed"
            exact.append(row)
        else:
            row["targetStatus"] = "human-continuously-reviewed-draft"
            drafts.append(row)
    seen_hashes = set()
    for row in exact + drafts:
        require(row["environment"] in {"grass", "indoor"} and row["sourceGroup"] != PROTECTED, "Excluded scope")
        verify(row["labelSource"])
        verify(row["normalizationProvenance"])
        provenance = read(Path(row["normalizationProvenance"]["path"]))
        require(row["contentSha256"] == provenance["normalized"]["sha256"], "Proxy provenance mismatch")
        require(row["sourceContentSha256"] == provenance["source"]["sha256"], "Raw provenance mismatch")
        require(Path(row["video"]).stat().st_size == provenance["normalized"]["sizeBytes"], "Proxy size changed")
        row_hashes = {row["contentSha256"], row["sourceContentSha256"]}
        require(not row_hashes & (protected_hashes | seen_hashes), "Protected or duplicate lineage")
        seen_hashes.update(row_hashes)
        row.setdefault("consentProvenance", {"parentManifest": identity(parent_path),
                                              "sourceManifest": parent["sourceManifest"]})
        verify(row["featureCaches"]["audiovisual"])
        row["featureCaches"]["audiovisual"] = av_cache(row, Path(row["featureCaches"]["audiovisual"]["path"]), config)
        is_draft = row in drafts
        row.update({"qualityTier": "human-continuously-reviewed-draft" if is_draft else "completed-exact-boundaries",
                    "trainingOnly": is_draft,
                    "targetMasks": {"live": True, "serve": not is_draft, "end": not is_draft, "keep": not is_draft},
                    "videoVerification": "stored full source/proxy hashes plus unchanged label/provenance hashes and proxy size; raw masters not rehashed",
                    "targetContract": {"live": "existing rally intervals on valid reviewed time; ignored intervals excluded",
                                       "serveAndEnd": "fully masked for draft rows" if is_draft else "existing exact rally start/end targets",
                                       "keep": "masked for draft rows" if is_draft else "derive from exact intervals using preregistered padding/join contract"}})
        if is_draft:
            row["reviewExtentEvidence"] = {"continuousVideoReviewed": True,
                                           "annotator": row["annotation"]["annotator"],
                                           "status": row["annotation"]["status"],
                                           "wholeTimelineLiveSupervision": True,
                                           "uncertainBoundaryHeadsMasked": ["serve", "end"]}
    coverage = []
    dataset_path = root / private_value('private-reference-0085')
    verify(supplement["exportDataset"])
    dataset = read(dataset_path)
    cache_by_id = {r["id"]: r for r in supplement["exportCaches"]}
    for item in dataset["records"]:
        reference_path = dataset_path.parent / item["referencePath"]
        require(sha256(reference_path) == item["referenceSha256"], "Reference revision changed")
        reference = read(reference_path)
        annotations = reference["annotations"]
        video = Path(reference["videoPath"])
        require(video.stat().st_size == reference["videoSizeBytes"], "Raw video size changed")
        require(sampled_fingerprint(video) == reference["sampledFingerprint"], "Raw fingerprint changed")
        feedback = {"path": reference["feedbackPath"], "sha256": reference["feedbackSha256"]}
        verify(feedback)
        require(reference["environment"] == "grass" and reference["sourceGroup"] == private_value('source-group-004'), "Unexpected export scope")
        require(reference["videoSha256"] not in protected_hashes | seen_hashes, "Protected/duplicate export lineage")
        seen_hashes.add(reference["videoSha256"])
        row = {"id": reference["recordingId"], "sourceGroup": reference["sourceGroup"],
               "environment": "grass", "video": str(video), "videoFilename": reference["videoFilename"],
               "contentSha256": reference["videoSha256"], "sourceContentSha256": reference["videoSha256"],
               "videoSizeBytes": reference["videoSizeBytes"], "sampledFingerprint": reference["sampledFingerprint"],
               "durationSeconds": annotations["durationSeconds"], "roi": reference["roi"],
               "split": "train", "sourceDatasetSplit": reference["split"], "trainingOnly": True,
               "consent": {"analyze": True, "train": True}, "consentProvenance": authorization,
               "qualityTier": "human-reviewed-export-coverage", "targetStatus": "reviewed-export-coverage",
               "referenceSource": identity(reference_path), "feedbackSource": feedback,
               "retainedCoverage": copy.deepcopy(annotations["associationCoreRanges"]),
               "rawRetainedCoreRanges": copy.deepcopy(annotations["retainedCoreRanges"]),
               "microRangeArtifacts": copy.deepcopy(annotations["microRangeArtifacts"]),
               "keepTargets": copy.deepcopy(annotations["finalExportIntervals"]),
               "finalExportIntervals": copy.deepcopy(annotations["finalExportIntervals"]),
               "ignoredIntervals": copy.deepcopy(annotations["ignoredIntervals"]),
               "approximateServeEvents": copy.deepcopy(annotations["serveEvents"]),
               "sideSwitches": copy.deepcopy(annotations["sideSwitches"]),
               "annotationScope": copy.deepcopy(reference["annotationScope"]), "gameWindow": reference["gameWindow"],
               "annotation": {"status": "manually-reviewed-kept-and-discarded", "continuousVideoReviewed": True,
                              "evidence": authorization},
               "targetMasks": {"live": False, "serve": False, "end": False, "keep": True},
               "targetContract": {"keep": "actual manually reviewed finalExportIntervals, already padded/joined; no additional padding",
                                  "positiveRangesField": "keepTargets", "alreadyPadded": True,
                                  "negativesOutsideKeepAuthorizedByFullManualReview": True,
                                  "validUniverse": "entire source timeline minus ignoredIntervals",
                                  "boundaryHeads": "fully masked; retainedCoverage is association metadata, not exact live boundaries",
                                  "serveMarkers": "approximate auxiliary evidence only; not used by this run"},
               "videoVerification": "source filename+size+sampled fingerprint; stored full raw SHA-256 inherited from immutable import"}
        cache_id = cache_by_id[row["id"]]["cache"]
        verify(cache_id)
        row["featureCaches"] = {"audiovisual": av_cache(row, Path(cache_id["path"]), config)}
        coverage.append(row)
    exact.sort(key=lambda r: r["id"])
    drafts.sort(key=lambda r: r["id"])
    coverage.sort(key=lambda r: r["id"])
    require((len(exact), sum(len(r["rallies"]) for r in exact), len({r["sourceGroup"] for r in exact})) == (8, 322, 4), "Unexpected exact cohort")
    require((len(drafts), sum(len(r["rallies"]) for r in drafts)) == (3, 123), "Unexpected draft cohort")
    require((len(coverage), sum(len(r["retainedCoverage"]) for r in coverage)) == (7, 275), "Unexpected coverage cohort")
    exact_meta = {"schemaVersion": 1, "name": "neural-expanded-exact-development-8-v1", "createdAt": now,
                  "annotationPolicy": parent["annotationPolicy"], "developmentOnly": True,
                  "protectedSourceGroups": [PROTECTED], "excludedEnvironments": ["beach"],
                  "parentManifest": identity(parent_path), "recordings": exact,
                  "evaluationProtocol": "nested group-held development; exact rows alone select and evaluate; no protected scope",
                  "splitInterpretation": "train/validation are loader-compatible inherited development categories; actual folds use sourceGroup"}
    fold_groups = sorted({r["sourceGroup"] for r in exact})
    all_rows = exact + drafts + coverage
    fold_policy = [{"outerHeldGroup": group,
                    "exactEvaluationIds": [r["id"] for r in exact if r["sourceGroup"] == group],
                    "forbiddenInFitOrSelectionIds": [r["id"] for r in all_rows if r["sourceGroup"] == group],
                    "availableOuterTrainingIds": [r["id"] for r in all_rows if r["sourceGroup"] != group]}
                   for group in fold_groups]
    manifest = {"schemaVersion": 1, "kind": "neural-expanded-quality-tier-manifest-v1", "createdAt": now,
                "name": "nonbeach-exact8-draft3-coverage7-development", "developmentOnly": True,
                "annotationPolicy": parent["annotationPolicy"], "authorization": authorization,
                "protectedSourceGroups": [PROTECTED], "excludedEnvironments": ["beach"],
                "parentManifest": identity(parent_path), "corpusAudit": identity(audit_path),
                "cacheSupplement": identity(supplement_path), "historicalTrainingConsentManifest": historical,
                "exactRows": exact, "draftRows": drafts, "coverageRows": coverage,
                "foldPolicy": fold_policy,
                "groupIsolationRule": "Exclude EVERY tier sharing outer-held OR inner-validation sourceGroup from that fit. Inner selection and outer metrics use exactRows only.",
                "coverageReadoutRule": "keep head learns reviewed export coverage; exact boundary/live readouts and weak keep readout must remain distinct unless a decoder combination is predeclared and selected using exact-only inner folds",
                "aug16Deferred": {"recordings": 11, "environment": "unknown in current metadata",
                                  "features": "embedded native-android-dsp-v1 104-channel features; not interchangeable with verified OpenCV caches",
                                  "reason": "Resolve environment and cross-runtime feature equivalence/regeneration before inclusion; source exports are now user-confirmed manually reviewed, so label review authorization is not the blocker."}}
    exact_text = json.dumps(exact_meta, indent=2, allow_nan=False) + "\n"
    manifest["exactManifest"] = {"path": str(output / "exact-manifest.json"),
                                 "sha256": hashlib.sha256(exact_text.encode()).hexdigest()}
    manifest_text = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    audit = {"schemaVersion": 1, "kind": "neural-expanded-dataset-audit-v1", "createdAt": now,
             "manifestSha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
             "exactManifestSha256": manifest["exactManifest"]["sha256"],
             "recordingCount": len(all_rows), "sourceGroupCount": len({r["sourceGroup"] for r in all_rows}),
             "exact": {"recordings": len(exact), "rallies": sum(len(r["rallies"]) for r in exact),
                       "sourceGroups": fold_groups, "durationSeconds": sum(r["durationSeconds"] for r in exact)},
             "draft": {"recordings": len(drafts), "intervals": sum(len(r["rallies"]) for r in drafts),
                       "durationSeconds": sum(r["durationSeconds"] for r in drafts), "trainingOnly": True},
             "coverage": {"recordings": len(coverage), "associationRanges": 275,
                          "actualExportIntervals": sum(len(r["keepTargets"]) for r in coverage),
                          "approximateServes": sum(len(r["approximateServeEvents"]) for r in coverage),
                          "durationSeconds": sum(r["durationSeconds"] for r in coverage), "trainingOnly": True},
             "groups": {g: [r["id"] for r in all_rows if r["sourceGroup"] == g] for g in sorted({r["sourceGroup"] for r in all_rows})},
             "cacheCounts": {"current104ChannelAV": len(all_rows), "existingOptionalDinoExact": sum("dino" in r["featureCaches"] for r in exact)},
             "verification": "All label/reference/provenance/AV/feedback byte hashes, normalized proxy size, raw-export sampled fingerprint, current cache key/schema/finite values, group plus raw/proxy hash exclusion, explicit quality and target masks. DINO inherited references are unused and were not rehashed.",
             "excluded": [manifest["aug16Deferred"], {"sourceGroup": PROTECTED, "reason": "protected group and lineage"}, {"environment": "beach", "reason": "user scope"}],
             "preparationScript": identity(Path(__file__).resolve()),
             "scopeNotes": ["Exact8 is the only evaluation/selection population; draft3 and coverage7 are training-only.",
                            "Four exact groups and seven total source groups; sources with multiple derivatives remain grouped.",
                            "Draft live labels use full continuous review, but draft serve/end heads remain masked.",
                            "Coverage negatives are now authorized by explicit full manual kept/discarded review; ignored time remains excluded.",
                            "275 association core ranges differ from actual reviewed export coverage; keepTargets stores 252 final intervals without extra padding.",
                            "This follows prior development results and user steering; not an untouched confirmation or production promotion."]}
    output.mkdir(parents=True, exist_ok=True)
    for name, text in (("exact-manifest.json", exact_text), ("manifest.json", manifest_text),
                       ("dataset-audit.json", json.dumps(audit, indent=2, allow_nan=False) + "\n")):
        with (output / name).open("x", encoding="utf-8", newline="\n") as f:
            f.write(text)
    load_manifest(output / "exact-manifest.json", require_videos=False)
    print(json.dumps({"manifest": str(output / "manifest.json"), "manifestSha256": audit["manifestSha256"],
                      "exactManifestSha256": audit["exactManifestSha256"], "exact": audit["exact"],
                      "draft": audit["draft"], "coverage": audit["coverage"], "groups": audit["sourceGroupCount"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(private_value('private-reference-0059')))
    parser.add_argument("--output-dir", type=Path, default=Path(private_value('private-reference-0070')))
    args = parser.parse_args()
    prepare(args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
