"""Qualify unchanged predictions and summarize paired-variant execution costs.

This is runtime qualification, not a model-ranking or new accuracy experiment.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
from private_ledger import private_value


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def qualification(single, paired):
    result = {}
    for candidate in paired["results"]:
        model = candidate["modelId"]
        reference = single[model]
        checks = {
            "ralliesAndConfidence": candidate["rallies"] == reference["rallies"],
            "embeddings": candidate["neural"]["embeddingSha256"] == reference["neural"]["embeddingSha256"],
            "quality": candidate["neural"]["video"]["quality"] == reference["neural"]["video"]["quality"],
            "selectedPresentationTimes": candidate["neural"]["video"]["sampleTimestamps"] == reference["neural"]["video"]["sampleTimestamps"],
            "audio": candidate["audioSha256"] == reference["audioSha256"],
            "servingSideFeaturesAndDecisions": candidate["servingSide"] == reference["servingSide"],
            "sideSwitchFeaturesAndDecisions": candidate["sideSwitch"] == reference["sideSwitch"],
            "noSuppression": not candidate["suppressionApplied"],
        }
        result[model] = checks
    return {"passed": all(all(checks.values()) for checks in result.values()), "checks": result}


def summarize(row):
    paired = "results" in row
    stages = row["primaryStageMilliseconds"] if paired else row["stageMilliseconds"]
    outputs = row["results"] if paired else [row]
    video = outputs[0]["neural"]["video"]
    profile = row["primaryProfileMilliseconds"] if paired else row["profileMilliseconds"]
    return dict(
        seconds=row["durationSeconds"], allReadySeconds=row["allReadyMs"] / 1000,
        sharedVideoAvEmbeddingsSeconds=stages.get("video_decode_and_features", 0) / 1000,
        audioSeconds=stages.get("audio_decode_and_features", 0) / 1000,
        contextualizeSeconds=stages.get("contextualize", 0) / 1000,
        rallyStageSeconds=stages.get("rally_inference", 0) / 1000,
        primaryScoresSeconds=stages.get("score_specialists", 0) / 1000,
        alternateScoresSeconds=row.get("alternateScoreMs", 0) / 1000,
        independentEmbeddingPassSeconds=profile.get("neural/embedding_video_pass", 0) / 1000,
        imagePreparationSeconds=video["prepareMs"] / 1000,
        primaryEncoderAndReadbackSeconds=video["encoderAndReadbackMs"] / 1000,
        encoderProfilesMilliseconds=video.get("encoderProfiles", {}),
        videoProfileMilliseconds={key: value for key, value in profile.items() if key.startswith("video/")},
        primaryScoreProfileMilliseconds={key: value for key, value in profile.items() if key.startswith("score/")},
        alternateScoreProfileMilliseconds=row.get("alternateScoreProfileMilliseconds", {}),
        queueBackpressureSeconds=video.get("queueBackpressureMs", 0) / 1000,
        uniquePreparedImages=video.get("uniqueSelectedImages", video["sampleCount"]),
        preparedBufferBytes=video.get("preparedBufferBytes"),
        sharedImagePreparation=row.get("sharedImagePreparation", False),
        encoderScheduling=video.get("encoderScheduling", "independent"),
        embeddingBytes=sum(value["neural"]["embeddingBytes"] for value in outputs),
        av104PayloadBytes=outputs[0]["sampleRows"] * 104 * 4,
        contextual520PayloadBytes=outputs[0]["sampleRows"] * 520 * 4,
        savedResultLoadMedianMs=statistics.median(row["savedResultLoadMs"]) if paired else None,
        savedResultLoadMaxMs=max(row["savedResultLoadMs"]) if paired else None,
        persistBothMs=row.get("persistBothMs"),
        cache=row["results"][0]["cache"] if paired else row["cache"],
        rallies={value["modelId"]: len(value["rallies"]) for value in outputs},
        temporalMilliseconds={value["modelId"]: value["profileMilliseconds"].get("neural/temporal_inference") for value in outputs},
        memoryAtPrimaryCompletion={key: outputs[0][key] for key in ("pssKilobytes", "javaHeapUsedBytes")},
    )


def compare_full_references(root, current):
    """Compare boundaries to earlier qualified outputs, never reuse their timings."""
    references = {
        "distilled-large-recall-v1": "shared-video-decoding/full-shared/result.json",
        "distilled-large-f1-v1": "visual-repair/full-f1/result.json",
    }
    checks = {}
    for row in current["results"]:
        reference = read(root / references[row["modelId"]])["results"][0]
        expected, actual = reference["rallies"], row["rallies"]
        boundaries = lambda rallies: [(item["start"], item["end"]) for item in rallies]
        same = boundaries(actual) == boundaries(expected)
        checks[row["modelId"]] = {
            "rallies": len(actual), "referenceRallies": len(expected), "boundariesExact": same,
            "maximumConfidenceDifference": max((abs(a["confidence"] - b["confidence"])
                for a, b in zip(actual, expected)), default=0) if same else None,
            "selectedPresentationTimesExact": row["neural"]["video"]["sampleTimestamps"] ==
                reference["neural"]["video"]["sampleTimestamps"],
        }
    return {"checks": checks, "scope": "Boundary and timestamp regression check against earlier frozen outputs. Not a same-session full single-variant performance control; the F1 reference is from a different device."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualify-short", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    indexed = private_value("private-reference-0223")
    root = Path(os.environ.get("VOLLEYCUT_DEVICE_ARTIFACT_ROOT", indexed)) / "paired-variant-experiment"
    names = ["short-recall-refresh", "short-f1-cached", "short-paired-refresh"]
    rows = {name: read(root / name / "result.json") for name in names}
    execution = [read(root / name / "execution.json") for name in names]
    assert all(item["status"] == "complete" for item in execution)
    assert len({item["apkSha256"] for item in execution}) == 1
    assert len({item["sourceSha256"] for item in execution}) == 1
    singles = {rows[name]["modelId"]: rows[name] for name in names[:2]}
    qualified = qualification(singles, rows[names[2]])
    qualified["passed"] = qualified["passed"] and all(rows[names[1]]["cache"].values())
    qualified["cachedSwitchReusedAv"] = all(rows[names[1]]["cache"].values())
    (root / "short-qualification.json").write_text(json.dumps(qualified, indent=2), encoding="utf-8")
    print(json.dumps(qualified))
    if not qualified["passed"]:
        raise SystemExit(1)
    if args.qualify_short:
        return
    full = read(root / "full-paired-refresh/result.json")
    full_execution = read(root / "full-paired-refresh/execution.json")
    assert full_execution["status"] == "complete"
    assert full_execution["apkSha256"] == execution[0]["apkSha256"]
    rows["full-paired-refresh"] = full
    summary = {"schema": "paired-neural-variants-v1", "recordingIndex": "recording-044",
        "artifactIndex": "private-reference-0223", "platform": "API 37 x86_64 Android emulator",
        "physicalPhoneTested": False, "apkSha256": execution[0]["apkSha256"],
        "qualification": qualified, "runs": {name: summarize(row) for name, row in rows.items()},
        "fullReferenceComparison": compare_full_references(root.parent, full),
        "timingNotes": ["Sequential fresh-process observations; one sample per configuration, no discarded warmup.",
            "Decode/AV/preparation and encoder work overlap; their nested counters must not be added to stage wall time.",
            "Saved-result loading measures local deserialization and editor-seed construction, not tap-to-first-paint.",
            "Source bytes and emulator virtual storage are NAS-backed. These are not phone latency measurements.",
            "Full paired timing is directly measured; no same-session full single-variant control is implied."]}
    summary["shortComparison"] = {
        "sequentialCachedSwitchTotalSeconds": (rows[names[0]]["allReadyMs"] + rows[names[1]]["allReadyMs"]) / 1000,
        "pairedTotalSeconds": rows[names[2]]["allReadyMs"] / 1000,
        "additionalVersusRecallSeconds": (rows[names[2]]["allReadyMs"] - rows[names[0]]["allReadyMs"]) / 1000,
    }
    if not args.output:
        raise ValueError("Choose the public report output explicitly")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["shortComparison"]))


if __name__ == "__main__":
    main()
