"""Publish allowlisted shared-decoder results using ledger-resolved evidence."""
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
from private_ledger import private_value


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    indexed = private_value("private-reference-0223")
    parent = Path(os.environ.get("VOLLEYCUT_EMULATOR_ARTIFACT_ROOT", indexed))
    root = parent / "shared-video-decoding"
    helper = module("native_report", "report-distilled-mobile-large-native.py")
    independent = module("independent_report", "report-production-emulator-comparison.py")
    phases = {phase: read(root / (phase + "-execution.json")) for phase in ("short", "full")}
    assert all(value["status"] == "complete" for value in phases.values())
    assert phases["short"]["apkSha256"] == phases["full"]["apkSha256"]
    gold_path = Path(os.environ.get("VOLLEYCUT_BENCHMARK_GOLD") or
        Path(private_value("private-reference-0218")).parent / "recording-044-label-snapshot.private.json")
    assert helper.digest(gold_path) == helper.GOLD_SHA256
    labels = read(gold_path)
    parity = {phase: read(root / (phase + "-parity.json")) for phase in phases}
    assert all(value["exact"] for value in parity.values())
    public = dict(schema="shared-video-decoding-report-v1", status="complete",
        recordingIndex="recording-044", artifactIndex="private-reference-0223",
        model="Frozen highest-recall Distilled MobileNetV3-Large + TCN FP32",
        selection=dict(draw=3407, epoch=15, calibrationTargetRecall=.99),
        targetPaddingSeconds=2, joinGapSeconds=3, goldSha256=helper.GOLD_SHA256,
        apkSha256=phases["full"]["apkSha256"], sourceSha256=phases["full"]["sourceSha256"],
        sourceBytes=phases["full"]["sourceBytes"], seconds=phases["full"]["seconds"],
        environment=phases["full"]["environment"], parity=parity, runs=[])
    public["sourceFingerprints"] = read(root / "source-hashes.json")
    contract = read(root / "full-shared/input-contract.json")
    assert contract["selectionMode"] == "recall"
    assert contract["studentWeightsSha256"] == "0be04fbf8e5633241a473f49176d9d27184f45ea94f90dd454d73fd4109d7bc6"
    assert contract["temporalWeightsSha256"] == "e79a626bb8a01a0de060d759721340b06f1bae2c90033ed8647448e4a9af5488"
    public["frozenGraphs"] = {key: contract[key] for key in (
        "hashes", "studentWeightsSha256", "temporalWeightsSha256")}
    locations = {
        "short-independent": root / "short-independent", "short-shared": root / "short-shared",
        "full-independent": parent / "native-area-full-video/native-recall",
        "full-shared": root / "full-shared",
    }
    for name, folder in locations.items():
        row = read(folder / "result.json")["results"][0]
        is_full = name.startswith("full")
        duration = public["seconds"] if is_full else 120
        truth = labels if is_full else helper.shifted_labels(labels, 180, 120)
        evaluation = helper.evaluate(row, truth, duration)
        storage = helper.storage(folder, row)
        video = row["neural"]["video"]
        timing = helper.timings(row)
        timing.update(sharedDecoder=video.get("sharedDecoding", False),
            sharedNestedVideoSpanSeconds=video.get("totalMs", 0)/1000 if video.get("sharedDecoding") else 0,
            sharedQueueBackpressureSeconds=video.get("queueBackpressureMs", 0)/1000,
            sharedFinishWaitSeconds=row["profileMs"].get("video/shared_consumer_finish_wait", 0)/1000,
            inventoryScanSeconds=row["profileMs"].get("video/sample_plan_scan", 0)/1000,
            separateEmbeddingInventorySeconds=video.get("samplePlanMs", 0)/1000,
            pixelHashDiagnosticSeconds=video.get("pixelHashMs", 0)/1000)
        replay_path = folder / "temporal-decoder-parity.json"
        raw_replay = read(replay_path)
        replay = {key: raw_replay[key] for key in ("passed", "checks")}
        replay["scope"] = "Actual emulator fused tensors: native temporal inference and rally decoder versus independent CPU replay"
        assert replay["passed"]
        observations = read(folder / "observations.json")
        valid = [o for o in observations if "wakefulness" in o]
        observed = dict(samples=len(observations), errors=len(observations)-len(valid),
            allObservedAwake=bool(valid) and all(any(s in ("mWakefulness=Awake", "mWakefulness=1")
                for s in o["wakefulness"]) for o in valid),
            minimumHostAvailableMemoryMB=min(o["hostAvailableMemoryMB"] for o in valid))
        result = dict(variant=name, timings=timing, evaluation=evaluation, storage=storage,
            observations=observed, temporalReplay=replay, avRows=row["sampleRows"],
            embeddingRows=video["sampleCount"], resultSha256=helper.digest(folder / "result.json"),
            sharedBufferBytes=video.get("preparedBufferBytes", 0),
            uniqueSelectedImages=video.get("uniqueSelectedImages"))
        if is_full:
            result["independentMetricCheck"] = independent.independent_metrics(evaluation, labels, duration)
            result["productionFromSameAv"] = helper.evaluate(dict(rallies=row["productionRallies"]), labels, duration)
        public["runs"].append(result)
    by_name = {row["variant"]: row for row in public["runs"]}
    old = by_name["full-independent"]["timings"]
    new = by_name["full-shared"]["timings"]
    public["fullSavings"] = dict(allReadySeconds=old["allReadySeconds"]-new["allReadySeconds"],
        allReadyPercent=100*(1-new["allReadySeconds"]/old["allReadySeconds"]),
        ralliesReadySeconds=old["ralliesReadySeconds"]-new["ralliesReadySeconds"],
        ralliesReadyPercent=100*(1-new["ralliesReadySeconds"]/old["ralliesReadySeconds"]))
    production_folder = parent / "production-emulator-full-comparison/optimized-native"
    production_row = read(production_folder / "result.json")["results"][0]
    public["previousProductionOnlyObservation"] = dict(
        timings=helper.timings(production_row), rallyCount=production_row["rallyCount"],
        resultSha256=helper.digest(production_folder / "result.json"),
        scope="Separate earlier emulator run; score specialists operate on production's own rally proposals")
    public["limitations"] = [
        "Emulator observation, not physical Pixel performance; API 37 x86_64, about 4 GB guest RAM and 24 GB configured storage.",
        "The native neural harness retains its existing ONNX Runtime CPU execution with four intra-op threads and one inter-op thread; this is not a phone GPU benchmark.",
        "One observation per path/window. Short A/B uses the same APK; full independent reference is the previously saved native-area run. Filesystem/JIT warmth and host scheduling are not controlled.",
        "Shared path currently qualifies fresh, unrotated, zero-start full-frame MobileNet input. Cached AV uses the independent path. Score-specialist decode remains a separate, already shared pass after rally selection.",
        "AV and embedding work overlap. Shared embedding span, inference and prepare measurements are nested within AV; do not add them to AV wall time.",
        "Short runs include optional prepared-pixel SHA256 diagnostics. All runs include saved diagnostic tensors and candidate reports.",
        "Fixed, repeatedly investigated recording and manually reviewed export-derived labels; no new generalization result or selection of model weights/thresholds.",
        "Short excerpt label alignment uses a nominal 180-second offset for a stream-copy excerpt; full-video metrics are the headline accuracy evidence.",
    ]
    target = REPO / "docs/research/android-shared-video-decoding.json"
    target.write_text(json.dumps(public, indent=2)+"\n", encoding="utf-8")
    lines = ["# Android shared video decoding experiment", "",
        "Frozen highest-recall Distilled MobileNetV3-Large + TCN FP32 on recording-044. "
        "AV and embeddings now share one timestamp inventory and asynchronous MediaCodec pass. "
        "Each consumer still prepares its own input directly from the original YUV image.", "",
        "The experimental decoder hook is in the actual Android app. The opt-in neural consumer "
        "is wired through the pipeline benchmark source set; the normal production app still runs its existing ensemble.", "",
        "## Runtime", "", "Seconds, including feature generation and both score specialists. "
        "Shared AV includes concurrent embedding preparation/inference; the separate embedding-pass column is zero for shared runs.", "",
        "| Window/path | AV incl. shared work | Audio | Separate embeddings | Rallies ready | Score specialists | All ready |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in public["runs"]:
        t = row["timings"]
        values = [t[k] for k in ("videoAvSeconds", "audioSeconds", "embeddingVideoPassSeconds", "ralliesReadySeconds", "scoreSeconds", "allReadySeconds")]
        lines.append("| "+row["variant"]+" | "+" | ".join(f"{v:.3f}" for v in values)+" |")
    savings = public["fullSavings"]
    lines += ["", f"Full-video all-ready savings: {savings['allReadySeconds']:.3f}s ({savings['allReadyPercent']:.2f}%). "
        f"Rallies-ready savings: {savings['ralliesReadySeconds']:.3f}s ({savings['ralliesReadyPercent']:.2f}%).", "",
        f"For context, the earlier optimized production-only run took {public['previousProductionOnlyObservation']['timings']['allReadySeconds']:.3f}s "
        f"for all results on {production_row['rallyCount']} rally proposals. Its score workload differs from the neural output; "
        "this is a separate observation, not an isolated measure of adding the neural model.", "",
        "## Output validation", "",
        "Short A/B requires exact equality of every prepared input SHA256, selected PTS, quality row, "
        "saved embedding, fused feature, probability, rally (including confidence), production rally, "
        "and both score-specialist inputs and decisions. Full comparison requires the same checks "
        "except pixel hashes, which the saved full reference did not capture. Every exact gate passed. "
        "Independent CPU temporal replay also reproduces rally boundaries.", "",
        "Primary accuracy uses symmetric 2s padding, joins positive gaps strictly below 3s, "
        "and subtracts ignored intervals without rejoining. The target was fixed before evaluation.", "",
        "| Full-video output | Rallies | Wholly missed saved human rallies | P_pad | R_core | F1_padP_coreR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in ("full-independent", "full-shared"):
        e = by_name[name]["evaluation"]
        m = e["primary"]
        lines.append(f"| {name} | {len(e['rallies'])} | {e['whollyMissedSavedHumanRallies']}/{e['humanRallies']} | {m['P_pad']:.2%} | {m['R_core']:.2%} | {m['F1_padP_coreR']:.2%} |")
    e = by_name["full-shared"]["productionFromSameAv"]
    m = e["primary"]
    lines.append(f"| Production from same AV | {len(e['rallies'])} | {e['whollyMissedSavedHumanRallies']}/{e['humanRallies']} | {m['P_pad']:.2%} | {m['R_core']:.2%} | {m['F1_padP_coreR']:.2%} |")
    lines += ["", "### Required padding sensitivity", "",
        "Full independent/shared outputs are identical; the following values apply to both.", "",
        "| Padding each side | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in by_name["full-shared"]["evaluation"]["padding"]:
        lines.append(f"| {m['paddingSecondsBeforeAndAfter']}s | {m['P_pad']:.2%} | {m['R_core']:.2%} | {m['F1_padP_coreR']:.2%} | {m['paddedModelExportSeconds']:.3f} | {m['paddedHumanExportSeconds']:.3f} | {m['exportDurationDifferenceSeconds']:+.3f} |")
    full = by_name["full-shared"]
    lines += ["", "## Storage and implementation", "",
        f"{full['avRows']} AV/fused rows; {full['embeddingRows']} embedding rows. "
        f"The bounded prepared-input pool holds four images ({full['sharedBufferBytes']:,} bytes). "
        "This is additional queue storage, not total process memory. It retains no decoder Image after the callback.", "",
        "| Saved diagnostic tensor | Bytes |", "| --- | ---: |"]
    for f in full["storage"]["savedDiagnosticFiles"]:
        lines.append(f"| {f['kind']} FP32 | {f['bytes']:,} |")
    lines += ["", "AV104 payload size is "
        f"{full['storage']['av104Float32BytesCalculated']:,} bytes; contextual AV520 is "
        f"{full['storage']['contextual520Float32BytesCalculated']:,} bytes. These are calculated "
        "float payloads, not additional saved cache files in this bypassed-cache run. "
        "The JSON report also records decoded and persisted specialist-feature payload sizes."]
    lines += ["", "Storage is unchanged by sharing. Separate preparation preserves AV area-resize and "
        "MobileNet bilinear-letterbox contracts. Repeated target rows reuse one source image and its embedding; "
        "the neural exclusive-end rule stays independent of AV nearest-frame selection. The encoder uses a "
        "bounded worker queue while the decoder owns and releases YUV images. Failures propagate and drain workers.", "",
        "## Scope and reproducibility", ""]
    lines += ["- "+item for item in public["limitations"]]
    lines += ["", "200 JVM tests and 8 emulator instrumentation tests passed, including shared-consumer "
        "AV parity, repeated/terminal samples, consumer failure propagation and native color/resize checks. "
        "Exact runtime inputs resolve through the external ledger and ignored environment. "
        "See the [machine-readable report](android-shared-video-decoding.json), "
        "[runner](../../scripts/benchmark-shared-video-decoding.py), "
        "[exact comparison](../../scripts/validate-shared-video-decoding.py), and "
        "[report generator](../../scripts/report-shared-video-decoding.py).", ""]
    target.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(public["fullSavings"]))


if __name__ == "__main__":
    main()
