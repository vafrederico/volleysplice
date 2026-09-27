"""Publish indexed physical-Pixel measurements of the optimized neural pipeline."""
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
    parent = Path(os.environ.get("VOLLEYCUT_DEVICE_ARTIFACT_ROOT", indexed))
    root = parent / "pixel-shared-video-decoding"
    helper = module("native_report", "report-distilled-mobile-large-native.py")
    independent = module("independent_metrics", "report-production-emulator-comparison.py")
    gold = Path(os.environ.get("VOLLEYCUT_BENCHMARK_GOLD") or
        Path(private_value("private-reference-0218")).parent / "recording-044-label-snapshot.private.json")
    assert helper.digest(gold) == helper.GOLD_SHA256
    labels = read(gold)
    build = read(root / "build-receipt.json")
    report = dict(schema="pixel-shared-video-decoding-v1", status="complete",
        recordingIndex="recording-044", artifactIndex="private-reference-0223",
        model="Frozen highest-recall Distilled MobileNetV3-Large + TCN FP32",
        device="Physical Pixel 10 Pro", transport="Tailscale ADB",
        execution="Native Android; existing ORT CPU execution, four intra-op threads and one inter-op thread; MediaCodec video decoding",
        targetPaddingSeconds=2, joinGapSeconds=3, goldSha256=helper.GOLD_SHA256,
        apkSha256=build["apkSha256"], sourceFingerprints=build["sourceHashes"], runs=[])
    report["shortPreparedPixelsVersusEmulator"] = read(root / "short-cross-device-pixels.json")
    assert report["shortPreparedPixelsVersusEmulator"]["exact"]
    for phase in ("short", "full"):
        folder = root / (phase + "-shared")
        execution = read(root / (phase + "-execution.json"))
        parity = read(root / (phase + "-parity.json"))
        assert execution["status"] == "complete" and parity["exact"]
        result = read(folder / "result.json")
        row = result["results"][0]
        assert result["device"] == "Pixel 10 Pro" and row["neural"]["sharedDecoding"]
        replay = read(folder / "temporal-decoder-parity.json")
        assert replay["passed"]
        observations = read(folder / "observations.json")
        valid = [o for o in observations if "awake" in o]
        foreground = [o["benchmarkForeground"] for o in valid if "benchmarkForeground" in o]
        observed = dict(samples=len(observations), errors=len(observations)-len(valid),
            allAwake=all(o["awake"] for o in valid), allUnlocked=all(not o["locked"] for o in valid),
            foregroundObservations=len(foreground), allObservedForeground=all(foreground) if foreground else None,
            thermalStatuses=sorted(set(o["thermalStatus"] for o in valid if o["thermalStatus"] is not None)),
            batteryTemperatureC=[min(o["batteryTemperatureTenthsC"] for o in valid)/10,
                                 max(o["batteryTemperatureTenthsC"] for o in valid)/10],
            batteryPercent=[valid[0]["batteryLevel"], valid[-1]["batteryLevel"]],
            chargingStates=sorted(set(line for o in valid for line in o["charging"])))
        assert observed["allAwake"] and observed["allUnlocked"] and observed["errors"] == 0
        duration = execution["seconds"]
        truth = labels if phase == "full" else helper.shifted_labels(labels, 180, duration)
        evaluation = helper.evaluate(row, truth, duration)
        t = helper.timings(row)
        video = row["neural"]["video"]
        t.update(areaConversionSeconds=row["profileMs"]["video/yuv_crop_scale_color"]/1000,
            inventoryScanSeconds=row["profileMs"]["video/sample_plan_scan"]/1000,
            encoderQueueBackpressureSeconds=video["queueBackpressureMs"]/1000,
            encoderLoadSeconds=video["encoderLoadMs"]/1000,
            sharedSpanSeconds=video["totalMs"]/1000)
        old_folder = parent / ("pixel-visual-retest/runs/repaired-1" if phase == "short" else "visual-repair/full-recall")
        old = read(old_folder / "result.json")["results"][0]
        item = dict(phase=phase, seconds=duration, sourceSha256=execution["sourceSha256"],
            sourceBytes=execution["sourceBytes"], timings=t, observations=observed,
            evaluation=evaluation, storage=helper.storage(folder, row), parity=parity,
            temporalReplay={key: replay[key] for key in ("passed", "checks")},
            resultSha256=helper.digest(folder / "result.json"),
            avRows=row["sampleRows"], embeddingRows=video["sampleCount"],
            decoder=video["decoder"], hardwareDecoder=video["hardwareDecoder"],
            preparedBufferBytes=video["preparedBufferBytes"],
            historicalPixelReference=dict(timings=helper.timings(old),
                thermalStart=old["thermalStart"], thermalEnd=old["thermalEnd"],
                resultSha256=helper.digest(old_folder / "result.json"),
                scope="Earlier corrected Java area plus independent embedding decode; conditions are not matched and both optimizations differ"))
        if phase == "full":
            item["independentMetricCheck"] = independent.independent_metrics(evaluation, labels, duration)
            item["productionFromSameAv"] = helper.evaluate(dict(rallies=row["productionRallies"]), labels, duration)
        report["runs"].append(item)
    report["frozenGraphs"] = read(root / "full-shared/input-contract.json")["hashes"]
    report["limitations"] = [
        "One fresh-process observation per duration; caches bypassed, no discarded warmup. Transfers and source hashing are excluded from device timing.",
        "Device-condition observations cover host orchestration, including transfer intervals; the measured stage and total times come from the app's internal clock.",
        "The phone was wirelessly charging. Thermal status and battery-temperature observations are recorded; this is not a controlled cool-device comparison.",
        "Historical Pixel reference times combine a different preprocessing implementation and separate decode passes, with unmatched device conditions; do not attribute their entire timing difference to shared decoding alone.",
        "This tests the existing FP32 CPU neural path in the native app harness, not a newly introduced GPU or NNAPI delegate.",
        "The exact comparison is against corrected same-phone outputs. Older Pixel references did not capture prepared-pixel hashes; the new short run does, but no same-phone historical pixel-hash equality is claimed.",
        "Fixed, repeatedly investigated recording with manually reviewed export-derived labels; this is runtime/output qualification, not model selection or a new generalization result.",
        "Short excerpt labels use a nominal 180-second offset for a stream-copy excerpt; full-video metrics are the headline accuracy evidence.",
        "Embedding preparation/inference overlap the AV stage. Their diagnostic timings must not be added again to AV wall time. Score specialists retain their later shared pass.",
    ]
    target = REPO / "docs/research/pixel-shared-video-decoding.json"
    target.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    lines = ["# Pixel shared-decoding Distilled Large benchmark", "",
        "Physical Pixel 10 Pro over Tailscale ADB, using the frozen highest-recall Distilled "
        "MobileNetV3-Large + TCN at FP32. The native area-conversion optimization and shared "
        "AV/embedding decoder are enabled. Both serving-side and side-switch results are included.", "",
        "| Window | Shared AV + embeddings s | Audio s | Rallies ready s | Score specialists s | All results s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in report["runs"]:
        t = item["timings"]
        lines.append("| "+item["phase"]+" | "+" | ".join(f"{t[k]:.3f}" for k in
            ("videoAvSeconds", "audioSeconds", "ralliesReadySeconds", "scoreSeconds", "allReadySeconds"))+" |")
    lines += ["", "For historical same-phone context only:", "",
        "| Window | Earlier corrected Java + independent decoding, all results s | Optimized, all results s |",
        "| --- | ---: | ---: |"]
    for item in report["runs"]:
        lines.append(f"| {item['phase']} | {item['historicalPixelReference']['timings']['allReadySeconds']:.3f} | {item['timings']['allReadySeconds']:.3f} |")
    lines += ["", "These earlier observations have unmatched device conditions and differ in both area conversion "
        "and decode sharing. Their entire timing difference cannot be attributed to shared decoding alone."]
    lines += ["", "## Outputs and accuracy", "",
        "Both runs match the earlier corrected Pixel outputs exactly: selected frame timestamps, "
        "quality features, saved embeddings, fused features, probabilities, rally boundaries and "
        "confidences, production proposals, and both score-specialist feature matrices and decisions. "
        "Independent CPU temporal replay also reproduces every rally boundary. All 240 prepared-input "
        "pixel hashes on the phone match the validated emulator's short run exactly.", ""]
    full = report["runs"][1]
    evaluation = full["evaluation"]
    m = evaluation["primary"]
    lines += [f"The full video produces **{len(evaluation['rallies'])} rallies**, with "
        f"**{evaluation['whollyMissedSavedHumanRallies']}/{evaluation['humanRallies']} wholly missed saved human rallies**. "
        f"At the predeclared 2s symmetric padding: P_pad **{m['P_pad']:.2%}**, R_core **{m['R_core']:.2%}**, "
        f"F1_padP_coreR **{m['F1_padP_coreR']:.2%}**.", "",
        "Positive padded gaps strictly below 3s are joined. Ignored intervals are removed without rejoining. "
        "All four padding cases are independently checked.", "",
        "| Padding each side | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in evaluation["padding"]:
        lines.append(f"| {m['paddingSecondsBeforeAndAfter']}s | {m['P_pad']:.2%} | {m['R_core']:.2%} | {m['F1_padP_coreR']:.2%} | {m['paddedModelExportSeconds']:.3f} | {m['paddedHumanExportSeconds']:.3f} | {m['exportDurationDifferenceSeconds']:+.3f} |")
    lines += ["", "## Device conditions", "",
        "| Window | Observations | Awake/unlocked throughout | Thermal statuses | Battery temperature °C |",
        "| --- | ---: | --- | --- | --- |"]
    for item in report["runs"]:
        o = item["observations"]
        lines.append(f"| {item['phase']} | {o['samples']} | {o['allAwake'] and o['allUnlocked']} | {o['thermalStatuses']} | {o['batteryTemperatureC']} |")
    lines += ["", "Android thermal status 0 means none, 1 light, 2 moderate, and 3 severe. "
        "No thermal override was used. The full run records whether the benchmark remains foreground.", "",
        "## Storage and nested timings", "",
        f"{full['avRows']} AV/fused rows and {full['embeddingRows']} embedding rows. "
        f"The four prepared-image buffers occupy {full['preparedBufferBytes']:,} bytes; this is queue storage, not process memory.", "",
        "| Saved FP32 diagnostic tensor | Bytes |", "| --- | ---: |"]
    for item in full["storage"]["savedDiagnosticFiles"]:
        lines.append(f"| {item['kind']} | {item['bytes']:,} |")
    t = full["timings"]
    lines += ["", f"Within the shared AV stage, area conversion took {t['areaConversionSeconds']:.3f}s, "
        f"neural image preparation {t['embeddingPreparationSeconds']:.3f}s, encoder/readback "
        f"{t['embeddingEncoderSeconds']:.3f}s, and encoder queue backpressure {t['encoderQueueBackpressureSeconds']:.3f}s. "
        f"The TCN took {t['temporalSeconds']:.3f}s. Overlapping measurements are not additive.", "",
        f"Calculated AV104 and contextual AV520 float payloads are {full['storage']['av104Float32BytesCalculated']:,} "
        f"and {full['storage']['contextual520Float32BytesCalculated']:,} bytes respectively. "
        "Specialist feature sizes are recorded in the JSON report.", "", "## Scope", ""]
    lines += ["- "+value for value in report["limitations"]]
    lines += ["", "See the [JSON report](pixel-shared-video-decoding.json), "
        "[runner](../../scripts/benchmark-pixel-shared-decoding.py), "
        "[report generator](../../scripts/report-pixel-shared-decoding.py), and "
        "[emulator qualification](android-shared-video-decoding.md). Exact private inputs resolve "
        "through the external ledger and ignored local environment.", ""]
    target.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(fullTimings=full["timings"], primary=full["evaluation"]["primary"])))


if __name__ == "__main__":
    main()
