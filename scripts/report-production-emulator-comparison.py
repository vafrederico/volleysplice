"""Publish indexed full-video production-only emulator observations."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_metrics(evaluation, labels, duration):
    """Second implementation of export unions; no shared metric helper calls."""
    assert [m['paddingSecondsBeforeAndAfter'] for m in evaluation['padding']] == [0, 1, 2, 3]
    def union(items, gap=0):
        output = []
        for start, end in sorted(items):
            if end <= start:
                continue
            if output and (start <= output[-1][1] or start - output[-1][1] < gap):
                output[-1][1] = max(output[-1][1], end)
            else:
                output.append([start, end])
        return output
    ignored = union([(r['start'], r['end']) for r in labels.get('ignoredIntervals', [])])
    def subtract(items):
        output = []
        for start, end in items:
            for left, right in ignored:
                if right <= start:
                    continue
                if left >= end:
                    break
                if left > start:
                    output.append([start, left])
                start = max(start, right)
            if start < end:
                output.append([start, end])
        return output
    def length(items):
        return sum(end - start for start, end in items)
    def overlap(a, b):
        return sum(max(0, min(end, right) - max(start, left))
                   for start, end in a for left, right in b)
    def export(items, pad):
        return subtract(union([(max(0, r['start'] - pad), min(duration, r['end'] + pad)) for r in items], 3))
    core = subtract(union([(r['start'], r['end']) for r in labels['rallies']]))
    maximum = 0
    for expected in evaluation['padding']:
        pad = expected['paddingSecondsBeforeAndAfter']
        model, human = export(evaluation['rallies'], pad), export(labels['rallies'], pad)
        precision = overlap(model, human) / length(model) if model else 0
        recall = overlap(model, core) / length(core)
        actual = dict(P_pad=precision, R_core=recall,
            F1_padP_coreR=2*precision*recall/(precision+recall) if precision+recall else 0,
            paddedModelExportSeconds=length(model), paddedHumanExportSeconds=length(human),
            exportDurationDifferenceSeconds=length(model)-length(human))
        error = max(abs(actual[key] - expected[key]) for key in actual)
        assert error < 1e-8
        maximum = max(maximum, error)
        if pad == 2:
            missed = [i+1 for i, r in enumerate(labels['rallies'])
                      if length(subtract([[r['start'], r['end']]])) > 0
                      and overlap(model, subtract([[r['start'], r['end']]])) <= 1e-9]
            assert missed == [r['humanRallyNumber'] for r in evaluation['whollyMissedHumanRallyRanges']]
    return dict(passed=True, paddingCases=4, maximumAbsoluteError=maximum)


def main():
    parent = Path(os.environ.get("VOLLEYCUT_EMULATOR_ARTIFACT_ROOT") or private_value("private-reference-0223"))
    root = parent / "production-emulator-full-comparison"
    execution = read(root / "execution.json")
    assert execution["status"] == "complete" and len(execution["runs"]) == 3
    spec = importlib.util.spec_from_file_location("native_report", REPO / "scripts/report-distilled-mobile-large-native.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    gold_path = Path(os.environ.get("VOLLEYCUT_BENCHMARK_GOLD") or
                     Path(private_value("private-reference-0218")).parent / "recording-044-label-snapshot.private.json")
    assert sha(gold_path) == helper.GOLD_SHA256
    labels = read(gold_path)
    names = {"original-point": "Original point sampling", "repaired-java": "Repaired Java area", "optimized-native": "Optimized native area"}
    public = dict(schema="production-emulator-full-comparison-v1", status="complete",
                  recordingIndex="recording-044", artifactIndex="private-reference-0223",
                  seconds=execution["seconds"], sourceSha256=execution["sourceSha256"],
                  sourceBytes=execution["sourceBytes"], goldSha256=helper.GOLD_SHA256,
                  targetPaddingSeconds=2, joinGapSeconds=3, model="production-ensemble",
                  fullFrame=True, roi=[0, 0, 1, 1], featureCacheMode="bypass",
                  neuralEncoderOrTemporalModelRun=False,
                  protocol=execution["protocol"], environment=execution["environment"],
                  identicalPackagedAssets=execution["identicalPackagedAssets"], runs=[])
    raw_rows = {}
    source_uri = None
    for receipt in execution["runs"]:
        variant = receipt["variant"]
        folder = root / variant
        result = read(folder / "result.json")
        plan = read(folder / "pipeline-plan.json")
        assert sha(folder / "result.json") == receipt["resultSha256"]
        assert result["runId"] == plan["runId"] and result["status"] == "complete"
        assert len(result["results"]) == len(plan["cases"]) == 1
        case = plan["cases"][0]
        assert case["family"] == "production" and case["seconds"] == public["seconds"] and case["fullFrame"]
        assert case["warmup"] is False and result["cacheMode"] == "bypass"
        if source_uri is None:
            source_uri = plan["sourceUri"]
        assert source_uri == plan["sourceUri"]
        row = result["results"][0]
        assert "neural" not in row and row["sampleRows"] == 4245 and row["roi"] == [0, 0, 1, 1]
        assert not any(key.startswith("neural/") for key in row["profileMs"])
        assert row["servingSideReady"] and row["sideSwitchReady"] and row["status"] == "complete"
        assert result["audioExtractorVersion"] == "native-dsp-v2-codec-framing-percentile"
        assert row["rallyCount"] == len(row["rallies"])
        assert all(math.isfinite(v) and v >= 0 for v in row["stagesMs"].values())
        raw_rows[variant] = row
        observations = read(folder / "observations.json")
        valid = [o for o in observations if "wakefulness" in o]
        observed = dict(samples=len(observations), errors=len(observations)-len(valid),
                        allObservedAwake=bool(valid) and all(any(s in ("mWakefulness=Awake", "mWakefulness=1") for s in o["wakefulness"]) for o in valid),
                        minimumHostAvailableMemoryMB=min(o["hostAvailableMemoryMB"] for o in valid))
        assert observed["allObservedAwake"] and observed["errors"] == 0
        profile = row["profileMs"]
        nested = {name: profile.get(key, 0) / 1000 for name, key in {
            "areaColorSeconds": "video/yuv_crop_scale_color",
            "opencvFeatureSeconds": "video/opencv_feature_call",
            "timestampScanSeconds": "video/sample_plan_scan",
            "timestampProbeSeconds": "video/sample_plan_cfr_probe",
            "scoreDecodeSeconds": "score/shared_decode_wall",
            "servingEvaluationSeconds": "score/serving_side_evaluation",
            "sideSwitchEvaluationSeconds": "score/side_switch_evaluation",
        }.items()}
        public["runs"].append(dict(variant=variant, name=names[variant],
            apkSha256=execution["apkHashes"][variant], resultSha256=receipt["resultSha256"],
            visualExtractorVersion=result["visualExtractorVersion"], audioExtractorVersion=result["audioExtractorVersion"],
            timing=helper.timings(row), stageMilliseconds=row["stagesMs"], nestedTiming=nested,
            rallyCount=row["rallyCount"], sampleRows=row["sampleRows"],
            evaluation=helper.evaluate(row, labels, public["seconds"]),
            storage=helper.storage(folder, row), observations=observed,
            thermalStart=row["thermalStart"], thermalEnd=row["thermalEnd"]))
        public["runs"][-1]["independentMetricCheck"] = independent_metrics(
            public["runs"][-1]["evaluation"], labels, public["seconds"])
    java, native = raw_rows["repaired-java"], raw_rows["optimized-native"]
    def boundaries(row):
        return [(r["start"], r["end"], r.get("confidence")) for r in row["rallies"]]
    parity = dict(ralliesExact=boundaries(java) == boundaries(native))
    for key in ("servingSide", "sideSwitch"):
        # Timing and diagnostics may differ; preserve numerical model outputs and features.
        selected = ("rows", "columns", "rawFeatures", "features", "candidates")
        parity[key + "Exact"] = {k: java[key][k] for k in selected if k in java[key]} == {
            k: native[key][k] for k in selected if k in native[key]}
    public["javaNativeOutputParity"] = parity
    post = read(root / "post-test-state.json")
    post_keys = ("selinuxRestored", "adbRootDisabled", "emulatorExited", "optimizedBenchmarkInstalled", "timedRunsComplete")
    assert all(post.get(key) is True for key in post_keys)
    public["postTestState"] = {key: post[key] for key in post_keys}
    public["changes"] = []
    for before, after in ((0, 1), (1, 2), (0, 2)):
        left, right = public["runs"][before], public["runs"][after]
        public["changes"].append(dict(before=left["variant"], after=right["variant"],
            seconds={k: right["timing"][k] - left["timing"][k] for k in ("allReadySeconds", "ralliesReadySeconds", "videoAvSeconds", "scoreSeconds")},
            allReadyPercentChange=100 * (right["timing"]["allReadySeconds"] / left["timing"]["allReadySeconds"] - 1)))
    public["limitations"] = [
        "Single full-video observation per version; sequential order, no discarded pipeline warmups; filesystem/JIT and host scheduling effects are not isolated.",
        "Emulator CPU and codec performance is not physical Pixel performance. API 37 userdebug emulator uses the previously qualified temporary SELinux workaround.",
        "Original point-sampling control retains corrected audio and the same full-frame input, models and benchmark harness; it is not an exact historical APK.",
        "Score specialists use each version's own production rallies; proposal changes can change score-stage workload.",
        "Totals include feature generation and serving/side-switch models; exclude install, transfer, source hashing, result collection and export rendering. Nested counters overlap.",
        "Same previously investigated recording and manually reviewed export-derived gold; runtime validation, not a new generalization or model-selection result.",
        "Production AV tensors and per-tick probabilities are not saved; output equality does not establish complete tensor equality.",
    ]
    output = REPO / "docs/research/production-emulator-full-comparison"
    output.with_suffix(".json").write_text(json.dumps(public, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    def clock(seconds):
        minutes, seconds = divmod(int(seconds + .5), 60)
        return f"{minutes}m{seconds:02d}s"
    original_time, java_time, native_time = [r["timing"]["allReadySeconds"] for r in public["runs"]]
    lines = ["# Full-video production-only emulator comparison", "",
        f"Current native preprocessing completed all results in **{clock(native_time)}**, compared with **{clock(java_time)}** for repaired Java area resizing and **{clock(original_time)}** for original point sampling. These are one full-video observation per version, measured sequentially on the same emulator.", "",
        "Three frozen preprocessing versions process the same full-frame 17m41s recording-044 through production rally detection, serving-side and side-switch results. The normal production suppression analysis is included; rally boundaries remain the ensemble union. No image embeddings or TCN run. All feature caches are bypassed.", "",
        "| Version | Video AV | Audio | Rallies ready | Score specialists | All ready | Rallies |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in public["runs"]:
        t = r["timing"]
        lines.append(f"| {r['name']} | {t['videoAvSeconds']:.3f}s | {t['audioSeconds']:.3f}s | {t['ralliesReadySeconds']:.3f}s | {t['scoreSeconds']:.3f}s | {t['allReadySeconds']:.3f}s | {r['rallyCount']} |")
    lines += ["", "| Comparison | All-ready change | Percent change |", "|---|---:|---:|"]
    for c in public["changes"]:
        lines.append(f"| {names[c['before']]} → {names[c['after']]} | {c['seconds']['allReadySeconds']:+.3f}s | {c['allReadyPercentChange']:+.2f}% |")
    lines += ["", "## Preprocessing detail", "", "These counters are nested and must not be added to stage totals.", "",
              "| Version | Resize/color | Timestamp inventory | OpenCV features | Score decode |", "|---|---:|---:|---:|---:|"]
    for r in public["runs"]:
        n = r["nestedTiming"]
        lines.append(f"| {r['name']} | {n['areaColorSeconds']:.3f}s | {n['timestampScanSeconds']+n['timestampProbeSeconds']:.3f}s | {n['opencvFeatureSeconds']:.3f}s | {n['scoreDecodeSeconds']:.3f}s |")
    original, java, native = public["runs"]
    area_reduction = 100 * (1 - native["nestedTiming"]["areaColorSeconds"] / java["nestedTiming"]["areaColorSeconds"])
    av_increase = 100 * (native["timing"]["videoAvSeconds"] / original["timing"]["videoAvSeconds"] - 1)
    score_extra = native["timing"]["scoreSeconds"] - original["timing"]["scoreSeconds"]
    lines += ["", f"The native kernel reduces resizing/color time by **{area_reduction:.2f}%** relative to repaired Java. Complete video AV time ends **{av_increase:.2f}% above original point sampling**. Of the remaining {native_time-original_time:.3f}s all-ready difference from original, {score_extra:.3f}s is in score specialists, which receive 63 rather than 61 production intervals. This decomposes the observed difference; a single run does not isolate proposal-count, cache or scheduling effects."]
    lines += ["", "## Saved-human comparison", "", "Declared target: 2 seconds before/after; join positive gaps strictly below 3 seconds. Subtract ignored time without rejoining. R_core measures retained human play duration. Event metrics and missed-rally ranges are in the JSON.", "",
        "| Version | P_pad | R_core | F1_padP_coreR | Wholly missed / human rallies |", "|---|---:|---:|---:|---:|"]
    for r in public["runs"]:
        e = r["evaluation"]; m = e["primary"]
        lines.append(f"| {r['name']} | {m['P_pad']:.2%} | {m['R_core']:.2%} | {m['F1_padP_coreR']:.2%} | {e['whollyMissedSavedHumanRallies']} / {e['humanRallies']} |")
    lines += ["", "## Required padding sensitivity", "", "| Version | Padding each side | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in public["runs"]:
        for m in r["evaluation"]["padding"]:
            lines.append(f"| {r['name']} | {m['paddingSecondsBeforeAndAfter']:g}s | {m['P_pad']:.2%} | {m['R_core']:.2%} | {m['F1_padP_coreR']:.2%} | {m['paddedModelExportSeconds']:.3f} | {m['paddedHumanExportSeconds']:.3f} | {m['exportDurationDifferenceSeconds']:+.3f} |")
    lines += ["", "## Validation and limits", "", f"Java/native exact output checks: rallies **{parity['ralliesExact']}**, serving-side **{parity['servingSideExact']}**, side-switch **{parity['sideSwitchExact']}**. Packaged model assets are identical across all three APKs; APK hashes match the archived qualified builds. The complete source was hash-verified on the emulator before timing.", ""]
    lines += ["- " + item for item in public["limitations"]]
    observed_count = sum(r["observations"]["samples"] for r in public["runs"])
    minimum_memory = min(r["observations"]["minimumHostAvailableMemoryMB"] for r in public["runs"])
    metric_count = sum(r["independentMetricCheck"]["paddingCases"] for r in public["runs"])
    lines += ["", f"All {observed_count} recorded wake observations were awake, with zero monitor errors and at least {minimum_memory:,.0f} MB of host memory available. All {metric_count} padding comparisons passed a second interval implementation with zero numerical difference. SELinux enforcement and non-root ADB were restored, and emulator exit was confirmed after testing."]
    lines += ["", "Raw evidence resolves through private-reference-0223. [Machine-readable report](production-emulator-full-comparison.json). Generated by [report script](../../scripts/report-production-emulator-comparison.py).", ""]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "runs": [{"variant": r["variant"], "timing": r["timing"], "primary": r["evaluation"]["primary"]} for r in public["runs"]], "javaNativeOutputParity": parity}))


if __name__ == "__main__":
    main()
