"""Publish indexed summaries of independently verified browser observations.

This consumes receipts; it does not train, select checkpoints or run inference.
Exact media and artifact roots remain in the external private ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.private_ledger import private_value

RUNS = (
    ("pilot-recall-file-wasm1", "120s / WASM 1 thread", "pilot", "recall", "wasm", 1, False),
    ("pilot-recall-file-wasm4", "120s / WASM 4 threads", "pilot", "recall", "wasm", 4, True),
    ("pilot-recall-file-webgpu1", "120s / WebGPU", "pilot", "recall", "webgpu", 1, False),
    ("full-recall-file-webgpu1", "Full / highest recall", "full", "recall", "webgpu", 1, False),
    ("full-f1-file-webgpu1", "Full / highest F1", "full", "f1", "webgpu", 1, False),
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_asset_map(value):
    output = {}
    for name, row in value.items():
        if Path(name).name != name or "\\" in name or ":" in name:
            raise ValueError("Asset receipt must use a component name")
        output[name] = {key: row[key] for key in ("sizeBytes", "sha256")}
    return output


def load_run(directory, label, scope, selection, provider, threads, isolation):
    result_path = directory / "result.json"
    result = read(result_path)
    summary = read(directory / "independent-summary.json")
    if not summary["passed"] or summary["resultSha256"] != sha(result_path):
        raise ValueError("Independent summary is not bound to the completed run")
    if summary.get("timingIntegrity", {}).get("clean") is not True:
        raise ValueError("Only clean timings belong in the matched report")
    runtime = summary["runtime"]
    if result["source"].get("transport") != "file" or result["timingIntegrity"].get("sourceTransport") != "file" \
            or result["sourceIo"]["requests"] != 0:
        raise ValueError("Primary measurements must use the product File path without HTTP media requests")
    if (summary["source"]["scope"], summary["model"]["selection"], runtime["requestedExecutionProviders"],
        runtime["wasmThreads"], runtime["crossOriginIsolated"]) != (scope, selection, [provider], threads, isolation):
        raise ValueError("Run identity does not match its published label")
    if summary["source"]["recordingIndex"] != "recording-044" or summary["targetPaddingSeconds"] != 2 \
            or summary["joinGapSeconds"] != 3 or summary["model"]["recallCalibrationTargetPercent"] != 99:
        raise ValueError("Unexpected source or metric configuration")
    if sorted(case["id"] for case in summary["avCases"]) != ["corrected", "legacy"]:
        raise ValueError("Both AV variants are required")
    if any("scores" not in case or "productionEvaluation" not in case for case in summary["avCases"]):
        raise ValueError("Score analysis and production comparison are required")
    if runtime["version"] != "1.22.0" or summary["host"]["browserVersion"] != "153.0.8010.48" \
            or summary["host"]["cpu"].strip() != "AMD Ryzen 9 5900X 12-Core Processor":
        raise ValueError("Runtime/host differs from the declared experiment environment")
    if provider == "webgpu" and (runtime["adapter"]["vendor"], runtime["adapter"]["architecture"],
                                 runtime["adapter"]["isFallbackAdapter"]) != ("nvidia", "ampere", False):
        raise ValueError("Unexpected WebGPU adapter")
    if result.get("sourceGuard", {}).get("passed") is not True:
        raise ValueError("Measurements must verify unchanged browser source")
    return dict(label=label, evaluation=summary,
                runtimeAssets=checked_asset_map(result["runtimeFiles"]),
                existingProductionAssets=checked_asset_map(result["productionRuntimeFiles"]),
                sourceCodeSha256=result["code"], productionCodeSha256=result["productionCode"],
                sourceCodeIntegrity=result.get("sourceGuard"),
                sourceIo={key: result["sourceIo"].get(key) for key in
                          ("requests", "requestedBytes", "sourceBytesRead", "closedBeforeFinish")})


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def pct(x):
    return f"{100 * x:.2f}%"


def sec(ms):
    return f"{ms / 1000:.3f}"


def markdown(report):
    runs = report["runs"]
    full = [r for r in runs if r["evaluation"]["source"]["scope"] == "full"]
    main_results = []
    for selection in ("recall", "f1"):
        run = next(r for r in full if r["evaluation"]["model"]["selection"] == selection)
        case = next(c for c in run["evaluation"]["avCases"] if c["id"] == "corrected")
        m = case["neuralEvaluation"]["primary"]
        total = case["scores"]["neuralAllReadyMs"] / 1000
        main_results.append(f'highest {"F1" if selection == "f1" else "recall"}: '
                            f'{int(total // 60)}m{total % 60:04.1f}s all ready, '
                            f'{pct(m["P_pad"])} P_pad / {pct(m["R_core"])} R_core / '
                            f'{pct(m["F1_padP_coreR"])} F1_padP_coreR, '
                            f'{case["neuralEvaluation"]["whollyMissedSavedHumanRallies"]} wholly missed human rallies')
    recall_cf = next(row for row in report["counterfactual"]["selections"] if row["selection"] == "recall")
    alignment = recall_cf["causalEmbeddingSelection"]
    alignment_note = (
        f'The recall diagnostic exposes a clock mismatch: at most '
        f'{recall_cf["maximumTimestampDisplacementSeconds"] * 1000:.1f}ms of AV timestamp displacement '
        f'causes {alignment["rowsUsingOlderTokensInLegacy"]:,}/{recall_cf["avRows"]:,} rows to select an '
        f'older nominal 2 Hz embedding, with a {alignment["maximumTokenTimestampDifferenceSeconds"]:.1f}s token lag. '
        'Keeping current AV values but using the corrected fusion/decode grid recovers most of the recall '
        'selection\'s F1 improvement. The highest-F1 choice does not improve under the bundled correction; '
        'the complete mode comparison must be retained rather than claiming a universal accuracy gain.')
    timings, quality, events, stages, padding, storage, counterfactual = [], [], [], [], [], [], []
    for run in runs:
        e = run["evaluation"]
        for case in e["avCases"]:
            av = "current AV" if case["id"] == "legacy" else "area/nearest AV"
            title = f'{run["label"]} / {av}'
            timings.append([title, sec(case["timings"]["neuralRalliesReadyMs"]),
                            sec(case["scores"]["neuralAllReadyMs"]),
                            sec(case["scores"]["productionAllReadyMs"])])
            for model, key in (("neural", "neuralEvaluation"), ("production", "productionEvaluation")):
                values = case[key]
                metric = values["primary"]
                if e["source"]["scope"] == "full":
                    quality.append([title + " / " + model, len(values["rallies"]),
                                    values["whollyMissedSavedHumanRallies"], pct(metric["P_pad"]),
                                    pct(metric["R_core"]), pct(metric["F1_padP_coreR"]),
                                    f'{metric["paddedModelExportSeconds"]:.3f}'])
                    event = values["event"]
                    events.append([title + " / " + model, event["matchedRallies"],
                                   pct(event["eventPrecision"]), pct(event["eventRecall"]), pct(event["eventF1"])])
                for p in values["padding"]:
                    padding.append([title + " / " + model, f'{p["paddingSecondsBeforeAndAfter"]:g}s',
                                    pct(p["P_pad"]), pct(p["R_core"]), pct(p["F1_padP_coreR"]),
                                    f'{p["paddedModelExportSeconds"]:.3f}', f'{p["paddedHumanExportSeconds"]:.3f}',
                                    f'{p["exportDurationDifferenceSeconds"]:+.3f}'])
            if e["source"]["scope"] == "full":
                t, b, s = case["timings"], e["embeddingTimings"], e["timings"]
                stages.append([title, sec(s["runtimeInitializeMs"] + s["sourceProbeMs"] + s["modelLoadingMs"]),
                               sec(b["pipelineMs"]), sec(t["avMs"]),
                               sec(b["tokenRoundingMs"] + t["fusionMs"] + t["temporalMs"] + t["decoderMs"]),
                               sec(t["productionMs"]), sec(case["scores"]["neural"]["allReadyAdditionalMs"])])
                n = case["rows"]
                storage.append([title, n, f'{n * 104 * 4 / 1e6:.3f}',
                                f'{n * 3952 * 4 / 1e6:.3f}', f'{n * 4 * 4 / 1e6:.3f}'])
    nested = [[r["label"], *[sec(r["evaluation"]["embeddingTimings"][k]) for k in
                            ("canvasReadbackMs", "prepareMs", "encoderAndReadbackMs", "decodeAndOtherMs")]] for r in full]
    assets = [[r["label"], f'{sum(x["bytes"] for x in r["evaluation"]["model"]["graphFiles"]) / 1e6:.3f}',
               f'{sum(x["sizeBytes"] for x in r["runtimeAssets"].values()) / 1e6:.3f}'] for r in full]
    for reference in report["desktopReferences"]:
        title = "Saved desktop / highest " + ("F1" if reference["selection"] == "f1" else "recall")
        values = reference["evaluation"]
        m = values["primary"]
        quality.append([title, len(values["rallies"]), values["whollyMissedSavedHumanRallies"],
                        pct(m["P_pad"]), pct(m["R_core"]), pct(m["F1_padP_coreR"]),
                        f'{m["paddedModelExportSeconds"]:.3f}'])
        for p in values["padding"]:
            padding.append([title, f'{p["paddingSecondsBeforeAndAfter"]:g}s', pct(p["P_pad"]),
                            pct(p["R_core"]), pct(p["F1_padP_coreR"]), f'{p["paddedModelExportSeconds"]:.3f}',
                            f'{p["paddedHumanExportSeconds"]:.3f}', f'{p["exportDurationDifferenceSeconds"]:+.3f}'])
    for selection in report["counterfactual"]["selections"]:
        for case in selection["cases"]:
            value = case["evaluation"]
            m = value["primary"]
            title = f'Frozen replay / {selection["selection"]} / AV={case["avValues"]}, time={case["fusionAndDecoderTimeline"]}'
            counterfactual.append([selection["selection"], case["avValues"], case["fusionAndDecoderTimeline"],
                                   "crossed" if case["counterfactual"] else "observed replay", case["rallyCount"],
                                   value["whollyMissedSavedHumanRallies"], pct(m["P_pad"]), pct(m["R_core"]),
                                   pct(m["F1_padP_coreR"])])
            for p in value["padding"]:
                padding.append([title, f'{p["paddingSecondsBeforeAndAfter"]:g}s', pct(p["P_pad"]),
                                pct(p["R_core"]), pct(p["F1_padP_coreR"]), f'{p["paddedModelExportSeconds"]:.3f}',
                                f'{p["paddedHumanExportSeconds"]:.3f}', f'{p["exportDurationDifferenceSeconds"]:+.3f}'])
    return "\n\n".join([
        "# Distilled MobileNetV3-Large in the browser",
        "Both frozen FP32 selections completed actual-video browser feature generation, TCN inference, "
        "and serving-side/side-switch analysis on the full 1061.016489-second recording-044. "
        "The Android emulator was stopped before browser work; measured browser runs and CPU replay were serialized.",
        "With the corrected AV option and qualified portable WebGPU export, " + "; ".join(main_results) + ". "
        "These complete totals include all required feature generation and both score specialists. Current AV "
        "and production comparisons remain separate below; the production default has not changed.",
        "This is a desktop Chrome experiment on one previously investigated recording, not a mobile-browser "
        "qualification or new training/selection result. Labels were hidden from inference. The saved 37-rally "
        "gold is manually reviewed export coverage with edits, not independently precise serve-contact/dead-ball annotation.",
        "## Readiness times",
        table(["Scope / runtime / AV", "Rallies ready s", "Neural all ready s", "Production shared-pass estimate s"], timings),
        "Each row includes all required feature generation for its proposal set. Neural all-ready adds production "
        "serve/state evidence and the real production score-specialist entry point, including its additional video "
        "sampling, serving-side features and side-switch features. Export rendering is excluded. Rows are single "
        "observations, with new browser profiles per run and no discarded pipeline warmup.",
        "Totals are reconstructed stage sums: both AV variants reuse one embedding pass, and later passes have "
        "warmer models/filesystem caches. Production is a shared-AV, warm-runtime estimate; it is not a separately "
        "launched cold production app. Corrected AV follows current AV in each run and reuses loaded score models. "
        "Diagnostic artifact writes are excluded from browser readiness but recorded separately in JSON; Android "
        "benchmark timing includes diagnostic writes. Do not interpret desktop/emulator totals as a hardware speed comparison.",
        "Measured runs use the actual file input and shipped browser File/Blob media path. The selected file is "
        "NAS-backed, without copying it to a local drive. Source I/O and browser decode are included; this is "
        "not phone-local-file latency. Earlier HTTP range-server experiments exposed media-fetch retries on the "
        "full source (`net::ERR_NO_BUFFER_SPACE` on two open-ended requests). They remain private diagnostics "
        "and are excluded from the primary timing matrix. The "
        "file path removes that artificial transport while preserving the same video and frozen models.",
        "WASM 1 thread and WebGPU ran without cross-origin isolation, matching current hosting capability. "
        "WASM 4 threads required isolation headers. WebGPU used a real NVIDIA Ampere adapter; operator placement "
        "was not profiled, so mixed WASM/CPU execution remains possible. The pinned runtime is ONNX Runtime Web "
        "1.22.0, Chrome 153.0.8010.48, Windows x64, Ryzen 9 5900X. The same all-backends runtime bundle was used "
        "for all measured profiles. WASM uses the original encoder graph; WebGPU uses the equivalent portable "
        "pooling graph described below. Timing compares these qualified deployment configurations, not an isolated "
        "execution-provider switch on byte-identical graphs.",
        "## Full-video accuracy",
        "Target padding is 2 seconds before and after. Positive gaps join only when strictly below 3 seconds. "
        "Ignored intervals are subtracted from model, human-core and padded-human unions without rejoining. "
        "R_core measures retained human-core time; it is not event recall. Wholly missed means zero retained "
        "nonignored human core after this padding and joining. Counts can therefore differ from event matching. "
        "Both selections retain their original 99% inner-calibration target; that target is not a promise of "
        "99% recall on each recording or runtime.",
        "Found-interval counts are raw model outputs, including intervals later excluded by ignored-time evaluation. "
        "Event denominators can be lower after censoring and overlap normalization.",
        table(["Model / AV", "Found intervals", "Wholly missed human rallies", "P_pad", "R_core", "F1_padP_coreR", "Export s"], quality),
        table(["Model / AV", "Matched events", "Event precision", "Event recall", "Event F1"], events),
        "Event metrics use chronological one-to-one matching of unpadded intervals at IoU >= 0.5. "
        "Human rallies touching ignored spans are censored from event matching, with their spans removed from "
        "event predictions. These guardrails do not replace the padded-export ranking metric.",
        "Saved desktop rows reuse the same gold revision and selected encoder/TCN/decoder identities from the "
        "[visual repair evaluation](android-visual-feature-repair-evaluation.json). They are output references, "
        "not timed runs in this matrix. Browser/native/desktop decoding and audio operations can still differ. "
        "The existing native encoder also uses float bilinear preparation and unrounded FP32 tokens, while this "
        "browser experiment follows training's uint8 INTER_LINEAR and FP16 cache round trip. Cross-platform "
        "differences therefore cannot be assigned solely to AV features or video decoding. The JNI optimization "
        "preserves native outputs and does not change those embedding contracts.",
        "Current AV is the shipped canvas/preceding-frame contract. Area/nearest AV is the existing opt-in correction. "
        "No production default, model threshold or selected checkpoint changed. Full-video accuracy has exact source "
        "alignment; the 120-second pilot uses a requested stream-copy offset and only approximate label alignment. "
        "The JSON includes all predicted intervals and wholly missed saved-human ranges. The AV comparison bundles "
        "resize, selected frames and row timestamps: legacy actual-PTS rows can causally align to an older image "
        "token, whereas corrected nominal 4 Hz rows remove that lag. Changes cannot be attributed to resize alone.",
        "## Frozen-input timestamp diagnostic",
        "After timed runs finished, the saved AV104 values and fusion/decoder time grids were crossed in a 2-by-2 "
        "CPU replay for each selection. Embeddings, quality values, fitted scalers, model weights and decoder "
        "settings remain fixed. Matching complete row grids are required; no interpolation, video decode or "
        "training occurs. The two observed diagonals reproduce their saved tensors, probabilities and boundaries.",
        table(["Selection", "AV values", "Fusion/decode grid", "Kind", "Rallies", "Wholly missed", "P_pad", "R_core", "F1_padP_coreR"], counterfactual),
        alignment_note,
        "AV values combine selected-frame, resize and audio-feature-window effects: the original extraction "
        "timestamps also drive audio features. The time grid jointly changes causal embedding "
        "selection, quality age and decoded boundary timestamps. Crossed inputs are diagnostics, not an implemented "
        "or timed product pipeline, and this known recording must not select a new model or default.",
        "## Full-video timing breakdown",
        table(["Model / AV", "Setup s", "Embedding pass s", "AV/audio s", "Round/fuse/TCN/decode s", "Production evidence s", "Score specialists s"], stages),
        table(["Selection", "Source RGB readback s", "224 image preparation s", "Encoder/readback s", "Decode/other s"], nested),
        "The second table contains nested embedding counters; do not add them again to complete stage totals. "
        "Decode/other is a residual wall-time bucket. Source-resolution image readback is measured separately from "
        "the neural encoder, so an encoder-only throughput claim would omit substantial work.",
        "## WebGPU pooling repair and qualification",
        "The original encoder graph failed WebGPU numerical validation. Its terminal `bchw,brhw->brc` Einsum "
        "was reduced incorrectly by the pinned runtime. A tiny uniform-input probe returned 1 instead of 2; "
        "a signed-input probe matched squared products rather than the required weighted sum. Inspection of the "
        "[ORT 1.22.0 shader generator](https://github.com/microsoft/onnxruntime/blob/v1.22.0/js/web/lib/wasm/jsep/webgpu/ops/einsum.ts) "
        "shows operand multiplication emitted for both reduced symbols. This is a correctness issue, not a model-precision experiment.",
        "A portable export replaces only that pooling operation with Reshape/Transpose/MatMul. Every trained "
        "initializer, TCN, scaler and decoder is unchanged; original graphs remain frozen. CPU comparison with "
        "the originals is exact on all eight image fixtures for both selections. Independent evaluation reconstructs "
        "the permitted rewrite from canonical bytes and checks the derived hash and preserved-initializer manifest.",
        "The image contract follows training: source-resolution RGB, ties-to-even ROI rounding, uint8 OpenCV "
        "INTER_LINEAR 224 letterboxing, ImageNet normalization, four regional pools, then FP16 cache rounding. "
        "Inference remains FP32. Two-Hz tokens align causally to AV104 plus eight quality/age/availability values "
        "at 4 Hz. The two selections have distinct encoder weights and cannot share image embeddings.",
        "They can still reuse AV/audio inputs and prepared image pixels when source, ROI, sampling window and "
        "preprocessing version match. Reusing those stages across selections was not timed here; each selection "
        "ran in a fresh browser profile.",
        "Independent checks cover eight synthetic images per selected encoder, twenty decoder cases, two real-scaler "
        "fusion fixtures, and every finite FP16 value. For each actual-video run, five saved browser RGB snapshots "
        "are reprocessed in Python and replayed through the original CPU encoder; all actual fused rows are replayed "
        "through the original CPU TCN. Numerical errors, browser-versus-Python boundaries, and CPU-probability "
        "decoded boundaries are recorded in JSON. Snapshot equality validates operations on those pixels, not full "
        "source color-conversion parity. Nearest-frame selection additionally relies on the tested sampling helper; "
        "there is no independent full-source PTS inventory in this browser report.",
        "All primary runs hash browser sources and decoder dependencies before navigation, disable HMR, and "
        "reject any source change afterward. The structured qualification evidence records fixture/probe hashes "
        "and maximum errors. Portable GPU FP32 outputs are numerically close to CPU; subsequent FP16 rounding "
        "can differ by a half-precision step near a rounding boundary and is not claimed byte-identical to CPU.",
        "Serving-side and side-switch outputs are generated and checked for completeness, but their classification "
        "accuracy is not independently scored here. The editor lab still displays precomputed neural predictions; "
        "this harness does not add neural inference to the production upload flow.",
        "## Download and feature storage",
        table(["Selection", "Model/config/pool MB", "Served ORT bundle MB"], assets),
        "Sizes are decimal, uncompressed bytes. The runtime is shared across selections, while each selection has "
        "its own encoder/head. Existing production OpenCV/audio/score assets are separately itemized in JSON and "
        "are not all incremental neural downloads. The experiment loads both OpenCV bundles: production prefers "
        "the worker, while neural image preparation forces the main bundle. That main OpenCV bundle is another "
        "10.873 MB when it is not already cached or needed by other work. Thus the full observed production-asset "
        "total must not be treated as already loaded in every baseline. A WASM-only deployment could use a smaller bundle; that "
        "alternative bundle was not the measured configuration.",
        table(["Selection / AV", "AV rows", "AV104 MB", "Fused3952 MB", "Probabilities MB"], storage),
        "The full source has 2,123 image samples: 32.609 MB of raw FP32 tokens, another 32.609 MB of FP16-rounded "
        "tokens stored in Float32Array, or 16.305 MB if packed as FP16. These are alternative/diagnostic representations; "
        "fused inputs duplicate the embeddings. They are not additive minimum memory. Sampled JavaScript heap "
        "excludes WASM linear memory, external buffers and GPU allocations, so it is not peak-memory qualification.",
        "## Deployment implications",
        "The portable pooling graph is a candidate for subsequent mobile-browser qualification, with WASM as "
        "a separately tested fallback. The original WebGPU graph must not be shipped on the strength of session "
        "creation or apparent speed. A physical phone still needs video-to-results timing, memory and output checks; "
        "the desktop NVIDIA GPU and NAS source cannot establish phone performance.",
        "Further optimization should target video passes and source-resolution RGB readback while preserving the "
        "qualified image operations. A faster resize/color path needs the same golden-image and frozen-model "
        "checks before acceptance. Streaming temporal chunks could also avoid retaining duplicated full-video "
        "embedding/fused diagnostic arrays. Neither optimization nor a production-default change is part of this experiment.",
        "## Required padding sensitivity",
        "Every compared observation uses the same gold revision, ignored-range revision and strict join threshold. "
        "The declared 2-second case remains primary; no per-model best-padding selection is performed.",
        table(["Observation / model", "Padding each side", "P_pad", "R_core", "F1_padP_coreR", "Model export s", "Human export s", "Difference s"], padding),
        "## Reproduction and artifacts",
        "Private inputs and raw receipts resolve through private-reference-0223. The "
        "[harness](../../scripts/benchmark-distilled-large-browser.mjs), "
        "[portable export preparer](../../scripts/prepare-distilled-browser-portable-pooling.py), "
        "[independent evaluator](../../scripts/evaluate-distilled-large-browser.py) and "
        "[report builder](../../scripts/report-distilled-large-browser.py) preserve indexed provenance. "
        "See the [structured report](distilled-large-browser-experiment.json), "
        "[earlier AV-only experiment](web-visual-preprocessing-validation.md), and "
        "[optimized Android full-video validation](android-native-area-full-video.md).",
    ]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-index", default="private-reference-0223")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "docs/research")
    args = parser.parse_args()
    directory = Path(private_value(args.root_index)) / "browser-distilled-large-runs"
    report = dict(schema="distilled-large-browser-experiment-v1", status="complete",
                  artifactIndex=args.root_index, recordingIndex="recording-044",
                  targetPaddingSeconds=2, joinGapSeconds=3, trainingPerformed=False,
                  selectionChanged=False, productionDefaultChanged=False,
                  runs=[load_run(directory / name, *expected) for name, *expected in RUNS])
    report["qualification"] = read(directory / "qualification-summary.json")
    report["counterfactual"] = read(directory / "av-timeline-counterfactual.json")
    report["qualificationReceiptSha256"] = sha(directory / "qualification-summary.json")
    report["counterfactualReceiptSha256"] = sha(directory / "av-timeline-counterfactual.json")
    q = report["qualification"]
    if not q["passed"] or {row["selection"] for row in q["cpuPoolingEquivalence"]} != {"recall", "f1"} \
            or any(row["caseCount"] != 8 or not row["rawExact"] or not row["roundedExact"]
                   for row in q["cpuPoolingEquivalence"]):
        raise ValueError("Exact CPU pooling-equivalence evidence is missing")
    if {row["selection"] for row in q["webgpuGoldenFixtures"]} != {"recall", "f1"} \
            or any(row["caseCount"] != 8 or not row["pixelsExact"] for row in q["webgpuGoldenFixtures"]):
        raise ValueError("Selected GPU image qualification is missing")
    if len({r["evaluation"]["goldSha256"] for r in report["runs"]}) != 1:
        raise ValueError("Compared runs use different gold revisions")
    cf = report["counterfactual"]
    if cf["goldSha256"] != report["runs"][0]["evaluation"]["goldSha256"] or \
            {row["selection"] for row in cf["selections"]} != {"recall", "f1"}:
        raise ValueError("Counterfactual source/gold scope differs")
    for row in cf["selections"]:
        if {(c["avValues"], c["fusionAndDecoderTimeline"]) for c in row["cases"]} != \
                {("legacy", "legacy"), ("legacy", "corrected"), ("corrected", "legacy"), ("corrected", "corrected")}:
            raise ValueError("Counterfactual matrix is incomplete")
        actual = next(r["evaluation"] for r in report["runs"]
                      if r["evaluation"]["source"]["scope"] == "full" and r["evaluation"]["model"]["selection"] == row["selection"])
        if row["resultSha256"] != actual["resultSha256"]:
            raise ValueError("Counterfactual uses different observed inputs")
    if len({r["evaluation"]["source"]["sha256"] for r in report["runs"]
            if r["evaluation"]["source"]["scope"] == "full"}) != 1:
        raise ValueError("Full runs use different source recordings")
    reference = read(ROOT / "docs/research/android-visual-feature-repair-evaluation.json")
    if reference["goldSha256"] != report["runs"][0]["evaluation"]["goldSha256"]:
        raise ValueError("Desktop reference uses different gold")
    report["desktopReferences"] = []
    for row in reference["models"]:
        identity = next(r["evaluation"]["model"] for r in report["runs"]
                        if r["evaluation"]["source"]["scope"] == "full" and
                        r["evaluation"]["model"]["selection"] == row["selection"])
        if row["encoderWeightsSha256"] != identity["encoderWeightsSha256"] or \
                row["weightsSha256"] != identity["temporalWeightsSha256"] or row["decoder"] != identity["decoder"]:
            raise ValueError("Desktop reference uses different frozen model identity")
        report["desktopReferences"].append(dict(selection=row["selection"], evaluation=row["savedDesktop"]["evaluation"]))
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "distilled-large-browser-experiment.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_directory / "distilled-large-browser-experiment.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(dict(passed=True, cleanRuns=len(report["runs"]))))


if __name__ == "__main__":
    main()
