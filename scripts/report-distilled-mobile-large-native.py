"""Report frozen native device observations using indexed, allowlisted evidence.

No training, model selection, device access, or mutation of benchmark inputs.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.neural_evaluation import evaluate_predictions
from analysis.private_ledger import private_value

ARTIFACT_INDEX = "private-reference-0223"
RECORDING_INDEX = "recording-044"
GOLD_SHA256 = "97ab7adce70f57ad521add3a18f18765abec61d36b2ba31cf0c739638164d94c"
APK_SHA256 = "293fc06007a41b9cc58318c259fa90764c5ac93171e5098f284b18ad9ace5061"
FULL_SECONDS = 1061.016489
DISPLAY = {"production": "Production ensemble", "mobile": "Frozen Small (highest recall)",
           "mobile-large": "Frozen Large (highest recall)",
           "distilled-f1": "Distilled Large (highest F1)",
           "distilled-recall": "Distilled Large (highest recall)"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shifted_labels(labels, offset, duration):
    """Clip source truth to an excerpt, retaining original rally numbers."""
    def shift(items, numbered=False):
        output = []
        for index, item in enumerate(items):
            start = max(0.0, float(item["start"]) - offset)
            end = min(duration, float(item["end"]) - offset)
            if end > start:
                row = dict(start=start, end=end)
                if numbered:
                    row["sourceHumanRallyNumber"] = index + 1
                output.append(row)
        return output
    return dict(rallies=shift(labels["rallies"], True),
                ignoredIntervals=shift(labels.get("ignoredIntervals", [])))


def evaluate(row, labels, duration):
    predictions = [dict(start=float(r["start"]), end=float(r["end"])) for r in row["rallies"]]
    report = evaluate_predictions([dict(id=RECORDING_INDEX, sourceGroup="benchmark-source",
        durationSeconds=duration, rallies=labels["rallies"],
        ignoredIntervals=labels.get("ignoredIntervals", []), predictions=predictions)])
    guard = report["guardrails"]
    # Copy only public aggregate fields and timestamp ranges, never raw metadata.
    metric_keys = ("paddingSecondsBeforeAndAfter", "joinGapSeconds", "P_pad", "R_core",
                   "F1_padP_coreR", "paddedModelExportSeconds", "paddedHumanExportSeconds",
                   "exportDurationDifferenceSeconds", "coreHumanSeconds")
    metric = lambda entry: {key: entry[key] for key in metric_keys}
    lost = []
    for rally in guard["primaryExportCoverage"]["rallies"]:
        if rally["completelyLost"]:
            truth = labels["rallies"][rally["truthIndex"]]
            lost.append(dict(humanRallyNumber=truth.get("sourceHumanRallyNumber", rally["truthIndex"] + 1),
                             start=rally["start"], end=rally["end"],
                             evaluableCoreSeconds=rally["evaluableCoreSeconds"]))
    return dict(primary=metric(report["primary"]), padding=[metric(m) for m in report["padding"]],
        humanRallies=len(labels["rallies"]), whollyMissedSavedHumanRallies=len(lost),
        whollyMissedHumanRallyRanges=lost, rallies=predictions,
        event={key: guard[key] for key in ("trueRallies", "predictedRallies", "matchedRallies",
                                         "eventPrecision", "eventRecall", "eventF1")})


def timings(row):
    stages, profile = row["stagesMs"], row["profileMs"]
    video = row.get("neural", {}).get("video", {})
    return dict(allReadySeconds=row["totalMs"] / 1000,
        ralliesReadySeconds=(row["totalMs"] - stages["score_specialists"]) / 1000,
        videoAvSeconds=stages["video_decode_and_features"] / 1000,
        audioSeconds=stages["audio_decode_and_features"] / 1000,
        contextSeconds=stages["contextualize"] / 1000,
        embeddingVideoPassSeconds=profile.get("neural/embedding_video_pass", 0) / 1000,
        embeddingDecodeOtherSeconds=video.get("decodeAndOtherMs", 0) / 1000,
        embeddingPreparationSeconds=video.get("prepareMs", 0) / 1000,
        embeddingEncoderSeconds=video.get("encoderAndReadbackMs", 0) / 1000,
        temporalSeconds=profile.get("neural/temporal_inference", 0) / 1000,
        fusionAndSaveSeconds=profile.get("neural/fusion_normalization_and_save", 0) / 1000,
        scoreSeconds=stages["score_specialists"] / 1000,
        scoreDecodeSeconds=profile.get("score/shared_decode_wall", 0) / 1000,
        servingEvaluationSeconds=profile.get("score/serving_side_evaluation", 0) / 1000,
        sideSwitchEvaluationSeconds=profile.get("score/side_switch_evaluation", 0) / 1000)


def storage(folder, row):
    n = row["sampleRows"]
    result = dict(av104Float32BytesCalculated=n * 104 * 4,
                  contextual520Float32BytesCalculated=n * 520 * 4,
                  savedDiagnosticFiles=[], specialistFeatures={})
    neural = row.get("neural")
    if neural:
        count, dimension = neural["featureRows"], neural["featureDimension"]
        expected = dict(tokens=neural["video"]["sampleCount"] * (dimension - 112) * 4,
                        features=count * dimension * 4, probabilities=count * 4 * 4)
        for kind, size in expected.items():
            path = folder / (row["id"] + "-" + kind + ".f32")
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError("Saved neural tensor absent or incorrectly sized: " + kind)
            result["savedDiagnosticFiles"].append(dict(kind=kind, bytes=path.stat().st_size,
                expectedFloat32BytesCalculated=size, sha256=digest(path)))
    for name, feature_key in (("servingSide", "rawFeatures"), ("sideSwitch", "features")):
        score = row[name]
        encoded = score[feature_key]
        result["specialistFeatures"][name] = dict(rows=score["rows"], columns=score["columns"],
            decodedPayloadBytes=len(base64.b64decode(encoded, validate=True)),
            persistedBase64Bytes=len(encoded.encode("ascii")),
            publishedPredictions=len(score["candidates"]))
    return result


def completed_rows(folder, families, duration, expected_runs, require_parity=True):
    result, plan = read(folder / "result.json"), read(folder / "pipeline-plan.json")
    if result.get("status") != "complete" or result.get("runId") != plan.get("runId"):
        raise ValueError("Completed matching result and plan required")
    specs = {item["id"]: item for item in plan["cases"]}
    if len(specs) != len(plan["cases"]) or len(result["results"]) != len(specs):
        raise ValueError("Duplicate or incomplete result cases")
    parity = {}
    if require_parity and any(f != "production" for f in families):
        check = read(folder / "temporal-decoder-parity.json")
        if check.get("passed") is not True:
            raise ValueError("Native temporal/decoder parity not validated")
        parity = {item["id"]: item for item in check["checks"]}
    grouped = {family: [] for family in families}
    warmups = {family: 0 for family in families}
    for row in result["results"]:
        spec = specs[row["id"]]
        if row["status"] != "complete" or spec["seconds"] != duration or not spec.get("fullFrame"):
            raise ValueError("Every case must finish on the declared full-frame duration")
        if not row.get("servingSideReady") or not row.get("sideSwitchReady"):
            raise ValueError("Both score specialists must be ready")
        if spec["family"] not in grouped:
            raise ValueError("Unexpected family in run")
        if row.get("neural") and require_parity:
            valid = parity.get(row["id"], {})
            if valid.get("decoderParity") is not True or valid.get("rows") != row["neural"]["featureRows"]:
                raise ValueError("Missing matching native tensor replay")
        if spec.get("warmup"):
            warmups[spec["family"]] += 1
        else:
            grouped[spec["family"]].append(row)
    if any(len(rows) != expected_runs for rows in grouped.values()):
        raise ValueError("Incorrect measured run count")
    if any(count != (1 if expected_runs == 3 else 0) for count in warmups.values()):
        raise ValueError("Incorrect warmup count")
    return grouped, plan


def graph_identity(folder, choice):
    contract = read(folder / "input-contract.json")
    config = read(folder / "mobile-large-pipeline.json")
    qualification = read(folder / "qualification.json")
    if (contract.get("selectionMode") != choice or config.get("selectionMode") != choice
            or qualification.get("selectionMode") != choice or qualification.get("passed") is not True
            or config.get("recallTargetPercent") != 99
            or config.get("dinoRequiredForInference") is not False
            or config.get("trainingProjectorIncluded") is not False
            or contract.get("roi") != dict(x=0, y=0, width=1, height=1)
            or qualification.get("studentWeightsSha256") != config.get("encoderWeightsSha256")
            or qualification.get("temporalWeightsSha256") != config.get("weightsSha256")):
        raise ValueError("Qualified selected distilled graph identity differs")
    files = []
    for name in ("mobile-large-encoder-fp32.onnx", "mobile-large-tcn-dynamic-fp32.onnx",
                 "mobile-large-pipeline.json", "mobile-large-encoder-pool_weights.f32"):
        path = folder / name
        actual = digest(path)
        if contract["hashes"].get(name) != actual:
            raise ValueError("Frozen graph file changed")
        files.append(dict(component=name, sha256=actual, bytes=path.stat().st_size))
    return dict(modelId="neural-distilled-mobile-large-tcn-fp32" + ("-high-recall" if choice == "recall" else ""),
        selection=choice, recallCalibrationTargetPercent=99, draw=config["draw"], epoch=config["epoch"],
        encoderWeightsSha256=config["encoderWeightsSha256"], temporalWeightsSha256=config["weightsSha256"],
        decoder=config["decoder"], graphFiles=files,
        inputContractSha256=digest(folder / "input-contract.json"),
        qualificationSha256=digest(folder / "qualification.json"))


def run_identity(folder, plan, identity, graphs):
    provenance = read(folder / "run-provenance.json")
    if (provenance.get("runId") != plan["runId"] or provenance.get("apkSha256") != APK_SHA256
            or provenance.get("inputContractSha256") != digest(folder / "input-contract.json")
            or provenance.get("planSha256") != digest(folder / "pipeline-plan.json")
            or provenance.get("resultSha256") != digest(folder / "result.json")
            or read(folder / "input-contract.json") != read(graphs / "input-contract.json")
            or provenance.get("remoteGraphDigestVerifiedByDriverBeforeLaunch") is not True):
        raise ValueError("Run is not bound to the verified APK and frozen graph contract")
    for item in identity["graphFiles"]:
        if provenance.get("graphHashes", {}).get(item["component"]) != item["sha256"]:
            raise ValueError("Run provenance graph hashes differ")
    for row in read(folder / "result.json")["results"]:
        if row.get("neural") and row["neural"].get("decoder") != identity["decoder"]:
            raise ValueError("Native run used a different frozen decoder")
    return dict(runProvenanceSha256=digest(folder / "run-provenance.json"),
                temporalDecoderParitySha256=digest(folder / "temporal-decoder-parity.json"))


def frozen_control_identity(graphs):
    """Bind only the Large files used by controls, not unused Small graph entries."""
    contract = read(graphs / "input-contract.json")
    config = read(graphs / "mobile-large-pipeline.json")
    if contract.get("roi") != dict(x=0, y=0, width=1, height=1):
        raise ValueError("Control graph geometry is cropped")
    files = []
    for name in ("mobile-large-encoder-fp32.onnx", "mobile-large-tcn-dynamic-fp32.onnx",
                 "mobile-large-pipeline.json", "mobile-large-encoder-pool_weights.f32"):
        path = graphs / name
        actual = digest(path)
        if contract["hashes"].get(name) != actual:
            raise ValueError("Frozen control graph file changed")
        files.append(dict(component=name, sha256=actual, bytes=path.stat().st_size))
    return dict(decoder=config["decoder"], graphFiles=files,
                inputContractSha256=digest(graphs / "input-contract.json"))


def exposure_for_task(record, task, selection_recordings):
    manifest = Path(task["manifest"]["path"])
    if digest(manifest) != task["manifest"]["sha256"]:
        raise ValueError("Exposure manifest changed")
    groups = {row["id"]: row["sourceGroup"] for row in read(manifest)["records"]}
    return dict(directTraining=record["id"] in task["trainIds"],
        sourceGroupTraining=record["sourceGroup"] in {groups[k] for k in task["trainIds"]},
        directCalibration=record["id"] in task["calibrationIds"],
        sourceGroupCalibration=record["sourceGroup"] in {groups[k] for k in task["calibrationIds"]},
        directModelSelection=(None if selection_recordings is None else
            any(r["id"] == record["id"] for r in selection_recordings)),
        sourceGroupModelSelection=(None if selection_recordings is None else
            any(r["sourceGroup"] == record["sourceGroup"] for r in selection_recordings)),
        reservedEvaluationGroup=record["sourceGroup"] in task["commonEvaluationGroups"])


def benchmark_exposure(source_sha256):
    study = Path(private_value("private-reference-0222"))
    records = read(study / "catalog-manifest.json")["records"]
    matches = [r for r in records if r.get("contentSha256") == source_sha256]
    if len(matches) != 1:
        raise ValueError("Benchmark bytes do not identify one registered recording")
    record = matches[0]
    platform = "WINDOWS" if os.name == "nt" else "POSIX"
    ledger_path = os.environ.get("VOLLEYCUT_PRIVATE_LEDGER_" + platform) or os.environ.get("VOLLEYCUT_PRIVATE_LEDGER")
    aliases = read(ledger_path)["originalToAlias"]
    source_group_index = aliases.get(record["sourceGroup"])
    if not source_group_index or not source_group_index.startswith("source-group-"):
        raise ValueError("Registered source-group index required")
    output = dict(recordingIndex=RECORDING_INDEX, sourceGroupIndex=source_group_index,
        recordingMatchedBySourceSha256=True, models=[],
        distinction="Reserved evaluation groups are not necessarily actual exact-label model-selection contributors.")
    panels = read(study / "all-video-evaluation.json")
    for index in ("private-reference-0222", "private-reference-0211"):
        root = Path(private_value(index)); plan = read(root / "plan.json"); selection = read(root / "evaluation.json")
        for chosen in selection["selected"]:
            if index == "private-reference-0211" and chosen["mode"] != "recall":
                continue
            task = next(t for t in plan["tasks"] if t["variant"] == chosen["variant"] and
                        t.get("splitSeed", t["seed"]) == chosen["draw"])
            item = exposure_for_task(record, task, chosen["evaluation"]["recordings"])
            item.update(model="distilled-" + chosen["mode"] if index.endswith("0222") else "mobile-large",
                        evidenceIndex=index, draw=chosen["draw"])
            if index.endswith("0222"):
                model = next(m for m in panels["models"] if m["mode"] == chosen["mode"])
                scoped = next(r for r in model["recordings"] if r["id"] == record["id"])
                item["recordedFitRole"] = scoped["fitRole"]
                item["labelPolicy"] = scoped["labelPolicy"]
            output["models"].append(item)
    small = Path(private_value("private-reference-0061")) / "randomized-variants-v1/fits/expanded-large/mobile-tcn/split-3407"
    ref = read(small / "fit-result.json")["task"]
    if digest(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("Frozen Small task identity changed")
    small_evaluation = read(small / "evaluation-fp32.json")
    floor = next(r for r in small_evaluation["floors"] if r["floorPercent"] == 99)
    operating_point = small_evaluation["operatingPoints"][floor["operatingPointKey"]]
    panel = next(p for p in operating_point["panels"] if p["panelId"] == "common-unseen" and
                 p["labelPolicy"] == "exact-rallies" and p["productionFilter"] == "all")
    small_graph = read(Path(private_value("private-reference-0217")) / "full-frame-graphs/mobile-pipeline.json")
    if (panel["status"] != "available" or operating_point["epoch"] != small_graph["epoch"] or
            operating_point["decoder"] != small_graph["decoder"] or
            digest(small / "temporal" / f"weights-{small_graph['epoch']}.npz") != small_graph["weightsSha256"]):
        raise ValueError("Frozen Small exposure lookup refers to a different selected model")
    selected_rows = [r for r in operating_point["rowsByPolicy"]["exact-rallies"] if r["id"] in panel["recordingIds"]]
    if len(selected_rows) != len(panel["recordingIds"]):
        raise ValueError("Frozen Small selection panel is incomplete")
    item = exposure_for_task(record, read(Path(ref["path"])), selected_rows)
    item.update(model="mobile", evidenceIndex="private-reference-0061", draw=3407)
    output["models"].append(item)
    output["production"] = {scope: {key: record["productionExposure"][scope][key] for key in
        ("status", "trainingOrRelated", "primaryTrainingClean", "strictNoFitOrCalibration")}
        for scope in ("rallyPipeline", "wholeProduct")}
    return output


def saved_desktop_references(source_sha256, identities, labels):
    from analysis.neural_development import decode

    study = Path(private_value("private-reference-0222"))
    catalog = read(study / "catalog-manifest.json")["records"]
    matches = [r for r in catalog if r.get("contentSha256") == source_sha256]
    if len(matches) != 1:
        raise ValueError("Saved desktop source identity is ambiguous")
    record = matches[0]
    selections = read(study / "evaluation.json")["selected"]
    output = []
    for identity in identities:
        choice = next(c for c in selections if c["mode"] == identity["selection"])
        epoch = str(identity["epoch"])
        path = study / "fits" / choice["variant"] / f"split-{choice['draw']}" / "inference" / f"{record['id']}.json"
        receipt = read(path)
        # The desktop manifest retains rational-container precision; the native
        # plan rounded that same source duration to microseconds.
        if (receipt.get("contentSha256") != source_sha256 or
                not math.isclose(receipt.get("durationSeconds", float("nan")), FULL_SECONDS, rel_tol=0, abs_tol=1e-6)
                or receipt.get("labelsUsed") is not False or receipt.get("ignoredIntervalsUsed") is not False
                or receipt["decoders"].get(epoch) != identity["decoder"]
                or receipt["studentWeights"]["sha256"] != identity["encoderWeightsSha256"]
                or receipt["weights"][epoch]["sha256"] != identity["temporalWeightsSha256"]):
            raise ValueError("Saved desktop receipt is not the benchmarked frozen selection")
        for ref in (receipt["studentWeights"], receipt["weights"][epoch], receipt["output"]):
            if digest(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("Saved desktop input or output artifact changed")
        ranges = receipt["decodedRallies"][epoch]
        with np.load(receipt["output"]["path"]) as archive:
            times = archive["times"]
            scores = archive["epoch_" + epoch]
        if not np.isfinite(times).all() or not np.isfinite(scores).all():
            raise ValueError("Saved desktop outputs contain nonfinite values")
        example = SimpleNamespace(times=times, valid=np.ones(len(times), dtype=bool), duration=receipt["durationSeconds"])
        decoded = [dict(start=r.start, end=r.end) for r in decode(example, scores, identity["decoder"])]
        np.testing.assert_allclose([(r["start"], r["end"]) for r in ranges],
                                   [(r["start"], r["end"]) for r in decoded], atol=1e-8, rtol=0)
        output.append(dict(model="distilled-" + identity["selection"],
            label=DISPLAY["distilled-" + identity["selection"]] + " — saved desktop",
            artifactIndex="private-reference-0222", recordingIndex=RECORDING_INDEX,
            receiptSha256=digest(path), outputSha256=receipt["output"]["sha256"],
            sourceDurationSeconds=receipt["durationSeconds"], benchmarkDurationSeconds=FULL_SECONDS,
            encoderWeightsSha256=identity["encoderWeightsSha256"], temporalWeightsSha256=identity["temporalWeightsSha256"],
            decoderVerifiedFromSavedProbabilities=True, inferenceRerun=False, rallyCount=len(ranges),
            evaluation=evaluate(dict(rallies=ranges), labels, FULL_SECONDS)))
    return output


def input_diagnostic(root, report):
    path = root / "native-input-diagnostic.json"
    data = read(path)
    identity = next(r for r in report["modelIdentities"] if r["selection"] == "f1")
    native = next(r for r in report["full"] if r["model"] == "distilled-f1")["cases"][0]
    desktop = next(r for r in report["savedDesktopReferences"] if r["model"] == "distilled-f1")
    files = {r["kind"]: r["sha256"] for r in native["storage"]["savedDiagnosticFiles"]}
    graph = next(r["sha256"] for r in identity["graphFiles"] if r["component"] == "mobile-large-tcn-dynamic-fp32.onnx")
    expected = dict(nativeFeatures=files["features"], nativeProbabilities=files["probabilities"],
                    desktopSavedProbabilities=desktop["outputSha256"], onnxGraph=graph)
    if (data.get("schema") != "frozen-native-input-substitution-v1" or data.get("complete") is not True
            or data.get("recordingIndex") != RECORDING_INDEX or data.get("labelSha256") != GOLD_SHA256
            or data.get("selectionMode") != "f1" or data.get("weightsSha256") != identity["temporalWeightsSha256"]
            or data.get("encoderWeightsSha256") != identity["encoderWeightsSha256"]
            or data.get("decoder") != identity["decoder"] or data.get("inputHashes") != expected
            or data.get("targetPaddingSeconds") != 2 or data.get("joinGapSeconds") != 3):
        raise ValueError("Input diagnostic is not bound to the benchmarked frozen tensors")
    for name, measured in (("native", native), ("desktop", desktop)):
        case = next(r for r in data["cases"] if r["name"] == name)
        if (case["rallyCount"] != measured["rallyCount"] or
                case["whollyMissedSavedHumanRallies"] != measured["evaluation"]["whollyMissedSavedHumanRallies"]):
            raise ValueError("Input diagnostic observed counts differ")
        for key in ("P_pad", "R_core", "F1_padP_coreR"):
            if abs(case["primary"][key] - measured["evaluation"]["primary"][key]) > 1e-12:
                raise ValueError("Input diagnostic observed metrics differ")
    names = ("native", "desktop", "av104", "av_audio", "embeddings", "quality", "av104_quality")
    fields = ("name", "kind", "rallyCount", "whollyMissedSavedHumanRallies", "whollyMissedHumanRallyNumbers", "primary", "padding")
    return dict(artifactIndex=ARTIFACT_INDEX, artifactSha256=digest(path), schema=data["schema"],
        inputHashes=expected, selection="f1", cases=[{key: r[key] for key in fields} for r in data["cases"] if r["name"] in names],
        scope="Bounded CPU substitutions of saved native/desktop input blocks with frozen weights, scaler and decoder. "
              "No new phone run, feature extraction, training, threshold tuning, or deployed correction.")


def summarize(folder, model, rows, labels, duration, source_index, accuracy_alignment):
    cases = []
    for index, row in enumerate(rows, 1):
        cases.append(dict(run=index, timing=timings(row), rallyCount=row["rallyCount"],
            thermalStart=row.get("thermalStart"), thermalEnd=row.get("thermalEnd"),
            servingCandidates=row["servingSide"]["rows"], sideSwitchCandidates=row["sideSwitch"]["rows"],
            storage=storage(folder, row), evaluation=evaluate(row, labels, duration)))
    times = {key: median(case["timing"][key] for case in cases) for key in cases[0]["timing"]}
    return dict(model=model, label=DISPLAY[model], artifactIndex=source_index,
        measuredRuns=len(cases), videoSeconds=duration, accuracyAlignment=accuracy_alignment,
        medianTiming=times, allReadyRangeSeconds=[min(c["timing"]["allReadySeconds"] for c in cases),
                                                max(c["timing"]["allReadySeconds"] for c in cases)],
        cases=cases)


def markdown(report):
    pilot_choices = {r["model"]: r["medianTiming"] for r in report["pilot"]}
    full_choices = {r["model"]: r["medianTiming"] for r in report["full"]}
    lines = ["# Distilled MobileNetV3-Large native benchmark", "",
        f"Both distilled selections completed the native pipeline. Highest-F1 / highest-recall all-ready times "
        f"were **{pilot_choices['distilled-f1']['allReadySeconds']:.3f}s / "
        f"{pilot_choices['distilled-recall']['allReadySeconds']:.3f}s** for the two-minute excerpt (three-run medians), "
        f"and **{full_choices['distilled-f1']['allReadySeconds']:.3f}s / "
        f"{full_choices['distilled-recall']['allReadySeconds']:.3f}s** for the complete recording (one run each). "
        "These totals include all feature generation and both score specialists, before export rendering.", "",
        "Physical Pixel 10 Pro, FP32 ONNX Runtime CPU with four threads, hardware MediaCodec decoding. "
        "Both frozen distilled selections are measured independently from video through rally, serving-side and side-switch results. "
        "DINO and the training projection are absent from inference.", "",
        "Target export padding is 2s each side; positive gaps strictly under 3s join. "
        "Ignored intervals are removed after joining, without rejoining across them. Recall is retained saved human core time.", ""]
    for scope, title in (("pilot", "Two-minute excerpt"), ("full", "Full 17m41s recording")):
        lines += ["## " + title, "",
            "| Model | Measured runs | AV video | Audio | Embedding pass | Rallies ready | Score specialists | All ready | Rallies |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for row in report[scope]:
            t = row["medianTiming"]
            numbers = [t[k] for k in ("videoAvSeconds", "audioSeconds", "embeddingVideoPassSeconds",
                                     "ralliesReadySeconds", "scoreSeconds", "allReadySeconds")]
            lines.append("| " + " | ".join([row["label"], str(row["measuredRuns"])] +
                [f"{n:.3f}s" for n in numbers] + [", ".join(str(c["rallyCount"]) for c in row["cases"])]) + " |")
        lines += ["", ("Pilot values are medians of three measured passes following one full warmup per model. "
            "All use the same excerpt and corrected APK; stage medians need not sum to median total."
            if scope == "pilot" else "Full-video values are single observations. Production and frozen Small/Large are prior corrected "
            "full-frame controls on the identical APK and source, not simultaneous controlled thermal comparisons. "
            "The new full runs execute highest F1 then highest recall; OS media cache and run-order effects are not isolated."), ""]
    lines += ["## Full-video saved-human comparison", "",
        "| Model | Found rallies | Wholly missed / human rallies | P_pad | R_core | F1_padP_coreR | Export (s) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["full"]:
        case = row["cases"][0]; ev = case["evaluation"]; m = ev["primary"]
        lines.append(f"| {row['label']} | {case['rallyCount']} | {ev['whollyMissedSavedHumanRallies']} / {ev['humanRallies']} | "
            f"{m['P_pad']:.2%} | {m['R_core']:.2%} | {m['F1_padP_coreR']:.2%} | {m['paddedModelExportSeconds']:.3f} |")
    for row in report["savedDesktopReferences"]:
        ev = row["evaluation"]; m = ev["primary"]
        lines.append(f"| {row['label']} | {row['rallyCount']} | {ev['whollyMissedSavedHumanRallies']} / {ev['humanRallies']} | "
            f"{m['P_pad']:.2%} | {m['R_core']:.2%} | {m['F1_padP_coreR']:.2%} | {m['paddedModelExportSeconds']:.3f} |")
    lines += ["", "Wholly missed means no nonignored saved human core retained after export padding and joining. "
        "The 37-rally gold snapshot contains manually reviewed export-derived boundaries with later edits, "
        "not independently precise serve-contact/dead-ball annotation. It is a different revision from the model-selection panel.", "",
        "Saved desktop rows use these same gold labels, source bytes, selected encoder/TCN weights and decoder thresholds. "
        "They are existing outputs with hash-verified provenance and replayed boundary decoding, with no training or inference rerun "
        "and no desktop speed measurement. A native/desktop accuracy gap therefore cannot be attributed solely to a different "
        "checkpoint or threshold; extraction/runtime-input differences remain material.", ""]
    lines += ["## Bounded native-input diagnosis", "",
        "The following highest-F1 checks replace selected blocks of already-saved native features with their desktop counterparts. "
        "Weights, normalization and decoder remain frozen. These are synthetic CPU replays, not new phone runs or usable corrected predictions.", "",
        "| Saved inputs / replacement | Rallies | Wholly missed | P_pad | R_core | F1_padP_coreR |",
        "|---|---:|---:|---:|---:|---:|"]
    diagnostic_labels = {"native": "Native observed", "desktop": "Desktop observed", "av104": "Replace AV104 only",
        "av_audio": "Replace AV audio only", "embeddings": "Replace embeddings only", "quality": "Replace quality scalars only",
        "av104_quality": "Replace AV104 + quality; keep native embeddings"}
    for row in report["nativeInputDiagnostic"]["cases"]:
        m = row["primary"]
        lines.append(f"| {diagnostic_labels[row['name']]} | {row['rallyCount']} | {row['whollyMissedSavedHumanRallies']} | "
            f"{m['P_pad']:.2%} | {m['R_core']:.2%} | {m['F1_padP_coreR']:.2%} |")
    lines += ["", "Replacing handcrafted AV inputs, particularly the audio block, recovers most of the gap; "
        "replacing image embeddings alone does not. AV104 plus quality scalars nearly reproduces desktop coverage while "
        "retaining native student embeddings. This localizes a material input-contract problem but does not identify its "
        "underlying audio decode/feature defect or qualify a fix. No feature change is installed by this benchmark.", "",
        "## Image and score-stage decomposition", "",
        "| Scope / model | Embedding decode / other | Image preparation | Encoder / readback | TCN | Specialist decode | Serving features / evaluation | Side-switch features / evaluation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for scope in ("pilot", "full"):
        for row in report[scope]:
            t = row["medianTiming"]
            values = [t[k] for k in ("embeddingDecodeOtherSeconds", "embeddingPreparationSeconds", "embeddingEncoderSeconds",
                "temporalSeconds", "scoreDecodeSeconds", "servingEvaluationSeconds", "sideSwitchEvaluationSeconds")]
            lines.append("| " + scope + " / " + row["label"] + " | " + " | ".join(f"{v:.3f}s" for v in values) + " |")
    lines += ["", "Embedding pass includes a separate decode pass in full. Decode/other is a residual wall-time bucket, "
        "not isolated decoder time. Specialist workloads use each model's actual rally boundaries; their cost is measured separately. "
        "Do not add nested or overlapping profile timings to wall-clock totals.", "", "## Full-video feature storage", "",
        "| Model | AV104 calculated | AV520 calculated | Tokens measured | Fused features measured | Probabilities measured | Specialist payloads measured |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["full"]:
        s = row["cases"][0]["storage"]; files = {f["kind"]: f["bytes"] for f in s["savedDiagnosticFiles"]}
        values = [s["av104Float32BytesCalculated"], s["contextual520Float32BytesCalculated"],
                  files.get("tokens"), files.get("features"), files.get("probabilities"),
                  sum(f["decodedPayloadBytes"] for f in s["specialistFeatures"].values())]
        lines.append("| " + row["label"] + " | " + " | ".join("n/a" if v is None else f"{v/1e6:.3f} MB" for v in values) + " |")
    lines += ["", "Decimal MB. AV arrays were not saved: sizes are calculated from actual row counts. "
        "Tokens are duplicated in fused features; these diagnostics are neither additive minimum memory nor measured peak RAM. "
        "Timed diagnostic writes add conservative overhead.", "", "## Required full-video padding sensitivities", "",
        "| Model | Padding each side | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["full"]:
        for m in row["cases"][0]["evaluation"]["padding"]:
            lines.append(f"| {row['label']} | {m['paddingSecondsBeforeAndAfter']:.0f}s | {m['P_pad']:.2%} | {m['R_core']:.2%} | "
                f"{m['F1_padP_coreR']:.2%} | {m['paddedModelExportSeconds']:.3f} | {m['paddedHumanExportSeconds']:.3f} | {m['exportDurationDifferenceSeconds']:+.3f} |")
    for row in report["savedDesktopReferences"]:
        for m in row["evaluation"]["padding"]:
            lines.append(f"| {row['label']} | {m['paddingSecondsBeforeAndAfter']:.0f}s | {m['P_pad']:.2%} | {m['R_core']:.2%} | "
                f"{m['F1_padP_coreR']:.2%} | {m['paddedModelExportSeconds']:.3f} | {m['paddedHumanExportSeconds']:.3f} | {m['exportDurationDifferenceSeconds']:+.3f} |")
    lines += ["", "## Observed variability and thermal state", "",
        "| Scope / model | All-ready range | Thermal start/end by measured run |",
        "|---|---:|---|"]
    for scope in ("pilot", "full"):
        for row in report[scope]:
            low, high = row["allReadyRangeSeconds"]
            thermal = ", ".join(f"{case['thermalStart']}/{case['thermalEnd']}" for case in row["cases"])
            lines.append(f"| {scope} / {row['label']} | {low:.3f}–{high:.3f}s | {thermal} |")
    lines += ["", "Android thermal status 0 means none; 1 means light. Values are observations, "
        "not a controlled cooling or battery experiment. A locked/dozing initial pilot attempt was stopped before any "
        "completed case, preserved separately and excluded. The reported sequence began again with the phone unlocked."]
    lines += ["", "## Limits and evidence", "",
        "- Every measurement bypasses AV caches. It includes all required features and both score specialists; "
        "app launch, transfer, installation, collection and export rendering are excluded.",
        "- Neural rally decisions use AV104 and their own encoder/TCN. Serving-side still requires the two production serve heads; "
        "side-switch requires production rally/dead-state evidence. Those auxiliary computations are included.",
        "- Real-device fused-tensor temporal and decoder replay must pass before publication. Native image/PTS/AV extraction "
        "is still not qualified against desktop; prior residual AV drift remains a limitation, documented in the "
        "[previous native comparison](mobile-tcn-follow-up.md). This is native CPU timing, not GPU or browser timing.",
        "- Pilot labels in the JSON are approximate diagnostics: source labels are shifted by the requested 180s stream-copy offset "
        "and clipped to 120s, without frame-accurate remux alignment. They are not the headline accuracy comparison. Full-video labels use exact source time.",
        "- The JSON preserves per-run timings, thermal start/end status, rally boundaries, all four padding cases, "
        "wholly missed ranges, event matching, and measured file sizes. No source filenames or runtime locations are included.",
        "- Accuracy on this one previously studied recording is a deployment diagnostic, not a new model selection or an independent generalization estimate.",
        f"- Recording `{RECORDING_INDEX}`; new measurements `{ARTIFACT_INDEX}`; previous corrected neural/production controls "
        "`private-reference-0218` / `private-reference-0219`; same-gold comparison `private-reference-0221`.",
        f"- Gold SHA-256: `{GOLD_SHA256}`. Corrected benchmark APK SHA-256: `{APK_SHA256}`.", ""]
    lines += ["## Frozen model identities", "",
        "| UI selection | Dataset draw | TCN epoch | Encoder bytes | TCN bytes |",
        "|---|---:|---:|---:|---:|"]
    for identity in report["modelIdentities"]:
        sizes = {f["component"]: f["bytes"] for f in identity["graphFiles"]}
        lines.append(f"| `{identity['modelId']}` | {identity['draw']} | {identity['epoch']} | "
            f"{sizes['mobile-large-encoder-fp32.onnx']:,} | {sizes['mobile-large-tcn-dynamic-fp32.onnx']:,} |")
    lines += ["", "The exact selected encoder/TCN weights, export qualification and per-run graph identities are frozen. "
        "No retraining, decoder adjustment or selection was performed for this benchmark. Raw graph sizes exclude "
        "the shared inference runtime and are not Play Store download sizes.", ""]
    exposure = report["benchmarkExposure"]
    lines += ["## Benchmark recording exposure", "",
        f"Source group `{exposure['sourceGroupIndex']}` is identified through the external ledger. "
        "Membership is checked against the selected fit manifests and actual scored common panels.", "",
        "| Model | Recording / source group in training | Recording / source group in calibration | Recording / source group in model selection |",
        "|---|---|---|---|"]
    flag = lambda value: "not re-audited" if value is None else ("yes" if value else "no")
    for row in exposure["models"]:
        values = [flag(row[a]) + " / " + flag(row[b]) for a, b in
            (("directTraining", "sourceGroupTraining"), ("directCalibration", "sourceGroupCalibration"),
             ("directModelSelection", "sourceGroupModelSelection"))]
        lines.append("| " + DISPLAY[row["model"]] + " | " + " | ".join(values) + " |")
    lines += ["", "The production lineage audit records no fit or calibration exposure for either the rally pipeline or "
        "the whole product. The benchmark recording and its source group were not used to fit, calibrate, or select "
        "either distilled choice or the frozen Small/Large choices. Reserved evaluation-group membership alone does not imply "
        "use in the actual exact-label selection panel. Historical diagnostic viewing and repeated pipeline fixes on this "
        "video remain separate exposure; these measurements are same-video deployment diagnostics, not a newly protected test.", ""]
    lines += ["Generated by [`report-distilled-mobile-large-native.py`](../../scripts/report-distilled-mobile-large-native.py). "
        "The [JSON companion](distilled-mobile-large-native-benchmark.json) contains the complete indexed observations.", ""]
    return "\n".join(lines)


def build(root, labels_path, neural_control, production_control, comparison):
    if digest(labels_path) != GOLD_SHA256:
        raise ValueError("Saved gold revision changed")
    labels = read(labels_path)
    contract = read(root / "benchmark-contract.json")
    required = dict(artifactIndex=ARTIFACT_INDEX, recordingIndex=RECORDING_INDEX,
                    apkSha256=APK_SHA256, fullSourceSeconds=FULL_SECONDS,
                    pilotSourceOffsetSeconds=180, pilotAnalysisSeconds=120)
    if any(contract.get(key) != value for key, value in required.items()):
        raise ValueError("Benchmark identity or source alignment contract differs")
    old = read(comparison / "corrected-native-comparison.json")
    if old["labelSha256"] != GOLD_SHA256 or old["recordingIndex"] != RECORDING_INDEX:
        raise ValueError("Control uses a different recording or gold revision")
    report = dict(schemaVersion=1, status="complete", artifactIndex=ARTIFACT_INDEX,
        recordingIndex=RECORDING_INDEX, labelSha256=GOLD_SHA256, apkSha256=APK_SHA256,
        precision="FP32", provider="ONNX Runtime CPU", threads=4,
        targetPaddingSeconds=2, joinGapSeconds=3, modelIdentities=[], pilot=[], full=[])
    report["benchmarkExposure"] = benchmark_exposure(contract["sourceIdentities"]["full"]["sha256"])
    plans = {"pilot": [], "full": []}
    clip_labels = shifted_labels(labels, 180, 120)
    controls, plan = completed_rows(root / "pilot-controls", ["production", "mobile-large"], 120, 3)
    plans["pilot"].append(plan)
    control_graphs = Path(private_value("private-reference-0217")) / "full-frame-graphs"
    control_evidence = run_identity(root / "pilot-controls", plan,
                                    frozen_control_identity(control_graphs), control_graphs)
    for family, rows in controls.items():
        value = summarize(root / "pilot-controls", family, rows, clip_labels, 120,
                           ARTIFACT_INDEX, "approximate shifted excerpt")
        value["evidence"] = control_evidence
        report["pilot"].append(value)
    for choice in ("f1", "recall"):
        identity = graph_identity(root / ("graphs-" + choice), choice)
        report["modelIdentities"].append(identity)
        for scope, duration, gold in (("pilot", 120, clip_labels), ("full", FULL_SECONDS, labels)):
            folder = root / (scope + "-" + choice)
            rows, plan = completed_rows(folder, ["mobile-large"], duration, 3 if scope == "pilot" else 1)
            plans[scope].append(plan)
            value = summarize(folder, "distilled-" + choice, rows["mobile-large"], gold,
                duration, ARTIFACT_INDEX, "exact source time" if scope == "full" else "approximate shifted excerpt")
            value["evidence"] = run_identity(folder, plan, identity, root / ("graphs-" + choice))
            report[scope].append(value)
    for folder, families, index in ((production_control, ["production"], "private-reference-0219"),
                                   (neural_control, ["mobile", "mobile-large"], "private-reference-0218")):
        # The old comparator already replayed these exact saved tensors; its identity is verified below.
        if read(folder / "replay-contract.json")["apkSha256"] != APK_SHA256:
            raise ValueError("Historical control used a different benchmark APK")
        rows, plan = completed_rows(folder, families, FULL_SECONDS, 1, require_parity=False)
        plans["full"].append(plan)
        for family in families:
            value = summarize(folder, family, rows[family], labels, FULL_SECONDS, index, "exact source time")
            old_row = next(m for m in old["models"] if m["family"] == family)
            for key in ("P_pad", "R_core", "F1_padP_coreR"):
                if abs(value["cases"][0]["evaluation"]["primary"][key] - old_row["observed"]["corrected"]["evaluation"]["primary"][key]) > 1e-12:
                    raise ValueError("Saved control changed since original comparison")
            report["full"].insert(0 if family == "production" else len(report["full"]) - 2, value)
    for scope in plans:
        if len({p["sourceUri"] for p in plans[scope]}) != 1:
            raise ValueError("Cases do not use the identical MediaStore source")
        if plans[scope][0]["sourceUri"] != contract["sourceIdentities"][scope]["uri"]:
            raise ValueError("Cases do not use the hash-verified source")
    report["savedDesktopReferences"] = saved_desktop_references(
        contract["sourceIdentities"]["full"]["sha256"], report["modelIdentities"], labels)
    report["nativeInputDiagnostic"] = input_diagnostic(root, report)
    report["speedComparisons"] = []
    for scope in ("pilot", "full"):
        by_model = {row["model"]: row for row in report[scope]}
        for choice in ("distilled-f1", "distilled-recall"):
            for baseline in ("production", "mobile-large"):
                for stage in ("ralliesReadySeconds", "allReadySeconds"):
                    actual = by_model[choice]["medianTiming"][stage]
                    control = by_model[baseline]["medianTiming"][stage]
                    report["speedComparisons"].append(dict(scope=scope, model=choice, baseline=baseline,
                        timing=stage, secondsDifference=actual - control, ratio=actual / control,
                        comparison="same corrected pilot sequence" if scope == "pilot" else "historical corrected full-video singleton"))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root or Path(private_value(ARTIFACT_INDEX))
    neural = Path(private_value("private-reference-0218"))
    result = build(root, neural.parent / "recording-044-label-snapshot.private.json", neural,
        Path(private_value("private-reference-0219")), Path(private_value("private-reference-0221")))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "distilled-mobile-large-native-benchmark.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "distilled-mobile-large-native-benchmark.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "pilotModels": len(result["pilot"]), "fullModels": len(result["full"])}))


if __name__ == "__main__":
    main()
