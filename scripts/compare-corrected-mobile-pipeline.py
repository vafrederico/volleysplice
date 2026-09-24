"""Evaluate completed native corrections against frozen before/desktop inputs.

This is an offline diagnostic. It does not train, calibrate, select models, or
access the phone. Private inputs/output locations come from arguments or the
external ledger; generated public reports contain stable indexes only.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.neural_evaluation import evaluate_predictions
from analysis.private_ledger import private_value
from analysis.crop_evaluation import pad_and_merge_intervals, subtract_intervals
from analysis.schema import Interval


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diagnostic_module():
    spec = importlib.util.spec_from_file_location(
        "native_input_diagnostic", Path(__file__).with_name("diagnose-native-neural-features.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def array_from_file(path, shape):
    if path.stat().st_size != int(np.prod(shape)) * 4:
        raise ValueError("Incomplete or incompatible saved native tensor")
    values = np.fromfile(path, dtype="<f4").reshape(shape)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite saved native tensor")
    return values


def ranges_from_probabilities(diagnostic, probabilities, times, duration, decoder):
    example = SimpleNamespace(times=times, valid=np.ones(len(times), bool), duration=duration)
    return [dict(start=r.start, end=r.end) for r in diagnostic.decode(example, probabilities, decoder)]


def local_summary(probabilities, ranges, times, decoder, truth, number):
    start, end = truth["start"], truth["end"]
    local = (times >= start) & (times <= end)
    if not local.any():
        raise ValueError("Target rally has no temporal samples")
    width = min(len(times), max(1, round(decoder["smoothing"] * 4)))
    smoothed = np.convolve(np.pad(probabilities[:, 0],
        (width // 2, (width - 1) // 2), mode="edge"), np.ones(width) / width, mode="valid")
    return dict(humanRallyNumber=number, start=start, end=end,
        maximumLiveProbability=float(probabilities[local, 0].max()),
        maximumSmoothedLiveProbability=float(smoothed[local].max()),
        nearbyRallies=[r for r in ranges if r["end"] > start - 5 and r["start"] < end + 5],
        humanCoreSecondsRetainedWithoutPadding=float(sum(
            max(0, min(end, r["end"]) - max(start, r["start"])) for r in ranges)))


def evaluate_ranges(ranges, duration, labels, recording_index):
    evaluation = evaluate_predictions([dict(id=recording_index, sourceGroup="benchmark-source",
        durationSeconds=duration, rallies=labels["rallies"],
        ignoredIntervals=labels.get("ignoredIntervals", []), predictions=ranges)])
    lost = [dict(humanRallyNumber=r["truthIndex"] + 1, start=r["start"], end=r["end"],
                 evaluableCoreSeconds=r["evaluableCoreSeconds"])
            for r in evaluation["guardrails"]["primaryExportCoverage"]["rallies"] if r["completelyLost"]]
    return dict(rallyCount=len(ranges), rallies=ranges, evaluation=evaluation,
        whollyMissedSavedHumanRallies=len(lost), whollyMissedHumanRallyRanges=lost,
        targetRallies=[])


def evaluate_case(diagnostic, probabilities, times, duration, decoder, labels, recording_index, targets):
    ranges = ranges_from_probabilities(diagnostic, probabilities, times, duration, decoder)
    result = evaluate_ranges(ranges, duration, labels, recording_index)
    result["targetRallies"] = [local_summary(probabilities, ranges, times, decoder,
        labels["rallies"][n - 1], n) for n in targets]
    return result


def range_changes(before, corrected, duration, labels):
    ignored = [Interval(r["start"], r["end"]) for r in labels.get("ignoredIntervals", [])]
    result = []
    for padding in (0, 1, 2, 3):
        def exported(rows):
            core = [Interval(r["start"], r["end"]) for r in rows]
            return subtract_intervals(pad_and_merge_intervals(core, duration, padding, 3), ignored)
        old, new = exported(before), exported(corrected)
        added, removed = subtract_intervals(new, old), subtract_intervals(old, new)
        result.append(dict(paddingSecondsBeforeAndAfter=padding,
            addedExportSeconds=sum(r.end - r.start for r in added),
            removedExportSeconds=sum(r.end - r.start for r in removed),
            addedExportRanges=[dict(start=r.start, end=r.end) for r in added],
            removedExportRanges=[dict(start=r.start, end=r.end) for r in removed]))
    return result


def specialist_comparison(before, corrected):
    output = {}
    for name, anchor, probability in (("servingSide", "anchor", "nearProbability"),
                                     ("sideSwitch", "timestamp", "probability")):
        old, new = before[name], corrected[name]
        if old["modelFingerprint"] != new["modelFingerprint"]:
            raise ValueError("Specialist model changed between native runs")
        old_by_time = {round(r[anchor], 9): r for r in old["candidates"]}
        new_by_time = {round(r[anchor], 9): r for r in new["candidates"]}
        if len(old_by_time) != len(old["candidates"]) or len(new_by_time) != len(new["candidates"]):
            raise ValueError("Ambiguous repeated specialist anchor")
        matched = []
        for time in sorted(old_by_time.keys() & new_by_time.keys()):
            first, second = old_by_time[time], new_by_time[time]
            item = dict(timestamp=time, beforeProbability=first[probability],
                        correctedProbability=second[probability],
                        probabilityDifference=second[probability] - first[probability])
            for field in ("verdict", "side", "serveDecisionSource", "kind"):
                if field in first or field in second:
                    item["before" + field[0].upper() + field[1:]] = first.get(field)
                    item["corrected" + field[0].upper() + field[1:]] = second.get(field)
            matched.append(item)
        entry = dict(modelFingerprint=old["modelFingerprint"],
            beforeFeatureVersion=old["featureVersion"], correctedFeatureVersion=new["featureVersion"],
            beforeEvaluatedCandidateCount=old["rows"], correctedEvaluatedCandidateCount=new["rows"],
            beforePublishedPredictionCount=len(old_by_time), correctedPublishedPredictionCount=len(new_by_time),
            sameAnchorCount=len(matched), matchedAnchors=matched,
            removedAnchors=sorted(old_by_time.keys() - new_by_time.keys()),
            addedAnchors=sorted(new_by_time.keys() - old_by_time.keys()))
        if name == "servingSide":
            entry.update(beforeVerdicts=dict(Counter(r["verdict"] for r in old["candidates"])),
                         correctedVerdicts=dict(Counter(r["verdict"] for r in new["candidates"])),
                         sameAnchorVerdictChanges=sum(r["beforeVerdict"] != r["correctedVerdict"] for r in matched))
        output[name] = entry
    output["caveat"] = "Exact timestamp comparisons only. Changed rally boundaries alter specialist anchors and candidate sets; these counts are not specialist accuracy without independent labels. Side-switch rows count evaluated proposals, while published predictions are selected switch markers after the native decoder."
    return output


def completed_case(folder, family, *, allow_completed_case_in_partial_run=False):
    result, plan = read(folder / "result.json"), read(folder / "pipeline-plan.json")
    if (result.get("runId") != plan.get("runId") or
            (result.get("status") != "complete" and not allow_completed_case_in_partial_run)):
        raise ValueError("Completed matching run required")
    specs = {r["id"]: r for r in plan["cases"]}
    rows = [r for r in result["results"] if specs[r["id"]]["family"] == family
            and not specs[r["id"]].get("warmup")]
    if (len(rows) != 1 or rows[0]["status"] != "complete" or
            not rows[0].get("servingSideReady") or not rows[0].get("sideSwitchReady")):
        raise ValueError("One completed measured case required")
    return rows[0], specs[rows[0]["id"]], plan


def feature_differences(before, corrected, desktop, times, config, labels, targets):
    blocks = dict(av104=slice(0, 104), embeddings=slice(104, corrected.shape[1] - 8),
                  quality=slice(corrected.shape[1] - 8, None))
    output = {}
    for name, block in blocks.items():
        output[name] = {
            "beforeVsDesktopMeanAbsoluteDifference": float(abs(before[:, block] - desktop[:, block]).mean()),
            "correctedVsDesktopMeanAbsoluteDifference": float(abs(corrected[:, block] - desktop[:, block]).mean()),
        }
    quality_names = ("content_fraction", "luminance_mean", "luminance_std", "laplacian_variance",
                     "clipped_pixel_fraction", "selected_pts_offset_seconds", "age", "available")
    local = []
    for number in targets:
        truth = labels["rallies"][number - 1]
        mask = (times >= truth["start"] - 16) & (times <= truth["end"] + 16)
        features = []
        for i, name in enumerate(quality_names):
            stats = dict(feature=name)
            for source, matrix in (("before", before), ("corrected", corrected), ("desktop", desktop)):
                standardized = matrix[mask, matrix.shape[1] - 8 + i]
                # Input standardization clamps to +/-10; inverse values are
                # reconstructed clipped inputs, never asserted to be raw pixels.
                reconstructed = standardized * config["scale"][104 + i] + config["mean"][104 + i]
                stats[source] = dict(standardizedMean=float(standardized.mean()),
                    reconstructedClippedInputMean=float(reconstructed.mean()),
                    samplesAtStandardizationLimit=int((abs(standardized) >= 10).sum()))
            features.append(stats)
        local.append(dict(humanRallyNumber=number, contextSecondsEachSide=16, features=features))
    return dict(blocks=output, targetQualityScalars=local,
        scalarCaveat="Reconstructed clipped inputs undo the fitted affine transform, not the +/-10 clamp. These are model-input values, not exact raw pixel statistics.")


def narrow_feature_groups(dimension):
    from analysis.features import feature_names
    from analysis.config import FeatureConfig, NOISE_NORMALIZED_AUDIO_FEATURE_SET
    names = feature_names(FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET))
    groups = dict(
        av_video=[i for i, name in enumerate(names) if not name.startswith("audio_")],
        av_audio=[i for i, name in enumerate(names) if name.startswith("audio_")],
        av_focus_blur=[names.index(name) for name in ("focus_quality", "blur_probability")],
        av_visibility=[names.index("visibility_quality")],
        clipped_pixel_fraction=[dimension - 4], selected_pts_offset_seconds=[dimension - 3],
        clipped_pixel_and_pts=[dimension - 4, dimension - 3])
    for i, name in enumerate(("content_fraction", "luminance_mean", "luminance_std", "laplacian_variance",
                              "clipped_pixel_fraction", "selected_pts_offset_seconds", "age", "available")):
        groups["quality_" + name] = [dimension - 8 + i]
    return groups


def markdown(report):
    lines = ["# Corrected native Mobile-TCN comparison", "",
        f"Recording `{report['recordingIndex']}`; label SHA-256 `{report['labelSha256']}`.", "",
        report["scope"], "", report["metricContract"], "",
        "| Model | Input pipeline | Found rallies | Wholly missed human rallies | P_pad | R_core | F1_padP_coreR |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for model in report["models"]:
        for name, case in model["observed"].items():
            metric = case["evaluation"]["primary"]
            lines.append(f"| {model['family']} | {name} | {case['rallyCount']} | {case['whollyMissedSavedHumanRallies']} | {metric['P_pad']:.2%} | {metric['R_core']:.2%} | {metric['F1_padP_coreR']:.2%} |")
    lines += ["", "The original native runs used a default crop. Corrected runs use full frame and corrected color handling, so before/after measures both changes together; it does not isolate either correction."]
    lines += ["", "Wholly missed means a saved human rally with nonignored core and no retained core overlap after export padding and joining. Recall measures retained human core time, not rally-event recall.", "",
        "## Previously missed rallies", "",
        "| Model | Input pipeline | Human rally | Maximum smoothed live | Entry threshold | Retained unpadded core |",
        "|---|---|---:|---:|---:|---:|"]
    for model in report["models"]:
        for name, case in model["observed"].items():
            for target in case["targetRallies"]:
                lines.append(f"| {model['family']} | {name} | {target['humanRallyNumber']} | {target['maximumSmoothedLiveProbability']:.4f} | {model['decoder']['enter']:.2f} | {target['humanCoreSecondsRetainedWithoutPadding']:.3f}s |")
    lines += ["", "## Padding sensitivity", "",
        "| Model | Input pipeline | Padding each side | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in report["models"]:
        for name, case in model["observed"].items():
            for metric in case["evaluation"]["padding"]:
                lines.append(f"| {model['family']} | {name} | {metric['paddingSecondsBeforeAndAfter']:.0f}s | {metric['P_pad']:.2%} | {metric['R_core']:.2%} | {metric['F1_padP_coreR']:.2%} | {metric['paddedModelExportSeconds']:.3f} | {metric['paddedHumanExportSeconds']:.3f} | {metric['exportDurationDifferenceSeconds']:+.3f} |")
    lines += ["", "The JSON includes every observed and counterfactual rally range, event diagnostics, all padding cases, native temporal/decoder parity, feature differences, and individual clipped-pixel/PTS substitutions. Counterfactual substitutions diagnose residual feature drift; they are not deployed predictions or newly selected models.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    for name in ("results", "graphs", "labels", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--production-results", type=Path,
                        help="Optional completed corrected production run using the new full-frame default")
    parser.add_argument("--production-before", type=Path)
    parser.add_argument("--small-before", type=Path)
    parser.add_argument("--large-before", type=Path)
    parser.add_argument("--recording-index", default="recording-044")
    parser.add_argument("--label-sha256", required=True)
    parser.add_argument("--target-rallies", type=int, nargs="+", default=[16, 17])
    args = parser.parse_args()
    if not re.fullmatch(r"recording-\d+", args.recording_index):
        raise ValueError("Use a recording ledger index")
    if sha256(args.labels) != args.label_sha256:
        raise ValueError("Label revision changed")
    diagnostic_root = args.diagnostics or Path(private_value("private-reference-0217"))
    result, plan = read(args.results / "result.json"), read(args.results / "pipeline-plan.json")
    if result.get("status") != "complete" or result.get("runId") != plan.get("runId"):
        raise ValueError("Completed matching run required before offline evaluation")
    labels = read(args.labels)
    if any(n < 1 or n > len(labels["rallies"]) for n in args.target_rallies):
        raise ValueError("Target human rally number outside label snapshot")
    specs = {r["id"]: r for r in plan["cases"]}
    measured = [r for r in result["results"] if not specs[r["id"]].get("warmup")]
    if sorted(specs[r["id"]]["family"] for r in measured) != ["mobile", "mobile-large"]:
        raise ValueError("Require exactly one completed Small and one Large full run")
    diagnostic = diagnostic_module()
    contract = read(args.graphs / "input-contract.json")
    if contract["recordingIndex"] != args.recording_index:
        raise ValueError("Input contract recording mismatch")
    for name, expected in contract["hashes"].items():
        if sha256(args.graphs / name) != expected:
            raise ValueError("Frozen graph/input contract changed")
    output = dict(recordingIndex=args.recording_index, labelSha256=args.label_sha256,
        scope="Completed corrected physical-device inputs compared with original native and frozen desktop inputs; same checkpoints and decoders, no training, calibration, or model selection. Single recording and run; not a general accuracy qualification.",
        metricContract="Primary: 2s padding each side, join positive gaps strictly under 3s, subtract ignored time without rejoining. Required 0/1/2/3s sensitivity uses identical model/human settings.",
        targetPaddingSeconds=2, joinGapSeconds=3, humanRallyCount=len(labels["rallies"]), models=[])
    for row in measured:
        spec = specs[row["id"]]
        family = spec["family"]
        if row["status"] != "complete" or not spec.get("fullFrame") or spec["seconds"] < 1000:
            raise ValueError("Complete full-frame full-video case required")
        subdirectory = "small" if family == "mobile" else "large"
        previous = read(diagnostic_root / subdirectory / "feature-counterfactuals.json")
        config = read(args.graphs / (family + "-pipeline.json"))
        if (previous["recordingIndex"] != args.recording_index or
                previous["weightsSha256"] != config["weightsSha256"] or
                previous["epoch"] != config["epoch"] or
                previous["decoder"] != config["decoder"] or
                row["neural"]["decoder"] != config["decoder"]):
            raise ValueError("Model/decoder lineage mismatch")
        with np.load(diagnostic_root / subdirectory / "diagnostic-tensors.private.npz") as old:
            before, desktop, times = old["native_features"], old["desktop_features"], old["times"]
            before_p = old["native"]
            desktop_p = old["native_with_desktop_av_embeddings_quality"]
        np.testing.assert_array_equal(times, row["neural"]["times"])
        shape = (row["neural"]["featureRows"], row["neural"]["featureDimension"])
        if before.shape != shape or desktop.shape != shape:
            raise ValueError("Feature layout mismatch")
        corrected = array_from_file(args.results / (row["id"] + "-features.f32"), shape)
        corrected_p = array_from_file(args.results / (row["id"] + "-probabilities.f32"), (len(times), 4))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(args.graphs / (family + "-tcn-dynamic-fp32.onnx")),
                                       sess_options=options, providers=["CPUExecutionProvider"])
        parity = {}
        for name, values, expected in (("before", before, before_p), ("corrected", corrected, corrected_p),
                                       ("desktop", desktop, desktop_p)):
            replay = diagnostic.predict(session, values)
            np.testing.assert_allclose(replay, expected, atol=3e-5, rtol=3e-4)
            parity[name] = float(abs(replay - expected).max())
        def evaluate(probabilities):
            return evaluate_case(diagnostic, probabilities, times, spec["seconds"], config["decoder"],
                                 labels, args.recording_index, args.target_rallies)
        observed = {name: evaluate(p) for name, p in
                    (("before", before_p), ("corrected", corrected_p), ("desktop", desktop_p))}
        expected_bounds = np.asarray([(r["start"], r["end"]) for r in observed["corrected"]["rallies"]]).reshape(-1, 2)
        actual_bounds = np.asarray([(r["start"], r["end"]) for r in row["rallies"]]).reshape(-1, 2)
        np.testing.assert_allclose(expected_bounds, actual_bounds, atol=1e-8, rtol=0)
        swaps = {}
        blocks = dict(av=slice(0, 104), embeddings=slice(104, shape[1] - 8), quality=slice(shape[1] - 8, None))
        for count in range(1, 4):
            for replaced in itertools.combinations(blocks, count):
                values = corrected.copy()
                for block in replaced:
                    values[:, blocks[block]] = desktop[:, blocks[block]]
                swaps["desktop_" + "_".join(replaced)] = evaluate(diagnostic.predict(session, values))
        for name, indexes in narrow_feature_groups(shape[1]).items():
            values = corrected.copy()
            values[:, indexes] = desktop[:, indexes]
            swaps["desktop_" + name] = evaluate(diagnostic.predict(session, values))
        output["models"].append(dict(family=family, modelId=previous["modelId"], epoch=config["epoch"],
            weightsSha256=config["weightsSha256"], decoder=config["decoder"],
            temporalReplayMaxAbsoluteErrors=parity, correctedDecoderParity=True,
            correctedRunTimingMs=dict(total=row["totalMs"], stages=row["stagesMs"]),
            correctedRunThermalStatus=dict(start=row.get("thermalStart"), end=row.get("thermalEnd")),
            observed=observed, correctedFeatureCounterfactuals=swaps,
            featureDifferences=feature_differences(before, corrected, desktop, times, config,
                                                  labels, args.target_rallies)))
        before_folder = ((args.small_before or Path(private_value("private-reference-0209")))
            if family == "mobile" else (args.large_before or Path(private_value("private-reference-0213"))))
        # The original Small run lives in a partial multi-family result whose
        # DINO case failed. Its individually completed Small row is valid.
        old_result = read(before_folder / "result.json")
        old_plan = read(before_folder / "pipeline-plan.json")
        old_specs = {r["id"]: r for r in old_plan["cases"]}
        old_rows = [r for r in old_result["results"] if old_specs[r["id"]]["family"] == family
                    and r["status"] == "complete" and not old_specs[r["id"]].get("warmup")]
        if len(old_rows) != 1 or old_specs[old_rows[0]["id"]]["seconds"] != spec["seconds"]:
            raise ValueError("Original individual native case unavailable or duration mismatch")
        np.testing.assert_allclose(np.asarray([(r["start"], r["end"]) for r in old_rows[0]["rallies"]]),
            np.asarray([(r["start"], r["end"]) for r in observed["before"]["rallies"]]), atol=1e-8, rtol=0)
        output["models"][-1]["specialistComparison"] = specialist_comparison(old_rows[0], row)
        output["models"][-1]["exportChanges"] = range_changes(observed["before"]["rallies"], row["rallies"], spec["seconds"], labels)
    if args.production_results:
        before_folder = args.production_before or Path(private_value("private-reference-0208"))
        # Historical production completed before a later neural case failed;
        # retain its complete per-case observation without requiring that old
        # multi-family queue to have completed successfully.
        before, before_spec, before_plan = completed_case(before_folder, "production",
            allow_completed_case_in_partial_run=True)
        corrected, corrected_spec, corrected_plan = completed_case(args.production_results, "production")
        if (before_spec.get("fullFrame", False) or not corrected_spec.get("fullFrame", False) or
                before_spec["seconds"] != corrected_spec["seconds"] or
                before_plan["sourceUri"] != corrected_plan["sourceUri"]):
            raise ValueError("Production comparison requires the same source/duration, old default crop, and corrected full frame")
        duration = corrected_spec["seconds"]
        observed = {name: evaluate_ranges(row["rallies"], duration, labels, args.recording_index)
                    for name, row in (("before", before), ("corrected", corrected))}
        output["models"].append(dict(family="production", observed=observed,
            scope="Production before used the original default crop; corrected production and neural runs all use full frame. Before/after combines the color-handling fix and removal of default cropping, so their effects are not isolated.",
            probabilityCaveat="Production per-tick rally probabilities and AV tensors were not persisted; rally outputs, confidence values, and specialist candidates are available.",
            correctedRunTimingMs=dict(total=corrected["totalMs"], stages=corrected["stagesMs"]),
            exportChanges=range_changes(before["rallies"], corrected["rallies"], duration, labels),
            specialistComparison=specialist_comparison(before, corrected)))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "corrected-native-comparison.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "corrected-native-comparison.md").write_text(markdown(output), encoding="utf-8")
    print(json.dumps({"complete": True, "recordingIndex": args.recording_index,
        "models": [{"family": m["family"], "corrected": m["observed"]["corrected"]["evaluation"]["primary"],
                    "whollyMissed": m["observed"]["corrected"]["whollyMissedSavedHumanRallies"]}
                   for m in output["models"]]}, indent=2))


if __name__ == "__main__":
    main()
