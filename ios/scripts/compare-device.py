#!/usr/bin/env python3
"""Frozen device implementation diagnostics; stdlib only, no decoding or tuning.

python compare-device.py --device analysis.json --gold reviewed.labels.json \
  --recording-id indoor-example --output report.json

--cases accepts a JSON array of objects with device, gold, recordingId and optional
reference, referenceKey, deviceReferenceKey, referenceLabel. Paths are relative to
that manifest. Every report includes the fixed 0/1/2/3 s sweep, primary 2 s, and
pooled numerators/denominators. Protected recordings require an explicit diagnostic
flag and are never eligible for model selection. --self-test exercises the metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

PADDINGS = (0, 1, 2, 3)
PRIMARY_PADDING = 2
JOIN_GAP = 3.0
VARIANTS = {"ensemble": "intervals", "previous": "previous.intervals", "allLabels": "allLabels.intervals"}


def load(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def at(payload, key):
    for part in key.split("."):
        if not isinstance(payload, dict) or part not in payload:
            return None
        payload = payload[part]
    return payload


def ranges(rows, duration):
    if not isinstance(rows, list):
        raise ValueError("Expected an interval array")
    result = []
    for row in rows:
        if isinstance(row, dict):
            start, end = row.get("start"), row.get("end")
            if row.get("included", True) is False:
                continue
        else:
            start, end = row
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (start, end)) or end <= start:
            raise ValueError("Expected finite positive intervals in source seconds")
        start, end = max(0.0, float(start)), min(duration, float(end))
        if end > start:
            result.append((start, end))
    return sorted(result)


def merge(intervals, gap=0.0):
    """Merge touching/overlap; join positive gaps only when strictly less than gap."""
    result = []
    for start, end in sorted(intervals):
        if result and (start <= result[-1][1] or start - result[-1][1] < gap):
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def subtract(intervals, ignored):
    result = list(intervals)
    for left, right in merge(ignored):
        pieces = []
        for start, end in result:
            if right <= start or left >= end:
                pieces.append((start, end))
            else:
                if start < left:
                    pieces.append((start, min(left, end)))
                if right < end:
                    pieces.append((max(right, start), end))
        result = pieces
    # Deliberately do not rejoin across the ignored time.
    return result


def seconds(intervals):
    return math.fsum(end - start for start, end in intervals)


def intersection(a, b):
    i = j = 0
    total = []
    while i < len(a) and j < len(b):
        total.append(max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0])))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return math.fsum(total)


def padded(intervals, padding, duration, ignored):
    clipped = [(max(0.0, a - padding), min(duration, b + padding)) for a, b in intervals]
    return subtract(merge(clipped, JOIN_GAP), ignored)


def ratios(stats):
    p = stats["intersectionPaddedHumanSeconds"] / stats["paddedModelExportSeconds"] if stats["paddedModelExportSeconds"] else 0.0
    r = stats["intersectionCoreHumanSeconds"] / stats["coreHumanSeconds"] if stats["coreHumanSeconds"] else 0.0
    return dict(stats, P_pad=p, R_core=r, F1_padP_coreR=2 * p * r / (p + r) if p + r else 0.0,
                exportDurationDifferenceSeconds=stats["paddedModelExportSeconds"] - stats["paddedHumanExportSeconds"])


def evaluate(model, human, duration, ignored, padding):
    model_plus, human_plus = padded(model, padding, duration, ignored), padded(human, padding, duration, ignored)
    core = subtract(merge(human), ignored)
    if not core:
        raise ValueError("Human core time is empty after ignored subtraction")
    return dict(ratios({"paddedModelExportSeconds": seconds(model_plus), "paddedHumanExportSeconds": seconds(human_plus),
                       "coreHumanSeconds": seconds(core), "intersectionPaddedHumanSeconds": intersection(model_plus, human_plus),
                       "intersectionCoreHumanSeconds": intersection(model_plus, core)}),
                beforePaddingSeconds=padding, afterPaddingSeconds=padding, joinGapSeconds=JOIN_GAP,
                paddedModelRangeCount=len(model_plus), paddedHumanRangeCount=len(human_plus))


def pooled(rows):
    keys = ("paddedModelExportSeconds", "paddedHumanExportSeconds", "coreHumanSeconds",
            "intersectionPaddedHumanSeconds", "intersectionCoreHumanSeconds")
    result = ratios({key: math.fsum(row[key] for row in rows) for key in keys})
    result.update(beforePaddingSeconds=rows[0]["beforePaddingSeconds"], afterPaddingSeconds=rows[0]["afterPaddingSeconds"],
                  joinGapSeconds=JOIN_GAP, recordings=len(rows))
    return result


def iou(a, b):
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return overlap / (max(a[1], b[1]) - min(a[0], b[0]))


def ordered_matches(truth, model, threshold=0.5):
    """Same maximum-cardinality/then-total-IoU chronological DP as analysis.metrics."""
    rows, cols = len(truth), len(model)
    scores = [[(0, 0.0) for _ in range(cols + 1)] for _ in range(rows + 1)]
    actions = [[0] * (cols + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            best, action = scores[row - 1][col], 1
            if scores[row][col - 1] > best:
                best, action = scores[row][col - 1], 2
            overlap = iou(truth[row - 1], model[col - 1])
            if overlap >= threshold:
                previous = scores[row - 1][col - 1]
                match = previous[0] + 1, previous[1] + overlap
                if match > best:
                    best, action = match, 3
            scores[row][col], actions[row][col] = best, action
    matches, row, col = [], rows, cols
    while row > 0 and col > 0:
        action = actions[row][col]
        if action == 3:
            matches.append((row - 1, col - 1, iou(truth[row - 1], model[col - 1])))
            row -= 1; col -= 1
        elif action == 1:
            row -= 1
        else:
            col -= 1
    return list(reversed(matches))


def guardrails(model, human, ignored):
    truth, predicted = subtract(merge(human), ignored), subtract(merge(model), ignored)
    matched = ordered_matches(truth, predicted)
    p, r = len(matched) / len(predicted) if predicted else 0, len(matched) / len(truth) if truth else 0
    starts = [predicted[b][0] - truth[a][0] for a, b, _ in matched]
    ends = [predicted[b][1] - truth[a][1] for a, b, _ in matched]
    overlap, model_time, truth_time = intersection(predicted, truth), seconds(predicted), seconds(truth)
    return {"eventIouThreshold": 0.5, "eventMatching": "maximum-cardinality-chronological-then-total-IoU",
            "eventUniverse": "unpadded-union-fragments-after-ignored-subtraction", "modelEventCount": len(predicted),
            "humanEventCount": len(truth), "matchedEventCount": len(matched), "eventPrecision": p, "eventRecall": r,
            "eventF1": 2 * p * r / (p + r) if p + r else 0,
            "medianAbsoluteStartErrorSeconds": statistics.median(map(abs, starts)) if starts else None,
            "medianAbsoluteEndErrorSeconds": statistics.median(map(abs, ends)) if ends else None,
            "startErrorsSeconds": starts, "endErrorsSeconds": ends,
            "unmatchedModelRanges": [item for i, item in enumerate(predicted) if i not in {b for _, b, _ in matched}],
            "unmatchedHumanRanges": [item for i, item in enumerate(truth) if i not in {a for a, _, _ in matched}],
            "unpaddedModelSeconds": model_time, "unpaddedHumanSeconds": truth_time,
            "unpaddedIntersectionSeconds": overlap, "unpaddedSymmetricDifferenceSeconds": model_time + truth_time - 2 * overlap,
            "unpaddedTimeIoU": overlap / (model_time + truth_time - overlap) if model_time + truth_time > overlap else 1}


def compare_features(device, reference):
    times, other = device.get("times"), reference.get("times")
    if not isinstance(times, list) or not isinstance(other, list):
        return {"available": False, "reason": "The reference has no timestamped feature matrices; counts cannot establish feature parity."}
    if len(times) != len(other) or any(abs(a - b) > 1e-6 for a, b in zip(times, other)):
        return {"available": False, "reason": "Feature timestamps differ; no positional comparison was performed."}
    result = {"available": False, "rows": len(times), "matrices": {}}
    for key in ("base", "contextual"):
        a, b = device.get(key), reference.get(key)
        if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b) or not times or len(a) % len(times):
            continue
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in a + b):
            raise ValueError("Nonfinite feature matrix")
        columns = len(a) // len(times)
        channel = []
        for column in range(columns):
            errors = [a[i] - b[i] for i in range(column, len(a), columns)]
            channel.append({"column": column, "mae": math.fsum(map(abs, errors)) / len(errors),
                            "rmse": math.sqrt(math.fsum(v * v for v in errors) / len(errors)), "maxAbsoluteError": max(map(abs, errors))})
        result["matrices"][key] = {"columns": columns, "channels": channel, "exactValueMatches": sum(x == y for x, y in zip(a, b)), "values": len(a)}
        result["available"] = True
    return result


def evaluate_case(case, allow_protected):
    device, device_file = load(case["device"])
    gold, gold_file = load(case["gold"])
    recording = gold.get("recording", {})
    if recording.get("id") != case["recordingId"]:
        raise ValueError("Gold recording identity does not match the declared recording")
    protected = recording.get("split") == "test"
    if protected and not allow_protected:
        raise ValueError("Protected test: pass --allow-protected-diagnostic only for a frozen implementation diagnostic, never selection")
    if gold.get("annotation", {}).get("status") != "complete":
        raise ValueError("Gold must be a completed reviewed label document; unvalidated working copies are not gold")
    duration = float(recording["durationSeconds"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid gold duration")
    media_duration = float(device["media"]["duration"])
    if not math.isfinite(media_duration) or media_duration <= 0:
        raise ValueError("Invalid device duration")
    if abs(media_duration - duration) > 0.25:
        raise ValueError("Device/gold durations differ by more than one analysis tick")
    if abs(float(device.get("start", 0))) > 1e-6 or abs(float(device.get("end", media_duration)) - media_duration) > 0.25:
        raise ValueError("Use a full-scope device result for this full-label diagnostic")
    human, ignored = ranges(gold["rallies"], duration), ranges(gold.get("ignoredIntervals", []), duration)
    predictions = {}
    for name, key in VARIANTS.items():
        rows = at(device, key)
        if rows is None:
            continue
        predicted = ranges(rows, duration)
        predictions[name] = {"deviceIntervalKey": key, "inputRangeCount": len(predicted), "coreRanges": predicted,
                             "padding": [evaluate(predicted, human, duration, ignored, padding) for padding in PADDINGS],
                             "guardrails": guardrails(predicted, human, ignored)}
    if not predictions:
        raise ValueError("No device interval outputs found")
    times = device.get("times", [])
    if not times or any(not math.isfinite(t) or t < 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Invalid device analysis timestamps")
    for key in ("base", "contextual"):
        matrix = device.get(key, [])
        if len(matrix) % len(times) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in matrix):
            raise ValueError(f"Invalid aligned finite device {key} matrix")
    stage = device.get("stageSeconds", {})
    result = {"recordingId": case["recordingId"], "sourceGroup": recording.get("sourceGroup"), "split": recording.get("split"),
              "selectionEligible": False, "evaluationDurationSeconds": duration, "deviceDurationSeconds": media_duration,
              "gold": gold_file, "goldAnnotation": gold.get("annotation"), "ignoredIntervals": ignored,
              "ignoredRevisionSha256": hashlib.sha256(json.dumps(gold.get("ignoredIntervals", []), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "device": device_file, "cacheIdentity": device.get("cacheIdentity"), "media": device.get("media"), "roi": device.get("roi"),
              "sourceIdentityEvidence": "Caller-declared original/proxy correspondence; device result has no source-content hash. Equal duration is not a fingerprint match.",
              "rows": len(times), "baseColumns": len(device.get("base", [])) // len(times),
              "contextualColumns": len(device.get("contextual", [])) // len(times),
              "stageSeconds": stage, "summedStageSeconds": math.fsum(stage.values()), "predictions": predictions}
    if case.get("reference"):
        reference, reference_file = load(case["reference"])
        if reference.get("recordingId", case["recordingId"]) != case["recordingId"]:
            raise ValueError("Reference recording identity does not match the case")
        reference_key = case.get("referenceKey", "rallies")
        device_key = case.get("deviceReferenceKey", "previous.intervals")
        ref = ranges(at(reference, reference_key), duration)
        candidate = ranges(at(device, device_key), duration)
        reference_metrics = {key.replace("Human", "Reference").replace("human", "reference"): value
                             for key, value in guardrails(candidate, ref, ignored).items()}
        result["referenceComparison"] = {"label": case.get("referenceLabel", "unclassified-reference"),
            "reference": reference_file, "referenceIntervalKey": reference_key, "deviceIntervalKey": device_key,
            "referenceSource": reference.get("source"), "referenceAnalysis": reference.get("analysis"),
            "metrics": reference_metrics, "features": compare_features(device, reference),
            "interpretation": "Reference agreement is implementation similarity, not human-label accuracy. Inspect source, ROI, model and frontend provenance."}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=Path); parser.add_argument("--gold", type=Path)
    parser.add_argument("--recording-id"); parser.add_argument("--cases", type=Path)
    parser.add_argument("--reference", type=Path); parser.add_argument("--reference-key", default="rallies")
    parser.add_argument("--device-reference-key", default="previous.intervals")
    parser.add_argument("--reference-label", default="unclassified-reference")
    parser.add_argument("--allow-protected-diagnostic", action="store_true")
    parser.add_argument("--output", type=Path); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return 0 if unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(MetricTests)).wasSuccessful() else 1
    if args.cases:
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
        for case in cases:
            for key in ("device", "gold", "reference"):
                if case.get(key): case[key] = str(args.cases.parent / case[key])
    else:
        if not args.device or not args.gold or not args.recording_id:
            parser.error("Provide --device, --gold, --recording-id or --cases")
        cases = [{"device": args.device, "gold": args.gold, "recordingId": args.recording_id,
                  "reference": args.reference, "referenceKey": args.reference_key, "deviceReferenceKey": args.device_reference_key,
                  "referenceLabel": args.reference_label}]
    if not cases or len({case["recordingId"] for case in cases}) != len(cases):
        raise ValueError("Cases must identify unique recordings")
    results = [evaluate_case(case, args.allow_protected_diagnostic) for case in cases]
    common = set.intersection(*(set(case["predictions"]) for case in results))
    report = {"schema": "volleycut-device-parity-diagnostic", "schemaVersion": 1,
              "generatedAt": datetime.now(timezone.utc).isoformat(), "purpose": "frozen-implementation-diagnostic-no-model-selection",
              "primaryPaddingSeconds": PRIMARY_PADDING, "requiredSymmetricPaddingSeconds": list(PADDINGS),
              "joinGapSeconds": JOIN_GAP, "joinComparison": "strictly-less-than", "ignoredSubtraction": "after-padding-and-joining-never-rejoin",
              "aggregation": "pool-intersection-numerators-and-duration-denominators-across-recordings", "cases": results,
              "pooled": {name: [pooled([case["predictions"][name]["padding"][i] for case in results]) for i in range(4)]
                         for name in VARIANTS if name in common}}
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}")
        for name, sweep in report["pooled"].items():
            row = sweep[PRIMARY_PADDING]
            print(f"{name}: P_pad={row['P_pad']:.6f} R_core={row['R_core']:.6f} F1_padP_coreR={row['F1_padP_coreR']:.6f} (fixed 2s diagnostic)")
    else:
        print(rendered)
    return 0


class MetricTests(unittest.TestCase):
    def test_strict_gap_and_retained_gap(self):
        self.assertEqual(merge([(0, 1), (3.999, 5)], 3), [(0, 5)])
        self.assertEqual(merge([(0, 1), (4, 5)], 3), [(0, 1), (4, 5)])
        self.assertEqual(merge([(0, 1), (1, 2)]), [(0, 2)])

    def test_ignored_after_join_never_rejoin(self):
        self.assertEqual(padded([(1, 3), (5, 8)], 1, 10, [(4, 4.5)]), [(0, 4), (4.5, 9)])
        row = evaluate([(1, 3), (5, 8)], [(1, 3), (5, 8)], 10, [(4, 4.5)], 1)
        self.assertEqual(row["paddedModelExportSeconds"], 8.5)
        self.assertEqual(row["F1_padP_coreR"], 1)

    def test_empty_prediction_and_padding_clip(self):
        self.assertEqual(evaluate([], [(1, 2)], 3, [], 2)["F1_padP_coreR"], 0)
        self.assertEqual(padded([(0, 1), (9, 10)], 3, 10, []), [(0, 10)])
        with self.assertRaises(ValueError): evaluate([(0, 1)], [(0, 1)], 3, [(0, 1)], 0)

    def test_pooled_uses_sufficient_statistics(self):
        a = evaluate([(0, 1)], [(0, 1)], 100, [], 0)
        b = evaluate([(0, 100)], [(0, 1)], 100, [], 0)
        result = pooled([a, b])
        self.assertAlmostEqual(result["P_pad"], 2 / 101)
        self.assertEqual(result["R_core"], 1)
        self.assertNotAlmostEqual(result["F1_padP_coreR"], (a["F1_padP_coreR"] + b["F1_padP_coreR"]) / 2)

    def test_event_matching_uses_chronological_maximum(self):
        self.assertEqual(len(ordered_matches([(0, 2), (3, 5)], [(0, 1), (3, 5)])), 2)
        self.assertEqual(len(ordered_matches([(0, 2), (3, 5)], [(0, 5)])), 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"compare-device: {error}", file=sys.stderr)
        sys.exit(2)
