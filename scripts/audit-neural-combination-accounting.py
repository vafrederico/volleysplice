#!/usr/bin/env python3
"""Independent, standard-library-only accounting oracle for combination studies.

No production, neural, or shared interval helpers are imported. An endpoint
sweep classifies each atomic span against model export, padded human export,
human core, and ignored time. Overrides are final exports, never raw proposals.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real


PADDINGS = (0, 1, 2, 3)
JOIN_GAP_SECONDS = 3.0
METRIC_FIELDS = (
    "P_pad", "R_core", "F1_padP_coreR", "paddedModelExportSeconds",
    "paddedHumanExportSeconds", "exportDurationDifferenceSeconds",
    "evaluableVideoSeconds", "correctlyRemovedSeconds",
    "incorrectlyRemovedSeconds", "incorrectExportSeconds", "missedCoreSeconds",
    "coreHumanSeconds", "paddedIntersectionSeconds", "coreIntersectionSeconds",
)
SUM_FIELDS = METRIC_FIELDS[3:]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, label):
    _require(isinstance(value, Real) and not isinstance(value, bool)
             and math.isfinite(value), f"{label} must be finite numeric data")
    return float(value)


def _close(actual, expected, label):
    expected = _number(expected, label)
    _require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8),
             f"{label}: independently computed {actual!r} != {expected!r}")


def _intervals(rows, duration, label):
    _require(isinstance(rows, (list, tuple)), f"{label} must be interval rows")
    result = []
    for index, row in enumerate(rows):
        if isinstance(row, Mapping):
            _require("start" in row and "end" in row, f"{label}[{index}] lacks boundaries")
            boundaries = (row["start"], row["end"])
        else:
            _require(isinstance(row, (list, tuple)) and len(row) == 2,
                     f"{label}[{index}] must be a start/end object or two-item pair")
            boundaries = row
        start = _number(boundaries[0], f"{label}[{index}].start")
        end = _number(boundaries[1], f"{label}[{index}].end")
        _require(start < end, f"{label}[{index}] must have positive duration")
        start, end = max(0.0, start), min(duration, end)
        if start < end:
            result.append((start, end))
    return result


def _atomic_spans(collections, *, bounds=()):
    """Yield consecutive endpoints and occupancy for independent collections."""
    changes = {point: [0] * len(collections) for point in bounds}
    for channel, intervals in enumerate(collections):
        for start, end in intervals:
            changes.setdefault(start, [0] * len(collections))[channel] += 1
            changes.setdefault(end, [0] * len(collections))[channel] -= 1
    points = sorted(changes)
    active = [0] * len(collections)
    for index, point in enumerate(points[:-1]):
        for channel, delta in enumerate(changes[point]):
            active[channel] += delta
            _require(active[channel] >= 0, "Invalid endpoint sweep activity")
        yield point, points[index + 1], tuple(value > 0 for value in active)


def _sweep_union(intervals, exclusions=()):
    output = []
    for start, end, (included, excluded) in _atomic_spans((intervals, exclusions)):
        if included and not excluded:
            if output and output[-1][1] == start:
                output[-1] = (output[-1][0], end)
            else:
                output.append((start, end))
    return output


def _canonical_export(intervals, ignored, duration, padding):
    expanded = [(max(0.0, start - padding), min(duration, end + padding))
                for start, end in intervals]
    joined = []
    for start, end in _sweep_union(expanded):
        if joined and 0 < start - joined[-1][1] < JOIN_GAP_SECONDS:
            joined[-1] = (joined[-1][0], end)
        else:
            joined.append((start, end))
    # Exclusions happen last. Never close a gap after subtracting ignored time.
    return _sweep_union(joined, ignored)


def _partition_checks(row, label):
    """Each identity uses independently accumulated atomic-span durations."""
    m, h, v = (row[key] for key in (
        "paddedModelExportSeconds", "paddedHumanExportSeconds", "evaluableVideoSeconds"))
    tp, tn, fn, fp = (row[key] for key in (
        "paddedIntersectionSeconds", "correctlyRemovedSeconds",
        "incorrectlyRemovedSeconds", "incorrectExportSeconds"))
    identities = (
        (tp + fp, m, "export = TP + FP"),
        (tp + fn, h, "human export = TP + FNpad"),
        (tp + tn + fn + fp, v, "four-way partition"),
        (m + tn + fn, v, "export + correctly removed + incorrectly removed"),
        (tn + fp, v - h, "unwanted-time partition"),
        (row["coreIntersectionSeconds"] + row["missedCoreSeconds"],
         row["coreHumanSeconds"], "core retained + missed"),
        (fp - fn, m - h, "export difference = FP - FNpad"),
    )
    for actual, expected, name in identities:
        _close(actual, expected, f"{label}: {name}")
    _require(row["coreHumanSeconds"] <= h + 1e-8, f"{label}: core exceeds human export")
    return len(identities)


def _record_accounting(record, padding, override):
    duration, ignored = record["duration"], record["ignored"]
    human = _canonical_export(record["rallies"], ignored, duration, padding)
    if override is None:
        model = _canonical_export(record["predictions"], ignored, duration, padding)
    else:
        # Actual-app and review exports may intentionally keep gaps below 3s.
        # Normalize their set representation, without re-padding or rejoining.
        model = _sweep_union(_intervals(override, duration, "export override"), ignored)
    core = _sweep_union(record["rallies"], ignored)
    row = {key: 0.0 for key in SUM_FIELDS}
    pieces = {key: [] for key in SUM_FIELDS}
    for start, end, (m, h, c, excluded) in _atomic_spans(
            (model, human, core, ignored), bounds=(0.0, duration)):
        if excluded:
            continue
        active_fields = {
            "evaluableVideoSeconds": True,
            "paddedModelExportSeconds": m,
            "paddedHumanExportSeconds": h,
            "correctlyRemovedSeconds": not m and not h,
            "incorrectlyRemovedSeconds": h and not m,
            "incorrectExportSeconds": m and not h,
            "missedCoreSeconds": c and not m,
            "coreHumanSeconds": c,
            "paddedIntersectionSeconds": m and h,
            "coreIntersectionSeconds": m and c,
        }
        for key, include in active_fields.items():
            if include:
                pieces[key].append(end - start)
    for key, values in pieces.items():
        row[key] = math.fsum(values)
    row["exportDurationDifferenceSeconds"] = row["paddedModelExportSeconds"] - row["paddedHumanExportSeconds"]
    return row


def audit_duration_records(records, expected_rows, export_overrides=None):
    """Audit four pooled metric rows; return counts, never a model selection.

    ``records`` are standard serialized recording dictionaries. ``expected_rows``
    contains exactly one dictionary for each numeric padding value 0, 1, 2, 3;
    additional fields are ignored. Every metric in ``METRIC_FIELDS`` is required.
    If provided, overrides must contain exactly the recording IDs and string
    padding keys ``"0"`` through ``"3"``. They represent final export intervals.
    All interval lists accept start/end objects or two-item numeric pairs.
    An empty model export has precision 0; an empty human core has recall 0.
    Call this function separately for any desired recording/source-group subset.
    """
    _require(isinstance(records, Sequence) and not isinstance(records, (str, bytes))
             and len(records) > 0, "Records must be a nonempty sequence")
    parsed, identifiers = [], set()
    for record in records:
        _require(isinstance(record, Mapping), "Each recording must be an object")
        _require(all(isinstance(record.get(key), str) and record[key]
                     for key in ("id", "sourceGroup")), "Missing recording identity")
        _require(record["id"] not in identifiers, "Duplicate recording ID")
        identifiers.add(record["id"])
        duration = _number(record.get("durationSeconds"), "durationSeconds")
        _require(duration > 0, "Video duration must be positive")
        _require(all(key in record for key in ("rallies", "predictions", "ignoredIntervals")),
                 "Recording missing required interval lists")
        parsed.append({"id": record["id"], "group": record["sourceGroup"], "duration": duration,
                       "rallies": _intervals(record["rallies"], duration, "rallies"),
                       "predictions": _intervals(record["predictions"], duration, "predictions"),
                       "ignored": _intervals(record["ignoredIntervals"], duration, "ignoredIntervals")})
    _require(isinstance(expected_rows, (list, tuple)) and len(expected_rows) == 4,
             "Expected exactly four padding rows")
    expected_by_pad = {}
    for row in expected_rows:
        _require(isinstance(row, Mapping), "Expected padding row must be an object")
        pad = _number(row.get("paddingSecondsBeforeAndAfter"), "paddingSecondsBeforeAndAfter")
        _require(pad in PADDINGS and pad not in expected_by_pad, "Invalid or duplicate padding case")
        _require(all(key in row for key in METRIC_FIELDS), "Expected padding row missing required metric")
        if "joinGapSeconds" in row:
            _close(JOIN_GAP_SECONDS, row["joinGapSeconds"], "joinGapSeconds")
        expected_by_pad[pad] = row
    if export_overrides is not None:
        _require(isinstance(export_overrides, Mapping) and set(export_overrides) == identifiers,
                 "Export override recording scope differs")
        for overrides in export_overrides.values():
            _require(isinstance(overrides, Mapping) and set(overrides) == {str(p) for p in PADDINGS},
                     "Export override requires exactly four string padding keys")
    checks = 0
    for padding in PADDINGS:
        per_record = []
        for record in parsed:
            override = None if export_overrides is None else export_overrides[record["id"]][str(padding)]
            values = _record_accounting(record, padding, override)
            checks += _partition_checks(values, f"{record['id']} pad={padding}")
            per_record.append(values)
        pooled = {key: math.fsum(row[key] for row in per_record) for key in SUM_FIELDS}
        checks += _partition_checks(pooled, f"pooled pad={padding}")
        model, core = pooled["paddedModelExportSeconds"], pooled["coreHumanSeconds"]
        precision = pooled["paddedIntersectionSeconds"] / model if model else 0.0
        recall = pooled["coreIntersectionSeconds"] / core if core else 0.0
        pooled.update(P_pad=precision, R_core=recall,
                      F1_padP_coreR=2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        for key in METRIC_FIELDS:
            _close(pooled[key], expected_by_pad[padding][key], f"pad={padding} {key}")
    return {"passed": True, "recordingCount": len(parsed),
            "sourceGroupCount": len({row["group"] for row in parsed}),
            "paddingCasesAudited": len(PADDINGS),
            "recordingPaddingRowsAudited": len(parsed) * len(PADDINGS),
            "metricValuesCompared": len(METRIC_FIELDS) * len(PADDINGS),
            "partitionIdentitiesChecked": checks,
            "overrideExportRowsAudited": len(parsed) * len(PADDINGS) if export_overrides is not None else 0}
