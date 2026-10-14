#!/usr/bin/env python3
"""Post-hoc scalar motion-quality diagnostics on frozen development predictions.

No model fitting, embedding loading, protected-label access, or operating-point
selection is performed. The large feature document is decoded one row at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value


DATASET_INDEX = "private-reference-0226"
HISTORICAL_INDEX = "private-reference-0231"
OUTPUT_INDEX = "private-reference-0236"
SEED = 3407
CONFIDENCE_THRESHOLD = 0.9
CONFIGURATION = "192x108-r4c6"
PHASES = ("launch", "early", "late")
QUALITY_NAMES = ("meanEnergy", "meanActiveFraction", "meanSmallComponentEnergyFraction", "meanAbsoluteFlowY")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def read_small(path):
    require(Path(path).stat().st_size < 1 << 20, "Metadata exceeds the small-file budget")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verified_reference(reference):
    path = Path(reference["path"])
    require(digest(path) == reference["sha256"], "Historical input content changed")
    return path


def stream_rows(path):
    """Decode the registered indented document without retaining its other rows."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip() == '"rows": [':
                break
        else:
            raise ValueError("Registered dataset row array not found")
        buffer = ""
        for chunk in iter(lambda: source.read(65536), ""):
            buffer += chunk
            require(len(buffer) < 1 << 20, "Single feature row exceeds streaming budget")
            while True:
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:].lstrip()
                if buffer.startswith("]"):
                    return
                try:
                    row, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                require(isinstance(row, dict), "Invalid dataset row")
                yield row
                buffer = buffer[end:]
    raise ValueError("Registered dataset row array is incomplete")


def recording_ranks(rows, values):
    """Label-free midpoint ranks for descriptive within-recording comparisons."""
    ranks = np.empty_like(values)
    for recording in sorted({row["recordingIndex"] for row in rows}):
        indexes = np.flatnonzero([row["recordingIndex"] == recording for row in rows])
        for column in range(values.shape[1]):
            local = values[indexes, column]
            ranks[indexes, column] = [
                0.5 if len(local) == 1 else
                ((local < value).sum() + 0.5 * ((local == value).sum() - 1)) / (len(local) - 1)
                for value in local
            ]
    return ranks


def describe(mask, truth, values):
    return {"rows": int(mask.sum()), "nearLabels": int(truth[mask].sum()),
            "qualityMedians": dict(zip(QUALITY_NAMES, np.median(values[mask], axis=0).tolist(), strict=True))
            if mask.any() else None}


def immutable(path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == text, "Preserve the existing diagnostic artifact")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    historical = Path(private_value(HISTORICAL_INDEX))
    output = Path(private_value(OUTPUT_INDEX))
    dataset = Path(private_value(DATASET_INDEX))
    require(historical.is_absolute() and output.is_absolute() and dataset.is_absolute(),
            "Private ledger values must be absolute runtime locations")
    registration_path = historical / "registration.json"
    registration = read_small(registration_path)
    prior_registration = read_small(verified_reference(registration["contract"]["priorStudy"]))
    registered_dataset = prior_registration["contract"]["inputs"]["dataset"]
    require(dataset.resolve() == Path(registered_dataset["path"]).resolve(), "Dataset ledger reference differs")
    verified_reference(registered_dataset)
    rows_path = verified_reference(registration["contract"]["data"]["rows"])
    rows = read_small(rows_path)
    require(len(rows) == 968, "Frozen population changed")
    for index, row in enumerate(rows):
        require(row["rowIndex"] == index and row["label"] in (0, 1), "Frozen row order or label differs")
        require(re.fullmatch(r"source-group-\d+", row["sourceGroup"]) is not None,
                "Source groups must use public ledger indexes")
    ledger_location = (os.environ.get("VOLLEYCUT_PRIVATE_LEDGER_WINDOWS" if os.name == "nt" else
                                     "VOLLEYCUT_PRIVATE_LEDGER_POSIX") or os.environ["VOLLEYCUT_PRIVATE_LEDGER"])
    aliases = json.loads(Path(ledger_location).read_text(encoding="utf-8"))["originalToAlias"]
    qualities = []
    for original in stream_rows(dataset):
        require(original["sourceSplit"] != "test", "Protected source cannot enter this development diagnostic")
        if original["environment"] == "beach":
            continue
        require(len(qualities) < len(rows), "Unexpected additional development row")
        row = rows[len(qualities)]
        require(row["label"] == original["label"] and original["decision"] in ("near", "far")
                and row["sourceGroup"] == aliases[original["sourceGroup"]]
                and row["recordingIndex"] == aliases[original["recordingId"]], "Frozen feature identity differs")
        raw = original["configurations"][CONFIGURATION]
        qualities.append([
            float(np.mean([raw[f"{phase}:{key}"] for phase in PHASES]))
            for key in ("energyMean", "activeFraction", "smallComponentEnergyFraction")
        ] + [float(np.mean([abs(raw[f"{phase}:flowY"]) for phase in PHASES]))])
    values = np.asarray(qualities, dtype=np.float64)
    require(values.shape == (968, 4) and np.isfinite(values).all(), "Invalid scalar quality matrix")
    predictions_path = historical / "nested-predictions.npz"
    with np.load(predictions_path, allow_pickle=False) as saved:
        probability = saved[f"logistic-{SEED}-probabilities"].copy()
        threshold = saved[f"logistic-{SEED}-thresholds"].copy()
    require(probability.shape == threshold.shape == (968,) and np.isfinite(probability).all()
            and np.isfinite(threshold).all() and np.all((probability > 0) & (probability < 1)),
            "Invalid historical logistic scores")
    truth = np.asarray([row["label"] for row in rows])
    correct = (probability >= 0.5) == truth
    confidence = np.maximum(probability, 1 - probability)
    high = confidence >= CONFIDENCE_THRESHOLD
    ranks = recording_ranks(rows, values)
    magnitude = np.abs(np.log(probability / (1 - probability)))
    floor = 1 / (1 + np.exp(-(magnitude - 3 - np.log(49))))
    sources = []
    for group in sorted({row["sourceGroup"] for row in rows}):
        scope = np.asarray([row["sourceGroup"] == group for row in rows]) & high
        sources.append({"sourceGroup": group, "highCorrect": describe(scope & correct, truth, values),
                        "highWrong": describe(scope & ~correct, truth, values)})
    result = {
        "schemaVersion": 1, "kind": "serving-side-quality-post-hoc-diagnostics-v1", "date": "2026-10-14",
        "studyIndex": OUTPUT_INDEX, "historicalStudyIndex": HISTORICAL_INDEX, "datasetIndex": DATASET_INDEX,
        "codeSha256": digest(__file__),
        "inputSha256": {"dataset": registered_dataset["sha256"], "historicalRegistration": digest(registration_path),
                        "rows": digest(rows_path), "historicalNestedPredictions": digest(predictions_path)},
        "rows": 968, "recordings": len({row["recordingIndex"] for row in rows}), "sourceGroups": len(sources),
        "diagnosticChoices": {
            "seed": SEED, "historicalModel": "nested-logistic", "sideThreshold": 0.5,
            "confidence": "max(p, 1-p)", "highConfidenceThreshold": CONFIDENCE_THRESHOLD,
            "configuration": CONFIGURATION, "phases": list(PHASES), "qualities": list(QUALITY_NAMES),
            "aggregation": "Equal-weight phase mean; flowY is made absolute before averaging.",
            "focusChosenBeforeInspectingTheseDiagnosticOutcomes": True,
            "qualification": "Post-hoc follow-up focus fixed before this diagnostic's values were examined; not a prospective training preregistration.",
        },
        "counts": {"highConfidence": int(high.sum()), "highCorrect": int((high & correct).sum()),
                   "highWrong": int((high & ~correct).sum()),
                   "highConfidencePrecision": float(correct[high].mean()),
                   "highWrongUsingHistoricalNestedThreshold": int((high & ((probability >= threshold) != truth)).sum()),
                   "highRoutesDifferUsingHistoricalNestedThreshold": int((high & ((probability >= .5) != (probability >= threshold))).sum()),
                   "allWrongFixedHalf": int((~correct).sum())},
        "pooled": {"highCorrect": describe(high & correct, truth, values),
                   "highWrong": describe(high & ~correct, truth, values), "lowerConfidence": describe(~high, truth, values)},
        "withinRecordingRanks": {"method": "Midpoint ranks of each phase-averaged quality over all frozen serves in the same recording; singleton rank 0.5.",
                                 "highCorrect": describe(high & correct, truth, ranks),
                                 "highWrong": describe(high & ~correct, truth, ranks)},
        "bySourceGroup": sources,
        "wrongConfidenceAndCorrectionBound": {
            "wrongConfidenceCounts": {str(cut): int((~correct & (confidence >= cut)).sum()) for cut in (.9, .99, .999, .9999)},
            "maximumWrongConfidence": float(confidence[~correct].max()),
            "maximumWrongAbsoluteLogit": float(magnitude[~correct].max()),
            "topTenWrongAbsoluteLogits": sorted(magnitude[~correct].tolist(), reverse=True)[:10],
            "cost49Residual3FloorFormula": "sigmoid(abs(base_logit) - 3 - log(49))",
            "floorCounts": {str(cut): int((~correct & (floor >= cut)).sum()) for cut in (.5, .9, .99)},
            "maximumWrongScoreFloor": float(floor[~correct].max()),
            "interpretation": "Hypothetical lower score with a maximum opposing residual and cost ratio 49, computed from historical scores; not an attainable trained-model guarantee or exact new FP32 replay.",
        },
        "limitations": [
            "Only ten high-confidence mistakes; half are from one source. Other sources can show stronger motion among wrong guesses.",
            "Seven of ten mistakes have a near label, so class composition differs as well as source composition.",
            "Flight activeFraction is approximately capped at 0.1 by the 90th-percentile motion threshold; depletion reflects the additional absolute floor.",
            "The four measures suggest testing motion-quality interactions but do not establish a universal rejection rule.",
            "Existing phase-wise recording ranks already contain related information; these descriptive associations do not prove incremental benefit from raw values.",
            "Within-recording ranks here are only a descriptive adjustment, not new model inputs or a selected operating point.",
            "Reviewed true serves only; no unclear or not-serve supervision, no fresh rally detection, and no protected-test claim.",
        ],
        "usedForSelection": False, "modelFittingPerformed": False, "embeddingsLoaded": False,
        "protectedLabelsOpened": False, "productionChanged": False,
    }
    private_report = output / "quality-diagnostics.json"
    immutable(private_report, result)
    immutable(args.report, result)
    require(digest(private_report) == digest(args.report), "Public and private diagnostics differ")
    print(json.dumps({"studyIndex": OUTPUT_INDEX, "counts": result["counts"],
                      "reportSha256": digest(args.report)}, sort_keys=True))


if __name__ == "__main__":
    main()
