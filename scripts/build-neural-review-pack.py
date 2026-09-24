#!/usr/bin/env python3
"""Create a bounded, label-aware descriptive TCN/linear error-review pack.

Uses fixed seed3407 as an engineering exemplar, not the best seed. Reads six
small proxy windows, at most five frames per window. Never changes labels,
predictions, training artifacts, or production models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value

from analysis.crop_evaluation import pad_and_merge_intervals, subtract_intervals
from analysis.schema import Interval


ROOT = Path(private_value('private-reference-0096'))
EXEMPLAR_SEED = 3407
PADDING = 2.0
JOIN_GAP = 3.0
MAX_CASES_PER_TYPE = 3
FRAMES_PER_CASE = 5


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def intervals(rows: list[dict]) -> tuple[Interval, ...]:
    return tuple(Interval(float(row["start"]), float(row["end"])) for row in rows)


def export_union(row: dict, key: str) -> tuple[Interval, ...]:
    return subtract_intervals(
        pad_and_merge_intervals(intervals(row[key]), row["durationSeconds"], PADDING, JOIN_GAP),
        intervals(row["ignoredIntervals"]),
    )


def removed_false_positives(tcn: dict, linear: dict) -> list[dict]:
    reference = {row["id"]: row for row in linear["predictions"]}
    spans = []
    for row in tcn["predictions"]:
        other = reference[row["id"]]
        if row["rallies"] != other["rallies"] or row["ignoredIntervals"] != other["ignoredIntervals"]:
            raise ValueError("compared predictions use different labels or ignored ranges")
        removed = subtract_intervals(export_union(other, "predictions"),
                                     (*export_union(row, "predictions"), *export_union(row, "rallies")))
        for span in removed:
            spans.append({"recordingId": row["id"], "sourceGroup": row["sourceGroup"],
                          "start": span.start, "end": span.end, "durationSeconds": span.end - span.start})
    return sorted(spans, key=lambda row: (-row["durationSeconds"], row["recordingId"], row["start"]))


def changed_rallies(tcn: dict, linear: dict) -> tuple[list[dict], dict]:
    reference = {(row["recordingId"], row["truthIndex"]): row
                 for row in linear["evaluation"]["guardrails"]["primaryExportCoverage"]["rallies"]}
    groups = {row["id"]: row["sourceGroup"] for row in tcn["predictions"]}
    losses = []
    counts = Counter()
    group_counts = Counter()
    for row in tcn["evaluation"]["guardrails"]["primaryExportCoverage"]["rallies"]:
        other = reference[(row["recordingId"], row["truthIndex"])]
        new_complete = row["completelyLost"] and not other["completelyLost"]
        new_partial = row["partiallyLost"] and other["fullyCovered"]
        counts["newCompleteLosses"] += int(new_complete)
        counts["newPartialLossesFromFullyCovered"] += int(new_partial)
        counts["recoveredCompleteLosses"] += int(other["completelyLost"] and not row["completelyLost"])
        if new_complete or new_partial:
            group = groups[row["recordingId"]]
            group_counts[group] += 1
            losses.append({
                **{key: row[key] for key in ("recordingId", "truthIndex", "start", "end", "tags")},
                "sourceGroup": group, "durationSeconds": row["end"] - row["start"],
                "lossType": "new-complete-loss" if new_complete else "new-partial-loss-from-fully-covered",
                "additionalCoreSecondsLost": other["retainedCoreSeconds"] - row["retainedCoreSeconds"],
                "linearCoverage": other["coverage"], "tcnCoverage": row["coverage"],
                "linearRetainedCoreSeconds": other["retainedCoreSeconds"],
                "tcnRetainedCoreSeconds": row["retainedCoreSeconds"],
            })
    for name, result in (("linear", linear), ("tcn", tcn)):
        coverage = result["evaluation"]["guardrails"]["primaryExportCoverage"]
        counts[f"{name}CompleteLosses"] = coverage["completeRallyLosses"]
        counts[f"{name}PartialLosses"] = coverage["partialRallyLosses"]
        counts[f"{name}FullyCovered"] = coverage["fullyCoveredRallies"]
    losses.sort(key=lambda row: (-row["additionalCoreSecondsLost"], row["recordingId"], row["start"]))
    return losses, {**dict(counts), "newLossesBySourceGroup": dict(group_counts),
                    "additionalCoreSecondsLostInNewLosses": sum(row["additionalCoreSecondsLost"] for row in losses)}


def nearby(rows: list[dict], start: float, end: float) -> list[dict]:
    return [row for row in rows if row["end"] > start and row["start"] < end]


def contact_sheet(video: Path, output: Path, start: float, end: float, duration: float) -> dict:
    import cv2

    timestamps = np.linspace(max(0.0, start), min(end, max(0.0, duration - 0.1)), FRAMES_PER_CASE)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"status": "unavailable", "reason": "OpenCV could not open exact normalized proxy"}
    tiles, frames = [], []
    try:
        for stamp in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(stamp) * 1000.0)
            ok, frame = cap.read()
            actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not ok:
                return {"status": "unavailable", "reason": f"Frame decode failed at {stamp:.3f}s"}
            height, width = frame.shape[:2]
            factor = min(320 / width, 180 / height)
            resized = cv2.resize(frame, (max(1, round(width * factor)), max(1, round(height * factor))),
                                 interpolation=cv2.INTER_AREA)
            tile = np.full((208, 320, 3), 245, dtype=np.uint8)
            top, left = (180 - resized.shape[0]) // 2, (320 - resized.shape[1]) // 2
            tile[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
            cv2.putText(tile, f"t={actual:.3f}s", (8, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (35, 35, 35), 1, cv2.LINE_AA)
            tiles.append(tile)
            frames.append({"requestedTimeSeconds": float(stamp), "decodedTimeSeconds": actual})
    finally:
        cap.release()
    if not cv2.imwrite(str(output), np.concatenate(tiles, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise OSError(f"could not write contact sheet: {output}")
    return {"status": "complete", "path": output.name, "sha256": digest(output),
            "width": 1600, "height": 208, "frames": frames,
            "limitation": "Five sampled frames support visual context, not an independent continuous-video boundary review."}


def build(root: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"refusing existing review pack: {destination}")
    manifest_path = root / "manifest.json"
    manifest = read(manifest_path)
    audit = read(root / "dataset-audit.json")
    if digest(manifest_path) != audit["manifestSha256"]:
        raise ValueError("manifest no longer matches its frozen audit")
    records = {row["id"]: row for row in manifest["recordings"]}
    study = root / "nested-study-v1"
    registration = read(study / "preregistration.json")
    if registration["contract"]["manifestSha256"] != audit["manifestSha256"]:
        raise ValueError("study and audited manifest differ")
    results, inputs, seed_counts = {}, {}, []
    for seed in registration["contract"]["seeds"]:
        pair = {}
        for kind in ("tcn", "linear"):
            path = study / f"result-{kind}-{seed}.json"
            result = read(path)
            if (result["kind"], result["seed"]) != (kind, seed):
                raise ValueError("result filename/identity mismatch")
            if {row["id"] for row in result["predictions"]} != set(records):
                raise ValueError("result scope differs from the manifest")
            for row in result["predictions"]:
                if row["sourceGroup"] != records[row["id"]]["sourceGroup"] or row["sourceGroup"] in manifest["protectedSourceGroups"]:
                    raise ValueError("protected or mismatched source group in review")
            pair[kind] = result
            inputs[f"{kind}-{seed}"] = {"path": str(path), "sha256": digest(path)}
        losses, counts = changed_rallies(pair["tcn"], pair["linear"])
        removed = removed_false_positives(pair["tcn"], pair["linear"])
        seed_counts.append({"seed": seed, **counts,
                            "removedFalsePositiveSpanCount": len(removed),
                            "removedFalsePositiveSeconds": sum(row["durationSeconds"] for row in removed)})
        results[seed] = (pair, losses, removed)
    pair, losses, removed = results[EXEMPLAR_SEED]
    linear = {row["id"]: row for row in pair["linear"]["predictions"]}
    tcn = {row["id"]: row for row in pair["tcn"]["predictions"]}
    selected = [("new-loss", row) for row in losses[:MAX_CASES_PER_TYPE]]
    selected += [("removed-false-positive", row) for row in removed[:MAX_CASES_PER_TYPE]]
    destination.mkdir(parents=True)
    cases = []
    for index, (kind, source) in enumerate(selected, 1):
        record = records[source["recordingId"]]
        duration = float(record["durationSeconds"])
        start, end = max(0.0, source["start"] - 2), min(duration, source["end"] + 2)
        image_path = destination / f"case-{index:02d}.jpg"
        case = {"caseId": f"case-{index:02d}", "kind": kind, "seed": EXEMPLAR_SEED, **source,
                "videoPath": record["video"], "videoContentSha256": record["contentSha256"],
                "labelSource": record["labelSource"], "evidenceWindow": {"start": start, "end": end},
                "nearbyTruth": nearby(tcn[source["recordingId"]]["rallies"], start - 8, end + 8),
                "nearbyLinearRawPredictions": nearby(linear[source["recordingId"]]["predictions"], start - 8, end + 8),
                "nearbyTcnRawPredictions": nearby(tcn[source["recordingId"]]["predictions"], start - 8, end + 8),
                "nearbyLinearExport": [row.to_dict() for row in export_union(linear[source["recordingId"]], "predictions") if row.end > start - 8 and row.start < end + 8],
                "nearbyTcnExport": [row.to_dict() for row in export_union(tcn[source["recordingId"]], "predictions") if row.end > start - 8 and row.start < end + 8]}
        case["contactSheet"] = contact_sheet(Path(record["video"]), image_path, start, end, duration)
        cases.append(case)
        print(f"Prepared {case['caseId']} {kind}: {source['recordingId']} {source['start']:.3f}-{source['end']:.3f}", flush=True)
    pack = {
        "schemaVersion": 1, "kind": "volleycut-neural-descriptive-review-pack-v1",
        "createdAt": datetime.now(timezone.utc).isoformat(), "manifestSha256": audit["manifestSha256"],
        "contractSha256": registration["sha256"], "fixedExemplarSeed": EXEMPLAR_SEED,
        "metricContext": {"symmetricPaddingSeconds": PADDING, "joinGapSeconds": JOIN_GAP,
                          "ignoredRule": "subtract ignored after padding/joining; never rejoin fragments"},
        "selectionPolicy": {
            "seed": "3407 fixed engineering exemplar, not selected by performance",
            "newLosses": "Top3 newly complete or newly partial-from-full original-rally losses by additional evaluable core seconds lost versus same-seed linear; tie ID,start.",
            "removedFalsePositives": "Top3 contiguous linear-export minus TCN-export minus padded-human-export spans by duration; tie ID,start.",
            "frameBudget": "At most6 windows x5 sampled frames; no continuous reannotation.",
        },
        "status": "descriptive-development-error-analysis-only",
        "labelsModified": False, "retrainingPerformed": False, "protectedInputsUsed": False,
        "perSeedCounts": seed_counts, "cases": cases, "inputs": inputs,
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__).resolve())},
    }
    json_path = destination / "review-pack.json"
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(pack, handle, indent=2, allow_nan=False)
        handle.write("\n")
    lines = ["# TCN versus linear: descriptive development review", "",
             "Fixed seed **3407**, chosen as an engineering exemplar. This is label-aware error review, not new labels, tuning, or a deployment decision.", "",
             "All comparisons use ±2-second export padding and strictly less than3-second gap joining. New losses refer to original gold rallies. Removed false positives are linear-only export spans outside padded human coverage.", "",
             "| Seed | Linear complete / partial loss | TCN complete / partial loss | Newly complete | Newly partial from full | Recovered complete | Removed FP seconds |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in seed_counts:
        lines.append(f"| {row['seed']} | {row['linearCompleteLosses']} / {row['linearPartialLosses']} | {row['tcnCompleteLosses']} / {row['tcnPartialLosses']} | {row['newCompleteLosses']} | {row['newPartialLossesFromFullyCovered']} | {row['recoveredCompleteLosses']} | {row['removedFalsePositiveSeconds']:.2f} |")
    for case in cases:
        lines += ["", f"## {case['caseId']}: {case['kind']}", "",
                  f"`{case['recordingId']}` — group `{case['sourceGroup']}` — **{case['start']:.3f}–{case['end']:.3f}s**, duration {case['durationSeconds']:.3f}s.", ""]
        if case["kind"] == "new-loss":
            lines.append(f"Original rally index {case['truthIndex']} (zero-based): {case['lossType']}; linear coverage {case['linearCoverage']:.1%}, TCN coverage {case['tcnCoverage']:.1%}; **{case['additionalCoreSecondsLost']:.3f}s additional core time lost**.")
        else:
            lines.append(f"**{case['durationSeconds']:.3f}s removed false-positive export** under the existing label contract. Sampled frames provide context, not an independent claim about the full interval.")
        sheet = case["contactSheet"]
        if sheet["status"] == "complete":
            lines += ["", f"![Five sampled proxy frames for {case['caseId']}]({sheet['path']})", ""]
        else:
            lines += ["", f"Contact sheet unavailable: {sheet['reason']}", ""]
        lines += [f"Source proxy: `{case['videoPath']}`", "",
                  "| Track | Nearby time ranges (seconds) |", "|---|---|"]
        for label, key in (("Gold rallies", "nearbyTruth"), ("Linear raw predictions", "nearbyLinearRawPredictions"),
                           ("TCN raw predictions", "nearbyTcnRawPredictions"), ("Linear export", "nearbyLinearExport"),
                           ("TCN export", "nearbyTcnExport")):
            ranges = "; ".join(f"{row['start']:.3f}–{row['end']:.3f}" for row in case[key]) or "none in the context window"
            lines.append(f"| {label} | {ranges} |")
    lines += ["", "JSON preserves exact original-rally identities, tags, source/label provenance, both prediction timelines, image hashes, requested/decoded frame times, and input-result hashes.",
              "", f"`review-pack.json` SHA-256: `{digest(json_path)}`", ""]
    markdown = destination / "review.md"
    with markdown.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    checksums = {path.name: digest(path) for path in sorted(destination.iterdir()) if path.is_file()}
    with (destination / "checksums.json").open("x", encoding="utf-8") as handle:
        json.dump(checksums, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"review": str(markdown), "json": str(json_path), "sha256": digest(json_path),
                      "cases": len(cases), "contactSheets": sum(case['contactSheet']['status'] == 'complete' for case in cases)}))
    return pack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "review-pack-v1")
    args = parser.parse_args()
    build(args.root, args.output_dir)


if __name__ == "__main__":
    main()
