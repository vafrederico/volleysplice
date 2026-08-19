#!/usr/bin/env python3
"""Evaluate corrected v3 as a production voter and suppression support gates.

This is a fixed-model ablation. It does not tune a decoder or choose a policy on
protected data. Candidate 1 is the unsuppressed v3 voter. Candidate 2 is reported
as an internally-suppressed diagnostic output, but is not treated as a fourth
independent voter.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAIN_SCRIPT = ROOT / "scripts" / "train-feedback-suppression-v3.py"
CORRECTED_ROOT = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "feedback-suppression-v3-corrected-2026-08-18"
)
SUPPRESSION_MODEL = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "feedback-suppression-v3-2026-08-16/models/"
    "suppression-overlap-exclusion-retrained"
)
DEFAULT_OUTPUT = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "v3-production-ensemble-support-2026-08-18"
)
PADDINGS = (0.0, 1.0, 2.0, 3.0)
TARGET_PADDING = 2.0
EPSILON = 1e-6
ROTATED_EXEMPTION = "PXL_20260816_164327879.mp4"

POLICIES: dict[str, dict[str, Any]] = {
    "raw-connected": {
        "label": "Raw connected",
        "padding": 0.0,
        "join": 0.0,
        "raw": True,
    },
    "aggressive-intermediate": {
        "label": "Aggressive intermediate",
        "padding": 1.5,
        "join": 0.5,
        "raw": False,
    },
    "zero-non-exempt-misses": {
        "label": "Zero non-exempt misses",
        "padding": 2.0,
        "join": 0.5,
        "raw": False,
    },
}


def metric_summary(metric: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "paddingSecondsBeforeAndAfter": metric["paddingSecondsBeforeAndAfter"],
        "joinGapSecondsStrictlyLessThan": metric["joinGapSeconds"],
        "recordings": metric["recordings"],
        "core": dict(metric["core"]),
        "padded": dict(metric["padded"]),
        "P_pad": metric["P_pad"],
        "R_core": metric["R_core"],
        "F1_padP_coreR": metric["F1_padP_coreR"],
        "paddedModelExportSeconds": metric["paddedModelExportSeconds"],
        "paddedHumanExportSeconds": metric["paddedHumanExportSeconds"],
        "paddedDurationDifferenceSeconds": metric[
            "paddedDurationDifferenceSeconds"
        ],
        "outputCropCount": metric["outputCropCount"],
    }


def intervals_equal(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return len(left) == len(right) and all(
        abs(a.start - b.start) <= 1e-9 and abs(a.end - b.end) <= 1e-9
        for a, b in zip(left, right, strict=True)
    )


def support_by_count(
    groups: Mapping[str, Sequence[Any]],
    duration: float,
    *,
    interval_type: Any,
    merge: Any,
    intersect: Any,
    intersection_duration: Any,
    pad_and_merge: Any,
    agreement_padding: float,
    agreement_join: float,
    raw_connected: bool,
) -> dict[int, tuple[Any, ...]]:
    """Return production time whose agreement component has N source models."""
    source_rows = {key: tuple(value) for key, value in groups.items()}
    raw_union = merge(row for rows in source_rows.values() for row in rows)
    result: dict[int, tuple[Any, ...]] = {
        count: () for count in range(1, len(groups) + 1)
    }
    if raw_connected:
        tagged = sorted(
            (float(row.start), float(row.end), source)
            for source, rows in source_rows.items()
            for row in rows
        )
        components: list[tuple[float, float, set[str]]] = []
        for start, end, source in tagged:
            # Raw-connected follows the retained rule: touching is not overlap.
            if not components or start >= components[-1][1]:
                components.append((start, end, {source}))
                continue
            c_start, c_end, sources = components[-1]
            components[-1] = (c_start, max(c_end, end), sources | {source})
        for count in result:
            result[count] = merge(
                interval_type(start, end)
                for start, end, sources in components
                if len(sources) == count
            )
        return result

    components = pad_and_merge(
        raw_union, duration, agreement_padding, agreement_join
    )
    for count in result:
        selected = tuple(
            component
            for component in components
            if sum(
                intersection_duration((component,), rows) > 1e-9
                for rows in source_rows.values()
            )
            == count
        )
        result[count] = intersect(raw_union, selected)
    return result


def markdown_report(report: Mapping[str, Any]) -> str:
    variants = report["variants"]
    labels = report["variantLabels"]
    baseline_export = variants["old+v2"]["metrics"]["all-evaluable"]["2"][
        "paddedModelExportSeconds"
    ]
    triple_export = variants["old+v2+v3-candidate1"]["metrics"]["all-evaluable"]["2"][
        "paddedModelExportSeconds"
    ]
    lines = [
        "# v3 production-ensemble and support-count ablation",
        "",
        "## Answer",
        "",
        report["conclusion"],
        "",
        "The earlier corrected-v3 run covered `old + v3` for both candidate outputs. "
        "It did not cover `v2 + v3` or `old + v2 + v3`; those rows are added here.",
        "",
        "Candidate 1 is the corrected, unsuppressed three-head v3 output and is the "
        "third voter. Candidate 2 is candidate 1 after its own trained suppression "
        "head, so candidate-2 unions are diagnostic and are not counted as another vote.",
        "",
        "For the three-voter suppression rows, `exactly 1` means only 1-of-3 models "
        "supports the complete agreement-connected component. `1 or 2` means anything "
        "short of unanimous 3-of-3 support. `exactly 2` is also reported to isolate the "
        "incremental risk of extending eligibility from 1-of-3 to 2-of-3.",
        "",
        "All models, the corrected specialist, and its held production decoder are fixed. "
        "The primary metric is `F1_padP_coreR`; the target product padding is 2 seconds "
        "before/after and export gaps strictly below 3 seconds are joined.",
        "",
        "## Target-padding comparison — all evaluable recordings",
        "",
        "Losses are measured relative to rallies covered by current production. "
        "For suppression rows, saved time and correct removals are measured relative "
        "to the unsuppressed `old + v2 + v3` union.",
        "",
        "| Variant | P_pad | R_core | F1_padP_coreR | Core P | Core R | Core F1 | Padded P | Padded R | Padded F1 | Export Δ vs prod | Saved vs triple | Correct FP removals | Complete / partial non-exempt losses | Lost core |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in report["displayVariantOrder"]:
        row = variants[variant]["metrics"]["all-evaluable"]["2"]
        audit = variants[variant]["recallAudit"]["all-evaluable"]["nonExempt"]
        removal = variants[variant]["correctRemovalAudit"]
        is_suppression = variant.startswith("triple-suppression:")
        saved = triple_export - row["paddedModelExportSeconds"] if is_suppression else 0.0
        lines.append(
            f"| {labels[variant]} | {row['P_pad']:.4f} | {row['R_core']:.4f} | "
            f"{row['F1_padP_coreR']:.4f} | {row['core']['precision']:.4f} | "
            f"{row['core']['recall']:.4f} | {row['core']['f1']:.4f} | "
            f"{row['padded']['precision']:.4f} | {row['padded']['recall']:.4f} | "
            f"{row['padded']['f1']:.4f} | "
            f"{row['paddedModelExportSeconds'] - baseline_export:+.1f}s | "
            f"{saved:.1f}s | {removal['count']} | "
            f"{audit['complete']} / {audit['partial']} | {audit['lostCoreSeconds']:.1f}s |"
        )

    lines.extend([
        "",
        "## Development selection scope at target padding",
        "",
        "This is the only scope appropriate for ranking alternatives. Other scopes below are post-hoc disclosure and guardrails.",
        "",
        "| Variant | P_pad | R_core | F1_padP_coreR | Export s |",
        "|---|---:|---:|---:|---:|",
    ])
    for variant in report["displayVariantOrder"]:
        row = variants[variant]["metrics"]["selection:development"]["2"]
        lines.append(
            f"| {labels[variant]} | {row['P_pad']:.4f} | {row['R_core']:.4f} | "
            f"{row['F1_padP_coreR']:.4f} | {row['paddedModelExportSeconds']:.1f} |"
        )

    lines.extend(["", "## Required padding sensitivity", ""])
    for scope in ("selection:development", "all-evaluable"):
        lines.extend([
            f"### {scope}",
            "",
            "| Variant | Pad | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for variant in report["displayVariantOrder"]:
            for padding in ("0", "1", "2", "3"):
                row = variants[variant]["metrics"][scope][padding]
                lines.append(
                    f"| {labels[variant]} | {padding}s | {row['P_pad']:.4f} | "
                    f"{row['R_core']:.4f} | {row['F1_padP_coreR']:.4f} | "
                    f"{row['paddedModelExportSeconds']:.1f} | "
                    f"{row['paddedHumanExportSeconds']:.1f} | "
                    f"{row['paddedDurationDifferenceSeconds']:+.1f} |"
                )

    lines.extend([
        "",
        "## Recall guardrails by scope",
        "",
        "| Scope | Variant | Core recall | R_core | Complete / partial non-exempt losses | Lost core |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for scope in report["guardrailScopes"]:
        for variant in report["suppressionVariantOrder"]:
            if scope not in variants[variant]["metrics"]:
                continue
            row = variants[variant]["metrics"][scope]["2"]
            audit = variants[variant]["recallAudit"][scope]["nonExempt"]
            lines.append(
                f"| {scope} | {labels[variant]} | {row['core']['recall']:.4f} | "
                f"{row['R_core']:.4f} | {audit['complete']} / {audit['partial']} | "
                f"{audit['lostCoreSeconds']:.1f}s |"
            )
    lines.extend([
        "",
        "## Exact loss rows",
        "",
        "The machine-readable report contains every affected rally under "
        "`variants.<id>.lossRows`, including file, range, duration, support rule, "
        "complete/partial status, and lost core seconds.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--corrected-root", type=Path, default=CORRECTED_ROOT)
    parser.add_argument("--suppression-model", type=Path, default=SUPPRESSION_MODEL)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    ns = runpy.run_path(str(TRAIN_SCRIPT))
    interval_type = ns["Interval"]
    item_type = ns["Item"]
    merge = ns["_merge_intervals"]
    subtract = ns["subtract_intervals"]
    intersect = ns["intersect_intervals"]
    intersection_duration = ns["_intersection_duration"]
    duration_of = ns["_duration"]
    pad_and_merge = ns["pad_and_merge_intervals"]

    def read_ranges(variant: str, recording_id: str) -> tuple[Any, ...]:
        payload = json.loads(
            (args.corrected_root / "inference" / variant / f"{recording_id}.json")
            .read_text(encoding="utf-8")
        )
        return tuple(
            interval_type(float(row["start"]), float(row["end"]))
            for row in payload["ranges"]
        )

    old_heads, v2_heads = ns["load_heads"]()
    suppression_model = ns["load_model"](args.suppression_model)
    suppression_config = ns["HELD_PRODUCTION_SUPPRESSION_DECODER"]
    if suppression_model.training_summary[
        "productionEnsembleDecoderSelection"
    ]["selectedConfig"] != suppression_config.to_dict():
        raise ValueError("suppression artifact does not record the held decoder")

    training_ids = {
        recording.id
        for recording in ns["load_manifest"](ns["ALL_LABELS_MANIFEST"]).recordings
    }
    split = json.loads(
        (args.corrected_root / "split-policy.json").read_text(encoding="utf-8")
    )
    feedback_fit = set(split["fitProjectIds"])
    entries: list[Any] = []
    for recording in sorted(ns["load_labeled_records"]().values(), key=lambda row: row.id):
        prepared = ns["prepare_recording"](
            recording, v2_heads.rally.feature_config, ns["FEATURE_CACHE"]
        )
        provenance = (
            "training-dataset"
            if recording.id in training_ids
            else "evaluation-validation-test-only"
        )
        entries.append(item_type(prepared, provenance, None, True))
    for path in sorted(ns["FEEDBACK_ROOT"].glob("*/bundle.json")):
        prepared, _ = ns["feedback_prepared"](path, v2_heads.rally.feature_config)
        partition = "fit" if prepared.recording.id in feedback_fit else "held-out"
        entries.append(item_type(prepared, "export-feedback", partition, True))

    variants: dict[str, dict[str, tuple[Any, ...]]] = {
        "old+v2": {},
        "v3-candidate1": {},
        "v3-candidate2": {},
        "old+v3-candidate1": {},
        "old+v3-candidate2": {},
        "v2+v3-candidate1": {},
        "v2+v3-candidate2": {},
        "old+v2+v3-candidate1": {},
        "old+v2+v3-candidate2": {},
    }
    for policy in POLICIES:
        for gate in ("exact1", "at-most2", "exact2"):
            variants[f"triple-suppression:{gate}:{policy}"] = {}

    support_durations: dict[str, dict[str, float]] = {
        f"{gate}:{policy}": {}
        for policy in POLICIES
        for gate in ("exact1", "at-most2", "exact2")
    }
    for index, entry in enumerate(entries, start=1):
        prepared = entry.prepared
        recording_id = prepared.recording.id
        old = ns["predict_heads"](prepared, old_heads)
        v2 = ns["predict_heads"](prepared, v2_heads)
        current = merge((*old, *v2))
        stored_current = read_ranges("current-production-ensemble", recording_id)
        if not intervals_equal(current, stored_current):
            raise ValueError(f"production reproduction mismatch: {recording_id}")
        v3 = read_ranges("v3-candidate1-three-head", recording_id)
        v3_suppressed = read_ranges("v3-candidate2-four-head", recording_id)
        old_v3 = merge((*old, *v3))
        if not intervals_equal(
            old_v3, read_ranges("old-plus-v3-candidate1", recording_id)
        ):
            raise ValueError(f"old+v3 reproduction mismatch: {recording_id}")

        triple = merge((*old, *v2, *v3))
        probabilities = suppression_model.predict(prepared.contextual_values)
        decoded, _ = ns["decode_probabilities"](
            prepared.sequence.times,
            probabilities,
            prepared.sequence.metadata.duration,
            suppression_config,
            4.0,
        )
        suppression = tuple(
            interval_type(float(row.start), float(row.end)) for row in decoded
        )
        full_cut = intersect(triple, suppression)

        variants["old+v2"][recording_id] = current
        variants["v3-candidate1"][recording_id] = v3
        variants["v3-candidate2"][recording_id] = v3_suppressed
        variants["old+v3-candidate1"][recording_id] = old_v3
        variants["old+v3-candidate2"][recording_id] = merge((*old, *v3_suppressed))
        variants["v2+v3-candidate1"][recording_id] = merge((*v2, *v3))
        variants["v2+v3-candidate2"][recording_id] = merge((*v2, *v3_suppressed))
        variants["old+v2+v3-candidate1"][recording_id] = triple
        variants["old+v2+v3-candidate2"][recording_id] = merge(
            (*old, *v2, *v3_suppressed)
        )

        for policy, config in POLICIES.items():
            by_count = support_by_count(
                {"old": old, "v2": v2, "v3": v3},
                prepared.sequence.metadata.duration,
                interval_type=interval_type,
                merge=merge,
                intersect=intersect,
                intersection_duration=intersection_duration,
                pad_and_merge=pad_and_merge,
                agreement_padding=config["padding"],
                agreement_join=config["join"],
                raw_connected=config["raw"],
            )
            gates = {
                "exact1": by_count[1],
                "at-most2": merge((*by_count[1], *by_count[2])),
                "exact2": by_count[2],
            }
            for gate, support in gates.items():
                cut = intersect(full_cut, support)
                variants[f"triple-suppression:{gate}:{policy}"][recording_id] = (
                    subtract(triple, cut)
                )
                support_durations[f"{gate}:{policy}"][recording_id] = duration_of(
                    support
                )
        print(f"Prepared {index}/{len(entries)}: {recording_id}", flush=True)

    groups = ns["group_items"](entries)
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for variant, predictions in variants.items():
        metrics[variant] = {
            scope: {
                f"{padding:g}": metric_summary(
                    ns["metric_for"](scoped_entries, predictions, padding)
                )
                for padding in PADDINGS
            }
            for scope, scoped_entries in groups.items()
            if scoped_entries
        }

    baseline_padded: dict[str, tuple[Any, ...]] = {}
    for entry in entries:
        prepared = entry.prepared
        recording_id = prepared.recording.id
        baseline_padded[recording_id] = subtract(
            pad_and_merge(
                variants["old+v2"][recording_id],
                prepared.sequence.metadata.duration,
                TARGET_PADDING,
                ns["JOIN_GAP_SECONDS"],
            ),
            prepared.recording.ignored_intervals,
        )

    def loss_rows(predictions: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for entry in entries:
            prepared = entry.prepared
            recording_id = prepared.recording.id
            candidate = subtract(
                pad_and_merge(
                    predictions[recording_id],
                    prepared.sequence.metadata.duration,
                    TARGET_PADDING,
                    ns["JOIN_GAP_SECONDS"],
                ),
                prepared.recording.ignored_intervals,
            )
            filename = prepared.recording.raw.get(
                "sourceFilename", prepared.recording.video.name
            )
            for rally_number, core in enumerate(prepared.recording.rallies, start=1):
                evaluable = subtract((core,), prepared.recording.ignored_intervals)
                if duration_of(evaluable) <= EPSILON:
                    continue
                before = intersection_duration(
                    evaluable, baseline_padded[recording_id]
                )
                after = intersection_duration(evaluable, candidate)
                lost = max(0.0, before - after)
                if lost <= EPSILON:
                    continue
                rows.append({
                    "recordingId": recording_id,
                    "file": filename,
                    "environment": prepared.recording.environment,
                    "provenance": entry.provenance,
                    "feedbackPartition": entry.feedback_partition,
                    "rallyNumber": rally_number,
                    "start": core.start,
                    "end": core.end,
                    "duration": core.end - core.start,
                    "productionCoveredSeconds": before,
                    "candidateCoveredSeconds": after,
                    "lostCoreSeconds": lost,
                    "completeMiss": after <= EPSILON,
                    "nonExempt": filename != ROTATED_EXEMPTION,
                })
        return sorted(
            rows,
            key=lambda row: (
                not row["completeMiss"], -row["lostCoreSeconds"], row["file"], row["start"]
            ),
        )

    scope_ids = {
        scope: {entry.prepared.recording.id for entry in scoped}
        for scope, scoped in groups.items()
        if scoped
    }

    def audit(rows: Sequence[Mapping[str, Any]], ids: set[str], non_exempt: bool) -> dict[str, Any]:
        selected = [
            row for row in rows
            if row["recordingId"] in ids and (not non_exempt or row["nonExempt"])
        ]
        return {
            "complete": sum(bool(row["completeMiss"]) for row in selected),
            "partial": sum(not bool(row["completeMiss"]) for row in selected),
            "affectedRallies": len(selected),
            "affectedRecordings": len({row["recordingId"] for row in selected}),
            "lostCoreSeconds": sum(float(row["lostCoreSeconds"]) for row in selected),
        }

    variant_payloads: dict[str, Any] = {}
    for variant, predictions in variants.items():
        losses = [] if variant == "old+v2" else loss_rows(predictions)
        correct_rows: list[dict[str, Any]] = []
        if variant.startswith("triple-suppression:"):
            for entry in entries:
                prepared = entry.prepared
                recording_id = prepared.recording.id
                human_export = subtract(
                    pad_and_merge(
                        prepared.recording.rallies,
                        prepared.sequence.metadata.duration,
                        TARGET_PADDING,
                        ns["JOIN_GAP_SECONDS"],
                    ),
                    prepared.recording.ignored_intervals,
                )
                candidate = subtract(
                    predictions[recording_id], prepared.recording.ignored_intervals
                )
                triple = subtract(
                    variants["old+v2+v3-candidate1"][recording_id],
                    prepared.recording.ignored_intervals,
                )
                for prediction in triple:
                    if (
                        intersection_duration((prediction,), human_export) <= EPSILON
                        and intersection_duration((prediction,), candidate) <= EPSILON
                    ):
                        correct_rows.append({
                            "recordingId": recording_id,
                            "file": prepared.recording.raw.get(
                                "sourceFilename", prepared.recording.video.name
                            ),
                            "start": prediction.start,
                            "end": prediction.end,
                            "duration": prediction.end - prediction.start,
                        })
        variant_payloads[variant] = {
            "metrics": metrics[variant],
            "lossRows": losses,
            "correctRemovalAudit": {
                "count": len(correct_rows),
                "rawSeconds": sum(row["duration"] for row in correct_rows),
                "rows": correct_rows,
            },
            "recallAudit": {
                scope: {
                    "all": audit(losses, ids, False),
                    "nonExempt": audit(losses, ids, True),
                }
                for scope, ids in scope_ids.items()
            },
        }

    labels = {
        "old+v2": "Current production (old + v2)",
        "v3-candidate1": "v3 candidate 1 alone",
        "v3-candidate2": "v3 candidate 2 alone (internally suppressed)",
        "old+v3-candidate1": "Old + v3 candidate 1",
        "old+v3-candidate2": "Old + v3 candidate 2",
        "v2+v3-candidate1": "v2 + v3 candidate 1",
        "v2+v3-candidate2": "v2 + v3 candidate 2",
        "old+v2+v3-candidate1": "Old + v2 + v3 candidate 1",
        "old+v2+v3-candidate2": "Old + v2 + v3 candidate 2",
    }
    for policy, config in POLICIES.items():
        for gate, gate_label in (
            ("exact1", "suppress exactly 1-of-3"),
            ("at-most2", "suppress 1-or-2-of-3"),
            ("exact2", "suppress exactly 2-of-3 only"),
        ):
            labels[f"triple-suppression:{gate}:{policy}"] = (
                f"Triple + specialist: {gate_label}; {config['label']}"
            )

    base_order = [
        "old+v2",
        "v3-candidate1",
        "v3-candidate2",
        "old+v3-candidate1",
        "old+v3-candidate2",
        "v2+v3-candidate1",
        "v2+v3-candidate2",
        "old+v2+v3-candidate1",
        "old+v2+v3-candidate2",
    ]
    suppression_order = [
        f"triple-suppression:{gate}:{policy}"
        for policy in POLICIES
        for gate in ("exact1", "at-most2", "exact2")
    ]
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "experiment": "v3-production-ensemble-support-2026-08-18",
        "createdAt": datetime.now(UTC).isoformat(),
        "rankingMetric": "F1_padP_coreR",
        "targetProductPaddingSecondsBeforeAndAfter": TARGET_PADDING,
        "requiredPaddingCases": list(PADDINGS),
        "joinGapSecondsStrictlyLessThan": ns["JOIN_GAP_SECONDS"],
        "fixedModels": {
            "v3Candidate1": "model-3a3738bffa6b",
            "v3Candidate2": "model-6dc36c67401a",
            "productionSuppressionArtifactSha256": suppression_model.artifact_sha256,
            "productionSuppressionDecoder": suppression_config.to_dict(),
        },
        "supportRule": {
            "voters": ["old", "all-labels-v2", "v3-candidate1"],
            "connectedComponentSemantics": "Any positive-duration overlap connects complete model intervals, including non-overlapping heads and tails; configured agreement padding and strict short-gap joining are then applied before counting distinct source models.",
            "exact1": "eligible only when exactly one of three models supports the component",
            "at-most2": "eligible when one or two of three models supports the component; unanimous three-model components are protected",
            "exact2": "diagnostic: eligible only when exactly two of three models supports the component",
            "policies": POLICIES,
        },
        "scopeRoles": {
            "selection:development": "selection/ranking",
            "disclosure:protected-test": "post-selection disclosure only",
            "all-evaluable": "post-hoc aggregate guardrail, not selection",
        },
        "variantLabels": labels,
        "displayVariantOrder": [*base_order, *suppression_order],
        "suppressionVariantOrder": ["old+v2+v3-candidate1", *suppression_order],
        "guardrailScopes": [
            "selection:development", "all-evaluable", "environment:grass",
            "environment:indoor", "environment:beach", "provenance:export-feedback",
            "feedback-partition:held-out", "disclosure:protected-test",
        ],
        "variants": variant_payloads,
        "supportEligibleRawSecondsByRecording": support_durations,
    }

    # Conclusion is computed from development ranking plus all-scope recall guardrails.
    dev_baseline = metrics["old+v2"]["selection:development"]["2"]
    safe: list[tuple[float, str]] = []
    for variant in suppression_order:
        dev = metrics[variant]["selection:development"]["2"]
        all_audit = variant_payloads[variant]["recallAudit"]["all-evaluable"]["nonExempt"]
        if (
            dev["core"]["recall"] + EPSILON >= dev_baseline["core"]["recall"]
            and dev["R_core"] + EPSILON >= dev_baseline["R_core"]
            and all_audit["complete"] == 0
            and all_audit["partial"] == 0
        ):
            safe.append((dev["F1_padP_coreR"], variant))
    production_all = metrics["old+v2"]["all-evaluable"]["2"]
    safe_and_better = [
        item for item in safe
        if metrics[item[1]]["all-evaluable"]["2"]["F1_padP_coreR"]
        > production_all["F1_padP_coreR"] + EPSILON
        and metrics[item[1]]["all-evaluable"]["2"]["paddedModelExportSeconds"]
        < production_all["paddedModelExportSeconds"] - EPSILON
    ]
    if safe_and_better:
        safe_and_better.sort(reverse=True)
        winner = safe_and_better[0][1]
        report["conclusion"] = (
            f"`{winner}` preserves the fixed recall guardrails while improving "
            "all-evaluable F1_padP_coreR and reducing export time versus current "
            "production. This is new research evidence only and does not reverse the "
            "recorded decision to reject v3 without a separate product decision."
        )
    elif safe:
        safe.sort(reverse=True)
        winner = safe[0][1]
        report["conclusion"] = (
            f"`{winner}` is the highest-development-F1 three-voter suppression option "
            "with no non-exempt complete or partial losses, but it does not improve "
            "both all-evaluable F1_padP_coreR and export duration versus current "
            "production. Allowing suppression on 2-of-3 support produces recall "
            "losses. The v3 rejection decision remains supported."
        )
    else:
        report["conclusion"] = (
            "No three-voter suppression support rule simultaneously preserved the "
            "development Core Recall/R_core baseline and avoided every non-exempt "
            "complete or partial loss across all evaluable recordings. The recorded "
            "decision to reject the v3 candidates remains supported."
        )

    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(output / "report.json")
    print(output / "report.md")


if __name__ == "__main__":
    main()
