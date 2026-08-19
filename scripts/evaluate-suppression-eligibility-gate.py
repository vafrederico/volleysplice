#!/usr/bin/env python3
"""Compare retained production suppression policies with and without their eligibility gate."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAIN_SCRIPT = ROOT / "scripts" / "train-feedback-suppression-v3.py"
DEFAULT_EXPERIMENT_ROOT = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "feedback-suppression-v3-2026-08-16"
)
DEFAULT_SUPPRESSION_MODEL = (
    DEFAULT_EXPERIMENT_ROOT / "models" / "suppression-overlap-exclusion-retrained"
)
DEFAULT_OUTPUT = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "suppression-eligibility-gate-ablation-2026-08-18"
)
ROTATED_CAMERA_EXEMPTION = "PXL_20260816_164327879.mp4"
TARGET_PADDING = 2.0
PADDINGS = (0.0, 1.0, 2.0, 3.0)
EPSILON = 1e-6
UNGATED_THRESHOLD_SWEEP = (
    0.75,
    0.80,
    0.85,
    0.875,
    0.90,
    0.925,
    0.95,
    0.96,
    0.97,
    0.975,
    0.98,
    0.985,
    0.99,
    0.9925,
    0.995,
    0.9975,
    0.999,
    0.9995,
    0.9999,
)

POLICIES: dict[str, dict[str, Any]] = {
    "raw-connected": {
        "label": "Raw connected",
        "agreementPaddingSeconds": 0.0,
        "agreementJoinGapSecondsStrictlyLessThan": 0.0,
        "rawConnected": True,
    },
    "aggressive-intermediate": {
        "label": "Aggressive intermediate",
        "agreementPaddingSeconds": 1.5,
        "agreementJoinGapSecondsStrictlyLessThan": 0.5,
        "rawConnected": False,
    },
    "zero-non-exempt-misses": {
        "label": "Zero non-exempt misses",
        "agreementPaddingSeconds": 2.0,
        "agreementJoinGapSecondsStrictlyLessThan": 0.5,
        "rawConnected": False,
    },
}


def metric_summary(metric: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "paddingSecondsBeforeAndAfter": metric["paddingSecondsBeforeAndAfter"],
        "joinGapSecondsStrictlyLessThan": metric["joinGapSeconds"],
        "recordings": metric["recordings"],
        "core": {
            key: metric["core"][key]
            for key in (
                "precision",
                "recall",
                "f1",
                "intersectionSeconds",
                "predictedSeconds",
                "truthSeconds",
            )
        },
        "padded": {
            key: metric["padded"][key]
            for key in (
                "precision",
                "recall",
                "f1",
                "intersectionSeconds",
                "predictedSeconds",
                "truthSeconds",
            )
        },
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


def intervals_equal(
    left: Sequence[Any], right: Sequence[Any], tolerance: float = 1e-9
) -> bool:
    return len(left) == len(right) and all(
        abs(first.start - second.start) <= tolerance
        and abs(first.end - second.end) <= tolerance
        for first, second in zip(left, right, strict=True)
    )


def markdown_report(report: Mapping[str, Any]) -> str:
    variants = report["variants"]
    labels = report["variantLabels"]
    ungated = "ungated-full-production"
    decision = report["decision"]
    lines = [
        "# Suppression one-model eligibility gate ablation",
        "",
        "## Outcome",
        "",
        (
            "**Retain the one-model-only eligibility gate.** "
            if decision["recommendation"] == "retain-one-model-only-gate"
            else "**The one-model-only eligibility gate can be removed.** "
        )
        + decision["reason"],
        "",
        "Removing the gate makes all three retained aggressiveness policies emit the "
        "same output, because agreement padding and joining only define which production "
        "time is eligible for suppression. The ungated result is therefore reported once.",
        "",
        "The corrected suppression weights and held prior decoder are fixed. No model "
        "or decoder was selected in this ablation. Threshold feasibility is selected "
        "only on development; the all-evaluable post-hoc diagnostic, protected test, "
        "and held feedback are checked afterward as final safety guardrails.",
        "",
        "## Target-padding result (all evaluable recordings)",
        "",
        "| Variant | Core P | Core R | Core F1 | Padded P | Padded R | Padded F1 | P_pad | R_core | F1_padP_coreR | Export saved | Correct FPs | Non-exempt complete / partial | Lost core |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    baseline_export = variants["current-production"]["metrics"]["all-evaluable"][
        "2"
    ]["paddedModelExportSeconds"]
    for variant in report["displayVariantOrder"]:
        row = variants[variant]
        metric = row["metrics"]["all-evaluable"]["2"]
        audit = row["recallAudit"]["all-evaluable"]["nonExempt"]
        saved = baseline_export - metric["paddedModelExportSeconds"]
        lines.append(
            f"| {labels[variant]} | {metric['core']['precision']:.4f} | "
            f"{metric['core']['recall']:.4f} | {metric['core']['f1']:.4f} | "
            f"{metric['padded']['precision']:.4f} | {metric['padded']['recall']:.4f} | "
            f"{metric['padded']['f1']:.4f} | {metric['P_pad']:.4f} | "
            f"{metric['R_core']:.4f} | {metric['F1_padP_coreR']:.4f} | "
            f"{saved:.1f}s | {row['correctRemovalAudit']['count']} | "
            f"{audit['complete']} / {audit['partial']} | {audit['lostCoreSeconds']:.1f}s |"
        )

    lines.extend(
        [
            "",
            "## Recall guardrails by scope",
            "",
            "All rows use the declared 2-second product padding and positive joins strictly below 3 seconds.",
            "",
            "| Scope | Variant | Core recall | R_core | Complete / partial losses | Lost core |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    recall_scopes = (
        "selection:development",
        "all-evaluable",
        "environment:grass",
        "environment:indoor",
        "environment:beach",
        "provenance:export-feedback",
        "feedback-partition:held-out",
        "disclosure:protected-test",
    )
    recall_variants = (
        "current-production",
        "gated:raw-connected",
        "gated:aggressive-intermediate",
        "gated:zero-non-exempt-misses",
        ungated,
    )
    for scope in recall_scopes:
        if scope not in variants["current-production"]["metrics"]:
            continue
        for variant in recall_variants:
            metric = variants[variant]["metrics"][scope]["2"]
            audit = variants[variant]["recallAudit"][scope]["nonExempt"]
            lines.append(
                f"| {scope} | {labels[variant]} | {metric['core']['recall']:.4f} | "
                f"{metric['R_core']:.4f} | {audit['complete']} / {audit['partial']} | "
                f"{audit['lostCoreSeconds']:.1f}s |"
            )

    lines.extend(["", "## Four required padding cases", ""])
    for scope in ("selection:development", "all-evaluable"):
        lines.extend(
            [
                f"### {scope}",
                "",
                "| Variant | Padding | P_pad | R_core | F1_padP_coreR | Model export | Human export | Difference |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for variant in recall_variants:
            for padding in ("0", "1", "2", "3"):
                metric = variants[variant]["metrics"][scope][padding]
                lines.append(
                    f"| {labels[variant]} | {padding}s | {metric['P_pad']:.4f} | "
                    f"{metric['R_core']:.4f} | {metric['F1_padP_coreR']:.4f} | "
                    f"{metric['paddedModelExportSeconds']:.1f}s | "
                    f"{metric['paddedHumanExportSeconds']:.1f}s | "
                    f"{metric['paddedDurationDifferenceSeconds']:+.1f}s |"
                )
        lines.append("")

    lines.extend(
        [
            "## Can a stricter ungated threshold recover recall?",
            "",
            report["thresholdSweep"]["interpretation"],
            "",
            "The feasibility constraint is evaluated only on the predeclared development scope: Core Recall and `R_core` must both be at least the no-suppression baseline, while `P_pad` must improve.",
            "",
            "| Enter threshold | Core recall | P_pad | R_core | F1_padP_coreR | Export saved | Feasible |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    sweep_baseline = report["thresholdSweep"]["developmentBaseline"]
    for row in report["thresholdSweep"]["rows"]:
        metric = row["development"]
        lines.append(
            f"| {row['enterThreshold']:.4f} | {metric['core']['recall']:.4f} | "
            f"{metric['P_pad']:.4f} | {metric['R_core']:.4f} | "
            f"{metric['F1_padP_coreR']:.4f} | "
            f"{sweep_baseline['paddedModelExportSeconds'] - metric['paddedModelExportSeconds']:.1f}s | "
            f"{'yes' if row['feasible'] else 'no'} |"
        )

    feasible_rows = [
        row for row in report["thresholdSweep"]["rows"] if row["feasible"]
    ]
    if feasible_rows:
        lines.extend(
            [
                "",
                "Development-feasible thresholds are then checked, without re-selection, on the all-evaluable safety diagnostic, protected test, and held feedback:",
                "",
                "| Threshold | Scope | Baseline Core R | Candidate Core R | Baseline R_core | Candidate R_core | Complete / partial losses | Promotion pass |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in feasible_rows:
            for scope in report["thresholdSweep"]["promotionGuardrailScopes"]:
                baseline = variants["current-production"]["metrics"][scope]["2"]
                candidate = row["scopeMetrics"][scope]
                audit = row["recallAudit"][scope]["nonExempt"]
                scope_passed = (
                    candidate["core"]["recall"] + EPSILON
                    >= baseline["core"]["recall"]
                    and candidate["R_core"] + EPSILON >= baseline["R_core"]
                    and not audit["complete"]
                    and not audit["partial"]
                )
                lines.append(
                    f"| {row['enterThreshold']:.4f} | {scope} | "
                    f"{baseline['core']['recall']:.4f} | {candidate['core']['recall']:.4f} | "
                    f"{baseline['R_core']:.4f} | {candidate['R_core']:.4f} | "
                    f"{audit['complete']} / {audit['partial']} | "
                    f"{'yes' if scope_passed else 'no'} |"
                )

    lines.extend(
        [
            "",
            "## Ungated policy-collapse verification",
            "",
            f"- Raw connected equals ungated common output: `{report['ungatedPolicyCollapse']['raw-connected']}`.",
            f"- Aggressive intermediate equals ungated common output: `{report['ungatedPolicyCollapse']['aggressive-intermediate']}`.",
            f"- Zero non-exempt misses equals ungated common output: `{report['ungatedPolicyCollapse']['zero-non-exempt-misses']}`.",
            "",
            "## Method",
            "",
            "- Base: current previous-production union all-labels-v2 ensemble.",
            "- Specialist: corrected overlap-safe suppression artifact and held prior decoder.",
            "- Gated cut: `P intersect S intersect E_policy`, where `E_policy` is one-model-only production time.",
            "- Ungated cut: `P intersect S`; agreement grouping no longer affects eligibility.",
            "- Metrics pool durations across recordings, subtract ignored intervals, apply identical padding, and join only positive gaps strictly below 3 seconds.",
            f"- `{ROTATED_CAMERA_EXEMPTION}` is excluded only from the non-exempt miss guardrail; it remains in pooled metrics.",
            "- Full per-rally loss rows and every evaluated scope are in `report.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--suppression-model", type=Path, default=DEFAULT_SUPPRESSION_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    experiment_root = args.experiment_root.resolve()
    suppression_model_path = args.suppression_model.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    ns = runpy.run_path(str(TRAIN_SCRIPT))
    interval_type = ns["Interval"]
    item_type = ns["Item"]
    merge = ns["_merge_intervals"]
    subtract = ns["subtract_intervals"]
    intersect = ns["intersect_intervals"]
    duration_of = ns["_duration"]
    intersection_duration = ns["_intersection_duration"]
    pad_and_merge = ns["pad_and_merge_intervals"]
    metric_for = ns["metric_for"]

    def read_ranges(path: Path) -> tuple[Any, ...]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return tuple(
            interval_type(float(row["start"]), float(row["end"]))
            for row in payload["ranges"]
        )

    old_heads, v2_heads = ns["load_heads"]()
    suppression_model = ns["load_model"](suppression_model_path)
    suppression_metadata = json.loads(
        (suppression_model_path / "model.json").read_text(encoding="utf-8")
    )
    suppression_config = ns["HELD_PRODUCTION_SUPPRESSION_DECODER"]
    selected_config = suppression_model.training_summary[
        "productionEnsembleDecoderSelection"
    ]["selectedConfig"]
    if selected_config != suppression_config.to_dict():
        raise ValueError("corrected suppression artifact does not record the held decoder")

    training_ids = {
        recording.id
        for recording in ns["load_manifest"](ns["ALL_LABELS_MANIFEST"]).recordings
    }
    split = json.loads(
        (experiment_root / "split-policy.json").read_text(encoding="utf-8")
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

    predictions: dict[str, dict[str, tuple[Any, ...]]] = {
        "current-production": {},
        "ungated-full-production": {},
        **{f"gated:{policy}": {} for policy in POLICIES},
        **{f"ungated:{policy}": {} for policy in POLICIES},
    }
    applied_cuts: dict[str, dict[str, tuple[Any, ...]]] = {
        variant: {} for variant in predictions if variant != "current-production"
    }
    suppression_probabilities: dict[str, Any] = {}
    inference_root = experiment_root / "inference" / "current-production-ensemble"

    for index, entry in enumerate(entries, start=1):
        prepared = entry.prepared
        recording_id = prepared.recording.id
        baseline = read_ranges(inference_root / f"{recording_id}.json")
        old = ns["predict_heads"](prepared, old_heads)
        v2 = ns["predict_heads"](prepared, v2_heads)
        probabilities = suppression_model.predict(prepared.contextual_values)
        suppression_probabilities[recording_id] = probabilities
        suppression_scored, _ = ns["decode_probabilities"](
            prepared.sequence.times,
            probabilities,
            prepared.sequence.metadata.duration,
            suppression_config,
            4.0,
        )
        suppression = tuple(
            interval_type(float(row.start), float(row.end))
            for row in suppression_scored
        )
        full_cut = intersect(baseline, suppression)
        ungated_candidate = subtract(baseline, full_cut)
        predictions["current-production"][recording_id] = baseline
        predictions["ungated-full-production"][recording_id] = ungated_candidate
        applied_cuts["ungated-full-production"][recording_id] = full_cut

        for policy, config in POLICIES.items():
            support = ns["one_model_support"](
                old,
                v2,
                prepared.sequence.metadata.duration,
                agreement_padding=config["agreementPaddingSeconds"],
                agreement_join_gap=config[
                    "agreementJoinGapSecondsStrictlyLessThan"
                ],
                raw_connected=config["rawConnected"],
            )
            gated_cut = intersect(full_cut, support)
            gated_variant = f"gated:{policy}"
            ungated_variant = f"ungated:{policy}"
            predictions[gated_variant][recording_id] = subtract(baseline, gated_cut)
            applied_cuts[gated_variant][recording_id] = gated_cut
            predictions[ungated_variant][recording_id] = ungated_candidate
            applied_cuts[ungated_variant][recording_id] = full_cut
        print(f"Prepared {index}/{len(entries)}: {recording_id}", flush=True)

    groups = ns["group_items"](entries)
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for variant, variant_predictions in predictions.items():
        metrics[variant] = {
            scope: {
                f"{padding:g}": metric_summary(
                    metric_for(scoped_entries, variant_predictions, padding)
                )
                for padding in PADDINGS
            }
            for scope, scoped_entries in groups.items()
            if scoped_entries
        }

    development_entries = groups["selection:development"]
    development_baseline = metrics["current-production"][
        "selection:development"
    ]["2"]
    threshold_rows: list[dict[str, Any]] = []
    feasible_threshold_predictions: dict[str, dict[str, tuple[Any, ...]]] = {}
    for threshold in UNGATED_THRESHOLD_SWEEP:
        config = ns["DecoderConfig"](
            smoothing_seconds=1.0,
            enter_threshold=threshold,
            exit_threshold=max(0.05, threshold - 0.1),
            min_live_seconds=0.5,
            bridge_gap_seconds=0.5,
            short_event_min_seconds=0.25,
            short_event_threshold=max(0.9, threshold),
        )
        threshold_predictions: dict[str, tuple[Any, ...]] = {}
        for entry in entries:
            prepared = entry.prepared
            recording_id = prepared.recording.id
            decoded, _ = ns["decode_probabilities"](
                prepared.sequence.times,
                suppression_probabilities[recording_id],
                prepared.sequence.metadata.duration,
                config,
                4.0,
            )
            cuts = intersect(
                predictions["current-production"][recording_id],
                tuple(
                    interval_type(float(row.start), float(row.end))
                    for row in decoded
                ),
            )
            threshold_predictions[recording_id] = subtract(
                predictions["current-production"][recording_id], cuts
            )
        scope_metrics = {
            scope: metric_summary(
                metric_for(scoped_entries, threshold_predictions, TARGET_PADDING)
            )
            for scope, scoped_entries in groups.items()
            if scoped_entries
        }
        development_metric = scope_metrics["selection:development"]
        all_metric = scope_metrics["all-evaluable"]
        feasible = (
            development_metric["core"]["recall"] + EPSILON
            >= development_baseline["core"]["recall"]
            and development_metric["R_core"] + EPSILON
            >= development_baseline["R_core"]
            and development_metric["P_pad"]
            > development_baseline["P_pad"] + EPSILON
        )
        threshold_rows.append(
            {
                "enterThreshold": threshold,
                "exitThreshold": config.exit_threshold,
                "feasible": feasible,
                "development": development_metric,
                "allEvaluable": all_metric,
                "scopeMetrics": scope_metrics,
            }
        )
        if feasible:
            feasible_threshold_predictions[f"{threshold:g}"] = threshold_predictions

    entry_by_id = {entry.prepared.recording.id: entry for entry in entries}
    baseline_padded: dict[str, tuple[Any, ...]] = {}
    for entry in entries:
        prepared = entry.prepared
        baseline_padded[prepared.recording.id] = subtract(
            pad_and_merge(
                predictions["current-production"][prepared.recording.id],
                prepared.sequence.metadata.duration,
                TARGET_PADDING,
                ns["JOIN_GAP_SECONDS"],
            ),
            prepared.recording.ignored_intervals,
        )

    def build_loss_rows(
        variant_predictions: Mapping[str, Sequence[Any]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for entry in entries:
            prepared = entry.prepared
            recording_id = prepared.recording.id
            candidate_padded = subtract(
                pad_and_merge(
                    variant_predictions[recording_id],
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
                evaluable_core = subtract(
                    (core,), prepared.recording.ignored_intervals
                )
                evaluable_seconds = duration_of(evaluable_core)
                if evaluable_seconds <= EPSILON:
                    continue
                production_covered = intersection_duration(
                    evaluable_core, baseline_padded[recording_id]
                )
                candidate_covered = intersection_duration(
                    evaluable_core, candidate_padded
                )
                lost = max(0.0, production_covered - candidate_covered)
                if lost <= EPSILON:
                    continue
                rows.append(
                    {
                        "recordingId": recording_id,
                        "file": filename,
                        "environment": prepared.recording.environment,
                        "provenance": entry.provenance,
                        "feedbackPartition": entry.feedback_partition,
                        "rallyNumber": rally_number,
                        "start": core.start,
                        "end": core.end,
                        "duration": evaluable_seconds,
                        "productionCoveredSeconds": production_covered,
                        "candidateCoveredSeconds": candidate_covered,
                        "lostCoreSeconds": lost,
                        "completeMiss": candidate_covered <= EPSILON,
                        "nonExempt": filename != ROTATED_CAMERA_EXEMPTION,
                    }
                )
        rows.sort(
            key=lambda row: (
                not row["completeMiss"],
                -row["lostCoreSeconds"],
                row["file"],
                row["start"],
            )
        )
        return rows

    loss_rows: dict[str, list[dict[str, Any]]] = {
        variant: (
            []
            if variant == "current-production"
            else build_loss_rows(variant_predictions)
        )
        for variant, variant_predictions in predictions.items()
    }

    scope_ids = {
        scope: {entry.prepared.recording.id for entry in scoped_entries}
        for scope, scoped_entries in groups.items()
        if scoped_entries
    }

    def recall_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        selected = list(rows)
        return {
            "complete": sum(bool(row["completeMiss"]) for row in selected),
            "partial": sum(not bool(row["completeMiss"]) for row in selected),
            "lostCoreSeconds": sum(float(row["lostCoreSeconds"]) for row in selected),
            "affectedRallies": len(selected),
            "affectedRecordings": len({str(row["recordingId"]) for row in selected}),
        }

    recall_audits: dict[str, dict[str, Any]] = {}
    for variant, rows in loss_rows.items():
        recall_audits[variant] = {}
        for scope, recording_ids in scope_ids.items():
            scoped = [row for row in rows if row["recordingId"] in recording_ids]
            recall_audits[variant][scope] = {
                "all": recall_summary(scoped),
                "nonExempt": recall_summary(
                    row for row in scoped if row["nonExempt"]
                ),
            }

    threshold_row_by_key = {
        f"{row['enterThreshold']:g}": row for row in threshold_rows
    }
    for threshold_key, threshold_predictions in feasible_threshold_predictions.items():
        rows = build_loss_rows(threshold_predictions)
        audit: dict[str, Any] = {}
        for scope, recording_ids in scope_ids.items():
            scoped = [row for row in rows if row["recordingId"] in recording_ids]
            audit[scope] = {
                "all": recall_summary(scoped),
                "nonExempt": recall_summary(
                    row for row in scoped if row["nonExempt"]
                ),
            }
        threshold_row_by_key[threshold_key]["recallAudit"] = audit
        threshold_row_by_key[threshold_key]["lossRows"] = rows

    correct_removal_audits: dict[str, dict[str, Any]] = {}
    for variant, variant_predictions in predictions.items():
        count = 0
        raw_seconds = 0.0
        rows: list[dict[str, Any]] = []
        if variant != "current-production":
            for entry in entries:
                prepared = entry.prepared
                recording_id = prepared.recording.id
                baseline = subtract(
                    predictions["current-production"][recording_id],
                    prepared.recording.ignored_intervals,
                )
                candidate = subtract(
                    variant_predictions[recording_id],
                    prepared.recording.ignored_intervals,
                )
                human_export = subtract(
                    pad_and_merge(
                        prepared.recording.rallies,
                        prepared.sequence.metadata.duration,
                        TARGET_PADDING,
                        ns["JOIN_GAP_SECONDS"],
                    ),
                    prepared.recording.ignored_intervals,
                )
                for prediction in baseline:
                    if (
                        intersection_duration((prediction,), human_export) <= EPSILON
                        and intersection_duration((prediction,), candidate) <= EPSILON
                    ):
                        count += 1
                        raw_seconds += prediction.end - prediction.start
                        rows.append(
                            {
                                "recordingId": recording_id,
                                "file": prepared.recording.raw.get(
                                    "sourceFilename", prepared.recording.video.name
                                ),
                                "start": prediction.start,
                                "end": prediction.end,
                                "duration": prediction.end - prediction.start,
                            }
                        )
        correct_removal_audits[variant] = {
            "count": count,
            "rawSeconds": raw_seconds,
            "rows": rows,
        }

    collapse = {
        policy: all(
            intervals_equal(
                predictions[f"ungated:{policy}"][recording_id],
                predictions["ungated-full-production"][recording_id],
            )
            for recording_id in entry_by_id
        )
        for policy in POLICIES
    }

    def variant_payload(variant: str) -> dict[str, Any]:
        return {
            "metrics": metrics[variant],
            "recallAudit": recall_audits[variant],
            "correctRemovalAudit": correct_removal_audits[variant],
            "lossRows": loss_rows[variant],
        }

    gated_variants = [f"gated:{policy}" for policy in POLICIES]
    development_scope = "selection:development"
    ungated_development = metrics["ungated-full-production"][development_scope]["2"]
    ungated_non_exempt = recall_audits["ungated-full-production"][development_scope][
        "nonExempt"
    ]
    gate_failures: list[str] = []
    for gated_variant in gated_variants:
        gated_metric = metrics[gated_variant][development_scope]["2"]
        gated_audit = recall_audits[gated_variant][development_scope]["nonExempt"]
        if ungated_development["core"]["recall"] + EPSILON < gated_metric["core"]["recall"]:
            gate_failures.append(f"development core recall is lower than {gated_variant}")
        if ungated_development["R_core"] + EPSILON < gated_metric["R_core"]:
            gate_failures.append(f"development R_core is lower than {gated_variant}")
        if (
            ungated_non_exempt["complete"] > gated_audit["complete"]
            or ungated_non_exempt["partial"] > gated_audit["partial"]
        ):
            gate_failures.append(
                f"development non-exempt misses increase versus {gated_variant}"
            )
    retain_gate = bool(gate_failures)
    feasible_thresholds = [
        row["enterThreshold"] for row in threshold_rows if row["feasible"]
    ]
    promotion_scopes = (
        "all-evaluable",
        "disclosure:protected-test",
        "feedback-partition:held-out",
    )
    promotable_thresholds: list[float] = []
    threshold_promotion_failures: dict[str, list[str]] = {}
    for threshold in feasible_thresholds:
        key = f"{threshold:g}"
        row = threshold_row_by_key[key]
        failures: list[str] = []
        for scope in promotion_scopes:
            candidate = row["scopeMetrics"][scope]
            baseline = metrics["current-production"][scope]["2"]
            audit = row["recallAudit"][scope]["nonExempt"]
            if candidate["core"]["recall"] + EPSILON < baseline["core"]["recall"]:
                failures.append(f"{scope} Core Recall decreased")
            if candidate["R_core"] + EPSILON < baseline["R_core"]:
                failures.append(f"{scope} R_core decreased")
            if audit["complete"] or audit["partial"]:
                failures.append(f"{scope} introduced non-exempt rally losses")
        row["promotionGuardrailPassed"] = not failures
        row["promotionGuardrailFailures"] = failures
        threshold_promotion_failures[key] = failures
        if not failures:
            promotable_thresholds.append(threshold)
    if not feasible_thresholds:
        gate_failures.append(
            "no stricter ungated threshold preserves development Core Recall and "
            "R_core while improving P_pad"
        )
        retain_gate = True
    elif not promotable_thresholds:
        gate_failures.append(
            "development-feasible ungated thresholds fail post-selection "
            "safety guardrails"
        )
        retain_gate = True
    decision = {
        "selectionScope": development_scope,
        "protectedTestUsedForThresholdSelection": False,
        "protectedTestUsedAsFinalPromotionGuardrail": True,
        "allEvaluableUsedAsPostSelectionSafetyDiagnostic": True,
        "recallGuardrail": (
            "Ungated output must not reduce development Core Recall or R_core and "
            "must not increase non-exempt complete or partial rally losses relative "
            "to any retained gated policy."
        ),
        "passed": not retain_gate,
        "failures": sorted(set(gate_failures)),
        "recommendation": (
            "retain-one-model-only-gate"
            if retain_gate
            else "remove-one-model-only-gate"
        ),
        "reason": (
            "The ungated full-production veto fails the predeclared development "
            "recall guardrail. The only stricter threshold that passes development "
            "recall does not pass the untouched promotion guardrails."
            if retain_gate
            else "The ungated full-production veto preserves every predeclared "
            "development recall guardrail."
        ),
    }

    labels = {
        "current-production": "No suppression",
        "gated:raw-connected": "Raw connected · one-model gate",
        "gated:aggressive-intermediate": "Aggressive intermediate · one-model gate",
        "gated:zero-non-exempt-misses": "Zero non-exempt misses · one-model gate",
        "ungated-full-production": "Gate removed · all production",
    }
    display_order = [
        "current-production",
        "gated:raw-connected",
        "gated:aggressive-intermediate",
        "gated:zero-non-exempt-misses",
        "ungated-full-production",
    ]
    report = {
        "schemaVersion": 1,
        "experiment": "suppression-eligibility-gate-ablation-2026-08-18",
        "objective": (
            "Test whether the three retained suppression policies still require "
            "one-model-only production eligibility to preserve recall."
        ),
        "decision": decision,
        "model": {
            "productionBase": "current-production-ensemble",
            "suppressionPath": str(suppression_model_path),
            "suppressionArtifactSha256": suppression_model.artifact_sha256,
            "suppressionWeightsSha256": suppression_metadata["weightsSha256"],
            "decoder": suppression_config.to_dict(),
        },
        "evaluation": {
            "targetPaddingSecondsBeforeAndAfter": TARGET_PADDING,
            "requiredPaddingCases": list(PADDINGS),
            "joinGapSecondsStrictlyLessThan": ns["JOIN_GAP_SECONDS"],
            "ignoredIntervalsExcluded": True,
            "recordings": len(entries),
            "rotatedCameraMissExemption": ROTATED_CAMERA_EXEMPTION,
            "fitProjectIds": sorted(feedback_fit),
            "protectedTestIds": sorted(ns["PROTECTED_TEST_IDS"]),
            "developmentIds": sorted(ns["DEVELOPMENT_IDS"]),
        },
        "policies": POLICIES,
        "thresholdSweep": {
            "selectionScope": "selection:development",
            "thresholds": list(UNGATED_THRESHOLD_SWEEP),
            "frozenDecoderFields": {
                "smoothingSeconds": 1.0,
                "exitThreshold": "enterThreshold - 0.1",
                "minimumLiveSeconds": 0.5,
                "bridgeGapSeconds": 0.5,
                "shortEventMinimumSeconds": 0.25,
                "shortEventThreshold": "max(0.9, enterThreshold)",
            },
            "constraint": (
                "Development Core Recall >= no-suppression baseline, development "
                "R_core >= no-suppression baseline, and development P_pad > "
                "no-suppression baseline."
            ),
            "developmentBaseline": development_baseline,
            "feasibleThresholds": feasible_thresholds,
            "promotionGuardrailScopes": list(promotion_scopes),
            "promotableThresholds": promotable_thresholds,
            "promotionFailures": threshold_promotion_failures,
            "interpretation": (
                "No tested ungated threshold both passes development selection "
                "and preserves recall on the untouched promotion guardrails."
                if not promotable_thresholds
                else "At least one tested ungated threshold passes development "
                "selection and the untouched promotion guardrails."
            ),
            "developmentInterpretation": (
                "No tested ungated threshold preserves both development recall "
                "measures while improving padded precision."
                if not feasible_thresholds
                else "At least one tested ungated threshold satisfies the "
                "development recall and precision constraint."
            ),
            "rows": threshold_rows,
        },
        "ungatedPolicyCollapse": collapse,
        "variantLabels": labels,
        "displayVariantOrder": display_order,
        "variants": {variant: variant_payload(variant) for variant in display_order},
        "ungatedAliases": {
            policy: f"ungated:{policy}" for policy in POLICIES
        },
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(output / "report.json")
    print(output / "report.md")


if __name__ == "__main__":
    main()
