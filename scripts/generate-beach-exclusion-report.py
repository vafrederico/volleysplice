#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis.model import load_model
from analysis.schema import load_manifest


FULL = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
V0 = Path("/mnt/freenas/volleycut/v0-2026-08-09")
NB = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12")
NBV0 = Path("/mnt/freenas/volleycut/v0-2026-08-09-no-beach-2026-08-12")
OLD_REPORTS = FULL / "reports"
NEW_REPORTS = NB / "reports"
METRICS = (
    "eventF1",
    "timeIoU",
    "liveTimeRecall",
    "liveTimePrecision",
    "predictedRallies",
    "trueRallies",
    "matchedRallies",
)


def metric(payload: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    value: Any = payload
    for key in path:
        value = value[key]
    return {key: value.get(key) for key in METRICS}


def load_metric(root: Path, filename: str, path: tuple[str, ...]) -> dict[str, Any]:
    return metric(json.loads((root / filename).read_text()), path)


def spec(
    name: str,
    family: str,
    old_val: tuple[str, tuple[str, ...]],
    old_test: tuple[str, tuple[str, ...]],
    new_val: tuple[str, tuple[str, ...]],
    new_test: tuple[str, tuple[str, ...]],
    *,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "oldVal": old_val,
        "oldTest": old_test,
        "newVal": new_val,
        "newTest": new_test,
        **({"note": note} if note else {}),
    }


def build_specs() -> list[dict[str, Any]]:
    standard = [
        ("pilot-baseline-v0", "pilot-baseline-v0"),
        ("pilot-percentile-v1", "pilot-percentile-v1"),
        ("full-percentile-v1", "full-percentile-v1"),
    ]
    result = [
        spec(
            name,
            "rally-live",
            (f"{name}-validation.json", ("aggregate",)),
            (f"{name}-test.json", ("aggregate",)),
            (f"{name}-evaluation-validation.json", ("aggregate",)),
            (f"{name}-evaluation-test.json", ("aggregate",)),
        )
        for name, _ in standard
    ]
    result += [
        spec(
            "full-audiovisual-v2",
            "rally-live",
            ("full-audiovisual-v2-original-evaluation-validation.json", ("aggregate",)),
            ("full-audiovisual-v2-original-evaluation-test.json", ("aggregate",)),
            ("full-audiovisual-v2-evaluation-validation.json", ("aggregate",)),
            ("full-audiovisual-v2-evaluation-test.json", ("aggregate",)),
        ),
        spec(
            "full-audiovisual-v2-final",
            "rally-live",
            ("full-audiovisual-v2-final-test.json", ("model", "validation")),
            ("full-audiovisual-v2-final-test.json", ("test", "aggregate")),
            ("full-audiovisual-v2-final-evaluation-validation.json", ("aggregate",)),
            ("full-audiovisual-v2-final-evaluation-test.json", ("aggregate",)),
        ),
        spec(
            "full-audiovisual-audio-normalized-v3",
            "rally-live",
            ("full-audiovisual-audio-normalized-v3-final-validation.json", ("aggregate",)),
            ("full-audiovisual-audio-normalized-v3-final-retrospective-test.json", ("aggregate",)),
            ("full-audiovisual-audio-normalized-v3-evaluation-validation.json", ("aggregate",)),
            ("full-audiovisual-audio-normalized-v3-evaluation-test.json", ("aggregate",)),
        ),
    ]

    serve_names = (
        "serve-specialist-v1",
        "serve-specialist-v2",
        "serve-specialist-audiovisual-v3",
        "serve-specialist-audiovisual-v4",
        "serve-specialist-audio-normalized-v5",
        "serve-specialist-audio-normalized-v6-no-legacy",
        "serve-specialist-audio-normalized-v7-new-only",
    )
    for name in serve_names:
        result.append(
            spec(
                name,
                "serve-contact+composition",
                (f"{name}-validation.json", ("composed", "aggregate")),
                (f"{name}-test.json", ("composed", "aggregate")),
                (f"{name}-evaluation-validation.json", ("composed", "aggregate")),
                (f"{name}-evaluation-test.json", ("composed", "aggregate")),
            )
        )

    stack_old_val = ("full-audiovisual-serve-prob-stack-v1-final-validation.json",)
    stack_old_test = ("full-audiovisual-serve-prob-stack-v1-final-retrospective-test.json",)
    stack_new_val = ("full-audiovisual-serve-prob-stack-v1-control-evaluation-validation.json",)
    stack_new_test = ("full-audiovisual-serve-prob-stack-v1-control-evaluation-test.json",)
    result += [
        spec(
            "full-audiovisual-serve-prob-stack-v1-control",
            "stacked-control",
            (stack_old_val[0], ("retrainedControl", "aggregate")),
            (stack_old_test[0], ("retrainedControl", "aggregate")),
            (stack_new_val[0], ("retrainedControl", "aggregate")),
            (stack_new_test[0], ("retrainedControl", "aggregate")),
        ),
        spec(
            "full-audiovisual-serve-prob-stack-v1-exploratory",
            "stacked-rally",
            (stack_old_val[0], ("stackedRally", "aggregate")),
            (stack_old_test[0], ("stackedRally", "aggregate")),
            ("full-audiovisual-serve-prob-stack-v1-exploratory-evaluation-validation.json", ("stackedRally", "aggregate")),
            ("full-audiovisual-serve-prob-stack-v1-exploratory-evaluation-test.json", ("stackedRally", "aggregate")),
        ),
        spec(
            "full-audiovisual-serve-peak-window-v1-exploratory",
            "peak-window",
            ("serve-evidence-v1-final-validation.json", ("peakWindowRally", "aggregate")),
            ("serve-evidence-v1-final-retrospective-test.json", ("peakWindowRally", "aggregate")),
            ("full-audiovisual-serve-peak-window-v1-exploratory-evaluation-validation.json", ("peakWindowRally", "aggregate")),
            ("full-audiovisual-serve-peak-window-v1-exploratory-evaluation-test.json", ("peakWindowRally", "aggregate")),
        ),
    ]

    dead_ball_names = (
        "dead-ball-specialist-audiovisual-v1",
        "dead-ball-specialist-audiovisual-v2",
        "dead-ball-specialist-audio-normalized-v2-legacy-only",
        "dead-ball-specialist-audio-normalized-v3",
        "dead-ball-specialist-audio-normalized-v4-no-legacy",
        "dead-ball-specialist-audio-normalized-v5-new-only",
    )
    for name in dead_ball_names:
        old_test = (
            (
                "dead-ball-specialist-audiovisual-v2-retrospective-test.json",
                ("v4Composition", "aggregate"),
            )
            if name == "dead-ball-specialist-audiovisual-v1"
            else (
                "dead-ball-specialist-audiovisual-v2-retrospective-test.json",
                ("selected", "aggregate"),
            )
            if name == "dead-ball-specialist-audiovisual-v2"
            else (f"{name}-final-retrospective-test.json", ("selected", "aggregate"))
        )
        note = None
        if name == "dead-ball-specialist-audiovisual-v1":
            note = "Original v1 selected the exact v4 no-op; its test metric is the v4 composition from the v2 retrospective report because the old v1 report was validation-only."
        result.append(
            spec(
                name,
                "dead-ball",
                (f"{name}-validation.json", ("selected", "aggregate")),
                old_test,
                (f"{name}-validation.json", ("selected", "aggregate")),
                (f"{name}-evaluation-test.json", ("selected", "aggregate")),
                note=note,
            )
        )

    state_names = (
        "dead-state-global-audio-normalized-v1-full-final",
        "dead-state-global-audio-normalized-v2-full-final",
        "dead-state-transition-audio-normalized-v0-legacy-only",
        "dead-state-transition-audio-normalized-v1-full",
        "dead-state-transition-audio-normalized-v2-no-legacy",
        "dead-state-transition-audio-normalized-v3-legacy-only-final",
        "dead-state-transition-audio-normalized-v4-full-final",
        "dead-state-transition-audio-normalized-v5-no-legacy-final",
    )
    for name in state_names:
        result.append(
            spec(
                name,
                "dead-state",
                (f"{name}-validation.json", ("selected", "aggregate")),
                (f"{name}-retrospective-test.json", ("selected", "aggregate")),
                (f"{name}-validation.json", ("selected", "aggregate")),
                (f"{name}-evaluation-test.json", ("selected", "aggregate")),
            )
        )

    result.append(
        spec(
            "audiovisual-v2-targeted-pruned-diagnostic",
            "targeted-pruning",
            ("audiovisual-v2-targeted-pruned-regression-test.json", ("validation", "metrics")),
            ("audiovisual-v2-targeted-pruned-regression-test.json", ("test", "aggregate")),
            ("audiovisual-v2-targeted-pruned-diagnostic-validation.json", ("aggregate",)),
            ("audiovisual-v2-targeted-pruned-diagnostic-test.json", ("aggregate",)),
        )
    )
    result += [
        spec(
            "real-rally-v0",
            "legacy-v0",
            ("real-rally-v0-original-evaluation-validation.json", ("aggregate",)),
            ("test-evaluation.json", ("aggregate",)),
            ("real-rally-v0-evaluation-validation.json", ("aggregate",)),
            ("real-rally-v0-evaluation-test.json", ("aggregate",)),
        ),
        spec(
            "real-rally-v0-no-flow",
            "legacy-v0",
            ("real-rally-v0-no-flow-original-evaluation-validation.json", ("aggregate",)),
            ("test-evaluation-no-flow.json", ("aggregate",)),
            ("real-rally-v0-no-flow-evaluation-validation.json", ("aggregate",)),
            ("real-rally-v0-no-flow-evaluation-test.json", ("aggregate",)),
        ),
    ]
    if len(result) != 33:
        raise RuntimeError(f"expected 33 model specs, got {len(result)}")
    return result


def main() -> int:
    specs = build_specs()
    names = {item["name"] for item in specs}
    full_manifest = load_manifest(FULL / "manifests/full-gold-v1.json")
    recording_meta = {
        row.id: {
            "environment": row.environment,
            "split": row.split,
            "video": str(row.video),
            "contentSha256": row.content_sha256,
        }
        for row in full_manifest.recordings
    }

    models: list[dict[str, Any]] = []
    new_hash_to_name: dict[str, str] = {}
    for item in specs:
        name = item["name"]
        root = (NBV0 if item["family"] == "legacy-v0" else NB) / "models" / name
        new_model = load_model(root)
        old_root = (V0 if item["family"] == "legacy-v0" else FULL) / "models" / name
        old_model = load_model(old_root)
        new_hash_to_name[new_model.artifact_sha256] = name
        old_reports_root = V0 / "reports" if item["family"] == "legacy-v0" else OLD_REPORTS
        new_reports_root = NBV0 / "reports" if item["family"] == "legacy-v0" else NEW_REPORTS

        def read(which: str, root_dir: Path) -> dict[str, Any]:
            filename, path = item[which]
            return load_metric(root_dir, filename, path)

        old_val = read("oldVal", old_reports_root)
        old_test = read("oldTest", old_reports_root)
        new_val = read("newVal", new_reports_root)
        new_test = read("newTest", new_reports_root)

        def delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
            return {
                key: None if before[key] is None or after[key] is None else after[key] - before[key]
                for key in METRICS
            }

        models.append(
            {
                "name": name,
                "family": item["family"],
                "predictionTask": new_model.prediction_task,
                "old": {
                    "artifactSha256": old_model.artifact_sha256,
                    "manifestSha256": old_model.training_summary.get("manifestSha256"),
                    "validation": old_val,
                    "test": old_test,
                    "reports": {
                        "validation": str(old_reports_root / item["oldVal"][0]),
                        "test": str(old_reports_root / item["oldTest"][0]),
                    },
                },
                "withoutBeach": {
                    "artifactSha256": new_model.artifact_sha256,
                    "manifestSha256": new_model.training_summary.get("manifestSha256"),
                    "validation": new_val,
                    "test": new_test,
                    "reports": {
                        "validation": str(new_reports_root / item["newVal"][0]),
                        "test": str(new_reports_root / item["newTest"][0]),
                    },
                    "modelPath": str(root),
                },
                "deltaWithoutBeachMinusOld": {
                    "validation": delta(old_val, new_val),
                    "test": delta(old_test, new_test),
                },
                **({"notes": [item["note"]]} if item.get("note") else {}),
            }
        )

    inference_rows: list[dict[str, Any]] = []
    for path in sorted(NB.glob("analyses/*/analysis.json")):
        payload = json.loads(path.read_text())
        recording_id = payload.get("recordingId")
        if recording_id not in recording_meta:
            continue
        analysis = payload.get("analysis", {})
        method = analysis.get("method", "")
        model_sha = analysis.get("modelSha256")
        model_name: str | None = None
        if method == "court-motion-temporal-logistic+serve-specialist-v1":
            serve = analysis.get("models", {}).get("serve", {})
            model_name = new_hash_to_name.get(serve.get("sha256"))
        else:
            model_name = new_hash_to_name.get(model_sha)
        if model_name is None:
            # Shared analysis directories contain older artifacts; they are not
            # part of this retraining run unless their hash is new.
            continue
        meta = recording_meta[recording_id]
        inference_rows.append(
            {
                "model": model_name,
                "recordingId": recording_id,
                "environment": meta["environment"],
                "split": meta["split"],
                "predictedRallies": len(payload.get("rallies", [])),
                "analysisPath": str(path),
                "outOfDomainBeachInference": meta["environment"] == "beach",
            }
        )

    by_model = {name: [] for name in names}
    for row in inference_rows:
        by_model[row["model"]].append(row)
    coverage = []
    for item in specs:
        name = item["name"]
        rows = sorted(by_model[name], key=lambda row: row["recordingId"])
        coverage.append(
            {
                "model": name,
                "recordings": len(rows),
                "expectedRecordings": len(recording_meta),
                "complete": len(rows) == len(recording_meta),
                "beachPredictedRallies": sum(row["predictedRallies"] for row in rows if row["environment"] == "beach"),
                "nonBeachPredictedRallies": sum(row["predictedRallies"] for row in rows if row["environment"] != "beach"),
                "rows": rows,
            }
        )
    if len(inference_rows) != len(specs) * len(recording_meta):
        incomplete = [row for row in coverage if not row["complete"]]
        raise RuntimeError(f"inference coverage incomplete: {incomplete}")

    test_f1 = [row["deltaWithoutBeachMinusOld"]["test"]["eventF1"] for row in models]
    val_f1 = [row["deltaWithoutBeachMinusOld"]["validation"]["eventF1"] for row in models]

    def f1_summary(values: list[float]) -> dict[str, Any]:
        ordered = sorted(values)
        return {
            "improved": sum(value > 1e-12 for value in values),
            "declined": sum(value < -1e-12 for value in values),
            "unchanged": sum(abs(value) <= 1e-12 for value in values),
            "meanDelta": sum(values) / len(values),
            "medianDelta": ordered[len(ordered) // 2],
        }

    summary = {
        "modelCount": len(models),
        "inferenceModelCount": len(coverage),
        "inferenceRecordingCount": len(recording_meta),
        "inferenceArtifactCount": len(inference_rows),
        "expectedInferenceArtifactCount": len(models) * len(recording_meta),
        "testEventF1": f1_summary(test_f1),
        "validationEventF1": f1_summary(val_f1),
        "largestTestEventF1Gains": sorted(
            ({"model": row["name"], "delta": row["deltaWithoutBeachMinusOld"]["test"]["eventF1"]} for row in models),
            key=lambda row: row["delta"],
            reverse=True,
        )[:5],
        "largestTestEventF1Losses": sorted(
            ({"model": row["name"], "delta": row["deltaWithoutBeachMinusOld"]["test"]["eventF1"]} for row in models),
            key=lambda row: row["delta"],
        )[:5],
    }
    excluded = [row.id for row in full_manifest.recordings if row.environment == "beach"]
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "title": "Beach-exclusion retraining comparison",
        "scope": {
            "modelCount": len(models),
            "models": [row["name"] for row in models],
            "excludedTrainingEnvironment": "beach",
            "excludedFullRecordingIds": excluded,
            "fullOriginalManifest": str(FULL / "manifests/full-gold-v1.json"),
            "fullNoBeachManifest": str(NB / "manifests/full-gold-v1-no-beach.json"),
            "fullOriginalManifestSha256": load_model(FULL / "models/full-percentile-v1").training_summary.get("manifestSha256"),
            "fullNoBeachManifestSha256": load_model(NB / "models/full-percentile-v1").training_summary.get("manifestSha256"),
            "pilotOriginalManifest": str(FULL / "manifests/pilot-gold-v1.json"),
            "pilotNoBeachManifest": str(NB / "manifests/pilot-gold-v1-no-beach.json"),
            "v0OriginalManifest": str(V0 / "manifests/real-v0.json"),
            "v0NoBeachManifest": str(NBV0 / "manifests/real-v0-no-beach.json"),
        },
        "protocol": {
            "validation": "same validation recordings; validation remains tuning/selection data",
            "test": "same held-out indoor test recording(s) within each lineage; retrospective reports because this corpus was previously inspected",
            "comparison": "withoutBeach metric minus original metric; positive is better",
            "inference": "original nine-recording full manifest, including both excluded beach recordings; beach predictions are qualitative/out-of-domain and not scored",
            "boundaryHeadFoldChange": "beach-free full-corpus training has two source groups, so dead-ball/dead-state epoch selection used two-fold leave-one-source-group-out instead of the historical three-fold selector",
        },
        "summary": summary,
        "models": models,
        "inferenceCoverage": coverage,
        "artifacts": {
            "withoutBeachWorkspace": str(NB),
            "withoutBeachModels": str(NB / "models"),
            "withoutBeachReports": str(NB / "reports"),
            "withoutBeachAnalyses": str(NB / "analyses"),
            "withoutBeachV0Workspace": str(NBV0),
        },
        "excludedDiagnostics": {
            "featureOrderOofModels": "not retrained: fold-level OOF research artifacts without a deployable single-manifest lineage",
            "ballPresenceDetector": "not retrained: separate detector, not one of the rally/serve/dead-state model versions",
        },
    }
    json_path = NEW_REPORTS / "beach-exclusion-retraining-comparison.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def fmt(value: Any) -> str:
        return "—" if value is None else f"{value:.3f}"

    def pp(value: Any) -> str:
        return "—" if value is None else f"{value * 100:+.1f} pp"

    lines = [
        "# Beach-exclusion retraining comparison",
        "",
        f"Generated {report['createdAt']}.",
        "",
        "## Result",
        "",
        f"Evaluated {summary['modelCount']} model versions and produced {summary['inferenceArtifactCount']}/{summary['expectedInferenceArtifactCount']} fresh inference artifacts (all models × all nine recordings).",
        f"On test Event F1, {summary['testEventF1']['improved']} models improved, {summary['testEventF1']['declined']} declined, and {summary['testEventF1']['unchanged']} were unchanged; mean delta was {summary['testEventF1']['meanDelta']:+.3f}, median {summary['testEventF1']['medianDelta']:+.3f}.",
        "Positive deltas mean the beach-free retrain scored higher on the same held-out test recording; this is evidence about the beach-data effect, not a causal proof.",
        "",
        "## Test comparison",
        "",
        "| Model | Family | Old F1 | No beach F1 | Δ F1 | Old IoU | No beach IoU | Δ IoU |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in models:
        old = row["old"]["test"]
        new = row["withoutBeach"]["test"]
        delta = row["deltaWithoutBeachMinusOld"]["test"]
        lines.append(f"| `{row['name']}` | {row['family']} | {fmt(old['eventF1'])} | {fmt(new['eventF1'])} | {pp(delta['eventF1'])} | {fmt(old['timeIoU'])} | {fmt(new['timeIoU'])} | {pp(delta['timeIoU'])} |")
    lines += [
        "",
        "## Validation comparison",
        "",
        "| Model | Old F1 | No beach F1 | Δ F1 | Old IoU | No beach IoU | Δ IoU | Old live recall | No beach live recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in models:
        old = row["old"]["validation"]
        new = row["withoutBeach"]["validation"]
        delta = row["deltaWithoutBeachMinusOld"]["validation"]
        lines.append(f"| `{row['name']}` | {fmt(old['eventF1'])} | {fmt(new['eventF1'])} | {pp(delta['eventF1'])} | {fmt(old['timeIoU'])} | {fmt(new['timeIoU'])} | {pp(delta['timeIoU'])} | {fmt(old['liveTimeRecall'])} | {fmt(new['liveTimeRecall'])} |")
    lines += [
        "",
        "## Inference coverage",
        "",
        "Every retrained model has nine fresh inference artifacts: two beach videos (qualitative only), four grass videos, two indoor development videos, and one indoor held-out test video. Full per-video paths and predicted rally counts are in the JSON report under `inferenceCoverage`.",
        "",
        "| Model | Recordings | Beach predicted rallies | Non-beach predicted rallies |",
        "|---|---:|---:|---:|",
    ]
    for row in coverage:
        lines.append(f"| `{row['model']}` | {row['recordings']}/{row['expectedRecordings']} | {row['beachPredictedRallies']} | {row['nonBeachPredictedRallies']} |")
    lines += [
        "",
        "## Artifacts and limitations",
        "",
        f"- Beach-free workspace: `{NB}`",
        f"- Models: `{NB / 'models'}`",
        f"- Evaluation reports: `{NB / 'reports'}`",
        f"- Inference timelines: `{NB / 'analyses'}`",
        f"- Machine-readable report: `{json_path}`",
        "",
        "- Beach videos were removed only from training; they remain in inference coverage and are not scored in the beach-free comparison.",
        "- Full-corpus boundary heads use two source-group epoch folds after beach removal; this is recorded in the JSON protocol.",
        "- Fold-level feature-order OOF artifacts and the separate ball-presence detector are outside this 33-version rally/serve/dead-state lineage.",
    ]
    md_path = NEW_REPORTS / "beach-exclusion-retraining-comparison.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(json_path)
    print(md_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
