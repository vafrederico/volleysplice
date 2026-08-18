#!/usr/bin/env python3
"""Add predeclared selection and protected-test scopes to an existing v3 report."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAIN_SCRIPT = ROOT / "scripts" / "train-feedback-suppression-v3.py"


def atomic_write(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()
    experiment_root = args.experiment_root.resolve()
    ns = runpy.run_path(str(TRAIN_SCRIPT))
    report_path = experiment_root / "report.json"
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))

    _, v2_heads = ns["load_heads"]()
    training_ids = {
        recording.id
        for recording in ns["load_manifest"](ns["ALL_LABELS_MANIFEST"]).recordings
    }
    split = json.loads(
        (experiment_root / "split-policy.json").read_text(encoding="utf-8")
    )
    fit_ids = set(split["fitProjectIds"])
    item_type = ns["Item"]
    entries = []
    for recording in ns["load_labeled_records"]().values():
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
        partition = "fit" if prepared.recording.id in fit_ids else "held-out"
        entries.append(item_type(prepared, "export-feedback", partition, True))

    interval_type = ns["Interval"]
    variants: dict[str, dict[str, tuple[Any, ...]]] = {}
    for variant in report["variantOrder"]:
        variants[variant] = {}
        for entry in entries:
            recording_id = entry.prepared.recording.id
            payload = json.loads(
                (
                    experiment_root
                    / "inference"
                    / variant
                    / f"{recording_id}.json"
                ).read_text(encoding="utf-8")
            )
            variants[variant][recording_id] = tuple(
                interval_type(float(row["start"]), float(row["end"]))
                for row in payload["ranges"]
            )

    scoped_entries = {
        "selection:development": [
            entry
            for entry in entries
            if entry.prepared.recording.id in ns["DEVELOPMENT_IDS"]
        ],
        "disclosure:protected-test": [
            entry
            for entry in entries
            if entry.prepared.recording.id in ns["PROTECTED_TEST_IDS"]
        ],
    }
    for scope, items in scoped_entries.items():
        report["metrics"][scope] = {}
        for variant in report["variantOrder"]:
            padding_metrics = {
                f"{padding:g}": ns["metric_for"](items, variants[variant], padding)
                for padding in ns["PADDING_CASES"]
            }
            baseline = ns["metric_for"](
                items,
                variants["current-production-ensemble"],
                ns["TARGET_PADDING_SECONDS"],
            )
            report["metrics"][scope][variant] = {
                "padding": padding_metrics,
                "deltaAtTargetVsCurrentProduction": ns["delta"](
                    padding_metrics["2"], baseline
                ),
            }

    report["displayScopes"] = [
        "all-evaluable",
        "selection:development",
        "disclosure:protected-test",
        *[
            scope
            for scope in report["displayScopes"]
            if scope
            not in {
                "all-evaluable",
                "selection:development",
                "disclosure:protected-test",
            }
        ],
    ]
    report["scopeRoles"] = {
        "selection:development": (
            "Predeclared tuning-safe ranking scope; use this scope to select variants."
        ),
        "disclosure:protected-test": (
            "Protected test disclosure only; never used to select a variant or decoder."
        ),
    }
    report["productDecision"] = {
        "status": "recorded-2026-08-18",
        "productionBase": "current-production-ensemble",
        "productionSuppressionModelPath": (
            "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
            "feedback-suppression-v3-2026-08-16/models/"
            "suppression-overlap-exclusion-retrained"
        ),
        "productionSuppressionArtifactSha256": (
            "39eddf58163901930434ea422a802686ae921ae1e8fe23c20a3c12e5f453da93"
        ),
        "productionSuppressionWeightsSha256": (
            "a943749b69c98a1bc926f8efc9fe60c67c1226534fe09a892c632519217aa3bb"
        ),
        "productionSuppressionDecoder": {
            "smoothing_seconds": 1.0,
            "enter_threshold": 0.75,
            "exit_threshold": 0.65,
            "min_live_seconds": 0.5,
            "bridge_gap_seconds": 0.5,
            "short_event_min_seconds": 0.25,
            "short_event_threshold": 0.9,
        },
        "v3Candidate1": "rejected-not-moving-forward",
        "v3Candidate2": "rejected-not-moving-forward",
        "reportVariantCaveat": (
            "The production-plus-suppression inference variants stored in this "
            "research experiment used the later re-tuned decoder. They are not "
            "the product decision. Use the held-decoder UI datasets and HTML "
            "report for the retained product-policy comparison."
        ),
        "htmlReport": (
            "https://internal.example/reports/"
            "volleycut-corrected-v3-comparison-2026-08-18.html"
        ),
    }
    atomic_write(report_path, json.dumps(report, indent=2, allow_nan=False) + "\n")
    atomic_write(experiment_root / "report.md", ns["markdown_report"](report))
    print(report_path)


if __name__ == "__main__":
    main()
