"""Frozen-production replay and source-group-held production-architecture refits.

Weights are fitted only on each outer fitting partition. Historical production
decoder settings and epoch caps stay fixed: they are not a fresh nested estimate
of decoder/hyperparameter selection, and their historical exposure is disclosed.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
import copy
import hashlib
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import TrainingConfig
from .dead_ball_experiment import _dead_ball_input_mask, _masked_dead_ball_values
from .dead_state_experiment import END_TRANSITION_TARGET_MODE, _target_for_mode
from .features import FeatureSequence, VideoMetadata, contextualize
from .model import DEAD_STATE_TASK, RALLY_LIVE_TASK, SERVE_CONTACT_TASK, load_model, train_logistic_model
from .neural_evaluation import evaluate_predictions
from .pipeline import PreparedRecording
from .runtime_models import load_production_runtime
from .schema import Interval, labels_for_times, load_manifest, mask_for_times
from .serve import serve_labels_for_times
from .side_switch_production_replay import ProductionHeads, _predict_ranges


PROTECTED_GROUPS = frozenset({private_value('source-group-008')})
RUNTIME_FILES = {"previous-production": "model-9c92b8e9333f.json", "all-labels-v2": "model-1ca43e38eefc.json"}
OLD_HEADS = {"rally": "full-audiovisual-audio-normalized-v3", "serve": "serve-specialist-audio-normalized-v5",
             "deadState": "dead-state-transition-audio-normalized-v5-no-legacy-final"}
TASKS = {"rally": RALLY_LIVE_TASK, "serve": SERVE_CONTACT_TASK, "deadState": DEAD_STATE_TASK}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def fitting_scope(items: list[PreparedRecording], held_group: str, allowed_ids: set[str]) -> list[PreparedRecording]:
    result = [item for item in items if item.recording.source_group != held_group and item.recording.id in allowed_ids]
    if not result or any(item.recording.source_group in PROTECTED_GROUPS for item in result):
        raise ValueError("empty or protected production fitting scope")
    if any(item.recording.consent.get("train") is not True for item in result):
        raise ValueError("production fitting requires explicit training consent")
    return result


def production_targets(item: PreparedRecording, role: str, historical: dict) -> tuple[np.ndarray, np.ndarray]:
    times, rallies = item.sequence.times, item.recording.rallies
    if role == "rally":
        return labels_for_times(times, rallies), item.sample_mask.copy()
    if role == "serve":
        return serve_labels_for_times(times, rallies, float(historical["serveTarget"]["radiusSeconds"])), item.sample_mask.copy()
    target = historical["deadStateTarget"]
    if target["mode"] != END_TRANSITION_TARGET_MODE:
        raise ValueError("expected the shipped local dead-state transition target")
    labels, mask = _target_for_mode(
        times, rallies, target_mode=target["mode"],
        before_end_seconds=float(target["beforeEndSeconds"]),
        after_end_seconds=float(target["afterEndSeconds"]),
        pre_serve_setup_seconds=float(target["preServeSetupSeconds"]),
    )
    return labels, mask & item.sample_mask


def hard_negative_augmented(item: PreparedRecording, values: np.ndarray, labels: np.ndarray,
                            multiplier: int) -> tuple[np.ndarray, np.ndarray]:
    indexes = np.flatnonzero(item.sample_mask)
    selected = np.zeros(len(item.sequence.times), dtype=bool)
    for interval in item.recording.raw.get("hardNegatives", []):
        selected |= (item.sequence.times >= interval["start"]) & (item.sequence.times < interval["end"])
    positions = np.flatnonzero(selected[indexes] & (item.labels[indexes] < 0.5))
    if multiplier > 1 and len(positions):
        values = np.concatenate([values, *([values[positions]] * (multiplier - 1))])
        labels = np.concatenate([labels, *([labels[positions]] * (multiplier - 1))])
    return values, labels


def refit_bundle(items: list[PreparedRecording], template: ProductionHeads, recipe: dict,
                 destination: Path, held_group: str) -> tuple[ProductionHeads, dict]:
    roles = {"rally": template.rally, "serve": template.serve, "deadState": template.dead}
    fitted, summaries = {}, {}
    for role, template_head in roles.items():
        matrices, targets = [], []
        metadata = recipe[role]
        for item in items:
            if item.recording.source_group == held_group:
                raise ValueError("held labels reached production fitting")
            labels, eligible = production_targets(item, role, metadata)
            values, labels = item.contextual_values[eligible], labels[eligible]
            if role == "deadState":
                retained = _dead_ball_input_mask(item.contextual_names, metadata["deadStateInputProfile"]["id"])
                values = _masked_dead_ball_values(values, retained)
            elif recipe["hardNegativeMultiplier"] > 1:
                values, labels = hard_negative_augmented(item, values, labels, recipe["hardNegativeMultiplier"])
            matrices.append(values)
            targets.append(labels)
        config = TrainingConfig(**metadata["fixedConfig"])
        model = train_logistic_model(matrices, targets, [], [], template_head.feature_config,
                                     template_head.feature_names, template_head.decoder, config,
                                     prediction_task=TASKS[role])
        # The production API retains the best training-loss epoch within the fixed
        # cap when no validation is supplied. This matches the v2 refit script.
        model.training_summary.update(copy.deepcopy(template_head.training_summary))
        model.training_summary.update({
            "heldOutSourceGroup": held_group,
            "trainingRecordingIds": [item.recording.id for item in items],
            "trainingSourceGroups": sorted({item.recording.source_group for item in items}),
            "validationRecordingIds": [], "heldLabelsUsedForFittingOrSelection": False,
            "historicalRecipe": metadata,
        })
        path = destination / role
        model.save(path)
        fitted[role] = model
        summaries[role] = {"path": str(path), "artifactSha256": model.artifact_sha256,
                           "training": model.training_summary}
    return ProductionHeads(fitted["rally"], fitted["serve"], fitted["deadState"]), summaries


def merge_overlap_union(groups: list[list[Interval]]) -> list[Interval]:
    # Product merges strictly overlapping candidate ranges; touching rows retain
    # event identity until the evaluation/export union applies its own rules.
    rows = sorted((row for group in groups for row in group), key=lambda row: (row.start, row.end))
    result: list[Interval] = []
    for row in rows:
        if result and row.start < result[-1].end:
            result[-1] = Interval(result[-1].start, max(result[-1].end, row.end))
        else:
            result.append(Interval(row.start, row.end))
    return result


def predict_bundle(item: PreparedRecording, heads: ProductionHeads) -> list[Interval]:
    # Explicitly remove all truth from the inference input, even though the replay
    # helper currently uses only cached contextual values and predicted scores.
    blind_record = replace(item.recording, rallies=(), ignored_intervals=(), raw={})
    blind = replace(item, recording=blind_record, labels=np.zeros_like(item.labels),
                    sample_mask=np.ones_like(item.sample_mask))
    return [Interval(row.start, row.end) for row in _predict_ranges(blind, heads)]


def load_recipes(data_root: Path, runtimes: dict) -> tuple[dict, dict]:
    previous_root = data_root / "labeling-v1-2026-08-09-no-beach-2026-08-12/models"
    current_root = data_root / "intake-2026-08-13/experiments/environment-specialists-v2/models/all-labels"
    recipes, lineage = {}, {}
    previous = {role: load_model(previous_root / name) for role, name in OLD_HEADS.items()}
    current = {role: load_model(current_root / name) for role, name in
               {"rally": "rally", "serve": "serve", "deadState": "dead-state"}.items()}
    for variant, models in (("previous-production", previous), ("all-labels-v2", current)):
        recipe = {"hardNegativeMultiplier": 1 if variant == "previous-production" else 4}
        info = {}
        for role, model in models.items():
            if model.artifact_sha256 != runtimes[variant][role]["artifactSha256"]:
                raise ValueError(f"historical model does not match shipped {variant}/{role}")
            training = model.training_summary
            cap = int(previous[role].training_summary["bestEpoch"])
            config = TrainingConfig(epochs=cap, batch_size=2048, learning_rate=0.02,
                                    l2=0.0001, patience=cap, seed=7)
            recipe[role] = {key: copy.deepcopy(training[key]) for key in
                            ("serveTarget", "serveInputProfile", "deadStateTarget", "deadStateInputProfile") if key in training}
            recipe[role]["fixedConfig"] = config.to_dict()
            info[role] = {"artifactSha256": model.artifact_sha256,
                          "trainingRecordingIds": training["trainingRecordingIds"],
                          "validationRecordingIds": training.get("validationRecordingIds", []),
                          "trainingSourceGroups": training.get("trainingSourceGroups", []),
                          "historicalConfig": training["config"], "historicalBestEpoch": training["bestEpoch"]}
        recipe["trainingRecordingIds"] = models["rally"].training_summary["trainingRecordingIds"]
        recipes[variant], lineage[variant] = recipe, info
    return recipes, lineage


def evaluation_rows(items: list[PreparedRecording], predictions: dict[str, list[Interval]]) -> list[dict]:
    return [{"id": item.recording.id, "sourceGroup": item.recording.source_group,
             "durationSeconds": item.sequence.metadata.duration, "rallies": item.recording.rallies,
             "ignoredIntervals": item.recording.ignored_intervals, "predictions": predictions[item.recording.id]}
            for item in items]


def run(manifest_path: Path, output_dir: Path, runtime_root: Path, data_root: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing existing immutable baseline directory: {output_dir}")
    started = time.perf_counter()
    manifest = load_manifest(manifest_path, require_videos=False)
    manifest_hash = file_sha256(manifest_path)
    audit = json.loads((manifest_path.parent / "dataset-audit.json").read_text())
    if audit["manifestSha256"] != manifest_hash:
        raise ValueError("manifest is not the frozen audited revision")
    runtimes, heads = {}, {}
    for variant, name in RUNTIME_FILES.items():
        runtimes[variant], heads[variant] = load_production_runtime(runtime_root / name)
    recipes, historical_lineage = load_recipes(data_root, runtimes)
    template = heads["previous-production"].rally
    items = []
    for record in manifest.recordings:
        if record.source_group in PROTECTED_GROUPS or record.split not in {"train", "validation"} or not record.consent.get("train"):
            raise ValueError(f"unauthorized/protected development row: {record.id}")
        entry = record.raw["featureCaches"]["audiovisual"]
        path = Path(entry["path"])
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"changed feature cache: {record.id}")
        with np.load(path, allow_pickle=False) as data:
            sequence = FeatureSequence(data["times"].astype(np.float64), data["values"].astype(np.float32),
                                       tuple(str(v) for v in data["names"]),
                                       VideoMetadata(**json.loads(str(data["metadata_json"].item()))))
        values, names = contextualize(sequence, template.feature_config)
        if names != template.feature_names or names != heads["all-labels-v2"].rally.feature_names:
            raise ValueError("production feature signature mismatch")
        items.append(PreparedRecording(record, sequence, values, names,
                                        labels_for_times(sequence.times, record.rallies),
                                        mask_for_times(sequence.times, record.ignored_intervals)))
    groups = sorted({item.recording.source_group for item in items})
    all_predictions = {"shipped-previous": {}, "shipped-all-labels": {}, "shipped-union": {},
                       "refit-previous": {}, "refit-all-labels": {}, "refit-union": {}}
    for item in items:
        old = predict_bundle(item, heads["previous-production"])
        current = predict_bundle(item, heads["all-labels-v2"])
        all_predictions["shipped-previous"][item.recording.id] = old
        all_predictions["shipped-all-labels"][item.recording.id] = current
        all_predictions["shipped-union"][item.recording.id] = merge_overlap_union([old, current])
    folds = []
    for group in groups:
        held = [item for item in items if item.recording.source_group == group]
        fitted, fold = {}, {"heldOutSourceGroup": group, "bundles": {}}
        for variant in RUNTIME_FILES:
            fit = fitting_scope(items, group, set(recipes[variant]["trainingRecordingIds"]))
            print(f"Refit {group} / {variant}: {len(fit)} fitting recordings", flush=True)
            fitted[variant], details = refit_bundle(fit, heads[variant], recipes[variant],
                                                    output_dir / "models" / group / variant, group)
            fold["bundles"][variant] = details
        for item in held:
            old, current = (predict_bundle(item, fitted[variant]) for variant in RUNTIME_FILES)
            all_predictions["refit-previous"][item.recording.id] = old
            all_predictions["refit-all-labels"][item.recording.id] = current
            all_predictions["refit-union"][item.recording.id] = merge_overlap_union([old, current])
        folds.append(fold)
    report = {
        "schemaVersion": 1, "kind": "volleycut-neural-production-baseline-v1",
        "createdAt": datetime.now(timezone.utc).isoformat(), "manifestPath": str(manifest_path),
        "manifestSha256": manifest_hash, "primaryPaddingSeconds": 2.0, "joinGapSeconds": 3.0,
        "trainingExposure": {
            "shipped": "Retrospective product reference; several development groups were training or historical selection data.",
            "refit": "Outer-group-held weights/scalers; no held labels used in current fitting or selection. Historical decoder/caps fixed.",
        },
        "differencesFromFullHistoricalProduction": [
            "Both production training allowlists exclude beach; beach folds measure unseen-environment transfer.",
            "Each refit uses the intersection of the historical training allowlist and the frozen eight-recording scope, minus its outer group.",
            "Later v2 training additions are excluded because they are outside this audited exact-label experiment.",
            "Previous-production early stopping is replaced with the fixed historical best-epoch cap; training-loss best checkpoint is retained by the existing API.",
            "Historical decoder settings and epoch caps were selected with historical development data, so refitting weights cannot erase that earlier selection exposure.",
            "Suppression is disabled, matching the default core ensemble product path; serving-side and switch outputs do not affect these rally ranges.",
        ],
        "runtimes": {variant: {"path": str(runtime_root / name), "sha256": file_sha256(runtime_root / name)}
                     for variant, name in RUNTIME_FILES.items()},
        "historicalLineage": historical_lineage, "recipes": recipes, "folds": folds,
        "results": {name: evaluate_predictions(evaluation_rows(items, predictions), primary_padding_seconds=2.0,
                                                join_gap_seconds=3.0)
                    for name, predictions in all_predictions.items()},
        "predictions": {name: {key: [row.to_dict() for row in rows] for key, rows in predictions.items()}
                        for name, predictions in all_predictions.items()},
        "wallSeconds": time.perf_counter() - started,
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__).resolve())},
    }
    write_new_json(output_dir / "baseline.json", report)
    print(json.dumps({"output": str(output_dir / "baseline.json"),
                      "F1_padP_coreR": {name: result["objective"] for name, result in report["results"].items()}}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(private_value('private-reference-0059'))
    parser.add_argument("--manifest", type=Path, default=root / "neural-experiments/2026-09-18-phase1/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=root / "neural-experiments/2026-09-18-phase1/production-baseline")
    parser.add_argument("--runtime-root", type=Path, default=Path(__file__).resolve().parents[1] / "prod/public/runtime")
    parser.add_argument("--data-root", type=Path, default=root)
    args = parser.parse_args()
    run(args.manifest, args.output_dir, args.runtime_root, args.data_root)


if __name__ == "__main__":
    main()
