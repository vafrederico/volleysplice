#!/usr/bin/env python3
"""Refit the production three-head architecture for environment experiments v2.

The experiment deliberately freezes the production feature extractor, model shapes,
epoch caps, decoder, serve composition, and dead-state refinement.  It trains on
explicit recording allowlists.  The Turkey Tourney, Forest Ridge, and YMCA KOB indoor
recordings are v2 training additions for the all-label and indoor-only variants; SPU
Match 1 Set 2 and the reserved grass recordings remain evaluation-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from analysis.annotations import load_label_document
from analysis.artifacts import atomic_write_text
from analysis.config import TrainingConfig
from analysis.dead_ball_experiment import (
    _dead_ball_input_mask,
    _masked_dead_ball_values,
    _prediction_inputs,
)
from analysis.dead_state import DeadStateDecoderConfig
from analysis.dead_state_experiment import (
    DEAD_STATE_TARGET_ID,
    END_TRANSITION_TARGET_MODE,
    DeadStateRefinementConfig,
    _eligible_target_mask,
    _predictions_for,
    _target_for_mode,
)
from analysis.features import camera_warnings, write_preview
from analysis.model import (
    DEAD_STATE_TASK,
    RALLY_LIVE_TASK,
    SERVE_CONTACT_TASK,
    LogisticModel,
    load_model,
    train_logistic_model,
)
from analysis.pipeline import PreparedRecording, _manifest_digest, prepare_recording
from analysis.schema import DatasetManifest, Recording, load_manifest
from analysis.serve import (
    ServeCompositionConfig,
    ServeDecoderConfig,
    serve_labels_for_times,
)
from analysis.version import __version__


EXPERIMENT_ID = "environment-specialists-v2"
PRODUCTION_MODEL_ID = "model-9c92b8e9333f"
PRODUCTION_HEADS = {
    "rally": "full-audiovisual-audio-normalized-v3",
    "serve": "serve-specialist-audio-normalized-v5",
    "deadState": "dead-state-transition-audio-normalized-v5-no-legacy-final",
}
LATEST_GRASS_TRAIN_IDS = (
    "grass-source-03",
    "grass-source-05",
)
LATEST_INDOOR_TRAIN_IDS = (
    "indoor-source-06",
    "indoor-source-04",
    "indoor-source-08",
)
INDOOR_EVALUATION_IDS = (
    "indoor-source-03",
)
LABEL_DOCUMENT_OVERRIDES = {
    "indoor-source-08": Path(
        "/mnt/freenas/volleycut/intake-2026-08-13/completed/full-v1/"
        "indoor-source-08.labels.json"
    ),
}
FUTURE_GRASS_EVALUATION_IDS = (
    "grass-source-02",
    "grass-source-06",
    "grass-source-08",
)
VARIANT_LABELS = {
    "all-labels": "All labels + latest grass v2",
    "grass-only": "Grass specialist",
    "indoor-only": "Indoor specialist v2",
}
HARD_NEGATIVE_MULTIPLIER = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _recording_row(recording: Recording, *, split: str) -> dict[str, Any]:
    row = dict(recording.raw)
    row["video"] = str(recording.video)
    row["split"] = split
    row["consent"] = {"analyze": True, "train": split == "train"}
    if recording.content_sha256 is not None:
        row["contentSha256"] = recording.content_sha256
    return row


def _label_row(path: Path, *, split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    document = load_label_document(
        path, require_complete=False, require_video=True
    )
    recording = document.payload["recording"]
    annotation = document.payload.get("annotation", {})
    row = {
        "id": document.recording_id,
        "video": str(document.video),
        "split": split,
        "sourceGroup": document.source_group,
        "environment": document.environment,
        "game": recording.get("game", {}),
        "consent": {"analyze": True, "train": split == "train"},
        "capture": recording.get("capture", {}),
        "roi": recording.get("roi"),
        "contentSha256": recording.get("contentSha256"),
        "rallies": document.payload["rallies"],
        "ignoredIntervals": document.payload.get("ignoredIntervals", []),
        "hardNegatives": document.payload.get("hardNegatives", []),
        "sideSwitches": document.payload.get("sideSwitches", []),
        "annotation": annotation,
    }
    provenance = {
        "recordingId": document.recording_id,
        "path": str(path.resolve()),
        "fileSha256": sha256_file(path),
        "annotationStatus": annotation.get("status"),
        "annotator": annotation.get("annotator"),
        "continuousVideoReviewed": annotation.get("continuousVideoReviewed"),
        "reviewedAt": annotation.get("reviewedAt"),
        "rallies": len(document.rallies),
        "ignoredIntervals": len(document.ignored_intervals),
        "hardNegatives": len(document.hard_negatives),
        "hardNegativeCategories": sorted(
            {
                str(item.get("category"))
                for item in document.payload.get("hardNegatives", [])
                if isinstance(item, Mapping) and item.get("category") is not None
            }
        ),
    }
    return row, provenance


def _write_new_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _manifest_payload(
    name: str,
    rows: Sequence[dict[str, Any]],
    *,
    role: str,
    source_documents: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "name": name,
        "annotationPolicy": {
            "id": "serve-contact-to-dead-ball-v1",
            "rallyStart": "serve-ball contact",
            "rallyEnd": "first instant live play has ended",
            "intervalConvention": "half-open [start,end) seconds on the normalized video",
        },
        "experiment": {
            "id": EXPERIMENT_ID,
            "role": role,
            "createdAt": datetime.now(UTC).isoformat(),
            "productionBaseline": PRODUCTION_MODEL_ID,
            "sourceDocuments": list(source_documents),
        },
        "recordings": list(rows),
    }


def prepare_manifests(
    historical_manifest_path: Path,
    intake_manifest_path: Path,
    labels_root: Path,
    destination: Path,
) -> dict[str, Path]:
    historical = load_manifest(historical_manifest_path)
    intake = load_manifest(intake_manifest_path)
    historical_development = [
        item for item in historical.recordings if item.split in {"train", "validation"}
    ]
    historical_grass = [
        item for item in historical_development if item.environment == "grass"
    ]
    historical_indoor = [
        item
        for item in historical.recordings
        if item.environment == "indoor" and item.split == "train"
    ]

    latest_grass_rows: list[dict[str, Any]] = []
    latest_indoor_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    label_provenance: list[dict[str, Any]] = []
    for recording_id in (
        *LATEST_GRASS_TRAIN_IDS,
        *LATEST_INDOOR_TRAIN_IDS,
        *INDOOR_EVALUATION_IDS,
    ):
        training_id = recording_id in {
            *LATEST_GRASS_TRAIN_IDS,
            *LATEST_INDOOR_TRAIN_IDS,
        }
        row, provenance = _label_row(
            LABEL_DOCUMENT_OVERRIDES.get(
                recording_id, labels_root / f"{recording_id}.labels.json"
            ),
            split=("train" if training_id else "challenge"),
        )
        label_provenance.append(provenance)
        if recording_id in LATEST_GRASS_TRAIN_IDS:
            latest_grass_rows.append(row)
        elif recording_id in LATEST_INDOOR_TRAIN_IDS:
            latest_indoor_rows.append(row)
        else:
            evaluation_rows.append(row)

    intake_by_id = {item.id: item for item in intake.recordings}
    missing_reserved = set(FUTURE_GRASS_EVALUATION_IDS) - set(intake_by_id)
    if missing_reserved:
        raise ValueError(f"reserved grass recordings are missing: {sorted(missing_reserved)}")

    variants = {
        "all-labels": [
            *(_recording_row(item, split="train") for item in historical_development),
            *latest_grass_rows,
            *latest_indoor_rows,
        ],
        "grass-only": [
            *(_recording_row(item, split="train") for item in historical_grass),
            *latest_grass_rows,
        ],
        "indoor-only": [
            *(_recording_row(item, split="train") for item in historical_indoor),
            *latest_indoor_rows,
        ],
    }
    paths: dict[str, Path] = {}
    for variant, rows in variants.items():
        path = destination / "manifests" / f"{variant}-train.json"
        _write_new_json(
            path,
            _manifest_payload(
                f"volleycut-{EXPERIMENT_ID}-{variant}-train",
                rows,
                role="training-only",
                source_documents=[
                    row
                    for row in label_provenance
                    if row["recordingId"] in {
                        *(LATEST_GRASS_TRAIN_IDS if variant != "indoor-only" else ()),
                        *(LATEST_INDOOR_TRAIN_IDS if variant != "grass-only" else ()),
                    }
                ],
            ),
        )
        load_manifest(path)
        paths[variant] = path

    evaluation_path = destination / "manifests" / "indoor-evaluation.json"
    _write_new_json(
        evaluation_path,
        _manifest_payload(
            f"volleycut-{EXPERIMENT_ID}-indoor-evaluation",
            evaluation_rows,
            role="evaluation-only",
            source_documents=[
                row
                for row in label_provenance
                if row["recordingId"] in INDOOR_EVALUATION_IDS
            ],
        ),
    )
    load_manifest(evaluation_path)
    paths["indoor-evaluation"] = evaluation_path

    inference_path = destination / "manifests" / "intake-inference.json"
    _write_new_json(
        inference_path,
        _manifest_payload(
            f"volleycut-{EXPERIMENT_ID}-intake-inference",
            [_recording_row(item, split="challenge") for item in intake.recordings],
            role="prediction-only",
        ),
    )
    load_manifest(inference_path)
    paths["intake-inference"] = inference_path

    policy_path = destination / "split-policy.json"
    _write_new_json(
        policy_path,
        {
            "schemaVersion": 1,
            "experiment": EXPERIMENT_ID,
            "createdAt": datetime.now(UTC).isoformat(),
            "latestGrassTrainingOnly": list(LATEST_GRASS_TRAIN_IDS),
            "latestIndoorTrainingOnly": list(LATEST_INDOOR_TRAIN_IDS),
            "indoorEvaluationOnly": list(INDOOR_EVALUATION_IDS),
            "futureGrassEvaluationOnly": list(FUTURE_GRASS_EVALUATION_IDS),
            "hardNegativeMultiplier": HARD_NEGATIVE_MULTIPLIER,
            "hardNegativeInterpretation": (
                "Each marked hard-negative sample appears four times total in rally-live "
                "and serve-contact fitting; ignored intervals remain excluded."
            ),
            "labelDocuments": label_provenance,
        },
    )
    paths["split-policy"] = policy_path
    return paths


def stage_feature_cache(source_dirs: Sequence[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in source_dirs:
        for path in sorted(source.glob("*.npz")):
            target = destination / path.name
            if target.exists():
                if sha256_file(target) != sha256_file(path):
                    raise ValueError(f"feature cache collision differs: {path.name}")
                continue
            shutil.copy2(path, target)


def _hard_negative_mask(item: PreparedRecording) -> np.ndarray:
    result = np.zeros(len(item.sequence.times), dtype=np.bool_)
    for interval in item.recording.raw.get("hardNegatives", []):
        if not isinstance(interval, Mapping):
            continue
        start = float(interval["start"])
        end = float(interval["end"])
        result |= (item.sequence.times >= start) & (item.sequence.times < end)
    return result & item.sample_mask & (item.labels < 0.5)


def hard_negative_augmented(
    item: PreparedRecording,
    values: np.ndarray,
    labels: np.ndarray,
    *,
    multiplier: int = HARD_NEGATIVE_MULTIPLIER,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Repeat marked negative rows without changing model architecture."""

    if multiplier < 1:
        raise ValueError("hard-negative multiplier must be at least one")
    base_indexes = np.flatnonzero(item.sample_mask)
    hard_indexes = np.flatnonzero(_hard_negative_mask(item))
    if values.shape[0] != len(base_indexes) or labels.shape != (len(base_indexes),):
        raise ValueError("training values must correspond to the prepared sample mask")
    if multiplier == 1 or len(hard_indexes) == 0:
        return values, labels, len(hard_indexes)
    source_to_base = {int(source): index for index, source in enumerate(base_indexes)}
    selected = np.asarray([source_to_base[int(index)] for index in hard_indexes])
    repeats = multiplier - 1
    return (
        np.concatenate([values, *([values[selected]] * repeats)], axis=0),
        np.concatenate([labels, *([labels[selected]] * repeats)], axis=0),
        len(hard_indexes),
    )


def _fixed_training_config(epochs: int) -> TrainingConfig:
    return TrainingConfig(
        epochs=epochs,
        batch_size=2048,
        learning_rate=0.02,
        l2=0.0001,
        patience=epochs,
        seed=7,
    )


def _training_lineage(
    variant: str,
    manifest: DatasetManifest,
    manifest_sha256: str,
    prepared: Sequence[PreparedRecording],
    production_models: Mapping[str, LogisticModel],
    hard_negative_samples: int,
) -> dict[str, Any]:
    return {
        "dataset": manifest.name,
        "manifestSha256": manifest_sha256,
        "variant": variant,
        "environmentScope": (
            "grass+indoor" if variant == "all-labels" else variant.removesuffix("-only")
        ),
        "trainingRecordingIds": [item.recording.id for item in prepared],
        "validationRecordingIds": [],
        "trainingSourceGroups": sorted(
            {item.recording.source_group for item in prepared}
        ),
        "validationSourceGroups": [],
        "recordingContentSha256": {
            item.recording.id: item.recording.content_sha256 for item in prepared
        },
        "productionArchitecture": {
            role: {
                "version": PRODUCTION_HEADS[role],
                "sha256": model.artifact_sha256,
                "bestEpoch": model.training_summary.get("bestEpoch"),
            }
            for role, model in production_models.items()
        },
        "selectionPolicy": (
            "No evaluation labels or protected test labels were used. Epoch caps, decoder, "
            "serve composition, and dead-state refinement are frozen from production."
        ),
        "hardNegativeWeighting": {
            "multiplier": HARD_NEGATIVE_MULTIPLIER,
            "uniqueSamples": hard_negative_samples,
            "appliedTo": ["rally-live", "serve-contact"],
        },
    }


def _prepare_training(
    manifest: DatasetManifest,
    production_rally: LogisticModel,
    cache_dir: Path,
) -> list[PreparedRecording]:
    result: list[PreparedRecording] = []
    for index, recording in enumerate(manifest.recordings, start=1):
        if recording.split != "train":
            raise ValueError(f"training manifest contains non-training row: {recording.id}")
        print(
            f"Preparing {manifest.name} {index}/{len(manifest.recordings)}: {recording.id}",
            flush=True,
        )
        item = prepare_recording(recording, production_rally.feature_config, cache_dir)
        if item.contextual_names != production_rally.feature_names:
            raise ValueError(f"feature signature mismatch for {recording.id}")
        result.append(item)
    return result


def train_variant(
    variant: str,
    manifest_path: Path,
    production_root: Path,
    cache_dir: Path,
    model_root: Path,
) -> dict[str, Any]:
    destination = model_root / variant
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite model variant: {destination}")
    production = {
        role: load_model(production_root / name)
        for role, name in PRODUCTION_HEADS.items()
    }
    production_rally = production["rally"]
    production_serve = production["serve"]
    production_dead = production["deadState"]
    manifest = load_manifest(manifest_path)
    manifest_sha256 = _manifest_digest(manifest)
    prepared = _prepare_training(manifest, production_rally, cache_dir)

    rally_values: list[np.ndarray] = []
    rally_labels: list[np.ndarray] = []
    hard_samples = 0
    for item in prepared:
        values, labels, count = hard_negative_augmented(
            item,
            item.contextual_values[item.sample_mask],
            item.labels[item.sample_mask],
        )
        rally_values.append(values)
        rally_labels.append(labels)
        hard_samples += count
    rally = train_logistic_model(
        rally_values,
        rally_labels,
        [],
        [],
        production_rally.feature_config,
        production_rally.feature_names,
        production_rally.decoder,
        _fixed_training_config(int(production_rally.training_summary["bestEpoch"])),
        prediction_task=RALLY_LIVE_TASK,
    )
    rally.feature_version = production_rally.feature_version
    lineage = _training_lineage(
        variant,
        manifest,
        manifest_sha256,
        prepared,
        production,
        hard_samples,
    )
    rally.training_summary.update(lineage)
    rally_path = destination / "rally"
    rally.save(rally_path)

    radius = float(production_serve.training_summary["serveTarget"]["radiusSeconds"])
    serve_values: list[np.ndarray] = []
    serve_labels: list[np.ndarray] = []
    for item in prepared:
        labels = serve_labels_for_times(item.sequence.times, item.recording.rallies, radius)
        values, labels, _ = hard_negative_augmented(
            item,
            item.contextual_values[item.sample_mask],
            labels[item.sample_mask],
        )
        serve_values.append(values)
        serve_labels.append(labels)
    serve = train_logistic_model(
        serve_values,
        serve_labels,
        [],
        [],
        rally.feature_config,
        rally.feature_names,
        rally.decoder,
        _fixed_training_config(int(production_serve.training_summary["bestEpoch"])),
        prediction_task=SERVE_CONTACT_TASK,
    )
    serve.feature_version = rally.feature_version
    serve.training_summary.update(
        {
            **lineage,
            "rallyModelSha256": rally.artifact_sha256,
            "serveTarget": production_serve.training_summary["serveTarget"],
            "serveInputProfile": production_serve.training_summary["serveInputProfile"],
            "serveDecoder": production_serve.training_summary["serveDecoder"],
            "composition": production_serve.training_summary["composition"],
        }
    )
    serve_path = destination / "serve"
    serve.save(serve_path)

    target = production_dead.training_summary["deadStateTarget"]
    dead_values: list[np.ndarray] = []
    dead_labels: list[np.ndarray] = []
    retained = _dead_ball_input_mask(
        rally.feature_names,
        str(production_dead.training_summary["deadStateInputProfile"]["id"]),
    )
    for item in prepared:
        labels, target_mask = _target_for_mode(
            item.sequence.times,
            item.recording.rallies,
            target_mode=END_TRANSITION_TARGET_MODE,
            before_end_seconds=float(target["beforeEndSeconds"]),
            after_end_seconds=float(target["afterEndSeconds"]),
            pre_serve_setup_seconds=float(target["preServeSetupSeconds"]),
        )
        eligible = _eligible_target_mask(item.sample_mask, target_mask)
        dead_values.append(
            _masked_dead_ball_values(item.contextual_values[eligible], retained)
        )
        dead_labels.append(labels[eligible])
    dead = train_logistic_model(
        dead_values,
        dead_labels,
        [],
        [],
        rally.feature_config,
        rally.feature_names,
        rally.decoder,
        _fixed_training_config(int(production_dead.training_summary["bestEpoch"])),
        prediction_task=DEAD_STATE_TASK,
    )
    dead.feature_version = rally.feature_version
    dead.training_summary.update(
        {
            **lineage,
            "rallyModelSha256": rally.artifact_sha256,
            "serveModelSha256": serve.artifact_sha256,
            "deadStateTarget": {
                **production_dead.training_summary["deadStateTarget"],
                "id": DEAD_STATE_TARGET_ID,
            },
            "deadStateInputProfile": production_dead.training_summary[
                "deadStateInputProfile"
            ],
            "selectedDeadStateDecoder": production_dead.training_summary[
                "selectedDeadStateDecoder"
            ],
            "selectedRefinement": production_dead.training_summary[
                "selectedRefinement"
            ],
        }
    )
    dead_path = destination / "dead-state"
    dead.save(dead_path)

    bundle = {
        "schemaVersion": 1,
        "experiment": EXPERIMENT_ID,
        "variant": variant,
        "label": VARIANT_LABELS[variant],
        "createdAt": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "manifestSha256": manifest_sha256,
        "hardNegativeMultiplier": HARD_NEGATIVE_MULTIPLIER,
        "heads": {
            "rally": {
                "path": str(rally_path),
                "sha256": rally.artifact_sha256,
            },
            "serve": {
                "path": str(serve_path),
                "sha256": serve.artifact_sha256,
            },
            "deadState": {
                "path": str(dead_path),
                "sha256": dead.artifact_sha256,
            },
        },
        "inferenceVariant": dead.artifact_sha256[:12],
        "modelId": f"model-{dead.artifact_sha256[:12]}",
    }
    _write_new_json(destination / "bundle.json", bundle)
    return bundle


def _existing_preview(output_root: Path, recording_id: str) -> Path | None:
    for candidate in output_root.glob(f"model-*--{recording_id}/court-preview.jpg"):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _write_analysis(
    recording: Recording,
    prepared: PreparedRecording,
    intervals: Sequence[Any],
    output_root: Path,
    bundle: Mapping[str, Any],
    rally: LogisticModel,
    serve: LogisticModel,
    dead: LogisticModel,
) -> str:
    variant = str(bundle["inferenceVariant"])
    analysis_id = f"model-{variant}--{recording.id}"
    destination = output_root / analysis_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {destination}")
    destination.mkdir(parents=True)
    metadata = prepared.sequence.metadata
    roi = recording.roi
    rows = [
        {
            "id": f"R{index:03d}",
            "start": max(0.0, round(float(interval.start), 3)),
            "end": min(metadata.duration, round(float(interval.end), 3)),
            "confidence": round(max(0.0, min(1.0, float(interval.confidence))), 5),
            "included": True,
        }
        for index, interval in enumerate(intervals, start=1)
        if interval.end > interval.start
    ]
    payload = {
        "schemaVersion": 1,
        "id": analysis_id,
        "recordingId": recording.id,
        "title": recording.id,
        "createdAt": datetime.now(UTC).isoformat(),
        "source": {
            "filename": recording.video.name,
            "contentSha256": recording.content_sha256,
            **metadata.to_dict(),
        },
        "assets": {"courtPreviewPath": "court-preview.jpg"},
        "analysis": {
            "method": "production-three-head-refit-v1",
            "modelVersion": f"{EXPERIMENT_ID}-{bundle['variant']}",
            "modelSha256": dead.artifact_sha256,
            "variantLabel": VARIANT_LABELS[str(bundle["variant"])],
            "variantDescription": (
                "Production three-head architecture refit with frozen production "
                "decoders and explicit environment-safe training scope."
            ),
            "producer": f"volleycut-analysis/{__version__}",
            "analysisFps": rally.feature_config.analysis_fps,
            "models": {
                "rally": {"version": "rally", "sha256": rally.artifact_sha256},
                "serve": {"version": "serve", "sha256": serve.artifact_sha256},
                "deadState": {
                    "version": "dead-state",
                    "sha256": dead.artifact_sha256,
                },
            },
            "selection": {
                "rallyDecoder": rally.decoder.to_dict(),
                "serveDecoder": serve.training_summary["serveDecoder"],
                "serveComposition": serve.training_summary["composition"],
                "deadStateDecoder": dead.training_summary[
                    "selectedDeadStateDecoder"
                ],
                "deadStateRefinement": dead.training_summary["selectedRefinement"],
            },
            "warnings": [
                *camera_warnings(metadata, recording.capture),
                "experimental environment-specialist model; not promoted to production",
            ],
            "court": {
                "source": "manual-roi" if roi is not None else "full-frame-fallback",
                "roi": (
                    {"x": roi[0], "y": roi[1], "width": roi[2], "height": roi[3]}
                    if roi is not None
                    else None
                ),
            },
        },
        "rallies": rows,
    }
    preview = destination / "court-preview.jpg"
    reference = _existing_preview(output_root, recording.id)
    if reference is not None:
        shutil.copyfile(reference, preview)
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".court-preview-", suffix=".jpg", dir=destination
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink()
        write_preview(recording.video, temporary, recording.roi)
        temporary.replace(preview)
    atomic_write_text(
        destination / "analysis.json",
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
    )
    return analysis_id


def infer_variant(
    bundle_path: Path,
    inference_manifest_path: Path,
    cache_dir: Path,
    output_root: Path,
    *,
    skip_existing: bool = False,
) -> list[str]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    heads = bundle["heads"]
    rally = load_model(heads["rally"]["path"])
    serve = load_model(heads["serve"]["path"])
    dead = load_model(heads["deadState"]["path"])
    serve_decoder = ServeDecoderConfig.from_dict(serve.training_summary["serveDecoder"])
    serve_composition = ServeCompositionConfig.from_dict(
        serve.training_summary["composition"]
    )
    dead_decoder = DeadStateDecoderConfig.from_dict(
        dead.training_summary["selectedDeadStateDecoder"]
    )
    refinement = DeadStateRefinementConfig.from_dict(
        dead.training_summary["selectedRefinement"]
    )
    manifest = load_manifest(inference_manifest_path)
    created: list[str] = []
    for index, recording in enumerate(manifest.recordings, start=1):
        analysis_id = f"model-{bundle['inferenceVariant']}--{recording.id}"
        if skip_existing and (output_root / analysis_id / "analysis.json").is_file():
            print(
                f"Skipping existing {bundle['variant']} {index}/{len(manifest.recordings)}: "
                f"{recording.id}",
                flush=True,
            )
            continue
        print(
            f"Inferring {bundle['variant']} {index}/{len(manifest.recordings)}: {recording.id}",
            flush=True,
        )
        prepared = prepare_recording(recording, rally.feature_config, cache_dir)
        prediction_prepared = replace(
            prepared,
            recording=replace(recording, rallies=(), ignored_intervals=()),
            labels=np.zeros(len(prepared.sequence.times), dtype=np.float32),
            sample_mask=np.ones(len(prepared.sequence.times), dtype=np.bool_),
        )
        inputs = _prediction_inputs(
            prediction_prepared,
            rally,
            serve,
            dead,
            serve_decoder,
            serve_composition,
        )
        prediction = _predictions_for([inputs], dead_decoder, refinement)[0]
        created.append(
            _write_analysis(
                recording,
                prediction_prepared,
                prediction.candidate,
                output_root,
                bundle,
                rally,
                serve,
                dead,
            )
        )
    return created


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-manifest",
        type=Path,
        default=Path(
            "/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/"
            "manifests/full-gold-v1-no-beach.json"
        ),
    )
    parser.add_argument(
        "--intake-manifest",
        type=Path,
        default=Path(
            "/mnt/freenas/volleycut/intake-2026-08-13/manifests/inference-only.json"
        ),
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("/mnt/freenas/volleycut/intake-2026-08-13/labels/full"),
    )
    parser.add_argument(
        "--production-models-root",
        type=Path,
        default=Path(
            "/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/models"
        ),
    )
    parser.add_argument(
        "--historical-cache",
        type=Path,
        default=Path(
            "/mnt/freenas/volleycut/labeling-v1-2026-08-09/features/"
            "audiovisual-audio-normalized-v3"
        ),
    )
    parser.add_argument(
        "--intake-cache",
        type=Path,
        default=Path(
            "/mnt/freenas/volleycut/intake-2026-08-13/features/"
            "audiovisual-audio-normalized-v3"
        ),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(
            "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
            f"{EXPERIMENT_ID}"
        ),
    )
    parser.add_argument(
        "--analysis-output-root",
        type=Path,
        default=Path("/mnt/freenas/volleycut/intake-2026-08-13/analyses"),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write and validate the split manifests without fitting models",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="reuse already prepared manifests and feature-cache staging",
    )
    parser.add_argument(
        "--infer-only",
        action="store_true",
        help="reuse fitted bundles and generate analyses without preparing or training",
    )
    parser.add_argument(
        "--inference-manifest",
        type=Path,
        help="manifest to use with --infer-only (defaults to the prepared intake manifest)",
    )
    parser.add_argument(
        "--skip-existing-analyses",
        action="store_true",
        help="leave already populated model/video analyses untouched",
    )
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        choices=tuple(VARIANT_LABELS),
        help="train or infer only this variant; repeat for multiple variants",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    selected_variants = tuple(dict.fromkeys(args.variants or VARIANT_LABELS))
    destination = args.destination.expanduser().resolve()
    cache = destination / "features" / "audiovisual-audio-normalized-v3"
    if not args.skip_prepare and not args.infer_only:
        paths = prepare_manifests(
            args.historical_manifest.expanduser().resolve(),
            args.intake_manifest.expanduser().resolve(),
            args.labels_root.expanduser().resolve(),
            destination,
        )
        stage_feature_cache(
            [
                args.historical_cache.expanduser().resolve(),
                args.intake_cache.expanduser().resolve(),
            ],
            cache,
        )
    else:
        paths = {
            variant: destination / "manifests" / f"{variant}-train.json"
            for variant in VARIANT_LABELS
        }
        paths["indoor-evaluation"] = destination / "manifests" / "indoor-evaluation.json"
        paths["intake-inference"] = destination / "manifests" / "intake-inference.json"
    if args.prepare_only:
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return 0

    if args.infer_only:
        inference_manifest = (
            args.inference_manifest.expanduser().resolve()
            if args.inference_manifest is not None
            else paths["intake-inference"]
        )
        created = {
            variant: infer_variant(
                destination / "models" / variant / "bundle.json",
                inference_manifest,
                cache,
                args.analysis_output_root.expanduser().resolve(),
                skip_existing=args.skip_existing_analyses,
            )
            for variant in selected_variants
        }
        print(json.dumps({"manifest": str(inference_manifest), "created": created}, indent=2))
        return 0

    bundles: dict[str, Any] = {}
    for variant in selected_variants:
        print(f"Training {variant}", flush=True)
        bundles[variant] = train_variant(
            variant,
            paths[variant],
            args.production_models_root.expanduser().resolve(),
            cache,
            destination / "models",
        )
    for variant, bundle in bundles.items():
        bundle_path = destination / "models" / variant / "bundle.json"
        infer_variant(
            bundle_path,
            paths["intake-inference"],
            cache,
            args.analysis_output_root.expanduser().resolve(),
        )
    summary = {
        "schemaVersion": 1,
        "experiment": EXPERIMENT_ID,
        "createdAt": datetime.now(UTC).isoformat(),
        "configurationSha256": _json_hash(
            {
                "latestGrassTraining": LATEST_GRASS_TRAIN_IDS,
                "latestIndoorTraining": LATEST_INDOOR_TRAIN_IDS,
                "indoorEvaluation": INDOOR_EVALUATION_IDS,
                "futureGrassEvaluation": FUTURE_GRASS_EVALUATION_IDS,
                "hardNegativeMultiplier": HARD_NEGATIVE_MULTIPLIER,
            }
        ),
        "bundles": bundles,
        "trainedVariants": list(selected_variants),
    }
    _write_new_json(destination / "experiment.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
