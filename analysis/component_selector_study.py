"""Leakage-safe v4/v5 component-selector development study.

The study has two deliberately separate phases:

``prepare_component_selector_oof``
    Fits fresh fold-specific rally and serve heads from the existing v2/v3 warm
    feature caches.  Full-development v4/v5 artifacts are configuration donors
    only; their predictions and weights never enter component rows.  For selector
    training rows, every generator also excludes every validation source group.

``run_component_selector_development``
    Fits the selector from gold action targets on training OOF rows, then compares
    it with v4 and the already frozen v4/v5 intersection rule on validation.  The
    validation labels never fit the selector or any generator.

Heuristic candidates and the local end refiner are intentionally omitted: neither
currently has a safe source-group-OOF regeneration path.  Uncovered additions and
union actions remain disabled, matching the preceding fusion experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .artifacts import atomic_write_text
from .component_selector import (
    ACTION_ORDER,
    FROZEN_INTERSECTION_CONFIG,
    INTERSECTION,
    KEEP_V4,
    BoundaryProvenance,
    CandidateFeatureRow,
    ComponentFeatureRow,
    ComponentSelectorModel,
    GeneratorProvenance,
    IntervalCandidate,
    ProposedInterval,
    RawScoreSummary,
    SelectorTarget,
    TransitionQualitySummary,
    build_component_rows,
    candidate_action_rows,
    fit_component_selector,
    frozen_intersection_action,
    materialize_selected,
    validate_oof_rows,
)
from .config import DecoderConfig, TrainingConfig
from .decoder import DecodedInterval
from .dual_serve_fusion_experiment import _clip_decoded
from .feature_experiments import objective, sha256_file
from .metrics import (
    aggregate_evaluations,
    aggregate_outcome_slices,
    evaluate_intervals,
    outcome_slice_metrics,
)
from .model import (
    RALLY_LIVE_TASK,
    SERVE_CONTACT_TASK,
    LogisticModel,
    ModelError,
    load_model,
    train_logistic_model,
)
from .pipeline import PreparedRecording, _manifest_digest, _prepare_many, _tune_decoder
from .schema import DatasetManifest, Interval, ManifestError, Recording, load_manifest
from .serve import serve_labels_for_times
from .serve_experiment import (
    FULL_SERVE_INPUT_PROFILE,
    _effective_analysis_fps,
    _masked_serve_values,
    _prediction_inputs,
    _serve_input_mask,
    _tune_composition,
    _validate_model_pair,
    _validate_serve_validation_targets,
)
from .version import __version__


STUDY_SCHEMA_VERSION = 1
EXPERIMENT_ID = "component-selector-oof-v1"
CACHE_FILENAME = "oof-components.json"
DEVELOPMENT_KIND = "volleycut-component-selector-development"
CACHE_KIND = "volleycut-component-selector-oof-cache"
GENERATOR_FAMILIES = ("v4", "v5")
HEAD_ROLES = ("rally", "serve")
DEFAULT_SELECTOR_L2 = 0.1


def _code_provenance() -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    tracked = (
        package / "component_selector_study.py",
        package / "component_selector.py",
        package / "dual_serve_fusion_experiment.py",
        package / "serve_experiment.py",
        package / "pipeline.py",
        package / "model.py",
        package / "metrics.py",
        package / "decoder.py",
    )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package.parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=package.parent,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        head, dirty = None, None
    return {
        "gitHead": head,
        "gitDirty": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "filesSha256": {
            str(path.relative_to(package.parent)): sha256_file(path)
            for path in tracked
            if path.is_file()
        },
    }


class ComponentSelectorStudyError(ValueError):
    """Raised when study lineage or split isolation is unsafe."""


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(
        "\0".join((str(seed), *parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


@dataclass(frozen=True, order=True)
class GeneratorFoldPlan:
    """One candidate-generator fold and every source group it consumes."""

    held_out_group: str
    role: str
    training_groups: tuple[str, ...]
    inner_validation_group: str
    excluded_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.role not in {"selector-training-oof", "selector-assessment"}:
            raise ValueError(f"unsupported generator fold role: {self.role!r}")
        training = tuple(sorted(set(self.training_groups)))
        excluded = tuple(sorted(set(self.excluded_groups)))
        object.__setattr__(self, "training_groups", training)
        object.__setattr__(self, "excluded_groups", excluded)
        if not self.held_out_group or not self.inner_validation_group or not training:
            raise ValueError("generator folds require held-out, train, and validation groups")
        consumed = set(training) | {self.inner_validation_group}
        if self.held_out_group in consumed:
            raise ComponentSelectorStudyError(
                f"generator fold consumes held-out group {self.held_out_group!r}"
            )
        if consumed & set(excluded):
            raise ComponentSelectorStudyError(
                "generator fold consumes a declared excluded source group"
            )

    @property
    def consumed_groups(self) -> tuple[str, ...]:
        return tuple(sorted((*self.training_groups, self.inner_validation_group)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "heldOutSourceGroup": self.held_out_group,
            "role": self.role,
            "trainingSourceGroups": list(self.training_groups),
            "innerValidationSourceGroup": self.inner_validation_group,
            "consumedSourceGroups": list(self.consumed_groups),
            "excludedSourceGroups": list(self.excluded_groups),
        }


def _rotating_choice(groups: Sequence[str], anchor: str) -> str:
    ordered = tuple(sorted(groups))
    if not ordered:
        raise ValueError("cannot choose from an empty source-group set")
    digest = hashlib.sha256(anchor.encode("utf-8")).digest()
    return ordered[int.from_bytes(digest[:4], "big") % len(ordered)]


def build_generator_fold_plan(
    recordings: Sequence[Recording],
) -> tuple[GeneratorFoldPlan, ...]:
    """Build the small nested plan used to create selector inputs.

    Training-row generators exclude both their own source group and all validation
    groups.  Validation-row generators consume training groups only.  Consequently
    validation cannot enter the selector through a base-model parameter fitted for
    some other selector-training row.
    """

    development = tuple(
        row for row in recordings if row.split in {"train", "validation"}
    )
    train_groups = tuple(
        sorted({row.source_group for row in development if row.split == "train"})
    )
    validation_groups = tuple(
        sorted(
            {row.source_group for row in development if row.split == "validation"}
        )
    )
    if len(train_groups) < 3:
        raise ComponentSelectorStudyError(
            "component selector OOF preparation requires at least three training "
            "source groups"
        )
    if not validation_groups:
        raise ComponentSelectorStudyError(
            "component selector assessment requires a validation source group"
        )
    plans: list[GeneratorFoldPlan] = []
    for held_out in train_groups:
        remaining = tuple(group for group in train_groups if group != held_out)
        inner_validation = _rotating_choice(remaining, f"train:{held_out}")
        training = tuple(group for group in remaining if group != inner_validation)
        plans.append(
            GeneratorFoldPlan(
                held_out_group=held_out,
                role="selector-training-oof",
                training_groups=training,
                inner_validation_group=inner_validation,
                excluded_groups=tuple(sorted((*validation_groups, held_out))),
            )
        )
    for held_out in validation_groups:
        inner_validation = _rotating_choice(train_groups, f"validation:{held_out}")
        training = tuple(group for group in train_groups if group != inner_validation)
        plans.append(
            GeneratorFoldPlan(
                held_out_group=held_out,
                role="selector-assessment",
                training_groups=training,
                inner_validation_group=inner_validation,
                excluded_groups=validation_groups,
            )
        )
    return tuple(sorted(plans, key=lambda row: (row.role, row.held_out_group)))


def _training_config_from_model(
    model: LogisticModel,
    *,
    epoch_cap: int | None,
    seed: int,
    family: str,
    head: str,
    held_out_group: str,
) -> TrainingConfig:
    raw = model.training_summary.get("config", {})
    try:
        config = TrainingConfig(**dict(raw)) if raw else TrainingConfig()
    except (TypeError, ValueError) as error:
        raise ComponentSelectorStudyError(
            f"{family} {head} template has invalid training configuration: {error}"
        ) from error
    epochs = config.epochs if epoch_cap in (None, 0) else min(config.epochs, epoch_cap)
    result = replace(
        config,
        epochs=epochs,
        patience=min(config.patience, epochs),
        seed=_stable_seed(seed, family, head, held_out_group),
    )
    result.validate()
    return result


def _validate_prepared_signature(
    prepared: Sequence[PreparedRecording], feature_names: tuple[str, ...]
) -> None:
    if not prepared:
        raise ComponentSelectorStudyError("generator fold has no prepared recordings")
    for item in prepared:
        if item.contextual_names != feature_names:
            raise ComponentSelectorStudyError(
                f"feature signature mismatch for {item.recording.id}"
            )


def _fit_rally_head(
    training: Sequence[PreparedRecording],
    validation: Sequence[PreparedRecording],
    template: LogisticModel,
    config: TrainingConfig,
    *,
    plan: GeneratorFoldPlan,
    family: str,
    manifest: DatasetManifest,
    manifest_sha256: str,
    destination: Path,
) -> tuple[LogisticModel, dict[str, Any]]:
    _validate_prepared_signature((*training, *validation), template.feature_names)
    validation_labels = [item.labels[item.sample_mask] for item in validation]
    positives = sum(int(np.sum(labels > 0.5)) for labels in validation_labels)
    samples = sum(len(labels) for labels in validation_labels)
    if not samples or positives in {0, samples}:
        raise ManifestError(
            f"rally inner-validation group {plan.inner_validation_group!r} must "
            "contain live and dead samples"
        )
    model = train_logistic_model(
        [item.contextual_values[item.sample_mask] for item in training],
        [item.labels[item.sample_mask] for item in training],
        [item.contextual_values[item.sample_mask] for item in validation],
        validation_labels,
        template.feature_config,
        template.feature_names,
        DecoderConfig(),
        config,
        prediction_task=RALLY_LIVE_TASK,
    )
    decoder, selection = _tune_decoder(validation, model, DecoderConfig())
    model.decoder = decoder
    model.training_summary.update(
        {
            "dataset": manifest.name,
            "manifestSha256": manifest_sha256,
            "study": EXPERIMENT_ID,
            "family": family,
            "role": "source-group-oof-candidate-generator",
            "heldOutSourceGroup": plan.held_out_group,
            "trainingRecordingIds": sorted(item.recording.id for item in training),
            "validationRecordingIds": sorted(
                item.recording.id for item in validation
            ),
            "trainingSourceGroups": list(plan.training_groups),
            "validationSourceGroups": [plan.inner_validation_group],
            "excludedSourceGroups": list(plan.excluded_groups),
            "decoderSelection": selection,
        }
    )
    model.save(destination)
    return model, selection


def _serve_target_radius(template: LogisticModel) -> float:
    target = template.training_summary.get("serveTarget", {})
    try:
        radius = float(target.get("radiusSeconds", 1.0))
    except (AttributeError, TypeError, ValueError) as error:
        raise ComponentSelectorStudyError(
            f"serve template has invalid target radius: {error}"
        ) from error
    if not math.isfinite(radius) or radius < 0.0:
        raise ComponentSelectorStudyError("serve target radius must be non-negative")
    return radius


def _serve_input_profile(template: LogisticModel) -> str:
    payload = template.training_summary.get("serveInputProfile")
    if payload is None:
        return FULL_SERVE_INPUT_PROFILE
    if not isinstance(payload, Mapping) or not isinstance(payload.get("id"), str):
        raise ComponentSelectorStudyError("serve template input profile is invalid")
    return str(payload["id"])


def _fit_serve_head(
    training: Sequence[PreparedRecording],
    validation: Sequence[PreparedRecording],
    rally_model: LogisticModel,
    template: LogisticModel,
    config: TrainingConfig,
    *,
    plan: GeneratorFoldPlan,
    family: str,
    manifest: DatasetManifest,
    manifest_sha256: str,
    destination: Path,
) -> tuple[LogisticModel, dict[str, Any]]:
    radius = _serve_target_radius(template)
    profile = _serve_input_profile(template)
    retained = _serve_input_mask(rally_model.feature_names, profile)

    def labels(item: PreparedRecording) -> np.ndarray:
        return serve_labels_for_times(item.sequence.times, item.recording.rallies, radius)

    train_labels = [labels(item)[item.sample_mask] for item in training]
    validation_labels = [labels(item)[item.sample_mask] for item in validation]
    _validate_serve_validation_targets(validation_labels)
    model = train_logistic_model(
        [
            _masked_serve_values(
                item.contextual_values[item.sample_mask], retained
            )
            for item in training
        ],
        train_labels,
        [
            _masked_serve_values(
                item.contextual_values[item.sample_mask], retained
            )
            for item in validation
        ],
        validation_labels,
        rally_model.feature_config,
        rally_model.feature_names,
        rally_model.decoder,
        config,
        prediction_task=SERVE_CONTACT_TASK,
    )
    model.feature_version = rally_model.feature_version
    model.training_summary.update(
        {
            "dataset": manifest.name,
            "manifestSha256": manifest_sha256,
            "study": EXPERIMENT_ID,
            "family": family,
            "role": "source-group-oof-candidate-generator",
            "heldOutSourceGroup": plan.held_out_group,
            "rallyModelSha256": rally_model.artifact_sha256,
            "trainingRecordingIds": sorted(item.recording.id for item in training),
            "validationRecordingIds": sorted(
                item.recording.id for item in validation
            ),
            "trainingSourceGroups": list(plan.training_groups),
            "validationSourceGroups": [plan.inner_validation_group],
            "excludedSourceGroups": list(plan.excluded_groups),
            "serveTarget": {
                "id": "serve-contact-window-v1",
                "radiusSeconds": radius,
                "contactDefinition": "rally start under serve-contact-to-dead-ball-v1",
            },
            "serveInputProfile": {
                "id": profile,
                "retainedInputs": int(np.sum(retained)),
                "removedInputs": int(np.sum(~retained)),
            },
        }
    )
    serve_decoder, composition, selection = _tune_composition(
        validation, rally_model, model
    )
    model.training_summary.update(
        {
            "serveDecoder": serve_decoder.to_dict(),
            "composition": composition.to_dict(),
            "selection": selection,
        }
    )
    model.save(destination)
    return model, selection


@dataclass(frozen=True)
class FamilyFoldOutput:
    family: str
    plan: GeneratorFoldPlan
    rally_model: LogisticModel
    serve_model: LogisticModel
    held_prepared: tuple[PreparedRecording, ...]
    composed: Mapping[str, tuple[DecodedInterval, ...]]
    rally_scores: Mapping[str, np.ndarray]
    serve_scores: Mapping[str, np.ndarray]
    generators: tuple[GeneratorProvenance, ...]
    metadata: Mapping[str, Any]


def _fit_family_fold(
    prepared: Sequence[PreparedRecording],
    rally_template: LogisticModel,
    serve_template: LogisticModel,
    *,
    family: str,
    plan: GeneratorFoldPlan,
    manifest: DatasetManifest,
    manifest_sha256: str,
    destination: Path,
    epoch_cap: int | None,
    seed: int,
    progress: Callable[[str], None] | None,
) -> FamilyFoldOutput:
    if family not in GENERATOR_FAMILIES:
        raise ValueError(f"unsupported generator family: {family!r}")
    by_group: dict[str, list[PreparedRecording]] = defaultdict(list)
    for item in prepared:
        by_group[item.recording.source_group].append(item)
    training = tuple(
        item for group in plan.training_groups for item in by_group.get(group, ())
    )
    validation = tuple(by_group.get(plan.inner_validation_group, ()))
    held = tuple(by_group.get(plan.held_out_group, ()))
    if not training or not validation or not held:
        raise ComponentSelectorStudyError(
            f"{family} fold {plan.held_out_group!r} has an empty partition"
        )
    if {item.recording.source_group for item in (*training, *validation)} & {
        plan.held_out_group,
        *plan.excluded_groups,
    }:
        raise ComponentSelectorStudyError(
            f"{family} fold {plan.held_out_group!r} leaks an excluded group"
        )
    destination.mkdir(parents=True)
    if progress is not None:
        progress(
            f"Fitting {family} rally head; hold out {plan.held_out_group}, "
            f"tune on {plan.inner_validation_group}"
        )
    rally_config = _training_config_from_model(
        rally_template,
        epoch_cap=epoch_cap,
        seed=seed,
        family=family,
        head="rally",
        held_out_group=plan.held_out_group,
    )
    rally, rally_selection = _fit_rally_head(
        training,
        validation,
        rally_template,
        rally_config,
        plan=plan,
        family=family,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        destination=destination / "rally",
    )
    if progress is not None:
        progress(
            f"Fitting {family} serve head; hold out {plan.held_out_group}, "
            f"tune on {plan.inner_validation_group}"
        )
    serve_config = _training_config_from_model(
        serve_template,
        epoch_cap=epoch_cap,
        seed=seed,
        family=family,
        head="serve",
        held_out_group=plan.held_out_group,
    )
    serve, serve_selection = _fit_serve_head(
        training,
        validation,
        rally,
        serve_template,
        serve_config,
        plan=plan,
        family=family,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        destination=destination / "serve",
    )
    serve_decoder, composition = _validate_model_pair(
        rally, serve, manifest_sha256=manifest_sha256
    )
    composed: dict[str, tuple[DecodedInterval, ...]] = {}
    rally_scores: dict[str, np.ndarray] = {}
    serve_scores: dict[str, np.ndarray] = {}
    for item in sorted(held, key=lambda row: row.recording.id):
        prediction = _prediction_inputs(
            item, rally, serve, serve_decoder, composition
        )
        composed[item.recording.id] = tuple(
            _clip_decoded(prediction.composed, item.recording.ignored_intervals)
        )
        rally_scores[item.recording.id] = rally.predict(item.contextual_values)
        serve_scores[item.recording.id] = serve.predict(item.contextual_values)

    consumed = plan.consumed_groups
    rally_provenance = GeneratorProvenance(
        f"{family}-rally", str(rally.artifact_sha256), consumed
    )
    serve_provenance = GeneratorProvenance(
        f"{family}-serve", str(serve.artifact_sha256), consumed
    )
    composition_payload = {
        "experiment": EXPERIMENT_ID,
        "family": family,
        "rallySha256": rally.artifact_sha256,
        "serveSha256": serve.artifact_sha256,
        "rallyDecoder": rally.decoder.to_dict(),
        "serveDecoder": serve_decoder.to_dict(),
        "composition": composition.to_dict(),
    }
    composition_provenance = GeneratorProvenance(
        f"{family}-composed", _sha256_payload(composition_payload), consumed
    )
    for generator in (
        rally_provenance,
        serve_provenance,
        composition_provenance,
    ):
        generator.validate_held_out(plan.held_out_group)
    return FamilyFoldOutput(
        family=family,
        plan=plan,
        rally_model=rally,
        serve_model=serve,
        held_prepared=tuple(sorted(held, key=lambda row: row.recording.id)),
        composed=composed,
        rally_scores=rally_scores,
        serve_scores=serve_scores,
        generators=(rally_provenance, serve_provenance, composition_provenance),
        metadata={
            "family": family,
            "fold": plan.to_dict(),
            "rally": {
                "path": f"{family}/rally",
                "artifactSha256": rally.artifact_sha256,
                "featureVersion": rally.feature_version,
                "trainingConfig": rally_config.to_dict(),
                "decoderSelection": rally_selection,
            },
            "serve": {
                "path": f"{family}/serve",
                "artifactSha256": serve.artifact_sha256,
                "featureVersion": serve.feature_version,
                "trainingConfig": serve_config.to_dict(),
                "compositionSelection": serve_selection,
            },
            "compositionSha256": composition_provenance.artifact_sha256,
            "generators": [row.to_dict() for row in (
                rally_provenance,
                serve_provenance,
                composition_provenance,
            )],
        },
    )


def _nearest_value(times: np.ndarray, values: np.ndarray, timestamp: float) -> float:
    index = int(np.argmin(np.abs(times - timestamp)))
    return float(values[index])


def _window_values(
    times: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
) -> np.ndarray:
    mask = (times >= start) & (times <= end)
    if np.any(mask):
        return values[mask]
    midpoint = (start + end) / 2.0
    return np.asarray([_nearest_value(times, values, midpoint)], dtype=np.float64)


def _window_mean(
    times: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
) -> float:
    return float(np.mean(_window_values(times, values, start, end)))


def _score_summaries(
    times: np.ndarray,
    scores: Mapping[str, np.ndarray],
    start: float,
    end: float,
) -> tuple[RawScoreSummary, ...]:
    summaries: list[RawScoreSummary] = []
    for key in sorted(scores):
        model_id, signal = key.rsplit(":", 1)
        values = np.asarray(scores[key], dtype=np.float64)
        if values.shape != times.shape or not np.isfinite(values).all():
            raise ComponentSelectorStudyError(
                f"score series {key!r} is not finite and aligned"
            )
        selected = _window_values(times, values, start, end)
        summaries.append(
            RawScoreSummary.from_values(
                model_id,
                signal,
                tuple(float(value) for value in selected),
                at_start=_nearest_value(times, values, start),
                at_end=_nearest_value(times, values, end),
            )
        )
    return tuple(summaries)


def _quality_channel(
    prepared: PreparedRecording, name: str, start: float, end: float
) -> float | None:
    try:
        index = prepared.sequence.names.index(name)
    except ValueError:
        return None
    return _window_mean(
        prepared.sequence.times,
        prepared.sequence.values[:, index],
        start,
        end,
    )


def _transition_quality(
    prepared: PreparedRecording,
    scores: Mapping[str, np.ndarray],
    start: float,
    end: float,
) -> TransitionQualitySummary:
    times = prepared.sequence.times
    live_channels = [
        np.asarray(values, dtype=np.float64)
        for key, values in sorted(scores.items())
        if key.endswith(":rally")
    ]
    if not live_channels:
        raise ComponentSelectorStudyError("transition summary requires rally scores")
    live = np.mean(np.vstack(live_channels), axis=0)
    visibility = _quality_channel(prepared, "visibility_quality", start, end)
    focus = _quality_channel(prepared, "focus_quality", start, end)
    audio = _quality_channel(prepared, "audio_available", start, end)
    visibility_value = 0.0 if visibility is None else visibility
    focus_value = visibility_value if focus is None else focus
    audio_value = 0.0 if audio is None else audio
    return TransitionQualitySummary(
        start_before=_window_mean(times, live, start - 1.0, start),
        start_after=_window_mean(times, live, start, start + 1.0),
        end_before=_window_mean(times, live, end - 1.0, end),
        end_after=_window_mean(times, live, end, end + 1.0),
        end_persistence=1.0
        - _window_mean(times, live, end, end + 2.0),
        visibility=visibility_value,
        audio_availability=min(1.0, max(0.0, audio_value)),
        camera_quality=math.sqrt(
            max(0.0, visibility_value) * max(0.0, focus_value)
        ),
    )


@dataclass(frozen=True)
class StudyComponentRow:
    component: ComponentFeatureRow
    actions: tuple[CandidateFeatureRow, ...]

    def __post_init__(self) -> None:
        expected = (
            self.component.recording_id,
            self.component.source_group,
            self.component.component_id,
        )
        if not self.actions:
            raise ValueError("study component must have at least one action")
        for row in self.actions:
            if (row.recording_id, row.source_group, row.component_id) != expected:
                raise ValueError("study component actions are not aligned")
        if tuple(row.action for row in self.actions) != tuple(
            action for action in ACTION_ORDER if action in {row.action for row in self.actions}
        ):
            raise ValueError("study component actions are not deterministically ordered")

    def validate_oof(self) -> None:
        self.component.validate_oof()
        validate_oof_rows(self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component.to_dict(),
            "actions": [row.to_dict() for row in self.actions],
        }


def build_recording_study_rows(
    prepared: PreparedRecording,
    v4_intervals: Sequence[DecodedInterval],
    v5_intervals: Sequence[DecodedInterval],
    *,
    v4_rally_scores: np.ndarray,
    v4_serve_scores: np.ndarray,
    v5_rally_scores: np.ndarray,
    v5_serve_scores: np.ndarray,
    generators: Sequence[GeneratorProvenance],
) -> tuple[StudyComponentRow, ...]:
    """Build deterministic, target-free component/action rows for one recording."""

    recording = prepared.recording
    times = prepared.sequence.times
    score_series = {
        "v4-rally:rally": np.asarray(v4_rally_scores, dtype=np.float64),
        "v4-serve:serve": np.asarray(v4_serve_scores, dtype=np.float64),
        "v5-rally:rally": np.asarray(v5_rally_scores, dtype=np.float64),
        "v5-serve:serve": np.asarray(v5_serve_scores, dtype=np.float64),
    }
    for name, values in score_series.items():
        if values.shape != times.shape:
            raise ComponentSelectorStudyError(
                f"{recording.id}: score series {name!r} is not time-aligned"
            )
    v4_candidates = tuple(
        IntervalCandidate.from_decoded(
            f"{recording.id}:v4:{index:04d}", "v4-composed", interval
        )
        for index, interval in enumerate(
            sorted(v4_intervals, key=lambda row: (row.start, row.end))
        )
    )
    v5_candidates = tuple(
        IntervalCandidate.from_decoded(
            f"{recording.id}:v5:{index:04d}", "v5-composed", interval
        )
        for index, interval in enumerate(
            sorted(v5_intervals, key=lambda row: (row.start, row.end))
        )
    )
    components = build_component_rows(
        v4_candidates,
        v5_candidates,
        recording_id=recording.id,
        source_group=recording.source_group,
        generators=generators,
    )
    result: list[StudyComponentRow] = []
    for component in components:
        component_start, component_end = component.component_envelope
        component = replace(
            component,
            raw_scores=_score_summaries(
                times, score_series, component_start, component_end
            ),
            transition_quality=_transition_quality(
                prepared, score_series, component_start, component_end
            ),
        )
        preliminary = candidate_action_rows(
            component,
            permit_union=False,
            allow_uncovered_additions=False,
        )
        raw_by_action: dict[str, tuple[RawScoreSummary, ...]] = {}
        transition_by_action: dict[str, TransitionQualitySummary] = {}
        for action in preliminary:
            if action.intervals:
                start = min(row.start for row in action.intervals)
                end = max(row.end for row in action.intervals)
            else:
                start, end = component_start, component_end
            raw_by_action[action.action] = _score_summaries(
                times, score_series, start, end
            )
            transition_by_action[action.action] = _transition_quality(
                prepared, score_series, start, end
            )
        actions = candidate_action_rows(
            component,
            permit_union=False,
            allow_uncovered_additions=False,
            raw_scores_by_action=raw_by_action,
            transition_quality_by_action=transition_by_action,
        )
        row = StudyComponentRow(component, actions)
        row.validate_oof()
        result.append(row)
    return tuple(result)


def _boundary_from_dict(payload: Mapping[str, Any]) -> BoundaryProvenance:
    try:
        parents = payload.get("parentCandidateIds", [])
        if not isinstance(parents, list):
            raise TypeError("parentCandidateIds must be an array")
        return BoundaryProvenance(
            str(payload["startGenerator"]),
            str(payload["endGenerator"]),
            str(payload["operation"]),
            tuple(str(row) for row in parents),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ComponentSelectorStudyError(
            f"invalid boundary provenance: {error}"
        ) from error


def _raw_score_from_dict(payload: Mapping[str, Any]) -> RawScoreSummary:
    try:
        return RawScoreSummary(
            model_id=str(payload["modelId"]),
            signal=str(payload["signal"]),
            minimum=float(payload["minimum"]),
            maximum=float(payload["maximum"]),
            mean=float(payload["mean"]),
            at_start=float(payload["atStart"]),
            at_end=float(payload["atEnd"]),
            sample_count=int(payload["sampleCount"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ComponentSelectorStudyError(f"invalid raw score summary: {error}") from error


def _transition_from_dict(payload: Mapping[str, Any]) -> TransitionQualitySummary:
    try:
        return TransitionQualitySummary(
            start_before=float(payload["startBefore"]),
            start_after=float(payload["startAfter"]),
            end_before=float(payload["endBefore"]),
            end_after=float(payload["endAfter"]),
            end_persistence=float(payload["endPersistence"]),
            visibility=float(payload["visibility"]),
            audio_availability=float(payload["audioAvailability"]),
            camera_quality=float(payload["cameraQuality"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ComponentSelectorStudyError(
            f"invalid transition/quality summary: {error}"
        ) from error


def _interval_candidate_from_dict(payload: Mapping[str, Any]) -> IntervalCandidate:
    try:
        provenance = payload["boundaryProvenance"]
        if not isinstance(provenance, Mapping):
            raise TypeError("boundaryProvenance must be an object")
        return IntervalCandidate(
            candidate_id=str(payload["candidateId"]),
            generator_id=str(payload["generatorId"]),
            start=float(payload["start"]),
            end=float(payload["end"]),
            confidence=float(payload["confidence"]),
            boundary_provenance=_boundary_from_dict(provenance),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ComponentSelectorStudyError(
            f"invalid interval candidate: {error}"
        ) from error


def _proposed_from_dict(payload: Mapping[str, Any]) -> ProposedInterval:
    try:
        provenance = payload["boundaryProvenance"]
        if not isinstance(provenance, Mapping):
            raise TypeError("boundaryProvenance must be an object")
        return ProposedInterval(
            start=float(payload["start"]),
            end=float(payload["end"]),
            confidence=float(payload["confidence"]),
            boundary_provenance=_boundary_from_dict(provenance),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ComponentSelectorStudyError(
            f"invalid proposed interval: {error}"
        ) from error


def _mapping_rows(payload: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(payload, list) or any(
        not isinstance(row, Mapping) for row in payload
    ):
        raise ComponentSelectorStudyError(f"{label} must be an object array")
    return payload


def _component_from_dict(payload: Mapping[str, Any]) -> ComponentFeatureRow:
    try:
        transition = payload["transitionQuality"]
        if not isinstance(transition, Mapping):
            raise TypeError("transitionQuality must be an object")
        component = ComponentFeatureRow(
            recording_id=str(payload["recordingId"]),
            source_group=str(payload["sourceGroup"]),
            component_id=str(payload["componentId"]),
            v4=tuple(
                _interval_candidate_from_dict(row)
                for row in _mapping_rows(payload["v4"], "v4")
            ),
            v5=tuple(
                _interval_candidate_from_dict(row)
                for row in _mapping_rows(payload["v5"], "v5")
            ),
            local_refined=tuple(
                _interval_candidate_from_dict(row)
                for row in _mapping_rows(
                    payload.get("localRefined", []), "localRefined"
                )
            ),
            raw_scores=tuple(
                _raw_score_from_dict(row)
                for row in _mapping_rows(payload["rawScores"], "rawScores")
            ),
            transition_quality=_transition_from_dict(transition),
            generators=tuple(
                GeneratorProvenance.from_dict(row)
                for row in _mapping_rows(payload["generators"], "generators")
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ComponentSelectorStudyError(
            f"invalid component feature row: {error}"
        ) from error
    if component.to_dict() != dict(payload):
        raise ComponentSelectorStudyError(
            f"derived component fields changed for {component.component_id!r}"
        )
    return component


def _action_from_dict(payload: Mapping[str, Any]) -> CandidateFeatureRow:
    try:
        cardinality = payload["cardinality"]
        transition = payload["transitionQuality"]
        if not isinstance(cardinality, Mapping) or not isinstance(
            transition, Mapping
        ):
            raise TypeError("cardinality and transitionQuality must be objects")
        action = CandidateFeatureRow(
            recording_id=str(payload["recordingId"]),
            source_group=str(payload["sourceGroup"]),
            component_id=str(payload["componentId"]),
            candidate_id=str(payload["candidateId"]),
            action=str(payload["action"]),
            intervals=tuple(
                _proposed_from_dict(row)
                for row in _mapping_rows(payload["intervals"], "intervals")
            ),
            topology=str(payload["topology"]),
            v4_count=int(cardinality["v4"]),
            v5_count=int(cardinality["v5"]),
            local_refined_count=int(cardinality["localRefined"]),
            raw_scores=tuple(
                _raw_score_from_dict(row)
                for row in _mapping_rows(payload["rawScores"], "rawScores")
            ),
            start_disagreement_seconds=float(
                payload["startDisagreementSeconds"]
            ),
            end_disagreement_seconds=float(payload["endDisagreementSeconds"]),
            component_duration_seconds=float(payload["componentDurationSeconds"]),
            duration_seconds=float(payload["durationSeconds"]),
            containment_by_v4=float(payload["containmentByV4"]),
            containment_by_v5=float(payload["containmentByV5"]),
            coverage_of_v4=float(payload["coverageOfV4"]),
            coverage_of_v5=float(payload["coverageOfV5"]),
            transition_quality=_transition_from_dict(transition),
            boundary_provenance=tuple(
                _boundary_from_dict(row)
                for row in _mapping_rows(
                    payload["boundaryProvenance"], "boundaryProvenance"
                )
            ),
            generators=tuple(
                GeneratorProvenance.from_dict(row)
                for row in _mapping_rows(payload["generators"], "generators")
            ),
            uncovered_addition=bool(payload["uncoveredAddition"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ComponentSelectorStudyError(
            f"invalid candidate action row: {error}"
        ) from error
    if action.to_dict() != dict(payload):
        raise ComponentSelectorStudyError(
            f"derived action fields changed for {action.candidate_id!r}"
        )
    return action


def _study_component_from_dict(payload: Mapping[str, Any]) -> StudyComponentRow:
    try:
        component_payload = payload["component"]
        if not isinstance(component_payload, Mapping):
            raise TypeError("component must be an object")
        result = StudyComponentRow(
            _component_from_dict(component_payload),
            tuple(
                _action_from_dict(row)
                for row in _mapping_rows(payload["actions"], "actions")
            ),
        )
        result.validate_oof()
        return result
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ComponentSelectorStudyError):
            raise
        raise ComponentSelectorStudyError(
            f"invalid cached study component: {error}"
        ) from error


def _load_templates(
    manifest_sha256: str,
    paths: Mapping[str, tuple[str | Path, str | Path]],
) -> dict[str, tuple[LogisticModel, LogisticModel]]:
    result: dict[str, tuple[LogisticModel, LogisticModel]] = {}
    for family in GENERATOR_FAMILIES:
        rally_path, serve_path = paths[family]
        rally = load_model(rally_path)
        serve = load_model(serve_path)
        _validate_model_pair(rally, serve, manifest_sha256=manifest_sha256)
        result[family] = rally, serve
    if result["v4"][0].feature_config == result["v5"][0].feature_config:
        raise ComponentSelectorStudyError(
            "v4 and v5 templates unexpectedly use the same feature configuration"
        )
    return result


def _validate_cache_destination(destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing OOF cache: {destination}"
        )


def prepare_component_selector_oof(
    manifest_path: str | Path,
    v4_rally_template_path: str | Path,
    v4_serve_template_path: str | Path,
    v4_cache_dir: str | Path,
    v5_rally_template_path: str | Path,
    v5_serve_template_path: str | Path,
    v5_cache_dir: str | Path,
    output_dir: str | Path,
    *,
    epoch_cap: int | None = 60,
    seed: int = 7,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate source-group-OOF v4/v5 candidates without full-model predictions."""

    destination = Path(output_dir).expanduser().resolve()
    _validate_cache_destination(destination)
    if epoch_cap is not None and epoch_cap < 0:
        raise ValueError("epoch_cap cannot be negative")
    started = time.perf_counter()
    manifest = load_manifest(manifest_path)
    manifest_sha256 = _manifest_digest(manifest)
    development = tuple(
        row for row in manifest.recordings if row.split in {"train", "validation"}
    )
    if not development:
        raise ComponentSelectorStudyError("manifest has no development recordings")
    plans = build_generator_fold_plan(manifest.recordings)
    plan_by_group = {row.held_out_group: row for row in plans}
    expected_groups = {row.source_group for row in development}
    if set(plan_by_group) != expected_groups:
        raise ComponentSelectorStudyError(
            "generator fold plan does not cover every development source group"
        )
    template_paths = {
        "v4": (v4_rally_template_path, v4_serve_template_path),
        "v5": (v5_rally_template_path, v5_serve_template_path),
    }
    templates = _load_templates(manifest_sha256, template_paths)
    if progress is not None:
        progress(
            f"Loading v4 warm features for {len(development)} development "
            "recordings; test is not prepared"
        )
    prepared_v4 = _prepare_many(
        development,
        templates["v4"][0].feature_config,
        v4_cache_dir,
        progress=progress,
    )
    if progress is not None:
        progress(
            f"Loading v5 warm features for {len(development)} development "
            "recordings; test is not prepared"
        )
    prepared_v5 = _prepare_many(
        development,
        templates["v5"][0].feature_config,
        v5_cache_dir,
        progress=progress,
    )
    v4_by_id = {row.recording.id: row for row in prepared_v4}
    v5_by_id = {row.recording.id: row for row in prepared_v5}
    expected_ids = {row.id for row in development}
    if set(v4_by_id) != expected_ids or set(v5_by_id) != expected_ids:
        raise ComponentSelectorStudyError(
            "prepared feature sets do not exactly cover development recordings"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        fold_metadata: list[dict[str, Any]] = []
        recording_payloads: list[dict[str, Any]] = []
        for fold_index, plan in enumerate(plans):
            fold_root = temporary / "models" / f"fold-{fold_index:02d}"
            outputs: dict[str, FamilyFoldOutput] = {}
            for family, prepared in (
                ("v4", prepared_v4),
                ("v5", prepared_v5),
            ):
                rally_template, serve_template = templates[family]
                outputs[family] = _fit_family_fold(
                    prepared,
                    rally_template,
                    serve_template,
                    family=family,
                    plan=plan,
                    manifest=manifest,
                    manifest_sha256=manifest_sha256,
                    destination=fold_root / family,
                    epoch_cap=epoch_cap,
                    seed=seed,
                    progress=progress,
                )
            v4_output, v5_output = outputs["v4"], outputs["v5"]
            fold_metadata.append(
                {
                    "foldIndex": fold_index,
                    "modelRoot": f"models/fold-{fold_index:02d}",
                    "plan": plan.to_dict(),
                    "families": {
                        family: dict(outputs[family].metadata)
                        for family in GENERATOR_FAMILIES
                    },
                }
            )
            held_recordings = sorted(
                (
                    row
                    for row in development
                    if row.source_group == plan.held_out_group
                ),
                key=lambda row: row.id,
            )
            generators = tuple(
                sorted((*v4_output.generators, *v5_output.generators))
            )
            for recording in held_recordings:
                item_v4, item_v5 = v4_by_id[recording.id], v5_by_id[recording.id]
                if (
                    item_v4.sequence.times.shape != item_v5.sequence.times.shape
                    or not np.allclose(
                        item_v4.sequence.times,
                        item_v5.sequence.times,
                        rtol=0.0,
                        atol=1e-9,
                    )
                ):
                    raise ComponentSelectorStudyError(
                        f"v4/v5 time grids differ for {recording.id}"
                    )
                rows = build_recording_study_rows(
                    item_v4,
                    v4_output.composed[recording.id],
                    v5_output.composed[recording.id],
                    v4_rally_scores=v4_output.rally_scores[recording.id],
                    v4_serve_scores=v4_output.serve_scores[recording.id],
                    v5_rally_scores=v5_output.rally_scores[recording.id],
                    v5_serve_scores=v5_output.serve_scores[recording.id],
                    generators=generators,
                )
                recording_payloads.append(
                    {
                        "recordingId": recording.id,
                        "sourceGroup": recording.source_group,
                        "split": recording.split,
                        "contentSha256": recording.content_sha256,
                        "generatorFoldIndex": fold_index,
                        "components": [row.to_dict() for row in rows],
                    }
                )

        payload: dict[str, Any] = {
            "schemaVersion": STUDY_SCHEMA_VERSION,
            "kind": CACHE_KIND,
            "experiment": EXPERIMENT_ID,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "producer": f"volleycut-analysis/{__version__}",
            "manifest": str(manifest.path),
            "manifestFileSha256": sha256_file(manifest.path),
            "manifestSnapshotSha256": manifest_sha256,
            "recordingContentSha256": {
                row.id: row.content_sha256 for row in manifest.recordings
            },
            "development": {
                "recordingIds": sorted(expected_ids),
                "trainingSourceGroups": sorted(
                    {row.source_group for row in development if row.split == "train"}
                ),
                "validationSourceGroups": sorted(
                    {
                        row.source_group
                        for row in development
                        if row.split == "validation"
                    }
                ),
            },
            "protected": {
                split: {
                    "recordingIds": sorted(
                        row.id for row in manifest.recordings if row.split == split
                    ),
                    "sourceGroups": sorted(
                        {
                            row.source_group
                            for row in manifest.recordings
                            if row.split == split
                        }
                    ),
                }
                for split in ("test", "challenge")
            },
            "testLabelsUsed": False,
            "testRecordingsPrepared": False,
            "protocol": {
                "selectorTrainingRows": (
                    "training-sourceGroup OOF; all validation source groups excluded "
                    "from every candidate generator"
                ),
                "selectorAssessmentRows": (
                    "validation source groups held out from rally, serve, decoder, "
                    "and composition fitting"
                ),
                "fullDevelopmentTemplatePredictionsUsed": False,
                "templatesAreConfigurationDonorsOnly": True,
                "epochCap": epoch_cap,
                "seed": seed,
                "uncoveredAdditions": "disabled",
                "union": "disabled",
            },
            "templates": {
                family: {
                    "rallyPath": str(Path(template_paths[family][0]).resolve()),
                    "rallySha256": templates[family][0].artifact_sha256,
                    "servePath": str(Path(template_paths[family][1]).resolve()),
                    "serveSha256": templates[family][1].artifact_sha256,
                    "featureVersion": templates[family][0].feature_version,
                    "featureConfig": templates[family][0].feature_config.to_dict(),
                    "weightsUsedForPredictions": False,
                }
                for family in GENERATOR_FAMILIES
            },
            "folds": fold_metadata,
            "recordings": sorted(
                recording_payloads, key=lambda row: str(row["recordingId"])
            ),
            "omittedCandidateFamilies": [
                {
                    "family": "heuristic",
                    "reason": "no safe source-group-OOF regeneration path is available",
                },
                {
                    "family": "local-end-refiner",
                    "reason": (
                        "existing artifact was trained/tuned on development groups; "
                        "fold-specific regeneration is not yet implemented"
                    ),
                },
            ],
            "provenance": _code_provenance(),
            "processing": {
                "wallClockSeconds": round(time.perf_counter() - started, 3),
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
        }
        atomic_write_text(
            temporary / CACHE_FILENAME,
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
        )
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing OOF cache: {destination}"
            )
        temporary.replace(destination)
        return payload
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


@dataclass(frozen=True)
class LoadedOOFCache:
    path: Path
    payload: Mapping[str, Any]
    components: tuple[StudyComponentRow, ...]
    recording_splits: Mapping[str, str]
    sha256: str


def _verify_cached_models(cache_root: Path, payload: Mapping[str, Any]) -> None:
    folds = _mapping_rows(payload.get("folds"), "folds")
    for fold in folds:
        root = fold.get("modelRoot")
        families = fold.get("families")
        if (
            not isinstance(root, str)
            or Path(root).is_absolute()
            or ".." in Path(root).parts
            or not isinstance(families, Mapping)
        ):
            raise ComponentSelectorStudyError("cached fold model paths are unsafe")
        for family in GENERATOR_FAMILIES:
            metadata = families.get(family)
            if not isinstance(metadata, Mapping):
                raise ComponentSelectorStudyError(
                    f"cached fold is missing {family} metadata"
                )
            for role in HEAD_ROLES:
                head = metadata.get(role)
                if not isinstance(head, Mapping) or not isinstance(
                    head.get("path"), str
                ):
                    raise ComponentSelectorStudyError(
                        f"cached {family} {role} metadata is invalid"
                    )
                relative = Path(str(head["path"]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ComponentSelectorStudyError("cached model path is unsafe")
                model = load_model(cache_root / str(root) / relative)
                if model.artifact_sha256 != head.get("artifactSha256"):
                    raise ComponentSelectorStudyError(
                        f"cached {family} {role} artifact digest changed"
                    )


def load_component_selector_oof(
    cache_dir: str | Path,
    manifest: DatasetManifest,
    *,
    verify_models: bool = True,
) -> LoadedOOFCache:
    root = Path(cache_dir).expanduser().resolve()
    path = root / CACHE_FILENAME
    try:
        cache_bytes = path.read_bytes()
        payload = json.loads(cache_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ComponentSelectorStudyError(
            f"cannot read component selector OOF cache {path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ComponentSelectorStudyError("OOF cache root must be an object")
    expected = (
        payload.get("schemaVersion") == STUDY_SCHEMA_VERSION
        and payload.get("kind") == CACHE_KIND
        and payload.get("experiment") == EXPERIMENT_ID
        and payload.get("manifestFileSha256") == sha256_file(manifest.path)
        and payload.get("manifestSnapshotSha256") == _manifest_digest(manifest)
        and payload.get("testLabelsUsed") is False
        and payload.get("testRecordingsPrepared") is False
    )
    if not expected:
        raise ComponentSelectorStudyError(
            "OOF cache does not match the immutable unopened-test study inputs"
        )
    cached_files = payload.get("provenance", {}).get("filesSha256")
    if cached_files != _code_provenance()["filesSha256"]:
        raise ComponentSelectorStudyError(
            "component-selector implementation changed after OOF cache preparation"
        )
    expected_content = {
        row.id: row.content_sha256 for row in manifest.recordings
    }
    if payload.get("recordingContentSha256") != expected_content:
        raise ComponentSelectorStudyError(
            "recording content changed after OOF cache preparation"
        )
    development = tuple(
        row for row in manifest.recordings if row.split in {"train", "validation"}
    )
    by_id = {row.id: row for row in development}
    rows = _mapping_rows(payload.get("recordings"), "recordings")
    if {row.get("recordingId") for row in rows} != set(by_id):
        raise ComponentSelectorStudyError(
            "OOF cache does not exactly cover development recordings"
        )
    components: list[StudyComponentRow] = []
    recording_splits: dict[str, str] = {}
    validation_groups = {
        row.source_group for row in development if row.split == "validation"
    }
    for cached in rows:
        recording_id = cached.get("recordingId")
        source = by_id[str(recording_id)]
        if (
            cached.get("sourceGroup") != source.source_group
            or cached.get("split") != source.split
            or cached.get("contentSha256") != source.content_sha256
        ):
            raise ComponentSelectorStudyError(
                f"cached recording lineage changed for {recording_id!r}"
            )
        recording_splits[source.id] = source.split
        parsed = tuple(
            _study_component_from_dict(row)
            for row in _mapping_rows(cached.get("components"), "components")
        )
        if any(row.component.recording_id != source.id for row in parsed):
            raise ComponentSelectorStudyError(
                f"component recording id differs for {source.id!r}"
            )
        if source.split == "train":
            for row in parsed:
                consumed = {
                    group
                    for generator in row.component.generators
                    for group in generator.training_source_groups
                }
                leaked = consumed & validation_groups
                if leaked:
                    raise ComponentSelectorStudyError(
                        "selector-training candidate generators consumed validation "
                        f"groups: {sorted(leaked)}"
                    )
        components.extend(parsed)
    components.sort(
        key=lambda row: (
            row.component.recording_id,
            row.component.component_envelope,
            row.component.component_id,
        )
    )
    if verify_models:
        _verify_cached_models(root, payload)
    return LoadedOOFCache(
        root,
        payload,
        tuple(components),
        recording_splits,
        hashlib.sha256(cache_bytes).hexdigest(),
    )


def _overlap_seconds(first: Interval, start: float, end: float) -> float:
    return max(0.0, min(first.end, end) - max(first.start, start))


def assign_truth_to_components(
    truth: Sequence[Interval], components: Sequence[ComponentFeatureRow]
) -> dict[str, tuple[Interval, ...]]:
    """Assign each overlapping truth interval to at most one component."""

    result: dict[str, list[Interval]] = {
        row.component_id: [] for row in components
    }
    ordered = tuple(
        sorted(
            components,
            key=lambda row: (row.component_envelope, row.component_id),
        )
    )
    for interval in truth:
        eligible: list[tuple[float, float, str, ComponentFeatureRow]] = []
        for component in ordered:
            start, end = component.component_envelope
            overlap = _overlap_seconds(interval, start, end)
            if overlap > 0.0:
                union = max(interval.end, end) - min(interval.start, start)
                eligible.append(
                    (overlap, overlap / union, component.component_id, component)
                )
        if eligible:
            selected = max(
                eligible,
                key=lambda row: (row[0], row[1], tuple(-ord(c) for c in row[2])),
            )[3]
            result[selected.component_id].append(interval)
    return {key: tuple(value) for key, value in result.items()}


def _target_rank(
    action: CandidateFeatureRow, truth: Sequence[Interval]
) -> tuple[Any, ...]:
    predictions = tuple(interval.to_decoded() for interval in action.intervals)
    metrics = evaluate_intervals(truth, predictions)
    return (
        int(metrics["matchedRallies"]),
        float(metrics["eventF1"]),
        int(metrics["matchedRalliesAtIou07"]),
        float(metrics["timeIoU"]),
        float(metrics["liveTimeRecall"]),
        -float(metrics["deadSecondsRetained"]),
        -ACTION_ORDER.index(action.action),
    )


def derive_selector_targets(
    recordings: Mapping[str, Recording],
    rows: Sequence[StudyComponentRow],
) -> tuple[SelectorTarget, ...]:
    """Derive fit-only action targets; no target is written back into feature rows."""

    by_recording: dict[str, list[StudyComponentRow]] = defaultdict(list)
    for row in rows:
        by_recording[row.component.recording_id].append(row)
    targets: list[SelectorTarget] = []
    for recording_id in sorted(by_recording):
        recording = recordings.get(recording_id)
        if recording is None:
            raise ComponentSelectorStudyError(
                f"no truth recording exists for {recording_id!r}"
            )
        components = by_recording[recording_id]
        assigned = assign_truth_to_components(
            recording.rallies, [row.component for row in components]
        )
        for row in sorted(components, key=lambda item: item.component.component_id):
            selected = max(
                row.actions,
                key=lambda action: _target_rank(
                    action, assigned[row.component.component_id]
                ),
            )
            targets.append(
                SelectorTarget(
                    recording_id,
                    row.component.component_id,
                    selected.action,
                )
            )
    return tuple(targets)


def _action_lookup(
    rows: Sequence[StudyComponentRow],
) -> dict[tuple[str, str, str], CandidateFeatureRow]:
    result: dict[tuple[str, str, str], CandidateFeatureRow] = {}
    for row in rows:
        for action in row.actions:
            key = (action.recording_id, action.component_id, action.action)
            if key in result:
                raise ComponentSelectorStudyError(f"duplicate action row {key!r}")
            result[key] = action
    return result


def select_frozen_intersection_actions(
    rows: Sequence[StudyComponentRow],
) -> tuple[CandidateFeatureRow, ...]:
    """Select cached action rows with exact frozen-rule parity."""

    selected: list[CandidateFeatureRow] = []
    for row in sorted(
        rows,
        key=lambda item: (
            item.component.recording_id,
            item.component.component_envelope,
            item.component.component_id,
        ),
    ):
        action = frozen_intersection_action(
            row.component, FROZEN_INTERSECTION_CONFIG
        )
        matching = [candidate for candidate in row.actions if candidate.action == action]
        if len(matching) != 1:
            raise ComponentSelectorStudyError(
                f"component {row.component.component_id!r} lacks frozen action {action!r}"
            )
        selected.append(matching[0])
    return tuple(selected)


def _select_constant_action(
    rows: Sequence[StudyComponentRow], action: str
) -> tuple[CandidateFeatureRow, ...]:
    result: list[CandidateFeatureRow] = []
    for row in rows:
        matching = [candidate for candidate in row.actions if candidate.action == action]
        if matching:
            result.append(matching[0])
        else:
            fallback = [candidate for candidate in row.actions if candidate.action == KEEP_V4]
            if len(fallback) != 1:
                raise ComponentSelectorStudyError(
                    f"component {row.component.component_id!r} lacks keep-v4"
                )
            result.append(fallback[0])
    return tuple(result)


def _evaluate_selected(
    recordings: Sequence[Recording],
    selected: Sequence[CandidateFeatureRow],
) -> dict[str, Any]:
    by_recording: dict[str, list[CandidateFeatureRow]] = defaultdict(list)
    for row in selected:
        by_recording[row.recording_id].append(row)
    per_recording: list[dict[str, Any]] = []
    for recording in sorted(recordings, key=lambda row: row.id):
        intervals = materialize_selected(by_recording.get(recording.id, ()))
        metrics = evaluate_intervals(recording.rallies, intervals)
        metrics["outcomeSlices"] = outcome_slice_metrics(
            recording.rallies, intervals
        )
        metrics.update(
            {
                "id": recording.id,
                "sourceGroup": recording.source_group,
                "environment": recording.environment,
            }
        )
        per_recording.append(metrics)
    aggregate = aggregate_evaluations(per_recording)
    aggregate["outcomeSlices"] = aggregate_outcome_slices(
        [row["outcomeSlices"] for row in per_recording]
    )
    aggregate["objective"] = objective(aggregate)
    return {"aggregate": aggregate, "recordings": per_recording}


def _metric_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "predictedRallies",
        "matchedRallies",
        "matchedRalliesAtIou03",
        "matchedRalliesAtIou07",
        "eventPrecision",
        "eventRecall",
        "eventF1",
        "eventF1AtIou03",
        "eventF1AtIou07",
        "timeIoU",
        "liveTimeRecall",
        "liveTimePrecision",
        "deadSecondsRetained",
    )
    return {key: float(candidate[key]) - float(baseline[key]) for key in keys}


def run_component_selector_development(
    manifest_path: str | Path,
    oof_cache_dir: str | Path,
    *,
    selector_l2: float = DEFAULT_SELECTOR_L2,
    verify_models: bool = True,
) -> dict[str, Any]:
    """Fit on training OOF components and assess once on validation components."""

    if not math.isfinite(selector_l2) or selector_l2 <= 0.0:
        raise ValueError("selector_l2 must be finite and positive")
    started = time.perf_counter()
    manifest = load_manifest(manifest_path)
    cache = load_component_selector_oof(
        oof_cache_dir, manifest, verify_models=verify_models
    )
    recordings = {row.id: row for row in manifest.recordings}
    training_components = tuple(
        row
        for row in cache.components
        if cache.recording_splits[row.component.recording_id] == "train"
    )
    assessment_components = tuple(
        row
        for row in cache.components
        if cache.recording_splits[row.component.recording_id] == "validation"
    )
    if not training_components or not assessment_components:
        raise ComponentSelectorStudyError(
            "OOF cache must contain selector-training and assessment components"
        )
    training_actions = tuple(
        action for row in training_components for action in row.actions
    )
    assessment_actions = tuple(
        action for row in assessment_components for action in row.actions
    )
    validate_oof_rows(training_actions)
    validate_oof_rows(assessment_actions)
    targets = derive_selector_targets(recordings, training_components)
    model = fit_component_selector(
        training_actions,
        targets,
        l2=selector_l2,
        permit_union=False,
        allow_uncovered_additions=False,
    )
    assessment_groups = {
        row.component.source_group for row in assessment_components
    }
    if set(model.training_source_groups) & assessment_groups:
        raise ComponentSelectorStudyError(
            "selector was fitted on an assessment source group"
        )
    for generator in model.generator_provenance:
        leaked = set(generator.training_source_groups) & assessment_groups
        if leaked:
            raise ComponentSelectorStudyError(
                "selector-training generator provenance consumed assessment groups: "
                f"{sorted(leaked)}"
            )
    selector_selected = model.select(assessment_actions)
    frozen_selected = select_frozen_intersection_actions(assessment_components)
    v4_selected = _select_constant_action(assessment_components, KEEP_V4)
    assessment_recording_ids = {
        row.component.recording_id for row in assessment_components
    }
    assessment_recordings = tuple(
        recordings[row_id] for row_id in sorted(assessment_recording_ids)
    )
    reports = {
        "v4": _evaluate_selected(assessment_recordings, v4_selected),
        "frozenIntersection": _evaluate_selected(
            assessment_recordings, frozen_selected
        ),
        "trainedSelector": _evaluate_selected(
            assessment_recordings, selector_selected
        ),
    }
    action_counts = Counter(row.action for row in selector_selected)
    target_counts = Counter(row.selected_action for row in targets)
    frozen_actions = Counter(row.action for row in frozen_selected)
    v4_aggregate = reports["v4"]["aggregate"]
    frozen_aggregate = reports["frozenIntersection"]["aggregate"]
    selector_aggregate = reports["trainedSelector"]["aggregate"]
    return {
        "schemaVersion": STUDY_SCHEMA_VERSION,
        "kind": DEVELOPMENT_KIND,
        "experiment": EXPERIMENT_ID,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "producer": f"volleycut-analysis/{__version__}",
        "freezeStatus": "frozen-development-selection",
        "assessmentRole": "validation-tuning-only-source-group-held-out",
        "testLabelsUsed": False,
        "testRecordingsPrepared": False,
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "manifestSnapshotSha256": _manifest_digest(manifest),
        "oofCache": str(cache.path),
        "oofCacheSha256": cache.sha256,
        "protocol": {
            "candidateGenerators": (
                "fresh fold-specific rally and serve heads; validation excluded "
                "from every selector-training row generator"
            ),
            "selectorFit": "training-split source-group-OOF component rows only",
            "selectorAssessment": (
                "validation component rows from generators that excluded validation"
            ),
            "targets": (
                "fit-only deterministic best available action after assigning each "
                "overlapping truth interval to at most one component"
            ),
            "selectorL2": selector_l2,
            "uncoveredAdditions": "disabled",
            "union": "disabled",
            "fullDevelopmentV4V5PredictionsUsed": False,
        },
        "selector": {
            "model": model.to_dict(),
            "artifactSha256": model.artifact_sha256,
            "trainingComponents": len(training_components),
            "trainingActionRows": len(training_actions),
            "targetActionCounts": dict(sorted(target_counts.items())),
            "assessmentComponents": len(assessment_components),
            "selectedActionCounts": dict(sorted(action_counts.items())),
        },
        "frozenIntersectionReference": {
            "config": FROZEN_INTERSECTION_CONFIG.to_dict(),
            "actionCounts": dict(sorted(frozen_actions.items())),
            "note": (
                "This rule was frozen by the earlier v4/v5 study. It is replayed "
                "unchanged on the new fold-generated candidates."
            ),
        },
        "assessment": {
            "recordingIds": sorted(assessment_recording_ids),
            "sourceGroups": sorted(assessment_groups),
            "variants": reports,
            "trainedSelectorDeltaVsV4": _metric_delta(
                selector_aggregate, v4_aggregate
            ),
            "trainedSelectorDeltaVsFrozenIntersection": _metric_delta(
                selector_aggregate, frozen_aggregate
            ),
        },
        "promotionDecision": {
            "promotable": False,
            "selectedForProduction": "frozenIntersection",
            "trainedSelectorRole": "diagnostic-validation-tuning-only",
            "reason": (
                "The assessment contains one grass-only validation source group, so "
                "it cannot satisfy the predeclared multi-source paired-evidence gate. "
                "Retain the already frozen intersection rule regardless of this "
                "diagnostic metric ordering."
            ),
            "testMetricsConsulted": False,
        },
        "omittedCandidateFamilies": cache.payload.get(
            "omittedCandidateFamilies", []
        ),
        "retrospectiveTestPlan": {
            "implemented": False,
            "reason": (
                "A sound test path first requires full-development v4/v5 generator "
                "refits whose epoch, decoder, and composition choices are frozen from "
                "development without opening test. This runner does not silently reuse "
                "the previously inspected full-development artifacts."
            ),
            "testRemainsUnopenedByThisStudy": True,
        },
        "limitations": [
            "The held-out validation source is grass-only and is tuning evidence, not a final generalization claim.",
            "Only v4/v5 composed candidates are included; heuristic and local-refiner OOF regeneration is unavailable.",
            "Uncovered additions remain disabled because the prior validation audit found no true addition.",
            "Rally and serve sigmoid outputs are uncalibrated score summaries.",
        ],
        "provenance": _code_provenance(),
        "processing": {
            "wallClockSeconds": round(time.perf_counter() - started, 3),
        },
    }


def write_development_report(
    path: str | Path, report: Mapping[str, Any]
) -> Path:
    destination = Path(path).expanduser().resolve()
    return atomic_write_text(
        destination,
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )


__all__ = [
    "CACHE_FILENAME",
    "CACHE_KIND",
    "ComponentSelectorStudyError",
    "DEFAULT_SELECTOR_L2",
    "DEVELOPMENT_KIND",
    "EXPERIMENT_ID",
    "FamilyFoldOutput",
    "GeneratorFoldPlan",
    "LoadedOOFCache",
    "STUDY_SCHEMA_VERSION",
    "StudyComponentRow",
    "assign_truth_to_components",
    "build_generator_fold_plan",
    "build_recording_study_rows",
    "derive_selector_targets",
    "load_component_selector_oof",
    "prepare_component_selector_oof",
    "run_component_selector_development",
    "select_frozen_intersection_actions",
    "write_development_report",
]
