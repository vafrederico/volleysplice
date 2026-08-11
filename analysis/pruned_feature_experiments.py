from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import FEATURE_VERSION, DecoderConfig, FeatureConfig, TrainingConfig
from .crop_evaluation import RecordingIntervals, evaluate_crop_padding
from .feature_experiments import (
    DEVELOPMENT_SPLITS,
    FeatureExperimentError,
    FeatureSet,
    _aggregate_artifacts,
    _fit,
    _fold_report,
    _manifest_digest,
    _padding_report,
    _run_candidate_fold,
    _scored_predictions,
    _select_decoder,
    _subset_prepared,
    build_fold_plan,
    classify_paired_deltas,
    code_provenance,
    objective,
    sha256_file,
)
from .feature_families import base_feature_name, feature_family
from .model import LogisticModel, load_model
from .pipeline import PreparedRecording, _evaluate_prepared_probabilities, _prepare_many
from .schema import DatasetManifest, load_manifest


PRUNING_EXPERIMENT_SCHEMA_VERSION = 1
BASELINE_KIND = "volleycut-feature-experiment-development"
PRUNING_KIND = "volleycut-targeted-pruning-development"
PRUNING_FINAL_KIND = "volleycut-targeted-pruning-final-test"
HARMFUL_FAMILIES = ("audio_onset", "legacy_appearance")
HELPFUL_EXCEPTIONS = ("audio_rms_novelty", "luma_grid_0", "luma_std")
HARMFUL_SIGNALS = ("audio_onset_cadence",)
ZERO_EMBEDDING_PROBABILITY_TOLERANCE = 5e-5


@dataclass(frozen=True)
class PruningVariant:
    name: str
    remove_families: tuple[str, ...] = ()
    keep_base_features: tuple[str, ...] = ()
    remove_base_features: tuple[str, ...] = ()
    rationale: str = ""

    def indexes(self, contextual_names: Sequence[str]) -> tuple[int, ...]:
        if set(self.keep_base_features) & set(self.remove_base_features):
            raise FeatureExperimentError(
                f"variant {self.name!r} both retains and removes a base feature"
            )
        available = {base_feature_name(item) for item in contextual_names}
        requested = set(self.keep_base_features) | set(self.remove_base_features)
        missing = requested - available
        if missing:
            raise FeatureExperimentError(
                f"variant {self.name!r} references missing signals: {sorted(missing)}"
            )
        selected: list[int] = []
        for index, contextual_name in enumerate(contextual_names):
            base = base_feature_name(contextual_name)
            removed = feature_family(contextual_name) in self.remove_families
            if base in self.keep_base_features:
                removed = False
            if base in self.remove_base_features:
                removed = True
            if not removed:
                selected.append(index)
        if not selected:
            raise FeatureExperimentError(
                f"variant {self.name!r} removes every feature column"
            )
        return tuple(selected)

    def to_dict(
        self, contextual_names: Sequence[str], indexes: Sequence[int]
    ) -> dict[str, Any]:
        retained = [contextual_names[index] for index in indexes]
        retained_set = set(indexes)
        removed = [
            name for index, name in enumerate(contextual_names) if index not in retained_set
        ]
        return {
            "name": self.name,
            "rationale": self.rationale,
            "removeFamilies": list(self.remove_families),
            "keepBaseFeatureExceptions": list(self.keep_base_features),
            "removeBaseFeatures": list(self.remove_base_features),
            "retainedFeatureCount": len(retained),
            "removedFeatureCount": len(removed),
            "retainedFeatureNames": retained,
            "removedFeatureNames": removed,
        }


TARGETED_VARIANTS = (
    PruningVariant(
        name="drop_audio_onset_cadence",
        remove_base_features=HARMFUL_SIGNALS,
        rationale=(
            "Conservative removal of only the individually harmful cadence signal."
        ),
    ),
    PruningVariant(
        name="audio_onset_rescue",
        remove_families=("audio_onset",),
        keep_base_features=("audio_rms_novelty",),
        rationale=(
            "Remove the harmful audio-onset family while rescuing its one signal "
            "with helpful held-group permutation evidence."
        ),
    ),
    PruningVariant(
        name="legacy_appearance_rescue",
        remove_families=("legacy_appearance",),
        keep_base_features=("luma_grid_0", "luma_std"),
        rationale=(
            "Remove the harmful legacy-appearance family while rescuing luma_std "
            "and luma_grid_0, which had helpful held-group permutation evidence."
        ),
    ),
    PruningVariant(
        name="targeted_pruned",
        remove_families=HARMFUL_FAMILIES,
        keep_base_features=HELPFUL_EXCEPTIONS,
        remove_base_features=HARMFUL_SIGNALS,
        rationale=(
            "Union of both family rescues plus removal of the individually harmful "
            "cadence signal; this is the primary targeted candidate."
        ),
    ),
)


def build_pruning_feature_sets(
    contextual_names: Sequence[str],
) -> tuple[tuple[PruningVariant, FeatureSet], ...]:
    result: list[tuple[PruningVariant, FeatureSet]] = []
    for variant in TARGETED_VARIANTS:
        indexes = variant.indexes(contextual_names)
        families = tuple(
            sorted({feature_family(contextual_names[index]) for index in indexes})
        )
        result.append(
            (
                variant,
                FeatureSet(
                    name=variant.name,
                    indexes=indexes,
                    families=families,
                    interpretation_subject=variant.name,
                ),
            )
        )
    signatures = [item.indexes for _, item in result]
    if len(set(signatures)) != len(signatures):
        raise FeatureExperimentError("targeted pruning variants are not distinct")
    return tuple(result)


def _load_baseline(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureExperimentError(
            f"cannot read frozen development report {resolved}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise FeatureExperimentError("frozen development report must be an object")
    return resolved, payload


def _base_classifications(report: dict[str, Any]) -> dict[str, str]:
    try:
        rows = report["baseFeaturePermutationImportance"]["ranking"]
        return {str(item["feature"]): str(item["classification"]) for item in rows}
    except (KeyError, TypeError) as error:
        raise FeatureExperimentError(
            "frozen report lacks per-signal importance classifications"
        ) from error


def validate_frozen_baseline(
    report: dict[str, Any], manifest: DatasetManifest
) -> None:
    if (
        report.get("kind") != BASELINE_KIND
        or report.get("testLabelsUsed") is not False
    ):
        raise FeatureExperimentError(
            "targeted pruning requires a development-only feature experiment report"
        )
    if report.get("manifestFileSha256") != sha256_file(manifest.path):
        raise FeatureExperimentError("frozen report does not match the manifest file")
    if report.get("manifestSnapshotSha256") != _manifest_digest(manifest):
        raise FeatureExperimentError("manifest snapshots changed after the frozen report")
    if report.get("recordingContentSha256") != {
        item.id: item.content_sha256 for item in manifest.recordings
    }:
        raise FeatureExperimentError("recording identities changed after the frozen report")
    if report.get("featureVersion") != FEATURE_VERSION:
        raise FeatureExperimentError("feature version changed after the frozen report")
    if report.get("selectionProtocol", {}).get("objective") != {
        "eventF1": 0.55,
        "timeIoU": 0.30,
        "liveTimeRecall": 0.15,
    }:
        raise FeatureExperimentError("frozen report uses a different selection objective")
    current_files = code_provenance().get("filesSha256")
    if report.get("provenance", {}).get("filesSha256") != current_files:
        raise FeatureExperimentError("core experiment code changed after baseline freezing")

    try:
        family_assessments = report["featureFamilyAssessments"]
        family_classes = {
            name: family_assessments[name]["classification"]
            for name in HARMFUL_FAMILIES
        }
    except (KeyError, TypeError) as error:
        raise FeatureExperimentError(
            "frozen report lacks the targeted family assessments"
        ) from error
    if any(value != "harmful" for value in family_classes.values()):
        raise FeatureExperimentError(
            f"frozen report no longer supports the targeted harmful families: "
            f"{family_classes}"
        )

    base_classes = _base_classifications(report)
    expected = {
        **{name: "helpful" for name in HELPFUL_EXCEPTIONS},
        **{name: "harmful" for name in HARMFUL_SIGNALS},
    }
    observed = {name: base_classes.get(name) for name in expected}
    if observed != expected:
        raise FeatureExperimentError(
            f"frozen report no longer supports the targeted signal plan: {observed}"
        )


def _paired_against_frozen_full(
    baseline_report: dict[str, Any],
    candidate_artifacts: Sequence[Any],
    *,
    margin: float,
    sign_consistency: float,
) -> dict[str, Any]:
    try:
        baseline_folds = {
            str(item["heldOutSourceGroup"]): item["metrics"]
            for item in baseline_report["candidates"]["full"]["outerFolds"]
        }
    except (KeyError, TypeError) as error:
        raise FeatureExperimentError("frozen report lacks full outer-fold metrics") from error
    candidate_folds = {
        item.fold.held_out_group: item.aggregate for item in candidate_artifacts
    }
    if set(baseline_folds) != set(candidate_folds):
        raise FeatureExperimentError("candidate and frozen full folds do not align")
    rows: list[dict[str, Any]] = []
    for group in sorted(baseline_folds):
        baseline = baseline_folds[group]
        candidate = candidate_folds[group]
        gain = float(candidate["objective"] - baseline["objective"])
        rows.append(
            {
                "sourceGroup": group,
                "frozenFullObjective": float(baseline["objective"]),
                "candidateObjective": float(candidate["objective"]),
                "deltaObjectiveCandidateMinusFull": gain,
                "deltaEventF1CandidateMinusFull": float(
                    candidate["eventF1"] - baseline["eventF1"]
                ),
                "deltaTimeIoUCandidateMinusFull": float(
                    candidate["timeIoU"] - baseline["timeIoU"]
                ),
                "deltaLiveTimeRecallCandidateMinusFull": float(
                    candidate["liveTimeRecall"] - baseline["liveTimeRecall"]
                ),
            }
        )
    classification = classify_paired_deltas(
        [item["deltaObjectiveCandidateMinusFull"] for item in rows],
        margin=margin,
        sign_consistency=sign_consistency,
    )
    return {
        "deltaDirection": "positive means the pruned candidate outperforms frozen full",
        "classification": classification,
        "pairedSourceGroups": rows,
    }


def select_targeted_candidate(
    frozen_full_metrics: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    primary_name = "targeted_pruned"
    if primary_name not in candidates:
        raise FeatureExperimentError("primary targeted_pruned candidate is missing")
    frozen_full_objective = float(frozen_full_metrics["objective"])
    guardrail_names = ("eventF1", "timeIoU", "liveTimeRecall")
    rows: list[dict[str, Any]] = []
    for name, candidate in sorted(candidates.items()):
        candidate_metrics = candidate["oof"]["aggregate"]
        candidate_objective = float(candidate_metrics["objective"])
        classification = str(
            candidate["pairedAgainstFrozenFull"]["classification"]["classification"]
        )
        feature_count = int(candidate["variant"]["retainedFeatureCount"])
        guardrails = {
            metric: {
                "frozenFull": float(frozen_full_metrics[metric]),
                "candidate": float(candidate_metrics[metric]),
                "deltaCandidateMinusFull": float(
                    candidate_metrics[metric] - frozen_full_metrics[metric]
                ),
                "passes": float(
                    candidate_metrics[metric] - frozen_full_metrics[metric]
                )
                >= -0.01,
            }
            for metric in guardrail_names
        }
        is_primary = name == primary_name
        is_eligible = is_primary and (
            classification == "helpful"
            and all(item["passes"] for item in guardrails.values())
        )
        rows.append(
            {
                "candidate": name,
                "objective": candidate_objective,
                "gainOverFrozenFull": candidate_objective - frozen_full_objective,
                "pairedClassification": classification,
                "retainedFeatureCount": feature_count,
                "primaryCandidate": is_primary,
                "aggregateMetricGuardrails": guardrails,
                "eligible": is_eligible,
            }
        )
    primary = next(item for item in rows if item["candidate"] == primary_name)
    if primary["eligible"]:
        selected = primary_name
        reason = (
            "the predeclared primary targeted-pruned candidate passed the paired "
            "source-group improvement rule"
        )
    else:
        selected = "full"
        reason = "no pruned candidate passed the predeclared paired improvement rule"
    return {
        "selectedCandidateForFinalTest": selected,
        "reason": reason,
        "rule": (
            "Only targeted_pruned may replace full. It must have classification=helpful "
            "under the frozen 0.01 objective margin / 3-of-4 source-group rule, and "
            "aggregate eventF1, timeIoU, and liveTimeRecall may each regress by at "
            "most 0.01. Other variants are diagnostics."
        ),
        "frozenFullObjective": float(frozen_full_objective),
        "candidates": rows,
    }


def expand_model_to_full_signature(
    model: LogisticModel, full_feature_names: Sequence[str]
) -> LogisticModel:
    """Zero-embed a subset model so the standard full extractor can serve it."""
    full_names = tuple(full_feature_names)
    if not full_names or len(set(full_names)) != len(full_names):
        raise FeatureExperimentError("full feature signature must be non-empty and unique")
    lookup = {name: index for index, name in enumerate(full_names)}
    try:
        retained_indexes = np.asarray(
            [lookup[name] for name in model.feature_names], dtype=np.int64
        )
    except KeyError as error:
        raise FeatureExperimentError(
            f"subset model feature is absent from the full signature: {error.args[0]}"
        ) from error
    if len(set(model.feature_names)) != len(model.feature_names):
        raise FeatureExperimentError("subset model signature contains duplicates")
    mean = np.zeros(len(full_names), dtype=np.float32)
    scale = np.ones(len(full_names), dtype=np.float32)
    weights = np.zeros(len(full_names), dtype=np.float32)
    mean[retained_indexes] = model.mean
    scale[retained_indexes] = model.scale
    weights[retained_indexes] = model.weights
    expanded = LogisticModel(
        feature_config=model.feature_config,
        feature_names=full_names,
        mean=mean,
        scale=scale,
        weights=weights,
        bias=model.bias,
        decoder=model.decoder,
        training_summary=dict(model.training_summary),
        feature_version=model.feature_version,
    )
    expanded.training_summary["zeroEmbeddedSubset"] = {
        "retainedFeatureCount": len(model.feature_names),
        "fullFeatureCount": len(full_names),
        "removedFeatureCount": len(full_names) - len(model.feature_names),
        "retainedFeatureNames": list(model.feature_names),
        "predictionParity": (
            "Removed full-signature columns have weight zero; retained normalization "
            "and weights are copied from the subset fit. Floating-point matrix "
            "multiplication can still differ slightly with the wider matrix."
        ),
    }
    return expanded


def validate_zero_embedding_predictions(
    subset_prepared: Sequence[PreparedRecording],
    subset_probabilities: Sequence[np.ndarray],
    full_prepared: Sequence[PreparedRecording],
    full_probabilities: Sequence[np.ndarray],
    decoder: DecoderConfig,
) -> float:
    """Require close probabilities and identical decoded intervals after embedding."""
    if not (
        len(subset_prepared)
        == len(subset_probabilities)
        == len(full_prepared)
        == len(full_probabilities)
    ):
        raise FeatureExperimentError("zero-embedding parity inputs are not aligned")
    maximum_error = 0.0
    for subset_item, subset, full_item, full in zip(
        subset_prepared,
        subset_probabilities,
        full_prepared,
        full_probabilities,
        strict=True,
    ):
        if subset.shape != full.shape:
            raise FeatureExperimentError(
                "zero-embedded probabilities have a different shape"
            )
        error = float(np.max(np.abs(subset - full))) if len(subset) else 0.0
        maximum_error = max(maximum_error, error)
        if not np.isfinite(error) or error > ZERO_EMBEDDING_PROBABILITY_TOLERANCE:
            raise FeatureExperimentError(
                "zero-embedded full model exceeded probability tolerance: "
                f"{error} > {ZERO_EMBEDDING_PROBABILITY_TOLERANCE}"
            )
        subset_intervals = _scored_predictions(subset_item, subset, decoder)
        full_intervals = _scored_predictions(full_item, full, decoder)
        if subset_intervals != full_intervals:
            raise FeatureExperimentError(
                "zero-embedded full model changed decoded validation intervals"
            )
    return maximum_error


def _local_provenance() -> dict[str, Any]:
    module = Path(__file__).resolve()
    script = module.parent.parent / "scripts" / "evaluate-pruned-features.py"
    return {
        "core": code_provenance(),
        "targetedPruningFilesSha256": {
            str(path.relative_to(module.parent.parent)): sha256_file(path)
            for path in (module, script)
            if path.is_file()
        },
    }


def run_targeted_pruning_experiments(
    manifest: DatasetManifest,
    prepared: Sequence[PreparedRecording],
    frozen_baseline_path: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    baseline_path, baseline = _load_baseline(frozen_baseline_path)
    validate_frozen_baseline(baseline, manifest)
    development_rows = [
        item for item in manifest.recordings if item.split in DEVELOPMENT_SPLITS
    ]
    expected_ids = {item.id for item in development_rows}
    if {item.recording.id for item in prepared} != expected_ids:
        raise FeatureExperimentError(
            "prepared recordings must contain exactly train+validation development rows"
        )
    if any(item.recording.split not in DEVELOPMENT_SPLITS for item in prepared):
        raise FeatureExperimentError("protected data entered targeted pruning")
    if not prepared:
        raise FeatureExperimentError("development data is empty")
    signature = prepared[0].contextual_names
    if any(item.contextual_names != signature for item in prepared):
        raise FeatureExperimentError("prepared feature signatures differ")
    if tuple(baseline.get("featureSignature", {}).get("names", ())) != signature:
        raise FeatureExperimentError("prepared signature differs from frozen full baseline")

    feature_config = FeatureConfig.from_dict(baseline["featureConfig"])
    training_config = TrainingConfig(**baseline["trainingConfig"])
    decoder_config = DecoderConfig.from_dict(baseline["baseDecoderConfig"])
    selection_protocol = baseline["selectionProtocol"]
    inner_fold_limit = selection_protocol.get("innerFoldLimit")
    margin = float(selection_protocol["objectiveMargin"])
    sign_consistency = float(selection_protocol["signConsistency"])
    paddings = tuple(
        float(item)
        for item in baseline["oofPadding"]["paddingSecondsBeforeAndAfter"]
    )
    folds = build_fold_plan(manifest.recordings)
    variants = build_pruning_feature_sets(signature)
    candidate_reports: dict[str, Any] = {}
    total = len(variants) * len(folds)
    completed = 0
    for variant, feature_set in variants:
        artifacts = []
        for fold in folds:
            completed += 1
            if progress is not None:
                progress(
                    f"Targeted pruning {completed}/{total}: {variant.name}, "
                    f"hold out {fold.held_out_group}"
                )
            artifacts.append(
                _run_candidate_fold(
                    prepared,
                    fold,
                    feature_set,
                    feature_config=feature_config,
                    training_config=training_config,
                    decoder_config=decoder_config,
                    inner_fold_limit=inner_fold_limit,
                )
            )
        aggregate = _aggregate_artifacts(artifacts)
        paired = _paired_against_frozen_full(
            baseline,
            artifacts,
            margin=margin,
            sign_consistency=sign_consistency,
        )
        candidate_reports[variant.name] = {
            "variant": variant.to_dict(signature, feature_set.indexes),
            "oof": aggregate,
            "pairedAgainstFrozenFull": paired,
            "outerFolds": [_fold_report(item) for item in artifacts],
            "oofPadding": _padding_report(artifacts, paddings),
        }

    frozen_full_metrics = baseline["candidates"]["full"]["oof"]["aggregate"]
    frozen_full_objective = float(frozen_full_metrics["objective"])
    selection = select_targeted_candidate(frozen_full_metrics, candidate_reports)
    selected = selection["selectedCandidateForFinalTest"]
    selected_signature = (
        list(signature)
        if selected == "full"
        else candidate_reports[selected]["variant"]["retainedFeatureNames"]
    )
    selected_padding = (
        baseline["oofPadding"]
        if selected == "full"
        else candidate_reports[selected]["oofPadding"]
    )
    return {
        "schemaVersion": PRUNING_EXPERIMENT_SCHEMA_VERSION,
        "kind": PRUNING_KIND,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "assessmentRole": "development-only-targeted-pruning-selection",
        "testLabelsUsed": False,
        "dataset": manifest.name,
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "manifestSnapshotSha256": _manifest_digest(manifest),
        "recordingContentSha256": {
            item.id: item.content_sha256 for item in manifest.recordings
        },
        "frozenFullBaseline": {
            "report": str(baseline_path),
            "reportSha256": sha256_file(baseline_path),
            "objective": frozen_full_objective,
            "featureCount": len(signature),
            "retrained": False,
            "aggregate": frozen_full_metrics,
            "macroSourceGroup": baseline["candidates"]["full"]["oof"][
                "macroSourceGroup"
            ],
            "bySourceGroup": baseline["candidates"]["full"]["oof"][
                "bySourceGroup"
            ],
        },
        "evidenceLockedBeforeRun": {
            "harmfulFamilies": list(HARMFUL_FAMILIES),
            "helpfulSignalsRetainedByTargetedVariants": list(HELPFUL_EXCEPTIONS),
            "harmfulSignals": list(HARMFUL_SIGNALS),
        },
        "featureVersion": FEATURE_VERSION,
        "fullFeatureSignature": {
            "count": len(signature),
            "names": list(signature),
        },
        "featureConfig": feature_config.to_dict(),
        "trainingConfig": training_config.to_dict(),
        "baseDecoderConfig": decoder_config.to_dict(),
        "selectionProtocol": {
            **selection_protocol,
            "baselineReuse": (
                "Full outer-fold results are reused byte-for-byte from the frozen "
                "report; candidates use the same nested source-group procedure and seeds."
            ),
            "targetedCandidateRule": selection["rule"],
            "paddingRole": (
                "0/1/2/3-second OOF padding is descriptive and does not participate "
                "in candidate selection; the existing 1-second operational default "
                "remains frozen."
            ),
            "seedSensitivity": (
                "Unmeasured in this pass: seed 7 is reused to preserve an exact paired "
                "comparison with the frozen full baseline."
            ),
        },
        "candidates": candidate_reports,
        "selection": selection,
        "selectedCandidateForFinalTest": selected,
        "selectedFeatureSignature": {
            "count": len(selected_signature),
            "names": selected_signature,
        },
        "selectedOofPadding": selected_padding,
        "development": baseline["development"],
        "evaluationCoverage": {
            "recordings": len(prepared),
            "sourceGroups": len(
                {item.recording.source_group for item in prepared}
            ),
            "videoSeconds": float(
                sum(item.sequence.metadata.duration for item in prepared)
            ),
            "trueRallies": sum(len(item.recording.rallies) for item in prepared),
            "selectionUnit": "sourceGroup",
            "paddingPredictions": "out-of-fold",
        },
        "protected": baseline["protected"],
        "provenance": _local_provenance(),
        "processing": {
            "wallClockSeconds": round(time.perf_counter() - started, 3),
            "candidateOuterFoldRuns": len(variants) * len(folds),
            "logisticModelFits": len(variants)
            * sum(
                (
                    len(fold.inner_folds)
                    if inner_fold_limit is None or inner_fold_limit <= 0
                    else min(len(fold.inner_folds), inner_fold_limit)
                )
                + 1
                for fold in folds
            ),
            "fullBaselineOuterFoldsRepeated": 0,
            "featureCacheNote": "Frozen audiovisual superset is column-subset in memory.",
        },
        "limitations": [
            "Only four independent development source groups are available.",
            (
                "These same development groups produced the original importance evidence; "
                "this targeted pass is post-selection exploratory evidence, not an "
                "independent confirmation."
            ),
            "Per-signal permutation evidence is predictive rather than causal.",
            "Seed sensitivity is unmeasured; this pass preserves frozen seed 7.",
            "The protected test remains unopened by this report.",
        ],
    }


def _load_pruning_report(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureExperimentError(
            f"cannot read targeted-pruning report {resolved}: {error}"
        ) from error
    if not isinstance(report, dict):
        raise FeatureExperimentError("targeted-pruning report must be an object")
    return resolved, report


def _validate_pruning_report(
    report: dict[str, Any], manifest: DatasetManifest
) -> None:
    if (
        report.get("schemaVersion") != PRUNING_EXPERIMENT_SCHEMA_VERSION
        or report.get("kind") != PRUNING_KIND
        or report.get("testLabelsUsed") is not False
    ):
        raise FeatureExperimentError(
            "final test requires a frozen, development-only targeted-pruning report"
        )
    if report.get("manifestFileSha256") != sha256_file(manifest.path):
        raise FeatureExperimentError("pruning report does not match the manifest file")
    if report.get("manifestSnapshotSha256") != _manifest_digest(manifest):
        raise FeatureExperimentError("manifest snapshots changed after pruning selection")
    if report.get("recordingContentSha256") != {
        item.id: item.content_sha256 for item in manifest.recordings
    }:
        raise FeatureExperimentError("recording identities changed after pruning selection")
    if report.get("featureVersion") != FEATURE_VERSION:
        raise FeatureExperimentError("feature version changed after pruning selection")
    expected_provenance = report.get("provenance", {})
    current = _local_provenance()
    if expected_provenance.get("core", {}).get("filesSha256") != current.get(
        "core", {}
    ).get("filesSha256"):
        raise FeatureExperimentError("core experiment code changed after pruning selection")
    if expected_provenance.get("targetedPruningFilesSha256") != current.get(
        "targetedPruningFilesSha256"
    ):
        raise FeatureExperimentError(
            "targeted-pruning code changed after development selection"
        )
    baseline_path = Path(report["frozenFullBaseline"]["report"])
    if sha256_file(baseline_path) != report["frozenFullBaseline"]["reportSha256"]:
        raise FeatureExperimentError("frozen full baseline report changed or disappeared")


def run_fixed_split_pruned_test(
    manifest_path: str | Path,
    pruning_report_path: str | Path,
    cache_dir: str | Path,
    model_destination: str | Path,
    *,
    candidate: str | None = None,
    allow_unpromoted_diagnostic: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fit a frozen pruned candidate, then open the single protected test source."""
    started = time.perf_counter()
    report_path, development = _load_pruning_report(pruning_report_path)
    manifest = load_manifest(manifest_path)
    _validate_pruning_report(development, manifest)
    selected = str(development["selectedCandidateForFinalTest"])
    requested = candidate or selected
    if requested == "full":
        raise FeatureExperimentError(
            "this runner is for pruned candidates; use the existing full-model final test"
        )
    if requested not in development.get("candidates", {}):
        raise FeatureExperimentError(f"unknown frozen pruning candidate: {requested!r}")
    promotion_eligible = selected == "targeted_pruned" and requested == selected
    if not promotion_eligible and not allow_unpromoted_diagnostic:
        raise FeatureExperimentError(
            "the requested candidate did not pass the primary development rule; "
            "use explicit diagnostic acknowledgement to run a retrospective test"
        )

    model_target = Path(model_destination).expanduser().resolve()
    if model_target.exists() and (
        not model_target.is_dir() or any(model_target.iterdir())
    ):
        raise FeatureExperimentError(
            f"model destination is not an empty directory: {model_target}"
        )
    feature_config = FeatureConfig.from_dict(development["featureConfig"])
    training_config = TrainingConfig(**development["trainingConfig"])
    decoder_config = DecoderConfig.from_dict(development["baseDecoderConfig"])
    full_signature = tuple(development["fullFeatureSignature"]["names"])
    retained_signature = tuple(
        development["candidates"][requested]["variant"]["retainedFeatureNames"]
    )
    full_lookup = {name: index for index, name in enumerate(full_signature)}
    try:
        retained_indexes = tuple(full_lookup[name] for name in retained_signature)
    except KeyError as error:
        raise FeatureExperimentError(
            f"frozen retained feature is absent from the full signature: {error.args[0]}"
        ) from error
    if tuple(sorted(retained_indexes)) != retained_indexes:
        raise FeatureExperimentError("frozen retained signature changed column ordering")

    train_rows = manifest.for_split("train")
    validation_rows = manifest.for_split("validation")
    if progress is not None:
        progress(f"Preparing {len(train_rows)} fixed-split training recordings")
    training_full = _prepare_many(
        train_rows, feature_config, cache_dir, progress=progress
    )
    if progress is not None:
        progress(f"Preparing {len(validation_rows)} fixed-split validation recordings")
    validation_full = _prepare_many(
        validation_rows, feature_config, cache_dir, progress=progress
    )
    for item in (*training_full, *validation_full):
        if item.contextual_names != full_signature:
            raise FeatureExperimentError(
                "fixed-split extracted signature differs from frozen development"
            )
    training = _subset_prepared(training_full, retained_indexes)
    validation = _subset_prepared(validation_full, retained_indexes)
    if progress is not None:
        progress(f"Fitting frozen pruned candidate {requested}")
    subset_model = _fit(
        training,
        validation,
        feature_config=feature_config,
        decoder=decoder_config,
        config=training_config,
        seed=training_config.seed,
    )
    validation_subset_probabilities = [
        subset_model.predict(item.contextual_values) for item in validation
    ]
    decoder, decoder_selection = _select_decoder(
        validation, validation_subset_probabilities, decoder_config
    )
    subset_model.decoder = decoder
    expanded_model = expand_model_to_full_signature(subset_model, full_signature)
    expanded_model.decoder = decoder
    validation_full_probabilities = [
        expanded_model.predict(item.contextual_values) for item in validation_full
    ]
    pre_save_parity_error = validate_zero_embedding_predictions(
        validation,
        validation_subset_probabilities,
        validation_full,
        validation_full_probabilities,
        decoder,
    )
    expanded_model.training_summary.update(
        {
            "dataset": manifest.name,
            "manifestSha256": _manifest_digest(manifest),
            "targetedPruningCandidate": requested,
            "targetedPruningDevelopmentReport": str(report_path),
            "targetedPruningDevelopmentReportSha256": sha256_file(report_path),
            "promotionEligible": promotion_eligible,
            "trainingRecordingIds": [item.recording.id for item in training_full],
            "validationRecordingIds": [
                item.recording.id for item in validation_full
            ],
            "trainingSourceGroups": sorted(
                {item.recording.source_group for item in training_full}
            ),
            "validationSourceGroups": sorted(
                {item.recording.source_group for item in validation_full}
            ),
            "recordingContentSha256": {
                item.recording.id: item.recording.content_sha256
                for item in (*training_full, *validation_full)
            },
            "decoderSelection": decoder_selection,
            "zeroEmbeddingPreSaveMaximumValidationProbabilityError": (
                pre_save_parity_error
            ),
            "zeroEmbeddingProbabilityTolerance": (
                ZERO_EMBEDDING_PROBABILITY_TOLERANCE
            ),
            "zeroEmbeddingDecodedValidationIntervalsIdentical": True,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": __import__("cv2").__version__,
            },
        }
    )
    expanded_model.save(model_target)
    deployed_model = load_model(model_target)
    deployed_validation_probabilities = [
        deployed_model.predict(item.contextual_values) for item in validation_full
    ]
    maximum_parity_error = validate_zero_embedding_predictions(
        validation,
        validation_subset_probabilities,
        validation_full,
        deployed_validation_probabilities,
        decoder,
    )
    validation_rows_report, validation_metrics = _evaluate_prepared_probabilities(
        validation_full, deployed_validation_probabilities, decoder
    )
    for row, item in zip(validation_rows_report, validation_full, strict=True):
        row["sourceGroup"] = item.recording.source_group
    validation_metrics["objective"] = objective(validation_metrics)

    test_rows = manifest.for_split("test")
    if progress is not None:
        progress(
            f"Opening {len(test_rows)} protected test recording(s) for retrospective "
            "single-source regression"
        )
    test_prepared = _prepare_many(test_rows, feature_config, cache_dir, progress=progress)
    for item in test_prepared:
        if item.contextual_names != full_signature:
            raise FeatureExperimentError("test extracted signature differs from frozen full")
    test_probabilities = [
        deployed_model.predict(item.contextual_values) for item in test_prepared
    ]
    per_recording, test_metrics = _evaluate_prepared_probabilities(
        test_prepared, test_probabilities, decoder
    )
    for row, item in zip(per_recording, test_prepared, strict=True):
        row["sourceGroup"] = item.recording.source_group
    test_metrics["objective"] = objective(test_metrics)
    paddings = tuple(
        float(item)
        for item in development["candidates"][requested]["oofPadding"][
            "paddingSecondsBeforeAndAfter"
        ]
    )
    padding_recordings = [
        RecordingIntervals(
            id=item.recording.id,
            split="test",
            duration=item.sequence.metadata.duration,
            truth=item.recording.rallies,
            predictions=_scored_predictions(item, probabilities, decoder),
        )
        for item, probabilities in zip(test_prepared, test_probabilities, strict=True)
    ]
    return {
        "schemaVersion": PRUNING_EXPERIMENT_SCHEMA_VERSION,
        "kind": PRUNING_FINAL_KIND,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "assessmentRole": "retrospective-single-source-regression-test",
        "testLabelsOpened": True,
        "promotionEligibleFromDevelopment": promotion_eligible,
        "candidate": requested,
        "developmentSelectedCandidate": selected,
        "developmentReport": str(report_path),
        "developmentReportSha256": sha256_file(report_path),
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "testCoverage": {
            "recordings": len(test_prepared),
            "recordingIds": [item.recording.id for item in test_prepared],
            "sourceGroups": sorted(
                {item.recording.source_group for item in test_prepared}
            ),
            "videoSeconds": float(
                sum(item.sequence.metadata.duration for item in test_prepared)
            ),
            "trueRallies": sum(
                len(item.recording.rallies) for item in test_prepared
            ),
        },
        "model": {
            "path": str(model_target),
            "artifactSha256": deployed_model.artifact_sha256,
            "fullFeatureCount": len(full_signature),
            "retainedFittedFeatureCount": len(retained_signature),
            "removedZeroWeightFeatureCount": len(full_signature)
            - len(retained_signature),
            "removedFeaturesHaveExactZeroWeights": True,
            "zeroEmbeddingMaximumValidationProbabilityError": maximum_parity_error,
            "zeroEmbeddingProbabilityTolerance": (
                ZERO_EMBEDDING_PROBABILITY_TOLERANCE
            ),
            "zeroEmbeddingDecodedValidationIntervalsIdentical": True,
        },
        "validation": {
            "assessmentRole": "decoder-and-early-stopping-tuning-only",
            "metrics": validation_metrics,
            "recordings": validation_rows_report,
        },
        "test": {"aggregate": test_metrics, "recordings": per_recording},
        "testPadding": evaluate_crop_padding(padding_recordings, paddings),
        "provenance": _local_provenance(),
        "processing": {
            "wallClockSeconds": round(time.perf_counter() - started, 3),
        },
        "warning": (
            "The protected test is one indoor source group whose labels were already "
            "opened for the earlier full-model evaluation. This is a retrospective "
            "regression result, not external generalization evidence, and it must not "
            "revise the frozen promotion decision."
        ),
    }
