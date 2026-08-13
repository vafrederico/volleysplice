"""Nested development study for rectangle-ROI court-relative features.

This module is deliberately downstream of a frozen transition-feature report.
It reconstructs that report's selected feature signature as the control, then
appends one compact, per-recording-ranked court-relative block.  No protected
split is prepared during selection and no recording metadata other than the
predeclared capture/ROI requirements enters a model matrix.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import FEATURE_VERSION, DecoderConfig, FeatureConfig, TrainingConfig
from .court_relative_experiment import (
    COURT_RELATIVE_SPEC_HASHES,
    COURT_RELATIVE_VARIANTS,
    court_relative_feature_spec,
    derive_court_relative_features,
)
from .feature_experiments import (
    DEFAULT_OBJECTIVE_MARGIN,
    DEFAULT_SIGN_CONSISTENCY,
    DEVELOPMENT_SPLITS,
    FeatureExperimentError,
    FeatureSet,
    _aggregate_artifacts,
    _fit,
    _fold_report,
    _model_fingerprint,
    _protected_summary,
    _run_candidate_fold,
    _seed,
    _select_decoder,
    _subset_prepared,
    build_fold_plan,
    objective,
    sha256_file,
)
from .pipeline import (
    PreparedRecording,
    _evaluate_prepared_probabilities,
    _manifest_digest,
    _prepare_many,
)
from .schema import DatasetManifest, load_manifest
from .transition_feature_experiment import (
    TRANSITION_EXPERIMENT_SCHEMA_VERSION,
    DerivedFeatureBlock,
    _multi_iou_metrics,
    _paired_candidate_comparison,
    _rank_nonconstant_columns,
    _signature_sha256,
    append_feature_blocks,
    prepare_transition_candidates,
)


COURT_STUDY_SCHEMA_VERSION = 1
CONTROL_CANDIDATE = "step1_control"
VARIANT_CANDIDATES = {
    "orientation_invariant": "control_plus_orientation_invariant",
    "fixed_endline": "control_plus_fixed_endline",
    "combined": "control_plus_combined",
}
PRIMARY_CANDIDATE_ORDER = (CONTROL_CANDIDATE, *VARIANT_CANDIDATES.values())


@dataclass(frozen=True)
class CourtStudyPrepared:
    prepared: tuple[PreparedRecording, ...]
    candidates: tuple[FeatureSet, ...]
    upstream_candidate: str
    control_names: tuple[str, ...]
    court_names: tuple[str, ...]
    definitions: Mapping[str, Any]


def load_frozen_step1_report(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load a frozen, unopened-test transition experiment report."""

    report_path = Path(path).expanduser().resolve()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureExperimentError(
            f"cannot read frozen step-1 report {report_path}: {error}"
        ) from error
    if (
        report.get("schemaVersion") != TRANSITION_EXPERIMENT_SCHEMA_VERSION
        or report.get("kind")
        != "volleycut-transition-feature-experiment-development"
        or report.get("freezeStatus") != "frozen-development-selection"
        or report.get("testLabelsUsed") is not False
        or not isinstance(report.get("selectedCandidateForRetrospectiveTest"), str)
        or not isinstance(report.get("finalizationPlan"), dict)
    ):
        raise FeatureExperimentError(
            "step-1 report is not a frozen unopened-test transition experiment"
        )
    inner_limit = report.get("selectionProtocol", {}).get("innerFoldLimit")
    if inner_limit not in (None, 0):
        raise FeatureExperimentError(
            "court study requires a full nested step-1 report, not an inner-fold smoke run"
        )
    return report_path, report


def _validate_upstream_manifest(
    manifest: DatasetManifest,
    upstream: Mapping[str, Any],
) -> None:
    if upstream.get("manifestFileSha256") != sha256_file(manifest.path):
        raise FeatureExperimentError("frozen step-1 report does not match the manifest")
    if upstream.get("manifestSnapshotSha256") != _manifest_digest(manifest):
        raise FeatureExperimentError(
            "recording snapshots changed after the step-1 report was frozen"
        )
    expected_content = {item.id: item.content_sha256 for item in manifest.recordings}
    if upstream.get("recordingContentSha256") != expected_content:
        raise FeatureExperimentError(
            "recording content identities changed after the step-1 report was frozen"
        )
    if upstream.get("featureVersion") != FEATURE_VERSION:
        raise FeatureExperimentError(
            "feature implementation version differs from the frozen step-1 report"
        )
    expected_folds = [item.to_dict() for item in build_fold_plan(manifest.recordings)]
    if upstream.get("selectionProtocol", {}).get("folds") != expected_folds:
        raise FeatureExperimentError("source-group folds differ from frozen step 1")


def _candidate_metadata(
    feature_set: FeatureSet,
    signature: Sequence[str],
) -> dict[str, Any]:
    names = [signature[index] for index in feature_set.indexes]
    return {
        "name": feature_set.name,
        "featureCount": len(names),
        "featureNames": names,
        "featureSignatureSha256": _signature_sha256(names),
        "families": list(feature_set.families),
        "interpretationSubject": feature_set.interpretation_subject,
    }


def prepare_court_study_candidates(
    prepared: Sequence[PreparedRecording],
    feature_config: FeatureConfig,
    upstream: Mapping[str, Any],
) -> CourtStudyPrepared:
    """Reconstruct the step-1 winner and append the combined court block once."""

    if not prepared:
        raise FeatureExperimentError("court-relative study data is empty")
    selected_name = upstream.get("selectedCandidateForRetrospectiveTest")
    finalization = upstream.get("finalizationPlan", {})
    if finalization.get("candidate") != selected_name:
        raise FeatureExperimentError("frozen step-1 candidate and finalization disagree")
    frozen_feature_config = FeatureConfig.from_dict(dict(upstream["featureConfig"]))
    if feature_config.to_dict() != frozen_feature_config.to_dict():
        raise FeatureExperimentError("feature configuration differs from frozen step 1")

    transition = prepare_transition_candidates(prepared, feature_config)
    selected = next(
        (item for item in transition.candidates if item.name == selected_name),
        None,
    )
    if selected is None:
        raise FeatureExperimentError(
            f"frozen step-1 candidate {selected_name!r} is no longer implemented"
        )
    controls = _subset_prepared(transition.prepared, selected.indexes)
    control_names = controls[0].contextual_names
    if (
        list(control_names) != finalization.get("featureNames")
        or _signature_sha256(control_names)
        != finalization.get("featureSignatureSha256")
    ):
        raise FeatureExperimentError("frozen step-1 feature signature changed")

    combined_spec = court_relative_feature_spec("combined")
    augmented: list[PreparedRecording] = []
    for raw, control in zip(prepared, controls, strict=True):
        if raw.recording.id != control.recording.id:
            raise FeatureExperimentError("step-1 and court rows are not aligned")
        raw_values, raw_names = derive_court_relative_features(raw, "combined")
        if raw_names != combined_spec.feature_names:
            raise FeatureExperimentError("court-relative combined signature changed")
        ranked = _rank_nonconstant_columns(raw_values.astype(np.float64, copy=False))
        block = DerivedFeatureBlock(
            values=ranked,
            names=raw_names,
            groups=combined_spec.groups,
            definitions={
                "normalization": (
                    "Each nonconstant court-relative column is percentile-ranked "
                    "within recording; structural constants remain zero."
                ),
                "appendPolicy": (
                    "The current-time court block is appended once after reconstructing "
                    "the frozen step-1 candidate; it is not expanded over context offsets."
                ),
                "spec": combined_spec.metadata(),
            },
        )
        augmented.append(append_feature_blocks(control, block))

    full_signature = augmented[0].contextual_names
    if any(item.contextual_names != full_signature for item in augmented):
        raise FeatureExperimentError("court-relative feature signatures differ by recording")
    control_count = len(control_names)
    local_by_name = {name: index for index, name in enumerate(combined_spec.feature_names)}

    def indexes_for_variant(variant: str) -> tuple[int, ...]:
        spec = court_relative_feature_spec(variant)
        return tuple(
            range(control_count)
        ) + tuple(control_count + local_by_name[name] for name in spec.feature_names)

    upstream_families = tuple(selected.families)
    candidates = [
        FeatureSet(
            CONTROL_CANDIDATE,
            tuple(range(control_count)),
            upstream_families,
            f"frozen step-1 candidate {selected_name}",
        )
    ]
    for variant in COURT_RELATIVE_VARIANTS:
        candidates.append(
            FeatureSet(
                VARIANT_CANDIDATES[variant],
                indexes_for_variant(variant),
                (*upstream_families, f"court_relative:{variant}"),
                f"court-relative {variant} bank",
            )
        )
    return CourtStudyPrepared(
        prepared=tuple(augmented),
        candidates=tuple(candidates),
        upstream_candidate=str(selected_name),
        control_names=control_names,
        court_names=combined_spec.feature_names,
        definitions={
            "normalization": (
                "Each nonconstant court-relative column is percentile-ranked within "
                "recording; structural constants remain zero."
            ),
            "appendPolicy": (
                "Court columns append once to the exact frozen step-1 feature matrix "
                "and are never contextualized again."
            ),
            "variantSpecs": {
                variant: court_relative_feature_spec(variant).metadata()
                for variant in COURT_RELATIVE_VARIANTS
            },
            "excludedInputs": [
                "rally labels",
                "outcome tags",
                "environment",
                "game metadata",
                "playersPerTeam",
                "sideSwitches",
                "annotation confidence",
            ],
        },
    )


def _court_code_provenance() -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    tracked = (
        package / "court_relative_feature_study.py",
        package / "court_relative_experiment.py",
        package / "transition_feature_experiment.py",
        package / "feature_experiments.py",
        package / "features.py",
        package / "model.py",
        package / "pipeline.py",
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
        head = None
        dirty = None
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


def run_development_court_experiments(
    manifest: DatasetManifest,
    prepared: Sequence[PreparedRecording],
    *,
    step1_report_path: str | Path,
    objective_margin: float = DEFAULT_OBJECTIVE_MARGIN,
    sign_consistency: float = DEFAULT_SIGN_CONSISTENCY,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run full nested LOGO court candidates using the frozen step-1 protocol."""

    started = time.perf_counter()
    upstream_path, upstream = load_frozen_step1_report(step1_report_path)
    _validate_upstream_manifest(manifest, upstream)
    if objective_margin < 0 or not 0.5 < sign_consistency <= 1.0:
        raise ValueError("invalid objective margin or sign consistency")
    development_rows = tuple(
        item for item in manifest.recordings if item.split in DEVELOPMENT_SPLITS
    )
    expected_ids = {item.id for item in development_rows}
    if {item.recording.id for item in prepared} != expected_ids:
        raise FeatureExperimentError(
            "prepared recordings must contain exactly train+validation development rows"
        )
    if any(item.recording.split not in DEVELOPMENT_SPLITS for item in prepared):
        raise FeatureExperimentError("protected test/challenge data entered development")

    feature_config = FeatureConfig.from_dict(dict(upstream["featureConfig"]))
    training_config = TrainingConfig(**dict(upstream["trainingConfig"]))
    decoder_config = DecoderConfig.from_dict(dict(upstream["baseDecoderConfig"]))
    feature_config.validate()
    training_config.validate()
    decoder_config.validate()
    study = prepare_court_study_candidates(prepared, feature_config, upstream)
    folds = build_fold_plan(manifest.recordings)

    artifacts_by_candidate: dict[str, list[Any]] = {}
    candidate_reports: dict[str, Any] = {}
    total = len(study.candidates) * len(folds)
    completed = 0
    signature = study.prepared[0].contextual_names
    for candidate in study.candidates:
        artifacts: list[Any] = []
        for fold in folds:
            completed += 1
            if progress is not None:
                progress(
                    f"Court-relative experiment {completed}/{total}: {candidate.name}, "
                    f"hold out {fold.held_out_group}"
                )
            artifacts.append(
                _run_candidate_fold(
                    study.prepared,
                    fold,
                    candidate,
                    feature_config=feature_config,
                    training_config=training_config,
                    decoder_config=decoder_config,
                    inner_fold_limit=None,
                )
            )
        artifacts_by_candidate[candidate.name] = artifacts
        aggregated = _aggregate_artifacts(artifacts)
        candidate_reports[candidate.name] = {
            "featureSet": _candidate_metadata(candidate, signature),
            "oof": aggregated,
            "multiIouMetrics": _multi_iou_metrics(aggregated["aggregate"]),
            "outcomeSlices": aggregated["aggregate"]["outcomeSlices"],
            "outerFolds": [_fold_report(item) for item in artifacts],
        }

    control_artifacts = artifacts_by_candidate[CONTROL_CANDIDATE]
    comparisons = {
        candidate.name: _paired_candidate_comparison(
            control_artifacts,
            artifacts_by_candidate[candidate.name],
            subject=candidate.interpretation_subject,
            margin=objective_margin,
            sign_consistency=sign_consistency,
        )
        for candidate in study.candidates
        if candidate.name != CONTROL_CANDIDATE
    }
    eligible = [CONTROL_CANDIDATE] + [
        name
        for name in PRIMARY_CANDIDATE_ORDER[1:]
        if comparisons[name]["classification"]["classification"] == "helpful"
    ]
    selected_name = max(
        eligible,
        key=lambda name: (
            float(candidate_reports[name]["oof"]["aggregate"]["objective"]),
            -PRIMARY_CANDIDATE_ORDER.index(name),
        ),
    )
    selected_artifacts = artifacts_by_candidate[selected_name]
    pooled_prepared = [
        item for artifact in selected_artifacts for item in artifact.held_prepared
    ]
    pooled_probabilities = [
        values for artifact in selected_artifacts for values in artifact.probabilities
    ]
    frozen_decoder, decoder_selection = _select_decoder(
        pooled_prepared, pooled_probabilities, decoder_config
    )
    epoch_cap = max(
        1,
        int(round(float(np.median([item.epoch_cap for item in selected_artifacts])))),
    )
    selected_feature_set = next(
        item for item in study.candidates if item.name == selected_name
    )
    selected_names = tuple(signature[index] for index in selected_feature_set.indexes)
    final_seed = _seed(
        training_config.seed,
        "court-relative-final-refit",
        study.upstream_candidate,
        selected_name,
    )
    upstream_sha256 = sha256_file(upstream_path)
    return {
        "schemaVersion": COURT_STUDY_SCHEMA_VERSION,
        "kind": "volleycut-court-relative-feature-study-development",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "freezeStatus": "frozen-development-selection",
        "assessmentRole": "development-only-source-group-out-of-fold-selection",
        "testLabelsUsed": False,
        "dataset": manifest.name,
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "manifestSnapshotSha256": _manifest_digest(manifest),
        "recordingContentSha256": {
            item.id: item.content_sha256 for item in manifest.recordings
        },
        "development": {
            "splits": sorted(DEVELOPMENT_SPLITS),
            "recordingIds": sorted(expected_ids),
            "sourceGroups": sorted({item.source_group for item in development_rows}),
        },
        "protected": _protected_summary(manifest.recordings),
        "featureVersion": FEATURE_VERSION,
        "featureConfig": feature_config.to_dict(),
        "trainingConfig": training_config.to_dict(),
        "baseDecoderConfig": decoder_config.to_dict(),
        "upstreamStep1": {
            "developmentReport": str(upstream_path),
            "developmentReportSha256": upstream_sha256,
            "selectedCandidate": study.upstream_candidate,
            "controlFeatureCount": len(study.control_names),
            "controlFeatureSignatureSha256": _signature_sha256(study.control_names),
            "selectionWasFrozenBeforeCourtStudy": True,
        },
        "substrate": {
            "controlFeatureCount": len(study.control_names),
            "controlFeatureNames": list(study.control_names),
            "controlSignatureSha256": _signature_sha256(study.control_names),
            "combinedCourtFeatureCount": len(study.court_names),
            "combinedCourtFeatureNames": list(study.court_names),
            "combinedCourtSignatureSha256": _signature_sha256(study.court_names),
        },
        "courtFeatureDefinitions": study.definitions,
        "courtSpecSha256": dict(COURT_RELATIVE_SPEC_HASHES),
        "selectionProtocol": {
            "outer": "leave one development sourceGroup out",
            "inner": "leave one remaining sourceGroup out",
            "innerFoldLimit": None,
            "folds": [item.to_dict() for item in folds],
            "trainingAndFoldSeeds": "identical frozen step-1 config and seed derivation",
            "epochRefit": (
                "median inner best epoch caps each outer refit; no outer labels select "
                "the fitted checkpoint"
            ),
            "decoder": "selected from pooled inner out-of-fold probabilities",
            "objective": {"eventF1": 0.55, "timeIoU": 0.30, "liveTimeRecall": 0.15},
            "objectiveMargin": objective_margin,
            "signConsistency": sign_consistency,
        },
        "metricProtocol": {
            "headlineIoU": 0.5,
            "reportedEventIoUThresholds": [0.3, 0.5, 0.7],
            "outcomeSlices": [
                "all",
                "shortAtMost3Seconds",
                "ace",
                "serviceFault",
                "ordinaryLong",
            ],
        },
        "candidates": candidate_reports,
        "pairedComparisonsAgainstControl": comparisons,
        "selectedCandidateForRetrospectiveTest": selected_name,
        "candidateSelection": {
            "rule": (
                "The frozen step-1 control is always eligible. A court variant is "
                "eligible only when its paired source-group deltas have mean and median "
                "at least +0.01 objective and improve at least 3 of 4 groups; select the "
                "highest-objective eligible candidate with declared-order tie breaking."
            ),
            "eligibleCandidates": eligible,
            "selected": selected_name,
            "testMetricsConsulted": False,
        },
        "finalizationPlan": {
            "candidate": selected_name,
            "featureCount": len(selected_names),
            "featureNames": list(selected_names),
            "featureSignatureSha256": _signature_sha256(selected_names),
            "epochCap": epoch_cap,
            "seed": final_seed,
            "decoder": frozen_decoder.to_dict(),
            "decoderSelectionOnDevelopmentOof": decoder_selection,
            "fitRows": "all train+validation development recordings",
            "checkpointSelection": "minimum training loss up to frozen epoch cap",
        },
        "guardrails": [
            "Only train and validation recordings are prepared during development.",
            "The exact frozen step-1 winner is the same-feature control.",
            "Source groups, not recordings, define every inner and outer split.",
            "Court columns append once and are never multiplied across context offsets.",
            "No labels, outcomes, confidence, environment, game fields, side switches, or player counts are inputs.",
            "The retrospective command accepts only this frozen report and never selects on test.",
        ],
        "provenance": _court_code_provenance(),
        "processing": {
            "wallClockSeconds": round(time.perf_counter() - started, 3),
            "featureCacheNote": (
                "Warm raw caches are reused; rectangle-ROI court columns are derived "
                "deterministically in memory."
            ),
        },
        "limitations": [
            "Only four independent development source groups are available.",
            "Rectangle thirds are proxies, not a calibrated court homography.",
            "Signed fixed-endline directions assume the declared stationary camera orientation.",
            "Reaction and migration features look forward and apply only to the offline cutter.",
            "Paired classifications are practical evidence categories, not significance tests.",
        ],
    }


def _load_frozen_court_report(path: str | Path) -> tuple[Path, dict[str, Any]]:
    report_path = Path(path).expanduser().resolve()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureExperimentError(
            f"cannot read frozen court report {report_path}: {error}"
        ) from error
    if (
        report.get("schemaVersion") != COURT_STUDY_SCHEMA_VERSION
        or report.get("kind")
        != "volleycut-court-relative-feature-study-development"
        or report.get("freezeStatus") != "frozen-development-selection"
        or report.get("testLabelsUsed") is not False
        or not isinstance(report.get("selectedCandidateForRetrospectiveTest"), str)
    ):
        raise FeatureExperimentError(
            "development report is not a frozen unopened-test court study"
        )
    return report_path, report


def run_retrospective_court_test(
    manifest_path: str | Path,
    development_report_path: str | Path,
    cache_dir: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fit the frozen court-study winner, then access the test split once."""

    report_path, development = _load_frozen_court_report(development_report_path)
    upstream_info = development.get("upstreamStep1", {})
    upstream_path, upstream = load_frozen_step1_report(
        upstream_info.get("developmentReport", "")
    )
    if sha256_file(upstream_path) != upstream_info.get("developmentReportSha256"):
        raise FeatureExperimentError("frozen step-1 report changed after court selection")

    manifest = load_manifest(manifest_path)
    if development.get("manifestFileSha256") != sha256_file(manifest.path):
        raise FeatureExperimentError("court report does not match the supplied manifest")
    if development.get("manifestSnapshotSha256") != _manifest_digest(manifest):
        raise FeatureExperimentError(
            "recording snapshots changed after the court report was frozen"
        )
    if development.get("recordingContentSha256") != {
        item.id: item.content_sha256 for item in manifest.recordings
    }:
        raise FeatureExperimentError("recording content identities changed after selection")
    if development.get("featureVersion") != FEATURE_VERSION:
        raise FeatureExperimentError("feature implementation version changed after selection")
    current_provenance = _court_code_provenance()
    if development.get("provenance", {}).get("filesSha256") != current_provenance.get(
        "filesSha256"
    ):
        raise FeatureExperimentError("court experiment code changed after development")

    feature_config = FeatureConfig.from_dict(dict(development["featureConfig"]))
    training_config = TrainingConfig(**dict(development["trainingConfig"]))
    finalization = development["finalizationPlan"]
    selected_name = development["selectedCandidateForRetrospectiveTest"]
    if finalization.get("candidate") != selected_name:
        raise FeatureExperimentError("frozen court candidate and finalization disagree")
    epoch_cap = int(finalization["epochCap"])
    if epoch_cap < 1:
        raise FeatureExperimentError("frozen court epoch cap is invalid")
    decoder = DecoderConfig.from_dict(dict(finalization["decoder"]))
    seed = int(finalization["seed"])

    development_rows = tuple(
        item for item in manifest.recordings if item.split in DEVELOPMENT_SPLITS
    )
    test_rows = manifest.for_split("test")
    if not development_rows or not test_rows:
        raise FeatureExperimentError("retrospective test requires development and test rows")
    if progress is not None:
        progress("Preparing frozen train+validation features; test remains unopened")
    development_prepared = (
        _prepare_many(development_rows, feature_config, cache_dir)
        if progress is None
        else _prepare_many(
            development_rows, feature_config, cache_dir, progress=progress
        )
    )
    development_study = prepare_court_study_candidates(
        development_prepared, feature_config, upstream
    )
    selected = next(
        (item for item in development_study.candidates if item.name == selected_name),
        None,
    )
    if selected is None:
        raise FeatureExperimentError("frozen court candidate is no longer implemented")
    selected_development = _subset_prepared(
        development_study.prepared, selected.indexes
    )
    selected_names = selected_development[0].contextual_names
    if (
        list(selected_names) != finalization.get("featureNames")
        or _signature_sha256(selected_names)
        != finalization.get("featureSignatureSha256")
    ):
        raise FeatureExperimentError("frozen court feature signature changed")
    fit_config = replace(
        training_config,
        epochs=epoch_cap,
        patience=max(training_config.patience, epoch_cap + 1),
        seed=seed,
    )
    if progress is not None:
        progress("Fitting the frozen court candidate on development rows only")
    model = _fit(
        selected_development,
        (),
        feature_config=feature_config,
        decoder=decoder,
        config=fit_config,
        seed=seed,
    )
    model.decoder = decoder

    if progress is not None:
        progress("Opening the retrospective test split once with selection frozen")
    test_prepared = (
        _prepare_many(test_rows, feature_config, cache_dir)
        if progress is None
        else _prepare_many(test_rows, feature_config, cache_dir, progress=progress)
    )
    test_study = prepare_court_study_candidates(test_prepared, feature_config, upstream)
    test_spec = next(
        item for item in test_study.candidates if item.name == selected_name
    )
    selected_test = _subset_prepared(test_study.prepared, test_spec.indexes)
    if any(item.contextual_names != selected_names for item in selected_test):
        raise FeatureExperimentError("test feature signature differs from frozen court study")
    probabilities = [model.predict(item.contextual_values) for item in selected_test]
    per_recording, aggregate = _evaluate_prepared_probabilities(
        selected_test, probabilities, decoder
    )
    aggregate["objective"] = objective(aggregate)
    return {
        "schemaVersion": COURT_STUDY_SCHEMA_VERSION,
        "kind": "volleycut-court-relative-feature-study-retrospective-test",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "assessmentRole": "explicit-single-source-retrospective-regression-test-access",
        "testLabelsOpened": True,
        "selectionLockedBeforeTest": True,
        "developmentReport": str(report_path),
        "developmentReportSha256": sha256_file(report_path),
        "upstreamStep1Report": str(upstream_path),
        "upstreamStep1ReportSha256": sha256_file(upstream_path),
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "selectedCandidate": selected_name,
        "featureCount": len(selected_names),
        "featureSignatureSha256": _signature_sha256(selected_names),
        "training": {
            "recordingIds": [item.recording.id for item in selected_development],
            "sourceGroups": sorted(
                {item.recording.source_group for item in selected_development}
            ),
            "splits": sorted(DEVELOPMENT_SPLITS),
            "testRowsUsedForFitting": False,
            "epochCap": epoch_cap,
            "seed": seed,
            "modelFingerprint": _model_fingerprint(model),
        },
        "decoder": decoder.to_dict(),
        "test": {
            "aggregate": aggregate,
            "multiIouMetrics": _multi_iou_metrics(aggregate),
            "outcomeSlices": aggregate["outcomeSlices"],
            "recordings": per_recording,
        },
        "provenance": current_provenance,
        "guardrails": [
            "Candidate, feature signature, epoch cap, seed, and decoder were frozen in development.",
            "Only train+validation rows fit the final model.",
            "Test metrics are reported once and cannot revise the selected candidate.",
        ],
        "warning": (
            "The protected split is a retrospective regression test, not evidence of "
            "external generalization. Do not revise this feature bank from its result."
        ),
    }
