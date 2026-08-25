#!/usr/bin/env python3
"""Run versioned side-switch feature-development profiles with nested LOO."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_feature_development import (
    BASELINE_PROFILE,
    BOUNDARY_BASELINE_PROFILE,
    C1_PROFILE,
    DECODER,
    GAP_SHAPE_PROFILE,
    INTERACTION_PROFILE,
    M1_PROFILE,
    P1_PROFILE,
    Q1_PROFILE,
    T2_CONDITIONAL_ONLY_PROFILE,
    T2_PROFILE,
    T2_SWAP_ONLY_PROFILE,
    T4_PROFILE,
    concise_metrics,
    evaluate_profile,
    evaluate_predictions,
    metric_delta,
)
from analysis.side_switch_t2_transport import T2_CORE_FEATURE_NAMES
from analysis.side_switch_t4_selective_far import T4_CORE_FEATURE_NAMES


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_FEATURES = REPORTS / "side-switch-full-union-v5-state-features-v1.json"
DEFAULT_AUDIT = REPORTS / "side-switch-full-video-marker-evaluation-2026-08-21-r2.json"
DEFAULT_MODEL = ROOT / "models/side-switch-hard-negative-mining-v1/model.json"
DEFAULT_EVALUATION = REPORTS / "side-switch-hard-negative-mining-v1-evaluation.json"
DEFAULT_IMPORTANCE = REPORTS / "side-switch-feature-importance-2026-08-24.json"
DEFAULT_GAP_FEATURES = REPORTS / "side-switch-gap-shape-features-v1.json"
DEFAULT_VISUAL_FEATURES = REPORTS / "side-switch-visual-summary-v2-features-v1.json"
DEFAULT_M1_FEATURES = REPORTS / "side-switch-m1-foreground-motion-features-v1.json"
DEFAULT_T2_FEATURES = (
    REPORTS / "side-switch-t2-conditional-identity-transport-features-v1.json"
)
DEFAULT_T4_FEATURES = (
    REPORTS / "side-switch-t4-selective-far-court-detection-features-v1.json"
)
DEFAULT_OUTPUTS = {
    "E0": REPORTS / "side-switch-feature-development-e0-baseline-v1.json",
    "E1": REPORTS / "side-switch-feature-development-e1-interactions-v1.json",
    "E2": REPORTS / "side-switch-feature-development-e2-gap-shape-v1.json",
    "E4": REPORTS / "side-switch-feature-development-e4-directional-q1-v1.json",
    "E5": REPORTS / "side-switch-feature-development-e5-camera-c1-v1.json",
    "E6": REPORTS / "side-switch-feature-development-e6-persistence-p1-v1.json",
    "M1": REPORTS / "side-switch-feature-development-m1-foreground-motion-v1.json",
    "T2": REPORTS / "side-switch-feature-development-t2-conditional-transport-opened-v1.json",
    "T4": REPORTS / "side-switch-feature-development-t4-selective-far-opened-v1.json",
}
EXPECTED_SHA256 = {
    "features": "9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551",
    "audit": "142d6617aed7f20b51cdcdfc9b3e61c8c76beaf79a31fc2d7ee54fb0fdf27b89",
    "model": "c2570481c30dec62f56ac3284cd4028763ffd8e72a9ad07e9908fec10df387e3",
    "evaluation": "e67088b36d177d68c24587efcb186eab4201b5294532be1fdf978ef13aaacc4b",
    "importance": "b2c501c61e9f7b3aeb2bbb04cf73f3f9793e1831053a7762c08993924daea14d",
    "gapFeatures": "0f89dbbf7d7cda896100e3f0730a17ecbaba5dfd1cd0525993199a5246c5a777",
    "visualFeatures": "6ce23b43018d04045ba783510ad86ef3c6d8767fdc435c7684481fca89ba4871",
    "m1Features": "b3ac34aaccec6395526a69341832e048e7ef763543790c2e87873236d569cb3b",
    "t2Features": "fc7e36306f4570e8c93b788f8471cd11d3fe865a1e8dc4699c673e8334aa281c",
    "t4Features": "443c0ded15cfddbb5e156f670c895375c8c43caede032a9b3ef5ae445ca4113a",
}


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _metric_counts(value: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return tuple(
        int(value[name])
        for name in ("proposals", "truePositives", "falsePositives", "falseNegatives")
    )


def _validate_e0(
    result: Mapping[str, Any],
    current_model: Mapping[str, Any],
    current_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    expected = current_evaluation["fixedVariantOuterResults"]["union34-top2-x2"]
    if _metric_counts(result["primary"]) != _metric_counts(expected["primary"]):
        raise ValueError("E0 primary event metrics do not reproduce the promoted control")
    if _metric_counts(result["strict"]) != _metric_counts(expected["strict"]):
        raise ValueError("E0 strict event metrics do not reproduce the promoted control")
    if not np.isclose(
        float(result["rowAveragePrecision"]),
        float(expected["rowAveragePrecision"]),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("E0 row AP does not reproduce the promoted control")

    expected_thresholds = {
        str(fold["heldRecordingId"]): float(
            next(
                row
                for row in fold["leaderboard"]
                if row["variant"]["id"] == "union34-top2-x2"
            )["threshold"]
        )
        for fold in current_evaluation["outerFolds"]
    }
    actual_thresholds = {
        str(fold["heldRecordingId"]): float(fold["threshold"])
        for fold in result["outerFolds"]
    }
    if actual_thresholds.keys() != expected_thresholds.keys() or any(
        not np.isclose(
            actual_thresholds[key], expected_thresholds[key], atol=1e-12, rtol=0.0
        )
        for key in actual_thresholds
    ):
        raise ValueError("E0 outer thresholds do not reproduce the promoted control")

    classifier = result["fullDevelopment"]["classifier"]
    expected_classifier = current_model["classifier"]
    if classifier["featureNames"] != expected_classifier["featureNames"]:
        raise ValueError("E0 final feature signature changed")
    for name in ("impute", "mean", "scale", "weights"):
        if not np.allclose(
            np.asarray(classifier[name], dtype=np.float64),
            np.asarray(expected_classifier[name], dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"E0 final classifier changed for {name}")
    for name in ("bias", "l2"):
        if not np.isclose(
            float(classifier[name]),
            float(expected_classifier[name]),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"E0 final classifier changed for {name}")
    if not np.isclose(
        float(classifier["threshold"]),
        float(current_model["threshold"]),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("E0 selected threshold changed")
    return {
        "status": "exact",
        "primaryCounts": list(_metric_counts(result["primary"])),
        "strictCounts": list(_metric_counts(result["strict"])),
        "rowAveragePrecision": result["rowAveragePrecision"],
        "outerThresholdsExact": True,
        "finalClassifierExact": True,
    }


def _selected_ids(result: Mapping[str, Any]) -> set[str]:
    return {
        str(row["eventId"])
        for fold in result["outerFolds"]
        for row in fold["heldCandidateScores"]
        if bool(row["selected"])
    }


def _recording_deltas(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for recording_id, before in baseline["primary"]["byRecording"].items():
        after = candidate["primary"]["byRecording"][recording_id]
        result[recording_id] = {
            name: int(after[name]) - int(before[name])
            for name in (
                "proposals",
                "truePositives",
                "falsePositives",
                "falseNegatives",
            )
        }
    return result


def _standalone_checks(delta: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "primaryF1GainAtLeast2pp": delta["primaryF1"] >= 0.02 - 1e-12,
        "precisionDeclineNoMoreThan2pp": delta["primaryPrecision"] >= -0.02 - 1e-12,
        "recallDeclineNoMoreThan2pp": delta["primaryRecall"] >= -0.02 - 1e-12,
        "strictF1DeclineNoMoreThan1pp": delta["strictF1"] >= -0.01 - 1e-12,
    }


def _evaluate_e1_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    importance: Mapping[str, Any],
) -> dict[str, Any]:
    delta = metric_delta(baseline, candidate)
    high_player_change_ids = {
        str(row["candidateId"])
        for row in importance["errorAnalysis"]["falsePositiveProposals"]
        if any(
            str(item["feature"]) == "playerGlobalAppearanceChange"
            for item in row["topPositiveContributors"]
        )
    }
    baseline_selected = _selected_ids(baseline)
    candidate_selected = _selected_ids(candidate)
    baseline_slice = len(high_player_change_ids & baseline_selected)
    candidate_slice = len(high_player_change_ids & candidate_selected)
    checks = {
        **_standalone_checks(delta),
        "highPlayerChangeFalsePositiveSliceReduced": candidate_slice < baseline_slice,
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "delta": delta,
        "targetSlice": {
            "definition": (
                "baseline false proposals where playerGlobalAppearanceChange was "
                "among the five strongest positive logit contributors"
            ),
            "frozenCandidateIds": sorted(high_player_change_ids),
            "baselineSelected": baseline_slice,
            "candidateSelected": candidate_slice,
            "change": candidate_slice - baseline_slice,
        },
        "byRecordingDelta": _recording_deltas(baseline, candidate),
    }


def _evaluate_e2_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    importance: Mapping[str, Any],
) -> dict[str, Any]:
    delta = metric_delta(baseline, candidate)
    gap_supported_ids = {
        str(row["candidateId"])
        for row in importance["errorAnalysis"]["falsePositiveProposals"]
        if any(
            str(item["feature"])
            in {
                "productionGapMeanDeadStateScore",
                "productionGapPeakDeadStateScore",
                "productionGapDurationSeconds",
            }
            for item in row["topPositiveContributors"]
        )
    }
    baseline_selected = _selected_ids(baseline)
    candidate_selected = _selected_ids(candidate)
    baseline_slice = len(gap_supported_ids & baseline_selected)
    candidate_slice = len(gap_supported_ids & candidate_selected)
    checks = {
        **_standalone_checks(delta),
        "gapSupportedFalsePositiveSliceReduced": candidate_slice < baseline_slice,
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "delta": delta,
        "targetSlice": {
            "definition": (
                "baseline false proposals where a current dead-state mean, peak, or "
                "gap-duration input was among the five strongest positive contributors"
            ),
            "frozenCandidateIds": sorted(gap_supported_ids),
            "baselineSelected": baseline_slice,
            "candidateSelected": candidate_slice,
            "change": candidate_slice - baseline_slice,
        },
        "byRecordingDelta": _recording_deltas(baseline, candidate),
    }


def _evaluate_e4_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    delta = metric_delta(baseline, candidate)
    baseline_false_selected = {
        str(row["eventId"])
        for fold in baseline["outerFolds"]
        for row in fold["heldCandidateScores"]
        if bool(row["selected"]) and int(row["label"]) == 0
    }
    post_observation_collapse = {
        str(row["eventId"])
        for row in rows
        if (
            float(row["features"]["afterProposalCoverage"])
            < float(row["features"]["beforeProposalCoverage"])
            or float(row["features"]["afterProposalCount"])
            < float(row["features"]["beforeProposalCount"])
        )
    }
    frozen_ids = baseline_false_selected & post_observation_collapse
    candidate_selected = _selected_ids(candidate)
    checks = {
        **_standalone_checks(delta),
        "postObservationCollapseFalsePositiveSliceReduced": len(
            frozen_ids & candidate_selected
        )
        < len(frozen_ids),
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "delta": delta,
        "targetSlice": {
            "definition": (
                "baseline false proposals where after-window proposal coverage or "
                "proposal count is lower than the before window"
            ),
            "frozenCandidateIds": sorted(frozen_ids),
            "baselineSelected": len(frozen_ids),
            "candidateSelected": len(frozen_ids & candidate_selected),
            "change": len(frozen_ids & candidate_selected) - len(frozen_ids),
        },
        "byRecordingDelta": _recording_deltas(baseline, candidate),
    }


def _evaluate_e5_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    from analysis.side_switch_visual_summary_v2 import C1_FEATURE_NAMES

    delta = metric_delta(baseline, candidate)
    thresholds = {
        name: float(
            np.quantile(
                [float(row["features"][name]) for row in rows],
                0.75,
            )
        )
        for name in C1_FEATURE_NAMES
    }
    baseline_false_selected = {
        str(row["eventId"])
        for fold in baseline["outerFolds"]
        for row in fold["heldCandidateScores"]
        if bool(row["selected"]) and int(row["label"]) == 0
    }
    scene_confounded = {
        str(row["eventId"])
        for row in rows
        if any(
            float(row["features"][name]) >= thresholds[name]
            for name in C1_FEATURE_NAMES
        )
    }
    frozen_ids = baseline_false_selected & scene_confounded
    candidate_selected = _selected_ids(candidate)
    checks = {
        **_standalone_checks(delta),
        "highSceneInstabilityFalsePositiveSliceReduced": len(
            frozen_ids & candidate_selected
        )
        < len(frozen_ids),
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "delta": delta,
        "targetSlice": {
            "definition": (
                "baseline false proposals at or above the all-row 75th percentile "
                "for at least one frozen C1 diagnostic"
            ),
            "featureThresholds": thresholds,
            "frozenCandidateIds": sorted(frozen_ids),
            "baselineSelected": len(frozen_ids),
            "candidateSelected": len(frozen_ids & candidate_selected),
            "change": len(frozen_ids & candidate_selected) - len(frozen_ids),
        },
        "byRecordingDelta": _recording_deltas(baseline, candidate),
    }


def _held_rows(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for fold in result["outerFolds"]
        for row in fold["heldCandidateScores"]
    ]


def _evaluate_e6_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    delta = metric_delta(baseline, candidate)
    baseline_scores = _held_rows(baseline)
    candidate_selected = _selected_ids(candidate)
    covered_miss_ids = {
        str(row["eventId"])
        for row in baseline_scores
        if int(row["label"]) == 1 and not bool(row["selected"])
    }
    persistent_values = {
        str(row["eventId"]): float(
            row["features"]["crossModalityPersistentMinimum"]
        )
        for row in rows
    }
    transient_false_ids = {
        str(row["eventId"])
        for row in baseline_scores
        if (
            int(row["label"]) == 0
            and bool(row["selected"])
            and persistent_values[str(row["eventId"])] <= 0.0
        )
    }
    recovered_covered = len(covered_miss_ids & candidate_selected)
    retained_transient_false = len(transient_false_ids & candidate_selected)
    recording_deltas = _recording_deltas(baseline, candidate)
    total_true_positive_gain = int(delta["truePositives"])
    maximum_recording_gain = max(
        (int(value["truePositives"]) for value in recording_deltas.values()),
        default=0,
    )
    regressing_recordings = sorted(
        recording_id
        for recording_id, value in recording_deltas.items()
        if int(value["truePositives"]) < 0
    )
    gain_depends_on_one_recording = (
        total_true_positive_gain > 0
        and total_true_positive_gain - maximum_recording_gain <= 0
        and len(regressing_recordings) >= 3
    )
    checks = {
        **_standalone_checks(delta),
        "coveredBoundaryMissesRecovered": recovered_covered > 0,
        "nonPersistentFalseBoundarySliceReduced": retained_transient_false
        < len(transient_false_ids),
        "recordingRobustness": not gain_depends_on_one_recording,
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "delta": delta,
        "targetSlices": {
            "coveredBoundaryMisses": {
                "definition": (
                    "positive boundary candidates not selected by the matched "
                    "boundary-only outer-held control"
                ),
                "frozenCandidateIds": sorted(covered_miss_ids),
                "baselineSelected": 0,
                "candidateSelected": recovered_covered,
                "change": recovered_covered,
            },
            "nonPersistentFalseBoundaries": {
                "definition": (
                    "matched-control selected false boundaries with "
                    "crossModalityPersistentMinimum <= 0"
                ),
                "frozenCandidateIds": sorted(transient_false_ids),
                "baselineSelected": len(transient_false_ids),
                "candidateSelected": retained_transient_false,
                "change": retained_transient_false - len(transient_false_ids),
            },
        },
        "recordingRobustness": {
            "gainDependsOnOneRecordingWhileAtLeastThreeRegress": gain_depends_on_one_recording,
            "totalTruePositiveGain": total_true_positive_gain,
            "maximumSingleRecordingTruePositiveGain": maximum_recording_gain,
            "truePositiveGainWithoutBestRecording": total_true_positive_gain
            - maximum_recording_gain,
            "regressingRecordingIds": regressing_recordings,
        },
        "byRecordingDelta": recording_deltas,
        "conflictRecording193307688": recording_deltas.get(
            "raw-no-backup-PXL_20260816_193307688"
        ),
    }


def _evaluate_m1_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    delta = metric_delta(baseline, candidate)
    baseline_scores = _held_rows(baseline)
    candidate_selected = _selected_ids(candidate)
    covered_miss_ids = {
        str(row["eventId"])
        for row in baseline_scores
        if int(row["label"]) == 1 and not bool(row["selected"])
    }
    exchange_values = {
        str(row["eventId"]): float(
            row["features"]["bidirectionalExchangeMinimum"]
        )
        for row in rows
    }
    exchange_median = float(np.median(list(exchange_values.values())))
    low_exchange_false_ids = {
        str(row["eventId"])
        for row in baseline_scores
        if (
            int(row["label"]) == 0
            and bool(row["selected"])
            and exchange_values[str(row["eventId"])] <= exchange_median
        )
    }
    recovered_covered = len(covered_miss_ids & candidate_selected)
    retained_low_exchange_false = len(low_exchange_false_ids & candidate_selected)
    recording_deltas = _recording_deltas(baseline, candidate)
    total_true_positive_gain = int(delta["truePositives"])
    maximum_recording_gain = max(
        (int(value["truePositives"]) for value in recording_deltas.values()),
        default=0,
    )
    regressing_recordings = sorted(
        recording_id
        for recording_id, value in recording_deltas.items()
        if int(value["truePositives"]) < 0
    )
    gain_depends_on_one_recording = (
        total_true_positive_gain > 0
        and total_true_positive_gain - maximum_recording_gain <= 0
        and len(regressing_recordings) >= 3
    )
    checks = {
        **_standalone_checks(delta),
        "coveredBoundaryMissesRecovered": recovered_covered > 0,
        "lowExchangeFalseBoundarySliceReduced": retained_low_exchange_false
        < len(low_exchange_false_ids),
        "recordingRobustness": not gain_depends_on_one_recording,
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "delta": delta,
        "targetSlices": {
            "coveredBoundaryMisses": {
                "definition": (
                    "positive boundary candidates not selected by the matched "
                    "boundary-only outer-held control"
                ),
                "frozenCandidateIds": sorted(covered_miss_ids),
                "baselineSelected": 0,
                "candidateSelected": recovered_covered,
                "change": recovered_covered,
            },
            "lowExchangeFalseBoundaries": {
                "definition": (
                    "matched-control selected false boundaries with "
                    "bidirectionalExchangeMinimum at or below the median across "
                    "all 624 eligible boundary rows"
                ),
                "featureMedian": exchange_median,
                "frozenCandidateIds": sorted(low_exchange_false_ids),
                "baselineSelected": len(low_exchange_false_ids),
                "candidateSelected": retained_low_exchange_false,
                "change": retained_low_exchange_false - len(low_exchange_false_ids),
            },
        },
        "recordingRobustness": {
            "gainDependsOnOneRecordingWhileAtLeastThreeRegress": gain_depends_on_one_recording,
            "totalTruePositiveGain": total_true_positive_gain,
            "maximumSingleRecordingTruePositiveGain": maximum_recording_gain,
            "truePositiveGainWithoutBestRecording": total_true_positive_gain
            - maximum_recording_gain,
            "regressingRecordingIds": regressing_recordings,
        },
        "byRecordingDelta": recording_deltas,
        "conflictRecording193307688": recording_deltas.get(
            "raw-no-backup-PXL_20260816_193307688"
        ),
    }


def _evaluate_t2_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    delta = metric_delta(baseline, candidate)
    baseline_scores = _held_rows(baseline)
    candidate_selected = _selected_ids(candidate)
    covered_miss_ids = {
        str(row["eventId"])
        for row in baseline_scores
        if int(row["label"]) == 1 and not bool(row["selected"])
    }
    feature_values = {
        str(row["eventId"]): {
            name: float(row["features"][name]) for name in T2_CORE_FEATURE_NAMES
        }
        for row in rows
    }
    medians = {
        name: float(np.median([value[name] for value in feature_values.values()]))
        for name in T2_CORE_FEATURE_NAMES
    }
    weak_transport_false_ids = {
        str(row["eventId"])
        for row in baseline_scores
        if (
            int(row["label"]) == 0
            and bool(row["selected"])
            and any(
                feature_values[str(row["eventId"])][name] <= medians[name]
                for name in T2_CORE_FEATURE_NAMES
            )
        )
    }
    recovered_covered = len(covered_miss_ids & candidate_selected)
    retained_weak_transport_false = len(
        weak_transport_false_ids & candidate_selected
    )
    recording_deltas = _recording_deltas(baseline, candidate)
    total_true_positive_gain = int(delta["truePositives"])
    maximum_recording_gain = max(
        (int(value["truePositives"]) for value in recording_deltas.values()),
        default=0,
    )
    regressing_recordings = sorted(
        recording_id
        for recording_id, value in recording_deltas.items()
        if int(value["truePositives"]) < 0
    )
    gain_depends_on_one_recording = (
        total_true_positive_gain > 0
        and total_true_positive_gain - maximum_recording_gain <= 0
        and len(regressing_recordings) >= 3
    )
    checks = {
        **_standalone_checks(delta),
        "coveredBoundaryMissesRecovered": recovered_covered > 0,
        "weakTransportFalseBoundarySliceReduced": retained_weak_transport_false
        < len(weak_transport_false_ids),
        "recordingRobustness": not gain_depends_on_one_recording,
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "interpretation": "exploratory-opened-development-only",
        "checks": checks,
        "delta": delta,
        "targetSlices": {
            "coveredBoundaryMisses": {
                "definition": (
                    "positive boundary candidates not selected by the matched "
                    "boundary-only outer-held control"
                ),
                "frozenCandidateIds": sorted(covered_miss_ids),
                "baselineSelected": 0,
                "candidateSelected": recovered_covered,
                "change": recovered_covered,
            },
            "weakTransportFalseBoundaries": {
                "definition": (
                    "matched-control selected false boundaries where either T2 "
                    "core value is at or below its all-624-row median"
                ),
                "featureMedians": medians,
                "frozenCandidateIds": sorted(weak_transport_false_ids),
                "baselineSelected": len(weak_transport_false_ids),
                "candidateSelected": retained_weak_transport_false,
                "change": retained_weak_transport_false
                - len(weak_transport_false_ids),
            },
        },
        "recordingRobustness": {
            "gainDependsOnOneRecordingWhileAtLeastThreeRegress": gain_depends_on_one_recording,
            "totalTruePositiveGain": total_true_positive_gain,
            "maximumSingleRecordingTruePositiveGain": maximum_recording_gain,
            "truePositiveGainWithoutBestRecording": total_true_positive_gain
            - maximum_recording_gain,
            "regressingRecordingIds": regressing_recordings,
        },
        "byRecordingDelta": recording_deltas,
        "conflictRecording193307688": recording_deltas.get(
            "raw-no-backup-PXL_20260816_193307688"
        ),
    }


def _t2_coefficient_importance(candidate: Mapping[str, Any]) -> dict[str, Any]:
    classifier = candidate["fullDevelopment"]["classifier"]
    names = [str(name) for name in classifier["featureNames"]]
    weights = [float(value) for value in classifier["weights"]]
    ranks = {
        name: rank
        for rank, name in enumerate(
            sorted(names, key=lambda name: (-abs(weights[names.index(name)]), name)),
            1,
        )
    }
    fold_weights = {name: [] for name in T2_CORE_FEATURE_NAMES}
    for fold in candidate["outerFolds"]:
        fit = fold["fitStandardizedWeights"]
        fit_names = [str(name) for name in fit["featureNames"]]
        for name in T2_CORE_FEATURE_NAMES:
            fold_weights[name].append(float(fit["weights"][fit_names.index(name)]))
    features = []
    for name in T2_CORE_FEATURE_NAMES:
        weight = weights[names.index(name)]
        values = fold_weights[name]
        expected_sign = 1 if weight > 0 else -1 if weight < 0 else 0
        same_sign = sum(
            (1 if value > 0 else -1 if value < 0 else 0) == expected_sign
            for value in values
        )
        features.append(
            {
                "feature": name,
                "fullDevelopmentStandardizedCoefficient": weight,
                "absoluteCoefficientRankAmong36": ranks[name],
                "outerFitCoefficients": values,
                "outerFitPositiveCount": sum(value > 0 for value in values),
                "outerFitNegativeCount": sum(value < 0 for value in values),
                "outerFitSameSignAsFullCount": same_sign,
                "outerFitSameSignAsFullFraction": same_sign / len(values),
                "outerFitMean": float(np.mean(values)),
                "outerFitStandardDeviation": float(np.std(values)),
                "outerFitMinimum": min(values),
                "outerFitMaximum": max(values),
            }
        )
    return {
        "coefficientContract": (
            "weights act on fold-standardized inputs; magnitudes are comparable "
            "within a fit but do not establish causal importance"
        ),
        "features": features,
    }


def _evaluate_t4_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    delta = metric_delta(baseline, candidate)
    baseline_scores = _held_rows(baseline)
    candidate_selected = _selected_ids(candidate)
    reliable_swap_name = "selectiveFarJerseyReliableSwapEvidence"
    feature_values = {
        str(row["eventId"]): float(row["features"][reliable_swap_name])
        for row in rows
    }
    covered_miss_ids = {
        str(row["eventId"])
        for row in baseline_scores
        if (
            int(row["label"]) == 1
            and not bool(row["selected"])
            and feature_values[str(row["eventId"])] > 0.0
        )
    }
    zero_evidence_false_ids = {
        str(row["eventId"])
        for row in baseline_scores
        if (
            int(row["label"]) == 0
            and bool(row["selected"])
            and feature_values[str(row["eventId"])] == 0.0
        )
    }
    recovered_covered = len(covered_miss_ids & candidate_selected)
    retained_zero_evidence_false = len(zero_evidence_false_ids & candidate_selected)
    recording_deltas = _recording_deltas(baseline, candidate)
    total_true_positive_gain = int(delta["truePositives"])
    maximum_recording_gain = max(
        (int(value["truePositives"]) for value in recording_deltas.values()),
        default=0,
    )
    regressing_recordings = sorted(
        recording_id
        for recording_id, value in recording_deltas.items()
        if int(value["truePositives"]) < 0
    )
    gain_depends_on_one_recording = (
        total_true_positive_gain > 0
        and total_true_positive_gain - maximum_recording_gain <= 0
        and len(regressing_recordings) >= 3
    )
    checks = {
        **_standalone_checks(delta),
        "coveredBoundaryMissesRecovered": recovered_covered > 0,
        "zeroReliableSwapEvidenceFalseBoundarySliceReduced": (
            retained_zero_evidence_false < len(zero_evidence_false_ids)
        ),
        "recordingRobustness": not gain_depends_on_one_recording,
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "interpretation": "exploratory-opened-development-only",
        "checks": checks,
        "delta": delta,
        "targetSlices": {
            "coveredBoundaryMisses": {
                "definition": (
                    "positive boundary candidates not selected by the matched "
                    "boundary-only outer-held control and having nonzero selective "
                    "far reliable swap evidence"
                ),
                "frozenCandidateIds": sorted(covered_miss_ids),
                "baselineSelected": 0,
                "candidateSelected": recovered_covered,
                "change": recovered_covered,
            },
            "zeroReliableSwapEvidenceFalseBoundaries": {
                "definition": (
                    "matched-control selected false boundaries with exactly zero "
                    "selectiveFarJerseyReliableSwapEvidence"
                ),
                "frozenCandidateIds": sorted(zero_evidence_false_ids),
                "baselineSelected": len(zero_evidence_false_ids),
                "candidateSelected": retained_zero_evidence_false,
                "change": retained_zero_evidence_false
                - len(zero_evidence_false_ids),
            },
        },
        "recordingRobustness": {
            "gainDependsOnOneRecordingWhileAtLeastThreeRegress": gain_depends_on_one_recording,
            "totalTruePositiveGain": total_true_positive_gain,
            "maximumSingleRecordingTruePositiveGain": maximum_recording_gain,
            "truePositiveGainWithoutBestRecording": total_true_positive_gain
            - maximum_recording_gain,
            "regressingRecordingIds": regressing_recordings,
        },
        "byRecordingDelta": recording_deltas,
        "conflictRecording193307688": recording_deltas.get(
            "raw-no-backup-PXL_20260816_193307688"
        ),
    }


def _t4_coefficient_importance(candidate: Mapping[str, Any]) -> dict[str, Any]:
    classifier = candidate["fullDevelopment"]["classifier"]
    names = [str(name) for name in classifier["featureNames"]]
    weights = [float(value) for value in classifier["weights"]]
    ranks = {
        name: rank
        for rank, name in enumerate(
            sorted(names, key=lambda name: (-abs(weights[names.index(name)]), name)),
            1,
        )
    }
    fold_weights = {name: [] for name in T4_CORE_FEATURE_NAMES}
    for fold in candidate["outerFolds"]:
        fit = fold["fitStandardizedWeights"]
        fit_names = [str(name) for name in fit["featureNames"]]
        for name in T4_CORE_FEATURE_NAMES:
            fold_weights[name].append(float(fit["weights"][fit_names.index(name)]))
    features = []
    for name in T4_CORE_FEATURE_NAMES:
        weight = weights[names.index(name)]
        values = fold_weights[name]
        expected_sign = 1 if weight > 0 else -1 if weight < 0 else 0
        same_sign = sum(
            (1 if value > 0 else -1 if value < 0 else 0) == expected_sign
            for value in values
        )
        features.append(
            {
                "feature": name,
                "fullDevelopmentStandardizedCoefficient": weight,
                "absoluteCoefficientRankAmong37": ranks[name],
                "outerFitCoefficients": values,
                "outerFitPositiveCount": sum(value > 0 for value in values),
                "outerFitNegativeCount": sum(value < 0 for value in values),
                "outerFitSameSignAsFullCount": same_sign,
                "outerFitSameSignAsFullFraction": same_sign / len(values),
                "outerFitMean": float(np.mean(values)),
                "outerFitStandardDeviation": float(np.std(values)),
                "outerFitMinimum": min(values),
                "outerFitMaximum": max(values),
            }
        )
    return {
        "coefficientContract": (
            "weights act on fold-standardized inputs; magnitudes are comparable "
            "within a fit but do not establish causal importance"
        ),
        "features": features,
    }


def _e6_composition_diagnostic(
    rows: list[Mapping[str, Any]],
    markers: Mapping[str, list[Mapping[str, Any]]],
    full_baseline: Mapping[str, Any],
    persistence: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = {
        str(row["eventId"])
        for row in _held_rows(persistence)
        if bool(row["selected"])
    }
    selected_ids.update(
        str(row["eventId"])
        for row in _held_rows(full_baseline)
        if bool(row["selected"]) and str(row["kind"]) == "internal-dead-state-peak"
    )
    predictions = np.asarray(
        [str(row["eventId"]) in selected_ids for row in rows], dtype=bool
    )
    return {
        "mergeContract": {
            "boundaryBranch": "P1 outer-held selection with boundary-fit threshold",
            "internalBranch": "exact full-union E0 outer-held internal selections",
            "probabilityCalibration": "none across branches; merge selected IDs only",
            "crossKindSuppression": "none in diagnostic; one-to-one event matching penalizes duplicates",
            "selectionUse": "diagnostic only; E6 decision uses boundary-only comparison",
        },
        "primary": evaluate_predictions(rows, predictions, markers, 4.0, inventory=True),
        "strict": evaluate_predictions(rows, predictions, markers, 0.0, inventory=True),
    }


def _m1_composition_diagnostic(
    rows: list[Mapping[str, Any]],
    markers: Mapping[str, list[Mapping[str, Any]]],
    full_baseline: Mapping[str, Any],
    motion: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = {
        str(row["eventId"])
        for row in _held_rows(motion)
        if bool(row["selected"])
    }
    selected_ids.update(
        str(row["eventId"])
        for row in _held_rows(full_baseline)
        if bool(row["selected"]) and str(row["kind"]) == "internal-dead-state-peak"
    )
    predictions = np.asarray(
        [str(row["eventId"]) in selected_ids for row in rows], dtype=bool
    )
    return {
        "mergeContract": {
            "boundaryBranch": "M1 outer-held selection with boundary-fit threshold",
            "internalBranch": "exact full-union E0 outer-held internal selections",
            "probabilityCalibration": "none across branches; merge selected IDs only",
            "crossKindSuppression": "none in diagnostic; one-to-one event matching penalizes duplicates",
            "selectionUse": "diagnostic only; M1 decision uses boundary-only comparison",
        },
        "primary": evaluate_predictions(rows, predictions, markers, 4.0, inventory=True),
        "strict": evaluate_predictions(rows, predictions, markers, 0.0, inventory=True),
    }


def _t2_composition_diagnostic(
    rows: list[Mapping[str, Any]],
    markers: Mapping[str, list[Mapping[str, Any]]],
    full_baseline: Mapping[str, Any],
    transport: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = {
        str(row["eventId"])
        for row in _held_rows(transport)
        if bool(row["selected"])
    }
    selected_ids.update(
        str(row["eventId"])
        for row in _held_rows(full_baseline)
        if bool(row["selected"]) and str(row["kind"]) == "internal-dead-state-peak"
    )
    predictions = np.asarray(
        [str(row["eventId"]) in selected_ids for row in rows], dtype=bool
    )
    return {
        "mergeContract": {
            "boundaryBranch": "T2 outer-held selection with boundary-fit threshold",
            "internalBranch": "exact full-union E0 outer-held internal selections",
            "probabilityCalibration": "none across branches; merge selected IDs only",
            "crossKindSuppression": "none in diagnostic; one-to-one event matching penalizes duplicates",
            "selectionUse": "diagnostic only; T2 decision uses boundary-only comparison",
        },
        "primary": evaluate_predictions(rows, predictions, markers, 4.0, inventory=True),
        "strict": evaluate_predictions(rows, predictions, markers, 0.0, inventory=True),
    }


def _t4_composition_diagnostic(
    rows: list[Mapping[str, Any]],
    markers: Mapping[str, list[Mapping[str, Any]]],
    full_baseline: Mapping[str, Any],
    selective_far: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = {
        str(row["eventId"])
        for row in _held_rows(selective_far)
        if bool(row["selected"])
    }
    selected_ids.update(
        str(row["eventId"])
        for row in _held_rows(full_baseline)
        if bool(row["selected"]) and str(row["kind"]) == "internal-dead-state-peak"
    )
    predictions = np.asarray(
        [str(row["eventId"]) in selected_ids for row in rows], dtype=bool
    )
    return {
        "mergeContract": {
            "boundaryBranch": "T4 outer-held selection with boundary-fit threshold",
            "internalBranch": "exact full-union E0 outer-held internal selections",
            "probabilityCalibration": "none across branches; merge selected IDs only",
            "crossKindSuppression": "none in diagnostic; one-to-one event matching penalizes duplicates",
            "selectionUse": "diagnostic only; T4 decision uses boundary-only comparison",
        },
        "primary": evaluate_predictions(rows, predictions, markers, 4.0, inventory=True),
        "strict": evaluate_predictions(rows, predictions, markers, 0.0, inventory=True),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "features": args.features.expanduser().resolve(),
        "audit": args.audit.expanduser().resolve(),
        "model": args.model.expanduser().resolve(),
        "evaluation": args.evaluation.expanduser().resolve(),
    }
    if args.experiment in {"E1", "E2"}:
        paths["importance"] = args.importance.expanduser().resolve()
    if args.experiment == "E2":
        paths["gapFeatures"] = args.gap_features.expanduser().resolve()
    if args.experiment in {"E4", "E5", "E6"}:
        paths["visualFeatures"] = args.visual_features.expanduser().resolve()
    if args.experiment == "M1":
        paths["m1Features"] = args.m1_features.expanduser().resolve()
    if args.experiment == "T2":
        paths["t2Features"] = args.t2_features.expanduser().resolve()
    if args.experiment == "T4":
        paths["t4Features"] = args.t4_features.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else DEFAULT_OUTPUTS[args.experiment].resolve()
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feature-development artifact: {output}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected_hashes = {
        name: EXPECTED_SHA256[name]
        for name in paths
    }
    if args.enforce_source_hash and hashes != expected_hashes:
        raise ValueError(f"feature-development source identity changed: {hashes}")

    feature_payload = _load(paths["features"])
    experiment_rows = feature_payload["rows"]
    if args.experiment == "E2":
        gap_payload = _load(paths["gapFeatures"])
        if gap_payload["scope"] != feature_payload["scope"] or [
            str(row["eventId"]) for row in gap_payload["rows"]
        ] != [str(row["eventId"]) for row in feature_payload["rows"]]:
            raise ValueError("E2 gap feature artifact does not match the frozen row universe")
        experiment_rows = gap_payload["rows"]
    elif args.experiment in {"E4", "E5", "E6"}:
        visual_payload = _load(paths["visualFeatures"])
        if visual_payload["scope"] != feature_payload["scope"] or [
            str(row["eventId"]) for row in visual_payload["rows"]
        ] != [str(row["eventId"]) for row in feature_payload["rows"]]:
            raise ValueError(
                f"{args.experiment} visual-summary artifact does not match the frozen row universe"
            )
        if visual_payload["parity"]["status"] != "exact-within-tolerance" or float(
            visual_payload["parity"]["maximumAbsoluteDifference"]
        ) > 1e-8:
            raise ValueError(
                f"{args.experiment} visual-summary artifact did not pass current parity"
            )
        experiment_rows = visual_payload["rows"]
    elif args.experiment == "M1":
        m1_payload = _load(paths["m1Features"])
        if m1_payload["scope"] != feature_payload["scope"] or [
            str(row["eventId"]) for row in m1_payload["rows"]
        ] != [str(row["eventId"]) for row in feature_payload["rows"]]:
            raise ValueError("M1 motion artifact does not match the frozen row universe")
        parity = m1_payload["parity"]
        if (
            int(parity["rows"]) != 704
            or int(parity["eligibleBoundaryRows"]) != 624
            or int(parity["ineligibleInternalRows"]) != 80
            or str(parity["existingFeatureValues"]) != "exact"
            or str(parity["candidateIdAndOrder"]) != "exact"
        ):
            raise ValueError("M1 motion artifact did not pass frozen parity")
        performance = m1_payload["performance"]
        budget = performance["budget"]
        if (
            int(performance["requestedFrames"]) != 7488
            or int(performance["flowPairs"]) != 3744
            or not bool(budget["wallTimePassed"])
            or not bool(budget["peakMemoryPassed"])
            or any(
                int(value["frameErrors"]) != 0
                for value in m1_payload["extractionAudit"].values()
            )
        ):
            raise ValueError("M1 motion artifact did not pass extraction gates")
        experiment_rows = m1_payload["rows"]
    elif args.experiment == "T2":
        t2_payload = _load(paths["t2Features"])
        if [str(row["eventId"]) for row in t2_payload["rows"]] != [
            str(row["eventId"]) for row in feature_payload["rows"]
        ]:
            raise ValueError("T2 artifact does not match the frozen row universe")
        parity = t2_payload["parity"]
        if (
            str(t2_payload["engineeringDecision"]) != "pass"
            or int(parity["rows"]) != 704
            or int(parity["eligibleBoundaryRows"]) != 624
            or int(parity["ineligibleInternalRows"]) != 80
            or str(parity["priorFeatureValues"]) != "exact"
            or str(parity["candidateIdAndOrder"]) != "exact"
            or tuple(t2_payload["contract"]["futureCoreFeatureNames"])
            != T2_CORE_FEATURE_NAMES
            or not all(bool(value) for value in t2_payload["engineering"]["checks"].values())
        ):
            raise ValueError("T2 artifact did not pass frozen engineering gates")
        experiment_rows = t2_payload["rows"]
    elif args.experiment == "T4":
        t4_payload = _load(paths["t4Features"])
        if [str(row["eventId"]) for row in t4_payload["rows"]] != [
            str(row["eventId"]) for row in feature_payload["rows"]
        ]:
            raise ValueError("T4 artifact does not match the frozen row universe")
        parity = t4_payload["parity"]
        if (
            str(t4_payload["engineeringDecision"]) != "pass"
            or int(parity["rows"]) != 704
            or int(parity["eligibleBoundaryRows"]) != 624
            or int(parity["ineligibleInternalRows"]) != 80
            or str(parity["priorFeatureValues"]) != "exact"
            or str(parity["candidateIdAndOrder"]) != "exact"
            or tuple(t4_payload["contract"]["futureFirstHeadFeatureNames"])
            != T4_CORE_FEATURE_NAMES
            or not all(bool(value) for value in t4_payload["engineering"]["checks"].values())
        ):
            raise ValueError("T4 artifact did not pass frozen engineering gates")
        experiment_rows = t4_payload["rows"]
    audit = _load(paths["audit"])
    current_model = _load(paths["model"])
    current_evaluation = _load(paths["evaluation"])
    recording_ids = tuple(
        str(value) for value in feature_payload["scope"]["recordingIds"]
    )
    markers = {
        recording_id: [
            {"time": float(value)}
            for value in audit["scope"]["humanEventsByRecording"][recording_id]
        ]
        for recording_id in recording_ids
    }

    baseline = evaluate_profile(experiment_rows, markers, BASELINE_PROFILE)
    parity = _validate_e0(baseline, current_model, current_evaluation)
    profiles: dict[str, Any] = {BASELINE_PROFILE.identifier: baseline}
    comparison: dict[str, Any] | None = None
    composition_diagnostic: dict[str, Any] | None = None
    feature_importance: dict[str, Any] | None = None
    if args.experiment == "E1":
        interaction = evaluate_profile(
            feature_payload["rows"], markers, INTERACTION_PROFILE
        )
        profiles[INTERACTION_PROFILE.identifier] = interaction
        comparison = _evaluate_e1_gate(
            baseline, interaction, _load(paths["importance"])
        )
    elif args.experiment == "E2":
        gap_shape = evaluate_profile(experiment_rows, markers, GAP_SHAPE_PROFILE)
        profiles[GAP_SHAPE_PROFILE.identifier] = gap_shape
        comparison = _evaluate_e2_gate(
            baseline, gap_shape, _load(paths["importance"])
        )
    elif args.experiment == "E4":
        directional = evaluate_profile(experiment_rows, markers, Q1_PROFILE)
        profiles[Q1_PROFILE.identifier] = directional
        comparison = _evaluate_e4_gate(baseline, directional, experiment_rows)
    elif args.experiment == "E5":
        camera = evaluate_profile(experiment_rows, markers, C1_PROFILE)
        profiles[C1_PROFILE.identifier] = camera
        comparison = _evaluate_e5_gate(baseline, camera, experiment_rows)
    elif args.experiment == "E6":
        boundary_rows = [
            row
            for row in experiment_rows
            if str(row["kind"]) == "adjacent-rally-boundary"
        ]
        boundary_baseline = evaluate_profile(
            boundary_rows, markers, BOUNDARY_BASELINE_PROFILE
        )
        persistence = evaluate_profile(boundary_rows, markers, P1_PROFILE)
        profiles[BOUNDARY_BASELINE_PROFILE.identifier] = boundary_baseline
        profiles[P1_PROFILE.identifier] = persistence
        comparison = _evaluate_e6_gate(
            boundary_baseline, persistence, boundary_rows
        )
        composition_diagnostic = _e6_composition_diagnostic(
            experiment_rows, markers, baseline, persistence
        )
    elif args.experiment == "M1":
        boundary_rows = [
            row
            for row in experiment_rows
            if str(row["kind"]) == "adjacent-rally-boundary"
        ]
        boundary_baseline = evaluate_profile(
            boundary_rows, markers, BOUNDARY_BASELINE_PROFILE
        )
        motion = evaluate_profile(boundary_rows, markers, M1_PROFILE)
        profiles[BOUNDARY_BASELINE_PROFILE.identifier] = boundary_baseline
        profiles[M1_PROFILE.identifier] = motion
        comparison = _evaluate_m1_gate(boundary_baseline, motion, boundary_rows)
        composition_diagnostic = _m1_composition_diagnostic(
            experiment_rows, markers, baseline, motion
        )
    elif args.experiment == "T2":
        boundary_rows = [
            row
            for row in experiment_rows
            if str(row["kind"]) == "adjacent-rally-boundary"
        ]
        boundary_baseline = evaluate_profile(
            boundary_rows, markers, BOUNDARY_BASELINE_PROFILE
        )
        transport = evaluate_profile(boundary_rows, markers, T2_PROFILE)
        swap_only = evaluate_profile(
            boundary_rows, markers, T2_SWAP_ONLY_PROFILE
        )
        conditional_only = evaluate_profile(
            boundary_rows, markers, T2_CONDITIONAL_ONLY_PROFILE
        )
        profiles[BOUNDARY_BASELINE_PROFILE.identifier] = boundary_baseline
        profiles[T2_PROFILE.identifier] = transport
        profiles[T2_SWAP_ONLY_PROFILE.identifier] = swap_only
        profiles[T2_CONDITIONAL_ONLY_PROFILE.identifier] = conditional_only
        comparison = _evaluate_t2_gate(
            boundary_baseline, transport, boundary_rows
        )
        feature_importance = _t2_coefficient_importance(transport)
        feature_importance["descriptiveSingleValueAblations"] = {
            T2_SWAP_ONLY_PROFILE.identifier: metric_delta(
                boundary_baseline, swap_only
            ),
            T2_CONDITIONAL_ONLY_PROFILE.identifier: metric_delta(
                boundary_baseline, conditional_only
            ),
        }
        composition_diagnostic = _t2_composition_diagnostic(
            experiment_rows, markers, baseline, transport
        )
    elif args.experiment == "T4":
        boundary_rows = [
            row
            for row in experiment_rows
            if str(row["kind"]) == "adjacent-rally-boundary"
        ]
        boundary_baseline = evaluate_profile(
            boundary_rows, markers, BOUNDARY_BASELINE_PROFILE
        )
        selective_far = evaluate_profile(boundary_rows, markers, T4_PROFILE)
        profiles[BOUNDARY_BASELINE_PROFILE.identifier] = boundary_baseline
        profiles[T4_PROFILE.identifier] = selective_far
        comparison = _evaluate_t4_gate(
            boundary_baseline, selective_far, boundary_rows
        )
        feature_importance = _t4_coefficient_importance(selective_far)
        composition_diagnostic = _t4_composition_diagnostic(
            experiment_rows, markers, baseline, selective_far
        )
    script_path = Path(__file__).resolve()
    module_path = (script_path.parent.parent / "analysis/side_switch_feature_development.py").resolve()
    created_at = datetime.now(UTC).isoformat()
    payload = {
        "schemaVersion": 1,
        "kind": f"volleycut-side-switch-feature-development-{args.experiment.lower()}-v1",
        "createdAt": created_at,
        "status": "opened-development-only",
        "experiment": {
            "id": args.experiment,
            "name": {
                "E0": "freeze-and-reproduce-control",
                "E1": "swap-specific-interactions",
                "E2": "production-gap-consensus-and-shape",
                "E4": "directional-observation-quality-replacement",
                "E5": "camera-and-scene-confounders",
                "E6": "boundary-multi-rally-persistence",
                "M1": "boundary-foreground-side-exchange-motion",
                "T2": "boundary-conditional-identity-transport-opened-development",
                "T4": "boundary-selective-far-jersey-transport-opened-development",
            }[args.experiment],
            "decision": "baseline-only" if comparison is None else comparison["decision"],
        },
        "scope": {
            **feature_payload["scope"],
            "humanMarkers": sum(len(value) for value in markers.values()),
            "positiveCandidateLabels": baseline["positiveCandidateLabels"],
        },
        "protocol": {
            "outer": "leave one recording out",
            "inner": "grouped leave-one-recording-out threshold selection on outer-fit recordings",
            "classifier": "fixed square-root logistic, L2 0.1, top-2/2x recording-balanced hard negatives",
            "decoder": DECODER.to_dict(),
            "primaryPaddingSeconds": 4.0,
        },
        "extractionPerformance": (
            _load(paths["m1Features"])["performance"]
            if args.experiment == "M1"
            else _load(paths["t2Features"])["performance"]
            if args.experiment == "T2"
            else _load(paths["t4Features"])["performance"]
            if args.experiment == "T4"
            else None
        ),
        "profiles": profiles,
        "parity": parity,
        "comparison": comparison,
        "compositionDiagnostic": composition_diagnostic,
        "featureImportance": feature_importance,
        "sources": {
            **{
                name: {"path": str(paths[name]), "sha256": hashes[name]}
                for name in paths
            },
            "runner": {"path": str(script_path), "sha256": _sha256(script_path)},
            "module": {"path": str(module_path), "sha256": _sha256(module_path)},
        },
        "limitations": [
            "All 11 recordings and 50 markers are opened development data.",
            (
                "E0 validates experiment machinery; it is not a new model result."
                if args.experiment == "E0"
                else (
                    f"{args.experiment} formulae and gates were frozen in the checked-in "
                    "development plan before this run."
                )
            ),
            *(
                [
                    "T2 was trained only because the user explicitly authorized reuse of the opened labels.",
                    "A T2 gain cannot authorize promotion or runtime porting without new recording-held gold.",
                    "The two single-value profiles are descriptive ablations, not model-selection contenders.",
                ]
                if args.experiment == "T2"
                else [
                    "T4 was trained only because selective detection passed every frozen engineering gate and the user authorized the experiment.",
                    "A T4 gain cannot authorize promotion or runtime porting without new recording-held gold.",
                    "The three-value T4 bundle is not pruned unless the full bundle passes every frozen model gate.",
                ]
                if args.experiment == "T4"
                else []
            ),
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--importance", type=Path, default=DEFAULT_IMPORTANCE)
    parser.add_argument("--gap-features", type=Path, default=DEFAULT_GAP_FEATURES)
    parser.add_argument("--visual-features", type=Path, default=DEFAULT_VISUAL_FEATURES)
    parser.add_argument("--m1-features", type=Path, default=DEFAULT_M1_FEATURES)
    parser.add_argument("--t2-features", type=Path, default=DEFAULT_T2_FEATURES)
    parser.add_argument("--t4-features", type=Path, default=DEFAULT_T4_FEATURES)
    parser.add_argument("--experiment", choices=tuple(DEFAULT_OUTPUTS), default="E0")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    payload = run(_parser().parse_args())
    metrics = {
        name: concise_metrics(result)
        for name, result in payload["profiles"].items()
    }
    print(
        json.dumps(
            {
                "experiment": payload["experiment"],
                "parity": payload["parity"],
                "metrics": metrics,
                "comparison": payload["comparison"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
