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
    T5_PROFILE,
    T14_PROFILE,
    T16_PROFILE,
    T18_PROFILE,
    T19_PROFILE,
    T20_PROFILE,
    concise_metrics,
    evaluate_profile,
    evaluate_predictions,
    metric_delta,
)
from analysis.side_switch_t2_transport import T2_CORE_FEATURE_NAMES
from analysis.side_switch_t4_selective_far import T4_CORE_FEATURE_NAMES
from analysis.side_switch_t5_court_tracking import T5_CORE_FEATURE_NAMES
from analysis.side_switch_t14_dominant_tracklet_medoid import T14_CORE_FEATURE_NAMES
from analysis.side_switch_t16_source_resolved import T16_CORE_FEATURE_NAMES
from analysis.side_switch_t18_representativeness import T18_CORE_FEATURE_NAMES
from analysis.side_switch_t19_cross_representation import T19_CORE_FEATURE_NAMES
from analysis.side_switch_t20_compact_candidate import T20_CORE_FEATURE_NAMES


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
DEFAULT_T5_FEATURES = (
    REPORTS / "side-switch-t5-court-constrained-temporal-team-tracking-features-v1.json"
)
DEFAULT_T14_FEATURES = (
    REPORTS / "side-switch-t14-dominant-tracklet-medoid-features-v1.json"
)
DEFAULT_T16_FEATURES = REPORTS / "side-switch-t16-source-resolved-medoid-features-v1.json"
DEFAULT_T18_FEATURES = REPORTS / "side-switch-t18-medoid-representativeness-features-v1.json"
DEFAULT_T19_FEATURES = REPORTS / "side-switch-t19-cross-representation-consensus-features-v1.json"
DEFAULT_T20_FEATURES = REPORTS / "side-switch-t20-compact-medoid-disagreement-features-v1.json"
DEFAULT_T4_MODEL_RESULT = (
    REPORTS / "side-switch-feature-development-t4-selective-far-opened-v1.json"
)
DEFAULT_T14_MODEL_RESULT = (
    REPORTS / "side-switch-feature-development-t14-dominant-tracklet-medoid-opened-v1.json"
)
DEFAULT_T19_MODEL_RESULT = (
    REPORTS / "side-switch-feature-development-t19-cross-representation-consensus-opened-v1.json"
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
    "T5": REPORTS / "side-switch-feature-development-t5-court-tracking-diagnostic-v1.json",
    "T14": REPORTS / "side-switch-feature-development-t14-dominant-tracklet-medoid-opened-v1.json",
    "T16": REPORTS / "side-switch-feature-development-t16-source-resolved-medoid-opened-v1.json",
    "T18": REPORTS / "side-switch-feature-development-t18-medoid-representativeness-opened-v1.json",
    "T19": REPORTS / "side-switch-feature-development-t19-cross-representation-consensus-opened-v1.json",
    "T20": REPORTS / "side-switch-feature-development-t20-compact-medoid-disagreement-opened-v1.json",
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
    "t5Features": "25d8a23563f803058f23ba2da23df0e82283ab893afd143b5f39200d96fcb460",
    "t14Features": "d68e0d9a2aa72e1fe3cb06f413fdb43dacd52b9090dd4241863596459f8b9753",
    "t16Features": "11bcedafe98284bd306398f44946720767a5323275aea38a7948ae43d7d01692",
    "t18Features": "0efab13db25849a74413fc595e2753c735d1c9eb00ae4afaf8b73b9007ef8949",
    "t19Features": "ab4b474fdbaee20bf884093a37eba25e79b6e93172f0dbfe619c3b0546c421e5",
    "t20Features": "f770eed53f85dd79f398a23e37d0706e80925971a300538082088ad59e0d5c1e",
    "t4ModelResult": "16d617fb8427517b45c31196877d7482cb019f8902b989e5b2db19cfcef0020b",
    "t14ModelResult": "ba6b7493f4d2c98a5d6bd755947a1f917ce4a5d0b69a9cb07c7b48c903d54f06",
    "t19ModelResult": "b67e2502f05cba27ba9a0b1fc19fd614506264b8319db0befd0bd17b33c615c7",
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


def _evaluate_t5_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    t4_result: Mapping[str, Any],
) -> dict[str, Any]:
    delta = metric_delta(baseline, candidate)
    baseline_scores = _held_rows(baseline)
    candidate_selected = _selected_ids(candidate)
    reliable_swap_name = "courtTrackedFarJerseyReliableSwapEvidence"
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
    t4_profile = t4_result["profiles"][
        "boundary-union34-plus-selective-far-jersey-t4"
    ]
    t4_f1 = float(t4_profile["primary"]["f1"])
    candidate_f1 = float(candidate["primary"]["f1"])
    checks = {
        **_standalone_checks(delta),
        "primaryF1GainOverT4AtLeast0_5pp": candidate_f1 - t4_f1
        >= 0.005 - 1e-12,
        "coveredBoundaryMissesRecovered": recovered_covered > 0,
        "zeroReliableSwapEvidenceFalseBoundarySliceReduced": (
            retained_zero_evidence_false < len(zero_evidence_false_ids)
        ),
        "recordingRobustness": not gain_depends_on_one_recording,
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "interpretation": "user-authorized-diagnostic-after-engineering-reject",
        "engineeringDecisionRemains": "fail",
        "checks": checks,
        "delta": delta,
        "immutableT4Comparison": {
            "t4PrimaryF1": t4_f1,
            "t5PrimaryF1": candidate_f1,
            "delta": candidate_f1 - t4_f1,
        },
        "targetSlices": {
            "coveredBoundaryMisses": {
                "definition": (
                    "positive T0 misses with nonzero court-tracked reliable swap evidence"
                ),
                "frozenCandidateIds": sorted(covered_miss_ids),
                "baselineSelected": 0,
                "candidateSelected": recovered_covered,
                "change": recovered_covered,
            },
            "zeroReliableSwapEvidenceFalseBoundaries": {
                "definition": (
                    "T0-selected false boundaries with exactly zero "
                    "courtTrackedFarJerseyReliableSwapEvidence"
                ),
                "frozenCandidateIds": sorted(zero_evidence_false_ids),
                "baselineSelected": len(zero_evidence_false_ids),
                "candidateSelected": retained_zero_evidence_false,
                "change": retained_zero_evidence_false - len(zero_evidence_false_ids),
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


def _t5_coefficient_importance(candidate: Mapping[str, Any]) -> dict[str, Any]:
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
    fold_weights = {name: [] for name in T5_CORE_FEATURE_NAMES}
    for fold in candidate["outerFolds"]:
        fit = fold["fitStandardizedWeights"]
        fit_names = [str(name) for name in fit["featureNames"]]
        for name in T5_CORE_FEATURE_NAMES:
            fold_weights[name].append(float(fit["weights"][fit_names.index(name)]))
    features = []
    for name in T5_CORE_FEATURE_NAMES:
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
            "weights act on fold-standardized inputs; this diagnostic does not "
            "override the failed T5 engineering decision"
        ),
        "features": features,
    }


def _evaluate_t14_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    t4_result: Mapping[str, Any],
) -> dict[str, Any]:
    delta = metric_delta(baseline, candidate)
    baseline_scores = _held_rows(baseline)
    candidate_selected = _selected_ids(candidate)
    reliable_swap_name = "dominantTrackletJerseyReliableSwapEvidence"
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
    t4_profile = t4_result["profiles"][
        "boundary-union34-plus-selective-far-jersey-t4"
    ]
    t4_f1 = float(t4_profile["primary"]["f1"])
    candidate_f1 = float(candidate["primary"]["f1"])
    classifier = candidate["fullDevelopment"]["classifier"]
    names = [str(name) for name in classifier["featureNames"]]
    full_reliable_weight = float(
        classifier["weights"][names.index(reliable_swap_name)]
    )
    outer_reliable_weights = []
    for fold in candidate["outerFolds"]:
        fit = fold["fitStandardizedWeights"]
        fit_names = [str(name) for name in fit["featureNames"]]
        outer_reliable_weights.append(
            float(fit["weights"][fit_names.index(reliable_swap_name)])
        )
    positive_outer_fits = sum(value > 0.0 for value in outer_reliable_weights)
    checks = {
        **_standalone_checks(delta),
        "primaryF1GainOverT4AtLeast0_5pp": candidate_f1 - t4_f1
        >= 0.005 - 1e-12,
        "coveredBoundaryMissesRecovered": recovered_covered > 0,
        "zeroReliableSwapEvidenceFalseBoundarySliceReduced": (
            retained_zero_evidence_false < len(zero_evidence_false_ids)
        ),
        "recordingRobustness": not gain_depends_on_one_recording,
        "reliableSwapCoefficientPositiveInFullFit": full_reliable_weight > 0.0,
        "reliableSwapCoefficientPositiveInAtLeast9Of11OuterFits": (
            positive_outer_fits >= 9
        ),
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "interpretation": "exploratory-opened-development-only",
        "checks": checks,
        "delta": delta,
        "immutableT4Comparison": {
            "t4PrimaryF1": t4_f1,
            "t14PrimaryF1": candidate_f1,
            "delta": candidate_f1 - t4_f1,
        },
        "reliableSwapCoefficientGate": {
            "feature": reliable_swap_name,
            "fullDevelopmentStandardizedCoefficient": full_reliable_weight,
            "outerFitCoefficients": outer_reliable_weights,
            "outerFitPositiveCount": positive_outer_fits,
            "requiredPositiveOuterFits": 9,
        },
        "targetSlices": {
            "coveredBoundaryMisses": {
                "definition": (
                    "positive T0 misses with nonzero dominant-tracklet reliable "
                    "swap evidence"
                ),
                "frozenCandidateIds": sorted(covered_miss_ids),
                "baselineSelected": 0,
                "candidateSelected": recovered_covered,
                "change": recovered_covered,
            },
            "zeroReliableSwapEvidenceFalseBoundaries": {
                "definition": (
                    "T0-selected false boundaries with exactly zero "
                    "dominantTrackletJerseyReliableSwapEvidence"
                ),
                "frozenCandidateIds": sorted(zero_evidence_false_ids),
                "baselineSelected": len(zero_evidence_false_ids),
                "candidateSelected": retained_zero_evidence_false,
                "change": retained_zero_evidence_false - len(zero_evidence_false_ids),
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


def _t14_coefficient_importance(candidate: Mapping[str, Any]) -> dict[str, Any]:
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
    fold_weights = {name: [] for name in T14_CORE_FEATURE_NAMES}
    for fold in candidate["outerFolds"]:
        fit = fold["fitStandardizedWeights"]
        fit_names = [str(name) for name in fit["featureNames"]]
        for name in T14_CORE_FEATURE_NAMES:
            fold_weights[name].append(float(fit["weights"][fit_names.index(name)]))
    features = []
    for name in T14_CORE_FEATURE_NAMES:
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


def _evaluate_t16_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    t14_result: Mapping[str, Any],
) -> dict[str, Any]:
    delta = metric_delta(baseline, candidate)
    baseline_scores = _held_rows(baseline)
    candidate_selected = _selected_ids(candidate)
    near_name, far_name = T16_CORE_FEATURE_NAMES
    feature_values = {
        str(row["eventId"]): (
            float(row["features"][near_name]),
            float(row["features"][far_name]),
        )
        for row in rows
    }
    covered_miss_ids = {
        str(row["eventId"])
        for row in baseline_scores
        if (
            int(row["label"]) == 1
            and not bool(row["selected"])
            and max(feature_values[str(row["eventId"])]) > 0.0
        )
    }
    disagreement_false_ids = {
        str(row["eventId"])
        for row in baseline_scores
        if (
            int(row["label"]) == 0
            and bool(row["selected"])
            and feature_values[str(row["eventId"])][0]
            * feature_values[str(row["eventId"])][1]
            <= 0.0
        )
    }
    recovered_covered = len(covered_miss_ids & candidate_selected)
    retained_disagreement_false = len(disagreement_false_ids & candidate_selected)
    recording_deltas = _recording_deltas(baseline, candidate)
    total_true_positive_gain = int(delta["truePositives"])
    maximum_recording_gain = max(
        (int(value["truePositives"]) for value in recording_deltas.values()), default=0
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
    t14_profile = t14_result["profiles"][
        "boundary-union34-plus-dominant-tracklet-medoid-t14"
    ]
    t14_f1 = float(t14_profile["primary"]["f1"])
    candidate_f1 = float(candidate["primary"]["f1"])
    classifier = candidate["fullDevelopment"]["classifier"]
    names = [str(name) for name in classifier["featureNames"]]
    full_weights = {
        name: float(classifier["weights"][names.index(name)])
        for name in T16_CORE_FEATURE_NAMES
    }
    outer_weights = {name: [] for name in T16_CORE_FEATURE_NAMES}
    for fold in candidate["outerFolds"]:
        fit = fold["fitStandardizedWeights"]
        fit_names = [str(name) for name in fit["featureNames"]]
        for name in T16_CORE_FEATURE_NAMES:
            outer_weights[name].append(float(fit["weights"][fit_names.index(name)]))
    positive_counts = {
        name: sum(value > 0.0 for value in values)
        for name, values in outer_weights.items()
    }
    checks = {
        **_standalone_checks(delta),
        "primaryF1GainOverT14AtLeast0_5pp": candidate_f1 - t14_f1
        >= 0.005 - 1e-12,
        "coveredBoundaryMissesRecovered": recovered_covered > 0,
        "sourceDisagreementFalseBoundarySliceReduced": (
            retained_disagreement_false < len(disagreement_false_ids)
        ),
        "recordingRobustness": not gain_depends_on_one_recording,
        "bothSourceCoefficientsPositiveInFullFit": all(
            value > 0.0 for value in full_weights.values()
        ),
        "bothSourceCoefficientsPositiveInAtLeast9Of11OuterFits": all(
            value >= 9 for value in positive_counts.values()
        ),
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "interpretation": "adaptive-opened-development-only",
        "checks": checks,
        "delta": delta,
        "immutableT14Comparison": {
            "t14PrimaryF1": t14_f1,
            "t16PrimaryF1": candidate_f1,
            "delta": candidate_f1 - t14_f1,
        },
        "sourceCoefficientGate": {
            name: {
                "fullDevelopmentStandardizedCoefficient": full_weights[name],
                "outerFitCoefficients": outer_weights[name],
                "outerFitPositiveCount": positive_counts[name],
                "requiredPositiveOuterFits": 9,
            }
            for name in T16_CORE_FEATURE_NAMES
        },
        "targetSlices": {
            "coveredBoundaryMisses": {
                "definition": "positive T0 misses with either source advantage positive",
                "frozenCandidateIds": sorted(covered_miss_ids),
                "baselineSelected": 0,
                "candidateSelected": recovered_covered,
                "change": recovered_covered,
            },
            "sourceDisagreementFalseBoundaries": {
                "definition": "T0-selected false boundaries whose source advantages disagree or include zero",
                "frozenCandidateIds": sorted(disagreement_false_ids),
                "baselineSelected": len(disagreement_false_ids),
                "candidateSelected": retained_disagreement_false,
                "change": retained_disagreement_false - len(disagreement_false_ids),
            },
        },
        "recordingRobustness": {
            "gainDependsOnOneRecordingWhileAtLeastThreeRegress": gain_depends_on_one_recording,
            "totalTruePositiveGain": total_true_positive_gain,
            "maximumSingleRecordingTruePositiveGain": maximum_recording_gain,
            "truePositiveGainWithoutBestRecording": total_true_positive_gain - maximum_recording_gain,
            "regressingRecordingIds": regressing_recordings,
        },
        "byRecordingDelta": recording_deltas,
    }


def _t16_coefficient_importance(candidate: Mapping[str, Any]) -> dict[str, Any]:
    classifier = candidate["fullDevelopment"]["classifier"]
    names = [str(name) for name in classifier["featureNames"]]
    weights = [float(value) for value in classifier["weights"]]
    ranks = {
        name: rank
        for rank, name in enumerate(
            sorted(names, key=lambda name: (-abs(weights[names.index(name)]), name)), 1
        )
    }
    fold_weights = {name: [] for name in T16_CORE_FEATURE_NAMES}
    for fold in candidate["outerFolds"]:
        fit = fold["fitStandardizedWeights"]
        fit_names = [str(name) for name in fit["featureNames"]]
        for name in T16_CORE_FEATURE_NAMES:
            fold_weights[name].append(float(fit["weights"][fit_names.index(name)]))
    return {
        "coefficientContract": (
            "weights act on fold-standardized inputs; magnitudes are comparable "
            "within a fit but do not establish causal importance"
        ),
        "features": [
            {
                "feature": name,
                "fullDevelopmentStandardizedCoefficient": weights[names.index(name)],
                "absoluteCoefficientRankAmong36": ranks[name],
                "outerFitCoefficients": fold_weights[name],
                "outerFitPositiveCount": sum(value > 0 for value in fold_weights[name]),
                "outerFitNegativeCount": sum(value < 0 for value in fold_weights[name]),
                "outerFitMean": float(np.mean(fold_weights[name])),
                "outerFitStandardDeviation": float(np.std(fold_weights[name])),
                "outerFitMinimum": min(fold_weights[name]),
                "outerFitMaximum": max(fold_weights[name]),
            }
            for name in T16_CORE_FEATURE_NAMES
        ],
    }


def _evaluate_t18_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    t14_result: Mapping[str, Any],
) -> dict[str, Any]:
    delta = metric_delta(baseline, candidate)
    baseline_scores = _held_rows(baseline)
    candidate_selected = _selected_ids(candidate)
    feature_name = T18_CORE_FEATURE_NAMES[0]
    margin_values = {
        str(row["eventId"]): float(row["features"][feature_name]) for row in rows
    }
    gate_values = {
        str(row["eventId"]): float(
            row["features"]["representativeMedoidJerseyGateMinimum"]
        )
        for row in rows
    }
    gate_quartile = float(np.quantile(list(gate_values.values()), 0.25))
    covered_miss_ids = {
        str(row["eventId"])
        for row in baseline_scores
        if int(row["label"]) == 1
        and not bool(row["selected"])
        and margin_values[str(row["eventId"])] > 0.0
    }
    low_gate_false_ids = {
        str(row["eventId"])
        for row in baseline_scores
        if int(row["label"]) == 0
        and bool(row["selected"])
        and gate_values[str(row["eventId"])] <= gate_quartile
    }
    recovered_covered = len(covered_miss_ids & candidate_selected)
    retained_low_gate_false = len(low_gate_false_ids & candidate_selected)
    recording_deltas = _recording_deltas(baseline, candidate)
    total_true_positive_gain = int(delta["truePositives"])
    maximum_recording_gain = max(
        (int(value["truePositives"]) for value in recording_deltas.values()), default=0
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
    t14_profile = t14_result["profiles"][
        "boundary-union34-plus-dominant-tracklet-medoid-t14"
    ]
    t14_f1 = float(t14_profile["primary"]["f1"])
    candidate_f1 = float(candidate["primary"]["f1"])
    classifier = candidate["fullDevelopment"]["classifier"]
    names = [str(name) for name in classifier["featureNames"]]
    full_weight = float(classifier["weights"][names.index(feature_name)])
    outer_weights = []
    for fold in candidate["outerFolds"]:
        fit = fold["fitStandardizedWeights"]
        fit_names = [str(name) for name in fit["featureNames"]]
        outer_weights.append(float(fit["weights"][fit_names.index(feature_name)]))
    positive_count = sum(value > 0.0 for value in outer_weights)
    checks = {
        **_standalone_checks(delta),
        "primaryF1GainOverT14AtLeast0_5pp": candidate_f1 - t14_f1 >= 0.005 - 1e-12,
        "coveredBoundaryMissesRecovered": recovered_covered > 0,
        "lowRepresentativenessFalseBoundarySliceReduced": retained_low_gate_false < len(low_gate_false_ids),
        "recordingRobustness": not gain_depends_on_one_recording,
        "representativeMarginCoefficientPositiveInFullFit": full_weight > 0.0,
        "representativeMarginCoefficientPositiveInAtLeast9Of11OuterFits": positive_count >= 9,
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "interpretation": "adaptive-opened-development-only",
        "checks": checks,
        "delta": delta,
        "immutableT14Comparison": {"t14PrimaryF1": t14_f1, "t18PrimaryF1": candidate_f1, "delta": candidate_f1 - t14_f1},
        "coefficientGate": {"feature": feature_name, "fullDevelopmentStandardizedCoefficient": full_weight, "outerFitCoefficients": outer_weights, "outerFitPositiveCount": positive_count, "requiredPositiveOuterFits": 9},
        "targetSlices": {
            "coveredBoundaryMisses": {"definition": "positive T0 misses with positive representative medoid margin", "frozenCandidateIds": sorted(covered_miss_ids), "baselineSelected": 0, "candidateSelected": recovered_covered, "change": recovered_covered},
            "lowRepresentativenessFalseBoundaries": {"definition": "T0-selected false boundaries at or below the all-row gate first quartile", "gateFirstQuartile": gate_quartile, "frozenCandidateIds": sorted(low_gate_false_ids), "baselineSelected": len(low_gate_false_ids), "candidateSelected": retained_low_gate_false, "change": retained_low_gate_false - len(low_gate_false_ids)},
        },
        "recordingRobustness": {"gainDependsOnOneRecordingWhileAtLeastThreeRegress": gain_depends_on_one_recording, "totalTruePositiveGain": total_true_positive_gain, "maximumSingleRecordingTruePositiveGain": maximum_recording_gain, "truePositiveGainWithoutBestRecording": total_true_positive_gain - maximum_recording_gain, "regressingRecordingIds": regressing_recordings},
        "byRecordingDelta": recording_deltas,
    }


def _t18_coefficient_importance(candidate: Mapping[str, Any]) -> dict[str, Any]:
    feature_name = T18_CORE_FEATURE_NAMES[0]
    classifier = candidate["fullDevelopment"]["classifier"]
    names = [str(name) for name in classifier["featureNames"]]
    weights = [float(value) for value in classifier["weights"]]
    ranks = {
        name: rank
        for rank, name in enumerate(
            sorted(names, key=lambda name: (-abs(weights[names.index(name)]), name)), 1
        )
    }
    outer_weights = []
    for fold in candidate["outerFolds"]:
        fit = fold["fitStandardizedWeights"]
        fit_names = [str(name) for name in fit["featureNames"]]
        outer_weights.append(float(fit["weights"][fit_names.index(feature_name)]))
    return {
        "coefficientContract": "weights act on fold-standardized inputs; magnitudes do not establish causal importance",
        "features": [{
            "feature": feature_name,
            "fullDevelopmentStandardizedCoefficient": weights[names.index(feature_name)],
            "absoluteCoefficientRankAmong35": ranks[feature_name],
            "outerFitCoefficients": outer_weights,
            "outerFitPositiveCount": sum(value > 0 for value in outer_weights),
            "outerFitNegativeCount": sum(value < 0 for value in outer_weights),
            "outerFitMean": float(np.mean(outer_weights)),
            "outerFitStandardDeviation": float(np.std(outer_weights)),
            "outerFitMinimum": min(outer_weights),
            "outerFitMaximum": max(outer_weights),
        }],
    }


def _evaluate_t19_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    t14_result: Mapping[str, Any],
) -> dict[str, Any]:
    delta = metric_delta(baseline, candidate)
    baseline_scores = _held_rows(baseline)
    candidate_selected = _selected_ids(candidate)
    consensus_name, disagreement_name = T19_CORE_FEATURE_NAMES
    consensus_values = {str(row["eventId"]): float(row["features"][consensus_name]) for row in rows}
    source_values = {
        str(row["eventId"]): (
            float(row["features"]["selectiveFarJerseyTeamTransportSwapMargin"]),
            float(row["features"]["dominantTrackletJerseyTeamTransportSwapMargin"]),
        )
        for row in rows
    }
    covered_miss_ids = {
        str(row["eventId"])
        for row in baseline_scores
        if int(row["label"]) == 1 and not bool(row["selected"]) and consensus_values[str(row["eventId"])] > 0.0
    }
    disagreement_false_ids = {
        str(row["eventId"])
        for row in baseline_scores
        if int(row["label"]) == 0 and bool(row["selected"]) and source_values[str(row["eventId"])][0] * source_values[str(row["eventId"])][1] <= 0.0
    }
    recovered_covered = len(covered_miss_ids & candidate_selected)
    retained_disagreement_false = len(disagreement_false_ids & candidate_selected)
    recording_deltas = _recording_deltas(baseline, candidate)
    total_true_positive_gain = int(delta["truePositives"])
    maximum_recording_gain = max((int(value["truePositives"]) for value in recording_deltas.values()), default=0)
    regressing_recordings = sorted(recording_id for recording_id, value in recording_deltas.items() if int(value["truePositives"]) < 0)
    gain_depends_on_one_recording = total_true_positive_gain > 0 and total_true_positive_gain - maximum_recording_gain <= 0 and len(regressing_recordings) >= 3
    t14_profile = t14_result["profiles"]["boundary-union34-plus-dominant-tracklet-medoid-t14"]
    t14_f1 = float(t14_profile["primary"]["f1"])
    candidate_f1 = float(candidate["primary"]["f1"])
    classifier = candidate["fullDevelopment"]["classifier"]
    names = [str(name) for name in classifier["featureNames"]]
    full_weights = {name: float(classifier["weights"][names.index(name)]) for name in T19_CORE_FEATURE_NAMES}
    outer_weights = {name: [] for name in T19_CORE_FEATURE_NAMES}
    for fold in candidate["outerFolds"]:
        fit = fold["fitStandardizedWeights"]
        fit_names = [str(name) for name in fit["featureNames"]]
        for name in T19_CORE_FEATURE_NAMES:
            outer_weights[name].append(float(fit["weights"][fit_names.index(name)]))
    consensus_positive_count = sum(value > 0.0 for value in outer_weights[consensus_name])
    disagreement_negative_count = sum(value < 0.0 for value in outer_weights[disagreement_name])
    checks = {
        **_standalone_checks(delta),
        "primaryF1GainOverT14AtLeast0_5pp": candidate_f1 - t14_f1 >= 0.005 - 1e-12,
        "coveredBoundaryMissesRecovered": recovered_covered > 0,
        "crossRepresentationSignDisagreementFalseSliceReduced": retained_disagreement_false < len(disagreement_false_ids),
        "recordingRobustness": not gain_depends_on_one_recording,
        "consensusCoefficientPositiveInFullFit": full_weights[consensus_name] > 0.0,
        "consensusCoefficientPositiveInAtLeast9Of11OuterFits": consensus_positive_count >= 9,
        "disagreementCoefficientNegativeInFullFit": full_weights[disagreement_name] < 0.0,
        "disagreementCoefficientNegativeInAtLeast9Of11OuterFits": disagreement_negative_count >= 9,
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "interpretation": "adaptive-opened-development-only",
        "checks": checks,
        "delta": delta,
        "immutableT14Comparison": {"t14PrimaryF1": t14_f1, "t19PrimaryF1": candidate_f1, "delta": candidate_f1 - t14_f1},
        "coefficientGates": {
            consensus_name: {"expectedSign": "positive", "fullDevelopmentStandardizedCoefficient": full_weights[consensus_name], "outerFitCoefficients": outer_weights[consensus_name], "outerFitExpectedSignCount": consensus_positive_count, "requiredExpectedSignOuterFits": 9},
            disagreement_name: {"expectedSign": "negative", "fullDevelopmentStandardizedCoefficient": full_weights[disagreement_name], "outerFitCoefficients": outer_weights[disagreement_name], "outerFitExpectedSignCount": disagreement_negative_count, "requiredExpectedSignOuterFits": 9},
        },
        "targetSlices": {
            "coveredBoundaryMisses": {"definition": "positive T0 misses with positive cross-representation consensus", "frozenCandidateIds": sorted(covered_miss_ids), "baselineSelected": 0, "candidateSelected": recovered_covered, "change": recovered_covered},
            "crossRepresentationSignDisagreementFalseBoundaries": {"definition": "T0-selected false boundaries whose T4/T14 raw signs disagree or include zero", "frozenCandidateIds": sorted(disagreement_false_ids), "baselineSelected": len(disagreement_false_ids), "candidateSelected": retained_disagreement_false, "change": retained_disagreement_false - len(disagreement_false_ids)},
        },
        "recordingRobustness": {"gainDependsOnOneRecordingWhileAtLeastThreeRegress": gain_depends_on_one_recording, "totalTruePositiveGain": total_true_positive_gain, "maximumSingleRecordingTruePositiveGain": maximum_recording_gain, "truePositiveGainWithoutBestRecording": total_true_positive_gain - maximum_recording_gain, "regressingRecordingIds": regressing_recordings},
        "byRecordingDelta": recording_deltas,
    }


def _t19_coefficient_importance(candidate: Mapping[str, Any]) -> dict[str, Any]:
    classifier = candidate["fullDevelopment"]["classifier"]
    names = [str(name) for name in classifier["featureNames"]]
    weights = [float(value) for value in classifier["weights"]]
    ranks = {name: rank for rank, name in enumerate(sorted(names, key=lambda name: (-abs(weights[names.index(name)]), name)), 1)}
    fold_weights = {name: [] for name in T19_CORE_FEATURE_NAMES}
    for fold in candidate["outerFolds"]:
        fit = fold["fitStandardizedWeights"]
        fit_names = [str(name) for name in fit["featureNames"]]
        for name in T19_CORE_FEATURE_NAMES:
            fold_weights[name].append(float(fit["weights"][fit_names.index(name)]))
    return {
        "coefficientContract": "weights act on fold-standardized inputs; magnitudes do not establish causal importance",
        "features": [{"feature": name, "fullDevelopmentStandardizedCoefficient": weights[names.index(name)], "absoluteCoefficientRankAmong36": ranks[name], "outerFitCoefficients": fold_weights[name], "outerFitPositiveCount": sum(value > 0 for value in fold_weights[name]), "outerFitNegativeCount": sum(value < 0 for value in fold_weights[name]), "outerFitMean": float(np.mean(fold_weights[name])), "outerFitStandardDeviation": float(np.std(fold_weights[name])), "outerFitMinimum": min(fold_weights[name]), "outerFitMaximum": max(fold_weights[name])} for name in T19_CORE_FEATURE_NAMES],
    }


def _evaluate_t20_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    t14_result: Mapping[str, Any],
    t19_result: Mapping[str, Any],
) -> dict[str, Any]:
    delta = metric_delta(baseline, candidate)
    baseline_scores = _held_rows(baseline)
    candidate_selected = _selected_ids(candidate)
    margin_name, disagreement_name = T20_CORE_FEATURE_NAMES
    margin_values = {
        str(row["eventId"]): float(row["features"][margin_name]) for row in rows
    }
    source_values = {
        str(row["eventId"]): (
            float(row["features"]["selectiveFarJerseyTeamTransportSwapMargin"]),
            float(row["features"]["dominantTrackletJerseyTeamTransportSwapMargin"]),
        )
        for row in rows
    }
    covered_miss_ids = {
        str(row["eventId"])
        for row in baseline_scores
        if int(row["label"]) == 1
        and not bool(row["selected"])
        and margin_values[str(row["eventId"])] > 0.0
    }
    disagreement_false_ids = {
        str(row["eventId"])
        for row in baseline_scores
        if int(row["label"]) == 0
        and bool(row["selected"])
        and source_values[str(row["eventId"])][0]
        * source_values[str(row["eventId"])][1]
        <= 0.0
    }
    recovered_covered = len(covered_miss_ids & candidate_selected)
    retained_disagreement_false = len(disagreement_false_ids & candidate_selected)
    recording_deltas = _recording_deltas(baseline, candidate)
    total_true_positive_gain = int(delta["truePositives"])
    maximum_recording_gain = max(
        (int(value["truePositives"]) for value in recording_deltas.values()), default=0
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
    t14_f1 = float(
        t14_result["profiles"][
            "boundary-union34-plus-dominant-tracklet-medoid-t14"
        ]["primary"]["f1"]
    )
    t19_f1 = float(
        t19_result["profiles"][
            "boundary-union34-plus-cross-representation-consensus-t19"
        ]["primary"]["f1"]
    )
    candidate_f1 = float(candidate["primary"]["f1"])
    classifier = candidate["fullDevelopment"]["classifier"]
    names = [str(name) for name in classifier["featureNames"]]
    full_weights = {
        name: float(classifier["weights"][names.index(name)])
        for name in T20_CORE_FEATURE_NAMES
    }
    outer_weights = {name: [] for name in T20_CORE_FEATURE_NAMES}
    for fold in candidate["outerFolds"]:
        fit = fold["fitStandardizedWeights"]
        fit_names = [str(name) for name in fit["featureNames"]]
        for name in T20_CORE_FEATURE_NAMES:
            outer_weights[name].append(float(fit["weights"][fit_names.index(name)]))
    margin_positive_count = sum(value > 0.0 for value in outer_weights[margin_name])
    disagreement_negative_count = sum(
        value < 0.0 for value in outer_weights[disagreement_name]
    )
    checks = {
        **_standalone_checks(delta),
        "primaryF1GainOverT14AtLeast0_5pp": candidate_f1 - t14_f1
        >= 0.005 - 1e-12,
        "primaryF1NoLowerThanT19": candidate_f1 >= t19_f1 - 1e-12,
        "coveredBoundaryMissesRecovered": recovered_covered > 0,
        "crossRepresentationSignDisagreementFalseSliceReduced": (
            retained_disagreement_false < len(disagreement_false_ids)
        ),
        "recordingRobustness": not gain_depends_on_one_recording,
        "compactMarginCoefficientPositiveInFullFit": full_weights[margin_name] > 0.0,
        "compactMarginCoefficientPositiveInAtLeast9Of11OuterFits": (
            margin_positive_count >= 9
        ),
        "compactDisagreementCoefficientNegativeInFullFit": (
            full_weights[disagreement_name] < 0.0
        ),
        "compactDisagreementCoefficientNegativeInAtLeast9Of11OuterFits": (
            disagreement_negative_count >= 9
        ),
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "interpretation": "adaptive-opened-development-screen-only",
        "independentValidation": False,
        "checks": checks,
        "delta": delta,
        "immutableComparisons": {
            "t14PrimaryF1": t14_f1,
            "t19PrimaryF1": t19_f1,
            "t20PrimaryF1": candidate_f1,
            "deltaOverT14": candidate_f1 - t14_f1,
            "deltaOverT19": candidate_f1 - t19_f1,
        },
        "coefficientGates": {
            margin_name: {
                "expectedSign": "positive",
                "fullDevelopmentStandardizedCoefficient": full_weights[margin_name],
                "outerFitCoefficients": outer_weights[margin_name],
                "outerFitExpectedSignCount": margin_positive_count,
                "requiredExpectedSignOuterFits": 9,
            },
            disagreement_name: {
                "expectedSign": "negative",
                "fullDevelopmentStandardizedCoefficient": full_weights[disagreement_name],
                "outerFitCoefficients": outer_weights[disagreement_name],
                "outerFitExpectedSignCount": disagreement_negative_count,
                "requiredExpectedSignOuterFits": 9,
            },
        },
        "targetSlices": {
            "coveredBoundaryMisses": {
                "definition": "positive T0 misses with positive compact medoid margin",
                "frozenCandidateIds": sorted(covered_miss_ids),
                "baselineSelected": 0,
                "candidateSelected": recovered_covered,
                "change": recovered_covered,
            },
            "crossRepresentationSignDisagreementFalseBoundaries": {
                "definition": "T0-selected false boundaries whose T4/T14 directions disagree or include zero",
                "frozenCandidateIds": sorted(disagreement_false_ids),
                "baselineSelected": len(disagreement_false_ids),
                "candidateSelected": retained_disagreement_false,
                "change": retained_disagreement_false - len(disagreement_false_ids),
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
    }


def _t20_coefficient_importance(candidate: Mapping[str, Any]) -> dict[str, Any]:
    classifier = candidate["fullDevelopment"]["classifier"]
    names = [str(name) for name in classifier["featureNames"]]
    weights = [float(value) for value in classifier["weights"]]
    ranks = {
        name: rank
        for rank, name in enumerate(
            sorted(names, key=lambda name: (-abs(weights[names.index(name)]), name)), 1
        )
    }
    fold_weights = {name: [] for name in T20_CORE_FEATURE_NAMES}
    for fold in candidate["outerFolds"]:
        fit = fold["fitStandardizedWeights"]
        fit_names = [str(name) for name in fit["featureNames"]]
        for name in T20_CORE_FEATURE_NAMES:
            fold_weights[name].append(float(fit["weights"][fit_names.index(name)]))
    return {
        "coefficientContract": (
            "weights act on fold-standardized inputs; this adaptive assembly was "
            "selected from opened T14/T19 results"
        ),
        "features": [
            {
                "feature": name,
                "fullDevelopmentStandardizedCoefficient": weights[names.index(name)],
                "absoluteCoefficientRankAmong36": ranks[name],
                "outerFitCoefficients": fold_weights[name],
                "outerFitPositiveCount": sum(value > 0 for value in fold_weights[name]),
                "outerFitNegativeCount": sum(value < 0 for value in fold_weights[name]),
                "outerFitMean": float(np.mean(fold_weights[name])),
                "outerFitStandardDeviation": float(np.std(fold_weights[name])),
                "outerFitMinimum": min(fold_weights[name]),
                "outerFitMaximum": max(fold_weights[name]),
            }
            for name in T20_CORE_FEATURE_NAMES
        ],
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


def _t5_composition_diagnostic(
    rows: list[Mapping[str, Any]],
    markers: Mapping[str, list[Mapping[str, Any]]],
    full_baseline: Mapping[str, Any],
    court_tracking: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = {
        str(row["eventId"])
        for row in _held_rows(court_tracking)
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
            "boundaryBranch": "T5 outer-held selection with boundary-fit threshold",
            "internalBranch": "exact full-union E0 outer-held internal selections",
            "selectionUse": "diagnostic only after engineering rejection",
        },
        "primary": evaluate_predictions(rows, predictions, markers, 4.0, inventory=True),
        "strict": evaluate_predictions(rows, predictions, markers, 0.0, inventory=True),
    }


def _t14_composition_diagnostic(
    rows: list[Mapping[str, Any]],
    markers: Mapping[str, list[Mapping[str, Any]]],
    full_baseline: Mapping[str, Any],
    dominant_tracklet: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = {
        str(row["eventId"])
        for row in _held_rows(dominant_tracklet)
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
            "boundaryBranch": "T14 outer-held selection with boundary-fit threshold",
            "internalBranch": "exact full-union E0 outer-held internal selections",
            "selectionUse": "diagnostic only; T14 decision uses boundary-only comparison",
        },
        "primary": evaluate_predictions(rows, predictions, markers, 4.0, inventory=True),
        "strict": evaluate_predictions(rows, predictions, markers, 0.0, inventory=True),
    }


def _t16_composition_diagnostic(
    rows: list[Mapping[str, Any]],
    markers: Mapping[str, list[Mapping[str, Any]]],
    full_baseline: Mapping[str, Any],
    source_resolved: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = {
        str(row["eventId"])
        for row in _held_rows(source_resolved)
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
            "boundaryBranch": "T16 outer-held selection with boundary-fit threshold",
            "internalBranch": "exact full-union E0 outer-held internal selections",
            "selectionUse": "diagnostic only; T16 decision uses boundary-only comparison",
        },
        "primary": evaluate_predictions(rows, predictions, markers, 4.0, inventory=True),
        "strict": evaluate_predictions(rows, predictions, markers, 0.0, inventory=True),
    }


def _t18_composition_diagnostic(
    rows: list[Mapping[str, Any]],
    markers: Mapping[str, list[Mapping[str, Any]]],
    full_baseline: Mapping[str, Any],
    representative: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = {
        str(row["eventId"])
        for row in _held_rows(representative)
        if bool(row["selected"])
    }
    selected_ids.update(
        str(row["eventId"])
        for row in _held_rows(full_baseline)
        if bool(row["selected"]) and str(row["kind"]) == "internal-dead-state-peak"
    )
    predictions = np.asarray([str(row["eventId"]) in selected_ids for row in rows], dtype=bool)
    return {
        "mergeContract": {"boundaryBranch": "T18 outer-held selection with boundary-fit threshold", "internalBranch": "exact full-union E0 outer-held internal selections", "selectionUse": "diagnostic only; T18 decision uses boundary-only comparison"},
        "primary": evaluate_predictions(rows, predictions, markers, 4.0, inventory=True),
        "strict": evaluate_predictions(rows, predictions, markers, 0.0, inventory=True),
    }


def _t19_composition_diagnostic(
    rows: list[Mapping[str, Any]],
    markers: Mapping[str, list[Mapping[str, Any]]],
    full_baseline: Mapping[str, Any],
    consensus: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = {str(row["eventId"]) for row in _held_rows(consensus) if bool(row["selected"])}
    selected_ids.update(str(row["eventId"]) for row in _held_rows(full_baseline) if bool(row["selected"]) and str(row["kind"]) == "internal-dead-state-peak")
    predictions = np.asarray([str(row["eventId"]) in selected_ids for row in rows], dtype=bool)
    return {
        "mergeContract": {"boundaryBranch": "T19 outer-held selection with boundary-fit threshold", "internalBranch": "exact full-union E0 outer-held internal selections", "selectionUse": "diagnostic only; T19 decision uses boundary-only comparison"},
        "primary": evaluate_predictions(rows, predictions, markers, 4.0, inventory=True),
        "strict": evaluate_predictions(rows, predictions, markers, 0.0, inventory=True),
    }


def _t20_composition_diagnostic(
    rows: list[Mapping[str, Any]],
    markers: Mapping[str, list[Mapping[str, Any]]],
    full_baseline: Mapping[str, Any],
    compact: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = {
        str(row["eventId"])
        for row in _held_rows(compact)
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
            "boundaryBranch": "T20 outer-held selection with boundary-fit threshold",
            "internalBranch": "exact full-union E0 outer-held internal selections",
            "selectionUse": "adaptive diagnostic only; T20 screen uses boundary comparison",
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
    if args.experiment == "T5":
        paths["t5Features"] = args.t5_features.expanduser().resolve()
        paths["t4ModelResult"] = args.t4_model_result.expanduser().resolve()
    if args.experiment == "T14":
        paths["t14Features"] = args.t14_features.expanduser().resolve()
        paths["t4ModelResult"] = args.t4_model_result.expanduser().resolve()
    if args.experiment == "T16":
        paths["t16Features"] = args.t16_features.expanduser().resolve()
        paths["t14ModelResult"] = args.t14_model_result.expanduser().resolve()
    if args.experiment == "T18":
        paths["t18Features"] = args.t18_features.expanduser().resolve()
        paths["t14ModelResult"] = args.t14_model_result.expanduser().resolve()
    if args.experiment == "T19":
        paths["t19Features"] = args.t19_features.expanduser().resolve()
        paths["t14ModelResult"] = args.t14_model_result.expanduser().resolve()
    if args.experiment == "T20":
        paths["t20Features"] = args.t20_features.expanduser().resolve()
        paths["t14ModelResult"] = args.t14_model_result.expanduser().resolve()
        paths["t19ModelResult"] = args.t19_model_result.expanduser().resolve()
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
    elif args.experiment == "T5":
        t5_payload = _load(paths["t5Features"])
        if [str(row["eventId"]) for row in t5_payload["rows"]] != [
            str(row["eventId"]) for row in feature_payload["rows"]
        ]:
            raise ValueError("T5 artifact does not match the frozen row universe")
        parity = t5_payload["parity"]
        checks = t5_payload["engineering"]["checks"]
        expected_failed_check = "reliableSwapEvidenceNonzeroGainAtLeast3pp"
        if (
            str(t5_payload["engineeringDecision"]) != "fail"
            or int(parity["rows"]) != 704
            or int(parity["eligibleBoundaryRows"]) != 624
            or int(parity["ineligibleInternalRows"]) != 80
            or str(parity["priorFeatureValues"]) != "exact"
            or str(parity["candidateIdAndOrder"]) != "exact"
            or tuple(t5_payload["contract"]["futureFirstHeadFeatureNames"])
            != T5_CORE_FEATURE_NAMES
            or bool(checks[expected_failed_check])
            or not all(
                bool(value) for name, value in checks.items() if name != expected_failed_check
            )
        ):
            raise ValueError("T5 artifact does not match the authorized engineering reject")
        experiment_rows = t5_payload["rows"]
    elif args.experiment == "T14":
        t14_payload = _load(paths["t14Features"])
        if [str(row["eventId"]) for row in t14_payload["rows"]] != [
            str(row["eventId"]) for row in feature_payload["rows"]
        ]:
            raise ValueError("T14 artifact does not match the frozen row universe")
        parity = t14_payload["parity"]
        if (
            str(t14_payload["engineeringDecision"]) != "pass"
            or int(parity["rows"]) != 704
            or int(parity["eligibleBoundaryRows"]) != 624
            or int(parity["ineligibleInternalRows"]) != 80
            or str(parity["priorFeatureValues"]) != "exact"
            or str(parity["candidateIdAndOrder"]) != "exact"
            or tuple(t14_payload["contract"]["futureFirstHeadFeatureNames"])
            != T14_CORE_FEATURE_NAMES
            or not all(bool(value) for value in t14_payload["engineering"]["checks"].values())
        ):
            raise ValueError("T14 artifact did not pass frozen engineering gates")
        experiment_rows = t14_payload["rows"]
    elif args.experiment == "T16":
        t16_payload = _load(paths["t16Features"])
        if [str(row["eventId"]) for row in t16_payload["rows"]] != [
            str(row["eventId"]) for row in feature_payload["rows"]
        ]:
            raise ValueError("T16 artifact does not match the frozen row universe")
        parity = t16_payload["parity"]
        if (
            str(t16_payload["engineeringDecision"]) != "pass"
            or int(parity["rows"]) != 704
            or int(parity["eligibleBoundaryRows"]) != 624
            or int(parity["ineligibleInternalRows"]) != 80
            or str(parity["priorFeatureValues"]) != "exact"
            or str(parity["candidateIdAndOrder"]) != "exact"
            or tuple(t16_payload["contract"]["futureFirstHeadFeatureNames"])
            != T16_CORE_FEATURE_NAMES
            or not all(bool(value) for value in t16_payload["engineering"]["checks"].values())
        ):
            raise ValueError("T16 artifact did not pass frozen engineering gates")
        experiment_rows = t16_payload["rows"]
    elif args.experiment == "T18":
        t18_payload = _load(paths["t18Features"])
        if [str(row["eventId"]) for row in t18_payload["rows"]] != [str(row["eventId"]) for row in feature_payload["rows"]]:
            raise ValueError("T18 artifact does not match the frozen row universe")
        parity = t18_payload["parity"]
        if (
            str(t18_payload["engineeringDecision"]) != "pass"
            or int(parity["rows"]) != 704
            or int(parity["eligibleBoundaryRows"]) != 624
            or int(parity["ineligibleInternalRows"]) != 80
            or str(parity["priorFeatureValues"]) != "exact"
            or str(parity["candidateIdAndOrder"]) != "exact"
            or tuple(t18_payload["contract"]["futureFirstHeadFeatureNames"]) != T18_CORE_FEATURE_NAMES
            or not all(bool(value) for value in t18_payload["engineering"]["checks"].values())
        ):
            raise ValueError("T18 artifact did not pass frozen engineering gates")
        experiment_rows = t18_payload["rows"]
    elif args.experiment == "T19":
        t19_payload = _load(paths["t19Features"])
        if [str(row["eventId"]) for row in t19_payload["rows"]] != [str(row["eventId"]) for row in feature_payload["rows"]]:
            raise ValueError("T19 artifact does not match the frozen row universe")
        parity = t19_payload["parity"]
        if (
            str(t19_payload["engineeringDecision"]) != "pass"
            or int(parity["rows"]) != 704
            or int(parity["eligibleBoundaryRows"]) != 624
            or int(parity["ineligibleInternalRows"]) != 80
            or str(parity["priorFeatureValues"]) != "exact"
            or str(parity["candidateIdAndOrder"]) != "exact"
            or tuple(t19_payload["contract"]["futureFirstHeadFeatureNames"]) != T19_CORE_FEATURE_NAMES
            or not all(bool(value) for value in t19_payload["engineering"]["checks"].values())
        ):
            raise ValueError("T19 artifact did not pass frozen engineering gates")
        experiment_rows = t19_payload["rows"]
    elif args.experiment == "T20":
        t20_payload = _load(paths["t20Features"])
        if [str(row["eventId"]) for row in t20_payload["rows"]] != [
            str(row["eventId"]) for row in feature_payload["rows"]
        ]:
            raise ValueError("T20 artifact does not match the frozen row universe")
        parity = t20_payload["parity"]
        if (
            str(t20_payload["engineeringDecision"]) != "pass"
            or int(parity["rows"]) != 704
            or int(parity["eligibleBoundaryRows"]) != 624
            or int(parity["ineligibleInternalRows"]) != 80
            or str(parity["priorFeatureValues"]) != "exact"
            or str(parity["candidateIdAndOrder"]) != "exact"
            or tuple(t20_payload["contract"]["futureFirstHeadFeatureNames"])
            != T20_CORE_FEATURE_NAMES
            or not all(bool(value) for value in t20_payload["engineering"]["checks"].values())
        ):
            raise ValueError("T20 artifact did not pass frozen engineering gates")
        experiment_rows = t20_payload["rows"]
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
    elif args.experiment == "T5":
        boundary_rows = [
            row
            for row in experiment_rows
            if str(row["kind"]) == "adjacent-rally-boundary"
        ]
        boundary_baseline = evaluate_profile(
            boundary_rows, markers, BOUNDARY_BASELINE_PROFILE
        )
        court_tracking = evaluate_profile(boundary_rows, markers, T5_PROFILE)
        profiles[BOUNDARY_BASELINE_PROFILE.identifier] = boundary_baseline
        profiles[T5_PROFILE.identifier] = court_tracking
        comparison = _evaluate_t5_gate(
            boundary_baseline,
            court_tracking,
            boundary_rows,
            _load(paths["t4ModelResult"]),
        )
        feature_importance = _t5_coefficient_importance(court_tracking)
        composition_diagnostic = _t5_composition_diagnostic(
            experiment_rows, markers, baseline, court_tracking
        )
    elif args.experiment == "T14":
        boundary_rows = [
            row
            for row in experiment_rows
            if str(row["kind"]) == "adjacent-rally-boundary"
        ]
        boundary_baseline = evaluate_profile(
            boundary_rows, markers, BOUNDARY_BASELINE_PROFILE
        )
        dominant_tracklet = evaluate_profile(boundary_rows, markers, T14_PROFILE)
        profiles[BOUNDARY_BASELINE_PROFILE.identifier] = boundary_baseline
        profiles[T14_PROFILE.identifier] = dominant_tracklet
        comparison = _evaluate_t14_gate(
            boundary_baseline,
            dominant_tracklet,
            boundary_rows,
            _load(paths["t4ModelResult"]),
        )
        feature_importance = _t14_coefficient_importance(dominant_tracklet)
        composition_diagnostic = _t14_composition_diagnostic(
            experiment_rows, markers, baseline, dominant_tracklet
        )
    elif args.experiment == "T16":
        boundary_rows = [
            row for row in experiment_rows if str(row["kind"]) == "adjacent-rally-boundary"
        ]
        boundary_baseline = evaluate_profile(
            boundary_rows, markers, BOUNDARY_BASELINE_PROFILE
        )
        source_resolved = evaluate_profile(boundary_rows, markers, T16_PROFILE)
        profiles[BOUNDARY_BASELINE_PROFILE.identifier] = boundary_baseline
        profiles[T16_PROFILE.identifier] = source_resolved
        comparison = _evaluate_t16_gate(
            boundary_baseline,
            source_resolved,
            boundary_rows,
            _load(paths["t14ModelResult"]),
        )
        feature_importance = _t16_coefficient_importance(source_resolved)
        composition_diagnostic = _t16_composition_diagnostic(
            experiment_rows, markers, baseline, source_resolved
        )
    elif args.experiment == "T18":
        boundary_rows = [row for row in experiment_rows if str(row["kind"]) == "adjacent-rally-boundary"]
        boundary_baseline = evaluate_profile(boundary_rows, markers, BOUNDARY_BASELINE_PROFILE)
        representative = evaluate_profile(boundary_rows, markers, T18_PROFILE)
        profiles[BOUNDARY_BASELINE_PROFILE.identifier] = boundary_baseline
        profiles[T18_PROFILE.identifier] = representative
        comparison = _evaluate_t18_gate(boundary_baseline, representative, boundary_rows, _load(paths["t14ModelResult"]))
        feature_importance = _t18_coefficient_importance(representative)
        composition_diagnostic = _t18_composition_diagnostic(experiment_rows, markers, baseline, representative)
    elif args.experiment == "T19":
        boundary_rows = [row for row in experiment_rows if str(row["kind"]) == "adjacent-rally-boundary"]
        boundary_baseline = evaluate_profile(boundary_rows, markers, BOUNDARY_BASELINE_PROFILE)
        cross_consensus = evaluate_profile(boundary_rows, markers, T19_PROFILE)
        profiles[BOUNDARY_BASELINE_PROFILE.identifier] = boundary_baseline
        profiles[T19_PROFILE.identifier] = cross_consensus
        comparison = _evaluate_t19_gate(boundary_baseline, cross_consensus, boundary_rows, _load(paths["t14ModelResult"]))
        feature_importance = _t19_coefficient_importance(cross_consensus)
        composition_diagnostic = _t19_composition_diagnostic(experiment_rows, markers, baseline, cross_consensus)
    elif args.experiment == "T20":
        boundary_rows = [
            row
            for row in experiment_rows
            if str(row["kind"]) == "adjacent-rally-boundary"
        ]
        boundary_baseline = evaluate_profile(
            boundary_rows, markers, BOUNDARY_BASELINE_PROFILE
        )
        compact = evaluate_profile(boundary_rows, markers, T20_PROFILE)
        profiles[BOUNDARY_BASELINE_PROFILE.identifier] = boundary_baseline
        profiles[T20_PROFILE.identifier] = compact
        comparison = _evaluate_t20_gate(
            boundary_baseline,
            compact,
            boundary_rows,
            _load(paths["t14ModelResult"]),
            _load(paths["t19ModelResult"]),
        )
        feature_importance = _t20_coefficient_importance(compact)
        composition_diagnostic = _t20_composition_diagnostic(
            experiment_rows, markers, baseline, compact
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
                "T5": "boundary-court-tracked-jersey-diagnostic-after-engineering-reject",
                "T14": "boundary-dominant-tracklet-medoid-opened-development",
                "T16": "boundary-source-resolved-medoid-opened-development",
                "T18": "boundary-representative-medoid-opened-development",
                "T19": "boundary-cross-representation-consensus-opened-development",
                "T20": "boundary-compact-medoid-disagreement-adaptive-development",
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
            else _load(paths["t5Features"])["performance"]
            if args.experiment == "T5"
            else _load(paths["t14Features"])["performance"]
            if args.experiment == "T14"
            else _load(paths["t16Features"])["performance"]
            if args.experiment == "T16"
            else _load(paths["t18Features"])["performance"]
            if args.experiment == "T18"
            else _load(paths["t19Features"])["performance"]
            if args.experiment == "T19"
            else _load(paths["t20Features"])["performance"]
            if args.experiment == "T20"
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
                else [
                    "T5 training is a user-authorized diagnostic override after a failed engineering gate.",
                    "This result cannot reverse the engineering rejection or authorize tuning, promotion, or runtime work.",
                ]
                if args.experiment == "T5"
                else [
                    "T14 was trained only because every frozen engineering gate passed.",
                    "A T14 gain cannot authorize promotion or runtime porting without new recording-held gold.",
                    "The exact three-value T14 bundle is not pruned or tuned on this opened result.",
                ]
                if args.experiment == "T14"
                else [
                    "T16 is an adaptive opened-development comparison after T14/T15.",
                    "A T16 pass cannot authorize promotion or runtime work without new recording-held gold.",
                    "The exact two source values are not pruned, reweighted, or role-swapped on this result.",
                ]
                if args.experiment == "T16"
                else [
                    "T18 is an adaptive opened-development comparison after T14-T17.",
                    "A T18 pass cannot authorize promotion or runtime work without new recording-held gold.",
                    "The frozen representativeness minimum is not retuned on this result.",
                ]
                if args.experiment == "T18"
                else [
                    "T19 is an adaptive opened-development comparison after T14-T18.",
                    "A T19 pass cannot authorize promotion or runtime work without new recording-held gold.",
                    "The equal fusion weights and disagreement semantics are not retuned on this result.",
                ]
                if args.experiment == "T19"
                else [
                    "T20 feature selection is explicitly adaptive to opened T14/T19 results.",
                    "The T20 screen is implementation and candidate-freeze evidence, not independent validation.",
                    "A pass requires new recording-held gold before promotion or runtime work.",
                ]
                if args.experiment == "T20"
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
    parser.add_argument("--t5-features", type=Path, default=DEFAULT_T5_FEATURES)
    parser.add_argument("--t14-features", type=Path, default=DEFAULT_T14_FEATURES)
    parser.add_argument("--t16-features", type=Path, default=DEFAULT_T16_FEATURES)
    parser.add_argument("--t18-features", type=Path, default=DEFAULT_T18_FEATURES)
    parser.add_argument("--t19-features", type=Path, default=DEFAULT_T19_FEATURES)
    parser.add_argument("--t20-features", type=Path, default=DEFAULT_T20_FEATURES)
    parser.add_argument("--t4-model-result", type=Path, default=DEFAULT_T4_MODEL_RESULT)
    parser.add_argument("--t14-model-result", type=Path, default=DEFAULT_T14_MODEL_RESULT)
    parser.add_argument("--t19-model-result", type=Path, default=DEFAULT_T19_MODEL_RESULT)
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
