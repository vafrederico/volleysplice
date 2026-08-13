from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from analysis.feature_experiments import FeatureExperimentError
from analysis.multistate import STATE_ORDER
from analysis.multistate_joint_emissions import (
    CANDIDATE_NAMES,
    PRIMARY_JOINT_CANDIDATE,
    REFERENCE_CANDIDATE,
    SECONDARY_JOINT_CANDIDATE,
    _median_epoch_cap,
    _promotion,
    _public_evaluation,
    _select_inner,
    _state_diagnostic_report,
    _state_diagnostics,
)


def slice_metrics(rallies: int, matches: int) -> dict[str, float | int]:
    return {
        "rallies": rallies,
        "strictMatchRecall": matches / rallies if rallies else 0.0,
    }


def metrics(objective: float) -> dict[str, object]:
    return {
        "objective": objective,
        "eventF1": objective,
        "eventPrecision": 0.75,
        "timeIoU": objective,
        "liveTimeRecall": 0.85,
        "deadSecondsRetained": 100.0,
        "outcomeSlices": {
            "ordinaryLong": slice_metrics(10, 8),
            "shortAtMost3Seconds": slice_metrics(4, 3),
            "ace": slice_metrics(2, 1),
            "serviceFault": slice_metrics(2, 1),
        },
    }


def candidate_report(objective: float) -> dict[str, object]:
    aggregate = metrics(objective)
    return {
        "aggregate": aggregate,
        "bySourceGroup": {
            group: metrics(objective) for group in ("a", "b", "c")
        },
        "macroSourceGroup": {"objective": objective},
        "stateDiagnostics": {"aggregate": {"samples": 12}},
        "recordings": [],
    }


class MultistateJointEmissionsTests(unittest.TestCase):
    def test_inner_selection_keeps_v1_without_one_point_macro_gain(self) -> None:
        reports = {
            REFERENCE_CANDIDATE: candidate_report(0.60),
            PRIMARY_JOINT_CANDIDATE: candidate_report(0.609),
            SECONDARY_JOINT_CANDIDATE: candidate_report(0.608),
        }

        selected, detail = _select_inner(reports)

        self.assertEqual(selected, REFERENCE_CANDIDATE)
        self.assertFalse(
            detail["candidates"][PRIMARY_JOINT_CANDIDATE]["gate"]["eligible"]
        )

    def test_primary_joint_wins_and_tie_margin_protects_it_from_secondary(self) -> None:
        reports = {
            REFERENCE_CANDIDATE: candidate_report(0.60),
            PRIMARY_JOINT_CANDIDATE: candidate_report(0.62),
            SECONDARY_JOINT_CANDIDATE: candidate_report(0.624),
        }

        selected, detail = _select_inner(reports)

        self.assertEqual(selected, PRIMARY_JOINT_CANDIDATE)
        self.assertTrue(
            detail["candidates"][SECONDARY_JOINT_CANDIDATE]["gate"]["eligible"]
        )
        self.assertFalse(detail["calibrationUsedForSelection"])

    def test_secondary_joint_must_clear_tie_margin_to_replace_primary(self) -> None:
        reports = {
            REFERENCE_CANDIDATE: candidate_report(0.60),
            PRIMARY_JOINT_CANDIDATE: candidate_report(0.62),
            SECONDARY_JOINT_CANDIDATE: candidate_report(0.63),
        }

        selected, _detail = _select_inner(reports)

        self.assertEqual(selected, SECONDARY_JOINT_CANDIDATE)

    def test_state_diagnostics_report_confusion_and_proper_scores(self) -> None:
        targets = np.asarray([0, 1, 2, 3], dtype=np.int64)
        probabilities = np.full((4, len(STATE_ORDER)), 0.01 / 3.0)
        probabilities[np.arange(4), targets] = 0.99
        rows = [
            {
                "recordingId": "r1",
                "sourceGroup": "a",
                "targets": targets[:2],
                "probabilities": probabilities[:2],
            },
            {
                "recordingId": "r2",
                "sourceGroup": "b",
                "targets": targets[2:],
                "probabilities": probabilities[2:],
            },
        ]

        aggregate = _state_diagnostics(rows)
        report = _state_diagnostic_report(rows)

        self.assertEqual(aggregate["samples"], 4)
        self.assertAlmostEqual(aggregate["accuracy"], 1.0)
        self.assertEqual(
            aggregate["confusionRowsTruthColumnsPrediction"],
            np.eye(4, dtype=np.int64).tolist(),
        )
        self.assertAlmostEqual(aggregate["multiclassLogLoss"], -math.log(0.99))
        self.assertEqual(set(report["bySourceGroup"]), {"a", "b"})

    def test_state_diagnostics_reject_non_normalized_probabilities(self) -> None:
        with self.assertRaisesRegex(FeatureExperimentError, "not normalized"):
            _state_diagnostics(
                [
                    {
                        "sourceGroup": "a",
                        "targets": np.asarray([0]),
                        "probabilities": np.asarray([[0.5, 0.5, 0.5, 0.5]]),
                    }
                ]
            )

    def test_epoch_cap_is_rounded_inner_median(self) -> None:
        self.assertEqual(_median_epoch_cap([7, 11, 20]), 11)
        self.assertEqual(_median_epoch_cap([1, 2]), 2)
        with self.assertRaisesRegex(FeatureExperimentError, "epochs are invalid"):
            _median_epoch_cap([])

    def test_public_evaluation_strips_private_state_rows(self) -> None:
        evaluation = {
            "aggregate": {"objective": 0.5},
            "recordings": [{"id": "r"}],
            "_stateDiagnosticRows": [{"targets": np.asarray([0])}],
        }
        self.assertEqual(
            _public_evaluation(evaluation),
            {
                "aggregate": {"objective": 0.5},
                "recordings": [{"id": "r"}],
            },
        )

    def test_promotion_requires_a_joint_and_unanimous_outer_selection(self) -> None:
        base = {
            "checks": {"baseGate": True},
            "promoteSelectedCandidate": True,
        }
        with patch(
            "analysis.multistate_joint_emissions._base_promotion",
            return_value=({"paired": True}, base),
        ):
            _paired, passing = _promotion(
                {},
                {},
                {},
                selected_name=PRIMARY_JOINT_CANDIDATE,
                unanimous=True,
                margin=0.01,
                consistency=0.75,
            )
            _paired, v1 = _promotion(
                {},
                {},
                {},
                selected_name=REFERENCE_CANDIDATE,
                unanimous=True,
                margin=0.01,
                consistency=0.75,
            )
            _paired, mixed = _promotion(
                {},
                {},
                {},
                selected_name=PRIMARY_JOINT_CANDIDATE,
                unanimous=False,
                margin=0.01,
                consistency=0.75,
            )

        self.assertTrue(passing["promoteSelectedCandidate"])
        self.assertFalse(v1["promoteSelectedCandidate"])
        self.assertFalse(mixed["promoteSelectedCandidate"])

    def test_promotion_calls_real_operational_helper_with_valid_signature(self) -> None:
        reference = candidate_report(0.60)
        candidate = candidate_report(0.62)
        binary = candidate_report(0.59)

        paired, promotion = _promotion(
            reference,
            candidate,
            binary,
            selected_name=PRIMARY_JOINT_CANDIDATE,
            unanimous=True,
            margin=0.01,
            consistency=0.75,
        )

        self.assertEqual(
            set(paired),
            {"candidateMinusV1", "candidateMinusOperationalBinary"},
        )
        self.assertIn("operationalBinaryGate", promotion)
        self.assertIn("selectedCandidateIsJointMulticlass", promotion["checks"])

    def test_candidate_family_keeps_primary_and_secondary_scientifically_separate(self) -> None:
        self.assertEqual(
            CANDIDATE_NAMES,
            (
                "v1_naive_ovr",
                "joint_empirical_softmax",
                "joint_balanced_prior_corrected_softmax",
            ),
        )


if __name__ == "__main__":
    unittest.main()
