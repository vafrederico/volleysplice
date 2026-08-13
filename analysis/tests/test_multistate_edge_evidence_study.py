from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from analysis.feature_experiments import FeatureExperimentError
from analysis.metrics import evaluate_intervals, outcome_slice_metrics
from analysis.multistate import MultistateState
from analysis.multistate_edge_evidence_study import (
    ARM_NAMES,
    BOTH_NAME,
    CONTROL_NAME,
    SERVE_NAME,
    TERMINAL_NAME,
    _aggregate_serve_anchor,
    _aggregate_outer_with_anchor,
    _arm_evidence,
    _assert_control,
    _median_epoch,
    _mechanism_gate,
    _select_inner,
    _terminal_target,
)
from analysis.schema import Interval


def slice_metrics(rallies: int, matches: int) -> dict[str, float | int]:
    return {
        "rallies": rallies,
        "strictMatchRecall": matches / rallies if rallies else 0.0,
    }


def aggregate(
    objective: float,
    *,
    start_mae: float = 0.7,
    end_mae: float = 1.5,
    anchor_recall: float = 0.5,
    short: int = 3,
    fault: int = 2,
) -> dict[str, object]:
    return {
        "objective": objective,
        "eventF1": objective,
        "eventPrecision": 0.75,
        "timeIoU": objective,
        "liveTimeRecall": 0.85,
        "liveTimePrecision": 0.8,
        "deadSecondsRetained": 100.0,
        "startBoundaryMaeSeconds": start_mae,
        "endBoundaryMaeSeconds": end_mae,
        "serveAnchorWithin05": {"recall": anchor_recall},
        "outcomeSlices": {
            "ordinaryLong": slice_metrics(10, 8),
            "shortAtMost3Seconds": slice_metrics(5, short),
            "serviceFault": slice_metrics(4, fault),
            "ace": slice_metrics(2, 1),
        },
    }


def report(value: dict[str, object]) -> dict[str, object]:
    return {
        "aggregate": value,
        "bySourceGroup": {
            group: value for group in ("a", "b", "c")
        },
        "macroSourceGroup": {
            "objective": value["objective"],
            "eventF1": value["eventF1"],
            "timeIoU": value["timeIoU"],
            "liveTimeRecall": value["liveTimeRecall"],
        },
        "recordings": [],
    }


class ConstantModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.full(len(values), self.probability, dtype=np.float64)


class MultistateEdgeEvidenceStudyTests(unittest.TestCase):
    def test_arm_family_is_the_frozen_four_way_ablation(self) -> None:
        self.assertEqual(
            ARM_NAMES,
            (CONTROL_NAME, SERVE_NAME, TERMINAL_NAME, BOTH_NAME),
        )

    def test_edge_mapping_is_fixed_bounded_and_local(self) -> None:
        item = SimpleNamespace(
            contextual_values=np.zeros((3, 2), dtype=np.float32),
            sequence=SimpleNamespace(times=np.asarray([0.0, 0.25, 0.5])),
        )
        serve = ConstantModel(0.75)
        terminal = ConstantModel(0.25)

        none = _arm_evidence(CONTROL_NAME, item, serve, terminal)
        both = _arm_evidence(BOTH_NAME, item, serve, terminal)

        self.assertIsNone(none)
        assert both is not None
        np.testing.assert_array_equal(
            both[(MultistateState.SETUP, MultistateState.SERVE)],
            np.full(3, 0.5),
        )
        np.testing.assert_array_equal(
            both[(MultistateState.LIVE, MultistateState.DEAD)],
            np.full(3, -0.5),
        )
        self.assertEqual(len(both), 2)

    def test_terminal_target_has_local_end_labels_and_pre_serve_negatives(self) -> None:
        times = np.arange(0.0, 12.25, 0.25)
        item = SimpleNamespace(
            sequence=SimpleNamespace(times=times),
            recording=SimpleNamespace(
                rallies=(SimpleNamespace(start=5.0, end=9.0),)
            ),
            sample_mask=np.ones(len(times), dtype=bool),
        )

        labels, mask = _terminal_target(item)

        self.assertTrue(np.all(mask[(times >= 4.0) & (times < 5.0)]))
        self.assertTrue(np.all(labels[(times >= 4.0) & (times < 5.0)] == 0.0))
        self.assertTrue(np.all(labels[(times >= 7.0) & (times < 9.0)] == 0.0))
        self.assertTrue(np.all(labels[(times >= 9.0) & (times < 11.0)] == 1.0))
        self.assertFalse(np.any(mask[times < 4.0]))

    def test_serve_mechanism_requires_anchor_gain_and_start_nonworsening(self) -> None:
        control = report(aggregate(0.60))
        passing = report(
            aggregate(0.62, start_mae=0.65, anchor_recall=0.55)
        )
        no_anchor_gain = report(
            aggregate(0.62, start_mae=0.65, anchor_recall=0.5)
        )

        self.assertTrue(
            _mechanism_gate(control, passing, arm=SERVE_NAME)["eligible"]
        )
        self.assertFalse(
            _mechanism_gate(control, no_anchor_gain, arm=SERVE_NAME)["eligible"]
        )

    def test_terminal_mechanism_accepts_boundary_or_slice_gain(self) -> None:
        control = report(aggregate(0.60))
        boundary = report(aggregate(0.62, end_mae=1.44))
        slices = report(aggregate(0.62, end_mae=1.5, short=4, fault=3))
        neither = report(aggregate(0.62, end_mae=1.49))

        self.assertTrue(
            _mechanism_gate(control, boundary, arm=TERMINAL_NAME)["eligible"]
        )
        self.assertTrue(
            _mechanism_gate(control, slices, arm=TERMINAL_NAME)["eligible"]
        )
        self.assertFalse(
            _mechanism_gate(control, neither, arm=TERMINAL_NAME)["eligible"]
        )

    def test_both_arm_requires_both_independent_arms(self) -> None:
        control = report(aggregate(0.60))
        serve = report(aggregate(0.62, start_mae=0.65, anchor_recall=0.55))
        terminal = report(aggregate(0.62, end_mae=1.44))
        both = report(
            aggregate(
                0.64, start_mae=0.65, end_mae=1.44, anchor_recall=0.55
            )
        )

        selected, detail = _select_inner(
            {
                CONTROL_NAME: control,
                SERVE_NAME: serve,
                TERMINAL_NAME: terminal,
                BOTH_NAME: both,
            }
        )

        self.assertEqual(selected, BOTH_NAME)
        self.assertTrue(detail["candidates"][BOTH_NAME]["gate"]["eligible"])

    def test_both_arm_cannot_hide_a_failed_independent_mechanism(self) -> None:
        control = report(aggregate(0.60))
        serve_fails = report(aggregate(0.62, start_mae=0.65, anchor_recall=0.5))
        terminal = report(aggregate(0.62, end_mae=1.44))
        both = report(
            aggregate(
                0.64, start_mae=0.65, end_mae=1.44, anchor_recall=0.55
            )
        )

        selected, detail = _select_inner(
            {
                CONTROL_NAME: control,
                SERVE_NAME: serve_fails,
                TERMINAL_NAME: terminal,
                BOTH_NAME: both,
            }
        )

        self.assertEqual(selected, TERMINAL_NAME)
        self.assertFalse(detail["candidates"][BOTH_NAME]["gate"]["eligible"])

    def test_serve_anchor_aggregation_uses_pooled_counts(self) -> None:
        result = _aggregate_serve_anchor(
            [
                {
                    "serveAnchorWithin05": {
                        "true": 2,
                        "predicted": 1,
                        "matched": 1,
                        "errorsSeconds": [0.25],
                    }
                },
                {
                    "serveAnchorWithin05": {
                        "true": 1,
                        "predicted": 2,
                        "matched": 1,
                        "errorsSeconds": [-0.5],
                    }
                },
            ]
        )
        self.assertEqual(result["matchedServes"], 2)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["timingMaeSeconds"], 0.375)

    def test_outer_aggregation_preserves_pooled_serve_anchor_metrics(self) -> None:
        row = aggregate(0.6)
        truth = [Interval(0.0, 2.0)]
        predictions = [Interval(0.0, 2.0)]
        recording = evaluate_intervals(truth, predictions)
        recording.update(
            {
                "id": "r1",
                "sourceGroup": "a",
                "environment": "grass",
                "outcomeSlices": outcome_slice_metrics(truth, predictions),
                "serveAnchorWithin05": {
                    "true": 1,
                    "predicted": 1,
                    "matched": 1,
                    "errorsSeconds": [0.1],
                },
            }
        )
        outer = [
            {
                "heldOutSourceGroup": "a",
                "candidate": {"aggregate": row, "recordings": [recording]},
            }
        ]

        result = _aggregate_outer_with_anchor(outer, "candidate")

        self.assertEqual(result["aggregate"]["serveAnchorWithin05"]["matchedServes"], 1)
        self.assertAlmostEqual(result["aggregate"]["serveAnchorWithin05"]["recall"], 1.0)

    def test_epoch_cap_and_control_reproduction_are_strict(self) -> None:
        self.assertEqual(_median_epoch([3, 7, 12]), 7)
        with self.assertRaises(FeatureExperimentError):
            _median_epoch([])
        expected = aggregate(0.6)
        _assert_control(expected, expected, "exact")
        changed = dict(expected)
        changed["liveTimeRecall"] = 0.8
        with self.assertRaisesRegex(FeatureExperimentError, "liveTimeRecall"):
            _assert_control(changed, expected, "changed")


if __name__ == "__main__":
    unittest.main()
