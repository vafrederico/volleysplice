from __future__ import annotations

import unittest
from unittest.mock import patch

from analysis.cli import build_parser
from analysis.decoder import DecodedInterval
from analysis.dual_serve_fusion_experiment import (
    BoundarySelectorConfig,
    EXPERIMENT_ID,
    _add_only_audit,
    _check_evaluation_groups,
    _clip_decoded,
    _validate_decision,
    add_only_candidates,
    apply_boundary_selector,
    evaluate_dual_serve_fusion_dataset,
    overlap_components,
)
from analysis.model import ModelError
from analysis.schema import Interval


def interval(start: float, end: float, confidence: float = 0.9) -> DecodedInterval:
    return DecodedInterval(start, end, confidence)


class DualServeFusionTests(unittest.TestCase):
    def test_touching_intervals_are_separate_components(self) -> None:
        components = overlap_components(
            [interval(0.0, 2.0), interval(2.0, 4.0)],
            [interval(1.0, 2.0), interval(2.0, 3.0)],
        )
        self.assertEqual(
            [(len(row.v4), len(row.v5)) for row in components], [(1, 1), (1, 1)]
        )

    def test_transitive_overlap_is_an_ambiguous_component(self) -> None:
        components = overlap_components(
            [interval(0.0, 3.0)],
            [interval(0.0, 1.0), interval(1.0, 2.0), interval(2.0, 3.0)],
        )
        self.assertEqual(len(components), 1)
        self.assertEqual((len(components[0].v4), len(components[0].v5)), (1, 3))

    def test_intersection_refines_eligible_one_to_one_pair(self) -> None:
        selected, changed = apply_boundary_selector(
            [interval(10.0, 15.0, 0.85)],
            [interval(10.5, 13.0, 0.95)],
            BoundarySelectorConfig(
                action="intersection",
                minimum_pair_iou=0.3,
                maximum_start_delta_seconds=1.0,
                minimum_v5_end_earlier_seconds=1.0,
                minimum_both_confidence=0.8,
            ),
        )
        self.assertEqual(changed, 1)
        self.assertEqual((selected[0].start, selected[0].end), (10.5, 13.0))

    def test_selector_retains_v4_when_any_gate_fails(self) -> None:
        original = interval(10.0, 15.0, 0.79)
        selected, changed = apply_boundary_selector(
            [original],
            [interval(10.5, 13.0, 0.95)],
            BoundarySelectorConfig(
                action="intersection",
                minimum_pair_iou=0.3,
                maximum_start_delta_seconds=1.0,
                minimum_v5_end_earlier_seconds=1.0,
                minimum_both_confidence=0.8,
            ),
        )
        self.assertEqual(changed, 0)
        self.assertEqual(selected, [original])

    def test_selector_never_changes_ambiguous_split(self) -> None:
        original = interval(10.0, 15.0)
        selected, changed = apply_boundary_selector(
            [original],
            [interval(10.0, 12.0), interval(12.0, 14.0)],
            BoundarySelectorConfig(action="intersection"),
        )
        self.assertEqual(changed, 0)
        self.assertEqual(selected, [original])

    def test_add_only_requires_zero_positive_overlap(self) -> None:
        candidates = add_only_candidates(
            [interval(0.0, 2.0)],
            [interval(1.0, 3.0), interval(2.0, 4.0), interval(5.0, 6.0)],
        )
        self.assertEqual(
            [(row.start, row.end) for row in candidates], [(2.0, 4.0), (5.0, 6.0)]
        )

    def test_config_round_trip_and_unknown_action_rejection(self) -> None:
        config = BoundarySelectorConfig(
            action="v5",
            minimum_pair_iou=0.5,
            maximum_start_delta_seconds=1.0,
            minimum_v5_end_earlier_seconds=0.5,
            minimum_both_confidence=0.9,
        )
        self.assertEqual(BoundarySelectorConfig.from_dict(config.to_dict()), config)
        with self.assertRaisesRegex(ValueError, "unsupported boundary action"):
            BoundarySelectorConfig(action="union").validate()

    def test_ignored_region_is_removed_before_fusion(self) -> None:
        clipped = _clip_decoded(
            [interval(0.0, 10.0, 0.7)], [Interval(3.0, 7.0)]
        )
        self.assertEqual(
            [(row.start, row.end, row.confidence) for row in clipped],
            [(0.0, 3.0, 0.7), (7.0, 10.0, 0.7)],
        )

    def test_cli_accepts_dual_fusion_decision(self) -> None:
        parsed = build_parser().parse_args(
            [
                "evaluate-dual-serve-fusion",
                "--manifest",
                "manifest.json",
                "--v4-rally-model",
                "v4-rally",
                "--v4-serve-model",
                "v4-serve",
                "--v4-cache-dir",
                "v4-cache",
                "--v5-rally-model",
                "v5-rally",
                "--v5-serve-model",
                "v5-serve",
                "--v5-cache-dir",
                "v5-cache",
                "--split",
                "test",
                "--decision",
                "validation.json",
                "--retrospective",
            ]
        )
        self.assertEqual(parsed.command, "evaluate-dual-serve-fusion")
        self.assertTrue(parsed.retrospective)

    def test_non_validation_requires_frozen_decision_before_loading_manifest(self) -> None:
        with patch(
            "analysis.dual_serve_fusion_experiment.load_manifest"
        ) as load_manifest:
            with self.assertRaisesRegex(
                ModelError, "requires a frozen validation decision"
            ):
                evaluate_dual_serve_fusion_dataset(
                    "manifest.json",
                    "v4-rally",
                    "v4-serve",
                    "v4-cache",
                    "v5-rally",
                    "v5-serve",
                    "v5-cache",
                    split="test",
                    retrospective=True,
                )
        load_manifest.assert_not_called()

    def test_decision_binds_manifest_and_models_and_replays_config(self) -> None:
        models = {"v4Rally": {"sha256": "a"}, "v5Rally": {"sha256": "b"}}
        config = BoundarySelectorConfig(
            action="intersection",
            minimum_pair_iou=0.3,
            maximum_start_delta_seconds=1.0,
            minimum_v5_end_earlier_seconds=1.0,
            minimum_both_confidence=0.8,
        )
        decision = {
            "experiment": EXPERIMENT_ID,
            "split": "validation",
            "manifestSha256": "manifest",
            "models": models,
            "selection": {
                "selectedOn": "validation",
                "addOnly": {"selectedPolicy": "disabled"},
                "boundarySelector": {"selectedConfig": config.to_dict()},
            },
        }
        self.assertEqual(_validate_decision(decision, "manifest", models), config)
        with self.assertRaisesRegex(ModelError, "immutable manifest"):
            _validate_decision(decision, "other", models)
        with self.assertRaisesRegex(ModelError, "different model artifacts"):
            _validate_decision(decision, "manifest", {"changed": True})

    def test_group_overlap_rejects_leaking_evaluation(self) -> None:
        recording = unittest.mock.Mock(source_group="held-out")
        leaking = unittest.mock.Mock(
            training_summary={
                "trainingSourceGroups": ["train"],
                "validationSourceGroups": ["held-out"],
            }
        )
        with self.assertRaisesRegex(ModelError, "leaks trained/tuned source groups"):
            _check_evaluation_groups("test", [recording], [leaking])
        _check_evaluation_groups("validation", [recording], [leaking])

    def test_test_audit_does_not_claim_validation_selection(self) -> None:
        with patch(
            "analysis.dual_serve_fusion_experiment._report",
            return_value={
                "aggregate": {
                    "predictedRallies": 0,
                    "matchedRallies": 0,
                    "eventPrecision": 1.0,
                    "eventRecall": 1.0,
                    "eventF1": 1.0,
                }
            },
        ):
            result = _add_only_audit([], select_policy=False)
        self.assertEqual(result["selectedPolicy"], "audit-only")
        self.assertIn("frozen", result["selectionReason"])


if __name__ == "__main__":
    unittest.main()
