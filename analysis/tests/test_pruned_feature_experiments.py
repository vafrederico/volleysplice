from __future__ import annotations

import unittest

import numpy as np

from analysis.config import DecoderConfig, FeatureConfig
from analysis.model import LogisticModel
from analysis.pruned_feature_experiments import (
    ZERO_EMBEDDING_PROBABILITY_TOLERANCE,
    build_pruning_feature_sets,
    expand_model_to_full_signature,
    select_targeted_candidate,
)


class PruningVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.names = (
            "t+0s/luma_mean",
            "t+0s/luma_grid_0",
            "t+0s/luma_std",
            "t+0s/diff_mean",
            "t+0s/audio_spectral_flux",
            "t+0s/audio_rms_novelty",
            "t+0s/audio_onset_cadence",
            "t+0s/audio_seconds_since_transient",
        )

    def test_variants_apply_family_and_signal_exceptions_exactly(self) -> None:
        sets = {
            variant.name: feature_set
            for variant, feature_set in build_pruning_feature_sets(self.names)
        }

        self.assertEqual(
            sets["drop_audio_onset_cadence"].indexes, (0, 1, 2, 3, 4, 5, 7)
        )
        self.assertEqual(
            sets["audio_onset_rescue"].indexes, (0, 1, 2, 3, 5, 6, 7)
        )
        self.assertEqual(
            sets["legacy_appearance_rescue"].indexes, (1, 2, 3, 4, 5, 6, 7)
        )
        self.assertEqual(sets["targeted_pruned"].indexes, (1, 2, 3, 5, 7))

    def test_every_context_column_for_a_removed_base_signal_is_removed(self) -> None:
        names = self.names + ("t-1s/audio_onset_cadence",)
        sets = {
            variant.name: feature_set
            for variant, feature_set in build_pruning_feature_sets(names)
        }

        retained = {
            names[index] for index in sets["drop_audio_onset_cadence"].indexes
        }
        self.assertNotIn("t+0s/audio_onset_cadence", retained)
        self.assertNotIn("t-1s/audio_onset_cadence", retained)


class CandidateSelectionTests(unittest.TestCase):
    @staticmethod
    def candidate(objective: float, classification: str, count: int) -> dict:
        return {
            "variant": {"retainedFeatureCount": count},
            "oof": {
                "aggregate": {
                    "objective": objective,
                    "eventF1": 0.51,
                    "timeIoU": 0.51,
                    "liveTimeRecall": 0.51,
                }
            },
            "pairedAgainstFrozenFull": {
                "classification": {"classification": classification}
            },
        }

    def test_primary_candidate_can_replace_full(self) -> None:
        selection = select_targeted_candidate(
            {
                "objective": 0.50,
                "eventF1": 0.50,
                "timeIoU": 0.50,
                "liveTimeRecall": 0.50,
            },
            {
                "higher_but_uncertain": self.candidate(0.58, "uncertain", 100),
                "eligible_a": self.candidate(0.54, "helpful", 120),
                "targeted_pruned": self.candidate(0.55, "helpful", 130),
            },
        )

        self.assertEqual(selection["selectedCandidateForFinalTest"], "targeted_pruned")

    def test_retains_full_when_no_pruned_candidate_passes_rule(self) -> None:
        selection = select_targeted_candidate(
            {
                "objective": 0.50,
                "eventF1": 0.50,
                "timeIoU": 0.50,
                "liveTimeRecall": 0.50,
            },
            {
                "helpful_but_lower": self.candidate(0.49, "helpful", 100),
                "higher_but_uncertain": self.candidate(0.55, "uncertain", 90),
                "targeted_pruned": self.candidate(0.55, "uncertain", 80),
            },
        )

        self.assertEqual(selection["selectedCandidateForFinalTest"], "full")

    def test_diagnostic_candidate_cannot_replace_full(self) -> None:
        selection = select_targeted_candidate(
            {
                "objective": 0.50,
                "eventF1": 0.50,
                "timeIoU": 0.50,
                "liveTimeRecall": 0.50,
            },
            {
                "diagnostic": self.candidate(0.70, "helpful", 100),
                "targeted_pruned": self.candidate(0.55, "uncertain", 120),
            },
        )

        self.assertEqual(selection["selectedCandidateForFinalTest"], "full")

    def test_primary_candidate_must_pass_all_metric_guardrails(self) -> None:
        primary = self.candidate(0.55, "helpful", 120)
        primary["oof"]["aggregate"]["eventF1"] = 0.48
        selection = select_targeted_candidate(
            {
                "objective": 0.50,
                "eventF1": 0.50,
                "timeIoU": 0.50,
                "liveTimeRecall": 0.50,
            },
            {"targeted_pruned": primary},
        )

        self.assertEqual(selection["selectedCandidateForFinalTest"], "full")
        row = selection["candidates"][0]
        self.assertFalse(row["aggregateMetricGuardrails"]["eventF1"]["passes"])


class ZeroEmbeddingTests(unittest.TestCase):
    def test_expanded_model_has_prediction_parity_with_subset_model(self) -> None:
        subset = LogisticModel(
            feature_config=FeatureConfig(),
            feature_names=("feature_b", "feature_d"),
            mean=np.asarray([0.2, -0.3], dtype=np.float32),
            scale=np.asarray([0.5, 1.7], dtype=np.float32),
            weights=np.asarray([1.2, -0.8], dtype=np.float32),
            bias=0.15,
            decoder=DecoderConfig(),
            training_summary={},
        )
        full_names = ("feature_a", "feature_b", "feature_c", "feature_d")
        full_values = np.asarray(
            [[9.0, 0.1, -5.0, 1.0], [-7.0, 0.8, 200.0, -0.2]],
            dtype=np.float32,
        )

        expanded = expand_model_to_full_signature(subset, full_names)

        expected = subset.predict(full_values[:, (1, 3)])
        actual = expanded.predict(full_values)
        np.testing.assert_array_equal(actual, expected)
        removed = np.asarray((0, 2), dtype=np.int64)
        np.testing.assert_array_equal(expanded.weights[removed], np.zeros(2))
        np.testing.assert_array_equal(expanded.mean[removed], np.zeros(2))
        np.testing.assert_array_equal(expanded.scale[removed], np.ones(2))

    def test_realistic_wide_embedding_is_numerically_close(self) -> None:
        rng = np.random.default_rng(17)
        full_names = tuple(f"feature_{index}" for index in range(450))
        retained_indexes = np.arange(85, 450, dtype=np.int64)
        subset = LogisticModel(
            feature_config=FeatureConfig(),
            feature_names=tuple(full_names[index] for index in retained_indexes),
            mean=rng.normal(0.0, 0.2, 365).astype(np.float32),
            scale=rng.uniform(0.5, 2.0, 365).astype(np.float32),
            weights=rng.normal(0.0, 0.1, 365).astype(np.float32),
            bias=0.15,
            decoder=DecoderConfig(),
            training_summary={},
        )
        full_values = rng.normal(0.0, 1.0, (10_000, 450)).astype(np.float32)

        expanded = expand_model_to_full_signature(subset, full_names)
        expected = subset.predict(full_values[:, retained_indexes])
        actual = expanded.predict(full_values)

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=ZERO_EMBEDDING_PROBABILITY_TOLERANCE,
        )


if __name__ == "__main__":
    unittest.main()
