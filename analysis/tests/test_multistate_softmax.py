from __future__ import annotations

import unittest

import numpy as np

from analysis.config import FeatureConfig, TrainingConfig
from analysis.model import ModelError
from analysis.multistate import STATE_ORDER
from analysis.multistate_softmax import (
    MultistateSoftmaxModel,
    multistate_softmax_fingerprint,
    train_multistate_softmax,
)


FEATURE_NAMES = ("x", "y")
FEATURE_CONFIG = FeatureConfig()


def separated_rows() -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        [
            [-2.0, -2.0],
            [-2.2, -1.8],
            [-1.8, -2.2],
            [-2.0, 2.0],
            [-2.2, 1.8],
            [-1.8, 2.2],
            [2.0, -2.0],
            [2.2, -1.8],
            [1.8, -2.2],
            [2.0, 2.0],
            [2.2, 1.8],
            [1.8, 2.2],
        ],
        dtype=np.float32,
    )
    targets = np.repeat(np.arange(len(STATE_ORDER), dtype=np.int64), 3)
    return values, targets


def training_config(*, seed: int = 13) -> TrainingConfig:
    return TrainingConfig(
        epochs=120,
        batch_size=12,
        learning_rate=0.05,
        l2=1e-4,
        patience=30,
        seed=seed,
    )


class MultistateSoftmaxTests(unittest.TestCase):
    def test_joint_model_learns_four_states_and_normalizes_probabilities(self) -> None:
        values, targets = separated_rows()
        model = train_multistate_softmax(
            [values],
            [targets],
            [values],
            [targets],
            FEATURE_CONFIG,
            FEATURE_NAMES,
            training_config(),
        )

        probabilities = model.predict_probabilities(values)
        log_scores = model.predict_log_scores(values)
        self.assertEqual(probabilities.shape, (len(values), len(STATE_ORDER)))
        np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(log_scores, np.log(probabilities), atol=1e-12)
        self.assertGreaterEqual(
            float(np.mean(np.argmax(probabilities, axis=1) == targets)), 0.99
        )

    def test_training_is_deterministic_for_the_same_seed(self) -> None:
        values, targets = separated_rows()
        arguments = (
            [values],
            [targets],
            [values],
            [targets],
            FEATURE_CONFIG,
            FEATURE_NAMES,
            training_config(),
        )
        first = train_multistate_softmax(*arguments)
        second = train_multistate_softmax(*arguments)

        np.testing.assert_array_equal(first.mean, second.mean)
        np.testing.assert_array_equal(first.scale, second.scale)
        np.testing.assert_array_equal(first.weights, second.weights)
        np.testing.assert_array_equal(first.bias, second.bias)
        self.assertEqual(
            multistate_softmax_fingerprint(first),
            multistate_softmax_fingerprint(second),
        )

    def test_normalization_and_priors_use_training_rows_only(self) -> None:
        values, targets = separated_rows()
        validation = np.full((4, 2), 10_000.0, dtype=np.float32)
        validation_targets = np.arange(4, dtype=np.int64)
        model = train_multistate_softmax(
            [values],
            [targets],
            [validation],
            [validation_targets],
            FEATURE_CONFIG,
            FEATURE_NAMES,
            TrainingConfig(
                epochs=1,
                batch_size=12,
                learning_rate=0.01,
                l2=0.0,
                patience=1,
                seed=1,
            ),
        )

        np.testing.assert_allclose(model.mean, np.mean(values, axis=0))
        self.assertEqual(
            model.training_summary["classSamples"],
            {state.name: 3 for state in STATE_ORDER},
        )
        self.assertEqual(model.training_summary["classWeighting"], "empirical")
        self.assertEqual(
            model.training_summary["classWeights"],
            {state.name: 1.0 for state in STATE_ORDER},
        )
        self.assertFalse(model.training_summary["priorCorrectionAvailable"])
        self.assertEqual(model.training_summary["validationSamples"], 4)

    def test_prior_correction_restores_fold_empirical_prevalence(self) -> None:
        counts = (1, 2, 3, 4)
        model = MultistateSoftmaxModel(
            feature_config=FEATURE_CONFIG,
            feature_names=FEATURE_NAMES,
            mean=np.zeros(2, dtype=np.float32),
            scale=np.ones(2, dtype=np.float32),
            weights=np.zeros((2, len(STATE_ORDER)), dtype=np.float32),
            bias=np.zeros(len(STATE_ORDER), dtype=np.float32),
            training_summary={
                "classWeighting": "balanced",
                "classSamples": {
                    state.name: count
                    for state, count in zip(STATE_ORDER, counts, strict=True)
                }
            },
        )
        values = np.zeros((2, 2), dtype=np.float32)

        np.testing.assert_allclose(
            model.predict_probabilities(values),
            np.full((2, 4), 0.25),
        )
        np.testing.assert_allclose(
            model.predict_probabilities(values, prior_corrected=True),
            np.tile(np.asarray(counts, dtype=np.float64) / sum(counts), (2, 1)),
        )

        changed_prior = MultistateSoftmaxModel(
            feature_config=model.feature_config,
            feature_names=model.feature_names,
            mean=model.mean,
            scale=model.scale,
            weights=model.weights,
            bias=model.bias,
            training_summary={
                "classWeighting": "balanced",
                "classSamples": {state.name: 1 for state in STATE_ORDER}
            },
        )
        self.assertNotEqual(
            multistate_softmax_fingerprint(model),
            multistate_softmax_fingerprint(changed_prior),
        )

    def test_empirical_model_rejects_prior_correction(self) -> None:
        values, targets = separated_rows()
        model = train_multistate_softmax(
            [values],
            [targets],
            [],
            [],
            FEATURE_CONFIG,
            FEATURE_NAMES,
            TrainingConfig(epochs=1, patience=1),
        )
        with self.assertRaisesRegex(ModelError, "only valid for balanced"):
            model.predict_probabilities(values, prior_corrected=True)

    def test_balanced_mode_uses_inverse_frequency_class_weights(self) -> None:
        values, targets = separated_rows()
        indexes = np.asarray([0, 3, 4, 6, 7, 8, 9, 10, 11], dtype=np.int64)
        selected_values = values[indexes]
        selected_targets = targets[indexes]
        model = train_multistate_softmax(
            [selected_values],
            [selected_targets],
            [],
            [],
            FEATURE_CONFIG,
            FEATURE_NAMES,
            TrainingConfig(epochs=1, batch_size=9, patience=1),
            class_weighting="balanced",
        )

        self.assertEqual(model.training_summary["classWeighting"], "balanced")
        self.assertTrue(model.training_summary["priorCorrectionAvailable"])
        expected_counts = np.asarray([1.0, 2.0, 3.0, 3.0])
        expected_weights = len(indexes) / (len(STATE_ORDER) * expected_counts)
        np.testing.assert_allclose(
            [
                model.training_summary["classWeights"][state.name]
                for state in STATE_ORDER
            ],
            expected_weights,
        )
        probabilities = model.predict_probabilities(
            selected_values, prior_corrected=True
        )
        np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)

    def test_unknown_class_weighting_is_rejected(self) -> None:
        values, targets = separated_rows()
        with self.assertRaisesRegex(ValueError, "class_weighting"):
            train_multistate_softmax(
                [values],
                [targets],
                [],
                [],
                FEATURE_CONFIG,
                FEATURE_NAMES,
                training_config(),
                class_weighting="automatic",
            )

    def test_training_rejects_a_missing_state_class(self) -> None:
        values, targets = separated_rows()
        selected = targets != len(STATE_ORDER) - 1
        with self.assertRaisesRegex(ModelError, "every state class"):
            train_multistate_softmax(
                [values[selected]],
                [targets[selected]],
                [],
                [],
                FEATURE_CONFIG,
                FEATURE_NAMES,
                training_config(),
            )

    def test_prediction_rejects_wrong_or_nonfinite_feature_matrices(self) -> None:
        values, targets = separated_rows()
        model = train_multistate_softmax(
            [values],
            [targets],
            [],
            [],
            FEATURE_CONFIG,
            FEATURE_NAMES,
            TrainingConfig(epochs=1, patience=1),
        )
        for invalid in (
            np.zeros((2, 3), dtype=np.float32),
            np.asarray([[0.0, np.nan]], dtype=np.float32),
        ):
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(ModelError):
                    model.predict_probabilities(invalid)


if __name__ == "__main__":
    unittest.main()
