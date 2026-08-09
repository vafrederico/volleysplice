from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.config import DecoderConfig, FeatureConfig, TrainingConfig
from analysis.model import ModelError, load_model, train_logistic_model


class LogisticModelTests(unittest.TestCase):
    @staticmethod
    def training_inputs() -> tuple[
        list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]
    ]:
        train_values = [
            np.asarray(
                [[-0.2, 0.1], [0.0, -0.1], [0.2, 0.1], [0.8, 0.9], [1.0, 1.2], [1.2, 0.8]],
                dtype=np.float32,
            ),
            np.asarray(
                [[-0.1, -0.2], [0.1, 0.2], [0.3, 0.0], [0.7, 1.1], [0.9, 0.8], [1.3, 1.0]],
                dtype=np.float32,
            ),
        ]
        train_labels = [
            np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float32),
            np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float32),
        ]
        validation_values = [
            np.asarray(
                [[-0.15, 0.0], [0.15, 0.05], [0.75, 0.85], [1.1, 1.05]],
                dtype=np.float32,
            )
        ]
        validation_labels = [np.asarray([0, 0, 1, 1], dtype=np.float32)]
        return train_values, train_labels, validation_values, validation_labels

    def train(self):
        train_values, train_labels, validation_values, validation_labels = self.training_inputs()
        return train_logistic_model(
            train_values,
            train_labels,
            validation_values,
            validation_labels,
            FeatureConfig(use_optical_flow=False, context_offsets_seconds=(0.0,)),
            ("motion", "posture"),
            DecoderConfig(smoothing_seconds=0.5, min_live_seconds=0.5),
            TrainingConfig(
                epochs=35,
                batch_size=3,
                learning_rate=0.03,
                l2=1e-4,
                patience=35,
                seed=23,
            ),
        )

    def test_training_is_deterministic_for_a_fixed_seed(self) -> None:
        first = self.train()
        second = self.train()

        np.testing.assert_array_equal(first.mean, second.mean)
        np.testing.assert_array_equal(first.scale, second.scale)
        np.testing.assert_array_equal(first.weights, second.weights)
        self.assertEqual(first.bias, second.bias)
        self.assertEqual(first.training_summary, second.training_summary)

    def test_save_load_preserves_predictions_and_configuration(self) -> None:
        model = self.train()
        probe = np.asarray(
            [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
            dtype=np.float32,
        )
        expected = model.predict(probe)

        with tempfile.TemporaryDirectory(prefix="volleycut-model-test-") as directory:
            model_path = model.save(Path(directory) / "model")
            restored = load_model(model_path)

        np.testing.assert_allclose(restored.predict(probe), expected, rtol=0.0, atol=1e-7)
        np.testing.assert_array_equal(restored.mean, model.mean)
        np.testing.assert_array_equal(restored.scale, model.scale)
        np.testing.assert_array_equal(restored.weights, model.weights)
        self.assertAlmostEqual(restored.bias, model.bias, places=7)
        self.assertEqual(restored.feature_names, model.feature_names)
        self.assertEqual(restored.feature_config, model.feature_config)
        self.assertEqual(restored.decoder, model.decoder)
        self.assertEqual(restored.training_summary, model.training_summary)
        self.assertEqual(restored.artifact_sha256, model.artifact_sha256)

    def test_load_rejects_non_object_metadata_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-model-root-test-") as directory:
            metadata_path = Path(directory) / "model.json"
            metadata_path.write_text("[]\n", encoding="utf-8")

            with self.assertRaisesRegex(ModelError, "metadata root must be an object"):
                load_model(metadata_path)

    def test_load_rejects_invalid_metadata_types(self) -> None:
        mutations = {
            "unsupported model type": lambda metadata: metadata.update({"modelType": 7}),
            "feature names are not an array": lambda metadata: metadata.update(
                {"featureNames": "motion"}
            ),
            "training metadata is not an object": lambda metadata: metadata.update(
                {"training": []}
            ),
        }
        for description, mutate in mutations.items():
            with self.subTest(description=description):
                with tempfile.TemporaryDirectory(prefix="volleycut-model-type-test-") as directory:
                    model_path = self.train().save(Path(directory) / "model")
                    metadata_path = model_path / "model.json"
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    mutate(metadata)
                    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

                    with self.assertRaises(ModelError):
                        load_model(model_path)

    def test_load_rejects_weights_with_a_valid_digest_but_wrong_shape(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-model-shape-test-") as directory:
            model_path = self.train().save(Path(directory) / "model")
            metadata_path = model_path / "model.json"
            weights_path = model_path / "weights.npz"
            with np.load(weights_path, allow_pickle=False) as artifact:
                mean = artifact["mean"].copy()
                scale = artifact["scale"].copy()
                weights = artifact["weights"].copy()
                bias = artifact["bias"].copy()
            np.savez_compressed(
                weights_path,
                mean=np.append(mean, np.float32(0.0)),
                scale=scale,
                weights=weights,
                bias=bias,
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["weightsSha256"] = hashlib.sha256(weights_path.read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ModelError, "dimensions do not agree"):
                load_model(model_path)

    def test_load_rejects_weights_that_do_not_match_the_recorded_digest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-model-digest-test-") as directory:
            model_path = self.train().save(Path(directory) / "model")
            weights_path = model_path / "weights.npz"
            weights_path.write_bytes(weights_path.read_bytes() + b"tampered")

            with self.assertRaisesRegex(ModelError, "weights digest.*does not match"):
                load_model(model_path)


if __name__ == "__main__":
    unittest.main()
