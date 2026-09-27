from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "onnx", "onnxruntime"))


@unittest.skipUnless(AVAILABLE, "PyTorch and ONNX qualification dependencies are not installed")
class ExportNeuralCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        path = Path(__file__).resolve().parents[2] / "scripts/export-neural-checkpoint.py"
        spec = importlib.util.spec_from_file_location("export_neural_checkpoint_script", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        import torch

        torch.set_num_threads(cls.previous_threads)

    def checkpoint(self, directory: Path) -> None:
        module = self.module
        model = module.model_for("mlp").eval()
        state = {f"model::{key}": value.detach().numpy() for key, value in model.state_dict().items()}
        np.savez_compressed(directory / "weights-5.npz", mean=np.zeros(104, dtype=np.float32),
                            scale=np.ones(104, dtype=np.float32), **state)
        np.savez_compressed(directory / "predictions-5.npz", held=np.zeros((4, 3), dtype=np.float32))
        metadata = {"kind": "mlp", "contractSha256": "contract", "epochs": [5],
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "artifacts": {name: module.file_sha256(directory / name)
                                  for name in ("weights-5.npz", "predictions-5.npz")}}
        (directory / "completed.json").write_text(json.dumps(metadata))

    def test_loader_rejects_tampered_prediction_artifact_before_using_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.checkpoint(directory)
            model, mean, scale, _ = self.module.load_checkpoint(directory, 5, "mlp", "contract")
            self.assertEqual(model.config.kind, "mlp")
            self.assertEqual(mean.shape, (104,))
            np.testing.assert_array_equal(scale, np.ones(104))
            with (directory / "predictions-5.npz").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.module.load_checkpoint(directory, 5, "mlp", "contract")

    def test_loader_rejects_unverified_or_wrong_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.checkpoint(directory)
            for kind, contract in (("tcn", "contract"), ("mlp", "different"), ("dino_tcn", "contract")):
                with self.subTest(kind=kind, contract=contract), self.assertRaises(ValueError):
                    self.module.load_checkpoint(directory, 5, kind, contract)

    def test_dynamic_export_accepts_short_real_edges_and_fused_scaling(self) -> None:
        import torch
        import onnxruntime as ort

        module = self.module
        torch.manual_seed(47)
        model = module.model_for("tcn").eval()
        mean = np.linspace(-1, 1, 104, dtype=np.float32)
        scale = np.linspace(0.1, 1.0, 104, dtype=np.float32)
        wrapper = module.ScaledProbabilityModel(model, mean, scale).eval()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dynamic.onnx"
            metadata = module.export_graph(wrapper, path)
            self.assertEqual(metadata["inputShape"], [1, "ticks", 104])
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            generator = np.random.default_rng(47)
            for ticks in (1, 7, 63, 252, 411):
                with self.subTest(ticks=ticks), torch.inference_mode():
                    values = generator.normal(0, 4, (1, ticks, 104)).astype(np.float32)
                    manual = torch.sigmoid(model(torch.from_numpy(np.clip((values-mean)/scale, -10, 10)))).numpy()
                    expected = wrapper(torch.from_numpy(values)).numpy()
                    actual = session.run(["probabilities"], {"features": values})[0]
                    module.compare(expected, manual, "fused scaler vs manual")
                    module.compare(actual, expected, "dynamic short/long graph")


if __name__ == "__main__":
    unittest.main()
