from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


PATH = Path(__file__).resolve().parents[2] / "scripts/trim-neural-auxiliary-head.py"
SPEC = importlib.util.spec_from_file_location("trim_auxiliary_tested", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def arrays_for(model):
    return {"mean": np.linspace(-1, 1, 104, dtype=np.float32), "scale": np.linspace(.1, 2, 104, dtype=np.float32),
            **{f"model::{name}": tensor.detach().numpy().copy() for name, tensor in model.state_dict().items()}}


class TrimAuxiliaryHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_linear_and_tcn_primary_logits_and_scaler_are_preserved(self):
        for kind in ("linear", "tcn"):
            with self.subTest(kind=kind):
                torch.manual_seed(47)
                original = module.ExpandedTemporalNetwork(module.ExpandedTemporalConfig(kind=kind)).eval()
                arrays = arrays_for(original)
                trimmed = module.trim_arrays(arrays, kind)
                derivative = module.CompactTemporalNetwork(module.CompactTemporalConfig(kind=kind)).eval()
                module.load_model(derivative, trimmed)
                head = "context_head" if kind == "linear" else "head"
                for name in arrays:
                    if name not in (f"model::{head}.weight", f"model::{head}.bias"):
                        np.testing.assert_array_equal(arrays[name], trimmed[name])
                generator = np.random.default_rng(47)
                for ticks in (1, 63, 252):
                    result = module.parity(original, derivative, generator.normal(size=(1, ticks, 104)).astype(np.float32), f"{kind}/{ticks}")
                    self.assertTrue(result["passed"])
                self.assertEqual(sum(p.numel() for p in derivative.parameters()), 29635 if kind == "tcn" else 1563)
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary)/"derivative.npz"
                    np.savez_compressed(path, **trimmed)
                    reloaded = module.strict_arrays(path, derivative)
                    np.testing.assert_array_equal(reloaded["mean"], arrays["mean"])
                    np.testing.assert_array_equal(reloaded["scale"], arrays["scale"])

    def test_strict_loader_rejects_schema_dtype_nonfinite_and_scaler_errors(self):
        model = module.ExpandedTemporalNetwork(module.ExpandedTemporalConfig(kind="linear"))
        valid = arrays_for(model)
        changes = [lambda a: a.update(unexpected=np.ones(1, dtype=np.float32)),
                   lambda a: a.update(mean=a["mean"].astype(np.float64)),
                   lambda a: a["scale"].__setitem__(0, 0),
                   lambda a: a["mean"].__setitem__(0, np.nan),
                   lambda a: a.update({"model::context_head.bias": np.ones(3, dtype=np.float32)})]
        for index, mutate in enumerate(changes):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                arrays = {key: value.copy() for key, value in valid.items()}
                mutate(arrays)
                path = Path(temporary)/"bad.npz"
                np.savez(path, **arrays)
                with self.assertRaises(ValueError):
                    module.strict_arrays(path, model)

    def test_existing_output_is_rejected_without_touching_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            marker = path/"keep.txt"
            marker.write_text("unchanged")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                module.convert(Path("missing-study"), Path("missing-checkpoint"), 5, "tcn", path)
            self.assertEqual(marker.read_text(), "unchanged")


if __name__ == "__main__":
    unittest.main()
