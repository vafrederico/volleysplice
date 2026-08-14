from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch


class InferDualServeFusionScriptTests(unittest.TestCase):
    def test_allow_manifest_mismatch_is_propagated_to_prediction_api(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "infer-dual-serve-fusion-dataset.py"
        )
        spec = importlib.util.spec_from_file_location(
            "infer_dual_serve_fusion_dataset_script", script_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        inference = Mock(return_value={"created": []})
        module.infer_dual_serve_fusion_dataset = inference

        with patch.object(
            sys,
            "argv",
            [script_path.name, "--allow-manifest-mismatch", "--limit", "1"],
        ), redirect_stdout(io.StringIO()):
            self.assertEqual(module.main(), 0)

        self.assertTrue(inference.call_args.kwargs["allow_manifest_mismatch"])
        self.assertEqual(inference.call_args.kwargs["limit"], 1)


if __name__ == "__main__":
    unittest.main()
