from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from analysis.compact_temporal_model import CompactTemporalError
from analysis.local_attention_temporal_model import (
    DINO_INPUT_DIMENSION, HEAD_NAMES, MODEL_KINDS, PARAMETER_COUNTS,
    DinoLocalAttentionNetwork, LocalAttentionTemporalConfig,
    model_for, model_metadata, trainable_parameter_count,
)


class LocalAttentionMetadataTests(unittest.TestCase):
    def test_metadata_is_explicit_about_context_mask_and_information_budget(self):
        self.assertEqual(HEAD_NAMES, ("live", "serve", "end", "keep"))
        for kind in MODEL_KINDS:
            metadata = model_metadata(kind)
            self.assertEqual(metadata["haloTicks"], 62)
            self.assertEqual(metadata["receptiveFieldTicks"], 125)
            self.assertEqual(metadata["attentionRadiusTicks"], 31)
            self.assertEqual(metadata["parameters"], PARAMETER_COUNTS[kind])
            self.assertFalse(metadata["featureExtractionIncluded"])
            self.assertIn("context, not supervision", metadata["segmentBoundaries"])
        dino = model_metadata("dino_transformer")
        self.assertEqual(dino["inputDimension"], 3944)
        self.assertEqual(dino["fusedDimension"], 264)
        self.assertEqual(dino["tokenProjectionDimension"], 16)
        for bad in ("tcn", "dino_tcn", "mlp"):
            with self.assertRaises(CompactTemporalError):
                model_metadata(bad)
            with self.assertRaises(CompactTemporalError):
                model_for(bad)

    def test_invalid_architecture_configuration_is_rejected(self):
        for kwargs in (
            {"kind": "tcn"}, {"input_dimension": 0}, {"hidden_dimension": True},
            {"hidden_dimension": 41}, {"attention_heads": 0}, {"block_count": -1},
            {"feedforward_dimension": 0}, {"attention_radius_ticks": 0},
            {"dropout": float("nan")}, {"dropout": 1}, {"dropout": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(CompactTemporalError):
                LocalAttentionTemporalConfig(**kwargs).validate()


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class LocalAttentionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.previous_threads)

    def test_parameter_counts_seed_repeatability_float32_and_gradients(self):
        import torch
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            for kind in MODEL_KINDS:
                torch.manual_seed(113)
                model = model_for(kind)
                torch.manual_seed(113)
                twin = model_for(kind)
                self.assertEqual(trainable_parameter_count(model), PARAMETER_COUNTS[kind])
                self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, twin.state_dict()[name], rtol=0, atol=0)
                values = torch.randn(2, 19, model.config.input_dimension, dtype=torch.float32, requires_grad=True)
                logits = model(values)
                self.assertEqual(tuple(logits.shape), (2, 19, 4))
                logits.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertGreater(torch.count_nonzero(values.grad[..., :104]).item(), 0)
                temporal = model.temporal if kind == "dino_transformer" else model
                self.assertTrue((temporal.head.weight.grad.abs().sum(dim=1) > 0).all())
                if kind == "dino_transformer":
                    self.assertGreater(torch.count_nonzero(values.grad[..., 104:]).item(), 0)
                    self.assertGreater(torch.count_nonzero(model.token_projection.weight.grad).item(), 0)
        finally:
            torch.set_default_dtype(original_dtype)

    def test_real_halo_chunks_match_whole_sequences_and_true_edges(self):
        import torch
        for kind in MODEL_KINDS:
            torch.manual_seed(127)
            model = model_for(kind).eval()
            for length in (1, 7, 63, 257, 411):
                with self.subTest(kind=kind, length=length), torch.no_grad():
                    values = torch.randn(1, length, model.config.input_dimension)
                    whole = model(values)
                    pieces = []
                    for start in range(0, length, 73):
                        stop = min(start + 73, length)
                        left, right = max(0, start - 62), min(length, stop + 62)
                        chunk = model(values[:, left:right])
                        pieces.append(chunk[:, start-left:stop-left])
                    torch.testing.assert_close(torch.cat(pieces, dim=1), whole, rtol=1e-5, atol=1e-6)

    def test_masked_padding_is_equivalent_to_short_segments_and_blocks_internal_gaps(self):
        import torch
        for kind in MODEL_KINDS:
            torch.manual_seed(131)
            model = model_for(kind).eval()
            values = torch.randn(2, 151, model.config.input_dimension)
            valid = torch.zeros(2, 151, dtype=torch.bool)
            valid[0, 13:34] = True
            valid[0, 35:130] = True
            # The other batch item is entirely invalid; no softmax row may NaN.
            values[~valid] = float("nan")
            with torch.no_grad():
                output = model(values, valid)
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(torch.count_nonzero(output[~valid]).item(), 0)
                for left, right in ((13, 34), (35, 130)):
                    expected = model(values[:1, left:right])
                    torch.testing.assert_close(output[:1, left:right], expected, rtol=1e-5, atol=1e-6)
                changed = values.clone()
                changed[0, 35:130] += 100
                torch.testing.assert_close(model(changed, valid)[0, 13:34], output[0, 13:34], rtol=0, atol=0)
            model.zero_grad(set_to_none=True)
            values.requires_grad_(True)
            model(values, valid)[valid].square().mean().backward()
            self.assertTrue(torch.isfinite(values.grad).all())
            self.assertEqual(torch.count_nonzero(values.grad[~valid]).item(), 0)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_exact_receptive_field_has_no_outside_dependency_and_reaches_both_edges(self):
        import torch
        torch.manual_seed(137)
        model = model_for("transformer").eval()
        values = torch.randn(1, 151, 104, requires_grad=True)
        center = 75
        model(values)[0, center, 0].backward()
        per_tick = values.grad.abs().sum(dim=-1)[0]
        self.assertEqual(torch.count_nonzero(per_tick[:center-62]).item(), 0)
        self.assertEqual(torch.count_nonzero(per_tick[center+63:]).item(), 0)
        self.assertGreater(per_tick[center-62].item(), 0)
        self.assertGreater(per_tick[center+62].item(), 0)

    def test_dino_preprocessing_matches_existing_fusion_exactly(self):
        import torch
        from analysis.transfer_temporal_model import model_for as tcn_model_for
        torch.manual_seed(139)
        tcn = tcn_model_for("dino_tcn").eval()
        torch.manual_seed(139)
        attention = model_for("dino_transformer").eval()
        for key in ("token_norm.weight", "token_norm.bias", "token_projection.weight", "token_projection.bias"):
            torch.testing.assert_close(tcn.state_dict()[key], attention.state_dict()[key], rtol=0, atol=0)
        values = torch.randn(1, 7, DINO_INPUT_DIMENSION)
        captured = []
        handles = [model.temporal.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
                   for model in (tcn, attention)]
        try:
            with torch.no_grad():
                tcn(values)
                attention(values)
            torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
        finally:
            for handle in handles:
                handle.remove()

    def test_input_and_context_mask_validation(self):
        import torch
        model = model_for("transformer")
        for values in (torch.zeros(4, 104), torch.zeros(1, 0, 104),
                       torch.zeros(1, 4, 90), torch.zeros(1, 4, 104, dtype=torch.float64)):
            with self.assertRaises(CompactTemporalError):
                model(values)
        values = torch.zeros(2, 4, 104)
        for mask in (torch.ones(2, 4), torch.ones(1, 4, dtype=torch.bool),
                     torch.ones(2, 4, 1, dtype=torch.bool), [[True] * 4] * 2):
            with self.assertRaises(CompactTemporalError):
                model(values, mask)
        with self.assertRaises(CompactTemporalError):
            DinoLocalAttentionNetwork(LocalAttentionTemporalConfig())

    @unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in ("onnx", "onnxruntime")),
                         "ONNX qualification dependencies are not installed")
    def test_static_onnx_export_uses_decomposed_attention_and_masked_cpu_parity(self):
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        for kind in MODEL_KINDS:
            torch.manual_seed(149)
            model = model_for(kind).eval()
            values = torch.randn(1, 256, model.config.input_dimension)
            valid = torch.ones(1, 256, dtype=torch.bool)
            valid[:, :11] = False
            valid[:, 73:75] = False
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "attention.onnx"
                torch.onnx.export(model, (values, valid), str(path), input_names=["features", "valid_mask"],
                                  output_names=["logits"], opset_version=17, dynamo=False)
                graph = onnx.load(str(path))
                onnx.checker.check_model(graph)
                operations = {node.op_type for node in graph.graph.node}
                self.assertTrue({"MatMul", "Softmax", "Where"} <= operations)
                self.assertFalse(any("Attention" in operation for operation in operations))
                session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
                for case in (valid, torch.zeros_like(valid), ~valid):
                    modified = values.clone()
                    modified[~case] = float("nan")
                    with torch.no_grad():
                        expected = model(modified, case).numpy()
                    actual = session.run(["logits"], {"features": modified.numpy(), "valid_mask": case.numpy()})[0]
                    self.assertTrue(np.isfinite(actual).all())
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
