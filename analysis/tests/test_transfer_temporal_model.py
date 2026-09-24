from __future__ import annotations

import importlib.util
import unittest

from analysis.compact_temporal_model import CompactTemporalError, DinoFusionNetwork
from analysis.expanded_temporal_model import ExpandedTemporalConfig, ExpandedTemporalNetwork
from analysis.transfer_temporal_model import (
    DINO_INPUT_DIMENSION, FUSED_DIMENSION, HEAD_NAMES, MODEL_KINDS,
    PARAMETER_COUNTS, PRIMARY_PARAMETER_COUNTS, model_for, model_metadata,
    primary_only_model, trainable_parameter_count,
)


class TransferMetadataTests(unittest.TestCase):
    def test_fixed_metadata_binds_feature_order_scaling_and_temporal_context(self):
        self.assertEqual(HEAD_NAMES, ("live", "serve", "end", "keep"))
        self.assertEqual(DINO_INPUT_DIMENSION, 3944)
        self.assertEqual(FUSED_DIMENSION, 264)
        for kind in MODEL_KINDS:
            metadata = model_metadata(kind)
            self.assertEqual(metadata["kind"], kind)
            self.assertEqual(metadata["parameters"], PARAMETER_COUNTS[kind])
            self.assertEqual(metadata["headNames"], list(HEAD_NAMES))
            self.assertEqual(metadata["receptiveFieldTicks"], 125)
            self.assertEqual(metadata["haloTicks"], 62)
            self.assertEqual(metadata["inputDimension"], 3944 if kind == "dino_tcn" else 104)
            self.assertFalse(metadata["featureExtractionIncluded"])
        dino = model_metadata("dino_tcn")
        self.assertEqual(dino["tokenCount"], 10)
        self.assertEqual(dino["tokenDimension"], 384)
        self.assertEqual(dino["tokenProjectionDimension"], 16)
        self.assertIn("AV104 only", dino["standardization"])
        self.assertIn("No percentile", dino["dinoStandardization"])
        with self.assertRaises(CompactTemporalError):
            model_metadata("mlp")
        with self.assertRaises(CompactTemporalError):
            model_for("linear")


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class TransferModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.previous_threads)

    def test_compact_initialization_forward_backward_and_rng_are_frozen_bit_exact(self):
        import torch
        torch.manual_seed(97)
        original = ExpandedTemporalNetwork(ExpandedTemporalConfig(kind="tcn"))
        original_rng = torch.get_rng_state().clone()
        torch.manual_seed(97)
        current = model_for("tcn")
        self.assertTrue(torch.equal(torch.get_rng_state(), original_rng))
        self.assertEqual(set(current.state_dict()), set(original.state_dict()))
        for name, tensor in original.state_dict().items():
            torch.testing.assert_close(tensor, current.state_dict()[name], rtol=0, atol=0)
        values = torch.randn(2, 139, 104)
        targets = torch.rand(2, 139, 4)
        outputs = []
        gradients = []
        for model in (original, current):
            torch.manual_seed(101)
            logits = model(values)
            torch.nn.functional.binary_cross_entropy_with_logits(logits, targets).backward()
            outputs.append(logits)
            gradients.append({name: parameter.grad for name, parameter in model.named_parameters()})
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)

    def test_dino_backbone_matches_original_fusion_and_only_head_shape_changes(self):
        import torch
        torch.manual_seed(107)
        original = DinoFusionNetwork().eval()
        torch.manual_seed(107)
        current = model_for("dino_tcn").eval()
        old, new = original.state_dict(), current.state_dict()
        self.assertEqual(set(old), set(new))
        for name in old:
            if name not in ("temporal.head.weight", "temporal.head.bias"):
                torch.testing.assert_close(old[name], new[name], rtol=0, atol=0)
        self.assertEqual(current.temporal.config.input_dimension, 264)
        self.assertEqual(current.token_norm.normalized_shape, (384,))
        self.assertEqual(current.token_norm.eps, 1e-5)
        self.assertEqual(current.config.to_dict()["headNames"], list(HEAD_NAMES))
        self.assertEqual(current.temporal.config.to_dict()["headNames"], list(HEAD_NAMES))
        with torch.no_grad():
            current.temporal.head.weight[:3].copy_(original.temporal.head.weight)
            current.temporal.head.bias[:3].copy_(original.temporal.head.bias)
            values = torch.randn(2, 131, 3944)
            torch.testing.assert_close(original(values), current(values)[..., :3], rtol=1e-6, atol=1e-6)

    def test_parameter_counts_float32_and_all_four_heads_transfer_gradients(self):
        import torch
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            for kind in MODEL_KINDS:
                torch.manual_seed(109)
                model = model_for(kind)
                self.assertEqual(trainable_parameter_count(model), PARAMETER_COUNTS[kind])
                self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
                for head_index in range(4):
                    with self.subTest(kind=kind, head=head_index):
                        model.zero_grad(set_to_none=True)
                        values = torch.randn(2, 37, model.config.input_dimension, dtype=torch.float32, requires_grad=True)
                        logits = model(values)
                        self.assertEqual(tuple(logits.shape), (2, 37, 4))
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits[..., head_index], torch.zeros_like(logits[..., head_index]))
                        loss.backward()
                        self.assertGreater(torch.count_nonzero(values.grad[..., :104]).item(), 0)
                        temporal = model.temporal if kind == "dino_tcn" else model
                        self.assertGreater(torch.count_nonzero(temporal.input_projection.weight.grad).item(), 0)
                        self.assertGreater(torch.count_nonzero(temporal.head.weight.grad[head_index]).item(), 0)
                        self.assertEqual(torch.count_nonzero(temporal.head.weight.grad[[i for i in range(4) if i != head_index]]).item(), 0)
                        if kind == "dino_tcn":
                            self.assertGreater(torch.count_nonzero(values.grad[..., 104:]).item(), 0)
                            self.assertGreater(torch.count_nonzero(model.token_norm.weight.grad).item(), 0)
                            self.assertGreater(torch.count_nonzero(model.token_projection.weight.grad).item(), 0)
                        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()))
        finally:
            torch.set_default_dtype(previous_dtype)

    def test_whole_real_halo_chunks_and_trimmed_heads_match_even_at_short_edges(self):
        import torch
        for kind in MODEL_KINDS:
            torch.manual_seed(113)
            model = model_for(kind).eval()
            saved = {name: tensor.clone() for name, tensor in model.state_dict().items()}
            rng = torch.get_rng_state().clone()
            primary = primary_only_model(model)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(sum(p.numel() for p in primary.parameters()), PRIMARY_PARAMETER_COUNTS[kind])
            self.assertEqual(trainable_parameter_count(primary), 0)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            self.assertFalse(primary.training)
            self.assertEqual(primary.config.to_dict()["headNames"], list(HEAD_NAMES[:3]))
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(saved[name], tensor, rtol=0, atol=0)
            for length in (1, 7, 63, 252, 411):
                with self.subTest(kind=kind, length=length), torch.no_grad():
                    values = torch.randn(2, length, model.config.input_dimension)
                    whole = model(values)
                    pieces = []
                    for start in range(0, length, 128):
                        stop = min(start+128, length)
                        left, right = max(0, start-62), min(length, stop+62)
                        output = model(values[:, left:right])
                        pieces.append(output[:, start-left:stop-left])
                    torch.testing.assert_close(torch.cat(pieces, dim=1), whole, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(primary(values), whole[..., :3], rtol=1e-5, atol=1e-6)
            with self.assertRaises(CompactTemporalError):
                primary_only_model(primary)

    def test_input_contract_rejects_wrong_dimensions_and_dtype(self):
        import torch
        for kind in MODEL_KINDS:
            model = model_for(kind)
            with self.assertRaises(CompactTemporalError):
                model(torch.zeros(1, 3, model.config.input_dimension-1))
            with self.assertRaises(CompactTemporalError):
                model(torch.zeros(1, 3, model.config.input_dimension, dtype=torch.float64))
            with self.assertRaises(CompactTemporalError):
                model(torch.zeros(1, 0, model.config.input_dimension))


if __name__ == "__main__":
    unittest.main()
