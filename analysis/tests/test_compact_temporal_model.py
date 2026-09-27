from __future__ import annotations

import importlib.util
import unittest

from analysis.compact_temporal_model import (
    CONTEXT_OFFSETS_TICKS,
    HEAD_NAMES,
    MODEL_KINDS,
    CompactTemporalConfig,
    CompactTemporalError,
    CompactTemporalNetwork,
    DinoFusionNetwork,
    contextualize_ticks,
    trainable_parameter_count,
)


class CompactTemporalConfigTests(unittest.TestCase):
    def test_exact_receptive_fields_and_invalid_configuration(self) -> None:
        self.assertEqual(CompactTemporalConfig().halo_ticks, 62)
        self.assertEqual(CompactTemporalConfig().receptive_field_ticks, 125)
        for kind in ("linear", "mlp"):
            self.assertEqual(CompactTemporalConfig(kind=kind).halo_ticks, 8)
            self.assertEqual(CompactTemporalConfig(kind=kind).receptive_field_ticks, 17)
        for kwargs in (
            {"kind": "unknown"}, {"input_dimension": 0}, {"hidden_dimension": True},
            {"kernel_size": 4}, {"dilations": ()}, {"dilations": (1, 0)},
            {"dropout": float("nan")}, {"dropout": 1.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(CompactTemporalError):
                CompactTemporalConfig(**kwargs).validate()


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class CompactTemporalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        import torch

        torch.set_num_threads(cls.previous_threads)

    def test_context_controls_match_offset_major_endpoint_clamping(self) -> None:
        import torch

        values = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
        contextual = contextualize_ticks(values)
        self.assertEqual(tuple(contextual.shape), (1, 6, 10))
        expected = torch.stack([
            torch.cat([values[0, min(5, max(0, tick + offset))] for offset in CONTEXT_OFFSETS_TICKS])
            for tick in range(6)
        ])
        torch.testing.assert_close(contextual[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            contextualize_ticks(values[:, :1]), values[:, :1].repeat(1, 1, 5), rtol=0, atol=0
        )

    def test_all_models_backpropagate_masked_three_head_loss(self) -> None:
        import torch

        for kind in MODEL_KINDS:
            with self.subTest(kind=kind):
                torch.manual_seed(7)
                model = CompactTemporalNetwork(CompactTemporalConfig(kind=kind))
                values = torch.randn(2, 17, 104)
                logits = model(values)
                self.assertEqual(tuple(logits.shape), (2, 17, len(HEAD_NAMES)))
                self.assertEqual(logits.dtype, torch.float32)
                targets = torch.randint(0, 2, logits.shape).float()
                valid = torch.ones(2, 17, dtype=torch.bool)
                valid[0, -4:] = False
                logits.retain_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[valid], targets[valid]
                )
                loss.backward()
                self.assertTrue(torch.isfinite(loss).item())
                self.assertEqual(torch.count_nonzero(logits.grad[~valid]).item(), 0)
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertTrue(any(torch.count_nonzero(p.grad).item() for p in model.parameters()))

    def test_seeded_models_have_bounded_float32_parameters(self) -> None:
        import torch

        expected_counts = {"linear": 1563, "mlp": 33539, "tcn": 29635}
        for kind in MODEL_KINDS:
            with self.subTest(kind=kind):
                torch.manual_seed(23)
                first = CompactTemporalNetwork(CompactTemporalConfig(kind=kind)).eval()
                torch.manual_seed(23)
                second = CompactTemporalNetwork(CompactTemporalConfig(kind=kind)).eval()
                self.assertEqual(trainable_parameter_count(first), expected_counts[kind])
                self.assertTrue(all(p.dtype == torch.float32 for p in first.parameters()))
                values = torch.randn(1, 9, 104)
                torch.testing.assert_close(first(values), second(values), rtol=0, atol=0)
                torch.testing.assert_close(first(values), first(values), rtol=0, atol=0)

    def test_eval_overlap_chunks_match_whole_sequence_including_real_edges(self) -> None:
        import torch

        for kind in MODEL_KINDS:
            with self.subTest(kind=kind), torch.no_grad():
                torch.manual_seed(11)
                model = CompactTemporalNetwork(CompactTemporalConfig(kind=kind)).eval()
                values = torch.randn(2, 411, 104)
                whole = model(values)
                stitched = []
                halo = model.config.halo_ticks
                for left in range(0, values.shape[1], 73):
                    right = min(values.shape[1], left + 73)
                    input_left = max(0, left - halo)
                    input_right = min(values.shape[1], right + halo)
                    chunk = model(values[:, input_left:input_right])
                    stitched.append(chunk[:, left - input_left:right - input_left])
                torch.testing.assert_close(torch.cat(stitched, dim=1), whole, rtol=1e-5, atol=1e-6)

    def test_model_has_no_dependency_outside_declared_tcn_halo(self) -> None:
        import torch

        torch.manual_seed(19)
        model = CompactTemporalNetwork().eval()
        values = torch.randn(1, 257, 104)
        changed = values.clone()
        target = 128
        halo = model.config.halo_ticks
        changed[:, :target - halo] += 100
        changed[:, target + halo + 1:] -= 100
        with torch.no_grad():
            torch.testing.assert_close(
                model(values)[:, target], model(changed)[:, target], rtol=0, atol=0
            )

    def test_invalid_tensor_contract_fails_before_inference(self) -> None:
        import torch

        model = CompactTemporalNetwork()
        for values in (
            torch.zeros(4, 104), torch.zeros(1, 0, 104), torch.zeros(1, 4, 90),
            torch.zeros(1, 4, 104, dtype=torch.float64),
        ):
            with self.subTest(shape=values.shape), self.assertRaises(CompactTemporalError):
                model(values)


    def test_dino_fusion_uses_matched_head_and_backpropagates(self) -> None:
        import torch

        torch.manual_seed(31)
        model = DinoFusionNetwork()
        self.assertEqual(model.config.input_dimension, 3944)
        self.assertEqual(model.temporal.config.input_dimension, 264)
        self.assertEqual(model.config.halo_ticks, 62)
        self.assertEqual(trainable_parameter_count(model), 46803)
        values = torch.randn(2, 11, 3944)
        logits = model(values)
        self.assertEqual(tuple(logits.shape), (2, 11, 3))
        logits.square().mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
        self.assertGreater(torch.count_nonzero(model.token_projection.weight.grad).item(), 0)
        with self.assertRaises(CompactTemporalError):
            DinoFusionNetwork(CompactTemporalConfig(input_dimension=104))

    def test_dino_token_normalization_preserves_chunk_and_batch_independence(self) -> None:
        import torch

        torch.manual_seed(37)
        model = DinoFusionNetwork().eval()
        values = torch.randn(1, 301, 3944)
        with torch.no_grad():
            whole = model(values)
            # 62 real ticks on either side of this 65-tick central crop.
            chunk = model(values[:, 56:245])
            torch.testing.assert_close(chunk[:, 62:127], whole[:, 118:183], rtol=1e-5, atol=1e-6)
            paired = model(torch.cat((values, torch.randn_like(values) * 100), dim=0))
            torch.testing.assert_close(paired[:1], whole, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
