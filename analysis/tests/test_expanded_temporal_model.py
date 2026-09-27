from __future__ import annotations

import importlib.util
import unittest

from analysis.compact_temporal_model import CompactTemporalConfig, CompactTemporalNetwork
from analysis.expanded_temporal_model import (
    HEAD_NAMES,
    KEEP_HEAD_INDEX,
    MODEL_KINDS,
    PRIMARY_HEAD_COUNT,
    ExpandedTemporalConfig,
    ExpandedTemporalNetwork,
    trainable_parameter_count,
)


class ExpandedTemporalConfigTests(unittest.TestCase):
    def test_metadata_declares_four_heads_and_unchanged_temporal_fields(self) -> None:
        self.assertEqual(HEAD_NAMES, ("live", "serve", "end", "keep"))
        self.assertEqual(PRIMARY_HEAD_COUNT, KEEP_HEAD_INDEX)
        for kind in MODEL_KINDS:
            with self.subTest(kind=kind):
                config = ExpandedTemporalConfig(kind=kind)
                metadata = config.to_dict()
                self.assertEqual(metadata["headNames"], list(HEAD_NAMES))
                self.assertEqual(metadata["auxiliaryHeadIndices"], [3])
                self.assertEqual(metadata["auxiliaryHeadNames"], ["keep"])
                self.assertEqual(metadata["primaryHeadCount"], 3)
                self.assertEqual(config.halo_ticks, 62 if kind == "tcn" else 8)
                self.assertEqual(config.receptive_field_ticks, 125 if kind == "tcn" else 17)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class ExpandedTemporalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        import torch

        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def output_layer(model):
        if model.config.kind == "linear":
            return model.context_head
        if model.config.kind == "mlp":
            return model.context_head[-1]
        return model.head

    def test_counts_float32_and_repeatable_initialization(self) -> None:
        import torch

        expected = {"linear": 2084, "mlp": 33604, "tcn": 29700}
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            for kind in MODEL_KINDS:
                with self.subTest(kind=kind):
                    torch.manual_seed(47)
                    first = ExpandedTemporalNetwork(CompactTemporalConfig(kind=kind)).eval()
                    torch.manual_seed(47)
                    second = ExpandedTemporalNetwork(ExpandedTemporalConfig(kind=kind)).eval()
                    self.assertIsInstance(first.config, ExpandedTemporalConfig)
                    self.assertEqual(trainable_parameter_count(first), expected[kind])
                    self.assertTrue(all(p.dtype == torch.float32 for p in first.parameters()))
                    values = torch.randn(2, 11, 104, dtype=torch.float32)
                    self.assertEqual(tuple(first(values).shape), (2, 11, 4))
                    torch.testing.assert_close(first(values), second(values), rtol=0, atol=0)
        finally:
            torch.set_default_dtype(previous_dtype)

    def test_only_output_layer_differs_from_frozen_compact_network(self) -> None:
        import torch

        for kind in MODEL_KINDS:
            with self.subTest(kind=kind), torch.no_grad():
                torch.manual_seed(53)
                original = CompactTemporalNetwork(CompactTemporalConfig(kind=kind)).eval()
                torch.manual_seed(53)
                expanded = ExpandedTemporalNetwork(ExpandedTemporalConfig(kind=kind)).eval()
                old_head, new_head = self.output_layer(original), self.output_layer(expanded)
                new_head.weight[:3].copy_(old_head.weight)
                new_head.bias[:3].copy_(old_head.bias)
                old_state, new_state = original.state_dict(), expanded.state_dict()
                self.assertEqual(set(old_state), set(new_state))
                for name in old_state:
                    if old_state[name].shape == new_state[name].shape:
                        torch.testing.assert_close(old_state[name], new_state[name], rtol=0, atol=0)
                values = torch.randn(2, 131, 104)
                torch.testing.assert_close(original(values), expanded(values)[..., :3], rtol=1e-6, atol=1e-6)

    def test_each_head_backpropagates_to_inputs_and_shared_representation(self) -> None:
        import torch

        for kind in MODEL_KINDS:
            torch.manual_seed(59)
            model = ExpandedTemporalNetwork(ExpandedTemporalConfig(kind=kind, dropout=0))
            for head_index in range(len(HEAD_NAMES)):
                with self.subTest(kind=kind, head=HEAD_NAMES[head_index]):
                    model.zero_grad(set_to_none=True)
                    values = torch.randn(2, 37, 104, requires_grad=True)
                    logits = model(values)
                    # Only one head is supervised, as with auxiliary-only data.
                    targets = torch.zeros_like(logits[..., head_index])
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[..., head_index], targets)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss).item())
                    self.assertGreater(torch.count_nonzero(values.grad).item(), 0)
                    output = self.output_layer(model)
                    self.assertGreater(torch.count_nonzero(output.weight.grad[head_index]).item(), 0)
                    others = [index for index in range(4) if index != head_index]
                    self.assertEqual(torch.count_nonzero(output.weight.grad[others]).item(), 0)
                    if kind != "linear":
                        shared = model.input_projection if kind == "tcn" else model.context_head[0]
                        self.assertGreater(torch.count_nonzero(shared.weight.grad).item(), 0)
                    for parameter in model.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_full_and_chunked_eval_match_all_four_heads_including_short_segments(self) -> None:
        import torch

        for kind in MODEL_KINDS:
            torch.manual_seed(61)
            model = ExpandedTemporalNetwork(ExpandedTemporalConfig(kind=kind)).eval()
            for length in (1, 7, 63, 252, 411):
                with self.subTest(kind=kind, length=length), torch.no_grad():
                    values = torch.randn(2, length, 104)
                    whole = model(values)
                    chunks = []
                    for core_left in range(0, length, 73):
                        core_right = min(length, core_left + 73)
                        left = max(0, core_left - model.config.halo_ticks)
                        right = min(length, core_right + model.config.halo_ticks)
                        chunk = model(values[:, left:right])
                        chunks.append(chunk[:, core_left - left:core_right - left])
                    torch.testing.assert_close(torch.cat(chunks, dim=1), whole, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
