from __future__ import annotations

import importlib.util
import unittest

from analysis import short_context_temporal_model as candidate
from analysis import transfer_temporal_model as frozen
from analysis.compact_temporal_model import CompactTemporalError, trainable_parameter_count


class ContextMetadataTests(unittest.TestCase):
    def test_explicit_temporal_context_and_unchanged_feature_contract(self):
        for kind in candidate.MODEL_KINDS:
            for context, field, halo in (("original", 125, 62), ("short", 33, 16)):
                md = candidate.model_metadata(kind, context=context)
                self.assertEqual(md["receptiveFieldTicks"], field)
                self.assertEqual(md["haloTicks"], halo)
                self.assertEqual(md["dilations"], list(candidate.CONTEXT_DILATIONS[context]))
                self.assertEqual(md["originalPairedTrainingChunkTicks"], 252)
                self.assertEqual(md["originalPairedTrainingHaloTicks"], 62)
                for key in ("inputDimension", "parameters", "headNames", "standardization", "featureExtractionIncluded"):
                    self.assertEqual(md[key], frozen.model_metadata(kind)[key])
                self.assertEqual(md["status"], "research-only-requires-registered-caller")
        with self.assertRaises(CompactTemporalError):
            candidate.model_metadata("tcn", context="unregistered-other-context")
        with self.assertRaises(CompactTemporalError):
            candidate.model_metadata("linear")


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class ShortContextModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.old_threads)

    def test_parameter_names_shapes_values_count_and_rng_exact_for_both_contexts(self):
        import torch
        for kind in candidate.MODEL_KINDS:
            for seed in (3407, 1729):
                torch.manual_seed(seed)
                original = frozen.model_for(kind)
                original_rng = torch.get_rng_state().clone()
                for context in candidate.CONTEXT_DILATIONS:
                    torch.manual_seed(seed)
                    model = candidate.model_for(kind, context=context)
                    self.assertTrue(torch.equal(torch.get_rng_state(), original_rng))
                    self.assertEqual(list(model.state_dict()), list(original.state_dict()))
                    self.assertEqual(trainable_parameter_count(model), frozen.PARAMETER_COUNTS[kind])
                    for name, parameter in model.state_dict().items():
                        self.assertEqual(parameter.dtype, torch.float32)
                        torch.testing.assert_close(parameter, original.state_dict()[name], rtol=0, atol=0)
                    temporal = model.temporal if kind == "dino_tcn" else model
                    self.assertEqual(model.config.dilations, temporal.config.dilations)
                    self.assertEqual(tuple(b.depthwise.dilation[0] for b in temporal.blocks), model.config.dilations)
                    self.assertEqual(model.halo_ticks, 16 if context == "short" else 62)
                    self.assertEqual(model.receptive_field_ticks, model.config.receptive_field_ticks)
                    for block in temporal.blocks:
                        self.assertEqual(block.depthwise.padding[0], 2 * block.depthwise.dilation[0])
                        self.assertEqual(block.depthwise._reversed_padding_repeated_twice, block.depthwise.padding * 2)

    def test_original_configuration_forward_backward_dropout_rng_bit_exact(self):
        import torch
        for kind in candidate.MODEL_KINDS:
            torch.manual_seed(101)
            original = frozen.model_for(kind)
            torch.manual_seed(101)
            adapted = candidate.model_for(kind, context="original")
            values = torch.randn(2, 45, original.config.input_dimension)
            targets = torch.rand(2, 45, 4)
            outputs, gradients, rngs = [], [], []
            for model in (original, adapted):
                torch.manual_seed(103)
                logits = model(values)
                torch.nn.functional.binary_cross_entropy_with_logits(logits, targets).backward()
                outputs.append(logits)
                gradients.append({n: p.grad for n, p in model.named_parameters()})
                rngs.append(torch.get_rng_state().clone())
            self.assertTrue(torch.equal(rngs[0], rngs[1]))
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            for name in gradients[0]:
                torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)

    def test_changed_context_retains_same_dropout_masks_and_forward_rng_consumption(self):
        import torch
        for kind in candidate.MODEL_KINDS:
            values = torch.ones(2, 37, 3944 if kind == "dino_tcn" else 104)
            masks, states = [], []
            for context in candidate.CONTEXT_DILATIONS:
                torch.manual_seed(107)
                model = candidate.model_for(kind, context=context).train()
                temporal = model.temporal if kind == "dino_tcn" else model
                captured, hooks = [], []
                for block in temporal.blocks:
                    hooks.append(block.dropout.register_forward_hook(
                        lambda _module, args, output: captured.append(((output == 0) & (args[0] != 0)).clone())))
                torch.manual_seed(109)
                model(values)
                states.append(torch.get_rng_state().clone())
                masks.append(captured)
                for hook in hooks:
                    hook.remove()
            self.assertTrue(torch.equal(states[0], states[1]))
            self.assertEqual(len(masks[0]), 5)
            for left, right in zip(*masks):
                self.assertTrue(torch.equal(left, right))

    def test_short_receptive_support_is_inclusive_33_ticks_for_both_inputs(self):
        import torch
        for kind in candidate.MODEL_KINDS:
            model = candidate.model_for(kind).eval()
            temporal = model.temporal if kind == "dino_tcn" else model
            # Positive weights/AV avoid inactive ReLUs hiding theoretical paths.
            with torch.no_grad():
                for parameter in temporal.parameters():
                    parameter.fill_(.02)
            values = torch.ones(1, 81, model.config.input_dimension, requires_grad=True)
            model(values)[0, 40, 0].backward()
            active_av_ticks = torch.nonzero(values.grad[0, :, 0] != 0).flatten().tolist()
            self.assertEqual(active_av_ticks, list(range(24, 57)))
            self.assertEqual(torch.count_nonzero(values.grad[:, :24]).item(), 0)
            self.assertEqual(torch.count_nonzero(values.grad[:, 57:]).item(), 0)
            with torch.no_grad():
                altered = values.detach().clone()
                altered[:, :24] = torch.randn_like(altered[:, :24]) * 50
                altered[:, 57:] = torch.randn_like(altered[:, 57:]) * 50
                torch.testing.assert_close(model(values)[0, 40], model(altered)[0, 40], rtol=0, atol=0)

    def test_short_chunks_match_whole_with_minimum_or_legacy_real_halo(self):
        import torch
        for kind in candidate.MODEL_KINDS:
            torch.manual_seed(113)
            model = candidate.model_for(kind).eval()
            for length in (1, 7, 33, 252, 407):
                values = torch.randn(1, length, model.config.input_dimension)
                with torch.no_grad():
                    whole = model(values)
                    self.assertEqual(tuple(whole.shape), (1, length, 4))
                    for halo in (16, 62):
                        parts = []
                        for start in range(0, length, 128):
                            stop = min(start + 128, length)
                            left, right = max(0, start - halo), min(length, stop + halo)
                            parts.append(model(values[:, left:right])[:, start-left:stop-left])
                        torch.testing.assert_close(torch.cat(parts, dim=1), whole, rtol=1e-5, atol=1e-6)

    def test_primary_derivative_preserves_context_parameters_and_rng(self):
        import torch
        for kind in candidate.MODEL_KINDS:
            model = candidate.model_for(kind).eval()
            rng = torch.get_rng_state().clone()
            primary = candidate.primary_only_model(model)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(primary.config.dilations, (1, 1, 2, 2, 2))
            self.assertEqual(primary.halo_ticks, 16)
            self.assertEqual(primary.receptive_field_ticks, 33)
            self.assertEqual(sum(p.numel() for p in primary.parameters()), frozen.PRIMARY_PARAMETER_COUNTS[kind])
            values = torch.randn(1, 51, model.config.input_dimension)
            with torch.no_grad():
                torch.testing.assert_close(primary(values), model(values)[..., :3], rtol=1e-5, atol=1e-6)

    def test_metadata_and_invalid_profile_do_not_consume_rng(self):
        import torch
        rng = torch.get_rng_state().clone()
        for kind in candidate.MODEL_KINDS:
            candidate.model_metadata(kind)
        with self.assertRaises(CompactTemporalError):
            candidate.model_for("tcn", context="different")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
