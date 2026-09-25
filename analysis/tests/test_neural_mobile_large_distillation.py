"""Synthetic, CPU-only checks; no dataset, NAS artifacts, or downloads required."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

# Historical shared helper modules resolve their experiment roots at import.
# These tests use only mocked roots and synthetic arrays, never a real ledger.
with patch('analysis.private_ledger.private_value', side_effect=lambda key: '/synthetic/' + key):
    from analysis import neural_mobile_large_distillation as study


def rows():
    return [SimpleNamespace(example=SimpleNamespace(id='recording-test', group='source-test'))]


def tiny_student(seed, device, checkpoint):
    torch.manual_seed(seed)
    encoder = nn.Sequential(nn.Conv2d(3, 960, 1), nn.BatchNorm2d(960)).to(device).eval()
    return encoder, nn.Linear(960, 384).to(device)


class LargeDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def training_inputs(self):
        rng = np.random.default_rng(8)
        pixels = rng.integers(0, 256, (17, 3, 3, 3), dtype=np.uint8)
        targets = rng.normal(size=(17, 10, 384)).astype(np.float32)
        return pixels, targets, np.linspace(1., 2., 17, dtype=np.float32), [{'id': 'recording-test'}]

    def fit(self, folder, *, seed=3, build=tiny_student, arrays=None):
        values = self.training_inputs() if arrays is None else arrays
        reference = {'path': 'synthetic-checkpoint.pth', 'sha256': '8738ca79' + '0' * 56}
        with patch.object(study, '_checkpoint_identity', return_value=reference), \
             patch.object(study, 'training_arrays', return_value=values), \
             patch.object(study, 'build_student', side_effect=build), redirect_stdout(io.StringIO()):
            result = study.fit_student(rows(), {'held-source'}, {'recording-test': {'version': 1}},
                {'recording-test': {'version': 1}}, seed, folder, 'contract-test', 'cpu', 'synthetic-checkpoint.pth')
        return result

    def test_actual_large_architecture_has_960_channel_features_and_training_only_projector(self):
        from torchvision.models import mobilenet_v3_large

        original = mobilenet_v3_large(weights=None)
        reference = {'path': 'synthetic-checkpoint.pth', 'sha256': '8738ca79' + '0' * 56}
        with patch.object(study, '_checkpoint_identity', return_value=reference), \
             patch('torchvision.models.mobilenet_v3_large', wraps=mobilenet_v3_large) as constructor, \
             patch.object(torch, 'load', return_value=original.state_dict()) as loader:
            encoder, projector = study.build_student(3, 'cpu', reference['path'])
        constructor.assert_called_once_with(weights=None)
        loader.assert_called_once_with(reference['path'], map_location='cpu', weights_only=True)
        with torch.no_grad():
            spatial = encoder(torch.zeros(1, 3, 224, 224))
        self.assertEqual(spatial.shape, (1, 960, 7, 7))
        self.assertEqual(projector.weight.shape, (384, 960))
        self.assertEqual(study.teaching_tokens(spatial, projector).shape, (1, 10, 384))
        self.assertTrue(all(parameter.requires_grad for parameter in encoder.parameters()))
        self.assertTrue(all(not layer.training for layer in encoder.modules() if isinstance(layer, nn.BatchNorm2d)))

    def test_checkpoint_missing_or_wrong_prefix_cannot_download(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(torch.hub, 'download_url_to_file') as download:
            path = Path(temporary) / 'checkpoint.pth'
            with self.assertRaisesRegex(ValueError, 'downloads are disabled'):
                study.build_student(3, 'cpu', path)
            path.write_bytes(b'synthetic wrong checkpoint')
            with self.assertRaisesRegex(ValueError, 'ImageNet V1 hash prefix'):
                study.build_student(3, 'cpu', path)
        download.assert_not_called()

    def test_fit_repeats_same_exposure_and_weights_with_frozen_batchnorm(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = self.fit(Path(temporary) / 'first')
            second = self.fit(Path(temporary) / 'second')
            self.assertEqual(len(first['history']), 8)
            self.assertEqual(first['history'], second['history'])
            self.assertEqual(first['recipe']['batchSize'], 16)
            self.assertEqual(first['trainingFrames'], 17)
            self.assertEqual(first['trainingOnlyProjectorParameters'], 960 * 384 + 384)
            with np.load(first['weights']['path'], allow_pickle=False) as a, \
                 np.load(second['weights']['path'], allow_pickle=False) as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
                np.testing.assert_array_equal(a['encoder::1.running_mean'], np.zeros(960))
                np.testing.assert_array_equal(a['encoder::1.running_var'], np.ones(960))
                self.assertEqual(a['encoder::1.num_batches_tracked'], 0)
            self.assertEqual(self.fit(Path(temporary) / 'first'), first)
            with self.assertRaisesRegex(ValueError, 'resume identity differs'):
                self.fit(Path(temporary) / 'first', seed=4)

    def test_resume_refuses_changed_input_mapping_and_incomplete_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'fit'
            result = self.fit(folder)
            with patch.object(study, '_checkpoint_identity', return_value=result['initialCheckpoint']):
                with self.assertRaisesRegex(ValueError, 'resume identity differs'):
                    study.fit_student(rows(), {'held-source'}, {'recording-test': {'version': 2}},
                        {'recording-test': {'version': 1}}, 3, folder, 'contract-test', 'cpu', 'checkpoint')
            incomplete = Path(temporary) / 'incomplete'
            incomplete.mkdir()
            with self.assertRaisesRegex(ValueError, 'Incomplete Large student fit'):
                self.fit(incomplete)

    def test_holdout_source_is_rejected_before_reading_artifacts(self):
        with self.assertRaisesRegex(ValueError, 'Held source'):
            study.fit_student(rows(), {'source-test'}, {}, {}, 3, Path('unused'), 'contract', 'cpu', 'unused')

    def test_nonfinite_loss_and_gradient_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            arrays = self.training_inputs()
            arrays[1][:] = np.nan
            with self.assertRaisesRegex(ValueError, 'Nonfinite distillation loss'):
                self.fit(Path(temporary) / 'loss', arrays=arrays)

            def bad_gradient(seed, device, checkpoint):
                encoder, projector = tiny_student(seed, device, checkpoint)
                next(encoder.parameters()).register_hook(lambda grad: torch.full_like(grad, float('nan')))
                return encoder, projector

            with self.assertRaisesRegex(ValueError, 'Nonfinite student gradient'):
                self.fit(Path(temporary) / 'gradient', build=bad_gradient)
            self.assertFalse((Path(temporary) / 'gradient' / 'completed.json').exists())

    def test_load_encoder_excludes_projector_and_freezes_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            student = self.fit(Path(temporary) / 'fit')
            with patch.object(study, '_checkpoint_identity', return_value=student['initialCheckpoint']), \
                 patch.object(study, 'build_student', side_effect=tiny_student):
                encoder = study.load_encoder(student, 'cpu', 'checkpoint')
            self.assertFalse(encoder.training)
            self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
            self.assertEqual(encoder(torch.zeros(1, 3, 3, 3)).shape, (1, 960, 3, 3))
            with patch.object(study, '_checkpoint_identity', return_value={'path': 'changed', 'sha256': 'changed'}):
                with self.assertRaisesRegex(ValueError, 'initial checkpoint differs'):
                    study.load_encoder(student, 'cpu', 'checkpoint')

    def test_extraction_preserves_media_alignment_and_uses_four_large_tokens_in_batches_of_32(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.batches = []

            def forward(self, values):
                self.batches.append(len(values))
                return values.mean(1, keepdim=True).expand(-1, 960, -1, -1)

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            pixels = np.random.default_rng(1).integers(0, 256, (33, 3, 3, 3), dtype=np.uint8)
            np.save(folder / 'images.npy', pixels)
            times = np.arange(33) / 2.
            quality = np.arange(33 * 6, dtype=np.float32).reshape(33, 6)
            np.savez(folder / 'timing.npz', times=times, selected_pts=times + .01,
                     quality=quality, boxes=np.tile([0., 0., 1., 1.], (33, 1)))
            record = {'id': 'recording-test', 'sourceGroup': 'source-test', 'arrays': {
                'images224': study.identity(folder / 'images.npy'), 'timing': study.identity(folder / 'timing.npz')}}
            student = {'recipe': study.RECIPE, 'weights': {'path': 'synthetic', 'sha256': 'synthetic'},
                       'contractSha256': 'contract-test'}
            encoder = Encoder()
            result = study.extract_student_record(encoder, student, record, folder / 'features', 'cpu')
            self.assertEqual(encoder.batches, [32, 1])
            self.assertEqual(result['shape'], [33, 4, 960])
            self.assertFalse(result['labelsUsed'])
            with np.load(result['output']['path'], allow_pickle=False) as payload:
                self.assertEqual(payload['tokens'].dtype, np.float16)
                np.testing.assert_array_equal(payload['timestamps'], times)
                np.testing.assert_array_equal(payload['selected_presentation_times'], times + .01)
                np.testing.assert_array_equal(payload['quality'], quality)
                normalized = study.normalize_pixels(pixels[:1], 'cpu').numpy()
                expected_global = normalized.mean()
                np.testing.assert_allclose(payload['tokens'][0, 0], expected_global, atol=1e-3)
            self.assertEqual(study.extract_student_record(encoder, student, record, folder / 'features', 'cpu'), result)
            self.assertEqual(encoder.batches, [32, 1])
            changed = {**student, 'weights': {'path': 'changed', 'sha256': 'changed'}}
            with self.assertRaisesRegex(ValueError, 'feature resume differs'):
                study.extract_student_record(encoder, changed, record, folder / 'features', 'cpu')


if __name__ == '__main__':
    unittest.main()
