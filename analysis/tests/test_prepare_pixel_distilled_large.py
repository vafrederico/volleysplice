"""Native artifact identity/geometry guards; generated inputs are synthetic."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

with patch('analysis.private_ledger.private_value', side_effect=lambda key: '/synthetic/' + key):
    spec = importlib.util.spec_from_file_location('native_distilled_large',
        Path(__file__).resolve().parents[2] / 'scripts/prepare-pixel-distilled-mobile-large.py')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)


class NativeDistilledLargeTests(unittest.TestCase):
    def test_selected_task_requires_strict99_and_exact_draw(self):
        tasks = [dict(variant='variant-test', seed=7, splitSeed=8),
                 dict(variant='variant-test', seed=7, splitSeed=9)]
        chosen = dict(mode='recall', variant='variant-test', draw=9, floorPercent=99,
                      setting=dict(innerR_core=.991))
        result, task = study.selected_task(dict(tasks=tasks), dict(selected=[chosen]), 'recall')
        self.assertIs(result, chosen)
        self.assertIs(task, tasks[1])
        chosen['setting']['innerR_core'] = .9899999
        with self.assertRaisesRegex(ValueError, 'strict target99'):
            study.selected_task(dict(tasks=tasks), dict(selected=[chosen]), 'recall')

    def test_duplicate_selection_or_task_fails_closed(self):
        chosen = dict(mode='f1', variant='variant-test', draw=9, floorPercent=99,
                      setting=dict(innerR_core=.999))
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            study.selected_task(dict(tasks=[]), dict(selected=[chosen, chosen]), 'f1')
        task = dict(variant='variant-test', seed=9)
        with self.assertRaisesRegex(ValueError, 'uniquely'):
            study.selected_task(dict(tasks=[task, task]), dict(selected=[chosen]), 'f1')

    def test_alias_resolution_uses_recording_manifest_and_no_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ledger.json'
            path.write_text(json.dumps(dict(originalToAlias={
                'source-recording': 'recording-test', 'source-stem': 'recording-test'})))
            platform = 'WINDOWS' if os.name == 'nt' else 'POSIX'
            with patch.dict(os.environ, {'VOLLEYCUT_PRIVATE_LEDGER_' + platform: str(path)}):
                self.assertEqual(study.recording_id('recording-test', {'source-recording'}), 'source-recording')
                with self.assertRaisesRegex(ValueError, 'uniquely'):
                    study.recording_id('recording-test', {'source-recording', 'source-stem'})
                with self.assertRaisesRegex(ValueError, 'uniquely'):
                    study.recording_id('unknown-index', {'source-recording'})

    def test_projector_is_excluded_and_regional_pool_shape_preserved(self):
        encoder = torch.nn.Conv2d(3, 960, 1)
        wrapped = study.RegionalEncoder(encoder)
        image = torch.ones(1, 3, 7, 7)
        weights = torch.full((1, 4, 7, 7), 1 / 49)
        with torch.inference_mode():
            result = wrapped(image, weights)
            expected = encoder(image).mean(dim=(2, 3))
        self.assertEqual(tuple(result.shape), (1, 4, 960))
        torch.testing.assert_close(result[:, 0], expected)
        self.assertEqual(sum(p.numel() for p in wrapped.parameters()), sum(p.numel() for p in encoder.parameters()))
        self.assertFalse(any('projector' in key or 'classifier' in key for key in wrapped.state_dict()))

    def test_cropped_input_rejected_before_cache_read(self):
        source = dict(id='recording-test', roi=dict(x=0, y=.1, width=1, height=.9))
        image = dict(id='recording-test', contract=dict(plan={'path': 'synthetic.json'}))
        with patch.object(study.inputs, 'verified', return_value='synthetic.json'), \
             patch.object(study, 'read', return_value=dict(records=[source])):
            with self.assertRaisesRegex(ValueError, 'full-frame'):
                study.checked_geometry(image)

    def test_graph_metadata_contains_model_identity_without_teacher_weights(self):
        import onnx
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'graph.onnx'
            graph = onnx.helper.make_graph([onnx.helper.make_node('Identity', ['x'], ['y'])], 'synthetic',
                [onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1])],
                [onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1])])
            onnx.save(onnx.helper.make_model(graph), str(path))
            study.annotate_graph(path, 'f1', 'a' * 64, 'regional-encoder')
            metadata = {row.key: row.value for row in onnx.load(str(path)).metadata_props}
            self.assertEqual(metadata['modelIdentity'], study.MODEL_ID)
            self.assertEqual(metadata['selectionMode'], 'f1')
            self.assertEqual(metadata['weightsSha256'], 'a' * 64)
            self.assertEqual(metadata['trainingProjectorIncluded'], 'false')


if __name__ == '__main__':
    unittest.main()
