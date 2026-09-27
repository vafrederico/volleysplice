from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from analysis import mobile_visual_features as mobile
from analysis import neural_mobile_distillation as study
from analysis.distillation_image_inputs import teaching_indexes, uint8_from_normalized
from analysis.neural_context_development import identity
from analysis.neural_development import Example
from analysis.neural_expanded_development import Supervised


def row(identifier='a', group='g', tier='exact'):
    times = np.arange(8)/4.
    e = Example(identifier, group, 2., times, np.arange(832, dtype=np.float32).reshape(8,104),
                np.zeros((8,4), np.float32), np.ones(8, bool), (), (), 'grass')
    return Supervised(e, np.ones((8,4), np.float32), tier)


class DistillationTests(unittest.TestCase):
    def test_pixel_roundtrip_preserves_exact_normalized_input_and_letterbox(self):
        rng = np.random.default_rng(2)
        for shape in ((23, 51, 3), (57, 21, 3), (32, 32, 3)):
            values, box, quality = mobile.preprocess_frame(rng.integers(0,256,shape,dtype=np.uint8))
            pixels = uint8_from_normalized(values, mobile.RGB_MEAN, mobile.RGB_STD)
            actual = study.normalize_pixels(pixels[None], 'cpu')[0].numpy()
            np.testing.assert_array_equal(actual, values)

    def test_teaching_selection_excludes_ignored_and_is_even_not_label_dependent(self):
        av = np.arange(80)/4
        valid = (av < 3) | (av >= 12)
        times = np.arange(40)/2
        indexes = teaching_indexes(times, av, valid, maximum=7)
        self.assertEqual(len(indexes), 7)
        self.assertEqual(indexes[0], 0)
        self.assertEqual(indexes[-1], 39)
        self.assertTrue(np.all(valid[indexes*2]))
        with self.assertRaisesRegex(ValueError, 'No usable'):
            teaching_indexes(times, av, np.zeros(80,bool))

    def test_teacher_projection_pool_matches_adaptive_reference_and_gradients(self):
        x = torch.randn(2, 576, 7, 7, requires_grad=True)
        projection = torch.nn.Linear(576, 384)
        expected = projection(torch.cat((x.mean((2,3))[:,None],
            torch.nn.functional.adaptive_avg_pool2d(x,(3,3)).flatten(2).transpose(1,2)),dim=1))
        actual = study.teaching_tokens(x, projection)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        teacher = torch.randn_like(actual)
        first = torch.autograd.grad(study.distillation_loss(actual,teacher).sum(), x, retain_graph=True)[0]
        second = torch.autograd.grad(study.distillation_loss(expected,teacher).sum(), x)[0]
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)

    def test_global_and_region_loss_each_contribute_half(self):
        teacher = torch.ones((1,10,384))
        identical = study.distillation_loss(teacher, teacher)
        self.assertLess(abs(identical.item()), 1e-6)
        student = teacher.clone()
        student[:,0] *= -1
        self.assertAlmostEqual(study.distillation_loss(student, teacher).item(), 1., places=6)
        student = -teacher.clone()
        student[:,0] *= -1
        self.assertAlmostEqual(study.distillation_loss(student, teacher).item(), 1., places=6)

    def test_auxiliary_same_source_is_excluded_from_student(self):
        data = {'exact':[row('a','held'), row('b','train')],
                'draft':[row('c','held','draft'),row('d','extra','draft')],
                'coverage':[row('e','held','coverage'),row('f','extra','coverage')]}
        train, aux, permitted = study.allowed_records(data, {'held'})
        self.assertEqual([r.example.id for r in permitted], ['b','d','f'])
        self.assertFalse(any(r.example.group == 'held' for r in permitted))

    def test_attachment_preserves_supervision_and_av_but_uses_new_features(self):
        original = row()
        original = replace(original, example=replace(original.example, valid=np.array([1,1,0,0,1,1,1,1],bool)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'features.npz'
            tokens = np.arange(4*4*576,dtype=np.float32).reshape(4,4,576).astype(np.float16)
            np.savez(path, timestamps=np.arange(4)/2, tokens=tokens, quality=np.zeros((4,6),np.float32),
                     selected_presentation_times=np.arange(4)/2)
            changed = study.attach_rows([original], {'a':{'sourceGroup':'g','output':identity(path)}})[0]
            self.assertEqual(changed.example.values.shape, (8,2416))
            np.testing.assert_array_equal(changed.example.values[:,:104],original.example.values)
            np.testing.assert_array_equal(changed.example.values[1,104:2408],tokens[0].flatten())
            self.assertEqual(changed.example.values[1,-2],.25)
            self.assertIs(changed.mask,original.mask)
            for name in ('targets','times','valid','truth','ignored'):
                self.assertIs(getattr(changed.example,name),getattr(original.example,name))

    def test_rejects_changed_feature_file_even_with_reused_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'x'
            path.write_bytes(b'first')
            reference=identity(path)
            study.verify_once(reference)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                study.verify_once(reference)


if __name__ == '__main__':
    unittest.main()
