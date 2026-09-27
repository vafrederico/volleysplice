"""Integrity helpers for the isolated encoder precision experiment."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

PATH=Path(__file__).resolve().parents[2]/'scripts/evaluate-dino-precision.py'
SPEC=importlib.util.spec_from_file_location('dino_precision_test',PATH)
module=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class PrecisionTests(unittest.TestCase):
    def test_normalization_matches_original_preprocessor(self):
        from analysis.dinov2_embeddings import preprocess_frames, letterbox_frame, _crop_frame
        frame=np.random.default_rng(5).integers(0,256,(412,731,3),dtype=np.uint8)
        roi=(.02,.18,.96,.8)
        rgb=letterbox_frame(_crop_frame(frame,roi),336)[None]
        np.testing.assert_array_equal(module.normalized(rgb),preprocess_frames([frame],[roi],input_size=336))

    def test_reject_non_pixel_input(self):
        with self.assertRaises(ValueError):
            module.normalized(np.zeros((1,336,336,3),np.float32))

    def test_errors_identical_and_shifted(self):
        a=np.arange(1,49).reshape(2,3,8)
        self.assertEqual(module.errors(a,a)['maximumAbsoluteError'],0.)
        self.assertAlmostEqual(module.errors(a,a+1)['meanAbsoluteError'],1.)

    def test_errors_reject_different_shapes_or_nonfinite(self):
        for b in (np.ones((2,4)),np.full((2,3),np.nan)):
            with self.assertRaises(ValueError):
                module.errors(np.ones((2,3)),b)


if __name__=='__main__':
    unittest.main()
