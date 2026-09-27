from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analysis.mobile_visual_features import (MobileVisualError, align_mobile_features, normalize_roi,
                                             preprocess_frame, regional_pool_weights, sample_selection, selected_frames,
                                             load_mobile_visual_cache)


class Capture:
    def __init__(self, pts, corrupt=0.):
        self.pts, self.ordinal, self.corrupt = pts, -1, corrupt

    def grab(self):
        self.ordinal += 1
        return self.ordinal < len(self.pts)

    def retrieve(self):
        return True, np.full((2, 2, 3), self.ordinal, np.uint8)

    def get(self, _):
        return (self.pts[self.ordinal] + self.corrupt) * 1000

    def set(self, *_):
        raise AssertionError("random seek is forbidden")


class MobileVisualTests(unittest.TestCase):
    def test_native_roi_letterbox_and_bgr_normalization(self):
        frame = np.zeros((40, 80, 3), np.uint8)
        frame[:, 40:] = [10, 20, 30]
        values, box, quality = preprocess_frame(frame, {"x": .5, "y": 0, "width": .5, "height": 1}, 20)
        np.testing.assert_allclose(box, [0, 0, 1, 1])
        np.testing.assert_allclose(values[:, 10, 10], (np.array([30, 20, 10]) / 255 - [.485, .456, .406]) / [.229, .224, .225], atol=1e-6)
        self.assertEqual(quality[0], 1.)
        _, box, quality = preprocess_frame(frame, None, 20)
        np.testing.assert_allclose(box, [0, .25, 1, .75])
        self.assertEqual(quality[0], .5)

    def test_roi_rejects_outside_nan_and_empty(self):
        for roi in ((-.1, 0, 1, 1), (0, 0, 0, 1), (0, 0, 1.1, 1), (0, 0, float("nan"), 1)):
            with self.subTest(roi=roi), self.assertRaises(MobileVisualError):
                normalize_roi(roi)

    def test_pools_exclude_padding_and_resolve_near_far_order(self):
        weights = regional_pool_weights(np.array([[0, .25, 1, .75]]), 8, 4)
        np.testing.assert_allclose(weights.sum(axis=(2, 3)), 1.)
        self.assertTrue(np.all(weights[:, :, :2] == 0))
        self.assertTrue(np.all(weights[:, :, 6:] == 0))
        # Padding is intentionally huge: it must not leak into any pooled value.
        image = np.full((8, 4), 1000.)
        image[2:4], image[4:6] = 2., 6.
        pooled = np.einsum("brhw,hw->br", weights, image)
        np.testing.assert_allclose(pooled, [[4., 6., 2., 4.]])

    def test_thin_content_and_net_band_never_have_empty_pool(self):
        weights = regional_pool_weights(np.array([[0, .48, 1, .52]]), 7, 7)
        self.assertTrue(np.isfinite(weights).all())
        np.testing.assert_allclose(weights.sum(axis=(2, 3)), 1., atol=1e-6)

    def test_media_pts_sampling_handles_vfr_earlier_tie_and_duration(self):
        pts = np.array([0., .1, .25, .375, .625, .9, 1.])
        times, indexes = sample_selection(pts, 1.)
        np.testing.assert_array_equal(times, [0., .5])
        np.testing.assert_array_equal(indexes, [0, 3])
        frames = list(selected_frames(Capture(pts), indexes, pts, 0))
        self.assertEqual([int(frame[0, 0, 0]) for _, frame, _ in frames], [0, 3])

    def test_pts_rejects_holes_nonzero_origin_and_decoder_drift(self):
        for pts in ([.1, .5], [0., 0., .5], [0., 2.]):
            with self.assertRaises(MobileVisualError):
                sample_selection(np.array(pts), 2.)
        pts = np.array([0., .5, 1.])
        with self.assertRaises(MobileVisualError):
            list(selected_frames(Capture(pts, .001), np.array([0, 1]), pts, 0))
        with self.assertRaises(MobileVisualError):
            list(selected_frames(Capture(pts[:-1]), np.array([0, 1]), pts, 0))

    def test_hold_does_not_select_future_features_and_marks_stale_missing(self):
        cache = SimpleNamespace(timestamps=np.array([0., .5, 1.]),
                                tokens=np.arange(3)[:, None, None].astype(np.float32),
                                quality=np.ones((3, 6), np.float32))
        aligned = align_mobile_features(cache, np.array([-.1, 0., .25, .5, .75, 1., 1.6]))
        np.testing.assert_array_equal(aligned["source_indexes"], [-1, 0, 0, 1, 1, 2, -1])
        np.testing.assert_allclose(aligned["feature_age_seconds"], [0, 0, .25, 0, .25, 0, 0])
        self.assertTrue(np.all(aligned["quality"][[0, -1]] == 0))

    def test_partial_decode_stops_after_declared_inventory_without_whole_video_read(self):
        capture = Capture(np.array([0., .25, .5, .75, 1., 1.25]))
        frames = list(selected_frames(capture, np.array([0, 2]), np.array([0., .25, .5, .75]), 0, partial=True))
        self.assertEqual(len(frames), 2)
        self.assertEqual(capture.ordinal, 3)

    def test_cache_rejects_wrong_identity_nonfinite_and_inconsistent_pts(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "visual.npz"
            metadata = {"schemaVersion": 1, "completed": True, "labelsUsed": False, "identity": {"source": "fixed"}}
            arrays = {"timestamps": np.array([0., .5]), "tokens": np.zeros((2, 4, 576), np.float16),
                      "quality": np.zeros((2, 6), np.float32), "selected_presentation_times": np.array([0., .5]),
                      "metadata_json": np.asarray(json.dumps(metadata))}
            np.savez_compressed(path, **arrays)
            self.assertEqual(load_mobile_visual_cache(path, expected_identity={"source": "fixed"}).tokens.dtype, np.float32)
            with self.assertRaises(MobileVisualError):
                load_mobile_visual_cache(path, expected_identity={"source": "different"})
            arrays["selected_presentation_times"][1] = .6
            np.savez_compressed(path, **arrays)
            with self.assertRaises(MobileVisualError):
                load_mobile_visual_cache(path)
            arrays["selected_presentation_times"][1] = .5
            arrays["tokens"][0, 0, 0] = np.nan
            np.savez_compressed(path, **arrays)
            with self.assertRaises(MobileVisualError):
                load_mobile_visual_cache(path)


if __name__ == "__main__":
    unittest.main()
