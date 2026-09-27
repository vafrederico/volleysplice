from __future__ import annotations

import importlib.util
import copy
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("pts_repair", Path(__file__).resolve().parents[2]/"scripts/prepare-neural-transfer-pts.py")
pts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pts)


class Capture:
    def __init__(self, times, corrupt=0.):
        self.times, self.ordinal, self.corrupt = times, -1, corrupt
    def grab(self):
        self.ordinal += 1
        return self.ordinal < len(self.times)
    def retrieve(self):
        return True, np.full((2,2,3), self.ordinal, np.uint8)
    def get(self, key):
        return (self.times[self.ordinal]+self.corrupt)*1000
    def set(self, key, value):
        raise AssertionError("random seek must never occur")


class PtsRepairTests(unittest.TestCase):
    def test_nearest_media_time_handles_rate_change_and_earlier_ties(self):
        times, indexes = pts.grid_and_selection(np.array([0.,.125,.375,.5,.75]), .8)
        np.testing.assert_array_equal(times, [0.,.25,.5,.75])
        np.testing.assert_array_equal(indexes, [0,1,3,4])
    def test_reject_nonmonotone_missing_origin_and_large_holes(self):
        for values in ([0.,0.,.25], [0.,.3,.2], [.05,.25], [0.,1.]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                pts.grid_and_selection(np.array(values), 1.)
    def test_sequential_ordinal_ignores_nominal_fps_and_never_seeks(self):
        frame_times = np.array([0.,.1,.2,.25,.3,.5,.7,.75])
        times, indexes = pts.grid_and_selection(frame_times, .8)
        result = list(pts.selected_frames(Capture(frame_times), indexes, frame_times, 0))
        self.assertEqual([r[0] for r in result], [0,3,5,7])
        self.assertEqual([int(r[1][0,0,0]) for r in result], [0,3,5,7])
        np.testing.assert_array_equal([r[2] for r in result], times)
    def test_reject_decoder_pts_mismatch_and_truncated_decode(self):
        frame_times = np.array([0.,.25,.5])
        with self.assertRaises(ValueError):
            list(pts.selected_frames(Capture(frame_times, .001), [0,1,2], frame_times, 0))
        with self.assertRaises(ValueError):
            list(pts.selected_frames(Capture(frame_times[:-1]), [0,1,2], frame_times, 0))
    def test_audio_positive_origin_pads_negative_origin_trims(self):
        a = np.array([1,2,3], np.float32)
        padded, shift = pts.align_audio(a, 100, .02)
        np.testing.assert_array_equal(padded, [0,0,1,2,3])
        self.assertEqual(shift, 2)
        trimmed, shift = pts.align_audio(a, 100, -.01)
        np.testing.assert_array_equal(trimmed, [2,3])
        self.assertEqual(shift, -1)
        with self.assertRaises(ValueError):
            pts.align_audio(a, 100, 2.)
    def test_grid_last_tick_strictly_before_media_duration(self):
        times, _ = pts.grid_and_selection(np.arange(61)/60, 1.)
        np.testing.assert_array_equal(times, [0.,.25,.5,.75])
    def test_retained_audit_requires_exact_eleven_cache_identities(self):
        base = {"records": [{"recordingId": str(i), "tier": "exact", "audiovisual": {"path": f"/{i}", "sha256": f"sha{i}"}} for i in range(11)]}
        audit = {"passed": True, "unchangedProxies": 11, "records": [{"recordingId": r["recordingId"], "AV": r["audiovisual"], "maximumAbsoluteClockErrorSeconds": 1e-6} for r in base["records"]]}
        pts.validate_retained_audit(audit, base)
        for field, value in (("AV", {"path":"/0","sha256":"wrong"}), ("maximumAbsoluteClockErrorSeconds", .2)):
            broken = copy.deepcopy(audit)
            broken["records"][0][field] = value
            with self.assertRaises(ValueError):
                pts.validate_retained_audit(broken, base)


if __name__ == "__main__":
    unittest.main()
