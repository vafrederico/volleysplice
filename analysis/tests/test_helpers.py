from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from analysis.cli import slugify
from analysis.court import estimate_court
from analysis.ffmpeg import proxy_backend


class HelperTests(unittest.TestCase):
    def test_slugify_handles_spaces_unicode_and_empty_names(self):
        self.assertEqual(slugify("Indoor Set #1"), "indoor-set-1")
        self.assertEqual(slugify("🏐"), "analysis")
        self.assertLessEqual(len(slugify("x" * 100)), 48)

    def test_blank_frame_uses_conservative_court_fallback(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        estimate = estimate_court(frame)
        self.assertEqual(estimate.source, "fallback-region")
        self.assertLess(estimate.confidence, 0.3)

    def test_long_stable_lines_produce_a_detected_region(self):
        frame = np.full((360, 640, 3), 40, dtype=np.uint8)
        for x in (80, 320, 560):
            cv2.line(frame, (x, 80), (x, 345), (245, 245, 245), 5)
        for y in (100, 220, 340):
            cv2.line(frame, (30, y), (610, y), (245, 245, 245), 5)
        estimate = estimate_court(frame)
        self.assertEqual(estimate.source, "detected-lines")
        self.assertGreaterEqual(len(estimate.lines), 3)

    def test_proxy_backend_defaults_to_software_and_honors_configuration(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(proxy_backend(), "software")
        with patch.dict("os.environ", {"VOLLEYCUT_PROXY_BACKEND": "jellyfin-vaapi"}, clear=True):
            self.assertEqual(proxy_backend(), "jellyfin-vaapi")


if __name__ == "__main__":
    unittest.main()
