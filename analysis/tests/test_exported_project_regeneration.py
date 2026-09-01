from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "regenerate-exported-project-inference.py"
SPEC = importlib.util.spec_from_file_location("exported_project_regeneration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExportedProjectRegenerationTests(unittest.TestCase):
    def test_eight_logical_cpus_never_enables_multiple_feature_workers(self) -> None:
        with patch.object(MODULE.os, "cpu_count", return_value=8):
            self.assertEqual(MODULE._feature_worker_layout(7, 0), (1, 8, 0))
            self.assertEqual(MODULE._feature_worker_layout(7, 6), (1, 8, 0))

    def test_large_host_reserves_two_cpus_and_uses_every_available_recording(self) -> None:
        with patch.object(MODULE.os, "cpu_count", return_value=24):
            self.assertEqual(MODULE._feature_worker_layout(7, 0), (7, 3, 2))

    def test_explicit_worker_limit_is_respected_on_a_large_host(self) -> None:
        with patch.object(MODULE.os, "cpu_count", return_value=24):
            self.assertEqual(MODULE._feature_worker_layout(20, 4), (4, 5, 2))

    def test_cpu_video_decoder_is_always_serial(self) -> None:
        with patch.object(MODULE.os, "cpu_count", return_value=24):
            self.assertEqual(
                MODULE._decoder_worker_layout(
                    20,
                    0,
                    MODULE.OPENCV_VIDEO_DECODER,
                ),
                (1, 24, 0),
            )
            self.assertEqual(
                MODULE._decoder_worker_layout(
                    7,
                    0,
                    MODULE.NVDEC_VIDEO_DECODER,
                ),
                (7, 3, 2),
            )


if __name__ == "__main__":
    unittest.main()
