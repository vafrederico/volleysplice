from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class CliSmokeTests(unittest.TestCase):
    def test_module_smoke_command_runs_without_opencv(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]

        completed = subprocess.run(
            [sys.executable, "-m", "analysis", "smoke"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["passed"])
        self.assertTrue(payload["modelSaveLoadParity"])
        self.assertGreaterEqual(payload["eventRecall"], 0.95)
        self.assertGreaterEqual(payload["timeIoU"], 0.8)


if __name__ == "__main__":
    unittest.main()
