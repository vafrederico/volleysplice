import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.private_ledger import private_value


class PrivateLedgerTests(unittest.TestCase):
    def test_missing_configuration_never_becomes_an_output_path(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "PRIVATE_LEDGER"):
                private_value("recording-001")

    def test_exact_runtime_identity_and_platform_override(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.json"
            ledger.write_text(json.dumps({"privateValues": {"recording-001": "synthetic-source"}}))
            platform = "WINDOWS" if os.name == "nt" else "POSIX"
            with patch.dict(os.environ, {f"VOLLEYCUT_PRIVATE_LEDGER_{platform}": str(ledger)}, clear=True):
                self.assertEqual(private_value("recording-001"), "synthetic-source")
                with self.assertRaisesRegex(ValueError, "recording-002"):
                    private_value("recording-002")


if __name__ == "__main__":
    unittest.main()
