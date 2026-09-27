import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "branch_privacy", Path(__file__).resolve().parents[2] / "scripts/audit-branch-privacy.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class BranchPrivacyTests(unittest.TestCase):
    def test_camera_and_mount_references_are_detected(self):
        self.assertIn("camera-filename", audit.categories("PXL" + "_20000101_000000000.mp4", set()))
        self.assertIn("private-mount", audit.categories("/" + "mnt/private-share/video.mp4", set()))

    def test_ledger_finds_names_without_camera_conventions(self):
        self.assertEqual(audit.categories("Example private title", {"Example private title"}), ["ledger-identity"])

    def test_credentials_and_network_addresses_are_detected(self):
        self.assertIn("credential", audit.categories("ghp_" + "A" * 36, set()))
        self.assertIn("lan-address", audit.categories("http://" + ".".join(["192", "168", "1", "2"]), set()))

    def test_dependency_version_is_not_a_network_address(self):
        self.assertEqual(audit.categories("example==" + ".".join(["10", "3", "9", "90"]), set()), [])
        self.assertEqual(audit.categories("other==1.2\nexample==" + ".".join(["10", "3", "9", "90"]) + "\n", set()), [])

    def test_public_indexes_are_safe(self):
        self.assertEqual(audit.categories("recording-001 private-reference-0001", set()), [])


if __name__ == "__main__":
    unittest.main()
