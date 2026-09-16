"""Credential-free checks for the release helper's signing preflight."""
import datetime
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("ci_release", Path(__file__).parents[1] / "ci-release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseValidationTests(unittest.TestCase):
    def test_base64_from_powershell_or_clipboard(self):
        with patch.dict(os.environ, {"TEST_RELEASE_SECRET": "aGVs\r\nbG8=\n"}):
            self.assertEqual(release.decode_secret("TEST_RELEASE_SECRET"), b"hello")

    def profile(self):
        return {
            "TeamIdentifier": ["ABCDE12345"],
            "ApplicationIdentifierPrefix": ["OLDPREFIX1"],
            "Entitlements": {"application-identifier": "OLDPREFIX1." + release.BUNDLE_ID},
            "ExpirationDate": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1),
            "UUID": "12345678-1234-1234-1234-123456789abc",
        }

    def test_valid_store_profile_with_legacy_app_prefix(self):
        release.validate_profile(self.profile(), "ABCDE12345")

    def test_wrong_team_app_and_expired_profile(self):
        for field, value in [
            ("TeamIdentifier", ["OTHER12345"]),
            ("Entitlements", {"application-identifier": "OLDPREFIX1.com.other.app"}),
            ("ExpirationDate", datetime.datetime(2020, 1, 1)),
            ("UUID", "../../unexpected"),
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                profile = self.profile()
                profile[field] = value
                release.validate_profile(profile, "ABCDE12345")

    def test_non_store_profiles(self):
        for extra in [{"ProvisionedDevices": ["device"]}, {"ProvisionsAllDevices": True},
                      {"Entitlements": {"application-identifier": "OLDPREFIX1." + release.BUNDLE_ID,
                                        "get-task-allow": True}}]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                release.validate_profile(self.profile() | extra, "ABCDE12345")

    def test_version_input(self):
        for version, build, valid in [("1.0.0", "1", True), ("1.2.3", "9999", True),
                                      ("1.0", "1", False), ("1.0.0", "0", False),
                                      ("1.0.0", "10000", False), ("1.0.0", "1\nEVIL=1", False)]:
            with self.subTest(version=version, build=build), patch.dict(os.environ, {
                "RELEASE_VERSION": version, "RELEASE_BUILD_NUMBER": build,
            }):
                if valid:
                    release.validate()
                else:
                    with self.assertRaises(ValueError):
                        release.validate()


if __name__ == "__main__":
    unittest.main()
