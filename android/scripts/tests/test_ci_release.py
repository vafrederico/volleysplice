"""Credential-free tests for the Android GitHub release helper."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "android_ci_release", Path(__file__).parents[1] / "ci-release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class MetadataTests(unittest.TestCase):
    def test_reads_literal_release_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            gradle = Path(directory) / "build.gradle.kts"
            gradle.write_text(
                'applicationId = "com.volleycut.nativeanalysis"\n'
                'versionCode = 42\nversionName = "1.2.3"\n'
                'versionNameSuffix = "-debug"\n',
                encoding="utf-8",
            )
            self.assertEqual(release.read_gradle_metadata(gradle), {
                "package_name": "com.volleycut.nativeanalysis",
                "version_code": 42,
                "version_name": "1.2.3",
            })

    def test_rejects_dynamic_or_wrong_metadata(self):
        cases = [
            'applicationId = "com.other"\nversionCode = 1\nversionName = "1.0.0"\n',
            'applicationId = "com.volleycut.nativeanalysis"\nversionCode = code\nversionName = "1.0.0"\n',
            'applicationId = "com.volleycut.nativeanalysis"\nversionCode = 1\nversionName = "1.0"\n',
        ]
        for source in cases:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                gradle = Path(directory) / "build.gradle.kts"
                gradle.write_text(source, encoding="utf-8")
                with self.assertRaises(ValueError):
                    release.read_gradle_metadata(gradle)

    def test_normalizes_certificate_fingerprint(self):
        fingerprint = ":".join(["AA"] * 32)
        self.assertEqual(release.normalize_sha256(fingerprint), "aa" * 32)
        with self.assertRaises(ValueError):
            release.normalize_sha256("AA:BB")


class ReleasePayloadTests(unittest.TestCase):
    def test_internal_release(self):
        payload = release.release_payload(42, "VolleySplice 1.2.3 (42)", "completed", "Fixes")
        self.assertEqual(payload["versionCodes"], ["42"])
        self.assertEqual(payload["releaseNotes"], [{"language": "en-US", "text": "Fixes"}])
        self.assertNotIn("userFraction", payload)

    def test_staged_production_release(self):
        payload = release.release_payload(
            42, "VolleySplice 1.2.3 (42)", "inProgress", user_fraction="0.10"
        )
        self.assertEqual(payload["userFraction"], 0.1)

    def test_rejects_invalid_rollout(self):
        for fraction in ["0", "1", "1.1", "not-a-number"]:
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                release.release_payload(42, "release", "inProgress", user_fraction=fraction)


class FakePublisherClient:
    def __init__(self):
        self.calls = []
        self.fail_update = False

    def create_edit(self, package):
        self.calls.append(("create", package))
        return "edit-1"

    def delete_edit(self, package, edit):
        self.calls.append(("delete", package, edit))

    def upload_bundle(self, package, edit, bundle):
        self.calls.append(("upload", package, edit, bundle.name))
        return 42

    def upload_deobfuscation_file(self, package, edit, version_code, file_type, file_path):
        self.calls.append(("symbols", package, edit, version_code, file_type, file_path.name))
        return {}

    def update_track(self, package, edit, track, payload):
        self.calls.append(("update", package, edit, track, payload))
        if self.fail_update:
            raise release.PublisherError("simulated update failure")
        return {}

    def commit_edit(self, package, edit):
        self.calls.append(("commit", package, edit))
        return {"id": edit}


class PublishingFlowTests(unittest.TestCase):
    def test_uploads_new_bundle_only_to_internal(self):
        client = FakePublisherClient()
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "release.aab"
            bundle.write_bytes(b"signed")
            mapping = Path(directory) / "mapping.txt"
            mapping.write_text("mapping", encoding="utf-8")
            result = release.publish_bundle(
                client, release.DEFAULT_PACKAGE_NAME, bundle, "internal", 42, "release",
                mapping_file=mapping,
            )
        self.assertEqual(result["track"], "internal")
        self.assertEqual(
            [call[0] for call in client.calls],
            ["create", "upload", "symbols", "update", "commit"],
        )
        self.assertEqual(client.calls[2][4], "proguard")

    def test_promotes_existing_internal_version(self):
        client = FakePublisherClient()
        result = release.promote_version(
            client, release.DEFAULT_PACKAGE_NAME, 42, "release", "0.10"
        )
        self.assertEqual(result["track"], "production")
        update = next(call for call in client.calls if call[0] == "update")
        self.assertEqual(update[3], "production")
        self.assertEqual(update[4]["versionCodes"], ["42"])

    def test_failed_edit_is_deleted(self):
        client = FakePublisherClient()
        client.fail_update = True
        with self.assertRaises(release.PublisherError):
            release.promote_version(
                client, release.DEFAULT_PACKAGE_NAME, 42, "release", "0.10"
            )
        self.assertEqual(client.calls[-1][0], "delete")


if __name__ == "__main__":
    unittest.main()
