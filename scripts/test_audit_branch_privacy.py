"""Check the narrow public-license exception without storing private test values."""
import hashlib
from pathlib import Path
import re
import runpy
import unittest
from urllib.parse import urlsplit


AUDIT = runpy.run_path(str(Path(__file__).with_name("audit-branch-privacy.py")))
CATEGORIES = AUDIT["categories"]
SOURCE_PATH = "android/app/src/main/java/com/volleycut/nativeanalysis/EditorActivity.kt"


class PublicLegalUrlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (Path(__file__).resolve().parents[1] / SOURCE_PATH).read_text(encoding="utf8")
        cls.urls = [re.search(r"private const val " + name + r'\s*=\s*"([^"]+)"', source).group(1)
                    for name in ("PROJECT_LICENSE_URL", "THIRD_PARTY_NOTICES_URL")]
        cls.owner = urlsplit(cls.urls[0]).path.split("/")[1]

    def test_only_the_established_two_links_are_approved(self):
        self.assertEqual({hashlib.sha256(url.encode()).hexdigest() for url in self.urls},
                         AUDIT["PUBLIC_LEGAL_URL_SHA256"][SOURCE_PATH])
        for url in self.urls:
            self.assertEqual([], CATEGORIES('"' + url + '"', {self.owner}, SOURCE_PATH))
        blob = "\n".join('"' + url + '"' for url in self.urls)
        self.assertEqual([], CATEGORIES(blob, {self.owner}, SOURCE_PATH))

    def test_altered_urls_other_files_and_commit_messages_stay_blocked(self):
        for url in self.urls:
            for suffix in ("/private", "?recording=1", "#private"):
                self.assertIn("ledger-identity", CATEGORIES(url + suffix, {self.owner}, SOURCE_PATH))
            self.assertIn("ledger-identity", CATEGORIES(url, {self.owner}, "other.kt"))
            self.assertIn("ledger-identity", CATEGORIES(url, {self.owner}))

    def test_other_occurrences_of_the_owner_are_still_private(self):
        line = '"' + self.urls[0] + '" // ' + self.owner
        self.assertIn("ledger-identity", CATEGORIES(line, {self.owner}, SOURCE_PATH))
        self.assertIn("ledger-identity", CATEGORIES(self.owner + "/report.md", {self.owner}))

    def test_generic_detectors_and_other_ledger_values_remain_active(self):
        probes = {
            "private-drive": chr(90) + ":/research/report.json",
            "private-mount": "/".join(("", "mnt", "dataset", "report.json")),
            "user-home": "/".join(("", "home", "tester", "report.json")),
            "lan-address": ".".join(("192", "168", "1", "20")),
            "camera-filename": "_".join(("PXL", "202001011234")) + ".mp4",
            "credential": "ghp" + "_" + "a" * 32,
        }
        for expected, private_value in probes.items():
            line = '"' + self.urls[0] + '" ' + private_value
            self.assertIn(expected, CATEGORIES(line, {self.owner}, SOURCE_PATH))
            self.assertIn("ledger-identity", CATEGORIES(line, {self.owner, private_value}, SOURCE_PATH))

    def test_generic_detector_still_sees_an_approved_token(self):
        # Prove URL filtering applies to ledger matches only, without changing
        # the production detector table or creating a generic allowlist.
        patterns = AUDIT["PATTERNS"]
        patterns["test-public-token"] = re.compile(re.escape(self.urls[0]))
        try:
            self.assertIn("test-public-token", CATEGORIES(self.urls[0], {self.owner}, SOURCE_PATH))
        finally:
            del patterns["test-public-token"]


if __name__ == "__main__":
    unittest.main()
