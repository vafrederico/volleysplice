"""Public CI bootstrap must reject missing, corrupt, or mismatched inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from onnx import helper


spec = importlib.util.spec_from_file_location(
    "prepare_android_neural_assets", Path(__file__).with_name("prepare-android-neural-assets.py")
)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class PublicAndroidBootstrapTest(unittest.TestCase):
    def test_missing_truncated_and_same_length_corruption_fail_closed(self):
        expected = b"public fixture"
        entry = {"sizeBytes": len(expected), "sha256": hashlib.sha256(expected).hexdigest()}
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "encoder.onnx"
            with self.assertRaises(FileNotFoundError):
                bootstrap.checked_bytes(source, entry)
            for payload in (expected[:-1], b"X" + expected[1:]):
                source.write_bytes(payload)
                with self.assertRaisesRegex(ValueError, "Pinned neural asset differs"):
                    bootstrap.checked_bytes(source, entry)
            source.write_bytes(expected)
            self.assertEqual(bootstrap.checked_bytes(source, entry), expected)

    def test_unqualified_encoder_graph_cannot_be_rewritten(self):
        graph = helper.make_graph([], "unqualified", [], [])
        model = helper.make_model(graph)
        with self.assertRaisesRegex(ValueError, "Unexpected portable regional pooling graph"):
            bootstrap.native_encoder(model.SerializeToString())

    def test_checkout_output_is_rejected_before_writing(self):
        with self.assertRaisesRegex(ValueError, "outside the checkout"):
            bootstrap.prepare(bootstrap.REPOSITORY / "ci-generated-assets")

    def test_manifest_mismatch_never_publishes_a_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests, public = root / "manifests", root / "public"
            manifests.mkdir()
            public.mkdir()
            pinned = {"variants": {}}
            for platform in ("web", "android"):
                (manifests / f"{platform}-manifest.json").write_text(json.dumps(pinned))
            (public / "manifest.json").write_text(json.dumps({"variants": {}, "changed": True}))
            output = root / "output"
            with patch.object(bootstrap, "MANIFESTS", manifests), patch.object(bootstrap, "WEB_ASSETS", public):
                with self.assertRaisesRegex(ValueError, "model manifests differ"):
                    bootstrap.prepare(output)
            self.assertFalse((output / "android" / "rally-models").exists())


if __name__ == "__main__":
    unittest.main()
