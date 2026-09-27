"""The release exporter must fail closed and strip research-only metadata."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/prepare-production-neural-assets.py"
spec = importlib.util.spec_from_file_location("production_neural_assets", SCRIPT)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class ProductionNeuralAssetsTest(unittest.TestCase):
    def config(self):
        return dict(
            mean=[0.] * 112, scale=[1.] * 112,
            decoder=dict(smoothing=.5, enter=.2, minimum=1., boundary=True),
            weightsSha256="a" * 64, encoderWeightsSha256="b" * 64,
            epoch=15, family="mobile-large", modelIdentity="distilled-large",
            selectionMode="recall", draw=3407, recallTargetPercent=99,
            tokenDimension=3840, config=dict(head="tcn"),
            trainingProjectorIncluded=False, dinoRequiredForInference=False,
            researchSource="private source must not be copied", dynamicParity=[dict(ticks=123)],
        )

    def test_exports_only_inference_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.json"
            source.write_text(json.dumps(self.config()), encoding="utf8")
            result = json.loads(assets.inference_config(source, "recall"))
            self.assertNotIn("researchSource", result)
            self.assertNotIn("dynamicParity", result)
            self.assertEqual(result["mean"], [0.] * 112)
            self.assertEqual(result["decoder"]["enter"], .2)
            with self.assertRaisesRegex(ValueError, "identity"):
                assets.inference_config(source, "f1")

    def test_invalid_normalizer_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.json"
            for value in (0., -1., float("nan"), float("inf")):
                config = self.config()
                config["scale"][8] = value
                source.write_text(json.dumps(config), encoding="utf8")
                with self.subTest(value=value), self.assertRaises(ValueError):
                    assets.inference_config(source, "recall")

    def test_changed_encoder_is_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repository = root / "checkout"
            manifests = repository / "models/distilled-large"
            manifests.mkdir(parents=True)
            original = root / "private/graphs-recall"
            original.mkdir(parents=True)
            config = self.config()
            (original / "mobile-large-pipeline.json").write_text(json.dumps(config), encoding="utf8")
            (original / "mobile-large-encoder-fp32.onnx").write_bytes(b"changed encoder")
            (original / "mobile-large-tcn-dynamic-fp32.onnx").write_bytes(b"head")
            entry = dict(name="encoder.onnx", sha256=assets.digest(b"frozen encoder"), sizeBytes=len(b"frozen encoder"))
            manifest = dict(schemaVersion=1, defaultVariant="high-recall", variants={
                "high-recall": dict(directory="recall", files=dict(encoder=entry)),
            })
            (manifests / "android-manifest.json").write_text(json.dumps(manifest), encoding="utf8")
            with patch.object(assets, "REPOSITORY", repository), self.assertRaisesRegex(ValueError, "identity"):
                assets.prepare(root / "private", root / "output")
            self.assertFalse((root / "output/android/rally-models/manifest.json").exists())

    def test_output_inside_repository_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "external output"):
            assets.prepare(Path("unused"), assets.REPOSITORY / "generated")


if __name__ == "__main__":
    unittest.main()
