from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis.mobile_visual_features import QUALITY_NAMES
from analysis.neural_recognition_inputs import PLAYER_FEATURE_NAMES, PLAYER_FEATURE_VERSION, attach_features
from analysis.recognition_temporal_model import RecognitionConfig


@dataclass
class Example:
    id: str
    group: str
    duration: float
    times: np.ndarray
    values: np.ndarray
    targets: np.ndarray
    valid: np.ndarray
    truth: tuple
    ignored: tuple
    environment: str


@dataclass
class Supervised:
    example: Example
    mask: np.ndarray
    tier: str


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class Fixture:
    def __init__(self, root, family):
        self.root, self.family = root, family
        self.source = root / "source.json"
        self.clean = root / "extraction.json"
        self.index = root / "features.json"
        self.data = {"exact": [], "draft": [], "coverage": []}
        self.manifest = {"protectedSourceGroups": ["protected"], "exactRows": [], "draftRows": [], "coverageRows": []}
        sanitized = []
        self.times = np.array([0., .25, .5, .75], np.float64)
        self.configuration = {"sampleFps": 2, "inputSize": 224}
        self.backbone = {"modelName": "mobilenet_v3_small", "checkpointSha256": "f" * 64}
        for tier, count in (("exact", 8), ("draft", 3), ("coverage", 7)):
            for i in range(count):
                identifier = f"{tier}-{i}"
                av_path = root / f"{identifier}-av.npz"
                values = np.full((4, 104), i, np.float32)
                np.savez_compressed(av_path, times=self.times, values=values)
                source = {"id": identifier, "sourceGroup": f"group-{i % 4}", "split": "train",
                          "environment": "indoor", "video": str(root / f"{identifier}.mp4"),
                          "contentSha256": hashlib.sha256(identifier.encode()).hexdigest(), "durationSeconds": 1.,
                          "roi": {"x": 0., "y": 0., "width": 1., "height": 1.},
                          "featureCaches": {"audiovisual": {"path": str(av_path), "sha256": sha(av_path)}}}
                self.manifest[tier + "Rows"].append(source)
                sanitized.append({key: value for key, value in source.items() if key != "featureCaches"}
                                 | {"avCache": source["featureCaches"]["audiovisual"]})
                example = Example(identifier, source["sourceGroup"], 1., self.times.copy(), values.copy(),
                                  np.full((4, 4), .3, np.float32), np.array([True, False, True, True]),
                                  ((.1, .6),), ((.25, .4),), "indoor")
                self.data[tier].append(Supervised(example, np.full((4, 4), .7, np.float32), tier))
        write(self.source, self.manifest)
        write(self.clean, {"schemaVersion": 1, "kind": "label-free-recognition-extraction-input-v1",
                           "sourceManifest": {"path": str(self.source), "sha256": sha(self.source)},
                           "labelsUsed": False, "protectedTestOpened": False, "beachIncluded": False,
                           "records": sanitized})
        self.clean_ref = {"path": str(self.clean), "sha256": sha(self.clean)}
        entries = []
        for clean in sanitized:
            path = root / f"{clean['id']}-features.npz"
            if family == "mobile":
                metadata = {"schemaVersion": 1, "labelsUsed": False, "completed": True, "partialVideo": False,
                    "recordingId": clean["id"], "recordingContentSha256": clean["contentSha256"],
                    "sourceVideoPath": clean["video"], "extractedDurationSeconds": 1., "video": {"duration": 1.},
                    "qualityNames": list(QUALITY_NAMES), "identity": {
                        "recordingId": clean["id"], "recordingContentSha256": clean["contentSha256"],
                        "roi": [0., 0., 1., 1.], "maximumExtractionSeconds": None,
                        "config": self.configuration, "backbone": self.backbone,
                        "extractorSourceSha256": sha(Path(__file__).resolve().parents[1] / "mobile_visual_features.py"),
                        "opencv": "fixed", "opencvBuildSha256": "b" * 64}}
                np.savez_compressed(path, timestamps=np.array([0., .5]), tokens=np.ones((2, 4, 576), np.float16),
                                    quality=np.zeros((2, 6), np.float32), selected_presentation_times=np.array([0., .5]),
                                    metadata_json=np.asarray(json.dumps(metadata)))
                entries.append({"id": clean["id"], "recordingId": clean["id"], "cachePath": str(path),
                                "cacheSha256": sha(path), "sourceGroup": clean["sourceGroup"],
                                "split": clean["split"], "recordingContentSha256": clean["contentSha256"]})
            else:
                metadata = {"schemaVersion": 1, "featureVersion": PLAYER_FEATURE_VERSION,
                    "recordingId": clean["id"], "sourceGroup": clean["sourceGroup"], "environment": "indoor",
                    "labelsUsed": False, "labelIndependent": True, "trainingEligible": True, "fullRecording": True,
                    "pilotMaximumSeconds": None, "manifest": self.clean_ref,
                    "sourceVideo": {"path": clean["video"], "sha256": clean["contentSha256"]},
                    "avCache": clean["avCache"], "roi": [0., 0., 1., 1.],
                    "alignment": {"avTimesExact": True, "timesSha256": hashlib.sha256(self.times.tobytes()).hexdigest()},
                    "media": {"stream": {"duration": 1.}}, "config": {"detector_fps": 2},
                    "detector": {"sha256": "f" * 64}, "detectorSelectionOverride": {"maximumDetections": 24},
                    "decoder": "fixed", "sourceCode": {"extractor.py": "e" * 64}, "runtime": {"opencv": "fixed"}}
                np.savez_compressed(path, times=self.times, values=np.ones((4, 49), np.float32),
                                    names=np.asarray(PLAYER_FEATURE_NAMES), metadata_json=np.asarray(json.dumps(metadata)))
                entries.append({"id": clean["id"], "path": str(path), "sha256": sha(path), "trainingEligible": True})
        if family == "mobile":
            index = {"schemaVersion": 1, "manifest": str(self.clean), "manifestSha256": sha(self.clean),
                     "config": self.configuration, "backbone": self.backbone, "records": entries}
        else:
            index = {"schemaVersion": 1, "manifest": self.clean_ref, "featureVersion": PLAYER_FEATURE_VERSION,
                     "featureNames": list(PLAYER_FEATURE_NAMES), "featureDimension": 49, "records": entries}
        helper = Path(__file__).resolve().parents[1] / "mobile_visual_features.py"
        archive = root / "mobile_visual_features.snapshot.py"
        archive.write_bytes(helper.read_bytes())
        index["runtimeDependencies"] = [{"path": str(helper), "sha256": sha(helper), "archivePath": str(archive)}]
        write(self.index, index)
        self.config = RecognitionConfig(family=family, head="tcn", scalar_dimension=8 if family == "mobile" else 49)

    def attach(self):
        return attach_features(self.data, self.index, self.config, self.source)

    def mutate_cache(self, transform, row=0):
        index = json.loads(self.index.read_text())
        entry = index["records"][row]
        path = Path(entry.get("cachePath", entry.get("path")))
        with np.load(path, allow_pickle=False) as cache:
            arrays = {key: cache[key].copy() for key in cache.files}
        metadata = json.loads(str(arrays["metadata_json"].item()))
        transform(arrays, metadata)
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(path, **arrays)
        entry["cacheSha256" if self.family == "mobile" else "sha256"] = sha(path)
        write(self.index, index)


class RecognitionInputsTests(unittest.TestCase):
    def test_both_families_copy_values_and_preserve_every_supervision_object(self):
        for family, dimension in (("player", 153), ("mobile", 2416)):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder), family)
                result = fixture.attach()
                self.assertIsNot(result, fixture.data)
                for tier in fixture.data:
                    self.assertIsNot(result[tier], fixture.data[tier])
                    for original, attached in zip(fixture.data[tier], result[tier]):
                        self.assertIsNot(original, attached)
                        self.assertIsNot(original.example, attached.example)
                        self.assertEqual(attached.example.values.shape, (4, dimension))
                        np.testing.assert_array_equal(attached.example.values[:, :104], original.example.values)
                        self.assertFalse(np.shares_memory(attached.example.values, original.example.values))
                        self.assertEqual(original.example.values.shape, (4, 104))
                        for name in ("targets", "valid", "truth", "ignored", "times"):
                            self.assertIs(getattr(attached.example, name), getattr(original.example, name))
                        self.assertIs(attached.mask, original.mask)
                        self.assertEqual(attached.example.group, original.example.group)
                        self.assertEqual(attached.tier, original.tier)
                if family == "mobile":
                    np.testing.assert_array_equal(result["exact"][0].example.values[:, -2], [0., .25, 0., .25])
                    np.testing.assert_array_equal(result["exact"][0].example.values[:, -1], 1.)

    def test_nonfinite_cache_values_are_rejected_even_with_updated_artifact_hash(self):
        for family in ("player", "mobile"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder), family)
                fixture.mutate_cache(lambda arrays, _: arrays["tokens" if family == "mobile" else "values"].flat.__setitem__(0, np.nan))
                with self.assertRaises((ValueError, RuntimeError)):
                    fixture.attach()

    def test_partial_cache_is_never_accepted_as_full_training_features(self):
        for family in ("player", "mobile"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder), family)
                fixture.mutate_cache(lambda _, md: md.update({"partialVideo": True} if family == "mobile" else {"trainingEligible": False}))
                with self.assertRaisesRegex(ValueError, "partial"):
                    fixture.attach()

    def test_missing_record_and_missing_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder), "mobile")
            index = json.loads(fixture.index.read_text())
            index["records"].pop()
            write(fixture.index, index)
            with self.assertRaisesRegex(ValueError, "18"):
                fixture.attach()
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder), "player")
            Path(json.loads(fixture.index.read_text())["records"][0]["path"]).unlink()
            with self.assertRaises(FileNotFoundError):
                fixture.attach()

    def test_source_manifest_lineage_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder), "mobile")
            clean = json.loads(fixture.clean.read_text())
            clean["sourceManifest"]["sha256"] = "0" * 64
            write(fixture.clean, clean)
            index = json.loads(fixture.index.read_text())
            index["manifestSha256"] = sha(fixture.clean)
            write(fixture.index, index)
            with self.assertRaisesRegex(ValueError, "lineage"):
                fixture.attach()

    def test_player_timestamp_shift_and_cache_source_identity_mismatch_are_rejected(self):
        for mutation in (
            lambda arrays, _: arrays["times"].__iadd__(.001),
            lambda _, md: md["sourceVideo"].update(sha256="0" * 64),
            lambda _, md: md["config"].update(detector_fps=4),
        ):
            with tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder), "player")
                fixture.mutate_cache(mutation, row=1)
                with self.assertRaises(ValueError):
                    fixture.attach()

    def test_mobile_model_and_source_content_mismatch_are_rejected(self):
        for mutation in (
            lambda _, md: md["identity"]["backbone"].update(checkpointSha256="0" * 64),
            lambda _, md: md["identity"].update(recordingContentSha256="0" * 64),
            lambda _, md: md["identity"].update(extractorSourceSha256="0" * 64),
        ):
            with tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder), "mobile")
                fixture.mutate_cache(mutation, row=1)
                with self.assertRaises(ValueError):
                    fixture.attach()

    def test_group_mismatch_and_preaugmented_inputs_are_rejected_without_mutation(self):
        for family in ("player", "mobile"):
            with tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder), family)
                fixture.data["exact"][0].example.group = "wrong-group"
                with self.assertRaisesRegex(ValueError, "group"):
                    fixture.attach()
                self.assertTrue(all(row.example.values.shape == (4, 104) for rows in fixture.data.values() for row in rows))

    def test_unbound_or_changed_runtime_helper_archive_is_rejected(self):
        for mode in ("missing", "archive", "current-hash"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                fixture = Fixture(Path(folder), "mobile")
                index = json.loads(fixture.index.read_text())
                if mode == "missing":
                    del index["runtimeDependencies"]
                elif mode == "archive":
                    Path(index["runtimeDependencies"][0]["archivePath"]).write_text("changed", encoding="utf-8")
                else:
                    index["runtimeDependencies"][0]["sha256"] = "0" * 64
                write(fixture.index, index)
                with self.assertRaises(ValueError):
                    fixture.attach()


if __name__ == "__main__":
    unittest.main()
