"""Preparation rejection tests and read-only checks of the frozen expanded cohort.

NAS readiness tests skip when the immutable artifacts are unavailable. They never
train, alter labels, or reread full source videos.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import copy
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.config import FeatureConfig, NOISE_NORMALIZED_AUDIO_FEATURE_SET
from analysis.features import feature_cache_path, feature_names
from analysis.schema import load_manifest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/prepare-neural-expanded.py"
SPEC = importlib.util.spec_from_file_location("prepare_neural_expanded", SCRIPT)
PREP = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(PREP)
DEFAULT = Path(private_value('private-reference-0070'))
FROZEN = Path(os.environ.get("VOLLEYCUT_EXPANDED_DATASET", str(DEFAULT)))


class PreparationRejectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET)
        self.row = {"id": "synthetic", "video": str(self.root / "source.mp4"),
                    "contentSha256": "0" * 64, "durationSeconds": 1.,
                    "roi": {"x": 0, "y": 0, "width": 1, "height": 1}}
        self.cache = feature_cache_path("synthetic", self.row["video"], self.config,
                                        (0., 0., 1., 1.), self.root,
                                        content_sha256=self.row["contentSha256"])

    def write_cache(self, **overrides):
        values = dict(times=np.array([0., .25, .5, .75]),
                      values=np.zeros((4, 104), dtype=np.float32),
                      names=np.array(feature_names(self.config)),
                      metadata_json=np.array(json.dumps({"duration": 1.})))
        values.update(overrides)
        np.savez(self.cache, **values)

    def test_modified_frozen_input_is_rejected(self):
        p = self.root / "input.json"
        p.write_text("original")
        original = PREP.identity(p)
        p.write_text("modified")
        with self.assertRaisesRegex(ValueError, "Changed frozen input"):
            PREP.verify(original)

    def test_existing_manifest_is_not_overwritten(self):
        output = self.root / "output"
        output.mkdir()
        (output / "manifest.json").write_text("do not overwrite")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            PREP.prepare(self.root / "missing-data", output)
        self.assertEqual((output / "manifest.json").read_text(), "do not overwrite")

    def test_unpinned_parent_is_rejected_before_outputs(self):
        parent = self.root / "neural-experiments/2026-09-19-nonbeach"
        parent.mkdir(parents=True)
        (parent / "manifest.json").write_text("{}")
        output = self.root / "output"
        with self.assertRaisesRegex(ValueError, "Unexpected parent"):
            PREP.prepare(self.root, output)
        self.assertFalse(output.exists())

    def test_integer_roi_uses_loader_float_cache_identity(self):
        self.write_cache()
        result = PREP.av_cache(self.row, self.cache, self.config)
        self.assertEqual(result["shape"], [4, 104])
        self.assertEqual(result["metadata"]["duration"], 1.)

    def test_wrong_source_identity_cannot_reuse_cache(self):
        self.write_cache()
        changed = {**self.row, "contentSha256": "1" * 64}
        with self.assertRaisesRegex(ValueError, "Cache identity mismatch"):
            PREP.av_cache(changed, self.cache, self.config)

    def test_wrong_channel_order_is_rejected(self):
        self.write_cache(names=np.array(tuple(reversed(feature_names(self.config)))))
        with self.assertRaisesRegex(ValueError, "Wrong AV schema"):
            PREP.av_cache(self.row, self.cache, self.config)

    def test_nonfinite_values_are_rejected(self):
        values = np.zeros((4, 104), dtype=np.float32)
        values[1, 1] = np.nan
        self.write_cache(values=values)
        with self.assertRaisesRegex(ValueError, "Invalid AV"):
            PREP.av_cache(self.row, self.cache, self.config)

    def test_duplicate_timestamps_are_rejected(self):
        self.write_cache(times=np.array([0., .25, .25, .75]))
        with self.assertRaisesRegex(ValueError, "Invalid AV"):
            PREP.av_cache(self.row, self.cache, self.config)

    def test_other_decoder_is_rejected(self):
        self.write_cache(video_decoder=np.array("native-android-dsp-v1"))
        with self.assertRaisesRegex(ValueError, "AV media mismatch"):
            PREP.av_cache(self.row, self.cache, self.config)


@unittest.skipUnless((FROZEN / "manifest.json").is_file(), "Expanded NAS dataset unavailable")
class FrozenExpandedReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((FROZEN / "manifest.json").read_text())
        cls.exact = json.loads((FROZEN / "exact-manifest.json").read_text())
        cls.audit = json.loads((FROZEN / "dataset-audit.json").read_text())
        cls.rows = sum((cls.manifest[k] for k in ("exactRows", "draftRows", "coverageRows")), [])

    def test_artifact_hashes_and_standard_exact_schema(self):
        self.assertEqual(PREP.sha256(FROZEN / "manifest.json"), self.audit["manifestSha256"])
        self.assertEqual(PREP.sha256(FROZEN / "exact-manifest.json"), self.audit["exactManifestSha256"])
        self.assertEqual(self.exact["recordings"], self.manifest["exactRows"])
        parsed = load_manifest(FROZEN / "exact-manifest.json", require_videos=False)
        self.assertEqual(len(parsed.recordings), 8)
        self.assertEqual(sum(len(row.rallies) for row in parsed.recordings), 322)
        self.assertEqual(len({row.source_group for row in parsed.recordings}), 4)

    def test_all_tiers_have_authorization_and_no_protected_or_duplicate_lineage(self):
        supplement = json.loads(Path(self.manifest["cacheSupplement"]["path"]).read_text())
        self.assertEqual(PREP.sha256(Path(self.manifest["cacheSupplement"]["path"])),
                         self.manifest["cacheSupplement"]["sha256"])
        protected = set()
        for row in supplement["lineageVerification"]["protectedMetadataOnly"]:
            protected.update((row["sourceSha256"], row["proxySha256"]))
        seen, ids = set(), set()
        for row in self.rows:
            self.assertNotIn(row["id"], ids)
            ids.add(row["id"])
            self.assertIn(row["environment"], {"grass", "indoor"})
            self.assertNotEqual(row["sourceGroup"], PREP.PROTECTED)
            self.assertEqual(row["consent"], {"analyze": True, "train": True})
            self.assertTrue(row["consentProvenance"])
            hashes = {row["contentSha256"], row["sourceContentSha256"]}
            self.assertFalse(hashes & protected)
            self.assertFalse(hashes & seen)
            seen.update(hashes)
        self.assertEqual(len(ids), 18)

    def test_held_source_group_excludes_every_quality_tier(self):
        rows = {row["id"]: row for row in self.rows}
        exact_ids = {row["id"] for row in self.manifest["exactRows"]}
        for fold in self.manifest["foldPolicy"]:
            held = {rid for rid, row in rows.items() if row["sourceGroup"] == fold["outerHeldGroup"]}
            self.assertEqual(set(fold["forbiddenInFitOrSelectionIds"]), held)
            self.assertEqual(set(fold["availableOuterTrainingIds"]), set(rows) - held)
            self.assertEqual(set(fold["exactEvaluationIds"]), held & exact_ids)
        kb = next(f for f in self.manifest["foldPolicy"] if f["outerHeldGroup"] == private_value('source-group-005'))
        self.assertIn(private_value('grass-source-05'), kb["forbiddenInFitOrSelectionIds"])

    def test_draft_review_and_boundary_masks_are_explicit(self):
        for row in self.manifest["draftRows"]:
            self.assertTrue(row["annotation"]["continuousVideoReviewed"])
            self.assertIn(row["annotation"]["annotator"].lower(), {"v", "vini"})
            self.assertTrue(row["trainingOnly"])
            self.assertEqual(row["targetMasks"], {"live": True, "serve": False, "end": False, "keep": False})
        for row in self.manifest["exactRows"]:
            self.assertEqual(row["annotation"]["status"], "complete")
            self.assertFalse(row["trainingOnly"])

    def test_actual_export_targets_preserve_reference_without_repadding(self):
        self.assertEqual(sum(len(r["retainedCoverage"]) for r in self.manifest["coverageRows"]), 275)
        self.assertEqual(sum(len(r["keepTargets"]) for r in self.manifest["coverageRows"]), 252)
        for row in self.manifest["coverageRows"]:
            source = row["referenceSource"]
            PREP.verify(source)
            reference = json.loads(Path(source["path"]).read_text())["annotations"]
            self.assertEqual(row["keepTargets"], reference["finalExportIntervals"])
            self.assertEqual(row["ignoredIntervals"], reference["ignoredIntervals"])
            self.assertTrue(row["targetContract"]["alreadyPadded"])
            self.assertTrue(row["targetContract"]["negativesOutsideKeepAuthorizedByFullManualReview"])
            self.assertEqual(row["targetMasks"], {"live": False, "serve": False, "end": False, "keep": True})
            self.assertEqual(row["consentProvenance"]["userReviewConfirmation"], "All manually reviewed")

    def test_game_window_mask_changes_no_current_coverage_ticks(self):
        for row in self.manifest["coverageRows"]:
            window = row["gameWindow"]
            self.assertEqual(window["start"], 0)
            self.assertLess(abs(window["end"] - row["durationSeconds"]), .001)
            for interval in row["keepTargets"]:
                self.assertGreaterEqual(interval["start"], window["start"])
                self.assertLessEqual(interval["end"], window["end"])
            with np.load(row["featureCaches"]["audiovisual"]["path"], allow_pickle=False) as n:
                times = n["times"]
                whole = (times >= 0) & (times < row["durationSeconds"])
                game = whole & (times >= window["start"]) & (times < window["end"])
                np.testing.assert_array_equal(whole, game)

    def test_all_eighteen_current_cache_identities_and_hashes(self):
        config = FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET)
        for row in self.rows:
            cache = row["featureCaches"]["audiovisual"]
            PREP.verify(cache)
            checked = PREP.av_cache(row, Path(cache["path"]), config)
            self.assertEqual(checked["sha256"], cache["sha256"])
            self.assertEqual(checked["shape"], cache["shape"])


if __name__ == "__main__":
    unittest.main()
