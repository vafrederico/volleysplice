from __future__ import annotations
from analysis.private_ledger import private_value

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.dinov2_embeddings import DinoExtractorConfig

REPO = Path(__file__).resolve().parents[2]


def script(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), REPO/"scripts"/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = script("prepare-neural-dino-transfer.py")
audit = script("audit-neural-dino-transfer.py")


class DinoTransferPreparationTests(unittest.TestCase):
    def test_nearest_tie_selects_earlier_and_grid_uses_actual_av_times(self):
        times = np.array([0., .125, .25, .49], np.float64)
        indexes, report = prep.alignment(times, np.array([0., .25, .5], np.float64))
        np.testing.assert_array_equal(indexes, [0, 0, 1, 2])
        self.assertEqual(report["tieCount"], 1)
        self.assertEqual(report["avTimesSha256"], audit.array_sha(times, "<f8"))
        self.assertEqual(report["nearestIndexesSha256"], audit.array_sha(indexes, "<i8"))

    def test_alignment_rejects_out_of_tolerance_and_nonmonotone_times(self):
        for av_times in (np.array([.8]), np.array([.25, .25]), np.array([float("nan")])):
            with self.assertRaises(ValueError):
                prep.alignment(av_times, np.array([0., .25]))

    def test_only_explicit_nas_prefix_alias_is_canonicalized(self):
        self.assertEqual(audit.canonical_path(private_value('private-reference-0069')), private_value('private-reference-0068'))
        self.assertEqual(audit.canonical_path("/synthetic-other/one.mp4"), "/synthetic-other/one.mp4")

    def fixture(self, root, mutate=None):
        source = root/"source.mp4"
        source.write_bytes(b"synthetic association fixture, not decoded video")
        digest = audit.sha(source)
        av_path, dino_path, sidecar_path = root/"av.npz", root/"dino.npz", root/"metadata.json"
        names = [f"feature-{i}" for i in range(104)]
        times, timestamps = np.array([0., .24]), np.array([0., .25])
        np.savez_compressed(av_path, times=times, values=np.zeros((2, 104), np.float32), names=names)
        config = DinoExtractorConfig().to_dict()
        runtime = {"python": "pinned", "numpy": "pinned", "torch": "pinned", "device": "cuda"}
        md = {"recordingId": "fixture", "recordingContentSha256": digest, "sourceVideoPath": str(source),
              "roi": [0., 0., 1., 1.], "preprocessing": config, "extractorConfigSha256": audit.CONFIG_SHA,
              "backbone": {**audit.BACKBONE, "repository": "/old/repository", "checkpoint": "/old/checkpoint"},
              "runtime": runtime, "completed": True, "labelsUsed": False, "video": {"duration": .5},
              "analysisTimestamps": {"count": 2, "sha256": audit.array_sha(timestamps, "<f8")}}
        if mutate:
            mutate(md)
        np.savez_compressed(dino_path, timestamps=timestamps, tokens=np.zeros((2, 10, 384), np.float16), metadata_json=json.dumps(md))
        _, join = prep.alignment(times, timestamps)
        source_identity = {"path": str(source), "sha256": digest, "sizeBytes": source.stat().st_size,
                           "mtimeNs": source.stat().st_mtime_ns}
        sidecar = {"recordingId": "fixture", "cache": {"path": str(dino_path), "sha256": audit.sha(dino_path)},
                   "sourceVideoVerified": source_identity, "extractionPlan": {"sha256": audit.PLAN_SHA},
                   "cacheMetadata": md, "nearestAlignment": join}
        sidecar_path.write_text(json.dumps(sidecar))
        av = {"path": str(av_path), "sha256": audit.sha(av_path), "names": names}
        row = {"id": "fixture", "sourceGroup": "development", "video": str(source), "contentSha256": digest,
               "sourceContentSha256": digest, "roi": {"x": 0., "y": 0., "width": 1., "height": 1.},
               "featureCaches": {"audiovisual": av}}
        record = {"recordingId": "fixture", "sourceGroup": "development", "dinoPath": str(dino_path),
                  "dinoSha256": audit.sha(dino_path), "metadataPath": str(sidecar_path),
                  "metadataSha256": audit.sha(sidecar_path), "sourceVideoVerified": source_identity,
                  "rawSourceContentSha256": digest, "audiovisualPath": str(av_path), "audiovisualSha256": av["sha256"],
                  "nearestAlignment": join}
        return record, row, {"extractorConfig": config, "historicalRuntime": runtime}

    def test_correct_association_passes_independent_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            record, row, plan = self.fixture(Path(temporary))
            result = audit.validate_record(record, row, plan)
            self.assertTrue(result["passed"])
            self.assertEqual(result["shape"], [2, 10, 384])

    def test_coherently_rehashed_wrong_embedded_id_source_roi_or_backbone_fails(self):
        changes = [lambda md: md.update(recordingId="another-record"),
                   lambda md: md.update(recordingContentSha256="0"*64),
                   lambda md: md.update(sourceVideoPath="/different/source.mp4"),
                   lambda md: md.update(roi=[0., 0., .5, 1.]),
                   lambda md: md["backbone"].update(checkpointSha256="0"*64)]
        for mutate in changes:
            with tempfile.TemporaryDirectory() as temporary:
                record, row, plan = self.fixture(Path(temporary), mutate)
                with self.assertRaises(ValueError):
                    audit.validate_record(record, row, plan)

    def test_changed_recipe_or_label_usage_fails_even_when_sidecar_matches(self):
        for mutate in (lambda md: md.update(labelsUsed=True), lambda md: md["preprocessing"].update(inputSize=224)):
            with tempfile.TemporaryDirectory() as temporary:
                record, row, plan = self.fixture(Path(temporary), mutate)
                with self.assertRaises(ValueError):
                    audit.validate_record(record, row, plan)

    def test_semantically_identical_asset_path_relocation_is_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            record, row, plan = self.fixture(Path(temporary), lambda md: md["backbone"].update(repository="/new/repo", checkpoint="/new/weights"))
            self.assertTrue(audit.validate_record(record, row, plan)["passed"])

    def test_wrong_alignment_index_digest_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            record, row, plan = self.fixture(Path(temporary))
            changed = copy.deepcopy(record)
            changed["nearestAlignment"]["nearestIndexesSha256"] = "0"*64
            with self.assertRaisesRegex(ValueError, "alignment"):
                audit.validate_record(changed, row, plan)


if __name__ == "__main__":
    unittest.main()
