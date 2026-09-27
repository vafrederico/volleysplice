from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis import neural_expanded_development as frozen
from analysis import neural_short_boost_transfer as study
from analysis import neural_short_boost_weighting as weighting
from analysis.tests.test_neural_expanded_development import example


def population(kind):
    def one(identifier, group, length, seed):
        e = example(identifier, group, length, seed)
        if kind == "dino_tcn":
            tokens = np.random.default_rng(seed+19).normal(size=(length, 3840)).astype(np.float32)
            e = replace(e, values=np.concatenate((e.values, tokens), axis=1))
        return e
    train = [frozen.exact_supervision(one("train", "fit", 137, 11)),
             frozen.exact_supervision(one("train2", "fit2", 149, 23))]
    auxiliary = {"draft": [frozen.auxiliary_supervision(one("draft", "draftgroup", 131, 13), "draft")],
                 "coverage": [frozen.auxiliary_supervision(one("keep", "keepgroup", 139, 17), "coverage")]}
    return train, auxiliary, [one("held", "validation", 133, 19)]


class TransferRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.threads = torch.get_num_threads()
        cls.deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.threads)
        torch.use_deterministic_algorithms(cls.deterministic)

    def test_compact_baseline_fitter_is_bit_exact_to_reference(self):
        train, aux, valid = population("tcn")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen.fit_model(train, aux, valid, "tcn", 71, (1, 2), root/"old", "cpu", "c")
            study.fit_model(train, aux, valid, "tcn", 71, (1, 2), root/"new", "cpu", "c", "baseline")
            old, new = [study.read(root/p/"completed.json") for p in ("old", "new")]
            for key in ("history", "optimizerSteps", "exposureSha256", "positiveWeight", "supervisedCounts", "parameters"):
                self.assertEqual(old[key], new[key], key)
            for stem in ("weights", "predictions"):
                for epoch in (1, 2):
                    with np.load(root/"old"/f"{stem}-{epoch}.npz") as a, np.load(root/"new"/f"{stem}-{epoch}.npz") as b:
                        self.assertEqual(a.files, b.files)
                        for key in a.files:
                            self.assertEqual(a[key].tobytes(), b[key].tobytes(), key)

    def test_both_representations_pair_rng_masks_and_sampling_across_arms(self):
        import torch
        original_loss = weighting.batch_loss
        for kind in study.KINDS:
            train, aux, valid = population(kind)
            calls, histories, predictions = {}, {}, {}
            with tempfile.TemporaryDirectory() as temporary:
                for arm in study.ARMS:
                    calls[arm] = []
                    def observe(model, chunks, indexes, pos_weight, head_weights, device):
                        call = (tuple(indexes), tuple(head_weights), torch.get_rng_state().numpy().tobytes(),
                                tuple(chunks[i][j].tobytes() for i in indexes for j in (0, 1, 2)),
                                tuple(chunks[i][3] for i in indexes))
                        result = original_loss(model, chunks, indexes, pos_weight, head_weights, device)
                        calls[arm].append((*call, torch.get_rng_state().numpy().tobytes()))
                        return result
                    location = Path(temporary)/arm
                    with patch.object(weighting, "batch_loss", side_effect=observe):
                        predictions[arm] = study.fit_model(train, aux, valid, kind, 73, (1, 2), location, "cpu", "c", arm)
                    histories[arm] = study.read(location/"completed.json")
                for arm in study.ARMS[1:]:
                    self.assertEqual(calls["baseline"], calls[arm])
                    for key in ("positiveWeight", "supervisedCounts", "parameters", "optimizerSteps", "exposureSha256"):
                        self.assertEqual(histories["baseline"][key], histories[arm][key])
                    self.assertNotEqual(histories["baseline"]["history"][0]["loss"], histories[arm]["history"][0]["loss"])
                    self.assertFalse(np.array_equal(predictions["baseline"][2]["held"], predictions[arm][2]["held"]))

    def test_resume_binds_architecture_arm_masks_and_artifacts(self):
        for kind in study.KINDS:
            train, aux, valid = population(kind)
            with tempfile.TemporaryDirectory() as temporary:
                args = [train, aux, valid, kind, 79, (1,), Path(temporary)/"fit", "cpu", "c"]
                original = study.fit_model(*args, weighting="baseline")
                with patch.object(weighting, "make_weighted_chunks", side_effect=AssertionError("retrained")):
                    resumed = study.fit_model(*args, weighting="baseline")
                np.testing.assert_array_equal(original[1]["held"], resumed[1]["held"])
                for arm in ("global_control", "short_boost"):
                    with self.assertRaises(ValueError):
                        study.fit_model(*args, weighting=arm)
                changed = copy.deepcopy(train)
                changed[0].mask[np.flatnonzero(changed[0].example.targets[:, 0])[0], 0] = 0
                with self.assertRaises(ValueError):
                    study.fit_model(changed, *args[1:], weighting="baseline")
                artifact = args[6]/"weights-1.npz"
                artifact.write_bytes(artifact.read_bytes()+b"tampered")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    study.fit_model(*args, weighting="baseline")

    def test_all_arms_use_same_nested_fold_exclusions(self):
        groups = ["a", "b", "c", "d"]
        data = {"exact": [frozen.exact_supervision(example(f"exact-{g}", g)) for g in groups]}
        for tier in ("draft", "coverage"):
            data[tier] = [frozen.auxiliary_supervision(example(f"{tier}-{g}", g), tier) for g in (*groups, "aux")]
        for kind in study.KINDS:
            for arm in study.ARMS:
                calls = []
                def fit(train, auxiliary, validation, fitted_kind, seed, epochs, destination, device, contract, mode):
                    outer = groups[int(destination.parent.name.removeprefix("outer-"))]
                    held = {e.group for e in validation}
                    excluded = held | {outer}
                    self.assertEqual({r.example.group for r in train}, set(groups)-excluded)
                    for tier, rows in auxiliary.items():
                        self.assertEqual([r.example.id for r in rows],
                                         [r.example.id for r in data[tier] if r.example.group not in excluded])
                    self.assertEqual((fitted_kind, mode), (kind, arm))
                    calls.append(destination)
                    return {ep: {e.id: np.zeros((len(e.times), 4), np.float32) for e in validation} for ep in epochs}
                def choose(examples, predictions):
                    self.assertEqual(len(examples), 3)
                    for values in predictions.values():
                        self.assertEqual(set(values), {e.id for e in examples})
                    return {"epoch": 15, "decoder": frozen.decoder_candidates()[0], "innerF1_padP_coreR": .8,
                            "innerR_core": .97, "recallEligibilityPassed": True, "recallEligibilityFloor": .95}
                with tempfile.TemporaryDirectory() as temporary:
                    with patch.object(study, "fit_model", side_effect=fit), \
                            patch.object(frozen, "choose_settings", side_effect=choose), \
                            patch.object(study.base, "decode", return_value=[]):
                        study.run_job(data, {"groups": groups}, "c", "reviewed_export", kind, arm,
                                      study.base.SEEDS[0], Path(temporary), "cpu")
                    self.assertEqual(len(set(calls)), 16)

    def test_reused_baseline_preserves_result_and_points_at_original_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = {"cohort": "exact", "kind": "tcn", "seed": 3407, "selections": [{"epoch": 15}],
                   "evaluation": {"objective": .7}, "predictions": [{"id": "original"}]}
            (root/"report.json").write_text(json.dumps({"results": [old]}))
            contract = {"referenceStudy": {"path": str(root), "reportSha256": "report", "contractSha256": "old"}}
            with patch.object(study, "fit_model", side_effect=AssertionError("retrained baseline")):
                study.run_job({}, contract, "new", "exact", "tcn", "baseline", 3407, root/"new", "cpu")
            result = study.read(study.result_path(root/"new", "exact", "tcn", "baseline", 3407))
            for key, value in old.items():
                self.assertEqual(result[key], value)
            self.assertEqual(result["origin"]["type"], "reused-reference")
            self.assertEqual(result["origin"]["fitRoot"], str(root/"fits/exact/tcn/3407"))
            self.assertFalse((root/"new/fits").exists())


class TransferInputTests(unittest.TestCase):
    def test_pts_revision_only_changes_coverage_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = {"exactRows": [{"id": "exact", "featureCaches": {"audiovisual": {"sha256": "exact"}}}],
                   "draftRows": [], "coverageRows": [
                       {"id": f"pixel-{i}", "featureCaches": {"audiovisual": {"sha256": "old"}},
                        "keepTargets": [{"start": 2, "end": 7}]} for i in range(7)]}
            original = root/"original.json"
            original.write_text(json.dumps(old))
            old_identity = {"path": str(original), "sha256": study.base.file_sha256(original)}
            document = root/"amendment.md"
            document.write_text("Prospective sampling repair")
            revision = {"recordingIds": [r["id"] for r in old["coverageRows"]]}
            amendment = root/"amendment.json"
            amendment.write_text(json.dumps({"originalManifest": old_identity, "featureRevision": revision,
                "document": {"path": str(document), "sha256": study.base.file_sha256(document)}}))
            amendment_identity = {"path": str(amendment), "sha256": study.base.file_sha256(amendment)}
            new = copy.deepcopy(old)
            for row in new["coverageRows"]:
                row["featureCaches"]["audiovisual"] = {"sha256": "corrected"}
            new.update(originalManifest=old_identity, protocolAmendment=amendment_identity, featureRevision=revision)
            path = root/"new.json"
            path.write_text(json.dumps(new))
            with patch.object(study, "AMENDMENT_SHA256", amendment_identity["sha256"]):
                self.assertEqual(study.verify_input_revision(path, old_identity["sha256"]), new)
                for field in ("gold", "exact", "unchanged_cache", "extra"):
                    changed = copy.deepcopy(new)
                    if field == "gold":
                        changed["coverageRows"][0]["keepTargets"][0]["end"] = 8
                    elif field == "exact":
                        changed["exactRows"][0]["featureCaches"]["audiovisual"]["sha256"] = "changed"
                    elif field == "unchanged_cache":
                        changed["coverageRows"][0]["featureCaches"]["audiovisual"] = {"sha256": "old"}
                    else:
                        changed["undeclared"] = True
                    path.write_text(json.dumps(changed))
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        study.verify_input_revision(path, old_identity["sha256"])

    def test_dino_alignment_uses_earlier_ties_and_rejects_wrong_source_association(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, sources, entries = [], [], []
            for index in range(18):
                e = example(f"record-{index}", f"group-{index}", 3, index)
                rows.append(frozen.exact_supervision(e))
                source_hash = f"{index:064x}"
                metadata = {"recordingId": e.id, "recordingContentSha256": source_hash,
                            "extractorConfigSha256": "config", "roi": None,
                            "labelsUsed": False, "completed": True, "backbone": {"modelName": "test"}}
                times = np.arange(4, dtype=np.float64)/4
                tokens = np.broadcast_to(np.arange(4, dtype=np.float16)[:, None, None], (4, 10, 384)).copy()
                cache = root/f"{e.id}.npz"
                np.savez(cache, timestamps=times, tokens=tokens, metadata_json=np.asarray(json.dumps(metadata)))
                digest = study.base.file_sha256(cache)
                wrapper = root/f"{e.id}.json"
                wrapper.write_text(json.dumps({"recordingId": e.id, "cacheMetadata": metadata, "cache": {"sha256": digest}}))
                source = {"id": e.id, "contentSha256": source_hash,
                          "featureCaches": {"audiovisual": {"sha256": "av"}}}
                sources.append(source)
                alignment = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in (
                    ("avTimesSha256", e.times.astype("<f8")), ("dinoTimesSha256", times.astype("<f8")),
                    ("nearestIndexesSha256", np.arange(3, dtype="<i8")))}
                entries.append({"recordingId": e.id, "tier": "exact", "sourceGroup": e.group, "passed": True,
                    "audiovisualSha256": "av", "sourceVideoVerified": {"sha256": source_hash},
                    "dinoPath": str(cache), "dinoSha256": digest, "metadataPath": str(wrapper),
                    "metadataSha256": study.base.file_sha256(wrapper), "nearestAlignment": alignment})
            parent = root/"expanded.json"
            parent.write_text(json.dumps({"exactRows": sources, "draftRows": [], "coverageRows": []}))
            manifest = {"expandedManifestPath": str(parent), "expandedManifestSha256": study.base.file_sha256(parent),
                        "records": entries, "protectedTestOpened": False, "beachIncluded": False,
                        "embeddingLabelsUsed": False, "extractorConfigSha256": "config",
                        "semanticBackbone": {"modelName": "test"}}
            path = root/"dino.json"
            path.write_text(json.dumps(manifest))
            with patch.object(frozen, "load_data", side_effect=lambda _: {"exact": rows.copy(), "draft": [], "coverage": []}):
                data = study.load_data(parent, path)
                for actual, original in zip(data["exact"], rows):
                    self.assertEqual(actual.example.values.shape, (3, 3944))
                    np.testing.assert_array_equal(actual.example.values[:, :104], original.example.values)
                    np.testing.assert_array_equal(actual.example.values[:, 104], np.arange(3))
                    np.testing.assert_array_equal(actual.mask, original.mask)
                manifest["records"][0]["sourceVideoVerified"]["sha256"] = "wrong-source"
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "source association"):
                    study.load_data(parent, path)


if __name__ == "__main__":
    unittest.main()
