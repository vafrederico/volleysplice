from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis import neural_context_fit as fit
from analysis import neural_short_boost_transfer as frozen
from analysis import neural_short_boost_weighting as weighting
from analysis.tests.test_neural_expanded_development import example
from analysis.tests.test_neural_short_boost_transfer import population


def expected(meta):
    return {key: meta[key] for key in ("contractSha256", "trainIds", "auxiliaryIds", "kind", "seed", "lossArm")}


class ContextFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.threads = torch.get_num_threads()
        cls.deterministic = torch.are_deterministic_algorithms_enabled()
        cls.benchmark = torch.backends.cudnn.benchmark
        cls.cudnn_tf32 = torch.backends.cudnn.allow_tf32
        cls.matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.threads)
        torch.use_deterministic_algorithms(cls.deterministic)
        torch.backends.cudnn.benchmark = cls.benchmark
        torch.backends.cudnn.allow_tf32 = cls.cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = cls.matmul_tf32

    def assert_archives_equal(self, left, right):
        with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
            self.assertEqual(a.files, b.files)
            for name in a.files:
                self.assertEqual(a[name].dtype, b[name].dtype)
                self.assertEqual(a[name].shape, b[name].shape)
                self.assertEqual(a[name].tobytes(), b[name].tobytes(), name)

    def test_original_profile_epoch5_baseline_matches_frozen_fitter_both_architectures(self):
        for kind in fit.MODEL_KINDS:
            train, auxiliary, validation = population(kind)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = (train, auxiliary, validation, kind, 3407, (1, 5))
                frozen.fit_model(*args, root/"frozen", "cpu", "synthetic-contract", "baseline")
                fit.fit_model(*args, root/"original", "cpu", "synthetic-contract", context="original")
                old, new = [fit.read(root/name/"completed.json") for name in ("frozen", "original")]
                for key in ("history", "optimizerSteps", "exposureSha256", "positiveWeight", "supervisedCounts", "parameters", "liveLossWeighting"):
                    self.assertEqual(old[key], new[key], key)
                for epoch in (1, 5):
                    for stem in ("weights", "predictions"):
                        self.assert_archives_equal(root/"frozen"/f"{stem}-{epoch}.npz", root/"original"/f"{stem}-{epoch}.npz")

    def test_short_context_changes_only_model_not_chunks_masks_sampling_or_rng(self):
        import torch
        original_loss = weighting.batch_loss
        for kind in fit.MODEL_KINDS:
            train, auxiliary, validation = population(kind)
            calls, metas, outputs = {}, {}, {}
            with tempfile.TemporaryDirectory() as temporary:
                for context in ("original", "short"):
                    calls[context] = []
                    def observe(model, chunks, indexes, pos_weight, head_weights, device):
                        before = (tuple(indexes), tuple(head_weights), pos_weight.detach().numpy().tobytes(),
                                  torch.get_rng_state().numpy().tobytes(),
                                  tuple(chunks[i][j].tobytes() for i in indexes for j in (0, 1, 2, 4)),
                                  tuple(chunks[i][3] for i in indexes))
                        loss = original_loss(model, chunks, indexes, pos_weight, head_weights, device)
                        calls[context].append((*before, torch.get_rng_state().numpy().tobytes()))
                        return loss
                    location = Path(temporary)/context
                    with patch.object(weighting, "batch_loss", side_effect=observe):
                        outputs[context] = fit.fit_model(train, auxiliary, validation, kind, 1729, (1, 2),
                                                        location, "cpu", "synthetic-contract", context=context)
                    metas[context] = fit.read(location/"completed.json")
                self.assertEqual(calls["original"], calls["short"])
                for key in ("optimizerSteps", "exposureSha256", "supervisedCounts", "positiveWeight", "liveLossWeighting"):
                    self.assertEqual(metas["original"][key], metas["short"][key])
                self.assertNotEqual(metas["original"]["history"][0]["loss"], metas["short"]["history"][0]["loss"])
                self.assertFalse(np.array_equal(outputs["original"][2]["held"], outputs["short"][2]["held"]))

    def test_union_validation_is_role_independent_and_excludes_every_training_tier(self):
        train, auxiliary, validation = population("tcn")
        union = validation + [example("held-second", "validation-second", 57, 201)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, held in (("one-view", validation), ("owner", union)):
                fit.fit_model(train, auxiliary, held, "tcn", 20260918, (1,), root/label, "cpu", "same-contract")
            one, owner = [fit.read(root/name/"completed.json") for name in ("one-view", "owner")]
            self.assertEqual(one["trainingIdentitySha256"], owner["trainingIdentitySha256"])
            self.assertEqual(owner["validationGroups"], ["validation", "validation-second"])
            self.assert_archives_equal(root/"one-view/weights-1.npz", root/"owner/weights-1.npz")
            for tier in ("exact", "draft", "coverage"):
                changed_train, changed_aux = copy.deepcopy(train), copy.deepcopy(auxiliary)
                records = changed_train if tier == "exact" else changed_aux[tier]
                records[0] = frozen.replace(records[0], example=frozen.replace(records[0].example, group="validation-second"))
                with self.assertRaisesRegex(ValueError, "source fold"):
                    fit.fit_model(changed_train, changed_aux, union, "tcn", 1, (1,), root/f"leak-{tier}", "cpu", "c")
                self.assertFalse((root/f"leak-{tier}").exists())

    def test_resume_binds_context_role_free_identity_and_every_artifact(self):
        train, auxiliary, validation = population("tcn")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/"fit"
            args = (train, auxiliary, validation, "tcn", 3407, (1,), destination, "cpu", "contract")
            original = fit.fit_model(*args)
            with patch.object(fit, "model_for", side_effect=AssertionError("resume allocated/retrained")):
                resumed = fit.fit_model(*args)
            np.testing.assert_array_equal(original[1]["held"], resumed[1]["held"])
            for changed in ({"context": "original"}, {"weighting": "short_boost"}):
                with self.assertRaisesRegex(ValueError, "resume"):
                    fit.fit_model(*args, **changed)
            meta_path = destination/"completed.json"
            meta = fit.read(meta_path)
            meta["trainingIdentitySha256"] = "different"
            meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "resume"):
                fit.fit_model(*args)
            meta["trainingIdentitySha256"] = fit.canonical_hash(meta["trainingIdentity"])
            meta_path.write_text(json.dumps(meta))
            weight = destination/"weights-1.npz"
            weight.write_bytes(weight.read_bytes()+b"tampered")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                fit.fit_model(*args)

    def test_checkpoint_replay_preserves_rng_context_and_ignored_zeroes(self):
        import torch
        for kind in fit.MODEL_KINDS:
            train, auxiliary, validation = population(kind)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for context in ("original", "short"):
                    destination = root/context
                    outputs = fit.fit_model(train, auxiliary, validation, kind, 3407, (1,), destination,
                                            "cpu", "contract", context=context)
                    meta = fit.read(destination/"completed.json")
                    identity = expected(meta)
                    rng = torch.get_rng_state().clone()
                    replayed = fit.predict_checkpoint(destination, 1, validation, kind=kind, context=context, expected_identity=identity)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    np.testing.assert_array_equal(replayed["held"], outputs[1]["held"])
                    self.assertTrue(np.all(replayed["held"][~validation[0].valid] == 0))
                    for wrong in ({**identity, "seed": 9}, {"kind": kind}):
                        with self.assertRaisesRegex(ValueError, "expected identity"):
                            fit.predict_checkpoint(destination, 1, validation, kind=kind, context=context, expected_identity=wrong)
                    with self.assertRaisesRegex(ValueError, "context/model"):
                        fit.predict_checkpoint(destination, 1, validation, kind=kind,
                                               context="short" if context == "original" else "original", expected_identity=identity)
                    with self.assertRaisesRegex(ValueError, "overlaps"):
                        fit.predict_checkpoint(destination, 1, [train[0].example], kind=kind, context=context, expected_identity=identity)

    def test_historical_transfer_replay_requires_explicit_original_context(self):
        train, auxiliary, validation = population("tcn")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/"frozen"
            outputs = frozen.fit_model(train, auxiliary, validation, "tcn", 3407, (1,), destination, "cpu", "old", "baseline")
            identity = expected(fit.read(destination/"completed.json"))
            replayed = fit.predict_checkpoint(destination, 1, validation, kind="tcn", context="original", expected_identity=identity)
            np.testing.assert_array_equal(replayed["held"], outputs[1]["held"])
            with self.assertRaisesRegex(ValueError, "historical checkpoint"):
                fit.predict_checkpoint(destination, 1, validation, kind="tcn", context="short", expected_identity=identity)

    def test_incomplete_path_and_invalid_epochs_fail_without_training(self):
        train, auxiliary, validation = population("tcn")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/"partial"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                fit.fit_model(train, auxiliary, validation, "tcn", 1, (1,), destination, "cpu", "c")
            for epochs in ((), (0,), (2, 1), (1, 1), (True,)):
                with self.assertRaisesRegex(ValueError, "checkpoint epochs"):
                    fit.fit_model(train, auxiliary, validation, "tcn", 1, epochs, destination, "cpu", "c")


if __name__ == "__main__":
    unittest.main()
