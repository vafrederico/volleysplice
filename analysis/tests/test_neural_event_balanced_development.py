from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis import neural_event_balanced_development as study
from analysis import neural_event_weighting as weighting
from analysis import neural_expanded_development as frozen
from analysis.schema import Interval
from analysis.tests.test_neural_expanded_development import example


def arrays_digest(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class EventBalancedRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.previous_threads = torch.get_num_threads()
        cls.previous_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch

        torch.set_num_threads(cls.previous_threads)
        torch.use_deterministic_algorithms(cls.previous_deterministic)

    def populations(self):
        training = [frozen.exact_supervision(example("train", "fit", 173, 11)),
                    frozen.exact_supervision(example("train-two", "fit-two", 211, 23))]
        auxiliary = {
            "draft": [frozen.auxiliary_supervision(example("draft", "draft-group", 137, 13), "draft")],
            "coverage": [frozen.auxiliary_supervision(example("keep", "keep-group", 141, 17),
                                                        "coverage", (Interval(0., 30.),))],
        }
        return training, auxiliary, [example("held", "validation", 131, 19)]

    def observed_fit(self, fitter, loss_owner, destination, mode=None):
        import torch

        train, auxiliary, validation = self.populations()
        original_loss = loss_owner.batch_loss
        calls = []

        def observe(model, chunks, indexes, positive_weight, head_weights, device):
            # Observe actual input supervision and RNG before/after every bucketed
            # forward; no hook consumes random numbers or modifies the graph.
            call = {"indexes": list(indexes), "headWeights": tuple(head_weights),
                    "seed": torch.initial_seed(), "training": model.training,
                    "rngBefore": torch.get_rng_state().numpy().tobytes(),
                    "inputDigest": arrays_digest([chunks[i][j] for i in indexes for j in range(3)]),
                    "samplingWeights": [chunks[i][3] for i in indexes],
                    "positiveWeight": positive_weight.detach().cpu().tolist()}
            result = original_loss(model, chunks, indexes, positive_weight, head_weights, device)
            call["rngAfter"] = torch.get_rng_state().numpy().tobytes()
            calls.append(call)
            return result

        kwargs = {} if mode is None else {"weighting": mode}
        with patch.object(loss_owner, "batch_loss", side_effect=observe):
            predictions = fitter(train, auxiliary, validation, "tcn", 71, (1, 2),
                                  destination, "cpu", "paired-preflight", **kwargs)
        return predictions, json.loads((destination/"completed.json").read_text()), calls

    def test_uniform_fit_is_bit_exact_to_frozen_fitter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_predictions, old_meta, old_calls = self.observed_fit(
                frozen.fit_model, frozen, root/"frozen")
            new_predictions, new_meta, new_calls = self.observed_fit(
                study.fit_model, weighting, root/"uniform", "uniform")
            self.assertEqual(old_calls, new_calls)
            self.assertGreater(len(old_calls), 3)  # Multiple steps and all tiers.
            self.assertEqual({c["headWeights"] for c in old_calls}, set(frozen.HEAD_WEIGHTS.values()))
            self.assertTrue(all(c["training"] for c in old_calls))
            for field in ("history", "optimizerSteps", "exposureSha256", "positiveWeight",
                          "supervisedCounts", "scalerTrainIds", "parameters"):
                self.assertEqual(old_meta[field], new_meta[field], field)
            self.assertEqual(new_meta["parameters"], 29700)
            self.assertFalse(new_meta["liveLossWeighting"]["rowDiagnosticsApply"])
            for epoch in (1, 2):
                np.testing.assert_array_equal(old_predictions[epoch]["held"], new_predictions[epoch]["held"])
                with np.load(root/"frozen"/f"weights-{epoch}.npz", allow_pickle=False) as old, \
                        np.load(root/"uniform"/f"weights-{epoch}.npz", allow_pickle=False) as new:
                    self.assertEqual(old.files, new.files)
                    for name in old.files:
                        np.testing.assert_array_equal(old[name], new[name], err_msg=name)

    def test_weighting_changes_objective_but_preserves_inputs_steps_and_dropout_streams(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            uniform, uniform_meta, uniform_calls = self.observed_fit(
                study.fit_model, weighting, root/"uniform", "uniform")
            weighted, weighted_meta, weighted_calls = self.observed_fit(
                study.fit_model, weighting, root/"weighted", "per_rally")
            self.assertEqual(uniform_calls, weighted_calls)
            for field in ("optimizerSteps", "exposureSha256", "positiveWeight",
                          "supervisedCounts", "scalerTrainIds", "parameters"):
                self.assertEqual(uniform_meta[field], weighted_meta[field], field)
            for left, right in zip(uniform_meta["history"], weighted_meta["history"]):
                self.assertEqual(left["optimizerSteps"], right["optimizerSteps"])
                self.assertEqual(left["exposureSha256"], right["exposureSha256"])
            self.assertNotEqual(uniform_meta["history"][0]["loss"], weighted_meta["history"][0]["loss"])
            self.assertGreater(float(np.max(np.abs(uniform[2]["held"]-weighted[2]["held"]))), 1e-7)
            self.assertTrue(weighted_meta["liveLossWeighting"]["rowDiagnosticsApply"])
            with np.load(root/"uniform"/"weights-2.npz", allow_pickle=False) as left, \
                    np.load(root/"weighted"/"weights-2.npz", allow_pickle=False) as right:
                np.testing.assert_array_equal(left["mean"], right["mean"])
                np.testing.assert_array_equal(left["scale"], right["scale"])
            # Shared TCN weights and global clipping can change the other heads'
            # learned outputs; only their targets, masks and loss inputs stay fixed.

    def test_resume_binds_mode_identity_epochs_seed_and_artifacts(self):
        train, auxiliary, validation = self.populations()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/"fit"
            args = [train, auxiliary, validation, "tcn", 73, (1,), destination, "cpu", "resume"]
            predictions = study.fit_model(*args)
            with patch.object(weighting, "make_weighted_chunks", side_effect=AssertionError("resume retrained")):
                replay = study.fit_model(*args)
            np.testing.assert_array_equal(predictions[1]["held"], replay[1]["held"])
            changes = [(3, "linear"), (4, 74), (5, (1, 2)), (8, "changed"),
                       (0, list(reversed(train))), (1, {}),
                       (2, [example("different", "validation", 131, 19)])]
            for index, value in changes:
                changed = args.copy()
                changed[index] = value
                with self.subTest(index=index), self.assertRaises(ValueError):
                    study.fit_model(*changed)
            with self.assertRaisesRegex(ValueError, "weighting"):
                study.fit_model(*args, weighting="uniform")
            changed = copy.deepcopy(train)
            tick = np.flatnonzero((changed[0].mask[:, 0] > 0) & (changed[0].example.targets[:, 0] > 0))[0]
            changed[0].mask[tick, 0] = 0
            with self.assertRaisesRegex(ValueError, "weighting"):
                study.fit_model(changed, *args[1:])
            metadata_path = destination/"completed.json"
            metadata = json.loads(metadata_path.read_text())
            removed = copy.deepcopy(metadata)
            removed["artifacts"].pop("weights-1.npz")
            metadata_path.write_text(json.dumps(removed))
            with self.assertRaisesRegex(ValueError, "inventory"):
                study.fit_model(*args)
            metadata_path.write_text(json.dumps(metadata))
            artifact = destination/"weights-1.npz"
            artifact.write_bytes(artifact.read_bytes()+b"tamper")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                study.fit_model(*args)

    def test_overlap_and_incomplete_destinations_are_refused_without_training(self):
        train, auxiliary, validation = self.populations()
        overlap = {"draft": [frozen.auxiliary_supervision(example("leak", "validation"), "draft")]}
        with patch.object(Path, "exists") as exists, self.assertRaisesRegex(ValueError, "overlapping"):
            study.fit_model(train, overlap, validation, "tcn", 3, (1,), Path("unused"), "cpu", "contract")
        exists.assert_not_called()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/"incomplete"
            destination.mkdir()
            marker = destination/"weights-1.npz"
            marker.write_bytes(b"partial")
            with patch.object(weighting, "make_weighted_chunks") as chunks, \
                    self.assertRaisesRegex(FileExistsError, "incomplete"):
                study.fit_model(train, auxiliary, validation, "tcn", 3, (1,), destination, "cpu", "contract")
            chunks.assert_not_called()
            self.assertEqual(marker.read_bytes(), b"partial")


class EventBalancedPartitionTests(unittest.TestCase):
    def test_jobs_retain_frozen_fold_membership_and_inner_only_selection(self):
        groups = ["a", "b", "c", "d"]
        data = {"exact": [frozen.exact_supervision(example(f"exact-{g}", g)) for g in groups]}
        for tier in ("draft", "coverage"):
            data[tier] = [frozen.auxiliary_supervision(example(f"{tier}-{g}", g), tier)
                          for g in (*groups, "aux-only")]
        fit_calls, choices = [], []
        decoder = frozen.decoder_candidates()[0]

        def fit(train, auxiliary, validation, kind, seed, epochs, destination, device, contract):
            held = {e.group for e in validation}
            fitting = {r.example.group for r in train}
            self.assertFalse(held & fitting)
            outer_index = int(destination.parent.name.removeprefix("outer-"))
            outer = groups[outer_index]
            excluded = held | {outer}
            for tier, rows in auxiliary.items():
                self.assertEqual([r.example.id for r in rows],
                                 [r.example.id for r in data[tier] if r.example.group not in excluded])
            if destination.name.startswith("inner-"):
                self.assertEqual(fitting, set(groups)-excluded)
                self.assertEqual(epochs, study.EPOCHS)
            else:
                self.assertEqual(held, {outer})
                self.assertEqual(fitting, set(groups)-{outer})
                self.assertEqual(epochs, (15,))
            self.assertEqual((kind, seed, contract), ("tcn", study.base.SEEDS[0], "contract"))
            fit_calls.append(destination)
            return {epoch: {e.id: np.zeros((len(e.times), 4), np.float32) for e in validation}
                    for epoch in epochs}

        def choose(examples, predictions):
            ids = {e.id for e in examples}
            self.assertEqual(len(ids), 3)
            self.assertEqual(set(predictions), set(study.EPOCHS))
            for probabilities in predictions.values():
                self.assertEqual(set(probabilities), ids)
            choices.append(ids)
            return {"epoch": 15, "decoder": decoder, "innerF1_padP_coreR": .8,
                    "innerR_core": .97, "recallEligibilityPassed": True, "recallEligibilityFloor": .95}

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(study, "fit_model", side_effect=fit), \
                    patch.object(frozen, "choose_settings", side_effect=choose), \
                    patch.object(study.base, "decode", return_value=[]):
                study.run_job(data, {"groups": groups}, "contract", "reviewed_export",
                              study.base.SEEDS[0], output, "cpu")
            result = json.loads((output/f"result-reviewed_export-tcn-{study.base.SEEDS[0]}.json").read_text())
            self.assertEqual(len(fit_calls), 16)
            self.assertEqual(len(set(fit_calls)), 16)
            self.assertEqual(len(choices), 4)
            self.assertEqual([r["heldSourceGroup"] for r in result["selections"]], groups)
            self.assertEqual({r["id"] for r in result["predictions"]}, {f"exact-{g}" for g in groups})
            self.assertEqual(result["contractSha256"], "contract")
        for cohort, seed in (("unknown", study.base.SEEDS[0]), ("exact", -1)):
            with self.assertRaisesRegex(ValueError, "grid"):
                study.run_job(data, {"groups": groups}, "contract", cohort, seed, Path("unused"), "cpu")


if __name__ == "__main__":
    unittest.main()
