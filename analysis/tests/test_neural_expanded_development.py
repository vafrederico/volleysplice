from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis import neural_expanded_development as study
from analysis.neural_development import Example, boundary_targets
from analysis.schema import Interval, labels_for_times, mask_for_times


def example(identifier="record", group="training", length=173, seed=7):
    times = np.arange(length, dtype=np.float64) / 4 + .125
    truth = (Interval(3., 6.), Interval(12., 12.5), Interval(21., 25.))
    ignored = (Interval(17., 18.),) if length > 72 else ()
    values = np.random.default_rng(seed).normal(size=(length, 104)).astype(np.float32)
    values[:, 0] = np.arange(length, dtype=np.float32) / length
    targets = np.stack((labels_for_times(times, truth),
                        boundary_targets(times, [r.start for r in truth]),
                        boundary_targets(times, [r.end for r in truth])), axis=1)
    return Example(identifier, group, length / 4, times, values, targets,
                   mask_for_times(times, ignored), truth, ignored, "grass")


class ExpandedSupervisionTests(unittest.TestCase):
    def test_exact_keeps_padded_gap_unions_but_never_ignored_ticks(self):
        source = example()
        row = study.exact_supervision(source)
        self.assertEqual(row.example.targets.shape, (173, 4))
        np.testing.assert_array_equal(row.example.targets[:, :3], source.targets)
        np.testing.assert_array_equal(row.mask[:, 0], source.valid)
        np.testing.assert_array_equal(row.mask[:, 3], source.valid)
        self.assertTrue(np.all(row.mask[~source.valid] == 0))
        halo = (source.times >= 16) & (source.times <= 19)
        self.assertTrue(np.all(row.mask[halo, 1:3] == 0))
        # [1,8) and [10,14.5) join across their two-second gap.
        self.assertTrue(np.all(row.example.targets[(source.times >= 8) & (source.times < 10), 3] == 1))
        self.assertTrue(np.all(row.example.targets[~source.valid, 3] == 0))
        self.assertEqual(source.targets.shape[1], 3)  # Source supervision was not mutated.

    def test_draft_censors_short_rallies_and_endpoint_uncertainty(self):
        source = example()
        row = study.auxiliary_supervision(source, "draft")
        np.testing.assert_array_equal(row.example.targets[:, 0], source.targets[:, 0])
        self.assertTrue(np.all(row.mask[:, 1:] == 0))
        self.assertTrue(np.all(row.example.targets[:, 1:] == 0))
        uncertain = np.zeros(len(source.times), dtype=bool)
        for interval in (*source.truth, *source.ignored):
            uncertain |= (np.abs(source.times-interval.start) <= 1) | (np.abs(source.times-interval.end) <= 1)
        np.testing.assert_array_equal(row.mask[:, 0], source.valid & ~uncertain)
        short = (source.times >= 12) & (source.times < 12.5)
        self.assertEqual(float(row.mask[short, 0].sum()), 0)

    def test_coverage_supervises_only_unpadded_keep_and_respects_valid_window(self):
        source = example()
        source.valid &= (source.times >= 2) & (source.times < 30)
        keep = (Interval(4., 6.), Interval(17., 20.))
        row = study.auxiliary_supervision(source, "coverage", keep)
        self.assertTrue(np.all(row.mask[:, :3] == 0))
        self.assertTrue(np.all(row.example.targets[:, :3] == 0))
        np.testing.assert_array_equal(row.mask[:, 3], source.valid)
        np.testing.assert_array_equal(row.example.targets[:, 3], labels_for_times(source.times, keep))
        self.assertEqual(float(row.example.targets[np.argmin(abs(source.times-3)), 3]), 0)

    def test_chunks_cover_each_supervised_tick_once_with_real_context_only(self):
        source = example(length=411)
        rows = [study.exact_supervision(source), study.auxiliary_supervision(source, "draft"),
                study.auxiliary_supervision(source, "coverage", (Interval(4., 20.),))]
        for row in rows:
            with self.subTest(tier=row.tier):
                chunks = study.make_chunks([row], np.zeros(104, np.float32), np.ones(104, np.float32), "tcn")
                observed = np.zeros((len(source.times), 4), np.float32)
                for values, targets, masks, weight in chunks:
                    indexes = np.rint(values[:, 0] * len(source.times)).astype(int)
                    self.assertTrue(np.all(source.valid[indexes]))
                    self.assertTrue(np.all(np.diff(indexes) == 1))
                    self.assertLessEqual(len(values), 252)
                    self.assertGreater(weight, 0)
                    np.testing.assert_array_equal(targets, row.example.targets[indexes])
                    observed[indexes] += masks
                np.testing.assert_array_equal(observed, row.mask)

    def test_fold_auxiliary_exclusion_applies_to_both_held_groups(self):
        data = {tier: [study.auxiliary_supervision(example(f"{tier}-{g}", g), tier)
                       for g in ("outer", "inner", "fit", "aux-only")]
                for tier in ("draft", "coverage")}
        self.assertEqual(study.auxiliary_for_fold(data, "exact", {"outer", "inner"}), {})
        draft = study.auxiliary_for_fold(data, "draft", {"outer", "inner"})
        self.assertEqual(set(draft), {"draft"})
        both = study.auxiliary_for_fold(data, "reviewed_export", {"outer", "inner"})
        for rows in both.values():
            self.assertEqual({r.example.group for r in rows}, {"fit", "aux-only"})
        refit = study.auxiliary_for_fold(data, "reviewed_export", {"outer"})
        self.assertEqual({r.example.group for r in refit["draft"]}, {"inner", "fit", "aux-only"})

    def test_selection_requires_recall_floor_then_ranks_f1_and_flags_fallback(self):
        rows = [{"epoch": 5, "innerR_core": .9499, "innerF1_padP_coreR": .99},
                {"epoch": 15, "innerR_core": .95, "innerF1_padP_coreR": .82},
                {"epoch": 30, "innerR_core": .99, "innerF1_padP_coreR": .81}]
        chosen = study.select_candidate(rows)
        self.assertEqual(chosen["epoch"], 15)
        self.assertTrue(chosen["recallEligibilityPassed"])
        fallback = study.select_candidate([rows[0], {"epoch": 60, "innerR_core": .94, "innerF1_padP_coreR": .9}])
        self.assertEqual(fallback["epoch"], 5)
        self.assertFalse(fallback["recallEligibilityPassed"])
        self.assertEqual(fallback["recallEligibilityFloor"], .95)
        tied = study.select_candidate([rows[1], {**rows[1], "epoch": 60}])
        self.assertEqual(tied["epoch"], 15)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class ExpandedTrainingTests(unittest.TestCase):
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

    def test_coverage_loss_is_valid_keep_mean_and_empty_heads_have_zero_gradient(self):
        import torch

        for length in (1, 91):
            logits = torch.zeros(2, length, 4, requires_grad=True)
            masks = torch.zeros_like(logits)
            masks[0, :, 3] = 1
            loss = study.masked_head_loss(logits, torch.zeros_like(logits), masks, torch.ones(4),
                                          torch.tensor(study.HEAD_WEIGHTS["coverage"]))
            self.assertAlmostEqual(float(loss.detach()), math.log(2)*.25, places=6)
            loss.backward()
            self.assertEqual(torch.count_nonzero(logits.grad[:, :, :3]).item(), 0)
            self.assertEqual(torch.count_nonzero(logits.grad[1]).item(), 0)
            self.assertGreater(torch.count_nonzero(logits.grad[0, :, 3]).item(), 0)
        logits = torch.randn(1, 7, 4, requires_grad=True)
        loss = study.masked_head_loss(logits, torch.zeros_like(logits), torch.zeros_like(logits),
                                      torch.ones(4), torch.ones(4))
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)

    def test_unequal_length_batches_reduce_per_head_over_all_real_ticks(self):
        import torch

        torch.manual_seed(67)
        model = torch.nn.Linear(104, 4)
        chunks = []
        for length in (3, 7):
            values = np.ones((length, 104), np.float32) * length
            targets = np.zeros((length, 4), np.float32)
            mask = np.zeros_like(targets)
            mask[:, 3] = 1
            mask[:1, 0] = 1
            chunks.append((values, targets, mask, 1))
        actual = study.batch_loss(model, chunks, [0, 1], torch.ones(4), (1, .5, .5, .25), "cpu")
        losses = []
        for x, y, mask, _ in chunks:
            elements = torch.nn.functional.binary_cross_entropy_with_logits(
                model(torch.from_numpy(x)), torch.from_numpy(y), reduction="none")
            losses.append(elements * torch.from_numpy(mask))
        expected = (sum(r.sum(dim=0) for r in losses) / torch.tensor([2, 1, 1, 10]) * torch.tensor([1, .5, .5, .25])).sum()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all().item())
        self.assertEqual(torch.count_nonzero(model.weight.grad[1:3]).item(), 0)

    def test_auxiliary_group_overlap_refused_before_files_or_model(self):
        training = [study.exact_supervision(example("exact", "training"))]
        auxiliary = {"coverage": [study.auxiliary_supervision(example("aux", "held"), "coverage")]}
        with patch.object(Path, "exists") as files, self.assertRaisesRegex(ValueError, "overlapping"):
            study.fit_model(training, auxiliary, [example("validation", "held")], "tcn", 3,
                            (1,), Path("unused"), "cpu", "contract")
        files.assert_not_called()

    def test_paired_fits_match_steps_exposure_and_dropout_streams_with_keep_transfer(self):
        import torch

        train = [study.exact_supervision(example("train", "fit", 173, 11))]
        draft = study.auxiliary_supervision(example("draft", "draft-group", 137, 13), "draft")
        coverage = study.auxiliary_supervision(example("keep", "keep-group", 141, 17), "coverage", (Interval(0., 30.),))
        validation = [example("validation", "held", 131, 19)]
        original_batch_loss = study.batch_loss
        with tempfile.TemporaryDirectory() as temporary:
            for kind in ("linear", "tcn"):
                outputs, metas, streams = {}, {}, {}
                for cohort, auxiliary in (("draft", {"draft": [draft]}),
                                           ("reviewed_export", {"draft": [draft], "coverage": [coverage]})):
                    calls = []
                    def observed_loss(model, chunks, indexes, pos_weight, weights, device):
                        calls.append((tuple(weights), torch.initial_seed(), model.training))
                        return original_batch_loss(model, chunks, indexes, pos_weight, weights, device)
                    destination = Path(temporary) / kind / cohort
                    with patch.object(study, "batch_loss", side_effect=observed_loss):
                        outputs[cohort] = study.fit_model(train, auxiliary, validation, kind, 71, (2,),
                                                        destination, "cpu", "paired-test")[2]["validation"]
                    metas[cohort] = json.loads((destination / "completed.json").read_text())
                    streams[cohort] = calls
                left, right = metas["draft"], metas["reviewed_export"]
                self.assertEqual(left["optimizerSteps"], right["optimizerSteps"])
                self.assertEqual([h["optimizerSteps"] for h in left["history"]], [h["optimizerSteps"] for h in right["history"]])
                for tier in ("exact", "draft"):
                    self.assertEqual(left["exposureSha256"][tier], right["exposureSha256"][tier])
                    weights = study.HEAD_WEIGHTS[tier]
                    self.assertEqual([c for c in streams["draft"] if c[0] == weights],
                                     [c for c in streams["reviewed_export"] if c[0] == weights])
                self.assertTrue(all(c[2] for c in streams["reviewed_export"]))
                self.assertEqual(left["scalerTrainIds"], ["train"])
                self.assertEqual(left["positiveWeight"], right["positiveWeight"])
                if kind == "linear":
                    np.testing.assert_array_equal(outputs["draft"][:, :3], outputs["reviewed_export"][:, :3])
                else:
                    self.assertGreater(float(np.max(abs(outputs["draft"][:, :3]-outputs["reviewed_export"][:, :3]))), 1e-7)

    def test_resume_refuses_seed_kind_contract_identity_and_tampered_artifact(self):
        train = [study.exact_supervision(example("train", "fit", 81))]
        validation = [example("validation", "held", 83)]
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            original = study.fit_model(train, {}, validation, "linear", 73, (1,), destination, "cpu", "resume")
            replay = study.fit_model(train, {}, validation, "linear", 73, (1,), destination, "cpu", "resume")
            np.testing.assert_array_equal(original[1]["validation"], replay[1]["validation"])
            for kind, seed, contract, held in (("tcn", 73, "resume", validation),
                                              ("linear", 74, "resume", validation),
                                              ("linear", 73, "changed", validation),
                                              ("linear", 73, "resume", [example("other", "held", 83)])):
                with self.subTest(kind=kind, seed=seed, contract=contract), self.assertRaises(ValueError):
                    study.fit_model(train, {}, held, kind, seed, (1,), destination, "cpu", contract)
            artifact = destination / "weights-1.npz"
            artifact.write_bytes(artifact.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                study.fit_model(train, {}, validation, "linear", 73, (1,), destination, "cpu", "resume")


if __name__ == "__main__":
    unittest.main()
