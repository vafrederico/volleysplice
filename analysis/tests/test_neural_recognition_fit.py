"""Synthetic CPU checks for frozen-recipe parity and fold/context isolation."""
from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class RecognitionFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from analysis import neural_recognition_fit as fit
        from analysis import neural_short_boost_transfer as frozen
        from analysis import neural_short_boost_weighting as weighting
        from analysis import recognition_temporal_model as models
        from analysis.tests.test_neural_short_boost_transfer import population
        from analysis.tests.test_neural_expanded_development import example
        cls.fit, cls.frozen, cls.weighting, cls.models = fit, frozen, weighting, models
        cls.population = staticmethod(population)
        cls.example = staticmethod(example)
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

    def assert_archives_identical(self, left, right):
        with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
            self.assertEqual(a.files, b.files)
            for name in a.files:
                self.assertEqual(a[name].dtype, b[name].dtype, name)
                self.assertEqual(a[name].shape, b[name].shape, name)
                self.assertEqual(a[name].tobytes(), b[name].tobytes(), name)

    def with_extra_features(self, data, config):
        """Add deliberately non-AV-scaled synthetic tokens/scalars to every tier."""
        train, auxiliary, validation = copy.deepcopy(data)

        def extend(example, offset):
            count = len(example.times)
            extras = []
            if config.family == "mobile":
                # These large raw values must not be passed through scalar clipping.
                extras.append(np.random.default_rng(offset).normal(100, 20,
                    (count, config.token_count * config.token_dimension)).astype(np.float32))
            if config.scalar_dimension:
                scalars = np.arange(count * config.scalar_dimension, dtype=np.float32).reshape(count, -1) + offset
                scalars[~example.valid] = 1e9
                extras.append(scalars)
            return replace(example, values=np.concatenate((example.values, *extras), axis=1))

        train = [replace(row, example=extend(row.example, 10+i)) for i, row in enumerate(train)]
        auxiliary = {tier: [replace(row, example=extend(row.example, 1000+i)) for i, row in enumerate(rows)]
                     for tier, rows in auxiliary.items()}
        validation = [extend(example, 10000+i) for i, example in enumerate(validation)]
        return train, auxiliary, validation

    def test_av_and_dino_preprocessing_chunks_and_two_epoch_fits_match_frozen_exactly(self):
        for family, kind in (("av", "tcn"), ("dino", "dino_tcn")):
            with self.subTest(family=family):
                config = self.models.RecognitionConfig(family=family, head="tcn")
                train, auxiliary, validation = self.population(kind)
                mean, scale = self.fit.fit_scaler(train, config)
                old_mean, old_scale = self.frozen.base.fit_scaler([row.example for row in train])
                np.testing.assert_array_equal(mean, old_mean)
                np.testing.assert_array_equal(scale, old_scale)
                for rows in (train, *auxiliary.values()):
                    old = self.weighting.make_weighted_chunks(rows, mean, scale, kind, "short_boost")
                    new = self.fit.make_chunks(rows, mean, scale, config, "short_boost")
                    self.assertEqual(len(old), len(new))
                    for old_chunk, new_chunk in zip(old, new):
                        for index in (0, 1, 2, 4):
                            self.assertEqual(old_chunk[index].dtype, new_chunk[index].dtype)
                            self.assertEqual(old_chunk[index].tobytes(), new_chunk[index].tobytes())
                        self.assertEqual(old_chunk[3], new_chunk[3])
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    self.frozen.fit_model(train, auxiliary, validation, kind, 3407, (1, 2),
                                          root/"frozen", "cpu", "synthetic", "short_boost")
                    self.fit.fit_model(train, auxiliary, validation, config, 3407, (1, 2),
                                       root/"recognition", "cpu", "synthetic", "short_boost")
                    old, new = [self.fit.read(root/name/"completed.json") for name in ("frozen", "recognition")]
                    for key in ("history", "optimizerSteps", "exposureSha256", "positiveWeight",
                                "supervisedCounts", "liveLossWeighting", "scalerTrainIds"):
                        self.assertEqual(old[key], new[key], key)
                    for epoch in (1, 2):
                        for stem in ("weights", "predictions"):
                            self.assert_archives_identical(root/"frozen"/f"{stem}-{epoch}.npz",
                                                           root/"recognition"/f"{stem}-{epoch}.npz")

    def test_attention_changes_head_but_not_context_examples_sampling_or_loss_weights(self):
        import torch
        original_loss = self.weighting.batch_loss
        for family, kind in (("av", "tcn"), ("dino", "dino_tcn")):
            train, auxiliary, validation = self.population(kind)
            observed, metadata, results = {}, {}, {}
            with tempfile.TemporaryDirectory() as temporary:
                for head in ("tcn", "transformer"):
                    config = self.models.RecognitionConfig(family=family, head=head)
                    observed[head] = []

                    def observe(model, chunks, indexes, pos_weight, head_weights, device):
                        observed[head].append((tuple(indexes), tuple(head_weights),
                            pos_weight.cpu().numpy().tobytes(), torch.get_rng_state().numpy().tobytes(),
                            tuple(chunks[i][j].tobytes() for i in indexes for j in (0, 1, 2, 4)),
                            tuple(chunks[i][3] for i in indexes)))
                        return original_loss(model, chunks, indexes, pos_weight, head_weights, device)

                    destination = Path(temporary)/head
                    with patch.object(self.weighting, "batch_loss", side_effect=observe):
                        results[head] = self.fit.fit_model(train, auxiliary, validation, config, 1729,
                                                          (1, 2), destination, "cpu", "synthetic")
                    metadata[head] = self.fit.read(destination/"completed.json")
                self.assertEqual(observed["tcn"], observed["transformer"])
                for key in ("optimizerSteps", "exposureSha256", "positiveWeight", "supervisedCounts", "liveLossWeighting"):
                    self.assertEqual(metadata["tcn"][key], metadata["transformer"][key])
                self.assertFalse(np.array_equal(results["tcn"][2]["held"], results["transformer"][2]["held"]))

    def test_scaler_uses_only_valid_exact_training_and_never_normalizes_mobile_tokens(self):
        for family in ("player", "mobile"):
            config = self.models.RecognitionConfig(family=family, scalar_dimension=3,
                token_count=2, token_dimension=8)
            train, auxiliary, validation = self.with_extra_features(self.population("tcn"), config)
            indexes = self.fit.scalar_indexes(config)
            expected_values = np.concatenate([row.example.values[row.example.valid][:, indexes] for row in train])
            mean, scale = self.fit.fit_scaler(train, config)
            np.testing.assert_array_equal(mean[104:], expected_values.mean(0, dtype=np.float64).astype(np.float32))
            np.testing.assert_array_equal(scale[104:], np.maximum(expected_values.std(0, dtype=np.float64), 1e-4).astype(np.float32))
            held_before = validation[0].values.copy()
            normalized = self.fit.standardized(validation[0], mean, scale, config)
            np.testing.assert_array_equal(validation[0].values, held_before)
            np.testing.assert_array_equal(normalized[:, indexes], np.clip(
                (held_before[:, indexes]-mean[104:])/scale[104:], -10, 10))
            if family == "mobile":
                np.testing.assert_array_equal(normalized[:, 104:indexes[0]], held_before[:, 104:indexes[0]])
            # Auxiliary/validation extremes cannot reach this exact-training-only scaler.
            for row in [row for rows in auxiliary.values() for row in rows]:
                row.example.values[:] = -9e8
            validation[0].values[:] = 9e8
            after_mean, after_scale = self.fit.fit_scaler(train, config)
            np.testing.assert_array_equal(mean, after_mean)
            np.testing.assert_array_equal(scale, after_scale)

    def test_held_features_do_not_change_fitted_player_weights_or_scaler(self):
        config = self.models.RecognitionConfig(family="player", head="transformer", scalar_dimension=2)
        train, auxiliary, validation = self.with_extra_features(self.population("tcn"), config)
        alternate = copy.deepcopy(validation)
        alternate[0].values[:] = np.random.default_rng(53).normal(100, 100, alternate[0].values.shape)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, held in (("original", validation), ("changed", alternate)):
                self.fit.fit_model(train, auxiliary, held, config, 79, (1, 2), root/label, "cpu", "synthetic")
            for epoch in (1, 2):
                self.assert_archives_identical(root/"original"/f"weights-{epoch}.npz",
                                               root/"changed"/f"weights-{epoch}.npz")

    def test_context_is_not_supervision_mask_and_prediction_resets_ignored_gaps(self):
        import torch
        from analysis import neural_expanded_development as expanded
        config = self.models.RecognitionConfig(family="av", head="transformer")
        source = self.example(length=411)
        row = expanded.exact_supervision(source)
        # Observable but wholly unsupervised ticks still carry input context.
        row.mask[80:100] = 0
        mean, scale = np.zeros(104, np.float32), np.ones(104, np.float32)
        chunks = self.fit.make_chunks([row], mean, scale, config, "short_boost")
        coverage = np.zeros_like(row.mask)
        observed_unknown = False
        for values, targets, masks, _weight, _live in chunks:
            indexes = np.rint(values[:, 0] * len(source.times)).astype(int)
            self.assertTrue(source.valid[indexes].all())
            self.assertTrue((np.diff(indexes) == 1).all())
            self.assertLessEqual(len(values), 252)
            coverage[indexes] += masks
            observed_unknown |= bool(np.isin(indexes, np.arange(80, 100)).any())
        self.assertTrue(observed_unknown)
        np.testing.assert_array_equal(coverage, row.mask)
        torch.manual_seed(83)
        model = self.models.model_for(config).eval()
        # Invalid features must never enter a model call or poison valid outputs.
        row.example.values[~row.example.valid] = np.nan
        actual = self.fit.predict(model, row.example, mean, scale, config, "cpu")
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue((actual[~source.valid] == 0).all())
        expected = np.zeros_like(actual)
        with torch.no_grad():
            for left, right in self.fit.base.segments(source.valid):
                expected[left:right] = torch.sigmoid(model(torch.from_numpy(source.values[left:right][None])))[0].numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_all_training_tiers_are_excluded_from_validation_before_artifacts(self):
        config = self.models.RecognitionConfig()
        for tier in ("exact", "draft", "coverage"):
            train, auxiliary, validation = self.population("tcn")
            rows = train if tier == "exact" else auxiliary[tier]
            rows[0] = replace(rows[0], example=replace(rows[0].example, group=validation[0].group))
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary)/"bad"
                with self.assertRaisesRegex(ValueError, "source fold"):
                    self.fit.fit_model(train, auxiliary, validation, config, 7, (1,), destination, "cpu", "synthetic")
                self.assertFalse(destination.exists())

    def test_resume_binds_configuration_weighting_and_artifacts(self):
        config = self.models.RecognitionConfig()
        train, auxiliary, validation = self.population("tcn")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/"fit"
            args = (train, auxiliary, validation, config, 89, (1,), destination, "cpu", "synthetic")
            original = self.fit.fit_model(*args)
            with patch.object(self.fit, "make_chunks", side_effect=AssertionError("resume retrained")):
                resumed = self.fit.fit_model(*args)
            np.testing.assert_array_equal(original[1]["held"], resumed[1]["held"])
            with self.assertRaisesRegex(ValueError, "Resume"):
                self.fit.fit_model(*args, mode="baseline")
            with self.assertRaisesRegex(ValueError, "Resume"):
                self.fit.fit_model(train, auxiliary, validation, replace(config, head="tcn"), *args[4:])
            artifact = destination/"weights-1.npz"
            artifact.write_bytes(artifact.read_bytes()+b"tampered")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                self.fit.fit_model(*args)

    def test_invalid_epoch_types_are_rejected_before_creating_fit(self):
        config = self.models.RecognitionConfig()
        train, auxiliary, validation = self.population("tcn")
        with tempfile.TemporaryDirectory() as temporary:
            for index, epochs in enumerate(((), (0,), (2, 1), (1, 1), (True,), (1.5,))):
                destination = Path(temporary)/str(index)
                with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                    self.fit.fit_model(train, auxiliary, validation, config, 7, epochs, destination, "cpu", "synthetic")
                self.assertFalse(destination.exists())

    def test_model_metadata_preserves_rng_and_rejects_noninteger_geometry(self):
        import torch
        for family in ("av", "dino", "player", "mobile"):
            config = self.models.RecognitionConfig(family=family, scalar_dimension=2 if family in ("player", "mobile") else 0)
            before = torch.get_rng_state().clone()
            metadata = self.models.model_metadata(config)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(metadata["inputDimension"], config.input_dimension)
            self.assertEqual(metadata["parameters"], sum(p.numel() for p in self.models.model_for(config).parameters()))
        for kwargs in ({"scalar_dimension": True}, {"scalar_dimension": 1.5},
                       {"token_count": True}, {"token_dimension": 0.5}, {"projection_dimension": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.models.RecognitionConfig(family="mobile", **kwargs).validate()

    def test_shared_inner_owners_exclude_both_groups_and_selection_sees_only_its_inner_views(self):
        from dataclasses import asdict
        from analysis import neural_expanded_development as expanded
        from analysis import neural_recognition_development as development
        groups = ["a", "b", "c", "d"]
        data = {"exact": [expanded.exact_supervision(self.example(f"exact-{group}", group, 37, i))
                          for i, group in enumerate(groups)]}
        for tier in ("draft", "coverage"):
            data[tier] = [expanded.auxiliary_supervision(self.example(f"{tier}-{group}", group, 37, i), tier)
                          for i, group in enumerate((*groups, "aux-only"))]
        config = self.models.RecognitionConfig()
        decoder = expanded.decoder_candidates()[0]
        registration = {"sha256": "synthetic-contract", "contract": {
            "config": asdict(config), "groups": groups, "population": {
                "exactRows": [row.example.id for row in data["exact"]]},
            "cohort": "reviewed_export", "lossArm": "short_boost",
            "checkpointEpochs": [1, 2], "decoderCandidates": [decoder],
            "model": self.models.model_metadata(config),
        }}
        fit_calls, selection_views, decoded = [], [], []

        def spy_fit(train, auxiliary, validation, received_config, seed, epochs,
                    destination, device, digest, mode):
            if destination.name.startswith("inner-"):
                indices = [int(item) for item in destination.name.split("-")[1:]]
                excluded = {groups[index] for index in indices}
                self.assertEqual(tuple(epochs), (1, 2))
            else:
                excluded = {groups[int(destination.name.split("-")[1])]}
                self.assertEqual(tuple(epochs), (2,))
            self.assertEqual(received_config, config)
            self.assertEqual((seed, device, digest, mode), (3407, "cpu", "synthetic-contract", "short_boost"))
            self.assertEqual({row.example.group for row in train}, set(groups)-excluded)
            self.assertEqual({example.group for example in validation}, excluded)
            for tier in ("draft", "coverage"):
                self.assertEqual([row.example.id for row in auxiliary[tier]],
                    [row.example.id for row in data[tier] if row.example.group not in excluded])
                self.assertIn(f"{tier}-aux-only", [row.example.id for row in auxiliary[tier]])
            fit_calls.append((destination.name, excluded))
            # Distinct group/epoch signals detect accidentally selecting the
            # owner's other (outer) group or using the wrong checkpoint view.
            return {epoch: {example.id: np.full((len(example.times), 4),
                        (groups.index(example.group)+1)/10 + epoch/100, dtype=np.float32)
                        for example in validation} for epoch in epochs}

        def spy_choose(examples, probabilities):
            seen_groups = {example.group for example in examples}
            self.assertEqual(len(seen_groups), 3)
            outer = (set(groups)-seen_groups).pop()
            expected_ids = {f"exact-{group}" for group in seen_groups}
            for epoch, by_recording in probabilities.items():
                self.assertEqual(set(by_recording), expected_ids)
                self.assertNotIn(f"exact-{outer}", by_recording)
                for example in examples:
                    expected = np.float32((groups.index(example.group)+1)/10 + epoch/100)
                    self.assertTrue((by_recording[example.id] == expected).all())
            selection_views.append((outer, seen_groups))
            return {"epoch": 2, "decoder": decoder, "innerF1_padP_coreR": .8,
                    "innerR_core": .97, "recallEligibilityPassed": True, "recallEligibilityFloor": .95}

        def spy_decode(example, probabilities, settings):
            self.assertEqual(settings, decoder)
            expected = np.float32((groups.index(example.group)+1)/10 + .02)
            self.assertTrue((probabilities == expected).all())
            decoded.append(example.id)
            return []

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            with patch.object(expanded, "choose_settings", side_effect=spy_choose), \
                    patch.object(expanded.base, "decode", side_effect=spy_decode), \
                    patch.object(development, "evaluate_predictions", return_value={"checked": True}):
                result = development.run_cell(data, registration, destination, 3407, "cpu", fit_fn=spy_fit)
            self.assertEqual(len(fit_calls), 10)
            self.assertEqual(sum(name.startswith("inner-") for name, _ in fit_calls), 6)
            self.assertEqual(len({name for name, _ in fit_calls}), 10)
            self.assertEqual({outer for outer, _ in selection_views}, set(groups))
            self.assertEqual(sum(len(inner) for _, inner in selection_views), 12)
            self.assertEqual(set(decoded), {row.example.id for row in data["exact"]})
            self.assertEqual(len(decoded), 4)
            self.assertEqual({selection["heldSourceGroup"] for selection in result["selections"]}, set(groups))
            self.assertFalse(result["protectedTestOpened"])
            self.assertFalse(result["productionPromotionAllowed"])
            with patch.object(expanded, "choose_settings", side_effect=AssertionError("completed cell selected again")):
                resumed = development.run_cell(data, registration, destination, 3407, "cpu",
                    fit_fn=lambda *args: self.fail("completed cell fitted again"))
            self.assertEqual(result, resumed)


if __name__ == "__main__":
    unittest.main()
