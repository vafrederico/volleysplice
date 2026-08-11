from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.config import DecoderConfig, FeatureConfig, TrainingConfig
from analysis.feature_experiments import (
    FeatureExperimentError,
    build_feature_sets,
    build_fold_plan,
    circular_shift_base_feature,
    circular_shift_family,
    classify_paired_deltas,
    run_development_experiments,
    run_fixed_split_final_test,
)
from analysis.features import FeatureSequence, VideoMetadata, contextualize
from analysis.pipeline import PreparedRecording
from analysis.schema import DatasetManifest, Interval, Recording, labels_for_times


def recording(
    root: Path,
    recording_id: str,
    source_group: str,
    split: str,
) -> Recording:
    video = root / f"{recording_id}.mp4"
    video.write_bytes(recording_id.encode("utf-8"))
    return Recording(
        id=recording_id,
        video=video,
        split=split,
        source_group=source_group,
        environment="indoor",
        game={"playersPerTeam": 4, "targetPoints": None},
        rallies=(Interval(2.0, 5.0, tags=("ace",)),),
        ignored_intervals=(),
        roi=(0.0, 0.0, 1.0, 1.0),
        capture={
            "position": "centered-behind-endline",
            "stationary": True,
            "fullCourtVisible": True,
            "serviceAreasVisible": True,
        },
        consent={"analyze": True, "train": split in {"train", "validation"}},
        content_sha256=None,
        raw={"id": recording_id, "split": split, "sourceGroup": source_group},
    )


def prepared_recording(item: Recording, config: FeatureConfig) -> PreparedRecording:
    times = np.arange(0.0, 8.0, 1.0, dtype=np.float64)
    labels = labels_for_times(times, item.rallies)
    values = np.column_stack(
        (
            0.1 + 0.8 * labels + np.linspace(0.0, 0.07, len(times)),
            0.8 * labels[::-1] + np.linspace(0.07, 0.0, len(times)),
        )
    ).astype(np.float32)
    sequence = FeatureSequence(
        times=times,
        values=values,
        names=("luma_mean", "audio_rms"),
        metadata=VideoMetadata(8.0, 1280, 720, 30.0, 240, True),
    )
    contextual_values, contextual_names = contextualize(sequence, config)
    return PreparedRecording(
        recording=item,
        sequence=sequence,
        contextual_values=contextual_values,
        contextual_names=contextual_names,
        labels=labels,
        sample_mask=np.ones(len(times), dtype=np.bool_),
    )


class FoldPlanningTests(unittest.TestCase):
    def test_outer_and_inner_folds_keep_sister_videos_in_one_source_group(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-fold-test-") as directory:
            root = Path(directory)
            rows = [
                recording(root, "a-one", "a", "train"),
                recording(root, "a-two", "a", "train"),
                recording(root, "b-one", "b", "train"),
                recording(root, "c-one", "c", "train"),
                recording(root, "d-one", "d", "validation"),
                recording(root, "protected", "test-group", "test"),
            ]

            folds = build_fold_plan(rows)

            self.assertEqual([item.held_out_group for item in folds], ["a", "b", "c", "d"])
            held_a = folds[0]
            self.assertNotIn("a", held_a.training_groups)
            self.assertEqual(len(held_a.inner_folds), 3)
            self.assertTrue(
                all(
                    inner.validation_group not in inner.training_groups
                    for inner in held_a.inner_folds
                )
            )
            self.assertNotIn("test-group", {item.held_out_group for item in folds})

    def test_feature_sets_cover_full_legacy_added_and_leave_one_family_out(self) -> None:
        names = (
            "t-1s/luma_mean",
            "t+0s/luma_mean",
            "t-1s/diff_mean",
            "t+0s/diff_mean",
            "t-1s/audio_rms",
            "t+0s/audio_rms",
        )

        sets = {item.name: item for item in build_feature_sets(names)}

        self.assertEqual(sets["full"].indexes, tuple(range(len(names))))
        self.assertEqual(sets["legacy_only"].indexes, (0, 1, 2, 3))
        self.assertEqual(sets["added_only"].indexes, (4, 5))
        self.assertEqual(sets["full_minus_audio_level"].indexes, (0, 1, 2, 3))
        self.assertEqual(
            sets["full_minus_legacy_frame_difference"].indexes,
            (0, 1, 4, 5),
        )


class ClassificationAndPermutationTests(unittest.TestCase):
    def test_paired_classification_requires_three_of_four_groups(self) -> None:
        helpful = classify_paired_deltas([0.02, 0.03, 0.011, -0.005])
        harmful = classify_paired_deltas([-0.02, -0.03, -0.011, 0.005])
        neutral = classify_paired_deltas([0.0, 0.001, -0.009, 0.008])
        uncertain = classify_paired_deltas([0.02, 0.02, -0.02, -0.02])

        self.assertEqual(helpful["classification"], "helpful")
        self.assertEqual(harmful["classification"], "harmful")
        self.assertEqual(neutral["classification"], "neutral")
        self.assertEqual(uncertain["classification"], "uncertain")
        self.assertEqual(helpful["requiredConsistentGroups"], 3)

    def test_paired_classification_rejects_an_overwhelming_counterfold(self) -> None:
        result = classify_paired_deltas([0.02, 0.03, 0.011, -0.2])

        self.assertEqual(result["classification"], "uncertain")
        self.assertLess(result["meanDelta"], 0.0)

    def test_circular_shift_moves_a_whole_family_and_rebuilds_context(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-permutation-test-") as directory:
            root = Path(directory)
            item = recording(root, "sample", "group", "train")
            config = FeatureConfig(
                analysis_fps=1.0,
                use_optical_flow=False,
                use_advanced_visual=False,
                use_audio=False,
                context_offsets_seconds=(0.0,),
                sequence_normalization="none",
            )
            times = np.arange(5, dtype=np.float64)
            sequence = FeatureSequence(
                times=times,
                values=np.column_stack(
                    (
                        np.arange(5),
                        np.arange(10, 15),
                        np.arange(20, 25),
                    )
                ).astype(np.float32),
                names=("luma_mean", "saturation_mean", "audio_rms"),
                metadata=VideoMetadata(5.0, 1280, 720, 30.0, 150, True),
            )
            values, names = contextualize(sequence, config)
            prepared = PreparedRecording(
                recording=item,
                sequence=sequence,
                contextual_values=values,
                contextual_names=names,
                labels=np.zeros(5, dtype=np.float32),
                sample_mask=np.ones(5, dtype=np.bool_),
            )

            shifted = circular_shift_family(
                prepared, "legacy_appearance", 2, config
            )

            np.testing.assert_array_equal(
                shifted.sequence.values[:, :2],
                np.roll(sequence.values[:, :2], 2, axis=0),
            )
            np.testing.assert_array_equal(
                shifted.sequence.values[:, 2], sequence.values[:, 2]
            )
            np.testing.assert_array_equal(
                shifted.contextual_values, shifted.sequence.values
            )
            self.assertEqual(shifted.contextual_names, prepared.contextual_names)

            single = circular_shift_base_feature(
                prepared, "saturation_mean", 1, config
            )
            np.testing.assert_array_equal(
                single.sequence.values[:, 1],
                np.roll(sequence.values[:, 1], 1),
            )
            np.testing.assert_array_equal(
                single.sequence.values[:, (0, 2)],
                sequence.values[:, (0, 2)],
            )


class DevelopmentExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="volleycut-feature-experiment-test-"
        )
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.config = FeatureConfig(
            analysis_fps=1.0,
            use_optical_flow=False,
            use_advanced_visual=False,
            use_audio=False,
            context_offsets_seconds=(0.0,),
            sequence_normalization="percentile-rank",
        )
        rows = [
            recording(self.root, "dev-a", "a", "train"),
            recording(self.root, "dev-b", "b", "train"),
            recording(self.root, "dev-c", "c", "train"),
            recording(self.root, "dev-d", "d", "validation"),
            recording(self.root, "test-e", "e", "test"),
        ]
        manifest_path = self.root / "manifest.json"
        manifest_payload = {
            "schemaVersion": 1,
            "name": "synthetic-feature-experiment",
            "annotationPolicy": {"id": "serve-contact-to-dead-ball-v1"},
            "recordings": [item.raw for item in rows],
        }
        manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
        self.manifest = DatasetManifest(
            path=manifest_path,
            name="synthetic-feature-experiment",
            recordings=tuple(rows),
            raw=manifest_payload,
        )
        self.prepared = [
            prepared_recording(item, self.config)
            for item in rows
            if item.split in {"train", "validation"}
        ]

    def test_development_suite_never_prepares_test_and_emits_all_required_sections(self) -> None:
        base_decoder = DecoderConfig(
            smoothing_seconds=0.0,
            enter_threshold=0.5,
            exit_threshold=0.4,
            min_live_seconds=1.0,
            bridge_gap_seconds=0.0,
            short_event_min_seconds=0.5,
            short_event_threshold=0.8,
        )

        with patch(
            "analysis.feature_experiments._select_decoder",
            side_effect=lambda _prepared, _probabilities, base: (
                base,
                {"status": "fixed-for-test"},
            ),
        ):
            report = run_development_experiments(
                self.manifest,
                self.prepared,
                feature_config=self.config,
                training_config=TrainingConfig(
                    epochs=2,
                    batch_size=4,
                    learning_rate=0.02,
                    patience=2,
                    seed=3,
                ),
                decoder_config=base_decoder,
                permutation_repeats=1,
                padding_seconds=(0, 1, 2, 3),
            )

        self.assertFalse(report["testLabelsUsed"])
        self.assertEqual(report["protected"]["test"]["recordingIds"], ["test-e"])
        self.assertNotIn("test-e", report["development"]["recordingIds"])
        self.assertIn("full", report["candidates"])
        self.assertIn("legacy_only", report["candidates"])
        self.assertIn("added_only", report["candidates"])
        self.assertIn("full_minus_audio_level", report["candidates"])
        self.assertEqual(len(report["candidates"]["full"]["outerFolds"]), 4)
        self.assertIn(
            "full_minus_audio_level", report["pairedComparisonsAgainstFull"]
        )
        self.assertIn("audio_level", report["permutationImportance"])
        self.assertEqual(
            report["baseFeaturePermutationImportance"]["features"], 2
        )
        self.assertIn(
            "audio_rms", report["baseFeaturePermutationImportance"]["byFeature"]
        )
        self.assertIn(
            "audio_rms", report["standardizedCoefficientProfiles"]["byFeature"]
        )
        self.assertIn("audio_level", report["featureFamilyAssessments"])
        self.assertIn(
            report["featureFamilyAssessments"]["audio_level"]["classification"],
            {"helpful", "harmful", "neutral", "uncertain"},
        )
        self.assertIn(
            "shortEventLogicDisabled", report["decoderComponentAblations"]
        )
        self.assertEqual(
            report["oofPadding"]["paddingSecondsBeforeAndAfter"],
            [0.0, 1.0, 2.0, 3.0],
        )
        prediction_ids = {
            item["id"] for item in report["oofPadding"]["oofPredictions"]
        }
        self.assertEqual(prediction_ids, {"dev-a", "dev-b", "dev-c", "dev-d"})
        json.dumps(report, allow_nan=False)

    def test_development_rejects_a_prepared_test_recording(self) -> None:
        test_item = next(
            item for item in self.manifest.recordings if item.split == "test"
        )
        contaminated = [*self.prepared, prepared_recording(test_item, self.config)]

        with self.assertRaisesRegex(
            FeatureExperimentError, r"exactly train\+validation"
        ):
            run_development_experiments(
                self.manifest,
                contaminated,
                feature_config=self.config,
                training_config=TrainingConfig(epochs=1, patience=1),
                decoder_config=DecoderConfig(),
                permutation_repeats=1,
            )

    def test_final_test_gate_rejects_a_report_that_already_used_test(self) -> None:
        report = self.root / "invalid-development.json"
        report.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "kind": "volleycut-feature-experiment-development",
                    "testLabelsUsed": True,
                }
            ),
            encoding="utf-8",
        )

        with patch(
            "analysis.feature_experiments.load_manifest",
            side_effect=AssertionError("invalid provenance must fail before manifest/test access"),
        ):
            with self.assertRaisesRegex(
                FeatureExperimentError, "unopened-test experiment"
            ):
                run_fixed_split_final_test(
                    self.manifest.path,
                    report,
                    self.root / "cache",
                    self.root / "model",
                )


if __name__ == "__main__":
    unittest.main()
