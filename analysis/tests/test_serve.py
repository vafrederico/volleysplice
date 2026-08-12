from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.cli import build_parser
from analysis.config import (
    FEATURE_VERSION,
    NOISE_NORMALIZED_AUDIO_FEATURE_SET,
    DecoderConfig,
    FeatureConfig,
)
from analysis.decoder import DecodedInterval
from analysis.features import FeatureSequence, VideoMetadata, feature_names
from analysis.model import (
    RALLY_LIVE_TASK,
    SERVE_CONTACT_TASK,
    LogisticModel,
    ModelError,
    load_model,
)
from analysis.serve import (
    ServeCompositionConfig,
    ServeDecoderConfig,
    ServeDetection,
    compose_serve_anchored_intervals,
    decode_serve_probabilities,
    match_serve_contacts,
    serve_labels_for_times,
)
from analysis.pipeline import PreparedRecording, evaluate_dataset, infer_video
from analysis.schema import Interval, ManifestError, Recording
from analysis.serve_experiment import (
    NEW_AUDIO_ONLY_SERVE_INPUT_PROFILE,
    NO_LEGACY_AUDIO_SERVE_INPUT_PROFILE,
    PairedPrediction,
    _delta,
    _effective_analysis_fps,
    _prediction_inputs,
    _selection_score,
    _serve_metrics,
    _serve_input_profile_metadata,
    _serve_input_mask,
    _slice_truth,
    train_serve_dataset,
    _validate_serve_validation_targets,
    _validate_model_pair,
)


@dataclass(frozen=True)
class IntervalValue:
    start: float
    end: float


class ServeTargetTests(unittest.TestCase):
    def test_evaluate_serve_cli_accepts_retrospective_marker(self) -> None:
        parsed = build_parser().parse_args(
            [
                "evaluate-serve",
                "--manifest",
                "manifest.json",
                "--rally-model",
                "rally",
                "--serve-model",
                "serve",
                "--retrospective",
            ]
        )

        self.assertTrue(parsed.retrospective)

    def test_training_rejects_report_inside_model_before_loading(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-serve-destinations-") as directory:
            model_path = Path(directory) / "model"
            with (
                patch("analysis.serve_experiment.load_model") as load,
                self.assertRaisesRegex(ModelError, "must not be inside"),
            ):
                train_serve_dataset(
                    "manifest.json",
                    "rally",
                    model_path,
                    "cache",
                    output_path=model_path / "report.json",
                )
            load.assert_not_called()

    def test_train_serve_cli_accepts_normalized_audio_ablation(self) -> None:
        parsed = build_parser().parse_args(
            [
                "train-serve",
                "--manifest",
                "manifest.json",
                "--rally-model",
                "rally",
                "--model",
                "serve",
                "--serve-input-profile",
                NO_LEGACY_AUDIO_SERVE_INPUT_PROFILE,
            ]
        )

        self.assertEqual(
            parsed.serve_input_profile, NO_LEGACY_AUDIO_SERVE_INPUT_PROFILE
        )

    def test_serve_input_profiles_ablate_only_the_requested_features(self) -> None:
        names = (
            "t+0s/player_motion_mean",
            "t+0s/audio_rms",
            "t+0s/audio_noise_floor",
            "t+0s/audio_noise_removed_broadband",
            "t+0s/audio_noise_normalized_flux",
            "t+0s/audio_band_80_250_snr",
        )

        no_legacy = _serve_input_mask(
            names, NO_LEGACY_AUDIO_SERVE_INPUT_PROFILE
        )
        new_only = _serve_input_mask(
            names, NEW_AUDIO_ONLY_SERVE_INPUT_PROFILE
        )

        np.testing.assert_array_equal(
            no_legacy, np.asarray([1, 0, 0, 1, 1, 1], dtype=np.bool_)
        )
        np.testing.assert_array_equal(
            new_only, np.asarray([0, 0, 0, 1, 1, 1], dtype=np.bool_)
        )

    def test_normalized_audio_profile_rejects_legacy_signature(self) -> None:
        with self.assertRaisesRegex(ModelError, "require normalized audio"):
            _serve_input_mask(
                ("t+0s/player_motion_mean", "t+0s/audio_rms"),
                NO_LEGACY_AUDIO_SERVE_INPUT_PROFILE,
            )

    def test_real_normalized_signature_has_expected_ablation_counts(self) -> None:
        config = FeatureConfig(
            audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET
        )
        names = tuple(
            f"t{offset:+g}s/{name}"
            for offset in config.context_offsets_seconds
            for name in feature_names(config)
        )

        self.assertEqual(len(names), 520)
        self.assertEqual(
            int(np.sum(_serve_input_mask(names, NO_LEGACY_AUDIO_SERVE_INPUT_PROFILE))),
            455,
        )
        self.assertEqual(
            int(np.sum(_serve_input_mask(names, NEW_AUDIO_ONLY_SERVE_INPUT_PROFILE))),
            70,
        )

    def test_labels_radius_and_nearest_sample_for_between_sample_contact(self) -> None:
        times = np.asarray([0.0, 0.25, 0.5, 0.75], dtype=np.float64)

        pulse = serve_labels_for_times(times, [IntervalValue(0.37, 0.6)], 0.0)
        window = serve_labels_for_times(times, [IntervalValue(0.5, 0.6)], 0.25)

        np.testing.assert_array_equal(pulse, np.asarray([0, 1, 0, 0], dtype=np.float32))
        np.testing.assert_array_equal(window, np.asarray([0, 1, 1, 1], dtype=np.float32))

    def test_rejects_unordered_sampling_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            serve_labels_for_times(
                np.asarray([0.0, 0.5, 0.25]), [IntervalValue(0.2, 0.3)], 0.5
            )

    def test_validation_requires_contact_and_non_contact_samples(self) -> None:
        for labels in (
            [np.zeros(4, dtype=np.float32)],
            [np.ones(4, dtype=np.float32)],
            [np.asarray([], dtype=np.float32)],
        ):
            with self.subTest(labels=labels):
                with self.assertRaisesRegex(ManifestError, "both contact-window"):
                    _validate_serve_validation_targets(labels)

        _validate_serve_validation_targets(
            [np.asarray([0.0, 1.0], dtype=np.float32)]
        )

    def test_effective_analysis_fps_uses_actual_sample_cadence(self) -> None:
        times = np.asarray([0.0, 1.0 / 3.75, 2.0 / 3.75], dtype=np.float64)
        sequence = FeatureSequence(
            times=times,
            values=np.zeros((len(times), 1), dtype=np.float32),
            names=("motion",),
            metadata=VideoMetadata(1.0, 1280, 720, 30.0, 30, False),
        )
        prepared = PreparedRecording(
            recording=Recording(
                id="cadence",
                video=Path("cadence.mp4"),
                split="validation",
                source_group="cadence-source",
                environment="indoor",
                game={},
                rallies=(),
                ignored_intervals=(),
                roi=None,
                capture={},
                consent={"analyze": True, "train": True},
                content_sha256=None,
                raw={"rallies": []},
            ),
            sequence=sequence,
            contextual_values=sequence.values,
            contextual_names=sequence.names,
            labels=np.zeros(len(times), dtype=np.float32),
            sample_mask=np.ones(len(times), dtype=np.bool_),
        )

        self.assertAlmostEqual(_effective_analysis_fps(prepared), 3.75)


class ServeDecoderTests(unittest.TestCase):
    def test_collapses_islands_and_applies_score_first_nms(self) -> None:
        times = np.arange(8, dtype=np.float64)
        probabilities = np.asarray([0.1, 0.8, 0.9, 0.1, 0.85, 0.1, 0.95, 0.1])

        detections = decode_serve_probabilities(
            times,
            probabilities,
            ServeDecoderConfig(threshold=0.8, min_separation_seconds=3.0),
        )

        self.assertEqual([(item.time, item.confidence) for item in detections], [(2.0, 0.9), (6.0, 0.95)])

    def test_peak_ties_use_earliest_time_and_offset_is_clipped(self) -> None:
        detections = decode_serve_probabilities(
            np.asarray([0.0, 0.5, 1.0]),
            np.asarray([0.9, 0.9, 0.1]),
            ServeDecoderConfig(0.8, 0.0, -0.5),
            duration=1.5,
        )

        self.assertEqual(detections, [ServeDetection(0.0, 0.9)])

    def test_contact_matching_prioritizes_lower_error_after_match_count(self) -> None:
        matches = match_serve_contacts(
            [1.0, 2.0], [ServeDetection(1.6, 0.9)], tolerance_seconds=0.6
        )

        self.assertEqual([(truth, predicted) for truth, predicted, _ in matches], [(1, 0)])
        self.assertAlmostEqual(matches[0][2], -0.4)


class ServeCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ServeCompositionConfig(
            association_seconds=1.0,
            fallback_seconds=2.0,
            max_rescue_seconds=5.0,
            permissive_decoder=DecoderConfig(
                smoothing_seconds=0.5,
                enter_threshold=0.4,
                exit_threshold=0.3,
                min_live_seconds=0.25,
                bridge_gap_seconds=0.0,
            ),
        )

    def test_anchors_rescues_and_clips_without_splitting_primary(self) -> None:
        composed = compose_serve_anchored_intervals(
            [DecodedInterval(5.0, 10.0, 0.7)],
            [DecodedInterval(20.0, 21.25, 0.55)],
            [
                ServeDetection(4.5, 0.9),
                ServeDetection(7.0, 0.99),
                ServeDetection(20.0, 0.8),
                ServeDetection(30.0, 0.75),
            ],
            31.0,
            self.config,
            sample_seconds=0.25,
        )

        self.assertEqual(
            [(item.start, item.end) for item in composed],
            [(4.5, 10.0), (20.0, 21.25), (30.0, 31.0)],
        )

    def test_overlapping_rescue_and_primary_are_merged(self) -> None:
        composed = compose_serve_anchored_intervals(
            [DecodedInterval(10.0, 15.0, 0.7)],
            [],
            [ServeDetection(8.5, 0.9)],
            20.0,
            self.config,
            sample_seconds=0.25,
        )

        self.assertEqual([(item.start, item.end) for item in composed], [(8.5, 15.0)])

    def test_touching_primary_and_rescue_remain_distinct_half_open_events(self) -> None:
        composed = compose_serve_anchored_intervals(
            [DecodedInterval(0.0, 2.0, 0.7)],
            [DecodedInterval(2.0, 3.0, 0.6)],
            [ServeDetection(2.0, 0.9)],
            5.0,
            self.config,
            sample_seconds=0.25,
        )

        self.assertEqual(
            [(item.start, item.end) for item in composed], [(0.0, 2.0), (2.0, 3.0)]
        )

    def test_selection_and_serve_metrics_exclude_ignored_region(self) -> None:
        recording = Recording(
            id="ignored-test",
            video=Path("ignored-test.mp4"),
            split="validation",
            source_group="ignored-source",
            environment="indoor",
            game={},
            rallies=(Interval(1.0, 2.0),),
            ignored_intervals=(Interval(3.0, 4.0),),
            roi=None,
            capture={},
            consent={"analyze": True, "train": True},
            content_sha256=None,
            raw={"rallies": [{"start": 1.0, "end": 2.0, "tags": []}]},
        )
        sequence = FeatureSequence(
            times=np.asarray([0.0, 1.0]),
            values=np.zeros((2, 1), dtype=np.float32),
            names=("motion",),
            metadata=VideoMetadata(10.0, 1280, 720, 30.0, 300, False),
        )
        prepared = PreparedRecording(
            recording=recording,
            sequence=sequence,
            contextual_values=sequence.values,
            contextual_names=sequence.names,
            labels=np.asarray([0.0, 1.0], dtype=np.float32),
            sample_mask=np.ones(2, dtype=np.bool_),
        )
        prediction = PairedPrediction(
            prepared=prepared,
            primary=(),
            serves=(ServeDetection(1.0, 0.9), ServeDetection(3.5, 0.99)),
            composed=(DecodedInterval(3.2, 3.8, 0.99),),
        )

        _, selection = _selection_score([prediction])
        serve_metrics = _serve_metrics([prediction], 0.5)

        self.assertEqual(selection["predictedRallies"], 0)
        self.assertEqual(serve_metrics["predictedServes"], 1)
        self.assertEqual(serve_metrics["matchedServes"], 1)

        clean_key, _ = _selection_score(
            [
                PairedPrediction(
                    prepared=prepared,
                    primary=(),
                    serves=(),
                    composed=(DecodedInterval(1.0, 2.0, 0.8),),
                )
            ]
        )
        early_key, _ = _selection_score(
            [
                PairedPrediction(
                    prepared=prepared,
                    primary=(),
                    serves=(),
                    composed=(DecodedInterval(0.0, 2.0, 0.8),),
                )
            ]
        )
        self.assertGreater(clean_key, early_key)

    def test_zero_count_outcome_delta_is_null(self) -> None:
        fields = (
            "eventPrecision",
            "eventRecall",
            "eventF1",
            "timeIoU",
            "liveTimeRecall",
            "liveTimePrecision",
            "missedLiveSeconds",
            "deadSecondsRetained",
        )
        baseline = {field: 0.0 for field in fields}
        composed = {field: 0.1 for field in fields}
        baseline["outcomeSlices"] = {
            "all": {"rallies": 1, "strictMatchRecall": 0.0},
            "ace": {"rallies": 0},
        }
        composed["outcomeSlices"] = {
            "all": {"rallies": 1, "strictMatchRecall": 1.0},
            "ace": {"rallies": 0},
        }

        result = _delta(composed, baseline)

        self.assertEqual(result["outcomeStrictRecall"]["all"], 1.0)
        self.assertIsNone(result["outcomeStrictRecall"]["ace"])

    def test_serve_slices_use_canonical_interval_tags(self) -> None:
        recording = Recording(
            id="slice-tags",
            video=Path("slice-tags.mp4"),
            split="validation",
            source_group="slice-tags-source",
            environment="indoor",
            game={},
            rallies=(
                Interval(0.0, 4.0, ("service-error",)),
                Interval(10.0, 11.0, ("service-fault",)),
            ),
            ignored_intervals=(),
            roi=None,
            capture={},
            consent={"analyze": True, "train": True},
            content_sha256=None,
            raw={"rallies": []},
        )

        slices = _slice_truth(recording)

        self.assertEqual(len(slices["serviceFault"]), 1)
        self.assertEqual(len(slices["ordinaryLong"]), 1)
        self.assertEqual(slices["serviceFault"][0].tags, ("service-fault",))
        self.assertEqual(slices["ordinaryLong"][0].tags, ("service-error",))


class ServeArtifactTests(unittest.TestCase):
    @staticmethod
    def model(
        task: str,
        *,
        artifact: str,
        rally_artifact: str | None = None,
        feature_version: str = FEATURE_VERSION,
    ) -> LogisticModel:
        training = {
            "manifestSha256": "manifest",
            "serveDecoder": ServeDecoderConfig().to_dict(),
            "composition": ServeCompositionConfig(
                1.0,
                2.0,
                5.0,
                DecoderConfig(min_live_seconds=0.5, bridge_gap_seconds=0.0),
            ).to_dict(),
        }
        if rally_artifact is not None:
            training["rallyModelSha256"] = rally_artifact
        return LogisticModel(
            feature_config=FeatureConfig(
                use_optical_flow=False, context_offsets_seconds=(0.0,)
            ),
            feature_names=("motion",),
            mean=np.zeros(1, dtype=np.float32),
            scale=np.ones(1, dtype=np.float32),
            weights=np.ones(1, dtype=np.float32),
            bias=0.0,
            decoder=DecoderConfig(),
            training_summary=training,
            artifact_sha256=artifact,
            feature_version=feature_version,
            prediction_task=task,
        )

    def test_legacy_artifact_defaults_to_rally_live(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-legacy-task-") as directory:
            model_path = self.model(RALLY_LIVE_TASK, artifact="unused").save(
                Path(directory) / "model"
            )
            metadata_path = model_path / "model.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            del metadata["predictionTask"]
            metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

            restored = load_model(model_path)

        self.assertEqual(restored.prediction_task, RALLY_LIVE_TASK)

    def test_pre_profile_serve_artifact_reports_full_input_profile(self) -> None:
        serve = self.model(SERVE_CONTACT_TASK, artifact="serve")

        profile = _serve_input_profile_metadata(serve)

        self.assertEqual(profile["id"], "full")
        self.assertEqual(profile["retainedInputs"], 1)
        self.assertEqual(profile["removedInputs"], 0)
        self.assertTrue(profile["legacyDefault"])

    def test_save_load_preserves_serve_task_and_rejects_role_swap(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-serve-task-") as directory:
            restored = load_model(
                self.model(SERVE_CONTACT_TASK, artifact="unused").save(Path(directory) / "model")
            )

        self.assertEqual(restored.prediction_task, SERVE_CONTACT_TASK)
        rally = self.model(RALLY_LIVE_TASK, artifact="rally")
        with self.assertRaisesRegex(ModelError, "specialist model"):
            _validate_model_pair(rally, rally)

    def test_pair_is_bound_to_exact_rally_artifact(self) -> None:
        rally = self.model(RALLY_LIVE_TASK, artifact="rally")
        serve = self.model(
            SERVE_CONTACT_TASK, artifact="serve", rally_artifact="different-rally"
        )

        with self.assertRaisesRegex(ModelError, "different rally model"):
            _validate_model_pair(rally, serve)

    def test_pair_rejects_different_feature_versions(self) -> None:
        rally = self.model(RALLY_LIVE_TASK, artifact="rally")
        serve = self.model(
            SERVE_CONTACT_TASK,
            artifact="serve",
            rally_artifact="rally",
            feature_version="court-motion-quality-v1",
        )

        with self.assertRaisesRegex(ModelError, "different feature versions"):
            _validate_model_pair(rally, serve)

    def test_paired_primary_decode_uses_effective_sample_cadence(self) -> None:
        rally = self.model(RALLY_LIVE_TASK, artifact="rally")
        rally.decoder = DecoderConfig(
            min_live_seconds=3.0,
            bridge_gap_seconds=0.0,
            short_event_min_seconds=3.0,
            short_event_threshold=1.0,
        )
        serve = self.model(
            SERVE_CONTACT_TASK, artifact="serve", rally_artifact="rally"
        )
        times = np.arange(12, dtype=np.float64) * (7.0 / 30.0)
        sequence = FeatureSequence(
            times=times,
            values=np.ones((len(times), 1), dtype=np.float32),
            names=("motion",),
            metadata=VideoMetadata(3.0, 1280, 720, 30.0, 90, False),
        )
        prepared = PreparedRecording(
            recording=Recording(
                id="cadence-primary",
                video=Path("cadence-primary.mp4"),
                split="test",
                source_group="cadence-primary-source",
                environment="indoor",
                game={},
                rallies=(),
                ignored_intervals=(),
                roi=None,
                capture={},
                consent={"analyze": True, "train": True},
                content_sha256=None,
                raw={"rallies": []},
            ),
            sequence=sequence,
            contextual_values=sequence.values,
            contextual_names=sequence.names,
            labels=np.zeros(len(times), dtype=np.float32),
            sample_mask=np.ones(len(times), dtype=np.bool_),
        )

        prediction = _prediction_inputs(
            prepared,
            rally,
            serve,
            ServeDecoderConfig(threshold=0.9),
            ServeCompositionConfig(
                1.0,
                0.0,
                5.0,
                DecoderConfig(min_live_seconds=0.5, bridge_gap_seconds=0.0),
            ),
        )

        self.assertEqual(prediction.primary, ())

    def test_noop_specialist_preserves_primary_inference_intervals(self) -> None:
        rally = self.model(RALLY_LIVE_TASK, artifact="rally")
        rally.decoder = DecoderConfig(
            min_live_seconds=3.0,
            bridge_gap_seconds=0.0,
            short_event_min_seconds=3.0,
            short_event_threshold=1.0,
        )
        serve = self.model(
            SERVE_CONTACT_TASK, artifact="serve", rally_artifact="rally"
        )
        rally.feature_names = ("t+0s/motion",)
        serve.feature_names = ("t+0s/motion",)
        serve.weights[:] = 0.0
        serve.bias = -10.0
        times = np.arange(12, dtype=np.float64) * (7.0 / 30.0)
        sequence = FeatureSequence(
            times=times,
            values=np.ones((len(times), 1), dtype=np.float32),
            names=("motion",),
            metadata=VideoMetadata(3.0, 1280, 720, 30.0, 90, False),
        )

        def write_preview_stub(
            _video: Path,
            destination: Path,
            _roi: tuple[float, float, float, float] | None,
        ) -> None:
            Path(destination).write_bytes(b"preview")

        with tempfile.TemporaryDirectory(prefix="volleycut-serve-noop-") as directory:
            root = Path(directory)
            with (
                patch("analysis.pipeline.load_model", side_effect=[rally, rally, serve]),
                patch("analysis.pipeline.extract_features", return_value=sequence),
                patch("analysis.pipeline.write_preview", side_effect=write_preview_stub),
            ):
                primary = infer_video(
                    root / "input.mp4", root / "rally", root / "primary"
                )
                paired = infer_video(
                    root / "input.mp4",
                    root / "rally",
                    root / "paired",
                    serve_model_path=root / "serve",
                )

        self.assertEqual(primary["rallies"], paired["rallies"])
        self.assertEqual(primary["rallies"], [])

    def test_inference_rejects_role_mismatch_before_feature_extraction(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-serve-infer-role-") as directory:
            root = Path(directory)
            rally_path = self.model(RALLY_LIVE_TASK, artifact="unused").save(
                root / "rally"
            )
            wrong_serve_path = self.model(RALLY_LIVE_TASK, artifact="unused").save(
                root / "wrong-serve"
            )
            with patch(
                "analysis.pipeline.extract_features",
                side_effect=AssertionError("role rejection must precede video decoding"),
            ):
                with self.assertRaisesRegex(ModelError, "specialist model"):
                    infer_video(
                        root / "missing.mp4",
                        rally_path,
                        root / "output",
                        serve_model_path=wrong_serve_path,
                    )

    def test_standard_evaluation_rejects_serve_model_before_manifest_loading(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-serve-evaluate-role-") as directory:
            root = Path(directory)
            serve_path = self.model(SERVE_CONTACT_TASK, artifact="unused").save(
                root / "serve"
            )

            with self.assertRaisesRegex(ModelError, "must predict rally-live"):
                evaluate_dataset(
                    root / "missing-manifest.json",
                    serve_path,
                    root / "cache",
                )


if __name__ == "__main__":
    unittest.main()
