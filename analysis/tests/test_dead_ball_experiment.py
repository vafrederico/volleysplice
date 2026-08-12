from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.cli import build_parser
from analysis.config import (
    NOISE_NORMALIZED_AUDIO_FEATURE_SET,
    DecoderConfig,
    FeatureConfig,
    TrainingConfig,
)
from analysis.dead_ball import DeadBallDecoderConfig, DeadBallDetection
from analysis.dead_ball_experiment import (
    LEGACY_AUDIO_DEAD_BALL_INPUT_PROFILE,
    NEW_AUDIO_ONLY_DEAD_BALL_INPUT_PROFILE,
    NO_LEGACY_AUDIO_DEAD_BALL_INPUT_PROFILE,
    DeadBallCompositionConfig,
    DeadBallInputs,
    _compose_candidate,
    _crossfit_epoch_cap,
    _dead_ball_input_mask,
    _dead_ball_input_profile_metadata,
    _masked_dead_ball_values,
    _validate_model_triplet,
    train_dead_ball_dataset,
)
from analysis.decoder import DecodedInterval
from analysis.features import FeatureSequence, VideoMetadata, feature_names
from analysis.model import (
    DEAD_BALL_TASK,
    RALLY_LIVE_TASK,
    SERVE_CONTACT_TASK,
    LogisticModel,
    ModelError,
    load_model,
)
from analysis.pipeline import PreparedRecording, evaluate_dataset
from analysis.schema import Interval, Recording
from analysis.serve import ServeCompositionConfig, ServeDecoderConfig, ServeDetection


def prepared(recording_id: str = "one", group: str = "group-one") -> PreparedRecording:
    times = np.arange(0.0, 30.0, 0.25, dtype=np.float64)
    sequence = FeatureSequence(
        times=times,
        values=np.arange(len(times), dtype=np.float32)[:, None],
        names=("feature",),
        metadata=VideoMetadata(30.0, 1280, 720, 30.0, 900, False),
    )
    recording = Recording(
        id=recording_id,
        video=Path(f"{recording_id}.mp4"),
        split="train",
        source_group=group,
        environment="indoor",
        game={},
        rallies=(Interval(10.0, 12.0),),
        ignored_intervals=(),
        roi=None,
        capture={},
        consent={"analyze": True, "train": True},
        content_sha256=None,
        raw={"rallies": [{"start": 10.0, "end": 12.0}]},
    )
    return PreparedRecording(
        recording,
        sequence,
        sequence.values,
        sequence.names,
        np.asarray((times >= 10.0) & (times < 12.0), dtype=np.float32),
        np.ones(len(times), dtype=bool),
    )


def inputs(
    *,
    primary: tuple[DecodedInterval, ...] = (DecodedInterval(10.0, 15.0, 0.8),),
    permissive: tuple[DecodedInterval, ...] = (DecodedInterval(9.75, 13.0, 0.7),),
    serves: tuple[ServeDetection, ...] = (ServeDetection(10.0, 0.9),),
    v4: tuple[DecodedInterval, ...] = (DecodedInterval(10.0, 12.0, 0.9),),
) -> DeadBallInputs:
    item = prepared()
    return DeadBallInputs(
        item,
        primary,
        permissive,
        serves,
        v4,
        np.zeros(len(item.sequence.times), dtype=np.float32),
    )


def model(task: str, artifact: str, summary: dict[str, object]) -> LogisticModel:
    return LogisticModel(
        feature_config=FeatureConfig(
            use_optical_flow=False,
            use_advanced_visual=False,
            use_audio=False,
            context_offsets_seconds=(0.0,),
            sequence_normalization="none",
        ),
        feature_names=("feature",),
        mean=np.zeros(1, dtype=np.float32),
        scale=np.ones(1, dtype=np.float32),
        weights=np.ones(1, dtype=np.float32),
        bias=0.0,
        decoder=DecoderConfig(),
        training_summary=dict(summary),
        artifact_sha256=artifact,
        prediction_task=task,
    )


def model_triplet() -> tuple[LogisticModel, LogisticModel, LogisticModel]:
    manifest = "manifest"
    rally = model(RALLY_LIVE_TASK, "rally", {"manifestSha256": manifest})
    serve = model(
        SERVE_CONTACT_TASK,
        "serve",
        {
            "manifestSha256": manifest,
            "rallyModelSha256": "rally",
            "serveDecoder": ServeDecoderConfig().to_dict(),
            "composition": ServeCompositionConfig(
                1.0, 2.0, 5.0, DecoderConfig()
            ).to_dict(),
        },
    )
    family = {
        "deadBallDecoder": DeadBallDecoderConfig().to_dict(),
        "composition": DeadBallCompositionConfig("gated-learned-end").to_dict(),
    }
    dead = model(
        DEAD_BALL_TASK,
        "dead",
        {
            "manifestSha256": manifest,
            "rallyModelSha256": "rally",
            "serveModelSha256": "serve",
            "deadBallDecoder": DeadBallDecoderConfig().to_dict(),
            "selectedDeadBallDecoder": DeadBallDecoderConfig().to_dict(),
            "selectedComposition": DeadBallCompositionConfig("v4-noop").to_dict(),
            "familySelections": {
                "learned": family,
                "refined": {
                    **family,
                    "composition": DeadBallCompositionConfig("refine-v4-end").to_dict(),
                },
                "safeRefined": {
                    **family,
                    "composition": DeadBallCompositionConfig(
                        "refine-v4-rescue-end"
                    ).to_dict(),
                },
                "fixed": {
                    **family,
                    "composition": DeadBallCompositionConfig(
                        "gated-fixed-duration", fixed_duration_seconds=2.0
                    ).to_dict(),
                },
                "noop": {
                    **family,
                    "composition": DeadBallCompositionConfig("v4-noop").to_dict(),
                },
            },
        },
    )
    return rally, serve, dead


class DeadBallCompositionTests(unittest.TestCase):
    def test_aggressive_refinement_can_trim_and_extend_an_associated_interval(self) -> None:
        config = DeadBallCompositionConfig("refine-v4-end")

        trimmed = _compose_candidate(
            inputs(v4=(DecodedInterval(10.0, 15.0, 0.8),)),
            [DeadBallDetection(12.0, 0.95)],
            config,
        )
        extended = _compose_candidate(
            inputs(v4=(DecodedInterval(10.0, 11.0, 0.8),)),
            [DeadBallDetection(12.0, 0.95)],
            config,
        )

        self.assertEqual((trimmed[0].start, trimmed[0].end), (10.0, 12.0))
        self.assertEqual((extended[0].start, extended[0].end), (10.0, 12.0))

    def test_gated_rescue_never_truncates_an_established_primary(self) -> None:
        original = (DecodedInterval(10.0, 15.0, 0.8),)

        result = _compose_candidate(
            inputs(primary=original),
            [DeadBallDetection(12.0, 0.95)],
            DeadBallCompositionConfig("gated-learned-end"),
        )

        self.assertEqual(result, original)

    def test_safe_refinement_only_changes_v4_rescues(self) -> None:
        config = DeadBallCompositionConfig("refine-v4-rescue-end")
        primary = (DecodedInterval(10.0, 15.0, 0.8),)
        protected = _compose_candidate(
            inputs(primary=primary, v4=primary),
            [DeadBallDetection(12.0, 0.95)],
            config,
        )
        rescue = _compose_candidate(
            inputs(primary=(), v4=(DecodedInterval(10.0, 12.0, 0.8),)),
            [DeadBallDetection(11.5, 0.95)],
            config,
        )

        self.assertEqual(protected, primary)
        self.assertEqual((rescue[0].start, rescue[0].end), (10.0, 11.5))

    def test_missing_end_is_exact_noop_for_primary(self) -> None:
        original = (DecodedInterval(10.0, 15.0, 0.8),)

        result = _compose_candidate(
            inputs(primary=original),
            [],
            DeadBallCompositionConfig("gated-learned-end"),
        )

        self.assertEqual(result, original)

    def test_unused_serve_requires_permissive_live_evidence(self) -> None:
        config = DeadBallCompositionConfig("gated-learned-end")
        detection = [DeadBallDetection(12.0, 0.95)]

        rescued = _compose_candidate(
            inputs(primary=(), permissive=(DecodedInterval(9.75, 12.5, 0.7),)),
            detection,
            config,
        )
        rejected = _compose_candidate(
            inputs(primary=(), permissive=()), detection, config
        )

        self.assertEqual((rescued[0].start, rescued[0].end), (10.0, 12.0))
        self.assertEqual(rejected, ())

    def test_v4_noop_preserves_composed_intervals(self) -> None:
        original = (DecodedInterval(10.0, 12.0, 0.9),)
        result = _compose_candidate(
            inputs(v4=original),
            [DeadBallDetection(11.0, 1.0)],
            DeadBallCompositionConfig("v4-noop"),
        )
        self.assertIs(result, original)

    def test_touching_intervals_remain_distinct(self) -> None:
        result = _compose_candidate(
            inputs(
                primary=(DecodedInterval(0.0, 10.0, 0.8),),
                serves=(ServeDetection(10.0, 0.9),),
                permissive=(DecodedInterval(10.0, 13.0, 0.7),),
            ),
            [DeadBallDetection(12.0, 0.9)],
            DeadBallCompositionConfig("gated-learned-end"),
        )
        self.assertEqual([(item.start, item.end) for item in result], [(0.0, 10.0), (10.0, 12.0)])


class DeadBallArtifactTests(unittest.TestCase):
    def test_training_rejects_report_inside_model_destination_before_loading(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-dead-ball-output-") as directory:
            model_path = Path(directory) / "model"
            with self.assertRaisesRegex(ModelError, "must not be inside"):
                train_dead_ball_dataset(
                    "missing-manifest.json",
                    "missing-rally",
                    "missing-serve",
                    model_path,
                    Path(directory) / "cache",
                    output_path=model_path / "report.json",
                )

    def test_dead_ball_input_profiles_preserve_the_full_signature(self) -> None:
        names = (
            "t+0s/player_motion_mean",
            "t+0s/audio_rms",
            "t+0s/audio_noise_floor",
            "t+0s/audio_noise_removed_broadband",
            "t+0s/audio_noise_normalized_flux",
            "t+0s/audio_band_80_250_snr",
        )
        values = np.arange(12, dtype=np.float32).reshape(2, 6)

        no_legacy = _dead_ball_input_mask(
            names, NO_LEGACY_AUDIO_DEAD_BALL_INPUT_PROFILE
        )
        new_only = _dead_ball_input_mask(
            names, NEW_AUDIO_ONLY_DEAD_BALL_INPUT_PROFILE
        )
        masked = _masked_dead_ball_values(values, new_only)
        legacy = _dead_ball_input_mask(
            names, LEGACY_AUDIO_DEAD_BALL_INPUT_PROFILE
        )

        np.testing.assert_array_equal(
            no_legacy, np.asarray([1, 0, 0, 1, 1, 1], dtype=np.bool_)
        )
        np.testing.assert_array_equal(
            new_only, np.asarray([0, 0, 0, 1, 1, 1], dtype=np.bool_)
        )
        np.testing.assert_array_equal(
            legacy, np.asarray([1, 1, 1, 0, 0, 0], dtype=np.bool_)
        )
        self.assertEqual(masked.shape, values.shape)
        np.testing.assert_array_equal(masked[:, ~new_only], 0.0)
        np.testing.assert_array_equal(masked[:, new_only], values[:, new_only])

    def test_real_normalized_signature_has_expected_dead_ball_ablation_counts(self) -> None:
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
            int(
                np.sum(
                    _dead_ball_input_mask(
                        names, LEGACY_AUDIO_DEAD_BALL_INPUT_PROFILE
                    )
                )
            ),
            450,
        )
        self.assertEqual(
            int(
                np.sum(
                    _dead_ball_input_mask(
                        names, NO_LEGACY_AUDIO_DEAD_BALL_INPUT_PROFILE
                    )
                )
            ),
            455,
        )
        self.assertEqual(
            int(
                np.sum(
                    _dead_ball_input_mask(
                        names, NEW_AUDIO_ONLY_DEAD_BALL_INPUT_PROFILE
                    )
                )
            ),
            70,
        )

    def test_normalized_dead_ball_profile_rejects_legacy_signature(self) -> None:
        with self.assertRaisesRegex(ModelError, "require normalized audio"):
            _dead_ball_input_mask(
                ("t+0s/player_motion_mean", "t+0s/audio_rms"),
                NO_LEGACY_AUDIO_DEAD_BALL_INPUT_PROFILE,
            )

    def test_pre_profile_dead_ball_artifact_reports_full_input_profile(self) -> None:
        _, _, dead = model_triplet()

        profile = _dead_ball_input_profile_metadata(dead)

        self.assertEqual(profile["id"], "full")
        self.assertEqual(profile["retainedInputs"], len(dead.feature_names))
        self.assertEqual(profile["removedInputs"], 0)
        self.assertTrue(profile["legacyDefault"])

    def test_triplet_requires_exact_roles_and_hashes(self) -> None:
        rally, serve, dead = model_triplet()
        _validate_model_triplet(rally, serve, dead, manifest_sha256="manifest")

        dead.training_summary["serveModelSha256"] = "wrong"
        with self.assertRaisesRegex(ModelError, "different serve model"):
            _validate_model_triplet(rally, serve, dead)

    def test_dead_ball_task_round_trips_and_standard_evaluate_rejects_it(self) -> None:
        _, _, dead = model_triplet()
        with tempfile.TemporaryDirectory(prefix="volleycut-dead-ball-") as directory:
            path = dead.save(Path(directory) / "model")
            restored = load_model(path)
            self.assertEqual(restored.prediction_task, DEAD_BALL_TASK)
            with self.assertRaisesRegex(ModelError, "must predict rally-live"):
                evaluate_dataset("missing.json", path, Path(directory) / "cache")

    def test_epoch_crossfit_excludes_held_out_group(self) -> None:
        items = [
            prepared("one", "one"),
            prepared("two", "two"),
            prepared("three", "three"),
        ]
        labels = {item.recording.id: item.labels for item in items}
        rally, _, _ = model_triplet()
        seen: list[set[float]] = []

        def fake_train(values, *_args, **_kwargs):
            seen.append({float(matrix[0, 0]) for matrix in values})
            result = model(DEAD_BALL_TASK, "fold", {})
            result.training_summary.update(
                {"bestEpoch": 2, "epochsCompleted": 2, "bestValidationLoss": 0.1}
            )
            return result

        with patch(
            "analysis.dead_ball_experiment.train_logistic_model", side_effect=fake_train
        ):
            epoch, folds = _crossfit_epoch_cap(
                items, labels, rally, TrainingConfig(epochs=3, patience=2)
            )

        self.assertEqual(epoch, 2)
        self.assertEqual(len(folds), 3)
        self.assertEqual(len(seen), 3)
        for fold in folds:
            self.assertNotIn(fold["heldOutSourceGroup"], fold["trainingSourceGroups"])

    def test_cli_parses_dead_ball_commands(self) -> None:
        train = build_parser().parse_args(
            [
                "train-dead-ball",
                "--manifest", "manifest.json",
                "--rally-model", "rally",
                "--serve-model", "serve",
                "--model", "dead",
                "--dead-ball-input-profile",
                NEW_AUDIO_ONLY_DEAD_BALL_INPUT_PROFILE,
            ]
        )
        evaluate = build_parser().parse_args(
            [
                "evaluate-dead-ball",
                "--manifest", "manifest.json",
                "--rally-model", "rally",
                "--serve-model", "serve",
                "--model", "dead",
                "--retrospective",
            ]
        )
        self.assertEqual(train.command, "train-dead-ball")
        self.assertEqual(
            train.dead_ball_input_profile,
            NEW_AUDIO_ONLY_DEAD_BALL_INPUT_PROFILE,
        )
        self.assertEqual(evaluate.command, "evaluate-dead-ball")
        self.assertTrue(evaluate.retrospective)


if __name__ == "__main__":
    unittest.main()
