from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from analysis.cli import build_parser
from analysis.config import DecoderConfig, FeatureConfig, TrainingConfig
from analysis.dead_state import DeadStateDetection
from analysis.dead_state_experiment import (
    GLOBAL_DEAD_STATE_TARGET_ID,
    DeadStateRefinementConfig,
    _InverseDeadStateModel,
    _eligible_target_mask,
    _global_dead_target,
    _global_dead_rally_decoder,
    _local_target_metrics,
    _refine_v5_ends,
    _target_metadata,
    _transition_target,
    _validate_model_triplet,
    evaluate_dead_state_dataset,
    train_dead_state_dataset,
)
from analysis.decoder import DecodedInterval
from analysis.model import (
    DEAD_STATE_TASK,
    RALLY_LIVE_TASK,
    SERVE_CONTACT_TASK,
    LogisticModel,
    ModelError,
    train_logistic_model,
)
from analysis.schema import Interval
from analysis.serve import ServeCompositionConfig, ServeDecoderConfig


def model(task: str, artifact: str, summary: dict[str, object]) -> LogisticModel:
    return LogisticModel(
        feature_config=FeatureConfig(
            use_optical_flow=False,
            use_advanced_visual=False,
            use_audio=False,
            context_offsets_seconds=(0.0,),
            sequence_normalization="none",
        ),
        feature_names=("kept", "removed"),
        mean=np.zeros(2, dtype=np.float32),
        scale=np.ones(2, dtype=np.float32),
        weights=np.ones(2, dtype=np.float32),
        bias=0.0,
        decoder=DecoderConfig(),
        training_summary=dict(summary),
        artifact_sha256=artifact,
        prediction_task=task,
    )


def model_triplet() -> tuple[LogisticModel, LogisticModel, LogisticModel]:
    rally = model(RALLY_LIVE_TASK, "rally", {"manifestSha256": "manifest"})
    serve = model(
        SERVE_CONTACT_TASK,
        "serve",
        {
            "manifestSha256": "manifest",
            "rallyModelSha256": "rally",
            "serveDecoder": ServeDecoderConfig().to_dict(),
            "composition": ServeCompositionConfig(
                1.0, 2.0, 5.0, DecoderConfig()
            ).to_dict(),
        },
    )
    dead = model(
        DEAD_STATE_TASK,
        "dead-state",
        {
            "manifestSha256": "manifest",
            "rallyModelSha256": "rally",
            "serveModelSha256": "serve",
            "selectedDeadStateDecoder": {
                "deadThreshold": 0.8,
                "liveResetThreshold": 0.4,
                "minimumLiveSamples": 1,
                "minimumDeadSamples": 2,
                "minAfterServeSeconds": 0.0,
                "maxAfterServeSeconds": 2.0,
                "timeOffsetSeconds": 0.0,
            },
            "selectedRefinement": DeadStateRefinementConfig(
                "refine-v5-end", 1.0
            ).to_dict(),
        },
    )
    return rally, serve, dead


class DeadStateExperimentTests(unittest.TestCase):
    def test_transition_target_adds_pre_serve_setup_as_negative(self) -> None:
        times = np.arange(0.0, 8.0, 0.25)
        labels, mask = _transition_target(
            times,
            (Interval(3.0, 5.0),),
            before_end_seconds=2.0,
            after_end_seconds=2.0,
            pre_serve_setup_seconds=1.0,
        )

        np.testing.assert_array_equal(times[mask], np.arange(2.0, 7.0, 0.25))
        self.assertTrue(np.all(labels[(times >= 2.0) & (times < 5.0)] == 0))
        self.assertTrue(np.all(labels[(times >= 5.0) & (times < 7.0)] == 1))

    def test_global_dead_target_is_inverse_rally_live_with_full_mask(self) -> None:
        times = np.arange(0.0, 4.0, 0.5)

        labels, mask = _global_dead_target(
            times,
            (Interval(1.0, 2.0), Interval(3.0, 3.5)),
        )

        np.testing.assert_array_equal(
            labels,
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        )
        np.testing.assert_array_equal(mask, np.ones(len(times), dtype=bool))

    def test_global_target_metadata_marks_algebraic_redundancy(self) -> None:
        metadata = _target_metadata(
            "global-dead",
            before_end_seconds=2.0,
            after_end_seconds=2.0,
            pre_serve_setup_seconds=1.0,
        )

        self.assertEqual(metadata["id"], GLOBAL_DEAD_STATE_TARGET_ID)
        self.assertTrue(metadata["algebraicRedundancy"]["isInverseRallyLiveControl"])
        self.assertIn("p_dead = 1 - p_live", metadata["algebraicRedundancy"]["note"])

    def test_inverse_dead_state_model_returns_complement(self) -> None:
        dead = model(DEAD_STATE_TASK, "dead-state", {})
        values = np.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32)

        np.testing.assert_allclose(
            _InverseDeadStateModel(dead).predict(values),
            1.0 - dead.predict(values),
        )

    def test_global_dead_rally_decoder_supports_legacy_and_validates_metadata(self) -> None:
        dead = model(DEAD_STATE_TASK, "dead-state", {})
        self.assertIsNone(_global_dead_rally_decoder(dead))

        dead.training_summary["globalDeadAsRallyControl"] = {
            "selectedDecoder": DecoderConfig().to_dict(),
            "selection": {"status": "selected-on-validation"},
        }
        decoder, metadata = _global_dead_rally_decoder(dead) or (None, None)
        self.assertEqual(decoder, DecoderConfig())
        self.assertEqual(metadata["selection"]["status"], "selected-on-validation")

        dead.training_summary["globalDeadAsRallyControl"] = {
            "selectedDecoder": {"enter_threshold": 2.0}
        }
        with self.assertRaisesRegex(ModelError, "decoder metadata is invalid"):
            _global_dead_rally_decoder(dead)

        dead.training_summary["globalDeadAsRallyControl"] = {
            "probabilityTransform": "identity",
            "selectedDecoder": DecoderConfig().to_dict(),
        }
        with self.assertRaisesRegex(ModelError, "probability transform is invalid"):
            _global_dead_rally_decoder(dead)

    def test_missing_transition_is_exact_noop_and_transition_is_bidirectional(self) -> None:
        original = (
            DecodedInterval(3.0, 5.0, 0.8),
            DecodedInterval(8.0, 10.0, 0.7),
        )

        refined = _refine_v5_ends(
            original,
            (DeadStateDetection(5.5, 0.9), None),
            duration=20.0,
            sample_seconds=0.25,
        )

        self.assertEqual((refined[0].start, refined[0].end), (3.0, 5.5))
        self.assertIs(refined[1], original[1])

    def test_refinement_never_deletes_an_interval(self) -> None:
        original = (DecodedInterval(3.0, 5.0, 0.8),)

        refined = _refine_v5_ends(
            original,
            (DeadStateDetection(3.0, 0.9),),
            duration=20.0,
            sample_seconds=0.25,
        )

        self.assertEqual(refined, original)

    def test_refinement_config_round_trips(self) -> None:
        original = DeadStateRefinementConfig("refine-v5-end", 1.5)
        restored = DeadStateRefinementConfig.from_dict(original.to_dict())
        self.assertEqual(restored, original)

    def test_ignored_samples_are_excluded_from_the_local_target(self) -> None:
        sample_mask = np.asarray([True, False, True, False])
        target_mask = np.asarray([False, True, True, True])

        eligible = _eligible_target_mask(sample_mask, target_mask)

        np.testing.assert_array_equal(eligible, [False, False, True, False])

    def test_global_target_metrics_use_every_valid_sample(self) -> None:
        dead = model(
            DEAD_STATE_TASK,
            "dead-state",
            {
                "deadStateTarget": _target_metadata(
                    "global-dead",
                    before_end_seconds=2.0,
                    after_end_seconds=2.0,
                    pre_serve_setup_seconds=1.0,
                )
            },
        )
        prepared = SimpleNamespace(
            sequence=SimpleNamespace(times=np.asarray([0.0, 1.0, 2.0, 3.0])),
            recording=SimpleNamespace(rallies=(Interval(1.0, 2.0),)),
            sample_mask=np.asarray([True, True, False, True]),
            contextual_values=np.zeros((4, 2), dtype=np.float32),
        )

        metrics = _local_target_metrics(
            (prepared,), dead, SimpleNamespace(dead_threshold=0.8)
        )

        self.assertEqual(metrics["targetMode"], "global-dead")
        self.assertEqual(metrics["scope"], "all-valid-samples")
        self.assertEqual(metrics["at0.5"]["samples"], 3)
        self.assertEqual(metrics["at0.5"]["positiveSamples"], 2)

    def test_masked_training_zero_embeds_removed_inputs_at_inference(self) -> None:
        raw = np.asarray(
            [[-2.0, 100.0], [-1.0, -80.0], [1.0, 60.0], [2.0, -40.0]],
            dtype=np.float32,
        )
        masked = raw.copy()
        masked[:, 1] = 0.0
        trained = train_logistic_model(
            [masked],
            [np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)],
            [],
            [],
            FeatureConfig(
                use_optical_flow=False,
                use_advanced_visual=False,
                use_audio=False,
                context_offsets_seconds=(0.0,),
                sequence_normalization="none",
            ),
            ("kept", "removed"),
            DecoderConfig(),
            TrainingConfig(epochs=3, patience=3, batch_size=4),
            prediction_task=DEAD_STATE_TASK,
        )

        self.assertEqual(float(trained.mean[1]), 0.0)
        self.assertEqual(float(trained.weights[1]), 0.0)
        np.testing.assert_array_equal(trained.predict(raw), trained.predict(masked))

    def test_artifact_validation_binds_roles_hashes_and_manifest(self) -> None:
        rally, serve, dead = model_triplet()
        _validate_model_triplet(
            rally, serve, dead, manifest_sha256="manifest"
        )

        dead.training_summary["serveModelSha256"] = "wrong"
        with self.assertRaisesRegex(ModelError, "different serve model"):
            _validate_model_triplet(rally, serve, dead)

    def test_evaluation_rejects_trained_group_leakage_before_preparation(self) -> None:
        rally, serve, dead = model_triplet()
        dead.training_summary["trainingSourceGroups"] = ["leak"]
        manifest = SimpleNamespace(
            for_split=lambda _split: (SimpleNamespace(source_group="leak"),),
            name="dataset",
        )
        with (
            patch(
                "analysis.dead_state_experiment.load_model",
                side_effect=[rally, serve, dead],
            ),
            patch(
                "analysis.dead_state_experiment.load_manifest", return_value=manifest
            ),
            patch(
                "analysis.dead_state_experiment._manifest_digest",
                return_value="manifest",
            ),
            patch("analysis.dead_state_experiment._prepare_many") as prepare,
            self.assertRaisesRegex(ModelError, "leaks trained/tuned source groups"),
        ):
            evaluate_dead_state_dataset(
                "manifest.json", "rally", "serve", "dead", "cache"
            )
        prepare.assert_not_called()

    def test_training_rejects_report_inside_model_before_loading(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="volleycut-dead-state-destinations-"
        ) as directory:
            destination = Path(directory) / "model"
            with (
                patch("analysis.dead_state_experiment.load_model") as load,
                self.assertRaisesRegex(ModelError, "must not be inside"),
            ):
                train_dead_state_dataset(
                    "manifest.json",
                    "rally",
                    "serve",
                    destination,
                    "cache",
                    output_path=destination / "report.json",
                )
            load.assert_not_called()

    def test_cli_parses_dead_state_commands_and_retrospective_role(self) -> None:
        default_train = build_parser().parse_args(
            [
                "train-dead-state",
                "--manifest",
                "manifest.json",
                "--rally-model",
                "rally",
                "--serve-model",
                "serve",
                "--model",
                "dead-state",
            ]
        )
        train = build_parser().parse_args(
            [
                "train-dead-state",
                "--manifest",
                "manifest.json",
                "--rally-model",
                "rally",
                "--serve-model",
                "serve",
                "--model",
                "dead-state",
                "--dead-state-input-profile",
                "visual-plus-legacy-audio",
                "--target-mode",
                "global-dead",
            ]
        )
        evaluate = build_parser().parse_args(
            [
                "evaluate-dead-state",
                "--manifest",
                "manifest.json",
                "--rally-model",
                "rally",
                "--serve-model",
                "serve",
                "--model",
                "dead-state",
                "--retrospective",
            ]
        )

        self.assertEqual(default_train.target_mode, "end-transition")
        self.assertEqual(train.command, "train-dead-state")
        self.assertEqual(train.dead_state_input_profile, "visual-plus-legacy-audio")
        self.assertEqual(train.target_mode, "global-dead")
        self.assertEqual(evaluate.command, "evaluate-dead-state")
        self.assertTrue(evaluate.retrospective)
        self.assertEqual(evaluate.cache_dir, Path("data/features"))


if __name__ == "__main__":
    unittest.main()
