from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from analysis.cli import build_parser
from analysis.config import DecoderConfig, FeatureConfig, TrainingConfig
from analysis.features import FeatureSequence, VideoMetadata
from analysis.model import (
    RALLY_LIVE_TASK,
    SERVE_CONTACT_TASK,
    STACKED_RALLY_TASK,
    LogisticModel,
    ModelError,
    load_model,
)
from analysis.pipeline import PreparedRecording, evaluate_dataset
from analysis.schema import Interval, Recording
from analysis.serve import ServeCompositionConfig, ServeDecoderConfig
from analysis.stacked_serve_experiment import (
    STACKED_FEATURE_NAME,
    _crossfit_training_scores,
    _metric_delta,
    _stack_prepared,
    _validate_stacked_models,
    evaluate_stacked_rally_dataset,
    train_stacked_rally_dataset,
)


def model(
    task: str,
    feature_names: tuple[str, ...],
    artifact: str,
    training: dict[str, object] | None = None,
) -> LogisticModel:
    return LogisticModel(
        feature_config=FeatureConfig(
            use_optical_flow=False,
            use_advanced_visual=False,
            use_audio=False,
            context_offsets_seconds=(0.0,),
            sequence_normalization="none",
        ),
        feature_names=feature_names,
        mean=np.zeros(len(feature_names), dtype=np.float32),
        scale=np.ones(len(feature_names), dtype=np.float32),
        weights=np.ones(len(feature_names), dtype=np.float32),
        bias=0.0,
        decoder=DecoderConfig(),
        training_summary=dict(training or {}),
        artifact_sha256=artifact,
        prediction_task=task,
    )


def prepared(recording_id: str, source_group: str, offset: float = 0.0) -> PreparedRecording:
    times = np.arange(4, dtype=np.float64)
    sequence = FeatureSequence(
        times=times,
        values=(np.arange(4, dtype=np.float32) + offset)[:, None],
        names=("feature",),
        metadata=VideoMetadata(4.0, 1280, 720, 30.0, 120, False),
    )
    recording = Recording(
        id=recording_id,
        video=Path(f"{recording_id}.mp4"),
        split="train",
        source_group=source_group,
        environment="indoor",
        game={},
        rallies=(Interval(1.0, 2.0),),
        ignored_intervals=(),
        roi=None,
        capture={},
        consent={"analyze": True, "train": True},
        content_sha256=None,
        raw={"rallies": [{"start": 1.0, "end": 2.0}]},
    )
    return PreparedRecording(
        recording=recording,
        sequence=sequence,
        contextual_values=sequence.values,
        contextual_names=sequence.names,
        labels=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        sample_mask=np.asarray([True, True, False, True]),
    )


def model_set() -> tuple[LogisticModel, LogisticModel, LogisticModel, LogisticModel]:
    manifest = "manifest"
    baseline = model(
        RALLY_LIVE_TASK,
        ("feature",),
        "baseline",
        {"manifestSha256": manifest},
    )
    serve = model(
        SERVE_CONTACT_TASK,
        ("feature",),
        "serve",
        {
            "manifestSha256": manifest,
            "rallyModelSha256": "baseline",
            "serveDecoder": ServeDecoderConfig().to_dict(),
            "composition": ServeCompositionConfig(
                1.0, 2.0, 5.0, DecoderConfig()
            ).to_dict(),
            "bestEpoch": 2,
            "config": TrainingConfig(epochs=3, patience=2).to_dict(),
            "serveTarget": {"radiusSeconds": 0.0},
        },
    )
    control = model(
        RALLY_LIVE_TASK,
        ("feature",),
        "control",
        {
            "manifestSha256": manifest,
            "baselineRallyModelSha256": "baseline",
        },
    )
    stacked = model(
        STACKED_RALLY_TASK,
        ("feature", STACKED_FEATURE_NAME),
        "stacked",
        {
            "manifestSha256": manifest,
            "baselineRallyModelSha256": "baseline",
            "serveModelSha256": "serve",
            "controlModelSha256": "control",
        },
    )
    return baseline, serve, control, stacked


class StackedFeatureTests(unittest.TestCase):
    def test_appends_serve_probability_and_preserves_mask(self) -> None:
        item = prepared("one", "group-one")
        serve = model(SERVE_CONTACT_TASK, ("feature",), "serve")

        stacked = _stack_prepared([item], serve)[0]

        self.assertEqual(stacked.contextual_names, ("feature", STACKED_FEATURE_NAME))
        self.assertEqual(stacked.contextual_values.shape, (4, 2))
        self.assertEqual(stacked.contextual_values.dtype, np.float32)
        np.testing.assert_array_equal(stacked.contextual_values[:, 0], item.contextual_values[:, 0])
        np.testing.assert_allclose(
            stacked.contextual_values[:, 1], serve.predict(item.contextual_values)
        )
        np.testing.assert_array_equal(stacked.sample_mask, item.sample_mask)

    def test_crossfit_holds_out_each_training_source_group(self) -> None:
        items = [
            prepared("one", "group-one", 0.0),
            prepared("two", "group-two", 0.2),
            prepared("three", "group-three", 0.4),
        ]
        _, serve, _, _ = model_set()

        scores, folds = _crossfit_training_scores(items, serve)

        self.assertEqual(set(scores), {"one", "two", "three"})
        self.assertTrue(all(value.shape == (4,) for value in scores.values()))
        self.assertEqual(
            {fold["heldOutSourceGroup"] for fold in folds},
            {"group-one", "group-two", "group-three"},
        )
        for fold in folds:
            self.assertNotIn(fold["heldOutSourceGroup"], fold["trainingSourceGroups"])

    def test_crossfit_never_supplies_held_out_rows_to_fold_trainer(self) -> None:
        items = [
            prepared("one", "group-one", 0.0),
            prepared("two", "group-two", 10.0),
            prepared("three", "group-three", 20.0),
        ]
        _, serve, _, _ = model_set()
        trained_offsets: list[set[float]] = []

        def fake_train(values, *_args, **_kwargs):
            trained_offsets.append({float(matrix[0, 0]) for matrix in values})
            fold = model(SERVE_CONTACT_TASK, ("feature",), "fold")
            fold.training_summary.update({"bestEpoch": 2, "epochsCompleted": 2})
            return fold

        with patch(
            "analysis.stacked_serve_experiment.train_logistic_model",
            side_effect=fake_train,
        ):
            _, folds = _crossfit_training_scores(items, serve)

        expected = {
            "group-one": {10.0, 20.0},
            "group-two": {0.0, 20.0},
            "group-three": {0.0, 10.0},
        }
        for offsets, fold in zip(trained_offsets, folds, strict=True):
            self.assertEqual(offsets, expected[fold["heldOutSourceGroup"]])

    def test_model_set_requires_exact_roles_signatures_and_hashes(self) -> None:
        baseline, serve, control, stacked = model_set()

        _validate_stacked_models(
            baseline, serve, control, stacked, manifest_sha256="manifest"
        )
        stacked.training_summary["serveModelSha256"] = "wrong"
        with self.assertRaisesRegex(ModelError, "different serve specialist"):
            _validate_stacked_models(baseline, serve, control, stacked)

    def test_stacked_task_round_trips_and_standard_evaluate_rejects_it(self) -> None:
        _, _, _, stacked = model_set()
        with tempfile.TemporaryDirectory(prefix="volleycut-stacked-task-") as directory:
            path = stacked.save(Path(directory) / "model")
            restored = load_model(path)
            self.assertEqual(restored.prediction_task, STACKED_RALLY_TASK)
            with self.assertRaisesRegex(ModelError, "must predict rally-live"):
                evaluate_dataset(
                    Path(directory) / "missing-manifest.json",
                    path,
                    Path(directory) / "cache",
                )

    def test_zero_count_slice_delta_is_null(self) -> None:
        fields = (
            "predictedRallies",
            "matchedRallies",
            "eventPrecision",
            "eventRecall",
            "eventF1",
            "timeIoU",
            "liveTimeRecall",
            "liveTimePrecision",
            "missedLiveSeconds",
            "deadSecondsRetained",
        )
        reference = {field: 0.0 for field in fields}
        candidate = {field: 1.0 for field in fields}
        reference["outcomeSlices"] = {"ace": {"rallies": 0}}
        candidate["outcomeSlices"] = {"ace": {"rallies": 0}}

        result = _metric_delta(candidate, reference)

        self.assertIsNone(result["outcomeStrictRecall"]["ace"])

    def test_public_evaluator_builds_report_and_checks_source_group_leakage(self) -> None:
        baseline, serve, control, stacked = model_set()
        item = prepared("one", "group-one")
        manifest = SimpleNamespace(
            name="synthetic",
            for_split=lambda _split: [item.recording],
        )
        patches = (
            patch(
                "analysis.stacked_serve_experiment.load_model",
                side_effect=[baseline, serve, control, stacked],
            ),
            patch("analysis.stacked_serve_experiment.load_manifest", return_value=manifest),
            patch("analysis.stacked_serve_experiment._manifest_digest", return_value="manifest"),
            patch("analysis.stacked_serve_experiment._prepare_many", return_value=[item]),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            report = evaluate_stacked_rally_dataset(
                "manifest.json", "baseline", "serve", "control", "stacked", "cache"
            )

        self.assertEqual(report["matching"], {"minimumIntervalIoU": 0.5})
        self.assertEqual(report["stackedFeature"]["featureCount"], 2)
        self.assertEqual(report["trainingScoreProtocol"], None)

        baseline.training_summary["trainingSourceGroups"] = ["group-one"]
        with (
            patch(
                "analysis.stacked_serve_experiment.load_model",
                side_effect=[baseline, serve, control, stacked],
            ),
            patch("analysis.stacked_serve_experiment.load_manifest", return_value=manifest),
            patch("analysis.stacked_serve_experiment._manifest_digest", return_value="manifest"),
            patch("analysis.stacked_serve_experiment._prepare_many") as prepare_many,
        ):
            with self.assertRaisesRegex(ModelError, "leaks trained/tuned source groups"):
                evaluate_stacked_rally_dataset(
                    "manifest.json", "baseline", "serve", "control", "stacked", "cache"
                )
        prepare_many.assert_not_called()

    def test_cli_parses_stacked_commands(self) -> None:
        train = build_parser().parse_args(
            [
                "train-rally-with-serve",
                "--manifest", "manifest.json",
                "--baseline-rally-model", "baseline",
                "--serve-model", "serve",
                "--control-model", "control",
                "--model", "stacked",
            ]
        )
        evaluate = build_parser().parse_args(
            [
                "evaluate-rally-with-serve",
                "--manifest", "manifest.json",
                "--baseline-rally-model", "baseline",
                "--serve-model", "serve",
                "--control-model", "control",
                "--model", "stacked",
            ]
        )

        self.assertEqual(train.command, "train-rally-with-serve")
        self.assertEqual(evaluate.command, "evaluate-rally-with-serve")

    def test_training_rejects_nested_artifact_destinations_before_loading(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-stack-destinations-") as directory:
            control = Path(directory) / "control"
            with (
                patch("analysis.stacked_serve_experiment.load_model") as load,
                self.assertRaisesRegex(ModelError, "destinations must be separate"),
            ):
                train_stacked_rally_dataset(
                    "manifest.json",
                    "baseline",
                    "serve",
                    control,
                    control / "stacked",
                    "cache",
                )
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
