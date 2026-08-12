from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.cli import build_parser
from analysis.config import DecoderConfig, FeatureConfig
from analysis.features import FeatureSequence, VideoMetadata
from analysis.model import (
    RALLY_LIVE_TASK,
    SERVE_CONTACT_TASK,
    STACKED_RALLY_TASK,
    LogisticModel,
    ModelError,
)
from analysis.pipeline import PreparedRecording
from analysis.schema import Interval, Recording
from analysis.serve import (
    ServeCompositionConfig,
    ServeDecoderConfig,
    ServeDetection,
    compose_serve_anchored_intervals,
)
from analysis.serve_evidence_experiment import (
    PEAK_FEATURE_NAMES,
    _gated_composition,
    _stack_peak_features,
    _validate_models,
    _validation_decision,
    serve_peak_features,
    train_serve_evidence_dataset,
)


def _model(
    task: str,
    names: tuple[str, ...],
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
        feature_names=names,
        mean=np.zeros(len(names), dtype=np.float32),
        scale=np.ones(len(names), dtype=np.float32),
        weights=np.ones(len(names), dtype=np.float32),
        bias=0.0,
        decoder=DecoderConfig(),
        training_summary=dict(training or {}),
        artifact_sha256=artifact,
        prediction_task=task,
    )


def _prepared() -> PreparedRecording:
    times = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    sequence = FeatureSequence(
        times=times,
        values=np.arange(4, dtype=np.float32)[:, None],
        names=("feature",),
        metadata=VideoMetadata(4.0, 1280, 720, 1.0, 4, False),
    )
    recording = Recording(
        id="recording",
        video=Path("recording.mp4"),
        split="train",
        source_group="group",
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
        sample_mask=np.ones(4, dtype=np.bool_),
    )


def _model_set() -> tuple[LogisticModel, LogisticModel, LogisticModel, LogisticModel]:
    manifest = "manifest"
    decoder = ServeDecoderConfig(0.5, 8.0, 0.25)
    composition = ServeCompositionConfig(1.0, 2.0, 5.0, DecoderConfig())
    baseline = _model(
        RALLY_LIVE_TASK, ("feature",), "baseline", {"manifestSha256": manifest}
    )
    serve = _model(
        SERVE_CONTACT_TASK,
        ("feature",),
        "serve",
        {
            "manifestSha256": manifest,
            "rallyModelSha256": "baseline",
            "serveDecoder": decoder.to_dict(),
            "composition": composition.to_dict(),
        },
    )
    control = _model(
        RALLY_LIVE_TASK,
        ("feature",),
        "control",
        {
            "manifestSha256": manifest,
            "baselineRallyModelSha256": "baseline",
        },
    )
    peak = _model(
        STACKED_RALLY_TASK,
        ("feature", *PEAK_FEATURE_NAMES),
        "peak",
        {
            "manifestSha256": manifest,
            "modelVariant": "rally-with-serve-peak-window-v1",
            "trainingSourceGroups": ["group-one", "group-two"],
            "baselineRallyModelSha256": "baseline",
            "serveModelSha256": "serve",
            "controlModelSha256": "control",
            "peakFeatureTransform": {
                "id": "decoded-serve-peak-window-v1",
                "names": list(PEAK_FEATURE_NAMES),
                "radiusSeconds": 2.0,
                "serveDecoder": decoder.to_dict(),
                "trainingScoreProtocol": "leave-one-training-source-group-out-v1",
                "validationAndInferenceScoreProtocol": "frozen-full-training-serve-specialist",
                "crossFitFolds": [
                    {
                        "heldOutSourceGroup": "group-one",
                        "trainingSourceGroups": ["group-two"],
                    },
                    {
                        "heldOutSourceGroup": "group-two",
                        "trainingSourceGroups": ["group-one"],
                    },
                ],
            },
            "evidenceGate": _gated_composition(composition).to_dict(),
        },
    )
    return baseline, serve, control, peak


class ServePeakFeatureTests(unittest.TestCase):
    def test_constructs_symmetric_peak_bundle_with_inclusive_boundaries(self) -> None:
        times = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

        result = serve_peak_features(times, [ServeDetection(0.0, 0.8)])

        np.testing.assert_array_equal(result[:, 0], [1, 1, 1, 1, 1, 0])
        np.testing.assert_allclose(result[:, 1], [0, 0.5, 1, 0.5, 0, 0])
        np.testing.assert_allclose(result[:, 2], [-1, -0.5, 0, 0.5, 1, 0])
        np.testing.assert_allclose(result[:, 3], [0.8, 0.8, 0.8, 0.8, 0.8, 0])
        self.assertEqual(result.dtype, np.float32)

    def test_nearest_peak_wins_then_confidence_breaks_ties(self) -> None:
        times = np.asarray([0.0, 0.25, 1.0])

        result = serve_peak_features(
            times, [ServeDetection(-1.0, 0.7), ServeDetection(1.0, 0.9)]
        )

        self.assertAlmostEqual(float(result[0, 2]), -0.5)
        self.assertAlmostEqual(float(result[0, 3]), 0.9)
        self.assertAlmostEqual(float(result[1, 2]), -0.375)

    def test_no_peaks_produce_finite_zero_columns(self) -> None:
        result = serve_peak_features(np.asarray([0.0, 1.0]), [])

        np.testing.assert_array_equal(result, np.zeros((2, 4), dtype=np.float32))

    def test_oof_score_override_is_decoded_without_calling_full_model(self) -> None:
        item = _prepared()
        serve = _model(SERVE_CONTACT_TASK, ("feature",), "serve")
        scores = np.asarray([0.1, 0.9, 0.2, 0.1], dtype=np.float32)

        with patch.object(serve, "predict", side_effect=AssertionError("must not run")):
            stacked = _stack_peak_features(
                [item],
                serve,
                ServeDecoderConfig(0.5, 0.0, 0.0),
                score_overrides={item.recording.id: scores},
            )[0]

        self.assertEqual(stacked.contextual_names, ("feature", *PEAK_FEATURE_NAMES))
        self.assertEqual(stacked.contextual_values.shape, (4, 5))
        np.testing.assert_array_equal(stacked.sample_mask, item.sample_mask)

    def test_score_overrides_require_exact_ids_and_bounded_values(self) -> None:
        item = _prepared()
        serve = _model(SERVE_CONTACT_TASK, ("feature",), "serve")

        with self.assertRaisesRegex(ModelError, "missing=.*recording"):
            _stack_peak_features(
                [item],
                serve,
                ServeDecoderConfig(),
                score_overrides={},
            )
        with self.assertRaisesRegex(ModelError, "scores are invalid"):
            _stack_peak_features(
                [item],
                serve,
                ServeDecoderConfig(),
                score_overrides={item.recording.id: np.full(4, np.nan)},
            )


class EvidenceGateTests(unittest.TestCase):
    def test_gate_fixes_two_second_association_and_removes_fallback(self) -> None:
        original = ServeCompositionConfig(1.0, 2.0, 5.0, DecoderConfig())

        gated = _gated_composition(original)

        self.assertEqual(gated.association_seconds, 2.0)
        self.assertEqual(gated.fallback_seconds, 0.0)
        self.assertEqual(gated.max_rescue_seconds, original.max_rescue_seconds)

    def test_unsupported_peak_vanishes_but_permissive_evidence_survives(self) -> None:
        config = _gated_composition(
            ServeCompositionConfig(1.0, 2.0, 5.0, DecoderConfig())
        )

        unsupported = compose_serve_anchored_intervals(
            [], [], [ServeDetection(5.0, 0.9)], 20.0, config, sample_seconds=0.25
        )
        supported = compose_serve_anchored_intervals(
            [],
            [Interval(6.9, 7.5)],
            [ServeDetection(5.0, 0.9)],
            20.0,
            config,
            sample_seconds=0.25,
        )
        too_far = compose_serve_anchored_intervals(
            [],
            [Interval(7.1, 7.5)],
            [ServeDetection(5.0, 0.9)],
            20.0,
            config,
            sample_seconds=0.25,
        )

        self.assertEqual(unsupported, [])
        self.assertEqual((supported[0].start, supported[0].end), (5.0, 7.5))
        self.assertEqual(too_far, [])


class ServeEvidenceArtifactTests(unittest.TestCase):
    def test_model_set_requires_exact_bundle_and_lineage(self) -> None:
        baseline, serve, control, peak = _model_set()

        _validate_models(
            baseline, serve, control, peak, manifest_sha256="manifest"
        )
        peak.training_summary["peakFeatureTransform"]["radiusSeconds"] = 1.0  # type: ignore[index]
        with self.assertRaisesRegex(ModelError, "different peak radius"):
            _validate_models(baseline, serve, control, peak)

    def test_validation_decision_rejects_live_recall_and_ordinary_regression(self) -> None:
        def metrics(
            *, matched: int, f1: float, live: float, short: int, ordinary: int
        ) -> dict[str, object]:
            return {
                "matchedRallies": matched,
                "eventF1": f1,
                "liveTimeRecall": live,
                "outcomeSlices": {
                    "shortAtMost3Seconds": {
                        "rallies": 10,
                        "strictMatchRecall": short / 10,
                    },
                    "ordinaryLong": {
                        "rallies": 20,
                        "strictMatchRecall": ordinary / 20,
                    },
                },
            }

        control = metrics(matched=20, f1=0.6, live=0.9, short=2, ordinary=18)
        v4 = metrics(matched=21, f1=0.5, live=0.9, short=4, ordinary=18)
        gated = metrics(matched=20, f1=0.7, live=0.9, short=3, ordinary=18)
        peak = metrics(matched=22, f1=0.7, live=0.8, short=5, ordinary=17)
        clamped = metrics(matched=19, f1=0.5, live=0.7, short=2, ordinary=17)

        decision = _validation_decision(control, v4, gated, peak, clamped)  # type: ignore[arg-type]

        self.assertFalse(decision["gate"]["promote"])
        self.assertFalse(decision["peakWindowRallyHead"]["promote"])

    def test_cli_parses_serve_evidence_commands(self) -> None:
        common = [
            "--manifest",
            "manifest.json",
            "--baseline-rally-model",
            "baseline",
            "--serve-model",
            "serve",
            "--control-model",
            "control",
            "--model",
            "peak",
        ]

        train = build_parser().parse_args(["train-serve-evidence", *common])
        evaluate = build_parser().parse_args(["evaluate-serve-evidence", *common])

        self.assertEqual(train.command, "train-serve-evidence")
        self.assertEqual(evaluate.command, "evaluate-serve-evidence")

    def test_training_rejects_report_inside_model_before_loading(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-serve-evidence-") as directory:
            model_path = Path(directory) / "model"
            with (
                patch("analysis.serve_evidence_experiment.load_model") as load,
                self.assertRaisesRegex(ModelError, "must not be inside"),
            ):
                train_serve_evidence_dataset(
                    "manifest.json",
                    "baseline",
                    "serve",
                    "control",
                    model_path,
                    "cache",
                    output_path=model_path / "report.json",
                )
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
