from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.config import DecoderConfig, FeatureConfig, TrainingConfig
from analysis.feature_experiments import FeatureExperimentError
from analysis.features import FeatureSequence, VideoMetadata, contextualize, feature_names
from analysis.pipeline import PreparedRecording
from analysis.schema import DatasetManifest, Interval, Recording, labels_for_times
from analysis.transition_feature_experiment import (
    EXPECTED_BASELINE_CONTEXT_COUNT,
    EXPECTED_BASE_SIGNAL_COUNT,
    INTERACTION_DEFINITIONS,
    append_feature_blocks,
    derive_transition_feature_block,
    prepare_transition_candidates,
    run_development_transition_experiments,
    run_retrospective_transition_test,
)


def _recording(
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
        game={"playersPerTeam": 4, "targetPoints": 25},
        rallies=(Interval(2.0, 5.0, tags=("ace",)),),
        ignored_intervals=(),
        roi=(0.05, 0.05, 0.9, 0.9),
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


def _feature_config() -> FeatureConfig:
    return FeatureConfig(
        analysis_fps=4.0,
        resize_width=192,
        resize_height=108,
        grid_size=3,
        use_optical_flow=True,
        use_advanced_visual=True,
        use_audio=True,
        audio_sample_rate=16000,
        context_offsets_seconds=(-2.0, -1.0, 0.0, 1.0, 2.0),
        sequence_normalization="percentile-rank",
    )


def _prepared(
    item: Recording,
    config: FeatureConfig,
    *,
    audio_available: float = 1.0,
) -> PreparedRecording:
    times = np.arange(0.0, 8.25, 0.25, dtype=np.float64)
    names = feature_names(config)
    phase = sum(item.id.encode("utf-8")) % 11 / 10.0
    columns: list[np.ndarray] = []
    for index, name in enumerate(names):
        values = (
            0.45
            + 0.25 * np.sin(times * (0.25 + index % 5 * 0.08) + phase)
            + 0.01 * (index % 7)
        )
        if name == "audio_available":
            values = np.full(len(times), audio_available)
        elif name == "visibility_quality":
            values = np.clip(0.85 - 0.05 * np.cos(times), 0.0, 1.0)
        elif name in {
            "audio_contact_like_transient",
            "audio_onset_strength",
            "player_motion_onset",
        }:
            values = np.where((times >= 2.0) & (times <= 2.5), 1.0, 0.05)
        elif name in {
            "player_motion_collapse",
            "synchronized_stand_down",
            "audio_cadence_collapse",
        }:
            values = np.where((times >= 4.75) & (times <= 5.5), 0.95, 0.05)
        columns.append(values)
    sequence = FeatureSequence(
        times=times,
        values=np.column_stack(columns).astype(np.float32),
        names=names,
        metadata=VideoMetadata(8.25, 1920, 1080, 30.0, 248, True),
    )
    contextual_values, contextual_names = contextualize(sequence, config)
    return PreparedRecording(
        recording=item,
        sequence=sequence,
        contextual_values=contextual_values,
        contextual_names=contextual_names,
        labels=labels_for_times(times, item.rallies),
        sample_mask=np.ones(len(times), dtype=np.bool_),
    )


class TransitionFeatureTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="volleycut-transition-feature-test-"
        )
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.config = _feature_config()
        self.item = _recording(self.root, "dev-a", "a", "train")
        self.prepared = _prepared(self.item, self.config)

    def test_derivation_is_deterministic_and_appended_once(self) -> None:
        first = derive_transition_feature_block(self.prepared, self.config)
        second = derive_transition_feature_block(self.prepared, self.config)
        augmented = append_feature_blocks(self.prepared, first)

        np.testing.assert_array_equal(first.values, second.values)
        self.assertEqual(first.names, second.names)
        self.assertEqual(len(self.prepared.sequence.names), EXPECTED_BASE_SIGNAL_COUNT)
        self.assertEqual(
            len(self.prepared.contextual_names), EXPECTED_BASELINE_CONTEXT_COUNT
        )
        self.assertEqual(
            augmented.contextual_values.shape[1],
            EXPECTED_BASELINE_CONTEXT_COUNT + len(first.names),
        )
        self.assertEqual(
            augmented.contextual_names[-len(first.names) :], first.names
        )
        self.assertTrue(
            all(not name.startswith("t+") and not name.startswith("t-") for name in first.names)
        )
        self.assertIn("multiscale", first.groups)
        self.assertEqual(
            len([key for key in first.groups if key.startswith("interaction:")]),
            len(INTERACTION_DEFINITIONS),
        )
        self.assertEqual(
            first.definitions["appendPolicy"],
            "Every derived value is appended exactly once at its aligned timestamp. "
            "Derived columns are not sampled again at -2,-1,0,+1,+2 seconds.",
        )

    def test_audio_unavailable_preserves_structural_zero_interactions(self) -> None:
        unavailable = _prepared(self.item, self.config, audio_available=0.0)
        block = derive_transition_feature_block(unavailable, self.config)
        by_name = {name: block.values[:, index] for index, name in enumerate(block.names)}

        for interaction in (
            "serve_peak_x_receiving_motion_onset",
            "audio_transient_x_coherent_flow",
            "terminal_transient_x_motion_collapse",
            "cadence_collapse_x_formation_contraction",
            "recent_serve_x_elapsed_x_dead_state",
        ):
            np.testing.assert_array_equal(
                by_name[f"transition/interaction/{interaction}"],
                np.zeros(len(unavailable.sequence.times), dtype=np.float32),
            )

    def test_candidate_plan_has_controls_combined_and_single_interaction_ablations(self) -> None:
        transition = prepare_transition_candidates(
            (self.prepared,), self.config
        )
        candidates = {item.name: item for item in transition.candidates}

        self.assertEqual(
            tuple(candidates)[:4],
            (
                "baseline_450",
                "baseline_plus_multiscale",
                "baseline_plus_interactions",
                "combined",
            ),
        )
        self.assertEqual(
            len(candidates["baseline_450"].indexes),
            EXPECTED_BASELINE_CONTEXT_COUNT,
        )
        self.assertEqual(
            len(candidates["baseline_plus_interactions"].indexes),
            EXPECTED_BASELINE_CONTEXT_COUNT + len(INTERACTION_DEFINITIONS),
        )
        for definition in INTERACTION_DEFINITIONS:
            ablation = candidates[f"combined_minus_{definition['id']}"]
            self.assertEqual(
                len(ablation.indexes), len(candidates["combined"].indexes) - 1
            )


class TransitionDevelopmentExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="volleycut-transition-development-test-"
        )
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.config = _feature_config()
        rows = [
            _recording(self.root, "dev-a", "a", "train"),
            _recording(self.root, "dev-b", "b", "train"),
            _recording(self.root, "dev-c", "c", "train"),
            _recording(self.root, "dev-d", "d", "validation"),
            _recording(self.root, "test-e", "e", "test"),
        ]
        manifest_path = self.root / "manifest.json"
        payload = {
            "schemaVersion": 1,
            "name": "synthetic-transition-experiment",
            "annotationPolicy": {"id": "serve-contact-to-dead-ball-v1"},
            "recordings": [item.raw for item in rows],
        }
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        self.manifest = DatasetManifest(
            path=manifest_path,
            name="synthetic-transition-experiment",
            recordings=tuple(rows),
            raw=payload,
        )
        self.prepared = [
            _prepared(item, self.config)
            for item in rows
            if item.split in {"train", "validation"}
        ]

    def test_nested_development_report_has_metrics_slices_deltas_and_frozen_plan(self) -> None:
        decoder = DecoderConfig(
            smoothing_seconds=0.0,
            enter_threshold=0.5,
            exit_threshold=0.4,
            min_live_seconds=1.0,
            bridge_gap_seconds=0.0,
            short_event_min_seconds=0.5,
            short_event_threshold=0.8,
        )
        fixed_decoder = lambda _prepared, _probabilities, base: (
            base,
            {"status": "fixed-for-test"},
        )
        with (
            patch("analysis.feature_experiments._select_decoder", side_effect=fixed_decoder),
            patch(
                "analysis.transition_feature_experiment._select_decoder",
                side_effect=fixed_decoder,
            ),
        ):
            report = run_development_transition_experiments(
                self.manifest,
                self.prepared,
                feature_config=self.config,
                training_config=TrainingConfig(
                    epochs=1,
                    batch_size=64,
                    learning_rate=0.01,
                    patience=1,
                    seed=5,
                ),
                decoder_config=decoder,
                inner_fold_limit=1,
            )

        self.assertFalse(report["testLabelsUsed"])
        self.assertEqual(report["freezeStatus"], "frozen-development-selection")
        self.assertEqual(report["protected"]["test"]["recordingIds"], ["test-e"])
        self.assertNotIn("test-e", report["development"]["recordingIds"])
        self.assertEqual(
            set(report["candidates"]),
            {
                "baseline_450",
                "baseline_plus_multiscale",
                "baseline_plus_interactions",
                "combined",
                *{
                    f"combined_minus_{item['id']}"
                    for item in INTERACTION_DEFINITIONS
                },
            },
        )
        self.assertEqual(
            set(report["candidates"]["combined"]["multiIouMetrics"]),
            {
                "eventF1AtIou03",
                "eventF1AtIou05",
                "eventF1AtIou07",
                "matchedRalliesAtIou03",
                "matchedRalliesAtIou05",
                "matchedRalliesAtIou07",
            },
        )
        self.assertIn(
            "shortAtMost3Seconds",
            report["candidates"]["combined"]["outcomeSlices"],
        )
        self.assertIn(
            "combined", report["pairedComparisonsAgainstBaseline"]
        )
        self.assertEqual(
            set(report["individualInteractionAblations"]),
            {item["id"] for item in INTERACTION_DEFINITIONS},
        )
        self.assertIn(
            report["selectedCandidateForRetrospectiveTest"],
            {
                "baseline_450",
                "baseline_plus_multiscale",
                "baseline_plus_interactions",
                "combined",
            },
        )
        self.assertFalse(report["candidateSelection"]["testMetricsConsulted"])
        self.assertGreater(report["finalizationPlan"]["epochCap"], 0)
        self.assertTrue(report["causalAndOfflineNotes"]["centeredWindowLookahead"])
        json.dumps(report, allow_nan=False)

    def test_development_rejects_a_prepared_test_recording(self) -> None:
        test_item = next(
            item for item in self.manifest.recordings if item.split == "test"
        )
        with self.assertRaisesRegex(
            FeatureExperimentError, r"exactly train\+validation"
        ):
            run_development_transition_experiments(
                self.manifest,
                [*self.prepared, _prepared(test_item, self.config)],
                feature_config=self.config,
                training_config=TrainingConfig(epochs=1, patience=1),
                decoder_config=DecoderConfig(),
            )

    def test_retrospective_gate_fails_before_manifest_or_test_access(self) -> None:
        invalid = self.root / "not-frozen.json"
        invalid.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "kind": "volleycut-transition-feature-experiment-development",
                    "freezeStatus": "draft",
                    "testLabelsUsed": False,
                }
            ),
            encoding="utf-8",
        )

        with patch(
            "analysis.transition_feature_experiment.load_manifest",
            side_effect=AssertionError("invalid gate must fail before manifest/test access"),
        ):
            with self.assertRaisesRegex(
                FeatureExperimentError, "frozen unopened-test"
            ):
                run_retrospective_transition_test(
                    self.manifest.path,
                    invalid,
                    self.root / "cache",
                )


if __name__ == "__main__":
    unittest.main()
