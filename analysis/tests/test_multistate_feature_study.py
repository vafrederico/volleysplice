from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from analysis.config import FEATURE_VERSION, DecoderConfig, TrainingConfig
from analysis.feature_experiments import (
    _subset_prepared,
    build_fold_plan,
    sha256_file,
)
from analysis.multistate import MultistateState
from analysis.multistate_feature_study import (
    _load_frozen_study,
    _load_frozen_upstream_report,
    estimate_fold_decoder,
    multistate_promotion_gate,
    reconstruct_frozen_upstream_features,
    run_development_multistate_study,
    state_log_scores,
    state_targets,
)
from analysis.pipeline import _manifest_digest
from analysis.schema import DatasetManifest
from analysis.tests.test_transition_feature_experiment import (
    _feature_config,
    _prepared as transition_prepared,
    _recording,
)
from analysis.transition_feature_experiment import prepare_transition_candidates


class _ProbabilityModel:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def predict(self, _matrix: np.ndarray) -> np.ndarray:
        return self.values


def _slice(rallies: int, matches: int) -> dict[str, float | int]:
    return {
        "rallies": rallies,
        "strictMatchRecall": matches / rallies if rallies else 0.0,
    }


class MultistateFeatureStudyTests(unittest.TestCase):
    def _prepared(self) -> SimpleNamespace:
        times = np.arange(0.0, 20.0, 0.25)
        rallies = (
            SimpleNamespace(start=5.02, end=7.8),
            SimpleNamespace(start=12.04, end=16.2),
        )
        return SimpleNamespace(
            sequence=SimpleNamespace(times=times),
            recording=SimpleNamespace(
                id="recording",
                source_group="group",
                rallies=rallies,
            ),
            sample_mask=np.ones(len(times), dtype=bool),
        )

    def test_serve_target_is_exactly_one_sample_per_rally(self) -> None:
        targets = state_targets(self._prepared())
        self.assertEqual(
            int(np.sum(targets == int(MultistateState.SERVE))), 2
        )
        serve_indexes = np.flatnonzero(targets == int(MultistateState.SERVE))
        self.assertEqual(list(serve_indexes), [20, 48])

    def test_fold_decoder_estimates_one_sample_serve_prior(self) -> None:
        prepared = self._prepared()
        config, summary = estimate_fold_decoder((prepared,), transition_bonus=1.0)
        serve = config.duration_priors[MultistateState.SERVE]
        self.assertEqual(serve.minimum_samples, 1)
        self.assertEqual(serve.maximum_samples, 1)
        self.assertEqual(summary["trainingSourceGroups"], ["group"])
        self.assertTrue(
            all(np.isfinite(value) for value in config.transition_log_scores.values())
        )

    def test_state_scores_are_row_normalized_log_probabilities(self) -> None:
        from analysis.multistate_feature_study import StateModelBundle

        bundle = StateModelBundle(
            tuple(
                _ProbabilityModel(np.asarray(values, dtype=np.float64))
                for values in (
                    [0.8, 0.1],
                    [0.1, 0.2],
                    [0.05, 0.3],
                    [0.05, 0.4],
                )
            )
        )
        scores = state_log_scores(bundle, np.zeros((2, 1), dtype=np.float32))
        np.testing.assert_allclose(np.sum(np.exp(scores), axis=1), 1.0)
        self.assertTrue(np.isfinite(scores).all())

    def test_reconstructs_frozen_step1_signature_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _feature_config()
            recording = _recording(root, "dev-a", "a", "train")
            raw = transition_prepared(recording, config)
            transition = prepare_transition_candidates((raw,), config)
            expected_spec = next(
                candidate
                for candidate in transition.candidates
                if candidate.name == "baseline_plus_interactions"
            )
            expected = transition.prepared[0].contextual_values[
                :, expected_spec.indexes
            ]
            names = tuple(
                transition.prepared[0].contextual_names[index]
                for index in expected_spec.indexes
            )
            actual = reconstruct_frozen_upstream_features((raw,), config, names)[0]
            self.assertEqual(actual.contextual_names, names)
            np.testing.assert_array_equal(actual.contextual_values, expected)

    def test_nested_smoke_report_is_cross_fitted_and_gated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _feature_config()
            rows = [
                _recording(root, "dev-a", "a", "train"),
                _recording(root, "dev-b", "b", "train"),
                _recording(root, "dev-c", "c", "train"),
                _recording(root, "dev-d", "d", "validation"),
                _recording(root, "test-e", "e", "test"),
            ]
            manifest_path = root / "manifest.json"
            manifest_payload = {
                "schemaVersion": 1,
                "name": "synthetic-multistate-study",
                "annotationPolicy": {"id": "serve-contact-to-dead-ball-v1"},
                "recordings": [item.raw for item in rows],
            }
            manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
            manifest = DatasetManifest(
                path=manifest_path,
                name="synthetic-multistate-study",
                recordings=tuple(rows),
                raw=manifest_payload,
            )
            prepared = [
                transition_prepared(item, config)
                for item in rows
                if item.split in {"train", "validation"}
            ]
            transition = prepare_transition_candidates(prepared, config)
            control_spec = next(
                item for item in transition.candidates if item.name == "baseline_450"
            )
            control = _subset_prepared(transition.prepared, control_spec.indexes)
            names = control[0].contextual_names
            upstream_payload = {
                "schemaVersion": 1,
                "kind": "volleycut-transition-feature-experiment-development",
                "freezeStatus": "frozen-development-selection",
                "testLabelsUsed": False,
                "manifestFileSha256": sha256_file(manifest.path),
                "manifestSnapshotSha256": _manifest_digest(manifest),
                "recordingContentSha256": {
                    item.id: item.content_sha256 for item in manifest.recordings
                },
                "featureVersion": FEATURE_VERSION,
                "featureConfig": config.to_dict(),
                "selectionProtocol": {
                    "innerFoldLimit": None,
                    "folds": [
                        item.to_dict() for item in build_fold_plan(manifest.recordings)
                    ],
                },
                "selectedCandidateForRetrospectiveTest": "baseline_450",
                "finalizationPlan": {
                    "candidate": "baseline_450",
                    "featureNames": list(names),
                    "featureSignatureSha256": __import__("hashlib").sha256(
                        "\0".join(names).encode("utf-8")
                    ).hexdigest(),
                },
            }
            upstream_path = root / "upstream.json"
            upstream_path.write_text(json.dumps(upstream_payload), encoding="utf-8")
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
                patch(
                    "analysis.feature_experiments._select_decoder",
                    side_effect=fixed_decoder,
                ),
                patch(
                    "analysis.multistate_feature_study._select_decoder",
                    side_effect=fixed_decoder,
                ),
            ):
                report = run_development_multistate_study(
                    manifest,
                    prepared,
                    upstream_report_path=upstream_path,
                    training_config=TrainingConfig(
                        epochs=1,
                        batch_size=64,
                        learning_rate=0.01,
                        patience=1,
                        seed=5,
                    ),
                    decoder_config=decoder,
                    inner_fold_limit=1,
                    transition_bonuses=(0.0,),
                )

            self.assertFalse(report["testLabelsUsed"])
            self.assertEqual(
                set(report["sameFeatureBinaryControl"]["oof"]["bySourceGroup"]),
                {"a", "b", "c", "d"},
            )
            self.assertEqual(
                set(report["serveAnchoredMultistate"]["oof"]["bySourceGroup"]),
                {"a", "b", "c", "d"},
            )
            self.assertIn("promoteMultistate", report["promotionDecision"])
            self.assertFalse(
                report["serveAnchoredMultistate"]["optionalImmediateResultHead"][
                    "implemented"
                ]
            )
            json.dumps(report, allow_nan=False)

    def test_promotion_gate_requires_objective_and_guardrails(self) -> None:
        control = {
            "objective": 0.50,
            "liveTimeRecall": 0.90,
            "outcomeSlices": {
                "shortAtMost3Seconds": _slice(5, 1),
                "serviceFault": _slice(3, 0),
                "ordinaryLong": _slice(10, 8),
            },
        }
        candidate = {
            "objective": 0.53,
            "liveTimeRecall": 0.895,
            "outcomeSlices": {
                "shortAtMost3Seconds": _slice(5, 2),
                "serviceFault": _slice(3, 1),
                "ordinaryLong": _slice(10, 8),
            },
        }
        passed = multistate_promotion_gate(
            control, candidate, {"classification": "helpful"}
        )
        self.assertTrue(passed["promoteMultistate"])
        self.assertEqual(
            passed["strictMatchCounts"]["shortAtMost3Seconds"],
            {"control": 1, "candidate": 2, "deltaCandidateMinusControl": 1},
        )
        self.assertEqual(
            passed["strictMatchCounts"]["serviceFault"],
            {"control": 0, "candidate": 1, "deltaCandidateMinusControl": 1},
        )

        candidate["liveTimeRecall"] = 0.88
        failed = multistate_promotion_gate(
            control, candidate, {"classification": "helpful"}
        )
        self.assertFalse(failed["promoteMultistate"])
        self.assertFalse(failed["checks"]["overallLiveRecallLossAtMostOnePoint"])

        candidate["liveTimeRecall"] = 0.895
        candidate["outcomeSlices"]["shortAtMost3Seconds"] = _slice(5, 1)
        no_short_gain = multistate_promotion_gate(
            control, candidate, {"classification": "helpful"}
        )
        self.assertFalse(no_short_gain["promoteMultistate"])
        self.assertFalse(no_short_gain["checks"]["shortStrictMatchesImproved"])

        candidate["outcomeSlices"]["shortAtMost3Seconds"] = _slice(5, 2)
        candidate["outcomeSlices"]["serviceFault"] = _slice(3, 0)
        no_fault_gain = multistate_promotion_gate(
            control, candidate, {"classification": "helpful"}
        )
        self.assertFalse(no_fault_gain["promoteMultistate"])
        self.assertFalse(
            no_fault_gain["checks"]["serviceFaultStrictMatchesImproved"]
        )

    def test_upstream_loader_rejects_report_that_used_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upstream.json"
            path.write_text(
                json.dumps(
                    {
                        "freezeStatus": "frozen-development-selection",
                        "kind": "volleycut-transition-feature-experiment-development",
                        "testLabelsUsed": True,
                        "featureConfig": {},
                        "finalizationPlan": {"featureNames": ["a"]},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "unopened-test"):
                _load_frozen_upstream_report(path)

    def test_upstream_loader_rejects_inner_fold_smoke_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upstream.json"
            path.write_text(
                json.dumps(
                    {
                        "freezeStatus": "frozen-development-selection",
                        "kind": "volleycut-transition-feature-experiment-development",
                        "testLabelsUsed": False,
                        "featureConfig": {},
                        "selectionProtocol": {"innerFoldLimit": 1},
                        "finalizationPlan": {"featureNames": ["a"]},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "full nested"):
                _load_frozen_upstream_report(path)

    def test_upstream_loader_rejects_generic_shaped_report_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upstream.json"
            path.write_text(
                json.dumps(
                    {
                        "kind": "generic-shaped-development-report",
                        "freezeStatus": "frozen-development-selection",
                        "testLabelsUsed": False,
                        "featureConfig": {},
                        "selectionProtocol": {"innerFoldLimit": None},
                        "finalizationPlan": {"featureNames": ["a"]},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "upstream report"):
                _load_frozen_upstream_report(path)

    def test_retrospective_loader_enforces_promotion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "volleycut-multistate-feature-study-development",
                        "freezeStatus": "frozen-development-selection",
                        "testLabelsUsed": False,
                        "finalizationPlan": {"architecture": "binary_control"},
                        "promotionDecision": {"promoteMultistate": False},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "gated off"):
                _load_frozen_study(path)

    def test_retrospective_loader_rejects_inner_fold_smoke_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "volleycut-multistate-feature-study-development",
                        "freezeStatus": "frozen-development-selection",
                        "testLabelsUsed": False,
                        "selectionProtocol": {"innerFoldLimit": 1},
                        "finalizationPlan": {
                            "architecture": "serve_anchored_multistate"
                        },
                        "promotionDecision": {"promoteMultistate": True},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "smoke ablation"):
                _load_frozen_study(path)


if __name__ == "__main__":
    unittest.main()
