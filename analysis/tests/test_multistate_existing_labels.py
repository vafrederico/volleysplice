from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import numpy as np

from analysis.multistate import MultistateState
from analysis.multistate_existing_labels import (
    CANDIDATE_NAMES,
    _empirical_prior,
    _normalize_duration_table_with_tail,
    _promotion,
    _selection_report,
    emission_log_scores,
    estimate_empirical_mixture_decoder,
    fit_two_geometric_mixture,
)
from analysis.multistate_feature_study import StateModelBundle


class _ProbabilityModel:
    def __init__(
        self, values: np.ndarray, *, positives: int, negatives: int
    ) -> None:
        self.values = values
        self.training_summary = {
            "positiveSamples": positives,
            "negativeSamples": negatives,
        }

    def predict(self, _matrix: np.ndarray) -> np.ndarray:
        return self.values


def _metrics(objective: float) -> dict[str, object]:
    return {
        "objective": objective,
        "eventF1": objective,
        "eventPrecision": 0.7,
        "timeIoU": objective,
        "liveTimeRecall": 0.8,
        "deadSecondsRetained": 100.0,
        "outcomeSlices": {
            "ordinaryLong": {"rallies": 10, "strictMatchRecall": 0.8},
            "shortAtMost3Seconds": {"rallies": 2, "strictMatchRecall": 0.5},
            "serviceFault": {"rallies": 2, "strictMatchRecall": 0.5},
        },
    }


class MultistateExistingLabelTests(unittest.TestCase):
    def test_prior_correction_restores_rare_state_odds(self) -> None:
        raw = np.asarray([0.8, 0.2], dtype=np.float64)
        bundle = StateModelBundle(
            tuple(
                _ProbabilityModel(
                    raw.copy(),
                    positives=(10 if state == MultistateState.SERVE else 50),
                    negatives=90 if state == MultistateState.SERVE else 50,
                )
                for state in MultistateState
            )
        )
        values = np.zeros((2, 1), dtype=np.float32)
        naive = np.exp(emission_log_scores(bundle, values, mode="naive_balanced_ovr"))
        corrected = np.exp(
            emission_log_scores(bundle, values, mode="fold_prior_corrected_ovr")
        )
        np.testing.assert_allclose(np.sum(corrected, axis=1), 1.0)
        self.assertLess(
            corrected[0, int(MultistateState.SERVE)],
            naive[0, int(MultistateState.SERVE)],
        )

    def test_prior_correction_rejects_unknown_mode(self) -> None:
        bundle = StateModelBundle(
            tuple(
                _ProbabilityModel(np.asarray([0.5]), positives=1, negatives=1)
                for _ in MultistateState
            )
        )
        with self.assertRaisesRegex(ValueError, "unknown emission"):
            emission_log_scores(bundle, np.zeros((1, 1)), mode="not-a-mode")

    def test_prior_correction_of_balanced_half_scores_is_fold_prevalence(self) -> None:
        counts = ((80, 20), (15, 85), (1, 99), (30, 70))
        bundle = StateModelBundle(
            tuple(
                _ProbabilityModel(
                    np.asarray([0.5]), positives=positive, negatives=negative
                )
                for positive, negative in counts
            )
        )
        corrected = np.exp(
            emission_log_scores(
                bundle,
                np.zeros((1, 1)),
                mode="fold_prior_corrected_ovr",
            )
        )[0]
        prevalences = np.asarray(
            [positive / (positive + negative) for positive, negative in counts]
        )
        np.testing.assert_allclose(corrected, prevalences / np.sum(prevalences))

    def test_naive_emissions_exactly_preserve_saturated_v1_path(self) -> None:
        columns = (
            np.asarray([1.0, 0.2]),
            np.asarray([0.3, 1.0]),
            np.asarray([0.1, 0.4]),
            np.asarray([0.5, 0.7]),
        )
        bundle = StateModelBundle(
            tuple(
                _ProbabilityModel(column, positives=1, negatives=1)
                for column in columns
            )
        )
        expected = np.column_stack(columns)
        expected /= np.sum(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(
            np.exp(
                emission_log_scores(
                    bundle,
                    np.zeros((2, 1)),
                    mode="naive_balanced_ovr",
                )
            ),
            expected,
            rtol=0.0,
            atol=1e-15,
        )

    def test_two_geometric_mixture_separates_bimodal_lengths(self) -> None:
        lengths = [4, 5, 5, 6, 6, 7] * 10 + [35, 40, 45, 50, 55] * 10
        weight, short_hazard, long_hazard, summary = fit_two_geometric_mixture(lengths)
        self.assertGreater(short_hazard, long_hazard)
        self.assertGreater(weight, 0.05)
        self.assertLess(weight, 0.95)
        self.assertTrue(summary["converged"])
        self.assertFalse(summary["componentsCollapsed"])
        self.assertLess(summary["shortMeanSamples"], summary["longMeanSamples"])

    def test_empirical_setup_prior_prefers_observed_19_or_20_samples(self) -> None:
        prior, summary = _empirical_prior([19] * 8 + [20] * 12)
        self.assertEqual(summary["modeSamples"], 20)
        self.assertGreater(prior.score_at(20), prior.score_at(1))
        self.assertTrue(np.isfinite(prior.score_at(40)))

    def test_duration_table_and_infinite_tail_are_normalized(self) -> None:
        original = np.asarray([0.2, 0.3, 0.1], dtype=np.float64)
        normalized = _normalize_duration_table_with_tail(original, 0.5)
        represented_mass = float(
            np.sum(normalized[:-1]) + normalized[-1] / (1.0 - 0.5)
        )
        self.assertAlmostEqual(represented_mass, 1.0)

    def test_empirical_mixture_decoder_preserves_dead_serve_and_transitions(self) -> None:
        targets = np.asarray(
            [0] * 10 + [1] * 19 + [2] + [3] * 5 + [0] * 10 + [1] * 20 + [2] + [3] * 40 + [0] * 10,
            dtype=np.int8,
        )
        times = np.arange(len(targets), dtype=np.float64) * 0.25
        prepared = SimpleNamespace(
            sequence=SimpleNamespace(times=times),
            recording=SimpleNamespace(
                id="recording",
                source_group="group",
                rallies=(
                    SimpleNamespace(start=29 * 0.25, end=35 * 0.25),
                    SimpleNamespace(start=65 * 0.25, end=106 * 0.25),
                ),
            ),
            sample_mask=np.ones(len(times), dtype=bool),
        )
        config, summary = estimate_empirical_mixture_decoder(
            [prepared], transition_bonus=0.0
        )
        self.assertEqual(
            config.duration_priors[MultistateState.SERVE].maximum_samples, 1
        )
        self.assertIn(
            (MultistateState.SERVE, MultistateState.LIVE),
            config.transition_log_scores,
        )
        self.assertIn("setupEmpirical", summary)
        self.assertIn("liveTwoGeometricMixture", summary)

    def test_selection_falls_back_to_reference_when_candidates_fail_guards(self) -> None:
        reference_metrics = _metrics(0.6)
        reports = {}
        for index, name in enumerate(CANDIDATE_NAMES):
            candidate_metrics = dict(reference_metrics)
            if index:
                candidate_metrics["objective"] = 0.59
            reports[name] = {
                "aggregate": candidate_metrics,
                "bySourceGroup": {
                    group: candidate_metrics for group in ("a", "b", "c")
                },
                "macroSourceGroup": {"objective": candidate_metrics["objective"]},
                "recordings": [],
            }
        selected, detail = _selection_report(reports)
        self.assertEqual(selected, CANDIDATE_NAMES[0])
        self.assertTrue(detail["candidates"][selected]["gate"]["eligible"])

    def test_selection_accepts_material_guarded_improvement(self) -> None:
        reports = {}
        for index, name in enumerate(CANDIDATE_NAMES):
            objective = 0.62 if index == 1 else 0.60
            metrics = _metrics(objective)
            reports[name] = {
                "aggregate": metrics,
                "bySourceGroup": {
                    group: metrics for group in ("a", "b", "c")
                },
                "macroSourceGroup": {"objective": objective},
                "recordings": [],
            }
        selected, detail = _selection_report(reports)
        self.assertEqual(selected, CANDIDATE_NAMES[1])
        self.assertTrue(detail["candidates"][selected]["gate"]["eligible"])

    def test_promotion_cannot_ignore_operational_binary_recall_floor(self) -> None:
        def report(
            objective: float, recall: float, *, auxiliary_recall: float
        ) -> dict[str, object]:
            metrics = _metrics(objective)
            metrics["liveTimeRecall"] = recall
            metrics["outcomeSlices"]["shortAtMost3Seconds"][
                "strictMatchRecall"
            ] = auxiliary_recall
            metrics["outcomeSlices"]["serviceFault"][
                "strictMatchRecall"
            ] = auxiliary_recall
            groups = {
                group: copy.deepcopy(metrics) for group in ("a", "b", "c", "d")
            }
            return {"aggregate": metrics, "bySourceGroup": groups}

        v1 = report(0.55, 0.78, auxiliary_recall=0.5)
        candidate = report(0.57, 0.79, auxiliary_recall=1.0)
        binary = report(0.54, 0.85, auxiliary_recall=0.5)
        paired, promotion = _promotion(
            v1,
            candidate,
            binary,
            margin=0.01,
            consistency=0.75,
        )
        self.assertEqual(
            paired["candidateMinusV1"]["classification"]["classification"],
            "helpful",
        )
        self.assertFalse(promotion["promoteSelectedCandidate"])
        self.assertFalse(
            promotion["checks"][
                "operationalBinary_overallLiveRecallLossAtMostOnePoint"
            ]
        )


if __name__ == "__main__":
    unittest.main()
