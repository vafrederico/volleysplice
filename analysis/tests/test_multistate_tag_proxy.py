from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import numpy as np

from analysis.feature_experiments import FeatureExperimentError
from analysis.multistate_tag_proxy import (
    TAG_PROXY_FEATURE_NAMES,
    extract_tag_proxy_anchors,
    predict_tag_proxy_sequence,
    tag_conditioned_live_runs,
    tag_proxy_metrics,
)
from analysis.schema import Interval


def prepared(
    rallies: tuple[Interval, ...], *, masked_index: int | None = None
) -> SimpleNamespace:
    times = np.arange(0.0, 21.0, 0.25, dtype=np.float64)
    values = np.arange(
        len(times) * len(TAG_PROXY_FEATURE_NAMES), dtype=np.float32
    ).reshape(len(times), len(TAG_PROXY_FEATURE_NAMES))
    mask = np.ones(len(times), dtype=bool)
    if masked_index is not None:
        mask[masked_index] = False
    return SimpleNamespace(
        recording=SimpleNamespace(
            id="recording",
            source_group="group",
            rallies=rallies,
        ),
        sequence=SimpleNamespace(times=times),
        contextual_values=values,
        contextual_names=TAG_PROXY_FEATURE_NAMES,
        sample_mask=mask,
    )


class _HalfModel:
    training_summary = {"positiveSamples": 1, "negativeSamples": 3}

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.full(len(values), 0.5, dtype=np.float32)


class MultistateTagProxyTests(unittest.TestCase):
    def test_anchor_target_is_only_ace_or_service_fault(self) -> None:
        item = prepared(
            (
                Interval(6.0, 8.0, tags=("ace",)),
                Interval(12.0, 18.0),
            )
        )
        anchors = extract_tag_proxy_anchors([item])
        np.testing.assert_array_equal(anchors.labels, np.asarray([1.0, 0.0]))
        self.assertEqual(anchors.outcome_tags, ("ace", "ordinary"))
        self.assertEqual(anchors.values.shape, (2, 17))
        self.assertEqual(anchors.summary()["positive"], 1)
        json.dumps(anchors.summary(), allow_nan=False)

    def test_both_outcome_tags_are_rejected(self) -> None:
        item = prepared(
            (Interval(6.0, 8.0, tags=("ace", "service-fault")),)
        )
        with self.assertRaisesRegex(FeatureExperimentError, "both ace"):
            extract_tag_proxy_anchors([item])

    def test_full_sequence_probabilities_restore_training_prior_and_clip(self) -> None:
        item = prepared((Interval(6.0, 8.0, tags=("ace",)),))
        scores = predict_tag_proxy_sequence(_HalfModel(), item)
        self.assertEqual(scores.shape, item.sequence.times.shape)
        np.testing.assert_allclose(scores, 0.25)

    def test_proxy_metrics_compare_against_fold_prevalence(self) -> None:
        labels = np.asarray([1.0, 0.0, 1.0, 0.0])
        strong = np.asarray([0.9, 0.1, 0.8, 0.2])
        metrics = tag_proxy_metrics(labels, strong, baseline_prevalence=0.5)
        self.assertLess(metrics["deltaLogLossProxyMinusBaseline"], 0.0)
        self.assertAlmostEqual(metrics["rocAuc"], 1.0)
        self.assertAlmostEqual(metrics["averagePrecision"], 1.0)

    def test_live_runs_are_tag_conditioned_and_mask_censored(self) -> None:
        rallies = (
            Interval(6.0, 8.0, tags=("service-fault",)),
            Interval(12.0, 18.0),
        )
        runs = tag_conditioned_live_runs([prepared(rallies)])
        self.assertEqual(runs.immediate_result, (7,))
        self.assertEqual(runs.ordinary, (23,))
        self.assertEqual(runs.excluded, ())

        censored = tag_conditioned_live_runs(
            [
                prepared(rallies),
                prepared(rallies, masked_index=52),
            ]
        )
        self.assertEqual(len(censored.excluded), 1)
        self.assertEqual(censored.excluded[0]["reason"], "ignored-span-censored")


if __name__ == "__main__":
    unittest.main()
