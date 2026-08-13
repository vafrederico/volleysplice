from __future__ import annotations

import unittest

from analysis.multistate_immediate_result import (
    CANDIDATE_NAMES,
    _geometric_prior,
    _select_inner,
)


def slice_metrics(rallies: int, matches: int) -> dict[str, float | int]:
    return {
        "rallies": rallies,
        "strictMatchRecall": matches / rallies if rallies else 0.0,
    }


def metrics(objective: float, immediate_matches: int) -> dict[str, object]:
    return {
        "objective": objective,
        "eventF1": objective,
        "eventPrecision": 0.7,
        "timeIoU": objective,
        "liveTimeRecall": 0.8,
        "deadSecondsRetained": 100.0,
        "outcomeSlices": {
            "ordinaryLong": slice_metrics(10, 8),
            "shortAtMost3Seconds": slice_metrics(4, immediate_matches),
            "ace": slice_metrics(2, immediate_matches // 2),
            "serviceFault": slice_metrics(2, immediate_matches - immediate_matches // 2),
        },
    }


def report(objective: float, immediate_matches: int) -> dict[str, object]:
    aggregate = metrics(objective, immediate_matches)
    return {
        "aggregate": aggregate,
        "bySourceGroup": {
            group: aggregate for group in ("a", "b", "c")
        },
        "macroSourceGroup": {"objective": objective},
        "recordings": [],
    }


def proxy_report(gain: float) -> dict[str, object]:
    return {
        "aggregate": {"logLossGainVsFoldPrevalence": gain},
        "bySourceGroup": {
            group: {"logLossGainVsFoldPrevalence": gain}
            for group in ("a", "b", "c")
        },
    }


class MultistateImmediateResultTests(unittest.TestCase):
    def test_geometric_result_prior_uses_training_runs_only(self) -> None:
        prior, summary = _geometric_prior([4, 8, 12])
        self.assertAlmostEqual(summary["meanSamples"], 8.0)
        self.assertAlmostEqual(summary["geometricExitHazard"], 0.125)
        self.assertGreater(prior.score_at(4), prior.score_at(12))

    def test_inner_selection_retains_v1_when_result_arm_lacks_gain(self) -> None:
        reports = {
            name: report(0.60 if index == 0 else 0.605, 2)
            for index, name in enumerate(CANDIDATE_NAMES)
        }
        selected, detail = _select_inner(reports, proxy_report(0.02))
        self.assertEqual(selected, "v1_noop")
        self.assertFalse(
            detail["candidates"]["tag_duration_mixture"]["gate"]["eligible"]
        )

    def test_inner_selection_can_choose_learned_proxy_after_all_guards(self) -> None:
        reports = {
            "v1_noop": report(0.60, 2),
            "tag_duration_mixture": report(0.615, 3),
            "bounded_proxy_tag_duration": report(0.63, 4),
        }
        selected, detail = _select_inner(reports, proxy_report(0.02))
        self.assertEqual(selected, "bounded_proxy_tag_duration")
        self.assertTrue(detail["candidates"][selected]["gate"]["eligible"])

    def test_learned_proxy_requires_log_loss_gain(self) -> None:
        reports = {
            "v1_noop": report(0.60, 2),
            "tag_duration_mixture": report(0.60, 2),
            "bounded_proxy_tag_duration": report(0.63, 4),
        }
        selected, detail = _select_inner(reports, proxy_report(0.0))
        self.assertEqual(selected, "v1_noop")
        self.assertFalse(
            detail["candidates"]["bounded_proxy_tag_duration"]["gate"][
                "checks"
            ]["proxyLogLossImprovesAtLeast001"]
        )


if __name__ == "__main__":
    unittest.main()
