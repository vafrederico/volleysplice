from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analysis.multistate import MultistateState
from analysis.multistate_followup import (
    _cache_name,
    _force_gold_state_anchors,
    _oracle_replace_boundaries,
    _quantiles,
    _serve_rows,
    _summarize_serve_rows,
    _validate_development_study,
    compose_conservative_hybrid,
    select_inner_hybrid,
    state_classification_report,
)
from analysis.schema import DatasetManifest, Interval


class MultistateFollowupTests(unittest.TestCase):
    def _hybrid_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[MultistateState], np.ndarray]:
        times = np.arange(0.0, 20.0, 0.25)
        probabilities = np.full((len(times), 4), 0.05, dtype=np.float64)
        probabilities[:, int(MultistateState.DEAD)] = 0.85
        states = [MultistateState.DEAD] * len(times)
        smoothed = np.full(len(times), 0.1, dtype=np.float64)
        return times, probabilities, states, smoothed

    def test_state_report_has_argmax_metrics_and_calibration(self) -> None:
        targets = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int8)
        probabilities = np.asarray(
            [
                [0.8, 0.1, 0.05, 0.05],
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.6, 0.2, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.1, 0.4, 0.4],
                [0.1, 0.1, 0.1, 0.7],
                [0.1, 0.1, 0.2, 0.6],
            ],
            dtype=np.float64,
        )
        report = state_classification_report(targets, probabilities)
        self.assertEqual(report["samples"], 8)
        self.assertEqual(report["accuracy"], 1.0)
        serve = report["states"]["SERVE"]
        self.assertEqual(serve["recall"], 1.0)
        self.assertEqual(serve["precision"], 1.0)
        self.assertEqual(serve["oneVsRestCalibration"]["samples"], 8)
        self.assertGreater(serve["oneVsRestCalibration"]["logLoss"], 0.0)

    def test_oracle_replaces_only_overlapping_matched_boundaries(self) -> None:
        truth = (Interval(1.0, 4.0), Interval(8.0, 12.0))
        predictions = (Interval(1.5, 3.0), Interval(8.5, 13.0), Interval(20.0, 21.0))
        starts = _oracle_replace_boundaries(
            truth, predictions, replace_start=True, replace_end=False
        )
        ends = _oracle_replace_boundaries(
            truth, predictions, replace_start=False, replace_end=True
        )
        both = _oracle_replace_boundaries(
            truth, predictions, replace_start=True, replace_end=True
        )
        self.assertEqual(starts, [Interval(1.0, 3.0), Interval(8.0, 13.0), predictions[2]])
        self.assertEqual(ends, [Interval(1.5, 4.0), Interval(8.5, 12.0), predictions[2]])
        self.assertEqual(both, [truth[0], truth[1], predictions[2]])

    def test_forced_state_oracles_pin_serve_and_first_post_end_dead(self) -> None:
        times = np.arange(0.0, 8.0, 0.25)
        item = SimpleNamespace(
            sequence=SimpleNamespace(times=times),
            recording=SimpleNamespace(
                id="recording",
                source_group="group",
                rallies=(Interval(2.02, 4.01),),
            ),
            sample_mask=np.ones(len(times), dtype=bool),
        )
        original = np.log(np.full((len(times), 4), 0.25, dtype=np.float64))
        forced = _force_gold_state_anchors(
            item, original, force_serves=True, force_ends=True
        )
        serve_index = int(np.argmin(np.abs(times - 2.02)))
        end_index = int(np.flatnonzero(times >= 4.01)[0])
        self.assertEqual(forced[serve_index, int(MultistateState.SERVE)], 0.0)
        self.assertTrue(
            np.isneginf(np.delete(forced[serve_index], int(MultistateState.SERVE))).all()
        )
        self.assertEqual(forced[end_index, int(MultistateState.DEAD)], 0.0)
        self.assertTrue(
            np.isneginf(np.delete(forced[end_index], int(MultistateState.DEAD))).all()
        )

    def test_serve_rejection_is_summarized_by_group_and_outcome(self) -> None:
        times = np.arange(0.0, 10.0, 0.25)
        probabilities = np.full((len(times), 4), 0.05, dtype=np.float64)
        probabilities[:, int(MultistateState.DEAD)] = 0.85
        probabilities[8, :] = [0.05, 0.05, 0.85, 0.05]
        probabilities[24, :] = [0.7, 0.1, 0.1, 0.1]
        states = [MultistateState.DEAD] * len(times)
        states[8] = MultistateState.SERVE
        item = SimpleNamespace(
            sequence=SimpleNamespace(times=times),
            sample_mask=np.ones(len(times), dtype=bool),
            recording=SimpleNamespace(
                id="recording",
                source_group="group",
                rallies=(
                    Interval(2.0, 4.0, ("service-fault",)),
                    Interval(6.0, 9.0),
                ),
            ),
        )
        rows = _serve_rows(item, probabilities, states, [Interval(2.0, 4.0)])
        summary = _summarize_serve_rows(rows)
        self.assertEqual(summary["aggregate"]["rallies"], 2)
        self.assertEqual(summary["aggregate"]["serveArgmaxCount"], 1)
        self.assertEqual(summary["aggregate"]["decodedServeWithin"]["0.5Seconds"], 1)
        self.assertEqual(summary["byOutcome"]["serviceFault"]["strictMatchCount"], 1)
        self.assertEqual(summary["bySourceGroup"]["group"]["rallies"], 2)

    def test_duration_quantiles_include_frozen_bins(self) -> None:
        report = _quantiles([0.5, 2.0, 4.0, 10.0])
        self.assertEqual(report["count"], 4)
        self.assertEqual(report["p50Seconds"], 3.0)
        self.assertEqual(
            report["bins"],
            {
                "atMost1Second": 1,
                "over1To3Seconds": 1,
                "over3To8Seconds": 1,
                "over8Seconds": 1,
            },
        )

    def test_cache_names_are_safe_and_content_disambiguated(self) -> None:
        first = _cache_name("recording/a")
        second = _cache_name("recording_a")
        self.assertNotIn("/", first)
        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith(".npz"))

    def test_reduced_inner_report_is_rejected_before_upstream_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            manifest = DatasetManifest(
                path=manifest_path,
                name="synthetic",
                recordings=(),
                raw={},
            )
            report_path = root / "multistate.json"
            report_path.write_text(
                json.dumps(
                    {
                        "kind": "volleycut-multistate-feature-study-development",
                        "freezeStatus": "frozen-development-selection",
                        "testLabelsUsed": False,
                        "selectionProtocol": {"innerFoldLimit": 1},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "full nested"):
                _validate_development_study(manifest, report_path)

    def test_hybrid_noop_preserves_binary_exactly(self) -> None:
        times, probabilities, states, smoothed = self._hybrid_arrays()
        binary = [Interval(2.0, 8.0)]
        output, summary = compose_conservative_hybrid(
            binary,
            [Interval(2.25, 7.75)],
            mode="binary_noop",
            times=times,
            state_probabilities=probabilities,
            decoded_states=states,
            smoothed_binary=smoothed,
        )
        self.assertEqual(output, binary)
        self.assertEqual(summary["outputIntervals"], 1)

    def test_hybrid_snaps_confident_boundaries_within_long_invariant(self) -> None:
        times, probabilities, states, smoothed = self._hybrid_arrays()
        candidate = Interval(2.25, 7.75)
        start_index = int(np.argmin(np.abs(times - candidate.start)))
        end_index = int(np.flatnonzero(times >= candidate.end)[0])
        live = (times >= candidate.start) & (times < candidate.end)
        probabilities[live] = [0.05, 0.05, 0.05, 0.85]
        probabilities[start_index] = [0.05, 0.05, 0.85, 0.05]
        probabilities[end_index] = [0.85, 0.05, 0.05, 0.05]
        states = [
            MultistateState.LIVE if selected else MultistateState.DEAD
            for selected in live
        ]
        states[start_index] = MultistateState.SERVE
        output, summary = compose_conservative_hybrid(
            [Interval(2.0, 8.0)],
            [candidate],
            mode="boundary_snap",
            times=times,
            state_probabilities=probabilities,
            decoded_states=states,
            smoothed_binary=smoothed,
        )
        self.assertEqual(output, [candidate])
        self.assertEqual(summary["snap"]["startSnaps"], 1)
        self.assertEqual(summary["snap"]["endSnaps"], 1)

    def test_hybrid_keeps_long_binary_when_snap_removes_too_much(self) -> None:
        times, probabilities, states, smoothed = self._hybrid_arrays()
        candidate = Interval(2.75, 7.25)
        start_index = int(np.argmin(np.abs(times - candidate.start)))
        end_index = int(np.flatnonzero(times >= candidate.end)[0])
        live = (times >= candidate.start) & (times < candidate.end)
        probabilities[live] = [0.05, 0.05, 0.05, 0.85]
        probabilities[start_index] = [0.05, 0.05, 0.85, 0.05]
        probabilities[end_index] = [0.85, 0.05, 0.05, 0.05]
        states = [
            MultistateState.LIVE if selected else MultistateState.DEAD
            for selected in live
        ]
        states[start_index] = MultistateState.SERVE
        base = Interval(2.0, 8.0)
        output, summary = compose_conservative_hybrid(
            [base],
            [candidate],
            mode="boundary_snap",
            times=times,
            state_probabilities=probabilities,
            decoded_states=states,
            smoothed_binary=smoothed,
        )
        self.assertEqual(output, [base])
        self.assertEqual(summary["snap"]["rejectedByLongInvariant"], 1)

    def test_hybrid_adds_only_isolated_dual_supported_short_rescue(self) -> None:
        times, probabilities, states, smoothed = self._hybrid_arrays()
        candidate = Interval(12.0, 14.0)
        selected = (times >= candidate.start) & (times < candidate.end)
        start_index = int(np.argmin(np.abs(times - candidate.start)))
        end_index = int(np.flatnonzero(times >= candidate.end)[0])
        probabilities[selected] = [0.05, 0.05, 0.10, 0.80]
        probabilities[start_index] = [0.02, 0.03, 0.90, 0.05]
        probabilities[end_index] = [0.90, 0.03, 0.02, 0.05]
        states = [
            MultistateState.LIVE if value else MultistateState.DEAD
            for value in selected
        ]
        states[start_index] = MultistateState.SERVE
        smoothed[selected] = 0.85
        output, summary = compose_conservative_hybrid(
            [Interval(2.0, 8.0)],
            [candidate],
            mode="short_rescue",
            times=times,
            state_probabilities=probabilities,
            decoded_states=states,
            smoothed_binary=smoothed,
        )
        self.assertEqual(output, [Interval(2.0, 8.0), candidate])
        self.assertEqual(summary["rescue"]["accepted"], 1)

    def test_short_rescue_rejects_high_binary_flank(self) -> None:
        times, probabilities, states, smoothed = self._hybrid_arrays()
        candidate = Interval(12.0, 14.0)
        selected = (times >= candidate.start) & (times < candidate.end)
        start_index = int(np.argmin(np.abs(times - candidate.start)))
        end_index = int(np.flatnonzero(times >= candidate.end)[0])
        probabilities[selected] = [0.05, 0.05, 0.10, 0.80]
        probabilities[start_index] = [0.02, 0.03, 0.90, 0.05]
        probabilities[end_index] = [0.90, 0.03, 0.02, 0.05]
        states = [
            MultistateState.LIVE if value else MultistateState.DEAD
            for value in selected
        ]
        states[start_index] = MultistateState.SERVE
        smoothed[selected] = 0.85
        smoothed[(times >= 11.0) & (times < 12.0)] = 0.8
        output, summary = compose_conservative_hybrid(
            [Interval(2.0, 8.0)],
            [candidate],
            mode="short_rescue",
            times=times,
            state_probabilities=probabilities,
            decoded_states=states,
            smoothed_binary=smoothed,
        )
        self.assertEqual(output, [Interval(2.0, 8.0)])
        self.assertEqual(summary["rescue"]["accepted"], 0)

    def test_inner_selector_keeps_noop_when_fixed_modes_do_not_clear_margin(self) -> None:
        outcomes = {
            "ordinaryLong": {
                "rallies": 10,
                "strictMatchRecall": 0.8,
                "anyOverlapRecall": 0.9,
                "meanCoverage": 0.8,
            },
            "shortAtMost3Seconds": {"rallies": 2, "strictMatchRecall": 0.5},
            "serviceFault": {"rallies": 1, "strictMatchRecall": 0.0},
        }
        metrics = {
            "objective": 0.6,
            "eventF1": 0.6,
            "timeIoU": 0.6,
            "liveTimeRecall": 0.8,
            "liveTimePrecision": 0.7,
            "eventPrecision": 0.6,
            "deadSecondsRetained": 10.0,
            "startBoundaryMaeSeconds": 1.0,
            "endBoundaryMaeSeconds": 1.0,
            "outcomeSlices": outcomes,
        }
        report = {
            "aggregate": metrics,
            "macroSourceGroup": {
                "objective": 0.6,
                "eventF1": 0.6,
                "timeIoU": 0.6,
                "liveTimeRecall": 0.8,
            },
            "bySourceGroup": {group: metrics for group in ("a", "b", "c")},
            "operations": [],
        }
        reports = {
            mode: report
            for mode in (
                "binary_noop",
                "boundary_snap",
                "short_rescue",
                "snap_and_rescue",
            )
        }
        selected, detail = select_inner_hybrid(reports)
        self.assertEqual(selected, "binary_noop")
        self.assertTrue(detail["candidates"]["binary_noop"]["gate"]["eligible"])
        self.assertFalse(detail["candidates"]["boundary_snap"]["gate"]["eligible"])


if __name__ == "__main__":
    unittest.main()
