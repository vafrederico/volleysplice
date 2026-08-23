import unittest

import numpy as np

from analysis.side_switch_full_union_ranker import (
    UnionDecoderSettings,
    add_derived_features,
    decode_ranked_candidates,
)


def _row(event_id: str, time: float) -> dict[str, object]:
    return {
        "eventId": event_id,
        "recordingId": "video",
        "kind": "adjacent-rally-boundary",
        "transitionTime": time,
        "features": {},
        "score": None,
    }


class SideSwitchFullUnionRankerTests(unittest.TestCase):
    def test_derived_candidate_features_distinguish_internal_peaks(self) -> None:
        row = add_derived_features(
            {
                "kind": "internal-dead-state-peak",
                "score": 0.99,
                "features": {"existing": 1.0},
            }
        )
        self.assertEqual(row["features"]["candidateIsInternalDeadStatePeak"], 1.0)
        self.assertEqual(row["features"]["candidateGeneratorScore"], 0.99)

    def test_decoder_uses_score_ranked_local_suppression(self) -> None:
        rows = [_row("a", 10.0), _row("b", 11.0), _row("c", 40.0)]
        selected = decode_ranked_candidates(
            rows,
            np.asarray([0.8, 0.9, 0.7]),
            0.5,
            UnionDecoderSettings(minimum_time_separation_seconds=5.0),
        )
        self.assertEqual(selected.tolist(), [False, True, True])

    def test_decoder_soft_count_penalizes_only_after_free_predictions(self) -> None:
        rows = [_row("a", 10.0), _row("b", 20.0)]
        selected = decode_ranked_candidates(
            rows,
            np.asarray([0.8, 0.7]),
            0.5,
            UnionDecoderSettings(
                free_predictions_per_recording=1,
                count_penalty_logit=1.0,
            ),
        )
        self.assertEqual(selected.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
