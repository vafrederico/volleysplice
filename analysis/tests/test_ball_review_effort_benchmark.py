from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark-ball-review-effort.py"
SPEC = importlib.util.spec_from_file_location("ball_review_effort_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def _annotation(state: str, *, x: float = 0.25) -> dict[str, object]:
    objects: list[dict[str, object]] = []
    if state == "localizable":
        objects.append(
            {
                "id": "ball",
                "category": "volleyball",
                "role": "primary-court",
                "bbox": {"x": x, "y": 0.25, "width": 0.1, "height": 0.1},
                "visibility": "clear",
                "truncated": False,
            }
        )
    return {
        "frameId": "unused",
        "status": "reviewed",
        "primaryBallState": state,
        "objects": objects,
        "notes": "",
    }


class BallReviewEffortBenchmarkTests(unittest.TestCase):
    def test_pairwise_similarity_is_symmetric_and_uses_active_states(self) -> None:
        first = {
            "a": _annotation("localizable"),
            "b": _annotation("indeterminate"),
            "c": _annotation("localizable"),
            "d": _annotation("indeterminate"),
        }
        second = {
            "a": _annotation("localizable"),
            "b": _annotation("localizable"),
            "c": _annotation("indeterminate"),
            "d": _annotation("indeterminate"),
        }

        forward = BENCHMARK.score_symmetric_similarity(first, second)
        reverse = BENCHMARK.score_symmetric_similarity(second, first)

        self.assertEqual(forward, reverse)
        self.assertAlmostEqual(forward["stateAccuracy"], 0.5)
        self.assertAlmostEqual(forward["stateMacroF1"], 0.5)
        self.assertAlmostEqual(forward["primaryPresenceF1"], 0.5)
        self.assertAlmostEqual(forward["boxF1Iou25"], 0.5)

    def test_iou_is_clamped_to_probability_range(self) -> None:
        box = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}
        self.assertEqual(BENCHMARK._iou(box, box), 1.0)

    def test_existing_pseudo_reference_is_loaded_as_sealed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            annotation = _annotation("indeterminate")
            (output / "sealed-reference.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "source": "prior blind Sol xhigh pseudo-reference",
                        "sourceTaskSha256": {"recording": "a" * 64},
                        "frames": {"frame": annotation},
                    }
                )
            )

            loaded = BENCHMARK.load_reference(output, ["frame"])

            self.assertEqual(loaded, {"frame": annotation})

    def test_receipt_identity_cannot_be_reassigned_to_another_effort(self) -> None:
        receipt = {
            "configId": "sol-low",
            "model": "gpt-5.6-sol",
            "effort": "low",
            "serviceTier": "default",
        }

        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            BENCHMARK._validate_receipt_identity(
                receipt,
                Path("receipt.json"),
                "sol-high",
                "gpt-5.6-sol",
                "high",
            )


if __name__ == "__main__":
    unittest.main()
