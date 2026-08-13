from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from analysis.court_relative_experiment import (
    COURT_RELATIVE_SPEC_HASHES,
    COURT_RELATIVE_VARIANTS,
    SOURCE_FEATURE_NAMES,
    append_court_relative_features,
    court_relative_feature_spec,
    derive_court_relative_features,
)
from analysis.features import FeatureSequence, VideoMetadata
from analysis.pipeline import PreparedRecording
from analysis.schema import Interval, Recording


def _prepared(
    times: np.ndarray,
    *,
    player_grid: np.ndarray | None = None,
    diff_grid: np.ndarray | None = None,
    flow_grid: np.ndarray | None = None,
    centroid_x: np.ndarray | None = None,
    centroid_y: np.ndarray | None = None,
    spread_x: np.ndarray | None = None,
    spread_y: np.ndarray | None = None,
    flow_x: np.ndarray | None = None,
    flow_y: np.ndarray | None = None,
    recording_id: str = "court-sample",
) -> PreparedRecording:
    count = len(times)
    default_grid = np.tile(
        np.asarray(
            [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]],
            dtype=np.float64,
        ),
        (count, 1, 1),
    )
    player_grid = default_grid.copy() if player_grid is None else player_grid
    diff_grid = default_grid[:, ::-1, :].copy() if diff_grid is None else diff_grid
    flow_grid = default_grid[:, :, ::-1].copy() if flow_grid is None else flow_grid

    def series(value: np.ndarray | None, default: float) -> np.ndarray:
        return (
            np.full(count, default, dtype=np.float64)
            if value is None
            else np.asarray(value, dtype=np.float64)
        )

    columns: dict[str, np.ndarray] = {"unused": np.linspace(0.0, 1.0, count)}
    for prefix, grid in (
        ("player_motion_grid_", player_grid),
        ("diff_grid_", diff_grid),
        ("flow_grid_", flow_grid),
    ):
        flat = np.asarray(grid, dtype=np.float64).reshape((count, 9))
        for index in range(9):
            columns[f"{prefix}{index}"] = flat[:, index]
    columns.update(
        {
            "flow_median_x": series(flow_x, -0.2),
            "flow_median_y": series(flow_y, 0.4),
            "player_motion_centroid_x": series(centroid_x, 0.7),
            "player_motion_centroid_y": series(centroid_y, 0.8),
            "player_motion_spread_x": series(spread_x, 0.1),
            "player_motion_spread_y": series(spread_y, 0.2),
        }
    )
    # Deliberately reverse the required columns to prove lookup is by name, not offset.
    names = ("unused", *reversed(SOURCE_FEATURE_NAMES))
    values = np.column_stack([columns[name] for name in names]).astype(np.float32)
    sequence = FeatureSequence(
        times=np.asarray(times, dtype=np.float64),
        values=values,
        names=names,
        metadata=VideoMetadata(
            duration=float(times[-1] + 0.25) if count else 0.0,
            width=1920,
            height=1080,
            fps=30.0,
            frame_count=max(count, 1),
            has_audio=True,
        ),
    )
    recording = Recording(
        id=recording_id,
        video=Path(f"/{recording_id}.mp4"),
        split="train",
        source_group="source-a",
        environment="indoor",
        game={"playersPerTeam": 4},
        rallies=(Interval(0.5, 1.5, tags=("ace",)),),
        ignored_intervals=(),
        roi=(0.1, 0.1, 0.8, 0.8),
        capture={
            "position": "centered-behind-endline",
            "stationary": True,
            "fullCourtVisible": True,
            "serviceAreasVisible": True,
        },
        consent={"analyze": True, "train": True},
        content_sha256=None,
        raw={"id": recording_id, "sideSwitches": [{"time": 1.0}]},
    )
    return PreparedRecording(
        recording=recording,
        sequence=sequence,
        contextual_values=np.arange(count, dtype=np.float32).reshape((-1, 1)),
        contextual_names=("t+0s/unused_context",),
        labels=np.ones(count, dtype=np.float32),
        sample_mask=np.ones(count, dtype=np.bool_),
    )


def _column(
    values: np.ndarray, names: tuple[str, ...], name: str
) -> np.ndarray:
    return values[:, names.index(name)]


class CourtRelativeSpecTests(unittest.TestCase):
    def test_specs_are_stable_unique_and_combined_is_a_deduplicated_union(self) -> None:
        specs = {
            variant: court_relative_feature_spec(variant)
            for variant in COURT_RELATIVE_VARIANTS
        }

        for variant, spec in specs.items():
            self.assertEqual(len(spec.feature_names), len(set(spec.feature_names)))
            canonical = json.dumps(
                spec.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            expected_hash = hashlib.sha256(canonical).hexdigest()
            self.assertEqual(spec.sha256, expected_hash)
            self.assertEqual(COURT_RELATIVE_SPEC_HASHES[variant], expected_hash)
            self.assertEqual(spec.metadata()["specSha256"], expected_hash)

        invariant = specs["orientation_invariant"].feature_names
        fixed = specs["fixed_endline"].feature_names
        combined = specs["combined"].feature_names
        self.assertEqual(
            combined,
            (*invariant, *(name for name in fixed if name not in invariant)),
        )
        metadata = specs["combined"].metadata()
        self.assertEqual(metadata["family"], "court_relative")
        self.assertEqual(metadata["grid"]["order"], "row-major")
        self.assertEqual(metadata["grid"]["rows"][0], "far-third")
        self.assertFalse(metadata["temporal"]["causal"])
        self.assertIn("sideSwitches", metadata["excludedInputs"])
        self.assertIn("playersPerTeam", metadata["excludedInputs"])
        grouped_indexes = [
            index for indexes in specs["combined"].groups.values() for index in indexes
        ]
        self.assertEqual(
            sorted(grouped_indexes), list(range(len(specs["combined"].feature_names)))
        )
        self.assertEqual(len(grouped_indexes), len(set(grouped_indexes)))

    def test_unknown_variant_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown court-relative variant"):
            court_relative_feature_spec("learned-from-outcomes")


class CourtRelativeDerivationTests(unittest.TestCase):
    def test_row_major_regions_and_axis_directions_have_declared_meaning(self) -> None:
        times = np.asarray([0.0], dtype=np.float64)
        player = np.asarray(
            [[[1.0, 1.0, 1.0], [0.0, 2.0, 0.0], [3.0, 3.0, 3.0]]]
        )
        prepared = _prepared(times, player_grid=player)

        values, names = derive_court_relative_features(prepared, "combined")

        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(values.flags.c_contiguous)
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(
            float(
                _column(
                    values,
                    names,
                    "derived/court_relative/fixed/player_motion_near_minus_far_contrast",
                )[0]
            ),
            0.5,
        )
        # Center=2 and the outer-ring mean=(14-2)/8=1.5.
        self.assertAlmostEqual(
            float(
                _column(
                    values,
                    names,
                    "derived/court_relative/player_motion_center_minus_margin_contrast",
                )[0]
            ),
            1.0 / 7.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(
                _column(
                    values,
                    names,
                    "derived/court_relative/flow_along_axis_fraction",
                )[0]
            ),
            2.0 / 3.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(
                _column(
                    values,
                    names,
                    "derived/court_relative/fixed/flow_toward_near_signed",
                )[0]
            ),
            0.4,
            places=6,
        )

    def test_invariant_bank_survives_both_axis_reflections_while_signed_bank_changes(self) -> None:
        times = np.arange(0.0, 2.25, 0.25, dtype=np.float64)
        count = len(times)
        player = np.zeros((count, 3, 3), dtype=np.float64)
        player[times >= 0.5, 2, :] = 0.03
        player[times >= 1.0, 0, :] = 0.03
        diff = np.tile(np.arange(1.0, 10.0).reshape((1, 3, 3)), (count, 1, 1))
        flow = np.tile(np.arange(9.0, 0.0, -1.0).reshape((1, 3, 3)), (count, 1, 1))
        centroid_x = np.linspace(0.55, 0.8, count)
        centroid_y = np.linspace(0.4, 0.75, count)
        original = _prepared(
            times,
            player_grid=player,
            diff_grid=diff,
            flow_grid=flow,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
        )
        reflected = _prepared(
            times,
            player_grid=player[:, ::-1, ::-1],
            diff_grid=diff[:, ::-1, ::-1],
            flow_grid=flow[:, ::-1, ::-1],
            centroid_x=1.0 - centroid_x,
            centroid_y=1.0 - centroid_y,
            flow_x=np.full(count, 0.2),
            flow_y=np.full(count, -0.4),
        )

        invariant, invariant_names = derive_court_relative_features(
            original, "orientation_invariant"
        )
        reflected_invariant, reflected_names = derive_court_relative_features(
            reflected, "orientation_invariant"
        )
        self.assertEqual(invariant_names, reflected_names)
        np.testing.assert_allclose(invariant, reflected_invariant, atol=1e-6)

        fixed, fixed_names = derive_court_relative_features(original, "fixed_endline")
        reflected_fixed, reflected_fixed_names = derive_court_relative_features(
            reflected, "fixed_endline"
        )
        self.assertEqual(fixed_names, reflected_fixed_names)
        signed_name = (
            "derived/court_relative/fixed/player_motion_centroid_toward_near_signed"
        )
        np.testing.assert_allclose(
            _column(fixed, fixed_names, signed_name),
            -_column(reflected_fixed, reflected_fixed_names, signed_name),
            atol=1e-6,
        )
        self.assertFalse(np.allclose(fixed, reflected_fixed))

    def test_near_then_far_motion_emits_onsets_and_half_second_lag(self) -> None:
        times = np.arange(0.0, 2.25, 0.25, dtype=np.float64)
        player = np.zeros((len(times), 3, 3), dtype=np.float64)
        player[times >= 0.5, 2, :] = 0.03
        player[times >= 1.0, 0, :] = 0.03
        prepared = _prepared(times, player_grid=player)

        values, names = derive_court_relative_features(prepared, "fixed_endline")

        near = _column(
            values, names, "derived/court_relative/fixed/near_reaction_onset"
        )
        far = _column(
            values, names, "derived/court_relative/fixed/far_reaction_onset"
        )
        lag = _column(
            values,
            names,
            "derived/court_relative/fixed/far_minus_near_reaction_lag_seconds",
        )
        self.assertGreater(float(np.max(near)), 0.0)
        self.assertGreater(float(np.max(far)), 0.0)
        self.assertAlmostEqual(float(lag[0]), 0.5, places=6)

    def test_annotations_and_confounded_metadata_cannot_change_features(self) -> None:
        times = np.arange(0.0, 1.25, 0.25, dtype=np.float64)
        first = _prepared(times)
        second_recording = replace(
            first.recording,
            environment="beach",
            game={"playersPerTeam": 2, "targetPoints": 21},
            rallies=(Interval(0.25, 0.5, tags=("service-fault",)),),
            raw={"id": first.recording.id, "sideSwitches": []},
        )
        second = replace(
            first,
            recording=second_recording,
            labels=np.zeros(len(times), dtype=np.float32),
            sample_mask=np.zeros(len(times), dtype=np.bool_),
        )

        first_values, first_names = derive_court_relative_features(first, "combined")
        second_values, second_names = derive_court_relative_features(
            second, "combined"
        )

        self.assertEqual(first_names, second_names)
        np.testing.assert_array_equal(first_values, second_values)

    def test_append_adds_current_time_bank_once_without_changing_sequence(self) -> None:
        prepared = _prepared(np.arange(0.0, 1.25, 0.25, dtype=np.float64))
        derived, names = derive_court_relative_features(
            prepared, "orientation_invariant"
        )

        appended = append_court_relative_features(prepared, "orientation_invariant")

        self.assertIs(appended.sequence, prepared.sequence)
        self.assertEqual(
            appended.contextual_names,
            (*prepared.contextual_names, *names),
        )
        np.testing.assert_array_equal(
            appended.contextual_values[:, :1], prepared.contextual_values
        )
        np.testing.assert_array_equal(appended.contextual_values[:, 1:], derived)
        self.assertFalse(any(name.startswith("t+") for name in names))
        with self.assertRaisesRegex(ValueError, "already present"):
            append_court_relative_features(appended, "orientation_invariant")

    def test_missing_roi_unsupported_capture_and_non_3x3_cache_are_rejected(self) -> None:
        prepared = _prepared(np.asarray([0.0, 0.25], dtype=np.float64))
        with self.assertRaisesRegex(ValueError, "rectangle ROI"):
            derive_court_relative_features(
                replace(prepared, recording=replace(prepared.recording, roi=None)),
                "orientation_invariant",
            )
        with self.assertRaisesRegex(ValueError, "fixed end-line capture"):
            derive_court_relative_features(
                replace(
                    prepared,
                    recording=replace(
                        prepared.recording,
                        capture={**prepared.recording.capture, "stationary": False},
                    ),
                ),
                "fixed_endline",
            )

        sequence = replace(
            prepared.sequence,
            values=np.column_stack(
                (prepared.sequence.values, np.zeros(len(prepared.sequence.times)))
            ),
            names=(*prepared.sequence.names, "flow_grid_9"),
        )
        with self.assertRaisesRegex(ValueError, "exact cached 3x3 grids"):
            derive_court_relative_features(
                replace(prepared, sequence=sequence), "combined"
            )


if __name__ == "__main__":
    unittest.main()
