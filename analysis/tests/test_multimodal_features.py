from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.config import DecoderConfig, FeatureConfig
from analysis.feature_families import (
    FEATURE_FAMILY_KINDS,
    base_feature_name,
    feature_family,
    grouped_base_features,
    grouped_feature_indexes,
)
from analysis.features import (
    ABSOLUTE_FEATURE_NAMES,
    FeatureSequence,
    VideoMetadata,
    _audio_feature_names,
    _audio_features_from_samples,
    _frame_feature_names,
    _frame_features,
    contextualize,
    feature_names,
)
from analysis.model import LogisticModel, load_model


try:
    import cv2
except Exception:  # dependency test must tolerate missing or ABI-incompatible OpenCV
    cv2 = None


def metadata(*, has_audio: bool | None = True) -> VideoMetadata:
    return VideoMetadata(
        duration=3.0,
        width=160,
        height=96,
        fps=30.0,
        frame_count=90,
        has_audio=has_audio,
    )


class FeatureFamilyCoverageTests(unittest.TestCase):
    def test_contextual_feature_families_are_deterministic_and_exhaustive(self) -> None:
        offsets = (-1.0, 0.0, 1.0)
        config = FeatureConfig(
            analysis_fps=4.0,
            resize_width=64,
            resize_height=36,
            grid_size=2,
            context_offsets_seconds=offsets,
            sequence_normalization="none",
        )
        base_names = feature_names(config)
        times = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        sequence = FeatureSequence(
            times=times,
            values=np.zeros((len(times), len(base_names)), dtype=np.float32),
            names=base_names,
            metadata=metadata(),
        )

        _, contextual_names = contextualize(sequence, config)
        expected_names = tuple(
            f"t{offset:+g}s/{name}" for offset in offsets for name in base_names
        )
        self.assertEqual(contextual_names, expected_names)
        self.assertEqual(len(contextual_names), len(set(contextual_names)))

        first = grouped_feature_indexes(contextual_names)
        second = grouped_feature_indexes(contextual_names)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(FEATURE_FAMILY_KINDS))
        covered = [index for indexes in first.values() for index in indexes]
        self.assertEqual(sorted(covered), list(range(len(contextual_names))))
        self.assertEqual(len(covered), len(set(covered)))

        grouped_bases = grouped_base_features(contextual_names)
        for family, indexes in first.items():
            self.assertTrue(indexes)
            self.assertTrue(
                all(feature_family(contextual_names[index]) == family for index in indexes)
            )
            expected_bases = grouped_bases[family]
            for name in expected_bases:
                occurrences = sum(
                    base_feature_name(contextual_name) == name
                    for contextual_name in contextual_names
                )
                self.assertEqual(occurrences, len(offsets))


class AudioFeatureTests(unittest.TestCase):
    @staticmethod
    def config() -> FeatureConfig:
        return FeatureConfig(
            analysis_fps=4.0,
            use_optical_flow=False,
            use_advanced_visual=False,
            use_audio=True,
            audio_sample_rate=16000,
            context_offsets_seconds=(0.0,),
            sequence_normalization="none",
        )

    def test_impulse_features_are_finite_aligned_and_distinct_from_quiet(self) -> None:
        config = self.config()
        times = np.arange(0.0, 3.0, 1.0 / config.analysis_fps, dtype=np.float64)
        samples = np.zeros(3 * config.audio_sample_rate, dtype=np.float32)
        impulse_time = 1.225  # center of a 50 ms analysis frame
        samples[round(impulse_time * config.audio_sample_rate)] = 1.0

        values = _audio_features_from_samples(samples, times, config, available=True)
        names = _audio_feature_names(config)
        indexes = {name: names.index(name) for name in names}

        self.assertEqual(values.shape, (len(times), len(names)))
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(
            values[:, indexes["audio_available"]], np.ones(len(times), dtype=np.float32)
        )

        quiet = times < 1.0
        impulse_row = int(np.argmin(np.abs(times - impulse_time)))
        self.assertLessEqual(abs(float(times[impulse_row]) - impulse_time), 0.125)
        for name in (
            "audio_rms",
            "audio_peak",
            "audio_spectral_flux",
            "audio_onset_strength",
            "audio_contact_like_transient",
        ):
            column = values[:, indexes[name]]
            self.assertEqual(int(np.argmax(column)), impulse_row, name)
            self.assertGreater(float(column[impulse_row]), float(np.max(column[quiet])), name)

        cadence = values[:, indexes["audio_onset_cadence"]]
        cadence_collapse = values[:, indexes["audio_cadence_collapse"]]
        self.assertGreater(
            float(np.mean(cadence[(times >= 1.25) & (times <= 1.75)])),
            float(np.mean(cadence[quiet])),
        )
        self.assertGreater(
            float(np.max(cadence_collapse[times >= 1.0])),
            float(np.max(cadence_collapse[quiet])),
        )

        elapsed = values[:, indexes["audio_seconds_since_transient"]]
        self.assertTrue(np.all(elapsed[quiet] == 10.0))
        after = elapsed[(times >= 1.5) & (times <= 2.5)]
        self.assertGreater(len(after), 1)
        self.assertLess(float(after[0]), 1.0)
        self.assertTrue(np.all(np.diff(after) >= -1e-6))

    def test_missing_audio_is_zero_with_zero_availability(self) -> None:
        config = self.config()
        times = np.arange(0.0, 2.0, 1.0 / config.analysis_fps, dtype=np.float64)

        values = _audio_features_from_samples(
            np.empty(0, dtype=np.float32),
            times,
            config,
            available=False,
        )

        self.assertEqual(values.shape, (len(times), len(_audio_feature_names(config))))
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(values, np.zeros_like(values))


class ContextNormalizationTests(unittest.TestCase):
    def test_absolute_quality_and_availability_channels_survive_percentile_ranking(self) -> None:
        absolute_names = tuple(sorted(ABSOLUTE_FEATURE_NAMES))
        names = (*absolute_names, "luma_mean")
        times = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        absolute_values = np.asarray(
            [
                np.linspace(0.10, 0.60, len(absolute_names)),
                np.linspace(0.80, 0.30, len(absolute_names)),
                np.linspace(0.35, 0.85, len(absolute_names)),
            ],
            dtype=np.float32,
        )
        values = np.column_stack(
            (absolute_values, np.asarray([30.0, 10.0, 20.0], dtype=np.float32))
        ).astype(np.float32)
        sequence = FeatureSequence(
            times=times,
            values=values,
            names=names,
            metadata=metadata(),
        )
        config = FeatureConfig(
            use_optical_flow=False,
            use_advanced_visual=False,
            use_audio=False,
            context_offsets_seconds=(0.0,),
            sequence_normalization="percentile-rank",
        )

        contextual, contextual_names = contextualize(sequence, config)

        for index, name in enumerate(absolute_names):
            contextual_index = contextual_names.index(f"t+0s/{name}")
            np.testing.assert_array_equal(
                contextual[:, contextual_index], values[:, index]
            )
        luma_index = contextual_names.index("t+0s/luma_mean")
        np.testing.assert_array_equal(
            contextual[:, luma_index],
            np.asarray([1.0, 0.0, 0.5], dtype=np.float32),
        )


@unittest.skipUnless(cv2 is not None, "OpenCV is not available to the test interpreter")
class CameraCompensationTests(unittest.TestCase):
    def test_global_translation_is_camera_motion_while_local_motion_remains_residual(self) -> None:
        assert cv2 is not None
        config = FeatureConfig(
            analysis_fps=4.0,
            resize_width=160,
            resize_height=96,
            grid_size=3,
            use_optical_flow=True,
            use_advanced_visual=True,
            use_audio=False,
            context_offsets_seconds=(0.0,),
            sequence_normalization="none",
        )
        rng = np.random.default_rng(431)
        texture = rng.integers(0, 256, size=(96, 160), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (5, 5), 0)
        background = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

        global_translation = cv2.warpAffine(
            background,
            np.float32([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0]]),
            (160, 96),
            borderMode=cv2.BORDER_REFLECT,
        )
        local_before = background.copy()
        local_after = background.copy()
        cv2.rectangle(local_before, (28, 28), (68, 68), (255, 255, 255), -1)
        cv2.rectangle(local_after, (44, 28), (84, 68), (255, 255, 255), -1)

        _, global_previous = _frame_features(background, None, config)
        global_values, _ = _frame_features(
            global_translation, global_previous, config
        )
        _, local_previous = _frame_features(local_before, None, config)
        local_values, _ = _frame_features(local_after, local_previous, config)
        names = _frame_feature_names(config)

        def value(row: np.ndarray, name: str) -> float:
            return float(row[names.index(name)])

        global_shift = value(global_values, "camera_shift_magnitude")
        local_shift = value(local_values, "camera_shift_magnitude")
        self.assertGreater(global_shift, local_shift + 0.005)
        self.assertGreater(value(local_values, "player_motion_mean"), 0.0)

        global_ratio = value(global_values, "player_motion_mean") / max(
            value(global_values, "flow_mean"), 1e-8
        )
        local_ratio = value(local_values, "player_motion_mean") / max(
            value(local_values, "flow_mean"), 1e-8
        )
        self.assertGreater(local_ratio, global_ratio + 0.10)


class LegacyCompatibilityTests(unittest.TestCase):
    @staticmethod
    def legacy_config_payload() -> dict[str, object]:
        return {
            "analysis_fps": 4.0,
            "resize_width": 192,
            "resize_height": 108,
            "grid_size": 3,
            "use_optical_flow": True,
            "context_offsets_seconds": [0.0],
            "sequence_normalization": "percentile-rank",
        }

    def test_v1_feature_config_defaults_new_families_off(self) -> None:
        config = FeatureConfig.from_dict(self.legacy_config_payload())

        self.assertFalse(config.use_advanced_visual)
        self.assertFalse(config.use_audio)
        self.assertEqual(config.audio_sample_rate, 16000)
        names = feature_names(config)
        self.assertEqual(len(names), 42)
        self.assertFalse(any(name.startswith("audio_") for name in names))
        self.assertFalse(any(name.startswith("player_motion_") for name in names))
        self.assertNotIn("focus_quality", names)

    def test_synthetic_v1_model_with_old_config_and_decoder_loads(self) -> None:
        config = FeatureConfig.from_dict(self.legacy_config_payload())
        model = LogisticModel(
            feature_config=config,
            feature_names=("t+0s/luma_mean",),
            mean=np.asarray([0.25], dtype=np.float32),
            scale=np.asarray([0.5], dtype=np.float32),
            weights=np.asarray([1.5], dtype=np.float32),
            bias=-0.2,
            decoder=DecoderConfig(
                smoothing_seconds=0.5,
                enter_threshold=0.5,
                exit_threshold=0.4,
                min_live_seconds=2.0,
                bridge_gap_seconds=1.0,
                short_event_min_seconds=2.0,
                short_event_threshold=1.0,
            ),
            training_summary={"dataset": "synthetic-legacy"},
            feature_version="court-motion-flow-v1",
        )
        probe = np.asarray([[0.0], [0.5], [1.0]], dtype=np.float32)
        expected = model.predict(probe)

        with tempfile.TemporaryDirectory(prefix="volleycut-v1-compat-") as directory:
            model_dir = model.save(Path(directory) / "model")
            metadata_path = model_dir / "model.json"
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in ("use_advanced_visual", "use_audio", "audio_sample_rate"):
                payload["featureConfig"].pop(key)
            for key in ("short_event_min_seconds", "short_event_threshold"):
                payload["decoder"].pop(key)
            metadata_path.write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )

            restored = load_model(model_dir)

        self.assertEqual(restored.feature_version, "court-motion-flow-v1")
        self.assertFalse(restored.feature_config.use_advanced_visual)
        self.assertFalse(restored.feature_config.use_audio)
        self.assertEqual(restored.decoder.short_event_min_seconds, 2.0)
        self.assertEqual(restored.decoder.short_event_threshold, 1.0)
        np.testing.assert_allclose(restored.predict(probe), expected, atol=1e-7, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
