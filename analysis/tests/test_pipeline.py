from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.config import DecoderConfig, FeatureConfig
from analysis.features import FeatureSequence, VideoMetadata
from analysis.model import LogisticModel, ModelError
from analysis.pipeline import PreparedRecording, _manifest_digest, evaluate_dataset
from analysis.schema import Recording, labels_for_times, load_manifest


class ImmutableEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="volleycut-evaluation-test-")
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        rows = []
        for split in ("train", "validation", "test"):
            video = self.root / f"{split}.mp4"
            video.write_bytes(f"synthetic-{split}".encode("ascii"))
            rows.append(
                {
                    "id": f"{split}-clip",
                    "video": video.name,
                    "split": split,
                    "sourceGroup": f"{split}-source",
                    "environment": "beach" if split == "test" else "indoor",
                    "game": {
                        "playersPerTeam": 2 if split == "test" else 4,
                        "targetPoints": None,
                    },
                    "consent": {
                        "analyze": True,
                        "train": split in {"train", "validation"},
                    },
                    "capture": {
                        "position": "centered-behind-endline",
                        "stationary": True,
                        "fullCourtVisible": True,
                        "serviceAreasVisible": True,
                    },
                    "roi": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
                    "rallies": [{"start": 0.5, "end": 1.5}],
                }
            )
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "name": "immutable-evaluation-test",
                    "annotationPolicy": {"id": "serve-contact-to-dead-ball-v1"},
                    "recordings": rows,
                }
            ),
            encoding="utf-8",
        )

    def save_model(
        self,
        *,
        validation_groups: list[str] | None = None,
    ) -> Path:
        manifest = load_manifest(self.manifest_path)
        model = LogisticModel(
            feature_config=FeatureConfig(
                analysis_fps=2.0,
                use_optical_flow=False,
                context_offsets_seconds=(0.0,),
            ),
            feature_names=("motion",),
            mean=np.asarray([0.0], dtype=np.float32),
            scale=np.asarray([1.0], dtype=np.float32),
            weights=np.asarray([0.0], dtype=np.float32),
            bias=-5.0,
            decoder=DecoderConfig(
                smoothing_seconds=0.0,
                enter_threshold=0.5,
                exit_threshold=0.4,
                min_live_seconds=0.0,
                bridge_gap_seconds=0.0,
            ),
            training_summary={
                "manifestSha256": _manifest_digest(manifest),
                "trainingSourceGroups": ["train-source"],
                "validationSourceGroups": validation_groups or ["validation-source"],
            },
        )
        return model.save(self.root / "model")

    @staticmethod
    def prepare_many(
        recordings: tuple[Recording, ...],
        _feature_config: FeatureConfig,
        _cache_dir: str | Path,
    ) -> list[PreparedRecording]:
        result = []
        times = np.asarray([0.25, 0.75, 1.25, 1.75], dtype=np.float64)
        for recording in recordings:
            values = np.zeros((len(times), 1), dtype=np.float32)
            sequence = FeatureSequence(
                times=times,
                values=values,
                names=("motion",),
                metadata=VideoMetadata(
                    duration=2.0,
                    width=1280,
                    height=720,
                    fps=2.0,
                    frame_count=4,
                    has_audio=False,
                ),
            )
            result.append(
                PreparedRecording(
                    recording=recording,
                    sequence=sequence,
                    contextual_values=values,
                    contextual_names=("motion",),
                    labels=labels_for_times(times, recording.rallies),
                )
            )
        return result

    def test_validation_is_tuning_only_and_test_is_held_out(self) -> None:
        model_path = self.save_model()
        with patch("analysis.pipeline._prepare_many", side_effect=self.prepare_many) as prepare:
            validation = evaluate_dataset(
                self.manifest_path,
                model_path,
                self.root / "cache",
                split="validation",
            )
            held_out = evaluate_dataset(
                self.manifest_path,
                model_path,
                self.root / "cache",
                split="test",
            )

        self.assertEqual(validation["assessmentRole"], "tuning-only")
        self.assertEqual(validation["recordings"][0]["id"], "validation-clip")
        self.assertEqual(held_out["assessmentRole"], "held-out-evaluation")
        self.assertEqual(held_out["recordings"][0]["id"], "test-clip")
        self.assertEqual(held_out["split"], "test")
        self.assertEqual(held_out["byPlayersPerTeam"]["2"]["recordings"], 1)
        self.assertEqual(held_out["byTargetPoints"]["unknown"]["recordings"], 1)
        self.assertEqual(prepare.call_count, 2)

    def test_evaluation_rejects_a_manifest_changed_after_training(self) -> None:
        model_path = self.save_model()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        payload["name"] = "mutated-after-training"
        self.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        with patch(
            "analysis.pipeline._prepare_many",
            side_effect=AssertionError("digest rejection must precede feature extraction"),
        ):
            with self.assertRaisesRegex(ModelError, "immutable manifest"):
                evaluate_dataset(
                    self.manifest_path,
                    model_path,
                    self.root / "cache",
                    split="test",
                )

    def test_held_out_evaluation_rejects_validation_group_leakage(self) -> None:
        model_path = self.save_model(validation_groups=["test-source"])

        with patch(
            "analysis.pipeline._prepare_many",
            side_effect=AssertionError("leakage rejection must precede feature extraction"),
        ):
            with self.assertRaisesRegex(ModelError, "leaks trained/tuned source groups"):
                evaluate_dataset(
                    self.manifest_path,
                    model_path,
                    self.root / "cache",
                    split="test",
                )


if __name__ == "__main__":
    unittest.main()
