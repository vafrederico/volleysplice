from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.config import FEATURE_VERSION, DecoderConfig, TrainingConfig
from analysis.feature_experiments import (
    FeatureExperimentError,
    _subset_prepared,
    build_fold_plan,
    sha256_file,
)
from analysis.highres_embedding_experiment import (
    CROP_SPECS,
    DEFAULT_BACKBONE_LAYER,
    HighresCache,
    OpenCvFrozenBackbone,
    _highres_code_provenance,
    _load_frozen_highres_study,
    _validated_highres_cache_inventory,
    _write_cache_no_replace,
    build_extraction_spec,
    cache_path_for,
    crop_views,
    highres_feature_block,
    prepare_highres_study_candidates,
    run_retrospective_highres_test,
    sample_times,
    signed_projection_matrix,
)
from analysis.pipeline import _manifest_digest
from analysis.schema import DatasetManifest, Recording
from analysis.tests.test_transition_feature_experiment import (
    _feature_config,
    _prepared,
    _recording,
)
from analysis.transition_feature_experiment import (
    _signature_sha256,
    prepare_transition_candidates,
)


class _FakeNet:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.input: np.ndarray | None = None

    def getLayerNames(self) -> tuple[str, ...]:
        return ("stem", DEFAULT_BACKBONE_LAYER, "output")

    def setInput(self, values: np.ndarray) -> None:
        self.input = values

    def setPreferableBackend(self, backend: int) -> None:
        assert backend == 3

    def setPreferableTarget(self, target: int) -> None:
        assert target == 0

    def forward(self, layer: str) -> np.ndarray:
        assert layer == DEFAULT_BACKBONE_LAYER
        assert self.input is not None
        base = float(np.mean(self.input))
        values = np.arange(1, self.dimension + 1, dtype=np.float32) + base
        return values.reshape(1, self.dimension, 1, 1)


class _FakeDnn:
    DNN_BACKEND_OPENCV = 3
    DNN_TARGET_CPU = 0

    def __init__(self, dimension: int) -> None:
        self.net = _FakeNet(dimension)

    def readNetFromONNX(self, _: str) -> _FakeNet:
        return self.net

class _FakeCv2:
    INTER_LINEAR = 1
    __version__ = "test-double-1"

    def __init__(self, dimension: int) -> None:
        self.dnn = _FakeDnn(dimension)

    @staticmethod
    def resize(
        image: np.ndarray,
        size: tuple[int, int],
        *,
        interpolation: int,
    ) -> np.ndarray:
        assert size == (9, 9)
        assert interpolation == _FakeCv2.INTER_LINEAR
        y_indexes = np.linspace(0, image.shape[0] - 1, size[1]).round().astype(int)
        x_indexes = np.linspace(0, image.shape[1] - 1, size[0]).round().astype(int)
        return image[y_indexes][:, x_indexes]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HighresEmbeddingExperimentTest(unittest.TestCase):
    def test_signed_projection_is_deterministic_and_content_addressed(self) -> None:
        first, first_spec = signed_projection_matrix(32, 7, 19)
        second, second_spec = signed_projection_matrix(32, 7, 19)
        other, other_spec = signed_projection_matrix(32, 7, 20)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_spec, second_spec)
        self.assertEqual(
            first_spec["matrixSha256"],
            hashlib.sha256(first.astype("<f4", copy=False).tobytes()).hexdigest(),
        )
        np.testing.assert_allclose(
            np.linalg.norm(first, axis=0), np.ones(7), atol=1e-6
        )
        self.assertFalse(np.array_equal(first, other))
        self.assertNotEqual(first_spec["specSha256"], other_spec["specSha256"])

    def test_crop_views_apply_roi_then_stable_proxy_rectangles(self) -> None:
        y, x = np.mgrid[:100, :200]
        frame = np.stack((x, y, x + y), axis=2).astype(np.uint16)
        crops = crop_views(frame, (0.25, 0.10, 0.50, 0.80))

        self.assertEqual(len(crops), len(CROP_SPECS))
        self.assertEqual(len(crops), 4)
        self.assertEqual(crops[0].shape, (80, 100, 3))
        self.assertEqual(crops[1].shape, (27, 100, 3))
        self.assertEqual(crops[2].shape, (40, 100, 3))
        self.assertEqual(crops[3].shape, (16, 100, 3))
        self.assertEqual(int(crops[0][0, 0, 0]), 50)
        self.assertEqual(int(crops[0][0, 0, 1]), 10)
        self.assertGreater(
            float(np.mean(crops[1][:, :, 1])),
            float(np.mean(crops[2][:, :, 1])),
        )

    def test_mocked_backbone_exposes_penultimate_layer_and_projects_crops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "backbone.onnx"
            model.write_bytes(b"frozen-test-backbone")
            backbone = OpenCvFrozenBackbone(
                model,
                expected_source_dimension=16,
                input_size=8,
                cv2_module=_FakeCv2(16),
            )
            spec, projection = build_extraction_spec(
                backbone,
                sample_fps=1.0,
                projection_dimension=5,
                projection_seed=3,
            )
            synthetic = np.zeros((20, 40, 3), dtype=np.uint8)
            synthetic[:, 20:] = 200
            values = backbone.project_crops(crop_views(synthetic, None), projection)

            self.assertEqual(backbone.sha256, sha256(model))
            self.assertEqual(spec.source_dimension, 16)
            self.assertEqual(spec.projection_dimension, 5)
            self.assertEqual(spec.output_kind, "penultimate-global-average-pool")
            self.assertEqual(spec.opencv_version, "test-double-1")
            self.assertEqual(spec.dnn_backend, "DNN_BACKEND_OPENCV")
            self.assertEqual(spec.dnn_target, "DNN_TARGET_CPU")
            self.assertEqual(values.shape, (4, 5))
            self.assertTrue(np.isfinite(values).all())

            colored = np.empty((10, 12, 3), dtype=np.uint8)
            colored[:, :, 0] = 10
            colored[:, :, 1] = 20
            colored[:, :, 2] = 30
            backbone._forward_one(colored)
            blob = backbone.net.input
            assert blob is not None
            self.assertEqual(blob.shape, (1, 3, 8, 8))
            self.assertAlmostEqual(
                float(blob[0, 0, 0, 0]),
                (30.0 / 255.0 - 0.485) / 0.229,
                places=5,
            )
            self.assertAlmostEqual(
                float(blob[0, 1, 0, 0]),
                (20.0 / 255.0 - 0.456) / 0.224,
                places=5,
            )
            self.assertAlmostEqual(
                float(blob[0, 2, 0, 0]),
                (10.0 / 255.0 - 0.406) / 0.225,
                places=5,
            )

    def test_temporal_block_aligns_one_fps_cache_to_four_fps_once(self) -> None:
        source_times = np.arange(5, dtype=np.float64)
        source = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
        cache = HighresCache(
            path=Path("synthetic.npz"),
            times=source_times,
            values=source,
            crop_names=tuple(item.name for item in CROP_SPECS),
            metadata={},
        )
        target_times = np.arange(17, dtype=np.float64) / 4.0
        block = highres_feature_block(
            cache,
            target_times,
            short_window_seconds=1.0,
            long_window_seconds=6.0,
        )

        self.assertEqual(block.values.shape, (17, 4 * 3 * 2))
        self.assertEqual(len(block.names), block.values.shape[1])
        self.assertEqual(len(set(block.names)), len(block.names))
        self.assertTrue(all(name.startswith("highres/") for name in block.names))
        self.assertTrue(np.isfinite(block.values).all())
        self.assertGreaterEqual(float(np.min(block.values)), 0.0)
        self.assertLessEqual(float(np.max(block.values)), 1.0)

    def test_sample_times_never_cross_duration(self) -> None:
        np.testing.assert_array_equal(
            sample_times(2.25, 1.0), np.asarray([0.0, 1.0, 2.0])
        )
        np.testing.assert_array_equal(sample_times(0.2, 1.0), np.asarray([0.0]))

    def test_retrospective_gate_fails_before_manifest_or_cache_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "not-promoted.json"
            report.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "volleycut-frozen-highres-embedding-study-development",
                        "freezeStatus": "frozen-development-selection",
                        "testLabelsUsed": False,
                        "testRecordingsPrepared": False,
                        "selectionProtocol": {"innerFoldLimit": None},
                        "candidateSelection": {
                            "selected": "frozen_binary_control",
                            "highresPromoted": False,
                        },
                        "selectedCandidateForRetrospectiveTest": "frozen_binary_control",
                        "finalizationPlan": {"candidate": "frozen_binary_control"},
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "analysis.highres_embedding_experiment.load_manifest",
                side_effect=AssertionError(
                    "failed promotion must stop before manifest/test access"
                ),
            ):
                with self.assertRaisesRegex(FeatureExperimentError, "gated off"):
                    run_retrospective_highres_test(
                        Path(temporary) / "manifest.json",
                        report,
                        Path(temporary) / "warm",
                        Path(temporary) / "highres",
                        spec=None,  # type: ignore[arg-type]
                        test_cache_index_path=Path(temporary) / "test-index.json",
                    )

    def test_retrospective_loader_rejects_inner_fold_smoke_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "smoke.json"
            report.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "volleycut-frozen-highres-embedding-study-development",
                        "freezeStatus": "frozen-development-selection",
                        "testLabelsUsed": False,
                        "testRecordingsPrepared": False,
                        "selectionProtocol": {"innerFoldLimit": 1},
                        "candidateSelection": {
                            "selected": "control_plus_frozen_highres",
                            "highresPromoted": True,
                        },
                        "selectedCandidateForRetrospectiveTest": (
                            "control_plus_frozen_highres"
                        ),
                        "finalizationPlan": {
                            "candidate": "control_plus_frozen_highres"
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FeatureExperimentError, "full nested"):
                _load_frozen_highres_study(report)

    def test_cache_inventory_revalidates_frozen_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "backbone.onnx"
            model.write_bytes(b"frozen-test-backbone")
            backbone = OpenCvFrozenBackbone(
                model,
                expected_source_dimension=16,
                input_size=8,
                cv2_module=_FakeCv2(16),
            )
            spec, _ = build_extraction_spec(
                backbone,
                sample_fps=1.0,
                projection_dimension=5,
                projection_seed=3,
            )
            recording = Recording(
                id="dev-a",
                video=root / "dev-a.mp4",
                split="train",
                source_group="a",
                environment="indoor",
                game={},
                rallies=(),
                ignored_intervals=(),
                roi=None,
                capture={},
                consent={},
                content_sha256="a" * 64,
                raw={},
            )
            cache_path = cache_path_for(root / "caches", recording, spec)
            metadata = {
                "schemaVersion": 1,
                "kind": "volleycut-frozen-highres-projected-representations",
                "recordingId": recording.id,
                "recordingContentSha256": recording.content_sha256,
                "extractorConfigSha256": spec.config_sha256,
                "extractor": spec.to_dict(),
                "labelsUsed": False,
            }
            _write_cache_no_replace(
                cache_path,
                times=np.asarray([0.0, 1.0], dtype=np.float64),
                values=np.zeros((2, len(CROP_SPECS), 5), dtype=np.float32),
                metadata=metadata,
            )
            actual = _validated_highres_cache_inventory(
                (recording,), root / "caches", spec
            )
            self.assertEqual(actual[recording.id]["path"], str(cache_path))
            self.assertEqual(
                actual[recording.id]["fileSha256"], sha256(cache_path)
            )

            tampered_frozen = {
                recording.id: {**actual[recording.id], "fileSha256": "0" * 64}
            }
            with self.assertRaisesRegex(FeatureExperimentError, "identity changed"):
                _validated_highres_cache_inventory(
                    (recording,),
                    root / "caches",
                    spec,
                    frozen_inventory=tampered_frozen,
                )

    def test_synthetic_promoted_retrospective_reports_frozen_test_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _feature_config()
            rows = []
            for recording_id, group, split in (
                ("dev-a", "a", "train"),
                ("dev-b", "b", "train"),
                ("dev-c", "c", "train"),
                ("dev-d", "d", "validation"),
                ("test-e", "e", "test"),
            ):
                raw = _recording(root, recording_id, group, split)
                rows.append(
                    replace(raw, content_sha256=sha256_file(raw.video))
                )
            manifest_payload = {
                "schemaVersion": 1,
                "name": "synthetic-highres-retrospective",
                "recordings": [item.raw for item in rows],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
            manifest = DatasetManifest(
                path=manifest_path,
                name="synthetic-highres-retrospective",
                recordings=tuple(rows),
                raw=manifest_payload,
            )
            prepared_by_id = {item.id: _prepared(item, config) for item in rows}
            development_rows = tuple(
                item for item in rows if item.split in {"train", "validation"}
            )
            development_prepared = [
                prepared_by_id[item.id] for item in development_rows
            ]
            transition = prepare_transition_candidates(development_prepared, config)
            control_spec = next(
                item for item in transition.candidates if item.name == "baseline_450"
            )
            control = _subset_prepared(transition.prepared, control_spec.indexes)
            frozen_names = control[0].contextual_names
            training = TrainingConfig(
                epochs=1,
                batch_size=64,
                learning_rate=0.01,
                patience=1,
                seed=11,
            )
            decoder = DecoderConfig(
                smoothing_seconds=0.0,
                enter_threshold=0.5,
                exit_threshold=0.4,
                min_live_seconds=1.0,
                bridge_gap_seconds=0.0,
                short_event_min_seconds=0.5,
                short_event_threshold=0.8,
            )
            folds = [item.to_dict() for item in build_fold_plan(manifest.recordings)]
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
                "trainingConfig": training.to_dict(),
                "baseDecoderConfig": decoder.to_dict(),
                "selectionProtocol": {"innerFoldLimit": None, "folds": folds},
                "selectedCandidateForRetrospectiveTest": "baseline_450",
                "finalizationPlan": {
                    "candidate": "baseline_450",
                    "featureNames": list(frozen_names),
                    "featureSignatureSha256": _signature_sha256(frozen_names),
                },
            }
            upstream_path = root / "upstream.json"
            upstream_path.write_text(json.dumps(upstream_payload), encoding="utf-8")

            model_path = root / "backbone.onnx"
            model_path.write_bytes(b"frozen-test-backbone")
            backbone = OpenCvFrozenBackbone(
                model_path,
                expected_source_dimension=16,
                input_size=8,
                cv2_module=_FakeCv2(16),
            )
            extraction_spec, _ = build_extraction_spec(
                backbone,
                sample_fps=1.0,
                projection_dimension=2,
                projection_seed=7,
            )
            highres_cache_dir = root / "highres"
            for index, recording in enumerate(rows):
                destination = cache_path_for(
                    highres_cache_dir, recording, extraction_spec
                )
                times = np.arange(9, dtype=np.float64)
                base = np.arange(
                    len(times) * len(CROP_SPECS) * 2,
                    dtype=np.float32,
                ).reshape(len(times), len(CROP_SPECS), 2)
                values = base + np.float32(index / 10.0)
                _write_cache_no_replace(
                    destination,
                    times=times,
                    values=values,
                    metadata={
                        "schemaVersion": 1,
                        "kind": "volleycut-frozen-highres-projected-representations",
                        "recordingId": recording.id,
                        "recordingContentSha256": recording.content_sha256,
                        "extractorConfigSha256": extraction_spec.config_sha256,
                        "extractor": extraction_spec.to_dict(),
                        "labelsUsed": False,
                    },
                )
            study = prepare_highres_study_candidates(
                development_prepared,
                config,
                frozen_names,
                extraction_spec,
                highres_cache_dir,
            )
            selected_spec = next(
                item
                for item in study.candidates
                if item.name == "control_plus_frozen_highres"
            )
            selected = _subset_prepared(study.prepared, selected_spec.indexes)
            selected_names = selected[0].contextual_names
            frozen_inventory = _validated_highres_cache_inventory(
                development_rows,
                highres_cache_dir,
                extraction_spec,
            )
            test_recording = next(item for item in rows if item.split == "test")
            test_cache_path = cache_path_for(
                highres_cache_dir, test_recording, extraction_spec
            )
            test_cache = _validated_highres_cache_inventory(
                (test_recording,), highres_cache_dir, extraction_spec
            )[test_recording.id]
            test_extraction_payload = {
                "schemaVersion": 1,
                "kind": "volleycut-frozen-highres-cache-extraction-index",
                "manifestFileSha256": sha256_file(manifest.path),
                "selectedSplits": ["test"],
                "protectedSplitAcknowledged": True,
                "labelsUsed": False,
                "extractorConfigSha256": extraction_spec.config_sha256,
                "extractor": extraction_spec.to_dict(),
                "recordings": [
                    {
                        "recordingId": test_recording.id,
                        "split": "test",
                        "sourceGroup": test_recording.source_group,
                        "recordingContentSha256": test_recording.content_sha256,
                        "cache": str(test_cache_path),
                        "cacheFileSha256": test_cache["fileSha256"],
                        "status": "created",
                        "samples": test_cache["samples"],
                    }
                ],
            }
            test_extraction_path = root / "test-extraction-index.json"
            test_extraction_path.write_text(
                json.dumps(test_extraction_payload), encoding="utf-8"
            )
            development_payload = {
                "schemaVersion": 1,
                "kind": "volleycut-frozen-highres-embedding-study-development",
                "freezeStatus": "frozen-development-selection",
                "testLabelsUsed": False,
                "testRecordingsPrepared": False,
                "manifestFileSha256": sha256_file(manifest.path),
                "manifestSnapshotSha256": _manifest_digest(manifest),
                "recordingContentSha256": {
                    item.id: item.content_sha256 for item in manifest.recordings
                },
                "featureVersion": FEATURE_VERSION,
                "featureConfig": config.to_dict(),
                "trainingConfig": training.to_dict(),
                "baseDecoderConfig": decoder.to_dict(),
                "selectionProtocol": {"innerFoldLimit": None, "folds": folds},
                "pairedComparisonAgainstControl": {
                    "classification": {"classification": "helpful"}
                },
                "candidates": {
                    "frozen_binary_control": {},
                    "control_plus_frozen_highres": {},
                },
                "selectedCandidateForRetrospectiveTest": (
                    "control_plus_frozen_highres"
                ),
                "candidateSelection": {
                    "selected": "control_plus_frozen_highres",
                    "highresPromoted": True,
                },
                "upstreamBinaryReport": {
                    "path": str(upstream_path),
                    "sha256": sha256_file(upstream_path),
                    "kind": upstream_payload["kind"],
                    "featureCount": len(frozen_names),
                    "featureSignatureSha256": _signature_sha256(frozen_names),
                    "selectionFrozenBeforeHighresStudy": True,
                },
                "highresFeatureDefinitions": study.definitions,
                "highresCaches": frozen_inventory,
                "finalizationPlan": {
                    "candidate": "control_plus_frozen_highres",
                    "featureCount": len(selected_names),
                    "featureNames": list(selected_names),
                    "featureSignatureSha256": _signature_sha256(selected_names),
                    "epochCap": 1,
                    "seed": 17,
                    "decoder": decoder.to_dict(),
                },
                "provenance": _highres_code_provenance(),
            }
            development_path = root / "highres-development.json"
            development_path.write_text(
                json.dumps(development_payload), encoding="utf-8"
            )

            def fake_prepare(recordings, _config, _cache_dir, *, progress=None):
                del progress
                return [prepared_by_id[item.id] for item in recordings]

            with (
                patch(
                    "analysis.highres_embedding_experiment.load_manifest",
                    return_value=manifest,
                ),
                patch(
                    "analysis.highres_embedding_experiment._prepare_many",
                    side_effect=fake_prepare,
                ),
            ):
                report = run_retrospective_highres_test(
                    manifest.path,
                    development_path,
                    root / "warm",
                    highres_cache_dir,
                    spec=extraction_spec,
                    test_cache_index_path=test_extraction_path,
                )

            self.assertTrue(report["testLabelsOpened"])
            self.assertTrue(report["highresPromotionGatePassed"])
            self.assertEqual(
                report["selectedCandidate"], "control_plus_frozen_highres"
            )
            self.assertIn("multiIouMetrics", report["test"])
            self.assertIn("outcomeSlices", report["test"])
            self.assertEqual(set(report["verifiedCaches"]["test"]), {"test-e"})
            self.assertFalse(
                report["verifiedCaches"]["cacheExtractionPerformedByEvaluator"]
            )


if __name__ == "__main__":
    unittest.main()
