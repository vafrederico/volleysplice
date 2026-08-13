from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.component_selector import GeneratorProvenance, materialize_selected
from analysis.component_selector_study import (
    CACHE_FILENAME,
    CACHE_KIND,
    EXPERIMENT_ID,
    STUDY_SCHEMA_VERSION,
    ComponentSelectorStudyError,
    LoadedOOFCache,
    assign_truth_to_components,
    build_generator_fold_plan,
    build_recording_study_rows,
    load_component_selector_oof,
    prepare_component_selector_oof,
    run_component_selector_development,
    select_frozen_intersection_actions,
    _code_provenance,
)
from analysis.decoder import DecodedInterval
from analysis.dual_serve_fusion_experiment import (
    FUSION_VARIANT_DESCRIPTION,
    apply_boundary_selector,
)
from analysis.features import FeatureSequence, VideoMetadata
from analysis.pipeline import PreparedRecording, _manifest_digest
from analysis.schema import DatasetManifest, Interval, Recording, labels_for_times


def recording(
    root: Path,
    recording_id: str,
    source_group: str,
    split: str,
    *,
    rallies: tuple[Interval, ...] = (Interval(2.0, 5.0),),
) -> Recording:
    video = root / f"{recording_id}.mp4"
    video.write_bytes(recording_id.encode("utf-8"))
    return Recording(
        id=recording_id,
        video=video,
        split=split,
        source_group=source_group,
        environment="indoor",
        game={"playersPerTeam": 4},
        rallies=rallies,
        ignored_intervals=(),
        roi=(0.05, 0.05, 0.9, 0.9),
        capture={"stationary": True},
        consent={"analyze": True, "train": split in {"train", "validation"}},
        content_sha256=None,
        raw={
            "id": recording_id,
            "video": str(video),
            "split": split,
            "sourceGroup": source_group,
        },
    )


def prepared(item: Recording) -> PreparedRecording:
    times = np.arange(0.0, 32.25, 0.25, dtype=np.float64)
    values = np.column_stack(
        (
            np.full(len(times), 0.9),
            np.full(len(times), 0.8),
            np.ones(len(times)),
        )
    ).astype(np.float32)
    sequence = FeatureSequence(
        times=times,
        values=values,
        names=("visibility_quality", "focus_quality", "audio_available"),
        metadata=VideoMetadata(32.25, 1920, 1080, 30.0, 968, True),
    )
    return PreparedRecording(
        recording=item,
        sequence=sequence,
        contextual_values=values,
        contextual_names=sequence.names,
        labels=labels_for_times(times, item.rallies),
        sample_mask=np.ones(len(times), dtype=np.bool_),
    )


def generators(
    training_groups: tuple[str, ...], *, suffix: str = "fold"
) -> tuple[GeneratorProvenance, ...]:
    return tuple(
        GeneratorProvenance(name, f"{name}-{suffix}", training_groups)
        for name in (
            "v4-composed",
            "v4-rally",
            "v4-serve",
            "v5-composed",
            "v5-rally",
            "v5-serve",
        )
    )


def scores(times: np.ndarray, phase: float = 0.0) -> tuple[np.ndarray, ...]:
    live_v4 = np.clip(0.45 + 0.35 * np.sin(times * 0.3 + phase), 0.01, 0.99)
    serve_v4 = np.clip(0.3 + 0.5 * np.cos(times * 0.5 + phase), 0.01, 0.99)
    live_v5 = np.clip(0.4 + 0.4 * np.sin(times * 0.25 + phase), 0.01, 0.99)
    serve_v5 = np.clip(0.25 + 0.6 * np.cos(times * 0.45 + phase), 0.01, 0.99)
    return tuple(row.astype(np.float32) for row in (live_v4, serve_v4, live_v5, serve_v5))


def study_rows(
    item: Recording,
    training_groups: tuple[str, ...],
    *,
    phase: float = 0.0,
    v4: tuple[DecodedInterval, ...] = (DecodedInterval(1.5, 6.0, 0.9),),
    v5: tuple[DecodedInterval, ...] = (DecodedInterval(2.0, 5.0, 0.9),),
):
    row = prepared(item)
    v4_live, v4_serve, v5_live, v5_serve = scores(row.sequence.times, phase)
    return build_recording_study_rows(
        row,
        v4,
        v5,
        v4_rally_scores=v4_live,
        v4_serve_scores=v4_serve,
        v5_rally_scores=v5_live,
        v5_serve_scores=v5_serve,
        generators=generators(training_groups, suffix=item.source_group),
    )


class ComponentSelectorFoldPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="volleycut-component-selector-study-test-"
        )
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def test_outer_assessment_group_is_absent_from_every_training_row_generator(self) -> None:
        rows = (
            recording(self.root, "train-a", "a", "train"),
            recording(self.root, "train-b", "b", "train"),
            recording(self.root, "train-c", "c", "train"),
            recording(self.root, "validation-d", "d", "validation"),
            recording(self.root, "test-e", "e", "test"),
        )
        plans = build_generator_fold_plan(rows)

        self.assertEqual({row.held_out_group for row in plans}, {"a", "b", "c", "d"})
        for plan in plans:
            self.assertNotIn(plan.held_out_group, plan.consumed_groups)
            self.assertNotIn("e", plan.consumed_groups)
            if plan.role == "selector-training-oof":
                self.assertNotIn("d", plan.consumed_groups)
                self.assertIn("d", plan.excluded_groups)
            else:
                self.assertEqual(plan.held_out_group, "d")
                self.assertEqual(set(plan.consumed_groups), {"a", "b", "c"})

    def test_cache_overwrite_refusal_happens_before_loading_inputs(self) -> None:
        destination = self.root / "already-exists"
        destination.mkdir()
        with patch(
            "analysis.component_selector_study.load_manifest"
        ) as load_manifest:
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                prepare_component_selector_oof(
                    "manifest.json",
                    "v4-rally",
                    "v4-serve",
                    "v4-cache",
                    "v5-rally",
                    "v5-serve",
                    "v5-cache",
                    destination,
                )
        load_manifest.assert_not_called()


class ComponentSelectorRowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="volleycut-component-selector-row-test-"
        )
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.item = recording(
            self.root,
            "recording",
            "held-out",
            "validation",
            rallies=(Interval(0.5, 3.5), Interval(20.0, 22.0)),
        )

    def test_rows_and_actions_are_deterministic_under_interval_reordering(self) -> None:
        v4 = (
            DecodedInterval(10.0, 11.0, 0.8),
            DecodedInterval(0.0, 5.0, 0.9),
        )
        v5 = (
            DecodedInterval(20.0, 21.0, 0.9),
            DecodedInterval(1.0, 4.0, 0.85),
        )
        first = study_rows(self.item, ("train-a", "train-b"), v4=v4, v5=v5)
        second = study_rows(
            self.item,
            ("train-b", "train-a"),
            v4=tuple(reversed(v4)),
            v5=tuple(reversed(v5)),
        )

        self.assertEqual(
            [row.to_dict() for row in first],
            [row.to_dict() for row in second],
        )
        self.assertEqual(
            [[action.action for action in row.actions] for row in first],
            [
                ["keep-v4", "keep-v5", "intersection"],
                ["keep-v4"],
                ["keep-v4"],
            ],
        )
        self.assertTrue(
            all(
                action.action != "union" and not action.uncovered_addition
                for row in first
                for action in row.actions
            )
        )

    def test_frozen_baseline_has_exact_existing_selector_parity(self) -> None:
        v4 = (
            DecodedInterval(0.0, 5.0, 0.90),
            DecodedInterval(8.0, 12.0, 0.95),
            DecodedInterval(20.0, 23.0, 0.99),
        )
        v5 = (
            DecodedInterval(0.5, 3.5, 0.85),
            DecodedInterval(8.0, 10.0, 0.70),
            DecodedInterval(20.0, 21.0, 0.99),
            DecodedInterval(21.0, 22.0, 0.99),
            DecodedInterval(30.0, 31.0, 0.99),
        )
        rows = study_rows(self.item, ("train-a", "train-b"), v4=v4, v5=v5)

        actual = materialize_selected(select_frozen_intersection_actions(rows))
        expected, _ = apply_boundary_selector(
            v4,
            v5,
            __import__(
                "analysis.component_selector", fromlist=["FROZEN_INTERSECTION_CONFIG"]
            ).FROZEN_INTERSECTION_CONFIG,
        )
        self.assertEqual(list(actual), expected)
        self.assertIn("frozen", FUSION_VARIANT_DESCRIPTION.lower())

    def test_truth_is_never_assigned_to_two_candidate_components(self) -> None:
        v4 = (
            DecodedInterval(0.0, 2.0, 0.9),
            DecodedInterval(3.0, 5.0, 0.9),
        )
        rows = study_rows(self.item, ("train-a", "train-b"), v4=v4, v5=())
        truth = (Interval(1.0, 4.0),)

        assigned = assign_truth_to_components(
            truth, [row.component for row in rows]
        )
        self.assertEqual(sum(len(value) for value in assigned.values()), 1)


class ComponentSelectorCacheAndDevelopmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="volleycut-component-selector-cache-test-"
        )
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.rows = (
            recording(self.root, "train-a", "a", "train"),
            recording(self.root, "train-b", "b", "train"),
            recording(self.root, "train-c", "c", "train"),
            recording(self.root, "validation-d", "d", "validation"),
            recording(self.root, "test-e", "e", "test"),
        )
        manifest_path = self.root / "manifest.json"
        raw = {
            "schemaVersion": 1,
            "name": "synthetic-selector-study",
            "annotationPolicy": {"id": "serve-contact-to-dead-ball-v1"},
            "recordings": [row.raw for row in self.rows],
        }
        manifest_path.write_text(json.dumps(raw), encoding="utf-8")
        self.manifest = DatasetManifest(
            path=manifest_path,
            name="synthetic-selector-study",
            recordings=self.rows,
            raw=raw,
        )

    def _cache_payload(self, *, leak_validation: bool = False):
        recording_payloads = []
        for index, item in enumerate(self.rows):
            if item.split not in {"train", "validation"}:
                continue
            if item.split == "train":
                groups = tuple(
                    group for group in ("a", "b", "c") if group != item.source_group
                )
                if leak_validation:
                    groups = (*groups, "d")
            else:
                groups = ("a", "b", "c")
            components = study_rows(item, groups, phase=index * 0.2)
            recording_payloads.append(
                {
                    "recordingId": item.id,
                    "sourceGroup": item.source_group,
                    "split": item.split,
                    "contentSha256": item.content_sha256,
                    "generatorFoldIndex": index,
                    "components": [row.to_dict() for row in components],
                }
            )
        return {
            "schemaVersion": STUDY_SCHEMA_VERSION,
            "kind": CACHE_KIND,
            "experiment": EXPERIMENT_ID,
            "manifestFileSha256": __import__(
                "analysis.feature_experiments", fromlist=["sha256_file"]
            ).sha256_file(self.manifest.path),
            "manifestSnapshotSha256": _manifest_digest(self.manifest),
            "recordingContentSha256": {
                row.id: row.content_sha256 for row in self.rows
            },
            "testLabelsUsed": False,
            "testRecordingsPrepared": False,
            "folds": [],
            "recordings": recording_payloads,
            "omittedCandidateFamilies": [],
            "provenance": _code_provenance(),
        }

    def _write_cache(self, payload) -> Path:
        cache = self.root / "cache"
        cache.mkdir()
        (cache / CACHE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
        return cache

    def test_loader_rejects_training_generators_that_consumed_validation(self) -> None:
        cache = self._write_cache(self._cache_payload(leak_validation=True))

        with self.assertRaisesRegex(
            ComponentSelectorStudyError, "consumed validation"
        ):
            load_component_selector_oof(
                cache, self.manifest, verify_models=False
            )

    def test_development_report_fits_train_oof_and_assesses_validation_only(self) -> None:
        payload = self._cache_payload()
        cache = self._write_cache(payload)
        loaded = load_component_selector_oof(
            cache, self.manifest, verify_models=False
        )

        with (
            patch(
                "analysis.component_selector_study.load_manifest",
                return_value=self.manifest,
            ),
            patch(
                "analysis.component_selector_study.load_component_selector_oof",
                return_value=loaded,
            ),
        ):
            report = run_component_selector_development(
                self.manifest.path,
                cache,
                verify_models=False,
            )

        self.assertFalse(report["testLabelsUsed"])
        self.assertFalse(report["testRecordingsPrepared"])
        self.assertTrue(report["selector"]["model"]["training"]["converged"])
        self.assertEqual(
            report["assessment"]["sourceGroups"], ["d"]
        )
        self.assertEqual(
            set(report["assessment"]["variants"]),
            {"v4", "frozenIntersection", "trainedSelector"},
        )
        self.assertNotIn(
            "d", report["selector"]["model"]["training"]["sourceGroups"]
        )
        for generator in report["selector"]["model"]["training"][
            "generatorProvenance"
        ]:
            self.assertNotIn("d", generator["trainingSourceGroups"])
        self.assertFalse(report["retrospectiveTestPlan"]["implemented"])
        self.assertIn(
            "objective",
            report["assessment"]["variants"]["trainedSelector"]["aggregate"],
        )
        self.assertFalse(report["promotionDecision"]["promotable"])
        self.assertEqual(
            report["promotionDecision"]["selectedForProduction"],
            "frozenIntersection",
        )
        self.assertTrue(
            report["retrospectiveTestPlan"]["testRemainsUnopenedByThisStudy"]
        )


if __name__ == "__main__":
    unittest.main()
