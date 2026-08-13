from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.config import FEATURE_VERSION, DecoderConfig, TrainingConfig
from analysis.court_relative_experiment import court_relative_feature_spec
from analysis.court_relative_feature_study import (
    CONTROL_CANDIDATE,
    PRIMARY_CANDIDATE_ORDER,
    VARIANT_CANDIDATES,
    load_frozen_step1_report,
    prepare_court_study_candidates,
    run_development_court_experiments,
    run_retrospective_court_test,
)
from analysis.feature_experiments import (
    FeatureExperimentError,
    _subset_prepared,
    build_fold_plan,
    sha256_file,
)
from analysis.pipeline import _manifest_digest
from analysis.schema import DatasetManifest
from analysis.tests.test_transition_feature_experiment import (
    _feature_config,
    _prepared,
    _recording,
)
from analysis.transition_feature_experiment import (
    _signature_sha256,
    prepare_transition_candidates,
)


class CourtRelativeFeatureStudyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="volleycut-court-study-test-"
        )
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.config = _feature_config()
        rows = [
            _recording(self.root, "dev-a", "a", "train"),
            _recording(self.root, "dev-b", "b", "train"),
            _recording(self.root, "dev-c", "c", "train"),
            _recording(self.root, "dev-d", "d", "validation"),
            _recording(self.root, "test-e", "e", "test"),
        ]
        manifest_path = self.root / "manifest.json"
        payload = {
            "schemaVersion": 1,
            "name": "synthetic-court-study",
            "annotationPolicy": {"id": "serve-contact-to-dead-ball-v1"},
            "recordings": [item.raw for item in rows],
        }
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        self.manifest = DatasetManifest(
            path=manifest_path,
            name="synthetic-court-study",
            recordings=tuple(rows),
            raw=payload,
        )
        self.development_prepared = [
            _prepared(item, self.config)
            for item in rows
            if item.split in {"train", "validation"}
        ]

    def _write_upstream(
        self,
        selected_name: str,
        *,
        inner_fold_limit: int | None = None,
    ) -> tuple[Path, dict[str, object]]:
        transition = prepare_transition_candidates(
            self.development_prepared, self.config
        )
        selected = next(
            item for item in transition.candidates if item.name == selected_name
        )
        subset = _subset_prepared(transition.prepared, selected.indexes)
        names = subset[0].contextual_names
        training = TrainingConfig(
            epochs=1,
            batch_size=64,
            learning_rate=0.01,
            patience=1,
            seed=5,
        )
        payload: dict[str, object] = {
            "schemaVersion": 1,
            "kind": "volleycut-transition-feature-experiment-development",
            "freezeStatus": "frozen-development-selection",
            "testLabelsUsed": False,
            "manifestFileSha256": sha256_file(self.manifest.path),
            "manifestSnapshotSha256": _manifest_digest(self.manifest),
            "recordingContentSha256": {
                item.id: item.content_sha256 for item in self.manifest.recordings
            },
            "featureVersion": FEATURE_VERSION,
            "featureConfig": self.config.to_dict(),
            "trainingConfig": training.to_dict(),
            "baseDecoderConfig": DecoderConfig(
                smoothing_seconds=0.0,
                enter_threshold=0.5,
                exit_threshold=0.4,
                min_live_seconds=1.0,
                bridge_gap_seconds=0.0,
                short_event_min_seconds=0.5,
                short_event_threshold=0.8,
            ).to_dict(),
            "selectionProtocol": {
                "innerFoldLimit": inner_fold_limit,
                "folds": [
                    item.to_dict() for item in build_fold_plan(self.manifest.recordings)
                ],
            },
            "selectedCandidateForRetrospectiveTest": selected_name,
            "finalizationPlan": {
                "candidate": selected_name,
                "featureNames": list(names),
                "featureSignatureSha256": _signature_sha256(names),
            },
        }
        path = self.root / f"step1-{selected_name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path, payload

    def test_reconstructs_every_primary_step1_winner_and_appends_variants_once(self) -> None:
        expected_added = {
            variant: len(court_relative_feature_spec(variant).feature_names)
            for variant in VARIANT_CANDIDATES
        }
        for selected_name in (
            "baseline_450",
            "baseline_plus_multiscale",
            "baseline_plus_interactions",
            "combined",
        ):
            with self.subTest(selected_name=selected_name):
                _, upstream = self._write_upstream(selected_name)
                study = prepare_court_study_candidates(
                    (self.development_prepared[0],), self.config, upstream
                )
                candidates = {item.name: item for item in study.candidates}

                self.assertEqual(tuple(candidates), PRIMARY_CANDIDATE_ORDER)
                self.assertEqual(study.upstream_candidate, selected_name)
                self.assertEqual(
                    len(candidates[CONTROL_CANDIDATE].indexes),
                    len(study.control_names),
                )
                for variant, candidate_name in VARIANT_CANDIDATES.items():
                    self.assertEqual(
                        len(candidates[candidate_name].indexes),
                        len(study.control_names) + expected_added[variant],
                    )
                court_columns = study.prepared[0].contextual_names[
                    len(study.control_names) :
                ]
                self.assertEqual(court_columns, study.court_names)
                self.assertTrue(
                    all(
                        not name.startswith("t+") and not name.startswith("t-")
                        for name in court_columns
                    )
                )
                self.assertLessEqual(
                    float(
                        study.prepared[0].contextual_values[
                            :, len(study.control_names) :
                        ].max()
                    ),
                    1.0,
                )
                self.assertEqual(
                    study.definitions["excludedInputs"],
                    [
                        "rally labels",
                        "outcome tags",
                        "environment",
                        "game metadata",
                        "playersPerTeam",
                        "sideSwitches",
                        "annotation confidence",
                    ],
                )

    def test_load_rejects_nonfrozen_and_reduced_inner_fold_reports(self) -> None:
        path, payload = self._write_upstream("baseline_450", inner_fold_limit=1)
        with self.assertRaisesRegex(FeatureExperimentError, "full nested"):
            load_frozen_step1_report(path)

        payload["selectionProtocol"] = {"innerFoldLimit": None, "folds": []}
        payload["freezeStatus"] = "draft"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(FeatureExperimentError, "frozen unopened-test"):
            load_frozen_step1_report(path)

    def test_nested_report_has_slices_paired_deltas_hashes_and_frozen_plan(self) -> None:
        path, _ = self._write_upstream("baseline_450")
        decoder = DecoderConfig(
            smoothing_seconds=0.0,
            enter_threshold=0.5,
            exit_threshold=0.4,
            min_live_seconds=1.0,
            bridge_gap_seconds=0.0,
            short_event_min_seconds=0.5,
            short_event_threshold=0.8,
        )
        fixed_decoder = lambda _prepared_rows, _probabilities, _base: (
            decoder,
            {"status": "fixed-for-test"},
        )
        with (
            patch("analysis.feature_experiments._select_decoder", side_effect=fixed_decoder),
            patch(
                "analysis.court_relative_feature_study._select_decoder",
                side_effect=fixed_decoder,
            ),
        ):
            report = run_development_court_experiments(
                self.manifest,
                self.development_prepared,
                step1_report_path=path,
            )

        self.assertFalse(report["testLabelsUsed"])
        self.assertEqual(report["freezeStatus"], "frozen-development-selection")
        self.assertEqual(report["protected"]["test"]["recordingIds"], ["test-e"])
        self.assertEqual(set(report["candidates"]), set(PRIMARY_CANDIDATE_ORDER))
        self.assertEqual(
            set(report["pairedComparisonsAgainstControl"]),
            set(PRIMARY_CANDIDATE_ORDER[1:]),
        )
        self.assertIn(
            "shortAtMost3Seconds",
            report["candidates"][CONTROL_CANDIDATE]["outcomeSlices"],
        )
        self.assertEqual(
            set(report["candidates"][CONTROL_CANDIDATE]["multiIouMetrics"]),
            {
                "eventF1AtIou03",
                "eventF1AtIou05",
                "eventF1AtIou07",
                "matchedRalliesAtIou03",
                "matchedRalliesAtIou05",
                "matchedRalliesAtIou07",
            },
        )
        self.assertEqual(
            set(report["courtSpecSha256"]), set(VARIANT_CANDIDATES)
        )
        self.assertEqual(
            report["upstreamStep1"]["developmentReportSha256"], sha256_file(path)
        )
        self.assertIn(
            report["selectedCandidateForRetrospectiveTest"],
            PRIMARY_CANDIDATE_ORDER,
        )
        self.assertFalse(report["candidateSelection"]["testMetricsConsulted"])
        self.assertGreater(report["finalizationPlan"]["epochCap"], 0)
        json.dumps(report, allow_nan=False)

    def test_development_rejects_a_prepared_test_recording(self) -> None:
        path, _ = self._write_upstream("baseline_450")
        test_row = next(item for item in self.manifest.recordings if item.split == "test")
        with self.assertRaisesRegex(
            FeatureExperimentError, r"exactly train\+validation"
        ):
            run_development_court_experiments(
                self.manifest,
                [*self.development_prepared, _prepared(test_row, self.config)],
                step1_report_path=path,
            )

    def test_retrospective_gate_fails_before_upstream_or_manifest_access(self) -> None:
        invalid = self.root / "not-frozen-court.json"
        invalid.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "kind": "volleycut-court-relative-feature-study-development",
                    "freezeStatus": "draft",
                    "testLabelsUsed": False,
                }
            ),
            encoding="utf-8",
        )
        with (
            patch(
                "analysis.court_relative_feature_study.load_frozen_step1_report",
                side_effect=AssertionError("invalid gate must fail before upstream access"),
            ),
            patch(
                "analysis.court_relative_feature_study.load_manifest",
                side_effect=AssertionError("invalid gate must fail before manifest access"),
            ),
        ):
            with self.assertRaisesRegex(FeatureExperimentError, "frozen unopened-test"):
                run_retrospective_court_test(
                    self.manifest.path,
                    invalid,
                    self.root / "cache",
                )


if __name__ == "__main__":
    unittest.main()
