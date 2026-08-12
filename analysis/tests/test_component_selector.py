from __future__ import annotations

import copy
import unittest
from dataclasses import FrozenInstanceError, fields, replace

from analysis.component_selector import (
    FROZEN_INTERSECTION_CONFIG,
    INTERSECTION,
    KEEP_V4,
    KEEP_V5,
    LOCAL_REFINED,
    UNION,
    BoundaryProvenance,
    CandidateFeatureRow,
    ComponentSelectorModel,
    GeneratorProvenance,
    IntervalCandidate,
    LeakageError,
    RawScoreSummary,
    TransitionQualitySummary,
    apply_frozen_intersection,
    assert_no_double_assignment,
    build_component_rows,
    candidate_action_rows,
    deterministic_choice,
    fit_component_selector,
    validate_oof_rows,
)
from analysis.decoder import DecodedInterval
from analysis.dual_serve_fusion_experiment import apply_boundary_selector


def provenance(
    generator_id: str,
    artifact: str,
    training_groups: tuple[str, ...] = (),
) -> GeneratorProvenance:
    return GeneratorProvenance(generator_id, artifact, training_groups)


def candidate(
    candidate_id: str,
    generator_id: str,
    start: float,
    end: float,
    confidence: float = 0.9,
    *,
    operation: str = "decoded",
) -> IntervalCandidate:
    return IntervalCandidate(
        candidate_id=candidate_id,
        generator_id=generator_id,
        start=start,
        end=end,
        confidence=confidence,
        boundary_provenance=BoundaryProvenance(
            generator_id,
            generator_id,
            operation,
            (candidate_id,),
        ),
    )


class ComponentPartitionTests(unittest.TestCase):
    def test_components_are_time_ordered_and_overlap_is_transitive(self) -> None:
        generators = (
            provenance("v4", "v4-fold"),
            provenance("v5", "v5-fold"),
        )
        components = build_component_rows(
            [candidate("late", "v4", 10.0, 11.0), candidate("bridge", "v4", 0.0, 3.0)],
            [
                candidate("third", "v5", 2.0, 3.0),
                candidate("uncovered", "v5", 20.0, 21.0),
                candidate("first", "v5", 0.0, 1.0),
                candidate("second", "v5", 1.0, 2.0),
            ],
            recording_id="recording",
            source_group="held-out",
            generators=generators,
        )

        self.assertEqual(
            [row.component_envelope for row in components],
            [(0.0, 3.0), (10.0, 11.0), (20.0, 21.0)],
        )
        self.assertEqual(
            [row.cardinality for row in components], [(1, 3, 0), (1, 0, 0), (0, 1, 0)]
        )
        self.assertEqual(components[0].topology, "one-to-many")
        assert_no_double_assignment(components, expected_candidate_count=6)
        assigned = [
            item.candidate_id
            for component_row in components
            for item in component_row.v4
            + component_row.v5
            + component_row.local_refined
        ]
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_touching_components_and_duplicate_input_rejection(self) -> None:
        generators = (provenance("v4", "v4-fold"),)
        first = candidate("first", "v4", 0.0, 1.0)
        second = candidate("second", "v4", 1.0, 2.0)
        components = build_component_rows(
            [second, first],
            [],
            recording_id="recording",
            source_group="held-out",
            generators=generators,
        )
        self.assertEqual(len(components), 2)
        with self.assertRaisesRegex(ValueError, "more than once"):
            build_component_rows(
                [first],
                [],
                local_refined=[first],
                recording_id="recording",
                source_group="held-out",
                generators=generators,
            )
        with self.assertRaisesRegex(ValueError, "duplicate component id"):
            assert_no_double_assignment((components[0], components[0]))


class ProvenanceAndActionTests(unittest.TestCase):
    def test_held_out_source_group_leakage_is_rejected_before_fit(self) -> None:
        leaking = provenance("v4", "leaked-fold", ("held-out", "other"))
        component = build_component_rows(
            [candidate("v4-0", "v4", 0.0, 3.0)],
            [],
            recording_id="recording",
            source_group="held-out",
            generators=(leaking,),
        )[0]
        rows = candidate_action_rows(component)

        with self.assertRaisesRegex(LeakageError, "trained on held-out"):
            validate_oof_rows(rows)
        with self.assertRaisesRegex(LeakageError, "trained on held-out"):
            fit_component_selector(rows, {component.component_id: KEEP_V4})

    def test_rows_record_fold_groups_hashes_and_contain_no_target_field(self) -> None:
        generators = (
            provenance("v4", "hash-v4", ("train-a",)),
            provenance("v5", "hash-v5", ("train-b",)),
        )
        component = build_component_rows(
            [candidate("v4-0", "v4", 2.0, 8.0)],
            [candidate("v5-0", "v5", 2.5, 6.0)],
            recording_id="recording",
            source_group="held-out",
            generators=generators,
        )[0]
        component = replace(
            component,
            raw_scores=(
                RawScoreSummary.from_values("v4", "rally", (0.2, 0.8, 0.5)),
                RawScoreSummary.from_values("v5", "serve", (0.1, 0.9)),
            ),
            transition_quality=TransitionQualitySummary(
                start_before=0.1,
                start_after=0.7,
                end_before=0.8,
                end_after=0.2,
                end_persistence=0.6,
                visibility=0.9,
                audio_availability=1.0,
                camera_quality=0.8,
            ),
        )
        row = candidate_action_rows(component)[0]

        self.assertEqual(
            row.generator_training_groups,
            (("v4", ("train-a",)), ("v5", ("train-b",))),
        )
        self.assertEqual(
            row.generator_hashes, (("v4", "hash-v4"), ("v5", "hash-v5"))
        )
        self.assertNotIn("target", {item.name for item in fields(CandidateFeatureRow)})
        self.assertNotIn("gold", {item.name for item in fields(CandidateFeatureRow)})
        with self.assertRaises(FrozenInstanceError):
            row.action = KEEP_V5  # type: ignore[misc]

    def test_uncovered_additions_and_union_are_disabled_by_default(self) -> None:
        generators = (
            provenance("v4", "v4-fold"),
            provenance("v5", "v5-fold"),
        )
        uncovered = build_component_rows(
            [],
            [candidate("v5-only", "v5", 4.0, 5.0)],
            recording_id="recording",
            source_group="held-out",
            generators=generators,
        )[0]
        default_rows = candidate_action_rows(uncovered)
        enabled_rows = candidate_action_rows(
            uncovered, allow_uncovered_additions=True
        )
        self.assertEqual([row.action for row in default_rows], [KEEP_V4])
        self.assertEqual(default_rows[0].intervals, ())
        self.assertEqual([row.action for row in enabled_rows], [KEEP_V4, KEEP_V5])
        self.assertTrue(enabled_rows[1].uncovered_addition)

        paired = build_component_rows(
            [candidate("v4", "v4", 0.0, 4.0)],
            [candidate("v5", "v5", 1.0, 3.0)],
            recording_id="paired",
            source_group="held-out",
            generators=generators,
        )[0]
        self.assertNotIn(UNION, [row.action for row in candidate_action_rows(paired)])
        self.assertIn(
            UNION,
            [row.action for row in candidate_action_rows(paired, permit_union=True)],
        )

    def test_action_order_and_boundary_provenance_are_deterministic(self) -> None:
        generators = (
            provenance("local", "local-fold"),
            provenance("v4", "v4-fold"),
            provenance("v5", "v5-fold"),
        )
        component = build_component_rows(
            [candidate("v4", "v4", 0.0, 5.0, 0.8)],
            [candidate("v5", "v5", 1.0, 4.0, 0.9)],
            local_refined=[
                candidate("local", "local", 0.5, 3.5, 0.85, operation="refined")
            ],
            recording_id="recording",
            source_group="held-out",
            generators=generators,
        )[0]
        rows = candidate_action_rows(component, permit_union=True)

        self.assertEqual(
            [row.action for row in rows],
            [KEEP_V4, KEEP_V5, INTERSECTION, UNION, LOCAL_REFINED],
        )
        intersection = next(row for row in rows if row.action == INTERSECTION)
        self.assertEqual(
            (
                intersection.intervals[0].start,
                intersection.intervals[0].end,
                intersection.boundary_provenance[0].start_generator,
                intersection.boundary_provenance[0].end_generator,
            ),
            (1.0, 4.0, "v5", "v5"),
        )


class FrozenParityTests(unittest.TestCase):
    def test_frozen_intersection_helper_matches_existing_selector(self) -> None:
        v4 = [
            DecodedInterval(0.0, 5.0, 0.90),
            DecodedInterval(8.0, 12.0, 0.95),
            DecodedInterval(20.0, 23.0, 0.99),
        ]
        v5 = [
            DecodedInterval(0.5, 3.5, 0.85),  # eligible
            DecodedInterval(8.0, 10.0, 0.70),  # confidence gate fails
            DecodedInterval(20.0, 21.0, 0.99),  # ambiguous split
            DecodedInterval(21.0, 22.0, 0.99),
            DecodedInterval(30.0, 31.0, 0.99),  # uncovered addition stays off
        ]

        expected = apply_boundary_selector(v4, v5, FROZEN_INTERSECTION_CONFIG)
        actual = apply_frozen_intersection(v4, v5, FROZEN_INTERSECTION_CONFIG)
        self.assertEqual(actual, expected)


def training_rows() -> tuple[list[CandidateFeatureRow], dict[str, str]]:
    groups = ("group-a", "group-b", "group-c", "group-d")
    rows: list[CandidateFeatureRow] = []
    targets: dict[str, str] = {}
    for index, held_out in enumerate(groups):
        training_groups = tuple(group for group in groups if group != held_out)
        generators = (
            provenance("v4", f"v4-fold-{held_out}", training_groups),
            provenance("v5", f"v5-fold-{held_out}", training_groups),
        )
        component = build_component_rows(
            [candidate(f"v4-{index}", "v4", index * 10.0, index * 10.0 + 6.0)],
            [
                candidate(
                    f"v5-{index}",
                    "v5",
                    index * 10.0 + 0.5,
                    index * 10.0 + (3.0 if index % 2 else 5.0),
                )
            ],
            recording_id=f"recording-{index}",
            source_group=held_out,
            generators=generators,
        )[0]
        component = replace(
            component,
            raw_scores=(
                RawScoreSummary.from_values(
                    "v4", "rally", (0.2 + index * 0.1, 0.8)
                ),
                RawScoreSummary.from_values(
                    "v5", "serve", (0.1, 0.6 + index * 0.08)
                ),
            ),
            transition_quality=TransitionQualitySummary(
                end_before=0.8,
                end_after=0.1 + index * 0.1,
                end_persistence=0.9 - index * 0.1,
                visibility=0.8,
                audio_availability=1.0,
                camera_quality=0.9,
            ),
        )
        component_rows = list(candidate_action_rows(component))
        rows.extend(component_rows)
        targets[component.component_id] = INTERSECTION if index < 2 else KEEP_V4
    return rows, targets


class SelectorFitTests(unittest.TestCase):
    def test_fit_and_serialization_are_deterministic_under_row_reordering(self) -> None:
        rows, targets = training_rows()

        first = fit_component_selector(rows, targets, max_iterations=400)
        second = fit_component_selector(
            list(reversed(rows)), targets, max_iterations=400
        )

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.artifact_sha256, second.artifact_sha256)
        self.assertEqual(first.spec_hash, second.spec_hash)
        self.assertEqual(first.training_component_count, 4)
        self.assertEqual(first.training_row_count, 12)
        self.assertFalse(first.allow_uncovered_additions)
        self.assertFalse(first.permit_union)
        restored = ComponentSelectorModel.from_dict(first.to_dict())
        self.assertEqual(restored.to_dict(), first.to_dict())
        self.assertEqual(
            [row.action for row in restored.select(rows)],
            [row.action for row in first.select(rows)],
        )

    def test_exact_score_ties_use_conservative_action_order(self) -> None:
        rows, _ = training_rows()
        component_rows = [
            row for row in rows if row.component_id == rows[0].component_id
        ]

        selected = deterministic_choice(
            list(reversed(component_rows)), [0.0] * len(component_rows)
        )
        self.assertEqual(selected.action, KEEP_V4)

    def test_serialization_rejects_changed_feature_spec_hash(self) -> None:
        rows, targets = training_rows()
        model = fit_component_selector(rows, targets, max_iterations=20)
        payload = copy.deepcopy(model.to_dict())
        payload["featureSpecSha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "specification hash"):
            ComponentSelectorModel.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
