import unittest

from analysis.neural_recall_operating_point import outer_status, strict_selection


def candidate(epoch, recall, f1):
    return {"epoch": epoch, "decoder": {"enter": .2}, "innerR_core": recall, "innerF1_padP_coreR": f1}


class StrictRecallSelectionTests(unittest.TestCase):
    def test_infeasible_does_not_relax_to_best_f1(self):
        result = strict_selection([candidate(5, .98, .97), candidate(15, .9899, .85)])
        self.assertFalse(result["feasible"])
        self.assertIsNone(result["selected"])
        self.assertEqual(result["eligibleCandidateCount"], 0)
        self.assertEqual(result["maximumInnerRecall"], .9899)

    def test_exact_floor_is_eligible_and_ranks_by_f1(self):
        rows = [candidate(5, .989, .99), candidate(15, .99, .90), candidate(30, 1., .87)]
        result = strict_selection(rows)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["eligibleCandidateCount"], 2)
        self.assertEqual(result["selected"], rows[1])

    def test_stable_tie_uses_registered_order(self):
        rows = [candidate(5, .99, .90), candidate(15, 1., .90)]
        self.assertEqual(strict_selection(rows)["selected"], rows[0])

    def test_outer_checkpoint_availability_is_separate_from_feasibility(self):
        result = strict_selection([candidate(30, .99, .90)])
        self.assertEqual(outer_status(result, [15]), "missing-selected-outer-checkpoint")
        self.assertEqual(outer_status(result, [30]), "available")
        self.assertEqual(outer_status(strict_selection([candidate(30, .98, .90)]), [30]), "infeasible-inner-recall")

    def test_invalid_scores_and_floor_fail(self):
        for row in (candidate(5, float("nan"), .9), candidate(5, .99, 1.01)):
            with self.assertRaises(ValueError):
                strict_selection([row])
        with self.assertRaises(ValueError):
            strict_selection([])
        with self.assertRaises(ValueError):
            strict_selection([candidate(5, .99, .9)], 1.1)


if __name__ == "__main__":
    unittest.main()
