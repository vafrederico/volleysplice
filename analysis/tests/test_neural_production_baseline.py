from types import SimpleNamespace
import unittest

import numpy as np

from analysis.neural_production_baseline import fitting_scope, merge_overlap_union, production_targets
from analysis.schema import Interval


class ProductionBaselineContractTests(unittest.TestCase):
    def test_held_group_removed_even_if_allowlisted(self):
        items = [SimpleNamespace(recording=SimpleNamespace(id=i, source_group=g, consent={"train": True}))
                 for i, g in (("a", "held"), ("b", "held"), ("c", "fit"))]
        self.assertEqual([row.recording.id for row in fitting_scope(items, "held", {"a", "b", "c"})], ["c"])

    def test_dead_target_is_local_transition_with_setup_negative_and_ignore_mask(self):
        times = np.arange(0, 12, 0.25)
        valid = np.ones(len(times), dtype=bool)
        valid[times == 6] = False
        item = SimpleNamespace(sequence=SimpleNamespace(times=times), sample_mask=valid,
                               recording=SimpleNamespace(rallies=(Interval(3, 6),)))
        labels, mask = production_targets(item, "deadState", {"deadStateTarget": {
            "mode": "end-transition", "beforeEndSeconds": 2, "afterEndSeconds": 2, "preServeSetupSeconds": 1}})
        self.assertFalse(mask[times == 0].item())
        self.assertTrue(mask[times == 2.5].item())
        self.assertEqual(labels[times == 2.5].item(), 0)
        self.assertEqual(labels[times == 6.5].item(), 1)
        self.assertFalse(mask[times == 6].item())

    def test_product_union_preserves_touching_events_but_merges_overlap(self):
        result = merge_overlap_union([[Interval(1, 3), Interval(5, 7)], [Interval(2, 4), Interval(4, 5)]])
        self.assertEqual([(r.start, r.end) for r in result], [(1, 4), (4, 5), (5, 7)])


if __name__ == "__main__":
    unittest.main()
