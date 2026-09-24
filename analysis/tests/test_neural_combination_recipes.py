"""Synthetic boundary cases for the independent construction audit."""
import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[2]/'scripts/audit-neural-combination-recipes.py'
spec=importlib.util.spec_from_file_location('independent_recipe_audit',PATH)
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class RecipeAuditTests(unittest.TestCase):
    def test_sweep_set_operations_handle_overlap_touch_and_duplicates(self):
        left=[(1,4),(3,5),(3,5)]
        right=[(5,8)]
        self.assertEqual(a.united(left,right),[(1,8)])
        self.assertEqual(a.overlap(left,right),[])
        self.assertEqual(a.minus(left,[(2,4)]),[(1,2),(4,5)])

    def test_graph_protects_full_transitive_source_component(self):
        self.assertEqual(a.agreement([(1,2),(5,6)],[(10,11)],20),[])
        self.assertEqual(a.agreement([(1,2)],[(6.5,7)],20),[[(1,2)],[(6.5,7)]])

    def test_guard_preserves_protected_and_already_removed_time(self):
        p=[(1,2),(7,9)]
        eligible=a.agreement([(1,2)],[(7,9)],20)
        self.assertEqual(a.construction('guarded_trim',p,[[]],eligible,20),[])
        eligible=a.agreement([(1,4)],[(8,9)],20)
        self.assertEqual(a.construction('guarded_trim',[(1,2)],[[]],eligible,20),[(1,2)])

    def test_component_rejection_and_partial_trim_are_distinct(self):
        p=[(2,12)]
        eligible=a.agreement(p,[],20)
        self.assertEqual(a.construction('guarded_trim',p,[[(5,6)]],eligible,20),[(3,8)])
        self.assertEqual(a.construction('guarded_component_rejection',p,[[(5,6)]],eligible,20),p)

    def test_majority_is_timewise_not_component_vote(self):
        self.assertEqual(a.construction('majority',[(1,5)],[[(4,8)],[(7,10)]],[],20),[(4,5),(7,8)])

    def test_exports_strict_three_gap_then_ignored_no_rejoin(self):
        rec={'durationSeconds':20,'ignoredIntervals':[(3,4)]}
        self.assertEqual(a.export([(1,5),(8,10)],rec,0),[(1,3),(4,5),(8,10)])
        self.assertEqual(a.expand([(1,2),(7,8)],2,20),[(0,4),(5,10)])

    def test_standalone_identity_rejects_event_collapsing(self):
        with self.assertRaisesRegex(ValueError,'count'):
            a.assert_ranges([(1,4)],[(1,2),(2,4)],'standalone')


if __name__=='__main__':
    unittest.main()
