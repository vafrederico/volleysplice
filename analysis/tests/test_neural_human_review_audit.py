import copy
import importlib.util
from pathlib import Path
import unittest


_PATH = Path(__file__).resolve().parents[2] / 'scripts/audit-neural-human-review.py'
_SPEC = importlib.util.spec_from_file_location('human_review_audit', _PATH)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def candidate(index, start, end, flag, kind='positive', score=None):
    return dict(id=f'c{index:05d}', start=start, end=end, flagged=flag, kind=kind, score=score)


def fixture():
    """Hand-computed whole-candidate case with a mixed kept candidate."""
    rec = dict(id='example', sourceGroup='group', durationSeconds=20.,
               ignoredIntervals=[], rallies=[dict(start=3., end=5.), dict(start=16., end=18.)])
    base = [[2., 4.], [10., 12.]]
    inventory = [candidate(0, 2, 4, True), candidate(1, 10, 12, True),
                 candidate(2, 16, 18, True, 'negative')]
    binary = {'0': [[2, 4], [16, 18]], '1': [[1, 5], [15, 19]],
              '2': [[0, 6], [14, 20]], '3': [[0, 7], [13, 20]]}
    boundary = {'0': [[3, 4], [16, 18]], '1': [[2, 5], [15, 19]],
                '2': [[1, 7], [14, 20]], '3': [[0, 8], [13, 20]]}
    decision = {'0': [[2, 4], [10, 12], [16, 18]], '1': [[1, 5], [9, 19]],
                '2': [[0, 20]], '3': [[0, 20]]}
    rows = []
    for pad in range(4):
        rows.append(dict(paddingSecondsBeforeAndAfter=pad,
                         decisionExportIntervals=decision[str(pad)], playbackIntervals=[[0, 20]],
                         reviewSeconds=20., decisionSeconds=[6., 14., 20., 20.][pad], reviewClips=1,
                         flaggedCandidates=3, flaggedPositiveCandidates=2, flaggedNegativeCandidates=1,
                         reviewedTrueRallies=2, playbackTrueRallies=2, binaryKeptCandidates=2,
                         binaryDroppedCandidates=1, mixedCandidates=1,
                         keptCandidateIds=['c00000', 'c00002'], droppedCandidateIds=['c00001'],
                         reviewedTrueRallyIds=[0, 1], playbackTrueRallyIds=[0, 1],
                         completeRallyLossesBinary=0, partialRallyLossesBinary=int(pad == 0),
                         completeRallyLossesBoundary=0, partialRallyLossesBoundary=int(pad == 0)))
    result = dict(candidates=inventory, selected=['c00000', 'c00001', 'c00002'],
                  binaryRaw=[[2, 4], [16, 18]], binaryExports=binary,
                  boundaryExports=boundary, padding=rows)
    return rec, base, result


class ReviewAuditTests(unittest.TestCase):
    def test_independent_binary_and_boundary_reconstruction(self):
        rec, base, result = fixture()
        self.assertTrue(audit.audit_record(rec, base, result)['passed'])

    def test_binary_cannot_silently_trim_mixed_candidate(self):
        rec, base, result = fixture()
        result['binaryRaw'][0][0] = 3
        with self.assertRaisesRegex(ValueError, 'binaryRaw'):
            audit.audit_record(rec, base, result)

    def test_playback_context_is_not_edit_permission(self):
        rec, base, result = fixture()
        result['boundaryExports']['0'][0][1] = 5
        with self.assertRaisesRegex(ValueError, 'boundaryExports'):
            audit.audit_record(rec, base, result)

    def test_decisions_not_merged_playback_clips(self):
        rec, base, result = fixture()
        result['padding'][2]['flaggedCandidates'] = 1
        with self.assertRaisesRegex(ValueError, 'flaggedCandidates'):
            audit.audit_record(rec, base, result)

    def test_gold_ordinals_not_candidate_ids(self):
        rec, base, result = fixture()
        result['padding'][0]['reviewedTrueRallyIds'] = ['c00000', 'c00002']
        with self.assertRaisesRegex(ValueError, 'reviewedTrueRallyIds'):
            audit.audit_record(rec, base, result)

    def test_ignored_split_cannot_be_candidate(self):
        rec, base, result = fixture()
        rec['ignoredIntervals'] = [[3, 3.5]]
        with self.assertRaisesRegex(ValueError, 'crosses ignored'):
            audit.audit_record(rec, base, result)

    def test_strict_three_second_join_and_ignored_holes(self):
        self.assertEqual(audit._export([(0, 1), (4, 5), (7.99, 8)], [], 12, 0),
                         [(0, 1), (4, 8)])
        self.assertEqual(audit._export([(0, 8)], [(2, 3)], 12, 0),
                         [(0, 2), (3, 8)])

    def test_suppression_geometry_tolerance(self):
        rec = dict(durationSeconds=30, ignoredIntervals=[])
        base, other = [[2, 8], [16, 20]], [[4, 6]]
        emitted = [candidate(0, 2, 8, False), candidate(1, 16, 20, True)]
        for policy in ('suppression_zero', 'suppression_half', 'suppression_any'):
            self.assertTrue(audit.audit_candidate_plan(rec, base, other, policy, emitted)['passed'])

    def test_strict_half_and_any_are_distinct(self):
        rec = dict(durationSeconds=20, ignoredIntervals=[])
        base, other = [[0, 10]], [[2, 3]]  # Dilation covers exactly half.
        for policy, flagged in [('suppression_half', False), ('suppression_any', True)]:
            audit.audit_candidate_plan(rec, base, other, policy, [candidate(0, 0, 10, flagged)])

    def test_bidirectional_candidate_expansion_and_labels_unused(self):
        rec = dict(durationSeconds=30, ignoredIntervals=[], rallies='poison: must not be read')
        base, other = [[2, 4], [12, 14]], [[2, 4], [20, 22]]
        emitted = [candidate(0, 2, 4, False), candidate(1, 12, 14, True),
                   candidate(2, 20, 22, True, 'negative')]
        audit.audit_candidate_plan(rec, base, other, 'bidirectional_any', emitted)

    def test_guard_protects_original_nonoverlapping_tails(self):
        rec = dict(durationSeconds=30, ignoredIntervals=[])
        base = [[1, 4], [8, 10], [20, 21]]
        emitted = [candidate(0, 1, 4, False), candidate(1, 8, 10, False),
                   candidate(2, 20, 21, True)]
        audit.audit_candidate_plan(rec, base, [], 'guarded_trim', emitted,
                                   previous=[[1, 4], [20, 21]], v2=[[8, 10]])

    def test_guard_exact_half_second_gap_is_not_protected(self):
        rec = dict(durationSeconds=30, ignoredIntervals=[])
        base = [[1, 4], [8.5, 10]]
        audit.audit_candidate_plan(rec, base, [], 'guarded_trim',
                                   [candidate(0, 1, 4, True), candidate(1, 8.5, 10, True)],
                                   previous=[[1, 4]], v2=[[8.5, 10]])

    def test_guard_replays_raw_combiner_before_ignored_subtraction(self):
        rec = dict(durationSeconds=20, ignoredIntervals=[[4, 6]])
        # A neural interval in ignored time can support neighboring visible
        # material in the existing raw combiner; excluded time is never queued.
        base = [[2, 8]]
        audit.audit_candidate_plan(rec, base, [[4, 6]], 'guarded_trim',
                                   [candidate(0, 2, 4, False), candidate(1, 6, 8, False)],
                                   previous=base, v2=[])

    def test_solo_absolute_grid_and_half_open_scores(self):
        rec = dict(durationSeconds=12, ignoredIntervals=[[6, 7]])
        base = [[3, 6]]
        emitted = [candidate(0, 0, 3, False, 'negative', .1),
                   candidate(1, 3, 6, True, 'positive', .6),
                   candidate(2, 7, 10, True, 'negative', .4),
                   candidate(3, 10, 12, True, 'negative', None)]
        audit.audit_candidate_plan(rec, base, None, 'uncertain_narrow', emitted,
                                   times=[0, 3, 5, 7, 9, 12], live=[.1, .5, .7, .2, .4, .9])

    def test_positive_only_does_not_flag_empty_negative(self):
        rec = dict(durationSeconds=5, ignoredIntervals=[])
        emitted = [candidate(0, 0, 1, False, 'negative', None),
                   candidate(1, 1, 2, True, 'positive', None),
                   candidate(2, 2, 5, False, 'negative', None)]
        audit.audit_candidate_plan(rec, [[1, 2]], None, 'positive_uncertain', emitted,
                                   times=[], live=[])

    def test_probability_threshold_equality_is_not_uncertain(self):
        rec = dict(durationSeconds=5, ignoredIntervals=[])
        emitted = [candidate(0, 0, 2, False, 'positive', .8),
                   candidate(1, 2, 5, False, 'negative', .2)]
        audit.audit_candidate_plan(rec, [[0, 2]], None, 'uncertain_medium', emitted,
                                   times=[0, 2], live=[.8, .2])

    def test_changed_flag_is_detected(self):
        rec = dict(durationSeconds=10, ignoredIntervals=[])
        with self.assertRaisesRegex(ValueError, 'flagged differs'):
            audit.audit_candidate_plan(rec, [[1, 2]], [], 'suppression_zero',
                                       [candidate(0, 1, 2, False)])

    def test_missing_pad_and_overlapping_inventory_rejected(self):
        rec, base, result = fixture()
        bad = copy.deepcopy(result)
        del bad['boundaryExports']['3']
        with self.assertRaisesRegex(ValueError, 'four padding'):
            audit.audit_record(rec, base, bad)
        result['candidates'][1]['start'] = 3
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            audit.audit_record(rec, base, result)


if __name__ == '__main__':
    unittest.main()
