"""Public-report privacy and label-policy safeguards on synthetic summaries."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('distilled_large_report', REPO/'scripts/report-distilled-mobile-large.py')
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)
PRIVATE = 'PRIVATE_RECORDING_CANARY'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def evaluation(precision=.94, recall=.97, missed=1):
    padding = [dict(paddingSecondsBeforeAndAfter=pad, joinGapSeconds=3,
                    P_pad=precision, R_core=recall, F1_padP_coreR=2*precision*recall/(precision+recall),
                    paddedPrecisionIntersectionSeconds=precision*10, paddedModelExportSeconds=10.,
                    coreRecallIntersectionSeconds=recall*10, coreHumanSeconds=10.,
                    paddedHumanExportSeconds=12., exportDurationDifferenceSeconds=-2.,
                    recordings=[dict(id=PRIVATE)], privatePath=PRIVATE)
               for pad in (0, 1, 2, 3)]
    coverage = dict(originalRallies=4, evaluableRallies=4, fullyIgnoredRallies=0,
                    completeRallyLosses=missed, partialRallyLosses=1, fullyCoveredRallies=3-missed,
                    evaluableCoreSeconds=10., retainedCoreSeconds=10*recall, coreRecall=recall,
                    rallies=[dict(recordingId=PRIVATE, start=1., end=2.)])
    return dict(primary=padding[2], padding=padding, recordingCount=1, sourceGroupCount=1,
                guardrails=dict(primaryExportCoverage=coverage, predictedRallies=4, trueRallies=4,
                                eventPrecision=.75, eventRecall=.75, eventF1=.75,
                                privatePath=PRIVATE),
                recordings=[dict(id=PRIVATE, sourceGroup=PRIVATE)], sourceGroups={PRIVATE: {}})


def approximate(policy):
    suffix = 'export' if policy == 'reviewed-export' else 'reviewed'
    padding = [dict(paddingSecondsBeforeAndAfter=pad, joinGapSeconds=3,
                    modelExportSeconds=10., humanExportSeconds=12., exportDurationDifferenceSeconds=-2.,
                    **{f'P_{suffix}': .9, f'R_{suffix}': .75, f'F1_{suffix}': .82},
                    humanPaddingSeconds=0, humanJoinGapSeconds=0, privatePath=PRIVATE)
               for pad in (0, 1, 2, 3)]
    return dict(labelPolicy=policy, recordingCount=1, sourceGroupCount=1, primary=padding[2], padding=padding,
                rallyCoreMetricsAvailable=False, eventMetricsAvailable=False, recordings=[dict(id=PRIVATE)])


class DistilledLargeReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.experiment = self.root/'experiment'
        rows = [dict(id=PRIVATE, sourceGroup=PRIVATE, durationSeconds=20., scoringPolicy='exact-core',
                     rallies=[dict(start=1., end=5.)], ignoredIntervals=[])]
        manifest = self.root/'manifest.json'
        save(manifest, dict(records=rows))
        plan = dict(floorsPercent=[99], targetPaddingSeconds=2, paddingSeconds=[0, 1, 2, 3], joinGapSeconds=3,
                    inferenceManifest=dict(path=str(manifest), sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()),
                    tasks=[dict(variant='private-variant-name', seed=seed) for seed in (7, 9)])
        save(self.experiment/'plan.json', plan)
        self.candidates = [dict(variant='private-variant-name', draw=seed, seed=seed, floorPercent=99,
                                setting=dict(epoch=5, innerR_core=.995, innerF1_padP_coreR=.91,
                                    decoder=dict(smoothing=.5, enter=.5, minimum=.25, boundary=False)),
                                evaluation=evaluation(precision, recall, missed))
                           for seed, precision, recall, missed in ((7, .94, .97, 1), (9, .86, .999, 0))]
        self.document = dict(candidates=self.candidates,
                             plan=dict(path=str(self.experiment/'plan.json'),
                                       sha256=hashlib.sha256((self.experiment/'plan.json').read_bytes()).hexdigest()),
                             selected=[dict(self.candidates[0], mode='f1'), dict(self.candidates[1], mode='recall')],
                             commonExactPanelRecordingCount=1, commonExactPanelSourceGroupCount=1)
        save(self.experiment/'evaluation.json', self.document)

    def test_public_summary_excludes_nested_identifiers_and_reports_both_counts(self):
        value = reporter.report(self.experiment)
        text = json.dumps(value)+reporter.markdown(value)
        for hidden in (PRIVATE, 'private-variant-name', str(self.root), 'privatePath'):
            self.assertNotIn(hidden, text)
        self.assertEqual(value['feasible99FitCount'], 2)
        self.assertEqual([row['whollyMissedSavedHumanRallies'] for row in value['selected']], [1, 0])
        self.assertEqual([row['draw'] for row in value['selected']], [7, 9])
        self.assertEqual(len(value['selected'][0]['padding']), 4)
        self.assertIn('not a guarantee', reporter.markdown(value))

    def test_all_padding_cases_and_fixed_primary_are_required(self):
        for mutation in ('missing-case', 'wrong-primary', 'below-floor'):
            document = copy.deepcopy(self.document)
            if mutation == 'missing-case':
                document['candidates'][0]['evaluation']['padding'].pop()
            elif mutation == 'wrong-primary':
                document['candidates'][0]['evaluation']['primary'] = document['candidates'][0]['evaluation']['padding'][1]
            else:
                document['candidates'][0]['setting']['innerR_core'] = .989
            save(self.experiment/'evaluation.json', document)
            with self.assertRaises(ValueError):
                reporter.report(self.experiment)

    def test_baseline_requires_identical_gold_and_ignored_scope(self):
        baseline = self.root/'baseline'
        save(baseline/'plan.json', reporter.read(self.experiment/'plan.json'))
        save(baseline/'evaluation.json', self.document)
        value = reporter.report(self.experiment, baseline)
        self.assertTrue(value['baseline']['sameCommonExactScopeAndGoldVerified'])
        self.assertEqual(value['baselineDeltas'][0]['F1_padP_coreR'], 0.)
        rows = reporter.read(self.root/'manifest.json')
        rows['records'][0]['ignoredIntervals'] = [dict(start=0., end=.5)]
        changed = self.root/'changed-manifest.json'
        save(changed, rows)
        plan = reporter.read(baseline/'plan.json')
        plan['inferenceManifest'] = dict(path=str(changed), sha256=hashlib.sha256(changed.read_bytes()).hexdigest())
        save(baseline/'plan.json', plan)
        with self.assertRaisesRegex(ValueError, 'gold labels or ignored intervals differ'):
            reporter.report(self.experiment, baseline)

    def test_separate_exact_draft_and_export_panels_never_invent_rally_truth(self):
        panels = [dict(labelPolicy=policy, scope='all', evaluation=(evaluation() if policy == 'exact-rallies' else approximate(policy)),
                       foundRallies=4, recordings=[dict(id=PRIVATE)]) for policy in reporter.POLICIES]
        panels.append(dict(labelPolicy='reviewed-export', scope='no-production-training', status='empty', evaluation=None))
        document = dict(models=[dict(mode=mode, panels=panels,
                                     fitRoleCounts={'training': 2, 'unseen-by-this-fit-and-selection': 1, PRIVATE: 1},
                                     scoredRecordingCount=3, unscoredRecordingCount=1)
                                for mode in ('f1', 'recall')],
                        selections=dict(path=PRIVATE, sha256=hashlib.sha256((self.experiment/'evaluation.json').read_bytes()).hexdigest()))
        save(self.experiment/'all-video-evaluation.json', document)
        value = reporter.report(self.experiment)
        text = json.dumps(value)+reporter.markdown(value)
        self.assertNotIn(PRIVATE, text)
        published = value['allVideoEvaluation']['models'][0]['panels']
        self.assertEqual(published[0]['whollyMissedSavedHumanRallies'], 1)
        for row in published[1:3]:
            self.assertIsNone(row['whollyMissedSavedHumanRallies'])
            self.assertNotIn('R_core', row['primary'])
            self.assertFalse(row['eventMetricsAvailable'])
        self.assertEqual(published[-1]['status'], 'empty')
        self.assertIn('P_export', published[2]['primary'])
        self.assertIn('P_reviewed', published[1]['primary'])

    def test_stale_all_video_evaluation_is_rejected(self):
        save(self.experiment/'all-video-evaluation.json', dict(selections=dict(sha256='0'*64), models=[]))
        with self.assertRaisesRegex(ValueError, 'another selected run'):
            reporter.report(self.experiment)


if __name__ == '__main__':
    unittest.main()
