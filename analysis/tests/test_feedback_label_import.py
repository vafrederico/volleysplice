from copy import deepcopy
import unittest

from analysis.feedback_label_import import human_layers
from analysis.tests.test_exported_project_dataset import corrected, marker, payload


class FeedbackLabelImportTests(unittest.TestCase):
    def sample(self):
        return payload([corrected('R1', 10., 20.), corrected('R2', 25., 30.)],
                       [marker('S1', 10., 'R1'), marker('S2', 25., 'R2')], cut_ids=['R1', 'R2'])

    def test_final_membership_excludes_included_but_suppressed_cut(self):
        value = self.sample(); value['finalExportIntervals'][0]['cutIds'] = ['R1']
        got = human_layers(value)
        self.assertEqual([r['id'] for r in got['rallies']], ['R1'])
        self.assertEqual([r['id'] for r in got['serveMarkers']], ['S1'])

    def test_partial_suppression_splits_coverage_only_for_its_own_cut(self):
        value = self.sample()
        value['finalExportProvenance'] = [{'kind': 'suppression-segment', 'start': 12., 'end': 27., 'cutIds': ['R1']}]
        got = human_layers(value)
        self.assertEqual([(r['start'], r['end']) for r in got['rallies']], [(10., 12.), (25., 30.)])

    def test_ignored_time_removed_without_converting_to_negative(self):
        value = self.sample(); value['corrections']['ignoredIntervals'] = [{'start': 0., 'end': 12., 'reason': 'non-game'}]
        got = human_layers(value)
        self.assertEqual(got['rallies'][0]['start'], 12.)
        self.assertEqual([r['id'] for r in got['serveMarkers']], ['S2'])
        self.assertEqual(got['hardNegatives'], [])

    def test_manual_timestamp_preserved_even_when_existing_normalizer_snaps(self):
        value = payload([corrected('M01', 20., 25.)], [marker('manual', 18., None, origin='manual')], cut_ids=['M01'])
        value['corrections']['correctedRanges'][0]['origin'] = 'manual'
        got = human_layers(value)
        self.assertEqual(got['serveMarkers'][0]['time'], 18.)
        self.assertEqual(got['normalizedReference']['serveEvents'][0]['time'], 20.)
        self.assertIn('manual-addition', got['rallies'][0]['tags'])
        self.assertFalse(got['importDetails']['frameRateConversionApplied'])

    def test_corrected_side_retains_original_model_side(self):
        value = self.sample(); raw = value['corrections']['scoreTracking']['state']['serveMarkers'][0]
        raw.update(side='far', modelSide='near', ignorePreviousPoint=True)
        got = human_layers(value)['serveMarkers'][0]
        self.assertEqual((got['side'], got['modelSide'], got['origin']), ('far', 'near', 'model'))
        self.assertTrue(got['ignorePreviousPoint'])

    def test_excluded_and_removed_markers_respected(self):
        value = self.sample(); state = value['corrections']['scoreTracking']
        state['excludedRallyIds'] = ['R1']; state['state']['removedModelMarkerIds'] = ['S2']
        self.assertEqual(human_layers(value)['serveMarkers'], [])

    def test_switches_preserved_with_exact_fractional_seconds(self):
        value = self.sample(); state = value['corrections']['scoreTracking']['state']
        state['sideSwitchMarkers'] = [{'id': 'switch', 'timestamp': 24.123456789, 'origin': 'manual'}]
        switch = human_layers(value)['sideSwitches'][0]
        self.assertEqual(switch['time'], 24.123456789)
        self.assertEqual(switch['origin'], 'manual')

    def test_false_positive_negative_never_overlaps_retained_or_ignored(self):
        value = self.sample(); value['corrections']['labels'] = {'falsePositives': [{'id': 'F1', 'start': 5., 'end': 15.}]}
        value['corrections']['ignoredIntervals'] = [{'start': 0., 'end': 7., 'reason': 'non-game'}]
        got = human_layers(value)['hardNegatives']
        self.assertEqual([(r['start'], r['end']) for r in got], [(7., 10.)])

    def test_original_payload_not_mutated(self):
        value = self.sample(); before = deepcopy(value); human_layers(value)
        self.assertEqual(value, before)


if __name__ == '__main__': unittest.main()
