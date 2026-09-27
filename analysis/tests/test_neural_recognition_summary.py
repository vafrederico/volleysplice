from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

from analysis.neural_evaluation import evaluate_predictions


def summarizer():
    path = Path(__file__).resolve().parents[2]/'scripts/summarize-neural-recognition.py'
    spec = importlib.util.spec_from_file_location('recognition_summary_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(identifier='one', group='a', predictions=None):
    return {'id': identifier, 'sourceGroup': group, 'durationSeconds': 100.,
            'rallies': [{'start': 12., 'end': 18.}, {'start': 30., 'end': 32.}],
            'ignoredIntervals': [{'start': 0., 'end': 8.}],
            'predictions': predictions if predictions is not None else [
                {'start': 12., 'end': 18.}, {'start': 50., 'end': 52.}]}


def result(seed, rows):
    return {'seed': seed, 'predictions': rows, 'evaluation': evaluate_predictions(rows)}


def reference_fixture(module, reference, rows):
    """Write explicitly synthetic control cells and the report that owns them."""
    controls = {}
    for family, kind in (('av', 'tcn'), ('dino', 'dino_tcn')):
        controls[family] = [{**result(seed, rows), 'cohort': 'reviewed_export', 'kind': kind,
                            'lossArm': 'short_boost', 'origin': {'syntheticFixtureOnly': True}}
                           for seed in module.SEEDS]
        for candidate in controls[family]:
            (reference/f'result-reviewed_export-{kind}-short_boost-{candidate["seed"]}.json').write_text(json.dumps(candidate))
    report = {'syntheticFixtureOnly': True, 'results': [*controls['av'], *controls['dino']]}
    (reference/'report.json').write_text(json.dumps(report))
    (reference/'preregistration.json').write_text(json.dumps({'syntheticFixtureOnly': True}))
    return controls, report


class RecognitionSummaryTests(unittest.TestCase):
    def test_metric_durations_separate_false_export_omitted_export_and_core(self):
        module = summarizer()
        measured = module.metrics(result(3407, [row()]))
        self.assertEqual(measured['paddedModelExportSeconds'], 16.)
        self.assertEqual(measured['paddedHumanExportSeconds'], 16.)
        self.assertEqual(measured['incorrectExportSeconds'], 6.)
        self.assertEqual(measured['wantedExportOmittedSeconds'], 6.)
        self.assertEqual(measured['missedCoreSeconds'], 2.)
        self.assertEqual(measured['completeLosses'], 1)
        self.assertEqual(measured['shortCompleteLosses'], 1)
        self.assertEqual(measured['longR_core'], 1.)
        self.assertEqual(92-measured['paddedHumanExportSeconds']-measured['incorrectExportSeconds'], 70.)

    def test_seed_means_use_pooled_metrics_and_preserve_each_padding_case(self):
        module = summarizer()
        rows = [row(), row('two', 'b', [{'start': 10., 'end': 60.}])]
        candidates = [result(seed, rows if index != 1 else [
            row(predictions=[]), row('two', 'b', [{'start': 12., 'end': 18.}])])
            for index, seed in enumerate(module.SEEDS)]
        summary = module.group_summary(candidates)
        for key in ('F1_padP_coreR', 'P_pad', 'R_core', 'paddedModelExportSeconds'):
            self.assertEqual(summary['mean'][key], statistics.mean(candidate['evaluation']['primary'][key] for candidate in candidates))
        per_video_mean = statistics.mean(evaluate_predictions([source])['primary']['F1_padP_coreR'] for source in rows)
        self.assertNotAlmostEqual(candidates[0]['evaluation']['primary']['F1_padP_coreR'], per_video_mean)
        self.assertEqual([entry['paddingSecondsBeforeAndAfter'] for entry in summary['padding']], [0., 1., 2., 3.])
        self.assertEqual(set(summary['sourceGroups']), {'a', 'b'})
        self.assertEqual([entry['seed'] for entry in summary['seeds']], list(module.SEEDS))
        for group in ('a', 'b'):
            for key in ('completeLosses', 'shortCompleteLosses', 'incompleteLosses', 'longR_core'):
                self.assertEqual(summary['sourceGroups'][group][key], statistics.mean(
                    module.metrics({'evaluation': candidate['evaluation']['sourceGroups'][group]})[key]
                    for candidate in candidates))

    def test_gold_identity_ignores_record_order_but_preserves_rally_identity_and_ignored_revision(self):
        module = summarizer()
        rows = [row(), row('two', 'b')]
        self.assertEqual(module.gold_identity(rows), module.gold_identity(list(reversed(rows))))
        changed = copy.deepcopy(rows)
        changed[0]['rallies'].reverse()
        self.assertNotEqual(module.gold_identity(rows), module.gold_identity(changed))
        changed = copy.deepcopy(rows)
        changed[0]['ignoredIntervals'][0]['end'] = 9.
        self.assertNotEqual(module.gold_identity(rows), module.gold_identity(changed))

    def test_production_complementarity_distinguishes_partial_full_and_new_misses(self):
        module = summarizer()
        base = {**row(), 'rallies': [{'start': 10., 'end': 20.}, {'start': 30., 'end': 36.},
                                   {'start': 60., 'end': 62.}]}
        production = result(None, [{**base, 'predictions': [{'start': 10., 'end': 20.}]}])
        predictions = [[{'start': 30., 'end': 30.5}, {'start': 60., 'end': 62.}],
                       [{'start': 10., 'end': 20.}],
                       [{'start': 10., 'end': 20.}, {'start': 30., 'end': 36.}]]
        candidates = [result(seed, [{**base, 'predictions': prediction}])
                      for seed, prediction in zip(module.SEEDS, predictions)]
        measured = module.production_complementarity(candidates, production)
        self.assertEqual(measured['productionCompleteMisses'], 2)
        self.assertEqual(measured['mean']['productionMissesRetainedAny'], 1.)
        self.assertEqual(measured['mean']['productionMissesRetainedPartly'], 1/3)
        self.assertEqual(measured['mean']['productionMissesRetainedFully'], 2/3)
        self.assertEqual(measured['mean']['newCompleteLosses'], 1/3)
        first = measured['seeds'][0]
        self.assertEqual([(r['recordingId'], r['truthIndex']) for r in first['productionMissesRetainedPartlyIds']], [('one', 1)])
        self.assertEqual(first['productionMissesRetainedPartlyIds'][0]['candidateRetainedCoreSeconds'], 2.5)
        self.assertEqual(first['productionMissesRetainedPartlyIds'][0]['productionRetainedCoreSeconds'], 0.)
        self.assertEqual([r['truthIndex'] for r in first['productionMissesRetainedFullyIds']], [2])
        self.assertEqual([r['truthIndex'] for r in first['newCompleteLossIds']], [0])
        changed = copy.deepcopy(candidates)
        changed[0]['evaluation']['guardrails']['primaryExportCoverage']['rallies'].pop()
        with self.assertRaisesRegex(ValueError, 'rally identities differ'):
            module.production_complementarity(changed, production)

    def test_acceptance_magnitudes_apply_to_means_and_direction_requires_two_seeds(self):
        module = summarizer()

        def values(seed, f1=.7, complete=10, short=5):
            return {'seed': seed, 'values': {'F1_padP_coreR': f1, 'R_core': .95, 'longR_core': .96,
                'completeLosses': complete, 'shortCompleteLosses': short, 'incompleteLosses': complete+5}}

        def means(rows):
            return {'mean': {key: statistics.mean(item['values'][key] for item in rows) for key in rows[0]['values']}}

        controls = [values(seed) for seed in module.SEEDS]
        candidates = [values(seed, f1, complete, short) for seed, f1, complete, short in zip(
            module.SEEDS, (.701, .701, .77), (9, 9, 1), (4, 4, 1))]
        with patch.object(module, 'metrics', side_effect=lambda item: item['values']), \
                patch.object(module, 'group_summary', side_effect=means):
            measured = module.compare(candidates, controls)
            self.assertTrue(measured['f1Passed'])
            self.assertTrue(measured['recoveryPassed'])
            self.assertTrue(all(item['f1DirectionPassed'] for item in measured['seeds']))
            self.assertIn('strictly fewer complete losses and short complete losses', measured['seedDirectionRule'])
            self.assertIn('0.02 F1 gain thresholds apply to seed means', measured['seedDirectionRule'])
            # A large gain in one seed cannot satisfy the directional replication gate.
            only_one = [values(seed, f1, complete, short) for seed, f1, complete, short in zip(
                module.SEEDS, (.7, .7, .8), (10, 10, 0), (5, 5, 0))]
            measured = module.compare(only_one, controls)
            self.assertFalse(measured['f1Passed'])
            self.assertFalse(measured['recoveryPassed'])

    def test_each_control_payload_is_bound_to_its_reference_report_cell(self):
        module = summarizer()
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary)
            controls, report = reference_fixture(module, reference, [row()])
            self.assertEqual(module.frozen_controls(reference, report), controls)
            for candidate in report['results']:
                with self.subTest(kind=candidate['kind'], seed=candidate['seed']):
                    path = reference/f'result-reviewed_export-{candidate["kind"]}-short_boost-{candidate["seed"]}.json'
                    changed = copy.deepcopy(candidate)
                    changed['origin']['unboundReplacement'] = True
                    # Gold, predictions and replayed metrics still agree: only provenance differs.
                    self.assertEqual(evaluate_predictions(changed['predictions']), candidate['evaluation'])
                    path.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, 'control payload differs from reference report'):
                        module.frozen_controls(reference, report)
                    path.write_text(json.dumps(candidate))

    def test_reference_report_requires_unique_exact_control_cells(self):
        module = summarizer()
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary)
            _, report = reference_fixture(module, reference, [row()])
            for mode in ('missing', 'duplicate', 'wrong-loss-arm', 'wrong-cohort'):
                changed = copy.deepcopy(report)
                if mode == 'missing':
                    changed['results'].pop(0)
                elif mode == 'duplicate':
                    changed['results'].append(copy.deepcopy(changed['results'][0]))
                else:
                    changed['results'][0]['lossArm' if mode == 'wrong-loss-arm' else 'cohort'] = 'different'
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'exactly one control cell'):
                    module.frozen_controls(reference, changed)

    def test_full_synthetic_summary_checks_audit_binding_and_correct_removed_time(self):
        """Temporary synthetic fixtures exercise validation, never certify real studies."""
        module = summarizer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, study = root/'synthetic-reference', root/'synthetic-study'
            reference.mkdir()
            study.mkdir()
            families, _ = reference_fixture(module, reference, [row(), row('two', 'b')])
            controls = families['av']
            production_path = root/'synthetic-production.json'
            production_path.write_text(json.dumps({'predictions': {source['id']: source['predictions'] for source in controls[0]['predictions']},
                                                   'evaluation': controls[0]['evaluation']}))
            reg = {'sha256': 'synthetic-only', 'contract': {'config': {'family': 'av', 'head': 'transformer'},
                    'referenceReport': module.identity(reference/'report.json')}}
            report = {'status': 'completed-recognition-development', 'contractSha256': 'synthetic-only', 'results': controls}
            (study/'preregistration.json').write_text(json.dumps(reg))
            (study/'report.json').write_text(json.dumps(report))
            fixture_audit = {'kind': 'independent-recognition-audit-v1', 'passed': True,
                'contractSha256': 'synthetic-only', 'syntheticFixtureOnly': True,
                'report': module.identity(study/'report.json'), 'registration': module.identity(study/'preregistration.json'),
                'counts': {'seeds': 3, 'physicalFits': 30, 'logicalInnerViews': 36}}
            (study/'audit.json').write_text(json.dumps(fixture_audit))
            output = root/'synthetic-summary.json'
            argv = ['summarize-neural-recognition.py', '--study', str(study), '--output', str(output)]
            with patch.object(module, 'REFERENCE', reference), patch.object(module, 'PRODUCTION', production_path), \
                    patch.object(sys, 'argv', argv):
                module.main()
            saved = json.loads(output.read_text())
            self.assertEqual(saved['models']['production_default']['mean']['correctlyRemovedSeconds'], 140.)
            self.assertEqual(saved['models']['av_transformer']['mean']['wantedExportOmittedSeconds'], 12.)
            self.assertEqual(saved['productionComplementarity']['av_transformer']['mean']['newCompleteLosses'], 0)
            self.assertEqual(saved['comparisons']['av_transformer']['productionCoverageMean'],
                             saved['productionComplementarity']['av_transformer']['mean'])
            self.assertTrue(any(item['path'].endswith('result-reviewed_export-tcn-short_boost-3407.json') for item in saved['references']))
            for mode in ('missing', 'wrong-hash', 'wrong-path'):
                changed = copy.deepcopy(reg)
                if mode == 'missing':
                    changed['contract'].pop('referenceReport')
                else:
                    changed['contract']['referenceReport']['sha256' if mode == 'wrong-hash' else 'path'] = 'different'
                (study/'preregistration.json').write_text(json.dumps(changed))
                changed_audit = {**fixture_audit, 'registration': module.identity(study/'preregistration.json')}
                (study/'audit.json').write_text(json.dumps(changed_audit))
                with self.subTest(mode=mode), patch.object(module, 'REFERENCE', reference), \
                        patch.object(module, 'PRODUCTION', production_path), \
                        patch.object(sys, 'argv', [*argv[:-1], str(root/'must-not-exist.json')]), \
                        self.assertRaisesRegex(ValueError, 'reference report binding differs'):
                    module.main()
                self.assertFalse((root/'must-not-exist.json').exists())
            (study/'preregistration.json').write_text(json.dumps(reg))
            (study/'audit.json').write_text(json.dumps(fixture_audit))
            # Even harmless bytes changing after an audit invalidate its binding.
            (study/'report.json').write_text(json.dumps(report, indent=2))
            with patch.object(module, 'REFERENCE', reference), patch.object(module, 'PRODUCTION', production_path), \
                    patch.object(sys, 'argv', [*argv[:-1], str(root/'must-not-exist.json')]):
                with self.assertRaisesRegex(ValueError, 'passing independent audit'):
                    module.main()
            self.assertFalse((root/'must-not-exist.json').exists())


if __name__ == '__main__':
    unittest.main()
