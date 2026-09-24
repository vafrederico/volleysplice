from __future__ import annotations

import copy
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/summarize-neural-context.py'
SPEC = importlib.util.spec_from_file_location('context_summary_tested', SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def fixture():
    contract = {'cohort': 'exact', 'lossArm': 'baseline', 'kinds': ['tcn', 'dino_tcn'],
                'seeds': [3407, 1729, 20260918], 'groups': ['a', 'b'],
                'primaryMetric': 'F1_padP_coreR', 'targetPaddingSeconds': 2,
                'joinGapSeconds': 3, 'paddingSweep': [0, 1, 2, 3],
                'retentionRecoveryScreen': {
                    'meanF1MinimumDelta': -.005, 'meanRCoreMinimumDelta': -.005,
                    'meanLongRCoreMinimumDelta': -.005, 'meanEventF1MinimumDelta': -.01,
                    'maximumCompleteLossRatio': .8, 'maximumShortCompleteLossRatio': .8,
                    'maximumIncompleteLossDelta': 0, 'maximumShortIncompleteLossDelta': 0,
                    'completeLossReductionSeedCount': 2, 'shortCompleteLossReductionSeedCount': 2,
                    'shortDurationSeconds': 3, 'coverageScope': 'primaryExportCoverage'}}
    results = []
    for kind in contract['kinds']:
        for context in ('original', 'short'):
            for seed in contract['seeds']:
                rows = [{'id': group, 'sourceGroup': group, 'durationSeconds': 50,
                         'rallies': [{'start': 2, 'end': 4}, {'start': 25, 'end': 35}],
                         'predictions': ([{'start': 2, 'end': 4}] if context == 'short' else [])
                                        + [{'start': 25, 'end': 35}],
                         'ignoredIntervals': [{'start': 11, 'end': 14}]} for group in contract['groups']]
                results.append({'cohort': 'exact', 'lossArm': 'baseline', 'kind': kind, 'context': context,
                                'seed': seed, 'origin': {'type': 'synthetic-test'},
                                'evaluation': summary.transfer.evaluate_predictions(rows), 'predictions': rows,
                                'selections': [{'recallEligibilityPassed': True, 'innerR_core': 1.,
                                                'recallEligibilityFloor': .95}]})
    return results, contract


class ContextSummaryTests(unittest.TestCase):
    def test_duplicate_or_mixed_recipe_is_rejected(self):
        results, contract = fixture()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            summary.verify_grid(results[:-1] + [results[0]], contract)
        results[-1]['lossArm'] = 'short_boost'
        with self.assertRaisesRegex(ValueError, 'loss arm'):
            summary.verify_grid(results, contract)

    def test_equal_benefits_have_zero_interaction_and_positive_recovery(self):
        results, contract = fixture()
        report, losses = summary.summarize_results(results, contract)
        self.assertEqual(len(report['aggregates']), 4)
        self.assertEqual(len(losses), 6)
        self.assertEqual(report['transfer']['meanSeedDifferenceInDifferences']['F1_padP_coreR'], 0.)
        self.assertEqual(report['transfer']['meanSeedDifferenceInDifferences']['all/completeLossRecovery'], 0.)
        for comparison in report['comparisons']:
            self.assertGreater(comparison['meanSeedPrimaryDelta']['F1_padP_coreR'], 0)
            self.assertEqual(comparison['perSeed'][0]['pointRecovery']['all']['completeLossRecovery'], 2)
        self.assertTrue(report['transfer']['replication']['retentionRecoveryScreen'])

    def test_each_seed_pools_recordings_and_does_not_select_best_padding(self):
        results, contract = fixture()
        report, _ = summary.summarize_results(results, contract)
        for row in report['aggregates']:
            self.assertEqual([p['paddingSecondsBeforeAndAfter'] for p in row['meanSeedPadding']], [0, 1, 2, 3])
            self.assertEqual(row['meanSeedPrimary']['F1_padP_coreR'], row['meanSeedPadding'][2]['F1_padP_coreR'])
        text = summary.markdown(report)
        self.assertIn('strictly <3s', text)
        self.assertIn('Human seconds', text)

    def test_independent_sweep_and_saved_metric_tampering(self):
        results, _ = fixture()
        result = results[-1]
        audit = summary.replay_metrics(result)
        self.assertEqual(audit['independentPaddingRows'], 20)
        changed = copy.deepcopy(result)
        changed['evaluation']['primary']['R_core'] = .9
        with self.assertRaisesRegex(ValueError, 'replay differs'):
            summary.replay_metrics(changed)

    def test_inner_infeasibility_prevents_replication(self):
        results, contract = fixture()
        results[-1]['selections'][0].update(recallEligibilityPassed=False, innerR_core=.9)
        report, _ = summary.summarize_results(results, contract)
        self.assertFalse(report['transfer']['replication']['retentionRecoveryScreen'])

    def test_runner_to_summary_replays_full_selection_refits_and_nested_reference_payloads(self):
        from analysis import neural_context_development as runner
        from analysis.tests.test_neural_context_development import fixture as runner_fixture, SyntheticFitter, save

        data, manifest, contract = runner_fixture()
        contract['retentionRecoveryScreen'] = fixture()[1]['retentionRecoveryScreen']
        # These are synthetic probabilities and tiny NPZ fixtures. No neural
        # model, real recording, or current-study outcome is opened or trained.
        by_id = {row.example.id: row.example for row in data['exact']}
        for row in manifest['exactRows']:
            e = by_id[row['id']]
            row.update(durationSeconds=e.duration, rallies=[i.to_dict() for i in e.truth],
                       ignoredIntervals=[i.to_dict() for i in e.ignored])
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            exact_path, manifest_path = root/'exact.json', root/'manifest.json'
            save(exact_path, {'recordings': manifest['exactRows']})
            manifest['exactManifest'] = runner.identity(exact_path)
            save(manifest_path, manifest)
            contract.update(manifest=runner.identity(manifest_path), manifestSha256=runner.digest(manifest_path), code={},
                            referenceStudy={'path': str(root/'reference'), 'reportSha256': 'synthetic-reference-report',
                                            'contractSha256': 'synthetic-reference-contract'})
            registration = {'contract': contract, 'sha256': runner.canonical_hash(contract)}
            contract_hash = registration['sha256']
            save(root/'preregistration.json', registration)
            fitter = SyntheticFitter()
            fresh = [runner.run_cell(data, manifest, contract, contract_hash, kind, seed, root, 'cpu', fit_fn=fitter)
                     for kind in contract['kinds'] for seed in contract['seeds']]
            self.assertEqual(len(fitter.executed), 60)
            sources = []
            for row in fresh:
                source_row = {key: copy.deepcopy(value) for key, value in row.items()
                              if key not in ('context', 'logicalInnerViews')}
                source_row['contractSha256'] = 'synthetic-reference-contract'
                source_row['origin'] = {'type': 'trained', 'fitRoot': '/synthetic-reference/'+row['kind']+'/'+str(row['seed'])}
                sources.append(source_row)
            originals = [runner.reference_result(row, contract['referenceStudy'], contract_hash) for row in sources]
            report = {'status': 'completed-context-development', 'contractSha256': contract_hash,
                      'manifestSha256': contract['manifestSha256'], 'sourceGroups': contract['groups'], 'records': 8,
                      'protectedTestOpened': False, 'productionPromotionAllowed': False, 'results': originals+fresh}
            save(root/'report.json', report)
            audit_path = root/'tensor-audit.json'
            save(audit_path, {'kind': 'independent-context-tensor-audit-v1', 'passed': True,
                              'contractSha256': contract_hash, 'inputs': [runner.identity(root/'report.json')]})
            # Launch-gate and real tensor audits have separate tests. Here their
            # prerequisite envelopes are fixtures; interval/choice/refit replay
            # and exact original-result payload comparison run without mocks.
            with patch.object(runner, 'validate_registration', return_value=(registration, manifest, {'results': sources})), \
                 patch.object(runner.expanded, 'load_data', return_value=data):
                result, losses = summary.summarize(root, audit_path)
            self.assertEqual(result['audit']['counts']['canonicalMetricReplays'], 12)
            self.assertEqual(result['audit']['counts']['reusedPayloadsVerified'], 6)
            self.assertEqual(result['audit']['counts']['freshSelectedRefitsReplayed'], 24)
            self.assertEqual(result['audit']['counts']['fullInnerGridSelectionsReplayed'], 24)
            self.assertEqual(result['audit']['counts']['independentPaddingScopeRows'], 624)
            self.assertEqual(len(losses['perSeed']), 6)

            # Epoch arrays are deliberately identical, so epoch60 has a fully
            # self-consistent score but loses the registered first-epoch tie.
            changed = copy.deepcopy(fresh[0])
            self.assertEqual(changed['selections'][0]['epoch'], 5)
            changed['selections'][0]['epoch'] = 60
            with self.assertRaisesRegex(ValueError, 'selection|winner|choice|grid'):
                summary.audit_selections(changed, data, manifest, contract, contract_hash)

            # Full source-origin equality is required, even when score/interval
            # payloads themselves remain identical.
            report['results'][0]['origin']['sourceOrigin']['fitRoot'] = '/different-reference'
            save(root/'report.json', report)
            save(audit_path, {'kind': 'independent-context-tensor-audit-v1', 'passed': True,
                              'contractSha256': contract_hash, 'inputs': [runner.identity(root/'report.json')]})
            with patch.object(runner, 'validate_registration', return_value=(registration, manifest, {'results': sources})), \
                 patch.object(runner.expanded, 'load_data', return_value=data):
                with self.assertRaisesRegex(ValueError, 'payload changed'):
                    summary.summarize(root, audit_path)


if __name__ == '__main__':
    unittest.main()
