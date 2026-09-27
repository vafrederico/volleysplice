"""Synthetic decoder/audit fixtures only; no real predictions or fitting."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from analysis import neural_keep_rescue as rescue
from analysis import neural_keep_rescue_development as runner
from analysis.schema import Interval
from analysis.tests.test_neural_keep_rescue_development import fixture

SCRIPT = Path(__file__).resolve().parents[2]/'scripts/summarize-neural-keep-rescue.py'
spec = importlib.util.spec_from_file_location('keep_rescue_summary_test', SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def enrich_fixture(root, **kwargs):
    examples, manifest, contract, baseline, previous, audited = fixture(root, **kwargs)
    examples = [replace(e, truth=e.truth+(Interval(18, 22),)) for e in examples]
    for row, e in zip(manifest['exactRows'], examples):
        row['rallies'] = [r.to_dict() for r in e.truth]
        row['durationSeconds'] = e.duration
    baseline['predictions'] = [summary.evaluation_row(e, []) for e in examples]
    baseline['evaluation'] = summary.transfer.evaluate_predictions(baseline['predictions'])
    # Synthetic fixture writes only the fields used by its runner. The actual
    # frozen completion schema additionally declares validationGroups.
    by_id = {e.id: e for e in examples}
    for path in audited:
        meta = summary.read(path)
        meta['validationGroups'] = sorted({by_id[rid].group for rid in meta['validationIds']})
        save(path, meta)
        audited[path] = summary.digest(path)
    return examples, manifest, contract, baseline, previous, audited


def metric_fixture(cohorts=('exact',)):
    contract = {'cohorts': list(cohorts), 'kinds': list(summary.KINDS), 'seeds': [3407, 1729, 20260918],
                'groups': ['A', 'B'], 'rescueOptions': list(summary.OPTIONS),
                'primaryMetric': 'F1_padP_coreR', 'targetPaddingSeconds': 2, 'joinGapSeconds': 3,
                'paddingSweep': [0, 1, 2, 3], 'retentionRecoveryScreen': deepcopy(summary.balanced.RECOVERY_CONTRACT)}
    results = []
    for cohort in cohorts:
        for kind in summary.KINDS:
            for variant in summary.VARIANTS:
                for seed in contract['seeds']:
                    rows = [{'id': group, 'sourceGroup': group, 'durationSeconds': 50,
                             'rallies': [{'start': 2, 'end': 4}, {'start': 25, 'end': 35}],
                             'ignoredIntervals': [{'start': 11, 'end': 14}],
                             'predictions': ([{'start': 2, 'end': 4}] if variant == 'selected' else [])
                                            + [{'start': 25, 'end': 35}]} for group in contract['groups']]
                    results.append({'cohort': cohort, 'kind': kind, 'architecture': kind, 'lossArm': 'baseline',
                        'seed': seed, 'rescueVariant': variant, 'origin': {'type': 'synthetic'},
                        'predictions': rows, 'evaluation': summary.transfer.evaluate_predictions(rows),
                        'selections': [{'recallEligibilityPassed': True, 'innerR_core': 1., 'recallEligibilityFloor': .95}]})
    return results, contract


class KeepRescueSummaryTests(unittest.TestCase):
    def test_exact_ties_eligibility_and_infeasible_fallback(self):
        def candidates(scores):
            return [{'optionIndex': i, 'keepThreshold': threshold, 'innerF1_padP_coreR': f, 'innerR_core': r}
                    for i, (threshold, (f, r)) in enumerate(zip(summary.OPTIONS, scores))]
        selected = summary.select_option(candidates([(.8, .96)]*5))
        self.assertIsNone(selected['keepThreshold'])
        selected = summary.select_option(candidates([(1, .94), (.8, .95), (.8, 1), (.7, 1), (.6, 1)]))
        self.assertEqual(selected['optionIndex'], 1)
        self.assertTrue(selected['recallEligibilityPassed'])
        selected = summary.select_option(candidates([(.5, .8), (.9, .94), (.9, .9), (.6, .7), (.4, .8)]))
        self.assertEqual(selected['optionIndex'], 1)
        self.assertFalse(selected['recallEligibilityPassed'])
        selected = summary.select_option(candidates([(.8, .96), (.8+1e-15, .96), (.7, 1), (.7, 1), (.7, 1)]))
        self.assertEqual(selected['optionIndex'], 1)
        with self.assertRaisesRegex(ValueError, 'order'):
            summary.select_option(list(reversed(candidates([(.8, .96)]*5))))

    def test_independent_components_match_helper_on_quantized_and_regular_grids(self):
        rng = np.random.default_rng(817)
        for quantized in (False, True):
            times = np.arange(320, dtype=np.float64)/4
            if quantized:
                times = np.round(times*30)/30
            ignored = (Interval(30, 33), Interval(50.06, 50.08))
            valid = np.ones(len(times), bool)
            for hole in ignored:
                valid &= ~((times >= hole.start) & (times < hole.end))
            e = SimpleNamespace(times=times, valid=valid, duration=80., ignored=ignored)
            for smoothing in (.5, 1.):
                for _ in range(5):
                    scores = rng.random((len(times), 4), dtype=np.float32)
                    for start, length in ((0, 21), (38, 25), (108, 25), (192, 25), (299, 21)):
                        scores[start:start+length, 3] = 1.
                    for threshold in summary.OPTIONS[1:]:
                        expected = rescue.keep_core_proposals(times, scores[:, 3], valid, 80., threshold=threshold,
                                                             smoothing_seconds=smoothing, ignored_intervals=ignored)
                        actual = summary.independent_proposals(e, scores, {'smoothing': smoothing}, threshold)
                        self.assertEqual(actual, [(i.start, i.end) for i in expected])
        # These are probability components only: no gold or feature fields exist.

    def test_independent_core_union_preserves_noop_and_does_not_join_positive_gap(self):
        times = np.arange(160)/4+.125
        e = SimpleNamespace(times=times, valid=np.ones(160, bool), duration=40., ignored=())
        scores = np.zeros((160, 4), np.float32)
        scores[(times >= 10) & (times < 16), 3] = 1
        decoder = {'smoothing': .5, 'enter': .5, 'minimum': .25, 'boundary': False}
        original = [Interval(5, 7), Interval(7, 8)]
        with patch('analysis.neural_development.decode', return_value=original):
            self.assertEqual(summary.independent_decode(e, scores, decoder, None), [i.to_dict() for i in original])
            actual = summary.independent_decode(e, scores, decoder, .5)
            self.assertEqual(actual[0], {'start': 5., 'end': 8.})
            self.assertGreater(actual[1]['start'], actual[0]['end'])
            contained = [Interval(0, 20)]
        with patch('analysis.neural_development.decode', return_value=contained):
            self.assertEqual(summary.independent_decode(e, scores, decoder, .5), [i.to_dict() for i in contained])

    def test_subset_grid_rejects_mixed_loss_duplicates_and_automatic_cohort_order(self):
        results, c = metric_fixture(('exact', 'reviewed_export'))
        self.assertEqual(len(summary.verify_grid(results, c)), 24)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            summary.verify_grid(results[:-1]+[results[0]], c)
        with self.assertRaisesRegex(ValueError, 'ordered subset'):
            summary.verify_grid(results, {**c, 'cohorts': ['reviewed_export', 'exact']})
        changed = deepcopy(results)
        changed[0]['lossArm'] = 'short_boost'
        with self.assertRaisesRegex(ValueError, 'loss arm'):
            summary.verify_grid(changed, c)

    def test_equal_benefits_have_zero_interaction_and_separate_replication(self):
        results, c = metric_fixture(summary.COHORTS)
        report, losses = summary.summarize_results(results, c)
        self.assertEqual(len(report['aggregates']), 12)
        self.assertEqual(len(report['comparisons']), 6)
        self.assertEqual(len(losses['perSeed']), 18)
        for interaction in report['transferInteractions']:
            self.assertEqual(interaction['meanSeedDifferenceInDifferences']['F1_padP_coreR'], 0.)
            self.assertEqual(interaction['meanSeedDifferenceInDifferences']['all/completeLossRecovery'], 0.)
            self.assertTrue(interaction['replication']['retentionRecoveryScreen'])
            self.assertFalse(interaction['positiveInteractionRequired'])
        for row in report['aggregates']:
            self.assertEqual([p['paddingSecondsBeforeAndAfter'] for p in row['meanSeedPadding']], [0, 1, 2, 3])
            self.assertEqual(row['meanSeedPrimary']['F1_padP_coreR'], row['meanSeedPadding'][2]['F1_padP_coreR'])
        text = summary.markdown(report)
        self.assertIn('zero new fits', text)
        self.assertIn('Human seconds', text)
        self.assertIn('strictly <3s', text)

    def test_canonical_and_endpoint_metrics_detect_tampering(self):
        results, _ = metric_fixture()
        checked = summary.replay_metrics(results[-1])
        self.assertEqual(checked['independentPaddingRows'], 20)
        changed = deepcopy(results[-1])
        changed['evaluation']['primary']['R_core'] = .9
        with self.assertRaisesRegex(ValueError, 'canonical evaluation'):
            summary.replay_metrics(changed)

    def test_full_cell_replay_binds_historical_and_current_npz_and_fixed_settings(self):
        for cohort, kind in (('exact', 'tcn'), ('draft', 'tcn'), ('reviewed_export', 'tcn'), ('exact', 'dino_tcn')):
            with tempfile.TemporaryDirectory() as temporary:
                examples, manifest, c, baseline, previous, audited = enrich_fixture(temporary, cohort=cohort, kind=kind)
                result = runner.run_cell(examples, manifest, c, 'new', baseline, previous, audited)
                checks = summary.audit_cell(result, baseline, examples, manifest, c, previous, audited)
                self.assertEqual(len(checks), 4)
                self.assertTrue(all(len(r['independentInnerOptions']) == 5 for r in checks))
                changed = deepcopy(result)
                changed['selections'][0]['decoder']['enter'] = .65
                with self.assertRaisesRegex(ValueError, 'epoch/decoder'):
                    summary.audit_cell(changed, baseline, examples, manifest, c, previous, audited)
                changed = deepcopy(result)
                changed['selections'][0]['optionIndex'] = 4
                with self.assertRaisesRegex(ValueError, 'winner/tie/fallback'):
                    summary.audit_cell(changed, baseline, examples, manifest, c, previous, audited)
                changed = deepcopy(result)
                changed['selections'][0]['innerPredictionBindings'][0]['membership']['validationIds'].append(examples[0].id)
                with self.assertRaisesRegex(ValueError, 'Inner evidence'):
                    summary.audit_cell(changed, baseline, examples, manifest, c, previous, audited)

    def test_original_prediction_hash_shape_and_ignored_mask_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            examples, manifest, c, baseline, previous, audited = enrich_fixture(temporary)
            root, contract_hash = runner.source_fit_root(baseline, previous, c['referenceStudy']['path'])
            folder = root/'outer-0'/'inner-0'
            held = [e for e in examples if e.group == 'B']
            members = summary.expected_membership(manifest, 'reviewed_export', {'A', 'B'}, 'B')
            kwargs = dict(manifest=manifest, kind='dino_tcn', seed=3407, source_contract=contract_hash, audited_fits=audited)
            scores, _ = summary.load_predictions(folder, 5, held, members, **kwargs)
            pp, cp = folder/'predictions-5.npz', folder/'completed.json'
            old = summary.read(cp)
            for defect in ('shape', 'ignored'):
                altered = {key: value.copy() for key, value in scores.items()}
                if defect == 'shape':
                    altered[held[0].id] = altered[held[0].id][:, :3]
                else:
                    altered[held[0].id][~held[0].valid] = .5
                np.savez_compressed(pp, **altered)
                with self.assertRaisesRegex(ValueError, 'NPZ changed'):
                    summary.load_predictions(folder, 5, held, members, **kwargs)
                meta = deepcopy(old)
                meta['artifacts'][pp.name] = summary.digest(pp)
                save(cp, meta)
                audited[cp.resolve()] = summary.digest(cp)
                with self.assertRaisesRegex(ValueError, 'probability shape/range/ignored'):
                    summary.load_predictions(folder, 5, held, members, **kwargs)
                # Restore the trusted synthetic envelope for next malformed NPZ.
                save(cp, old)
                audited[cp.resolve()] = summary.digest(cp)

    def test_baseline_wrapper_retains_nested_origin_and_real_summary_refuses_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, c, baseline, _, _ = enrich_fixture(temporary, cohort='exact', kind='tcn')
            wrapped = runner.reference_result(baseline, c['referenceStudy'], 'new')
            summary.reference_copy(wrapped, baseline, c['referenceStudy'], 'new')
            wrapped['origin']['sourceOrigin']['fitRoot'] = '/wrong'
            with self.assertRaisesRegex(ValueError, 'nested provenance'):
                summary.reference_copy(wrapped, baseline, c['referenceStudy'], 'new')
            with self.assertRaisesRegex(ValueError, 'completed explicitly registered'):
                summary.summarize(Path(temporary)/'not-started')

    def test_complete_synthetic_study_audit_counts_zero_fits_and_all_scopes(self):
        # The prior 54-cell launch qualification has separate runner tests.
        # Here its audit envelope is synthetic; the complete 12-cell decoder,
        # NPZ, membership, selection, cuts, metrics and reporting path is real.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = root/'rescue-study'
            study.mkdir()
            fixtures = [enrich_fixture(root, cohort='exact', kind=kind, seed=seed)
                        for kind in summary.KINDS for seed in (3407, 1729, 20260918)]
            examples, manifest, c, _, previous, _ = fixtures[0]
            old_registration = {'contract': previous, 'sha256': summary.canonical(previous)}
            old_reg_path = root/'reference'/'preregistration.json'
            save(old_reg_path, old_registration)
            audited, baselines = {}, []
            for _, _, _, baseline, _, bindings in fixtures:
                baseline['contractSha256'] = old_registration['sha256']
                for path in bindings:
                    meta = summary.read(path)
                    if meta['contractSha256'] == 'current-contract':
                        meta['contractSha256'] = old_registration['sha256']
                        save(path, meta)
                    audited[path] = summary.digest(path)
                baselines.append(baseline)
            old_report = {'results': baselines}
            old_report_path = root/'reference'/'report.json'
            save(old_report_path, old_report)
            old_summary_path = root/'reference'/'summary.json'
            save(old_summary_path, {'synthetic': True})
            # Unreferenced prior-fit identities fill the separately qualified
            # reference audit envelope. No files or predictions are invented
            # for these entries or counted as new fitted checkpoints.
            for index in range(864-len(audited)):
                audited[(root/'unreferenced-prior'/str(index)/'completed.json').resolve()] = 'synthetic-unused'
            tensor_path, interval_path = root/'prior-tensor.json', root/'prior-interval.json'
            save(tensor_path, {'kind': 'neural-short-boost-independent-tensor-audit-v1', 'passed': True,
                               'fits': [{'completed': {'path': str(p), 'sha256': sha}} for p, sha in audited.items()]})
            save(interval_path, {'kind': 'independent-short-boost-transfer-interval-audit-v1', 'passed': True})
            reference = {'path': str(root/'reference'), 'contractSha256': old_registration['sha256'],
                         'preregistrationFileSha256': summary.digest(old_reg_path), 'reportSha256': summary.digest(old_report_path),
                         'summarySha256': summary.digest(old_summary_path),
                         'independentAudits': [summary.identity(tensor_path), summary.identity(interval_path)]}
            av = root/'synthetic-timestamps.npz'
            np.savez(av, times=examples[0].times, metadata_json=json.dumps({'duration': 30.}))
            for row in manifest['exactRows']:
                row['featureCaches'] = {'audiovisual': summary.identity(av)}
            exact_path, manifest_path = root/'exact.json', root/'manifest.json'
            save(exact_path, {'recordings': manifest['exactRows']})
            manifest['exactManifest'] = summary.identity(exact_path)
            save(manifest_path, manifest)
            preflight, protocol = root/'preflight.json', root/'protocol.md'
            save(preflight, {'syntheticQualifiedEnvelope': True})
            protocol.write_text('Synthetic prospective zero-fit protocol.\n', encoding='utf-8')
            c.update(referenceStudy=reference, seeds=[3407, 1729, 20260918],
                     experiment='keep-head-short-rescue-development-v1', manifest=summary.identity(manifest_path),
                     manifestSha256=summary.digest(manifest_path), preflight=summary.identity(preflight),
                     protocolSnapshot=summary.identity(protocol), code={}, primaryMetric='F1_padP_coreR',
                     targetPaddingSeconds=2, joinGapSeconds=3, paddingSweep=[0, 1, 2, 3],
                     retentionRecoveryScreen=deepcopy(summary.balanced.RECOVERY_CONTRACT))
            reg = {'contract': c, 'sha256': summary.canonical(c)}
            save(study/'preregistration.json', reg)
            results = []
            for baseline in baselines:
                results.append(runner.reference_result(baseline, reference, reg['sha256']))
                results.append(runner.run_cell(examples, manifest, c, reg['sha256'], baseline, previous, audited))
            for row in results:
                save(runner.result_path(study, row), row)
            report = {'status': 'completed-keep-rescue-development', 'contractSha256': reg['sha256'],
                      'manifestSha256': c['manifestSha256'], 'records': 8, 'sourceGroups': c['groups'],
                      'protectedTestOpened': False, 'productionPromotionAllowed': False,
                      'execution': {'newFits': 0, 'device': 'cpu', 'candidateCells': 6, 'referenceCells': 6},
                      'results': results}
            save(study/'report.json', report)
            with patch.object(runner, 'validate_registration', return_value=(reg, manifest, old_report, audited)):
                output, losses = summary.summarize(study)
            counts = output['audit']['counts']
            self.assertTrue(output['passed'])
            self.assertEqual(output['newFits'], 0)
            self.assertEqual(counts['canonicalMetricReplays'], 12)
            self.assertEqual(counts['matchedBaselinePayloadsVerified'], 6)
            self.assertEqual(counts['outerRescueSelectionsReplayed'], 24)
            self.assertEqual(counts['innerOptionScoresIndependentlyReplayed'], 120)
            self.assertEqual(counts['independentFinalPaddingScopeRows'], 624)
            self.assertEqual(len(output['audit']['originalCompletedFitsReferenced']), 96)
            self.assertEqual(len(output['audit']['originalNPZArtifactsReferenced']), 192)
            self.assertEqual(len(losses['perSeed']), 6)
            self.assertFalse((study/'fits').exists())


if __name__ == '__main__':
    unittest.main()
