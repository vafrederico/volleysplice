from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis import neural_context_development as study
from analysis.tests.test_neural_expanded_development import example


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def fixture():
    groups = ['A', 'B', 'C', 'D']
    exact = [study.expanded.exact_supervision(example(f'exact-{g}-{i}', g, 132, i + 10))
             for i in range(2) for g in groups]
    data = {'exact': exact}
    for tier in ('draft', 'coverage'):
        data[tier] = [study.expanded.auxiliary_supervision(example(f'{tier}-{g}', g, 132, 20), tier)
                      for g in [*groups, 'aux-only']]
    manifest = {tier + 'Rows': [{'id': r.example.id, 'sourceGroup': r.example.group,
                                'featureCaches': {'audiovisual': {'sha256': r.example.id + '-cache'}},
                                'rallies': [{'start': i.start, 'end': i.end} for i in r.example.truth]}
                               for r in rows] for tier, rows in data.items()}
    contract = {'experiment': study.EXPERIMENT, 'contexts': list(study.CONTEXTS),
                'cohort': 'reviewed_export', 'cohorts': ['reviewed_export'], 'lossArm': 'baseline',
                'kinds': list(study.KINDS), 'seeds': [3407, 1729, 20260918], 'groups': groups,
                'checkpointEpochs': [5, 15, 30, 60], 'decoderCandidates': study.expanded.decoder_candidates(),
                'selection': {'recallEligibilityFloor': .95}, 'primaryMetric': 'F1_padP_coreR',
                'targetPaddingSeconds': 2, 'joinGapSeconds': 3, 'paddingSweep': [0, 1, 2, 3],
                'models': {k: study.adapter.model_metadata(k, context='short') for k in study.KINDS},
                'contextModels': {c: {k: study.adapter.model_metadata(k, context=c) for k in study.KINDS}
                                  for c in study.CONTEXTS},
                'manifestSha256': study.canonical_hash(manifest), 'fitReuse': deepcopy(study.FIT_REUSE),
                'training': {'haloTicks': 62, 'coreTicks': 128}, 'environment': {'synthetic': True},
                'screen': {'meanF1Gain': .02}, 'retentionRecoveryScreen': {'synthetic': True},
                'evaluationPopulation': {'records': 8, 'groups': groups}}
    return data, manifest, contract


class SyntheticFitter:
    """Writes tiny deterministic fixtures; never allocates or trains a model."""
    def __init__(self):
        self.executed = []

    def __call__(self, train, auxiliary, validation, kind, seed, epochs, destination,
                 device, contract_hash, arm, *, context):
        train_groups = {r.example.group for r in train}
        auxiliary_groups = {g for rows in auxiliary.values() for g in (r.example.group for r in rows)}
        valid_groups = {e.group for e in validation}
        if valid_groups & (train_groups | auxiliary_groups):
            raise AssertionError('synthetic fitter received a leaking fold')
        training = {'contractSha256': contract_hash, 'kind': kind, 'seed': seed, 'contextProfile': context,
                    'lossArm': arm, 'trainIds': [r.example.id for r in train],
                    'auxiliaryIds': {tier: [r.example.id for r in rows] for tier, rows in auxiliary.items()},
                    'trainGroups': sorted(train_groups),
                    'auxiliaryGroups': {tier: sorted({r.example.group for r in rows}) for tier, rows in auxiliary.items()},
                    'model': study.adapter.model_metadata(kind, context=context)}
        expected = {**training, 'validationIds': [e.id for e in validation], 'epochs': list(epochs)}
        if destination.exists():
            study.validate_completion(destination, expected)
            return None
        self.executed.append((destination, deepcopy(expected)))
        destination.mkdir(parents=True)
        for epoch in epochs:
            arrays = {e.id: np.full((len(e.times), 4), .4, np.float32) for e in validation}
            for e in validation:
                arrays[e.id][~e.valid] = 0
            np.savez_compressed(destination/f'predictions-{epoch}.npz', **arrays)
            np.savez_compressed(destination/f'weights-{epoch}.npz', synthetic=np.array([epoch, seed], dtype=np.float32))
        save(destination/'completed.json', {
            **expected, 'scalerTrainIds': training['trainIds'], 'trainingIdentity': training,
            'trainingIdentitySha256': study.canonical_hash(training),
            'artifacts': {f'{stem}-{epoch}.npz': study.digest(destination/f'{stem}-{epoch}.npz')
                          for epoch in epochs for stem in ('predictions', 'weights')}})


def reference_fixture(root):
    _, _, contract = fixture()
    contract['cohorts'] = list(study.expanded.COHORTS)
    registration = {'contract': contract, 'sha256': study.canonical_hash(contract)}
    save(root/'preregistration.json', registration)
    rows = [{'cohort': c, 'kind': k, 'lossArm': a, 'seed': s, 'contractSha256': registration['sha256']}
            for c in study.expanded.COHORTS for k in study.KINDS for a in study.source.ARMS for s in contract['seeds']]
    report = {'status': 'completed-short-boost-transfer-development', 'contractSha256': registration['sha256'],
              'protectedTestOpened': False, 'productionPromotionAllowed': False, 'results': rows}
    save(root/'report.json', report)
    report_identity = study.identity(root/'report.json')
    save(root/'summary.json', {'status': 'completed-short-boost-transfer-audit', 'contractSha256': registration['sha256'],
                               'inputs': [report_identity]})
    audit_paths = []
    for kind in ('neural-short-boost-independent-tensor-audit-v1', 'independent-short-boost-transfer-interval-audit-v1'):
        path = root/(kind+'.json')
        save(path, {'kind': kind, 'passed': True, 'contractSha256': registration['sha256'], 'inputs': [report_identity]})
        audit_paths.append(path)
    return {'path': str(root), 'contractSha256': registration['sha256'],
            'preregistrationFileSha256': study.digest(root/'preregistration.json'),
            'reportSha256': study.digest(root/'report.json'), 'summarySha256': study.digest(root/'summary.json'),
            'independentAudits': [study.identity(p) for p in audit_paths]}


class ContextOwnershipTests(unittest.TestCase):
    def test_six_owners_supply_twelve_views_and_exclude_both_groups_in_all_tiers(self):
        _, manifest, contract = fixture()
        owners, views = study.fold_graph(contract, 'contract', manifest, 'tcn', 3407)
        self.assertEqual((len(owners), len(views)), (6, 12))
        for owner in owners.values():
            excluded = set(owner['excludedGroups'])
            self.assertEqual(sum(v['ownerId'] == owner['ownerId'] for v in views), 2)
            self.assertFalse(excluded & set(owner['training']['trainGroups']))
            for tier, rows in (('draft', manifest['draftRows']), ('coverage', manifest['coverageRows'])):
                self.assertEqual(owner['training']['auxiliaryIds'][tier], [r['id'] for r in rows if r['sourceGroup'] not in excluded])
                self.assertFalse(excluded & set(owner['training']['auxiliaryGroups'][tier]))
                self.assertIn(tier+'-aux-only', owner['training']['auxiliaryIds'][tier])
        for view in views:
            self.assertEqual(view['validationIds'], [r['id'] for r in manifest['exactRows'] if r['sourceGroup'] == view['innerValidationGroup']])
            self.assertTrue(all(view['outerHeldSourceGroup'] not in rid for rid in view['validationIds']))
        for cohort, tiers in (('exact', set()), ('draft', {'draft'})):
            changed = {**contract, 'cohort': cohort}
            plan = study.owner_plan(changed, 'contract', manifest, 'tcn', 3407, ['A', 'B'])
            self.assertEqual(set(plan['training']['auxiliaryIds']), tiers)

    def test_owner_identity_ignores_role_but_binds_recipe_order_and_feature_label_identity(self):
        _, manifest, contract = fixture()
        def plan(c=contract, m=manifest, seed=3407, kind='tcn', excluded=('A', 'B')):
            return study.owner_plan(c, 'contract', m, kind, seed, excluded)
        original = plan()
        self.assertEqual(original, plan(excluded=('B', 'A')))
        mutations = []
        for field, value in (('lossArm', 'short_boost'), ('manifestSha256', 'new-cache-manifest')):
            mutations.append(plan(c={**contract, field: value}))
        for modification in ('cache', 'label', 'order'):
            m = deepcopy(manifest)
            if modification == 'cache':
                m['exactRows'][2]['featureCaches']['audiovisual']['sha256'] = 'repaired'
            elif modification == 'label':
                m['exactRows'][2]['rallies'][0]['end'] = 6.125
            else:
                m['exactRows'].reverse()
            mutations.append(plan(m=m))
        mutations.extend((plan(seed=1729), plan(kind='dino_tcn')))
        for changed in mutations:
            self.assertNotEqual(original['trainingKeySha256'], changed['trainingKeySha256'])
        with self.assertRaisesRegex(ValueError, 'excluded'):
            plan(excluded=('A', 'A'))

    def test_end_to_end_owner_execution_logical_selection_and_exact_resume(self):
        data, manifest, contract = fixture()
        fitter = SyntheticFitter()
        selected_scopes = []
        def choose(examples, probabilities):
            identifiers = {e.id for e in examples}
            self.assertEqual(len({e.group for e in examples}), 3)
            self.assertEqual(set(probabilities), {5, 15, 30, 60})
            self.assertTrue(all(set(values) == identifiers for values in probabilities.values()))
            selected_scopes.append(identifiers)
            return {'epoch': 5, 'decoder': contract['decoderCandidates'][0], 'innerR_core': .96,
                    'innerF1_padP_coreR': .8, 'recallEligibilityPassed': True, 'recallEligibilityFloor': .95}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = (data, manifest, contract, 'contract', 'tcn', 3407, root, 'cpu')
            result = study.run_cell(*args, fit_fn=fitter, choose_fn=choose)
            self.assertEqual(len(fitter.executed), 10)
            self.assertEqual(sum('inner-owners' in p.parts for p, _ in fitter.executed), 6)
            self.assertEqual(len(result['logicalInnerViews']), 12)
            self.assertEqual(len(result['predictions']), 8)
            self.assertEqual(len(result['selections']), 4)
            self.assertEqual(set(result['evaluation']['sourceGroups']), set(contract['groups']))
            for index, scope in enumerate(selected_scopes):
                self.assertEqual(scope, {r.example.id for r in data['exact'] if r.example.group != contract['groups'][index]})
            repeated = study.run_cell(*args, fit_fn=fitter, choose_fn=choose)
            self.assertEqual(result, repeated)
            self.assertEqual(len(fitter.executed), 10, 'resume must not create physical fits')
            first = study.read_verified(result['logicalInnerViews'][0])
            owner = Path(first['ownerPath'])
            checkpoint = owner/'weights-5.npz'
            checkpoint.write_bytes(checkpoint.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact hash'):
                study.run_cell(*args, fit_fn=fitter, choose_fn=choose)

    def test_logical_view_rejects_outer_scores_stale_metadata_and_wrong_context(self):
        data, manifest, contract = fixture()
        examples = {r.example.id: r.example for r in data['exact']}
        owners, views = study.fold_graph(contract, 'contract', manifest, 'tcn', 3407)
        view = views[0]
        plan = owners[view['ownerId']]
        excluded = set(plan['excludedGroups'])
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'owner'
            SyntheticFitter()([r for r in data['exact'] if r.example.group not in excluded],
                              study.expanded.auxiliary_for_fold(data, contract['cohort'], excluded),
                              [e for e in examples.values() if e.group in excluded], 'tcn', 3407, (5, 15, 30, 60),
                              folder, 'cpu', 'contract', 'baseline', context='short')
            meta = study.validate_completion(folder, {**plan['training'], 'validationIds': plan['validationIds']})
            reference = study.verify_logical_reference(view, plan, folder, meta)
            actual = study.load_logical_predictions(reference, plan, examples, 5)
            self.assertEqual(list(actual), view['validationIds'])
            leak = {**reference, 'validationIds': plan['validationIds']}
            with self.assertRaisesRegex(ValueError, 'outer group'):
                study.load_logical_predictions(leak, plan, examples, 5)
            wrong = {**reference, 'trainingKeySha256': 'changed'}
            with self.assertRaisesRegex(ValueError, 'training identity'):
                study.load_logical_predictions(wrong, plan, examples, 5)
            meta['contextProfile'] = 'original'
            save(folder/'completed.json', meta)
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                study.load_logical_predictions(reference, plan, examples, 5)
            with self.assertRaisesRegex(ValueError, 'contextProfile'):
                study.validate_completion(folder, {**plan['training'], 'validationIds': plan['validationIds']})

    def test_prediction_validation_rejects_ignored_nonzero_and_wrong_inventory(self):
        data, _, _ = fixture()
        e = data['exact'][0].example
        values = np.zeros((len(e.times), 4), np.float32)
        values[~e.valid] = .4
        with self.assertRaisesRegex(ValueError, 'Ignored'):
            study.prediction_view({e.id: values}, [e.id], [e.id], {e.id: e})
        values[:] = 0
        with self.assertRaisesRegex(ValueError, 'inventory'):
            study.prediction_view({e.id: values, 'outer': values}, [e.id], [e.id], {e.id: e})
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            study.prediction_view({e.id: values.astype(np.float64)}, [e.id], [e.id], {e.id: e})

    def test_immutable_write_rejects_conflict_and_reference_wrapper_preserves_nested_origin(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'view.json'
            study.write_immutable(path, {'owner': 1})
            study.write_immutable(path, {'owner': 1})
            with self.assertRaisesRegex(ValueError, 'Immutable'):
                study.write_immutable(path, {'owner': 2})
        row = {'cohort': 'exact', 'kind': 'tcn', 'seed': 3407, 'lossArm': 'baseline', 'contractSha256': 'old',
               'origin': {'type': 'reused-reference', 'fitRoot': '/historical/fits', 'reportSha256': 'historical'},
               'evaluation': {'primary': .8}, 'predictions': [1], 'selections': [2]}
        reference = {'path': '/completed/current', 'reportSha256': 'current-report', 'contractSha256': 'current-contract'}
        wrapped = study.reference_result(row, reference, 'new-contract')
        self.assertEqual(wrapped['origin']['sourceOrigin'], row['origin'])
        self.assertEqual(wrapped['origin']['sourceResult'], {k: row[k] for k in ('cohort', 'kind', 'lossArm', 'seed')})
        self.assertEqual(wrapped['context'], 'original')
        self.assertEqual(row['contractSha256'], 'old')
        self.assertNotIn('context', row)
        for field in ('evaluation', 'predictions', 'selections'):
            self.assertEqual(row[field], wrapped[field])


class ContextLaunchGateTests(unittest.TestCase):
    def test_complete_reference_requires_both_independent_audits_bound_to_exact_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = reference_fixture(root)
            _, report = study.completed_reference(reference)
            self.assertEqual(len(report['results']), 54)
            with self.assertRaisesRegex(ValueError, 'Both independent'):
                study.completed_reference({**reference, 'independentAudits': reference['independentAudits'][:1]})
            item = reference['independentAudits'][0]
            audit = study.read(item['path'])
            for change in ({'passed': False}, {'inputs': [{'path': str(root/'report.json'), 'sha256': 'different'}]}):
                save(Path(item['path']), {**audit, **change})
                reference['independentAudits'][0] = study.identity(item['path'])
                with self.assertRaisesRegex(ValueError, 'Independent audit'):
                    study.completed_reference(reference)

    def test_duplicate_reference_cell_cannot_masquerade_as_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = reference_fixture(root)
            report = study.read(root/'report.json')
            report['results'][-1] = deepcopy(report['results'][0])
            save(root/'report.json', report)
            reference['reportSha256'] = study.digest(root/'report.json')
            with self.assertRaisesRegex(ValueError, 'population'):
                study.completed_reference(reference)

    def test_registration_requires_same_preflight_cohort_arm_and_code(self):
        _, manifest, contract = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = reference_fixture(root/'reference')
            source_repo = root/'repo'
            old_code = {'frozen.py': 'old'}
            new_code = {**old_code, **{name: 'new' for name in study.NEW_SOURCES}}
            for name, value in new_code.items():
                path = source_repo/'analysis'/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
            code = {name: study.digest(source_repo/'analysis'/name) for name in new_code}
            save(root/'manifest.json', manifest)
            save(root/'dino.json', {'synthetic': True})
            (root/'protocol.md').write_text('A synthetic protocol for testing file identity.\n')
            contract.update(manifest=study.identity(root/'manifest.json'), dinoManifest=study.identity(root/'dino.json'),
                            manifestSha256=study.digest(root/'manifest.json'), code=code, referenceStudy=reference,
                            protocolSnapshot=study.identity(root/'protocol.md'))
            old = deepcopy(contract)
            old['code'] = {'frozen.py': code['frozen.py']}
            old['cohorts'] = list(study.expanded.COHORTS)
            old_reg = {'contract': old, 'sha256': study.canonical_hash(old)}
            preflight = {'passed': True, 'ownerReuseQualified': True, 'originalProfileReplayQualified': True,
                         'code': code, 'manifest': contract['manifest'], 'dinoManifest': contract['dinoManifest'],
                         'referenceContractSha256': old_reg['sha256'], 'cohort': contract['cohort'], 'lossArm': contract['lossArm'],
                         'referenceReport': study.identity(root/'reference/report.json'),
                         'referenceAudits': reference['independentAudits']}
            def registration():
                save(root/'preflight.json', preflight)
                contract['preflight'] = study.identity(root/'preflight.json')
                save(root/'preregistration.json', {'contract': contract, 'sha256': study.canonical_hash(contract)})
                return root/'preregistration.json'
            with patch.object(study, 'REPO', source_repo), patch.object(study, 'completed_reference', return_value=(old_reg, {})):
                study.validate_registration(registration())
                for key, value in (('cohort', 'exact'), ('lossArm', 'short_boost'), ('ownerReuseQualified', False)):
                    previous = preflight[key]
                    preflight[key] = value
                    with self.assertRaisesRegex(ValueError, 'Preflight|preflight'):
                        study.validate_registration(registration())
                    preflight[key] = previous
                preflight['referenceReport']['sha256'] = 'wrong-report'
                with self.assertRaisesRegex(ValueError, 'report/audit bindings'):
                    study.validate_registration(registration())
                preflight['referenceReport'] = study.identity(root/'reference/report.json')
                (root/'protocol.md').write_text('changed protocol')
                with self.assertRaisesRegex(ValueError, 'protocol changed'):
                    study.validate_registration(registration())
                contract['protocolSnapshot'] = study.identity(root/'protocol.md')
                (source_repo/'analysis/neural_context_fit.py').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    study.validate_registration(registration())

    def test_execution_terminal_record_covers_process_and_final_validation_failures(self):
        for failure in ('process', 'final-validation', None):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, manifest, contract = fixture()
                contract['referenceStudy'] = {'path': '/reference', 'reportSha256': 'old-report', 'contractSha256': 'old-contract'}
                reg = {'contract': contract, 'sha256': study.canonical_hash(contract)}
                original = [{'cohort': contract['cohort'], 'kind': k, 'architecture': k, 'lossArm': contract['lossArm'],
                             'seed': s, 'contractSha256': 'old-contract', 'origin': {'type': 'trained', 'fitRoot': '/old'},
                             'predictions': [], 'evaluation': {}, 'selections': []}
                            for k in study.KINDS for s in contract['seeds']]
                report = {'records': 8, 'results': original}
                def process(command, **kwargs):
                    if failure == 'process':
                        raise RuntimeError('synthetic worker failure')
                    kind, seed = command[-1].split(':')
                    row = {**next(r for r in original if r['kind'] == kind and r['seed'] == int(seed)),
                           'contractSha256': reg['sha256'], 'context': 'short'}
                    save(root/f"result-{contract['cohort']}-{kind}-{contract['lossArm']}-short-{seed}.json", row)
                returns = [(reg, manifest, report), ValueError('late registered source change') if failure == 'final-validation' else (reg, manifest, report)]
                with patch.object(study, 'validate_registration', side_effect=returns), \
                     patch.object(study.subprocess, 'run', side_effect=process), redirect_stdout(io.StringIO()):
                    if failure:
                        with self.assertRaisesRegex((RuntimeError, ValueError), 'worker failure|source change'):
                            study.run_study(root/'preregistration.json', 'cpu', 1)
                        self.assertFalse((root/'report.json').exists())
                    else:
                        final = study.run_study(root/'preregistration.json', 'cpu', 2)
                        self.assertEqual(len(final['results']), 12)
                        self.assertEqual(final['execution']['physicalFreshFits'], 60)
                status = 'failed' if failure else 'completed'
                terminal = list(root.glob(f'execution-*-{status}.json'))
                self.assertEqual(len(terminal), 1)
                evidence = study.read(terminal[0])
                self.assertEqual(evidence['status'], status)
                self.assertEqual(study.read_verified(evidence['started'])['status'], 'running')
                if failure:
                    self.assertIn('errorType', evidence)
                else:
                    self.assertEqual(study.read_verified(evidence['report']), final)


if __name__ == '__main__':
    unittest.main()
