"""Synthetic auditor checks only: no real data, training or model outcomes."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis import neural_context_development as runner
from analysis.tests.test_neural_context_development import fixture

SCRIPT = Path(__file__).resolve().parents[2]/'scripts/audit-neural-context-tensors.py'
spec = importlib.util.spec_from_file_location('context_tensor_auditor_test', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class ContextTensorAuditTests(unittest.TestCase):
    def test_independent_graph_matches_all_cohorts_and_representations(self):
        _, manifest, contract = fixture()
        for cohort in ('exact', 'draft', 'reviewed_export'):
            c = {**contract, 'cohort': cohort}
            for kind in audit.KINDS:
                for seed in c['seeds']:
                    graph = audit.independent_graph(c, 'contract', manifest, kind, seed)
                    self.assertEqual(graph, runner.fold_graph(c, 'contract', manifest, kind, seed))
                    owners, views = graph
                    self.assertEqual((len(owners), len(views)), (6, 12))
                    for plan in owners.values():
                        self.assertEqual(sum(v['ownerId'] == plan['ownerId'] for v in views), 2)
                        self.assertFalse(set(plan['excludedGroups']) & set(plan['training']['trainGroups']))
                        for groups in plan['training']['auxiliaryGroups'].values():
                            self.assertFalse(set(plan['excludedGroups']) & set(groups))

    def test_owner_ids_are_role_independent_but_reject_duplicate_exclusion(self):
        self.assertEqual(audit.owner_key(['A', 'B']), audit.owner_key(['B', 'A']))
        with self.assertRaisesRegex(ValueError, 'distinct'):
            audit.owner_key(['A', 'A'])

    def test_membership_rejects_auxiliary_group_leakage_and_scaler_scope(self):
        _, manifest, _ = fixture()
        by_id = {r['id']: r for tier in audit.TIERS for r in manifest[tier+'Rows']}
        expected = audit.membership(manifest, 'reviewed_export', ['A', 'B'])
        meta = {**deepcopy(expected), 'scalerTrainIds': expected['trainIds']}
        audit.validate_membership(meta, expected, by_id)
        changed = deepcopy(meta)
        changed['auxiliaryIds']['draft'].append('draft-A')
        with self.assertRaisesRegex(ValueError, 'membership'):
            audit.validate_membership(changed, expected, by_id)
        # Even a colluding changed expected membership cannot bypass group check.
        tampered_expected = {k: changed[k] for k in expected}
        with self.assertRaisesRegex(ValueError, 'Source-group leakage'):
            audit.validate_membership(changed, tampered_expected, by_id)
        changed = deepcopy(meta)
        changed['scalerTrainIds'] = changed['trainIds']+['draft-aux-only']
        with self.assertRaisesRegex(ValueError, 'Scaler'):
            audit.validate_membership(changed, expected, by_id)

    def test_logical_view_rejects_outer_scores_wrong_owner_and_stale_hash(self):
        _, manifest, c = fixture()
        owners, views = audit.independent_graph(c, 'contract', manifest, 'tcn', 3407)
        view = views[0]
        plan = owners[view['ownerId']]
        meta = {'trainingIdentitySha256': 'identity', 'artifacts': {'predictions-5.npz': 'hash'}}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save(folder/'completed.json', meta)
            actual = {**view, 'ownerPath': str(folder), 'ownerCompletedSha256': audit.digest(folder/'completed.json'),
                      'trainingKeySha256': plan['trainingKeySha256'], 'ownerValidationIds': plan['validationIds'],
                      'trainingIdentitySha256': 'identity', 'checkpointArtifacts': meta['artifacts']}
            audit.validate_logical_reference(actual, view, plan, folder, meta)
            for field, value in [('validationIds', plan['validationIds']), ('ownerId', 'another-owner'),
                                 ('ownerCompletedSha256', 'stale')]:
                changed = deepcopy(actual)
                changed[field] = value
                with self.assertRaisesRegex(ValueError, 'Logical view'):
                    audit.validate_logical_reference(changed, view, plan, folder, meta)
            wrong_view = {**view, 'ownerId': 'invalid'}
            with self.assertRaisesRegex(ValueError, 'Wrong logical owner'):
                audit.validate_logical_reference(actual, wrong_view, plan, folder, meta)

    def test_rf33_profile_rejects_wrong_dilation_or_training_halo(self):
        c = fixture()[2]
        for kind in audit.KINDS:
            audit.validate_context_metadata(c['models'][kind])
            for key, value in [('dilations', [1, 2, 4, 8, 16]), ('receptiveFieldTicks', 125),
                               ('originalPairedTrainingHaloTicks', 16), ('haloTicks', 62)]:
                with self.assertRaisesRegex(ValueError, 'context/RF'):
                    audit.validate_context_metadata({**c['models'][kind], key: value})

    def test_external_protocol_both_profiles_and_preflight_reference_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            protocol = Path(temporary)/'protocol.md'
            protocol.write_text('Synthetic prospective protocol.\n', encoding='utf-8')
            reference = {'path': 'synthetic-reference', 'reportSha256': 'report',
                         'independentAudits': [{'path': 'audit.json', 'sha256': 'audit'}]}
            models = {context: {kind: runner.adapter.model_metadata(kind, context=context) for kind in audit.KINDS}
                      for context in ('original', 'short')}
            contract = {'contextModels': models, 'protocolSnapshot': audit.identity(protocol), 'referenceStudy': reference}
            preflight = {'referenceReport': {'path': str(Path(reference['path'])/'report.json'), 'sha256': 'report'},
                         'referenceAudits': reference['independentAudits']}
            audit.verify_context_bindings(contract, preflight)
            changed = deepcopy(contract)
            changed['contextModels']['original']['tcn']['receptiveFieldTicks'] = 33
            with self.assertRaisesRegex(ValueError, 'Both context model'):
                audit.verify_context_bindings(changed, preflight)
            with self.assertRaisesRegex(ValueError, 'Preflight reference'):
                audit.verify_context_bindings(contract, {**preflight, 'referenceAudits': []})
            protocol.write_text('Changed.\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'protocol snapshot'):
                audit.verify_context_bindings(contract, preflight)

    def test_sampler_geometry_and_exposure_match_frozen_helpers_without_training(self):
        data, _, _ = fixture()
        rows = {tier: records for tier, records in data.items()}
        inputs = {r.example.id: {'group': r.example.group, 'valid': r.example.valid} for records in rows.values() for r in records}
        supervision = {r.example.id: {'mask': r.mask} for records in rows.values() for r in records}
        pools = {tier: audit.chunk_geometry([r.example.id for r in records], inputs, supervision) for tier, records in rows.items()}
        chunks = {tier: runner.expanded.make_chunks(records, np.zeros(104, np.float32), np.ones(104, np.float32), 'tcn')
                  for tier, records in rows.items()}
        for tier in audit.TIERS:
            self.assertEqual([(r['right']-r['left'], r['samplingWeight']) for r in pools[tier]],
                             [(len(c[0]), c[3]) for c in chunks[tier]])
            self.assertTrue(all(r['right']-r['left'] <= 252 for r in pools[tier]))
        for seed in (3407, 1729, 20260918):
            expected = audit.independent_exposure(pools, seed, 3)
            rngs = {tier: np.random.default_rng(np.random.SeedSequence([seed, i])) for i, tier in enumerate(audit.TIERS)}
            hashes = {tier: hashlib.sha256() for tier in chunks}
            observed, steps = [], 0
            for epoch in range(1, 4):
                for exact in runner.expanded.epoch_batches(chunks['exact'], rngs['exact']):
                    for tier in audit.TIERS:
                        selected = exact if tier == 'exact' else rngs[tier].choice(
                            len(chunks[tier]), len(exact), replace=True, p=runner.expanded.sampling_weights(chunks[tier])).tolist()
                        hashes[tier].update(json.dumps([epoch, steps, selected], separators=(',', ':')).encode())
                    steps += 1
                observed.append({'epoch': epoch, 'optimizerSteps': steps, 'exposureSha256': {t: h.hexdigest() for t, h in hashes.items()}})
            self.assertEqual(expected, observed)
            meta = {'history': [{**r, 'loss': .5} for r in observed], **observed[-1]}
            audit.validate_history(meta, expected)
            changed = deepcopy(meta)
            changed['history'][0]['exposureSha256']['exact'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'sampling/step'):
                audit.validate_history(changed, expected)

    def test_geometry_does_not_cross_ignored_spans_or_create_unsupervised_chunks(self):
        valid = np.ones(700, bool)
        valid[260:300] = False
        mask = np.repeat(valid[:, None], 4, axis=1).astype(np.float32)
        inputs = {'first': {'group': 'same', 'valid': valid},
                  'censored': {'group': 'same', 'valid': np.ones(700, bool)}}
        supervision = {'first': {'mask': mask}, 'censored': {'mask': np.zeros((700, 4), np.float32)}}
        rows = audit.chunk_geometry(['first', 'censored'], inputs, supervision)
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(r['id'] == 'first' for r in rows))
        self.assertTrue(all(r['right'] <= 260 or r['left'] >= 300 for r in rows))
        self.assertEqual(max(r['right']-r['left'] for r in rows), 252)
        # The censored recording stays in the original group record count.
        self.assertTrue(all(r['samplingWeight'] == 1/14 for r in rows))

    def test_context_tensors_keep_shapes_and_parameter_counts_cpu_only(self):
        try:
            import torch
        except ImportError:
            self.skipTest('CPU torch unavailable')
        before = torch.random.get_rng_state().clone()
        architecture = audit.architecture_contract(audit.low_level())
        self.assertEqual({k: v['parameterCount'] for k, v in architecture.items()}, {'tcn': 29700, 'dino_tcn': 46868})
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_both_loader_representations_check_actual_multiplier_bytes(self):
        from analysis.neural_short_boost_weighting import live_event_weights
        data, manifest, contract = fixture()
        inputs, supervision, diagnostics = {}, {}, {}
        for tier, rows in data.items():
            for row in rows:
                e = row.example
                inputs[e.id] = {'group': e.group, 'times': e.times, 'normalized': e.values}
                supervision[e.id] = {'valid': e.valid, 'targets': e.targets, 'mask': row.mask}
                diagnostics[e.id] = live_event_weights(row, 'baseline')[1]
        def loader(_manifest, _dino, with_dino):
            return {tier: [replace(row, example=replace(row.example, values=np.concatenate(
                [row.example.values, np.zeros((len(row.example.times), 3840), np.float32)], axis=1)))
                if with_dino else row for row in rows] for tier, rows in data.items()}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root/'synthetic-dino.npz'
            e = data['exact'][0].example
            np.savez(cache, timestamps=e.times, tokens=np.zeros((len(e.times), 10, 384), np.float16))
            wrapper = root/'dino-manifest.json'
            save(wrapper, {'records': [{'recordingId': rid, 'dinoPath': str(cache)} for rid in inputs]})
            contract = {**contract, 'manifest': {'path': str(root/'not-read.json')}, 'dinoManifest': {'path': str(wrapper)}}
            with patch('analysis.neural_short_boost_transfer.load_data', side_effect=loader):
                checked = audit.verify_loader(contract, manifest, inputs, supervision, diagnostics)
                self.assertEqual(len(checked), 2*len(inputs))
                self.assertEqual({r['shape'][1] for r in checked}, {104, 3944})
                def bad_vector(row, arm):
                    values, diagnostic = live_event_weights(row, arm)
                    values[0] += 1
                    return values, diagnostic
                with patch('analysis.neural_short_boost_weighting.live_event_weights', side_effect=bad_vector):
                    with self.assertRaisesRegex(ValueError, 'Actual live loss multipliers'):
                        audit.verify_loader(contract, manifest, inputs, supervision, diagnostics)

    def test_checkpoint_structural_checks_reject_wrong_shape_and_ignored_values(self):
        low = audit.low_level()
        mean, scale = np.zeros(104, np.float32), np.ones(104, np.float32)
        shape = {'stateShapes': {'model::tiny': [2, 4]}}
        valid = np.array([True, False, True])
        inputs = {'synthetic': {'tier': 'exact', 'times': np.arange(3)/4, 'valid': valid}}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            wp, pp = folder/'weights-5.npz', folder/'predictions-5.npz'
            values = np.full((3, 4), .5, np.float32)
            values[~valid] = 0
            np.savez(wp, mean=mean, scale=scale, **{'model::tiny': np.zeros((2, 4), np.float32)})
            np.savez(pp, synthetic=values)
            meta = {'validationIds': ['synthetic'], 'artifacts': {p.name: audit.digest(p) for p in (wp, pp)}}
            low.audit_checkpoint(folder, 5, meta, shape, mean, scale, inputs)
            np.savez(wp, mean=mean, scale=scale, **{'model::tiny': np.zeros((2, 3), np.float32)})
            meta['artifacts'][wp.name] = audit.digest(wp)
            with self.assertRaisesRegex(ValueError, 'tensor'):
                low.audit_checkpoint(folder, 5, meta, shape, mean, scale, inputs)
            np.savez(wp, mean=mean, scale=scale, **{'model::tiny': np.zeros((2, 4), np.float32)})
            values[1] = .1
            np.savez(pp, synthetic=values)
            meta['artifacts'] = {p.name: audit.digest(p) for p in (wp, pp)}
            with self.assertRaisesRegex(ValueError, 'Ignored'):
                low.audit_checkpoint(folder, 5, meta, shape, mean, scale, inputs)

    def test_nested_reference_origin_is_preserved(self):
        original = {'cohort': 'draft', 'kind': 'tcn', 'lossArm': 'baseline', 'seed': 3407,
                    'contractSha256': 'old', 'origin': {'type': 'reused-reference', 'fitRoot': 'historical'},
                    'predictions': [{'id': 'synthetic'}]}
        reference = {'path': 'previous-study', 'reportSha256': 'report', 'contractSha256': 'old'}
        wrapped = runner.reference_result(original, reference, 'new')
        audit.reference_copy(wrapped, original, reference, 'new')
        changed = deepcopy(wrapped)
        changed['origin']['sourceOrigin'] = {'type': 'trained'}
        with self.assertRaisesRegex(ValueError, 'nested provenance'):
            audit.reference_copy(changed, original, reference, 'new')

    def test_actual_audit_refuses_without_completed_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'completed context report'):
                audit.audit(Path(temporary)/'preregistration.json')


if __name__ == '__main__':
    unittest.main()
