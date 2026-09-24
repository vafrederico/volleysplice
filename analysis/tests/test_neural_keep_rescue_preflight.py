"""Synthetic file-only preflight/registration integration; no actual outcomes."""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis import neural_keep_rescue_development as runner
from analysis.tests.test_neural_keep_rescue_development import fixture, gate_fixture

REPO = Path(__file__).resolve().parents[2]


def script(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO/'scripts'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = script('synthetic_keep_preflight_cli', 'audit-neural-keep-rescue-preflight.py')
registration = script('synthetic_keep_registration_cli', 'register-neural-keep-rescue-study.py')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def complete_fixture(root):
    """Combine a full synthetic evidence envelope with six tiny baseline cells."""
    gate, repo = gate_fixture(root)
    reference = root/'reference'
    old = runner.read(reference/'preregistration.json')['contract']
    cells, fit_ids = [], {}
    for kind in runner.KINDS:
        for seed in old['seeds']:
            examples, manifest, contract, baseline, previous, fits = fixture(root, cohort='exact', kind=kind, seed=seed)
            cells.append(baseline)
            fit_ids.update(fits)
    for row, e in zip(manifest['exactRows'], examples):
        path = root/'timestamps'/f'{e.id}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, times=e.times, metadata_json=np.asarray(json.dumps({'duration': e.duration})))
        row['durationSeconds'] = e.duration
        row['featureCaches'] = {'audiovisual': runner.identity(path)}
    save(root/'exact.json', {'recordings': manifest['exactRows']})
    manifest['exactManifest'] = runner.identity(root/'exact.json')
    save(root/'manifest-pts-v1.json', manifest)
    old.update(manifestSha256=runner.digest(root/'manifest-pts-v1.json'),
               decoderCandidates=contract['decoderCandidates'], referenceStudy=previous['referenceStudy'])
    old_hash = runner.canonical_hash(old)
    save(reference/'preregistration.json', {'contract': old, 'sha256': old_hash})
    for cell in cells:
        cell['contractSha256'] = old_hash
    for path in fit_ids:
        meta = runner.read(path)
        if meta['contractSha256'] == 'current-contract':
            meta['contractSha256'] = old_hash
            save(path, meta)
    report = runner.read(reference/'report.json')
    report['contractSha256'] = old_hash
    selected = {(c['cohort'], c['kind'], c['lossArm'], c['seed']): c for c in cells}
    for index, cell in enumerate(report['results']):
        key = tuple(cell[k] for k in ('cohort', 'kind', 'lossArm', 'seed'))
        report['results'][index] = selected.get(key, {**cell, 'contractSha256': old_hash})
    save(reference/'report.json', report)
    bound_report = runner.identity(reference/'report.json')
    save(reference/'summary.json', {'status': 'completed-short-boost-transfer-audit',
         'contractSha256': old_hash, 'inputs': {'report': bound_report}})
    for name in ('tensor', 'interval'):
        value = runner.read(reference/(name+'.json'))
        value.update(contractSha256=old_hash, report=bound_report)
        if name == 'tensor':
            value['fits'] = [{'completed': runner.identity(path)} for path in fit_ids] + value['fits'][len(fit_ids):]
        save(reference/(name+'.json'), value)
    # The integration verifies the CLI envelopes and actual tiny NPZ replay.
    # Running the real qualification test suite is checked separately; its
    # subprocess is stubbed here to avoid recursive unittest invocation.
    for module in preflight.TESTS:
        path = repo/Path(*module.split('.')).with_suffix('.py')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Synthetic qualification-source fixture.\n')
    protocol = root/'protocol.md'
    protocol.write_text('# Synthetic prospective protocol\n' + 'Fixed checkpoint, decoder and inner-only selection.\n'*8)
    artifacts = {str(path): runner.digest(path) for completed in fit_ids for path in completed.parent.glob('*.npz')}
    return repo, reference, protocol, artifacts


def preflight_args(root, reference):
    return ['preflight', '--reference', str(reference), '--tensor-audit', str(reference/'tensor.json'),
            '--interval-audit', str(reference/'interval.json'), '--cohorts', 'exact', '--output', str(root/'qualified')]


def registration_args(root, reference, protocol, qualified, output):
    return ['register', '--reference', str(reference), '--tensor-audit', str(reference/'tensor.json'),
            '--interval-audit', str(reference/'interval.json'), '--cohorts', 'exact', '--protocol', str(protocol),
            '--preflight', str(qualified), '--output', str(output)]


class KeepRescuePreflightTests(unittest.TestCase):
    def test_noop_preflight_registers_bound_recipe_without_selection_training_or_input_writes(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            repo, reference, protocol, artifacts = complete_fixture(root)
            original_loader = runner.load_bound_predictions
            calls = []
            def outer_only(folder, *args, **kwargs):
                self.assertEqual(folder.name, 'refit', 'preflight must not read inner candidate predictions')
                calls.append(folder)
                return original_loader(folder, *args, **kwargs)
            original_proposals = runner.rescue.keep_core_proposals
            def noop_only(*args, **kwargs):
                self.assertIsNone(kwargs['threshold'], 'preflight must not evaluate rescue thresholds')
                return original_proposals(*args, **kwargs)
            with patch.object(runner, 'REPO', repo), patch.object(preflight, 'REPO', repo), \
                 patch.object(runner, 'load_bound_predictions', side_effect=outer_only), \
                 patch.object(runner.rescue, 'keep_core_proposals', side_effect=noop_only), \
                 patch.object(runner, 'select_rescue', side_effect=AssertionError('preflight selected a candidate')), \
                 patch.object(runner.source, 'fit_model', side_effect=AssertionError('preflight trained a model')), \
                 patch.object(preflight.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout='synthetic test subprocess fixture\n')) as tests, \
                 patch.object(sys, 'argv', preflight_args(root, reference)):
                preflight.main()
            result = runner.read(root/'qualified/report.json')
            self.assertEqual(result['kind'], 'keep-rescue-engineering-preflight-v1')
            self.assertTrue(result['passed'])
            self.assertFalse(result['candidateQualityMetricsInspected'])
            self.assertEqual((result['noOpRecordingsReplayed'], result['selectedRefitNPZs'], len(calls)), (48, 24, 24))
            self.assertEqual(len(result['checks']), 6)
            self.assertEqual(tests.call_args.args[0][3:-1], list(preflight.TESTS))
            destination = root/'registered'
            with patch.object(runner, 'REPO', repo), patch.object(registration, 'REPO', repo), \
                 patch.object(sys, 'argv', registration_args(root, reference, protocol, root/'qualified/report.json', destination)):
                registration.main()
                reg, manifest, report, fits = runner.validate_registration(destination/'preregistration.json')
            self.assertEqual((len(manifest['exactRows']), len(report['results']), len(fits)), (8, 54, 864))
            self.assertEqual(reg['contract']['cohorts'], ['exact'])
            self.assertEqual(reg['contract']['rescueOptions'], [None, .35, .5, .65, .8])
            self.assertEqual(reg['contract']['preflight'], runner.identity(root/'qualified/report.json'))
            self.assertFalse((destination/'report.json').exists(), 'registration must not run the experiment')
            self.assertEqual(len(list((destination/'registered-sources').glob('*.py'))), 18)
            self.assertTrue(all(runner.digest(destination/'registered-sources'/name) == sha
                                for name, sha in reg['contract']['code'].items()))
            self.assertEqual({path: runner.digest(path) for path in artifacts}, artifacts)

    def test_registrar_rejects_wrong_preflight_kind_or_environment_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            repo, reference, protocol, _ = complete_fixture(root)
            old = runner.read(reference/'preregistration.json')
            binding = preflight.reference_binding(reference, reference/'tensor.json', reference/'interval.json')
            qualified = {'kind': 'keep-rescue-engineering-preflight-v1', 'passed': True,
                         'noOpReplayQualified': True, 'selectionIsolationQualified': True,
                         'code': {**old['contract']['code'], **{name: runner.digest(repo/'analysis'/name) for name in runner.NEW_SOURCES}},
                         'manifest': runner.identity(root/'manifest-pts-v1.json'), 'cohorts': ['exact'],
                         'referenceContractSha256': old['sha256'], 'referenceReport': runner.identity(reference/'report.json'),
                         'referenceAudits': binding['independentAudits'],
                         'executionEnvironment': {'python': runner.platform.python_version(), 'numpy': np.__version__, 'device': 'cpu'}}
            for key, value in (('kind', 'unrelated-preflight'), ('executionEnvironment', {'device': 'different'})):
                candidate = root/(key+'.json')
                save(candidate, {**qualified, key: value})
                destination = root/('rejected-'+key)
                with self.subTest(key=key), patch.object(runner, 'REPO', repo), patch.object(registration, 'REPO', repo), \
                     patch.object(sys, 'argv', registration_args(root, reference, protocol, candidate, destination)):
                    with self.assertRaisesRegex(ValueError, 'qualify|preflight|environment'):
                        registration.main()
                self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
