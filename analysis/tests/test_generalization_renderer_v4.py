import importlib.util
import json
from pathlib import Path
import unittest
from analysis.tests.test_generalization_renderer_v3 import fixture, embedded

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('scoped_report_v4_tests', REPO/'scripts/render-neural-generalization-report-v4.py')
renderer = importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)


def membership_fixture():
    data = fixture(); tasks = []
    models = ('av-tcn','av-transformer','mobile-tcn','dino-tcn','dino-transformer','distilled-mobile-tcn')
    variants = ('original-medium','expanded-medium','expanded-large','expanded-wide-validation','export-rally-training','export-rally-selection')
    for variant in ('original-corpus', *variants):
        for model in models:
            for draw in (3407,1729,20260918) if variant == 'original-corpus' else (3407,1729,20260918,20260923):
                for precision in ('fp32','fp16','int8') if model.startswith('dino-') else ('fp32',):
                    task_id = f'{variant}/{model}/{draw}'
                    tasks.append({'id': task_id, 'model': model, 'variant': variant, 'draw': draw, 'precision': precision,
                        'trainingExecution': {'trainingPerformedForThisTask': True, 'physicalOwnerTaskId': task_id, 'reusedAllFixedCheckpointEpochs': []},
                        'memberships': {'fit':['synthetic-A'], 'calibrate':['synthetic-B'], 'evaluate':['synthetic-C'], 'infer':['synthetic-A','synthetic-B','synthetic-C','synthetic-never']}})
    for model in ('productionDefault','productionUnion'):
        tasks.append({'id':model,'model':model,'variant':'fixed-production','draw':'fixed','precision':'shipped',
            'memberships':{'fit':['synthetic-A'],'calibrate':['synthetic-B'],'evaluate':['synthetic-C'],'infer':['synthetic-A','synthetic-B','synthetic-C','synthetic-never']}})
    data['tasks'] = tasks
    return data


class RendererV4Tests(unittest.TestCase):
    def test_payload_download_and_metrics_are_identical_to_v3(self):
        raw = json.dumps(membership_fixture()).encode()
        old, _ = renderer.load_v3().render(raw, 'fixture'); new, receipt = renderer.render(raw, 'fixture')
        for key in ('packedReport','canonicalReport'): self.assertEqual(embedded(new,key),embedded(old,key))
        self.assertEqual(receipt['kind'],'scoped-task-report-view-v4')
        self.assertEqual(len(membership_fixture()['tasks']),272)
        self.assertNotIn("const selected=$('task').value==='*'?TASKS:",new)
        self.assertIn("['*','all registered draws'].includes(selectedDraw)",new)

    def test_real_audit_and_frozen_v3_pin_remain_required(self):
        data=membership_fixture();data['metadata'].pop('syntheticFixture')
        with self.assertRaisesRegex(ValueError,'auditPassed'):renderer.render(json.dumps(data).encode(),'fixture')
        self.assertEqual(renderer.sha((REPO/'scripts/render-neural-generalization-report-v3.py').read_bytes()),renderer.V3_SHA)


if __name__=='__main__': unittest.main()
