from analysis.private_ledger import private_value
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('production_panel_tests', REPO/'scripts/audit-neural-production-panel.py')
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)


def fixture():
    rows = [{'id':str(i), 'sourceGroup':'g'+str(i), 'contentSha256':'a'*64,
             'eligibleRoles':['infer', 'evaluate'] if i < 34 else ['infer'],
             'scoringPolicy':'exact-core' if i < 34 else 'none'} for i in range(42)]
    records = [{'recordingId':r['id'], 'sourceGroup':r['sourceGroup'], 'contentSha256':r['contentSha256'],
        'features':{'audiovisual':{'path':private_value('private-reference-0060')+r['id']+'.npz','sha256':'b'*64}}} for r in rows]
    core = {'kind':'frozen-independent-inference-panel-v1', 'features':{'hash':'core'}, 'featureAudit':{}, 'registrar':{},
        'manifest':{'gold':'unchanged'}, 'recordingIds':[r['id'] for r in rows], 'scoredRecordingIds':[r['id'] for r in rows[:34]],
        'inferenceUsesLabels':False,'allInferenceTicksValid':True}
    panel = {**core,'features':{'hash':'av'},'modelFamily':'av','continuityReferencePanel':{},'productionProtocol':{},'interpretation':'fixture'}
    av = {'publicationMode':'av','encoderPlan':None,'records':records,'publicationScope':'all'}
    visual = {**deepcopy(av),'publicationMode':'core','encoderPlan':{'hash':'encoder'}}
    for row in visual['records']: row['features']['mobile'] = {'hash':'extra'}
    return panel,core,{'records':rows},av,visual


class ProductionPanelTests(unittest.TestCase):
    def test_same_av_population_with_additional_unused_embeddings_passes(self):
        self.assertEqual(len(gate.compare_contract(*fixture())),42)

    def test_changed_av_bytes_source_order_and_gold_fail(self):
        for mutation in ('av','source','order','gold','scored','timing'):
            values = list(fixture()); panel,core,manifest,av,visual = values
            if mutation == 'av': visual['records'][0]['features']['audiovisual']['sha256'] = 'c'*64
            elif mutation == 'source': visual['records'][0]['contentSha256'] = 'd'*64
            elif mutation == 'order': visual['records'].reverse()
            elif mutation == 'gold': panel['manifest'] = {'gold':'different'}
            elif mutation == 'scored': panel['scoredRecordingIds'] = panel['scoredRecordingIds'][:-1]
            else: visual['records'][0]['features']['imageInput'] = {'path':'other'}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): gate.compare_contract(*values)

    def test_unknown_panel_policy_or_qualification_change_fails(self):
        values = list(fixture()); values[0]['ignoredOverride'] = []
        with self.assertRaises(ValueError): gate.compare_contract(*values)
        values = list(fixture()); values[4]['publicationScope'] = 'fit'
        with self.assertRaises(ValueError): gate.compare_contract(*values)


if __name__ == '__main__': unittest.main()
