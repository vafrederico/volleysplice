import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unittest

REPO=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('report_v3_tests',REPO/'scripts/render-neural-generalization-report-v3.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
VARIANTS=['original-medium','expanded-medium','expanded-large','expanded-wide-validation','export-rally-training','export-rally-selection']


def fixture():
    rows=[]
    for model in ('av-tcn','dino-tcn'):
        for index,variant in enumerate(VARIANTS):
            for draw in (3407,'all registered draws'):
                for pad in (2,3):
                    for policy in ('exact-rallies','reviewed-export'):
                        if model=='dino-tcn' and variant=='export-rally-selection' and policy=='reviewed-export': continue
                        for panel in ('common-unseen','task-unseen-sources'):
                            for floor in range(90,101):
                                status='incomplete-registered-draws' if floor==100 and variant=='expanded-large' and isinstance(draw,str) else 'available'
                                precision=.61+index*.035+(floor-90)*.003;recall=.94+(floor-90)*.004
                                rows.append({'model':model,'variant':variant,'draw':draw,'precision':'fp32','floorPercent':floor,'panelId':panel,'labelPolicy':policy,'productionFilter':'all','paddingSeconds':pad,'status':status,
                                    'precisionValue':precision if status=='available' else None,'recallValue':recall if status=='available' else None,'f1Value':2*precision*recall/(precision+recall) if status=='available' else None,
                                    'exportSeconds':600.,'humanExportSeconds':620.,'correctlyRemovedSeconds':560.,'incorrectExportSeconds':20.,'wantedExportOmittedSeconds':40.,'missedCoreSeconds':12.,'completeRallyLosses':1,'eventF1':.8,
                                    'scopeId':'other' if panel=='task-unseen-sources' and variant=='expanded-wide-validation' else 'common','aggregationScope':'saved complete-draw mean' if isinstance(draw,str) else 'one saved draw'})
    template=rows[0]
    for model,scope in (('productionDefault','common'),('production-unmatched','never')):
        for pad in (2,3):
            for policy in ('exact-rallies','reviewed-export'):
                for panel in ('common-unseen','task-unseen-sources'):
                    for floor in range(90,101):
                        rows.append({**template,'model':model,'variant':'fixed-production','draw':'fixed','precision':'shipped','floorPercent':floor,'paddingSeconds':pad,'labelPolicy':policy,'panelId':panel,'scopeId':scope})
    for floor in range(90,101): rows.append({**template,'variant':'historical-nested','draw':3407,'floorPercent':floor,'panelId':'common-unseen'})
    inventory=[{'id':key,'sourceGroup':'fixture-'+key,'labelTier':'exact','protected':True,'scoringPolicy':'exact-core'} for key in ('synthetic-A','synthetic-B','synthetic-C','synthetic-never')]
    return {'metadata':{'syntheticFixture':True,'auditPassed':False,'defaultScenario':'original-medium','defaultPanelId':'common-unseen','defaultLabelPolicy':'exact-rallies','defaultProductionFilter':'all','defaultPaddingSeconds':2,'defaultDraw':'all registered draws','defaultPrecision':'fp32'},
        'inventory':inventory,'scopes':{'common':['synthetic-A','synthetic-B'],'other':['synthetic-C'],'never':['synthetic-never']},'series':rows,'tasks':[]}


def embedded(html,key):
    return re.search(r'<script id="'+key+r'"[^>]*>([^<]*)</script>',html).group(1)


class RendererV3Tests(unittest.TestCase):
    def test_transport_and_exact_original_download_are_unchanged(self):
        raw=json.dumps(fixture(),indent=2).encode();v2=renderer.load_v2()
        old,_=v2.render(raw,'fixture-renderer');html,receipt=renderer.render(raw,'fixture-renderer')
        for key in ('packedReport','canonicalReport'):self.assertEqual(embedded(html,key),embedded(old,key))
        self.assertEqual(gzip.decompress(base64.b64decode(embedded(html,'canonicalReport'))),raw)
        packed=json.loads(gzip.decompress(base64.b64decode(embedded(html,'packedReport'))))
        self.assertEqual(v2.unpack_series(packed['series']),json.loads(raw)['series'])
        self.assertEqual(receipt['sourceSha256'],hashlib.sha256(raw).hexdigest())
        self.assertIn('SOURCE SCOPE MISMATCH',html);self.assertIn('filter-model',html)

    def test_frozen_v2_identity_and_real_audit_validation_are_retained(self):
        data=fixture();data['metadata'].pop('syntheticFixture')
        with self.assertRaisesRegex(ValueError,'auditPassed'):renderer.render(json.dumps(data).encode(),'fixture')
        self.assertEqual(renderer.sha((REPO/'scripts/render-neural-generalization-report-v2.py').read_bytes()),renderer.V2_SHA)


if __name__=='__main__':unittest.main()
