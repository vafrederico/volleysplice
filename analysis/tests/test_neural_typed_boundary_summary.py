from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

from analysis import neural_production_combinations as iv
from analysis.neural_boundary_metrics import evaluate_boundary_proposals
from analysis.neural_rally_identity_metrics import evaluate_rally_identities
from analysis.neural_split_metrics import evaluate_split_proposals
from analysis.tests.test_neural_boundary_metrics import record

PATH=Path(__file__).resolve().parents[2]/'scripts/summarize-neural-typed-boundaries.py'
SPEC=importlib.util.spec_from_file_location('typed_summary_under_test',PATH)
summary=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(summary)


def evaluation(records,exports):
    return {'durationMetrics':iv.duration_rows(records,export_overrides=exports),
            'identityMetrics':evaluate_rally_identities(records),
            'typedMetrics':evaluate_boundary_proposals(records),
            'rawCoverageMetrics':evaluate_split_proposals(records),
            **{k:{'passed':True} for k in ('fixedExportAudit','identityAudit','rawCoverageAudit','typedMetricAudit')}}


def fixture():
    records=[record(identity='a',group='g1'),record(identity='b',group='g2')]
    for r in records:r['predictions']=r['productionEvents'];r['splitProposals']=[]
    exports={r['id']:{str(p):iv.export(r['productionEvents'],r,p) for p in (0,1,2,3)} for r in records}
    base=evaluation(records,exports)
    contract={'seeds':[3407,1729,20260918],
              'policies':['first_start','typed_starts','separate_ends','head_refined'],
              'humanModes':['proposal_confirmation','full_parent'],
              'rankers':['chronological','evidence'],'budgetFractions':[.05,.1,.2,.4]}
    cells=[]
    for seed in contract['seeds']:
        automatic=[{**deepcopy(base),'id':'automatic--'+p,'policy':p} for p in contract['policies']]
        reviewed=[]
        for identity,config in summary.expected_reviews(contract).items():
            per=[]
            for r in records:
                row={'id':r['id'],'sourceGroup':r['sourceGroup'],'workload':{k:0 for k in summary.WORKLOAD}}
                if config['humanMode']=='proposal_confirmation':row.update({k:0 for k in summary.ACTIONS})
                per.append(row)
            reviewed.append({**deepcopy(base),'id':identity,**config,'perRecording':per,
                             'workload':{k:0 for k in summary.WORKLOAD}})
        cells.append({'seed':seed,'baseline':deepcopy(base),'automatic':automatic,'reviewed':reviewed})
    return contract,cells,records,exports


class TypedBoundarySummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.contract,cls.cells,cls.records,cls.exports=fixture()

    def test_full_matrix_all_scopes_and_null_full_parent_actions(self):
        result=summary.aggregate(self.contract,self.cells)
        self.assertEqual(result['automaticOutcomes'],12)
        self.assertEqual(result['reviewedOutcomes'],192)
        self.assertEqual(result['allArmsIncludingBaseline'],69)
        self.assertEqual(set(result['baseline']['sourceGroups']),{'g1','g2'})
        self.assertEqual(set(result['baseline']['recordings']),{'a','b'})
        self.assertEqual(result['baseline']['coverage']['resultRawCoreRecall'],1)
        full=next(a for a in result['reviewed'] if a['humanMode']=='full_parent')
        self.assertIsNone(full['humanActions']['acceptedCount'])
        self.assertEqual(full['humanActionsSeedStatistics']['acceptedCount']['availableSeeds'],0)
        self.assertIn('NOT corrected human output',result['typedMetricInterpretation'])
        json.dumps(result,allow_nan=False)

    def test_real_automatic_core_loss_is_retained_not_rejected(self):
        cells=deepcopy(self.cells);records=deepcopy(self.records)
        records[0]['predictions']=[{'id':'retained','start':8.,'end':25.}]
        loss=evaluation(records,self.exports)
        for cell in cells:cell['automatic'][0].update(deepcopy(loss))
        result=summary.aggregate(self.contract,cells)
        arm=next(a for a in result['automatic'] if a['policy']=='first_start')
        self.assertEqual(arm['coverage']['rawCoreSecondsLostFromBaseline'],10)
        self.assertEqual(arm['coverage']['additionalCompleteMisses'],1)
        self.assertEqual(arm['coverage']['resultRawCoreRecall'],.75)
        self.assertEqual(arm['primary'],result['baseline']['primary'])

    def test_proposal_only_loss_allowed(self):
        cells=deepcopy(self.cells)
        arm=next(a for a in cells[0]['reviewed'] if a['humanMode']=='proposal_confirmation')
        records=deepcopy(self.records);records[0]['predictions']=[]
        arm.update(evaluation(records,self.exports))
        result=summary.aggregate(self.contract,cells)
        row=next(a for a in result['reviewed'] if a['id']==arm['id'])
        self.assertGreater(row['coverage']['rawCoreSecondsLostFromBaseline'],0)

    def test_full_parent_loss_rejected(self):
        cells=deepcopy(self.cells)
        arm=next(a for a in cells[0]['reviewed'] if a['humanMode']=='full_parent')
        arm['typedMetrics']['pooled']['rawCoreSecondsLostFromBaseline']=.1
        with self.assertRaisesRegex(ValueError,'Full-parent review lost'):summary.aggregate(self.contract,cells)

    def test_export_change_rejected_even_when_event_core_loss_allowed(self):
        cells=deepcopy(self.cells);cells[0]['automatic'][0]['durationMetrics'][1]['paddedModelExportSeconds']+=1
        with self.assertRaisesRegex(ValueError,'Export changed'):summary.aggregate(self.contract,cells)

    def test_missing_review_arm_rejected(self):
        cells=deepcopy(self.cells);cells[0]['reviewed'].pop()
        with self.assertRaisesRegex(ValueError,'reviewed scope'):summary.aggregate(self.contract,cells)

    def test_missing_seed_rejected(self):
        with self.assertRaisesRegex(ValueError,'seed'):summary.aggregate(self.contract,self.cells[:2])

    def test_typed_audit_required(self):
        cells=deepcopy(self.cells);cells[0]['automatic'][0]['typedMetricAudit']['passed']=False
        with self.assertRaisesRegex(ValueError,'typedMetricAudit'):summary.aggregate(self.contract,cells)

    def test_nested_workload_closure_required(self):
        cells=deepcopy(self.cells);cells[0]['reviewed'][0]['perRecording'][0]['workload']['reviewSeconds']=1
        with self.assertRaisesRegex(ValueError,'Workload total'):summary.aggregate(self.contract,cells)

    def test_qualified_statistics_mean_and_range_on_nested_workload(self):
        samples=[]
        for cell,value in zip(self.cells,(0,1,5)):
            arm=deepcopy(cell['reviewed'][0])
            arm['perRecording'][0]['workload']['reviewSeconds']=value
            arm['workload']['reviewSeconds']=value
            samples.append(arm)
        output=summary.summarize(samples)
        self.assertEqual(output['workload']['reviewSeconds'],2)
        self.assertEqual(output['workloadSeedStatistics']['reviewSeconds']['min'],0)
        self.assertEqual(output['workloadSeedStatistics']['reviewSeconds']['max'],5)

    def test_physical_coverage_uses_typed_scope_not_censored_identity_scope(self):
        r=record(gold=((10,30),),ignored=((18,22),),predictions=[[22,42]])
        exports={r['id']:{str(p):iv.export(r['productionEvents'],r,p) for p in (0,1,2,3)}}
        e=evaluation([r],exports);out=summary.summarize([e,e,e])
        self.assertEqual(out['coverage']['rawCoreSecondsLostFromBaseline'],8)
        self.assertEqual(out['legacyIdentityCoverage']['rawCoreSecondsLostFromBaseline'],0)

    def test_markdown_has_all_modes_and_separates_original_candidate_metrics(self):
        result=summary.aggregate(self.contract,self.cells);doc=summary.markdown(result)
        for arm in result['reviewed']:self.assertIn(arm['id'],doc)
        self.assertIn('Raw core R %',doc)
        self.assertIn('Original candidate type and paired-boundary quality',doc)
        self.assertIn('not gold-corrected output',doc)
        self.assertIn('including rallies absent from model proposals',doc)
        self.assertIn('69 aggregate arms',doc)


if __name__=='__main__':unittest.main()
