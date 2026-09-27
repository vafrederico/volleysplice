from copy import deepcopy
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch
from analysis.neural_selection_serialization import load_module

REPO = Path(__file__).resolve().parents[2]
workflow = load_module('explicit_finalization_tests', REPO/'scripts/finalize-neural-generalization-v2.py')


class FinalizationTests(unittest.TestCase):
    def test_real_frozen_preflight_and_v2_audit_signatures(self):
        functions = [(workflow.cache.global_cache_gate,1),(workflow.execution.publication_preflight,2),
                     (workflow.containment.preflight_index,1)]
        numerical = load_module('actual_v2_signature_tests',REPO/'scripts/audit-neural-generalization-results-v2.py')
        functions.append((numerical.audit,4))
        for function,count in functions: inspect.signature(function).bind(*[None]*count)

    def test_restored_numerical_payload_is_exact_including_digest(self):
        base={'kind':'independent-generalization-result-report-audit-v2','passed':True,
              'reportContentSha256':'a'*64,'references':[{'path':'original','sha256':'b'*64}]}
        wrapped={**deepcopy(base),'kind':workflow.KIND,**{k:None for k in workflow.EXTRA}}
        workflow.restore_numerical(wrapped,base)
        for key,value in [('reportContentSha256','c'*64),('references',[]),('extraUnregisteredValue',True)]:
            bad={**wrapped,key:value}
            with self.subTest(key=key), self.assertRaises(ValueError): workflow.restore_numerical(bad,base)

    def test_global_cache_plan_binding_cannot_be_substituted(self):
        plan={'cachePlan':{'path':'registered','sha256':'a'*64}}
        with patch.object(workflow.cache,'global_cache_gate',return_value=({}, {'path':'side','sha256':'b'*64})), \
             patch.object(workflow.io,'read',return_value={'plan':{'path':'another','sha256':'c'*64}}):
            with self.assertRaisesRegex(ValueError,'another hash-cache plan'):workflow.global_preflight(plan,'global')

    def test_every_existing_publication_preflight_is_required(self):
        plan={'productionPanelAudit':{}}; global_refs={'globalSelectionGate':{},'globalIdentityCacheGate':{}}
        with patch.object(workflow,'verify_plan',return_value=plan), \
             patch.object(workflow,'global_preflight',return_value=global_refs), \
             patch.object(workflow.containment,'preflight_index',return_value=['all162']) as corrections, \
             patch.object(workflow.execution,'publication_preflight',return_value=['all270']) as executions, \
             patch.object(workflow,'production_publication',return_value={'production':'gate'}) as production, \
             patch.object(workflow.io,'identity',return_value={'plan':'ref'}):
            result=workflow.final_preflight('plan','index','global')
            corrections.assert_called_once_with('index');executions.assert_called_once_with('index','global');production.assert_called_once()
            self.assertEqual(result['selectionCorrectionGates'],['all162']);self.assertEqual(result['evaluationPublicationGates'],['all270'])


if __name__ == '__main__': unittest.main()
