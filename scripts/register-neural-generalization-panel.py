#!/usr/bin/env python3
"""Freeze a complete label-blind inference population after feature qualification."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value
from analysis.neural_generalization_inputs import manifest_rows, feature_entries, verified
from analysis.neural_recall_sweep import identity, read, require, write_new

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--features', type=Path, required=True)
p.add_argument('--audit', type=Path, required=True)
p.add_argument('--inventory', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--precision-complete', action='store_true')
a = p.parse_args()
_, rows = manifest_rows(a.manifest)
index = read(a.features); audit = read(a.audit)
require(len(rows) == 42 and all('infer' in r['eligibleRoles'] for r in rows), 'Expected all42 nonbeach inference sources')
require(index['publicationScope'] == 'all' and {r['recordingId'] for r in index['records']} == {r['id'] for r in rows},
        'Partial features cannot define the all-video inference panel')
require(audit['passed'] and audit['kind'] == 'independent-expansion-feature-audit-v1'
        and audit['inputs'] == identity(a.manifest) and audit['features'] == identity(a.features)
        and audit['auditedRecordingCount'] == 42 and audit['protectedAndPanelTeacherTargetsAbsent'],
        'Complete independent feature audit absent')
entries = feature_entries(rows, a.features)
for row in rows:
    values = entries[row['id']]
    for key in ('audiovisual', 'mobile', 'imageInput'):
        verified(values[key])
    require('fp32' in values['dino'], 'FP32 DINO missing')
    for precision in ('fp32', 'fp16', 'int8') if a.precision_complete else ('fp32',):
        verified(values['dino'][precision])
    if row['protected'] or row['sourceGroup'] == private_value('source-group-001') or 'fit' not in row['eligibleRoles']:
        require(not values.get('teacherTargets'), 'Evaluation-only source has teaching targets')
require(sum('evaluate' in r['eligibleRoles'] and r['scoringPolicy'] != 'none' for r in rows) == 34,
        'Scored label population changed')
write_new(a.output, {'kind': 'frozen-independent-inference-panel-v1',
    'manifest': identity(a.manifest), 'features': identity(a.features), 'featureAudit': identity(a.audit),
    'inventory': identity(a.inventory), 'registrar': identity(__file__),
    'precisionVariants': ['fp32', 'fp16', 'int8'] if a.precision_complete else ['fp32'],
    'recordingIds': [r['id'] for r in rows], 'scoredRecordingIds': [r['id'] for r in rows if 'evaluate' in r['eligibleRoles'] and r['scoringPolicy'] != 'none'],
    'commonEvaluationGroups': [private_value('source-group-001'), private_value('source-group-008')],
    'inferenceUsesLabels': False, 'allInferenceTicksValid': True,
    'outsideFittingSelection': 'Common evaluation groups excluded from every fit/student/calibration; all additional groups excluded from original-corpus fitting/selection.',
    'protectedSourcesOpenedForFinalEvaluationOnly': True, 'productionPromotionAllowed': False})
print(str(a.output))
