#!/usr/bin/env python3
"""Bind production's already-qualified AV index to the identical frozen population."""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_recall_sweep as io
from analysis.neural_generalization_inputs import verified


def register(core_panel, features, feature_audit, production_protocol, output):
    io.require(output.resolve().is_relative_to('/mnt/freenas') and not output.exists(), 'New NAS panel required')
    core = io.read(core_panel); audit = io.read(feature_audit); production = io.read(production_protocol)
    for key in ('manifest', 'inventory', 'features', 'featureAudit', 'registrar'): verified(core[key])
    manifest = io.read(core['manifest']['path']); index = io.read(features)
    io.require(core['kind'] == 'frozen-independent-inference-panel-v1' and core['precisionVariants'] == ['fp32'], 'Frozen FP32 panel required')
    io.require(index['publicationScope'] == 'all' and index['publicationMode'] == 'av'
        and index['labelsUsedForFeatures'] is False and index['protectedImageryUsedForTraining'] is False,
        'Complete blind AV feature index required')
    io.require(production['manifest'] == core['manifest'] and production['features'] == io.identity(features)
        and production['labelsUsedForInference'] is False, 'Production protocol AV/source binding differs')
    ids = [r['id'] for r in manifest['records']]
    io.require(len(ids) == len(set(ids)) == 42 and ids == core['recordingIds']
        and [r['recordingId'] for r in index['records']] == ids, 'Same ordered all42 source population required')
    io.require(audit['kind'] == 'independent-expansion-feature-audit-v1' and audit['passed'] is True
        and audit['mode'] == 'av' and audit['scope'] == 'all' and audit['inputs'] == core['manifest']
        and audit['features'] == io.identity(features) and audit['auditedRecordingCount'] == 42
        and audit['protectedAndPanelTeacherTargetsAbsent'] is True and audit['inferenceMaskAlwaysAllTrue'] is True
        and audit['source'] == io.identity(REPO/'scripts/audit-neural-generalization-features.py'), 'Current all42 AV audit required')
    panel = {**core, 'features': io.identity(features), 'featureAudit': io.identity(feature_audit),
        'registrar': io.identity(__file__), 'modelFamily': 'av',
        'continuityReferencePanel': io.identity(core_panel), 'productionProtocol': io.identity(production_protocol),
        'interpretation': 'Same source/gold/order and exact AV cache bytes as the neural FP32 core panel; visual embeddings are outside this production AV contract.'}
    io.write_new(output, panel)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('core-panel', 'features', 'feature-audit', 'production-protocol', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args(); register(a.core_panel, a.features, a.feature_audit, a.production_protocol, a.output)
    print(io.identity(a.output), flush=True)
