#!/usr/bin/env python3
"""Project audited precision cells into the common historical report schema."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value
from analysis import neural_recall_sweep as sweep
from analysis import neural_recall_sweep_precision as precision
from analysis.neural_recall_sweep_adapters import bind


def project(root):
    protocol = precision.verify(root)
    audit_ref = sweep.identity(root / 'audit.json')
    audit = sweep.read(audit_ref['path'])
    source_ref = sweep.identity(root / 'report.json')
    sweep.require(audit['passed'] is True and audit['kind'] == 'independent-historical-dino-precision-floor-audit-v1'
                  and audit['report'] == source_ref and audit['protocol'] == sweep.identity(root / 'protocol.json'),
                  'Passing precision report audit required')
    for ref in audit['references']:
        bind(ref)
    report = sweep.read(source_ref['path'])
    original = sweep.read(protocol['historicalProtocol']['path'])
    ids = [r['id'] for r in sweep.read(original['records']['path'])['records']]
    outputs = []
    for arm in ('fp16', 'int8'):
        cells = []
        for source in [cell for cell in report['cells'] if cell['precision'] == arm]:
            cells.append({key: source[key] for key in ('model', 'variant', 'seed', 'selectionDesign', 'precision')} |
                {'precisionSpecificSelection': False, 'panels': {'historical-nested-exact': {
                    'panelId': 'historical-nested-exact', 'labelPolicy': 'exact-rallies', 'reviewStatus': 'manually-reviewed',
                    'sourcePolicy': 'source-held', 'expectedRecordingIds': ids, 'result': source['result']}}})
        sweep.require(len(cells) == 6, 'Precision projection omitted model/seed cells')
        result = {'kind': 'audited-historical-precision-sweep-projection-v1', 'protocol': report['protocol'],
                  'sourceReport': source_ref, 'sourceAudit': audit_ref, 'precision': arm,
                  'code': sweep.identity(Path(__file__)), 'cells': cells,
                  'comparisons': [c for c in report['comparisons'] if c['precision'] == arm],
                  'selectionPolicy': 'FP32-selected epochs/decoders transferred unchanged; precision arms were not recalibrated.',
                  'precisionSpecificSelection': False, 'projectionOnly': True, 'newMetricsComputed': False,
                  'trainingPerformed': False, 'gpuUsed': False, 'protectedTestOpened': False}
        path = root / f'report-{arm}.json'
        sweep.write_new(path, result)
        copied = sweep.read(path)
        sweep.require(copied == result, 'Precision projection serialization differs')
        outputs.append(sweep.identity(path))
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    sweep.require(str(args.study.resolve()).startswith(private_value('private-reference-0060')), 'Projection outputs must use direct NAS')
    print(project(args.study))
