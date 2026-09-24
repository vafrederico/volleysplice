#!/usr/bin/env python3
"""Register a separately chosen context study after its audited reference exists."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_short_boost_transfer as frozen
from analysis.short_context_temporal_model import model_metadata

SPEC = importlib.util.spec_from_file_location('context_preflight_registration', REPO/'scripts/audit-neural-context-preflight.py')
preflight_tools = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight_tools)
require, read, identity, write_new = (getattr(preflight_tools, name) for name in ('require', 'read', 'identity', 'write_new'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--cohort', choices=frozen.COHORTS, required=True)
    parser.add_argument('--arm', choices=frozen.ARMS, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--tensor-audit', type=Path, required=True)
    parser.add_argument('--interval-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reg, report, summary = [read(args.reference/name) for name in ('preregistration.json', 'report.json', 'summary.json')]
    old, preflight = reg['contract'], read(args.preflight)
    require(frozen.canonical_hash(old) == reg['sha256']
            and report['status'] == 'completed-short-boost-transfer-development'
            and report['contractSha256'] == reg['sha256'] and len(report['results']) == 54
            and summary['status'] == 'completed-short-boost-transfer-audit'
            and summary['contractSha256'] == reg['sha256']
            and report['protectedTestOpened'] is False and report['productionPromotionAllowed'] is False,
            'Preceding study must be complete and audited')
    code = {**old['code'], **{name: frozen.base.file_sha256(REPO/'analysis'/name) for name in preflight_tools.NEW_SOURCES}}
    require(all(frozen.base.file_sha256(REPO/'analysis'/name) == sha for name, sha in code.items()), 'Frozen source changed')
    manifest = Path(old['dinoManifest']['path']).parent/'manifest-pts-v1.json'
    manifest_identity, dino_identity = identity(manifest), identity(Path(old['dinoManifest']['path']))
    require(manifest_identity['sha256'] == old['manifestSha256'] and dino_identity == old['dinoManifest'], 'Data identity changed')
    require(preflight.get('passed') is True and preflight.get('ownerReuseQualified') is True
            and preflight.get('originalProfileReplayQualified') is True
            and preflight.get('kind') == 'context-engineering-preflight-v1'
            and preflight.get('environment') == old['environment']
            and preflight['code'] == code and preflight['manifest'] == manifest_identity
            and preflight['dinoManifest'] == dino_identity and preflight['referenceContractSha256'] == reg['sha256']
            and preflight['cohort'] == args.cohort and preflight['lossArm'] == args.arm,
            'Context preflight does not qualify this recipe')
    report_identity = identity(args.reference/'report.json')
    require(args.tensor_audit.resolve() != args.interval_audit.resolve(), 'Distinct independent audits are required')
    audits = [preflight_tools.verify_reference_audit(path, args.reference/'report.json', kind, reg['sha256'])
              for path, kind in zip((args.tensor_audit, args.interval_audit), preflight_tools.AUDIT_KINDS)]
    require(preflight['referenceAudits'] == audits and preflight['referenceReport'] == report_identity,
            'Preflight reference changed')
    require(args.protocol.is_file() and args.protocol.stat().st_size > 200, 'A concrete prospective protocol is required')
    contract = copy.deepcopy(old)
    contract.update({
        'experiment': 'short-context-development-v1', 'cohort': args.cohort, 'cohorts': [args.cohort],
        'lossArm': args.arm, 'lossArms': [args.arm], 'contexts': ['original', 'short'],
        'manifest': manifest_identity, 'manifestSha256': manifest_identity['sha256'], 'dinoManifest': dino_identity,
        'models': {kind: model_metadata(kind, context='short') for kind in old['kinds']},
        'contextModels': {context: {kind: model_metadata(kind, context=context) for kind in old['kinds']}
                          for context in ('original', 'short')},
        'code': code, 'preflight': identity(args.preflight), 'protocolSnapshot': identity(args.protocol),
        'registrationTool': identity(Path(__file__)),
        'referenceStudy': {'path': str(args.reference), 'contractSha256': reg['sha256'],
                           'preregistrationFileSha256': identity(args.reference/'preregistration.json')['sha256'],
                           'reportSha256': report_identity['sha256'],
                           'summarySha256': identity(args.reference/'summary.json')['sha256'],
                           'independentAudits': audits},
        'fitReuse': {'unorderedInnerExclusions': True, 'physicalInnerOwnersPerCell': 6,
                     'logicalInnerViewsPerCell': 12, 'outerRefitsPerCell': 4,
                     'trainingHaloTicks': 62, 'trainingCoreTicks': 128},
        'contextIntervention': {'originalDilations': [1, 2, 4, 8, 16], 'shortDilations': [1, 1, 2, 2, 2],
                                'kernelSize': 5, 'originalReceptiveFieldTicks': 125, 'shortReceptiveFieldTicks': 33,
                                'parameterCountChanges': 0,
                                'unchanged': 'Inputs, supervision, loss arm, parameter initialization and shapes, '
                                             'training chunks/62 halo, sampling, optimizer, checkpoint/decoder selection.',
                                'scope': 'Temporal network only; existing audiovisual feature windows unchanged'},
        'transferScreen': {'withinArchitectureComparison': 'short-minus-original',
                           'architectures': old['kinds'], 'sameCohortAndLossArmRequired': True,
                           'standardF1Replication': 'Both architecture standardF1Screen.screenPassedAndInnerFeasible',
                           'retentionReplication': 'Both architecture retentionRecoveryScreen.screenPassedAndInnerFeasible',
                           'differenceInDifferences': 'Descriptive; positive interaction is not required',
                           'promotionAllowed': False},
    })
    registration = {'sha256': frozen.canonical_hash(contract), 'contract': contract}
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output/'preregistration.json'
    require(not destination.exists(), 'Refusing to overwrite a context registration')
    sources = args.output/'registered-sources'
    sources.mkdir(exist_ok=False)
    for name, sha in code.items():
        data = (REPO/'analysis'/name).read_bytes()
        require(frozen.base.file_sha256(REPO/'analysis'/name) == sha, 'Source changed during registration')
        with (sources/name).open('xb') as handle:
            handle.write(data)
    write_new(destination, registration)
    print(json.dumps({'contractSha256': registration['sha256'], 'registration': identity(destination),
                      'freshPhysicalFits': 60, 'logicalFreshFits': 96, 'referenceCells': 6}, indent=2))


if __name__ == '__main__':
    main()
