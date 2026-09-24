#!/usr/bin/env python3
"""Register a separately chosen keep-output decoder comparison; never fit models."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import platform
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_keep_rescue_development as runner
from analysis import neural_keep_rescue as rescue

SPEC = importlib.util.spec_from_file_location('keep_rescue_registration_preflight', REPO/'scripts/audit-neural-keep-rescue-preflight.py')
preflight_tools = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight_tools)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--cohorts', nargs='+', choices=runner.COHORTS, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--tensor-audit', type=Path, required=True)
    parser.add_argument('--interval-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runner.require(args.cohorts == [c for c in runner.COHORTS if c in args.cohorts], 'Cohorts must be unique and in canonical order')
    reference = preflight_tools.reference_binding(args.reference, args.tensor_audit, args.interval_audit)
    reg, _, _ = runner.completed_reference(reference)
    old, preflight = reg['contract'], runner.read(args.preflight)
    code = {**old['code'], **{name: runner.digest(REPO/'analysis'/name) for name in runner.NEW_SOURCES}}
    runner.require(all(runner.digest(REPO/'analysis'/name) == sha for name, sha in code.items()), 'Frozen source changed')
    manifest = runner.identity(Path(old['dinoManifest']['path']).parent/'manifest-pts-v1.json')
    runner.require(manifest['sha256'] == old['manifestSha256'], 'Data manifest changed')
    execution_environment = {'python': platform.python_version(), 'numpy': np.__version__, 'device': 'cpu'}
    runner.require(preflight.get('kind') == 'keep-rescue-engineering-preflight-v1'
        and preflight.get('executionEnvironment') == execution_environment
        and preflight.get('passed') is True and preflight.get('noOpReplayQualified') is True
        and preflight.get('selectionIsolationQualified') is True and preflight['code'] == code
        and preflight['manifest'] == manifest and preflight['cohorts'] == args.cohorts
        and preflight['referenceContractSha256'] == reg['sha256']
        and preflight['referenceReport'] == runner.identity(args.reference/'report.json')
        and preflight['referenceAudits'] == reference['independentAudits'], 'Keep-rescue preflight does not qualify this recipe')
    runner.require(args.protocol.is_file() and args.protocol.stat().st_size > 200, 'Concrete prospective protocol required')
    contract = copy.deepcopy(old)
    contract.update({'experiment': runner.EXPERIMENT, 'cohorts': args.cohorts, 'kinds': list(runner.KINDS),
        'variants': list(runner.VARIANTS), 'baselineLossArm': 'baseline', 'lossArms': ['baseline'],
        'rescueOptions': list(rescue.KEEP_RESCUE_OPTIONS), 'rescueRecipe': rescue.rescue_metadata(),
        'rescueSelection': runner.RESCUE_SELECTION, 'referenceStudy': reference,
        'manifest': manifest, 'preflight': runner.identity(args.preflight), 'code': code,
        'protocolSnapshot': runner.identity(args.protocol), 'registrationTool': runner.identity(Path(__file__)),
        'executionEnvironment': execution_environment,
        'modelTraining': 'None; all model checkpoints and baseline epoch/live-boundary decoders are immutable references.',
        'transferScreen': {'withinArchitectureComparison': 'selected-minus-reference',
            'architectures': list(runner.KINDS), 'sameCohortRequired': True,
            'standardF1Replication': 'Both architecture standardF1Screen.screenPassedAndInnerFeasible',
            'retentionReplication': 'Both architecture retentionRecoveryScreen.screenPassedAndInnerFeasible',
            'differenceInDifferences': 'Descriptive; positive interaction is not required', 'promotionAllowed': False}})
    registration = {'sha256': runner.canonical_hash(contract), 'contract': contract}
    args.output.mkdir(parents=True, exist_ok=True)
    runner.require(not (args.output/'preregistration.json').exists(), 'Refusing to overwrite a rescue registration')
    sources = args.output/'registered-sources'
    sources.mkdir(exist_ok=False)
    for name, sha in code.items():
        data = (REPO/'analysis'/name).read_bytes()
        runner.require(runner.digest(REPO/'analysis'/name) == sha, 'Source changed during registration')
        with (sources/name).open('xb') as handle:
            handle.write(data)
    runner.write_immutable(args.output/'preregistration.json', registration)
    runner.validate_registration(args.output/'preregistration.json')
    print(json.dumps({'contractSha256': registration['sha256'], 'registration': runner.identity(args.output/'preregistration.json'),
                      'newFits': 0, 'referenceCells': 6*len(args.cohorts), 'candidateCells': 6*len(args.cohorts)}, indent=2))


if __name__ == '__main__':
    main()
