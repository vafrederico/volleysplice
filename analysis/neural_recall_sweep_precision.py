"""CPU-only transfer of historical FP32 recall-floor choices to DINO precision caches."""
from __future__ import annotations
from analysis.private_ledger import private_value

from dataclasses import replace
from pathlib import Path
import numpy as np

from . import neural_recall_sweep as sweep
from . import neural_recall_sweep_historical as historical
from .neural_recall_sweep_adapters import NAS, bind

MODELS = ('dino_tcn_short_boost', 'dino_transformer')
ARMS = ('fp32', 'fp16', 'int8')
PUBLICATION_SHA = '23d278b536c36227a9d7c1537e6eb5840e3bb5a4420aa9042ac0474c2092f73c'
RTOL, ATOL = 1e-4, 2e-5


def prepare(source, stage_plan, output):
    source, output = Path(source), Path(output)
    sweep.require(str(output.resolve()).startswith(private_value('private-reference-0060')) and not (output / 'protocol.json').exists(), 'Need new direct-NAS precision protocol')
    parent = historical.verify(source)
    report_ref = sweep.identity(source / 'report-complete.json')
    report = sweep.read(report_ref['path'])
    gates = [sweep.identity(source / name) for name in ('audit-refits.json', 'audit-complete.json', 'audit-owners.json')]
    for ref in gates:
        gate = sweep.read(ref['path'])
        sweep.require(gate['passed'] is True, 'Historical audit failed')
        if 'report' in gate:
            sweep.require(gate['report'] == report_ref, 'Metric audit belongs to another report')
        if 'protocol' in gate:
            sweep.require(gate['protocol'] == sweep.identity(source / 'protocol.json'), 'Refit audit belongs to another study')
    refit_audit = sweep.read(gates[0]['path'])
    sweep.require(refit_audit['counts'] == {'outerFits': 72, 'checkpoints': 288, 'exactOriginalEpochs': 86, 'newStudents': 2},
                  'Historical refit audit scope incomplete')
    checkpoints = []
    for checked in refit_audit['checks']:
        if checked['task']['model'] not in MODELS:
            continue
        bind(checked['completed']); bind(checked['parity'])
        owner = sweep.read(checked['completed']['path']); folder = Path(checked['completed']['path']).parent
        artifacts = [{'path': str(folder / name), 'sha256': sha} for name, sha in owner['artifacts'].items()]
        for ref in artifacts:
            bind(ref)
        checkpoints.append({'task': checked['task'], 'completed': checked['completed'], 'parity': checked['parity'], 'artifacts': artifacts})
    sweep.require(len(checkpoints) == 24, 'DINO outer checkpoint scope incomplete')
    publication = {'path': str(NAS / '2026-09-23-recall-distillation/dino-precision-v1/publication-v1/manifest.json'), 'sha256': PUBLICATION_SHA}
    published = sweep.read(bind(publication))
    input_ref = next(ref for ref in published['artifacts'] if ref['path'].endswith('/audit-inputs.json'))
    input_audit = sweep.read(bind(input_ref))
    precision_root = NAS / '2026-09-23-recall-distillation/dino-precision-v1'
    control_reports = [ref for ref in published['artifacts'] if Path(ref['path']) in
                       (precision_root / 'report.json', precision_root / 'recall99-v1/report.json')]
    sweep.require(len(control_reports) == 2, 'Prior precision control reports missing')
    for ref in control_reports:
        previous = sweep.read(bind(ref))
        for result in previous['results']:
            bind(result)
    sweep.require(input_audit['passed'] is True and input_audit['allPrecisionTimestampsExactlyOriginal']
                  and input_audit['allCacheShapesDtypesFinite'], 'Precision cache qualification absent')
    checked = {(ref['path'], ref['sha256']) for ref in input_audit['checked']}
    gold = sweep.read(parent['records']['path'])['records']
    by_id = {r['id']: r for r in sweep.read(stage_plan)['records']}
    caches = {}
    for row in gold:
        source_row = by_id[row['id']]
        sweep.require(source_row['sourceGroup'] == row['sourceGroup'] and source_row['sampling'] == 'historical-ordinal', 'Precision source association differs')
        caches[row['id']] = {}
        timestamps = None
        for arm in ARMS:
            ref = source_row['reuse']['dino'][arm]
            sweep.require((ref['path'], ref['sha256']) in checked, 'Cache is absent from passing historical precision audit')
            with np.load(bind(ref), allow_pickle=False) as arrays:
                times, tokens = arrays['timestamps'], arrays['tokens']
                sweep.require(times.ndim == 1 and len(times) and tokens.dtype == np.float16 and tokens.shape == (len(times), 10, 384)
                              and np.isfinite(times).all() and np.isfinite(tokens).all() and np.all(np.diff(times) > 0), 'Precision cache schema changed')
                sweep.require(timestamps is None or np.array_equal(timestamps, times), 'Precision timestamps differ')
                timestamps = times.copy()
            caches[row['id']][arm] = {key: ref[key] for key in ('path', 'sha256')}
    code = {ref['path']: ref for ref in parent['code']}
    for name in ('analysis/neural_recall_sweep_precision.py', 'scripts/run-neural-historical-precision-sweep.py',
                 'scripts/audit-neural-historical-precision-sweep.py', 'scripts/audit-neural-historical-controls.py',
                 'scripts/project-neural-historical-precision-sweep.py'):
        ref = sweep.identity(historical.REPO / name); code[ref['path']] = ref
    cells = [c for c in report['cells'] if c['model'] in MODELS]
    sweep.require(len(cells) == 6 and {(c['model'], c['seed']) for c in cells} == {(m, s) for m in MODELS for s in historical.SEEDS}, 'DINO control model/seed scope differs')
    protocol = {'kind': 'historical-dino-precision-recall-floor-transfer-v1',
        'historicalProtocol': sweep.identity(source / 'protocol.json'), 'historicalReport': report_ref,
        'historicalAudits': gates, 'historicalPlan': sweep.identity(source / 'refit-plan.json'),
        'cachePublication': publication, 'cacheAudit': input_ref, 'historicalPrecisionReports': control_reports, 'stagePlan': sweep.identity(stage_plan),
        'cacheReferences': caches, 'checkpointOwners': checkpoints, 'code': list(code.values()), 'models': list(MODELS), 'seeds': list(historical.SEEDS),
        'arms': list(ARMS), 'epochs': list(sweep.EPOCHS), 'floors': list(sweep.FLOORS),
        'headDevice': 'cpu', 'headThreads': 2, 'scoreParityTolerance': {'rtol': RTOL, 'atol': ATOL},
        'fp32ControlGate': 'Every checkpoint score within original head replay tolerance; every selected FP32 raw interval and full fold evaluation exactly equals canonical GPU control. Fail closed on any mismatch.',
        'selectionPolicy': 'Transfer every FP32 epoch/decoder choice unchanged; no precision-specific calibration, fallback, or ranking on held sources.',
        'historicalInferencePolicy': parent['historicalInferencePolicy'], 'paddingCases': [0, 1, 2, 3], 'targetPaddingSeconds': 2, 'joinGapSeconds': 3,
        'encoderExecuted': False, 'trainingPerformed': False, 'gpuUsed': False, 'protectedTestOpened': False}
    for ref in code.values():
        path = Path(ref['path']); dest = output / 'registered-sources' / path.relative_to(historical.REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as stream:
            stream.write(path.read_bytes())
    sweep.write_new(output / 'protocol.json', protocol)
    return protocol


def verify(output):
    output = Path(output); protocol = sweep.read(output / 'protocol.json')
    refs = [protocol[k] for k in ('historicalProtocol', 'historicalReport', 'historicalPlan', 'cachePublication', 'cacheAudit', 'stagePlan')]
    refs += protocol['historicalAudits'] + protocol['code']
    refs += protocol['historicalPrecisionReports']
    refs += [result for ref in protocol['historicalPrecisionReports'] for result in sweep.read(ref['path'])['results']]
    refs += [ref for row in protocol['cacheReferences'].values() for ref in row.values()]
    refs += [ref for owner in protocol['checkpointOwners'] for ref in (owner['completed'], owner['parity'], *owner['artifacts'])]
    for ref in refs:
        bind(ref)
    for ref in protocol['code']:
        archived = output / 'registered-sources' / Path(ref['path']).relative_to(historical.REPO)
        sweep.require(sweep.identity(archived)['sha256'] == ref['sha256'], 'Precision archived source changed')
    return protocol


def attach(example, reference):
    with np.load(bind(reference), allow_pickle=False) as arrays:
        times, tokens = arrays['timestamps'], arrays['tokens']
        right = np.clip(np.searchsorted(times, example.times), 0, len(times) - 1)
        left = np.maximum(0, right - 1)
        nearest = np.where(abs(times[left] - example.times) <= abs(times[right] - example.times), left, right)
        sweep.require(np.max(abs(times[nearest] - example.times)) <= .125 + 1e-8, 'Precision feature alignment changed')
        values = np.concatenate((example.values, tokens[nearest].reshape(len(example.times), -1)), axis=1).astype(np.float32)
    sweep.require(values.shape == (len(example.times), 3944) and np.array_equal(values[:, :104], example.values), 'AV features changed')
    return replace(example, values=values)


def run(output):
    import torch
    from . import neural_expanded_development as expanded
    from .neural_recognition_fit import predict
    from .recognition_temporal_model import RecognitionConfig, model_for
    from .transfer_temporal_model import model_for as legacy_model
    torch.set_num_threads(2)
    output = Path(output); protocol = verify(output)
    source = Path(protocol['historicalProtocol']['path']).parent
    parent = sweep.read(protocol['historicalProtocol']['path'])
    report = sweep.read(protocol['historicalReport']['path'])
    examples = expanded.base.load_examples(Path(parent['exactManifest']['path']), False)
    cells = []
    for control in [c for c in report['cells'] if c['model'] in MODELS]:
        arms = {arm: [] for arm in ARMS}; fold_checks = []
        for outer, fold in enumerate(control['folds']):
            print(f"PRECISION_SWEEP {control['model']} seed={control['seed']} outer={outer}", flush=True)
            folder = source / 'refits' / control['model'] / str(control['seed']) / f'outer-{outer}'
            done = sweep.read(folder / 'temporal/completed.json')
            parity = sweep.read(folder / 'parity.json')
            sweep.require(parity['passed'] is True and parity['plan'] == protocol['historicalPlan']
                          and parity['completed'] == sweep.identity(folder / 'temporal/completed.json')
                          and done['seed'] == control['seed'] and done['validationGroups'] == [fold['foldId']], 'Wrong historical checkpoint owner')
            held = [e for e in examples if e.group == fold['foldId']]
            dest = output / 'scores' / control['model'] / str(control['seed']) / f'outer-{outer}'
            complete = dest / 'completed.json'
            references, score_sets = [], {arm: {} for arm in ARMS}
            if complete.exists():
                receipt = sweep.read(complete)
                sweep.require(receipt['protocol'] == sweep.identity(output / 'protocol.json')
                              and receipt['owner'] == sweep.identity(folder / 'temporal/completed.json'), 'Precision replay resume changed')
                for row in receipt['scores']:
                    sweep.require(row['arm'] in ARMS and row['epoch'] in sweep.EPOCHS
                                  and row['weights'] == {'path': str(folder / f"temporal/weights-{row['epoch']}.npz"),
                                      'sha256': done['artifacts'][f"weights-{row['epoch']}.npz"]}, 'Resumed precision checkpoint lineage differs')
                    path = bind(row['archive'])
                    with np.load(path, allow_pickle=False) as arrays:
                        score_sets[row['arm']][row['epoch']] = {key: arrays[key].copy() for key in arrays.files}
                references = receipt['scores']
                sweep.require(len(references) == 12 and {(r['arm'], r['epoch']) for r in references}
                              == {(a, e) for a in ARMS for e in sweep.EPOCHS}, 'Resumed precision archive inventory differs')
            else:
                config = RecognitionConfig(family='dino', head='transformer')
                legacy = control['model'] == 'dino_tcn_short_boost'
                with torch.random.fork_rng(devices=[]):
                    model = (legacy_model('dino_tcn') if legacy else model_for(config)).cpu().eval()
                for arm in ARMS:
                    attached = [attach(e, protocol['cacheReferences'][e.id][arm]) for e in held]
                    for epoch in sweep.EPOCHS:
                        weight = {'path': str(folder / f'temporal/weights-{epoch}.npz'), 'sha256': done['artifacts'][f'weights-{epoch}.npz']}
                        with np.load(bind(weight), allow_pickle=False) as weights:
                            model.load_state_dict({k: torch.from_numpy(weights['model::' + k].copy()) for k in model.state_dict()}, strict=True)
                            mean, scale = weights['mean'].copy(), weights['scale'].copy()
                        values = {e.id: (expanded.predict(model, e, mean, scale, 'dino_tcn', 'cpu') if legacy
                                        else predict(model, e, mean, scale, config, 'cpu')) for e in attached}
                        sweep.validate_scores(held, {epoch: values}, require_all_epochs=False)
                        if arm == 'fp32':
                            saved_ref = {'path': str(folder / f'temporal/predictions-{epoch}.npz'), 'sha256': done['artifacts'][f'predictions-{epoch}.npz']}
                            with np.load(bind(saved_ref), allow_pickle=False) as saved:
                                sweep.require(all(np.allclose(values[e.id], saved[e.id], rtol=RTOL, atol=ATOL) for e in held), 'FP32 CPU/GPU score control failed')
                        target = dest / arm / f'predictions-{epoch}.npz'
                        target.parent.mkdir(parents=True, exist_ok=True)
                        sweep.require(not target.exists(), 'Incomplete precision score artifact; investigate before resume')
                        np.savez_compressed(target, **values)
                        references.append({'arm': arm, 'epoch': epoch, 'archive': sweep.identity(target), 'weights': weight})
                        score_sets[arm][epoch] = values
                receipt = {'protocol': sweep.identity(output / 'protocol.json'),
                           'owner': sweep.identity(folder / 'temporal/completed.json'), 'scores': references,
                           'fp32AllEpochScoresWithinTolerance': True}
                sweep.write_new(complete, receipt)
            for arm in ARMS:
                measured = sweep.evaluate_selected(held, score_sets[arm], fold['decisions'])
                if arm == 'fp32':
                    sweep.require(measured == fold['panels']['historical-nested-exact'], 'FP32 CPU raw intervals/full fold evaluation differs from canonical GPU control')
                arms[arm].append(measured)
            fold_checks.append({'foldId': fold['foldId'], 'scores': sweep.identity(complete), 'fp32CanonicalIntervalParity': True})
        for arm in ARMS:
            gold = arms[arm][0]['expectedGold'].copy()
            for fold in arms[arm][1:]:
                gold.update(fold['expectedGold'])
            pooled = sweep.pool_fold_results(arms[arm], gold)
            cells.append({'model': control['model'], 'variant': 'historical-original18-' + arm, 'seed': control['seed'],
                          'selectionDesign': 'transferred-nested-original-four-source', 'panelId': 'historical-nested-exact',
                          'precision': arm, 'folds': arms[arm], 'foldChecks': fold_checks, 'result': pooled})
    comparisons = []
    for model in MODELS:
        for arm in ARMS:
            selected = [c for c in cells if c['model'] == model and c['precision'] == arm]
            comparisons.append({'model': model, 'precision': arm, 'summary': sweep.summarize_seed_cells(selected, historical.SEEDS)})
    verify(output)
    result = {'kind': 'historical-dino-precision-recall-floor-results-v1', 'protocol': sweep.identity(output / 'protocol.json'),
              'cells': cells, 'comparisons': comparisons, 'fp32FullControlPassed': True, 'precisionSpecificSelection': False,
              'trainingPerformed': False, 'gpuUsed': False, 'encoderExecuted': False, 'protectedTestOpened': False}
    sweep.write_new(output / 'report.json', result)
    return result
