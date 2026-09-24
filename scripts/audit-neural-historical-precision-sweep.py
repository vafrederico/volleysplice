#!/usr/bin/env python3
"""Independent interval, provenance and sampled-head audit of DINO floor transfer."""
from pathlib import Path
import argparse
import statistics
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
from analysis import neural_recall_sweep as sweep
from analysis import neural_recall_sweep_historical as historical
from analysis import neural_recall_sweep_precision as precision
from analysis.neural_recall_sweep_adapters import bind


def audit(root):
    import torch
    from analysis import neural_expanded_development as expanded
    from analysis.recognition_temporal_model import RecognitionConfig, model_for
    from analysis.transfer_temporal_model import model_for as legacy_model
    torch.set_num_threads(2)
    protocol = precision.verify(root)
    report_ref = sweep.identity(root / 'report.json'); report = sweep.read(report_ref['path'])
    sweep.require(report['protocol'] == sweep.identity(root / 'protocol.json') and report['fp32FullControlPassed'] is True
                  and report['precisionSpecificSelection'] is False and report['gpuUsed'] is False, 'Precision result contract differs')
    controls = sweep.read(protocol['historicalReport']['path'])
    parents = {(c['model'], c['seed']): c for c in controls['cells'] if c['model'] in precision.MODELS}
    expected = {(m, s, a) for m in precision.MODELS for s in historical.SEEDS for a in precision.ARMS}
    cells = {(c['model'], c['seed'], c['precision']): c for c in report['cells']}
    sweep.require(set(cells) == expected and len(cells) == len(report['cells']), 'Missing/duplicate precision result cell')
    parent_protocol = sweep.read(protocol['historicalProtocol']['path'])
    examples = expanded.base.load_examples(Path(parent_protocol['exactManifest']['path']), False)
    metric = historical.load_script('audit-neural-recall-sweep.py')
    helper_path = REPO / 'scripts/audit-neural-short-boost-intervals.py'
    sweep.require(sweep.identity(helper_path)['sha256'] == metric.HELPER_SHA, 'Independent endpoint helper changed')
    helper = metric.module('precision_interval_auditor', helper_path)
    replay = historical.load_script('audit-neural-recognition.py')
    control_audit = historical.load_script('audit-neural-historical-controls.py')
    references, checks, point_count = {}, [], 0
    source = Path(protocol['historicalProtocol']['path']).parent
    for (name, seed), control in parents.items():
        for outer, fold in enumerate(control['folds']):
            held = [e for e in examples if e.group == fold['foldId']]
            folder = source / 'refits' / name / str(seed) / f'outer-{outer}' / 'temporal'
            meta_ref = sweep.identity(folder / 'completed.json'); meta = sweep.read(meta_ref['path'])
            score_folder = root / 'scores' / name / str(seed) / f'outer-{outer}'
            receipt_ref = sweep.identity(score_folder / 'completed.json')
            receipt = sweep.read(bind(receipt_ref, references))
            sweep.require(receipt['protocol'] == report['protocol'] and receipt['owner'] == meta_ref
                          and receipt['fp32AllEpochScoresWithinTolerance'] is True, 'Precision score owner differs')
            sweep.require(len(receipt['scores']) == 12 and {(r['arm'], r['epoch']) for r in receipt['scores']}
                          == {(a, e) for a in precision.ARMS for e in sweep.EPOCHS}, 'Precision checkpoint inventory differs')
            config = RecognitionConfig(family='dino', head='transformer')
            legacy = name == 'dino_tcn_short_boost'
            with torch.random.fork_rng(devices=[]):
                model = (legacy_model('dino_tcn') if legacy else model_for(config)).cpu().eval()
            for arm in precision.ARMS:
                attached = {e.id: precision.attach(e, protocol['cacheReferences'][e.id][arm]) for e in held}
                scores, samples, maximum = {}, [], 0.
                for epoch in sweep.EPOCHS:
                    record = next(r for r in receipt['scores'] if (r['arm'], r['epoch']) == (arm, epoch))
                    expected_weight = {'path': str(folder / f'weights-{epoch}.npz'), 'sha256': meta['artifacts'][f'weights-{epoch}.npz']}
                    sweep.require(record['weights'] == expected_weight, 'Precision scores refer to another checkpoint')
                    with np.load(bind(record['archive'], references), allow_pickle=False) as archive:
                        scores[epoch] = {key: archive[key].copy() for key in archive.files}
                    sweep.validate_scores(held, {epoch: scores[epoch]}, require_all_epochs=False)
                    sweep.require(all(np.all(scores[epoch][e.id][~e.valid] == 0) and scores[epoch][e.id].dtype == np.float32 for e in held),
                                  'Historical invalid tick/dtype contract changed')
                    with np.load(bind(expected_weight, references), allow_pickle=False) as weight:
                        model.load_state_dict({key: torch.from_numpy(weight['model::' + key].copy()) for key in model.state_dict()}, strict=True)
                        mean, scale = weight['mean'].copy(), weight['scale'].copy()
                    for e in held:
                        samples.append({'epoch': epoch, **replay.replay_sample(model, attached[e.id], scores[epoch][e.id], mean, scale, config)})
                    if arm == 'fp32':
                        reference = {'path': str(folder / f'predictions-{epoch}.npz'), 'sha256': meta['artifacts'][f'predictions-{epoch}.npz']}
                        with np.load(bind(reference, references), allow_pickle=False) as archive:
                            for e in held:
                                difference = np.abs(archive[e.id] - scores[epoch][e.id])
                                maximum = max(maximum, float(difference.max()))
                                sweep.require(np.allclose(archive[e.id], scores[epoch][e.id], rtol=precision.RTOL, atol=precision.ATOL),
                                              'FP32 numerical control failed')
                cell = cells[name, seed, arm]
                measured = cell['folds'][outer]
                sweep.require([r['selection'] for r in measured['floors']] == fold['decisions'], 'Precision changed FP32 choices')
                for key, point in measured['operatingPoints'].items():
                    sweep.require(key == sweep.operating_point_key(point), 'Precision point identity differs')
                    rows = sweep.serial_rows([e.row(expanded.base.decode(e, scores[point['epoch']][e.id], point['decoder'])) for e in held])
                    sweep.require(rows == point['predictions'], 'Precision decoded intervals changed')
                    metric.check_evaluation(helper, rows, point['evaluation'], 'exact-rallies')
                    point_count += 1
                for point, decision in zip(measured['floors'], fold['decisions'], strict=True):
                    chosen = decision['selected']
                    sweep.require(point['status'] == ('available' if chosen else 'infeasible-inner-recall')
                                  and point['operatingPointKey'] == (sweep.operating_point_key(chosen) if chosen else None),
                                  'Precision fell back or altered feasibility')
                if arm == 'fp32':
                    sweep.require(measured == fold['panels']['historical-nested-exact'], 'FP32 full fold does not equal canonical control')
                checks.append({'model': name, 'seed': seed, 'outer': outer, 'precision': arm, 'receipt': receipt_ref,
                               'sampledHeadReplay': samples, 'fp32MaximumScoreDifference': maximum if arm == 'fp32' else None})
        for arm in precision.ARMS:
            cell = cells[name, seed, arm]
            result = cell['result']
            sweep.require(len(result['floors']) == 11, 'Pooled precision floor inventory differs')
            for index, pooled in enumerate(result['floors']):
                rows, statuses = [], []
                for fold in cell['folds']:
                    choice = fold['floors'][index]; statuses.append(choice['status'])
                    if choice['status'] == 'available':
                        rows.extend(fold['operatingPoints'][choice['operatingPointKey']]['predictions'])
                complete = all(status == 'available' for status in statuses)
                sweep.require(pooled['floorPercent'] == 90 + index and pooled['foldStatuses'] == statuses
                              and pooled['predictions'] == rows and pooled['completeEvaluationScope'] == complete
                              and pooled['partialScopeNotRankable'] == (not complete), 'Precision pooled scope differs')
                sweep.require((pooled['evaluation'] is not None) == complete, 'Precision incomplete scope received complete metrics')
                if complete:
                    sweep.require({r['id'] for r in rows} == {e.id for e in examples}, 'Complete precision scope lacks an original recording')
                    metric.check_evaluation(helper, rows, pooled['evaluation'], 'exact-rallies')
            if arm == 'fp32':
                sweep.require(result == control['panels']['historical-nested-exact']['result'], 'FP32 pooled result differs from historical control')
            if name == 'dino_tcn_short_boost':
                previous_root = precision.NAS / '2026-09-23-recall-distillation/dino-precision-v1'
                old95 = sweep.read(previous_root / f'result-{arm}-{seed}.json')
                sweep.require(control_audit.by_recording(result['floors'][5]['predictions']) == control_audit.by_recording(old95['predictions']),
                              'Old95 precision raw intervals changed')
                control_audit.compare_metrics(result['floors'][5]['evaluation'], old95['evaluation'])
                old99 = sweep.read(previous_root / f'recall99-v1/result-{seed}.json')
                old_rows = [row for fold in old99['folds'] if fold['feasible'] for row in fold['arms'][arm]['predictions']]
                sweep.require(control_audit.by_recording(result['floors'][9]['predictions']) == control_audit.by_recording(old_rows),
                              'Old99 precision raw intervals changed')
                control_audit.compare_metrics(result['floors'][9]['evaluation'], old99['arms'][arm]['fullEightRecordingEvaluation'])
    sweep.require(len(report['comparisons']) == 6, 'Precision comparison inventory differs')
    for comparison in report['comparisons']:
        selected = [cells[comparison['model'], seed, comparison['precision']] for seed in historical.SEEDS]
        for index, row in enumerate(comparison['summary']['floors']):
            scopes = [cell['result']['floors'][index] for cell in selected]
            complete = all(s['completeEvaluationScope'] for s in scopes)
            sweep.require(row['allRegisteredSeedsComplete'] == complete and (row['meanPadding'] is not None) == complete
                          and row['completeSeeds'] == [seed for seed, scope in zip(historical.SEEDS, scopes) if scope['completeEvaluationScope']],
                          'Precision seed mean hides incomplete scope')
            if complete:
                for padding, pad in enumerate(row['meanPadding']):
                    for key in set(pad) - {'paddingSecondsBeforeAndAfter', 'joinGapSeconds'}:
                        helper.close(pad[key], statistics.mean(s['evaluation']['padding'][padding][key] for s in scopes), 'precision mean/' + key)
    for reference in references.values():
        bind(reference)
    precision.verify(root)
    return {'kind': 'independent-historical-dino-precision-floor-audit-v1', 'passed': True,
            'report': report_ref, 'protocol': report['protocol'], 'checks': checks,
            'counts': {'modelSeedPrecisionCells': len(cells), 'foldArmChecks': len(checks), 'checkpoints': len(checks) * 4,
                       'distinctDecodedPoints': point_count}, 'references': list(references.values()),
            'auditor': sweep.identity(Path(__file__)), 'independentIntervalArithmetic': sweep.identity(helper_path),
            'precisionSpecificSelection': False, 'trainingPerformed': False, 'gpuUsed': False,
            'scope': 'All raw intervals, endpoint metrics, scopes and means; all-epoch FP32 score/control parity; sampled real-context CPU head replay for every arm/checkpoint. Frozen prior encoder cache audit reused, no encoder or training replay.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sweep.require(str(args.output.resolve()).startswith(private_value('private-reference-0060')) and not args.output.exists(), 'Audit needs new NAS output')
    sweep.write_new(args.output, audit(args.study))
