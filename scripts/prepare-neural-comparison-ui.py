"""Publish frozen predictions; user-requested common-unseen selection, no training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.neural_development import decode
from analysis.neural_recall_sweep import SweepExample


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False))


def choose(root, report, model, precision, mode='f1', variant_filter=None):
    # User-requested UI selection. Common-unseen is now development data.
    for floor in (99, 98):
        candidates = [r for r in report['series'] if r['model']==model and r['precision']==precision
            and isinstance(r['draw'], int) and r['panelId']=='common-unseen'
            and r['labelPolicy']=='exact-rallies' and r['productionFilter']=='all'
            and (variant_filter is None or r['variant']==variant_filter)
            and r['paddingSeconds']==2 and r['floorPercent']==floor and r['status']=='available']
        if candidates:
            best = sorted(candidates, key=lambda r: (
                -r['recallValue'] if mode=='recall' else -r['f1Value'],
                -r['f1Value'], r['variant'], r['draw']))[0]
            variant, seed = best['variant'], best['draw']
            fit = (root / 'original-corpus-v1/fits' / model / f'seed-{seed}' if variant=='original-corpus'
                else root / ('export-proxy-v1' if variant.startswith('export-') else 'randomized-variants-v1')
                / 'fits' / variant / model / f'split-{seed}')
            row = next(r for r in read(fit / 'selection.json')['floors'] if r['floorPercent']==floor)
            assert row['feasible']
            return fit, seed, floor, dict(**row['selected'], variant=variant,
                selectionPanel='common-unseen/exact-rallies/all', selectionF1=best['f1Value'],
                selectionRecall=best['recallValue'], selectionPrecision=best['precisionValue'],
                selectionMode=mode, commonUnseenUsedForUiSelection=True)
    raise ValueError(f'No 99% or 98% calibrated option: {model}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    args = parser.parse_args()
    inventory = read(args.study / 'inventory-v1/inventory-v2.json')['records']
    report = read(args.study / 'final-report-v1/report.json')
    models = [('dino-transformer', 'int8', 'INT8 DINO-transformer'),
              ('dino-transformer', 'fp32', 'FP32 DINO-transformer'),
              ('dino-tcn', 'fp32', 'DINO-TCN'), ('mobile-tcn', 'fp32', 'Mobile-TCN'),
              ('distilled-mobile-tcn', 'fp32', 'Distilled Mobile-TCN')]
    selected = []
    sensitivity = {}
    choices = []
    for model, precision, label in models:
        first = choose(args.study, report, model, precision)
        choices.append((model, precision, label, 'f1', first))
        choices.append((model, precision, label, 'recall', choose(args.study, report, model, precision,
            mode='recall', variant_filter=first[3]['variant'])))
    for model, precision, label, mode, choice in choices:
        fit, seed, floor, setting = choice
        model_id = f'neural-{model}-{precision}' + ('-high-recall' if mode=='recall' else '')
        evaluation = read(fit / f'evaluation-{precision}.json')
        entry = next(r for r in evaluation['floors'] if r['floorPercent'] == floor)
        assert entry['status'] == 'available'
        op = evaluation['operatingPoints'][entry['operatingPointKey']]
        sensitivity[model_id] = [p for p in op['panels']
            if p['panelId']=='common-unseen' and p['productionFilter']=='all']
        expected = {r['id']: r['predictions'] for rows in op['rowsByPolicy'].values() for r in rows}
        selected.append((fit, expected, dict(modelId=model_id, model=model,
            precision=precision, modelLabel=f'{label} · highest {"recall" if mode=="recall" else "F1"} · target {floor}%', seed=seed,
            recallTargetPercent=floor, **setting)))
    del report
    index = dict(schemaVersion=1, kind='volleycut-neural-comparison-index',
                 models=[s[2] for s in selected], recordings=[])
    verified = 0
    for row in inventory:
        if row['environment'] == 'beach':
            continue
        references = []
        previous_path = args.output / 'recordings' / f"{row['id']}.json"
        previous = read(previous_path)['recordings'][0]['references'] if previous_path.exists() else []
        previous_revision = hashlib.sha256(previous_path.read_bytes()).hexdigest() if previous else None
        for fit, expected, model in selected:
            folder = 'c32b91f984ae76a4' if model['precision'] == 'int8' else 'be60564b4090c9b8'
            source = fit / 'inference' / folder / model['precision'] / f"{row['id']}.npz"
            if not source.exists():
                matches = list((fit / 'inference').glob(f"*/{model['precision']}/{row['id']}.npz"))
                assert len(matches)==1, (fit, row['id'], matches)
                source = matches[0]
            receipt = read(source.with_suffix('.json'))
            assert receipt['labelsUsed'] is False
            with np.load(source) as data:
                times = data['times']
                scores = data[f"epoch_{model['epoch']}"]
            assert scores.shape == (len(times), 4) and np.isfinite(scores).all()
            example = SweepExample(row['id'], row['sourceGroup'], row['durationSeconds'],
                                   times, np.ones(len(times), dtype=bool), (), ())
            rallies = [dict(start=r.start, end=r.end) for r in decode(example, scores, model['decoder'])]
            if row['id'] in expected:
                assert rallies == expected[row['id']], (row['id'], model['modelId'])
                verified += 1
            description = (f"Frozen {model['precision'].upper()} embeddings with FP32 temporal head, {model['variant']} draw {model['seed']}, "
                           f"epoch {model['epoch']}. Calibration target {model['recallTargetPercent']}% retained-play recall "
                           "at 2s padding; not a guarantee for this video. Core boundaries remain separate; export padding joins gaps under 3s. "
                           f"Selected for highest common-unseen exact-label {'recall within this variant (F1 breaks ties)' if model['selectionMode']=='recall' else 'F1_padP_coreR among available fits'}; this panel is now selection data. "
                           "Serve/start is a timing signal, not a serving-side prediction.")
            references.append(dict(modelId=model['modelId'], modelLabel=model['modelLabel'], description=description,
                rallies=rallies, exportRallies=rallies, exportPolicy='model-predictions', research=dict(
                    recommendation=description,
                    signals=dict(times=times.tolist(), **{head:scores[:,i].tolist() for i,head in enumerate(['live','serve','end','keep'])}),
                    boundaryFlags=[], reviewRegions=[], queue=dict(budgetFraction=0,reviewSeconds=0,selectedParentCount=0),
                    provenance=dict(**model, labelsUsed=False, source=str(source),
                                    sourceSha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                    productionExposure=row.get('productionExposure'), labelTier=row['tier']))))
            old = next((r for r in previous if r['modelId']==model['modelId']), None)
            if old and old['rallies']==rallies and old['research']['signals']==references[-1]['research']['signals']:
                references[-1]['research']['provenance']['uiDraftRevision'] = old['research'].get('provenance',{}).get('uiDraftRevision', previous_revision)
        recording = dict(recordingId=row['id'], videoFilename=Path(row['video']).name,
                         durationSeconds=row['durationSeconds'], references=references)
        # Legacy raw imports identify the source with their feedback-document hash.
        # Match that existing UI identity without claiming it is a video-byte hash.
        recording['contentSha256'] = row['contentSha256'] or row['labelSource']['sha256']
        filename = f"recordings/{row['id']}.json"
        write(args.output / filename, dict(schemaVersion=1, kind='volleycut-labeling-research-references', recordings=[recording]))
        index['recordings'].append(dict(id=row['id'], name=recording['videoFilename'], file=filename, tier=row['tier']))
    write(args.output / 'index.json', index)
    catalog = read(args.catalog)
    old = {r['recordingId']: r for r in catalog['records']}
    for row in inventory:
        entry = old.setdefault(row['id'], {})
        previous_source = entry.get('sourceType')
        previous_status = entry.get('targetStatus')
        human = row['tier'] in ('completed-exact','human-continuously-reviewed-draft','human-partially-reviewed-draft','reviewed-export-coverage')
        entry.update(recordingId=row['id'], environment=row['environment'], sourceGroup=row['sourceGroup'],
                     split=entry.get('split','challenge'), sourceType=('human-reviewed-model-feedback-export' if row['tier']=='reviewed-export-coverage'
                     else 'imported-human-labels' if human else entry.get('sourceType','model-candidate')),
                     targetStatus=row['tier'], labelPath=row['labelSource']['path'], labelSha256=row['labelSource']['sha256'],
                     videoPath=row['video'], videoFilename=Path(row['video']).name, videoSha256=row['contentSha256'],
                     durationSeconds=row['durationSeconds'], roi=row.get('roi'), rallies=row['rallies'],
                     ignoredIntervals=row['ignoredIntervals'], serveMarkers=row.get('serveMarkers',[]),
                     sideSwitches=row.get('sideSwitches',[]), candidateSource=entry.get('candidateSource',{}))
        entry['humanReviewedImport'] = human
        if previous_source and row['tier']=='reviewed-export-coverage':
            entry['sourceType'] = previous_source
        if previous_status:
            entry['targetStatus'] = previous_status
        merged = []
        for interval in sorted(entry['ignoredIntervals'], key=lambda r:r['start']):
            if merged and interval['start'] <= merged[-1]['end']:
                merged[-1]['end'] = max(merged[-1]['end'], interval['end'])
            else:
                merged.append(dict(interval))
        entry['ignoredIntervals'] = merged
    catalog['records'] = list(old.values())
    write(args.output / 'labeling-catalog.json', catalog)
    write(args.output / 'audit.json', dict(recordings=len(index['recordings']), models=index['models'],
        verifiedSavedEvaluationPairs=verified, selectionPolicy='99 then 98; highest common-unseen exact-label F1 across variant/draw fits, plus highest-recall draw within each chosen variant (F1 tiebreak); target 2s padding; user requested selection',
        targetPaddingSeconds=2, joinGapSeconds=3, commonUnseenSensitivity=sensitivity,
        trainingRun=False, sourceStudy=str(args.study)))
    print(json.dumps(dict(recordings=len(index['recordings']), verifiedSavedEvaluationPairs=verified,
                          models=[(x[2]['modelId'],x[2]['recallTargetPercent']) for x in selected])))


if __name__ == '__main__':
    main()
