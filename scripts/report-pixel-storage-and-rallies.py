"""Measure saved benchmark tensors and score native rallies against a label snapshot."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.neural_evaluation import evaluate_predictions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,action='append',required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--recording-index',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert re.fullmatch(r'recording-\d+',a.recording_index)
    raw=a.labels.read_bytes(); labels=json.loads(raw)
    rows=[]
    for folder in a.results:
        result=json.loads((folder/'result.json').read_text())
        plan=json.loads((folder/'pipeline-plan.json').read_text())
        specs={s['id']:s for s in plan['cases']}
        for row in result['results']:
            spec=specs[row['id']]
            if row['status']!='complete' or spec['warmup']:
                continue
            duration=spec['seconds']
            # This report deliberately evaluates full recordings only. The pilot
            # starts later in the source and must have its own alignment contract.
            assert duration > 1000, 'Full-video source alignment required'
            report=evaluate_predictions([dict(id=a.recording_index,sourceGroup='benchmark-source',
                durationSeconds=duration,rallies=labels['rallies'],
                ignoredIntervals=labels.get('ignoredIntervals',[]),predictions=row['rallies'])])
            n=row['sampleRows']
            storage=dict(av104Float32BytesCalculated=n*104*4,
                         contextual520Float32BytesCalculated=n*520*4,
                         savedDiagnosticFiles=[], specialistFeatures={})
            neural=row.get('neural')
            if neural:
                storage['fusedFeatureFloat32BytesCalculated']=neural['featureRows']*neural['featureDimension']*4
                storage['temporalProbabilityFloat32BytesCalculated']=neural['featureRows']*4*4
                storage['embeddingFloat32BytesCalculated']=neural['video']['sampleCount']*(neural['featureDimension']-104-(8 if spec['family'].startswith('mobile') else 0))*4
                for suffix in ('tokens','features','probabilities'):
                    path=folder/(row['id']+'-'+suffix+'.f32')
                    storage['savedDiagnosticFiles'].append(dict(kind=suffix,exists=path.exists(),
                         bytes=path.stat().st_size if path.exists() else None))
            for name,key in (('servingSide','rawFeatures'),('sideSwitch','features')):
                score=row[name]
                encoded=score[key]
                storage['specialistFeatures'][name]=dict(rows=score['rows'],columns=score['columns'],
                    persistedBase64Bytes=len(encoded.encode('ascii')),
                    decodedPayloadBytes=len(base64.b64decode(encoded,validate=True)))
            rows.append(dict(family=spec['family'],rallyCount=row['rallyCount'],
                humanRallyCount=len(labels['rallies']),rallies=row['rallies'],
                whollyMissedSavedHumanRallies=report['guardrails']['primaryExportCoverage']['completeRallyLosses'],
                whollyMissedSavedHumanRalliesUnpadded=report['guardrails']['coreCoverage']['completeRallyLosses'],
                whollyMissedHumanRallyRanges=[dict(humanRallyNumber=r['truthIndex']+1,
                    start=r['start'],end=r['end'],evaluableCoreSeconds=r['evaluableCoreSeconds'])
                    for r in report['guardrails']['primaryExportCoverage']['rallies'] if r['completelyLost']],
                storage=storage,evaluation=report))
    artifact=dict(recordingIndex=a.recording_index,labelSha256=hashlib.sha256(raw).hexdigest(),
        labelSource='current saved label document; manually reviewed imported export with subsequent edits',
        labelBoundaryCaveat='Reviewed export boundaries are not independently precise serve-contact/dead-ball annotations.',
        precision='FP32',scope='full recording',targetPaddingSeconds=2,joinGapSeconds=3,
        whollyMissedDefinition='Saved human rallies with nonignored core time and zero retained core overlap (1e-9s tolerance) after model 2s padding and strict-under-3s gap joining; fully ignored rallies excluded. This is not one-to-one event matching.',
        memoryCaveat='Tensor byte calculations and measured persisted diagnostics, not peak RAM. Tokens are also embedded in fused features; these are alternative/redundant representations, not additive minimum memory requirements.',
        avCaveat='AV arrays were not persisted by the benchmark; byte counts are sampleRows times dimensions times float32 bytes.',
        rows=rows)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'storage-and-rallies.json').write_text(json.dumps(artifact,indent=2))
    lines=['# Native full-video feature storage and human comparison','',
           f'Recording `{a.recording_index}`; label SHA-256 `{artifact["labelSha256"]}`.',
           '',artifact['labelBoundaryCaveat'],'',
           'Target padding: 2s before/after; join positive gaps strictly less than 3s. Ignored time is excluded. Recall measures retained human core time, not a count of rallies.',
           '', '| Model | Found rallies | Human rallies | Wholly missed human rallies (export) | P_pad | R_core | F1_padP_coreR |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m=row['evaluation']['primary']
        lines.append(f"| {row['family']} | {row['rallyCount']} | {row['humanRallyCount']} | {row['whollyMissedSavedHumanRallies']} | {m['P_pad']:.2%} | {m['R_core']:.2%} | {m['F1_padP_coreR']:.2%} |")
    lines+=['',artifact['whollyMissedDefinition'],
        'Without padding or gap joining, wholly missed counts are '+', '.join(f"{r['family']}: {r['whollyMissedSavedHumanRalliesUnpadded']}" for r in rows)+'.',
        'Exact wholly missed saved-human ranges and both counts are included in the JSON.']
    lines+=['','## Padding sensitivity','',
        '| Model | Padding each side | P_pad | R_core | F1_padP_coreR | Model export | Human export | Difference |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        for m in row['evaluation']['padding']:
            lines.append(f"| {row['family']} | {m['paddingSecondsBeforeAndAfter']:.0f}s | {m['P_pad']:.2%} | {m['R_core']:.2%} | {m['F1_padP_coreR']:.2%} | {m['paddedModelExportSeconds']:.2f}s | {m['paddedHumanExportSeconds']:.2f}s | {m['exportDurationDifferenceSeconds']:+.2f}s |")
    lines+=['','## Rally separation diagnostics','',
        'One-to-one event matching at IoU ≥0.5; rallies touched by ignored time are excluded. These are separate from retained-play recall and sensitive to the reviewed-export boundary caveat.',
        '', '| Model | Evaluable human rallies | Matched | Event precision | Event recall | Event F1 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        g=row['evaluation']['guardrails']
        fmt=lambda v:'unavailable' if v is None else f'{v:.2%}'
        lines.append(f"| {row['family']} | {g['trueRallies']} | {g['matchedRallies']} | {fmt(g['eventPrecision'])} | {fmt(g['eventRecall'])} | {fmt(g['eventF1'])} |")
    lines+=['','## Feature storage','', '| Model | AV104 raw calculated | AV520 context calculated | Embeddings calculated | Fused features calculated | Saved diagnostic tensors | Specialist decoded payloads |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        s=row['storage']; mb=lambda value:f'{value/1e6:.3f} MB'
        lines.append('| '+ ' | '.join([row['family'],mb(s['av104Float32BytesCalculated']),mb(s['contextual520Float32BytesCalculated']),
            mb(s.get('embeddingFloat32BytesCalculated',0)),mb(s.get('fusedFeatureFloat32BytesCalculated',0)),
            mb(sum(f['bytes'] or 0 for f in s['savedDiagnosticFiles'])),mb(sum(f['decodedPayloadBytes'] for f in s['specialistFeatures'].values()))])+' |')
    lines+=['',artifact['memoryCaveat'],artifact['avCaveat'],'',
            'The JSON includes actual rally ranges, event matching diagnostics, all four padding cases (0/1/2/3s), export durations/differences, and saved-file existence and byte counts. These are observed native outputs; pixel/PTS/AV parity against the training extractor remains unqualified.','']
    (a.output/'storage-and-rallies.md').write_text('\n'.join(lines))
    print(json.dumps({'models':len(rows),'humanRallies':len(labels['rallies'])}))


if __name__=='__main__':main()
