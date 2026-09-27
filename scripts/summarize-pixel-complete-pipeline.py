"""Summarize complete video-to-prediction timings without adding overlapping stages."""
import argparse
import json
import re
from pathlib import Path
from statistics import median

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recording-index',required=True)
    p.add_argument('--source-index',required=True)
    a=p.parse_args();source=json.loads((a.results/'result.json').read_text());plan=json.loads((a.results/'pipeline-plan.json').read_text())
    for index in (a.recording_index,a.source_index):
        if not re.fullmatch(r'(?:recording|private-reference|(?:indoor|grass|beach)-source)-[0-9]+',index):
            p.error('Use a private ledger index, not a filename or path')
    specs={r['id']:r for r in plan['cases']};summary=[]
    for family in ('production','mobile','mobile-large','dino'):
        rows=[r for r in source['results'] if specs[r['id']]['family']==family and not specs[r['id']]['warmup'] and r['status']=='complete']
        if not rows:continue
        m=lambda fn:median(fn(r) for r in rows)/1000
        summary.append({'family':family,'runs':len(rows),'videoSeconds':specs[rows[0]['id']]['seconds'],
            'videoAvSeconds':m(lambda r:r['stagesMs']['video_decode_and_features']),
            'audioSeconds':m(lambda r:r['stagesMs']['audio_decode_and_features']),
            'contextSeconds':m(lambda r:r['stagesMs']['contextualize']),
            'embeddingVideoPassSeconds':m(lambda r:r['profileMs'].get('neural/embedding_video_pass',0)),
            'embeddingDecodeOtherSeconds':m(lambda r:r.get('neural',{}).get('video',{}).get('decodeAndOtherMs',0)),
            'embeddingPreparationSeconds':m(lambda r:r.get('neural',{}).get('video',{}).get('prepareMs',0)),
            'embeddingEncoderSeconds':m(lambda r:r.get('neural',{}).get('video',{}).get('encoderAndReadbackMs',0)),
            'neuralTemporalSeconds':m(lambda r:r['profileMs'].get('neural/temporal_inference',0)),
            'neuralFusionSeconds':m(lambda r:r['profileMs'].get('neural/fusion_normalization_and_save',0)),
            'ralliesReadySeconds':m(lambda r:r['totalMs']-r['stagesMs']['score_specialists']),
            'scoreSeconds':m(lambda r:r['stagesMs']['score_specialists']),
            'scoreDecodeSeconds':m(lambda r:r['profileMs'].get('score/shared_decode_wall',0)),
            'servingEvaluationSeconds':m(lambda r:r['profileMs'].get('score/serving_side_evaluation',0)),
            'sideSwitchEvaluationSeconds':m(lambda r:r['profileMs'].get('score/side_switch_evaluation',0)),
            'allReadySeconds':m(lambda r:r['totalMs']),
            'totalRangeSeconds':[min(r['totalMs'] for r in rows)/1000,max(r['totalMs'] for r in rows)/1000],
            'rallyCounts':[r['rallyCount'] for r in rows],
            'thermalStatuses':[[r['thermalStart'],r['thermalEnd']] for r in rows]})
    a.output.mkdir(parents=True,exist_ok=True)
    artifact={'status':source['status'],'sourceIndex':a.source_index,'recordingIndex':a.recording_index,'precision':'FP32','summary':summary}
    (a.output/'complete-pipeline-summary.json').write_text(json.dumps(artifact,indent=2))
    durations=sorted({c['seconds'] for c in plan['cases']})
    lines=['# Pixel native complete FP32 pipeline benchmark','',f"Status: {source['status']}. Physical Pixel 10 Pro / Tensor G5. Video scope: {durations} seconds. No training.",'',
        'Every case starts from video with the AV cache bypassed. Neural execution uses ONNX Runtime native CPU with four threads; video decoding uses hardware MediaCodec. This is not a browser benchmark. The complete total includes selected rally predictions, serving-side and side-switch predictions and their necessary frame decoding and feature generation. It excludes video export/rendering, app launch, and ADB result collection. Timed neural diagnostics add conservative overhead.','',
        '| Model | Measured runs | AV video | Audio | Image feature pass | Rallies ready | Score specialists | All results ready | Rallies |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in summary:
        label='Production ensemble' if r['family']=='production' else r['family'].upper()+' FP32'
        lines.append(f"| {label} | {r['runs']} | {r['videoAvSeconds']:.2f}s | {r['audioSeconds']:.2f}s | {r['embeddingVideoPassSeconds']:.2f}s | {r['ralliesReadySeconds']:.2f}s | {r['scoreSeconds']:.2f}s | **{r['allReadySeconds']:.2f}s** | {r['rallyCounts']} |")
    lines+=['','Medians exclude the first full-length pass when the plan contains repeated runs. Medians of individual stages need not sum to median total. Rallies-ready is measured total less the separately timed score stage.','',
        '## Neural image feature pass','',
        '| Model | Decode / other | Preparation / quality | Embeddings / readback | TCN inference |',
        '|---|---:|---:|---:|---:|']
    for r in summary:
        if r['family']!='production':lines.append(f"| {r['family']} | {r['embeddingDecodeOtherSeconds']:.2f}s | {r['embeddingPreparationSeconds']:.2f}s | {r['embeddingEncoderSeconds']:.2f}s | {r['neuralTemporalSeconds']:.4f}s |")
    families={r['family'] for r in summary}
    image_contracts={'mobile':'MobileNetV3 Small uses 224px at 2Hz.',
                     'mobile-large':'MobileNetV3 Large uses 224px at 2Hz.',
                     'dino':'DINO uses 336px at 4Hz.'}
    image_description=' '.join(text for family,text in image_contracts.items() if family in families)
    lines+=['',image_description+' The initial complete prototype has a separate embedding decode pass, charged in full. No hypothetical shared-decode savings are deducted. Decode/other is a residual wall-time bucket, not isolated decoder time.','',
        '## Score stage','',
        '| Model | Shared specialist decode | Serving feature extraction + evaluation | Side-switch feature extraction + evaluation |',
        '|---|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['family']} | {r['scoreDecodeSeconds']:.2f}s | {r['servingEvaluationSeconds']:.2f}s | {r['sideSwitchEvaluationSeconds']:.2f}s |")
    lines+=['','Specialists use each selected model\'s actual rally boundaries. Counts and boundary locations change workload, so score cost is not borrowed from production or assumed equal. Planning and other small overhead are included in the full score-stage timer.','',
        '## Dependencies and validation','',
        '- Neural rally decisions are independent of production rally decisions. Neural models consume AV104 plus their own image embeddings; MobileNetV3 Small and Large also use eight quality/age/availability scalars.',
        '- The frozen serving-side model requires both production serve heads. The frozen side-switch model requires production rally/dead-state evidence. Those auxiliary production computations are included in neural totals. Current research integration also retains the small suppression/merge overhead.',
        '- Checkpoint selection and the calibration recall target are not claims of measured recall on this clip. See the corresponding selection report for the selected fit and decoder.',
        '- Variable-length temporal ONNX exports are qualified against PyTorch before timing. The temporal-decoder-parity.json in each evaluated result folder records real-device fused-input temporal and decoder validation; pixel/PTS/AV extraction qualification is separate.',
        '- Native pixel conversion and forward-frame selection still require parity/accuracy qualification against training-time OpenCV and nearest-PTS extraction. This is a functional timing prototype, not an accuracy-qualified deployment.',
        '- Phone was charging. Thermal status is retained per run; 0 = none, 1 = light. These are real device observations, not a controlled laboratory thermal comparison.','']
    checkpoints={'dino':'DINO original-medium / seed 3407 / epoch 30; previously selected target-99% decoder.',
                 'mobile':'MobileNetV3 Small expanded-large / seed 3407 / epoch 15; previously selected target-99% decoder.',
                 'mobile-large':'MobileNetV3 Large uses its independently selected highest-recall fit and frozen decoder. The Large selection report records its variant, draw, epoch, and 99%-or-98% calibration target.'}
    for family,text in checkpoints.items():
        if family in families:lines.append('- '+text)
    for r in summary:lines.append(f"- {r['family']}: total range {r['totalRangeSeconds'][0]:.2f}-{r['totalRangeSeconds'][1]:.2f}s; thermal start/end {r['thermalStatuses']}.")
    size_path=a.output/'model-size-summary.json'
    if size_path.exists():
        sizes=json.loads(size_path.read_text(encoding='utf-8-sig'))
        lines+=['','## Added model size','', '| FP32 model | Raw files | Gzip level 6 |','|---|---:|---:|']
        for family in ('mobile','dino'):
            files=[f for f in sizes['modelFiles'] if f['file'].startswith(family+'-')]
            lines.append(f"| {family} encoder + TCN + scalers | {sum(f['bytes'] for f in files)/1e6:.2f} MB | {sum(f['gzipBytes'] for f in files)/1e6:.2f} MB |")
        lines+=['','Decimal MB. Mobile also has a 784-byte pooling tensor. These are weights/configuration, not inference runtime size.',
            'The full ARM64 ONNX runtime in this benchmark contributes 33.10 MB of native libraries (12.44 MB gzip). Browser runtime assets measured separately are about 3.68 MB gzip for WASM-only or 6.72 MB for the GPU-capable variant, plus a small JavaScript loader. Runtime costs are shared between model choices. These are component measurements, not final Play Store delivery sizes.']
    lines+=['',f"Recording: `{a.recording_index}`. Raw results: private ledger `{a.source_index}`.",'']
    (a.output/'complete-pipeline-report.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(artifact,indent=2))
if __name__=='__main__':main()
