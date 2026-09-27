#!/usr/bin/env python3
"""Replay decoded metrics and report the bounded encoder precision experiment."""
from pathlib import Path
import importlib.util
import math
import statistics
import sys

HERE=Path(__file__).resolve()
sys.path.insert(0,str(HERE.parents[1]))
from analysis.private_ledger import private_value
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,HERE.with_name(file))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
p=module('precision','evaluate-dino-precision.py')
summary=module('recognition_summary','summarize-neural-recognition.py')


def main():
    import numpy as np
    from analysis import neural_development as base
    from analysis.neural_evaluation import evaluate_predictions
    from analysis.crop_evaluation import subtract_intervals
    from analysis.schema import Interval
    root=p.DEFAULT;p.ensure_nas(root);protocol=p.verify(root)
    audit=p.read(root/'audit-inputs.json')
    if audit.get('passed') is not True or audit['protocol']!=p.ident(root/'protocol.json'):
        raise ValueError('A passing bound precision-input audit is required')
    manifest=p.read(p.OLD/'manifest-pts-v1.json')
    examples={e.id:e for e in base.load_examples(Path(manifest['exactManifest']['path']),False)}
    universe=sum(i.end-i.start for e in examples.values() for i in subtract_intervals([Interval(0,e.duration)],e.ignored))
    labels={'fp32':'FP32 encoder reference','fp16':'FP16 CUDA encoder','int8':'Mixed dynamic INT8 CPU encoder'}
    arms={};references=[];raw_results={}
    for arm in labels:
        results=[]
        for seed in p.SEEDS:
            path=root/f'result-{arm}-{seed}.json';r=p.read(path)
            if evaluate_predictions(r['predictions'])!=r['evaluation']:
                raise ValueError('Saved result metric replay differs')
            if arm=='fp32':
                control=p.read(r['control']['path'])
                if p.sha(r['control']['path'])!=r['control']['sha256']:
                    raise ValueError('Historical control changed')
                if ({row['id']:row for row in r['predictions']}
                        !={row['id']:row for row in control['predictions']}
                        or r['evaluation']!=control['evaluation']):
                    raise ValueError('FP32 raw boundaries or complete evaluation differ from historical control')
            settings={s['heldSourceGroup']:s['decoder'] for s in r['selections']}
            with np.load(root/f'probabilities-{arm}-{seed}.npz',allow_pickle=False) as n:
                if set(n.files)!=set(examples): raise ValueError('Probability inventory differs')
                replay={e.id:base.decode(e,n[e.id],settings[e.group]) for e in examples.values()}
            for row in r['predictions']:
                if [i.to_dict() for i in replay[row['id']]]!=row['predictions']:
                    raise ValueError('Decoded probabilities differ from published intervals')
            results.append(r);references.append(p.ident(path))
        arms[arm]=summary.group_summary(results)
        raw_results[arm]=results
        m=arms[arm]['mean']
        m['correctlyRemovedSeconds']=universe-m['paddedHumanExportSeconds']-m['incorrectExportSeconds']
    baseline=arms['fp32']['mean']
    deltas={arm:{k:arms[arm]['mean'][k]-baseline[k] for k in baseline} for arm in ('fp16','int8')}
    coverage={}
    for arm in ('fp16','int8'):
        paired=[]
        for reference,current in zip(raw_results['fp32'],raw_results[arm],strict=True):
            key=lambda r:(r['recordingId'],r['truthIndex'])
            old={key(r):r for r in reference['evaluation']['guardrails']['primaryExportCoverage']['rallies']}
            new={key(r):r for r in current['evaluation']['guardrails']['primaryExportCoverage']['rallies']}
            if set(old)!=set(new):raise ValueError('Paired rally inventory differs')
            paired.append({'seed':current['seed'],
                'worseRetainedCore':sum(new[k]['retainedCoreSeconds']<old[k]['retainedCoreSeconds']-1e-9 for k in old),
                'betterRetainedCore':sum(new[k]['retainedCoreSeconds']>old[k]['retainedCoreSeconds']+1e-9 for k in old),
                'newCompleteLosses':sum(new[k]['completelyLost'] and not old[k]['completelyLost'] for k in old)})
        coverage[arm]={'seeds':paired,'mean':{k:statistics.mean(r[k] for r in paired) for k in paired[0] if k!='seed'}}
    extraction={arm:p.read(root/f'extraction-{arm}.json') for arm in ('fp16','int8')}
    drift={}
    for arm,extract in extraction.items():
        count=sum(r['samples'] for r in extract['records'])
        rows=[(r['samples'],r['driftVsFp32Reference']) for r in extract['records']]
        drift[arm]={'samples':count,
            'maximumAbsoluteError':max(r['maximumAbsoluteError'] for _,r in rows),
            'meanAbsoluteError':sum(n*r['meanAbsoluteError'] for n,r in rows)/count,
            'rootMeanSquareError':math.sqrt(sum(n*r['rootMeanSquareError']**2 for n,r in rows)/count),
            'meanCosineSimilarity':sum(n*r['meanCosineSimilarity'] for n,r in rows)/count,
            'minimumCosineSimilarity':min(r['minimumCosineSimilarity'] for _,r in rows)}
        references.append(p.ident(root/f'extraction-{arm}.json'))
    for file in ('protocol.json','audit-inputs.json','qualification.json','cuda-pilot.json','browser-v1/report.json',
                 'int8-saturation-diagnostic-v1/report.json','sequential-rgb-amendment-v1/report.json',
                 'parallel-int8-amendment-v1.json'):
        references.append(p.ident(root/file))
    q=p.read(root/'qualification.json');cuda=p.read(root/'cuda-pilot.json');browser=p.read(root/'browser-v1/report.json')
    report={'kind':'dino-encoder-precision-summary-v1','arms':arms,'deltasVsFp32':deltas,
        'embeddingDrift':drift,'pairedRallyCoverageVsFp32':coverage,'references':references,'probabilityDecodeAndMetricReplayPassed':True,
        'inputAudit':p.ident(root/'audit-inputs.json'),
        'targetPaddingSeconds':2,'paddingSensitivity':[0,1,2,3],'joinGapStrictlyLessThanSeconds':3,
        'protectedTestOpened':False,'mobileDeploymentQualified':False,'browserQualification':browser,
        'aggregation':'Pool eight recording intersections/denominators per seed, then average three seeds; all precisions use identical original95%-selected heads and decoders.',
        'strict99Replay':'Separate recall99-v1 artifacts preserve independently selected99% controls and explicit infeasible folds.'}
    strict99=p.read(root/'recall99-v1/report.json')
    strict99_results=[]
    for ref in strict99['results']:
        if p.sha(ref['path'])!=ref['sha256']:
            raise ValueError('Strict99 precision result changed')
        strict99_results.append(p.read(ref['path']))
    report['strict99']={'report':p.ident(root/'recall99-v1/report.json'),
        'completeSeeds':strict99['completeSeeds'],
        'results':[{k:r[k] for k in ('seed','completeEvaluationScope','arms')} for r in strict99_results]}
    if (root/'summary.json').exists():
        if p.read(root/'summary.json')!=report:
            raise ValueError('Immutable numerical summary changed')
    else:
        p.write(root/'summary.json',report)
    lines=['# DINO encoder precision experiment — 2026-09-23','',
        'FP16 CUDA is the useful result from this precision experiment: it halves encoder weight storage and preserves every rally\'s retained core time at the primary 2-second padding across all three original operating points. Mixed dynamic INT8 reduces file size further, but lowers mean recall and F1 in that native CPU comparison and fails the measured CPU/browser parity check. In the sole complete strict99% seed it slightly raises recall while reducing precision and F1. Keep FP16 as the desktop candidate; this INT8 graph is not qualified for deployment.','',
        'This isolates encoder precision while keeping all three historical DINO+TCN seeds, fold scalers, checkpoints and decoders fixed. These are the original 95%-inner-recall-selected operating points. A separate strict99% replay uses the newly selected controls without precision-specific tuning. No encoder training or protected-test evaluation occurs here.','',
        'The scope is eight development recordings from four source groups, 33,382 sampled frames, 322 rallies, with the existing ignored ranges excluded from rally evaluation. FP32 and FP16 refer to encoder arithmetic: every resulting embedding cache is stored as float16, matching the historical representation.','',
        '## Frozen-head rally results','',
        'Target export padding is 2 seconds before and after, and padded gaps are joined only when strictly below 3 seconds. The table averages three per-seed pooled evaluations. Recall means retained human core play time, not the count of separately identified rallies.','',
        '| Encoder | P_pad | R_core | F1_padP_coreR | Export min | Correctly removed min | Incorrect export min | Wanted export omitted min |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,label in labels.items():
        m=arms[arm]['mean']
        lines.append(f"| {label} | {100*m['P_pad']:.3f}% | {100*m['R_core']:.3f}% | {100*m['F1_padP_coreR']:.3f}% | {m['paddedModelExportSeconds']/60:.3f} | {m['correctlyRemovedSeconds']/60:.3f} | {m['incorrectExportSeconds']/60:.3f} | {m['wantedExportOmittedSeconds']/60:.3f} |")
    lines += ['', 'Correctly removed is evaluable time outside both padded human and model exports. Incorrect export is extra footage outside the padded human export. Wanted export omitted is padded human footage missing from the model export.']
    lines += ['', '| Encoder | Complete losses | Incomplete losses | Short complete losses | Long-rally core recall | Event F1 |',
              '|---|---:|---:|---:|---:|---:|']
    for arm,label in labels.items():
        m=arms[arm]['mean']
        lines.append(f"| {label} | {m['completeLosses']:.2f} | {m['incompleteLosses']:.2f} | {m['shortCompleteLosses']:.2f} | {100*m['longR_core']:.3f}% | {100*m['eventF1']:.3f}% |")
    lines += ['', 'Complete loss means no human rally core retained by the padded export. Incomplete includes complete and partial losses. Short means original duration at most 3 seconds. Event F1 uses uncensored one-to-one IoU at least 0.5 and does not certify precise serve boundaries.','',
              '| Encoder | ΔP percentage points | ΔR percentage points | ΔF1 percentage points | Δ missed core seconds |',
              '|---|---:|---:|---:|---:|']
    for arm,d in deltas.items():
        lines.append(f"| {labels[arm]} | {100*d['P_pad']:+.4f} | {100*d['R_core']:+.4f} | {100*d['F1_padP_coreR']:+.4f} | {d['missedCoreSeconds']:+.3f} |")
    lines += ['', 'Paired human-rally coverage, averaged over the three seeds with padding of 2 seconds:', '',
        '| Encoder | Rallies retaining less core | Rallies retaining more core | New complete losses versus FP32 |',
        '|---|---:|---:|---:|']
    for arm,c in coverage.items():
        m=c['mean'];lines.append(f"| {labels[arm]} | {m['worseRetainedCore']:.2f} | {m['betterRetainedCore']:.2f} | {m['newCompleteLosses']:.2f} |")
    lines += ['', '## Encoder size and engineering runtime','',
        '| Artifact | Actual bytes | Measurement | Median image inference |', '|---|---:|---|---:|',
        f"| FP32 ONNX | {q['graphs']['fp32']['sizeBytes']:,} | Desktop ORT CPU, 2 threads | {1000*q['timing']['fp32']['medianSeconds']:.2f} ms |",
        f"| FP16 PyTorch state | {cuda['fp16Checkpoint']['sizeBytes']:,} | RTX 3080 CUDA | {1000*cuda['timing']['fp16']['medianSeconds']:.2f} ms |",
        f"| Dynamic INT8 ONNX | {q['graphs']['dynamicInt8']['sizeBytes']:,} | Desktop ORT CPU, 2 threads | {1000*q['timing']['dynamicInt8']['medianSeconds']:.2f} ms |",'',
        f"The same CUDA pilot measured FP32 at {1000*cuda['timing']['fp32']['medianSeconds']:.2f} ms/image. CPU and GPU rows use different backends and are not a precision-only speed comparison across rows. These are prepared-input, batch-one encoder timings on the Ryzen 9 5900X/RTX 3080 desktop. They exclude decoding, preprocessing, AV features, the temporal head and UI. Concurrent experiments share the host, and phone battery, thermals and memory remain unmeasured.",'',
        'The quantized graph contains 48 MatMulInteger and 48 DynamicQuantizeLinear nodes. Weights are per-channel signed INT8 and activations dynamically quantized to unsigned INT8. The 24 attention MatMuls, patch Conv, LayerNorm, Softmax and other operations remain FP32. It is a mixed graph, not full INT8. No activation-calibration dataset or label fitting is used.','',
        '## Browser qualification','',
        'Unlike the earlier temporal-head-only experiment, this runs the actual 336px DINO image encoder in desktop Chrome with ONNX Runtime Web 1.22.0, WASM CPU, one thread, on two fixed label-blind grass/indoor inputs.','',
        '| Graph | CPU↔browser parity | Browser median | Maximum output difference |','|---|---|---:|---:|']
    for m in browser['models']:
        lines.append(f"| {m['name']} | {'Pass' if all(c['passed'] for c in m['cases']) else 'FAIL'} | {m['medianMs']:.1f} ms/image | {max(c['maxAbsoluteError'] for c in m['cases']):.6f} |")
    lines += ['', '**The INT8 CPU accuracy results cannot be assumed to transfer to the browser:** the graph runs but fails CPU/browser numerical parity. FP32 executes with parity, but this desktop single-thread timing does not establish an acceptable phone experience. FP16 was tested with PyTorch CUDA, not a phone or browser delegate.','',
        '## Embedding drift and bounded INT8 alternatives','',
        '| Encoder | Mean cosine | Minimum token cosine | RMSE | Maximum absolute difference |','|---|---:|---:|---:|---:|']
    for arm,d in drift.items():
        lines.append(f"| {labels[arm]} | {d['meanCosineSimilarity']:.8f} | {d['minimumCosineSimilarity']:.6f} | {d['rootMeanSquareError']:.6f} | {d['maximumAbsoluteError']:.6f} |")
    lines += ['', 'Embedding statistics include all sampled frames, including ignored time; rally metrics exclude ignored time. On the fixed 32-frame pilot, original signed full-range INT8 had RMSE 0.945 and mean cosine 0.8864. Two predeclared label-blind alternatives—signed reduced range and unsigned full range—gave RMSE 0.962 and 0.948. Neither solved the drift; this does not support AVX2 U8S8 saturation as its main cause. The original graph had the lowest pilot RMSE and remained the full replay candidate. No rally outcomes selected this choice.','',
        '## Required padding sensitivity','',
        '| Encoder | Before/after s | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in labels:
        for r in arms[arm]['padding']:
            lines.append(f"| {labels[arm]} | {r['paddingSecondsBeforeAndAfter']:.0f} | {100*r['P_pad']:.3f}% | {100*r['R_core']:.3f}% | {100*r['F1_padP_coreR']:.3f}% | {r['paddedModelExportSeconds']:.3f} | {r['paddedHumanExportSeconds']:.3f} | {r['exportDurationDifferenceSeconds']:+.3f} |")
    lines += ['', '## Independently selected 99% recall controls','',
        'This replay holds the new FP32-selected strict99% checkpoint and decoder selection fixed across precisions. Eligibility is measured on inner validation groups; held-source recall is not guaranteed. No precision-specific recalibration or requalification of the inner 99% constraint was performed. Two of twelve DINO seed/source folds were infeasible within the frozen grid and remain absent. No relaxed threshold or production fallback fills those folds.','',
        '**Compare precisions within a seed only.** The full-scope row and partial feasible-subset rows have different recording populations, so no three-seed aggregate is reported here. The original95% eight-recording table above remains the complete matched precision-isolation comparison.','',
        '| Seed | Scope | Encoder | P_pad | R_core | F1_padP_coreR | Complete losses |',
        '|---:|---|---|---:|---:|---:|---:|']
    for r in strict99_results:
        for arm in labels:
            branch=r['arms'][arm]
            e=branch['fullEightRecordingEvaluation'] or branch['partialFeasibleScopeEvaluation']
            m=summary.metrics({'evaluation':e})
            scope='All 8 recordings' if r['completeEvaluationScope'] else f"Partial: {len(branch['scopeRecordingIds'])}/8 recordings"
            lines.append(f"| {r['seed']} | {scope} | {labels[arm]} | {100*m['P_pad']:.3f}% | {100*m['R_core']:.3f}% | {100*m['F1_padP_coreR']:.3f}% | {m['completeLosses']} |")
    lines += ['', 'Guardrails for those same strict99% seed/scope combinations:', '',
        '| Seed | Encoder | Incomplete losses | Short complete losses | Long-rally core recall | Event F1 | Missed core s |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in strict99_results:
        for arm in labels:
            branch=r['arms'][arm]
            e=branch['fullEightRecordingEvaluation'] or branch['partialFeasibleScopeEvaluation']
            m=summary.metrics({'evaluation':e})
            lines.append(f"| {r['seed']} | {labels[arm]} | {m['incompleteLosses']} | {m['shortCompleteLosses']} | {100*m['longR_core']:.3f}% | {100*m['eventF1']:.3f}% | {m['missedCoreSeconds']:.3f} |")
    lines += ['', 'Strict99% sensitivity retains exactly the same per-seed feasible scope and the same FP32-selected decoder for all three precision arms within each fold:', '',
        '| Seed | Encoder | Before/after s | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in strict99_results:
        for arm in labels:
            branch=r['arms'][arm];e=branch['fullEightRecordingEvaluation'] or branch['partialFeasibleScopeEvaluation']
            for pad in e['padding']:
                lines.append(f"| {r['seed']} | {labels[arm]} | {pad['paddingSecondsBeforeAndAfter']:.0f} | {100*pad['P_pad']:.3f}% | {100*pad['R_core']:.3f}% | {100*pad['F1_padP_coreR']:.3f}% | {pad['paddedModelExportSeconds']:.3f} | {pad['paddedHumanExportSeconds']:.3f} | {pad['exportDurationDifferenceSeconds']:+.3f} |")
    lines += ['', '## Integrity and artifacts','',
        'The original FP32 caches, videos and checkpoints were not overwritten. The 32-frame engineering gate passed FP32 ONNX/PyTorch parity and reproduced historical frame selection within the preregistered float16-cache tolerance. Sequential decoding preserved the historical `round(timestamp × fps)` ordinal sampler, ROI and letterbox; 6,723 comparisons against historical seeking passed byte-for-byte. This changes IO, not frame-selection semantics.','',
        'Full INT8 replay used two disjoint CPU workers, each with two ORT threads and batch one, using the exact same graph. All caches and RGB inputs are on the NAS. A separate audit verified every chunk/receipt/cache hash, timestamps, finite tensor dimensions, source associations, and that FP16 parameters are exactly the original weights cast to half. Probability-to-interval decoding and every padding evaluation were independently replayed by the summary script.','',
        'Artifacts: private ledger private-reference-0103. Detailed immutable receipts remain private.','',
        'Key artifact hashes (SHA-256; paths relative to the experiment folder):','',
        '| Artifact | SHA-256 |','|---|---|']
    for file in ('protocol.json','audit-inputs.json','report.json','recall99-v1/protocol.json','recall99-v1/report.json','summary.json','browser-v1/report.json'):
        lines.append(f"| `{file}` | `{p.sha(root/file)}` |")
    lines += ['', 'ONNX Runtime documents dynamic quantization and mixed operator support in its [quantization guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html). Quantization quality and target-backend parity require measurement; file size alone is insufficient.','']
    path=HERE.parents[1]/'docs/research/dino-precision-results-2026-09-23.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    print(str(path),flush=True)


if __name__=='__main__': main()
