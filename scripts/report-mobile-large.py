"""Public, identifier-free summary of the Large substitution selection panel."""
import argparse
import json
from pathlib import Path


def metric(value):
    return {k:v for k,v in value.items() if k!='recordings'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args(); evaluation=json.loads((a.experiment/'evaluation.json').read_text())
    rows=[]
    for c in evaluation['candidates']:
        rows.append({**{k:c[k] for k in ('variant','draw','seed','floorPercent')},
                     'epoch':c['setting']['epoch'],'decoder':c['setting']['decoder'],
                     'primary':metric(c['evaluation']['primary']),
                     'padding':[metric(x) for x in c['evaluation']['padding']]})
    selected=[]
    for c in evaluation['selected']:
        selected.append(dict(mode=c['mode'],**next(r for r in rows if (r['variant'],r['draw'],r['floorPercent'])==(c['variant'],c['draw'],c['floorPercent']))))
    output=dict(model='Frozen MobileNetV3 Large V1 + TCN',precision='FP32',floorsPercent=[98,99],
        primaryPaddingSeconds=2,joinGapSeconds=3,commonUnseenUsedForSelection=True,
        commonExactPanelRecordingCount=evaluation['commonExactPanelRecordingCount'],
        commonExactPanelSourceGroupCount=evaluation['commonExactPanelSourceGroupCount'],
        selectionRule='99% calibrated target first, 98% fallback; highest common-unseen F1, then highest-recall draw within that variant with F1 tie-break',
        scope='24 matched randomized and export-proxy split configurations; original-corpus OOF refits excluded',
        candidates=rows,selected=selected)
    (a.experiment/'selection-report.json').write_text(json.dumps(output,indent=2))
    lines=['# MobileNetV3 Large substitution','',
        'Frozen ImageNet V1 encoder at 224px and 2Hz, four 960-value regional pools; AV104 plus eight quality/age/availability values; FP32 TCN. Identical sampled frames and training recipe to Small. No 90–100% recall sweep: only calibration targets 98% and 99%.','',
        output['scope']+'. '+output['selectionRule']+'. Common-unseen is selection data, not an untouched test set. The target is calibrated retained-play recall; measured common-panel recall can be lower.','',
        f"The common-unseen exact-label selection panel contains {output['commonExactPanelRecordingCount']} recording(s) from {output['commonExactPanelSourceGroupCount']} source group(s). This small selection panel limits generalization claims.",'',
        '| UI option | Dataset variant | Draw | Target | Epoch | P_pad | R_core | F1_padP_coreR |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in selected:
        m=c['primary'];lines.append(f"| Highest {c['mode']} | {c['variant']} | {c['draw']} | {c['floorPercent']}% | {c['epoch']} | {m['P_pad']:.2%} | {m['R_core']:.2%} | {m['F1_padP_coreR']:.2%} |")
    lines+=['','Target product padding is 2s each side; gaps strictly under 3s join. Ignored time is subtracted after padding/joining without rejoining. The primary metric pools time across videos.','',
        '| UI option | Padding | P_pad | R_core | F1_padP_coreR | Model export | Human export | Difference |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in selected:
        for m in c['padding']:
            lines.append(f"| {c['mode']} | {m['paddingSecondsBeforeAndAfter']:.0f}s | {m['P_pad']:.2%} | {m['R_core']:.2%} | {m['F1_padP_coreR']:.2%} | {m['paddedModelExportSeconds']:.2f}s | {m['paddedHumanExportSeconds']:.2f}s | {m['exportDurationDifferenceSeconds']:+.2f}s |")
    lines+=['','The accompanying JSON contains every feasible fit at both targets and all four required padding cases. Native timing and feature-storage reports are separate until the phone runs complete.','']
    (a.experiment/'selection-report.md').write_text('\n'.join(lines))
    print(json.dumps({'selected':[{k:c[k] for k in ('mode','variant','draw','floorPercent')} for c in selected]}))


if __name__=='__main__':main()
