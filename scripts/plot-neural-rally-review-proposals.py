#!/usr/bin/env python3
"""Plot audited fixed-budget export, event and observed-start tradeoffs."""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(private_value('private-reference-0092'))
MODELS = ('compact_boost', 'dino_global', 'dino_boost')
INVENTORIES = ('legacy', 'local_events', 'local_heads')
MODEL_LABELS = {'compact_boost': 'Compact short boost', 'dino_global': 'DINO global', 'dino_boost': 'DINO short boost'}
INVENTORY_LABELS = {'legacy': 'Prior flags', 'local_events': 'Local event proposals', 'local_heads': 'Local proposals + boundary heads'}
COLORS = {'compact_boost': '#2864b7', 'dino_global': '#c86b17', 'dino_boost': '#16877a'}
MARKERS = {'legacy': 'o', 'local_events': 's', 'local_heads': '^'}
STYLES = {'legacy': ':', 'local_events': '--', 'local_heads': '-'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ref(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'sizeBytes': path.stat().st_size}


def quality(row, kind):
    if kind == 'export':
        return 100*row['primary']['F1_padP_coreR']
    if kind == 'event':
        return 100*row['identity']['eventF1']
    return 100*row['identity']['observedStartLocalization']['1']['f1']


def render(summary, mode, directory):
    arms = [row for row in summary['arms'] if row['mode'] == mode and row['ranker'] == 'evidence']
    if len(arms) != 36:
        raise ValueError('Expected 36 evidence-ranked arms per mode')
    baselines = {row['id']: row for row in summary['automaticBaselines']}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.7))
    fig.patch.set_facecolor('#fbfcfe')
    fig.subplots_adjust(left=.055, right=.985, bottom=.235, top=.755, wspace=.245)
    title = 'Production with model-guided review' if mode == 'production' else 'Neural detector with review'
    fig.suptitle(title, fontsize=18, fontweight='bold', y=.975, color='#182532')
    fig.text(.5, .925, 'Fixed evidence ranking | actual playback workload | perfect localized human correction',
             ha='center', fontsize=11, color='#3b4c5e')
    model_handles = [Line2D([0], [0], color=COLORS[model], lw=2.4, label=MODEL_LABELS[model]) for model in MODELS]
    inventory_handles = [Line2D([0], [0], color='#49596b', marker=MARKERS[inventory], linestyle=STYLES[inventory],
                                markersize=6, lw=1.7, label=INVENTORY_LABELS[inventory]) for inventory in INVENTORIES]
    inventory_handles.append(Line2D([0], [0], color='#49596b', marker='D', linestyle='None',
                                    markerfacecolor='white', markersize=6, label='Automatic baseline'))
    fig.legend(handles=model_handles, loc='upper center', bbox_to_anchor=(.5, .9), ncol=3, frameon=False, fontsize=10)
    fig.legend(handles=inventory_handles, loc='upper center', bbox_to_anchor=(.5, .85), ncol=4, frameon=False, fontsize=9.5)
    panels = (('export', 'Export quality', 'F1_padP_coreR (%)'),
              ('event', 'Separate rally detection', 'Event F1, IoU >= 0.5 (%)'),
              ('start', 'Observed rally-start localization', 'Start F1 within 1 second (%)'))
    for axis, (kind, title, label) in zip(axes, panels):
        values = []
        for model in MODELS:
            for inventory in INVENTORIES:
                selected = sorted([row for row in arms if row['model'] == model and row['inventory'] == inventory],
                                  key=lambda row: row['budgetFraction'])
                if [row['budgetFraction'] for row in selected] != [.05, .1, .2, .4]:
                    raise ValueError('Budget scope differs')
                x = [row['workload']['reviewSeconds']/60 for row in selected]
                y = [quality(row, kind) for row in selected]
                values.extend(y)
                axis.plot(x, y, color=COLORS[model], marker=MARKERS[inventory], linestyle=STYLES[inventory],
                          linewidth=1.5, markersize=5, alpha=.9)
        for name in (('production',) if mode == 'production' else MODELS):
            value = quality(baselines[name], kind)
            values.append(value)
            color = '#334155' if name == 'production' else COLORS[name]
            axis.scatter([0], [value], marker='D', facecolors='white', edgecolors=color, s=52, linewidths=1.6, zorder=8)
        axis.set_ylim(max(0, 5*((min(values)-4)//5)), 101)
        axis.set_xlim(-1.6, max(row['workload']['budgetSeconds']/60 for row in arms)*1.025)
        axis.set_title(title, fontsize=11, fontweight='bold', pad=12)
        axis.set_ylabel(label, fontsize=10)
        axis.set_xlabel('Playback to review (minutes)', fontsize=10, labelpad=9)
        axis.grid(True, color='#dce3eb', linewidth=.65, alpha=.75)
        axis.set_axisbelow(True)
        axis.set_facecolor('white')
        axis.tick_params(labelsize=9, colors='#334155')
        for spine in axis.spines.values():
            spine.set_color('#cbd5e1')
    fig.text(.055, .13,
             'Each curve includes 5%, 10%, 20% and 40% per-video review caps; unused cap is allowed. Points are three-seed means on 8 videos / 322 rallies.',
             fontsize=9, color='#334155')
    fig.text(.055, .091,
             'Export: symmetric 2-second padding, strict gap <3-second joining. Event identities stay separate. Synthetic review-edge starts are excluded from observed-start predictions.',
             fontsize=9, color='#334155')
    fig.text(.055, .052,
             'Start timing is a serve-contact proxy, not serving-side or score accuracy. Hypothetical perfect review; development data, not fresh protected-test validation.',
             fontsize=9, color='#334155')
    files = []
    for extension in ('png', 'svg'):
        path = directory/f'rally-review-{mode}-quality-vs-workload.{extension}'
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=170 if extension == 'png' else None, facecolor=fig.get_facecolor(), metadata={'Creator': 'VolleyCut fixed rally review experiment'})
        files.append(ref(path))
    plt.close(fig)
    return files


def main(root):
    summary_path = root/'summary.json'
    summary = read(summary_path)
    audit_path = root/'summary-audit-v1.json'
    audit = read(audit_path)
    if not audit['passed'] or audit['summary']['sha256'] != ref(summary_path)['sha256'] or audit['contractSha256'] != summary['contractSha256']:
        raise ValueError('Independent audited summary binding does not match')
    directory = root/'figures'; directory.mkdir(exist_ok=True)
    assets = []
    for mode in ('production', 'individual'):
        assets.extend(render(summary, mode, directory))
    manifest = {'kind': 'rally-review-quality-vs-workload-figures-v1', 'createdAt': datetime.now(timezone.utc).isoformat(),
                'contractSha256': summary['contractSha256'], 'summary': ref(summary_path), 'summaryAudit': ref(audit_path),
                'sourceScript': ref(Path(__file__)), 'matplotlibVersion': matplotlib.__version__,
                'selection': 'Every evidence-ranked arm, 3 models x 3 inventories x 4 caps; automatic baselines at zero workload',
                'assetCount': len(assets), 'assets': assets}
    manifest_path = directory/'manifest.json'
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2); stream.write('\n')
    print(json.dumps({'completed': True, 'manifest': ref(manifest_path), 'assets': assets}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    main(args.root)
