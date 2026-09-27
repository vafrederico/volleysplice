#!/usr/bin/env python3
"""Render completed typed-boundary results without changing numerical artifacts."""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0079'))
POLICIES = ('first_start', 'typed_starts', 'separate_ends', 'head_refined')
LABELS = {'first_start': 'First-start correction', 'typed_starts': 'Typed starts',
          'separate_ends': 'Separate ends', 'head_refined': 'Head refinement'}
COLORS = {'first_start': '#1675a9', 'typed_starts': '#c66d18',
          'separate_ends': '#248466', 'head_refined': '#8453a6'}
MARKERS = {'first_start': 'o', 'typed_starts': 's', 'separate_ends': 'D', 'head_refined': '^'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'sizeBytes': path.stat().st_size}


def style(axis, direction='y'):
    axis.spines[['top', 'right']].set_visible(False)
    axis.spines[['left', 'bottom']].set_color('#b7bec7')
    axis.grid(axis=direction, color='#e3e7ec', linewidth=.7)
    axis.set_axisbelow(True)
    axis.tick_params(colors='#374151')


def save(fig, output, assets, stem):
    artifacts = []
    for extension in ('png', 'svg'):
        path = output/f'{stem}.{extension}'
        copy = assets/path.name
        require(not path.exists() and not copy.exists(), 'Refusing to overwrite figure '+str(path))
        fig.savefig(path, dpi=170, facecolor='white')
        shutil.copyfile(path, copy)
        source_ref, copy_ref = reference(path), reference(copy)
        require(source_ref['sha256'] == copy_ref['sha256'], 'Figure copy changed')
        artifacts.append({'study': source_ref, 'repository': copy_ref})
    plt.close(fig)
    return artifacts


def automatic(summary, output, assets):
    lookup = {a['policy']: a for a in summary['automatic']}
    require(set(lookup) == set(POLICIES), 'Automatic scope differs')
    rows = [summary['baseline'], *(lookup[p] for p in POLICIES)]
    labels = ['Production', *(LABELS[p] for p in POLICIES)]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.0))
    fig.subplots_adjust(left=.17, right=.965, top=.78, bottom=.20, wspace=.27)
    fig.suptitle('More precise rally records can still omit wanted play', x=.04, y=.96,
                 ha='left', fontsize=18, fontweight='bold', color='#162536')
    fig.text(.04, .896, 'Automatic compact advice on production · event timeline and exported footage are separate outputs',
             fontsize=11, color='#4b5563')
    positions = np.arange(len(rows))
    for index, (name, label, color) in enumerate((('eventPrecision', 'Rally precision', '#1675a9'),
                                                ('eventRecall', 'Rally recall', '#c66d18'),
                                                ('eventF1', 'Rally F1', '#248466'))):
        values = [100*r['identity'][name] for r in rows]
        bars = axes[0].barh(positions+(index-1)*.22, values, .20, color=color, label=label)
        for bar, value in zip(bars, values):
            axes[0].text(value+.7, bar.get_y()+bar.get_height()/2, f'{value:.1f}',
                         ha='left', va='center', fontsize=8.5)
    axes[0].set_xlim(0, 106)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('One-to-one rally detection (%)')
    axes[0].set_title('Individual rally quality', loc='left', fontsize=12, fontweight='bold', pad=12)
    axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=3, frameon=False, fontsize=9)
    raw = [100*r['coverage']['resultRawCoreRecall'] for r in rows]
    colors = ['#65748a', *(COLORS[p] for p in POLICIES)]
    bars = axes[1].barh(positions, raw, .55, color=colors)
    for bar, value in zip(bars, raw):
        axes[1].text(value-1.0, bar.get_y()+bar.get_height()/2, f'{value:.2f}%',
                     ha='right', va='center', fontsize=10, color='white', fontweight='bold')
    fixed = 100*summary['baseline']['primary']['R_core']
    axes[1].axvline(fixed, color='#172536', linestyle='--', linewidth=1.4,
                    label=f'Fixed padded-export recall: {fixed:.2f}%')
    axes[1].set_xlim(max(0, min(70, min(raw)-5)), 102)
    axes[1].set_yticks(positions, [''] * len(rows))
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Human play covered by raw rally intervals (%)')
    axes[1].set_title('Physical raw-core recall', loc='left', fontsize=12, fontweight='bold', pad=12)
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.13), frameon=False, fontsize=9)
    for axis in axes:
        style(axis, 'x')
    fig.text(.04, .065, 'Three-seed means after pooling recordings within each seed. Rally matching uses raw event identities. '
             'Physical core recall excludes ignored time only.', fontsize=9, color='#536171')
    fig.text(.04, .036, 'The dashed reference uses unchanged production exports with 2-second padding and gaps strictly below 3 seconds joined. '
             'It does not certify the new rally timeline.', fontsize=9, color='#536171')
    fig.text(.04, .009, 'Development footage; no serving-side, point-winner, or score-tracking accuracy is measured.',
             fontsize=9, color='#536171')
    return save(fig, output, assets, 'compact-typed-boundaries-automatic')


def reviewed(summary, output, assets):
    rows = [r for r in summary['reviewed'] if r['ranker'] == 'evidence']
    require(len(rows) == 32, 'Evidence-order reviewed scope differs')
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.98, top=.78, bottom=.27, wspace=.15)
    fig.suptitle('What review can fix depends on the permitted human action', x=.055, y=.96,
                 ha='left', fontsize=18, fontweight='bold', color='#162536')
    fig.text(.055, .896, 'Ideal human review · evidence order · whole flagged production parent plus playback context',
             fontsize=11, color='#4b5563')
    handles = []
    for axis, mode, title in zip(axes, ('proposal_confirmation', 'full_parent'),
                                 ('Confirm proposed rally starts', 'Edit every rally in the reviewed parent')):
        for policy in POLICIES:
            selected = sorted([r for r in rows if r['policy'] == policy and r['humanMode'] == mode],
                              key=lambda r: r['budgetFraction'])
            require([r['budgetFraction'] for r in selected] == [.05, .1, .2, .4], 'Budget scope differs')
            x = [r['workload']['reviewSeconds']/60 for r in selected]
            y = [100*r['identity']['eventF1'] for r in selected]
            line, = axis.plot(x, y, marker=MARKERS[policy], color=COLORS[policy],
                              markersize=5, linewidth=1.9, label=LABELS[policy])
            if axis is axes[0]:
                handles.append(line)
        baseline = 100*summary['baseline']['identity']['eventF1']
        axis.axhline(baseline, color='#737b87', linestyle=(0, (2, 4)), linewidth=1.)
        axis.scatter([0], [baseline], color='#172536', marker='X', s=60, zorder=5)
        axis.annotate(f'Production {baseline:.2f}%', (0, baseline), xytext=(6, -17),
                      textcoords='offset points', fontsize=9, color='#374151')
        axis.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=12)
        axis.set_xlabel('Actual playback minutes reviewed')
        axis.margins(x=.04, y=.18)
        style(axis)
    axes[0].set_ylabel('Rally event F1 (%)')
    fig.legend(handles, [LABELS[p] for p in POLICIES], loc='lower center', bbox_to_anchor=(.53, .155),
               ncol=4, frameon=False, fontsize=10, columnspacing=1.6, handlelength=3.)
    fig.text(.055, .109, 'Left: accept a proposed start within 1 second with material rally overlap; correct its type. '
             'Separate-end policies also correct that accepted rally’s end.', fontsize=9, color='#536171')
    fig.text(.055, .081, 'Confirmation can omit unproposed rallies. Right: a stronger upper bound that discovers and edits all rallies inside the reviewed parent.',
             fontsize=9, color='#536171')
    fig.text(.055, .052, 'Points use 5%, 10%, 20%, and 40% per-video review caps; actual workload may be lower. '
             'Three-seed means, pooled within each seed.', fontsize=9, color='#536171')
    fig.text(.055, .023, 'Exports remain exactly production’s. Review footage is not measured labor. Development results; '
             'chronological-order comparisons are in the full report.', fontsize=9, color='#536171')
    return save(fig, output, assets, 'compact-typed-boundaries-human-review')


def boundary_quality(summary, output, assets):
    lookup = {r['policy']: r for r in summary['automatic']}
    rows = [lookup[p] for p in POLICIES]
    fig, axis = plt.subplots(figsize=(11.8, 6.7))
    fig.subplots_adjust(left=.08, right=.975, top=.77, bottom=.25)
    fig.suptitle('Start type and the matching rally end both matter', x=.055, y=.96,
                 ha='left', fontsize=18, fontweight='bold', color='#162536')
    fig.text(.055, .894, 'Automatic candidate quality · 1-second boundary tolerance · endpoints scored on the same matched rally',
             fontsize=11, color='#4b5563')
    specifications = [('Initial-start F1', '#1675a9', lambda r:r['typed']['byType']['initial_start']['1']['f1']),
                      ('Additional-rally start F1', '#c66d18', lambda r:r['typed']['byType']['additional_start']['1']['f1']),
                      ('Joint start-and-end F1', '#248466', lambda r:r['typed']['samePairBoundaries']['1']['f1'])]
    positions = np.arange(len(rows))
    for index, (label, color, getter) in enumerate(specifications):
        values = [100*getter(r) if getter(r) is not None else 0. for r in rows]
        bars = axis.bar(positions+(index-1)*.24, values, .22, label=label, color=color)
        for bar, row, value in zip(bars, rows, values):
            axis.annotate('n/a' if getter(row) is None else f'{value:.1f}',
                          (bar.get_x()+bar.get_width()/2, value), xytext=(0, 5),
                          textcoords='offset points', ha='center', fontsize=10)
    axis.set_xticks(positions, [LABELS[p] for p in POLICIES])
    axis.set_ylim(0, 108); axis.set_yticks(np.arange(0, 101, 20))
    axis.set_ylabel('F1 (%)')
    axis.legend(loc='upper center', bbox_to_anchor=(.5, -.14), ncol=3, frameon=False)
    style(axis)
    fig.text(.055, .076, 'The first gold rally within a production parent component is the initial target; later rallies are additional targets. '
             'Incorrect type assignments cannot match.', fontsize=9, color='#536171')
    fig.text(.055, .047, 'Joint F1 requires both observed endpoints correct on that same typed start match. '
             'Artificial partition boundaries and clipped endpoints cannot count as observed predictions.', fontsize=9, color='#536171')
    fig.text(.055, .018, 'Three-seed means after pooling recordings. Development timing proxies only; these are automatic candidates, '
             'not corrected human outcomes.', fontsize=9, color='#536171')
    return save(fig, output, assets, 'compact-typed-boundaries-start-and-end-quality')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--assets', type=Path, default=REPO/'docs/research/assets')
    args = parser.parse_args()
    report_path, summary_path = args.root/'report.json', args.root/'summary.json'
    require(report_path.exists() and summary_path.exists(), 'Completed report and summary are required')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    require(report['passed'] and report['automaticOutcomes'] == 12 and report['reviewedOutcomes'] == 192, 'Incomplete outcome matrix')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    require(len(summary['automatic']) == 4 and len(summary['reviewed']) == 64, 'Unexpected summary scope')
    output = args.output or args.root/'figures'
    output.mkdir(parents=True, exist_ok=True); args.assets.mkdir(parents=True, exist_ok=True)
    require(not (output/'manifest.json').exists(), 'Figures already captured')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.labelcolor':'#374151'})
    artifacts = (automatic(summary, output, args.assets) + reviewed(summary, output, args.assets)
                 + boundary_quality(summary, output, args.assets))
    manifest = {'kind':'compact-typed-boundary-figures-v1','report':reference(report_path),
                'summary':reference(summary_path),'source':reference(Path(__file__).resolve()),
                'artifacts':artifacts,'python':platform.python_version(),'matplotlib':matplotlib.__version__,
                'reviewRanker':'evidence','newNumericalOutcomes':False,'productionExportsFixed':True}
    with (output/'manifest.json').open('x',encoding='utf-8') as stream:
        json.dump(manifest,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'figures':len(artifacts),'manifest':str(output/'manifest.json')}))


if __name__ == '__main__':
    main()
