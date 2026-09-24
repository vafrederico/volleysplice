#!/usr/bin/env python3
"""Plot an already-completed, immutable split-adviser summary (no fitting)."""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(private_value('private-reference-0093'))
POLICIES = ('event_starts', 'head_evidence', 'corroborated')
LABELS = {'event_starts': 'Neural event starts', 'head_evidence': 'Boundary heads',
          'corroborated': 'Corroborated starts'}
COLORS = {'event_starts': '#1675a9', 'head_evidence': '#c66d18', 'corroborated': '#248466'}
MARKERS = {'event_starts': 'o', 'head_evidence': 's', 'corroborated': 'D'}


def ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'sizeBytes': path.stat().st_size}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def style_axis(axis):
    axis.spines[['top', 'right']].set_visible(False)
    axis.spines[['left', 'bottom']].set_color('#b7bec7')
    axis.grid(axis='y', color='#e3e7ec', linewidth=.7)
    axis.set_axisbelow(True)
    axis.tick_params(colors='#374151')


def save(fig, output, stem):
    files = []
    for extension in ('png', 'svg'):
        path = output/f'{stem}.{extension}'
        require(not path.exists(), 'Refusing to overwrite '+str(path))
        fig.savefig(path, dpi=170, facecolor='white')
        files.append(ref(path))
    plt.close(fig)
    return files


def automatic_plot(summary, output):
    lookup = {row['policy']: row for row in summary['automatic']}
    require(set(lookup) == set(POLICIES), 'Automatic policy scope differs')
    arms = [lookup[p] for p in POLICIES]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.7))
    fig.subplots_adjust(left=.075, right=.98, bottom=.26, top=.78, wspace=.25)
    fig.suptitle('Can compact identify a missing rally boundary?', x=.06, y=.97,
                 ha='left', fontsize=18, fontweight='bold', color='#162536')
    fig.text(.06, .903, 'Automatic proposals inside production rallies · matching within 1 second',
             fontsize=11, color='#4b5563')
    positions = np.arange(len(POLICIES))
    for index, (name, label, color) in enumerate((('precision', 'Split precision', '#1675a9'),
                                                ('recall', 'Split recall', '#c66d18'))):
        values = [100 * row['split']['splitLocalization']['1'][name] for row in arms]
        stats = [row['splitSeedStatistics']['splitLocalization']['1'][name] for row in arms]
        errors = np.asarray([[value - 100 * stat['min'] for value, stat in zip(values, stats)],
                             [100 * stat['max'] - value for value, stat in zip(values, stats)]])
        bars = axes[0].bar(positions + (index - .5) * .32, values, .3, label=label,
                           color=color, yerr=errors, capsize=3, error_kw={'linewidth': 1})
        for bar, value, stat in zip(bars, values, stats):
            axes[0].text(bar.get_x()+bar.get_width()/2, max(value, 100*stat['max'])+2,
                         f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    axes[0].set_ylim(0, 108)
    axes[0].set_yticks(np.arange(0, 101, 20))
    axes[0].set_ylabel('Percentage of proposals / missing boundaries')
    axes[0].set_title('Precision and recall', loc='left', fontweight='bold', pad=12)
    axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.17), frameon=False, ncol=2)
    largest = 0.
    for index, (name, label, color) in enumerate((('matched', 'Correct proposals', '#248466'),
                                                ('falsePositive', 'Spurious proposals', '#b6424a'),
                                                ('falseNegative', 'Missed boundaries', '#8190a1'))):
        values = [row['split']['splitLocalization']['1'][name] for row in arms]
        largest = max(largest, *values)
        bars = axes[1].bar(positions + (index - 1) * .24, values, .22, label=label, color=color)
        for bar, value in zip(bars, values):
            axes[1].annotate(f'{value:.1f}', (bar.get_x()+bar.get_width()/2, value),
                             xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
    axes[1].set_ylim(0, max(1., largest)*1.15)
    axes[1].set_title('Extra rally starts: useful and spurious', loc='left', fontweight='bold', pad=12)
    axes[1].set_ylabel('Count across the evaluated videos')
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.17), frameon=False, ncol=1)
    for axis in axes:
        axis.set_xticks(positions, [LABELS[p].replace(' ', '\n', 1) for p in POLICIES])
        style_axis(axis)
    fig.text(.06, .055, 'Three-seed means; precision/recall error bars show seed ranges. '
             'Each target is an additional labeled rally start within a production parent.',
             fontsize=9, color='#536171')
    fig.text(.06, .025, 'Development footage. All export ranges remain exactly production’s. '
             'These start labels do not measure serving side, point winner, or score accuracy.',
             fontsize=9, color='#536171')
    return save(fig, output, 'compact-automatic-split-proposals')


def review_plot(summary, output):
    rows = [row for row in summary['reviewed'] if row['ranker'] == 'evidence']
    require(len(rows) == 28, 'Evidence-order reviewed scope differs')
    baseline = summary['baseline']
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.7))
    fig.subplots_adjust(left=.065, right=.98, bottom=.31, top=.8, wspace=.23)
    fig.suptitle('Review can clean the rally timeline while keeping every export range',
                 x=.055, y=.966, ha='left', fontsize=17, fontweight='bold', color='#162536')
    fig.text(.055, .904, 'Compact advice on production · evidence order · entire flagged parent reviewed with playback context',
             fontsize=11, color='#4b5563')
    selectors = (lambda row: row['identity']['eventF1'],
                 lambda row: row['identity']['observedStartLocalization']['1']['recall'])
    titles = ('Rally precision/recall balance', 'Observed rally-start recall within 1 second')
    ylabels = ('Rally event F1 (%)', 'Observed rally-start recall (%)')
    handles, labels = [], []
    for policy, inventory in [*( (p, inv) for p in POLICIES for inv in ('split_only', 'combined')),
                               ('none', 'cleanup_only')]:
        arm = sorted([row for row in rows if row['policy'] == policy and row['inventory'] == inventory],
                     key=lambda row: row['budgetFraction'])
        require([row['budgetFraction'] for row in arm] == [.05, .1, .2, .4], 'Budget scope differs')
        x = [row['workload']['reviewSeconds']/60 for row in arm]
        label = ('Cleanup flags only' if inventory == 'cleanup_only' else
                 LABELS[policy] + (' · splits only' if inventory == 'split_only' else ' + cleanup'))
        for axis, select in zip(axes, selectors):
            y = [100*select(row) for row in arm]
            line, = axis.plot(x, y, color=COLORS.get(policy, '#535d6e'), linewidth=1.9,
                              linestyle='--' if inventory == 'split_only' else ':' if inventory == 'cleanup_only' else '-',
                              marker=MARKERS.get(policy, '^'), markersize=5, label=label)
            if axis is axes[0]:
                handles.append(line); labels.append(label)
    for axis, select, title, ylabel in zip(axes, selectors, titles, ylabels):
        value = 100*select(baseline)
        axis.axhline(value, color='#717784', linewidth=1., linestyle=(0, (2, 4)))
        axis.scatter([0], [value], color='#111827', marker='X', s=65, zorder=5)
        axis.annotate(f'Production {value:.2f}%', xy=(0, value), xytext=(8, -16),
                      textcoords='offset points', fontsize=9, color='#374151')
        axis.set_xlabel('Actual playback minutes reviewed')
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc='left', fontweight='bold', fontsize=11, pad=12)
        axis.margins(x=.04, y=.18)
        style_axis(axis)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.53, .118),
               ncol=2, frameon=False, fontsize=9, columnspacing=2.2, handlelength=3.2)
    fig.text(.055, .063, 'Points use 5%, 10%, 20%, and 40% per-video review caps; actual playback may use less. '
             'Three-seed means, pooled within each seed.', fontsize=9, color='#536171')
    fig.text(.055, .038, 'Hypothetical perfect human accepts/snaps proposed splits and rejects false whole-rally records. '
             'Export footage stays fixed; review time is not measured labor.', fontsize=9, color='#536171')
    fig.text(.055, .013, 'Development results. Dashed curves: split guidance only. Solid curves: split guidance plus cleanup. '
             'Chronological-order comparisons remain in the full report.', fontsize=9, color='#536171')
    return save(fig, output, 'compact-review-rally-quality-vs-workload')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    summary_path = args.root/'summary.json'
    require(summary_path.is_file(), 'Completed summary.json is required; plotting does not execute outcomes')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    require(summary['kind'] == 'production-preserving-compact-split-adviser-summary-v1', 'Summary type differs')
    require(all(summary['verification'].values()), 'Summary verification failed')
    require(summary['automaticOutcomes'] == 9 and summary['reviewedOutcomes'] == 168, 'Matrix incomplete')
    output = args.output or args.root/'figures'
    output.mkdir(parents=True, exist_ok=True)
    require(not (output/'manifest.json').exists(), 'Figure manifest already exists')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'axes.labelcolor': '#374151'})
    artifacts = automatic_plot(summary, output) + review_plot(summary, output)
    manifest = {'kind': 'compact-split-adviser-figures-v1', 'input': ref(summary_path),
                'source': ref(Path(__file__).resolve()), 'artifacts': artifacts,
                'python': platform.python_version(), 'matplotlib': matplotlib.__version__,
                'automaticToleranceSeconds': 1, 'reviewRanker': 'evidence',
                'hypotheticalPerfectHuman': True, 'productionExportsUnchanged': True}
    with (output/'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'files': [row['path'] for row in artifacts], 'manifest': str(output/'manifest.json')}))


if __name__ == '__main__':
    main()
