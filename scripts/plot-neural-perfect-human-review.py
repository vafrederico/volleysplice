#!/usr/bin/env python3
"""Plot audited, complete perfect-human-review quality against playback effort.

Requires matplotlib in the caller's environment. This is a presentation artifact:
all plotted values are read from the independently audited numerical summary.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(private_value('private-reference-0091'))
COLORS = {'production_compact': '#235b9b', 'production_dino': '#8550a5',
          'individual_compact': '#d17c18', 'individual_dino': '#25856d'}
FAMILY_LABELS = {'production_compact': 'Production + compact TCN',
                 'production_dino': 'Production + DINO+TCN',
                 'individual_compact': 'Individual compact TCN',
                 'individual_dino': 'Individual DINO+TCN'}
MODEL_NAMES = {'compact_boost': 'compact boost', 'compact_keep': 'compact keep',
               'dino_global': 'DINO global', 'dino_boost': 'DINO boost', 'dino_keep': 'DINO keep'}
POLICY_NAMES = {'guarded_trim': 'guarded review', 'suppression_zero': 'zero support',
                'suppression_half': 'half support', 'suppression_any': 'any disagreement',
                'bidirectional_half': 'two-way half support',
                'bidirectional_any': 'two-way disagreement',
                'positive_uncertain': 'positive uncertainty',
                'uncertain_narrow': 'narrow uncertainty',
                'uncertain_medium': 'medium uncertainty',
                'uncertain_wide': 'wide uncertainty',
                'all_positive': 'all positive candidates', 'all_candidates': 'full review'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': sha(path), 'sizeBytes': path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_ref(ref, path):
    require(Path(ref['path']).resolve() == Path(path).resolve(), 'Bound artifact path differs')
    require(ref['sha256'] == sha(path), 'Bound artifact hash differs')


def load_complete(root):
    paths = {name: root / name for name in ('report.json', 'summary.json', 'summary-audit-v1.json')}
    require(all(path.is_file() for path in paths.values()),
            'Wait for completed report, summary, and passed summary audit')
    # Check audit/completion status before reading any quality values.
    receipt = read(paths['summary-audit-v1.json'])
    require(receipt.get('passed') is True and not receipt.get('failures'), 'Summary audit did not pass')
    verify_ref(receipt['summary'], paths['summary.json'])
    verify_ref(receipt['report'], paths['report.json'])
    report = read(paths['report.json'])
    require(report.get('status') == 'completed-perfect-human-review'
            and report.get('resultCells') == 360 and report.get('policyCount') == 120,
            'Complete fixed matrix required')
    summary = read(paths['summary.json'])
    require(summary.get('passed') is True and summary.get('targetPaddingSeconds') == 2
            and summary.get('joinGapSeconds') == 3, 'Summary contract differs')
    require(summary.get('recordings') == 8 and summary.get('rallies') == 322
            and summary.get('seedCount') == 3, 'Summary recording/seed scope differs')
    require(summary['contractSha256'] == report['contractSha256'] == receipt['contractSha256'],
            'Study contract identity differs')
    return summary, {name: identity(path) for name, path in paths.items()}


def family(row):
    mode = 'production' if row['family'] == 'combination' else 'individual'
    architecture = 'compact' if row['neuralId'].startswith('compact_') else 'dino'
    return mode + '_' + architecture


def marker(model):
    return {'global': 'o', 'boost': '^', 'keep': 'D'}[model.rsplit('_', 1)[1]]


def pareto_labels(rows, scenario, maximum=4, automatic_scores=()):
    """Select at most four evenly spaced non-dominated policies deterministically."""
    front = []
    for row in rows:
        x = row['primary']['workload']['reviewSeconds']
        y = row['primary'][scenario]['F1_padP_coreR']
        dominated = any(other['primary']['workload']['reviewSeconds'] <= x
                        and other['primary'][scenario]['F1_padP_coreR'] >= y
                        and (other['primary']['workload']['reviewSeconds'] < x
                             or other['primary'][scenario]['F1_padP_coreR'] > y)
                        for other in rows)
        dominated = dominated or any(score >= y and (x > 0 or score > y)
                                     for score in automatic_scores)
        if not dominated:
            front.append(row)
    # Exact coincident points need only one label; break ties by frozen ID.
    unique = {}
    for row in sorted(front, key=lambda value: value['id']):
        key = (row['primary']['workload']['reviewSeconds'], row['primary'][scenario]['F1_padP_coreR'])
        unique.setdefault(key, row)
    ordered = [unique[key] for key in sorted(unique)]
    if len(ordered) <= maximum:
        return ordered
    indices = sorted({round(i * (len(ordered) - 1) / (maximum - 1)) for i in range(maximum)})
    return [ordered[i] for i in indices]


def short_label(row):
    source = 'Prod + ' if row['family'] == 'combination' else ''
    return f"{source}{MODEL_NAMES[row['neuralId']]}\n{POLICY_NAMES[row['policy']]}"


def main(root):
    summary, inputs = load_complete(root)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = [r for r in summary['configurations'] if r['family'] == 'individual'
            or r['anchor'] == 'productionDefault']
    require(len(rows) == 60, 'Expected 30 current-production and 30 individual policies')
    rows.sort(key=lambda row: row['id'])
    full_minutes = summary['evaluableVideoSeconds'] / 60
    baseline_keys = ['productionDefault', *MODEL_NAMES]
    baselines = {key: summary['baselines'][key] for key in baseline_keys}
    minimum = min([r['primary'][scenario]['F1_padP_coreR'] * 100
                   for r in rows for scenario in ('binary', 'boundary')]
                  + [r['F1_padP_coreR'] * 100 for r in baselines.values()])
    lower = max(0, 5 * math.floor((minimum - 1) / 5))
    upper = 101
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.06, right=.985, top=.79, bottom=.20, wspace=.13)
    fig.suptitle('Perfect-human review: quality versus footage watched',
                 x=.06, y=.985, ha='left', fontsize=20, fontweight='bold')
    fig.text(.06, .941, '8 videos · 322 rallies · three-seed means · ±2 s export padding · gaps joined only below 3 s',
             ha='left', fontsize=11, color='#444444')
    legend = [Line2D([], [], marker='o', linestyle='none', color=color, markersize=7,
                     label=FAMILY_LABELS[name]) for name, color in COLORS.items()]
    legend += [Line2D([], [], marker='s', markerfacecolor='none', markeredgecolor='#555555',
                      linestyle='none', markersize=7, label='Automatic baseline (no review)')]
    fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(.055, .906),
               ncol=3, frameon=False, columnspacing=2.5, handletextpad=.6)
    labels, points = {}, []
    for axis, scenario, title in zip(axes, ('binary', 'boundary'),
                                      ('A  Correct whole-candidate keep/remove',
                                       'B  Perfect editing inside flagged regions')):
        axis.set_title(title, loc='left', fontsize=13, pad=12)
        axis.set_xlim(-4, full_minutes + 8)
        axis.set_ylim(lower, upper)
        axis.grid(axis='both', color='#e9e9e9', linewidth=.7, zorder=0)
        axis.set_xlabel('Video playback to review (minutes at 1×)', labelpad=10)
        axis.axvline(full_minutes, color='#888888', linestyle=(0, (4, 4)), linewidth=1, zorder=1)
        axis.text(full_minutes - 1.5, lower + .65, f'All footage: {full_minutes:.2f} min',
                  rotation=90, va='bottom', ha='right', color='#666666', fontsize=9)
        for row in rows:
            x = row['primary']['workload']['reviewSeconds'] / 60
            y = row['primary'][scenario]['F1_padP_coreR'] * 100
            axis.scatter(x, y, c=COLORS[family(row)], marker=marker(row['neuralId']),
                         s=44, alpha=.83, edgecolors='white', linewidths=.5, zorder=3)
            points.append({'id': row['id'], 'scenario': scenario, 'family': family(row),
                           'playbackMinutes': x, 'F1Percent': y,
                           'PPercent': row['primary'][scenario]['P_pad'] * 100,
                           'RPercent': row['primary'][scenario]['R_core'] * 100,
                           'flaggedCandidateDecisions': row['primary']['workload']['flaggedCandidates'],
                           'distinctTrueRalliesReviewed': row['primary']['workload']['reviewedTrueRallies']})
        for key, metrics in baselines.items():
            color = '#555555' if key == 'productionDefault' else COLORS[
                'individual_compact' if key.startswith('compact_') else 'individual_dino']
            axis.scatter(0, metrics['F1_padP_coreR'] * 100, facecolors='white', edgecolors=color,
                         marker='s' if key == 'productionDefault' else marker(key),
                         linewidths=1.3, s=48, zorder=4)
        selected = pareto_labels(rows, scenario, automatic_scores=[
            value['F1_padP_coreR'] for value in baselines.values()])
        labels[scenario] = [row['id'] for row in selected]
        for i, row in enumerate(selected):
            x = row['primary']['workload']['reviewSeconds'] / 60
            y = row['primary'][scenario]['F1_padP_coreR'] * 100
            left = x > full_minutes * .68
            offset_x = -12 if left else 12
            # Labels sit beneath high-performing points to remain within axes.
            offset_y = -25 - (i % 2) * 24
            axis.annotate(short_label(row), xy=(x, y), xytext=(offset_x, offset_y),
                          textcoords='offset points', ha='right' if left else 'left', va='top',
                          fontsize=8.5, color=COLORS[family(row)],
                          bbox=dict(boxstyle='round,pad=.25', fc='white', ec='none', alpha=.9),
                          arrowprops=dict(arrowstyle='-', color=COLORS[family(row)],
                                          linewidth=.7, alpha=.75), zorder=5)
    axes[0].set_ylabel('F1_padP_coreR (%)', labelpad=10)
    fig.text(.06, .09, 'Shapes: ▲ short boost   ◆ keep rescue   ● DINO global. Labels mark a deterministic subset of non-dominated policies.',
             fontsize=9.5, color='#444444')
    fig.text(.06, .058, 'Hypothetical perfect humans; no reviewer timing measured. Flags use fixed disagreement or live-score rules, not a trained review head.',
             fontsize=9.5, color='#444444')
    fig.text(.06, .026, 'Current production means the checked-in fresh-project setting. Historical shipped/refit comparisons remain in the complete report.',
             fontsize=9.5, color='#444444')
    destination = root / 'figures'
    destination.mkdir(exist_ok=True)
    files = []
    for suffix in ('png', 'svg'):
        path = destination / f'perfect-human-review-quality-vs-workload.{suffix}'
        require(not path.exists(), 'Figure exists; do not overwrite evidence')
        fig.savefig(path, dpi=180, facecolor='white', metadata={'Creator': 'VolleyCut audited research plot'})
        files.append({'name': path.name, **identity(path)})
    plt.close(fig)
    for ref in inputs.values():
        verify_ref(ref, ref['path'])
    receipt = {'kind': 'perfect-human-review-figures-v1', 'passed': True,
               'createdAt': datetime.now(timezone.utc).isoformat(),
               'contractSha256': summary['contractSha256'], 'sourceScript': identity(Path(__file__)),
               'summary': inputs['summary.json'], 'summaryAudit': inputs['summary-audit-v1.json'],
               'report': inputs['report.json'], 'matplotlibVersion': matplotlib.__version__,
               'files': files, 'targetPaddingSeconds': 2, 'joinGapSeconds': 3,
               'recordings': 8, 'rallies': 322, 'seedCount': 3, 'policyCount': len(rows),
               'scenarios': ['binary', 'boundary'], 'fullFootageMinutes': full_minutes,
               'axisF1RangePercent': [lower, upper], 'labeledPolicyIds': labels, 'points': points,
               'automaticBaselines': {key: {'playbackMinutes': 0, 'F1Percent': value['F1_padP_coreR'] * 100}
                                      for key, value in baselines.items()},
               'scopeNote': 'Current production and individual models only; omitted shipped/refit scopes remain in full report.',
               'interpretation': 'Hypothetical perfect-human outcomes; fixed review rules; no learned needs-review head.'}
    manifest = destination / 'manifest.json'
    with manifest.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'manifest': identity(manifest), 'files': files, 'policyCount': len(rows)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    main(args.root)
