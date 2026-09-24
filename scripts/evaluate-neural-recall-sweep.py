#!/usr/bin/env python3
"""Evaluate a frozen, hash-bound saved-score bundle at integer recall floors90..100."""
from pathlib import Path
import argparse
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
from analysis import neural_recall_sweep as sweep
from analysis.neural_recall_sweep_adapters import bind, load_score_shards


def load_bundle(path, evidence):
    bundle_ref = sweep.identity(path)
    bind(bundle_ref, evidence)
    bundle = sweep.read(path)
    sweep.require(bundle['kind'] == 'recall-floor-sweep-bundle-v1'
                  and bundle['floors'] == list(sweep.FLOORS) and bundle['epochs'] == list(sweep.EPOCHS)
                  and bundle['targetPaddingSeconds'] == 2 and bundle['joinGapSeconds'] == 3
                  and bundle['protectedTestOpened'] is False, 'Frozen sweep contract differs')
    bind(bundle['protocol'], evidence)
    sweep.require(bundle['code'], 'Numerical sources must be frozen in the input bundle')
    for ref in bundle['code']:
        bind(ref, evidence)
    for name in ('analysis/neural_recall_sweep.py', 'analysis/neural_recall_sweep_adapters.py',
                 'analysis/neural_development.py', 'analysis/neural_evaluation.py', 'analysis/crop_evaluation.py',
                 'analysis/neural_recall_operating_point.py', 'analysis/neural_expanded_development.py',
                 'analysis/decoder.py', 'analysis/config.py', 'analysis/schema.py', 'analysis/metrics.py',
                 'scripts/evaluate-neural-recall-sweep.py'):
        sweep.require(sweep.identity(REPO / name) in bundle['code'], 'Sweep numerical dependency unbound: ' + name)
    records = sweep.read(bind(bundle['records'], evidence))
    records = records['records'] if isinstance(records, dict) else records
    examples = {e.id: e for e in sweep.examples_from_rows(records)}
    provenance = {r['id']: r['reviewStatus'] for r in records}
    sweep.require(all(status in ('manually-reviewed', 'model-feedback-unreviewed') for status in provenance.values()),
                  'Recording review provenance must be explicit')
    return bundle, examples, provenance


def subset(examples, ids):
    sweep.require(ids and len(set(ids)) == len(ids) and set(ids) <= set(examples), 'Invalid declared recording subset')
    return [examples[key] for key in ids]


def run_bundle(bundle_path, output):
    evidence = {}
    bundle, examples, provenance = load_bundle(bundle_path, evidence)
    output_cells = []
    for cell in bundle['cells']:
        sweep.require(all(cell[k] for k in ('model', 'variant', 'selectionDesign')) and cell['folds'], 'Missing cell identity/folds')
        panel_specs = {p['panelId']: p for p in cell['panels']}
        sweep.require(len(panel_specs) == len(cell['panels']), 'Duplicate panel declaration')
        collected = {key: [] for key in panel_specs}
        folds = []
        for fold in cell['folds']:
            selection_examples = subset(examples, fold['selectionRecordingIds'])
            sweep.require(all(provenance[e.id] == 'manually-reviewed' for e in selection_examples),
                          'Unreviewed model feedback cannot select a calibrated operating point')
            inner = load_score_shards(selection_examples, fold['selectionScores'], evidence=evidence)
            candidates = sweep.build_candidate_table(selection_examples, inner)
            decisions = sweep.select_floors(candidates)
            held = {}
            for panel in fold['panels']:
                spec = panel_specs[panel['panelId']]
                rows = subset(examples, panel['recordingIds'])
                sweep.require(all(provenance[e.id] == spec['reviewStatus'] for e in rows), 'Mixed panel review provenance')
                sweep.require(spec['sourcePolicy'] != 'source-held' or not {e.group for e in rows} & {e.group for e in selection_examples},
                              'A source-held evaluation panel overlaps operating-point selection sources')
                scores = load_score_shards(rows, panel['scores'], evidence=evidence,
                                          source_policy=spec['sourcePolicy'], require_all_epochs=False)
                evaluated = sweep.evaluate_selected(rows, scores, decisions, spec['labelPolicy'])
                held[panel['panelId']] = evaluated
                collected[panel['panelId']].append(evaluated)
            folds.append({'foldId': fold['foldId'], 'candidates': candidates, 'decisions': decisions, 'panels': held})
        panels = {}
        for name, spec in panel_specs.items():
            rows = subset(examples, spec['expectedRecordingIds'])
            gold = sweep.gold_signature(sweep.panel_rows(rows, {e.id: [] for e in rows}, spec['labelPolicy']), spec['labelPolicy'])
            panels[name] = {**spec, 'result': sweep.pool_fold_results(collected[name], gold, spec['labelPolicy'])}
        output_cells.append({k: cell[k] for k in ('model', 'variant', 'seed', 'selectionDesign')} | {'folds': folds, 'panels': panels})
    comparisons = []
    for request in bundle['comparisons']:
        cells = [c for c in output_cells if all(c[k] == request[k] for k in ('model', 'variant', 'selectionDesign'))]
        by_seed = {c['seed']: c for c in cells}
        sweep.require(len(by_seed) == len(cells) and set(by_seed) == set(request['seeds']), 'Comparison seed population differs')
        selected = [{k: by_seed[seed][k] for k in ('model', 'variant', 'seed', 'selectionDesign')} |
                    {'panelId': request['panelId'], 'result': by_seed[seed]['panels'][request['panelId']]['result']}
                    for seed in request['seeds']]
        comparisons.append({**request, 'summary': sweep.summarize_seed_cells(selected, request['seeds'])})
    for ref in list(evidence.values()):
        bind(ref)
    result = {'kind': 'recall-floor-sweep-results-v1', 'bundle': sweep.identity(bundle_path),
              'protocol': bundle['protocol'], 'floors': list(sweep.FLOORS), 'cells': output_cells, 'comparisons': comparisons,
              'references': list(evidence.values()),
              'heldPanelsUsedForSelection': False, 'trainingPerformed': False, 'gpuUsed': False, 'protectedTestOpened': False}
    sweep.write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sweep.require(str(args.output.resolve()).startswith(private_value('private-reference-0060')), 'Sweep outputs must use direct NAS')
    run_bundle(args.bundle, args.output)


if __name__ == '__main__':
    main()
