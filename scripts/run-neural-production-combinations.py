#!/usr/bin/env python3
"""Register/run a fixed production+NN interval matrix; no fitting or tuning."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
from analysis import neural_production_combinations as core
from analysis.neural_evaluation import evaluate_predictions

ROOT = Path(private_value('private-reference-0057'))
SEEDS = (3407, 1729, 20260918)
ANCHORS = ('productionDefault', 'shippedUnion', 'refitUnion')
PRODUCTION = ('shippedPrevious', 'shippedV2', 'shippedUnion', 'refitPrevious', 'refitV2',
              'refitUnion', 'productionDefault', 'productionBalanced', 'productionConservative')
NN = (
    ('compact_boost', 'tcn', 'reviewed_export', 'short_boost', False),
    ('compact_keep', 'tcn', 'reviewed_export', 'baseline', True),
    ('dino_global', 'dino_tcn', 'draft', 'global_control', False),
    ('dino_boost', 'dino_tcn', 'reviewed_export', 'short_boost', False),
    ('dino_keep', 'dino_tcn', 'reviewed_export', 'baseline', True),
    ('compact_baseline', 'tcn', 'reviewed_export', 'baseline', False),
    ('dino_baseline', 'dino_tcn', 'reviewed_export', 'baseline', False),
)
PARTICIPANTS = tuple(r[0] for r in NN[:5])
PAIRS = {'best_f1_pair': ('compact_boost', 'dino_global'),
         'recovery_pair': ('compact_keep', 'dino_boost')}
SOURCE_PATHS = (
    'analysis/neural_production_combinations.py',
    'scripts/run-neural-production-combinations.py',
    'scripts/audit-neural-combination-accounting.py',
    'scripts/prepare-neural-production-comparison.py',
    'analysis/neural_evaluation.py', 'analysis/crop_evaluation.py',
    'analysis/schema.py', 'analysis/metrics.py',
    'analysis/tests/test_neural_production_combinations.py',
    'analysis/tests/test_neural_combination_accounting.py',
    'analysis/tests/test_neural_production_comparison_input.py',
    'analysis/tests/test_neural_combination_runner.py',
)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    path = Path(path)
    return {'path': str(path), 'sha256': digest(path), 'sizeBytes': path.stat().st_size}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write('\n')


def verified(ref):
    require(digest(ref['path']) == ref['sha256'], 'Bound artifact changed: '+ref['path'])
    return read(ref['path'])


def gold(rows):
    return [{'start': float(x['start']), 'end': float(x['end']), 'tags': x.get('tags', [])} for x in rows]


def same_labels(a, b):
    return (a['id'] == b['id'] and a['sourceGroup'] == b['sourceGroup']
            and gold(a['rallies']) == gold(b['rallies'])
            and core.serial(core.intervals(a.get('ignoredIntervals', []))) ==
                core.serial(core.intervals(b.get('ignoredIntervals', []))))


def contains_binding(value, expected):
    if isinstance(value, dict):
        return (value.get('path') == expected['path'] and value.get('sha256') == expected['sha256']) or any(
            contains_binding(child, expected) for child in value.values())
    return isinstance(value, list) and any(contains_binding(child, expected) for child in value)


def verify_reference_result(result, report):
    keys = ('cohort', 'kind', 'lossArm', 'seed', 'rescueVariant')
    matched = [row for row in report['results'] if all(row.get(k) == result.get(k) for k in keys)]
    require(len(matched) == 1 and matched[0] == result,
            'Neural source result is not the exact uniquely audited report payload')


def configurations():
    rows = [{'id': name, 'family': 'production', 'anchor': name, 'neuralIds': []} for name in PRODUCTION]
    rows += [{'id': name, 'family': 'neural', 'anchor': None, 'neuralIds': [name]} for name, *_ in NN]
    for name, pair in PAIRS.items():
        for family in ('nn_union', 'nn_intersection'):
            rows.append({'id': f'{name}--{family}', 'family': family, 'anchor': None, 'neuralIds': list(pair)})
    for anchor in ANCHORS:
        for nn in PARTICIPANTS:
            for family in core.FAMILIES:
                rows.append({'id': f'{anchor}--{nn}--{family}', 'family': family,
                             'anchor': anchor, 'neuralIds': [nn]})
        for name, pair in PAIRS.items():
            for family in ('three_union', 'majority'):
                rows.append({'id': f'{anchor}--{name}--{family}', 'family': family,
                             'anchor': anchor, 'neuralIds': list(pair)})
    require(len(rows) == len({r['id'] for r in rows}) == 107, 'Unexpected fixed recipe inventory')
    return rows


def register(output):
    output = Path(output)
    require(not (output/'registration.json').exists(), 'Registration already exists')
    production_path = output/'production-input.json'
    production = read(production_path)
    require(production['protectedTestOpened'] is False and production['productionChanged'] is False
            and production['trainingPerformed'] is False, 'Production adapter scope differs')
    production_by = {r['id']: r for r in production['recordings']}
    require(len(production_by) == len(production['recordings']) == 8, 'Production scope differs')
    sources, normalized, bindings = {}, {}, []
    references, reference_reports, reference_contracts = {}, {}, {}
    for study in ('2026-09-19-short-boost-transfer', '2026-09-19-keep-rescue'):
        leaf = ROOT/study/'study'
        references[study] = {name: identity(leaf/name) for name in ('preregistration.json', 'summary.json', 'report.json')}
        reg = read(leaf/'preregistration.json')
        report = read(leaf/'report.json')
        summary = read(leaf/'summary.json')
        require(canonical(reg['contract']) == reg['sha256'] == report['contractSha256'] == summary['contractSha256']
                and contains_binding(summary, references[study]['report.json']), 'Reference contract/report binding differs')
        reference_reports[study], reference_contracts[study] = report, reg['sha256']
        require(summary['protectedTestOpened'] is False and summary['productionPromotionAllowed'] is False,
                'Reference scope differs')
        if study.endswith('keep-rescue'):
            require(summary['passed'] is True, 'Keep-rescue independent summary is not passed')
        else:
            for name in ('tensor-scaler-audit-v1.json', 'interval-audit-v1.json'):
                # Prior study preserves independent audits at the study root.
                path = leaf/name
                if not path.exists():
                    path = leaf.parent/name
                audit = read(path)
                require(audit['passed'] is True and audit['contractSha256'] == reg['sha256']
                        and contains_binding(audit, references[study]['report.json']),
                        'Reference independent audit did not pass or binds a different report')
                references[study][name] = identity(path)
    first_records = None
    for name, kind, cohort, arm, rescue in NN:
        normalized[name] = {}
        for seed in SEEDS:
            study = '2026-09-19-keep-rescue' if rescue else '2026-09-19-short-boost-transfer'
            leaf = ROOT/study/'study'
            filename = f'result-{cohort}-{kind}-{arm}-'+('selected-' if rescue else '')+f'{seed}.json'
            path = leaf/filename
            result = read(path)
            require(result['kind'] == kind and result['cohort'] == cohort and result['seed'] == seed
                    and result['lossArm'] == arm and result['contractSha256'] == reference_contracts[study]
                    and (not rescue or result['rescueVariant'] == 'selected'), 'Neural result identity differs')
            verify_reference_result(result, reference_reports[study])
            rows = {r['id']: r for r in result['predictions']}
            require(set(rows) == set(production_by) and len(rows) == 8, 'Neural evaluation population differs')
            if first_records is None:
                first_records = [{key: r[key] for key in ('id', 'sourceGroup', 'durationSeconds', 'rallies', 'ignoredIntervals')}
                                 for _, r in sorted(rows.items())]
            for r in first_records:
                require(same_labels(r, rows[r['id']]) and r['durationSeconds'] == rows[r['id']]['durationSeconds'],
                        'Neural gold/duration revision differs')
                p = production_by[r['id']]
                require(same_labels(r, p) and abs(r['durationSeconds']-p['durationSeconds']) <= 5e-7,
                        'Production gold/duration revision differs')
                require(r['durationSeconds'] == p['featureMetadataDurationSeconds'],
                        'Canonical exact feature duration differs from production metadata')
                for values in [rows[r['id']]['predictions'], *p['cores'].values(),
                               r['rallies'], r.get('ignoredIntervals', [])]:
                    require(all(x.start >= 0 and x.end <= r['durationSeconds']+1e-9
                                for x in core.intervals(values)), 'Raw input interval outside video bounds')
                require(p['environment'] in ('grass', 'indoor') and r['sourceGroup'] != private_value('source-group-008'),
                        'Forbidden evaluation group/environment')
            normalized[name][str(seed)] = {rid: core.serial(core.intervals(r['predictions'])) for rid, r in rows.items()}
            bindings.append({'model': name, 'seed': seed, **identity(path)})
            sources[f'{name}:{seed}'] = result['evaluation']['primary']
    groups = sorted({r['sourceGroup'] for r in first_records})
    require(len(groups) == 4 and sum(len(r['rallies']) for r in first_records) == 322, 'Gold scope differs')
    recipes = configurations()
    normalized_path = output/'neural-input.json'
    write(normalized_path, {'records': first_records, 'predictions': normalized,
                           'sourcePrimaryMetrics': sources, 'sourceBindings': bindings,
                           'durationNormalization': 'Use exact cached metadata durations for every row; production manifest display rounding differs by at most 0.000000333334 seconds.'})
    protocol = REPO/'docs/research/neural-production-combinations-protocol-2026-09-19.md'
    snapshot = output/'protocol-initial.md'
    with snapshot.open('xb') as f:
        f.write(protocol.read_bytes())
    contract = {
        'kind': 'fixed-production-neural-combination-development-v1',
        'createdAt': datetime.now(timezone.utc).isoformat(),
        'primaryMetric': 'F1_padP_coreR', 'targetPaddingSeconds': 2, 'joinGapSeconds': 3,
        'paddingSweep': list(core.PADS), 'seeds': list(SEEDS), 'sourceGroups': groups,
        'protectedTestOpened': False, 'productionPromotionAllowed': False,
        'newFits': 0, 'records': 8, 'rallies': 322,
        'productionInput': identity(production_path), 'neuralInput': identity(normalized_path),
        'referenceStudies': references, 'protocol': identity(snapshot),
        'code': {p: identity(REPO/p) for p in SOURCE_PATHS},
        'configurations': recipes, 'anchors': list(ANCHORS), 'participants': list(PARTICIPANTS),
        'reviewModes': ['suppression_review', 'bidirectional_review'],
        'automaticConfigurations': 107, 'automaticResultCells': 303, 'reviewResultCells': 90,
        'conservativeScreen': {'meanF1MinimumGain': .02, 'meanRecallMinimumDelta': -.005,
                               'meanLongRecallMinimumDelta': -.005, 'meanEventF1MinimumDelta': -.01,
                               'maximumNewCompleteLossesAnySeed': 0, 'maximumWorsenedRalliesAnySeed': 0},
        'actualAppExportsRole': 'Separate fidelity diagnostic; never substituted for canonical primary ranking.',
        'reviewOracleRole': 'Perfect correction of disputed export seconds only; ground-truth upper bound, excluded from automatic ranking.',
    }
    write(output/'registration.json', {'contract': contract, 'sha256': canonical(contract)})
    print(json.dumps({'registered': True, 'contractSha256': canonical(contract),
                      'configurations': 107, 'automaticCells': 303, 'reviewCells': 90}, indent=2))


def load_contract(output):
    reg = read(output/'registration.json')
    c = reg['contract']
    require(canonical(c) == reg['sha256'], 'Registration changed')
    for ref in c['code'].values():
        require(digest(ref['path']) == ref['sha256'], 'Registered code changed')
    for refs in c['referenceStudies'].values():
        for ref in refs.values():
            require(digest(ref['path']) == ref['sha256'], 'Reference study artifact changed')
    production, neural = verified(c['productionInput']), verified(c['neuralInput'])
    for ref in neural['sourceBindings']:
        require(digest(ref['path']) == ref['sha256'], 'Neural result changed')
    require(digest(c['protocol']['path']) == c['protocol']['sha256'], 'Frozen protocol changed')
    return reg, production, neural


def predictions_for(recipe, seed, production, neural):
    output = {}
    for rec in neural['records']:
        rid, kind = rec['id'], recipe['family']
        p = production[rid]['cores']
        ns = [neural['predictions'][n][str(seed)][rid] for n in recipe['neuralIds']]
        anchor = recipe['anchor']
        if kind == 'production':
            value = p[anchor]
        elif kind == 'neural':
            value = ns[0]
        elif kind == 'nn_union':
            value = core.union(*ns)
        elif kind == 'nn_intersection':
            value = core.intersection(*ns)
        elif kind == 'three_union':
            value = core.union(p[anchor], *ns)
        elif kind == 'majority':
            value = core.majority(p[anchor], *ns)
        else:
            prefix = 'refit' if anchor == 'refitUnion' else 'shipped'
            value = core.combine(p[anchor], ns[0], p[prefix+'Previous'], p[prefix+'V2'],
                                 rec['durationSeconds'], kind)
        output[rid] = core.serial(core.intervals(value))
    return output


def with_predictions(records, predictions):
    return [{**r, 'predictions': predictions[r['id']]} for r in records]


def audit_module():
    spec = importlib.util.spec_from_file_location('independent_combination_accounting',
        REPO/'scripts/audit-neural-combination-accounting.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact_guards(evaluation):
    g = evaluation['guardrails']
    coverage = g['primaryExportCoverage']
    long = [r for r in coverage['rallies'] if r['end']-r['start'] > 3]
    short = [r for r in coverage['rallies'] if r['end']-r['start'] <= 3]
    denom = sum(r['evaluableCoreSeconds'] for r in long)
    return {'eventF1': g['eventF1'], 'completeLosses': coverage['completeRallyLosses'],
            'partialLosses': coverage['partialRallyLosses'],
            'incompleteLosses': coverage['completeRallyLosses']+coverage['partialRallyLosses'],
            'shortCompleteLosses': sum(r['completelyLost'] for r in short),
            'shortIncompleteLosses': sum(not r['fullyCovered'] for r in short),
            'longR_core': sum(r['retainedCoreSeconds'] for r in long)/denom if denom else 0.}


def paired_guards(evaluation, baseline):
    a = {(r['recordingId'], r['truthIndex']): r for r in baseline['guardrails']['primaryExportCoverage']['rallies']}
    b = {(r['recordingId'], r['truthIndex']): r for r in evaluation['guardrails']['primaryExportCoverage']['rallies']}
    require(set(a) == set(b), 'Paired rally universe differs')
    lost, gained = [], []
    for key in a:
        delta = b[key]['retainedCoreSeconds']-a[key]['retainedCoreSeconds']
        if delta < -1e-9:
            lost.append({'recordingId': key[0], 'truthIndex': key[1], 'lostSeconds': -delta,
                         'newComplete': not a[key]['completelyLost'] and b[key]['completelyLost'],
                         'newPartial': a[key]['fullyCovered'] and b[key]['partiallyLost'],
                         'short': b[key]['end']-b[key]['start'] <= 3})
        elif delta > 1e-9:
            gained.append({'recordingId': key[0], 'truthIndex': key[1], 'recoveredSeconds': delta})
    ga, gb = compact_guards(baseline), compact_guards(evaluation)
    return {'newCompleteLosses': sum(r['newComplete'] for r in lost),
            'newPartialLosses': sum(r['newPartial'] for r in lost), 'worsenedRallies': len(lost),
            'additionalLostCoreSeconds': sum(r['lostSeconds'] for r in lost),
            'recoveredCoreSeconds': sum(r['recoveredSeconds'] for r in gained),
            'newShortCompleteLosses': sum(r['newComplete'] and r['short'] for r in lost),
            'guardrailDelta': {key: gb[key]-ga[key] for key in ga},
            'lostRallyIdentities': lost, 'recoveredRallyIdentities': gained}


def audit_all_scopes(auditor, records, rows, overrides=None):
    checks = [auditor.audit_duration_records(records, rows, overrides)]
    groups = sorted({r['sourceGroup'] for r in records})
    for group in groups:
        subset = [r for r in records if r['sourceGroup'] == group]
        scoped = None if overrides is None else {r['id']: overrides[r['id']] for r in subset}
        checks.append(auditor.audit_duration_records(subset, core.duration_rows(subset, scoped), scoped))
    for rec in records:
        scoped = None if overrides is None else {rec['id']: overrides[rec['id']]}
        checks.append(auditor.audit_duration_records([rec], core.duration_rows([rec], scoped), scoped))
    return {'scopeCount': len(checks), 'checks': checks}


def run(output):
    reg, production, neural = load_contract(output)
    require(not (output/'report.json').exists(), 'Completed study exists')
    c, started = reg['contract'], time.perf_counter()
    began = datetime.now(timezone.utc).isoformat()
    production = {r['id']: r for r in production['recordings']}
    auditor = audit_module()
    base, artifacts, reviews, fidelity = {}, [], [], []
    for recipe in c['configurations']:
        for seed in ([None] if recipe['family'] == 'production' else c['seeds']):
            pred = predictions_for(recipe, seed, production, neural)
            records = with_predictions(neural['records'], pred)
            evaluation = evaluate_predictions(records)
            durations = core.duration_rows(records)
            for actual, measured in zip(evaluation['padding'], durations):
                for key in ('P_pad', 'R_core', 'F1_padP_coreR', 'paddedModelExportSeconds',
                            'paddedHumanExportSeconds', 'exportDurationDifferenceSeconds'):
                    require(abs(actual[key]-measured[key]) <= 1e-7, 'Canonical accounting disagreement')
            audit = audit_all_scopes(auditor, records, durations)
            if recipe['family'] == 'neural':
                source = neural['sourcePrimaryMetrics'][f"{recipe['neuralIds'][0]}:{seed}"]
                for key in ('P_pad', 'R_core', 'F1_padP_coreR', 'paddedModelExportSeconds'):
                    require(abs(evaluation['primary'][key]-source[key]) <= 1e-7, 'Standalone source metric replay differs')
            result = {'configuration': recipe, 'seed': seed, 'contractSha256': reg['sha256'],
                      'predictions': pred, 'evaluation': evaluation, 'durationMetrics': durations,
                      'guardrails': compact_guards(evaluation), 'independentAccounting': audit}
            if recipe['family'] == 'production':
                base[recipe['id']] = evaluation
            else:
                anchor = recipe['anchor'] or 'productionDefault'
                result['pairedAgainst'] = anchor
                result['pairedGuardrails'] = paired_guards(evaluation, base[anchor])
            path = output/'results'/f"{recipe['id']}--{seed if seed is not None else 'fixed'}.json"
            write(path, result)
            artifacts.append(identity(path))
        print(f"COMPLETED configuration {len({Path(x['path']).name.rsplit('--', 1)[0] for x in artifacts})}/107", flush=True)
    for anchor in c['anchors']:
        records = with_predictions(neural['records'], {rid: r['cores'][anchor] for rid, r in production.items()})
        automatic = core.duration_rows(records)
        for name in c['participants']:
            for seed in c['seeds']:
                for mode in c['reviewModes']:
                    value = core.review_diagnostic(records, neural['predictions'][name][str(seed)], mode)
                    audit = audit_all_scopes(auditor, records, value['oracleDurationMetrics'], value['oracleExports'])
                    for row, origin in zip(value['oracleDurationMetrics'], automatic):
                        require(row['incorrectExportSeconds'] <= origin['incorrectExportSeconds']+1e-7
                                and row['missedCoreSeconds'] <= origin['missedCoreSeconds']+1e-7,
                                'Perfect disputed review worsens errors')
                    path = output/'reviews'/f'{anchor}--{name}--{mode}--{seed}.json'
                    write(path, {'anchor': anchor, 'neuralId': name, 'seed': seed,
                                 'contractSha256': reg['sha256'], 'automaticDurationMetrics': automatic,
                                 **value, 'independentAccounting': audit})
                    reviews.append(identity(path))
        print(f'COMPLETED review anchor {anchor}', flush=True)
    for name in ('shippedUnion', 'productionDefault', 'productionBalanced', 'productionConservative'):
        records = with_predictions(neural['records'], {rid: r['cores'][name] for rid, r in production.items()})
        overrides = {rid: r['productExportsByPadding'][name] for rid, r in production.items()}
        rows = core.duration_rows(records, overrides)
        canonical_rows = core.duration_rows(records)
        parity = []
        for pad in core.PADS:
            app_only = canonical_only = 0.
            for rec in records:
                app = core.intersection(overrides[rec['id']][str(pad)],
                    core.difference([(0, rec['durationSeconds'])], rec.get('ignoredIntervals', [])))
                canon = core.export(rec['predictions'], rec, pad)
                app_only += core.duration(core.difference(app, canon))
                canonical_only += core.duration(core.difference(canon, app))
            parity.append({'padding': pad, 'appOnlySeconds': app_only, 'canonicalOnlySeconds': canonical_only})
        fidelity.append({'variant': name, 'actualAppDurationMetrics': rows, 'canonicalDurationMetrics': canonical_rows,
                         'exportDifferences': parity, 'exports': overrides,
                         'independentAccounting': audit_all_scopes(auditor, records, rows, overrides)})
    # Recheck every registration dependency after computation, before completion.
    require(load_contract(output)[0] == reg, 'Registration changed during execution')
    require(len(artifacts) == 303 and len(reviews) == 90, 'Incomplete fixed matrix')
    report = {'kind': c['kind'], 'status': 'completed-fixed-combinations', 'contractSha256': reg['sha256'],
              'startedAt': began, 'completedAt': datetime.now(timezone.utc).isoformat(),
              'wallSeconds': time.perf_counter()-started, 'protectedTestOpened': False,
              'productionChanged': False, 'newFits': 0, 'automaticResults': artifacts,
              'reviewResults': reviews, 'actualAppFidelity': fidelity,
              'counts': {'configurations': 107, 'automaticCells': 303, 'reviewCells': 90,
                         'accountingScopesPerCell': 13, 'paddingCasesPerScope': 4}}
    write(output/'report.json', report)
    print(json.dumps({'completed': True, 'report': identity(output/'report.json'), 'counts': report['counts']}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'2026-09-19-production-combinations')
    p.add_argument('--register', action='store_true')
    args = p.parse_args()
    register(args.output) if args.register else run(args.output)


if __name__ == '__main__':
    main()
