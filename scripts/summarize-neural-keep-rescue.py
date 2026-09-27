#!/usr/bin/env python3
"""Prospective CPU-only keep-rescue artifact audit and descriptive report.

No fitting, automatic registration or default study path. Full mode requires a
completed explicitly registered study and its already-audited reference study.
The keep proposal is a heuristic; export coverage is not invertible rally truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def import_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO/'scripts'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transfer = import_script('keep_rescue_transfer_reporting', 'summarize-neural-short-boost-transfer.py')
interval = import_script('keep_rescue_independent_intervals', 'audit-neural-short-boost-intervals.py')
tensor = import_script('keep_rescue_reference_tensor_helpers', 'audit-neural-short-boost-tensors.py')
expanded, balanced, helpers = transfer.expanded, transfer.balanced, transfer.helpers
require, mean = transfer.require, transfer.mean
KINDS = ('tcn', 'dino_tcn')
VARIANTS = ('reference', 'selected')
COHORTS = ('exact', 'draft', 'reviewed_export')
OPTIONS = (None, .35, .5, .65, .8)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    return {'path': str(path), 'sha256': digest(path)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verified(bound):
    require(digest(bound['path']) == bound['sha256'], 'Bound file changed: '+bound['path'])
    return read(bound['path'])


def runs(mask):
    # Independent edge-vector implementation; no helper or baseline.segments.
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def independent_proposals(example, scores, settings, threshold):
    if threshold is None:
        return []
    require(threshold in OPTIONS[1:] and settings['smoothing'] in (.5, 1.), 'Unregistered rescue option/smoothing')
    output = []
    for left, right in runs(example.valid):
        values = scores[left:right, 3]
        width = min(max(1, round(settings['smoothing']*4)), len(values))
        # Reproduce the registered frozen edge-padded float32 moving mean.
        # Proposal runs, edge exclusions, erosion and unions below are separate
        # implementations, not calls into the prospective rescue helper.
        if width > 1:
            padded = np.pad(values, (width//2, width-1-width//2), mode='edge')
            values = np.convolve(padded, np.ones(width, np.float32)/width, mode='valid').astype(np.float32)
        for start_index, end_index in runs(values >= np.float32(threshold)):
            if start_index == 0 or end_index == right-left:
                continue
            start = max(0., float(example.times[left+start_index])-.125)
            end = min(example.duration, float(example.times[left+end_index-1])+.125)
            if start <= 0 or end >= example.duration or any(start <= i.end and end >= i.start for i in example.ignored):
                continue
            start, end = start+2., end-2.
            if 0 < end-start <= 3.:
                output.append((start, end))
    return output


def independent_decode(example, probabilities, decoder, threshold):
    from analysis.neural_development import decode
    original = [r.to_dict() for r in decode(example, probabilities, decoder)]
    if threshold is None:
        return original
    proposals = independent_proposals(example, probabilities, decoder, threshold)
    base = [(r['start'], r['end']) for r in original]
    # No proposal/novel duration preserves the original list exactly, including
    # its cut identities. A positive gap is never joined in this core stage.
    if not interval.boolean_intervals(proposals, base, 'difference'):
        return original
    return [{'start': start, 'end': end} for start, end in interval.boolean_intervals(base, proposals, 'union')]


def evaluation_row(example, predictions):
    return {'id': example.id, 'sourceGroup': example.group, 'durationSeconds': example.duration,
            'rallies': [i.to_dict() for i in example.truth], 'ignoredIntervals': [i.to_dict() for i in example.ignored],
            'predictions': predictions}


def select_option(candidates):
    """Independent stable first-option tie and infeasible highest-F1 fallback."""
    require(len(candidates) == 5 and [r['keepThreshold'] for r in candidates] == list(OPTIONS)
            and [r['optionIndex'] for r in candidates] == list(range(5)), 'Option order or explicit no-op differs')
    require(all(np.isfinite(r['innerF1_padP_coreR']) and np.isfinite(r['innerR_core']) for r in candidates), 'Nonfinite inner score')
    eligible = [r for r in candidates if r['innerR_core'] >= .95]
    pool = eligible if eligible else candidates
    chosen = pool[0]
    for row in pool[1:]:
        if row['innerF1_padP_coreR'] > chosen['innerF1_padP_coreR']:
            chosen = row
    return {**chosen, 'recallEligibilityPassed': bool(eligible), 'recallEligibilityFloor': .95,
            'candidates': candidates}


def replay_options(examples, probabilities, decoder):
    from analysis.neural_keep_rescue import decode_with_keep_rescue
    from analysis.crop_evaluation import RecordingIntervals, evaluate_f1_pad_p_core_r
    from analysis.schema import Interval
    require(set(probabilities) == {e.id for e in examples}, 'Inner option pool population differs')
    candidates, independent_checks = [], []
    for index, threshold in enumerate(OPTIONS):
        rows = []
        for e in examples:
            cuts = independent_decode(e, probabilities[e.id], decoder, threshold)
            shared = [r.to_dict() for r in decode_with_keep_rescue(e, probabilities[e.id], decoder, keep_threshold=threshold)]
            require(cuts == shared, 'Independent proposal/core union differs from registered decoder')
            rows.append(evaluation_row(e, cuts))
        canonical_rows = [RecordingIntervals(e.id, 'development', e.duration, e.truth,
                          tuple(Interval(r['start'], r['end']) for r in row['predictions']), e.ignored)
                          for e, row in zip(examples, rows)]
        score = evaluate_f1_pad_p_core_r(canonical_rows, [2.], 3.)[0]
        independent = interval.pooled_metric([interval.parse_record(r) for r in rows], 2)
        interval.compare_metric(independent, score, 'keep-rescue/inner-option')
        candidates.append({'keepThreshold': threshold, 'optionIndex': index,
                           'innerF1_padP_coreR': score['F1_padP_coreR'], 'innerR_core': score['R_core']})
        independent_checks.append({'optionIndex': index, 'keepThreshold': threshold,
                                   'pooledEndpointMetric': independent, 'decodedCoresSha256': canonical(rows)})
    return select_option(candidates), independent_checks


def result_key(row):
    return row['cohort'], row['kind'], row['rescueVariant'], row['seed']


def verify_grid(results, contract):
    cohorts = contract['cohorts']
    require(cohorts and cohorts == [c for c in COHORTS if c in cohorts], 'Cohorts must be an explicit ordered subset')
    expected = {(c, k, v, s) for c in cohorts for k in KINDS for v in VARIANTS for s in contract['seeds']}
    require(len(results) == len(expected) and {result_key(r) for r in results} == expected,
            'Incomplete or duplicate keep-rescue grid')
    require(contract['kinds'] == list(KINDS) and all(r['lossArm'] == 'baseline' for r in results), 'Reference architecture/loss arm differs')
    require(contract['rescueOptions'] == list(OPTIONS) and contract['primaryMetric'] == 'F1_padP_coreR'
            and contract['targetPaddingSeconds'] == 2 and contract['joinGapSeconds'] == 3
            and contract['paddingSweep'] == [0, 1, 2, 3], 'Fixed option/metric/padding contract differs')
    return {result_key(r): r for r in results}


def replay_metrics(result):
    rows = result['predictions']
    computed = transfer.evaluate_predictions(rows)
    require(computed == result['evaluation'], 'Saved canonical evaluation differs')
    parsed = [interval.parse_record(r) for r in rows]
    scopes = {'pooled': interval.compare_scope(parsed, computed, 'keep-rescue/pooled')}
    for group, evaluation in computed['sourceGroups'].items():
        scopes['group:'+group] = interval.compare_scope([r for r in parsed if r['sourceGroup'] == group], evaluation, 'keep-rescue/group/'+group)
    for row in rows:
        scopes['recording:'+row['id']] = interval.compare_scope([interval.parse_record(row)], transfer.evaluate_predictions([row]), 'keep-rescue/recording/'+row['id'])
    return {'independentScopes': scopes, 'independentPaddingRows': 4*len(scopes)}


def summarize_results(results, contract):
    by_key = verify_grid(results, contract)
    aggregates, per_seed, comparisons, losses, interactions = [], [], [], [], []
    for cohort in contract['cohorts']:
        cohort_comparisons = []
        for kind in KINDS:
            for variant in VARIANTS:
                rows = []
                for seed in contract['seeds']:
                    result = by_key[(cohort, kind, variant, seed)]
                    evaluation = result['evaluation']
                    compact = {'cohort': cohort, 'kind': kind, 'rescueVariant': variant, 'seed': seed,
                               'origin': result['origin'], **expanded.compact_evaluation(evaluation),
                               'slices': balanced.event_slices(evaluation), 'selections': result['selections'],
                               'sourceGroups': {g: {**expanded.compact_evaluation(e), 'slices': balanced.event_slices(e)}
                                                for g, e in evaluation['sourceGroups'].items()}}
                    rows.append(compact)
                    per_seed.append(compact)
                aggregates.append({'cohort': cohort, 'kind': kind, 'rescueVariant': variant, 'seedCount': len(rows),
                    'meanSeedPrimary': expanded.metric_means([r['primary'] for r in rows]),
                    'meanSeedPadding': [{'paddingSecondsBeforeAndAfter': pad, **expanded.metric_means([r['padding'][pad] for r in rows])} for pad in range(4)],
                    'meanSeedEventF1': mean(r['eventF1'] for r in rows), 'meanSeedSlices': balanced.mean_slices([r['slices'] for r in rows])})
            candidates = [by_key[(cohort, kind, 'selected', seed)] for seed in contract['seeds']]
            references = [by_key[(cohort, kind, 'reference', seed)] for seed in contract['seeds']]
            pairs = []
            for candidate, reference in zip(candidates, references):
                a, b = candidate['evaluation'], reference['evaluation']
                pairs.append({'seed': candidate['seed'], **expanded.paired_evaluation(a, b),
                              'sliceDelta': balanced.slice_delta(balanced.event_slices(a), balanced.event_slices(b)),
                              'pointRecovery': {name: transfer.recovery_effect(a, b, name) for name in balanced.short.SLICES}})
                losses.append({'cohort': cohort, 'kind': kind, 'seed': candidate['seed'],
                               'candidate': balanced.event_slices(a, identities=True), 'reference': balanced.event_slices(b, identities=True),
                               'paired': {scope: helpers.coverage_comparison(a, b, scope) for scope in balanced.SCOPES}})
            comparison = {'cohort': cohort, 'kind': kind, 'name': f'{cohort}/{kind}/selected-minus-reference', 'perSeed': pairs,
                'meanSeedPrimaryDelta': expanded.metric_means([r['primaryDelta'] for r in pairs]),
                'meanSeedEventF1Delta': mean(r['eventF1Delta'] for r in pairs),
                'meanSeedSliceDelta': balanced.mean_slices([r['sliceDelta'] for r in pairs]),
                'sourceGroupMeanPairedDelta': {g: {f: mean(r['sourceGroups'][g][f] for r in pairs)
                    for f in ('F1_padP_coreRDelta', 'R_coreDelta', 'paddedModelExportSecondsDelta')} for g in contract['groups']},
                'standardF1Screen': expanded.feasibility(candidates, pairs, .95),
                'retentionRecoveryScreen': balanced.recovery_screen(candidates, references, contract['retentionRecoveryScreen'])}
            comparisons.append(comparison)
            cohort_comparisons.append(comparison)
        effects = []
        for seed in contract['seeds']:
            values = [by_key[(cohort, k, v, seed)]['evaluation'] for k, v in
                      (('dino_tcn', 'selected'), ('dino_tcn', 'reference'), ('tcn', 'selected'), ('tcn', 'reference'))]
            effects.append({'seed': seed, **transfer.difference_in_differences(*values),
                            'sourceGroups': {g: transfer.difference_in_differences(*(e['sourceGroups'][g] for e in values)) for g in contract['groups']}})
        fields = effects[0]['differenceInDifferences']
        interactions.append({'cohort': cohort, 'perSeed': effects,
            'meanSeedDifferenceInDifferences': {f: expanded.mean_available(r['differenceInDifferences'][f] for r in effects) for f in fields},
            'sourceGroupMeanDifferenceInDifferences': {g: {f: expanded.mean_available(r['sourceGroups'][g]['differenceInDifferences'][f] for r in effects)
                                                        for f in fields} for g in contract['groups']},
            'replication': {name: all(r[name]['screenPassedAndInnerFeasible'] for r in cohort_comparisons)
                            for name in ('standardF1Screen', 'retentionRecoveryScreen')},
            'positiveInteractionRequired': False})
    return {'aggregates': aggregates, 'perSeed': per_seed, 'comparisons': comparisons,
            'transferInteractions': interactions}, {'perSeed': losses}


def markdown(summary):
    lines = ['# Keep-head short-rally rescue', '',
             'CPU decoder-only development study; zero new fits. Baseline epochs and live/boundary decoders stay fixed. '
             'Each outer fold selects among an explicit no-op and four keep thresholds using inner predictions only. '
             'The eroded keep component is a heuristic proposal, not an exact inverse of export coverage.', '',
             'Primary ranking uses fixed 2s symmetric padding and strictly <3s gap joining, after ignored-time exclusions. '
             'Each seed pools recordings before scoring; averages below are means of the three seed scores.', '',
             '| Cohort | Architecture | Variant | F1_padP_coreR | R_core | P_pad | Event F1 |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in summary['aggregates']:
        p = row['meanSeedPrimary']
        lines.append(f"| {row['cohort']} | {row['kind']} | {row['rescueVariant']} | {p['F1_padP_coreR']:.6f} | {p['R_core']:.6f} | {p['P_pad']:.6f} | {row['meanSeedEventF1']:.6f} |")
    lines += ['', '| Cohort | Architecture | F1 change | Recall change | F1 screen | Retention screen |', '|---|---|---:|---:|---|---|']
    for row in summary['comparisons']:
        p = row['meanSeedPrimaryDelta']
        lines.append(f"| {row['cohort']} | {row['kind']} | {p['F1_padP_coreR']:+.6f} | {p['R_core']:+.6f} | {row['standardF1Screen']['screenPassedAndInnerFeasible']} | {row['retentionRecoveryScreen']['screenPassedAndInnerFeasible']} |")
    lines += ['', '| Cohort | Architecture | Variant | Slice | Complete losses | Partial losses | Core recall |', '|---|---|---|---|---:|---:|---:|']
    for row in summary['aggregates']:
        for name in ('all', 'duration_le_3s', 'duration_gt_3s', 'ace', 'service_fault'):
            value = row['meanSeedSlices']['primaryExportCoverage'][name]
            recall = f"{value['coreRecall']:.6f}" if value['coreRecall'] is not None else 'n/a'
            lines.append(f"| {row['cohort']} | {row['kind']} | {row['rescueVariant']} | {name} | {value['completeRallyLosses']:.2f} | {value['partialRallyLosses']:.2f} | {recall} |")
    lines += ['', '| Cohort | Architecture | Variant | Padding | P_pad | R_core | F1_padP_coreR | Model seconds | Human seconds | Difference |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary['aggregates']:
        for p in row['meanSeedPadding']:
            lines.append(f"| {row['cohort']} | {row['kind']} | {row['rescueVariant']} | {p['paddingSecondsBeforeAndAfter']} | {p['P_pad']:.6f} | {p['R_core']:.6f} | {p['F1_padP_coreR']:.6f} | {p['paddedModelExportSeconds']:.3f} | {p['paddedHumanExportSeconds']:.3f} | {p['exportDurationDifferenceSeconds']:+.3f} |")
    lines += ['', 'All threshold candidates, infeasible fallbacks, source-group effects, original-event loss identities and separate replication/interaction screens are retained in JSON. '
              'Three seeds and four source groups support descriptive development comparisons. No protected test or production promotion.']
    return '\n'.join(lines)+'\n'


def load_examples(manifest):
    """Only exact AV timestamps/metadata, frozen annotations and validity."""
    from types import SimpleNamespace
    from analysis.schema import Interval
    output = []
    for row in manifest['exactRows']:
        require(row['consent']['train'] is True and row['environment'] in ('grass', 'indoor')
                and row['sourceGroup'] not in manifest['protectedSourceGroups'], 'Forbidden exact input')
        cache = row['featureCaches']['audiovisual']
        require(digest(cache['path']) == cache['sha256'], 'Exact AV timeline source changed')
        with np.load(cache['path'], allow_pickle=False) as data:
            times = data['times'].astype(np.float64)
            duration = float(json.loads(str(data['metadata_json'].item()))['duration'])
        require(times.ndim == 1 and len(times) and np.isfinite(times).all() and np.isfinite(duration) and duration > 0
                and np.all(np.diff(times) > 0) and np.all(np.abs(np.diff(times)-.25) <= .10)
                and times.min() >= 0 and times.max() <= duration, 'Invalid nominal4Hz source timeline')
        truth = tuple(Interval(r['start'], r['end'], tuple(r.get('tags', []))) for r in row['rallies'])
        ignored = tuple(Interval(r['start'], r['end']) for r in row.get('ignoredIntervals', []))
        valid = np.ones(len(times), bool)
        for interval_ in ignored:
            valid &= ~((times >= interval_.start) & (times < interval_.end))
        output.append(SimpleNamespace(id=row['id'], group=row['sourceGroup'], times=times,
                                      duration=duration, truth=truth, ignored=ignored, valid=valid))
    return output


def expected_membership(manifest, cohort, excluded, held):
    require(cohort in COHORTS, 'Unknown cohort')
    active = {'exact': (), 'draft': ('draft',), 'reviewed_export': ('draft', 'coverage')}[cohort]
    return {'trainIds': [r['id'] for r in manifest['exactRows'] if r['sourceGroup'] not in excluded],
            'auxiliaryIds': {tier: [r['id'] for r in manifest[tier+'Rows'] if r['sourceGroup'] not in excluded] for tier in active},
            'validationIds': [r['id'] for r in manifest['exactRows'] if r['sourceGroup'] == held]}


def load_predictions(folder, epoch, examples, membership, *, manifest, kind, seed, source_contract, audited_fits):
    """Independent read-only artifact/shape/mask/fold proof; no tensor refitting."""
    completion = folder/'completed.json'
    require(audited_fits.get(completion.resolve()) == digest(completion), 'Completion no longer matches bound prior tensor audit')
    meta = read(completion)
    require(meta['contractSha256'] == source_contract and meta['kind'] == kind and meta['seed'] == seed
            and meta.get('lossArm', 'baseline') == 'baseline' and epoch in meta['epochs']
            and all(meta[k] == v for k, v in membership.items()) and meta['scalerTrainIds'] == membership['trainIds'],
            'Baseline checkpoint epoch/membership/identity differs')
    by_id = {r['id']: r for key in ('exactRows', 'draftRows', 'coverageRows') for r in manifest[key]}
    train = membership['trainIds']+[rid for ids in membership['auxiliaryIds'].values() for rid in ids]
    held = membership['validationIds']
    require(len(train) == len(set(train)) and len(held) == len(set(held))
            and not set(train) & set(held) and [e.id for e in examples] == held
            and not {by_id[r]['sourceGroup'] for r in train} & {e.group for e in examples}, 'Source group/record leakage')
    require(meta['trainGroups'] == sorted({by_id[r]['sourceGroup'] for r in membership['trainIds']})
            and meta['auxiliaryGroups'] == {tier: sorted({by_id[r]['sourceGroup'] for r in ids}) for tier, ids in membership['auxiliaryIds'].items()}
            and meta['validationGroups'] == sorted({e.group for e in examples}), 'Checkpoint group metadata differs')
    names = {f'{stem}-{ep}.npz' for ep in meta['epochs'] for stem in ('weights', 'predictions')}
    require(set(meta['artifacts']) == names, 'Original checkpoint inventory differs')
    paths = {name: folder/f'{name}-{epoch}.npz' for name in ('weights', 'predictions')}
    for path in paths.values():
        require(digest(path) == meta['artifacts'][path.name], 'Selected original NPZ changed')
    probabilities = {}
    with np.load(paths['predictions'], allow_pickle=False) as data:
        require(len(data.files) == len(held) and set(data.files) == set(held), 'NPZ contains missing or outer prediction rows')
        for e in examples:
            values = data[e.id]
            require(values.shape == (len(e.times), 4) and values.dtype == np.float32 and np.isfinite(values).all()
                    and np.all((values >= 0) & (values <= 1)) and np.all(values[~e.valid] == 0), 'Invalid probability shape/range/ignored mask')
            probabilities[e.id] = values.copy()
    return probabilities, {'completed': identity(completion), **{name: identity(path) for name, path in paths.items()},
                           'epoch': epoch, 'membership': membership}


def reference_copy(result, baseline, reference, contract_hash):
    expected = {**baseline, 'rescueVariant': 'reference', 'contractSha256': contract_hash,
                'origin': {'type': 'reused-keep-rescue-reference', 'studyPath': reference['path'],
                           'reportSha256': reference['reportSha256'], 'referenceContractSha256': reference['contractSha256'],
                           'sourceOrigin': baseline['origin'],
                           'sourceResult': {k: baseline[k] for k in ('cohort', 'kind', 'lossArm', 'seed')}}}
    require(result == expected, 'Matched baseline payload or nested provenance changed')


def audit_cell(result, baseline, examples, manifest, contract, previous_contract, audited_fits):
    cohort, kind, seed = (result[k] for k in ('cohort', 'kind', 'seed'))
    root, historical = tensor.validate_origin(baseline, Path(contract['referenceStudy']['path']), previous_contract['referenceStudy'])
    source_contract = previous_contract['referenceStudy']['contractSha256'] if historical else contract['referenceStudy']['contractSha256']
    expected_origin = {'type': 'rescued-reference-checkpoints', 'fitRoot': str(root),
                       'sourceContractSha256': source_contract, 'sourceOrigin': baseline['origin'],
                       'referenceStudy': contract['referenceStudy']}
    require(result['origin'] == expected_origin and result['baselineNoOpReplayPassed'] is True
            and result['newFits'] == 0, 'Selected rescue origin/replay/new-fit claim differs')
    old_selection = {r['heldSourceGroup']: r for r in baseline['selections']}
    selections = {r['heldSourceGroup']: r for r in result['selections']}
    require(len(baseline['selections']) == len(old_selection) == len(result['selections']) == len(selections) == 4
            and set(selections) == set(old_selection) == set(contract['groups']), 'Outer fold selections differ')
    all_rows, baseline_rows, checks = [], [], []
    for outer_index, outer in enumerate(contract['groups']):
        choice, fixed = selections[outer], old_selection[outer]
        require(choice['baselineSelection'] == fixed and choice['epoch'] == fixed['epoch']
                and choice['decoder'] == fixed['decoder'] and choice['epoch'] in contract['checkpointEpochs']
                and choice['decoder'] in contract['decoderCandidates'], 'Baseline epoch/decoder was reselected')
        inner, bindings = {}, []
        for inner_index, group in enumerate(g for g in contract['groups'] if g != outer):
            held = [e for e in examples if e.group == group]
            scores, binding = load_predictions(root/f'outer-{outer_index}'/f'inner-{inner_index}', choice['epoch'], held,
                expected_membership(manifest, cohort, {outer, group}, group), manifest=manifest, kind=kind,
                seed=seed, source_contract=source_contract, audited_fits=audited_fits)
            require(not set(inner) & set(scores), 'Duplicated inner validation recordings')
            inner.update(scores)
            bindings.append({'innerValidationGroup': group, **binding})
        fitting = [e for e in examples if e.group != outer]
        require(set(inner) == {e.id for e in fitting} and choice['innerPredictionBindings'] == bindings,
                'Inner evidence or outer exclusion differs')
        selected, option_checks = replay_options(fitting, inner, choice['decoder'])
        selection_keys = set(selected)
        require({k: choice[k] for k in selection_keys} == selected, 'Saved option winner/tie/fallback differs from independent replay')
        require(set(choice) == selection_keys | {'heldSourceGroup', 'epoch', 'decoder', 'baselineSelection',
                                                'innerPredictionBindings', 'outerPredictionBinding'}, 'Unexpected selection fields')
        # Deliberately open held outer probabilities only after inner selection.
        held = [e for e in examples if e.group == outer]
        probabilities, outer_bound = load_predictions(root/f'outer-{outer_index}'/'refit', choice['epoch'], held,
            expected_membership(manifest, cohort, {outer}, outer), manifest=manifest, kind=kind,
            seed=seed, source_contract=source_contract, audited_fits=audited_fits)
        require(choice['outerPredictionBinding'] == outer_bound, 'Outer NPZ evidence differs')
        for e in held:
            baseline_rows.append(evaluation_row(e, independent_decode(e, probabilities[e.id], choice['decoder'], None)))
            all_rows.append(evaluation_row(e, independent_decode(e, probabilities[e.id], choice['decoder'], selected['keepThreshold'])))
        checks.append({'heldSourceGroup': outer, 'baselineEpoch': choice['epoch'], 'baselineDecoder': choice['decoder'],
                       'selection': selected, 'independentInnerOptions': option_checks,
                       'innerPredictionBindings': bindings, 'outerPredictionBinding': outer_bound})
    require(baseline_rows == baseline['predictions'] and transfer.evaluate_predictions(baseline_rows) == baseline['evaluation'],
            'Actual original NPZ no-op replay changes immutable baseline')
    require(all_rows == result['predictions'], 'Saved final cuts differ from independent proposal replay')
    return checks


def summarize(study):
    from analysis import neural_keep_rescue_development as runner
    require((study/'report.json').is_file(), 'Summary requires a completed explicitly registered rescue study')
    registration, manifest, reference_report, runner_audited = runner.validate_registration(study/'preregistration.json')
    c, contract_hash = registration['contract'], registration['sha256']
    require(canonical(c) == contract_hash and c['experiment'] == 'keep-head-short-rescue-development-v1', 'Registration identity differs')
    report = read(study/'report.json')
    require(report['status'] == 'completed-keep-rescue-development' and report['contractSha256'] == contract_hash
            and report['manifestSha256'] == c['manifestSha256'] and report['records'] == 8
            and report['sourceGroups'] == c['groups'] and report['protectedTestOpened'] is False
            and report['productionPromotionAllowed'] is False and report['execution']['newFits'] == 0
            and report['execution']['device'] == 'cpu', 'Incomplete or mismatched zero-fit report')
    exact = verified(manifest['exactManifest'])
    require(exact['recordings'] == manifest['exactRows'], 'Exact label revision differs')
    reference = c['referenceStudy']
    old_reg_bound = {'path': str(Path(reference['path'])/'preregistration.json'), 'sha256': reference['preregistrationFileSha256']}
    old_registration = verified(old_reg_bound)
    require(old_registration['sha256'] == canonical(old_registration['contract']) == reference['contractSha256'], 'Reference contract changed')
    report_binding = {'path': str(Path(reference['path'])/'report.json'), 'sha256': reference['reportSha256']}
    require(verified(report_binding) == reference_report, 'Reference report binding differs')
    prior_audits = [verified(bound) for bound in reference['independentAudits']]
    prior_tensor = next(a for a in prior_audits if a['kind'] == 'neural-short-boost-independent-tensor-audit-v1')
    audited_fits = {Path(r['completed']['path']).resolve(): r['completed']['sha256'] for r in prior_tensor['fits']}
    require(len(audited_fits) == len(prior_tensor['fits']) == 864 and audited_fits == runner_audited, 'Prior audited fit map differs')
    locked = [identity(study/'preregistration.json'), identity(study/'report.json'), c['manifest'], manifest['exactManifest'],
              c['protocolSnapshot'], c['preflight'], old_reg_bound, report_binding,
              {'path': str(Path(reference['path'])/'summary.json'), 'sha256': reference['summarySha256']},
              *reference['independentAudits']]
    dependency_paths = {Path(__file__), Path(transfer.__file__), Path(interval.__file__), Path(tensor.__file__),
                        Path(expanded.__file__), Path(balanced.__file__), Path(helpers.__file__), Path(balanced.short.__file__)}
    locked += [identity(path) for path in sorted(dependency_paths)]
    results = report['results']
    verify_grid(results, c)
    baseline = {(r['cohort'], r['kind'], r['seed']): r for r in reference_report['results']
                if r['cohort'] in c['cohorts'] and r['lossArm'] == 'baseline'}
    require(len(baseline) == 6*len(c['cohorts']), 'Wrong baseline cell population')
    examples = load_examples(manifest)
    counts, metrics, selections = Counter(), [], []
    used_completions, used_npz = {}, {}
    for row in results:
        require(row['contractSha256'] == contract_hash and row['architecture'] == row['kind'], 'Result contract/architecture differs')
        result_path = study/f"result-{row['cohort']}-{row['kind']}-baseline-{row['rescueVariant']}-{row['seed']}.json"
        require(read(result_path) == row, 'Result file differs from final report')
        helpers.assert_result_revision(exact, row['predictions'], str(result_key(row)))
        original = baseline[(row['cohort'], row['kind'], row['seed'])]
        if row['rescueVariant'] == 'reference':
            reference_copy(row, original, reference, contract_hash)
            counts['matchedBaselinePayloadsVerified'] += 1
        else:
            checks = audit_cell(row, original, examples, manifest, c, old_registration['contract'], audited_fits)
            selections.append({'cohort': row['cohort'], 'kind': row['kind'], 'seed': row['seed'], 'outerFolds': checks})
            counts['outerRescueSelectionsReplayed'] += len(checks)
            counts['innerOptionScoresIndependentlyReplayed'] += sum(len(r['independentInnerOptions']) for r in checks)
            counts['selectedRefitsDecoded'] += len(checks)
            counts['infeasibleOuterFallbacks'] += sum(not r['selection']['recallEligibilityPassed'] for r in checks)
            counts['explicitNoOpSelections'] += sum(r['selection']['keepThreshold'] is None for r in checks)
            for check in checks:
                for bound in [*check['innerPredictionBindings'], check['outerPredictionBinding']]:
                    used_completions[bound['completed']['path']] = bound['completed']['sha256']
                    for name in ('weights', 'predictions'):
                        used_npz[bound[name]['path']] = bound[name]['sha256']
        values = replay_metrics(row)
        metrics.append({'cohort': row['cohort'], 'kind': row['kind'], 'rescueVariant': row['rescueVariant'], 'seed': row['seed'], **values})
        counts['canonicalMetricReplays'] += 1
        counts['independentFinalPaddingScopeRows'] += values['independentPaddingRows']
    cohorts = len(c['cohorts'])
    expected = {'canonicalMetricReplays': 12*cohorts, 'matchedBaselinePayloadsVerified': 6*cohorts,
                'outerRescueSelectionsReplayed': 24*cohorts, 'innerOptionScoresIndependentlyReplayed': 120*cohorts,
                'selectedRefitsDecoded': 24*cohorts, 'independentFinalPaddingScopeRows': 624*cohorts}
    require(all(counts[k] == value for k, value in expected.items()) and len(used_completions) == 96*cohorts
            and len(used_npz) == 192*cohorts, 'Rescue artifact audit counts differ')
    require(report['execution']['candidateCells'] == report['execution']['referenceCells'] == 6*cohorts
            and not (study/'fits').exists(), 'Zero-fit study created or reports unexpected fits')
    values, losses = summarize_results(results, c)
    for bound in locked:
        require(digest(bound['path']) == bound['sha256'], 'Audit input changed during summary')
    for name, sha in c['code'].items():
        require(digest(REPO/'analysis'/name) == sha, 'Registered code changed during summary')
    return {'kind': 'independent-neural-keep-rescue-summary-v1', 'status': 'completed-keep-rescue-development-audit',
            'passed': True, 'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': contract_hash,
            'report': identity(study/'report.json'), 'primaryMetric': 'F1_padP_coreR', 'primaryPaddingSeconds': 2, 'joinGapSeconds': 3,
            'protectedTestOpened': False, 'productionPromotionAllowed': False, 'newFits': 0,
            'scope': {'cohorts': c['cohorts'], 'architectures': c['kinds'], 'seeds': c['seeds'],
                      'exactRecordings': 8, 'exactSourceGroups': c['groups'], 'resultCells': len(results)},
            'inputs': locked, 'referenceStudy': reference,
            'audit': {'counts': dict(counts), 'expected': expected, 'metricReplays': metrics, 'selections': selections,
                      'originalCompletedFitsReferenced': used_completions, 'originalNPZArtifactsReferenced': used_npz,
                      'freshFits': 0, 'freshNPZArtifacts': 0},
            'limitations': ['Prospective optional decoder study; only execute after explicit registration and completed reference audits.',
                            'Adaptive development on four already inspected source groups; three seeds are not independent population evidence.',
                            'Baseline epoch and live/serve/end decoder remain fixed. Only the ordered rescue threshold/no-op option is selected on inner predictions.',
                            'Keep export coverage is not invertible; two-second erosion is a heuristic, not independently exact rally labels.',
                            'Baseline payloads and nested historical compact provenance are retained. There are no new neural fits or independent baseline replicates.',
                            'Prior bound tensor/scaler audits cover original weights. This audit rechecks selected NPZ hashes and probability shape/masks, not all neural forward passes.',
                            'Core proposals are identical across all four padding cases. Final interval arithmetic and inner primary metrics use an independent endpoint sweep.',
                            'Within-architecture F1 and ten-condition retention screens are separate from interaction/replication; no screen permits production promotion.'],
            **values}, {'scope': {'cohorts': c['cohorts'], 'seeds': c['seeds']}, **losses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    names = ('summary.json', 'summary.md', 'loss-identities.json')
    require(not any((args.study/name).exists() for name in names), 'Refuse overwriting immutable rescue summaries')
    summary, losses = summarize(args.study)
    values = (json.dumps(summary, indent=2, allow_nan=False)+'\n', markdown(summary), json.dumps(losses, indent=2, allow_nan=False)+'\n')
    for name, value in zip(names, values):
        with (args.study/name).open('x', encoding='utf-8') as stream:
            stream.write(value)
    print(json.dumps({'passed': True, 'counts': summary['audit']['counts'],
                      'artifacts': {name: identity(args.study/name) for name in names}}, indent=2))


if __name__ == '__main__':
    main()
