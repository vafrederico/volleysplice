#!/usr/bin/env python3
"""Evaluate small side/switch heads on a selected frozen image encoder.

All private inputs are explicit CLI arguments. The output contains aggregate
metrics, content hashes, and head size/norm summaries, never source identifiers/paths.
The existing rally encoder/TCN are read only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.side_switch_full_video import event_metric_counts, monotonic_interval_match


PROJECTION_DIMENSION = 128
PROJECTION_SEED = 3407
L2 = 0.1
SWITCH_MATCH_SECONDS = 4.0


@dataclass(frozen=True)
class UnionDecoderSettings:
    minimum_index_separation: int
    minimum_time_separation_seconds: float
    free_predictions_per_recording: int
    count_penalty_logit: float

    def to_dict(self):
        return dict(minimumIndexSeparation=self.minimum_index_separation,
                    minimumTimeSeparationSeconds=self.minimum_time_separation_seconds,
                    freePredictionsPerRecording=self.free_predictions_per_recording,
                    countPenaltyLogit=self.count_penalty_logit,
                    usesCadence=False, reanchorOnSelection=False,
                    hardMaximumPredictions=None)


def logit(value):
    clipped = min(1-1e-9, max(1e-9, float(value)))
    return math.log(clipped/(1-clipped))


def decode_ranked_candidates(rows, probabilities, threshold, settings):
    """The frozen union decoder, applied to new out-of-fold scores."""
    scores = np.asarray(probabilities, np.float64)
    require(scores.shape == (len(rows),) and np.isfinite(scores).all()
            and np.all((scores >= 0)&(scores <= 1)), 'Invalid candidate scores')
    predicted = np.zeros(len(rows), bool)
    for recording_id in sorted({r['recordingId'] for r in rows}):
        indexes = [i for i, row in enumerate(rows) if row['recordingId'] == recording_id]
        chronological = sorted(indexes, key=lambda i: (float(rows[i]['transitionTime']), rows[i]['eventId']))
        ordinal = {i: order for order, i in enumerate(chronological)}
        ranked = sorted(indexes, key=lambda i: (-float(scores[i]),
                                                float(rows[i]['transitionTime']), rows[i]['eventId']))
        selected = []
        for index in ranked:
            timestamp = float(rows[index]['transitionTime'])
            if any((settings.minimum_index_separation > 0
                    and abs(ordinal[index]-ordinal[other]) < settings.minimum_index_separation)
                   or (settings.minimum_time_separation_seconds > 0
                       and abs(timestamp-float(rows[other]['transitionTime']))
                       < settings.minimum_time_separation_seconds)
                   for other in selected):
                continue
            excess = max(0, len(selected)+1-settings.free_predictions_per_recording)
            margin = logit(scores[index])-logit(threshold)-settings.count_penalty_logit*excess
            if margin >= -1e-12:
                selected.append(index)
                predicted[index] = True
    return predicted


def average_precision(labels, probabilities):
    truth, scores = np.asarray(labels, np.int64), np.asarray(probabilities, np.float64)
    positives = int(truth.sum())
    require(positives > 0, 'Average precision needs a positive label')
    ranked = truth[np.argsort(-scores, kind='stable')]
    return float(np.sum(np.cumsum(ranked)*ranked/np.arange(1, len(ranked)+1))/positives)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def frozen_tokens(folder, recording_id, expected_encoder):
    path = Path(folder)/(recording_id+'.npz')
    receipt = load(path.with_suffix('.json'))
    require(receipt['id'] == recording_id and receipt['labelsUsed'] is False,
            'Feature provenance or label-blind contract differs')
    require(receipt['encoder'] == expected_encoder and receipt['output']['sha256'] == digest(path),
            'Frozen encoder or feature bytes differ')
    with np.load(path, allow_pickle=False) as data:
        times = data['timestamps'].copy()
        tokens = data['tokens'].astype(np.float32)
    require(tokens.shape == (len(times), 4, 960) and len(times) > 2
            and np.isfinite(tokens).all() and np.all(np.diff(times) > 0),
            'Invalid frozen regional embeddings')
    return times, tokens


def nearest(times, tokens, requested):
    query = np.asarray(requested, np.float64)
    high = np.searchsorted(times, query, side='left').clip(0, len(times)-1)
    low = (high-1).clip(0, len(times)-1)
    chosen = np.where(np.abs(times[low]-query) <= np.abs(times[high]-query), low, high)
    errors = np.abs(times[chosen]-query)
    return tokens[chosen], errors


def serving_features(times, tokens, anchor):
    samples, errors = nearest(times, tokens, anchor+np.asarray((-1., -.5, 0., .5, 1., 1.5)))
    before, contact, after = (samples[0:2].mean(axis=0),
                              samples[2:4].mean(axis=0), samples[4:6].mean(axis=0))
    vector = np.concatenate((contact[1]-contact[2],
                             (after[1]-after[2])-(before[1]-before[2]),
                             after[0]-before[0], contact[0])).astype(np.float32)
    return vector, float(errors.max())


def window_mean(times, tokens, start, end):
    require(end > start, 'Invalid side-switch comparison window')
    subset = tokens[(times >= start) & (times <= end)]
    if len(subset):
        return subset.mean(axis=0), 0
    fallback, _ = nearest(times, tokens, [(start+end)/2])
    return fallback[0], 1


def switch_features(times, tokens, windows):
    before, miss_a = window_mean(times, tokens, **windows['before'])
    after, miss_b = window_mean(times, tokens, **windows['after'])
    cosine = (np.sum(before*after, axis=1) /
              np.maximum(np.linalg.norm(before, axis=1)*np.linalg.norm(after, axis=1), 1e-8))
    vector = np.concatenate(((after-before).reshape(-1), cosine)).astype(np.float32)
    return vector, miss_a+miss_b


def project(matrix, dimension=PROJECTION_DIMENSION, seed=PROJECTION_SEED):
    require(matrix.ndim == 2 and np.isfinite(matrix).all(), 'Nonfinite feature matrix')
    if dimension == 0:
        return matrix.copy(), None
    rng = np.random.default_rng(seed)
    directions = rng.choice(np.asarray([-1., 1.], np.float32),
                            size=(matrix.shape[1], dimension))
    directions /= math.sqrt(matrix.shape[1])
    return matrix @ directions, directions


def fit_head(matrix, labels):
    x = np.asarray(matrix, np.float64)
    y = np.asarray(labels, np.float64)
    require(x.ndim == 2 and len(x) == len(y) and len(np.unique(y)) == 2,
            'Head fit needs aligned examples of both classes')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-6, 1., scale)
    x = (x-mean)/scale
    positives = int(y.sum())
    negatives = len(y)-positives
    weights = np.where(y == 1, math.sqrt(len(y)/(2*positives)),
                       math.sqrt(len(y)/(2*negatives)))
    weights /= weights.sum()

    def loss_gradient(params):
        logit = x @ params[:-1]+params[-1]
        probability = 1/(1+np.exp(-np.clip(logit, -35, 35)))
        loss = (weights*np.logaddexp(0, logit)).sum()-(weights*y*logit).sum()
        loss += L2*np.dot(params[:-1], params[:-1])/2
        residual = weights*(probability-y)
        grad = np.r_[x.T@residual+L2*params[:-1], residual.sum()]
        return loss, grad

    result = minimize(loss_gradient, np.zeros(x.shape[1]+1), method='L-BFGS-B',
                      jac=True, options={'maxiter': 300, 'ftol': 1e-11})
    require(result.success or np.linalg.norm(result.jac) < 1e-5,
            'Lightweight head did not converge')
    return (mean, scale, result.x[:-1], float(result.x[-1]))


def predict(head, matrix):
    mean, scale, weights, bias = head
    logits = ((np.asarray(matrix, np.float64)-mean)/scale)@weights+bias
    return 1/(1+np.exp(-np.clip(logits, -35, 35)))


def binary_metrics(labels, scores):
    truth = np.asarray(labels, np.int64)
    predicted = np.asarray(scores) >= .5
    tp = int(np.sum((truth == 1)&predicted))
    tn = int(np.sum((truth == 0)&~predicted))
    fp = int(np.sum((truth == 0)&predicted))
    fn = int(np.sum((truth == 1)&~predicted))
    return dict(rows=len(truth), positives=int(truth.sum()), truePositive=tp,
                trueNegative=tn, falsePositive=fp, falseNegative=fn,
                balancedAccuracy=(tp/(tp+fn)+tn/(tn+fp))/2)


def serving_experiment(folder, encoder, dataset_path, baseline_path, projection_dimension):
    source = load(dataset_path)
    require(source['scope'] == 'development' and len(source['rows']) == 1027,
            'Unknown serving-side development population')
    rows = [r for r in source['rows'] if r['environment'] != 'beach']
    require(len(rows) == 968 and {r['label'] for r in rows} == {0, 1},
            'Non-beach serving-side population differs')
    baseline = {r['rallyId']: r for r in load(baseline_path)['selectedPredictions']}
    require(len(baseline) == len(source['rows']), 'Serving baseline population differs')
    tokens_by_id = {}
    values, max_error = [], 0.
    for row in rows:
        key = row['recordingId']
        if key not in tokens_by_id:
            tokens_by_id[key] = frozen_tokens(folder, key, encoder)
        vector, error = serving_features(*tokens_by_id[key], float(row['serveAnchor']))
        values.append(vector)
        max_error = max(max_error, error)
    x, directions = project(np.stack(values), projection_dimension)
    labels = np.asarray([r['label'] for r in rows], np.int64)
    groups = np.asarray([r['sourceGroup'] for r in rows])
    oof = np.full(len(rows), np.nan)
    fold_metrics = []
    for group in sorted(set(groups)):
        held = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        require(len(set(labels[train])) == 2 and len(set(labels[held])) == 2,
                'Serving-side fold lacks a class')
        oof[held] = predict(fit_head(x[train], labels[train]), x[held])
        fold_metrics.append(binary_metrics(labels[held], oof[held]))
    require(np.isfinite(oof).all(), 'Serving side has unscored rows')
    baseline_scores = np.asarray([float(baseline[r['rallyId']]['prediction'] == 'near') for r in rows])
    require(all(baseline[r['rallyId']]['recordingId'] == r['recordingId']
                and baseline[r['rallyId']]['decision'] == r['decision'] for r in rows),
            'Serving baseline identity differs')
    baseline_folds = [binary_metrics(labels[groups == group], baseline_scores[groups == group])
                      for group in sorted(set(groups))]
    final = fit_head(x, labels)
    return dict(scope='non-beach development; leave one source group out',
                rows=len(rows), recordings=len(tokens_by_id), sourceGroups=len(set(groups)),
                maxNearestFrameErrorSeconds=max_error, featureDimension=len(values[0]),
                head=binary_metrics(labels, oof),
                baseline=binary_metrics(labels, baseline_scores),
                groupMacroBalancedAccuracy=float(np.mean([m['balancedAccuracy'] for m in fold_metrics])),
                baselineGroupMacroBalancedAccuracy=float(np.mean([m['balancedAccuracy'] for m in baseline_folds])),
                finalWeights=compact_weights(final, directions),
                sourceHashes={'labels': digest(dataset_path), 'baseline': digest(baseline_path)})


def compact_weights(head, directions):
    mean, scale, weights, bias = head
    effective = (directions@(weights/scale) if directions is not None else weights/scale)
    intercept = bias-np.dot(mean/scale, weights)
    return dict(inputDimension=len(effective), nonzero=int(np.count_nonzero(effective)),
                bytesFloat32=int((len(effective)+1)*4),
                weightNorm=float(np.linalg.norm(effective)), intercept=float(intercept))


def switch_event_metrics(rows, predictions, markers):
    tp = fp = fn = proposals = 0
    for recording_id, truth in markers.items():
        chosen = [dict(eventId=r['eventId'], recordingId=recording_id, kind=r['kind'],
                       gapStart=float(r['gapStart']), gapEnd=float(r['gapEnd']),
                       transitionTime=float(r['transitionTime']))
                  for r, selected in zip(rows, predictions, strict=True)
                  if selected and r['recordingId'] == recording_id]
        chosen.sort(key=lambda r: (r['transitionTime'], r['eventId']))
        result = event_metric_counts(monotonic_interval_match(chosen, truth, SWITCH_MATCH_SECONDS))
        tp += result['truePositives']; fp += result['falsePositives']; fn += result['falseNegatives']
        proposals += len(chosen)
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    return dict(proposals=proposals, truePositives=tp, falsePositives=fp,
                falseNegatives=fn, precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.)


def switch_experiment(folder, encoder, features_path, audit_path, model_path, baseline_path,
                      projection_dimension):
    source, audit, model = load(features_path), load(audit_path), load(model_path)
    require(source['scope']['status'] == 'opened-development-only'
            and len(source['rows']) == 704 and source['scope']['recordings'] == 11,
            'Unknown switch candidate population')
    require(model['sources']['features']['sha256'] == digest(features_path)
            and model['sources']['fullAudit']['sha256'] == digest(audit_path),
            'Switch source identity differs from promoted control')
    rows = source['rows']
    ids = source['scope']['recordingIds']
    markers = {key: [{'time':float(value)} for value in audit['scope']['humanEventsByRecording'][key]]
               for key in ids}
    labels = {r['eventId']: 0 for r in rows}
    for recording_id in ids:
        local = [r for r in rows if r['recordingId'] == recording_id]
        proposals = [dict(eventId=r['eventId'], recordingId=recording_id, kind=r['kind'],
                          gapStart=float(r['gapStart']), gapEnd=float(r['gapEnd']),
                          transitionTime=float(r['transitionTime'])) for r in local]
        result = monotonic_interval_match(proposals, markers[recording_id], SWITCH_MATCH_SECONDS)
        for pair in result.pairs:
            labels[local[pair.proposal_index]['eventId']] = 1
    y = np.asarray([labels[r['eventId']] for r in rows], np.int64)
    require(y.sum() == 46 and sum(map(len, markers.values())) == 50,
            'Switch candidate labels differ from frozen development universe')
    tokens_by_id = {}
    values, fallback_windows = [], 0
    for row in rows:
        key = row['recordingId']
        if key not in tokens_by_id:
            tokens_by_id[key] = frozen_tokens(folder, key, encoder)
        value, fallback = switch_features(*tokens_by_id[key], row['comparisonWindows'])
        values.append(value)
        fallback_windows += fallback
    x, directions = project(np.stack(values), projection_dimension)
    recordings = np.asarray([r['recordingId'] for r in rows])
    oof = np.full(len(rows), np.nan)
    for recording_id in ids:
        held = np.flatnonzero(recordings == recording_id)
        train = np.flatnonzero(recordings != recording_id)
        oof[held] = predict(fit_head(x[train], y[train]), x[held])
    require(np.isfinite(oof).all(), 'Side switch has unscored rows')
    settings = UnionDecoderSettings(
        minimum_index_separation=int(model['decoder']['minimumIndexSeparation']),
        minimum_time_separation_seconds=float(model['decoder']['minimumTimeSeparationSeconds']),
        free_predictions_per_recording=int(model['decoder']['freePredictionsPerRecording']),
        count_penalty_logit=float(model['decoder']['countPenaltyLogit']))
    predictions = decode_ranked_candidates(rows, oof, .5, settings)
    optimistic = None
    for threshold in np.unique(np.quantile(oof, np.linspace(.02, .98, 49))):
        trial = switch_event_metrics(rows,
            decode_ranked_candidates(rows, oof, float(threshold), settings), markers)
        if optimistic is None or (trial['f1'], trial['recall']) > (optimistic['f1'], optimistic['recall']):
            optimistic = dict(threshold=float(threshold), **trial)
    baseline = load(baseline_path)['fixedVariantOuterResults']['union34-top2-x2']
    final = fit_head(x, y)
    return dict(scope='opened development; leave one recording out; one source group',
                rows=len(rows), recordings=len(ids), positiveCandidates=int(y.sum()),
                humanMarkers=sum(map(len, markers.values())), fallbackWindows=fallback_windows,
                featureDimension=len(values[0]), rowAveragePrecision=average_precision(y, oof),
                baselineRowAveragePrecision=baseline['rowAveragePrecision'],
                event=switch_event_metrics(rows, predictions, markers),
                postHocThresholdUpperBound=optimistic,
                baselineEvent={key: value for key, value in baseline['primary'].items()
                               if key != 'byRecording'},
                decoder={'threshold':.5, **settings.to_dict()},
                finalWeights=compact_weights(final, directions),
                sourceHashes={'features': digest(features_path), 'audit': digest(audit_path),
                              'baselineModel':digest(model_path),'baselineEvaluation':digest(baseline_path)})


def main():
    global L2
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study-root','serving-dataset','serving-baseline','switch-features',
                 'switch-audit','switch-model','switch-baseline','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--projection-dimension', type=int, default=PROJECTION_DIMENSION,
                        help='Zero uses the full embedding descriptor; otherwise use a fixed sign projection')
    parser.add_argument('--l2', type=float, default=L2)
    args = parser.parse_args()
    require(args.projection_dimension == 0 or args.projection_dimension > 0,
            'Projection dimension must be nonnegative')
    require(math.isfinite(args.l2) and args.l2 > 0, 'L2 must be finite and positive')
    L2 = args.l2
    root = args.study_root
    evaluation = load(root/'evaluation.json')
    selected = next(row for row in evaluation['selected'] if row['mode'] == 'recall')
    require(selected['floorPercent'] == 99 and selected['variant'] == 'expanded-large'
            and selected['draw'] == 3407,
            'Frozen highest-recall Large selection differs')
    fit = root/'fits'/selected['variant']/f"split-{selected['draw']}"
    encoder = load(fit/'student/completed.json')['weights']
    require(encoder['sha256'] == digest(encoder['path']), 'Selected encoder checkpoint differs')
    folder = fit/'student-features'
    serving = serving_experiment(folder, encoder, args.serving_dataset, args.serving_baseline,
                                 args.projection_dimension)
    switch = switch_experiment(folder, encoder, args.switch_features, args.switch_audit,
                               args.switch_model, args.switch_baseline, args.projection_dimension)
    result = dict(schemaVersion=1, kind='distilled-large-frozen-specialist-head-screen-v1',
                  selectedModel='neural-distilled-mobile-large-tcn-fp32-high-recall',
                  encoderSha256=encoder['sha256'], evaluationSha256=digest(root/'evaluation.json'),
                  projectionDimension=args.projection_dimension, projectionSeed=PROJECTION_SEED,
                  logisticL2=L2, serving=serving, sideSwitch=switch,
                  limitations=['Frozen encoder is label-blind for side/switch but may have seen evaluation video frames during DINO distillation.',
                               'Serving-side anchors are human-reviewed candidates, not neural-rally starts.',
                               'Switch rows come from one source group; recording holdout does not establish source transfer.',
                               'This study fits heads on saved desktop embeddings, not native-device embeddings.'])
    path = args.output
    require(not path.exists(), 'Refusing to overwrite experiment output')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'output':str(path),'serving':serving['head']['balancedAccuracy'],
                      'servingBaseline':serving['baseline']['balancedAccuracy'],
                      'switchAp':switch['rowAveragePrecision'],
                      'switchBaselineAp':switch['baselineRowAveragePrecision'],
                      'switchF1':switch['event']['f1'],
                      'switchBaselineF1':switch['baselineEvent']['f1']}))


if __name__ == '__main__':
    main()
