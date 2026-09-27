"""Optional keep-head short-core proposals; no selection or data access.

The keep head predicts retained export coverage, including training padding and
gap joins. That operation is not invertible: clipping loses boundaries, joining
loses cuts, and a learned probability component need not equal a labeled export.
Removing two seconds per end is therefore a fixed proposal heuristic, not a
reconstruction of true rally boundaries. No performance claim is implied.

The baseline model, checkpoint and live/serve/end decoder remain fixed. Only the
fourth probability head supplies additional proposals. The same resulting core
intervals must enter the canonical evaluator for every 0/1/2/3-second padding
case. This module never pads predictions or joins a positive gap.
"""
from __future__ import annotations

from math import isfinite

import numpy as np

from . import neural_development as baseline
from .crop_evaluation import subtract_intervals
from .decoder import smooth_probabilities
from .schema import Interval


KEEP_RESCUE_OPTIONS = (None, .35, .5, .65, .8)
ANALYSIS_FPS = 4.
ERODE_SECONDS = 2.
MAX_CORE_SECONDS = 3.
MAX_CADENCE_ERROR_SECONDS = .10


def rescue_metadata():
    """Implementation identity; a future protocol must lock its own option grid."""
    return {
        'kind': 'keep-short-core-rescue-v1', 'options': list(KEEP_RESCUE_OPTIONS),
        'keepHeadIndex': 3, 'analysisFps': ANALYSIS_FPS,
        'maximumNominalCadenceErrorSeconds': MAX_CADENCE_ERROR_SECONDS,
        'erodeSeconds': ERODE_SECONDS, 'maxCoreSeconds': MAX_CORE_SECONDS,
        'thresholdComponentMode': 'smoothed keep >= threshold; no hysteresis, bridging or minimum-length cleanup',
        'thresholdPrecision': 'float32 smoothed probabilities and float32 threshold',
        'smoothing': 'unchanged selected baseline decoder smoothing_seconds; frozen edge-padded moving mean',
        'smoothingReset': 'independent within every contiguous valid tick segment',
        'sampleIntervalConvention': 'first tick minus .125s through last tick plus .125s, clipped to video bounds',
        'timestampConvention': 'actual nominal4Hz frame-quantized AV timestamps or repaired exact4Hz timestamps; never retimed',
        'edgeRule': 'discard components touching either valid-segment edge or either video edge before erosion',
        'ignoredRule': 'also discard whole components intersecting or touching exact ignored intervals, including sub-tick holes',
        'durationRule': 'retain predicted eroded duration strictly greater than zero and at most three seconds',
        'unionRule': 'baseline cores plus proposals, merge overlapping/touching only; never join a positive gap',
        'noOp': 'None returns the unchanged frozen baseline core list; no proposals also preserves that list',
        'gates': 'no live, boundary, gold-duration, feature or label gate',
        'evaluation': 'same fixed proposal cores for all0/1/2/3 padding; padding and strict<3 gap joining only in canonical evaluation',
        'invertibility': 'erosion is heuristic; clipping, gap joining and prediction error prevent an exact inverse of keep coverage',
    }


def _validate_inputs(times, keep_probabilities, valid, duration):
    if (not isinstance(times, np.ndarray) or times.ndim != 1
            or not isinstance(keep_probabilities, np.ndarray) or keep_probabilities.shape != times.shape
            or not isinstance(valid, np.ndarray) or valid.shape != times.shape or valid.dtype != np.bool_):
        raise ValueError('times, keep probabilities and boolean validity must be aligned one-dimensional arrays')
    if (not isfinite(duration) or duration <= 0 or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(times > duration)
            or (len(times) > 1 and (np.any(np.diff(times) <= 0)
                or np.any(np.abs(np.diff(times) - 1 / ANALYSIS_FPS) > MAX_CADENCE_ERROR_SECONDS + 1e-8)))):
        raise ValueError('timestamps must have ordered nominal4Hz cadence within a positive finite video duration')
    if not np.isfinite(keep_probabilities).all() or np.any((keep_probabilities < 0) | (keep_probabilities > 1)):
        raise ValueError('keep probabilities must be finite and between zero and one')


def keep_core_proposals(times, keep_probabilities, valid, duration, *, threshold,
                        smoothing_seconds, ignored_intervals=()):
    """Return label-free short core proposals from the fourth probability head.

    A component's support must lie strictly inside one valid segment. Explicit
    ignored intervals are checked in continuous time as well: a very short
    ignored hole might not contain any tick center, so validity alone is not
    enough. The entire intersecting/touching component is rejected, not split
    into newly invented padded components. None is the explicit no-op option.
    Retained AV caches use frame-quantized nominal4Hz timestamps; repaired caches
    use an exact4Hz grid. Both retain their actual timestamps for interval edges.
    """
    _validate_inputs(times, keep_probabilities, valid, duration)
    if threshold is None:
        return ()
    if isinstance(threshold, bool) or not isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('keep threshold must be None or a finite probability in (0,1]')
    if smoothing_seconds not in (.5, 1.):
        raise ValueError('keep smoothing must equal an existing selected baseline option (.5 or1second)')
    ignored = tuple(ignored_intervals)
    if any(not isfinite(i.start) or not isfinite(i.end) or i.end <= i.start for i in ignored):
        raise ValueError('ignored intervals must have finite ordered endpoints')
    proposals = []
    half_sample = .5 / ANALYSIS_FPS
    samples = max(1, round(smoothing_seconds * ANALYSIS_FPS))
    threshold_value = np.float32(threshold)
    for left, right in baseline.segments(valid):
        smoothed = smooth_probabilities(keep_probabilities[left:right], samples)
        for component_left, component_right in baseline.segments(smoothed >= threshold_value):
            if component_left == 0 or component_right == right-left:
                continue
            start = max(0., float(times[left+component_left]) - half_sample)
            end = min(duration, float(times[left+component_right-1]) + half_sample)
            if start <= 0 or end >= duration:
                continue
            # Closed comparison deliberately rejects mere contact as well as
            # overlap: clipped padding at an ignored edge is not invertible.
            if any(start <= hole.end and end >= hole.start for hole in ignored):
                continue
            core_start, core_end = start + ERODE_SECONDS, end - ERODE_SECONDS
            if 0 < core_end-core_start <= MAX_CORE_SECONDS:
                proposals.append(Interval(core_start, core_end))
    return tuple(proposals)


def decode_with_keep_rescue(example, scores, settings, *, keep_threshold=None):
    """Decode with fixed baseline settings and optionally union keep proposals.

    Compact and DINO models share the same four-head probability contract. This
    function reads only times, duration, validity and ignored ranges from the
    recording. It never reads rally labels, target arrays or input features.
    """
    if (not isinstance(scores, np.ndarray) or scores.shape != (len(example.times), 4)
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
        raise ValueError('rescue requires finite aligned [time,4] probability scores')
    original = baseline.decode(example, scores, settings)
    if keep_threshold is None:
        return original
    proposals = keep_core_proposals(example.times, scores[:, 3], example.valid, example.duration,
                                    threshold=keep_threshold, smoothing_seconds=settings['smoothing'],
                                    ignored_intervals=example.ignored)
    if not proposals or not subtract_intervals(proposals, original):
        return original
    # With no exclusions this helper forms only the overlap/touching union.
    # Unlike canonical export processing, it retains every positive gap here.
    return list(subtract_intervals((*original, *proposals), ()))
