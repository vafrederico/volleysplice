export type TimestampedClosableSample<Sample> = {
  timestamp: number;
  clone(): Sample;
  close(): void;
};

export type SequentialSampleSource<Sample> = {
  samples(startTimestamp?: number, endTimestamp?: number): AsyncGenerator<Sample, void, unknown>;
};

async function* selectFromSequentialPass<
  Sample extends TimestampedClosableSample<Sample>,
>(
  source: SequentialSampleSource<Sample>,
  timestamps: readonly number[],
  onSourceFrame: () => void,
  exclusiveEnd: number | undefined,
  nearest: boolean,
): AsyncGenerator<Sample | null, void, unknown> {
  if (timestamps.length === 0) return;
  const sampleIterator = source.samples(timestamps[0], exclusiveEnd);
  let candidate: Sample | null = null;
  let lookahead: Sample | null = null;
  let targetIndex = 0;
  try {
    while (targetIndex < timestamps.length) {
      const next = await sampleIterator.next();
      if (next.done) break;
      const sample = next.value;
      lookahead = sample;
      onSourceFrame();
      while (
        targetIndex < timestamps.length &&
        (nearest ? sample.timestamp >= timestamps[targetIndex]! - 1e-10
          : sample.timestamp - timestamps[targetIndex]! > 1e-10)
      ) {
        const target = timestamps[targetIndex]!;
        // Training/native use nearest presentation time; ties choose earlier.
        // Preserve one row per target even when several targets share a frame.
        const selected = !nearest ? candidate
          : candidate && target - candidate.timestamp <= sample.timestamp - target + 1e-10
            ? candidate : sample;
        yield selected?.clone() ?? null;
        targetIndex += 1;
      }
      candidate?.close();
      candidate = sample;
      lookahead = null;
    }
    while (targetIndex < timestamps.length) {
      yield candidate?.clone() ?? null;
      targetIndex += 1;
    }
  } finally {
    candidate?.close();
    lookahead?.close();
    await sampleIterator.return(undefined);
  }
}

/** Preserve the frozen score-specialist preceding-frame contract. */
export function samplesAtTimestampsFromSequentialPass<Sample extends TimestampedClosableSample<Sample>>(
  source: SequentialSampleSource<Sample>, timestamps: readonly number[], onSourceFrame: () => void,
  _exclusiveEnd?: number,
) {
  return selectFromSequentialPass(source, timestamps, onSourceFrame, undefined, false);
}

/** AV rows use nearest actual PTS without decoding beyond the analysis window. */
export function nearestSamplesAtTimestampsFromSequentialPass<Sample extends TimestampedClosableSample<Sample>>(
  source: SequentialSampleSource<Sample>, timestamps: readonly number[], onSourceFrame: () => void,
  exclusiveEnd?: number,
) {
  return selectFromSequentialPass(source, timestamps, onSourceFrame, exclusiveEnd, true);
}
