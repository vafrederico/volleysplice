export type TimestampedClosableSample<Sample> = {
  timestamp: number;
  clone(): Sample;
  close(): void;
};

export type SequentialSampleSource<Sample> = {
  samples(startTimestamp?: number): AsyncGenerator<Sample, void, unknown>;
};

export async function* samplesAtTimestampsFromSequentialPass<
  Sample extends TimestampedClosableSample<Sample>,
>(
  source: SequentialSampleSource<Sample>,
  timestamps: readonly number[],
  onSourceFrame: () => void,
): AsyncGenerator<Sample | null, void, unknown> {
  if (timestamps.length === 0) return;
  const sampleIterator = source.samples(timestamps[0]);
  let candidate: Sample | null = null;
  let targetIndex = 0;
  try {
    while (targetIndex < timestamps.length) {
      const next = await sampleIterator.next();
      if (next.done) break;
      const sample = next.value;
      onSourceFrame();
      while (
        targetIndex < timestamps.length &&
        sample.timestamp - timestamps[targetIndex]! > 1e-10
      ) {
        yield candidate?.clone() ?? null;
        targetIndex += 1;
      }
      candidate?.close();
      candidate = sample;
    }
    while (targetIndex < timestamps.length) {
      yield candidate?.clone() ?? null;
      targetIndex += 1;
    }
  } finally {
    candidate?.close();
    await sampleIterator.return(undefined);
  }
}
