import assert from "node:assert/strict";
import test from "node:test";

import { samplesAtTimestampsFromSequentialPass } from "../../lib/on-device/sequential-samples.ts";

class FakeSample {
  closed = false;
  readonly timestamp: number;

  constructor(timestamp: number) {
    this.timestamp = timestamp;
  }

  clone(): FakeSample {
    assert.equal(this.closed, false);
    return new FakeSample(this.timestamp);
  }

  close(): void {
    this.closed = true;
  }
}

class FakeSource {
  startTimestamp: number | undefined;
  yielded: FakeSample[] = [];
  private readonly timestamps: readonly number[];

  constructor(timestamps: readonly number[]) {
    this.timestamps = timestamps;
  }

  async *samples(startTimestamp?: number): AsyncGenerator<FakeSample, void, unknown> {
    this.startTimestamp = startTimestamp;
    for (const timestamp of this.timestamps) {
      const sample = new FakeSample(timestamp);
      this.yielded.push(sample);
      yield sample;
    }
  }
}

test("sequential selection returns the final source frame at or before each target", async () => {
  const source = new FakeSource([0, 0.1, 0.25, 0.4, 0.5, 0.6]);
  let traversed = 0;
  const selected: number[] = [];
  for await (const sample of samplesAtTimestampsFromSequentialPass(
    source,
    [0, 0.25, 0.5],
    () => {
      traversed += 1;
    },
  )) {
    assert.ok(sample);
    selected.push(sample.timestamp);
    sample.close();
  }
  assert.deepEqual(selected, [0, 0.25, 0.5]);
  assert.equal(source.startTimestamp, 0);
  assert.equal(traversed, 6);
  assert.equal(source.yielded.every((sample) => sample.closed), true);
});

test("sequential selection emits null before the first frame and reuses the last frame at EOF", async () => {
  const source = new FakeSource([0.1, 0.4]);
  const selected: Array<number | null> = [];
  for await (const sample of samplesAtTimestampsFromSequentialPass(
    source,
    [0, 0.2, 0.5],
    () => undefined,
  )) {
    selected.push(sample?.timestamp ?? null);
    sample?.close();
  }
  assert.deepEqual(selected, [null, 0.1, 0.4]);
  assert.equal(source.yielded.every((sample) => sample.closed), true);
});
