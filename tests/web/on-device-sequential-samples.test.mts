import assert from "node:assert/strict";
import test from "node:test";

import { nearestSamplesAtTimestampsFromSequentialPass as samplesAtTimestampsFromSequentialPass } from "../../lib/on-device/sequential-samples.ts";
import { nearestSamplesAtTimestampsFromSequentialPass as productionSelect, samplesAtTimestampsFromSequentialPass as specialistSelect } from "../../prod/src/lib/on-device/sequential-samples.ts";
import { nearestFrameTimestamps } from "../../prod/src/lib/on-device/nearest-frame-timestamps.ts";

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

test("sequential selection stops at an exact final target without decoding another frame", async () => {
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
  assert.equal(traversed, 5);
  assert.equal(source.yielded.every((sample) => sample.closed), true);
});

test("sequential selection clamps to the first/last frame without dropping grid rows", async () => {
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
  assert.deepEqual(selected, [0.1, 0.1, 0.4]);
  assert.equal(source.yielded.every((sample) => sample.closed), true);
});

for (const [name, select] of [["lab", samplesAtTimestampsFromSequentialPass], ["production", productionSelect]] as const) {
  test(`${name}: irregular PTS choose nearest with earlier ties and repeated frames`, async () => {
    const pts = [0, 0.12, 0.41, 0.9];
    const targets = [0.05, 0.06, 0.1, 0.25, 0.3, 0.3, 0.5, 0.8, 1];
    const source = new FakeSource(pts);
    const result: number[] = [];
    for await (const sample of select(source, targets, () => undefined)) {
      assert.ok(sample);
      result.push(sample.timestamp);
      sample.close();
    }
    const expected = [0, 0, 0.12, 0.12, 0.41, 0.41, 0.41, 0.9, 0.9];
    assert.deepEqual(result, expected);
    assert.deepEqual(nearestFrameTimestamps(pts, targets), expected);
    assert.ok(source.yielded.every(sample => sample.closed));
  });
  test(`${name}: early cancellation closes both the preceding and lookahead frames`, async () => {
    const source = new FakeSource([0, 0.3, 0.6]);
    for await (const sample of select(source, [0.25, 0.5], () => undefined)) {
      assert.equal(sample?.timestamp, 0.3);
      sample?.close();
      break;
    }
    assert.equal(source.yielded.length, 2);
    assert.ok(source.yielded.every(sample => sample.closed));
  });
  test(`${name}: empty input is represented explicitly`, async () => {
    const output = [];
    for await (const sample of select(new FakeSource([]), [0, 0.25], () => undefined)) output.push(sample);
    assert.deepEqual(output, [null, null]);
  });
}

test("score specialists retain their frozen preceding-frame choices", async () => {
  const source = new FakeSource([0.1, 0.4, 0.7]);
  const result: Array<number | null> = [];
  for await (const sample of specialistSelect(source, [0, 0.35, 0.6, 1], () => undefined)) {
    result.push(sample?.timestamp ?? null);
    sample?.close();
  }
  assert.deepEqual(result, [null, 0.1, 0.4, 0.7]);
  assert.ok(source.yielded.every(sample => sample.closed));
});
