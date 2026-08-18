import assert from "node:assert/strict";
import test from "node:test";
import { AudioSample } from "mediabunny";

import {
  AudioAccumulator,
  planTimestampedAudioChunk,
  type StreamingAudioResampler,
} from "../../lib/on-device/audio-features.ts";

const SAMPLE_RATE = 16_000;
const QUANTIZED_ONE = 32767 / 32768;

function approximately(actual: number, expected: number, epsilon = 1e-6): void {
  assert.ok(
    Math.abs(actual - expected) <= epsilon,
    `expected ${actual} to be within ${epsilon} of ${expected}`,
  );
}

test("timestamp placement removes negative AAC priming", () => {
  assert.deepEqual(planTimestampedAudioChunk(-1024 / 48_000, 1024, 48_000, 0), {
    gapFrames: 0,
    trimFrames: 1024,
    appendFrames: 0,
    nextFrame: 0,
  });
  assert.deepEqual(planTimestampedAudioChunk(-512 / 48_000, 1024, 48_000, 0), {
    gapFrames: 0,
    trimFrames: 512,
    appendFrames: 512,
    nextFrame: 512,
  });
});

test("timestamp placement inserts gaps and trims partial or complete overlaps", () => {
  assert.deepEqual(planTimestampedAudioChunk(960 / 48_000, 480, 48_000, 480), {
    gapFrames: 480,
    trimFrames: 0,
    appendFrames: 480,
    nextFrame: 1440,
  });
  assert.deepEqual(planTimestampedAudioChunk(480 / 48_000, 960, 48_000, 960), {
    gapFrames: 0,
    trimFrames: 480,
    appendFrames: 480,
    nextFrame: 1440,
  });
  assert.deepEqual(planTimestampedAudioChunk(0, 960, 48_000, 1440), {
    gapFrames: 0,
    trimFrames: 960,
    appendFrames: 0,
    nextFrame: 1440,
  });
});

test("the accumulator trims priming before the existing resampler", () => {
  const accumulator = new AudioAccumulator();
  const sample = new AudioSample({
    data: new Float32Array(800).fill(1),
    format: "f32",
    numberOfChannels: 1,
    sampleRate: SAMPLE_RATE,
    timestamp: -400 / SAMPLE_RATE,
  });
  try {
    accumulator.push(sample);
  } finally {
    sample.close();
  }

  const features = accumulator.finish();
  assert.equal(features.rms.length, 1);
  approximately(
    features.rms[0],
    Math.sqrt((399 * QUANTIZED_ONE ** 2 + 1) / 800),
  );
  // finish() preserves the existing final-source-sample behavior, including its lack of re-quantization.
  approximately(features.peak[0], 1);
});

test("the accumulator represents timestamp gaps as silence", () => {
  const accumulator = new AudioAccumulator();
  accumulator.pushMono(new Float32Array(400).fill(1), 0, SAMPLE_RATE);
  accumulator.pushMono(
    new Float32Array(400).fill(0.5),
    800 / SAMPLE_RATE,
    SAMPLE_RATE,
  );

  const features = accumulator.finish();
  assert.equal(features.rms.length, 2);
  approximately(features.rms[0], QUANTIZED_ONE / Math.sqrt(2));
  approximately(features.rms[1], 0.5 / Math.sqrt(2));
});

test("the accumulator discards overlap before resampling", () => {
  const accumulator = new AudioAccumulator();
  accumulator.pushMono(new Float32Array(800).fill(1), 0, SAMPLE_RATE);
  accumulator.pushMono(
    new Float32Array(800).fill(0.5),
    400 / SAMPLE_RATE,
    SAMPLE_RATE,
  );

  const features = accumulator.finish();
  assert.equal(features.rms.length, 2);
  approximately(features.rms[0], QUANTIZED_ONE);
  approximately(features.rms[1], 0.5 / Math.sqrt(2));
});

test("the reusable linear-resampler buffer handles uneven decoded chunks", () => {
  const sampleRate = 48_000;
  const samples = Float32Array.from(
    { length: Math.round(sampleRate * 1.3) },
    (_, index) =>
      0.4 * Math.sin((2 * Math.PI * 440 * index) / sampleRate) +
      (index % 12_000 < 24 ? 0.5 : 0),
  );
  const chunked = new AudioAccumulator();
  const chunkSizes = [1024, 960, 1536, 511];
  let offset = 0;
  let chunk = 0;
  while (offset < samples.length) {
    const end = Math.min(
      samples.length,
      offset + chunkSizes[chunk % chunkSizes.length],
    );
    chunked.pushMono(
      samples.subarray(offset, end),
      offset / sampleRate,
      sampleRate,
    );
    offset = end;
    chunk += 1;
  }
  const actual = chunked.finish();

  assert.ok(actual.rms.length > 0);
  assert.equal(actual.rms.length, actual.peak.length);
  assert.equal(actual.rms.length, actual.spectralFlux.length);
  assert.ok(actual.rms.every(Number.isFinite));
  assert.ok(actual.bandPower.every((band) => band.length === actual.rms.length));
});

test("timestamp repair happens before planar WASM resampling", () => {
  class CapturingResampler implements StreamingAudioResampler {
    readonly inputs: number[][][] = [];
    closed = false;

    configure(): void {}

    push(planes: readonly Float32Array[]): Int16Array {
      this.inputs.push(planes.map((plane) => Array.from(plane)));
      return new Int16Array(0);
    }

    flush(): Int16Array {
      return new Int16Array(0);
    }

    close(): void {
      this.closed = true;
    }
  }

  const resampler = new CapturingResampler();
  const accumulator = new AudioAccumulator(resampler);
  accumulator.pushPlanar(
    [
      Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]),
      Float32Array.from([8, 7, 6, 5, 4, 3, 2, 1]),
    ],
    -0.5,
    8,
  );
  accumulator.pushPlanar(
    [Float32Array.from([9, 10]), Float32Array.from([11, 12])],
    0.75,
    8,
  );
  accumulator.finish();

  assert.deepEqual(resampler.inputs, [
    [
      [5, 6, 7, 8],
      [4, 3, 2, 1],
    ],
    [
      [0, 0],
      [0, 0],
    ],
    [
      [9, 10],
      [11, 12],
    ],
  ]);
  assert.equal(resampler.closed, true);
});
