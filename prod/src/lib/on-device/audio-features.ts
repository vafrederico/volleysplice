import FFT from "fft.js";
import { AudioSampleSink, type AudioSample, type InputAudioTrack } from "mediabunny";

import { ANALYSIS_FPS, AUDIO_FEATURE_NAMES } from "./feature-schema.ts";
import type { AnalysisWindow } from "./analysis-window.ts";
import { mean, percentileRanks, quantile, rollingMean } from "./feature-math.ts";
import { LibswresampleWasmResampler } from "./libswresample-wasm.ts";
import type { OnDeviceRuntimeVariant } from "./runtime-variants.ts";
import type { AnalysisProgress } from "./types.ts";

const TARGET_SAMPLE_RATE = 16_000;
const FRAME_SAMPLES = 800;
const FRAME_SECONDS = 0.05;
const FFT_SIZE = 1024;
const BAND_BOUNDS = [
  [80, 250],
  [250, 500],
  [500, 1000],
  [1000, 2000],
  [2000, 4000],
  [4000, 7800],
] as const;

type AudioFrameFeatures = {
  rms: number[];
  peak: number[];
  spectralFlux: number[];
  bandPower: number[][];
};

export type StreamingAudioResampler = {
  configure(inputSampleRate: number, inputChannels: number): void;
  push(planes: readonly Float32Array[]): Int16Array;
  flush(): Int16Array;
  close(): void;
};

export type TimestampedAudioChunkPlan = {
  gapFrames: number;
  trimFrames: number;
  appendFrames: number;
  nextFrame: number;
};

/**
 * Places one decoded chunk on the source-rate timeline before resampling.
 * Negative timestamps are encoder priming, while gaps and overlaps are real
 * discontinuities that must not be hidden by concatenating decoded chunks.
 */
export function planTimestampedAudioChunk(
  timestamp: number,
  frameCount: number,
  sampleRate: number,
  nextFrame: number,
): TimestampedAudioChunkPlan {
  if (!Number.isFinite(timestamp)) throw new Error("Audio sample timestamp is not finite.");
  if (!Number.isInteger(frameCount) || frameCount < 0) {
    throw new Error("Audio sample frame count is invalid.");
  }
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new Error("Audio sample rate is invalid.");
  }
  if (!Number.isInteger(nextFrame) || nextFrame < 0) {
    throw new Error("Audio timeline position is invalid.");
  }
  if (frameCount === 0) {
    return { gapFrames: 0, trimFrames: 0, appendFrames: 0, nextFrame };
  }

  const startFrame = Math.round(timestamp * sampleRate);
  const primingFrames = Math.min(frameCount, Math.max(0, -startFrame));
  const presentedFrames = frameCount - primingFrames;
  if (presentedFrames === 0) {
    return { gapFrames: 0, trimFrames: frameCount, appendFrames: 0, nextFrame };
  }

  const presentedStart = startFrame + primingFrames;
  const gapFrames = Math.max(0, presentedStart - nextFrame);
  const overlapFrames = Math.min(presentedFrames, Math.max(0, nextFrame - presentedStart));
  const trimFrames = primingFrames + overlapFrames;
  const appendFrames = frameCount - trimFrames;

  return {
    gapFrames,
    trimFrames,
    appendFrames,
    nextFrame: Math.max(nextFrame, presentedStart + presentedFrames),
  };
}

function lowerBound(values: ArrayLike<number>, target: number): number {
  let lower = 0;
  let upper = values.length;
  while (lower < upper) {
    const middle = (lower + upper) >>> 1;
    if (values[middle] < target) lower = middle + 1;
    else upper = middle;
  }
  return lower;
}

function upperBound(values: ArrayLike<number>, target: number): number {
  let lower = 0;
  let upper = values.length;
  while (lower < upper) {
    const middle = (lower + upper) >>> 1;
    if (values[middle] <= target) lower = middle + 1;
    else upper = middle;
  }
  return lower;
}

function rankVector(values: Float32Array): Float32Array {
  return percentileRanks(values, values.length, 1);
}

function rollingPercentile(values: Float32Array, window: number, percentile: number): Float32Array {
  const output = new Float32Array(values.length);
  const sorted: number[] = [];
  for (let index = 0; index < values.length; index += 1) {
    const insertion = lowerBound(sorted, values[index]);
    sorted.splice(insertion, 0, values[index]);
    if (index >= window) {
      const removal = lowerBound(sorted, values[index - window]);
      sorted.splice(removal, 1);
    }
    const position = (sorted.length - 1) * percentile;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    output[index] =
      sorted[lower] * (upper - position) + sorted[upper] * (position - lower);
  }
  return output;
}

export class AudioAccumulator {
  private readonly wasmResampler: StreamingAudioResampler | null;
  private readonly fft = new FFT(FFT_SIZE);
  private readonly window = Float32Array.from(
    { length: FRAME_SAMPLES },
    (_, index) => 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / (FRAME_SAMPLES - 1)),
  );
  private readonly bandIndexes = BAND_BOUNDS.map(([lower, upper]) => {
    const indexes: number[] = [];
    for (let bin = 0; bin <= FFT_SIZE / 2; bin += 1) {
      const frequency = (bin * TARGET_SAMPLE_RATE) / FFT_SIZE;
      if (frequency >= lower && frequency < upper) indexes.push(bin);
    }
    return indexes;
  });
  private previousSpectrum: Float32Array | null = null;
  private sourceRate = 0;
  private nextSourceFrame = 0;
  private sourceBuffer = new Float32Array(0);
  private sourcePosition = 0;
  private outputFrame: number[] = [];
  readonly features: AudioFrameFeatures = {
    rms: [],
    peak: [],
    spectralFlux: [],
    bandPower: BAND_BOUNDS.map(() => []),
  };

  constructor(wasmResampler: StreamingAudioResampler | null = null) {
    this.wasmResampler = wasmResampler;
  }

  push(sample: AudioSample, timestampOffset = 0): void {
    if (this.wasmResampler) {
      const planes: Float32Array[] = [];
      for (let channel = 0; channel < sample.numberOfChannels; channel += 1) {
        const allocation = sample.allocationSize({ format: "f32-planar", planeIndex: channel });
        const plane = new Float32Array(allocation / Float32Array.BYTES_PER_ELEMENT);
        sample.copyTo(plane, { format: "f32-planar", planeIndex: channel });
        planes.push(plane);
      }
      this.pushPlanar(
        planes,
        sample.timestamp + timestampOffset,
        sample.sampleRate,
      );
      return;
    }
    const allocation = sample.allocationSize({ format: "f32", planeIndex: 0 });
    const interleaved = new Float32Array(allocation / Float32Array.BYTES_PER_ELEMENT);
    sample.copyTo(interleaved, { format: "f32", planeIndex: 0 });
    const mono = new Float32Array(sample.numberOfFrames);
    for (let frame = 0; frame < sample.numberOfFrames; frame += 1) {
      let total = 0;
      for (let channel = 0; channel < sample.numberOfChannels; channel += 1) {
        total += interleaved[frame * sample.numberOfChannels + channel];
      }
      mono[frame] = total / sample.numberOfChannels;
    }
    this.pushMono(mono, sample.timestamp + timestampOffset, sample.sampleRate);
  }

  pushMono(mono: Float32Array, timestamp: number, sampleRate: number): void {
    if (this.wasmResampler) {
      this.pushPlanar([mono], timestamp, sampleRate);
      return;
    }
    if (this.sourceRate && this.sourceRate !== sampleRate) {
      throw new Error("Audio sample rate changed within the source track.");
    }
    this.sourceRate = sampleRate;

    const plan = planTimestampedAudioChunk(
      timestamp,
      mono.length,
      sampleRate,
      this.nextSourceFrame,
    );
    if (plan.gapFrames > 0) this.appendSilence(plan.gapFrames);
    if (plan.appendFrames > 0) this.appendSource(mono.subarray(plan.trimFrames));
    this.nextSourceFrame = plan.nextFrame;
  }

  pushPlanar(planes: readonly Float32Array[], timestamp: number, sampleRate: number): void {
    if (!this.wasmResampler) {
      throw new Error("Planar audio input requires the libswresample WASM runtime.");
    }
    if (planes.length < 1 || planes.length > 2) {
      throw new Error("The libswresample experiment supports mono or stereo audio.");
    }
    const frameCount = planes[0].length;
    if (planes.some((plane) => plane.length !== frameCount)) {
      throw new Error("Decoded audio planes have different frame counts.");
    }
    if (this.sourceRate && this.sourceRate !== sampleRate) {
      throw new Error("Audio sample rate changed within the source track.");
    }
    this.sourceRate = sampleRate;
    this.wasmResampler.configure(sampleRate, planes.length);
    const plan = planTimestampedAudioChunk(
      timestamp,
      frameCount,
      sampleRate,
      this.nextSourceFrame,
    );
    if (plan.gapFrames > 0) this.appendPlanarSilence(plan.gapFrames, planes.length);
    if (plan.appendFrames > 0) {
      this.pushResampled(
        this.wasmResampler.push(planes.map((plane) => plane.subarray(plan.trimFrames))),
      );
    }
    this.nextSourceFrame = plan.nextFrame;
  }

  finish(): AudioFrameFeatures {
    if (this.wasmResampler) {
      try {
        this.pushResampled(this.wasmResampler.flush());
      } finally {
        this.wasmResampler.close();
      }
    }
    if (this.sourceBuffer.length) {
      while (this.sourcePosition < this.sourceBuffer.length) {
        this.pushOutput(this.sourceBuffer[Math.min(this.sourceBuffer.length - 1, Math.floor(this.sourcePosition))]);
        this.sourcePosition += this.sourceRate / TARGET_SAMPLE_RATE;
      }
    }
    if (this.outputFrame.length) {
      while (this.outputFrame.length < FRAME_SAMPLES) this.outputFrame.push(0);
      this.processFrame(this.outputFrame);
      this.outputFrame = [];
    }
    return this.features;
  }

  dispose(): void {
    this.wasmResampler?.close();
  }

  private appendSource(mono: Float32Array): void {
    const combined = new Float32Array(this.sourceBuffer.length + mono.length);
    combined.set(this.sourceBuffer);
    combined.set(mono, this.sourceBuffer.length);
    this.sourceBuffer = combined;
    const step = this.sourceRate / TARGET_SAMPLE_RATE;
    while (this.sourcePosition + 1 < this.sourceBuffer.length) {
      const lower = Math.floor(this.sourcePosition);
      const fraction = this.sourcePosition - lower;
      const value =
        this.sourceBuffer[lower] * (1 - fraction) + this.sourceBuffer[lower + 1] * fraction;
      // The reference asks FFmpeg for signed 16-bit PCM. Quantization here reduces one avoidable mismatch.
      this.pushOutput(Math.max(-32768, Math.min(32767, Math.round(value * 32768))) / 32768);
      this.sourcePosition += step;
    }
    const consumed = Math.floor(this.sourcePosition);
    if (consumed > 0) {
      this.sourceBuffer = this.sourceBuffer.slice(consumed);
      this.sourcePosition -= consumed;
    }
  }

  private appendSilence(frameCount: number): void {
    const block = new Float32Array(Math.min(frameCount, Math.max(1, Math.round(this.sourceRate))));
    let remaining = frameCount;
    while (remaining > 0) {
      const length = Math.min(remaining, block.length);
      this.appendSource(length === block.length ? block : block.subarray(0, length));
      remaining -= length;
    }
  }

  private appendPlanarSilence(frameCount: number, channels: number): void {
    if (!this.wasmResampler) return;
    const blockFrames = Math.min(frameCount, Math.max(1, Math.round(this.sourceRate)));
    const block = Array.from({ length: channels }, () => new Float32Array(blockFrames));
    let remaining = frameCount;
    while (remaining > 0) {
      const length = Math.min(remaining, blockFrames);
      this.pushResampled(
        this.wasmResampler.push(
          length === blockFrames ? block : block.map((plane) => plane.subarray(0, length)),
        ),
      );
      remaining -= length;
    }
  }

  private pushResampled(samples: Int16Array): void {
    for (const sample of samples) this.pushOutput(sample / 32768);
  }

  private pushOutput(value: number): void {
    this.outputFrame.push(value);
    if (this.outputFrame.length === FRAME_SAMPLES) {
      this.processFrame(this.outputFrame);
      this.outputFrame = [];
    }
  }

  private processFrame(frame: number[]): void {
    let squareTotal = 0;
    let peak = 0;
    const fftInput = new Array<number>(FFT_SIZE).fill(0);
    for (let index = 0; index < FRAME_SAMPLES; index += 1) {
      const value = frame[index];
      squareTotal += value * value;
      peak = Math.max(peak, Math.abs(value));
      fftInput[index] = value * this.window[index];
    }
    this.features.rms.push(Math.sqrt(squareTotal / FRAME_SAMPLES));
    this.features.peak.push(peak);
    const complex = this.fft.createComplexArray();
    this.fft.realTransform(complex, fftInput);
    const spectrum = new Float32Array(FFT_SIZE / 2 + 1);
    let spectrumTotal = 0;
    for (let bin = 0; bin <= FFT_SIZE / 2; bin += 1) {
      const magnitude = Math.hypot(complex[bin * 2], complex[bin * 2 + 1]);
      spectrum[bin] = magnitude;
      spectrumTotal += magnitude;
    }
    for (let band = 0; band < this.bandIndexes.length; band += 1) {
      let power = 0;
      for (const bin of this.bandIndexes[band]) power += spectrum[bin] * spectrum[bin];
      this.features.bandPower[band].push(power);
    }
    const divisor = Math.max(spectrumTotal, 1e-8);
    for (let bin = 0; bin < spectrum.length; bin += 1) spectrum[bin] /= divisor;
    let fluxSquares = 0;
    if (this.previousSpectrum) {
      for (let bin = 0; bin < spectrum.length; bin += 1) {
        const difference = Math.max(spectrum[bin] - this.previousSpectrum[bin], 0);
        fluxSquares += difference * difference;
      }
    }
    this.features.spectralFlux.push(Math.sqrt(fluxSquares));
    this.previousSpectrum = spectrum;
  }
}

function buildFrameFeatureSources(frames: AudioFrameFeatures): Record<string, Float32Array> {
  const rms = Float32Array.from(frames.rms);
  const peak = Float32Array.from(frames.peak);
  const spectralFlux = Float32Array.from(frames.spectralFlux);
  const count = rms.length;
  const rmsNovelty = new Float32Array(count);
  for (let index = 1; index < count; index += 1) {
    rmsNovelty[index] = Math.max(rms[index] - rms[index - 1], 0);
  }
  const peakRank = rankVector(peak);
  const noveltyRank = rankVector(rmsNovelty);
  const fluxRank = rankVector(spectralFlux);
  const onsetStrength = new Float32Array(count);
  const contactLike = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    onsetStrength[index] = 0.45 * fluxRank[index] + 0.35 * noveltyRank[index] + 0.2 * peakRank[index];
    contactLike[index] = onsetStrength[index] * Math.sqrt(peakRank[index]);
  }
  const strongThreshold = Math.max(0.72, quantile(contactLike, 0.85));
  const cadence = rollingMean(contactLike, Math.round(2 / FRAME_SECONDS), false);
  const futureCadence = rollingMean(contactLike, Math.round(1 / FRAME_SECONDS), true);
  const cadenceCollapse = new Float32Array(count);
  const elapsed = new Float32Array(count).fill(10);
  let lastTime: number | null = null;
  for (let index = 0; index < count; index += 1) {
    cadenceCollapse[index] = Math.max(cadence[index] - futureCadence[index], 0);
    const timestamp = (index + 0.5) * FRAME_SECONDS;
    if (contactLike[index] >= strongThreshold && peakRank[index] >= 0.6) {
      lastTime = timestamp;
      elapsed[index] = 0;
    } else if (lastTime !== null) {
      elapsed[index] = Math.min(10, timestamp - lastTime);
    }
  }

  const noiseWindow = Math.round(10 / FRAME_SECONDS);
  const noiseFloor = rollingPercentile(rms, noiseWindow, 0.2);
  const snr = new Float32Array(count);
  const peakToRms = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    snr[index] = Math.log1p(Math.max(rms[index] - noiseFloor[index], 0) / (noiseFloor[index] + 1e-4));
    peakToRms[index] = Math.min(30, Math.max(0, peak[index] / (rms[index] + 1e-5)));
  }

  const sources: Record<string, Float32Array> = {
    audio_rms: rms,
    audio_peak: peak,
    audio_peak_to_rms: peakToRms,
    audio_noise_floor: noiseFloor,
    audio_snr: snr,
    audio_spectral_flux: spectralFlux,
    audio_rms_novelty: rmsNovelty,
    audio_onset_strength: onsetStrength,
    audio_contact_like_transient: contactLike,
    audio_onset_cadence: cadence,
    audio_cadence_collapse: cadenceCollapse,
    audio_seconds_since_transient: elapsed,
  };

  const bandLabels = ["80_250", "250_500", "500_1000", "1000_2000", "2000_4000", "4000_7800"];
  const broadbandNumerator = new Float32Array(count);
  const broadbandDenominator = new Float32Array(count);
  const normalizedFluxSquares = new Float32Array(count);
  for (let band = 0; band < frames.bandPower.length; band += 1) {
    const power = Float32Array.from(frames.bandPower[band]);
    const floor = rollingPercentile(power, noiseWindow, 0.2);
    const snrValues = new Float32Array(count);
    const fluxValues = new Float32Array(count);
    for (let index = 0; index < count; index += 1) {
      const excess = Math.max(power[index] - floor[index], 0);
      snrValues[index] = Math.log1p(excess / (floor[index] + 1e-8));
      fluxValues[index] = Math.max(snrValues[index] - snrValues[Math.max(0, index - 1)], 0);
      broadbandNumerator[index] += excess;
      broadbandDenominator[index] += floor[index];
      normalizedFluxSquares[index] += fluxValues[index] * fluxValues[index];
    }
    sources[`audio_band_${bandLabels[band]}_snr`] = snrValues;
    sources[`audio_band_${bandLabels[band]}_snr_flux`] = fluxValues;
  }
  const broadband = new Float32Array(count);
  const normalizedFlux = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    broadband[index] = Math.log1p(broadbandNumerator[index] / (broadbandDenominator[index] + 1e-8));
    normalizedFlux[index] = Math.sqrt(normalizedFluxSquares[index]);
  }
  sources.audio_noise_removed_broadband = broadband;
  sources.audio_noise_normalized_flux = normalizedFlux;
  return sources;
}

export async function extractAudioFeatures(
  audioTrack: InputAudioTrack | null,
  times: Float64Array,
  analysisWindow: AnalysisWindow,
  runtimeVariant: OnDeviceRuntimeVariant = "linear-v1",
  onProgress?: (progress: AnalysisProgress) => void,
): Promise<Float32Array> {
  const columns = AUDIO_FEATURE_NAMES.length;
  const output = new Float32Array(times.length * columns);
  if (!audioTrack || !(await audioTrack.canDecode())) return output;

  const accumulator = new AudioAccumulator(
    runtimeVariant === "libswresample-wasm-v1"
      ? await LibswresampleWasmResampler.create()
      : null,
  );
  const sink = new AudioSampleSink(audioTrack);
  const analysisDuration = analysisWindow.end - analysisWindow.start;
  let lastProgress = -1;
  let frames: AudioFrameFeatures;
  try {
    for await (const sample of sink.samples(analysisWindow.start)) {
      try {
        if (sample.timestamp >= analysisWindow.end) break;
        accumulator.push(sample, -analysisWindow.start);
        const progress = Math.min(
          analysisDuration,
          Math.max(0, sample.timestamp + sample.duration - analysisWindow.start),
        );
        const rounded = Math.floor(progress);
        if (rounded !== lastProgress) {
          lastProgress = rounded;
          onProgress?.({
            stage: "audio",
            completed: progress,
            total: analysisDuration,
            detail:
              runtimeVariant === "libswresample-wasm-v1"
                ? "Decoding audio through libswresample WASM"
                : "Decoding audio at 16 kHz",
          });
        }
      } finally {
        sample.close();
      }
    }
    frames = accumulator.finish();
  } catch (error) {
    accumulator.dispose();
    throw error;
  }
  if (frames.rms.length === 0) return output;
  const sources = buildFrameFeatureSources(frames);
  const audioTimes = Float64Array.from(
    { length: frames.rms.length },
    (_, index) => analysisWindow.start + (index + 0.5) * FRAME_SECONDS,
  );
  const halfWidth = 0.5 / ANALYSIS_FPS;
  const meanPooled = new Set([
    "audio_rms",
    "audio_noise_floor",
    "audio_snr",
    "audio_onset_cadence",
    "audio_cadence_collapse",
    "audio_seconds_since_transient",
  ]);
  for (let row = 0; row < times.length; row += 1) {
    output[row * columns] = 1;
    let left = lowerBound(audioTimes, times[row] - halfWidth);
    let right = upperBound(audioTimes, times[row] + halfWidth);
    if (right <= left) {
      const nearest = Math.min(
        audioTimes.length - 1,
        Math.max(0, Math.round(times[row] / FRAME_SECONDS - 0.5)),
      );
      left = nearest;
      right = nearest + 1;
    }
    for (let column = 1; column < columns; column += 1) {
      const name = AUDIO_FEATURE_NAMES[column];
      const source = sources[name];
      const segment = source.subarray(left, right);
      output[row * columns + column] = meanPooled.has(name)
        ? mean(segment)
        : Math.max(...segment);
    }
  }
  return output;
}
